import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import main
from server.main import (
    GROQ_MODEL,
    GROQ_TEXT_MODEL,
    AnalyzeRequest,
    ZeroPiiLoggingFilter,
    app,
    generate_groq_action,
    generate_groq_plan,
    generate_groq_verification,
)
from server.nvidia_client import (
    NVIDIA_BASE_URL,
    RETRYABLE_ERROR_TYPES,
    NvidiaClient,
    NvidiaError,
    classify_failure,
)

IMAGE = "data:image/png;base64,iVBORw0KGgo="
FAKE_NVIDIA_KEY = "nvapi-" + "0123456789abcdef" * 4
# A stand-in model for the tests only. The server has no default NVIDIA model.
TEST_MODEL = "vendor/test-vision-model"
ACTION_JSON = json.dumps({"action": "click", "selector": "#buy", "target": "Buy", "reasoning": "next step"})
PLAN_JSON = json.dumps({"reasoning": "short", "steps": ["Open the cart", "Check out"]})
PROVIDER_VARS = (
    "AI_PROVIDER",
    "NVIDIA_API_KEY",
    "NVIDIA_MODEL",
    "GROQ_API_KEY",
    "AI_FALLBACK",
    "NVIDIA_FALLBACK_MODELS",
)


def environment(**values):
    """Replace every provider variable with exactly `values`."""
    env = {key: value for key, value in os.environ.items() if key not in PROVIDER_VARS}
    env.update(values)
    return patch.dict(os.environ, env, clear=True)


def nvidia_env(**extra):
    """NVIDIA as the only provider: fallback off unless a test turns it on."""
    extra.setdefault("NVIDIA_MODEL", TEST_MODEL)
    extra.setdefault("AI_FALLBACK", "0")
    extra.setdefault("NVIDIA_FALLBACK_MODELS", "")
    return environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, **extra)


def completion_body(content, model=TEST_MODEL):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def action_request():
    return AnalyzeRequest(image=IMAGE, task="Add the item", elements=[{"selector": "#buy", "tag": "button"}])


class Recorder:
    """Mock NVIDIA HTTP endpoint: records requests, replays one response."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    def factory(self):
        return lambda api_key: NvidiaClient(api_key, transport=httpx.MockTransport(self))

    def body(self, index=-1):
        return json.loads(self.requests[index].content)


class FakeGroq:
    calls = []

    def __init__(self, api_key=None, max_retries=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        FakeGroq.calls.append(kwargs)
        content = PLAN_JSON if kwargs["model"] == GROQ_TEXT_MODEL else ACTION_JSON
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        )


class NoGroq:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Groq was reached although fallback is disabled")


def no_nvidia(api_key):
    raise AssertionError("Groq configuration must not create an NVIDIA client")


class ProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        FakeGroq.calls = []

    def test_default_provider_is_groq(self):
        with environment(GROQ_API_KEY="test-key"), patch("server.main.Groq", FakeGroq), \
                patch("server.main.create_nvidia_client", no_nvidia):
            self.assertEqual(main.selected_ai_provider(), "groq")
            response = generate_groq_action(action_request())

        self.assertEqual(response.actions[0].selector, "#buy")
        self.assertEqual(FakeGroq.calls[0]["model"], GROQ_MODEL)
        self.assertEqual(FakeGroq.calls[0]["max_completion_tokens"], 500)
        self.assertEqual(FakeGroq.calls[0]["reasoning_effort"], "none")

    def test_explicit_groq_keeps_existing_models(self):
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="test-key"), patch("server.main.Groq", FakeGroq), \
                patch("server.main.create_nvidia_client", no_nvidia):
            plan = generate_groq_plan(action_request())
            generate_groq_action(action_request())

        self.assertEqual(plan.steps, ["Open the cart", "Check out"])
        self.assertEqual([call["model"] for call in FakeGroq.calls], ["groq/compound-mini", GROQ_MODEL])

    def test_ai_provider_selects_nvidia(self):
        with nvidia_env(GROQ_API_KEY="groq-key"):
            self.assertEqual(main.selected_ai_provider(), "nvidia")
            self.assertEqual(main.configured_ai_provider_label(), "nvidia")

    def test_unknown_provider_is_a_clear_configuration_error(self):
        with environment(AI_PROVIDER="openai", GROQ_API_KEY="test-key"), patch("server.main.Groq", NoGroq):
            with self.assertRaises(HTTPException) as raised:
                generate_groq_action(action_request())
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("AI_PROVIDER", raised.exception.detail)
        self.assertIn("nvidia", raised.exception.detail)

    def test_migrated_away_providers_are_not_configurable(self):
        self.assertEqual(main.AI_PROVIDERS, ("groq", "nvidia"))
        for retired in ("gemini", "openrouter"):
            with self.subTest(provider=retired):
                with environment(AI_PROVIDER=retired, GROQ_API_KEY="test-key"):
                    self.assertEqual(main.configured_ai_provider_label(), "invalid")


class NvidiaRequestTests(unittest.TestCase):
    def test_action_request_uses_nvidia_url_key_configured_model_and_image(self):
        recorder = Recorder(completion_body(ACTION_JSON))
        with nvidia_env(GROQ_API_KEY="groq-key"), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()):
            response = generate_groq_action(action_request())

        request = recorder.requests[0]
        body = recorder.body()
        self.assertEqual(str(request.url), f"{NVIDIA_BASE_URL}/chat/completions")
        self.assertEqual(request.headers["authorization"], f"Bearer {FAKE_NVIDIA_KEY}")
        self.assertEqual(body["model"], TEST_MODEL)
        # The same sanitized screenshot data URL and prompt structure as Groq:
        # the multimodal pipeline is preserved, not downgraded to text.
        self.assertEqual(body["messages"][1]["content"][1], {"type": "image_url", "image_url": {"url": IMAGE}})
        self.assertIn("Interactive Elements", body["messages"][1]["content"][0]["text"])
        # Token cap unchanged; only parameter names are adapted.
        self.assertEqual(body["max_tokens"], 500)
        self.assertNotIn("max_completion_tokens", body)
        # reasoning_effort has no NVIDIA equivalent and is dropped, not forwarded.
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        # Existing action schema parses unchanged.
        self.assertEqual(response.actions[0].action, "click")
        self.assertEqual(response.actions[0].selector, "#buy")

    def test_nvidia_model_variable_applies_to_plan_action_and_verify(self):
        recorder = Recorder(completion_body(PLAN_JSON))
        with nvidia_env(NVIDIA_MODEL="vendor/chosen-model"), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()):
            plan = generate_groq_plan(action_request())
            recorder.payload = completion_body(ACTION_JSON)
            action = generate_groq_action(action_request())
            recorder.payload = completion_body('{"success": false, "reason": "not yet", "evidence": []}')
            verdict = generate_groq_verification(AnalyzeRequest(image=IMAGE, task="t", verify=True))

        self.assertEqual(plan.steps, ["Open the cart", "Check out"])
        self.assertEqual(action.actions[0].selector, "#buy")
        self.assertFalse(verdict.success)
        self.assertEqual(
            [json.loads(request.content)["model"] for request in recorder.requests],
            ["vendor/chosen-model"] * 3,
        )

    def test_structured_json_and_recoverable_wrappers_parse_into_the_existing_schema(self):
        recorder = Recorder(completion_body("```json\n" + ACTION_JSON + "\n```"))
        with nvidia_env(), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()):
            fenced = generate_groq_action(action_request())
            # A response cut off at the token limit goes through the existing
            # salvage path: the first complete action is recovered.
            recorder.payload = {"model": TEST_MODEL, "choices": [{
                "finish_reason": "length",
                "message": {"content": '{"action": "fill", "selector": "#q", "value": "hat", "reasoning": "ok"}, {"action": "cli'},
            }]}
            salvaged = generate_groq_action(action_request())

        self.assertEqual(fenced.actions[0].action, "click")
        self.assertEqual(salvaged.actions[0].action, "fill")
        self.assertEqual(salvaged.actions[0].value, "hat")

    def test_unparseable_response_is_a_provider_failure_not_a_fabricated_action(self):
        recorder = Recorder(completion_body("I cannot help with that."))
        with nvidia_env(), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()):
            with self.assertRaises(HTTPException) as raised:
                generate_groq_action(action_request())
        self.assertEqual(raised.exception.status_code, 502)

    def test_verification_schema_and_evidence_guard_are_preserved(self):
        evidence = [{"id": "E1", "kind": "control", "text": "Cart quantity 2"}]
        request = AnalyzeRequest(image=IMAGE, task="t", verify=True, verification_evidence=evidence, page_context="p")

        grounded = Recorder(completion_body('{"success": true, "reason": "quantity is 2", "evidence": ["E1"]}'))
        with nvidia_env(), patch("server.main.Groq", NoGroq), patch("server.main.create_nvidia_client", grounded.factory()):
            verdict = generate_groq_verification(request)
        self.assertTrue(verdict.success)
        body = grounded.body()
        self.assertEqual(body["messages"][1]["content"][1]["image_url"]["url"], IMAGE)
        self.assertEqual(body["max_tokens"], 150)

        ungrounded = Recorder(completion_body('{"success": true, "reason": "looks done", "evidence": []}'))
        with nvidia_env(), patch("server.main.Groq", NoGroq), patch("server.main.create_nvidia_client", ungrounded.factory()):
            verdict = generate_groq_verification(request)
        self.assertFalse(verdict.success)

    def test_missing_nvidia_key_is_clear_and_does_not_fall_back(self):
        with environment(AI_PROVIDER="nvidia", NVIDIA_MODEL=TEST_MODEL, GROQ_API_KEY="groq-key"), \
                patch("server.main.Groq", NoGroq):
            with self.assertRaises(HTTPException) as raised:
                generate_groq_action(action_request())
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("NVIDIA_API_KEY", raised.exception.detail)

    def test_missing_nvidia_model_is_a_clear_error_rather_than_a_guessed_model(self):
        with environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, GROQ_API_KEY="groq-key"), \
                patch("server.main.Groq", NoGroq), patch("server.main.create_nvidia_client", no_nvidia):
            with self.assertRaises(HTTPException) as raised:
                generate_groq_action(action_request())
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("NVIDIA_MODEL", raised.exception.detail)


class NvidiaErrorClassificationTests(unittest.TestCase):
    """Upstream failures are classified once, in the client, and not retried."""

    def call(self, payload, status):
        recorder = Recorder(payload, status=status)
        client = recorder.factory()(FAKE_NVIDIA_KEY)
        with self.assertRaises(NvidiaError) as raised:
            client.chat.completions.create(model=TEST_MODEL, messages=[{"role": "user", "content": "hi"}])
        return raised.exception

    def error(self, message, status):
        return self.call({"error": {"message": message, "code": status}}, status)

    def test_status_and_message_together_decide_the_classification(self):
        cases = {
            ("invalid api key", 401): "key_invalid",
            ("caller lacks permission", 403): "permission_denied",
            ("Model not found", 400): "model_unavailable",
            ("Invalid JSON payload", 400): "bad_request",
            ("unknown model", 404): "model_unavailable",
            ("Too many requests", 429): "rate_limited",
            ("gateway timeout", 504): "timeout",
            ("backend error", 500): "upstream_error",
            ("service unavailable", 503): "upstream_error",
        }
        for (detail, status), expected in cases.items():
            with self.subTest(status=status, detail=detail):
                self.assertEqual(classify_failure(status, detail), expected)

    def test_each_failure_is_actionable_and_names_the_provider(self):
        cases = [
            ("invalid api key", 401, "key_invalid", "authentication failed"),
            ("no permission", 403, "permission_denied", "denied access"),
            ("Model not found", 404, "model_unavailable", "not available"),
            ("Too many requests", 429, "rate_limited", "rate limit"),
            ("backend error", 500, "upstream_error", "unavailable"),
        ]
        for detail, status, expected_type, phrase in cases:
            with self.subTest(status=status):
                exc = self.error(detail, status)
                self.assertEqual(exc.error_type, expected_type)
                self.assertEqual(exc.status_code, status)
                self.assertIn(phrase, exc.message)
                self.assertIn("provider=nvidia", exc.message)
                self.assertIn(f"error_type={expected_type}", exc.message)
                self.assertIn(f"HTTP {status}", exc.message)

    def test_only_transient_failures_are_retryable(self):
        self.assertEqual(
            RETRYABLE_ERROR_TYPES,
            frozenset({"rate_limited", "timeout", "upstream_error", "unreachable"}),
        )
        for status in (401, 403, 404):
            with self.subTest(status=status):
                self.assertFalse(self.error("nope", status).retryable)
        for status in (429, 500, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(self.error("try later", status).retryable)

    def test_malformed_responses_are_classified_not_parsed(self):
        no_choices = self.call({"model": TEST_MODEL, "choices": []}, 200)
        self.assertEqual(no_choices.error_type, "invalid_response")

        client = NvidiaClient(FAKE_NVIDIA_KEY, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        ))
        with self.assertRaises(NvidiaError) as raised:
            client.chat.completions.create(model=TEST_MODEL, messages=[])
        self.assertEqual(raised.exception.error_type, "invalid_response")

    def test_network_and_timeout_failures_are_classified_without_the_key(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)

        def time_out(request):
            raise httpx.ReadTimeout("too slow", request=request)

        for handler, expected in ((refuse, "unreachable"), (time_out, "timeout")):
            with self.subTest(expected=expected):
                client = NvidiaClient(FAKE_NVIDIA_KEY, transport=httpx.MockTransport(handler))
                with self.assertRaises(NvidiaError) as raised:
                    client.chat.completions.create(model=TEST_MODEL, messages=[])
                self.assertEqual(raised.exception.error_type, expected)
                self.assertTrue(raised.exception.retryable)
                self.assertIsNone(raised.exception.status_code)
                self.assertEqual(raised.exception.server_status, 502)
                self.assertNotIn(FAKE_NVIDIA_KEY, str(raised.exception) + repr(client))

    def test_both_provider_key_formats_are_scrubbed_from_logs(self):
        # The log filter is the last line of defence for any text that reaches
        # a logger without passing through the client's own scrub.
        for key in (FAKE_NVIDIA_KEY, "gsk_" + "x" * 40):
            with self.subTest(prefix=key[:6]):
                scrubbed = ZeroPiiLoggingFilter._scrub(f"authorization=Bearer {key} tail")
                self.assertNotIn(key, scrubbed)
                self.assertIn("[API_KEY_REDACTED]", scrubbed)
                self.assertIn("tail", scrubbed)


class NvidiaServerTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def post_action(self, recorder, **extra):
        payload = {"image": IMAGE, "task": "Add the item", "elements": [{"selector": "#buy", "tag": "button"}], **extra}
        with nvidia_env(), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()), \
                patch("server.main.time.sleep") as slept:
            return self.client.post("/analyze", json=payload), slept

    def test_invalid_image_fails_closed_before_any_provider_call(self):
        recorder = Recorder(completion_body(ACTION_JSON))
        for extra in ({"image": "not-a-data-url"}, {"image": "data:text/plain;base64,aGVsbG8=", "verify": True}):
            response, _ = self.post_action(recorder, **extra)
            self.assertEqual(response.status_code, 400)
            self.assertIn("Fail-Closed", response.json()["detail"])
        self.assertEqual(recorder.requests, [])

    def test_upstream_statuses_are_reported_by_cause(self):
        cases = {
            401: (502, "authentication failed"),
            403: (502, "denied access"),
            404: (502, "not available"),
            429: (429, "rate limit"),
            503: (502, "provider is unavailable"),
        }
        for upstream, (expected_status, phrase) in cases.items():
            with self.subTest(upstream=upstream):
                recorder = Recorder({"error": {"message": "upstream said no", "code": upstream}}, status=upstream)
                response, _ = self.post_action(recorder)
                self.assertEqual(response.status_code, expected_status)
                self.assertIn(phrase, response.json()["detail"])
                self.assertIn(f"HTTP {upstream}", response.json()["detail"])
                self.assertIn("provider=nvidia", response.json()["detail"])

    def test_non_retryable_failures_cost_exactly_one_call_per_request(self):
        for label, extra in (
            ("action", {}),
            ("plan", {"plan": True}),
            ("verify", {"verify": True, "page_context": "p"}),
        ):
            with self.subTest(path=label):
                recorder = Recorder({"error": {"message": "no permission", "code": 403}}, status=403)
                response, slept = self.post_action(recorder, **extra)
                self.assertEqual(len(recorder.requests), 1)
                slept.assert_not_called()
                self.assertNotIn(FAKE_NVIDIA_KEY, response.text)

    def test_a_retryable_rate_limit_still_retries_as_before(self):
        recorder = Recorder({"error": {"message": "Too many requests", "code": 429}}, status=429)
        response, slept = self.post_action(recorder)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(recorder.requests), 4)
        self.assertTrue(slept.called)

    def test_non_nvidia_exceptions_keep_their_existing_retry_behaviour(self):
        self.assertFalse(main.provider_error_is_permanent(ValueError("boom")))
        self.assertFalse(main.provider_error_is_permanent(RuntimeError("upstream 403")))

    def test_api_key_never_appears_in_errors_or_logs(self):
        recorder = Recorder({"error": {"message": f"Invalid credentials {FAKE_NVIDIA_KEY}", "code": 401}}, status=401)
        with self.assertLogs(level="INFO") as logs:
            response, _ = self.post_action(recorder)
            plan, _ = self.post_action(recorder, plan=True)
            verify, _ = self.post_action(recorder, verify=True, page_context="p")

        for result in (response, plan):
            self.assertEqual(result.status_code, 502)
            self.assertNotIn(FAKE_NVIDIA_KEY, result.text)
        self.assertFalse(verify.json()["success"])
        self.assertNotIn(FAKE_NVIDIA_KEY, verify.text)
        self.assertNotIn(FAKE_NVIDIA_KEY, "\n".join(logs.output))
        self.assertNotIn(FAKE_NVIDIA_KEY, ZeroPiiLoggingFilter._scrub(f"key={FAKE_NVIDIA_KEY}"))

    def test_health_reports_provider_without_secrets_or_a_live_call(self):
        with nvidia_env(), patch("server.main.create_nvidia_client", no_nvidia):
            health = self.client.get("/health").json()
        self.assertEqual(health["ai_provider"], "nvidia")
        self.assertEqual(health["nvidia_model"], TEST_MODEL)
        self.assertTrue(health["nvidia_configured"])
        self.assertNotIn("ok", health)
        self.assertNotIn(FAKE_NVIDIA_KEY, json.dumps(health))
        self.assertNotIn("gemini", json.dumps(health).lower())


class NvidiaPreflightTests(unittest.TestCase):
    ENDPOINT = "/providers/nvidia/preflight"

    def setUp(self):
        self.client = TestClient(app)

    def preflight(self, recorder=None, **env):
        factory = recorder.factory() if recorder else no_nvidia
        with environment(**env), patch("server.main.create_nvidia_client", factory):
            response = self.client.get(self.ENDPOINT)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_missing_api_key_is_reported_without_calling_nvidia(self):
        body = self.preflight(AI_PROVIDER="nvidia", NVIDIA_MODEL=TEST_MODEL, GROQ_API_KEY="groq-key")
        self.assertEqual(body["provider"], "nvidia")
        self.assertFalse(body["configured"])
        self.assertFalse(body["ok"])
        self.assertEqual(body["error_type"], "key_missing")
        self.assertIsNone(body["status_code"])
        self.assertIn("NVIDIA_API_KEY", body["message"])

    def test_missing_model_is_reported_without_calling_nvidia(self):
        body = self.preflight(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error_type"], "model_missing")
        self.assertIn("NVIDIA_MODEL", body["message"])

    def test_successful_preflight_is_one_minimal_text_only_request(self):
        recorder = Recorder(completion_body("OK"))
        body = self.preflight(recorder, AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL)

        self.assertEqual(body, {
            "provider": "nvidia",
            "configured": True,
            "ok": True,
            "model": TEST_MODEL,
            "error_type": None,
            "status_code": 200,
            "message": "NVIDIA preflight successful.",
        })
        self.assertEqual(len(recorder.requests), 1)
        request, sent = recorder.requests[0], recorder.body()
        self.assertEqual(request.url.path, "/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], f"Bearer {FAKE_NVIDIA_KEY}")
        # Exactly one trivial text message: no screenshot, DOM, task or context.
        self.assertEqual(sent["messages"], [{"role": "user", "content": "Reply with OK."}])
        self.assertNotIn("image_url", json.dumps(sent))
        self.assertEqual(sent["max_tokens"], 16)
        self.assertEqual(sent["model"], TEST_MODEL)

    def failing_preflight(self, payload, status):
        return self.preflight(
            Recorder(payload, status=status),
            AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL,
        )

    def test_each_upstream_failure_gets_its_own_error_type(self):
        cases = [
            ("invalid api key", 401, "key_invalid", "authentication failed"),
            ("caller lacks permission", 403, "permission_denied", "denied access"),
            ("Model not found", 404, "model_unavailable", "not available"),
            ("Too many requests", 429, "rate_limited", "rate limit"),
            ("Internal error", 500, "upstream_error", "unavailable"),
            ("Service unavailable", 503, "upstream_error", "unavailable"),
            ("gateway timeout", 504, "timeout", "timed out"),
        ]
        for detail, status, expected_type, phrase in cases:
            with self.subTest(status=status, expected=expected_type):
                body = self.failing_preflight({"error": {"message": detail, "code": status}}, status)
                self.assertFalse(body["ok"])
                self.assertTrue(body["configured"])
                self.assertEqual(body["error_type"], expected_type)
                self.assertEqual(body["status_code"], status)
                self.assertIn(phrase, body["message"])

    def test_malformed_and_empty_responses_are_unexpected_format(self):
        no_choices = self.failing_preflight({"model": TEST_MODEL, "choices": []}, 200)
        self.assertEqual(no_choices["error_type"], "invalid_response")

        empty = self.failing_preflight(completion_body(""), 200)
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["error_type"], "invalid_response")
        self.assertEqual(empty["status_code"], 502)

    def test_the_key_never_appears_in_the_response_or_the_logs(self):
        recorder = Recorder(
            {"error": {"message": f"Invalid credentials {FAKE_NVIDIA_KEY}", "code": 401}}, status=401
        )
        with self.assertLogs(level="INFO") as logs:
            with environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL), \
                    patch("server.main.create_nvidia_client", recorder.factory()):
                response = self.client.get(self.ENDPOINT)

        self.assertEqual(response.json()["error_type"], "key_invalid")
        self.assertNotIn(FAKE_NVIDIA_KEY, response.text)
        joined = "\n".join(logs.output)
        self.assertNotIn(FAKE_NVIDIA_KEY, joined)
        self.assertIn("error_type=key_invalid", joined)
        self.assertIn("upstream_status=401", joined)

    def test_no_retired_provider_preflight_endpoint_remains(self):
        for retired in ("/providers/gemini/preflight", "/providers/openrouter/preflight"):
            with self.subTest(endpoint=retired):
                self.assertEqual(self.client.get(retired).status_code, 404)


class ExtensionProviderSelectionTests(unittest.TestCase):
    """The popup sends its selector value as `provider`; the server routes on it."""

    def setUp(self):
        self.client = TestClient(app)
        FakeGroq.calls = []

    @staticmethod
    def payload(provider=None, **extra):
        body = {"image": IMAGE, "task": "Add the item", "elements": [{"selector": "#buy", "tag": "button"}], **extra}
        if provider is not None:
            body["provider"] = provider
        return body

    def test_nvidia_selection_routes_plan_action_and_verify_to_nvidia(self):
        recorder = Recorder(completion_body(PLAN_JSON))
        # The server default stays Groq: the extension's explicit choice decides.
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="groq-key",
                         NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL), \
                patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()), \
                patch("server.main.generate_ollama_action", AsyncMock(side_effect=AssertionError("not ollama"))), \
                self.assertLogs(level="INFO") as logs:
            plan = self.client.post("/analyze", json=self.payload("nvidia", plan=True))
            recorder.payload = completion_body(ACTION_JSON)
            action = self.client.post("/analyze", json=self.payload("nvidia"))
            recorder.payload = completion_body('{"success": false, "reason": "not yet", "evidence": []}')
            verify = self.client.post("/analyze", json=self.payload("nvidia", verify=True, page_context="p"))

        self.assertEqual([result.status_code for result in (plan, action, verify)], [200, 200, 200])
        self.assertEqual(plan.json()["steps"], ["Open the cart", "Check out"])
        self.assertEqual(action.json()["actions"][0]["selector"], "#buy")
        self.assertIn("success", verify.json())
        self.assertEqual(len(recorder.requests), 3)
        self.assertTrue(all(
            str(request.url).startswith("https://integrate.api.nvidia.com/v1/") for request in recorder.requests
        ))
        for result in (plan, action, verify):
            self.assertNotIn(FAKE_NVIDIA_KEY, result.text)
        joined = "\n".join(logs.output)
        self.assertNotIn(FAKE_NVIDIA_KEY, joined)
        self.assertIn(f"[AI-PROVIDER] provider=nvidia model={TEST_MODEL}", joined)

    def test_groq_selection_keeps_groq_even_when_server_default_is_nvidia(self):
        with nvidia_env(GROQ_API_KEY="groq-key"), patch("server.main.Groq", FakeGroq), \
                patch("server.main.create_nvidia_client", no_nvidia):
            response = self.client.post("/analyze", json=self.payload("groq"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(FakeGroq.calls[0]["model"], GROQ_MODEL)

    def test_ollama_selection_is_unchanged(self):
        ollama = AsyncMock(return_value={"action": "click", "selector": "#buy", "reasoning": "local"})
        with nvidia_env(GROQ_API_KEY="groq-key"), \
                patch("server.main.generate_ollama_action", ollama), \
                patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", no_nvidia):
            response = self.client.post("/analyze", json=self.payload("ollama"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["actions"][0]["selector"], "#buy")
        self.assertEqual(ollama.await_args.args[3], main.OLLAMA_MODEL)

    def test_request_without_provider_uses_server_default(self):
        recorder = Recorder(completion_body(ACTION_JSON))
        with nvidia_env(), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()):
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(recorder.requests), 1)

    def test_unknown_provider_value_is_rejected(self):
        with patch("server.main.Groq", NoGroq), patch("server.main.create_nvidia_client", no_nvidia):
            for unknown in ("gemini", "openrouter", "openai"):
                with self.subTest(provider=unknown):
                    response = self.client.post("/analyze", json=self.payload(unknown))
                    self.assertEqual(response.status_code, 422)

    def test_providers_endpoint_lists_nvidia_and_no_retired_provider(self):
        with nvidia_env(GROQ_API_KEY="gsk_" + "x" * 40):
            response = self.client.get("/providers")

        self.assertEqual([provider["id"] for provider in response.json()["providers"]], ["groq", "nvidia", "ollama"])
        self.assertIn(TEST_MODEL, response.text)
        for retired in ("gemini", "openrouter"):
            self.assertNotIn(retired, response.text.lower())
        self.assertNotIn(FAKE_NVIDIA_KEY, response.text)
        self.assertNotIn("gsk_", response.text)


class NvidiaCapacityTests(unittest.TestCase):
    """An exhausted NVIDIA worker is a capacity failure, not a transient blip."""

    CAPACITY_MESSAGES = (
        "ResourceExhausted: Worker local total request limit reached (1296/16)",
        "ResourceExhausted: Worker local total request limit reached (16/16)",
        "resourceexhausted: worker local total request limit reached (3/16)",
    )

    def setUp(self):
        self.client = TestClient(app)

    def capacity_recorder(self, message=CAPACITY_MESSAGES[0], status=503):
        return Recorder({"error": {"message": message, "code": status}}, status=status)

    def raise_from(self, recorder):
        client = recorder.factory()(FAKE_NVIDIA_KEY)
        with self.assertRaises(NvidiaError) as raised:
            client.chat.completions.create(model=TEST_MODEL, messages=[{"role": "user", "content": "hi"}])
        return raised.exception

    def test_capacity_messages_are_upstream_capacity_and_not_retryable(self):
        for message in self.CAPACITY_MESSAGES:
            with self.subTest(message=message):
                self.assertEqual(classify_failure(503, message), "upstream_capacity")
                exc = self.raise_from(self.capacity_recorder(message))
                self.assertEqual(exc.error_type, "upstream_capacity")
                self.assertEqual(exc.status_code, 503)
                self.assertFalse(exc.retryable)
                self.assertNotIn("upstream_capacity", RETRYABLE_ERROR_TYPES)

    def test_the_message_is_concise_and_carries_provider_metadata(self):
        exc = self.raise_from(self.capacity_recorder())
        self.assertIn("NVIDIA is temporarily unavailable: upstream worker capacity exhausted", exc.message)
        self.assertIn("provider=nvidia", exc.message)
        self.assertIn("error_type=upstream_capacity", exc.message)
        self.assertIn("HTTP 503", exc.message)
        # The raw worker ratio is internal noise and is not echoed at the user.
        self.assertNotIn("1296/16", exc.message)

    def test_an_ordinary_503_keeps_its_existing_classification_and_retryability(self):
        self.assertEqual(classify_failure(503, "Service temporarily unavailable"), "upstream_error")
        exc = self.raise_from(Recorder({"error": {"message": "Service temporarily unavailable", "code": 503}}, status=503))
        self.assertEqual(exc.error_type, "upstream_error")
        self.assertTrue(exc.retryable)
        # Not every 5xx became non-retryable.
        for status in (500, 502, 503):
            with self.subTest(status=status):
                self.assertEqual(classify_failure(status, "backend error"), "upstream_error")
                self.assertIn("upstream_error", RETRYABLE_ERROR_TYPES)

    def test_capacity_does_not_reclassify_the_non_5xx_band(self):
        # A capacity phrase alongside a rate limit keeps rate-limit handling.
        self.assertEqual(classify_failure(429, self.CAPACITY_MESSAGES[0]), "rate_limited")
        self.assertEqual(classify_failure(401, self.CAPACITY_MESSAGES[0]), "key_invalid")
        self.assertEqual(classify_failure(404, self.CAPACITY_MESSAGES[0]), "model_unavailable")

    def post(self, recorder, **extra):
        payload = {"image": IMAGE, "task": "Add the item", "elements": [{"selector": "#buy", "tag": "button"}], **extra}
        with nvidia_env(GROQ_API_KEY="groq-key"), patch("server.main.Groq", NoGroq), \
                patch("server.main.create_nvidia_client", recorder.factory()), \
                patch("server.main.generate_ollama_action", AsyncMock(side_effect=AssertionError("no ollama fallback"))), \
                patch("server.main.time.sleep") as slept:
            return self.client.post("/analyze", json=payload), slept

    def test_planning_makes_no_second_immediate_attempt_on_capacity(self):
        recorder = self.capacity_recorder()
        response, slept = self.post(recorder, plan=True)
        self.assertEqual(len(recorder.requests), 1)
        slept.assert_not_called()

    def test_action_and_verify_also_stop_after_one_attempt(self):
        for label, extra in (("action", {}), ("verify", {"verify": True, "page_context": "p"})):
            with self.subTest(path=label):
                recorder = self.capacity_recorder()
                response, slept = self.post(recorder, **extra)
                self.assertEqual(len(recorder.requests), 1)
                slept.assert_not_called()

    def test_the_extension_receives_a_clean_provider_unavailable_error(self):
        for label, extra in (("action", {}), ("plan", {"plan": True})):
            with self.subTest(path=label):
                response, _ = self.post(self.capacity_recorder(), **extra)
                detail = response.json()["detail"]
                self.assertEqual(response.status_code, 502)
                self.assertIn("NVIDIA is temporarily unavailable: upstream worker capacity exhausted", detail)
                self.assertIn("error_type=upstream_capacity", detail)
                self.assertIn("provider=nvidia", detail)
                # Not dressed up as a model/action failure, and no key leaks.
                self.assertNotIn("couldn't produce", detail)
                self.assertNotIn(FAKE_NVIDIA_KEY, response.text)

    def test_ai_fallback_off_reports_the_failure_without_trying_another_provider(self):
        # nvidia_env sets AI_FALLBACK=0; NoGroq and the ollama AsyncMock both
        # raise if either is reached anyway.
        recorder = self.capacity_recorder()
        response, _ = self.post(recorder, plan=True)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(recorder.requests), 1)
        self.assertNotIn("groq", response.json()["detail"].lower())
        self.assertNotIn("ollama", response.json()["detail"].lower())


class ModelRouter:
    """Mock NVIDIA endpoint that answers per model, recording the order tried."""

    def __init__(self, responses):
        self.responses = responses
        self.models = []

    def __call__(self, request):
        model = json.loads(request.content)["model"]
        self.models.append(model)
        status, payload = self.responses.get(model, (404, {"error": {"message": "unknown model", "code": 404}}))
        return httpx.Response(status, json=payload)

    def factory(self, fallbacks):
        def create(api_key):
            return NvidiaClient(
                api_key, transport=httpx.MockTransport(self), fallback_models=fallbacks
            )
        return create


class RateLimitedGroq:
    """A Groq client whose every call reports a long rate-limit wait."""

    retry_after = "19"
    calls = 0

    def __init__(self, api_key=None, max_retries=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        RateLimitedGroq.calls += 1
        error = RuntimeError("Rate limit reached. Try again in 19s")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": RateLimitedGroq.retry_after})
        raise error


class RecoveringGroq:
    """Rate-limited until the wait is taken, then serves the request."""

    calls = 0
    limited_calls = 2

    def __init__(self, api_key=None, max_retries=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        RecoveringGroq.calls += 1
        if RecoveringGroq.calls <= RecoveringGroq.limited_calls:
            error = RuntimeError("Rate limit reached. Try again in 11s")
            error.status_code = 429
            error.response = SimpleNamespace(headers={"retry-after": "11"})
            raise error
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=ACTION_JSON))],
        )


CAPACITY_503 = (503, {"error": {"message": "ResourceExhausted: Worker local total request limit reached (1296/16)", "code": 503}})


class FallbackTests(unittest.TestCase):
    """An exhausted worker ends the request only when nothing else can serve it."""

    def setUp(self):
        self.client = TestClient(app)
        FakeGroq.calls = []

    def payload(self, **extra):
        return {"image": IMAGE, "task": "Add the item", "elements": [{"selector": "#buy", "tag": "button"}], **extra}

    def test_the_client_retries_a_capacity_failure_on_the_next_model(self):
        router = ModelRouter({TEST_MODEL: CAPACITY_503, "vendor/backup-vision": (200, completion_body(ACTION_JSON, "vendor/backup-vision"))})
        client = router.factory(("vendor/backup-vision",))(FAKE_NVIDIA_KEY)

        completion = client.chat.completions.create(model=TEST_MODEL, messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(router.models, [TEST_MODEL, "vendor/backup-vision"])
        self.assertEqual(completion.model, "vendor/backup-vision")

    def test_a_bad_request_is_not_retried_on_another_model(self):
        router = ModelRouter({TEST_MODEL: (400, {"error": {"message": "malformed", "code": 400}})})
        client = router.factory(("vendor/backup-vision",))(FAKE_NVIDIA_KEY)

        with self.assertRaises(NvidiaError):
            client.chat.completions.create(model=TEST_MODEL, messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(router.models, [TEST_MODEL])

    def test_every_nvidia_model_exhausted_falls_through_to_groq(self):
        router = ModelRouter({TEST_MODEL: CAPACITY_503, "vendor/backup-vision": CAPACITY_503})
        with nvidia_env(GROQ_API_KEY="groq-key", AI_FALLBACK="1"), patch("server.main.Groq", FakeGroq), \
                patch("server.main.create_nvidia_client", router.factory(("vendor/backup-vision",))), \
                patch("server.main.generate_ollama_action", AsyncMock(side_effect=AssertionError("groq should have served this"))), \
                patch("server.main.time.sleep"):
            response = self.client.post("/analyze", json=self.payload(plan=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["steps"], ["Open the cart", "Check out"])
        self.assertEqual(router.models, [TEST_MODEL, "vendor/backup-vision"])
        self.assertEqual(FakeGroq.calls[0]["model"], GROQ_TEXT_MODEL)

    def test_ollama_serves_an_action_only_when_opted_into(self):
        # No GROQ_API_KEY here, so with AI_FALLBACK_OLLAMA the chain is nvidia then ollama.
        router = ModelRouter({TEST_MODEL: CAPACITY_503})
        with nvidia_env(AI_FALLBACK="1", AI_FALLBACK_OLLAMA="1", NVIDIA_FALLBACK_MODELS=""), \
                patch("server.main.create_nvidia_client", router.factory(())), \
                patch("server.main.generate_ollama_action", AsyncMock(return_value=json.loads(ACTION_JSON))), \
                patch("server.main.time.sleep"):
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selector"], "#buy")
        self.assertEqual(router.models, [TEST_MODEL])

    def test_the_reported_error_is_the_selected_providers_own(self):
        router = ModelRouter({TEST_MODEL: CAPACITY_503})
        with nvidia_env(AI_FALLBACK="1", NVIDIA_FALLBACK_MODELS=""), \
                patch("server.main.create_nvidia_client", router.factory(())), \
                patch("server.main.generate_ollama_action", AsyncMock(side_effect=HTTPException(status_code=502, detail="Ollama is not running"))), \
                patch("server.main.time.sleep"):
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 502)
        detail = response.json()["detail"]
        self.assertIn("error_type=upstream_capacity", detail)
        self.assertNotIn("Ollama", detail)

    def test_a_long_rate_limit_hands_off_instead_of_sleeping(self):
        RateLimitedGroq.calls = 0
        RateLimitedGroq.retry_after = "19"
        router = ModelRouter({TEST_MODEL: (200, completion_body(ACTION_JSON))})
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="groq-key", NVIDIA_API_KEY=FAKE_NVIDIA_KEY,
                         NVIDIA_MODEL=TEST_MODEL, NVIDIA_FALLBACK_MODELS=""), \
                patch("server.main.Groq", RateLimitedGroq), \
                patch("server.main.create_nvidia_client", router.factory(())), \
                patch("server.main.time.sleep") as slept:
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selector"], "#buy")
        # Groq was asked once and then dropped; nothing waited 19 seconds.
        self.assertEqual(RateLimitedGroq.calls, 1)
        slept.assert_not_called()
        self.assertEqual(router.models, [TEST_MODEL])

    def test_a_short_rate_limit_is_still_waited_out_on_the_same_provider(self):
        RateLimitedGroq.calls = 0
        RateLimitedGroq.retry_after = "1"
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="groq-key"), \
                patch("server.main.Groq", RateLimitedGroq), \
                patch("server.main.generate_ollama_action", AsyncMock(side_effect=AssertionError("no handoff expected"))), \
                patch("server.main.time.sleep") as slept:
            self.client.post("/analyze", json=self.payload())

        self.assertEqual(RateLimitedGroq.calls, 4)
        self.assertEqual(slept.call_count, 3)

    def test_the_last_provider_waits_rather_than_giving_up(self):
        RateLimitedGroq.calls = 0
        RateLimitedGroq.retry_after = "19"
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="groq-key", AI_FALLBACK="0"), \
                patch("server.main.Groq", RateLimitedGroq), \
                patch("server.main.time.sleep") as slept:
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 429)
        self.assertEqual(slept.call_count, 3)

    def test_the_chain_skips_providers_that_are_not_configured(self):
        request = AnalyzeRequest(image=IMAGE, task="t")
        with environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL):
            self.assertEqual(main.provider_chain(request), ["nvidia"])
        with environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL, GROQ_API_KEY="k"):
            self.assertEqual(main.provider_chain(request), ["nvidia", "groq"])
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="k"):
            self.assertEqual(main.provider_chain(request), ["groq"])
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="k", AI_FALLBACK="0"):
            self.assertEqual(main.provider_chain(request), ["groq"])

    def test_the_slow_local_model_is_never_an_automatic_fallback(self):
        request = AnalyzeRequest(image=IMAGE, task="t")
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="k", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL):
            self.assertEqual(main.provider_chain(request), ["groq", "nvidia"])
        # Opt in explicitly, and it takes its place at the end of the chain.
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="k", AI_FALLBACK_OLLAMA="1"):
            self.assertEqual(main.provider_chain(request), ["groq", "ollama"])
        # An explicit Ollama request is untouched by any of this.
        local = AnalyzeRequest(image=IMAGE, task="t", provider="ollama")
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="k"):
            self.assertEqual(main.provider_chain(local), ["ollama", "groq"])

    def test_when_every_fallback_fails_the_rate_limit_is_waited_out(self):
        RateLimitedGroq.calls = 0
        RateLimitedGroq.retry_after = "11"
        # Groq hands off to dodge an 11s wait, NVIDIA is exhausted, so the
        # round is served by waiting on Groq rather than lost.
        router = ModelRouter({TEST_MODEL: CAPACITY_503})
        with environment(AI_PROVIDER="groq", GROQ_API_KEY="groq-key", NVIDIA_API_KEY=FAKE_NVIDIA_KEY,
                         NVIDIA_MODEL=TEST_MODEL, NVIDIA_FALLBACK_MODELS=""), \
                patch("server.main.Groq", RecoveringGroq), \
                patch("server.main.create_nvidia_client", router.factory(())), \
                patch("server.main.time.sleep") as slept:
            RecoveringGroq.calls = 0
            response = self.client.post("/analyze", json=self.payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selector"], "#buy")
        self.assertEqual(router.models, [TEST_MODEL])
        # First call handed off; the retry waited out the limit and succeeded.
        self.assertEqual(RecoveringGroq.calls, 3)
        slept.assert_called_once()

    def test_verification_never_falls_back_to_ollama(self):
        request = AnalyzeRequest(image=IMAGE, task="t", verify=True, page_context="p")
        with environment(AI_PROVIDER="nvidia", NVIDIA_API_KEY=FAKE_NVIDIA_KEY, NVIDIA_MODEL=TEST_MODEL, GROQ_API_KEY="k"):
            self.assertEqual(main.provider_chain(request), ["nvidia", "groq"])

    def test_the_models_endpoint_answers_and_hides_credentials(self):
        with nvidia_env(GROQ_API_KEY="groq-key", NVIDIA_FALLBACK_MODELS="vendor/backup-vision"):
            response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        ids = [model["id"] for model in response.json()["data"]]
        self.assertIn(TEST_MODEL, ids)
        self.assertIn("vendor/backup-vision", ids)
        self.assertNotIn(FAKE_NVIDIA_KEY, response.text)


if __name__ == "__main__":
    unittest.main()
