"""NVIDIA API chat-completions client for the Privacy Vision Agent server.

The NVIDIA API is OpenAI-compatible, so this client exposes the one call the
server's generation functions already make on a Groq client,
``client.chat.completions.create(**kwargs)``, and raises errors carrying the
same ``status_code`` / ``body`` / ``response`` attributes their retry handling
reads. Prompts, JSON parsing, salvage and verification stay in server/main.py
and are shared with the Groq path.

Messages pass through unchanged, including the ``image_url`` part holding the
redacted screenshot data URL, so the existing multimodal pipeline is preserved
end to end. Only parameter names differ.

The API key is only placed in the Authorization header. It is never logged and
is scrubbed from any error text before that text leaves this module.
"""
import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import httpx

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
REQUEST_TIMEOUT_SECONDS = 90.0
MAX_ERROR_CHARS = 300

logger = logging.getLogger("server.nvidia_client")

# Every provider failure is classified into one of these types. Callers switch
# on the type rather than re-reading the provider's prose, and the preflight
# endpoint reports it verbatim.
ERROR_TYPE_SUMMARIES = {
    "key_missing": "NVIDIA_API_KEY is not configured. Set it in .env.",
    "model_missing": "NVIDIA_MODEL is not configured. Set it in .env.",
    "key_invalid": "NVIDIA authentication failed - check NVIDIA_API_KEY",
    "permission_denied": "NVIDIA denied access to this model or key",
    "model_unavailable": "NVIDIA model is not available for this API key",
    "bad_request": "NVIDIA rejected the request",
    "rate_limited": "NVIDIA rate limit or quota reached",
    "timeout": "NVIDIA request timed out",
    "upstream_error": "NVIDIA provider is unavailable",
    "upstream_capacity": "NVIDIA is temporarily unavailable: upstream worker capacity exhausted",
    "unreachable": "NVIDIA is unreachable",
    "invalid_response": "NVIDIA returned an unexpected response format",
    "invalid_image": "NVIDIA image input was rejected before sending",
}

# Only these are worth trying again. A rejected or unpermitted key is a
# configuration problem: retrying burns agent rounds and changes nothing, and
# neither does hammering a worker that just reported its request limit reached.
RETRYABLE_ERROR_TYPES = frozenset({"rate_limited", "timeout", "upstream_error", "unreachable"})

# Retrying the *same* model cannot fix an exhausted worker, but another model
# served by the same account usually answers immediately, and so does another
# provider. These are the failures worth escaping rather than reporting: the
# request itself was fine, only the machine behind this model was not.
FALLBACK_ERROR_TYPES = frozenset({
    "rate_limited",
    "timeout",
    "upstream_error",
    "upstream_capacity",
    "unreachable",
    "model_unavailable",
    "permission_denied",
    "invalid_response",
})

# For these the summary is already the whole actionable message.
_DETAIL_SUPPRESSED = frozenset({"key_invalid", "key_missing", "model_missing", "upstream_capacity"})

_MODEL_MARKERS = ("model not found", "unknown model", "model does not exist", "no such model")

# NVIDIA reports an exhausted worker as a 5xx whose body names the condition,
# e.g. "ResourceExhausted: Worker local total request limit reached (1296/16)".
# The counts vary, so the phrases are matched rather than any ratio.
_CAPACITY_MARKERS = (
    "resourceexhausted",
    "resource exhausted",
    "worker local total request limit reached",
    "total request limit reached",
)


def classify_failure(status_code: int | None, detail: str = "") -> str:
    """Map an upstream status plus its (already scrubbed) message to an error type."""
    text = (detail or "").lower()
    if status_code in (401, 403):
        # NVIDIA returns 401 for a bad key and 403 once a valid key lacks access.
        return "key_invalid" if status_code == 401 else "permission_denied"
    if status_code == 400:
        return "model_unavailable" if any(m in text for m in _MODEL_MARKERS) else "bad_request"
    if status_code == 404:
        return "model_unavailable"
    if status_code in (408, 504):
        return "timeout"
    if status_code == 429:
        return "rate_limited"
    # Capacity exhaustion only reclassifies the 5xx/unknown band, so ordinary
    # transient upstream failures keep their existing retryable classification.
    if any(marker in text for marker in _CAPACITY_MARKERS):
        return "upstream_capacity"
    return "upstream_error"


class NvidiaError(Exception):
    """A classified provider failure whose message never contains the API key."""

    def __init__(
        self,
        status_code: int | None,
        message: str,
        body: Any = None,
        response: httpx.Response | None = None,
        error_type: str = "upstream_error",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body
        self.response = response
        self.error_type = error_type

    @property
    def retryable(self) -> bool:
        """False when retrying cannot possibly change the outcome."""
        return self.error_type in RETRYABLE_ERROR_TYPES

    @property
    def fallback_worthy(self) -> bool:
        """True when another model or provider could still serve this request."""
        return self.error_type in FALLBACK_ERROR_TYPES

    @property
    def server_status(self) -> int:
        # Rate limits keep 429 so the extension still recognises them; every
        # other upstream failure is reported as a bad gateway, like Groq's.
        return 429 if self.status_code == 429 else 502


def describe_failure(status_code: int | None, detail: str = "", error_type: str | None = None) -> tuple[str, str]:
    """Return (error_type, message) for a provider failure.

    The message states the actionable summary and always names the provider,
    the classification and the upstream status, so a caller that only forwards
    the text still reports enough to diagnose the failure.
    """
    error_type = error_type or classify_failure(status_code, detail)
    summary = ERROR_TYPE_SUMMARIES.get(error_type, "NVIDIA request failed")
    status = f" (HTTP {status_code})" if status_code is not None else ""
    text = f"{summary}{status} [provider=nvidia error_type={error_type}]"
    if detail and error_type not in _DETAIL_SUPPRESSED:
        text = f"{text}: {detail}"
    return error_type, text


def provider_failure(
    status_code: int | None,
    detail: str = "",
    *,
    error_type: str | None = None,
    body: Any = None,
    response: httpx.Response | None = None,
) -> NvidiaError:
    """Build a classified NvidiaError from an upstream status and message."""
    error_type, message = describe_failure(status_code, detail, error_type)
    return NvidiaError(status_code, message, body=body, response=response, error_type=error_type)


def build_request_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Adapt Groq-style request kwargs to NVIDIA's OpenAI-compatible body.

    Messages (including the redacted screenshot's image_url part), model,
    temperature and response_format pass through unchanged. Only parameter
    names differ, and the token cap value itself is untouched.
    """
    body = dict(kwargs)
    # NVIDIA's token cap parameter is max_tokens; the cap value is unchanged.
    if "max_completion_tokens" in body:
        body["max_tokens"] = body.pop("max_completion_tokens")
    # Groq's reasoning_effort has no OpenAI-compatible equivalent on NVIDIA and
    # is rejected as an unknown field, so it is dropped rather than translated.
    body.pop("reasoning_effort", None)
    return body


def message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


class NvidiaClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = NVIDIA_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        fallback_models: Sequence[str] = (),
    ):
        self._api_key = api_key
        # Tried in order when the requested model is reachable but unusable
        # (worker capacity exhausted, rate limited, not served to this key).
        self._fallback_models = tuple(dict.fromkeys(m for m in fallback_models if m))
        self._http = httpx.Client(
            base_url=base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_chat_completion)
        )

    def __repr__(self) -> str:
        return f"NvidiaClient(base_url={str(self._http.base_url)!r})"

    def _scrub(self, text: Any) -> str:
        value = str(text or "")
        return value.replace(self._api_key, "[API_KEY_REDACTED]") if self._api_key else value

    def create_chat_completion(self, **kwargs: Any) -> SimpleNamespace:
        """Send the request, moving to a fallback model if this one cannot serve it.

        The messages, including the redacted screenshot, are unchanged between
        attempts: only the model name differs, so a failover is invisible to
        the pipeline apart from the served_model it reports back.
        """
        requested = kwargs.get("model")
        candidates = [requested, *(m for m in self._fallback_models if m != requested)]
        last_failure: NvidiaError | None = None
        for index, candidate in enumerate(candidates):
            try:
                return self._attempt({**kwargs, "model": candidate})
            except NvidiaError as failure:
                last_failure = failure
                remaining = candidates[index + 1:]
                if not remaining or not failure.fallback_worthy:
                    raise
                logger.warning(
                    f"[AI-PROVIDER] provider=nvidia model={candidate} "
                    f"error_type={failure.error_type} falling_back_to={remaining[0]}"
                )
        raise last_failure  # unreachable: candidates always holds the requested model

    def _attempt(self, kwargs: dict[str, Any]) -> SimpleNamespace:
        model = kwargs.get("model")
        body = build_request_body(kwargs)
        try:
            response = self._http.post(
                "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException:
            # Not chained: the httpx request object carries the auth header.
            raise provider_failure(None, "request timed out", error_type="timeout") from None
        except httpx.HTTPError as exc:
            raise provider_failure(
                None, f"transport error ({type(exc).__name__})", error_type="unreachable"
            ) from None

        try:
            data = response.json()
        except ValueError:
            data = None

        error = data.get("error") if isinstance(data, dict) else None
        choices = data.get("choices") if isinstance(data, dict) else None

        if response.status_code >= 400 or (error and not choices):
            status_code = response.status_code
            if status_code < 400:
                code = error.get("code") if isinstance(error, dict) else None
                status_code = code if isinstance(code, int) and code >= 400 else 502
            if isinstance(error, dict):
                raw_detail = error.get("message")
            elif isinstance(data, dict) and data.get("detail"):
                # NVIDIA reports some gateway failures as a bare `detail` field.
                raw_detail = data.get("detail")
            else:
                raw_detail = response.text
            detail = self._scrub(raw_detail)[:MAX_ERROR_CHARS]
            failure = provider_failure(
                status_code,
                detail,
                body={"error": {"message": detail}},
                response=response,
            )
            # Classification only; the key and the full body stay out of the log.
            logger.warning(
                f"[AI-PROVIDER] provider=nvidia model={model} error_type={failure.error_type} "
                f"upstream_status={status_code} retryable={failure.retryable}"
            )
            raise failure from None

        if not choices:
            raise provider_failure(
                502, "response contained no choices", error_type="invalid_response"
            ) from None

        served_model = data.get("model") or model
        logger.info(f"[AI-PROVIDER] provider=nvidia served_model={served_model}")
        return SimpleNamespace(
            model=served_model,
            choices=[
                SimpleNamespace(
                    finish_reason=choice.get("finish_reason"),
                    message=SimpleNamespace(content=message_text(choice.get("message"))),
                )
                for choice in choices
                if isinstance(choice, dict)
            ],
        )
