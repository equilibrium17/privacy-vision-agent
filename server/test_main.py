import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from server.main import GROQ_MODEL, AnalyzeRequest, build_plan_prompt, build_verify_prompt, build_user_prompt, decode_model_json, describe_element, generate_groq_action, generate_groq_verification, parse_analyze_response, salvage_json_payload


class FakeCompletion:
    def __init__(self, payload: str, finish_reason: str = "stop"):
        self.choices = [type("Choice", (), {
            "message": type("Message", (), {"content": payload})(),
            "finish_reason": finish_reason,
        })()]


class FakeChatCompletions:
    def __init__(self, payload: str):
        self.payload = payload

    def create(self, **kwargs):
        return FakeCompletion(self.payload)


class FakeGroqClient:
    def __init__(self, api_key: str, max_retries: int = 2):
        self.api_key = api_key
        self.max_retries = max_retries
        self.chat = type("Chat", (), {"completions": FakeChatCompletions('{"reasoning":"done","complete":true}')})()


class GenerateGroqActionTests(unittest.TestCase):
    def test_invalid_image_is_rejected_before_groq_configuration(self):
        request = AnalyzeRequest(image="not-a-data-url", pii_fields=[], task="test")

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with self.assertRaises(HTTPException) as raised:
                generate_groq_action(request)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("base64 data URL", raised.exception.detail)

    def test_complete_response_without_actions_is_allowed(self):
        request = AnalyzeRequest(image="data:image/png;base64,AA==", pii_fields=[], task="test")

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            with patch("server.main.Groq", FakeGroqClient):
                response = generate_groq_action(request)

        self.assertTrue(response.complete)
        self.assertEqual(response.actions, [])

    def test_groq_client_disables_sdk_retries(self):
        request = AnalyzeRequest(image="data:image/png;base64,AA==", pii_fields=[], task="test")

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            with patch("server.main.Groq", FakeGroqClient):
                from server.main import create_groq_client

                client = create_groq_client("test-key")

        self.assertEqual(client.max_retries, 0)

    def test_user_prompt_includes_bounded_execution_context(self):
        request = AnalyzeRequest(
            image="data:image/png;base64,AA==",
            pii_fields=[],
            task="Complete the task",
            task_plan=["Fill the form", "Submit it"],
            action_history=[{"action": "fill", "selector": "#email", "target": "Email"}],
            failed_action="click #continue failed",
        )

        prompt = build_user_prompt(request)

        self.assertIn('Task Plan: ["Fill the form", "Submit it"]', prompt)
        self.assertIn('Completed Actions: [{"action": "fill", "target": "Email", "selector": "#email"}]', prompt)
        self.assertIn('Failed Action: "click #continue failed"', prompt)

    def test_decode_model_json_accepts_fenced_and_list_responses(self):
        self.assertEqual(decode_model_json('```json\n[{"action":"click"}]\n```'), [{"action": "click"}])

    def test_post_navigation_round_uses_new_page_elements(self):
        """Page 1 (username/password/submit) -> navigation -> Page 2 (logout).

        The second round must be built from the NEW page's elements, and the
        logout action must use the real selector from that list.
        """
        page_one = [
            {"selector": "#username", "tag": "input", "type": "text", "label": "Username"},
            {"selector": "#password", "tag": "input", "type": "password", "label": "Password"},
            {"selector": "#submit", "tag": "button", "label": "Submit"},
        ]
        page_two = [{"selector": "#logout", "tag": "button", "label": "Log out"}]
        task = "fill the credentials and then click on submit then click on logout"

        round_one = build_user_prompt(AnalyzeRequest(
            image="data:image/png;base64,AA==",
            task=task,
            elements=page_one,
        ))
        self.assertIn("#submit", round_one)
        self.assertNotIn("#logout", round_one)

        round_two = build_user_prompt(AnalyzeRequest(
            image="data:image/png;base64,AA==",
            task=task,
            elements=page_two,
            action_history=[
                {"action": "fill", "selector": "#username", "target": "Username"},
                {"action": "fill", "selector": "#password", "target": "Password"},
                {"action": "click", "selector": "#submit", "target": "Submit"},
            ],
        ))
        # Page 1's controls must not leak into the post-navigation round.
        self.assertIn("#logout", round_two)
        self.assertNotIn('"selector": "#username"', round_two.split("Interactive Elements:")[1])
        self.assertNotIn('"selector": "#submit"', round_two.split("Interactive Elements:")[1])

        # The real selector from the new page survives action validation.
        response = parse_analyze_response({
            "actions": [{"action": "click", "selector": "#logout", "target": "Log out"}]
        })
        self.assertEqual(response.actions[0].selector, "#logout")

    def test_truncated_action_response_is_salvaged_not_retried(self):
        """A response cut off mid-explanation still yields the first action."""
        truncated = (
            '{"action": "click", "selector": "#logout", "target": "Log out", '
            '"reasoning": "clicking logout"} I could not find the element so let me '
            'explain at length why this page does not contain'
        )
        salvaged = salvage_json_payload(truncated)
        response = parse_analyze_response(salvaged)

        self.assertEqual(response.actions[0].action, "click")
        self.assertEqual(response.actions[0].selector, "#logout")

    def test_salvage_returns_none_for_unrecoverable_prose(self):
        self.assertIsNone(salvage_json_payload("I am unable to locate the logout button."))
        self.assertIsNone(salvage_json_payload(""))

    def test_missing_element_escape_hatch_is_a_valid_action(self):
        """The prompt's fallback for an absent element must parse cleanly."""
        response = parse_analyze_response({
            "action": "wait",
            "selector": "body",
            "reasoning": "target element not on this page yet",
            "complete": False,
        })

        self.assertEqual(response.actions[0].action, "wait")
        self.assertFalse(response.complete)

    def test_frame_id_survives_request_and_action_validation(self):
        request = AnalyzeRequest(
            image="data:image/png;base64,AA==",
            pii_fields=[],
            task="Click continue",
            elements=[{"selector": "#continue", "tag": "button", "frame_id": 7}],
        )
        response = parse_analyze_response({
            "actions": [{"action": "click", "selector": "#continue", "frame_id": 7}]
        })

        self.assertEqual(request.elements[0].frame_id, 7)
        self.assertEqual(response.actions[0].frame_id, 7)

    def test_plan_progress_is_numbered_and_points_at_next_step(self):
        request = AnalyzeRequest(
            image="data:image/png;base64,AA==",
            task="Search, open product, add to cart",
            task_plan=["Search", "Open product", "Add to cart"],
            completed_steps=["Search"],
            remaining_steps=["Open product", "Add to cart"],
            last_action_result="executed fill Search for step 1",
            current_url="https://shop.test/search",
        )

        prompt = build_user_prompt(request)

        self.assertIn('Completed Steps: ["1. Search"]', prompt)
        self.assertIn('Next Incomplete Step: "2. Open product"', prompt)
        self.assertIn('Remaining Steps: ["2. Open product", "3. Add to cart"]', prompt)
        self.assertIn('Current URL: "https://shop.test/search"', prompt)
        self.assertIn("done_through", prompt)
        # Existing sections are unchanged.
        self.assertIn('Task Plan: ["Search", "Open product", "Add to cart"]', prompt)

    def test_prompt_without_progress_keeps_legacy_closing(self):
        prompt = build_user_prompt(AnalyzeRequest(image="data:image/png;base64,AA==", task="Click continue"))

        self.assertNotIn("Plan Progress", prompt)
        self.assertIn("Return only the remaining optimal action steps as JSON.", prompt)

    def test_replan_prompt_asks_only_for_remaining_work(self):
        prompt = build_user_prompt(
            AnalyzeRequest(
                image="data:image/png;base64,AA==",
                task="Search then open product",
                completed_steps=["Search"],
                remaining_steps=["Open product"],
            ),
            for_plan=True,
        )

        self.assertIn("Plan only the work that remains", prompt)
        self.assertNotIn("done_through", prompt)

    def test_replan_and_progress_only_responses_are_not_coerced_to_complete(self):
        replan = parse_analyze_response({"reasoning": "not on this page", "replan": True})
        self.assertTrue(replan.replan)
        self.assertFalse(replan.complete)
        self.assertEqual(replan.actions, [])

        progress = parse_analyze_response({"reasoning": "all planned steps visible", "done_through": 3})
        self.assertEqual(progress.done_through, 3)
        self.assertFalse(progress.complete)

        escape_hatch = parse_analyze_response({
            "action": "wait", "selector": "body", "reasoning": "not here", "replan": True, "complete": False,
        })
        self.assertTrue(escape_hatch.replan)
        self.assertEqual(escape_hatch.actions[0].action, "wait")

    # --- final verification: fresh screenshot + cited DOM evidence ---------------

    VERIFY_IMAGE = "data:image/png;base64,AA=="

    def _run_verification(self, request, verdicts):
        """Run the real verifier against a fake Groq client returning `verdicts` in order."""
        calls = []
        queue = list(verdicts)

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                verdict = queue.pop(0)
                if isinstance(verdict, Exception):
                    raise verdict
                return FakeCompletion(json.dumps(verdict))

        client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), patch("server.main.create_groq_client", return_value=client):
            return generate_groq_verification(request), calls

    def _cart_request(self, evidence, completed_steps=None, image=VERIFY_IMAGE):
        return AnalyzeRequest(
            image=image,
            task="Open the cart and set the quantity to 2",
            verify=True,
            page_context="Page title: Products",
            completed_steps=completed_steps or [],
            verification_evidence=evidence,
        )

    def test_verify_cart_drawer_open_with_quantity_two_succeeds(self):
        request = self._cart_request([
            {"id": "E1", "kind": "url", "text": "Current URL: https://shop.test/products"},
            {"id": "E2", "kind": "open-overlay", "text": 'Open overlay #cart-drawer role=dialog "Your cart"'},
            {"id": "E3", "kind": "numeric-control", "text": 'Input #quantity-1 "Quantity" = 2'},
            {"id": "E4", "kind": "action-control", "text": 'Visible button #proceed-to-buy "Proceed to buy"'},
        ], completed_steps=["Open cart", "Set quantity to 2"])

        result, calls = self._run_verification(request, [
            {"success": True, "reason": "cart drawer open with quantity 2", "evidence": ["E2", "E3"]},
        ])

        self.assertTrue(result.success)
        # The verifier saw the fresh screenshot AND the structured evidence.
        self.assertEqual(calls[0]["model"], GROQ_MODEL)
        content = calls[0]["messages"][1]["content"]
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": self.VERIFY_IMAGE}})
        self.assertIn('E2 [open-overlay] Open overlay #cart-drawer role=dialog "Your cart"', content[0]["text"])
        self.assertIn('E3 [numeric-control] Input #quantity-1 "Quantity" = 2', content[0]["text"])
        self.assertIn("Current URL: https://shop.test/products", content[0]["text"])
        self.assertEqual(calls[0]["max_completion_tokens"], 150)

    def test_verify_cart_not_open_fails(self):
        closed = [
            {"id": "E1", "kind": "url", "text": "Current URL: https://shop.test/products"},
            {"id": "E2", "kind": "overlay-state", "text": "No open dialog, drawer or modal is visible"},
            {"id": "E3", "kind": "action-control", "text": 'Visible button #add-to-cart-1 "Add to cart"'},
        ]

        result, _calls = self._run_verification(self._cart_request(closed), [
            {"success": False, "reason": "no cart is open", "evidence": []},
        ])
        self.assertFalse(result.success)

        # A success that cites evidence which does not exist on the page is not success.
        result, _calls = self._run_verification(self._cart_request(closed), [
            {"success": True, "reason": "cart open", "evidence": ["E9"]},
        ])
        self.assertFalse(result.success)

    def test_verify_history_says_cart_opened_but_dom_shows_closed_fails(self):
        request = self._cart_request([
            {"id": "E1", "kind": "url", "text": "Current URL: https://shop.test/products"},
            {"id": "E2", "kind": "overlay-state", "text": "No open dialog, drawer or modal is visible"},
        ], completed_steps=["Open the cart", "Increase quantity to 2"])

        # A verdict resting on the step history alone.
        result, calls = self._run_verification(request, [
            {"success": True, "reason": "history shows the cart was opened", "evidence": []},
        ])
        self.assertFalse(result.success)
        self.assertIn("current page evidence", result.reason)

        # Citing a claimed step instead of an evidence id is also rejected.
        result, _calls = self._run_verification(request, [
            {"success": True, "reason": "step done", "evidence": ["Open the cart"]},
        ])
        self.assertFalse(result.success)

        # Claims are labelled as claims in what the verifier receives.
        text = calls[0]["messages"][1]["content"][0]["text"]
        self.assertIn('Completed plan steps (claims, not proof): ["Open the cart", "Increase quantity to 2"]', text)
        self.assertIn("E2 [overlay-state] No open dialog, drawer or modal is visible", text)

    def test_verify_without_screenshot_keeps_text_model(self):
        request = self._cart_request([], image="")
        result, calls = self._run_verification(request, [{"success": False, "reason": "not done"}])

        self.assertFalse(result.success)
        self.assertEqual(calls[0]["model"], "groq/compound-mini")
        self.assertIsInstance(calls[0]["messages"][1]["content"], str)

    def test_verify_image_failure_falls_back_to_dom_evidence(self):
        request = self._cart_request([
            {"id": "E1", "kind": "open-overlay", "text": 'Open overlay #cart-drawer role=dialog "Your cart"'},
        ])
        result, calls = self._run_verification(request, [
            RuntimeError("image rejected"),
            {"success": True, "reason": "drawer open", "evidence": ["E1"]},
        ])

        self.assertTrue(result.success)
        self.assertEqual(calls[0]["model"], GROQ_MODEL)
        self.assertEqual(calls[1]["model"], "groq/compound-mini")


class ElementDescriptionTests(unittest.TestCase):
    """Each control reaches the model under exactly one name."""

    def element(self, **fields):
        return AnalyzeRequest(image="", task="t", elements=[fields]).elements[0]

    def test_a_placeholder_that_repeats_the_label_is_dropped(self):
        described = describe_element(self.element(
            selector="#username", tag="input", label="Username", placeholder="Username", name="username"
        ))

        self.assertEqual(described["label"], "Username")
        self.assertNotIn("placeholder", described)
        self.assertNotIn("name", described)

    def test_case_and_spacing_do_not_hide_the_repeat(self):
        described = describe_element(self.element(
            selector="#pw", tag="input", label="Password", placeholder=" password "
        ))

        self.assertNotIn("placeholder", described)

    def test_wording_the_label_does_not_carry_is_kept(self):
        described = describe_element(self.element(
            selector="#user", tag="input", label="Username", placeholder="you@example.com", name="login_id"
        ))

        self.assertEqual(described["placeholder"], "you@example.com")
        self.assertEqual(described["name"], "login_id")

    def test_geometry_never_reaches_the_model(self):
        described = describe_element(self.element(
            selector="#buy", tag="button", label="Buy", rect={"x": 1, "y": 2}
        ))

        self.assertNotIn("rect", described)

    def test_the_prompt_lists_a_repeated_name_only_once(self):
        request = AnalyzeRequest(image="", task="Log in", elements=[
            {"selector": "#username", "tag": "input", "label": "Username", "placeholder": "Username"}
        ])

        self.assertEqual(build_user_prompt(request, for_plan=True).count("Username"), 1)


class ProseActionTests(unittest.TestCase):
    """Plan text a model leaves in "actions" must not cost a whole round."""

    def test_string_entries_are_dropped_and_the_real_action_survives(self):
        response = parse_analyze_response({
            "action": "fill",
            "selector": "#username",
            "value": "rahul",
            "reasoning": "enter username",
            "actions": [
                {"action": "fill", "selector": "#username", "value": "rahul", "reasoning": "enter username"},
                "2. Enter Password",
                "3. Click Submit",
            ],
        })

        self.assertEqual([action.selector for action in response.actions], ["#username"])

    def test_an_all_prose_actions_list_falls_back_to_the_top_level_action(self):
        response = parse_analyze_response({
            "action": "click",
            "selector": "#submit",
            "reasoning": "submit the form",
            "actions": ["2. Enter Password"],
        })

        self.assertEqual([action.selector for action in response.actions], ["#submit"])


class TargetMismatchTests(unittest.TestCase):
    """The element list decides which selector wears which label."""

    MENU = [
        {"selector": "#menu-item-20 a", "tag": "a", "label": "Practice"},
        {"selector": "#menu-item-21 a", "tag": "a", "label": "Courses"},
        {"selector": "#submit", "tag": "button", "label": "Submit"},
    ]

    def elements(self, raw=None):
        return AnalyzeRequest(image="", task="t", elements=raw or self.MENU).elements

    def test_a_selector_that_belongs_to_another_label_is_corrected(self):
        response = parse_analyze_response(
            {"action": "click", "selector": "#menu-item-21 a", "target": "Practice", "reasoning": "navigate"},
            self.elements(),
        )

        self.assertEqual(response.actions[0].selector, "#menu-item-20 a")

    def test_a_matching_selector_is_left_alone(self):
        response = parse_analyze_response(
            {"action": "click", "selector": "#menu-item-20 a", "target": "Practice", "reasoning": "navigate"},
            self.elements(),
        )

        self.assertEqual(response.actions[0].selector, "#menu-item-20 a")

    def test_an_ambiguous_target_is_left_to_the_model(self):
        twins = self.MENU + [{"selector": "#footer-practice a", "tag": "a", "label": "Practice"}]
        response = parse_analyze_response(
            {"action": "click", "selector": "#menu-item-21 a", "target": "Practice", "reasoning": "navigate"},
            self.elements(twins),
        )

        self.assertEqual(response.actions[0].selector, "#menu-item-21 a")

    def test_a_partial_name_still_counts_as_a_match(self):
        response = parse_analyze_response(
            {"action": "click", "selector": "#submit", "target": "Submit button", "reasoning": "submit"},
            self.elements(),
        )

        self.assertEqual(response.actions[0].selector, "#submit")

    def test_an_unlisted_selector_is_not_second_guessed(self):
        response = parse_analyze_response(
            {"action": "click", "selector": "#not-on-page", "target": "Practice", "reasoning": "navigate"},
            self.elements(),
        )

        self.assertEqual(response.actions[0].selector, "#not-on-page")

    def test_without_an_element_list_nothing_is_rewritten(self):
        response = parse_analyze_response(
            {"action": "click", "selector": "#menu-item-21 a", "target": "Practice", "reasoning": "navigate"}
        )

        self.assertEqual(response.actions[0].selector, "#menu-item-21 a")


class VerifyPromptTests(unittest.TestCase):
    def test_the_verifier_judges_the_outcome_not_the_vanished_controls(self):
        prompt = build_verify_prompt()

        self.assertIn("Judge the task's RESULT", prompt)
        self.assertIn("gone because the action succeeded is evidence of success", prompt)
        self.assertIn("still shows the work undone", prompt)


class PlanPromptTests(unittest.TestCase):
    def test_the_planner_is_told_not_to_invent_navigation_or_join_names(self):
        prompt = build_plan_prompt()

        self.assertIn("Name each target exactly once", prompt)
        self.assertIn("never join an element's label", prompt)
        self.assertIn("only because it appears in the list", prompt)
        self.assertIn("Plan only the steps the task", prompt)

    def test_the_planner_may_not_invent_credentials_or_split_a_field_in_two(self):
        prompt = build_plan_prompt()

        self.assertIn("names the field to fill, never the data to type", prompt)
        self.assertIn("Never invent sample values", prompt)
        self.assertIn("filling a field is one step", prompt)


if __name__ == "__main__":
    unittest.main()
