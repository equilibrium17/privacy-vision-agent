import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from server.main import AnalyzeRequest, generate_groq_action


class FakeCompletion:
    def __init__(self, payload: str):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": payload})()})()]


class FakeChatCompletions:
    def __init__(self, payload: str):
        self.payload = payload

    def create(self, **kwargs):
        return FakeCompletion(self.payload)


class FakeGroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
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


if __name__ == "__main__":
    unittest.main()
