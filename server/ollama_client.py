import json
import logging
import os
import httpx
from typing import Any

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2-vl:7b")


async def generate_ollama_action(system_prompt: str, user_prompt: str, image_b64: str, model_name: str | None = None) -> dict[str, Any]:
    """
    Sends multimodal vision prompt to local Ollama server instance.
    Returns parsed JSON dictionary response matching agent action schema.
    """
    model = model_name or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_HOST}/api/generate"

    # Strip data URL header if present
    if "," in image_b64:
        clean_b64 = image_b64.split(",", 1)[1]
    else:
        clean_b64 = image_b64

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "images": [clean_b64],
        "stream": False,
        "format": "json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            response_text = data.get("response", "")
            return json.loads(response_text)
    except Exception as exc:
        logger.warning(f"Ollama request to {url} failed ({exc}). Returning fallback action structure.")
        return {
            "reasoning": "Ollama local model unreachable or offline; fallback response generated.",
            "actions": [
                {
                    "action": "wait",
                    "selector": "body",
                    "reasoning": "Local offline VLM service pending startup."
                }
            ]
        }
