import json
import logging
import os
from typing import Any

import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream")
MAX_OLLAMA_USER_PROMPT_CHARS = 3000


async def generate_ollama_action(system_prompt: str, user_prompt: str, image_b64: str, model_name: str | None = None) -> dict[str, Any]:
    """
    Sends multimodal vision prompt to local Ollama server instance.
    Returns parsed JSON dictionary response matching agent action schema.
    """
    model = model_name or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    url = f"{OLLAMA_HOST}/api/generate"

    # Strip data URL header if present
    if "," in image_b64:
        clean_b64 = image_b64.split(",", 1)[1]
    else:
        clean_b64 = image_b64

    logger.info("Resolved Ollama model name: %s", model)
    payload = {
        "model": model,
        "prompt": (
            f"{system_prompt}\n\n{user_prompt[:MAX_OLLAMA_USER_PROMPT_CHARS]}\n\n"
            "Output only one complete JSON object. Do not explain the JSON, use markdown, "
            "or leave any string unfinished."
        ),
        "images": [clean_b64],
        "stream": False,
        # Leave headroom for image tokens and the schema-constrained prompt.
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 256},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            log_payload = {**payload, "images": ["<base64 image data omitted>"]}
            logger.info("Ollama request payload: %s", json.dumps(log_payload, ensure_ascii=False))
            logger.info("Final Ollama payload model: %s", payload["model"])
            res = await client.post(url, json=payload)
            response_body = res.text
            logger.info("Ollama raw response body (HTTP %s): %s", res.status_code, response_body)
            if "model not found" in response_body.lower():
                try:
                    raise RuntimeError(f"Ollama response indicates model not found: {response_body[:1000]}")
                except RuntimeError as exc:
                    logger.exception("Ollama model %r was not found", model)
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{model}' not found in Ollama — run `ollama pull {model}`",
                    ) from exc
            res.raise_for_status()
            data = res.json()
            response_text = data.get("response", "")
            logger.info("Ollama raw response field: %s", response_text)
            return json.loads(response_text)
    except httpx.ConnectError as exc:
        logger.exception("Ollama is not running; request to %s could not connect", url)
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running — start it with `ollama serve`",
        ) from exc
    except httpx.TimeoutException as exc:
        logger.exception("Ollama request to %s timed out", url)
        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama request timed out — the model may be too slow for this hardware "
                "or the image payload too large"
            ),
        ) from exc
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        response_body = exc.response.text
        if exc.response.status_code == 404 or "model not found" in response_body.lower():
            logger.exception("Ollama model %r was not found", model)
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' not found in Ollama — run `ollama pull {model}`",
            ) from exc

        if "exceed_context_size_error" in response_body:
            n_prompt_tokens = None
            try:
                error_payload = json.loads(response_body)
                nested_error = error_payload.get("error")
                if isinstance(nested_error, str):
                    nested_error = json.loads(nested_error)
                if isinstance(nested_error, dict):
                    n_prompt_tokens = nested_error.get("n_prompt_tokens")
                    if n_prompt_tokens is None and isinstance(nested_error.get("error"), dict):
                        n_prompt_tokens = nested_error["error"].get("n_prompt_tokens")
            except (TypeError, ValueError):
                pass
            logger.warning(
                "Ollama context limit exceeded; reported n_prompt_tokens=%s",
                n_prompt_tokens,
            )

        logger.exception(
            "Ollama returned unexpected HTTP %s: %s",
            exc.response.status_code,
            response_body[:1000],
        )
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {str(exc)}") from exc
    except Exception as exc:
        logger.exception("Unexpected Ollama request failure to %s", url)
        error_message = str(exc).strip() or repr(exc)
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {type(exc).__name__}: {error_message}",
        ) from exc
