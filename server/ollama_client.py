import json
import logging
import os
import re
from typing import Any

import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream")
MAX_OLLAMA_USER_PROMPT_CHARS = 3000
MAX_OLLAMA_ELEMENTS = 5


def repair_json(raw: str) -> str:
    """Best-effort repair of truncated or malformed JSON from small models."""
    s = raw.strip()
    if not s:
        raise ValueError("Empty response")

    # Find the first { to strip any preamble
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    s = s[start:]

    # Count open braces/brackets to find where it got cut off
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    last_valid = 0

    for i, ch in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        if depth_brace == 0 and depth_bracket == 0 and i > 0:
            last_valid = i + 1
            break

    if last_valid:
        return s[:last_valid]

    # JSON was truncated — close any open strings, brackets, braces
    if in_string:
        s += '"'
    while depth_bracket > 0:
        s += "]"
        depth_bracket -= 1
    while depth_brace > 0:
        s += "}"
        depth_brace -= 1
    return s


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

    # Moondream is a 1B model with limited context — use a condensed prompt
    ollama_system_prompt = (
        "You are a browser automation assistant. "
        "Given a sanitized screenshot and interactive elements, respond with "
        "ONLY a valid JSON object. "
        "For a single action: {\"action\": \"click\"|\"fill\"|\"scroll\"|\"wait\"|\"press\", "
        "\"selector\": \"<CSS selector>\", \"value\": \"<value or token>\", "
        "\"target\": \"<label>\", \"reasoning\": \"<one sentence>\", \"complete\": true|false}. "
        "For multiple actions: {\"reasoning\": \"<one sentence>\", \"actions\": [...]}. "
        "For PII fields use tokens: [TOKEN:USER_EMAIL], [TOKEN:USER_PASSWORD], "
        "[TOKEN:USER_PHONE], [TOKEN:USER_NAME]. "
        "Prefer [TOKEN:USER_EMAIL] for email, email-or-phone, or login identifier fields. "
        "Reserve [TOKEN:USER_NAME] only for fields explicitly asking for a username or user name. "
        "Do not infer USER_NAME from generic login/sign-in context. Set complete=true when done."
    )

    import re as _re
    task_match = _re.search(r"User Task:.*?(?=\n\n)", user_prompt, _re.DOTALL)
    plan_match = _re.search(r"Task Plan: (\[.*?\])", user_prompt, _re.DOTALL)
    history_match = _re.search(r"Completed Actions: (\[.*?\])", user_prompt, _re.DOTALL)
    failed_match = _re.search(r"Failed Action: (.*?)(?=\n\n)", user_prompt, _re.DOTALL)
    elements_match = _re.search(r"Interactive Elements: (\[.*?\])", user_prompt, _re.DOTALL)
    pii_match = _re.search(r"Redacted PII Fields: (\[.*?\])", user_prompt, _re.DOTALL)
    try:
        elements_list = json.loads(elements_match.group(1)) if elements_match else []
        pii_list = json.loads(pii_match.group(1)) if pii_match else []
        task_plan = json.loads(plan_match.group(1)) if plan_match else []
        action_history = json.loads(history_match.group(1)) if history_match else []
        failed_action = failed_match.group(1).strip() if failed_match else ""
        trimmed_elements = [
            {k: v for k, v in el.items() if k in ("selector", "frame_id", "tag", "type", "label", "reason")}
            for el in elements_list[:MAX_OLLAMA_ELEMENTS]
        ]
        trimmed_pii = [
            {k: v for k, v in p.items() if k in ("selector", "reason")}
            for p in pii_list
        ]
        task_line = task_match.group(0) if task_match else f"User Task: {user_prompt[:80]}"
        ollama_user_prompt = (
            f"{task_line}\n\n"
            f"Task Plan: {json.dumps(task_plan[:6])}\n\n"
            f"Completed Actions: {json.dumps(action_history[-6:])}\n\n"
            f"Failed Action: {failed_action[:160]}\n\n"
            f"Redacted PII Fields: {json.dumps(trimmed_pii)}\n\n"
            f"Interactive Elements: {json.dumps(trimmed_elements)}\n\n"
            "Continue from the current page. Do not repeat completed actions. "
            "Return the optimal remaining action as JSON."
        )
    except Exception:
        ollama_user_prompt = user_prompt[:MAX_OLLAMA_USER_PROMPT_CHARS]

    logger.info("Resolved Ollama model name: %s", model)
    payload = {
        "model": model,
        "prompt": (
            f"{ollama_system_prompt}\n\n{ollama_user_prompt}\n\n"
            "Output only one complete JSON object. Do not explain the JSON, use markdown, "
            "or leave any string unfinished."
        ),
        "images": [clean_b64],
        "stream": False,
        # Leave headroom for image tokens and the schema-constrained prompt.
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            log_payload = {**payload, "images": ["<base64 image data omitted>"]}
            logger.info("Ollama request payload: %s", json.dumps(log_payload, ensure_ascii=False))
            logger.info("Final Ollama payload model: %s", payload["model"])
            clean_b64_preview = clean_b64[:80] if clean_b64 else "EMPTY"
            clean_b64_len = len(clean_b64) if clean_b64 else 0
            logger.info(f"[OLLAMA-DEBUG] Image b64 length: {clean_b64_len}, prefix: {clean_b64_preview}")
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
            try:
                repaired = repair_json(response_text)
                if repaired != response_text:
                    logger.warning(f"[OLLAMA-REPAIR] Repaired truncated JSON response")
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                logger.error(f"[OLLAMA-JSON-FAIL] Original: {response_text}")
                logger.error(f"[OLLAMA-JSON-FAIL] Repaired: {repaired if 'repaired' in dir() else 'N/A'}")
                logger.error(f"[OLLAMA-REPAIR-RESULT] Full repaired string: {repaired}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama request failed: {type(exc).__name__}: {exc}"
                ) from exc
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
        if exc.response.status_code == 400:
            logger.error(f"[OLLAMA-400] Full error response body: {exc.response.text[:2000]}")
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
