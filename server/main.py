import base64
import json
import logging
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

# Keep the model configurable. Qwen 3.6 27B is Groq's current production
# multimodal model and is available for image + JSON requests.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2-vl:7b")
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB max image payload
MAX_INTERACTIVE_ELEMENTS = 20
MAX_CONTEXT_STRING_CHARS = 120

# Setup Zero-PII Logger
class ZeroPiiLoggingFilter(logging.Filter):
    """Filter to ensure base64 image strings, credentials, and raw PII never enter server logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            # Redact data URLs & Base64 blocks
            record.msg = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{20,}", "[IMAGE_DATA_REDACTED]", record.msg)
            # Redact bearer tokens and keys
            record.msg = re.sub(r"(gsk_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_]{20,})", "[API_KEY_REDACTED]", record.msg)
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrivacyVisionServer")
logger.addFilter(ZeroPiiLoggingFilter())

AgentActionName = Literal["click", "fill", "scroll", "wait", "press"]
ALLOWED_ACTIONS = {"click", "fill", "scroll", "wait", "press"}


class PiiField(BaseModel):
    selector: str
    rect: dict[str, Any]
    reason: str


class PageElement(BaseModel):
    selector: str
    tag: str
    role: str | None = None
    type: str | None = None
    label: str | None = None
    placeholder: str | None = None
    name: str | None = None
    reason: str | None = None
    rect: dict[str, Any] | None = None


class AnalyzeRequest(BaseModel):
    image: str = Field(..., description="Base64 data URL of the redacted screenshot")
    pii_fields: list[PiiField] = Field(default_factory=list)
    elements: list[PageElement] = Field(default_factory=list)
    task: str


class AgentAction(BaseModel):
    action: AgentActionName
    selector: str
    value: str | None = None
    target: str | None = None
    reasoning: str = ""


class AnalyzeResponse(BaseModel):
    action: AgentActionName | None = None
    selector: str | None = None
    value: str | None = None
    target: str | None = None
    reasoning: str = ""
    complete: bool = False
    actions: list[AgentAction] = Field(default_factory=list)


app = FastAPI(title="Privacy Vision Agent Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "Privacy Vision Agent Server",
        "version": "1.0.0",
        "groq_model": GROQ_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
        "zero_pii_logging": True
    }


@app.get("/providers")
async def list_providers():
    return {
        "providers": [
            {"id": "groq", "name": f"Groq Cloud VLM ({GROQ_MODEL})", "type": "cloud"},
            {"id": "ollama", "name": f"Ollama Offline VLM ({OLLAMA_MODEL})", "type": "local"}
        ]
    }


def parse_image_data_url(image: str) -> tuple[bytes, str]:
    if not image or not image.startswith("data:") or "," not in image:
        raise HTTPException(
            status_code=400,
            detail="Fail-Closed Error: image must be a valid base64 data URL (e.g. data:image/png;base64,...)",
        )

    header, encoded = image.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Fail-Closed Error: image data URL must be base64 encoded")

    mime_type = header.removeprefix("data:").split(";", 1)[0]
    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fail-Closed Error: unsupported non-image MIME type")

    # Enforce base64 length limit before decoding (~13.3MB base64 string for 10MB raw)
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 1024):
        raise HTTPException(
            status_code=413,
            detail=f"Fail-Closed Error: Image payload exceeds maximum size limit of {MAX_IMAGE_BYTES // (1024 * 1024)}MB."
        )

    try:
        decoded = base64.b64decode(encoded, validate=True)
        if len(decoded) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Fail-Closed Error: Decoded image exceeds maximum 10MB limit.")
        return decoded, mime_type
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Fail-Closed Error: Image contains invalid base64 data") from exc


def build_system_prompt() -> str:
    return (
        "You are a careful browser automation planner operating on anonymized visual context. "
        "Sensitive user PII (passwords, emails, faces) have been redacted and masked locally by the client. "
        "Given a sanitized screenshot, interactive DOM elements, and the user's task, "
        "respond with ONLY valid JSON matching the requested action schema. "
        'Allowed actions: "click", "fill", "scroll", "wait", "press". '
        'For one-step tasks, use format: {"action": "click"|"fill"|"scroll"|"wait"|"press", '
        '"selector": "<CSS selector>", "value": "<text to fill or token ID>", '
        '"target": "<visible label>", "reasoning": "<explanation>", "complete": true|false}. '
        'For multi-step tasks, use format: {"reasoning": "<explanation>", "actions": ['
        '{"action": "click"|"fill"|"scroll"|"wait"|"press", "selector": "<CSS selector>", '
        '"value": "<text to fill or token ID>", "target": "<visible label>", "reasoning": "<step explanation>"}]}. '
        "Never invent arbitrary selectors when elements contains valid DOM selectors. "
        "The user's task wording may differ from an element's exact visible label (e.g. the task says 'login' but the button says 'Log in', or 'sign up' vs 'Register') — match the task to the closest semantically equivalent element in the Interactive Elements list rather than requiring exact text matches. "
        "If the task only requires this one action to be fully satisfied, set complete=true in this same response — do not default to false or omit this field. "
        "If a field is marked as a redacted PII field, use token identifiers like [TOKEN:USER_EMAIL] in the value field so the client can safely fill it locally. "
        "Set complete=true when the task is fully satisfied."
    )


def build_user_prompt(request: AnalyzeRequest) -> str:
    def compact(value: Any) -> Any:
        if isinstance(value, str):
            return value[:MAX_CONTEXT_STRING_CHARS]
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    pii_context = [
        compact({"selector": field.selector, "reason": field.reason})
        for field in request.pii_fields
    ]
    elements = [
        compact({
            key: value
            for key, value in element.model_dump(exclude_none=True).items()
            if key != "rect"
        })
        # Defense in depth: bound DOM controls even if an older/future client
        # sends more than the client-side network cap.
        for element in request.elements[:MAX_INTERACTIVE_ELEMENTS]
    ]

    return (
        f"User Task: {request.task[:MAX_CONTEXT_STRING_CHARS]}\n\n"
        f"Redacted PII Fields: {json.dumps(pii_context, ensure_ascii=True)}\n\n"
        f"Interactive Elements: {json.dumps(elements, ensure_ascii=True)}\n\n"
        "Return the optimal action steps as JSON."
    )


def generate_groq_action(request: AnalyzeRequest) -> AnalyzeResponse:
    parse_image_data_url(request.image)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on server.",
        )

    client = Groq(api_key=api_key)
    request_messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(request)},
                {"type": "image_url", "image_url": {"url": request.image}},
            ],
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=request_messages,
            response_format={"type": "json_object"},
            max_completion_tokens=512,
        )
        response_text = completion.choices[0].message.content
        payload = json.loads(response_text)
        return parse_analyze_response(payload)
    except Exception as exc:
        logger.error(f"Groq API call failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Groq VLM Provider Error: {exc}") from exc


def parse_analyze_response(payload: dict[str, Any]) -> AnalyzeResponse:
    response = AnalyzeResponse.model_validate(payload)
    if not response.actions and response.action and response.selector:
        response.actions = [
            AgentAction(
                action=response.action,
                selector=response.selector,
                value=response.value,
                target=response.target,
                reasoning=response.reasoning,
            )
        ]

    # Validate action allowlist
    valid_actions = []
    for act in response.actions:
        if act.action not in ALLOWED_ACTIONS:
            logger.warning(f"Rejecting unallowed action type: {act.action}")
            continue
        # Filter dangerous script patterns in selectors
        if re.search(r"(javascript:|script|eval\(|window\.|onerror|onload)", act.selector, re.I):
            logger.warning(f"Rejecting dangerous selector: {act.selector}")
            continue
        valid_actions.append(act)

    response.actions = valid_actions

    if not response.actions and not response.complete:
        raise ValueError("Response must include at least one valid action or set complete=true")

    return response


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    parse_image_data_url(request.image)
    logger.info("Processing analyze request for task")
    return await run_in_threadpool(generate_groq_action, request)
