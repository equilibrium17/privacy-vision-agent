import base64
import json
import logging
import os
import re
import time
from contextvars import ContextVar
from collections.abc import Callable
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from server.ollama_client import generate_ollama_action
from server.nvidia_client import NvidiaClient, NvidiaError, provider_failure

# Keep the model configurable. Qwen 3.8 27B is Groq's current production
# multimodal model and is available for image + JSON requests. (Its
# predecessor qwen3.6 has been retired and now 404s as model_not_found.)
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
# Groq's lightweight text model for planning and screenshot-free verification.
GROQ_TEXT_MODEL = "groq/compound-mini"
# Cloud provider for plan/action/verify requests, chosen by AI_PROVIDER (default groq).
AI_PROVIDERS = ("groq", "nvidia")
# Vision models this account can fall back to when the configured NVIDIA model
# is reachable but cannot serve the request (worker capacity exhausted, rate
# limited, not served to this key). Override with NVIDIA_FALLBACK_MODELS.
DEFAULT_NVIDIA_FALLBACK_MODELS = ("meta/llama-3.2-11b-vision-instruct",)
# Provider order tried after the selected one when a request fails for
# infrastructure reasons. Set AI_FALLBACK=0 to report the first failure instead.
# Ollama is deliberately absent: a small local model answers slowly (a minute
# is normal) and poorly, so falling back to it costs more than the cloud
# failure it was meant to rescue. Set AI_FALLBACK_OLLAMA=1 to include it.
FALLBACK_PROVIDER_ORDER = ("nvidia", "groq")
# Upstream statuses that mean "this provider is unwell", not "this request is bad".
FALLBACK_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Waiting out a rate limit is only worth it when nothing else can serve the
# request. A longer wait than this costs the agent more than switching provider.
MAX_RATE_LIMIT_WAIT_SECONDS = 3.0
# The providers analyze() can still try after the one running right now. Set per
# attempt and read by the generation loops, which decide whether a rate-limit
# wait is worth it. anyio copies the context into the worker thread.
remaining_providers: ContextVar[tuple[str, ...]] = ContextVar("remaining_providers", default=())
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream")
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB max image payload
MAX_INTERACTIVE_ELEMENTS = 20
MAX_CONTEXT_STRING_CHARS = 120
MAX_VERIFY_EVIDENCE_ITEMS = 40
MAX_VERIFY_EVIDENCE_CHARS = 160

# Setup Zero-PII Logger
class ZeroPiiLoggingFilter(logging.Filter):
    """Filter to ensure base64 image strings, credentials, and raw PII never enter server logs."""

    @staticmethod
    def _scrub(value: str) -> str:
        # Redact data URLs & Base64 blocks
        value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{20,}", "[IMAGE_DATA_REDACTED]", value)
        # Redact bearer tokens and keys (Groq gsk_..., NVIDIA nvapi-...,
        # and OpenAI-style sk-... credentials).
        return re.sub(
            r"(gsk_[A-Za-z0-9_]{20,}|nvapi-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})",
            "[API_KEY_REDACTED]",
            value,
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        # %s-style arguments bypass record.msg entirely, so scrub them too.
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._scrub(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                key: self._scrub(arg) if isinstance(arg, str) else arg
                for key, arg in record.args.items()
            }
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrivacyVisionServer")
logger.addFilter(ZeroPiiLoggingFilter())
# The Ollama client logs raw request/response bodies under its own logger name.
logging.getLogger("server.ollama_client").addFilter(ZeroPiiLoggingFilter())
logging.getLogger("server.nvidia_client").addFilter(ZeroPiiLoggingFilter())

AgentActionName = Literal["click", "fill", "scroll", "wait", "press"]
ALLOWED_ACTIONS = {"click", "fill", "scroll", "wait", "press"}


class PiiField(BaseModel):
    selector: str
    rect: dict[str, Any]
    reason: str


class PageElement(BaseModel):
    selector: str
    tag: str
    frame_id: int | None = None
    role: str | None = None
    type: str | None = None
    label: str | None = None
    placeholder: str | None = None
    name: str | None = None
    reason: str | None = None
    rect: dict[str, Any] | None = None


class VerificationEvidence(BaseModel):
    id: str
    kind: str
    text: str


class AnalyzeRequest(BaseModel):
    image: str = Field(default="", description="Base64 data URL of the redacted screenshot")
    pii_fields: list[PiiField] = Field(default_factory=list)
    elements: list[PageElement] = Field(default_factory=list)
    task: str
    # The extension's explicit selection; None uses the server's AI_PROVIDER default.
    provider: Literal["groq", "nvidia", "ollama"] | None = None
    verify: bool = Field(default=False, description="If true, verify task completion instead of planning actions")
    page_context: str | None = Field(default=None, description="Text context of current page for verification")
    action_history: list[dict] = Field(default_factory=list, description="History of actions taken in previous rounds")
    failed_action: str | None = Field(default=None, description="Description of the action that just failed, for self-correction requests")
    task_plan: list[str] = Field(default_factory=list, description="The task plan generated at the start of the run")
    plan: bool = Field(default=False, description="If true, return a step-by-step plan instead of actions")
    completed_steps: list[str] = Field(default_factory=list, description="Plan steps already verified complete, in order")
    remaining_steps: list[str] = Field(default_factory=list, description="Plan steps not yet complete, starting with the next one")
    last_action_result: str | None = Field(default=None, description="Outcome of the previous round's execution")
    current_url: str | None = Field(default=None, description="URL of the freshly captured page")
    verification_evidence: list[VerificationEvidence] = Field(
        default_factory=list,
        description="Numbered, fresh DOM evidence items the verifier must cite",
    )


class AgentAction(BaseModel):
    action: AgentActionName
    selector: str
    frame_id: int | None = None
    value: str | None = None
    target: str | None = None
    reasoning: str = ""


class AnalyzeResponse(BaseModel):
    action: AgentActionName | None = None
    selector: str | None = None
    frame_id: int | None = None
    value: str | None = None
    target: str | None = None
    reasoning: str = ""
    complete: bool = False
    actions: list[AgentAction] = Field(default_factory=list)
    done_through: int | None = Field(default=None, description="Highest 1-based plan step number now complete")
    replan: bool = False


class VerifyResponse(BaseModel):
    success: bool
    reason: str = ""


class PlanResponse(BaseModel):
    steps: list[str] = Field(default_factory=list)
    reasoning: str = ""


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
        "ai_provider": configured_ai_provider_label(),
        "nvidia_model": os.getenv("NVIDIA_MODEL") or "",
        "nvidia_configured": bool(os.getenv("NVIDIA_API_KEY")),
        "nvidia_fallback_models": list(nvidia_fallback_models()),
        "ai_fallback": ai_fallback_enabled(),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
        "zero_pii_logging": True
    }


@app.get("/providers")
async def list_providers():
    return {
        "providers": [
            {"id": "groq", "name": f"Groq Cloud VLM ({GROQ_MODEL})", "type": "cloud"},
            {"id": "nvidia", "name": f"NVIDIA Cloud VLM ({os.getenv('NVIDIA_MODEL') or 'model not configured'})", "type": "cloud"},
            {"id": "ollama", "name": f"Ollama Offline VLM ({OLLAMA_MODEL})", "type": "local"}
        ]
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-shaped model listing, so probes of this path get an answer.

    It reports the routes this server can actually serve right now, including
    the NVIDIA fallback models, and never the credentials behind them.
    """
    models = [
        {"id": GROQ_MODEL, "object": "model", "owned_by": "groq"},
        {"id": OLLAMA_MODEL, "object": "model", "owned_by": "ollama"},
    ]
    nvidia_model = os.getenv("NVIDIA_MODEL")
    if nvidia_model:
        models.append({"id": nvidia_model, "object": "model", "owned_by": "nvidia"})
    models += [
        {"id": model, "object": "model", "owned_by": "nvidia"}
        for model in nvidia_fallback_models()
        if model != nvidia_model
    ]
    return {"object": "list", "data": models}


# A fixed, trivial prompt. The preflight never carries a screenshot, DOM data,
# task text, credentials or any browser context - only this constant.
NVIDIA_PREFLIGHT_PROMPT = "Reply with OK."


def nvidia_preflight_result(
    ok: bool,
    model: str,
    error_type: str | None,
    status_code: int | None,
    message: str,
    configured: bool = True,
) -> dict[str, Any]:
    return {
        "provider": "nvidia",
        "configured": configured,
        "ok": ok,
        "model": model,
        "error_type": error_type,
        "status_code": status_code,
        "message": message,
    }


def run_nvidia_preflight() -> dict[str, Any]:
    """Check NVIDIA reachability with one minimal text-only request.

    Reports the configured model and a classified error type so a provider
    problem is visible before a browser-agent task is started. Never sends page
    content and never returns or logs the API key.
    """
    model = os.getenv("NVIDIA_MODEL") or ""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.warning(f"[AI-PREFLIGHT] provider=nvidia model={model} ok=false error_type=key_missing")
        return nvidia_preflight_result(
            False, model, "key_missing",
            None,
            "NVIDIA_API_KEY is not configured. Set it in .env.",
            configured=False,
        )
    if not model:
        logger.warning("[AI-PREFLIGHT] provider=nvidia ok=false error_type=model_missing")
        return nvidia_preflight_result(
            False, model, "model_missing",
            None,
            "NVIDIA_MODEL is not configured. Set it in .env to the NVIDIA model to use.",
            configured=False,
        )

    try:
        completion = create_nvidia_client(api_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": NVIDIA_PREFLIGHT_PROMPT}],
            max_completion_tokens=16,
            temperature=0,
        )
    except NvidiaError as exc:
        # The client already logged the classification; nothing more is needed.
        return nvidia_preflight_result(False, model, exc.error_type, exc.status_code, exc.message)
    except Exception as exc:
        failure = provider_failure(None, type(exc).__name__, error_type="unreachable")
        logger.warning(f"[AI-PREFLIGHT] provider=nvidia model={model} ok=false error_type={failure.error_type}")
        return nvidia_preflight_result(False, model, failure.error_type, None, failure.message)

    choices = getattr(completion, "choices", None) or []
    text = (choices[0].message.content or "").strip() if choices else ""
    if not text:
        failure = provider_failure(502, "preflight returned no usable text", error_type="invalid_response")
        logger.warning(f"[AI-PREFLIGHT] provider=nvidia model={model} ok=false error_type={failure.error_type}")
        return nvidia_preflight_result(False, model, failure.error_type, 502, failure.message)

    logger.info(f"[AI-PREFLIGHT] provider=nvidia model={model} ok=true")
    return nvidia_preflight_result(True, model, None, 200, "NVIDIA preflight successful.")


@app.get("/providers/nvidia/preflight")
async def nvidia_preflight():
    """Explicit connectivity check. /health stays static and makes no API call."""
    return await run_in_threadpool(run_nvidia_preflight)


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
        '"selector": "<CSS selector>", "frame_id": <optional frame ID>, "value": "<text to fill or token ID>", '
        '"target": "<visible label>", "reasoning": "<explanation>", "complete": true|false}. '
        'For multi-step tasks, use format: {"reasoning": "<explanation>", "actions": ['
        '{"action": "click"|"fill"|"scroll"|"wait"|"press", "selector": "<CSS selector>", "frame_id": <optional frame ID>, '
        '"value": "<text to fill or token ID>", "target": "<visible label>", "reasoning": "<step explanation>"}]}. '
        "Never invent arbitrary selectors when elements contains valid DOM selectors. "
        "When an Interactive Element includes frame_id, return that same frame_id with the action. "
        "When a selector from the Interactive Elements list appears truncated (ends mid-word or mid-attribute), prefer using the element's aria-label attribute as a selector instead, e.g. div[aria-label='Continue Prince Kumar'] or button[aria-label='Log in']. Always prefer short, unique selectors over long nested div chains. "
        "The user's task wording may differ from an element's exact visible label (e.g. the task says 'login' but the button says 'Log in', or 'sign up' vs 'Register') — match the task to the closest semantically equivalent element in the Interactive Elements list rather than requiring exact text matches. "
        "Set complete=true only when the ENTIRE User Task is satisfied (by this response's actions or already on the page); finishing one plan step is not finishing the task. "
        "If a field is marked as a redacted PII field, use exactly one of these client-resolved token identifiers in the value field: [TOKEN:USER_EMAIL], [TOKEN:USER_PASSWORD], [TOKEN:USER_PHONE], or [TOKEN:USER_NAME]. Never invent other token names, and never put a real credential value in the response. "
        "Prefer [TOKEN:USER_EMAIL] for email, email-or-phone, or login identifier fields. Reserve [TOKEN:USER_NAME] only for fields explicitly asking for a username or user name. Do not infer USER_NAME from generic login/sign-in context. Only use [TOKEN:USER_NAME] when the field is clearly username-like and not email-like. "
        "Keep each reasoning field brief: under 15 words, never more than one short sentence. "
        "Output the JSON object and nothing else — no prose, no markdown, no commentary before or after it. "
        "Only use selectors that appear in the Interactive Elements list. "
        "The Interactive Elements list describes the CURRENT page and is authoritative: selectors from Completed Actions, the Task Plan, Failed Action or earlier rounds may be stale, so never reuse one that is not in the current list; use the matching current element instead (for example a renumbered row such as #item-4 instead of #item-1). "
        "If an element in the current list can perform the next step, return that click or fill action; never return wait for a target that is present. Use wait only when the needed element is genuinely absent or the page is still loading. "
        "If the element the task needs is NOT in the Interactive Elements list, do not explain why and do not invent a selector. "
        'Return exactly this compact JSON instead: {"action": "wait", "selector": "body", '
        '"reasoning": "target element not on this page", "replan": true, "complete": false}. '
        "The client keeps the plan and progress, executes your actions, re-reads the page after any click or press, and calls you again, so act only on the next incomplete step. "
        'When Plan Progress is given, always include "done_through": <highest completed plan step number> in the JSON. '
        "Set complete=true when the task is fully satisfied."
    )


def describe_element(element: PageElement) -> dict[str, Any]:
    """One element as the model sees it: no geometry, no repeated names.

    A control whose label, placeholder and name all say "Username" is one
    target, but listing all three reads to the model as a longer name and
    comes back as "the 'Username Username' field". Only distinct wording is
    kept, so every target has exactly one visible name.
    """
    described = {
        key: value
        for key, value in element.model_dump(exclude_none=True).items()
        if key != "rect"
    }
    label = str(described.get("label") or "").strip().casefold()
    if label:
        for key in ("placeholder", "name"):
            if str(described.get(key) or "").strip().casefold() == label:
                described.pop(key, None)
    return described


def build_user_prompt(request: AnalyzeRequest, for_plan: bool = False) -> str:
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
        compact(describe_element(element))
        # Defense in depth: bound DOM controls even if an older/future client
        # sends more than the client-side network cap.
        for element in request.elements[:MAX_INTERACTIVE_ELEMENTS]
    ]
    completed_actions = [
        compact({
            key: action[key]
            for key in ("action", "target", "selector", "frame_id")
            if action.get(key) not in (None, "")
        })
        for action in request.action_history[-6:]
        if isinstance(action, dict)
    ]
    task_plan = [step[:MAX_CONTEXT_STRING_CHARS] for step in request.task_plan[:6]]
    failed_action = (request.failed_action or "")[:MAX_CONTEXT_STRING_CHARS]

    # Explicit plan progress, numbered against the full plan, so the model
    # continues from the next incomplete step instead of inferring its position
    # from raw action history.
    progress_section = ""
    if request.completed_steps or request.remaining_steps:
        completed_count = len(request.completed_steps)
        completed = [
            f"{number}. {step[:MAX_CONTEXT_STRING_CHARS]}"
            for number, step in enumerate(request.completed_steps, start=1)
        ][-6:]
        remaining = [
            f"{completed_count + number}. {step[:MAX_CONTEXT_STRING_CHARS]}"
            for number, step in enumerate(request.remaining_steps, start=1)
        ][:6]
        next_step = remaining[0] if remaining else "none - every planned step is done"
        progress_section = (
            "Plan Progress:\n"
            f"Completed Steps: {json.dumps(completed, ensure_ascii=True)}\n"
            f"Next Incomplete Step: {json.dumps(next_step, ensure_ascii=True)}\n"
            f"Remaining Steps: {json.dumps(remaining, ensure_ascii=True)}\n"
            f"Last Round Result: {json.dumps((request.last_action_result or 'none')[:MAX_CONTEXT_STRING_CHARS], ensure_ascii=True)}\n"
            f"Current URL: {json.dumps((request.current_url or '')[:MAX_CONTEXT_STRING_CHARS], ensure_ascii=True)}\n\n"
        )

    if progress_section and for_plan:
        closing = (
            "Re-plan: the Completed Steps are already done. "
            "Plan only the work that remains, starting from the current page."
        )
    elif progress_section:
        closing = (
            "This is one iteration of an execution loop. First set done_through to the highest plan step number "
            "that is visibly complete on THIS screenshot and element list (never below the Completed Steps count). "
            "Then return actions only for the next incomplete step after done_through, plus immediately following "
            "steps only if they are on this same page. Never repeat Completed Steps. "
            "Interactive Elements are the current page and authoritative: never reuse a selector that is not in that list, "
            "and if a listed element performs the step, act on it instead of waiting. "
            "If Failed Action is present, choose a different valid selector or approach. "
            'If that step cannot be done from this page, return {"action": "wait", "selector": "body", '
            '"reasoning": "<short>", "replan": true, "complete": false}.'
        )
    else:
        closing = (
            "Continue from the current screenshot. Do not repeat Completed Actions. "
            "Interactive Elements are the current page and authoritative: never reuse a selector that is not in that list. "
            "If Failed Action is present, choose a different valid selector or approach. "
            "Return only the remaining optimal action steps as JSON."
        )

    return (
        f"User Task: {request.task[:MAX_CONTEXT_STRING_CHARS]}\n\n"
        f"Task Plan: {json.dumps(task_plan, ensure_ascii=True)}\n\n"
        f"Completed Actions: {json.dumps(completed_actions, ensure_ascii=True)}\n\n"
        f"Failed Action: {json.dumps(failed_action, ensure_ascii=True)}\n\n"
        f"{progress_section}"
        f"Redacted PII Fields: {json.dumps(pii_context, ensure_ascii=True)}\n\n"
        f"Interactive Elements: {json.dumps(elements, ensure_ascii=True)}\n\n"
        f"{closing}"
    )


def decode_model_json(response_text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[match.start():])
                return value
            except json.JSONDecodeError:
                continue
        raise exc


def build_plan_prompt() -> str:
    return (
        "You are a browser automation planner. "
        "Your response must be a single valid JSON object and nothing else. "
        "No thinking, no markdown, no explanation — only the JSON object. "
        "Given interactive elements and the task, create a concise step-by-step plan. "
        "Use exactly this format: "
        "{\"reasoning\": \"one sentence about the overall approach\", "
        "\"steps\": [\"step 1\", \"step 2\", \"step 3\"]} "
        "Each step must be a short clear instruction. Maximum 6 steps. "
        "If Completed Steps are listed, they are already done: return steps only for the work that remains. "
        "Describe targets by their visible label, not by CSS selectors: selectors change between pages. "
        "Name each target exactly once, using its label as listed: never join an element's label, "
        "placeholder or name into one longer name. "
        "The Interactive Elements are the page the user is already on. Plan only the steps the task "
        "itself requires, and use no more steps than that: if every control the task needs is already "
        "listed, the plan is just those actions. "
        "Never add a step for a menu item, tab, link or button only because it appears in the list - "
        "include a navigation step only when the task needs that destination, or when a control the "
        "task needs is missing from the current page. "
        "A step names the field to fill, never the data to type. Never invent sample values such as "
        "usernames, passwords, emails or phone numbers: the credentials are supplied locally and a "
        "made-up one would be typed into the real form. Write \"Enter the username\", not "
        "\"Enter 'testuser' in the username field\". "
        "Do not split one field into separate click and type steps: filling a field is one step."
    )


def create_groq_client(api_key: str) -> Groq:
    """Create a client whose retries are controlled by the server."""
    # The SDK retries 429 responses internally, sometimes after a long
    # Retry-After delay. Each endpoint already has a bounded retry loop.
    return Groq(api_key=api_key, max_retries=0)


def nvidia_fallback_models() -> tuple[str, ...]:
    """Models the NVIDIA client may retry with, in order."""
    configured = os.getenv("NVIDIA_FALLBACK_MODELS")
    if configured is None:
        return DEFAULT_NVIDIA_FALLBACK_MODELS
    return tuple(model.strip() for model in configured.split(",") if model.strip())


def create_nvidia_client(api_key: str) -> NvidiaClient:
    return NvidiaClient(api_key, fallback_models=nvidia_fallback_models())


def ai_fallback_enabled() -> bool:
    return (os.getenv("AI_FALLBACK") or "1").strip().lower() not in ("0", "false", "off", "no")


def ollama_fallback_enabled() -> bool:
    return (os.getenv("AI_FALLBACK_OLLAMA") or "").strip().lower() in ("1", "true", "on", "yes")


def provider_chain(request: "AnalyzeRequest") -> list[str]:
    """The providers to try for this request, most preferred first.

    The explicit choice (or AI_PROVIDER) is always first and is the only one
    used when AI_FALLBACK is off. Otherwise every other configured provider
    follows, so a single provider outage does not end the agent run.
    """
    primary = request.provider if request.provider in ("groq", "nvidia", "ollama") else configured_ai_provider_label()
    if primary == "invalid":
        primary = "groq"
    chain = [primary]
    if not ai_fallback_enabled():
        return chain
    candidates = FALLBACK_PROVIDER_ORDER + (("ollama",) if ollama_fallback_enabled() else ())
    for candidate in candidates:
        if candidate in chain:
            continue
        if candidate == "nvidia" and not (os.getenv("NVIDIA_API_KEY") and os.getenv("NVIDIA_MODEL")):
            continue
        if candidate == "groq" and not os.getenv("GROQ_API_KEY"):
            continue
        # Ollama has no verification prompt of its own; it only serves plan/action.
        if candidate == "ollama" and request.verify:
            continue
        chain.append(candidate)
    return chain


def rate_limit_handoff(delay: float) -> HTTPException | None:
    """The failure to raise instead of sleeping out a rate limit, if any.

    A provider that says "come back in 19 seconds" has effectively ended its
    turn: another configured provider answers in a fraction of that. Returns
    None when the wait is short, or when this is the last provider left and
    waiting really is the only way to serve the request.
    """
    alternatives = remaining_providers.get()
    if delay <= MAX_RATE_LIMIT_WAIT_SECONDS or not alternatives:
        return None
    logger.warning(
        f"AI request rate-limited for {delay:.1f}s; handing off to {alternatives[0]} instead of waiting"
    )
    return HTTPException(
        status_code=429,
        detail=f"AI provider rate limit reached. Retry in {int(delay) + 1} seconds.",
        headers={"Retry-After": str(int(delay) + 1)},
    )


def provider_failure_is_transient(exc: Exception) -> bool:
    """True when another provider could still serve a request this one refused."""
    if isinstance(exc, NvidiaError):
        return exc.fallback_worthy
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, NvidiaError):
        return cause.fallback_worthy
    if isinstance(exc, HTTPException):
        # 500 covers a provider whose key or model is not configured here.
        return exc.status_code in FALLBACK_STATUS_CODES or exc.status_code == 500
    return False


def configured_ai_provider_label() -> str:
    provider = (os.getenv("AI_PROVIDER") or "groq").strip().lower()
    return provider if provider in AI_PROVIDERS else "invalid"


def selected_ai_provider() -> str:
    """The configured cloud provider. Defaults to Groq and never switches on its own."""
    provider = configured_ai_provider_label()
    if provider == "invalid":
        raise HTTPException(status_code=500, detail="AI_PROVIDER must be 'groq' or 'nvidia'.")
    return provider


def create_ai_client(groq_missing_key_detail: str, requested_provider: str | None = None) -> tuple[Any, str, str]:
    """Return (client, text_model, vision_model) for the selected cloud provider.

    An explicit groq/nvidia choice from the request wins; otherwise the
    server's AI_PROVIDER applies. Credentials are only ever read here.

    Both clients expose the same chat.completions.create call, so request
    construction, parsing, salvage and verification stay shared.
    """
    provider = requested_provider if requested_provider in AI_PROVIDERS else selected_ai_provider()
    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="NVIDIA_API_KEY is not configured. Set it in .env or use AI_PROVIDER=groq.",
            )
        # No default model: NVIDIA serves many models and silently picking one
        # would send requests to a model the account may not have access to.
        model = os.getenv("NVIDIA_MODEL")
        if not model:
            raise HTTPException(
                status_code=500,
                detail="NVIDIA_MODEL is not configured. Set it in .env to the NVIDIA model to use.",
            )
        logger.info(f"[AI-PROVIDER] provider=nvidia model={model}")
        return create_nvidia_client(api_key), model, model

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail=groq_missing_key_detail)
    logger.info(f"[AI-PROVIDER] provider=groq text_model={GROQ_TEXT_MODEL} vision_model={GROQ_MODEL}")
    return create_groq_client(api_key), GROQ_TEXT_MODEL, GROQ_MODEL


def call_ai_provider(generate: Callable[[Any], Any], request: Any) -> Any:
    """Run a generation function; report NVIDIA failures by their real cause."""
    try:
        return generate(request)
    except HTTPException as exc:
        cause = exc.__cause__
        if isinstance(cause, NvidiaError):
            logger.error(
                f"[AI-PROVIDER] provider=nvidia error_type={cause.error_type} "
                f"upstream_status={cause.status_code} retryable={cause.retryable}"
            )
            raise HTTPException(
                status_code=cause.server_status,
                detail=cause.message,
                headers=exc.headers,
            ) from cause
        raise


def provider_error_is_permanent(exc: Exception) -> bool:
    """True when a provider rejected the request in a way retrying cannot fix.

    Only classified NVIDIA failures answer to this; every other exception keeps
    the existing retry behaviour unchanged.
    """
    return isinstance(exc, NvidiaError) and not exc.retryable


def groq_retry_delay(exc: Exception) -> float:
    """Read Groq's suggested retry delay, capped for request handling."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for header_name in ("retry-after", "x-ratelimit-reset-tokens"):
            header_value = headers.get(header_name)
            if header_value:
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(header_value))
                if match:
                    return min(float(match.group(1)), 60.0)
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), re.I)
    if not match:
        return 2.0
    return min(float(match.group(1)), 60.0)


def coerce_plan_steps(raw: Any) -> list[str]:
    def describe_step(step: Any) -> str | None:
        if isinstance(step, str):
            text = step.strip()
            return text or None

        if isinstance(step, dict):
            explicit = step.get("step") or step.get("description")
            if isinstance(explicit, str) and explicit.strip():
                return explicit.strip()

            action = step.get("action")
            target = step.get("target") or step.get("selector")
            value = step.get("value")
            reasoning = step.get("reasoning")

            parts: list[str] = []
            if action:
                parts.append(str(action).capitalize())
            if target:
                parts.append(str(target))
            if value not in (None, ""):
                parts.append(f"with {value}")

            text = " ".join(parts).strip()
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                text = f"{text} - {reasoning.strip()}" if text else reasoning.strip()
            return text or None

        return None

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("steps") or raw.get("actions") or []
        if not items and raw.get("action"):
            items = [raw]
    else:
        items = []

    steps = [step for step in (describe_step(item) for item in items) if step]
    return steps[:6]


def generate_groq_plan(request: AnalyzeRequest) -> PlanResponse:
    parse_image_data_url(request.image)
    client, text_model, _ = create_ai_client("GROQ_API_KEY is not configured.", request.provider)
    
    user_content = build_user_prompt(request, for_plan=True)
    
    request_messages = [
        {"role": "system", "content": build_plan_prompt()},
        {
            "role": "user",
            # Planning only needs the task and sanitized DOM metadata. Keeping
            # the screenshot for action generation avoids two vision requests
            # competing for the same Groq token window.
            "content": user_content,
        },
    ]
    
    for attempt in range(1, 4):
        try:
            request_kwargs = {
                # Planning is text-only and uses the lightweight model so it
                # does not consume the visual action model's token window.
                "model": text_model,
                "messages": request_messages,
                # Keep the planning response compact for fast, focused steps.
                "max_completion_tokens": 250,
                "temperature": 0,
            }
            if attempt == 1:
                request_kwargs["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(
                **request_kwargs,
            )
            choice = completion.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            response_text = choice.message.content or ""
            if finish_reason == "length":
                raise ValueError("AI plan response was truncated at the token limit")
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            response_text = re.sub(r"```(?:json)?", "", response_text, flags=re.IGNORECASE).replace("```", "").strip()
            logger.info(f"[AI-PLAN] Raw response (attempt {attempt}): {response_text}")
            try:
                raw = json.loads(response_text)
            except json.JSONDecodeError as exc:
                raw = None
                decoder = json.JSONDecoder()
                for match in re.finditer(r"[\[{]", response_text):
                    try:
                        raw, _ = decoder.raw_decode(response_text[match.start():])
                        break
                    except json.JSONDecodeError:
                        continue
                if raw is None:
                    raise exc
            steps = coerce_plan_steps(raw)
            reasoning = raw.get("reasoning", "") if isinstance(raw, dict) else ""
            if not steps:
                raise ValueError("Plan returned no steps")
            logger.info(f"AI plan attempt {attempt}/3 succeeded: {len(steps)} steps")
            return PlanResponse(steps=steps, reasoning=reasoning)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"AI plan attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                raise HTTPException(
                    status_code=502,
                    detail="The AI model couldn't produce a valid plan. Try a simpler task.",
                ) from exc
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 and attempt < 3:
                delay = groq_retry_delay(exc)
                handoff = rate_limit_handoff(delay)
                if handoff:
                    raise handoff from exc
                logger.warning(f"AI plan request rate-limited; retrying in {delay:.1f}s")
                time.sleep(delay + 1)
                continue
            if status_code == 429:
                delay = groq_retry_delay(exc)
                raise HTTPException(
                    status_code=429,
                    detail=f"AI provider rate limit reached. Retry in {int(delay) + 1} seconds.",
                    headers={"Retry-After": str(int(delay) + 1)},
                ) from exc
            is_json_validate_failed = (
                isinstance(status_code, int)
                and 400 <= status_code < 500
                and "json_validate_failed" in str(getattr(exc, "body", None) or exc)
            )
            if is_json_validate_failed and attempt < 2:
                logger.warning(
                    "AI plan JSON mode was rejected; retrying without response_format"
                )
                continue
            logger.error(f"AI plan API call failed on attempt {attempt}/2: {exc}")
            raise HTTPException(
                status_code=502,
                detail="The AI provider could not produce a valid plan. Try a simpler task.",
            ) from exc


def build_verify_prompt() -> str:
    return (
        "You verify whether a browser automation task is FULLY satisfied by the CURRENT page. "
        "You receive a fresh redacted screenshot (when available) and structured DOM evidence items with ids like E1. "
        "Judge the entire task, not a single step: if any requested part is not reflected in the current screenshot or evidence, success is false. "
        "Judge the task's RESULT, not whether the controls it used are still on screen. A form, button or field "
        "that is gone because the action succeeded is evidence of success, not of failure: when the page shows the "
        "expected outcome - a confirmation, a success message, the resulting page or the changed state - success is true. "
        "Report false when that outcome is missing or contradicted, or when the page still shows the work undone. "
        "Completed plan steps are claims from the agent's history, not proof, and cannot be cited. "
        "If current evidence contradicts a claimed step (for example, the history says a panel was opened but no open panel is present now), success is false. "
        "When the screenshot is ambiguous, rely on the structured evidence: open dialogs, drawers or modals, visible controls, input values and the URL. "
        "Output ONLY valid JSON: "
        "{\"success\": true, \"reason\": \"short explanation\", \"evidence\": [\"E2\", \"E3\"]} "
        "or {\"success\": false, \"reason\": \"short explanation\", \"evidence\": []}. "
        "When success is true, evidence must list the ids of the current evidence items that prove it. "
        "Nothing else. No markdown. Just the JSON object."
    )


def build_verify_user_content(request: AnalyzeRequest) -> str:
    evidence_lines = [
        f"{item.id} [{item.kind}] {item.text[:MAX_VERIFY_EVIDENCE_CHARS]}"
        for item in request.verification_evidence[:MAX_VERIFY_EVIDENCE_ITEMS]
    ]
    claims = [step[:MAX_CONTEXT_STRING_CHARS] for step in request.completed_steps[-8:]]

    return (
        f"Task: {request.task}\n\n"
        f"Completed plan steps (claims, not proof): {json.dumps(claims, ensure_ascii=True)}\n\n"
        "Current DOM evidence:\n"
        f"{chr(10).join(evidence_lines) if evidence_lines else 'none provided'}\n\n"
        f"Current page:\n{request.page_context or 'No page context provided.'}\n\n"
        "Is the ENTIRE task satisfied right now? Output JSON only."
    )


def apply_verification_evidence_guard(request: AnalyzeRequest, success: bool, reason: Any, cited: Any) -> VerifyResponse:
    """Accept success only when it is grounded in cited CURRENT evidence.

    Completed plan steps are never evidence ids, so a verdict that rests on the
    agent's history alone - or cites nothing that exists now - is not success.
    """
    if success and request.verification_evidence:
        valid_ids = {item.id.strip().upper() for item in request.verification_evidence}
        cited_ids = {str(value).strip().upper() for value in cited} if isinstance(cited, list) else set()

        if not cited_ids & valid_ids:
            logger.warning("Verifier claimed success without citing current page evidence; not verified")
            return VerifyResponse(
                success=False,
                reason="Verifier did not ground success in current page evidence.",
            )

    return VerifyResponse(success=success, reason=str(reason)[:300])


def generate_groq_verification(request: AnalyzeRequest) -> VerifyResponse:
    client, text_model, vision_model = create_ai_client("GROQ_API_KEY is not configured.", request.provider)
    user_text = build_verify_user_content(request)
    # With a screenshot, verify on the configured vision model; without one,
    # keep the lightweight text model used before.
    use_image = bool(request.image)

    for attempt in range(1, 4):
        try:
            if use_image:
                request_kwargs = {
                    "model": vision_model,
                    "messages": [
                        {"role": "system", "content": build_verify_prompt()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {"type": "image_url", "image_url": {"url": request.image}},
                            ],
                        },
                    ],
                    "reasoning_effort": "none",
                }
            else:
                request_kwargs = {
                    "model": text_model,
                    "messages": [
                        {"role": "system", "content": build_verify_prompt()},
                        {"role": "user", "content": user_text},
                    ],
                }

            completion = client.chat.completions.create(
                **request_kwargs,
                response_format={"type": "json_object"},
                max_completion_tokens=150,
                temperature=0,
            )
            response_text = completion.choices[0].message.content or ""
            logger.info(f"[AI-VERIFY] Raw response (attempt {attempt}, image={use_image}): {response_text}")
            raw = decode_model_json(response_text)
            if not isinstance(raw, dict):
                raise json.JSONDecodeError("Verification response was not a JSON object", response_text, 0)
            success_val = raw.get("success", False)
            reason_val = raw.get("reason", raw.get("message", ""))
            if not isinstance(success_val, bool):
                success_val = str(success_val).lower() in ("true", "1", "yes")
            logger.info(f"AI verification attempt {attempt}/3 succeeded: success={success_val}")
            return apply_verification_evidence_guard(request, success_val, reason_val, raw.get("evidence"))
        except json.JSONDecodeError as exc:
            logger.warning(f"AI verification attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                return VerifyResponse(success=False, reason="Could not parse verification response.")
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 and attempt < 3:
                delay = groq_retry_delay(exc)
                handoff = rate_limit_handoff(delay)
                if handoff:
                    raise handoff from exc
                logger.warning(f"AI verification request rate-limited; retrying in {delay:.1f}s")
                time.sleep(delay + 1)
                continue

            if provider_error_is_permanent(exc):
                # A rejected or suspended key is not a model failure: retrying
                # it, with or without the image, only burns agent rounds.
                logger.error(f"AI verification stopped on a non-retryable provider error: {exc}")
                return VerifyResponse(success=False, reason=str(getattr(exc, "message", exc))[:300])

            if use_image and attempt < 3:
                # e.g. a content filter on the image: retry on text evidence only.
                logger.warning(f"AI image verification failed; retrying with DOM evidence only: {exc}")
                use_image = False
                continue

            logger.error(f"AI verification failed: {exc}")
            if attempt == 3:
                return VerifyResponse(success=False, reason=f"Verification error: {str(exc)[:100]}")

    return VerifyResponse(success=False, reason="Verification did not produce a verdict.")


def generate_groq_action(request: AnalyzeRequest) -> AnalyzeResponse:
    parse_image_data_url(request.image)
    client, _, vision_model = create_ai_client("GROQ_API_KEY is not configured on server.", request.provider)
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

    for attempt in range(1, 5):
        try:
            request_kwargs = {
                "model": vision_model,
                "messages": request_messages,
                # Groq's current on-demand OTPM limit for this model is 1000.
                # Leave headroom so the request is accepted rather than turned
                # into a 429 before generation starts.
                "max_completion_tokens": 500,
                "reasoning_effort": "none",
            }
            if attempt == 1:
                request_kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            logger.info(f"[AI-DEBUG] finish_reason (attempt {attempt}): {finish_reason}")
            response_text = choice.message.content or ""
            if finish_reason == "length":
                # The model ran past the cap (usually by explaining itself instead
                # of emitting JSON). Try to recover the first complete action
                # before burning another identical request on a retry.
                salvaged = salvage_json_payload(response_text)
                if salvaged is not None:
                    try:
                        recovered = parse_analyze_response(
                            {"actions": salvaged} if isinstance(salvaged, list) else salvaged,
                            request.elements,
                        )
                        logger.info(f"[SALVAGE] Recovered truncated response on attempt {attempt}")
                        # TEMPORARY debug logging of the successful action response.
                        logger.info(
                            f"[AI-ACTION-DEBUG] attempt={attempt} salvaged-truncated "
                            f"parsed={json.dumps(recovered.model_dump(exclude_none=True), ensure_ascii=True)}"
                        )
                        return recovered
                    except ValueError as salvage_exc:
                        logger.warning(f"[SALVAGE] Truncated payload unusable: {salvage_exc}")
                raise json.JSONDecodeError(
                    "AI action response was truncated at the token limit",
                    response_text,
                    len(response_text),
                )
            raw = decode_model_json(response_text)
            logger.info(f"[AI-DEBUG] Parsed action response (attempt {attempt})")
            # If model returns a bare list, wrap it in the expected dict format
            if isinstance(raw, list):
                payload = {"actions": raw}
            else:
                payload = raw
            response = parse_analyze_response(payload, request.elements)
            logger.info(f"AI attempt {attempt}/4 succeeded")
            # TEMPORARY debug logging of the successful action response.
            logger.info(
                f"[AI-ACTION-DEBUG] attempt={attempt} raw={response_text[:1000]} "
                f"parsed={json.dumps(response.model_dump(exclude_none=True), ensure_ascii=True)}"
            )
            return response
        except json.JSONDecodeError as exc:
            logger.warning(f"AI attempt {attempt}/4 failed: invalid JSON response ({exc})")
            logger.warning(f"[AI-DEBUG] finish_reason on failure (attempt {attempt}): {getattr(completion.choices[0], 'finish_reason', None)}")
            logger.warning(f"[AI-DEBUG] Failed raw response_text (attempt {attempt}): {response_text}")
            if attempt == 4:
                raise HTTPException(
                    status_code=502,
                    detail="The AI model couldn't produce a valid response after 3 attempts. Try a simpler task or page.",
                ) from exc
        except ValueError as exc:
            # parse_analyze_response rejected the payload (no valid action, or a
            # schema mismatch). Retry like any other recoverable model failure.
            logger.warning(f"AI attempt {attempt}/4 failed: unusable action payload ({exc})")
            if attempt == 4:
                raise HTTPException(
                    status_code=502,
                    detail="The AI model couldn't produce a valid response after 3 attempts. Try a simpler task or page.",
                ) from exc
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            error_body = getattr(exc, "body", None)
            if status_code == 429 and attempt < 4:
                delay = groq_retry_delay(exc)
                handoff = rate_limit_handoff(delay)
                if handoff:
                    raise handoff from exc
                logger.warning(f"AI action request rate-limited; retrying in {delay:.1f}s")
                time.sleep(delay + 1)
                continue
            if status_code == 429:
                delay = groq_retry_delay(exc)
                raise HTTPException(
                    status_code=429,
                    detail=f"AI provider rate limit reached. Retry in {int(delay) + 1} seconds.",
                    headers={"Retry-After": str(int(delay) + 1)},
                ) from exc
            is_json_validate_failed = (
                400 <= status_code < 500
                and "json_validate_failed" in str(error_body or exc)
            ) if isinstance(status_code, int) else False

            if is_json_validate_failed:
                logger.warning(
                    f"AI attempt {attempt}/2 failed: json_validate_failed ({exc})"
                )
                # Try to salvage the failed_generation content
                failed_gen = ""
                if isinstance(error_body, dict):
                    error_payload = error_body.get("error", {})
                    if isinstance(error_payload, dict):
                        failed_gen = str(error_payload.get("failed_generation", "") or "")
                    if not failed_gen:
                        failed_gen = str(error_body.get("failed_generation", "") or "")
                if failed_gen.strip():
                    try:
                        cleaned = failed_gen.strip()
                        salvaged = None

                        # Strategy 1: parse as-is
                        try:
                            salvaged = json.loads(cleaned)
                        except json.JSONDecodeError:
                            pass

                        # Strategy 2: wrap as {"actions": [...]} if it looks like one or more objects
                        if salvaged is None:
                            try:
                                # Only wrap if it doesn't already look like a list
                                if not cleaned.startswith("["):
                                    salvaged = json.loads('{"actions": [' + cleaned + ']}')
                                else:
                                    # If it was a truncated list, Strategy 3 might handle it better
                                    pass
                            except json.JSONDecodeError:
                                pass

                        # Strategy 3: extract first complete object via brace counting
                        if salvaged is None:
                            depth = 0
                            end_idx = -1
                            for i, ch in enumerate(cleaned):
                                if ch == '{':
                                    depth += 1
                                elif ch == '}':
                                    depth -= 1
                                    if depth == 0:
                                        end_idx = i + 1
                                        break
                            if end_idx > 0:
                                try:
                                    first_obj = json.loads(cleaned[:end_idx])
                                    # Wrap single action object in actions array
                                    if 'action' in first_obj and 'actions' not in first_obj:
                                        salvaged = {'actions': [first_obj]}
                                    else:
                                        salvaged = first_obj
                                except json.JSONDecodeError:
                                    pass

                        if salvaged is not None:
                            try:
                                payload = {"actions": salvaged} if isinstance(salvaged, list) else salvaged
                                response = parse_analyze_response(payload, request.elements)
                                logger.info(f"[SALVAGE] Succeeded on attempt {attempt}")
                                # TEMPORARY debug logging of the successful action response.
                                logger.info(
                                    f"[AI-ACTION-DEBUG] attempt={attempt} salvaged-failed-generation "
                                    f"raw={failed_gen[:1000]} "
                                    f"parsed={json.dumps(response.model_dump(exclude_none=True), ensure_ascii=True)}"
                                )
                                return response
                            except Exception as parse_exc:
                                logger.warning(f"[SALVAGE] parse_analyze_response failed: {parse_exc}")

                        logger.warning(f"[SALVAGE] All strategies failed: {cleaned[:150]}")
                    except Exception as salvage_exc:
                        logger.warning(f"[SALVAGE] Outer error: {salvage_exc}")
                if attempt == 4:
                    logger.error(f"[ACTION-FAIL] json_validate_failed body: {error_body}")
                    raise HTTPException(
                        status_code=502,
                        detail="The AI model couldn't produce a valid response after 2 attempts. Try a simpler task or page.",
                    ) from exc
                continue

            logger.error(f"AI action API call failed on attempt {attempt}/2: {exc}")
            logger.error(f"[ACTION-FAIL] General error body: {exc}")
            raise HTTPException(status_code=502, detail=f"AI VLM Provider Error: {exc}") from exc


def salvage_json_payload(raw_text: str) -> Any | None:
    """Recover the first usable JSON object from a truncated or wrapped response."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if not cleaned.startswith("["):
        try:
            return json.loads('{"actions": [' + cleaned + "]}")
        except json.JSONDecodeError:
            pass

    # Brace-count out the first complete object from a stream that was cut off.
    depth = 0
    for index, char in enumerate(cleaned):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    first_object = json.loads(cleaned[: index + 1])
                except json.JSONDecodeError:
                    return None
                if "action" in first_object and "actions" not in first_object:
                    return {"actions": [first_object]}
                return first_object

    return None


def normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def correct_target_mismatch(action: AgentAction, elements: list[PageElement]) -> AgentAction:
    """Point an action at the element it says it is targeting.

    Models sometimes pair the right visible name with the wrong selector - a
    click on "Practice" carrying the selector of the Courses menu item next to
    it. The element list is the authority on which selector wears which label,
    so when the named target belongs to exactly one other element, the selector
    is corrected to it. When the target is ambiguous or unknown, the action is
    left untouched: guessing further would be worse than the model's own guess.
    """
    target = normalize_label(action.target)
    if not target:
        return action

    chosen = next((element for element in elements if element.selector == action.selector), None)
    if chosen is None:
        return action

    chosen_label = normalize_label(chosen.label)
    if not chosen_label or target in chosen_label or chosen_label in target:
        return action

    matches = [
        element for element in elements
        if element.selector != action.selector and normalize_label(element.label) == target
    ]
    if len(matches) != 1:
        logger.warning(
            f"Action targets {action.target!r} but selector {action.selector} is {chosen.label!r}; "
            "no single element carries that name, leaving the action unchanged"
        )
        return action

    logger.warning(
        f"Correcting selector for {action.target!r}: {action.selector} is {chosen.label!r}, "
        f"using {matches[0].selector} instead"
    )
    return action.model_copy(update={"selector": matches[0].selector})


def parse_analyze_response(
    payload: dict[str, Any], elements: list[PageElement] | None = None
) -> AnalyzeResponse:
    # Some models answer with the remaining plan steps as bare strings in
    # "actions" ("2. Enter Password") next to a real action object. Those are
    # prose, not instructions: dropping them keeps the usable action instead of
    # failing the whole payload and spending another round.
    actions = payload.get("actions")
    if isinstance(actions, list):
        usable = [action for action in actions if isinstance(action, dict)]
        if len(usable) != len(actions):
            logger.warning(
                f"Dropping {len(actions) - len(usable)} non-action entries from the model's actions list"
            )
            payload = {**payload, "actions": usable}

    response = AnalyzeResponse.model_validate(payload)
    if not response.actions and response.action and response.selector:
        response.actions = [
            AgentAction(
                action=response.action,
                selector=response.selector,
                frame_id=response.frame_id,
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
        if re.search(r"(javascript:|<script|eval\(|window\.|onerror\s*=|onload\s*=)", act.selector, re.I):
            logger.warning(f"Rejecting dangerous selector: {act.selector}")
            continue
        valid_actions.append(correct_target_mismatch(act, elements) if elements else act)

    response.actions = valid_actions

    if not response.actions and not response.complete and (response.replan or response.done_through is not None):
        # A re-plan request or a pure progress report is a valid response
        # without an action; completion is decided by client-side verification.
        return response

    if not response.actions and not response.complete:
        # Model returned reasoning only — treat as task complete
        if response.reasoning:
            logger.info("Model returned reasoning only with no actions — treating as complete")
            response.complete = True
        else:
            raise ValueError("Response must include at least one valid action or set complete=true")

    return response


@app.post("/analyze", response_model=PlanResponse | VerifyResponse | AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> PlanResponse | VerifyResponse | AnalyzeResponse:
    if not request.verify:
        parse_image_data_url(request.image)
    elif request.image:
        # A verification screenshot passes the same fail-closed validation.
        parse_image_data_url(request.image)

    chain = provider_chain(request)
    logger.info(f"Processing analyze request with {chain[0]} provider (fallbacks: {', '.join(chain[1:]) or 'none'})")
    if not request.verify and not request.plan:
        logger.info(f"[AI-DEBUG] Interactive Elements: {json.dumps([element.model_dump() for element in request.elements], ensure_ascii=True)}")

    # The failure reported to the extension is always the selected provider's:
    # it is the one the user picked, and a fallback's own error would only
    # describe a provider they never asked for.
    primary_error: Exception | None = None
    for index, provider in enumerate(chain):
        remaining_providers.set(tuple(chain[index + 1:]))
        try:
            return await run_with_provider(provider, request)
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
            remaining = chain[index + 1:]
            if not remaining or not provider_failure_is_transient(exc):
                break
            status = getattr(exc, "status_code", type(exc).__name__)
            logger.warning(
                f"[AI-PROVIDER] provider={provider} failed ({status}); falling back to {remaining[0]}"
            )
    # Every alternative failed too. A provider that only stepped aside to dodge
    # a wait has not actually been tried, so take the wait now rather than
    # losing the round: handing off must never cost more than staying put.
    if isinstance(primary_error, HTTPException) and primary_error.status_code == 429 and len(chain) > 1:
        logger.warning(
            f"[AI-PROVIDER] no fallback could serve the request; waiting out {chain[0]}'s rate limit instead"
        )
        remaining_providers.set(())
        return await run_with_provider(chain[0], request)

    raise primary_error  # the chain always holds the selected provider


async def run_with_provider(
    provider: str, request: AnalyzeRequest
) -> PlanResponse | VerifyResponse | AnalyzeResponse:
    """Serve one analyze request with exactly one provider."""
    if provider == "ollama":
        payload = await generate_ollama_action(
            build_system_prompt(),
            build_user_prompt(request),
            request.image,
            OLLAMA_MODEL,
        )
        action_response = parse_analyze_response(payload, request.elements)
        if not request.plan:
            return action_response
        steps = coerce_plan_steps({
            "actions": [action.model_dump() for action in action_response.actions]
        })
        if not steps:
            raise HTTPException(status_code=502, detail="Ollama could not produce a valid plan.")
        return PlanResponse(steps=steps, reasoning=action_response.reasoning)

    # Pin the cloud provider for this attempt so a fallback really switches.
    pinned = request.model_copy(update={"provider": provider})
    if request.verify:
        return await run_in_threadpool(call_ai_provider, generate_groq_verification, pinned)
    if request.plan:
        return await run_in_threadpool(call_ai_provider, generate_groq_plan, pinned)
    return await run_in_threadpool(call_ai_provider, generate_groq_action, pinned)
