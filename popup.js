import {
  FaceDetector,
  FilesetResolver
} from "./vendor/mediapipe/vision_bundle.mjs";

const chrome = globalThis.browser ?? globalThis.chrome;
const captureButton = document.getElementById("captureButton");
const runAgentButton = document.getElementById("runAgentButton");
const taskInput = document.getElementById("taskInput");
const providerSelect = document.getElementById("providerSelect");
const screenshotCanvas = document.getElementById("screenshotCanvas");
const canvasContext = screenshotCanvas.getContext("2d");
const statusMessage = document.getElementById("status");
const agentResult = document.getElementById("agentResult");
const perfStats = document.getElementById("perfStats");
const saveCredentialsButton = document.getElementById("saveCredentialsButton");
const credentialsStatus = document.getElementById("credentialsStatus");
const credentialInputs = [...document.querySelectorAll("[data-credential-token]")];

const CREDENTIALS_STORAGE_KEY = "credentialsVault";

const piiText = globalThis.__privacyVisionPii;
// Presentation hooks from popup-ui.js. Optional: the pipeline never depends on them.
const ui = globalThis.__privacyVisionUi;

if (!piiText) {
  throw new Error("privacy-vision-agent: pii-patterns.js must load before popup.js");
}

const analyzeUrl = "http://127.0.0.1:8000/analyze";

const modelPath = chrome.runtime.getURL("models/blaze_face_full_range.tflite");
const wasmPath = chrome.runtime.getURL("vendor/mediapipe/wasm");
const faceBlurRadius = 16;
const minDetectionConfidence = 0.5;
const MAX_INTERACTIVE_ELEMENTS = 20;
// Settle window before goal verification (unchanged from the previous inline wait).
const VERIFY_SETTLE_MS = 6000;

// TEMPORARY debug logging for action selection - remove after diagnosis.
// Local console only; resolved credential values are never logged.
const DEBUG_ACTION_LOGGING = true;

function debugLog(event, details) {
  if (DEBUG_ACTION_LOGGING) {
    console.log(`[AGENT-DEBUG] ${event} ${JSON.stringify(details)}`);
  }
}
const MAX_ELEMENT_TEXT_LENGTH = 30;
const MAX_MODEL_IMAGE_DIMENSION = 512;

let faceDetectorPromise;
let originalScreenshotDataUrl = "";
let latestPiiFields = [];
let latestElements = [];
// The sanitized task text of the current run. Used only to tell a value the
// user actually asked for from one the model made up.
let latestTaskText = "";
const performanceMetrics = {
  faceModelLoadMs: null,
  faceDelegate: "CPU",
  detectionMs: null,
  domScanMs: null,
  imageSizeKb: null,
  serverRoundTripMs: null
};

async function resolveCredentialToken(value) {
  if (typeof value !== "string") {
    return value;
  }

  const tokenMatch = value.match(/^\[TOKEN:([A-Z0-9_]+)\]$/);
  if (!tokenMatch) {
    return value;
  }

  const tokenName = tokenMatch[1];
  const stored = await chrome.storage.local.get(CREDENTIALS_STORAGE_KEY);
  const vault = stored[CREDENTIALS_STORAGE_KEY] ?? {};
  const resolvedValue = vault[tokenName];

  if (typeof resolvedValue !== "string" || resolvedValue.length === 0) {
    throw new Error(`No saved value for ${tokenName} — set it in the Credentials Vault first.`);
  }

  return resolvedValue;
}

async function loadCredentialVault() {
  try {
    const stored = await chrome.storage.local.get(CREDENTIALS_STORAGE_KEY);
    const vault = stored[CREDENTIALS_STORAGE_KEY] ?? {};

    for (const input of credentialInputs) {
      input.value = typeof vault[input.dataset.credentialToken] === "string"
        ? vault[input.dataset.credentialToken]
        : "";
    }
  } catch (error) {
    credentialsStatus.textContent = "Unable to load the local Credentials Vault.";
    console.error(error);
  }
}

async function saveCredentialVault() {
  const vault = Object.fromEntries(
    credentialInputs.map((input) => [input.dataset.credentialToken, input.value])
  );

  try {
    // chrome.storage.local is device-local but not encrypted at rest by the browser.
    // This demo keeps values off the server; production should use native credentials
    // storage or an encrypted vault instead.
    await chrome.storage.local.set({ [CREDENTIALS_STORAGE_KEY]: vault });
    credentialsStatus.textContent = "Credentials saved on this device.";
  } catch (error) {
    credentialsStatus.textContent = "Unable to save the local Credentials Vault.";
    console.error(error);
  }
}

function looksEmailLike(text) {
  if (typeof text !== "string") {
    return false;
  }

  return /\b(?:e-?mail|email address|mail address|email)\b/i.test(text);
}

// The credential token each field kind may legitimately receive. The model is
// never trusted to make this choice: it is decided locally from the field's own
// metadata, so an email credential can never land in a username field.
const CREDENTIAL_TOKEN_FOR_FIELD = {
  email: "[TOKEN:USER_EMAIL]",
  username: "[TOKEN:USER_USERNAME]",
  password: "[TOKEN:USER_PASSWORD]"
};

function looksUsernameLike(text) {
  return typeof text === "string" && /\b(?:user\s?name|username|userid|user\s?id|login\s?id|login\s?name)\b/i.test(text);
}

function looksPasswordLike(text) {
  return typeof text === "string" && /\b(?:password|passwd|pwd|passphrase)\b/i.test(text);
}

/**
 * Classify a fill target as an email, username or password field from its own
 * DOM metadata. Returns null when the evidence is not clear, so ordinary text
 * inputs are never guessed into a credential field.
 */
function classifyCredentialField(element, action) {
  const type = typeof element?.type === "string" ? element.type.toLowerCase() : "";

  // An explicit input type is unambiguous and outranks any label text.
  if (type === "email") {
    return "email";
  }
  if (type === "password") {
    return "password";
  }

  const metadata = [
    element?.name,
    element?.label,
    element?.placeholder,
    element?.selector,
    // Only used when the element is not in the current scan; never the model's
    // free-text reasoning, which is what produced the wrong token to begin with.
    element ? "" : action?.selector,
    element ? "" : action?.target
  ]
    .filter((value) => typeof value === "string" && value.length > 0)
    .join(" ");

  if (!metadata) {
    return null;
  }

  if (looksPasswordLike(metadata)) {
    return "password";
  }
  // Checked before username: "email address" must not match a bare user-id rule.
  if (looksEmailLike(metadata)) {
    return "email";
  }
  if (looksUsernameLike(metadata)) {
    return "username";
  }

  return null;
}

/** True when the run's own task text asked for this value. */
function taskMentionsValue(value) {
  const text = typeof value === "string" ? value.trim().toLowerCase() : "";
  return text.length > 0 && latestTaskText.toLowerCase().includes(text);
}

function findElementForAction(action) {
  const selector = typeof action?.selector === "string" ? action.selector : "";
  return selector ? latestElements.find((element) => element.selector === selector) : undefined;
}

/**
 * Remap a credential token the model chose for a field whose semantics say
 * otherwise. Only unambiguous corrections are made; anything unclassified is
 * left exactly as the model proposed.
 */
function enforceCredentialTokenForField(action) {
  const proposed = action?.value;
  if (typeof proposed !== "string") {
    return proposed;
  }

  const isToken = /^\[TOKEN:[A-Z0-9_]+\]$/.test(proposed);
  const fieldKind = classifyCredentialField(findElementForAction(action), action);

  if (!isToken) {
    // A literal in a credential field is either the user's own words or, far
    // more often, an invented sample ("testuser", "testpass") from a plan step.
    // Invented data must never be typed into a login form: the stored
    // credential is the only legitimate value there.
    if (fieldKind && proposed.trim() && !taskMentionsValue(proposed)) {
      const required = CREDENTIAL_TOKEN_FOR_FIELD[fieldKind];
      if (required) {
        // The rejected literal is never logged: it may be real data.
        console.warn(
          `Replacing a literal value the model invented for a ${fieldKind} field (${action?.selector ?? ""}) with ${required}.`
        );
        return required;
      }
    }

    // Not a credential token (a search term, a date): nothing to enforce.
    return proposed;
  }

  if (!fieldKind) {
    return proposed;
  }

  const required = CREDENTIAL_TOKEN_FOR_FIELD[fieldKind];
  if (required && required !== proposed) {
    // Token names only - no credential value is read or logged here.
    console.warn(
      `Credential token ${proposed} is incompatible with a ${fieldKind} field (${action?.selector ?? ""}); using ${required}.`
    );
    return required;
  }

  return proposed;
}

async function resolveCredentialValueForAction(action) {
  if (action?.action !== "fill") {
    return action?.value;
  }

  const tokenValue = typeof action?.value === "string" ? action.value : action?.value;
  if (typeof tokenValue !== "string") {
    return tokenValue;
  }

  return resolveCredentialToken(enforceCredentialTokenForField(action));
}

function formatMilliseconds(value) {
  return value == null ? "—" : `${Math.round(value)}ms`;
}

function formatSecondsOrMilliseconds(value) {
  if (value == null) {
    return "—";
  }

  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}

function updatePerformanceStats() {
  perfStats.textContent = [
    `Face model: ${formatMilliseconds(performanceMetrics.faceModelLoadMs)}`,
    `Face delegate: ${performanceMetrics.faceDelegate ?? "—"}`,
    `Detection: ${formatMilliseconds(performanceMetrics.detectionMs)}`,
    `DOM scan: ${formatMilliseconds(performanceMetrics.domScanMs)}`,
    `Image: ${performanceMetrics.imageSizeKb == null ? "—" : `${performanceMetrics.imageSizeKb.toFixed(1)}KB`}`,
    `Server: ${formatSecondsOrMilliseconds(performanceMetrics.serverRoundTripMs)}`
  ].join(" | ");
}

async function getFaceDetector() {
  if (!faceDetectorPromise) {
    const pending = (async () => {
      const loadStart = performance.now();
      const vision = await FilesetResolver.forVisionTasks(wasmPath);
      // MediaPipe also supports delegate: "GPU" (WebGL-backed). It measured
      // slower locally for this lightweight model, so CPU is the deliberate default.
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: "CPU"
        },
        minDetectionConfidence,
        runningMode: "IMAGE"
      });

      performanceMetrics.faceModelLoadMs = performance.now() - loadStart;
      console.log(
        `Face detector initialized with CPU delegate; load time: ${performanceMetrics.faceModelLoadMs.toFixed(1)} ms`
      );
      return detector;
    })();

    pending.catch(() => {
      // Don't cache a failed load; allow the next capture to retry it.
      if (faceDetectorPromise === pending) {
        faceDetectorPromise = undefined;
      }
    });
    faceDetectorPromise = pending;
  }

  return faceDetectorPromise;
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();

    image.addEventListener("load", () => resolve(image), { once: true });
    image.addEventListener("error", () => reject(new Error("Unable to load captured screenshot.")), {
      once: true
    });

    image.src = dataUrl;
  });
}

// Keep the model payload small while preserving the full-quality canvas used by
// the on-screen preview and the local privacy checks.
function getModelImageDataUrl() {
  const scale = Math.min(1, MAX_MODEL_IMAGE_DIMENSION / Math.max(screenshotCanvas.width, screenshotCanvas.height));
  const modelCanvas = document.createElement("canvas");
  modelCanvas.width = Math.max(1, Math.round(screenshotCanvas.width * scale));
  modelCanvas.height = Math.max(1, Math.round(screenshotCanvas.height * scale));
  const modelContext = modelCanvas.getContext("2d");
  modelContext.drawImage(screenshotCanvas, 0, 0, modelCanvas.width, modelCanvas.height);
  return modelCanvas.toDataURL("image/jpeg", 0.6);
}

function prepareModelElements(elements) {
  return [...(elements ?? [])]
    // Keep the existing viewport-first behavior explicit at the network
    // boundary, including results merged from multiple frames.
    .sort((first, second) => Number(Boolean(second.rect)) - Number(Boolean(first.rect)))
    // A selector that embeds PII (e.g. [aria-label="a@b.com"]) cannot be sent
    // as-is and cannot be redacted without becoming unusable, so drop it.
    .filter((element) => {
      if (typeof element.selector === "string" && piiText.containsPii(element.selector)) {
        console.warn("Dropping interactive element whose selector embeds PII.");
        return false;
      }
      return true;
    })
    .slice(0, MAX_INTERACTIVE_ELEMENTS)
    .map((element) => {
      const compactElement = { ...element };

      for (const textField of ["text", "label"]) {
        if (typeof compactElement[textField] === "string") {
          compactElement[textField] = compactElement[textField].slice(0, MAX_ELEMENT_TEXT_LENGTH);
        }
      }

      // Structure (tag, role, type, selector, rect) is useful to the model and
      // carries no PII. Free text can, so it is replaced with a placeholder.
      for (const textField of ["text", "label", "placeholder", "name", "reason"]) {
        if (typeof compactElement[textField] === "string") {
          compactElement[textField] = piiText.redactPiiText(compactElement[textField]);
        }
      }

      if (Number.isInteger(compactElement.frameId)) {
        compactElement.frame_id = compactElement.frameId;
        delete compactElement.frameId;
      }

      return compactElement;
    });
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  return tabs[0];
}

async function getPiiFields(tabId) {
  if (!tabId) {
    return {
      fields: [],
      elements: [],
      viewport: null,
      scanFailed: true,
      scanError: "no active tab to scan"
    };
  }

  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      files: ["pii-patterns.js", "content.js"]
    });

    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      func: () => window.__privacyVisionAgent?.scanPiiFields?.() ?? null
    });

    return mergeFramePiiResults(injectionResults);
  } catch (error) {
    console.warn("Unable to scan DOM PII fields:", error);
    return {
      fields: [],
      elements: [],
      viewport: null,
      scanFailed: true,
      scanError: error.message
    };
  }
}

function getFramePathKey(framePath) {
  return Array.isArray(framePath) ? framePath.join("/") : "";
}

function mergeFramePiiResults(injectionResults) {
  const frameResults = injectionResults
    .map(({ frameId, result }) => ({
      frameId,
      result
    }))
    .filter(({ result }) => result);
  const topFrame =
    frameResults.find(({ frameId, result }) => frameId === 0 || (result.framePath?.length ?? 0) === 0)?.result
    ?? frameResults[0]?.result;
  const frameOffsets = new Map([["", { x: 0, y: 0 }]]);

  let changed = true;
  while (changed) {
    changed = false;

    for (const { result } of frameResults) {
      const parentPathKey = getFramePathKey(result.framePath);
      const parentOffset = frameOffsets.get(parentPathKey);

      if (!parentOffset) {
        continue;
      }

      for (const childFrame of result.childFrameRects ?? []) {
        const childPathKey = getFramePathKey(childFrame.framePath);

        if (frameOffsets.has(childPathKey)) {
          continue;
        }

        frameOffsets.set(childPathKey, {
          x: parentOffset.x + childFrame.rect.x,
          y: parentOffset.y + childFrame.rect.y
        });
        changed = true;
      }
    }
  }

  const fields = [];
  const elements = [];
  const textPiiFailure = frameResults.find(({ result }) => result.textPiiError)?.result?.textPiiError;

  for (const { frameId, result } of frameResults) {
    const framePathKey = getFramePathKey(result.framePath);
    const frameOffset = frameOffsets.get(framePathKey) ?? { x: 0, y: 0 };

    if (!frameOffsets.has(framePathKey)) {
      console.warn("Missing iframe offset; using local frame coordinates", {
        frameId,
        framePath: result.framePath
      });
    }

    for (const field of result.fields ?? []) {
      fields.push({
        ...field,
        frameId,
        framePath: result.framePath,
        rect: {
          ...field.rect,
          x: field.rect.x + frameOffset.x,
          y: field.rect.y + frameOffset.y
        }
      });
    }

    for (const element of result.elements ?? []) {
      elements.push({
        ...element,
        frameId,
        framePath: result.framePath,
        rect: element.rect ? {
          ...element.rect,
          x: element.rect.x + frameOffset.x,
          y: element.rect.y + frameOffset.y
        } : null
      });
    }

  }

  return {
    fields,
    elements,
    viewport: topFrame?.viewport ?? null,
    textPiiCount: frameResults.reduce((total, { result }) => total + (result.textPiiCount ?? 0), 0),
    scanFailed: frameResults.length === 0 || Boolean(textPiiFailure),
    scanError: frameResults.length === 0
      ? "no frame returned a scan result"
      : (textPiiFailure ? `text PII scan failed: ${textPiiFailure}` : undefined)
  };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function blurFaceRegions(image, detections) {
  const blurExpansion = faceBlurRadius * 2;

  for (const detection of detections) {
    const boundingBox = detection.boundingBox;

    if (!boundingBox) {
      continue;
    }

    const left = clamp(Math.floor(boundingBox.originX), 0, screenshotCanvas.width);
    const top = clamp(Math.floor(boundingBox.originY), 0, screenshotCanvas.height);
    const right = clamp(Math.ceil(boundingBox.originX + boundingBox.width), 0, screenshotCanvas.width);
    const bottom = clamp(Math.ceil(boundingBox.originY + boundingBox.height), 0, screenshotCanvas.height);
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0) {
      continue;
    }

    const expandedLeft = clamp(left - blurExpansion, 0, screenshotCanvas.width);
    const expandedTop = clamp(top - blurExpansion, 0, screenshotCanvas.height);
    const expandedRight = clamp(right + blurExpansion, 0, screenshotCanvas.width);
    const expandedBottom = clamp(bottom + blurExpansion, 0, screenshotCanvas.height);
    const expandedWidth = expandedRight - expandedLeft;
    const expandedHeight = expandedBottom - expandedTop;

    canvasContext.save();
    canvasContext.beginPath();
    canvasContext.rect(left, top, width, height);
    canvasContext.clip();
    canvasContext.filter = `blur(${faceBlurRadius}px)`;
    canvasContext.drawImage(
      image,
      expandedLeft,
      expandedTop,
      expandedWidth,
      expandedHeight,
      expandedLeft,
      expandedTop,
      expandedWidth,
      expandedHeight
    );
    canvasContext.restore();
  }

  canvasContext.filter = "none";
}

function drawPiiRedactions(fields, viewport) {
  if (!viewport?.width || !viewport?.height) {
    return;
  }

  const devicePixelRatio = viewport.devicePixelRatio || 1;
  const expectedCanvasWidth = viewport.width * devicePixelRatio;
  const expectedCanvasHeight = viewport.height * devicePixelRatio;
  const ratioScaleX = screenshotCanvas.width / viewport.width;
  const ratioScaleY = screenshotCanvas.height / viewport.height;
  const canvasMatchesDevicePixels =
    Math.abs(screenshotCanvas.width - expectedCanvasWidth) <= 2 &&
    Math.abs(screenshotCanvas.height - expectedCanvasHeight) <= 2;
  const scaleX = canvasMatchesDevicePixels ? devicePixelRatio : ratioScaleX;
  const scaleY = canvasMatchesDevicePixels ? devicePixelRatio : ratioScaleY;

  canvasContext.save();
  canvasContext.fillStyle = "#000000";

  for (const field of fields) {
    const rect = field.rect;

    if (!rect) {
      continue;
    }

    const left = clamp(Math.floor(rect.x * scaleX), 0, screenshotCanvas.width);
    const top = clamp(Math.floor(rect.y * scaleY), 0, screenshotCanvas.height);
    const right = clamp(Math.ceil((rect.x + rect.width) * scaleX), 0, screenshotCanvas.width);
    const bottom = clamp(Math.ceil((rect.y + rect.height) * scaleY), 0, screenshotCanvas.height);
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0) {
      continue;
    }

    canvasContext.fillRect(left, top, width, height);
  }

  canvasContext.restore();
}

function drawRedactedScreenshot(image, detections, piiScan) {
  screenshotCanvas.width = image.naturalWidth;
  screenshotCanvas.height = image.naturalHeight;
  screenshotCanvas.style.display = "block";

  canvasContext.clearRect(0, 0, screenshotCanvas.width, screenshotCanvas.height);
  canvasContext.drawImage(image, 0, 0);
  blurFaceRegions(image, detections);
  drawPiiRedactions(piiScan.fields, piiScan.viewport);
}

// Per-category totals for the popup summary, classified the same way as the
// audit entries below. Numbers only; no field, selector or value leaves here.
function countRedactions(detections, fields) {
  const counts = { faces: detections.length, formFields: 0, textPii: 0, entities: 0 };

  for (const field of fields) {
    const reason = String(field.reason ?? "");
    const textCategory = reason.startsWith("text-pii:") ? reason.slice("text-pii:".length) : null;

    if (textCategory == null) {
      counts.formFields += 1;
    } else if (textCategory in (piiText.ENTITY_PLACEHOLDERS ?? {})) {
      counts.entities += 1;
    } else {
      counts.textPii += 1;
    }
  }

  return counts;
}

function verifySanitizedCanvasOutput(canvas, originalDataUrl, piiCount = 0, faceCount = 0) {
  if (!canvas || canvas.width <= 0 || canvas.height <= 0) {
    return { valid: false, error: "Fail-Closed Privacy Error: Invalid or zero-sized screenshot canvas." };
  }

  let sanitizedDataUrl = "";
  try {
    sanitizedDataUrl = canvas.toDataURL("image/png");
  } catch (e) {
    return { valid: false, error: `Fail-Closed Privacy Error: Canvas conversion failed: ${e.message}` };
  }

  if (!sanitizedDataUrl || !sanitizedDataUrl.startsWith("data:image/")) {
    return { valid: false, error: "Fail-Closed Privacy Error: Generated screenshot payload is invalid." };
  }

  if ((piiCount > 0 || faceCount > 0) && originalDataUrl && sanitizedDataUrl === originalDataUrl) {
    return {
      valid: false,
      error: "Fail-Closed Privacy Error: Outbound validation failed! Redaction output matches raw capture."
    };
  }

  return { valid: true, sanitizedDataUrl };
}

async function captureRedactedScreenshot() {
  // Invalidate up front. If this capture throws part-way, the agent must not
  // fall back to the previous page's interactive elements.
  latestPiiFields = [];
  latestElements = [];
  ui?.captureStarted();

  try {
    const activeTab = await getActiveTab();
    const dataUrl = await chrome.tabs.captureVisibleTab(activeTab?.windowId, { format: "png" });
    let originalDataUrl = dataUrl;

    statusMessage.textContent = "Loading detector...";
    const faceDetector = await getFaceDetector();
    const image = await loadImage(dataUrl);

    statusMessage.textContent = "Detecting faces...";
    const inferenceStart = performance.now();
    const result = faceDetector.detect(image);
    performanceMetrics.detectionMs = performance.now() - inferenceStart;
    console.log(`Face detector inference time: ${performanceMetrics.detectionMs.toFixed(1)} ms`);

    statusMessage.textContent = "Scanning form fields...";
    const domScanStart = performance.now();
    const piiScan = await getPiiFields(activeTab?.id);
    performanceMetrics.domScanMs = performance.now() - domScanStart;

    if (piiScan.scanFailed) {
      throw new Error(
        `DOM PII scan did not run (${piiScan.scanError || "unknown reason"}); refusing to transmit an unverified screenshot.`
      );
    }

    if (piiScan.fields.length > 0 && !(piiScan.viewport?.width > 0 && piiScan.viewport?.height > 0)) {
      throw new Error(
        "PII fields were detected but no viewport was available to place their redactions."
      );
    }

    drawRedactedScreenshot(image, result.detections, piiScan);
    latestPiiFields = piiScan.fields;
    latestElements = piiScan.elements;

    // Outbound validation check
    const outboundCheck = verifySanitizedCanvasOutput(
      screenshotCanvas,
      originalDataUrl,
      piiScan.fields.length,
      result.detections.length
    );

    // Zero out raw original screenshot immediately to ensure raw data is never stored
    originalDataUrl = null;
    originalScreenshotDataUrl = "";

    if (!outboundCheck.valid) {
      throw new Error(outboundCheck.error);
    }

    // Local-only audit of what was redacted (metadata only, never values).
    // Recorded only after outbound validation confirmed the redactions.
    const audit = globalThis.__privacyVisionRedactionAudit;
    if (audit) {
      const pageOrigin = activeTab?.url;

      for (const detection of result.detections) {
        const box = detection.boundingBox;
        audit.recordRedaction({
          type: "face",
          source: "face-detector",
          reason: "face blurred",
          rect: box ? { x: box.originX, y: box.originY, width: box.width, height: box.height } : undefined,
          pageOrigin
        });
      }

      for (const field of piiScan.fields) {
        const reason = String(field.reason ?? "");
        const textCategory = reason.startsWith("text-pii:") ? reason.slice("text-pii:".length) : null;
        const isEntity = textCategory != null && textCategory in (piiText.ENTITY_PLACEHOLDERS ?? {});
        audit.recordRedaction({
          type: textCategory == null ? "form-field" : isEntity ? "entity" : "text-pii",
          source: textCategory == null ? "dom-field-scan" : "text-scan",
          reason: textCategory == null ? reason : "redacted visible text",
          category: textCategory ?? undefined,
          rect: field.rect,
          pageOrigin
        });
      }
    }

    const modelImageDataUrl = getModelImageDataUrl();
    performanceMetrics.imageSizeKb = modelImageDataUrl.length / 1024;
    updatePerformanceStats();

    const textPiiCount = piiScan.textPiiCount ?? 0;
    statusMessage.textContent =
      `${result.detections.length} face(s) blurred, ${piiScan.fields.length - textPiiCount} field(s) redacted, ` +
      `${textPiiCount} text PII region(s) redacted`;
    ui?.captureCompleted(countRedactions(result.detections, piiScan.fields));
    return { activeTab, piiScan, sanitizedDataUrl: outboundCheck.sanitizedDataUrl };
  } catch (error) {
    statusMessage.textContent = `Privacy Error: ${error.message}`;
    ui?.captureBlocked(error.message);
    throw new Error(`Privacy Fail-Closed: Screenshot transmission blocked. (${error.message})`);
  }
}

function sanitizePromptText(text) {
  if (!text || typeof text !== "string") return text;
  return piiText.redactPiiText(text);
}

// Text-PII region selectors are informational only (the model never acts on
// them), so any PII/entity text embedded in them - e.g. an aria-label holding a
// person's name - is redacted. Form-field selectors stay intact: the model
// needs them for tokenised fills.
function prepareModelPiiFields(piiFields) {
  return (piiFields ?? []).map((field) =>
    typeof field.reason === "string" && field.reason.startsWith("text-pii:")
      ? { ...field, selector: sanitizePromptText(field.selector) }
      : field
  );
}

// Action history echoes model-produced strings. If the model ever repeats an
// entity it read from an undetected source (e.g. text inside an image), scrub
// it here instead of letting the outbound gate block the whole next round.
function prepareModelActionHistory(actionHistory) {
  return (actionHistory ?? []).map((entry) => ({
    ...entry,
    selector: sanitizePromptText(entry.selector),
    target: sanitizePromptText(entry.target),
    reasoning: sanitizePromptText(entry.reasoning)
  }));
}

// Plan progress travels with every request so the model continues the plan
// instead of re-deriving it from raw action history. Every string is page- or
// model-derived, so it goes through the same sanitizer as the task.
function prepareModelStepContext(stepContext) {
  if (!stepContext) {
    return {};
  }

  return {
    completed_steps: (stepContext.completed_steps ?? []).map(sanitizePromptText),
    remaining_steps: (stepContext.remaining_steps ?? []).map(sanitizePromptText),
    last_action_result: sanitizePromptText(stepContext.last_action_result || "") || null,
    current_url: sanitizePromptText(stepContext.current_url || "") || null
  };
}

// Last gate before any fetch. Everything except the (already pixel-redacted)
// screenshot is walked; a surviving match means redaction did not do its job,
// so the request is blocked rather than sent.
function assertOutboundPayloadClean(body, label) {
  const findings = piiText.findPiiInPayload(body, ["image"]);

  if (findings.length > 0) {
    const summary = findings
      .map((finding) => `${finding.type} at ${finding.path}`)
      .join(", ");
    throw new Error(
      `Fail-Closed Privacy Error: ${label} payload still contains unredacted PII (${summary}). Request blocked.`
    );
  }

  return body;
}

async function requestAgentAction(imageDataUrl, piiFields, elements, task, actionHistory = [], failedAction = null, taskPlan = [], stepContext = null) {
  let response;
  const sanitizedTask = sanitizePromptText(task);
  const modelElements = prepareModelElements(elements);

  try {
    const requestBody = assertOutboundPayloadClean({
      image: imageDataUrl,
      pii_fields: prepareModelPiiFields(piiFields),
      elements: modelElements,
      task: sanitizedTask,
      action_history: prepareModelActionHistory(actionHistory),
      ...(failedAction ? { failed_action: sanitizePromptText(failedAction) } : {}),
      task_plan: taskPlan.map(sanitizePromptText),
      ...prepareModelStepContext(stepContext),
      provider: providerSelect.value
    }, "action");

    const serverStart = performance.now();
    response = await fetch(analyzeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody)
    });
    performanceMetrics.serverRoundTripMs = performance.now() - serverStart;
    performanceMetrics.imageSizeKb = imageDataUrl.length / 1024;
    updatePerformanceStats();
  } catch (error) {
    throw new Error(`Unable to reach the agent server. Is it running at ${analyzeUrl}? ${error.message}`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`The agent server returned invalid JSON (HTTP ${response.status}).`);
  }

  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : "Request failed.";
    throw new Error(`Agent server error (HTTP ${response.status}): ${detail}`);
  }

  if (Array.isArray(payload.actions) && payload.actions.length > 0) {
    const invalidStep = payload.actions.find(
      (step) => !step.action || !step.selector || typeof step.reasoning !== "string"
    );

    if (invalidStep) {
      throw new Error("The agent server returned an incomplete action step.");
    }

    return {
      ...payload,
      actions: payload.actions.slice(0, 10)
    };
  }

  // A re-plan request or a pure progress report (done_through) legitimately
  // carries no action.
  const reportsProgressOnly = payload.replan === true || Number.isInteger(payload.done_through);

  if (!payload.complete && !reportsProgressOnly && (!payload.action || !payload.selector || typeof payload.reasoning !== "string")) {
    throw new Error("The agent server returned an incomplete action.");
  }

  return {
    ...payload,
    actions: payload.action && payload.selector ? [payload] : []
  };
}

async function requestTaskPlan(imageDataUrl, piiFields, elements, task, planContext = null) {
  const sanitizedTask = sanitizePromptText(task);
  const modelElements = prepareModelElements(elements);

  let response;
  try {
    response = await fetch(analyzeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(assertOutboundPayloadClean({
        image: imageDataUrl,
        pii_fields: prepareModelPiiFields(piiFields),
        elements: modelElements,
        task: sanitizedTask,
        plan: true,
        ...(planContext ? { task_plan: (planContext.task_plan ?? []).map(sanitizePromptText) } : {}),
        ...prepareModelStepContext(planContext),
        provider: providerSelect.value
      }, "plan"))
    });
  } catch (error) {
    throw new Error(`Unable to reach agent server for planning: ${error.message}`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`Plan response was not valid JSON (HTTP ${response.status}).`);
  }

  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : "Planning failed.";
    throw new Error(`Planning error (HTTP ${response.status}): ${detail}`);
  }

  if (!Array.isArray(payload.steps) || payload.steps.length === 0) {
    throw new Error("Agent returned an empty plan.");
  }

  return payload;
}

async function verifyTaskCompletion(task, actionHistory = [], options = {}) {
  try {
    statusMessage.textContent = "Verifying task outcome...";
    // The agent loop settles and captures a fresh observation itself, so the
    // page text read below and that observation describe the same moment.
    if (!options.skipInitialWait) {
      await wait(VERIFY_SETTLE_MS);
    }
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    let pageContext = { title: activeTab?.title || "", url: activeTab?.url || "" };

    try {
      const results = await chrome.scripting.executeScript({
        target: { tabId: activeTab.id },
        func: () => {
          // The first 800 chars of <body> are usually header/navigation, which
          // is why the verifier kept "seeing" the listing page. Prefer the main
          // landmark for the verifier; keep bodyText for the login shortcuts.
          const main = document.querySelector("main, [role='main']");

          const isShown = (element) => {
            const rect = element.getBoundingClientRect();

            if (rect.width <= 0 || rect.height <= 0 ||
                rect.right <= 0 || rect.bottom <= 0 || rect.left >= innerWidth || rect.top >= innerHeight) {
              return false;
            }

            const style = getComputedStyle(element);
            return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) !== 0;
          };

          const selectorFor = (element) => {
            const tag = element.tagName.toLowerCase();

            if (element.id) {
              return `#${CSS.escape(element.id)}`;
            }

            for (const attribute of ["data-testid", "name", "aria-label"]) {
              const value = element.getAttribute(attribute);
              if (value) {
                return `${tag}[${attribute}="${CSS.escape(value)}"]`;
              }
            }

            return tag;
          };

          const labelFor = (element) => (
            element.getAttribute("aria-label") || element.getAttribute("title") || element.innerText ||
            element.getAttribute("placeholder") || element.getAttribute("name") || ""
          ).replace(/\s+/g, " ").trim().slice(0, 60);

          // Open dialogs / drawers / modals, by semantics first and by common UI
          // vocabulary second. Only visible, container-sized, outermost matches.
          const overlayElements = [];
          const overlayCandidates = document.querySelectorAll(
            "dialog[open], [role='dialog'], [role='alertdialog'], [aria-modal='true'], " +
            "[class*='drawer' i], [class*='modal' i], [class*='offcanvas' i], [class*='popover' i], " +
            "[id*='drawer' i], [id*='modal' i]"
          );

          for (const element of overlayCandidates) {
            if (overlayElements.length >= 5) {
              break;
            }

            const tag = element.tagName.toLowerCase();
            const rect = element.getBoundingClientRect();

            if (element === document.body || element === document.documentElement ||
                ["button", "a", "input", "label", "span", "img", "svg", "i"].includes(tag) ||
                element.getAttribute("aria-hidden") === "true" ||
                rect.width < 150 || rect.height < 100 || !isShown(element) ||
                overlayElements.some((outer) => outer.contains(element))) {
              continue;
            }

            overlayElements.push(element);
          }

          const overlays = overlayElements.map((element) => ({
            selector: selectorFor(element),
            role: element.getAttribute("role") || (element.tagName === "DIALOG" ? "dialog" : ""),
            label: (element.getAttribute("aria-label") ||
              element.querySelector("h1, h2, h3, h4, [role='heading']")?.innerText || "")
              .replace(/\s+/g, " ").trim().slice(0, 60)
          }));

          const expanded = [];
          for (const element of document.querySelectorAll("[aria-expanded='true']")) {
            if (expanded.length >= 8) {
              break;
            }
            if (isShown(element)) {
              expanded.push({ selector: selectorFor(element), label: labelFor(element) });
            }
          }

          const numericControls = [];
          const checkedControls = [];
          for (const control of document.querySelectorAll("input, select")) {
            const type = (control.type || "").toLowerCase();

            if (type === "hidden" || type === "password" || !isShown(control)) {
              continue;
            }

            if (type === "checkbox" || type === "radio") {
              if (checkedControls.length < 10) {
                checkedControls.push({ selector: selectorFor(control), label: labelFor(control), type, checked: control.checked });
              }
              continue;
            }

            const value = String(control.value ?? "").trim();

            // Only short numeric values (quantities, counts) are read: free-text
            // values can hold PII the sanitizer may not recognise.
            if (numericControls.length < 20 && /^\d{1,4}$/.test(value)) {
              numericControls.push({ selector: selectorFor(control), label: labelFor(control), type, value });
            }
          }

          // Controls inside open overlays first, so a long page header cannot
          // crowd them out of the bounded list.
          const actionControls = [];
          const listedControls = new Set();
          const actionSelector = "button, a[href], [role='button'], input[type='submit'], input[type='button']";

          for (const root of [...overlayElements, document]) {
            for (const element of root.querySelectorAll(actionSelector)) {
              if (actionControls.length >= 30 || listedControls.has(element) || !isShown(element)) {
                continue;
              }

              listedControls.add(element);
              const label = element.tagName === "INPUT"
                ? String(element.value || "").trim().slice(0, 60)
                : labelFor(element);

              if (label) {
                actionControls.push({ tag: element.tagName.toLowerCase(), selector: selectorFor(element), label });
              }
            }
          }

          return {
            title: document.title,
            url: location.href,
            bodyText: document.body?.innerText?.slice(0, 800) || "",
            contentText: (main?.innerText || document.body?.innerText || "").slice(0, 800),
            overlays,
            expanded,
            numericControls,
            checkedControls,
            actionControls
          };
        }
      });
      if (results?.[0]?.result) {
        pageContext = results[0].result;
      }
    } catch {
      // fallback to tab title/url only
    }

    console.log("[VERIFY] Page context captured:", pageContext.url, pageContext.title);
    const currentUrl = pageContext.url || "";
    const currentTitle = pageContext.title || "";
    
    // If URL changed away from any login/signin page, consider it success
    const wasLoginPage = currentUrl.includes("/login") || 
                         currentUrl.includes("/signin") ||
                         currentUrl.includes("/auth") ||
                         currentUrl.includes("practice-test-login");
    
    console.log("[VERIFY] wasLoginPage:", wasLoginPage, "url:", currentUrl);
    
    // Only use URL shortcut for simple login-only tasks
    const taskLower = task.toLowerCase();
    const isAuthTask = taskLower.includes("login") || 
                       taskLower.includes("log in") || 
                       taskLower.includes("sign in") || 
                       taskLower.includes("signin") ||
                       taskLower.includes("auth");
    
    const hasPostLoginSteps = taskLower.includes("logout") || 
                              taskLower.includes("log out") ||
                              taskLower.includes("then ") ||
                              taskLower.includes("after") ||
                              taskLower.includes("next") ||
                              taskLower.includes("and then") ||
                              taskLower.includes("also");
    const requestsLogout = taskLower.includes("logout") || taskLower.includes("log out");
    const logoutWasCompleted = requestsLogout && actionHistory.some((action) =>
      action.action === "click" && /log\s*out|sign\s*out/i.test(`${action.target} ${action.selector}`)
    );
    const visibleText = (pageContext.bodyText || "").toLowerCase();
    const hasLoginSuccessMessage = /(?:login|log\s*in|sign\s*in)\s+(?:was\s+)?(?:successful|success|complete|completed)|(?:you(?:'|’)re|you\s+are)\s+(?:now\s+)?(?:logged\s*in|signed\s*in)/i.test(visibleText);

    if (wasLoginPage && logoutWasCompleted) {
      console.log("[VERIFY] Logout completed: returned to login page");
      return { success: true, reason: "Task completed: logout returned to the login page." };
    }

    if (isAuthTask && !hasPostLoginSteps && hasLoginSuccessMessage) {
      console.log("[VERIFY] Page-based success: login success message detected");
      return { success: true, reason: "Task completed: page confirms a successful login." };
    }

    if (isAuthTask && !wasLoginPage && !hasPostLoginSteps) {
      console.log("[VERIFY] URL-based success: redirected away from login");
      return { success: true, reason: `Task completed. Current page: ${currentTitle} (${currentUrl})` };
    }

    const verificationEvidence = globalThis.__privacyVisionVerificationEvidence;

    if (!verificationEvidence) {
      throw new Error("privacy-vision-agent: verification-evidence.js must load before popup.js");
    }

    // Numbered, fresh DOM evidence from the same moment as the screenshot in
    // options.observation. Completed steps come back separately as claims.
    const evidence = verificationEvidence.buildVerificationEvidence(pageContext, {
      elements: prepareModelElements(options.observation?.elements ?? []),
      completedSteps: options.completedSteps ?? []
    });
    const evidenceItems = evidence.items.map((item) => ({ ...item, text: sanitizePromptText(item.text) }));

    // Page text and the task are raw user/page data. Scrub them with the same
    // sanitizer the action and plan requests use before anything leaves the device.
    const verifyText = sanitizePromptText(
      `Page title: ${currentTitle}\n` +
      `URL: ${currentUrl}\n` +
      `Visible text (first 800 chars): ${pageContext.contentText ?? pageContext.bodyText}`
    );

    // The observation's image is the already pixel-redacted, outbound-validated capture.
    const verifyImage = typeof options.observation?.image === "string" ? options.observation.image : "";

    let response;
    try {
      response = await fetch(analyzeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(assertOutboundPayloadClean({
          task: sanitizePromptText(task),
          page_context: verifyText,
          ...(verifyImage ? { image: verifyImage } : {}),
          completed_steps: evidence.claims.map(sanitizePromptText),
          verification_evidence: evidenceItems,
          provider: providerSelect.value,
          verify: true
        }, "verify"))
      });
    } catch (err) {
      console.error("[VERIFY] fetch failed:", err);
      return { success: false, reason: "Verification network error: " + err.message };
    }

    let payload;
    try {
      payload = await response.json();
    } catch (err) {
      console.error("[VERIFY] response.json() failed:", err);
      return { success: false, reason: "Verification response was not JSON" };
    }

    console.log("[VERIFY] HTTP status:", response.status);
    console.log("[VERIFY] Response payload:", JSON.stringify(payload));

    if (!response.ok) {
      console.error("[VERIFY] Non-ok response:", payload);
      return { success: false, reason: "Verification request failed: " + (payload?.detail || response.status) };
    }

    if (typeof payload.success !== "boolean") {
      console.error("[VERIFY] Unexpected payload shape:", payload);
      return { success: false, reason: "Verification returned unexpected response shape" };
    }

    return { success: payload.success, reason: payload.reason || "" };
  } catch (err) {
    console.error("[VERIFY] Unexpected error:", err);
    return { success: false, reason: "Verification error: " + err.message };
  }
}

const ALLOWED_ACTIONS = new Set(["click", "fill", "scroll", "wait", "press"]);
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// A click can replace the document. chrome.scripting.executeScript resolves
// against whichever document is committed at call time, so a DOM scan issued
// before the new page commits silently returns the PREVIOUS page's elements --
// while captureVisibleTab may already be painting the new one. Block until the
// tab reports a settled document before any capture that follows an action.
async function waitForTabSettled(tabId, timeoutMs = 10000) {
  if (!tabId) {
    return null;
  }

  const deadline = Date.now() + timeoutMs;
  let lastUrl = null;

  while (Date.now() < deadline) {
    let tab;
    try {
      tab = await chrome.tabs.get(tabId);
    } catch {
      return lastUrl;
    }

    lastUrl = tab?.url ?? lastUrl;

    if (tab?.status === "complete") {
      try {
        const [probe] = await chrome.scripting.executeScript({
          target: { tabId },
          func: () => ({ readyState: document.readyState, url: location.href })
        });

        const readyState = probe?.result?.readyState;
        if (readyState === "complete" || readyState === "interactive") {
          return probe.result.url ?? lastUrl;
        }
      } catch {
        // The document is swapping under us; keep polling until the deadline.
      }
    }

    await wait(150);
  }

  return lastUrl;
}

const UNSAFE_SELECTOR_PATTERNS = [
  /javascript:/i,
  /<script/i,
  /\beval\s*\(/i,
  /\bwindow\./i,
  /\bonerror\s*=/i,
  /\bonload\s*=/i,
  /\bdocument\.cookie/i
];

function validateActionSafety(action) {
  if (!action || !action.action || !ALLOWED_ACTIONS.has(action.action)) {
    throw new Error(`Action rejected: action "${action?.action}" is not allowed.`);
  }

  const selector = action.selector || "";
  for (const pattern of UNSAFE_SELECTOR_PATTERNS) {
    if (pattern.test(selector)) {
      throw new Error(`Action rejected: selector contains unsafe pattern "${selector}".`);
    }
  }

  const textToTest = `${action.selector || ""} ${action.target || ""} ${action.reasoning || ""}`;
  const isRiskySubmit = action.action === "click" && /submit|delete|remove|pay|buy|purchase|send|confirm|checkout/i.test(textToTest);
  const isRiskyCredentialFill = action.action === "fill" && /password|pwd|ssn|creditcard|cardnumber|cvv|cvc/i.test(textToTest) && action.value;

  return {
    isRisky: isRiskySubmit || isRiskyCredentialFill,
    riskReason: isRiskySubmit ? "Form submission / payment / data modification" : (isRiskyCredentialFill ? "Direct credential or sensitive field fill" : null)
  };
}

async function executeAgentAction(tabId, action) {
  if (!tabId) {
    throw new Error("Could not identify the active tab.");
  }

  const resolvedAction = action.action === "fill"
    ? { ...action, value: await resolveCredentialValueForAction(action) }
    : action;
  const safety = validateActionSafety(resolvedAction);
  if (safety.isRisky) {
    const userConfirmed = globalThis.confirm?.(
      `⚠️ Risky Action Warning!\n\nThe agent is attempting to execute: ${resolvedAction.action.toUpperCase()}\nTarget: ${resolvedAction.target || resolvedAction.selector}\nRisk: ${safety.riskReason}\n\nDo you authorize this action?`
    ) ?? false;

    if (!userConfirmed) {
      throw new Error("Action cancelled by user safety confirmation.");
    }
  }

  const target = Number.isInteger(resolvedAction.frame_id)
    ? { tabId, frameIds: [resolvedAction.frame_id] }
    : { tabId, allFrames: true };
  const injectionResults = await chrome.scripting.executeScript({
    target,
    func: ({ action: actionName, selector, value, task, reasoning, target }) => {
      function normalizeText(text) {
        return (text ?? "")
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, " ")
          .replace(/\s+/g, " ")
          .trim();
      }

      function cssEscape(value) {
        if (window.CSS?.escape) {
          return CSS.escape(value);
        }

        return value.replace(/["\\]/g, "\\$&");
      }

      function getVisibleRect(element) {
        const rect = element.getBoundingClientRect();

        if (rect.width <= 0 || rect.height <= 0) {
          return null;
        }

        const style = window.getComputedStyle(element);

        if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) {
          return null;
        }

        const x = Math.max(rect.x, 0);
        const y = Math.max(rect.y, 0);
        const right = Math.min(rect.right, window.innerWidth);
        const bottom = Math.min(rect.bottom, window.innerHeight);

        if (right - x <= 0 || bottom - y <= 0) {
          return null;
        }

        return rect;
      }

      function getElementLabel(element) {
        return [
          element.innerText,
          element.getAttribute("aria-label"),
          element.getAttribute("title"),
          element.getAttribute("placeholder"),
          element.value,
          element.getAttribute("name"),
          element.id
        ]
          .filter((label) => typeof label === "string")
          .map(normalizeText)
          .filter(Boolean)
          .join(" ");
      }

      function getTaskTargets() {
        const quotedText = `${selector ?? ""} ${reasoning ?? ""} ${target ?? ""}`
          .match(/"([^"]{2,60})"/g)
          ?.map((match) => normalizeText(match.slice(1, -1))) ?? [];
        const taskTarget = normalizeText(target || task)
          .replace(/\b(click|press|tap|select|choose|open|go|on|the|a|an|button|link|field|input|please|fill|type|enter|with|into)\b/g, " ")
          .replace(/\s+/g, " ")
          .trim();
        const selectorText = normalizeText(selector)
          .replace(/\b(html|body|div|span|button|input|form|main|section|type|submit|text|password)\b/g, " ")
          .replace(/\s+/g, " ")
          .trim();
        const explicitTarget = normalizeText(target);

        return [explicitTarget, taskTarget, ...quotedText, selectorText]
          .flatMap((target) => {
            if (!target) {
              return [];
            }

            const compact = target.replace(/\s+/g, "");
            return compact && compact !== target ? [target, compact] : [target];
          })
          .filter((target, index, targets) => target.length >= 2 && targets.indexOf(target) === index);
      }

      function isClickableElement(element) {
        const tagName = element.tagName.toLowerCase();

        return (
          tagName === "button" ||
          (tagName === "a" && element.hasAttribute("href")) ||
          (tagName === "input" && ["button", "submit", "reset"].includes((element.type || "").toLowerCase())) ||
          (element.getAttribute("role") || "").toLowerCase() === "button" ||
          element.hasAttribute("onclick")
        );
      }

      function isFillableElement(element) {
        const tagName = element.tagName.toLowerCase();
        return tagName === "input" || tagName === "textarea" || tagName === "select" || element.isContentEditable;
      }

      function getAssociatedLabelText(element) {
        const id = element.id;
        const labels = [];

        if (element.labels) {
          labels.push(...Array.from(element.labels).map((label) => label.innerText));
        }

        if (id) {
          try {
            labels.push(
              ...Array.from(document.querySelectorAll(`label[for="${cssEscape(id)}"]`)).map((label) => label.innerText)
            );
          } catch (error) {
            // Ignore label lookup failures from unusual IDs; other attributes still help.
          }
        }

        return labels.join(" ");
      }

      function findClickableByText() {
        const targets = getTaskTargets();

        if (targets.length === 0) {
          return null;
        }

        const candidates = Array.from(
          document.querySelectorAll('button, a[href], input[type="button"], input[type="submit"], input[type="reset"], [role="button"], [onclick]')
        ).filter((element) => isClickableElement(element) && getVisibleRect(element));
        let bestMatch = null;
        let bestScore = 0;

        for (const element of candidates) {
          const label = getElementLabel(element);

          if (!label) {
            continue;
          }

          for (const target of targets) {
            let score = 0;

            if (label === target) {
              score = 100;
            } else if (label.replace(/\s+/g, "") === target) {
              score = 95;
            } else if (label.includes(target)) {
              score = 80;
            } else if (target.includes(label) && label.length >= 3) {
              score = 60;
            }

            if (score > bestScore) {
              bestScore = score;
              bestMatch = element;
            }
          }
        }

        return bestMatch;
      }

      function findFillableByText() {
        const targets = getTaskTargets();

        if (targets.length === 0) {
          return null;
        }

        const candidates = Array.from(document.querySelectorAll("input, textarea, select, [contenteditable]"))
          .filter((element) => isFillableElement(element) && getVisibleRect(element));
        let bestMatch = null;
        let bestScore = 0;

        for (const element of candidates) {
          const label = [
            getElementLabel(element),
            getAssociatedLabelText(element),
            (element.type || "").toLowerCase()
          ]
            .map(normalizeText)
            .filter(Boolean)
            .join(" ");

          if (!label) {
            continue;
          }

          for (const target of targets) {
            let score = 0;

            if (label === target) {
              score = 100;
            } else if (label.includes(target)) {
              score = 85;
            } else if (target.includes(label) && label.length >= 3) {
              score = 55;
            }

            if (score > bestScore) {
              bestScore = score;
              bestMatch = element;
            }
          }
        }

        return bestMatch;
      }

      function querySelector(selectorText) {
        const selectorParts = (selectorText || "body").split("::shadow");
        let root = document;
        let element = null;

        for (let index = 0; index < selectorParts.length; index += 1) {
          const part = selectorParts[index];
          element = root.querySelector(part.trim());
          if (!element) {
            return null;
          }
          root = element.shadowRoot;
          if (index < selectorParts.length - 1 && !root) {
            return null;
          }
        }

        return element;
      }

      let element;
      let usedFallback = false;

      if (actionName === "wait") {
        return { found: true };
      }

      try {
        element = querySelector(selector);
      } catch (error) {
        if (actionName !== "click" && actionName !== "scroll" && actionName !== "fill" && actionName !== "press") {
          return { found: false, error: `Invalid CSS selector: ${error.message}` };
        }
      }

      if (!element) {
        if (actionName === "fill") {
          element = findFillableByText();
          usedFallback = Boolean(element);
        } else if (actionName === "click" || actionName === "scroll") {
          element = findClickableByText();
          usedFallback = Boolean(element);
        } else if (actionName === "press") {
          element = document.activeElement || document.body;
        }
      }

      if (!element) {
        return { found: false };
      }

      if (actionName === "click") {
        if (typeof element.click !== "function") {
          return { found: true, error: "The selected element cannot be clicked." };
        }
        element.scrollIntoView({ behavior: "auto", block: "center", inline: "center" });
        element.focus?.();
        element.click();
      } else if (actionName === "fill") {
        if (!isFillableElement(element)) {
          return { found: true, error: "The selected element cannot be filled." };
        }

        const normalizedValue = normalizeText(value);
        const tagName = element.tagName.toLowerCase();
        const type = (element.type || "").toLowerCase();

        element.focus?.();

        if (tagName === "select") {
          const options = Array.from(element.options);
          const matchingOption = options.find((option) => {
            const optionText = normalizeText(`${option.label} ${option.textContent} ${option.value}`);
            return optionText === normalizedValue || optionText.includes(normalizedValue);
          });

          if (matchingOption) {
            element.value = matchingOption.value;
          } else {
            return { found: true, error: `No select option matched: ${value}` };
          }
        } else if (tagName === "input" && ["checkbox", "radio"].includes(type)) {
          const truthyValues = new Set(["1", "true", "yes", "y", "on", "checked", "select"]);
          const falsyValues = new Set(["0", "false", "no", "n", "off", "unchecked", "clear"]);

          if (truthyValues.has(normalizedValue)) {
            element.checked = true;
          } else if (falsyValues.has(normalizedValue)) {
            element.checked = false;
          } else {
            element.checked = true;
          }
        } else if (element.isContentEditable) {
          element.textContent = value ?? "";
        } else {
          const valueSetter = Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(element),
            "value"
          )?.set;

          if (valueSetter) {
            valueSetter.call(element, value ?? "");
          } else {
            element.value = value ?? "";
          }
        }

        element.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
      } else if (actionName === "scroll") {
        if (selector === "body" || selector === "html" || element === document.body || element === document.documentElement) {
          window.scrollBy({ top: Math.round(window.innerHeight * 0.75), behavior: "smooth" });
        } else {
          element.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      } else if (actionName === "press") {
        const key = value || "Enter";
        const targetElement = element || document.activeElement || document.body;
        targetElement.focus?.();
        for (const eventType of ["keydown", "keyup"]) {
          targetElement.dispatchEvent(new KeyboardEvent(eventType, {
            key,
            code: key.length === 1 ? `Key${key.toUpperCase()}` : key,
            bubbles: true,
            cancelable: true,
            composed: true
          }));
        }
      } else {
        return { found: true, error: `Unsupported action: ${actionName}` };
      }

      return { found: true, usedFallback };
    },
    args: [resolvedAction]
  });

  const successfulResult = injectionResults.find(({ result }) => result?.found && !result.error);
  const actionError = injectionResults.find(({ result }) => result?.error)?.result?.error;

  if (actionError && !successfulResult) {
    throw new Error(actionError);
  }

  if (!successfulResult) {
    throw new Error(`No element matched the selector: ${action.selector}`);
  }

  debugLog("executed", {
    action: resolvedAction.action,
    selector: resolvedAction.selector,
    target: resolvedAction.target ?? null,
    frame_id: resolvedAction.frame_id ?? null,
    // The value may be a resolved credential by now: never log it.
    value: resolvedAction.value == null ? null : "[value omitted]",
    usedTextFallback: Boolean(successfulResult.result?.usedFallback),
    executedInFrameId: successfulResult.frameId ?? null
  });
}

async function executeAgentActions(tabId, response, task, actionHistory = [], taskPlan = [], stepContext = null) {
  const actions = response.actions ?? [response];
  const executedActions = [];
  let deferToNextRound = false;

  for (let index = 0; index < actions.length; index += 1) {
    const action = {
      ...actions[index],
      task
    };
    let lastError;
    const tabBeforeAction = await getActiveTab();
    const actionTabId = tabBeforeAction?.id ?? tabId;
    const urlBeforeAction = tabBeforeAction?.url ?? "";

    statusMessage.textContent = `Executing step ${index + 1} of ${actions.length}: ${action.action}`;

    if (action.action === "wait") {
      await wait(900);
      executedActions.push(action);
      debugLog("executed", {
        action: "wait",
        selector: action.selector ?? null,
        target: action.target ?? null,
        frame_id: action.frame_id ?? null,
        value: null,
        usedTextFallback: false
      });
      continue;
    }

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      try {
        const currentTab = await getActiveTab();
        const currentTabId = currentTab?.id ?? tabId;
        await executeAgentAction(currentTabId, action);
        executedActions.push(action);
        lastError = null;
        break;
      } catch (error) {
        lastError = error;

        if (attempt === 1) {
          const settledUrl = await waitForTabSettled(actionTabId);

          if (settledUrl && urlBeforeAction && settledUrl !== urlBeforeAction) {
            console.warn(
              `[SELF-CORRECT] Skipped: page navigated to ${settledUrl}. Deferring to the next round so the model sees fresh DOM metadata.`
            );
            statusMessage.textContent = "Page changed — re-reading it before continuing...";
            deferToNextRound = true;
            lastError = null;
            break;
          }

          // First failure on the same page - try self-correction before giving up
          try {
            statusMessage.textContent = `Step failed: ${error.message}. Asking agent to retry differently...`;
            console.warn(`[SELF-CORRECT] Action failed: ${error.message}. Requesting correction.`);
            
            // Re-capture screen so agent sees current state
            await captureRedactedScreenshot();
            await wait(500);
            
            // Ask VLM to try again with failure context
            const correctedResponse = await requestAgentAction(
              getModelImageDataUrl(),
              latestPiiFields,
              latestElements,
              action.task || task,
              actionHistory,
              `Action failed: "${action.action}" on selector "${action.selector}" — error: ${error.message}`,
              taskPlan,
              stepContext
            );
            
            if (correctedResponse.actions && correctedResponse.actions.length > 0) {
              // Try the corrected action(s)
              for (const correctedAction of correctedResponse.actions.slice(0, 1)) {
                const correctedTab = await getActiveTab();
                const correctedTabId = correctedTab?.id ?? tabId;
                await executeAgentAction(correctedTabId, { ...correctedAction, task });
                executedActions.push({ ...correctedAction, task });
              }
              lastError = null;
              break;
            }
          } catch (correctionError) {
            console.warn(`[SELF-CORRECT] Correction also failed: ${correctionError.message}`);
            lastError = correctionError;
          }
        } else {
          await wait(500);
        }
      }
    }

    if (deferToNextRound) {
      break;
    }

    if (lastError) {
      throw lastError;
    }

    // A click or key press can replace the document. Wait for any resulting
    // navigation to commit, then stop this round: the next round re-scans the
    // current page instead of applying this page's stale selectors.
    if (["click", "press"].includes(executedActions.at(-1)?.action)) {
      await waitForTabSettled(actionTabId);
      break;
    }

    if (index < actions.length - 1) {
      await wait(450);
    }
  }

  return executedActions;
}

async function runCapture() {
  captureButton.disabled = true;
  runAgentButton.disabled = true;
  statusMessage.textContent = "Capturing...";

  try {
    await captureRedactedScreenshot();
    updatePerformanceStats();
  } catch (error) {
    statusMessage.textContent = error.message;
    console.error(error);
  } finally {
    captureButton.disabled = false;
    runAgentButton.disabled = false;
  }
}

async function runAgent() {
  const task = taskInput.value.trim();

  if (!task) {
    agentResult.hidden = false;
    agentResult.textContent = "Enter a task before running the agent.";
    taskInput.focus();
    return;
  }

  captureButton.disabled = true;
  runAgentButton.disabled = true;
  agentResult.hidden = false;
  agentResult.textContent = "";
  latestTaskText = sanitizePromptText(task);
  ui?.agentStarted(latestTaskText, latestTaskText !== task);
  statusMessage.textContent = "Capturing...";

  try {
    const agentLoop = globalThis.__privacyVisionAgentLoop;

    if (!agentLoop) {
      throw new Error("privacy-vision-agent: agent-loop.js must load before popup.js");
    }

    const maxRounds = 5;
    const actionHistory = [];
    let activeTab = null;

    // OBSERVE: settle on the current document, then run the full local privacy
    // pipeline. Action selection, step verification and goal verification all
    // consume observations produced here, so none of them can use stale state.
    const observe = async ({ purpose }) => {
      if (purpose === "verify") {
        await wait(VERIFY_SETTLE_MS);
      }

      if (activeTab?.id) {
        await waitForTabSettled(activeTab.id);
      }

      const capture = await captureRedactedScreenshot();
      activeTab = capture.activeTab;

      return {
        url: activeTab?.url ?? "",
        signature: agentLoop.pageSignature(activeTab?.url, latestElements),
        image: getModelImageDataUrl(),
        piiFields: latestPiiFields,
        elements: latestElements
      };
    };

    const renderPlan = (state) => {
      const lines = state.plan.map((step, index) => {
        const executedThrough = Math.max(state.executedThroughIndex ?? 0, state.currentStepIndex);
        const marker = index < state.currentStepIndex
          ? "✓"
          : index === state.currentStepIndex
            ? "→"
            : index < executedThrough ? "◐" : "·";
        return `${marker} ${index + 1}. ${step.description}`;
      });

      agentResult.textContent = `Plan:\n${lines.join("\n")}${state.lastActionResult ? `\n\nLast: ${state.lastActionResult}` : ""}`;
    };

    const result = await agentLoop.runAgentLoop({
      goal: task,
      maxRounds,
      observe,
      requestPlan: async ({ observation, context }) => {
        const plan = await requestTaskPlan(observation.image, observation.piiFields, observation.elements, task, context);
        return plan.steps;
      },
      // Selection feedback (stale selector / present target) rides the existing
      // sanitized failed_action channel.
      requestAction: ({ observation, context, feedback }) => requestAgentAction(
        observation.image,
        observation.piiFields,
        observation.elements,
        task,
        actionHistory,
        feedback ?? null,
        context.task_plan,
        context
      ),
      executeActions: async ({ response, context, round }) => {
        const executedActions = await executeAgentActions(activeTab?.id, response, task, actionHistory, context.task_plan, context);

        for (const action of executedActions) {
          actionHistory.push({
            round,
            action: action.action,
            selector: action.selector,
            frame_id: action.frame_id ?? null,
            target: action.target || "",
            value: action.value ? "[value provided]" : "",
            reasoning: action.reasoning || ""
          });
        }

        return executedActions;
      },
      verifyGoal: ({ observation, completedSteps }) => verifyTaskCompletion(task, actionHistory, {
        skipInitialWait: true,
        observation,
        completedSteps
      }),
      onStatus: (text) => {
        statusMessage.textContent = text;
      },
      onDebug: debugLog,
      onState: (state) => {
        renderPlan(state);
        ui?.agentState(state);
        updatePerformanceStats();
      }
    });

    if (result.status === "complete") {
      statusMessage.textContent = "✅ Task verified complete.";
    } else if (result.status === "failed") {
      statusMessage.textContent = `❌ Stopped: ${result.reason}`;
    } else {
      statusMessage.textContent = `⚠️ Task may be incomplete after ${maxRounds} rounds: ${result.reason}`;
    }
    ui?.agentFinished(result);
  } catch (error) {
    statusMessage.textContent = error.message;
    agentResult.textContent = `Error: ${error.message}`;
    ui?.agentFailed(error.message);
    console.error(error);
  } finally {
    captureButton.disabled = false;
    runAgentButton.disabled = false;
  }
}

captureButton.addEventListener("click", runCapture);
runAgentButton.addEventListener("click", runAgent);
saveCredentialsButton.addEventListener("click", saveCredentialVault);
loadCredentialVault();
