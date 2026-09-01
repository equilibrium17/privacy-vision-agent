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

const analyzeUrl = "http://127.0.0.1:8000/analyze";

const modelPath = chrome.runtime.getURL("models/blaze_face_full_range.tflite");
const wasmPath = chrome.runtime.getURL("vendor/mediapipe/wasm");
const faceBlurRadius = 16;
const minDetectionConfidence = 0.5;
const MAX_INTERACTIVE_ELEMENTS = 20;
const MAX_ELEMENT_TEXT_LENGTH = 30;

let faceDetectorPromise;
let originalScreenshotDataUrl = "";
let latestPiiFields = [];
let latestElements = [];
const performanceMetrics = {
  faceModelLoadMs: null,
  faceDelegate: "CPU",
  detectionMs: null,
  domScanMs: null,
  imageSizeKb: null,
  serverRoundTripMs: null
};

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
    faceDetectorPromise = (async () => {
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
  const maxDimension = 800;
  const scale = Math.min(1, maxDimension / Math.max(screenshotCanvas.width, screenshotCanvas.height));
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
    .slice(0, MAX_INTERACTIVE_ELEMENTS)
    .map((element) => {
      const compactElement = { ...element };

      for (const textField of ["text", "label"]) {
        if (typeof compactElement[textField] === "string") {
          compactElement[textField] = compactElement[textField].slice(0, MAX_ELEMENT_TEXT_LENGTH);
        }
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
      viewport: null
    };
  }

  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      files: ["content.js"]
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
      viewport: null
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
  const topFrame = frameResults[0]?.result;
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
    viewport: topFrame?.viewport ?? null
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

    const modelImageDataUrl = getModelImageDataUrl();
    performanceMetrics.imageSizeKb = modelImageDataUrl.length / 1024;
    updatePerformanceStats();

    statusMessage.textContent = `${result.detections.length} face(s) blurred, ${piiScan.fields.length} field(s) redacted`;
    return { activeTab, piiScan, sanitizedDataUrl: outboundCheck.sanitizedDataUrl };
  } catch (error) {
    statusMessage.textContent = `Privacy Error: ${error.message}`;
    throw new Error(`Privacy Fail-Closed: Screenshot transmission blocked. (${error.message})`);
  }
}

function sanitizePromptText(text) {
  if (!text || typeof text !== "string") return text;
  return text
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[EMAIL_REDACTED]")
    .replace(/\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b/g, "[PHONE_REDACTED]")
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[SSN_REDACTED]")
    .replace(/\b(?:\d[ -]*?){13,19}\b/g, "[CARD_REDACTED]")
    .replace(/\bsk-[a-zA-Z0-9]{20,}\b/g, "[API_KEY_REDACTED]")
    .replace(/\b(?:api[_-]?key|secret|access[_-]?token|bearer)\s*[:=]\s*["']?[A-Za-z0-9_\-]{16,}["']?\b/gi, "[TOKEN_REDACTED]");
}

async function requestAgentAction(imageDataUrl, piiFields, elements, task) {
  let response;
  const sanitizedTask = sanitizePromptText(task);
  const modelElements = prepareModelElements(elements);

  try {
    const serverStart = performance.now();
    response = await fetch(analyzeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageDataUrl,
        pii_fields: piiFields,
        elements: modelElements,
        task: sanitizedTask,
        provider: providerSelect.value
      })
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

  if (!payload.complete && (!payload.action || !payload.selector || typeof payload.reasoning !== "string")) {
    throw new Error("The agent server returned an incomplete action.");
  }

  return {
    ...payload,
    actions: payload.action && payload.selector ? [payload] : []
  };
}

const ALLOWED_ACTIONS = new Set(["click", "fill", "scroll", "wait", "press"]);
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
  const isRiskyCredentialFill = action.action === "fill" && /password|pwd|ssn|creditcard|cardnumber|cvv|cvc/i.test(textToTest) && action.value && !action.value.startsWith("[TOKEN:");

  return {
    isRisky: isRiskySubmit || isRiskyCredentialFill,
    riskReason: isRiskySubmit ? "Form submission / payment / data modification" : (isRiskyCredentialFill ? "Direct credential or sensitive field fill" : null)
  };
}

async function executeAgentAction(tabId, action) {
  if (!tabId) {
    throw new Error("Could not identify the active tab.");
  }

  const safety = validateActionSafety(action);
  if (safety.isRisky) {
    const userConfirmed = globalThis.confirm?.(
      `⚠️ Risky Action Warning!\n\nThe agent is attempting to execute: ${action.action.toUpperCase()}\nTarget: ${action.target || action.selector}\nRisk: ${safety.riskReason}\n\nDo you authorize this action?`
    ) ?? true;

    if (!userConfirmed) {
      throw new Error("Action cancelled by user safety confirmation.");
    }
  }

  const injectionResults = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
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

      let element;
      let usedFallback = false;

      if (actionName === "wait") {
        return { found: true };
      }

      try {
        element = document.querySelector(selector || "body");
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
    args: [action]
  });

  const successfulResult = injectionResults.find(({ result }) => result?.found && !result.error);
  const actionError = injectionResults.find(({ result }) => result?.error)?.result?.error;

  if (actionError && !successfulResult) {
    throw new Error(actionError);
  }

  if (!successfulResult) {
    throw new Error(`No element matched the selector: ${action.selector}`);
  }
}

async function executeAgentActions(tabId, response, task) {
  const actions = response.actions ?? [response];

  for (let index = 0; index < actions.length; index += 1) {
    const action = {
      ...actions[index],
      task
    };
    let lastError;

    statusMessage.textContent = `Executing step ${index + 1} of ${actions.length}: ${action.action}`;

    if (action.action === "wait") {
      await wait(900);
      continue;
    }

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      try {
        await executeAgentAction(tabId, action);
        lastError = null;
        break;
      } catch (error) {
        lastError = error;

        if (attempt < 6) {
          await wait(500);
        }
      }
    }

    if (lastError) {
      throw lastError;
    }

    if (index < actions.length - 1) {
      await wait(450);
    }
  }
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
  statusMessage.textContent = "Capturing...";

  try {
    const { activeTab } = await captureRedactedScreenshot();
    const maxRounds = 2;
    let completed = false;

    for (let round = 1; round <= maxRounds; round += 1) {
      statusMessage.textContent = `Step ${round} of ${maxRounds}: ${round === 1 ? "Asking the agent..." : "Re-reading page..."}`;
      const response = await requestAgentAction(
        getModelImageDataUrl(),
        latestPiiFields,
        latestElements,
        task
      );
      updatePerformanceStats();
      agentResult.textContent = `${response.reasoning}${response.complete ? "\n\nTask complete." : ""}`;

      if (response.complete && (!response.actions || response.actions.length === 0)) {
        completed = true;
        statusMessage.textContent = "Task complete.";
        break;
      }

      if (!response.actions || response.actions.length === 0) {
        throw new Error("The agent server returned no actions and did not mark the task complete.");
      }

      await executeAgentActions(activeTab?.id, response, task);

      if (response.complete) {
        completed = true;
        statusMessage.textContent = "Task complete.";
        break;
      }

      if (round < maxRounds) {
        statusMessage.textContent = "Re-reading page...";
        await captureRedactedScreenshot();
      }
    }

    if (!completed) {
      statusMessage.textContent = `Stopped after ${maxRounds} steps — task may be incomplete.`;
    }
  } catch (error) {
    statusMessage.textContent = error.message;
    agentResult.textContent = `Error: ${error.message}`;
    console.error(error);
  } finally {
    captureButton.disabled = false;
    runAgentButton.disabled = false;
  }
}

captureButton.addEventListener("click", runCapture);
runAgentButton.addEventListener("click", runAgent);
