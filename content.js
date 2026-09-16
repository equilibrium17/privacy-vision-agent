(() => {
  if (window.__privacyVisionAgentContentLoaded) {
    return;
  }

  const piiText = globalThis.__privacyVisionPii;

  if (!piiText) {
    // pii-patterns.js must be injected before this script. Without it there is
    // no text-PII detection, so refuse to register a scanner that would look
    // like it had checked the page.
    throw new Error("privacy-vision-agent: pii-patterns.js was not loaded before content.js");
  }

  window.__privacyVisionAgentContentLoaded = true;

  const documentedPiiAutocompleteTokens = new Set([
    "name",
    "honorific-prefix",
    "given-name",
    "additional-name",
    "family-name",
    "honorific-suffix",
    "nickname",
    "email",
    "tel",
    "tel-country-code",
    "tel-national",
    "tel-area-code",
    "tel-local",
    "tel-local-prefix",
    "tel-local-suffix",
    "tel-extension",
    "street-address",
    "address-line1",
    "address-line2",
    "address-line3",
    "address-level1",
    "address-level2",
    "address-level3",
    "address-level4",
    "country",
    "country-name",
    "postal-code",
    "cc-name",
    "cc-given-name",
    "cc-additional-name",
    "cc-family-name",
    "cc-number",
    "cc-exp",
    "cc-exp-month",
    "cc-exp-year",
    "cc-csc",
    "cc-type",
    "transaction-currency",
    "transaction-amount",
    "bday",
    "bday-day",
    "bday-month",
    "bday-year",
    "sex",
    "url",
    "photo",
    "username",
    "current-password",
    "new-password",
    "one-time-code"
  ]);

  const piiAttributePatterns = [
    {
      pattern: /(?:^|[\s_:.\-])(?:e-?mail|emailaddress|mailaddress)(?:$|[\s_:.\-])/i,
      reason: "email-like attribute"
    },
    {
      pattern: /(?:^|[\s_:.\-])(?:tel|telephone|phone|mobile|cell|contactnumber)(?:$|[\s_:.\-])/i,
      reason: "phone-like attribute"
    },
    {
      pattern: /(?:^|[\s_:.\-])(?:ssn|socialsecurity|social-security|taxid|tax-id)(?:$|[\s_:.\-])/i,
      reason: "SSN/tax-id-like attribute"
    },
    {
      pattern: /(?:^|[\s_:.\-])(?:cc|card|cardnumber|creditcard|credit-card|ccnum|cc-number|cvc|cvv|cvn|securitycode|security-code)(?:$|[\s_:.\-])/i,
      reason: "payment-card-like attribute"
    },
    {
      pattern: /(?:^|[\s_:.\-])(?:pass|pwd|passwd|password|secret|token|apikey|api-key|auth)(?:$|[\s_:.\-])/i,
      reason: "password/secret/token-like attribute"
    },
    {
      pattern: /(?:^|[\s_:.\-])(?:username|user-name|user_id|userid|bday|dob|birthdate|address|street|zip|postal)(?:$|[\s_:.\-])/i,
      reason: "PII/identity-like attribute"
    }
  ];

  function cssEscape(value) {
    if (window.CSS?.escape) {
      return CSS.escape(value);
    }

    return value.replace(/["\\]/g, "\\$&");
  }

  function normalizeWhitespace(value) {
    return (value ?? "").replace(/\s+/g, " ").trim();
  }

  function truncateText(value, maxLength = 140) {
    const text = normalizeWhitespace(value);
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
  }

  function getSelectorPart(element) {
    if (element.id) {
      return `#${cssEscape(element.id)}`;
    }

    for (const attributeName of ["data-testid", "data-test"]) {
      const attributeValue = element.getAttribute(attributeName);
      if (attributeValue) {
        return `${element.tagName.toLowerCase()}[${attributeName}="${cssEscape(attributeValue)}"]`;
      }
    }

    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel) {
      return `${element.tagName.toLowerCase()}[aria-label="${cssEscape(ariaLabel)}"]`;
    }

    if (element.name) {
      return `${element.tagName.toLowerCase()}[name="${cssEscape(element.name)}"]`;
    }

    const parent = element.parentElement;
    if (!parent) {
      return element.tagName.toLowerCase();
    }

    const sameTagSiblings = Array.from(parent.children)
      .filter((child) => child.tagName === element.tagName);

    if (sameTagSiblings.length <= 1) {
      return element.tagName.toLowerCase();
    }

    return `${element.tagName.toLowerCase()}:nth-of-type(${sameTagSiblings.indexOf(element) + 1})`;
  }

  function getAutocompleteTokens(input) {
    return input
      .getAttribute("autocomplete")
      ?.toLowerCase()
      .split(/\s+/)
      .filter(Boolean) ?? [];
  }

  function getPiiReason(input) {
    const inputType = (input.getAttribute("type") || "text").toLowerCase();

    if (input.tagName.toLowerCase() === "input" && inputType === "password") {
      return "input type=password";
    }

    if (input.tagName.toLowerCase() === "input" && inputType === "email") {
      return "input type=email";
    }

    if (input.tagName.toLowerCase() === "input" && inputType === "tel") {
      return "input type=tel";
    }

    for (const token of getAutocompleteTokens(input)) {
      if (documentedPiiAutocompleteTokens.has(token)) {
        return `autocomplete=${token}`;
      }
    }

    const attributeText = [
      input.getAttribute("type"),
      input.getAttribute("name"),
      input.id,
      input.getAttribute("autocomplete"),
      input.getAttribute("aria-label"),
      input.getAttribute("placeholder")
    ]
      .filter(Boolean)
      .join(" ");

    for (const { pattern, reason } of piiAttributePatterns) {
      if (pattern.test(attributeText)) {
        return reason;
      }
    }

    return "";
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
    const width = right - x;
    const height = bottom - y;

    if (width <= 0 || height <= 0) {
      return null;
    }

    return { x, y, width, height };
  }

  function getFramePath() {
    const path = [];
    let currentWindow = window;

    while (currentWindow !== currentWindow.parent) {
      const parentWindow = currentWindow.parent;
      let frameIndex = -1;

      for (let index = 0; index < parentWindow.frames.length; index += 1) {
        if (parentWindow.frames[index] === currentWindow) {
          frameIndex = index;
          break;
        }
      }

      if (frameIndex === -1) {
        break;
      }

      path.unshift(frameIndex);
      currentWindow = parentWindow;
    }

    return path;
  }

  function getChildFrameRects(framePath) {
    const frames = [];
    const frameElements = document.querySelectorAll("iframe, frame");

    for (const frameElement of frameElements) {
      let frameIndex = -1;

      for (let index = 0; index < window.frames.length; index += 1) {
        if (window.frames[index] === frameElement.contentWindow) {
          frameIndex = index;
          break;
        }
      }

      if (frameIndex === -1) {
        continue;
      }

      const rect = frameElement.getBoundingClientRect();

      frames.push({
        framePath: [...framePath, frameIndex],
        selector: getSelectorPart(frameElement),
        rect: {
          x: rect.x + frameElement.clientLeft,
          y: rect.y + frameElement.clientTop,
          width: rect.width,
          height: rect.height
        }
      });
    }

    return frames;
  }

  function getAssociatedLabelText(element) {
    // The same <label> is usually reachable more than one way (element.labels,
    // label[for=...], aria-labelledby). Collecting them all without dedup is
    // what turned one "Username" caption into a "Username Username" target.
    const labels = [];
    const seenNodes = new Set();
    const seenText = new Set();

    const addLabel = (labelElement) => {
      if (!labelElement || seenNodes.has(labelElement)) {
        return;
      }
      seenNodes.add(labelElement);
      const text = normalizeWhitespace(labelElement.innerText);
      const key = text.toLowerCase();
      if (!text || seenText.has(key)) {
        return;
      }
      seenText.add(key);
      labels.push(text);
    };

    if (element.labels) {
      Array.from(element.labels).forEach(addLabel);
    }

    if (element.id) {
      try {
        Array.from(document.querySelectorAll(`label[for="${cssEscape(element.id)}"]`)).forEach(addLabel);
      } catch (error) {
        // Unusual IDs can break selector construction; nearby text still helps.
      }
    }

    const labelledBy = element.getAttribute("aria-labelledby");
    if (labelledBy) {
      for (const id of labelledBy.split(/\s+/).filter(Boolean)) {
        addLabel(document.getElementById(id));
      }
    }

    return normalizeWhitespace(labels.join(" "));
  }

  function getElementLabel(element) {
    const tagName = element.tagName.toLowerCase();
    const type = (element.getAttribute("type") || "").toLowerCase();
    const candidates = [
      element.textContent,
      element.getAttribute("aria-label"),
      tagName === "input" && ["button", "submit", "reset"].includes(type) ? element.value : "",
      element.getAttribute("placeholder"),
      getAssociatedLabelText(element),
      element.getAttribute("title")
    ];

    return truncateText(candidates.map(normalizeWhitespace).find(Boolean) || "");
  }

  function isInteractiveElement(element) {
    const tagName = element.tagName.toLowerCase();
    const role = (element.getAttribute("role") || "").toLowerCase();
    const type = (element.getAttribute("type") || "").toLowerCase();

    return (
      ["button", "a", "input", "textarea", "select", "summary"].includes(tagName) ||
      ["button", "link", "menuitem", "option", "tab", "checkbox", "radio", "combobox", "textbox"].includes(role) ||
      element.isContentEditable ||
      element.hasAttribute("onclick") ||
      (tagName === "input" && type !== "hidden")
    );
  }

  function walkDomTree(root, controls, interactiveElements, selectorPath = "", namingRoot = document) {
    const children = root instanceof Document || root instanceof ShadowRoot
      ? root.children
      : root.children ?? [];

    for (const child of children) {
      const selectorPart = getSelectorPart(child);
      // Attribute selectors are only self-sufficient when they identify one
      // element in the current naming root (document or shadow root).
      let isUniquePart = false;
      try {
        isUniquePart = namingRoot.querySelectorAll(selectorPart).length === 1;
      } catch {
        isUniquePart = false;
      }
      // A unique part replaces its ancestor chain, but it must keep any
      // "::shadow" hops: a unique id inside a shadow root is not reachable
      // from the document, so dropping the prefix produced a dead selector.
      const shadowHopEnd = selectorPath.lastIndexOf("::shadow");
      const shadowPrefix = shadowHopEnd === -1
        ? ""
        : selectorPath.slice(0, shadowHopEnd + "::shadow".length);
      const currentPath = (selectorPath && !isUniquePart)
        ? `${selectorPath} ${selectorPart}`
        : (shadowPrefix ? `${shadowPrefix} ${selectorPart}` : selectorPart);
      const tagName = child.tagName.toLowerCase();

      if (tagName === "input" || tagName === "textarea" || tagName === "select") {
        controls.push({
          element: child,
          selector: currentPath
        });
      }

      if (isInteractiveElement(child)) {
        interactiveElements.push({
          element: child,
          selector: currentPath
        });
      }

      if (child.shadowRoot) {
        walkDomTree(child.shadowRoot, controls, interactiveElements, `${currentPath}::shadow`, child.shadowRoot);
      } else {
        walkDomTree(child, controls, interactiveElements, currentPath, namingRoot);
      }
    }
  }

  function scanDocumentElements() {
    const controls = [];
    const interactiveElements = [];
    walkDomTree(document, controls, interactiveElements);
    return { controls, interactiveElements };
  }

  const MAX_TEXT_PII_REGIONS = 300;
  const MAX_TEXT_NODES_SCANNED = 4000;

  function buildElementSelector(element) {
    const parts = [];
    let current = element;

    for (let depth = 0; current && current.nodeType === 1 && depth < 5; depth += 1) {
      parts.unshift(getSelectorPart(current));

      try {
        if (document.querySelectorAll(parts.join(" ")).length === 1) {
          break;
        }
      } catch {
        break;
      }

      current = current.parentElement;
    }

    return parts.join(" ");
  }

  function isTextHostVisible(element, visibilityCache) {
    if (visibilityCache.has(element)) {
      return visibilityCache.get(element);
    }

    let visible = false;

    if (!piiText.isNonRenderedTag(element.tagName)) {
      const style = window.getComputedStyle(element);
      visible =
        style.visibility !== "hidden" &&
        style.display !== "none" &&
        Number(style.opacity) !== 0 &&
        Boolean(getVisibleRect(element));
    }

    visibilityCache.set(element, visible);
    return visible;
  }

  // Turn every PII span found in visible text into viewport rectangles so the
  // popup's existing redaction pass can black those pixels out.
  function scanVisibleTextPii() {
    const regions = [];

    if (!document.body) {
      return regions;
    }

    const visibilityCache = new Map();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;

        if (!parent || !node.nodeValue || !node.nodeValue.trim()) {
          return NodeFilter.FILTER_REJECT;
        }

        if (piiText.isNonRenderedTag(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }

        return NodeFilter.FILTER_ACCEPT;
      }
    });

    let scanned = 0;
    let previousText = "";
    let previousParent = null;

    while (walker.nextNode() && scanned < MAX_TEXT_NODES_SCANNED && regions.length < MAX_TEXT_PII_REGIONS) {
      scanned += 1;

      const node = walker.currentNode;
      const parent = node.parentElement;
      // Labels and values are often split across sibling nodes
      // (<dt>Name</dt><dd>...</dd>, <label>Username:</label> <span>...</span>).
      // Offer the previous nearby text node as context so contextual entity
      // rules can see the label; matches are still reported for this node only.
      const sharesContext = Boolean(previousParent && parent) && (
        previousParent === parent ||
        previousParent.parentElement === parent.parentElement ||
        previousParent.parentElement === parent ||
        parent.parentElement === previousParent
      );
      const matches = piiText.detectPiiMatches(
        node.nodeValue,
        sharesContext ? { prefix: previousText } : undefined
      );

      previousText = node.nodeValue;
      previousParent = parent;

      if (matches.length === 0) {
        continue;
      }

      if (!parent || !isTextHostVisible(parent, visibilityCache)) {
        continue;
      }

      const selector = buildElementSelector(parent);

      for (const match of matches) {
        if (regions.length >= MAX_TEXT_PII_REGIONS) {
          break;
        }

        const range = document.createRange();
        range.setStart(node, match.start);
        range.setEnd(node, match.end);

        // A span can wrap across lines, so each client rect is redacted separately.
        for (const clientRect of range.getClientRects()) {
          const x = Math.max(clientRect.x - 1, 0);
          const y = Math.max(clientRect.y - 1, 0);
          const right = Math.min(clientRect.right + 1, window.innerWidth);
          const bottom = Math.min(clientRect.bottom + 1, window.innerHeight);
          const width = right - x;
          const height = bottom - y;

          if (width <= 0 || height <= 0) {
            continue;
          }

          regions.push({
            selector,
            rect: { x, y, width, height },
            reason: `text-pii:${match.type}`
          });
        }

        range.detach?.();
      }
    }

    return regions;
  }

  function scanPiiFields() {
    const fields = [];
    const elements = [];
    const { controls, interactiveElements } = scanDocumentElements();
    const framePath = getFramePath();

    for (const { element: control, selector } of controls) {
      const reason = getPiiReason(control);

      if (!reason) {
        continue;
      }

      const rect = getVisibleRect(control);

      if (!rect) {
        continue;
      }

      fields.push({
        selector,
        rect,
        reason
      });
    }

    for (const { element, selector } of interactiveElements) {
      const rect = getVisibleRect(element);

      if (!rect) {
        continue;
      }

      const tagName = element.tagName.toLowerCase();
      const type = element.getAttribute("type");
      const label = getElementLabel(element);
      const piiReason = getPiiReason(element);

      if (!label && !element.getAttribute("name") && !element.id && !element.getAttribute("placeholder")) {
        continue;
      }

      // A placeholder or name that only repeats the label is noise the model
      // reads as a second word in the target's name.
      const isEcho = (value) => !value || normalizeWhitespace(value).toLowerCase() === label.toLowerCase();
      const placeholder = truncateText(element.getAttribute("placeholder") || "");
      const name = truncateText(element.getAttribute("name") || "", 80);

      elements.push({
        selector,
        tag: tagName,
        role: element.getAttribute("role") || undefined,
        type: type || undefined,
        label: label || undefined,
        placeholder: isEcho(placeholder) ? undefined : placeholder,
        name: isEcho(name) ? undefined : name,
        reason: piiReason || undefined,
        rect
      });

      if (elements.length >= 20) {
        break;
      }
    }

    let textPiiError;
    let textPiiCount = 0;

    try {
      const textRegions = scanVisibleTextPii();
      textPiiCount = textRegions.length;
      fields.push(...textRegions);
    } catch (error) {
      // Reported to the popup, which fails closed rather than transmitting a
      // screenshot whose visible text was never checked.
      textPiiError = error?.message || String(error);
    }

    return {
      fields,
      elements,
      textPiiCount,
      textPiiError,
      framePath,
      childFrameRects: getChildFrameRects(framePath),
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio
      }
    };
  }

  function sanitizeTaskText(text) {
    if (!text || typeof text !== "string") return text;
    let sanitized = text;
    // Replace emails
    sanitized = sanitized.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[EMAIL_REDACTED]");
    // Replace phone numbers
    sanitized = sanitized.replace(/\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b/g, "[PHONE_REDACTED]");
    // Replace SSNs
    sanitized = sanitized.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[SSN_REDACTED]");
    // Replace Credit Cards
    sanitized = sanitized.replace(/\b(?:\d[ -]*?){13,19}\b/g, "[CARD_REDACTED]");
    // Replace API keys & tokens
    sanitized = sanitized.replace(/\bsk-[a-zA-Z0-9]{20,}\b/g, "[API_KEY_REDACTED]");
    sanitized = sanitized.replace(/\b(?:api[_-]?key|secret|access[_-]?token|bearer)\s*[:=]\s*["']?[A-Za-z0-9_\-]{16,}["']?\b/gi, "[TOKEN_REDACTED]");
    return sanitized;
  }

  window.__privacyVisionAgent = {
    scanPiiFields,
    sanitizeTaskText
  };
})();
