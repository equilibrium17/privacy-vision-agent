// Structured evidence for final task verification.
//
// Turns a fresh, read-only DOM snapshot into numbered evidence items (E1, E2,
// ...) that the verifier must cite. Completed plan steps are returned
// separately as `claims`: they describe the agent's history, not the current
// page, so they are never citable evidence.
//
// Pure and generic: no site-specific names. Loaded as a classic script (like
// pii-patterns.js) so popup.js needs no build step.
(() => {
  const MAX_ITEMS = 40;
  const MAX_LABEL_CHARS = 60;
  const MAX_SELECTOR_CHARS = 100;
  const MAX_URL_CHARS = 160;
  const SAFE_NUMERIC_VALUE = /^\d{1,4}$/;

  const clean = (value, max = MAX_LABEL_CHARS) => String(value ?? "").replace(/\s+/g, " ").trim().slice(0, max);
  const quoted = (label) => (label ? ` "${label}"` : "");

  function buildVerificationEvidence(snapshot = {}, { elements = [], completedSteps = [] } = {}) {
    const items = [];
    const seen = new Set();

    const add = (kind, text, selector) => {
      if (items.length >= MAX_ITEMS) {
        return;
      }

      if (selector) {
        // A control listed by the verifier read and by the agent's element scan
        // is the same thing; list it once.
        const group = kind === "action-control" || kind === "element" ? "control" : kind;
        const key = `${group}|${selector}`;

        if (seen.has(key)) {
          return;
        }
        seen.add(key);
      }

      items.push({ id: `E${items.length + 1}`, kind, text });
    };

    if (snapshot.url) {
      add("url", `Current URL: ${clean(snapshot.url, MAX_URL_CHARS)}`);
    }

    if (snapshot.title) {
      add("title", `Page title: ${clean(snapshot.title)}`);
    }

    const overlays = (snapshot.overlays ?? []).filter((overlay) => overlay?.selector);

    // An explicit "nothing is open" item lets the verifier see a closed state
    // instead of inferring it from absence.
    if (overlays.length === 0) {
      add("overlay-state", "No open dialog, drawer or modal is visible");
    }

    for (const overlay of overlays) {
      const selector = clean(overlay.selector, MAX_SELECTOR_CHARS);
      const role = clean(overlay.role);
      add("open-overlay", `Open overlay ${selector}${role ? ` role=${role}` : ""}${quoted(clean(overlay.label))}`, selector);
    }

    for (const control of snapshot.expanded ?? []) {
      if (control?.selector) {
        const selector = clean(control.selector, MAX_SELECTOR_CHARS);
        add("expanded-control", `Expanded control ${selector}${quoted(clean(control.label))} (aria-expanded=true)`, selector);
      }
    }

    for (const control of snapshot.numericControls ?? []) {
      const value = String(control?.value ?? "").trim();

      // Only short numeric values (quantities, counts) are exposed. Free-text
      // values can carry PII the redactor may not recognise.
      if (!control?.selector || String(control.type ?? "").toLowerCase() === "password" || !SAFE_NUMERIC_VALUE.test(value)) {
        continue;
      }

      const selector = clean(control.selector, MAX_SELECTOR_CHARS);
      add("numeric-control", `Input ${selector}${quoted(clean(control.label))} = ${value}`, selector);
    }

    for (const control of snapshot.checkedControls ?? []) {
      if (control?.selector) {
        const selector = clean(control.selector, MAX_SELECTOR_CHARS);
        const noun = control.type === "radio" ? "Radio" : "Checkbox";
        add("checked-control", `${noun} ${selector}${quoted(clean(control.label))} is ${control.checked ? "checked" : "unchecked"}`, selector);
      }
    }

    for (const control of snapshot.actionControls ?? []) {
      const label = clean(control?.label);

      if (control?.selector && label) {
        const selector = clean(control.selector, MAX_SELECTOR_CHARS);
        add("action-control", `Visible ${clean(control.tag) || "control"} ${selector}${quoted(label)}`, selector);
      }
    }

    for (const element of elements ?? []) {
      if (element?.selector) {
        const selector = clean(element.selector, MAX_SELECTOR_CHARS);
        const label = clean(element.label || element.placeholder || element.name);
        add("element", `Interactive ${clean(element.tag) || "element"} ${selector}${quoted(label)}`, selector);
      }
    }

    return {
      items,
      claims: (completedSteps ?? []).map((step) => String(step ?? "").trim()).filter(Boolean)
    };
  }

  const api = { MAX_ITEMS, buildVerificationEvidence };

  globalThis.__privacyVisionVerificationEvidence = api;

  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  }
})();
