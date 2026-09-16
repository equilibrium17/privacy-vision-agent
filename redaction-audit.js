// Local-only audit log of what the privacy pipeline redacted (feature #8).
//
// Entries hold METADATA ONLY: what kind of thing was redacted, where it came
// from, why, and the region it covered. They never hold the matched text, an
// input value, a screenshot, DOM text or a URL. The log lives in extension
// storage (chrome.storage.local when available, memory otherwise) and is never
// added to any outbound request.
//
// Loaded as a classic script (like pii-patterns.js) so popup.js needs no build
// step.
(() => {
  const STORAGE_KEY = "redactionAuditLog";
  const MAX_ENTRIES = 500;
  const ALLOWED_TYPES = new Set(["face", "form-field", "text-pii", "entity"]);
  // Pipeline reason/category strings are short lowercase labels. Capitals,
  // quotes and other punctuation mark free text, which is never stored.
  const SAFE_TEXT = /^[a-z0-9 _:./=+-]{1,80}$/;

  let entries = [];

  function storageArea() {
    const api = globalThis.browser ?? globalThis.chrome;
    return api?.storage?.local ?? null;
  }

  async function persist() {
    const area = storageArea();

    if (!area) {
      return;
    }

    try {
      await area.set({ [STORAGE_KEY]: entries });
    } catch (error) {
      console.warn("Unable to persist the redaction audit log:", error);
    }
  }

  async function loadRedactionAuditLog() {
    const area = storageArea();

    if (!area) {
      return entries;
    }

    try {
      const stored = await area.get(STORAGE_KEY);
      entries = Array.isArray(stored?.[STORAGE_KEY]) ? stored[STORAGE_KEY] : [];
    } catch (error) {
      console.warn("Unable to load the redaction audit log:", error);
    }

    return entries;
  }

  // Short labels only (reason/category strings the pipeline already emits);
  // anything else is dropped rather than risk storing a value.
  function safeLabel(value) {
    const text = String(value ?? "").trim();

    if (!SAFE_TEXT.test(text)) {
      return undefined;
    }

    // Defence in depth: the shared detector, when loaded, gets a veto.
    const detector = globalThis.__privacyVisionPii;
    return detector?.containsPii?.(text) ? undefined : text;
  }

  function safeRect(rect) {
    if (!rect) {
      return undefined;
    }

    const numbers = ["x", "y", "width", "height"].map((key) => Number(rect[key]));
    return numbers.every(Number.isFinite)
      ? { x: Math.round(numbers[0]), y: Math.round(numbers[1]), width: Math.round(numbers[2]), height: Math.round(numbers[3]) }
      : undefined;
  }

  /** Origin only (scheme + host), never a path or query. */
  function safeOrigin(url) {
    try {
      const parsed = new URL(String(url ?? ""));
      return /^https?:$/.test(parsed.protocol) ? parsed.origin : undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Record one redaction. Only whitelisted metadata fields are kept, so a
   * caller cannot store a value by accident.
   */
  function recordRedaction(entry = {}) {
    const type = ALLOWED_TYPES.has(entry.type) ? entry.type : "text-pii";
    const record = {
      timestamp: new Date().toISOString(),
      type,
      source: safeLabel(entry.source) ?? "unknown",
      reason: safeLabel(entry.reason) ?? type,
      category: safeLabel(entry.category),
      rect: safeRect(entry.rect),
      pageOrigin: safeOrigin(entry.pageOrigin)
    };

    for (const key of Object.keys(record)) {
      if (record[key] === undefined) {
        delete record[key];
      }
    }

    entries.push(record);

    if (entries.length > MAX_ENTRIES) {
      entries = entries.slice(-MAX_ENTRIES);
    }

    void persist();
    return record;
  }

  function getRedactionAuditLog() {
    return entries.map((record) => ({ ...record, ...(record.rect ? { rect: { ...record.rect } } : {}) }));
  }

  function clearRedactionAuditLog() {
    entries = [];
    return persist();
  }

  const api = {
    STORAGE_KEY,
    MAX_ENTRIES,
    recordRedaction,
    getRedactionAuditLog,
    clearRedactionAuditLog,
    loadRedactionAuditLog
  };

  globalThis.__privacyVisionRedactionAudit = api;

  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  }
})();
