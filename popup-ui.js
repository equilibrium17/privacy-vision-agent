// Presentation layer for popup.html (classic script, loaded before popup.js).
//
// Renders state the pipeline already produces and nothing else. popup.js calls
// the hooks exported below where it already updates the page, and the legacy
// #status / #perfStats / #agentResult nodes are mirrored into the layout. No
// detection, redaction, agent or network logic lives here.
//
// Every string that came from a page, a model or an error is passed through
// safeText() before it is shown: URLs are cut down to their host and the
// shared PII redactor runs over the rest.
(() => {
  const extension = globalThis.browser ?? globalThis.chrome;
  const piiText = globalThis.__privacyVisionPii;
  const audit = globalThis.__privacyVisionRedactionAudit;
  const byId = (id) => document.getElementById(id);

  const app = byId("app");
  const el = {
    settingsToggle: byId("settingsToggle"),
    settingsPanel: byId("settingsPanel"),
    providerSelect: byId("providerSelect"),
    routeLabel: byId("routeLabel"),
    shieldState: byId("shieldState"),
    shieldHeadline: byId("shieldHeadline"),
    shieldReason: byId("shieldReason"),
    pageHost: byId("pageHost"),
    pageNote: byId("pageNote"),
    ledgerStamp: byId("ledgerStamp"),
    ledgerNote: byId("ledgerNote"),
    counts: {
      faces: byId("countFaces"),
      formFields: byId("countFields"),
      textPii: byId("countText"),
      entities: byId("countEntities")
    },
    agentPhase: byId("agentPhase"),
    agentIdle: byId("agentIdle"),
    agentRun: byId("agentRun"),
    agentDetail: byId("agentDetail"),
    taskEcho: byId("taskEcho"),
    taskNote: byId("taskNote"),
    planBlock: byId("planBlock"),
    stepCount: byId("stepCount"),
    stepTicks: byId("stepTicks"),
    planList: byId("planList"),
    outcome: byId("outcome"),
    outcomeTitle: byId("outcomeTitle"),
    outcomeReason: byId("outcomeReason"),
    taskInput: byId("taskInput"),
    taskHint: byId("taskHint"),
    runAgentButton: byId("runAgentButton"),
    runShortcut: byId("runShortcut"),
    timeline: byId("timeline"),
    activityEmpty: byId("activityEmpty"),
    activityCount: byId("activityCount"),
    timelineMore: byId("timelineMore"),
    pipelineStats: byId("pipelineStats"),
    frameEmpty: byId("frameEmpty"),
    canvas: byId("screenshotCanvas"),
    auditAside: byId("auditAside"),
    auditTotal: byId("auditTotal"),
    auditTypes: byId("auditTypes"),
    auditLog: byId("auditLog"),
    auditEmpty: byId("auditEmpty"),
    versionLabel: byId("versionLabel"),
    liveAnnouncer: byId("liveAnnouncer"),
    statusSource: byId("status"),
    agentResultSource: byId("agentResult"),
    perfSource: byId("perfStats")
  };

  const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)");
  const prefersStill = () => Boolean(reducedMotion?.matches);

  const pad = (value) => String(value).padStart(2, "0");
  const clock = (date = new Date()) =>
    Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("en-GB", { hour12: false });

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text != null) {
      element.textContent = text;
    }
    return element;
  }

  function replayEntrance(element) {
    if (prefersStill()) {
      return;
    }
    element.classList.remove("enter");
    void element.offsetWidth;
    element.classList.add("enter");
  }

  // ---------------------------------------------------------------- sanitizing

  function safeText(value, maxLength = 240) {
    let text = String(value ?? "")
      // Status strings lead with emoji markers; tone is conveyed separately.
      .replace(/^[\s✅❌⚠️]+/u, "")
      // The one error that can echo a resolved fill value.
      .replace(/(No select option matched:).*/i, "$1 [value hidden]")
      // Paths and query strings can carry identifiers; keep the host only.
      .replace(/\b(?:https?|file):\/\/[^\s)"'<>]+/gi, (url) => {
        try {
          const parsed = new URL(url);
          // The agent server's loopback address is not personal data.
          if (["127.0.0.1", "[::1]", "localhost"].includes(parsed.hostname)) {
            return `localhost${parsed.port ? `:${parsed.port}` : ""}`;
          }
          return parsed.host || "local file";
        } catch {
          return "[url]";
        }
      });

    if (typeof piiText?.redactPiiText === "function") {
      text = piiText.redactPiiText(text);
    }

    text = text.replace(/\s+/g, " ").trim();
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
  }

  const isFailClosed = (text) => /fail-closed/i.test(String(text ?? ""));

  // -------------------------------------------------------------------- shield

  const SHIELD_VIEWS = {
    idle: {
      label: "Standing by",
      headline: "Redacted <em>here</em>, before any model sees the page.",
      note: "Privacy layer ready · no capture yet"
    },
    restricted: {
      label: "Unavailable",
      headline: "Browser pages can’t be captured, <em>so nothing is sent.</em>",
      note: "Restricted page"
    },
    scanning: {
      label: "Scanning",
      headline: "Redacting the visible frame <em>on this device.</em>",
      note: "Local pass in progress"
    },
    protected: {
      label: "Protected",
      headline: "Last frame redacted <em>and checked</em> on this device.",
      note: "Privacy layer active"
    },
    blocked: {
      label: "Blocked",
      headline: "Transmission stopped. <em>This frame was not sent.</em>",
      note: "Failed closed"
    }
  };

  let shieldView = "idle";
  let pageRestricted = false;

  function setShield(view, { reason = "", stamp = "" } = {}) {
    const copy = SHIELD_VIEWS[view];
    shieldView = view;
    app.dataset.shield = view;
    el.shieldState.textContent = copy.label;

    if (el.shieldHeadline.dataset.view !== view) {
      el.shieldHeadline.dataset.view = view;
      el.shieldHeadline.innerHTML = copy.headline; // static markup from SHIELD_VIEWS only
      replayEntrance(el.shieldHeadline);
    }

    el.pageNote.textContent = stamp ? `${copy.note} · ${stamp}` : copy.note;
    el.shieldReason.textContent = reason;
    el.shieldReason.hidden = !reason;
  }

  async function readActivePage() {
    let url = "";
    try {
      const [tab] = await extension.tabs.query({ active: true, currentWindow: true });
      url = tab?.url ?? "";
    } catch {
      // Leave the page unknown rather than guess.
    }

    let parsed = null;
    try {
      parsed = new URL(url);
    } catch {
      parsed = null;
    }

    const protocol = parsed?.protocol ?? "";
    pageRestricted = Boolean(parsed) && !["http:", "https:", "file:"].includes(protocol);

    if (!parsed) {
      el.pageHost.textContent = "—";
    } else if (protocol === "file:") {
      el.pageHost.textContent = "Local file";
    } else if (/^https?:$/.test(protocol)) {
      el.pageHost.textContent = parsed.host;
    } else {
      el.pageHost.textContent = parsed.host ? `${protocol}//${parsed.host}` : protocol;
    }

    if (shieldView === "idle" || shieldView === "restricted") {
      setShield(pageRestricted ? "restricted" : "idle");
    }
  }

  // -------------------------------------------------------------------- ledger

  function showCount(target, value) {
    const next = Number.isFinite(value) ? value : 0;
    const from = Number(target.dataset.value ?? 0);
    target.dataset.value = String(next);
    target.parentElement.dataset.zero = String(next === 0);

    if (prefersStill() || from === next) {
      target.textContent = pad(next);
      return;
    }

    const start = performance.now();
    const duration = 420;
    const frame = (now) => {
      if (target.dataset.value !== String(next)) {
        return; // superseded by a newer capture
      }
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - progress) ** 3;
      target.textContent = pad(Math.round(from + (next - from) * eased));
      if (progress < 1) {
        requestAnimationFrame(frame);
      }
    };
    requestAnimationFrame(frame);
  }

  function syncFrame() {
    el.frameEmpty.hidden = el.canvas.style.display === "block";
  }

  // --------------------------------------------------------------------- agent

  const PHASE_LABELS = {
    idle: "Idle",
    observing: "Observing",
    planning: "Planning",
    acting: "Acting",
    verifying: "Verifying",
    completed: "Completed",
    incomplete: "Incomplete",
    failed: "Failed",
    blocked: "Blocked"
  };

  // Ordered: a step description can itself contain words like "verify".
  const PHASE_PATTERNS = [
    [/^\s*(step \d+ of|executing step)|step failed|page changed/i, "acting"],
    [/verif/i, "verifying"],
    [/plan/i, "planning"],
    [/captur|detector|detecting faces|scanning form|re-reading|face\(s\) blurred/i, "observing"]
  ];

  const OUTCOMES = {
    complete: { phase: "completed", tone: "success", title: "Verified complete" },
    incomplete: { phase: "incomplete", tone: "warn", title: "May be incomplete" },
    failed: { phase: "failed", tone: "error", title: "Stopped" }
  };

  let agentRunning = false;

  function setPhase(phase) {
    app.dataset.agent = phase;
    el.agentPhase.textContent = PHASE_LABELS[phase];
  }

  function renderAgentState(state) {
    const plan = Array.isArray(state?.plan) ? state.plan : [];
    const total = plan.length;
    const index = Math.min(Math.max(Number(state?.currentStepIndex) || 0, 0), total);
    // Steps already acted on but not yet confirmed by a fresh page read. They
    // are shown as underway so a round that runs several steps at once does
    // not leave the list sitting on step 1 until verification catches up.
    const executed = Math.min(Math.max(Number(state?.executedThroughIndex) || 0, index), total);
    const statusOf = (position) => {
      if (position < index) {
        return "done";
      }
      if (position === index) {
        return "current";
      }
      return position < executed ? "executed" : "pending";
    };

    el.planBlock.hidden = total === 0;
    el.stepCount.textContent = total ? `Step ${Math.min(Math.max(index + 1, executed), total)} / ${total}` : "Step — / —";

    el.stepTicks.replaceChildren(...plan.map((_, position) => {
      const tick = node("span", "tick");
      tick.dataset.status = statusOf(position);
      return tick;
    }));

    el.planList.replaceChildren(...plan.map((step, position) => {
      const item = node("li");
      item.dataset.status = statusOf(position);
      if (position === index) {
        item.setAttribute("aria-current", "step");
      }
      item.append(
        node("span", "step-index mono", pad(position + 1)),
        node("span", null, safeText(step?.description, 180))
      );
      return item;
    }));
  }

  function finishAgent(outcome, reason) {
    agentRunning = false;
    el.agentDetail.hidden = true;
    el.outcome.dataset.tone = outcome.tone;
    el.outcomeTitle.textContent = outcome.title;
    el.outcomeReason.textContent = safeText(reason, 280);
    el.outcome.hidden = false;
    replayEntrance(el.outcome);

    if (isFailClosed(reason)) {
      setPhase("blocked");
      setShield("blocked", { reason: safeText(reason) });
    } else {
      setPhase(outcome.phase);
    }
  }

  // ------------------------------------------------------------------ activity

  const VISIBLE_EVENTS = 6;
  const MAX_EVENTS = 60;
  const TONE_PREFIX = { error: "Error: ", warn: "Warning: " };
  const events = [];
  let eventSequence = 0;
  let timelineExpanded = false;

  function toneOf(raw) {
    const text = String(raw ?? "");
    if (/^\s*✅/u.test(text)) return "success";
    if (/^\s*❌/u.test(text)) return "error";
    if (/^\s*⚠/u.test(text) || /step failed|page changed|planning unavailable/i.test(text)) return "warn";
    if (/fail-closed|privacy error|unable to|error|rejected|cancelled|no element matched/i.test(text)) return "error";
    if (/face\(s\) blurred/i.test(text)) return "done";
    return "active";
  }

  function logEvent(text, tone) {
    const previous = events.at(-1);
    if (previous?.tone === "active") {
      previous.tone = "done";
    }
    // A message that restates the previous one (e.g. a wrapped error) replaces it.
    const core = (value) => value.replace(/^[\w -]*error:\s*/i, "");
    if (previous && text.includes(core(previous.text))) {
      events.pop();
    }
    events.push({ id: ++eventSequence, text, tone, time: clock() });
    if (events.length > MAX_EVENTS) {
      events.shift();
    }
    renderTimeline();
  }

  function renderTimeline() {
    const shown = timelineExpanded ? events : events.slice(-VISIBLE_EVENTS);

    el.timeline.replaceChildren(...shown.map((event) => {
      const item = node("li", "event");
      item.dataset.tone = event.tone;
      if (!event.rendered && !prefersStill()) {
        item.classList.add("enter");
      }
      event.rendered = true;

      const label = event.tone === "active"
        ? event.text.replace(/\.{3}$/, "…")
        : event.text.replace(/(\.{3}|…)$/, "");
      const text = node("span", "event-text");
      if (TONE_PREFIX[event.tone] && !/error|fail/i.test(label)) {
        text.append(node("span", "visually-hidden", TONE_PREFIX[event.tone]));
      }
      text.append(label);

      item.append(node("span", "event-dot"), text, node("time", "event-time", event.time));
      item.firstChild.setAttribute("aria-hidden", "true");
      return item;
    }));

    const earlier = events.length - VISIBLE_EVENTS;
    el.activityEmpty.hidden = events.length > 0;
    el.activityCount.textContent = events.length ? `${events.length} event${events.length === 1 ? "" : "s"}` : "";
    el.timelineMore.hidden = earlier <= 0;
    el.timelineMore.textContent = earlier <= 0 ? "" : timelineExpanded ? "Show recent only" : `Show earlier · ${earlier}`;
    el.timelineMore.setAttribute("aria-expanded", String(timelineExpanded));
  }

  // --------------------------------------------------------- mirrored outputs

  function onStatusText(raw) {
    const text = safeText(raw, 220);
    if (!text) {
      return;
    }

    el.liveAnnouncer.textContent = text;
    logEvent(text, toneOf(raw));

    if (agentRunning) {
      el.agentDetail.textContent = text;
      const match = PHASE_PATTERNS.find(([pattern]) => pattern.test(raw));
      // Goal verification captures a fresh frame; that capture belongs to verifying.
      if (match && !(match[1] === "observing" && app.dataset.agent === "verifying")) {
        setPhase(match[1]);
      }
    }
  }

  function onAgentResultText(raw) {
    const missingTask = /^Enter a task/.test(raw ?? "");
    el.taskHint.textContent = missingTask ? "Describe a task before running the agent." : "";
    el.taskHint.hidden = !missingTask;
    if (missingTask) {
      el.taskInput.setAttribute("aria-invalid", "true");
    }
  }

  const STAT_LABELS = {
    "Face model": "Face model load",
    "Face delegate": "Face delegate",
    Detection: "Face detection",
    "DOM scan": "DOM scan",
    Image: "Model image",
    Server: "Server round-trip"
  };

  function renderStats(raw) {
    const rows = String(raw ?? "")
      .split("|")
      .map((part) => part.split(/:(.*)/s).map((piece) => (piece ?? "").trim()))
      .filter(([label, value]) => label && value);

    el.pipelineStats.replaceChildren(...rows.flatMap(([label, value]) => [
      node("dt", null, STAT_LABELS[label] ?? label),
      node("dd", null, value)
    ]));
  }

  function mirror(source, render) {
    new MutationObserver(() => render(source.textContent))
      .observe(source, { childList: true, characterData: true, subtree: true });
  }

  // --------------------------------------------------------------------- audit

  const AUDIT_TYPES = [
    ["face", "Faces", "Face"],
    ["form-field", "Form fields", "Field"],
    ["text-pii", "Text PII", "Text"],
    ["entity", "Entities", "Entity"]
  ];

  // Read-only view of the stored log. The module's own state is not touched.
  async function renderAudit() {
    const key = audit?.STORAGE_KEY;
    let entries = null;

    if (key && extension?.storage?.local) {
      try {
        const stored = await extension.storage.local.get(key);
        entries = Array.isArray(stored?.[key]) ? stored[key] : [];
      } catch {
        entries = null;
      }
    }

    if (!entries) {
      el.auditAside.textContent = "unavailable";
      el.auditTotal.textContent = "—";
      el.auditTypes.replaceChildren();
      el.auditLog.replaceChildren();
      el.auditEmpty.textContent = "The local audit log could not be read.";
      el.auditEmpty.hidden = false;
      return;
    }

    el.auditAside.textContent = `${entries.length} recorded`;
    el.auditTotal.textContent = pad(entries.length);

    el.auditTypes.replaceChildren(...AUDIT_TYPES.flatMap(([type, label]) => [
      node("dt", null, label),
      node("dd", null, pad(entries.filter((entry) => entry?.type === type).length))
    ]));

    el.auditLog.replaceChildren(...entries.slice(-6).reverse().map((entry) => {
      const typeLabel = AUDIT_TYPES.find(([type]) => type === entry?.type)?.[2] ?? "Other";
      let host = "";
      try {
        host = entry?.pageOrigin ? new URL(entry.pageOrigin).host : "";
      } catch {
        host = "";
      }
      const detail = safeText(entry?.category ?? entry?.source ?? "", 40);

      const item = node("li");
      item.append(
        node("time", null, clock(new Date(entry?.timestamp))),
        node("span", null, detail ? `${typeLabel} · ${detail}` : typeLabel),
        node("span", "audit-host", host)
      );
      return item;
    }));

    el.auditEmpty.textContent = "Nothing recorded yet.";
    el.auditEmpty.hidden = entries.length > 0;
  }

  // ---------------------------------------------------------------- controls

  function setSettingsOpen(open) {
    el.settingsPanel.hidden = !open;
    el.settingsToggle.setAttribute("aria-expanded", String(open));
    if (open) {
      el.providerSelect.focus();
    }
  }

  el.settingsToggle.addEventListener("click", () => setSettingsOpen(el.settingsPanel.hidden));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !el.settingsPanel.hidden) {
      setSettingsOpen(false);
      el.settingsToggle.focus();
    }
  });

  const syncRoute = () => {
    el.routeLabel.textContent = el.providerSelect.selectedOptions[0]?.textContent ?? "—";
  };
  el.providerSelect.addEventListener("change", syncRoute);

  el.taskInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!el.runAgentButton.disabled) {
        el.runAgentButton.click();
      }
    }
  });

  el.taskInput.addEventListener("input", () => {
    if (!el.taskHint.hidden) {
      onAgentResultText("");
      el.taskInput.removeAttribute("aria-invalid");
    }
  });

  el.timelineMore.addEventListener("click", () => {
    timelineExpanded = !timelineExpanded;
    renderTimeline();
  });

  if (/Mac|iPhone|iPad/.test(navigator.platform)) {
    el.runShortcut.textContent = "⌘ ↵";
  }

  try {
    const version = extension?.runtime?.getManifest?.().version;
    el.versionLabel.textContent = version ? `v${version}` : "";
  } catch {
    el.versionLabel.textContent = "";
  }

  extension?.storage?.onChanged?.addListener?.((changes, area) => {
    if (area === "local" && audit?.STORAGE_KEY in changes) {
      void renderAudit();
    }
  });

  mirror(el.statusSource, onStatusText);
  mirror(el.agentResultSource, onAgentResultText);
  mirror(el.perfSource, renderStats);

  renderStats(el.perfSource.textContent);
  renderTimeline();
  syncRoute();
  void renderAudit();
  void readActivePage();

  // ------------------------------------------------------------ popup.js hooks

  globalThis.__privacyVisionUi = {
    captureStarted() {
      setShield("scanning");
      void readActivePage();
    },

    /** Numbers only: { faces, formFields, textPii, entities }. */
    captureCompleted(counts = {}) {
      const values = Object.fromEntries(
        Object.keys(el.counts).map((key) => [key, Number(counts[key]) || 0])
      );
      const total = Object.values(values).reduce((sum, value) => sum + value, 0);
      const stamp = clock();

      for (const [key, target] of Object.entries(el.counts)) {
        showCount(target, values[key]);
      }

      el.ledgerStamp.textContent = `Last capture ${stamp}`;
      el.ledgerNote.dataset.tone = total === 0 ? "clean" : "redacted";
      el.ledgerNote.textContent = total === 0
        ? "Clean frame. No sensitive regions detected."
        : `${total} region${total === 1 ? "" : "s"} redacted on this device.`;

      setShield("protected", { stamp });
      syncFrame();
      void readActivePage();
    },

    captureBlocked(message) {
      setShield("blocked", { reason: safeText(String(message ?? "").replace(/^Privacy Error:\s*/i, "")) });
      syncFrame();
    },

    /** task is the sanitized text actually sent; changed is true when redaction altered it. */
    agentStarted(task, changed) {
      agentRunning = true;
      el.agentIdle.hidden = true;
      el.agentRun.hidden = false;
      el.agentDetail.hidden = false;
      el.agentDetail.textContent = "Starting…";
      el.taskEcho.textContent = safeText(task, 600);
      el.taskNote.hidden = !changed;
      el.outcome.hidden = true;
      renderAgentState(null);
      setPhase("observing");
      replayEntrance(el.agentRun);
    },

    agentState(state) {
      renderAgentState(state);
    },

    agentFinished(result) {
      finishAgent(OUTCOMES[result?.status] ?? OUTCOMES.incomplete, result?.reason);
    },

    agentFailed(message) {
      finishAgent(OUTCOMES.failed, message);
    }
  };
})();
