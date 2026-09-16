// Explicit plan-execution controller for the browser agent.
//
//   Goal -> Plan -> Observe -> Act -> Verify step -> Advance / Re-plan -> Verify goal
//
// The controller owns task state across rounds, so the VLM never has to
// reconstruct progress from raw action history: every request carries the
// plan, the completed steps and the next incomplete step. The environment
// (privacy-preserving capture, model calls, action execution, goal
// verification) is injected, so the same controller runs in the popup and
// under the Node test runner.
//
// Loaded as a classic script (like pii-patterns.js) so popup.js needs no build
// step.
(() => {
  const DEFAULT_MAX_REPLANS = 2;
  const DEFAULT_MAX_STEP_ATTEMPTS = 2;
  const MAX_RESULT_CHARS = 160;
  const MAX_NO_PROGRESS_WAITS = 2;
  const WAIT_NO_PROGRESS_REASON = "Previous wait actions produced no page progress.";

  function isRateLimitError(error) {
    return /HTTP 429|rate limit/i.test(String(error?.message ?? error ?? ""));
  }

  function errorText(error) {
    return String(error?.message ?? error ?? "unknown error").slice(0, 100);
  }

  function hashString(text) {
    let hash = 5381;

    for (let index = 0; index < text.length; index += 1) {
      hash = ((hash << 5) + hash + text.charCodeAt(index)) | 0;
    }

    return (hash >>> 0).toString(16);
  }

  /**
   * Local-only fingerprint of a page: URL plus interactive-element structure.
   * Used to notice "nothing changed"; it is never sent anywhere.
   */
  function pageSignature(url, elements) {
    const structure = (elements ?? [])
      .map((element) => [element.tag, element.selector, element.type, element.label]
        .map((part) => part ?? "")
        .join("|"))
      .join("\n");

    return hashString(`${url ?? ""}\n${structure}`);
  }

  function actionFingerprint(actions) {
    return (actions ?? [])
      .map((action) => `${action.action}|${action.selector}|${action.value ? "v" : ""}`)
      .join(";");
  }

  function isWaitOnly(actions) {
    return Array.isArray(actions) && actions.length > 0 && actions.every((action) => action?.action === "wait");
  }

  /** URL or interactive-element structure/labels differ between two observations. */
  function hasMeaningfulChange(before, after) {
    return !before || !after || before.url !== after.url || before.signature !== after.signature;
  }

  const SELF_TARGET_SELECTORS = new Set(["", "body", "html", "document"]);
  const STEP_STOP_WORDS = new Set([
    "the", "and", "then", "for", "with", "into", "onto", "from", "that", "this", "its", "your",
    "click", "tap", "press", "open", "select", "choose", "enter", "type", "fill", "set", "use",
    "button", "link", "field", "input", "page", "item", "now", "view", "navigate", "visit"
  ]);
  const WAIT_STEP_PATTERN = /\b(?:wait|waiting|load|loading|loads|appear|appears|refresh)\b/i;

  const normalizeSelector = (selector) => String(selector ?? "").trim().replace(/'/g, "\"").replace(/\s+/g, " ");
  const normalizeLabel = (text) => String(text ?? "").toLowerCase().replace(/\s+/g, " ").trim();

  // Compact, generic web-UI vocabulary: words that name the same control are
  // grouped into one concept, so "basket" finds a cart button and "checkout"
  // finds "Proceed to buy". An explainable lexicon, not a language model.
  const UI_CONCEPTS = {
    cart: ["cart", "basket", "bag", "trolley"],
    checkout: ["checkout", "proceed", "buy", "purchase", "pay", "payment", "billing"],
    login: ["login", "signin"],
    logout: ["logout", "signout"],
    signup: ["signup", "register", "registration"],
    account: ["account", "profile"],
    search: ["search", "find", "lookup"],
    settings: ["setting", "preference", "option", "config", "configuration"],
    menu: ["menu", "navigation", "nav", "hamburger"],
    quantity: ["quantity", "qty"],
    wishlist: ["wishlist", "favorite", "favourite"],
    home: ["home", "homepage"],
    // Action concepts: a control's own action must not contradict the step.
    add: ["add", "increase", "increment", "plus"],
    remove: ["remove", "delete", "trash", "decrease", "decrement", "minus"],
    next: ["next", "continue"],
    back: ["back", "previous", "prev"],
    close: ["close", "dismiss"],
    cancel: ["cancel"],
    submit: ["submit", "send", "confirm"]
  };
  const ACTION_CONCEPTS = new Set(["add", "remove", "next", "back", "close", "cancel", "submit"]);
  const CONCEPT_OF_WORD = new Map(
    Object.entries(UI_CONCEPTS).flatMap(([concept, words]) => words.map((word) => [word, concept]))
  );
  const UI_PHRASES = [
    [/\b(?:sign|log)\s*-?\s*(?:in|on)\b/g, " login "],
    [/\b(?:sign|log)\s*-?\s*out\b/g, " logout "],
    [/\bsign\s*-?\s*up\b/g, " signup "],
    [/\bcheck\s*-?\s*out\b/g, " checkout "],
    [/\bmy\s*-?\s*account\b/g, " account "],
    [/\bwish\s*-?\s*list\b/g, " wishlist "]
  ];

  const stemWord = (word) => (word.length > 4 && word.endsWith("s") && !/(?:ss|us|is)$/.test(word) ? word.slice(0, -1) : word);

  /** Concepts and remaining content words of a piece of UI or step text. */
  function meaningOf(text) {
    let normalized = String(text ?? "").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

    for (const [pattern, replacement] of UI_PHRASES) {
      normalized = normalized.replace(pattern, replacement);
    }

    const concepts = new Set();
    const terms = new Set();

    for (const word of normalized.split(/[^a-z0-9]+/).map(stemWord)) {
      const concept = CONCEPT_OF_WORD.get(word);

      if (concept) {
        concepts.add(concept);
      } else if (word.length >= 3 && !STEP_STOP_WORDS.has(word) && !/^\d+$/.test(word)) {
        terms.add(word);
      }
    }

    return { concepts, terms };
  }

  /**
   * Resolve model-chosen selectors against the CURRENT page, which is
   * authoritative. A present selector (or an aria-label selector naming a
   * present element) is kept. A selector absent from the page is remapped only
   * when exactly one current element matches - by the action's target label, or
   * by the same selector with a different number (a renumbered row). Anything
   * else is reported stale and must not be executed.
   */
  function resolveActionsAgainstObservation(actions, observation) {
    const elements = (observation?.elements ?? []).filter((element) => element?.selector);
    const known = new Set(
      [...elements, ...(observation?.piiFields ?? [])]
        .filter((entry) => entry?.selector)
        .map((entry) => normalizeSelector(entry.selector))
    );

    // Without DOM metadata there is nothing to judge selectors against.
    if (known.size === 0) {
      return { actions, stale: [], remapped: [] };
    }

    const resolved = [];
    const stale = [];
    const remapped = [];

    for (const action of actions ?? []) {
      const selector = normalizeSelector(action?.selector);

      if (
        action?.action === "wait" ||
        (["scroll", "press"].includes(action?.action) && SELF_TARGET_SELECTORS.has(selector.toLowerCase())) ||
        known.has(selector)
      ) {
        resolved.push(action);
        continue;
      }

      const ariaLabel = selector.match(/\[aria-label="([^"]+)"\]$/)?.[1];

      if (ariaLabel && elements.some((element) => normalizeLabel(element.label) === normalizeLabel(ariaLabel))) {
        resolved.push(action);
        continue;
      }

      const byLabel = action?.target
        ? elements.filter((element) => element.label && normalizeLabel(element.label) === normalizeLabel(action.target))
        : [];
      const stem = /\d/.test(selector) ? selector.replace(/\d+/g, "#") : null;
      const byRenumbering = stem
        ? elements.filter((element) => normalizeSelector(element.selector).replace(/\d+/g, "#") === stem)
        : [];
      const match = byLabel.length === 1 ? byLabel[0] : byRenumbering.length === 1 ? byRenumbering[0] : null;

      if (match) {
        remapped.push({ from: action.selector, to: match.selector });
        resolved.push({
          ...action,
          selector: match.selector,
          ...(Number.isInteger(match.frameId) ? { frame_id: match.frameId } : {})
        });
      } else {
        stale.push(String(action?.selector ?? ""));
      }
    }

    return { actions: resolved, stale, remapped };
  }

  /**
   * Current elements that plausibly perform the step. Meaning comes from the
   * step text plus the model's own target hints, compared by concept (so
   * "basket" matches "Cart") and by remaining content words. A control whose
   * own action contradicts the step ("Add to cart" or "Continue shopping" for
   * "Open the basket") is not a candidate.
   */
  function findStepCandidates(stepDescription, elements, hints = []) {
    if (!stepDescription || WAIT_STEP_PATTERN.test(stepDescription)) {
      return [];
    }

    const intent = meaningOf([stepDescription, ...(hints ?? [])].join(" "));

    if (intent.concepts.size === 0 && intent.terms.size === 0) {
      return [];
    }

    return (elements ?? [])
      .filter((element) => element?.selector)
      .map((element) => {
        const meaning = meaningOf(`${element.label ?? ""} ${element.selector} ${element.placeholder ?? ""} ${element.name ?? ""}`);
        let score = 0;

        for (const concept of intent.concepts) {
          if (meaning.concepts.has(concept)) {
            score += 2;
          }
        }

        for (const term of intent.terms) {
          if (meaning.terms.has(term)) {
            score += 1;
          }
        }

        for (const concept of meaning.concepts) {
          if (ACTION_CONCEPTS.has(concept) && !intent.concepts.has(concept)) {
            score -= 2;
          }
        }

        return { element, score };
      })
      .filter((candidate) => candidate.score > 0)
      .sort((first, second) => second.score - first.score)
      .slice(0, 2)
      .map((candidate) => candidate.element);
  }

  // Controls that belong to a site-wide utility rather than to page content.
  const UTILITY_CONCEPTS = new Set(["cart", "checkout", "login", "logout", "signup", "account", "search", "settings", "menu", "wishlist", "home"]);
  // Actions whose presence contradicts a step that does not ask for them.
  const CONFLICTING_ACTION_CONCEPTS = new Set(["add", "remove", "close", "cancel", "back"]);

  /**
   * Is a chosen control plausibly relevant to a step? Rejects only clear
   * mismatches, so arbitrary content names (product titles, article links)
   * are always accepted:
   * - the control performs a conflicting action the step does not ask for
   *   ("Add to Cart" for "Open cart");
   * - the control is a site utility (account, cart, search, ...) the step does
   *   not mention and shares nothing with it ("Account & Lists" for "Click
   *   first laptop product link").
   */
  function assessActionRelevance(stepDescription, controlText) {
    if (!stepDescription || !controlText) {
      return { relevant: true, reason: "" };
    }

    const intent = meaningOf(stepDescription);
    const control = meaningOf(controlText);

    for (const concept of control.concepts) {
      if (CONFLICTING_ACTION_CONCEPTS.has(concept) && !intent.concepts.has(concept)) {
        return { relevant: false, reason: `performs "${concept}"` };
      }
    }

    const shared = [...control.concepts].some((concept) => intent.concepts.has(concept)) ||
      [...control.terms].some((term) => intent.terms.has(term));
    const utility = [...control.concepts].find((concept) => UTILITY_CONCEPTS.has(concept) && !intent.concepts.has(concept));

    if (utility && !shared) {
      return { relevant: false, reason: `is a "${utility}" control` };
    }

    return { relevant: true, reason: "" };
  }

  /**
   * First action that is clearly unrelated to the plan. The first action must
   * fit the pending step; later actions in a same-page group may fit any
   * remaining step.
   */
  function findIrrelevantAction(state, actions, observation) {
    const steps = state.plan.slice(state.currentStepIndex).map((step) => step.description);

    if (steps.length === 0) {
      return null;
    }

    const elements = indexElements(observation?.elements);

    for (let index = 0; index < (actions ?? []).length; index += 1) {
      const action = actions[index];

      if (!action || action.action === "wait" || action.action === "scroll" ||
          (action.action === "press" && SELF_TARGET_SELECTORS.has(normalizeSelector(action.selector).toLowerCase()))) {
        continue;
      }

      const label = elements.get(normalizeSelector(action.selector))?.label;
      // Labels, not selectors, describe meaning; container selectors can carry
      // unrelated words.
      const text = [action.target, label].filter(Boolean).join(" ") || String(action.selector ?? "");
      const verdicts = (index === 0 ? [steps[0]] : steps).map((step) => assessActionRelevance(step, text));

      if (verdicts.every((verdict) => !verdict.relevant)) {
        const name = String(action.target || label || action.selector || "").replace(/\s+/g, " ").trim().slice(0, 30);
        return { action, name, reason: verdicts[0].reason };
      }
    }

    return null;
  }

  // --- Local action predictor (deterministic, no model) ---------------------
  //
  // Handles only the obvious case: a click-like step naming exactly one
  // clickable element on the current page. Anything involving values,
  // choice ("first", "suitable", "cheapest"), or more than one plausible
  // target is left to the VLM. A prediction is only a proposal: it goes
  // through the same validation, staleness, relevance and execution path as a
  // model response.
  const CLICK_VERBS = /^(?:click|tap|press|hit|select|choose|open|go to|navigate to|visit|activate|submit|toggle|expand)(?:\s+on)?\s+/i;
  const VALUE_STEP = /\b(?:fill|type|enter|input|write|paste|search for|set|change|increase|decrease|update|scroll|wait|upload|drag)\b/i;
  const CHOICE_WORDS = new Set([
    "first", "second", "third", "last", "next", "previous", "any", "some", "suitable", "best",
    "cheapest", "top", "random", "matching", "relevant", "appropriate", "available", "one"
  ]);
  const CLICKABLE_TAGS = new Set(["button", "a", "summary"]);
  const CLICKABLE_ROLES = new Set(["button", "link", "menuitem", "tab"]);
  const LOCAL_TRAILING = /\s+(?:button|link|tab|icon|option|menu item)$/i;

  function isClickable(element) {
    const tag = String(element?.tag ?? "").toLowerCase();
    const role = String(element?.role ?? "").toLowerCase();
    const type = String(element?.type ?? "").toLowerCase();
    return CLICKABLE_TAGS.has(tag) || CLICKABLE_ROLES.has(role) || (tag === "input" && ["submit", "button"].includes(type));
  }

  function predictLocalAction(stepDescription, observation) {
    const step = String(stepDescription ?? "").trim();

    if (!step || !CLICK_VERBS.test(step) || VALUE_STEP.test(step) || WAIT_STEP_PATTERN.test(step)) {
      return null;
    }

    const objectPhrase = normalizeLabel(step.replace(CLICK_VERBS, "").replace(/^(?:the|a|an)\s+/i, "").replace(LOCAL_TRAILING, "").replace(/["'.]/g, ""));
    const objectMeaning = meaningOf(objectPhrase);
    const stepWords = objectPhrase.split(" ").filter(Boolean);

    if (!objectPhrase || stepWords.some((word) => CHOICE_WORDS.has(word)) || /\d/.test(objectPhrase)) {
      return null;
    }

    const elements = (observation?.elements ?? []).filter((element) => element?.selector && isClickable(element));
    const byExactLabel = elements.filter((element) => normalizeLabel(element.label) === objectPhrase);
    let candidates = byExactLabel;

    if (candidates.length === 0 && (objectMeaning.concepts.size > 0 || objectMeaning.terms.size > 0)) {
      // Every content word of the step must be accounted for by the label. If
      // more than one label covers the step, the choice is not obvious.
      const covering = elements.filter((element) => {
        const meaning = meaningOf(element.label);
        return [...objectMeaning.concepts].every((concept) => meaning.concepts.has(concept)) &&
          [...objectMeaning.terms].every((term) => meaning.terms.has(term));
      });

      // A label with much more in it than the step ("ASUS Vivobook Laptop"
      // for "laptop") is a description, not the named control.
      candidates = covering.length === 1 && meaningOf(covering[0].label).terms.size <= objectMeaning.terms.size + 1
        ? covering
        : [];
    }

    if (candidates.length !== 1) {
      return null;
    }

    const element = candidates[0];

    if (!assessActionRelevance(step, element.label).relevant) {
      return null;
    }

    return {
      action: "click",
      selector: element.selector,
      ...(Number.isInteger(element.frameId) ? { frame_id: element.frameId } : {}),
      target: String(element.label ?? "").slice(0, 60),
      reasoning: "local: step names exactly this element",
      source: "local"
    };
  }

  function describeCandidates(elements) {
    return elements
      .map((element) => {
        const label = String(element.label ?? "").replace(/\s+/g, " ").trim().slice(0, 30);
        return label && label !== element.selector ? `${element.selector} ("${label}")` : element.selector;
      })
      .join(" or ");
  }

  // --- TEMPORARY debug logging helpers (no effect on behaviour) -------------
  const TOKEN_VALUE = /^\[TOKEN:[A-Z0-9_]+\]$/;
  const maskDebugValue = (value) => (value == null ? null : TOKEN_VALUE.test(String(value)) ? value : "[value omitted]");

  function debugActions(actions) {
    return (Array.isArray(actions) ? actions : []).map((action) => ({
      action: action?.action ?? null,
      selector: action?.selector ?? null,
      target: action?.target ?? null,
      frame_id: action?.frame_id ?? null,
      value: maskDebugValue(action?.value)
    }));
  }

  function debugResponse(response) {
    if (!response || typeof response !== "object") {
      return response ?? null;
    }

    return {
      ...response,
      ...(Object.prototype.hasOwnProperty.call(response, "value") ? { value: maskDebugValue(response.value) } : {}),
      actions: debugActions(response.actions)
    };
  }

  function describeActions(actions) {
    return (actions ?? [])
      .map((action) => `${action.action} ${action.target || action.selector || ""}`.trim())
      .join(", ");
  }

  function normalizeSteps(steps) {
    return (Array.isArray(steps) ? steps : [])
      .map((step) => (typeof step === "string" ? step.trim() : ""))
      .filter(Boolean);
  }

  function comparableText(text) {
    return String(text ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  }

  /** A step that merely restates the goal is not a plan. */
  function isEchoOfGoal(step, goal) {
    const stepText = comparableText(step);
    const goalText = comparableText(goal);

    if (!stepText || !goalText) {
      return false;
    }

    if (stepText === goalText) {
      return true;
    }

    // One containing the other only counts as an echo when almost nothing was
    // added or removed, so a genuine sub-step is never mistaken for the goal.
    const [shorter, longer] = stepText.length <= goalText.length ? [stepText, goalText] : [goalText, stepText];
    return longer.includes(shorter) && shorter.length >= longer.length * 0.8;
  }

  /**
   * Goals joining several independent actions ("fill the form and click
   * submit") must be broken down; an atomic goal ("click Login") may legitimately
   * stay a single step.
   */
  function goalLooksMultiAction(goal) {
    return /\b(?:and|then|after that|followed by)\b|[;,]/i.test(String(goal ?? ""));
  }

  /**
   * Planner output is only a plan once it says something the goal did not.
   * Returns [] when the result is unusable, which the caller fails closed on.
   */
  function validatePlanSteps(steps, goal) {
    const descriptions = normalizeSteps(steps);

    if (descriptions.length === 0) {
      return [];
    }

    if (descriptions.length === 1 && isEchoOfGoal(descriptions[0], goal) && goalLooksMultiAction(goal)) {
      return [];
    }

    return descriptions;
  }

  function createAgentState(goal, steps) {
    const descriptions = normalizeSteps(steps);

    return {
      goal,
      // The raw goal is never used as a plan step: planning is a distinct stage
      // and a run without a usable plan fails closed instead of executing it.
      plan: descriptions
        .map((description) => ({ description, status: "pending" })),
      currentStepIndex: 0,
      // Steps whose actions have run but whose effect the next observation has
      // not confirmed yet. Display only: the loop still trusts
      // currentStepIndex, which moves when the page proves the work landed.
      executedThroughIndex: 0,
      stepAttempts: {},
      replanCount: 0,
      needsReplan: false,
      lastExecution: null,
      lastActionResult: "",
      consecutiveModelErrors: 0,
      noProgressCount: 0,
      round: 0,
      status: "running",
      reason: ""
    };
  }

  function completedSteps(state) {
    return state.plan.slice(0, state.currentStepIndex).map((step) => step.description);
  }

  function remainingSteps(state) {
    return state.plan.slice(state.currentStepIndex).map((step) => step.description);
  }

  function isPlanFinished(state) {
    return state.currentStepIndex >= state.plan.length;
  }

  /** Plan progress sent with every model request. */
  function buildStepContext(state, observation) {
    return {
      task_plan: state.plan.map((step) => step.description),
      completed_steps: completedSteps(state),
      remaining_steps: remainingSteps(state),
      last_action_result: state.lastActionResult,
      current_url: observation?.url ?? ""
    };
  }

  /** Steps already acted on, so the UI can show them as underway, not untouched.
   *  step.status stays reserved for verified completion. */
  function markExecuted(state, stepIndex, executedCount) {
    const reached = Math.min(stepIndex + Math.max(executedCount, 0), state.plan.length);
    state.executedThroughIndex = Math.max(state.executedThroughIndex, reached, state.currentStepIndex);
  }

  /** Monotonic: steps are never un-completed and the plan never restarts. */
  function advanceTo(state, stepCount) {
    const target = Math.min(Math.max(stepCount, state.currentStepIndex), state.plan.length);

    if (target === state.currentStepIndex) {
      return false;
    }

    for (let index = state.currentStepIndex; index < target; index += 1) {
      state.plan[index].status = "done";
    }

    state.currentStepIndex = target;
    state.executedThroughIndex = Math.max(state.executedThroughIndex, target);
    return true;
  }

  function indexElements(elements) {
    const index = new Map();

    for (const element of elements ?? []) {
      if (element?.selector) {
        index.set(normalizeSelector(element.selector), element);
      }
    }

    return index;
  }

  const elementState = (element) =>
    [element.label, element.type, element.role, element.expanded, element.checked, element.pressed, element.selected]
      .map((part) => part ?? "")
      .join("|");

  /**
   * Did the fresh post-action observation show the intended state change?
   * Generic evidence only:
   * - navigation: the URL changed;
   * - fill: the target's observed value changed;
   * - click / press / scroll: interactive elements appeared or disappeared (a
   *   modal, drawer, panel or view opened or closed), or the acted-on element
   *   changed label/state or disappeared.
   */
  function observedExecutionProgress(execution, observation) {
    if (!execution || !execution.ok || execution.waitOnly || !observation) {
      return false;
    }

    if (observation.url !== execution.urlBefore) {
      return true;
    }

    const before = indexElements(execution.elementsBefore);
    const after = indexElements(observation.elements);
    const hasValue = (element) => element && Object.prototype.hasOwnProperty.call(element, "value");

    for (const action of execution.actions ?? []) {
      const key = normalizeSelector(action.selector);
      const was = before.get(key);
      const now = after.get(key);

      if (action.action === "fill") {
        if (hasValue(was) && hasValue(now) && String(was.value) !== String(now.value)) {
          return true;
        }
        continue;
      }

      if (action.action === "wait") {
        continue;
      }

      if (was && (!now || elementState(was) !== elementState(now))) {
        return true;
      }
    }

    const actedBeyondFill = (execution.actions ?? []).some((action) => !["fill", "wait"].includes(action.action));

    return actedBeyondFill && (
      before.size !== after.size || [...after.keys()].some((key) => !before.has(key))
    );
  }

  /**
   * Step verification. The model's done_through is a CLAIM: the plan advances
   * only when the fresh observation shows the last execution's intended state
   * change, and never further than that execution can account for (one step
   * per executed action). Each execution is verified once, so a duplicated or
   * re-asked response cannot advance the plan twice.
   */
  function applyStepVerification(state, response, observation) {
    const execution = state.lastExecution;

    if (!execution || !execution.ok || execution.consumed) {
      return false;
    }

    const verified = observedExecutionProgress(execution, observation);
    execution.consumed = true;

    if (!verified) {
      return false;
    }

    const claimed = Number.isInteger(response?.done_through) ? response.done_through : execution.stepIndex + 1;
    const verifiedLimit = execution.stepIndex + Math.max(1, execution.executedCount);

    return advanceTo(state, Math.min(claimed, verifiedLimit));
  }

  function recordExecution(state, { stepIndex, actions, executed, error, observation }) {
    state.stepAttempts[stepIndex] = (state.stepAttempts[stepIndex] ?? 0) + 1;
    state.lastExecution = {
      stepIndex,
      ok: !error,
      fingerprint: actionFingerprint(actions),
      waitOnly: isWaitOnly(actions),
      actions: (actions ?? []).map((action) => ({ action: action?.action, selector: action?.selector })),
      elementsBefore: observation?.elements ?? [],
      executedCount: (executed ?? []).filter((action) => action?.action !== "wait").length,
      signatureBefore: observation?.signature,
      urlBefore: observation?.url,
      consumed: false,
      goalChecked: false
    };
    if (!error) {
      markExecuted(state, stepIndex, state.lastExecution.executedCount);
    }

    state.lastActionResult = (error
      ? `step ${stepIndex + 1} failed: ${errorText(error)}`
      : `executed ${describeActions(executed)} for step ${stepIndex + 1}`
    ).slice(0, MAX_RESULT_CHARS);
  }

  /** Same step, same action, and the page is identical to before that action ran. */
  function isRepeatedNoProgress(state, actions, observation) {
    const execution = state.lastExecution;

    // Waits have their own no-progress counter.
    return Boolean(
      !isWaitOnly(actions) &&
      execution &&
      execution.ok &&
      execution.stepIndex === state.currentStepIndex &&
      execution.fingerprint === actionFingerprint(actions) &&
      observation?.signature === execution.signatureBefore &&
      observation?.url === execution.urlBefore
    );
  }

  /** The last round executed at least one action per remaining step. */
  function lastExecutionCoversRemainingPlan(state) {
    const execution = state.lastExecution;

    return Boolean(
      execution &&
      execution.ok &&
      !execution.goalChecked &&
      execution.stepIndex === state.currentStepIndex &&
      execution.executedCount >= state.plan.length - state.currentStepIndex
    );
  }

  async function runAgentLoop(deps) {
    const maxRounds = deps.maxRounds;
    const maxReplans = deps.maxReplans ?? DEFAULT_MAX_REPLANS;
    const maxStepAttempts = deps.maxStepAttempts ?? DEFAULT_MAX_STEP_ATTEMPTS;
    const status = deps.onStatus ?? (() => {});
    // TEMPORARY: optional debug sink; absent in tests and when disabled.
    const debug = deps.onDebug ?? (() => {});
    const publish = deps.onState ?? (() => {});

    let observation = await deps.observe({ round: 1, purpose: "initial" });

    // PLAN ONCE. Planning is a required stage: no plan, no action generation.
    status("Planning task steps...");
    let initialSteps = [];
    let planFailure = "";
    try {
      initialSteps = validatePlanSteps(
        await deps.requestPlan({ observation, replan: false, context: null }),
        deps.goal
      );

      if (initialSteps.length === 0) {
        planFailure = "the planner did not return usable steps for this goal";
      }
    } catch (error) {
      if (isRateLimitError(error)) {
        throw error;
      }
      planFailure = `planning request failed: ${error?.message ?? error}`;
    }

    const state = createAgentState(deps.goal, initialSteps);
    publish(state);

    const finish = (outcome, reason) => {
      state.status = outcome;
      state.reason = String(reason ?? "").slice(0, 300);
      publish(state);
      return state;
    };

    // Fail closed rather than executing the raw goal as if it were a plan.
    if (planFailure || state.plan.length === 0) {
      status("Planning did not produce a usable plan.");
      return finish("failed", planFailure || "the planner did not return usable steps for this goal");
    }

    // GOAL VERIFICATION always runs on a fresh observation of its own.
    const verifyGoal = async () => {
      status("Verifying the whole task on a fresh page read...");
      observation = await deps.observe({ round: state.round, purpose: "verify" });

      if (state.lastExecution) {
        state.lastExecution.goalChecked = true;
      }

      const verdict = await deps.verifyGoal({
        observation,
        completedSteps: completedSteps(state),
        plan: state.plan.map((step) => step.description)
      });

      return verdict ?? { success: false, reason: "no verification verdict" };
    };

    // RE-PLAN from the current observation; completed steps are kept.
    const replan = async ({ keepCurrentStep = false, reason = "" } = {}) => {
      if (state.replanCount >= maxReplans) {
        return false;
      }

      state.replanCount += 1;
      status(`Re-planning from the current page (${state.replanCount}/${maxReplans})...`);

      const context = buildStepContext(state, observation);

      if (reason) {
        context.last_action_result = reason;
      }

      let steps;
      try {
        steps = validatePlanSteps(await deps.requestPlan({ observation, replan: true, context }), deps.goal);
      } catch (error) {
        if (isRateLimitError(error)) {
          throw error;
        }
        steps = [];
      }

      state.needsReplan = false;

      // Completed steps are always kept. A wait-triggered re-plan also keeps
      // the pending step and replaces only the steps after it.
      const keepCount = Math.min(state.currentStepIndex + (keepCurrentStep ? 1 : 0), state.plan.length);

      if (keepCount > state.currentStepIndex) {
        const pending = state.plan[state.currentStepIndex].description.toLowerCase();

        while (steps.length > 0 && steps[0].toLowerCase() === pending) {
          steps.shift();
        }
      }

      if (steps.length === 0) {
        return false;
      }

      state.plan = [
        ...state.plan.slice(0, keepCount),
        ...steps.map((description) => ({ description, status: "pending" }))
      ];
      state.stepAttempts = {};
      state.lastExecution = null;
      // The replaced steps were never verified, so nothing beyond the kept
      // ones counts as executed any more.
      state.executedThroughIndex = Math.max(state.currentStepIndex, Math.min(state.executedThroughIndex, keepCount));

      if (reason) {
        state.lastActionResult = reason;
      }

      publish(state);
      return true;
    };

    // Set after a wait: that settled capture is reused instead of re-capturing.
    let pendingObservation = null;

    for (let round = 1; round <= maxRounds; round += 1) {
      state.round = round;
      let goalVerifiedThisRound = false;

      // OBSERVE the page the previous round's actions produced.
      if (round > 1) {
        if (pendingObservation) {
          observation = pendingObservation;
          pendingObservation = null;
        } else {
          status(`Round ${round} of ${maxRounds}: re-reading the page...`);
          observation = await deps.observe({ round, purpose: "act" });
        }
      }

      // A round that already attempted every remaining step only needs the
      // goal check - no extra action request.
      if (lastExecutionCoversRemainingPlan(state)) {
        goalVerifiedThisRound = true;
        const verdict = await verifyGoal();

        if (verdict.success) {
          advanceTo(state, state.plan.length);
          return finish("complete", verdict.reason);
        }

        state.lastActionResult = `whole task not satisfied yet: ${verdict.reason}`.slice(0, MAX_RESULT_CHARS);
      }

      let replannedThisRound = false;
      let actionResponse = null;
      // One corrective re-request per round when the model picks a stale
      // selector or waits although the step's target is on the page.
      let correctedThisRound = false;
      let feedback = null;

      // At most one re-plan (and so at most two action requests) per round.
      while (true) {
        if (state.needsReplan && !replannedThisRound) {
          replannedThisRound = true;
          await replan();
        }

        const canReplanNow = () => !replannedThisRound && state.replanCount < maxReplans;
        let response;

        const requestFeedback = feedback;
        feedback = null;
        const requestStepIndex = state.currentStepIndex;
        const requestedStep = state.plan[requestStepIndex]?.description ?? null;

        // Obvious steps are resolved locally on a clean first attempt only;
        // any retry, correction or replan goes to the model.
        const localAction = (
          deps.allowLocalPrediction !== false &&
          !requestFeedback &&
          !state.needsReplan &&
          (state.stepAttempts[requestStepIndex] ?? 0) === 0 &&
          !(state.lastExecution && !state.lastExecution.consumed && !state.lastExecution.ok)
        ) ? predictLocalAction(requestedStep, observation) : null;

        try {
          if (localAction) {
            response = { reasoning: localAction.reasoning, actions: [localAction], source: "local" };
            debug("local-prediction", { round, requestedStep, action: debugActions([localAction])[0] });
          } else {
            response = await deps.requestAction({
              observation,
              context: buildStepContext(state, observation),
              state,
              feedback: requestFeedback
            });
          }
          state.consecutiveModelErrors = 0;
        } catch (error) {
          // The server already honoured Groq's retry window; a 429 reaching
          // here aborts the run instead of re-sending the same logical step.
          if (isRateLimitError(error)) {
            throw error;
          }

          state.consecutiveModelErrors += 1;
          state.lastActionResult = `model error: ${errorText(error)}`.slice(0, MAX_RESULT_CHARS);
          break;
        }

        // VERIFY STEP on the same fresh observation the action was chosen from.
        const advanced = applyStepVerification(state, response, observation);
        publish(state);

        const debugBase = {
          round,
          requestedStepNumber: requestStepIndex + 1,
          requestedStep,
          feedbackSent: requestFeedback,
          parsedResponse: debugResponse(response),
          selectedAction: debugActions(response?.actions)[0] ?? null,
          stepAdvanced: advanced
        };
        const report = (decision, extra = {}) => debug("action-response", {
          ...debugBase,
          stepAfterVerification: state.currentStepIndex + 1,
          currentStep: state.plan[state.currentStepIndex]?.description ?? null,
          decision,
          ...extra
        });

        if (isPlanFinished(state)) {
          report("plan-finished");

          if (goalVerifiedThisRound) {
            break;
          }

          goalVerifiedThisRound = true;
          const verdict = await verifyGoal();

          if (verdict.success) {
            return finish("complete", verdict.reason);
          }

          state.lastActionResult = `planned steps look done but the task is not satisfied: ${verdict.reason}`
            .slice(0, MAX_RESULT_CHARS);
          state.needsReplan = true;

          if (canReplanNow()) {
            continue;
          }
          break;
        }

        const stepNumber = state.currentStepIndex + 1;
        let actions = Array.isArray(response?.actions) ? response.actions : [];

        if (!advanced && (state.stepAttempts[state.currentStepIndex] ?? 0) >= maxStepAttempts) {
          state.lastActionResult = `step ${stepNumber} made no progress after ${maxStepAttempts} attempts`;
          state.needsReplan = true;
          report("replan-no-progress");

          if (canReplanNow()) {
            continue;
          }
        }

        if (response?.replan) {
          report("replan-requested");
          state.lastActionResult = `step ${stepNumber} cannot be done from this page`;
          state.needsReplan = true;

          if (canReplanNow()) {
            continue;
          }
          break;
        }

        if (response?.complete && actions.length === 0) {
          report("model-complete");

          if (goalVerifiedThisRound) {
            break;
          }

          goalVerifiedThisRound = true;
          const verdict = await verifyGoal();

          if (verdict.success) {
            advanceTo(state, state.plan.length);
            return finish("complete", verdict.reason);
          }

          state.lastActionResult = `model reported completion but verification disagreed: ${verdict.reason}`
            .slice(0, MAX_RESULT_CHARS);
          break;
        }

        if (actions.length === 0) {
          report("no-action");
          state.lastActionResult = `model returned no action for step ${stepNumber}`;
          break;
        }

        // The CURRENT page is authoritative: never execute a selector that is
        // not on it.
        const resolution = resolveActionsAgainstObservation(actions, observation);

        if (resolution.stale.length > 0) {
          report("selector-rejected", { staleSelectors: resolution.stale, reAsk: !correctedThisRound });
          state.lastActionResult = `selector ${resolution.stale[0]} is not on the current page`.slice(0, MAX_RESULT_CHARS);

          if (!correctedThisRound) {
            correctedThisRound = true;
            feedback = `Selector ${resolution.stale[0]} is not on the current page; choose only from the current Interactive Elements.`;
            continue;
          }
          break;
        }

        actions = resolution.actions;

        // Prefer the real action over waiting when the step's target is present.
        if (isWaitOnly(actions) && !correctedThisRound) {
          const candidates = findStepCandidates(
            state.plan[state.currentStepIndex]?.description,
            observation?.elements,
            actions.map((action) => action?.target).filter(Boolean)
          );

          if (candidates.length > 0) {
            report("wait-correction", {
              remapped: resolution.remapped,
              candidates: candidates.map((candidate) => candidate.selector)
            });
            correctedThisRound = true;
            feedback = `Do not wait: the current page has ${describeCandidates(candidates)} for this step; act on it.`;
            continue;
          }
        }

        // Semantic relevance: never execute a control that clearly serves a
        // different intent than the pending step.
        const irrelevant = findIrrelevantAction(state, actions, observation);

        if (irrelevant) {
          report("action-irrelevant", {
            rejectedAction: debugActions([irrelevant.action])[0],
            rejectionReason: irrelevant.reason,
            reAsk: !correctedThisRound
          });
          state.lastActionResult = `rejected "${irrelevant.name}": ${irrelevant.reason}, not step ${stepNumber}`.slice(0, MAX_RESULT_CHARS);

          if (!correctedThisRound) {
            correctedThisRound = true;
            feedback = `Rejected "${irrelevant.name}": it ${irrelevant.reason}, unrelated to the current step. Choose an element for this step.`;
            continue;
          }
          break;
        }

        // LOOP DETECTION: never re-run an action that already had no effect.
        if (isRepeatedNoProgress(state, actions, observation)) {
          report("loop-detected", { resolvedActions: debugActions(actions) });
          state.lastActionResult = `repeating the same action on step ${stepNumber} had no effect`;
          state.needsReplan = true;

          if (canReplanNow()) {
            continue;
          }
          break;
        }

        report("accepted", {
          remapped: resolution.remapped,
          resolvedActions: debugActions(actions)
        });
        actionResponse = { ...response, actions };
        break;
      }

      if (state.consecutiveModelErrors >= 2) {
        return finish("failed", `The model failed twice in a row (${state.lastActionResult}).`);
      }

      if (!actionResponse) {
        const index = Math.min(state.currentStepIndex, state.plan.length - 1);
        state.stepAttempts[index] = (state.stepAttempts[index] ?? 0) + 1;
        publish(state);
        continue;
      }

      const waitOnlyResponse = isWaitOnly(actionResponse.actions);

      // Never run another wait once waits have stopped producing progress.
      if (waitOnlyResponse && state.noProgressCount >= MAX_NO_PROGRESS_WAITS) {
        state.lastActionResult = WAIT_NO_PROGRESS_REASON;
        const index = Math.min(state.currentStepIndex, state.plan.length - 1);
        state.stepAttempts[index] = (state.stepAttempts[index] ?? 0) + 1;
        publish(state);
        continue;
      }

      // ACT on the current step only.
      const stepIndex = state.currentStepIndex;
      status(`Step ${stepIndex + 1} of ${state.plan.length}: ${state.plan[stepIndex].description}`);

      let executed = [];
      let executionError = null;

      try {
        executed = (await deps.executeActions({
          response: actionResponse,
          context: buildStepContext(state, observation),
          stepIndex,
          round
        })) ?? [];
      } catch (error) {
        if (isRateLimitError(error)) {
          throw error;
        }
        executionError = error;
      }

      recordExecution(state, { stepIndex, actions: actionResponse.actions, executed, error: executionError, observation });

      if (!executionError && waitOnlyResponse) {
        // OBSERVE right after the wait and judge whether it achieved anything.
        const beforeWait = observation;
        observation = await deps.observe({ round, purpose: "after-wait" });
        pendingObservation = observation;

        if (hasMeaningfulChange(beforeWait, observation)) {
          state.noProgressCount = 0;
        } else {
          state.noProgressCount += 1;
          state.lastActionResult =
            `wait on step ${stepIndex + 1} produced no page progress (${state.noProgressCount}/${MAX_NO_PROGRESS_WAITS})`;

          if (
            state.noProgressCount >= MAX_NO_PROGRESS_WAITS &&
            await replan({ keepCurrentStep: true, reason: WAIT_NO_PROGRESS_REASON })
          ) {
            state.noProgressCount = 0;
          }
        }
      } else if (!executionError) {
        state.noProgressCount = 0;
      }

      publish(state);
    }

    // Rounds exhausted: the last actions have not been checked yet.
    if (state.lastExecution?.ok && !state.lastExecution.goalChecked) {
      const verdict = await verifyGoal();

      if (verdict.success) {
        advanceTo(state, state.plan.length);
        return finish("complete", verdict.reason);
      }

      return finish("incomplete", verdict.reason);
    }

    return finish("incomplete", state.lastActionResult || `Stopped after ${maxRounds} rounds.`);
  }

  const api = {
    DEFAULT_MAX_REPLANS,
    DEFAULT_MAX_STEP_ATTEMPTS,
    MAX_NO_PROGRESS_WAITS,
    WAIT_NO_PROGRESS_REASON,
    runAgentLoop,
    resolveActionsAgainstObservation,
    findStepCandidates,
    assessActionRelevance,
    predictLocalAction,
    createAgentState,
    validatePlanSteps,
    isEchoOfGoal,
    goalLooksMultiAction,
    buildStepContext,
    applyStepVerification,
    observedExecutionProgress,
    recordExecution,
    isRepeatedNoProgress,
    pageSignature,
    actionFingerprint,
    completedSteps,
    remainingSteps
  };

  globalThis.__privacyVisionAgentLoop = api;

  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  }
})();
