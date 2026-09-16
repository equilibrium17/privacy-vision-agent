// Tests for the explicit plan-execution controller (agent-loop.js).
// The environment (capture, model, executor, verifier) is faked, so these tests
// exercise plan progression logic, not a browser or a model.
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../agent-loop.js", import.meta.url), "utf8");
new Function(source)();

const Loop = globalThis.__privacyVisionAgentLoop;

const page = (url, selectors) => ({
  url,
  elements: selectors.map((selector) => ({ tag: "button", selector, label: selector }))
});

const click = (selector) => ({ action: "click", selector, target: selector, reasoning: "" });

/**
 * Fake environment. `responses` are consumed one per action request (an Error
 * is thrown, a function is called with the request). `onExecute` may move the
 * fake browser to another page.
 */
function selectorsIn(responses) {
  const selectors = new Set(["#a"]);

  for (const response of responses) {
    for (const action of response?.actions ?? []) {
      if (action?.selector && action.action !== "wait") {
        selectors.add(action.selector);
      }
    }
  }

  return [...selectors];
}

function makeEnv({ plan = [], pages: explicitPages, responses = [], verify, onExecute, maxRounds = 5, navigateOnExecute = false, allowLocalPrediction, goal = "the user goal" }) {
  const pages = explicitPages ?? [page("https://site.test/", selectorsIn(responses))];
  const events = [];
  const snapshots = [];
  let pageIndex = 0;
  let observeCount = 0;

  const deps = {
    goal,
    maxRounds,
    ...(allowLocalPrediction === false ? { allowLocalPrediction: false } : {}),
    observe: async ({ purpose }) => {
      observeCount += 1;
      const current = pages[pageIndex];
      const executions = events.filter((event) => event.type === "execute").length;
      const url = navigateOnExecute && executions > 0 ? `${current.url}#state-${executions}` : current.url;
      const observation = {
        id: observeCount,
        purpose,
        url,
        elements: current.elements,
        signature: Loop.pageSignature(url, current.elements)
      };
      events.push({ type: "observe", observation });
      return observation;
    },
    requestPlan: async (args) => {
      events.push({ type: "plan", args });
      if (plan instanceof Error) {
        throw plan;
      }
      return typeof plan === "function" ? plan(args) : plan;
    },
    requestAction: async (args) => {
      events.push({ type: "action", args });
      const next = responses.length > 0 ? responses.shift() : { reasoning: "idle", actions: [] };
      if (next instanceof Error) {
        throw next;
      }
      return typeof next === "function" ? next(args) : next;
    },
    executeActions: async (args) => {
      events.push({ type: "execute", args });
      if (onExecute) {
        const moveTo = await onExecute(args, events.filter((event) => event.type === "execute").length);
        if (Number.isInteger(moveTo)) {
          pageIndex = moveTo;
        }
      }
      return args.response.actions;
    },
    verifyGoal: async (args) => {
      events.push({ type: "verify", args });
      return verify ? verify(args, events) : { success: false, reason: "not satisfied" };
    },
    onState: (state) => snapshots.push(JSON.parse(JSON.stringify(state)))
  };

  const of = (type) => events.filter((event) => event.type === type);
  return { deps, events, snapshots, of };
}

// --- TEST 1 ----------------------------------------------------------------

test("TEST 1: after step 1 is verified, the next request targets step 2, not step 1", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    plan: ["Search", "Open product", "Add to cart"],
    responses: [
      { actions: [{ action: "fill", selector: "#q", value: "laptop", reasoning: "" }] },
      { done_through: 1, actions: [click("#product")] },
      { done_through: 2, actions: [click("#add")] }
    ],
    verify: () => ({ success: true, reason: "cart shows the item" })
  });

  const state = await Loop.runAgentLoop(env.deps);
  const requests = env.of("action").map((event) => event.args.context);

  assert.equal(requests[0].remaining_steps[0], "Search");
  assert.deepEqual(requests[2].completed_steps, ["Search"]);
  assert.equal(requests[2].remaining_steps[0], "Open product");
  assert.equal(requests.some((context, index) => index >= 2 && context.remaining_steps[0] === "Search"), false);

  assert.equal(state.status, "complete");
  assert.deepEqual(env.of("execute").map((event) => event.args.stepIndex), [0, 1, 2]);
});

// --- TEST 2 ----------------------------------------------------------------

test("TEST 2: after an action navigates, step 2 is chosen from the fresh page", async () => {
  const listing = page("https://shop.test/search", ["#q", "#result-1"]);
  const product = page("https://shop.test/product/1", ["#add-to-cart"]);

  const env = makeEnv({
    plan: ["Open the first product", "Add it to cart"],
    pages: [listing, product],
    onExecute: (_args, count) => (count === 1 ? 1 : undefined),
    responses: [
      { actions: [click("#result-1")] },
      { done_through: 1, actions: [click("#add-to-cart")] }
    ],
    verify: () => ({ success: true, reason: "added" })
  });

  await Loop.runAgentLoop(env.deps);

  const secondRequest = env.of("action")[1].args;
  assert.equal(secondRequest.observation.url, product.url);
  assert.deepEqual(secondRequest.observation.elements.map((element) => element.selector), ["#add-to-cart"]);
  assert.equal(secondRequest.context.current_url, product.url);

  // The observation feeding request 2 was captured after execution 1.
  const firstExecute = env.events.findIndex((event) => event.type === "execute");
  const observationIndex = env.events.findIndex((event) => event.type === "observe" && event.observation === secondRequest.observation);
  assert.ok(observationIndex > firstExecute);
});

// --- TEST 3 ----------------------------------------------------------------

test("TEST 3: a failed action never marks its step complete, even if the model claims it", async () => {
  const env = makeEnv({
    plan: ["A", "B", "C"],
    maxRounds: 2,
    responses: [
      { actions: [click("#a")] },
      { done_through: 1, actions: [click("#a-again")] }
    ],
    onExecute: (_args, count) => {
      if (count === 1) {
        throw new Error("No element matched the selector: #a");
      }
    },
    verify: () => ({ success: false, reason: "step A not done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.currentStepIndex, 0);
  assert.equal(state.plan[0].status, "pending");
  assert.notEqual(state.status, "complete");
  // The retry still targets step 1.
  assert.deepEqual(env.of("execute").map((event) => event.args.stepIndex), [0, 0]);
});

// --- TEST 4 ----------------------------------------------------------------

test("TEST 4: repeating an action with no page change triggers re-planning, not endless repetition", async () => {
  const env = makeEnv({
    plan: (args) => (args.replan ? ["A (revised)", "B"] : ["A", "B"]),
    responses: Array.from({ length: 20 }, () => ({ actions: [click("#same")] })),
    verify: () => ({ success: false, reason: "nothing happened" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  const replans = env.of("plan").filter((event) => event.args.replan);
  assert.ok(replans.length >= 1, "expected at least one re-plan");
  assert.ok(state.replanCount <= Loop.DEFAULT_MAX_REPLANS);

  // Between two executions of the same action on an unchanged page there must
  // be a re-plan.
  let lastExecuteAt = -1;
  for (let index = 0; index < env.events.length; index += 1) {
    if (env.events[index].type !== "execute") {
      continue;
    }
    if (lastExecuteAt >= 0) {
      const between = env.events.slice(lastExecuteAt + 1, index);
      assert.ok(between.some((event) => event.type === "plan"), "identical no-effect action re-executed without re-planning");
    }
    lastExecuteAt = index;
  }

  assert.ok(env.of("execute").length <= env.deps.maxRounds);
  assert.notEqual(state.status, "complete");
});

// --- TEST 5 ----------------------------------------------------------------

test("TEST 5: the task is marked complete only after the final step and a passing goal check", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    plan: ["Open cart", "Set quantity to 2"],
    responses: [
      { actions: [click("#cart")] },
      { done_through: 1, actions: [{ action: "fill", selector: "#quantity-1", value: "2", reasoning: "" }] }
    ],
    verify: (_args, events) => {
      const lastStepExecuted = events.some((event) => event.type === "execute" && event.args.stepIndex === 1);
      return { success: lastStepExecuted, reason: lastStepExecuted ? "quantity is 2" : "not yet" };
    }
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "complete");
  const verifyAt = env.events.findIndex((event) => event.type === "verify");
  const finalExecuteAt = env.events.findIndex((event) => event.type === "execute" && event.args.stepIndex === 1);
  assert.ok(verifyAt > finalExecuteAt);
  assert.equal(env.snapshots.slice(0, -1).some((snapshot) => snapshot.status === "complete"), false);
});

test("TEST 5b: a failing goal check after the last step does not produce success", async () => {
  const env = makeEnv({
    plan: ["Open cart", "Set quantity to 2"],
    responses: [
      { actions: [click("#cart")] },
      { done_through: 1, actions: [{ action: "fill", selector: "#quantity-1", value: "2", reasoning: "" }] },
      { done_through: 2, reasoning: "looks done", actions: [] }
    ],
    verify: () => ({ success: false, reason: "quantity still 1" })
  });

  const state = await Loop.runAgentLoop(env.deps);
  assert.notEqual(state.status, "complete");
});

// --- TEST 6 ----------------------------------------------------------------

test("TEST 6: goal verification receives a fresh observation taken after the last action", async () => {
  const listing = page("https://shop.test/", ["#cart"]);
  const cart = page("https://shop.test/cart", ["#quantity-1", "#checkout"]);

  const env = makeEnv({
    plan: ["Open cart"],
    pages: [listing, cart],
    onExecute: () => 1,
    responses: [{ actions: [click("#cart")] }],
    verify: () => ({ success: true, reason: "cart open" })
  });

  await Loop.runAgentLoop(env.deps);

  const verifyEvent = env.of("verify")[0];
  const observations = env.of("observe").map((event) => event.observation);
  const latest = observations[observations.length - 1];

  assert.equal(verifyEvent.args.observation, latest);
  assert.equal(latest.purpose, "verify");
  assert.equal(latest.url, cart.url);

  const executeAt = env.events.findIndex((event) => event.type === "execute");
  const observeAt = env.events.findIndex((event) => event.type === "observe" && event.observation === latest);
  const verifyAt = env.events.indexOf(verifyEvent);
  assert.ok(executeAt < observeAt && observeAt < verifyAt);
});

// --- TEST 7 ----------------------------------------------------------------

test("TEST 7: a 429 surfacing from the server aborts without re-running or double-advancing a step", async () => {
  const env = makeEnv({
    plan: ["A", "B"],
    responses: [
      { actions: [click("#a")] },
      new Error("Agent server error (HTTP 429): Groq rate limit reached. Retry in 3 seconds.")
    ]
  });

  await assert.rejects(Loop.runAgentLoop(env.deps), /429/);

  assert.equal(env.of("execute").length, 1);
  assert.equal(env.of("action").length, 2);
  const last = env.snapshots[env.snapshots.length - 1];
  assert.equal(last.currentStepIndex, 0);
});

test("TEST 7b: applying the same step verification twice advances the plan once", () => {
  const state = Loop.createAgentState("goal", ["A", "B", "C"]);
  const observation = { url: "https://site.test/next", signature: "after" };

  Loop.recordExecution(state, {
    stepIndex: 0,
    actions: [click("#a")],
    executed: [click("#a")],
    observation: { url: "https://site.test/", signature: "before" }
  });

  // Reported progress (absolute step number).
  Loop.applyStepVerification(state, { done_through: 1 }, observation);
  Loop.applyStepVerification(state, { done_through: 1 }, observation);
  assert.equal(state.currentStepIndex, 1);

  // Fallback progress (observable change) is consumed after one use.
  const fallback = Loop.createAgentState("goal", ["A", "B", "C"]);
  Loop.recordExecution(fallback, {
    stepIndex: 0,
    actions: [click("#a")],
    executed: [click("#a")],
    observation: { url: "https://site.test/", signature: "before" }
  });
  Loop.applyStepVerification(fallback, {}, observation);
  Loop.applyStepVerification(fallback, {}, observation);
  assert.equal(fallback.currentStepIndex, 1);
});

// --- regressions -------------------------------------------------------------

test("regression: a same-page action group still finishes with a single action request", async () => {
  const env = makeEnv({
    plan: ["Fill email", "Fill password", "Click login"],
    responses: [{
      actions: [
        { action: "fill", selector: "#email", value: "[TOKEN:USER_EMAIL]", reasoning: "" },
        { action: "fill", selector: "#password", value: "[TOKEN:USER_PASSWORD]", reasoning: "" },
        click("#login")
      ]
    }],
    verify: () => ({ success: true, reason: "logged in" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "complete");
  assert.equal(env.of("action").length, 1);
  assert.equal(env.of("verify").length, 1);
});

test("regression: a model failure is never reported as task completion", async () => {
  const modelFailure = () => new Error("Agent server error (HTTP 502): The AI model couldn't produce a valid response after 2 attempts.");
  const env = makeEnv({
    plan: ["A", "B"],
    responses: [modelFailure(), modelFailure()],
    verify: () => ({ success: true, reason: "should not be asked" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "failed");
  assert.equal(env.of("verify").length, 0);
});

test("regression: when planning is unavailable the run fails closed instead of executing the goal", async () => {
  // Previously the raw goal became a single plan step. Planning is a required
  // stage now, so an unavailable planner stops the run before any action.
  const env = makeEnv({
    plan: new Error("Planning error (HTTP 502): the AI provider could not produce a valid plan."),
    responses: [{ actions: [click("#go")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "failed");
  assert.deepEqual(state.plan, []);
  // The goal never became an executable step, and nothing was executed.
  assert.equal(env.of("action").length, 0);
  assert.equal(env.of("execute").length, 0);
});

test("a replan keeps completed steps and replaces only the remaining ones", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    plan: (args) => (args.replan ? ["Use the search box instead", "Open product"] : ["Search", "Click missing link", "Open product"]),
    responses: [
      { actions: [{ action: "fill", selector: "#q", value: "x", reasoning: "" }] },
      { done_through: 1, reasoning: "link not here", replan: true, actions: [{ action: "wait", selector: "body", reasoning: "" }] },
      { done_through: 1, actions: [click("#search-go")] }
    ],
    verify: () => ({ success: false, reason: "unfinished" }),
    maxRounds: 2
  });

  const state = await Loop.runAgentLoop(env.deps);

  const replan = env.of("plan").find((event) => event.args.replan);
  assert.ok(replan, "expected a re-plan");
  assert.deepEqual(replan.args.context.completed_steps, ["Search"]);
  assert.deepEqual(state.plan.map((step) => step.description), ["Search", "Use the search box instead", "Open product"]);
  assert.equal(state.plan[0].status, "done");
  // The wait escape hatch from the replan response was never executed.
  assert.equal(env.of("execute").some((event) => event.args.response.actions[0].action === "wait"), false);
});

// --- wait -> no progress -> replan ---------------------------------------------

const waitAction = () => ({ action: "wait", selector: "body", reasoning: "" });

test("wait: a page change after the wait resets no-progress and continues normally", async () => {
  const loading = page("https://shop.test/search?loading", ["#spinner"]);
  const results = page("https://shop.test/search", ["#result-1"]);

  const env = makeEnv({
    plan: ["Open result"],
    pages: [loading, results],
    onExecute: (_args, count) => (count === 1 ? 1 : undefined),
    responses: [
      { done_through: 0, actions: [waitAction()] },
      { done_through: 0, actions: [click("#result-1")] }
    ],
    verify: () => ({ success: true, reason: "result opened" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  const afterWait = env.of("observe").find((event) => event.observation.purpose === "after-wait");
  assert.ok(afterWait, "expected an observation right after the wait");
  assert.equal(afterWait.observation.url, results.url);
  // The post-wait capture feeds the next request; it is not captured twice.
  assert.equal(env.of("action")[1].args.observation, afterWait.observation);

  assert.equal(env.of("plan").some((event) => event.args.replan), false);
  assert.equal(state.noProgressCount, 0);
  assert.equal(state.status, "complete");
});

test("wait: an unchanged page after one wait records no-progress without replanning", async () => {
  const env = makeEnv({
    plan: ["Wait for the form", "Submit"],
    maxRounds: 2,
    responses: [
      { done_through: 0, actions: [waitAction()] },
      { done_through: 0, actions: [click("#submit")] }
    ],
    verify: () => ({ success: false, reason: "not submitted" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  const afterFirstWait = env.snapshots.find((snapshot) => snapshot.noProgressCount === 1);
  assert.ok(afterFirstWait, "expected one no-progress wait to be recorded");
  assert.match(afterFirstWait.lastActionResult, /no page progress \(1\/2\)/);
  assert.equal(afterFirstWait.currentStepIndex, 0); // a wait never completes a step

  assert.equal(env.of("plan").some((event) => event.args.replan), false);
  // The next real action resets the consecutive-wait counter.
  assert.equal(state.noProgressCount, 0);
});

test("wait: a second unchanged wait triggers a replan that keeps completed and pending steps", async () => {
  const home = page("https://shop.test/", ["#q"]);
  const results = page("https://shop.test/search", ["#filters"]);

  const env = makeEnv({
    plan: (args) => (args.replan
      ? ["Wait for results", "Scroll to reveal results", "Open result"]
      : ["Search", "Wait for results", "Open result"]),
    pages: [home, results],
    maxRounds: 3,
    onExecute: (_args, count) => (count === 1 ? 1 : undefined),
    responses: [
      { actions: [{ action: "fill", selector: "#q", value: "laptop", reasoning: "" }] },
      { done_through: 1, actions: [waitAction()] },
      { done_through: 1, actions: [waitAction()] },
      { done_through: 1, actions: [waitAction()] }
    ],
    verify: () => ({ success: false, reason: "unfinished" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  const replanAt = env.events.findIndex((event) => event.type === "plan" && event.args.replan);
  assert.ok(replanAt > 0, "expected a re-plan");

  const { context } = env.events[replanAt].args;
  assert.equal(context.last_action_result, "Previous wait actions produced no page progress.");
  assert.deepEqual(context.completed_steps, ["Search"]);
  assert.equal(context.remaining_steps[0], "Wait for results");

  // Exactly two waits ran, and nothing executed between the second
  // no-progress observation and the re-plan.
  const isWaitExecute = (event) => event.type === "execute" && event.args.response.actions[0].action === "wait";
  assert.equal(env.events.slice(0, replanAt).filter(isWaitExecute).length, 2);
  assert.equal(env.events.filter(isWaitExecute).length, 2);

  const lastAfterWaitAt = env.events.reduce(
    (found, event, index) => (index < replanAt && event.type === "observe" && event.observation.purpose === "after-wait" ? index : found),
    -1
  );
  assert.equal(env.events.slice(lastAfterWaitAt + 1, replanAt).some((event) => event.type === "execute"), false);

  assert.deepEqual(state.plan.map((step) => [step.description, step.status]), [
    ["Search", "done"],
    ["Wait for results", "pending"],
    ["Scroll to reveal results", "pending"],
    ["Open result", "pending"]
  ]);
  assert.equal(state.currentStepIndex, 1);
  assert.equal(state.noProgressCount, 0);
});

// --- current DOM is authoritative for action selection -------------------------

const element = (selector, label, tag = "button") => ({ tag, selector, label });

test("selection: cart button present -> the step clicks it instead of waiting", async () => {
  const products = { url: "https://shop.test/products", elements: [
    element("#ecom-cart-button", "Cart"),
    element("#search", "Search", "input")
  ] };

  const env = makeEnv({
    // These exercise the VLM path; the local predictor would resolve the step itself.
    allowLocalPrediction: false,
    plan: ["Open the cart"],
    pages: [products],
    responses: [
      { done_through: 0, actions: [waitAction()] },
      (args) => (args.feedback?.includes("#ecom-cart-button")
        ? { done_through: 0, actions: [click("#ecom-cart-button")] }
        : { done_through: 0, actions: [waitAction()] })
    ],
    verify: () => ({ success: true, reason: "cart open" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  const executed = env.of("execute").flatMap((event) => event.args.response.actions);
  assert.deepEqual(executed.map((action) => `${action.action} ${action.selector}`), ["click #ecom-cart-button"]);
  assert.match(env.of("action")[1].args.feedback, /Do not wait: the current page has #ecom-cart-button \("Cart"\)/);
  assert.equal(state.status, "complete");
});

test("selection: stale #quantity-1 absent but #quantity-4 present -> #quantity-4 is used", async () => {
  const cart = { url: "https://shop.test/cart", elements: [
    element("#quantity-4", "Quantity", "input"),
    element("#remove-4", "Remove")
  ] };

  const env = makeEnv({
    plan: ["Set quantity to 2"],
    pages: [cart],
    responses: [{ done_through: 0, actions: [{ action: "fill", selector: "#quantity-1", target: "Quantity", value: "2", reasoning: "" }] }],
    verify: () => ({ success: true, reason: "quantity 2" })
  });

  await Loop.runAgentLoop(env.deps);

  const executed = env.of("execute").flatMap((event) => event.args.response.actions);
  assert.deepEqual(executed.map((action) => action.selector), ["#quantity-4"]);
  assert.equal(executed.some((action) => action.selector === "#quantity-1"), false);
  assert.equal(env.of("action").length, 1); // remapped locally, no extra model call
});

test("selection: an ambiguous stale selector is never executed; the model is re-asked with current elements", async () => {
  const cart = { url: "https://shop.test/cart", elements: [
    element("#quantity-3", "Qty", "input"),
    element("#quantity-4", "Qty", "input")
  ] };

  const env = makeEnv({
    plan: ["Set the second item's quantity to 2"],
    pages: [cart],
    responses: [
      { done_through: 0, actions: [{ action: "fill", selector: "#quantity-1", value: "2", reasoning: "" }] },
      { done_through: 0, actions: [{ action: "fill", selector: "#quantity-4", value: "2", reasoning: "" }] }
    ],
    verify: () => ({ success: true, reason: "quantity 2" })
  });

  await Loop.runAgentLoop(env.deps);

  const executed = env.of("execute").flatMap((event) => event.args.response.actions);
  assert.deepEqual(executed.map((action) => action.selector), ["#quantity-4"]);
  assert.match(env.of("action")[1].args.feedback, /Selector #quantity-1 is not on the current page/);
});

test("selection: target genuinely absent -> wait still runs and replan still follows", async () => {
  const unrelated = { url: "https://shop.test/", elements: [element("#home", "Home"), element("#search", "Search", "input")] };

  const env = makeEnv({
    plan: (args) => (args.replan ? ["Use search to find the cart"] : ["Open the cart", "Set quantity to 2"]),
    pages: [unrelated],
    maxRounds: 2,
    responses: [
      { done_through: 0, actions: [waitAction()] },
      { done_through: 0, actions: [waitAction()] }
    ],
    verify: () => ({ success: false, reason: "no cart" })
  });

  await Loop.runAgentLoop(env.deps);

  const waits = env.of("execute").filter((event) => event.args.response.actions[0].action === "wait");
  assert.equal(waits.length, 2);
  assert.equal(env.of("action").some((event) => event.args.feedback), false); // no nudge without a matching target
  assert.ok(env.of("plan").some((event) => event.args.replan), "expected the existing wait -> replan path");
});

test("selection: a stale selector with no current match is not executed and replan remains possible", async () => {
  const unrelated = { url: "https://shop.test/", elements: [element("#home", "Home")] };

  const env = makeEnv({
    plan: (args) => (args.replan ? ["Go to the shop home"] : ["Open the cart"]),
    pages: [unrelated],
    maxRounds: 1,
    responses: [
      { done_through: 0, actions: [click("#ecom-cart-button")] },
      (args) => (args.feedback?.includes("not on the current page")
        ? { done_through: 0, reasoning: "no cart here", replan: true, actions: [waitAction()] }
        : { done_through: 0, actions: [click("#ecom-cart-button")] })
    ]
  });

  await Loop.runAgentLoop(env.deps);

  assert.equal(env.of("execute").length, 0);
  assert.ok(env.of("plan").some((event) => event.args.replan));
});

test("selection helpers: present, aria-label and body selectors are kept as-is", () => {
  const observation = { elements: [element("#login", "Log in"), element("div:nth-of-type(2) button", "Continue Prince Kumar")] };
  const actions = [
    click("#login"),
    click("div[aria-label='Continue Prince Kumar']"),
    { action: "scroll", selector: "body", reasoning: "" },
    waitAction()
  ];

  const result = Loop.resolveActionsAgainstObservation(actions, observation);
  assert.deepEqual(result.stale, []);
  assert.deepEqual(result.actions, actions);

  // A step that is itself about waiting never triggers the prefer-action nudge.
  assert.deepEqual(Loop.findStepCandidates("Wait for the cart to load", [element("#cart", "Cart")]), []);
});

// --- semantic target matching for the wait nudge ---------------------------------

const shopControls = [
  element("#ecom-cart-button", "Cart"),
  element('button[data-testid="add-to-cart-1"]', "Add to cart"),
  element('button[data-testid="remove-from-cart-1"]', "Remove"),
  element("#continue-shopping", "Continue shopping"),
  element("#ecom-proceed-to-buy", "Proceed to buy"),
  element("#home", "Home")
];
const pick = (step, hints) => Loop.findStepCandidates(step, shopControls, hints).map((candidate) => candidate.selector);

test("semantic: 'Open the basket' finds the cart button, not add/remove/continue controls", () => {
  assert.deepEqual(pick("Open the basket"), ["#ecom-cart-button"]);
});

test("semantic: 'Go to checkout' finds the proceed-to-buy control", () => {
  assert.deepEqual(pick("Go to checkout"), ["#ecom-proceed-to-buy"]);
});

test("semantic: 'View shopping cart' finds the cart button, not 'Continue shopping'", () => {
  assert.deepEqual(pick("View shopping cart"), ["#ecom-cart-button"]);
});

test("semantic: the model's own target meaning is used when the step text is vague", () => {
  assert.deepEqual(pick("Finish the task"), []);
  assert.deepEqual(pick("Finish the task", ["shopping basket button"]), ["#ecom-cart-button"]);
});

test("semantic: no plausible actionable target -> no candidates", () => {
  const unrelated = [element("#home", "Home"), element("#search", "Search", "input"), element("#logo", "Brand")];
  assert.deepEqual(Loop.findStepCandidates("Open the basket", unrelated), []);
  assert.deepEqual(Loop.findStepCandidates("Go to checkout", unrelated), []);
});

test("semantic loop: bare wait on 'Open the basket' is re-asked once with the cart button, then clicked", async () => {
  const env = makeEnv({
    plan: ["Open the basket"],
    pages: [{ url: "https://shop.test/products", elements: shopControls }],
    responses: [
      { done_through: 0, actions: [waitAction()] },
      (args) => (args.feedback?.includes("#ecom-cart-button")
        ? { done_through: 0, actions: [click("#ecom-cart-button")] }
        : { done_through: 0, actions: [waitAction()] })
    ],
    verify: () => ({ success: true, reason: "basket open" })
  });

  await Loop.runAgentLoop(env.deps);

  const executed = env.of("execute").flatMap((event) => event.args.response.actions);
  assert.deepEqual(executed.map((action) => `${action.action} ${action.selector}`), ["click #ecom-cart-button"]);
  assert.match(env.of("action")[1].args.feedback, /#ecom-cart-button \("Cart"\)/);
  assert.equal(env.of("action")[1].args.feedback.includes("add-to-cart"), false);
});

test("semantic loop: with no actionable target the wait is accepted without a re-ask", async () => {
  const env = makeEnv({
    plan: ["Open the basket"],
    pages: [{ url: "https://shop.test/", elements: [element("#home", "Home"), element("#search", "Search", "input")] }],
    maxRounds: 1,
    responses: [{ done_through: 0, actions: [waitAction()] }]
  });

  await Loop.runAgentLoop(env.deps);

  assert.equal(env.of("action").length, 1);
  assert.equal(env.of("execute")[0].args.response.actions[0].action, "wait");
});

// --- temporary debug logging -----------------------------------------------------

test("debug hook: reports step, parsed action, remap, wait re-ask and accepted action; execution unchanged", async () => {
  const debugEvents = [];
  const env = makeEnv({
    plan: ["Open the basket", "Set quantity to 2"],
    pages: [
      { url: "https://shop.test/products", elements: shopControls },
      { url: "https://shop.test/products", elements: [...shopControls, element("#cart-drawer", "Your cart", "div"), element("#quantity-4", "Quantity", "input")] }
    ],
    onExecute: (_args, count) => (count === 1 ? 1 : undefined),
    responses: [
      { done_through: 0, actions: [waitAction()] },
      { done_through: 0, actions: [click("#ecom-cart-button")] },
      { done_through: 1, actions: [{ action: "fill", selector: "#quantity-1", target: "Quantity", value: "2", reasoning: "" }] }
    ],
    verify: () => ({ success: true, reason: "done" })
  });
  env.deps.onDebug = (event, details) => debugEvents.push({ event, details });

  const state = await Loop.runAgentLoop(env.deps);

  // Behaviour is identical to the undebugged run.
  assert.equal(state.status, "complete");
  assert.deepEqual(
    env.of("execute").flatMap((event) => event.args.response.actions).map((action) => action.selector),
    ["#ecom-cart-button", "#quantity-4"]
  );

  const reports = debugEvents.filter((event) => event.event === "action-response").map((event) => event.details);
  assert.deepEqual(reports.map((report) => report.decision), ["wait-correction", "accepted", "accepted"]);

  const [waitReport, cartReport, quantityReport] = reports;
  assert.equal(waitReport.requestedStep, "Open the basket");
  assert.equal(waitReport.selectedAction.action, "wait");
  assert.deepEqual(waitReport.candidates, ["#ecom-cart-button"]);

  assert.match(cartReport.feedbackSent, /Do not wait/);
  assert.equal(cartReport.selectedAction.selector, "#ecom-cart-button");
  assert.equal(cartReport.selectedAction.action, "click");

  assert.equal(quantityReport.requestedStepNumber, 1);
  assert.equal(quantityReport.stepAfterVerification, 2);
  assert.equal(quantityReport.selectedAction.selector, "#quantity-1");
  assert.equal(quantityReport.selectedAction.target, "Quantity");
  assert.deepEqual(quantityReport.remapped, [{ from: "#quantity-1", to: "#quantity-4" }]);
  assert.equal(quantityReport.resolvedActions[0].selector, "#quantity-4");
  // Non-token values are not written to logs.
  assert.equal(quantityReport.selectedAction.value, "[value omitted]");
  assert.equal(quantityReport.parsedResponse.actions[0].value, "[value omitted]");
});

// --- step progress requires an observed state change ---------------------------------

function executedState(stepCount, action, before) {
  const state = Loop.createAgentState("goal", Array.from({ length: stepCount }, (_item, index) => `Step ${index + 1}`));
  Loop.recordExecution(state, { stepIndex: 0, actions: [action], executed: [action], observation: before });
  return state;
}

test("progress 1: click with no state change does not advance, even when the model claims it", () => {
  const before = { url: "https://app.test/", elements: [element("#open-panel", "Open panel")] };
  const state = executedState(2, click("#open-panel"), before);

  assert.equal(Loop.applyStepVerification(state, { done_through: 1 }, { url: before.url, elements: before.elements }), false);
  assert.equal(state.currentStepIndex, 0);
  assert.equal(state.plan[0].status, "pending");
});

test("progress 2: click with the expected state change (a panel appears) advances", () => {
  const before = { url: "https://app.test/", elements: [element("#open-panel", "Open panel")] };
  const state = executedState(2, click("#open-panel"), before);
  const after = { url: before.url, elements: [...before.elements, element("#settings-dialog", "Settings", "div")] };

  assert.equal(Loop.applyStepVerification(state, { done_through: 1 }, after), true);
  assert.equal(state.currentStepIndex, 1);
});

test("progress 3: done_through beyond verified progress is clamped", () => {
  const before = { url: "https://app.test/", elements: [element("#toggle", "Show details")] };
  const state = executedState(3, click("#toggle"), before);
  const after = { url: before.url, elements: [element("#toggle", "Hide details")] };

  Loop.applyStepVerification(state, { done_through: 3 }, after);
  assert.equal(state.currentStepIndex, 1); // one action verified -> one step, not three
});

test("progress 4: fill advances only when the target's observed value changed", () => {
  const fill = { action: "fill", selector: "#amount", value: "2", reasoning: "" };
  const before = { url: "https://app.test/form", elements: [{ tag: "input", selector: "#amount", label: "Amount", value: "1" }] };

  const changed = executedState(2, fill, before);
  Loop.applyStepVerification(changed, { done_through: 1 }, { url: before.url, elements: [{ ...before.elements[0], value: "2" }] });
  assert.equal(changed.currentStepIndex, 1);

  const unchanged = executedState(2, fill, before);
  Loop.applyStepVerification(unchanged, { done_through: 1 }, { url: before.url, elements: before.elements });
  assert.equal(unchanged.currentStepIndex, 0);
});

test("progress 5: a URL change after the action advances", () => {
  const before = { url: "https://app.test/list", elements: [element("#item-link", "Item")] };
  const state = executedState(2, click("#item-link"), before);

  Loop.applyStepVerification(state, { done_through: 1 }, { url: "https://app.test/item/7", elements: before.elements });
  assert.equal(state.currentStepIndex, 1);
});

// --- semantic relevance of the chosen control ---------------------------------------

test("relevance: account link for a product-link step is rejected", () => {
  assert.equal(Loop.assessActionRelevance("Click first laptop product link", "Hello, Mayank Account & Lists").relevant, false);
});

test("relevance: arbitrary product link names are accepted", () => {
  assert.equal(Loop.assessActionRelevance("Click first laptop product link", "ASUS Vivobook Laptop 15.6 inch").relevant, true);
  assert.equal(Loop.assessActionRelevance("Click first laptop product link", "Zenith Nova X2").relevant, true);
});

test("relevance: 'Go' for a submit-search step is accepted", () => {
  assert.equal(Loop.assessActionRelevance("Click Go button to submit search", "Go").relevant, true);
});

test("relevance: 'Add to Cart' for 'Open cart' is rejected", () => {
  assert.equal(Loop.assessActionRelevance("Open cart", "Add to Cart").relevant, false);
  assert.equal(Loop.assessActionRelevance("Open cart", "Cart").relevant, true);
});

test("relevance loop: an unrelated control is never executed; the model is re-asked with feedback", async () => {
  const results = { url: "https://shop.test/s?k=laptop", elements: [
    element("#nav-link-accountList", "Hello, Mayank Account & Lists", "a"),
    element("#result-0", "ASUS Vivobook Laptop 15.6 inch", "a")
  ] };

  const env = makeEnv({
    plan: ["Click first laptop product link"],
    pages: [results],
    responses: [
      { done_through: 0, actions: [{ action: "click", selector: "#nav-link-accountList", target: "Hello, Mayank Account & Lists", reasoning: "" }] },
      { done_through: 0, actions: [{ action: "click", selector: "#result-0", target: "ASUS Vivobook Laptop 15.6 inch", reasoning: "" }] }
    ],
    verify: () => ({ success: true, reason: "product open" })
  });

  await Loop.runAgentLoop(env.deps);

  const executed = env.of("execute").flatMap((event) => event.args.response.actions).map((action) => action.selector);
  assert.deepEqual(executed, ["#result-0"]);
  assert.match(env.of("action")[1].args.feedback, /Rejected "Hello, Mayank Account & Lists"/);
});

test("relevance loop: a repeated unrelated choice is not executed and falls to retry/replan", async () => {
  const cartPage = { url: "https://shop.test/p/1", elements: [element("#add-to-cart", "Add to Cart"), element("#cart", "Cart")] };
  const addToCart = { done_through: 0, actions: [{ action: "click", selector: "#add-to-cart", target: "Add to Cart", reasoning: "" }] };

  const env = makeEnv({
    // These exercise the VLM path; the local predictor would resolve the step itself.
    allowLocalPrediction: false,
    plan: (args) => (args.replan ? ["Open cart via the Cart link"] : ["Open cart"]),
    pages: [cartPage],
    maxRounds: 3,
    responses: Array.from({ length: 10 }, () => addToCart)
  });

  await Loop.runAgentLoop(env.deps);

  assert.equal(env.of("execute").length, 0);
  assert.ok(env.of("plan").some((event) => event.args.replan), "expected the existing replan path");
});

// --- local action predictor -----------------------------------------------------

const loginPage = { url: "https://app.test/login", elements: [
  element("#login", "Login"), element("#help", "Help", "a"), element("#email", "Email", "input")
] };

test("local: 'Click Login' with one Login button is predicted locally and runs through the normal pipeline", async () => {
  const env = makeEnv({
    plan: ["Click Login"],
    pages: [loginPage],
    navigateOnExecute: true,
    responses: [],
    verify: () => ({ success: true, reason: "logged in" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(env.of("action").length, 0, "no VLM action request expected");
  const executed = env.of("execute").flatMap((event) => event.args.response.actions);
  assert.deepEqual(executed.map((action) => `${action.action} ${action.selector}`), ["click #login"]);
  assert.equal(state.status, "complete");
});

test("local: choice steps with several product links go to the VLM", () => {
  const results = { url: "https://shop.test/s", elements: [
    element("#r0", "ASUS Vivobook Laptop", "a"), element("#r1", "HP Pavilion Laptop", "a"), element("#cart", "Cart")
  ] };
  assert.equal(Loop.predictLocalAction("Open the first suitable laptop", results), null);
  assert.equal(Loop.predictLocalAction("Click the laptop link", results), null); // ambiguous
});

test("local: value, wait and ambiguous-label steps are never predicted", () => {
  const twoLogins = { elements: [element("#a", "Login"), element("#b", "Login")] };
  assert.equal(Loop.predictLocalAction("Click Login", twoLogins), null);
  assert.equal(Loop.predictLocalAction("Enter the email", loginPage), null);
  assert.equal(Loop.predictLocalAction("Wait for the login page", loginPage), null);
  assert.equal(Loop.predictLocalAction("Click Login", { elements: [element("#email", "Login", "input")] }), null); // not clickable
});

test("local: a stale-page prediction still obeys the pipeline (relevance gate)", () => {
  const productPage = { elements: [element("#add", "Add to Cart"), element("#cart", "Cart")] };
  const prediction = Loop.predictLocalAction("Open cart", productPage);
  assert.equal(prediction?.selector, "#cart");
  assert.equal(Loop.predictLocalAction("Open cart", { elements: [element("#add", "Add to Cart")] }), null);
});

test("local: after a failed attempt the step goes to the VLM, not the predictor", async () => {
  const env = makeEnv({
    plan: ["Click Login"],
    pages: [loginPage],
    maxRounds: 2,
    onExecute: (_args, count) => { if (count === 1) throw new Error("No element matched the selector: #login"); },
    responses: [{ done_through: 0, actions: [click("#login")] }],
    verify: () => ({ success: false, reason: "not yet" })
  });

  await Loop.runAgentLoop(env.deps);

  assert.equal(env.of("action").length, 1, "retry must use the VLM");
  assert.equal(env.of("action")[0].args.context.remaining_steps[0], "Click Login");
});

// --- BUG 1: planning is a required, distinct stage -------------------------

test("BUG 1: planning runs before the first action request", async () => {
  const env = makeEnv({
    plan: ["Fill the username field", "Fill the password field", "Click submit"],
    responses: [{ done_through: 3, actions: [click("#submit")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  await Loop.runAgentLoop(env.deps);

  const order = env.of("plan").length && env.of("action").length;
  assert.ok(order, "both a plan and an action request were made");
  // The plan event is recorded before any action event.
  const events = env.of("plan").concat(env.of("action"));
  assert.equal(events[0].type, "plan");
});

test("BUG 1: a multi-action goal echoed back as one step is not a plan", async () => {
  const goal = "fill the form and click submit";
  const env = makeEnv({
    goal,
    // The planner returns the task verbatim, which is what collapsed the plan.
    plan: [goal],
    responses: [{ actions: [click("#submit")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "failed");
  assert.deepEqual(state.plan, []);
  assert.equal(env.of("action").length, 0, "no action was generated from the raw goal");
});

test("BUG 1: a multi-action goal with real substeps produces a persisted plan", async () => {
  const env = makeEnv({
    goal: "fill the form and click submit",
    plan: ["Fill the username field", "Fill the password field", "Click the submit button"],
    responses: [{ done_through: 3, actions: [click("#submit")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.deepEqual(
    state.plan.map((step) => step.description),
    ["Fill the username field", "Fill the password field", "Click the submit button"]
  );
  assert.ok(state.plan.length > 1, "a multi-action goal was broken down");
});

test("BUG 1: a genuinely single-action goal may still be one plan step", async () => {
  const env = makeEnv({
    goal: "click Login",
    plan: ["Click Login"],
    responses: [{ done_through: 1, actions: [click("#a")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.deepEqual(state.plan.map((step) => step.description), ["Click Login"]);
  assert.equal(state.status, "complete");
});

test("BUG 1: an empty planner result never becomes raw-goal execution", async () => {
  const env = makeEnv({
    plan: [],
    responses: [{ actions: [click("#go")] }],
    verify: () => ({ success: true, reason: "done" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.equal(state.status, "failed");
  assert.deepEqual(state.plan, []);
  assert.equal(env.of("action").length, 0);
});

test("BUG 1: plan validation accepts substeps and rejects goal echoes", () => {
  const goal = "fill the form and click submit";

  assert.deepEqual(Loop.validatePlanSteps(["Fill username", "Click submit"], goal), ["Fill username", "Click submit"]);
  // Echo of a multi-action goal, in several shapes.
  assert.deepEqual(Loop.validatePlanSteps([goal], goal), []);
  assert.deepEqual(Loop.validatePlanSteps(["Fill the form and click submit."], goal), []);
  assert.deepEqual(Loop.validatePlanSteps(["  "], goal), []);
  assert.deepEqual(Loop.validatePlanSteps(null, goal), []);
  // An atomic goal echoed back is legitimate.
  assert.deepEqual(Loop.validatePlanSteps(["click Login"], "click Login"), ["click Login"]);
  assert.equal(Loop.goalLooksMultiAction("fill the form and click submit"), true);
  assert.equal(Loop.goalLooksMultiAction("click Login"), false);
});

test("BUG 1: replan behaviour still works after plan validation", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    goal: "search and open the product",
    plan: (args) => (args.replan ? ["Use the search box instead", "Open product"] : ["Search", "Click missing link", "Open product"]),
    responses: [
      { actions: [{ action: "fill", selector: "#q", value: "x", reasoning: "" }] },
      { done_through: 1, reasoning: "link not here", replan: true, actions: [{ action: "wait", selector: "body", reasoning: "" }] },
      { done_through: 1, actions: [click("#search-go")] }
    ],
    verify: () => ({ success: false, reason: "unfinished" })
  });

  const state = await Loop.runAgentLoop(env.deps);

  assert.ok(env.of("plan").some((event) => event.args.replan), "a replan was requested");
  assert.ok(state.plan.some((step) => step.description === "Use the search box instead"));
});

// --- UI PROGRESS -----------------------------------------------------------

test("UI progress: steps run in one round show as executed before verification confirms them", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    plan: ["Enter Username", "Enter Password", "Click Submit"],
    responses: [
      {
        actions: [
          { action: "fill", selector: "#username", value: "u", target: "#username", reasoning: "" },
          { action: "fill", selector: "#password", value: "p", target: "#password", reasoning: "" },
          click("#submit")
        ]
      }
    ],
    verify: () => ({ success: true, reason: "logged in" })
  });

  await Loop.runAgentLoop(env.deps);

  // The snapshot published right after the round that ran all three actions:
  // the loop has verified nothing yet, but the UI can already show 2 and 3 as
  // underway instead of leaving the list sitting on step 1.
  const afterExecution = env.snapshots.find((snapshot) => snapshot.executedThroughIndex === 3);

  assert.ok(afterExecution, "a snapshot reports all three steps as executed");
  assert.equal(afterExecution.currentStepIndex, 0, "verified progress has not moved yet");
  assert.deepEqual(afterExecution.plan.map((step) => step.status), ["pending", "pending", "pending"]);
});

test("UI progress: a failed execution marks nothing as executed", async () => {
  const env = makeEnv({
    plan: ["Enter Username", "Click Submit"],
    responses: [{ actions: [click("#submit")] }],
    onExecute: () => {
      throw new Error("element detached");
    },
    verify: () => ({ success: false, reason: "not done" })
  });

  await Loop.runAgentLoop(env.deps);

  assert.equal(
    env.snapshots.every((snapshot) => snapshot.executedThroughIndex === 0),
    true,
    "nothing is shown as underway when the actions never ran"
  );
});

test("UI progress: executed steps never run ahead of the plan or behind verified progress", async () => {
  const env = makeEnv({
    navigateOnExecute: true,
    plan: ["Search", "Open product"],
    responses: [
      { actions: [click("#result-1"), click("#open"), click("#extra")] },
      { done_through: 2, actions: [] }
    ],
    verify: () => ({ success: true, reason: "done" })
  });

  await Loop.runAgentLoop(env.deps);

  for (const snapshot of env.snapshots) {
    assert.ok(snapshot.executedThroughIndex <= snapshot.plan.length, "never past the last step");
    assert.ok(snapshot.executedThroughIndex >= snapshot.currentStepIndex, "never behind verified progress");
  }
});
