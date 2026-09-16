// Regression tests for credential-token selection (BUG 2).
// The model must never decide that a username field takes an email credential;
// the local safety layer decides it from the field's own DOM metadata.
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (file) => readFileSync(new URL(`../${file}`, import.meta.url), "utf8");
const popupSource = read("popup.js");
const popupHtml = read("popup.html");

// Load only the pure credential-mapping helpers, with `latestElements` supplied
// per test. Nothing here touches chrome.* or the DOM.
function loadMapper(elements, taskText = "") {
  const names = [
    "looksEmailLike",
    "looksUsernameLike",
    "looksPasswordLike",
    "classifyCredentialField",
    "taskMentionsValue",
    "findElementForAction",
    "enforceCredentialTokenForField",
    "CREDENTIAL_TOKEN_FOR_FIELD"
  ];
  const blocks = names.map((name) => {
    const pattern = name === "CREDENTIAL_TOKEN_FOR_FIELD"
      ? new RegExp(`^const ${name} = \\{[\\s\\S]*?^\\};$`, "m")
      : new RegExp(`^function ${name}\\([\\s\\S]*?^\\}$`, "m");
    const match = popupSource.match(pattern);
    assert.ok(match, `${name} found in popup.js`);
    return match[0];
  });

  const warnings = [];
  const factory = new Function(
    "latestElements",
    "latestTaskText",
    "console",
    `${blocks.join("\n\n")}\nreturn { classifyCredentialField, enforceCredentialTokenForField, CREDENTIAL_TOKEN_FOR_FIELD };`
  );
  const api = factory(elements, taskText, { warn: (message) => warnings.push(String(message)) });
  return { ...api, warnings };
}

// The demo form from the live run that produced the bug.
const DEMO_FORM = [
  { selector: "#username", tag: "input", type: "text", name: "username", label: "Username Username" },
  { selector: "#password", tag: "input", type: "password", name: "password", label: "Password Password" },
  { selector: "#submit", tag: "button", label: "Submit" }
];

const fill = (selector, value, extra = {}) => ({ action: "fill", selector, value, ...extra });

test("BUG 2: the demo form's username field never receives the email credential", () => {
  const mapper = loadMapper(DEMO_FORM);

  // Exactly the action the NVIDIA model returned in the live run.
  const proposed = fill("#username", "[TOKEN:USER_EMAIL]", { target: "Username Username" });
  const enforced = mapper.enforceCredentialTokenForField(proposed);

  assert.notEqual(enforced, "[TOKEN:USER_EMAIL]", "#username must not receive USER_EMAIL");
  assert.equal(enforced, "[TOKEN:USER_USERNAME]");
});

test("BUG 2: password and email fields keep their own compatible tokens", () => {
  const mapper = loadMapper([
    ...DEMO_FORM,
    { selector: "#email", tag: "input", type: "email", name: "email", label: "Email address" }
  ]);

  assert.equal(mapper.enforceCredentialTokenForField(fill("#password", "[TOKEN:USER_PASSWORD]")), "[TOKEN:USER_PASSWORD]");
  assert.equal(mapper.enforceCredentialTokenForField(fill("#email", "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_EMAIL]");
  // A password field is corrected just as firmly in the other direction.
  assert.equal(mapper.enforceCredentialTokenForField(fill("#password", "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_PASSWORD]");
});

test("field semantics come from metadata, not from label string equality", () => {
  const mapper = loadMapper([
    { selector: "#a", tag: "input", type: "text", name: "username", label: "Sign in" },
    { selector: "#b", tag: "input", type: "text", name: "login", label: "User name" },
    { selector: "#c", tag: "input", type: "text", name: "id", placeholder: "E-mail" },
    { selector: "#d", tag: "input", type: "email", name: "x", label: "Sign in" },
    { selector: "#e", tag: "input", type: "password", name: "x", label: "Sign in" }
  ]);

  assert.equal(mapper.enforceCredentialTokenForField(fill("#a", "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_USERNAME]", 'name="username"');
  assert.equal(mapper.enforceCredentialTokenForField(fill("#b", "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_USERNAME]", 'label "User name"');
  assert.equal(mapper.enforceCredentialTokenForField(fill("#c", "[TOKEN:USER_USERNAME]")), "[TOKEN:USER_EMAIL]", 'placeholder "E-mail"');
  assert.equal(mapper.enforceCredentialTokenForField(fill("#d", "[TOKEN:USER_USERNAME]")), "[TOKEN:USER_EMAIL]", 'type="email"');
  assert.equal(mapper.enforceCredentialTokenForField(fill("#e", "[TOKEN:USER_USERNAME]")), "[TOKEN:USER_PASSWORD]", 'type="password"');
});

test("inference is not aggressive: unclassified inputs keep the proposed token", () => {
  const mapper = loadMapper([
    { selector: "#q", tag: "input", type: "text", name: "q", label: "Search" },
    { selector: "#city", tag: "input", type: "text", name: "city", label: "City" }
  ]);

  for (const selector of ["#q", "#city"]) {
    assert.equal(mapper.enforceCredentialTokenForField(fill(selector, "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_EMAIL]");
  }

  // A field absent from the current scan is not guessed at from model prose.
  assert.equal(mapper.enforceCredentialTokenForField(fill("#gone", "[TOKEN:USER_EMAIL]")), "[TOKEN:USER_EMAIL]");
});

test("values for fields that are not credentials are left completely alone", () => {
  const mapper = loadMapper([
    ...DEMO_FORM,
    { selector: "#search", tag: "input", type: "text", name: "q", label: "Search" }
  ]);

  assert.equal(mapper.enforceCredentialTokenForField(fill("#search", "plain text")), "plain text");
  assert.equal(mapper.enforceCredentialTokenForField(fill("#nowhere", "plain text")), "plain text");
  // Clearing a credential field stays a clear.
  assert.equal(mapper.enforceCredentialTokenForField(fill("#username", "")), "");
});

test("BUG 3: a credential a plan step invented is never typed into the form", () => {
  // The NVIDIA plan said: fill Username field with placeholder "testuser".
  const mapper = loadMapper(DEMO_FORM, "fill the form and click submit");

  assert.equal(mapper.enforceCredentialTokenForField(fill("#username", "testuser")), "[TOKEN:USER_USERNAME]");
  assert.equal(mapper.enforceCredentialTokenForField(fill("#password", "testpass")), "[TOKEN:USER_PASSWORD]");
  // The rejected literal never reaches the log.
  assert.equal(mapper.warnings.some((warning) => /testuser|testpass/.test(warning)), false);
});

test("a literal the user asked for in the task is still honoured", () => {
  const mapper = loadMapper(DEMO_FORM, "log in as demo_account_42 and submit");

  assert.equal(mapper.enforceCredentialTokenForField(fill("#username", "demo_account_42")), "demo_account_42");
  // Case and surrounding words do not matter; invention still does.
  assert.equal(mapper.enforceCredentialTokenForField(fill("#username", "DEMO_ACCOUNT_42")), "DEMO_ACCOUNT_42");
  assert.equal(mapper.enforceCredentialTokenForField(fill("#username", "someone_else")), "[TOKEN:USER_USERNAME]");
});

test("model reasoning cannot steer the token away from the field's semantics", () => {
  const mapper = loadMapper(DEMO_FORM);

  const enforced = mapper.enforceCredentialTokenForField(fill("#username", "[TOKEN:USER_EMAIL]", {
    target: "Email address",
    reasoning: "this is the email login field, use the email credential"
  }));

  assert.equal(enforced, "[TOKEN:USER_USERNAME]");
});

test("the remap warning names tokens only, never a credential value", () => {
  const mapper = loadMapper(DEMO_FORM);
  mapper.enforceCredentialTokenForField(fill("#username", "[TOKEN:USER_EMAIL]"));

  assert.equal(mapper.warnings.length, 1);
  assert.match(mapper.warnings[0], /\[TOKEN:USER_EMAIL\][\s\S]*\[TOKEN:USER_USERNAME\]/);
  assert.equal(/@|password|secret/i.test(mapper.warnings[0].replace("USER_PASSWORD", "")), false);
});

test("the vault defines a username credential distinct from email and name", () => {
  const tokens = [...popupHtml.matchAll(/data-credential-token="([^"]+)"/g)].map((match) => match[1]);

  assert.ok(tokens.includes("USER_USERNAME"), "USER_USERNAME exists in the vault");
  assert.ok(tokens.includes("USER_EMAIL") && tokens.includes("USER_PASSWORD"));
  // Every token the mapper can require must be storable in the vault.
  const mapper = loadMapper([]);
  for (const token of Object.values(mapper.CREDENTIAL_TOKEN_FOR_FIELD)) {
    assert.ok(tokens.includes(token.slice(7, -1)), `${token} is storable in the vault`);
  }
});

test("credential values live only in the vault, never in popup source", () => {
  // The mapping layer deals in token names; no raw credential ever appears.
  assert.equal(/\[TOKEN:[A-Z0-9_]+\]\s*=\s*["'][^"']+["']/.test(popupSource), false);
  assert.ok(popupSource.includes("resolveCredentialToken(enforceCredentialTokenForField(action))"));
});

// --- The specific end-to-end regression from the live run ------------------

const agentSource = read("agent-loop.js");
new Function(agentSource)();
const Loop = globalThis.__privacyVisionAgentLoop;

// Values that must never leave the vault. Distinctive so a substring search is
// conclusive across every payload, history entry and log line.
const VAULT = {
  USER_EMAIL: "real-person-7f3a@example.test",
  USER_USERNAME: "realuser-9c2b",
  USER_PASSWORD: "correct-horse-battery-4d1e"
};

test("REGRESSION: the demo form plans, then fills each field with its own credential", async () => {
  const mapper = loadMapper(DEMO_FORM);
  const logs = [];
  const actionHistory = [];
  const typed = [];

  // Exactly what the NVIDIA model returned in the live run, email token included.
  const modelResponse = {
    reasoning: "Fill username and password then submit",
    done_through: 0,
    complete: true,
    actions: [
      fill("#username", "[TOKEN:USER_EMAIL]", { target: "Username Username", reasoning: "" }),
      fill("#password", "[TOKEN:USER_PASSWORD]", { target: "Password Password", reasoning: "" }),
      { action: "click", selector: "#submit", target: "Submit", reasoning: "" }
    ]
  };

  const events = [];
  const observation = {
    url: "https://demo.test/login",
    elements: DEMO_FORM,
    signature: Loop.pageSignature("https://demo.test/login", DEMO_FORM)
  };

  const state = await Loop.runAgentLoop({
    goal: "fill the form and click submit",
    maxRounds: 5,
    observe: async () => observation,
    requestPlan: async (args) => {
      events.push({ type: "plan", args });
      return ["Fill the username field", "Fill the password field", "Click the submit button"];
    },
    requestAction: async (args) => {
      events.push({ type: "action", args });
      return modelResponse;
    },
    executeActions: async ({ response, round }) => {
      for (const action of response.actions) {
        if (action.action === "fill") {
          // The popup's local safety layer, then the vault lookup.
          const token = mapper.enforceCredentialTokenForField(action);
          const tokenName = token.slice(7, -1);
          const value = VAULT[tokenName];
          assert.ok(value, `${tokenName} is resolvable from the vault`);
          typed.push({ selector: action.selector, tokenName, value });
        }
        // Action history records that a value was provided, never the value.
        actionHistory.push({
          round,
          action: action.action,
          selector: action.selector,
          target: action.target || "",
          value: action.value ? "[value provided]" : "",
          reasoning: action.reasoning || ""
        });
      }
      logs.push(...mapper.warnings);
      return response.actions;
    },
    verifyGoal: async () => ({ success: true, reason: "logged in" })
  });

  // PLANNING produced meaningful steps, not the raw goal.
  assert.equal(events[0].type, "plan");
  assert.equal(state.plan.length, 3);
  assert.equal(state.plan.some((step) => step.description === "fill the form and click submit"), false);

  // ACTION GENERATION mapped each field to its own credential.
  const bySelector = Object.fromEntries(typed.map((entry) => [entry.selector, entry]));
  assert.equal(bySelector["#username"].tokenName, "USER_USERNAME");
  assert.equal(bySelector["#password"].tokenName, "USER_PASSWORD");
  // The explicit failure condition from the bug report.
  assert.notEqual(bySelector["#username"].tokenName, "USER_EMAIL");
  assert.notEqual(bySelector["#username"].value, VAULT.USER_EMAIL);
  assert.ok(actionHistory.some((entry) => entry.action === "click" && entry.selector === "#submit"));

  // PRIVACY: no raw credential value in any model-facing or recorded structure.
  const surfaces = {
    "planner payload": events.filter((event) => event.type === "plan"),
    "action payload": events.filter((event) => event.type === "action"),
    "action history": actionHistory,
    "model response": modelResponse,
    "agent state": state,
    logs
  };

  for (const [name, surface] of Object.entries(surfaces)) {
    const serialized = JSON.stringify(surface);
    for (const [tokenName, value] of Object.entries(VAULT)) {
      assert.equal(serialized.includes(value), false, `${tokenName} value leaked into ${name}`);
    }
  }

  // Only token placeholders ever reached the model-facing action payload.
  assert.ok(JSON.stringify(modelResponse).includes("[TOKEN:USER_PASSWORD]"));
  assert.equal(typed.length, 2);
});
