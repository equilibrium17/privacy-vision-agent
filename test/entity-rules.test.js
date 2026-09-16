// Focused tests for contextual entity redaction (feature #6).
//
// This is rule-based contextual entity detection, not statistical NER: every
// assertion below checks either that an explicit cue fires, or that ordinary
// UI/page text without a cue is left alone.
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../pii-patterns.js", import.meta.url), "utf8");
new Function(source)();

const {
  detectPiiMatches,
  detectEntityMatches,
  containsPii,
  redactPiiText,
  findPiiInPayload,
  placeholderFor,
  ENTITY_PLACEHOLDERS,
  REDACTION_PLACEHOLDER
} = globalThis.__privacyVisionPii;

/** "type:matched text" pairs - built here in the test, never by the detector. */
const found = (text, options) =>
  detectPiiMatches(text, options).map((match) => `${match.type}:${text.slice(match.start, match.end)}`);

const DEMO = [
  "Contact Ujjwal Jha in Kolkata at 123 Park Street, Kolkata.",
  "He works at Example Technologies.",
  "Appointment: 15 September 2026 at 3:30 PM.",
  "Age: 21.",
  "Username: ujjwal_jha."
].join("\n");

const DEMO_RAW_VALUES = [
  "Ujjwal Jha",
  "Ujjwal",
  "Kolkata",
  "123 Park Street",
  "Park Street",
  "Example Technologies",
  "15 September 2026",
  "3:30 PM",
  "ujjwal_jha"
];

// --- entity types ----------------------------------------------------------

test("PERSON: labels, honorifics, greetings and relationship cues", () => {
  assert.deepEqual(found("Name: Priya Sharma"), ["person:Priya Sharma"]);
  assert.deepEqual(found("Dr. Anil Kumar will see you"), ["person:Anil Kumar"]);
  assert.deepEqual(found("Welcome, Ujjwal"), ["person:Ujjwal"]);
  assert.deepEqual(found("Posted by Ananya Roy"), ["person:Ananya Roy"]);
  assert.deepEqual(found("Father's name: Ramesh Jha"), ["person:Ramesh Jha"]);
  assert.deepEqual(found("Contact Ujjwal Jha today"), ["person:Ujjwal Jha"]);
});

test("LOCATION: needs a preposition or label, never a bare place name", () => {
  assert.deepEqual(found("Lives in Siliguri"), ["location:Siliguri"]);
  assert.deepEqual(found("City: Durgapur"), ["location:Durgapur"]);
  assert.deepEqual(found("She is from Kolkata"), ["location:Kolkata"]);
  assert.deepEqual(found("Location: Mumbai"), ["location:Mumbai"]);

  assert.deepEqual(found("Kolkata"), []);
  assert.deepEqual(found("New Delhi"), []);
  assert.deepEqual(found("Made in India"), []);
  assert.deepEqual(found("Location: Remote"), []); // weak label needs a known place
});

test("ORGANIZATION: needs a work/study cue or a corporate suffix after 'at'", () => {
  assert.deepEqual(found("He works at Example Technologies."), ["organization:Example Technologies"]);
  assert.deepEqual(found("Employer: Acme Robotics"), ["organization:Acme Robotics"]);
  assert.deepEqual(found("She studies at Jadavpur University"), ["organization:Jadavpur University"]);

  // A site's own footer is not personal context.
  assert.deepEqual(found("© 2026 Example Technologies Pvt Ltd"), []);
  assert.deepEqual(found("Samsung Galaxy S24 Ultra"), []);
});

test("ADDRESS: numbered street shapes, address labels and postal codes", () => {
  assert.deepEqual(found("Office at 123 Park Street, Kolkata."), ["address:123 Park Street, Kolkata"]);
  assert.deepEqual(
    found("Address: Flat 4B, Lake Town, Kolkata 700089"),
    ["address:Flat 4B, Lake Town, Kolkata 700089"]
  );
  assert.deepEqual(found("Pincode: 700016"), ["address:700016"]);

  // "IP address" / "Email address" keep their precise deterministic type.
  assert.deepEqual(found("IP Address: 192.168.1.10"), ["ipv4:192.168.1.10"]);
  assert.deepEqual(found("Email address: person@example.com"), ["email:person@example.com"]);
  assert.deepEqual(found("Top 10 Street Foods"), []);
});

test("DATE: only with a personal-event cue", () => {
  assert.deepEqual(found("Appointment: 15 September 2026"), ["date:15 September 2026"]);
  assert.deepEqual(found("Date of birth: 12/05/2004"), ["date:12/05/2004"]);
  assert.deepEqual(found("Check-in 10 Oct 2026"), ["date:10 Oct 2026"]);

  assert.deepEqual(found("Published on 15 September 2026"), []);
  assert.deepEqual(found("Updated 2026-09-12"), []);
  assert.deepEqual(found("Ordered on 2024-01-15"), []);
});

test("TIME: only with a personal-event cue", () => {
  assert.deepEqual(found("Booking slot: 11:45 AM"), ["time:11:45 AM"]);
  assert.deepEqual(found("Check-in 10 Oct 2026, 2 PM"), ["date:10 Oct 2026", "time:2 PM"]);

  assert.deepEqual(found("Today 3:30 PM"), []);
  assert.deepEqual(found("Time: 10:30"), []);
  assert.deepEqual(found("Reading time: 5 min"), []);
});

test("AGE: labelled ages and 'N years old', not page/stage counters", () => {
  assert.deepEqual(found("Age: 21."), ["age:21"]);
  assert.deepEqual(found("aged 34"), ["age:34"]);
  assert.deepEqual(found("He is 21 years old"), ["age:21 years old"]);

  assert.deepEqual(found("Page 21 of 40"), []);
  assert.deepEqual(found("Stage: 3"), []);
  assert.deepEqual(found("Posted 3 days ago"), []);
  assert.deepEqual(found("Age: 400"), []); // implausible
});

test("USERNAME: labelled values, session banners and @handles", () => {
  assert.deepEqual(found("Username: ujjwal_jha."), ["username:ujjwal_jha"]);
  assert.deepEqual(found("Signed in as ujjwal_jha"), ["username:ujjwal_jha"]);
  assert.deepEqual(found("Follow @ujjwal_jha"), ["username:@ujjwal_jha"]);

  // Login-form labels and selectors carry no value.
  assert.deepEqual(found("Username"), []);
  assert.deepEqual(found("Username or email"), []);
  assert.deepEqual(found("#username"), []);
  assert.deepEqual(found("person@example.com"), ["email:person@example.com"]);
});

test("JOB_TITLE: labelled designations and 'works as a ...'", () => {
  assert.deepEqual(found("Designation: Senior Software Engineer"), ["job-title:Senior Software Engineer"]);
  assert.deepEqual(
    found("He works as a Data Analyst at Example Technologies"),
    ["job-title:Data Analyst", "organization:Example Technologies"]
  );
  assert.deepEqual(found("Position: 3"), []);
});

test("MISC_PERSONAL: sensitive attributes, without swallowing the next label", () => {
  assert.deepEqual(found("Gender: Male Age: 21"), ["misc-personal:Male", "age:21"]);
  assert.deepEqual(found("Blood group: O+"), ["misc-personal:O+"]);
  assert.deepEqual(found("Category: Electronics"), []);
});

// --- false-positive safety ---------------------------------------------------

test("benign UI and page text is not redacted", () => {
  const benign = [
    "Home", "Settings", "Login", "Submit", "Log out", "Log in to continue",
    "Contact Us", "Contact Customer Care", "Welcome back!", "Welcome to Example",
    "Sign in with Google", "Hello World", "Dear Customer", "Posted by Admin",
    "Name", "Full name", "Enter your name", "Address", "Product Name: Widget",
    "File name: report.pdf", "Apple iPhone 15 Pro", "The Road Ahead", "5 Star Hotels",
    "Price ₹1,299.00", "Order #1234567890",
    // The live-tested multi-step task and typical model/plan strings.
    "fill the credentials and then click on submit then click on logout",
    "Click Log out to end the session",
    "Fill the Username field with [TOKEN:USER_EMAIL]",
    'button[aria-label="Log in"]', 'input[name="first_name"]', "#submit", "#logout"
  ];

  for (const text of benign) {
    assert.deepEqual(found(text), [], `false positive on: ${text}`);
    assert.equal(redactPiiText(text), text);
  }
});

// --- placeholders and redaction ------------------------------------------------

test("entities get type-specific placeholders; deterministic PII keeps the generic one", () => {
  assert.equal(placeholderFor("person"), "[REDACTED_PERSON]");
  assert.equal(placeholderFor("job-title"), "[REDACTED_JOB_TITLE]");
  assert.equal(placeholderFor("misc-personal"), "[REDACTED_PERSONAL]");
  assert.equal(placeholderFor("email"), REDACTION_PLACEHOLDER);

  assert.equal(
    redactPiiText(DEMO),
    [
      "Contact [REDACTED_PERSON] in [REDACTED_LOCATION] at [REDACTED_ADDRESS].",
      "He works at [REDACTED_ORGANIZATION].",
      "Appointment: [REDACTED_DATE] at [REDACTED_TIME].",
      "Age: [REDACTED_AGE].",
      "Username: [REDACTED_USERNAME]."
    ].join("\n")
  );

  assert.deepEqual(Object.keys(ENTITY_PLACEHOLDERS).sort(), [
    "address", "age", "date", "job-title", "location", "misc-personal",
    "organization", "person", "time", "username"
  ]);
});

test("redacted output never re-triggers detection (the outbound gate re-scans it)", () => {
  const samples = [
    DEMO,
    "Name: Priya Sharma, Age: 30, City: Durgapur",
    "Address: Flat 4B, Lake Town, Kolkata 700089 Phone: 9876543210",
    "Check-in 10 Oct 2026, 2 PM. Signed in as ujjwal_jha",
    "Gender: Male Age: 21 Blood group: O+ Designation: Senior Software Engineer"
  ];

  for (const text of samples) {
    const redacted = redactPiiText(text);
    assert.equal(containsPii(redacted), false, `re-triggered on: ${redacted}`);
    assert.equal(redactPiiText(redacted), redacted);
  }
});

test("sibling-node label context: a 'Name:' label in the previous node exposes the value", () => {
  assert.deepEqual(found("Priya Sharma"), []);
  assert.deepEqual(found("Priya Sharma", { prefix: "Name:" }), ["person:Priya Sharma"]);
  assert.deepEqual(found("ujjwal_jha", { prefix: "Username:" }), ["username:ujjwal_jha"]);

  // Matches that live wholly in the prefix are not reported for this node.
  assert.deepEqual(found("Home", { prefix: "Name: Priya Sharma" }), []);
});

// --- critical privacy test -------------------------------------------------------

test("detector results carry only {type, start, end} - never the raw entity", () => {
  const matches = detectPiiMatches(DEMO);

  assert.equal(matches.length, 8);
  for (const match of matches) {
    assert.deepEqual(Object.keys(match).sort(), ["end", "start", "type"]);
  }

  const serialized = JSON.stringify(matches) + JSON.stringify(detectEntityMatches(DEMO));
  for (const raw of DEMO_RAW_VALUES) {
    assert.equal(serialized.includes(raw), false, `detector result leaked: ${raw}`);
  }

  // Outbound-gate findings likewise report {path, type} only.
  const findings = findPiiInPayload({ elements: [{ label: DEMO }] });
  assert.ok(findings.length >= 8);
  for (const raw of DEMO_RAW_VALUES) {
    assert.equal(JSON.stringify(findings).includes(raw), false, `gate finding leaked: ${raw}`);
  }
});

test("demo entities never reach an action, plan or verify payload", () => {
  // These mirror popup.js's network-boundary preparation (prepareModelElements,
  // prepareModelPiiFields, prepareModelActionHistory, sanitizePromptText), which
  // cannot be imported under Node because popup.js binds the DOM at load.
  const scannedElements = [
    { selector: "#contact-card", tag: "div", role: "region", label: DEMO, rect: { x: 0, y: 0, width: 300, height: 80 } },
    { selector: "#logout", tag: "button", label: "Log out", rect: { x: 10, y: 10, width: 60, height: 20 } }
  ];
  const piiFields = [
    { selector: 'div[aria-label="Contact Ujjwal Jha"] p', reason: "text-pii:person", rect: { x: 1, y: 2, width: 3, height: 4 } },
    { selector: "#password", reason: "input type=password", rect: { x: 5, y: 6, width: 7, height: 8 } }
  ];
  const actionHistory = [
    { action: "click", selector: "#submit", target: "Submit", reasoning: "Submit the form submitted by Ujjwal Jha", value: "" }
  ];

  const elements = scannedElements.map((element) => ({ ...element, label: redactPiiText(element.label) }));
  const pii_fields = piiFields.map((field) =>
    field.reason.startsWith("text-pii:") ? { ...field, selector: redactPiiText(field.selector) } : field
  );
  const action_history = actionHistory.map((entry) => ({
    ...entry,
    selector: redactPiiText(entry.selector),
    target: redactPiiText(entry.target),
    reasoning: redactPiiText(entry.reasoning)
  }));

  const payloads = {
    action: {
      image: "data:image/jpeg;base64,AAAA",
      pii_fields,
      elements,
      task: redactPiiText("click on logout"),
      action_history,
      task_plan: ["Click Log out"].map(redactPiiText)
    },
    plan: { image: "data:image/jpeg;base64,AAAA", pii_fields, elements, task: "click on logout", plan: true },
    verify: {
      task: "click on logout",
      page_context: redactPiiText(`Page title: Profile\nURL: https://example.test/profile\nVisible text (first 800 chars): ${DEMO}`),
      verify: true
    }
  };

  for (const [label, payload] of Object.entries(payloads)) {
    const serialized = JSON.stringify(payload);

    for (const raw of DEMO_RAW_VALUES) {
      assert.equal(serialized.includes(raw), false, `${label} payload leaked: ${raw}`);
    }

    assert.deepEqual(findPiiInPayload(payload, ["image"]), [], `${label} payload failed the gate`);
  }

  // Structure the VLM needs survives redaction.
  assert.equal(payloads.action.elements[1].selector, "#logout");
  assert.equal(payloads.action.elements[1].label, "Log out");
  assert.equal(payloads.action.elements[0].role, "region");
  assert.equal(payloads.action.pii_fields[1].selector, "#password"); // form field untouched
});

test("fail-closed: an unredacted entity in a payload is caught by the outbound gate", () => {
  const gate = (payload) => {
    const findings = findPiiInPayload(payload, ["image"]);
    if (findings.length > 0) {
      throw new Error(`Fail-Closed Privacy Error: ${findings.map((f) => `${f.type} at ${f.path}`).join(", ")}`);
    }
    return payload;
  };

  assert.throws(() => gate({ elements: [{ label: "Name: Priya Sharma" }] }), /person at elements\[0\]\.label/);
  assert.throws(() => gate({ page_context: "Appointment: 15 September 2026" }), /date at page_context/);
  assert.throws(() => gate({ action_history: [{ target: "Username: ujjwal_jha" }] }), /username/);
  assert.doesNotThrow(() => gate({ elements: [{ label: redactPiiText("Name: Priya Sharma") }] }));
});

test("known limitation: a name with no contextual cue is NOT detected", () => {
  // Rule-based, not statistical NER. Without a label, honorific, greeting or
  // relationship phrase there is no signal, so these stay visible. Kept as an
  // explicit test so the limitation is documented rather than hidden.
  assert.equal(containsPii("Ujjwal Jha"), false);
  assert.equal(containsPii("Submit the form for Ujjwal Jha"), false);
  assert.equal(containsPii('div[aria-label="Card for Ujjwal Jha"] p'), false);
});

// --- regression: existing text-PII behaviour ----------------------------------------

test("regression: deterministic text-PII detection is unchanged by the entity layer", () => {
  assert.deepEqual(found("Email: person@example.com"), ["email:person@example.com"]);
  assert.deepEqual(found("Call +91 98765 43210 today"), ["phone:+91 98765 43210"]);
  assert.deepEqual(found("Mobile 9876543210"), ["phone:9876543210"]);
  assert.deepEqual(found("Card 4111 1111 1111 1111"), ["payment-card:4111 1111 1111 1111"]);
  assert.deepEqual(found("Aadhaar 2345 6789 0123"), ["aadhaar:2345 6789 0123"]);
  assert.deepEqual(found("PAN ABCDE1234F"), ["pan:ABCDE1234F"]);
  assert.deepEqual(found("Server at 192.168.1.100"), ["ipv4:192.168.1.100"]);
  assert.deepEqual(found("SSN 123-45-6789"), ["ssn:123-45-6789"]);

  assert.equal(
    redactPiiText("Contact person@example.com for help"),
    `Contact ${REDACTION_PLACEHOLDER} for help`
  );

  // Mixed: both layers in one string, each with its own placeholder.
  assert.equal(
    redactPiiText("Name: Priya Sharma, Email: priya@example.com, Mobile 9876543210"),
    `Name: [REDACTED_PERSON], Email: ${REDACTION_PLACEHOLDER}, Mobile ${REDACTION_PLACEHOLDER}`
  );
});

// --- performance -------------------------------------------------------------------

test("performance: 4000 text nodes (the scanner bound) stay well inside one capture budget", () => {
  const samples = [
    "Home", "Add to cart", "Price ₹1,299.00", "Samsung Galaxy S24 Ultra 256GB", "Contact Us",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
    "Name: Priya Sharma", "Appointment: 15 September 2026 at 3:30 PM", "Posted 3 days ago"
  ];

  let previous = "";
  const started = performance.now();
  for (let index = 0; index < 4000; index += 1) {
    const text = samples[index % samples.length];
    detectPiiMatches(text, { prefix: previous });
    previous = text;
  }
  const elapsed = performance.now() - started;

  // Measured ~30ms warm on the dev machine; the bound only catches regressions
  // such as catastrophic regex backtracking.
  assert.ok(elapsed < 1500, `4000-node pass took ${elapsed.toFixed(1)}ms`);
});

test("performance: adversarial inputs to the entity rules do not backtrack catastrophically", () => {
  const adversarial = [
    `1 ${"Aaaa ".repeat(2000)}x`,
    `Address: ${"1,".repeat(5000)}`,
    `Name: ${"Aa ".repeat(3000)}`,
    `works at ${"Aa of ".repeat(2000)}`,
    `Appointment ${"1/1/".repeat(3000)}`,
    `Username: ${"a".repeat(10000)}`
  ];

  for (const text of adversarial) {
    const started = performance.now();
    detectPiiMatches(text);
    const elapsed = performance.now() - started;
    assert.ok(elapsed < 500, `${elapsed.toFixed(1)}ms on input starting ${JSON.stringify(text.slice(0, 20))}`);
  }
});
