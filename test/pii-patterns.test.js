// Focused tests for the local text-PII detection layer (feature #5).
// Run with: node --test test/
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// pii-patterns.js is a classic script (it must stay loadable as a content
// script), so it is evaluated rather than imported.
const source = readFileSync(new URL("../pii-patterns.js", import.meta.url), "utf8");
new Function(source)();

const {
  detectPiiMatches,
  containsPii,
  redactPiiText,
  findPiiInPayload,
  isNonRenderedTag,
  REDACTION_PLACEHOLDER
} = globalThis.__privacyVisionPii;

const typesIn = (text) => detectPiiMatches(text).map((match) => match.type);

// --- detection ---------------------------------------------------------------

test("detects email addresses", () => {
  assert.deepEqual(typesIn("Email: person@example.com"), ["email"]);
  assert.ok(containsPii("reach me at first.last+tag@sub.domain.co.in"));
});

test("detects phone numbers in the shapes we support", () => {
  assert.deepEqual(typesIn("Call +91 98765 43210 today"), ["phone"]);
  assert.deepEqual(typesIn("Mobile 9876543210"), ["phone"]); // Indian 10-digit
  assert.deepEqual(typesIn("Tel 555-867-5309"), ["phone"]); // grouped
});

test("detects card-like numbers only when they pass Luhn", () => {
  assert.deepEqual(typesIn("Card 4111 1111 1111 1111"), ["payment-card"]);
  assert.deepEqual(typesIn("Card 4539578763621486"), ["payment-card"]);
  // Same shape, bad check digit: not a card, so not redacted.
  assert.equal(containsPii("Reference 4111111111111112"), false);
});

test("detects Aadhaar-like numbers", () => {
  // Grouped form is accepted on format alone.
  assert.deepEqual(typesIn("Aadhaar 2345 6789 0123"), ["aadhaar"]);
  assert.deepEqual(typesIn("Aadhaar 2345-6789-0123"), ["aadhaar"]);
  // Bare 12-digit requires the Verhoeff check digit.
  const { passesVerhoeff } = globalThis.__privacyVisionPii;
  const valid = ["234567890123", "999999999999", "234567890120", "234567890121"].find(
    (candidate) => passesVerhoeff(candidate)
  );
  assert.ok(valid, "expected at least one Verhoeff-valid sample");
  assert.deepEqual(typesIn(`UID ${valid}`), ["aadhaar"]);
});

test("detects PAN identifiers", () => {
  assert.deepEqual(typesIn("PAN ABCDE1234F"), ["pan"]);
  assert.equal(containsPii("CODE ABCD1234F"), false); // wrong shape
});

test("detects IP addresses", () => {
  assert.deepEqual(typesIn("Server at 192.168.1.100"), ["ipv4"]);
  assert.deepEqual(typesIn("host 10.0.0.1"), ["ipv4"]);
  assert.equal(containsPii("not an ip 999.1.1.1"), false);
  assert.equal(containsPii("version 1.2.3.4"), false); // version-string guard
});

test("detects sensitive URL query parameters", () => {
  assert.deepEqual(
    typesIn("https://x.test/cb?access_token=abcd1234efgh5678"),
    ["sensitive-url-param"]
  );
});

test("detects SSNs and API keys", () => {
  assert.deepEqual(typesIn("SSN 123-45-6789"), ["ssn"]);
  assert.deepEqual(typesIn("key sk-abcdefghijklmnopqrstuvwx"), ["api-key"]);
});

// --- false-positive safety ----------------------------------------------------

test("leaves ordinary numbers, prices, dates and IDs visible", () => {
  const benign = [
    "Total: $1,234.56",
    "Price ₹1299.00",
    "Ordered on 2024-01-15",
    "Expires 12/25/2027",
    "SKU-12345",
    "1234 views",
    "Order #1234567890",
    "Version 2.10.4",
    "Quantity: 42",
    "Invoice 000123456",
    "Published 2019"
  ];

  for (const text of benign) {
    assert.equal(containsPii(text), false, `false positive on: ${text}`);
    assert.equal(redactPiiText(text), text);
  }
});

// --- redaction ----------------------------------------------------------------

test("redaction replaces PII with a neutral placeholder and keeps the rest", () => {
  assert.equal(
    redactPiiText("Contact person@example.com for help"),
    `Contact ${REDACTION_PLACEHOLDER} for help`
  );
  assert.equal(
    redactPiiText("Email: a@b.com Phone: 9876543210"),
    `Email: ${REDACTION_PLACEHOLDER} Phone: ${REDACTION_PLACEHOLDER}`
  );
});

test("overlapping matches are redacted exactly once", () => {
  const redacted = redactPiiText("Card 4111 1111 1111 1111 here");
  assert.equal(redacted, `Card ${REDACTION_PLACEHOLDER} here`);
  assert.equal(containsPii(redacted), false);
});

// --- non-rendered text hosts ---------------------------------------------------

test("script/style/template text hosts are not treated as visible page text", () => {
  for (const tag of ["script", "style", "noscript", "template", "head", "title", "meta"]) {
    assert.equal(isNonRenderedTag(tag), true, `${tag} should be skipped`);
    assert.equal(isNonRenderedTag(tag.toUpperCase()), true);
  }

  for (const tag of ["div", "span", "p", "button", "td", "label"]) {
    assert.equal(isNonRenderedTag(tag), false, `${tag} should be scanned`);
  }
});

// --- outbound payload guarantees -----------------------------------------------

test("PII in an outbound payload is reported by path and type, never by value", () => {
  const leaky = {
    task: "log in",
    elements: [{ selector: "#a", tag: "button", label: "Email: person@example.com" }]
  };

  const findings = findPiiInPayload(leaky);
  assert.equal(findings.length, 1);
  assert.equal(findings[0].type, "email");
  assert.equal(findings[0].path, "elements[0].label");
  // The finding must not carry the value itself.
  assert.equal(JSON.stringify(findings).includes("person@example.com"), false);
});

test('a page reading "Email: person@example.com" never reaches the outbound payload', () => {
  // Mirrors prepareModelElements: free-text fields are redacted, structure kept.
  const scanned = [
    { selector: "#contact", tag: "div", role: "text", label: "Email: person@example.com" },
    { selector: "#phone", tag: "span", label: "Call 9876543210" }
  ];

  const outbound = {
    task: redactPiiText("email person@example.com about my card 4111 1111 1111 1111"),
    elements: scanned.map((element) => ({
      ...element,
      label: redactPiiText(element.label)
    })),
    // pii_fields carry selector + reason only: never the matched text.
    pii_fields: [{ selector: "#contact", reason: "text-pii:email", rect: { x: 1, y: 2, width: 3, height: 4 } }]
  };

  const serialized = JSON.stringify(outbound);
  assert.equal(serialized.includes("person@example.com"), false);
  assert.equal(serialized.includes("9876543210"), false);
  assert.equal(serialized.includes("4111"), false);
  assert.deepEqual(findPiiInPayload(outbound), []);

  // Structural context the VLM needs is preserved.
  assert.equal(outbound.elements[0].selector, "#contact");
  assert.equal(outbound.elements[0].tag, "div");
  assert.equal(outbound.elements[0].role, "text");
});

test("the base64 screenshot field is skipped, not scanned", () => {
  // Random base64 reliably trips alphanumeric patterns; walking it would
  // block every request. It is already pixel-redacted upstream.
  const body = { image: "data:image/jpeg;base64,ABCDE1234F/9j/4AAQSkZJRg==", task: "go" };
  assert.deepEqual(findPiiInPayload(body, ["image"]), []);
  assert.ok(findPiiInPayload(body, []).length > 0, "without the skip it would flag");
});

test("redaction failure is detectable, which is what drives fail-closed", () => {
  // Simulates the text-PII pass not running: the raw value survives, and the
  // outbound guard sees it. popup.js turns a non-empty result into a throw.
  const unredacted = { task: "ok", elements: [{ label: "person@example.com" }] };
  const findings = findPiiInPayload(unredacted);

  assert.ok(findings.length > 0, "an unredacted payload must be flagged");

  const gate = (payload) => {
    const found = findPiiInPayload(payload, ["image"]);
    if (found.length > 0) {
      throw new Error(`Fail-Closed Privacy Error: ${found.map((f) => f.type).join(", ")}`);
    }
    return payload;
  };

  assert.throws(() => gate(unredacted), /Fail-Closed Privacy Error/);
  assert.doesNotThrow(() => gate({ task: "ok", elements: [{ label: REDACTION_PLACEHOLDER }] }));
});
