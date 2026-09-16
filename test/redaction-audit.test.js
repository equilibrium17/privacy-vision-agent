// Tests for the local-only redaction audit log (feature #8).
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

for (const file of ["../pii-patterns.js", "../redaction-audit.js"]) {
  new Function(readFileSync(new URL(file, import.meta.url), "utf8"))();
}

const audit = globalThis.__privacyVisionRedactionAudit;
const { findPiiInPayload } = globalThis.__privacyVisionPii;

test("recordRedaction stores metadata only", async () => {
  await audit.clearRedactionAuditLog();
  const record = audit.recordRedaction({
    type: "text-pii",
    source: "text-scan",
    reason: "redacted visible text",
    category: "email",
    rect: { x: 10.4, y: 20.6, width: 100, height: 14 },
    pageOrigin: "https://shop.test/account?email=person@example.com&token=abcd1234efgh5678"
  });

  assert.deepEqual(Object.keys(record).sort(), ["category", "pageOrigin", "reason", "rect", "source", "timestamp", "type"]);
  assert.equal(record.pageOrigin, "https://shop.test"); // origin only, never path/query
  assert.deepEqual(record.rect, { x: 10, y: 21, width: 100, height: 14 });
  assert.equal(audit.getRedactionAuditLog().length, 1);
});

test("matched text and values never reach the stored entry", async () => {
  await audit.clearRedactionAuditLog();
  const secrets = ["person@example.com", "9876543210", "Ujjwal Jha", "hunter2", "sk-abcdefghijklmnopqrstuvwx"];

  audit.recordRedaction({
    type: "entity",
    source: "text-scan",
    reason: "Name: Ujjwal Jha",          // rejected: not a safe label shape
    category: "person",
    matchedText: "Ujjwal Jha",           // not a whitelisted field
    value: "hunter2",
    text: "person@example.com 9876543210",
    pageOrigin: "https://x.test/?token=sk-abcdefghijklmnopqrstuvwx"
  });

  const serialized = JSON.stringify(audit.getRedactionAuditLog());
  for (const secret of secrets) {
    assert.equal(serialized.includes(secret), false, `leaked: ${secret}`);
  }
  assert.deepEqual(findPiiInPayload(audit.getRedactionAuditLog()), []);
});

test("clearRedactionAuditLog empties the local log", async () => {
  audit.recordRedaction({ type: "face", source: "face-detector", reason: "face blurred" });
  assert.ok(audit.getRedactionAuditLog().length > 0);
  await audit.clearRedactionAuditLog();
  assert.deepEqual(audit.getRedactionAuditLog(), []);
});

test("the audit log is not part of any outbound payload shape", () => {
  // Mirrors the bodies popup.js builds for action, plan and verify requests.
  const bodies = [
    { image: "data:image/jpeg;base64,AA", pii_fields: [], elements: [], task: "t", action_history: [], task_plan: [], provider: "groq" },
    { image: "data:image/jpeg;base64,AA", pii_fields: [], elements: [], task: "t", plan: true, provider: "groq" },
    { task: "t", page_context: "p", completed_steps: [], verification_evidence: [], provider: "groq", verify: true }
  ];

  for (const body of bodies) {
    assert.equal(JSON.stringify(body).includes(audit.STORAGE_KEY), false);
    assert.equal("redactionAuditLog" in body || "audit" in body, false);
  }

  // Static guard: popup.js only ever reads the audit module to record entries.
  const popup = readFileSync(new URL("../popup.js", import.meta.url), "utf8");
  assert.equal(/getRedactionAuditLog|redactionAuditLog/.test(popup), false);
});
