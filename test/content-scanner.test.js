// Runs the REAL content.js text scanner against a minimal fake DOM, so the
// TreeWalker filter, visibility filtering, sibling-label context and
// fail-closed error reporting are exercised as shipped - not re-implemented.
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const patternsSource = readFileSync(new URL("../pii-patterns.js", import.meta.url), "utf8");
const contentSource = readFileSync(new URL("../content.js", import.meta.url), "utf8");

const NodeFilter = { SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3 };
const CHAR_WIDTH = 7;

class Document {}
class ShadowRoot {}

function element(tagName, parent, { style = {}, rect = { x: 10, y: 10, width: 400, height: 20 } } = {}) {
  const node = {
    nodeType: 1,
    tagName: tagName.toUpperCase(),
    parentElement: parent ?? null,
    id: "",
    children: [],
    computedStyle: { display: "block", visibility: "visible", opacity: "1", ...style },
    getAttribute: () => null,
    hasAttribute: () => false,
    getBoundingClientRect: () => ({ ...rect, right: rect.x + rect.width, bottom: rect.y + rect.height })
  };
  parent?.children.push(node);
  return node;
}

function textNode(parent, nodeValue) {
  return { nodeType: 3, nodeValue, parentElement: parent };
}

/** Build a sandboxed page, load pii-patterns.js + content.js into it, and scan. */
function scanPage(buildTextNodes, { breakDetector = false, skipPatterns = false } = {}) {
  const body = element("body", null, { rect: { x: 0, y: 0, width: 1280, height: 800 } });
  const textNodes = buildTextNodes(body);

  const document = Object.assign(new Document(), {
    body,
    children: [],
    getElementById: () => null,
    querySelectorAll: (selector) => (selector === "iframe, frame" ? [] : [body]),
    createTreeWalker: (_root, _whatToShow, filter) => {
      let index = -1;
      return {
        currentNode: null,
        nextNode() {
          while (++index < textNodes.length) {
            if (filter.acceptNode(textNodes[index]) === NodeFilter.FILTER_ACCEPT) {
              this.currentNode = textNodes[index];
              return true;
            }
          }
          return false;
        }
      };
    },
    createRange: () => {
      let node;
      let start;
      let end;
      return {
        setStart: (target, offset) => { node = target; start = offset; },
        setEnd: (_target, offset) => { end = offset; },
        getClientRects: () => {
          const host = node.parentElement.getBoundingClientRect();
          return [{ x: host.x + start * CHAR_WIDTH, y: host.y, right: host.x + end * CHAR_WIDTH, bottom: host.bottom }];
        },
        detach: () => {}
      };
    }
  });

  const sandbox = {
    console,
    document,
    Document,
    ShadowRoot,
    NodeFilter,
    innerWidth: 1280,
    innerHeight: 800,
    devicePixelRatio: 1,
    frames: { length: 0 },
    getComputedStyle: (target) => target.computedStyle
  };
  sandbox.window = sandbox;
  sandbox.parent = sandbox;
  vm.createContext(sandbox);

  if (!skipPatterns) {
    vm.runInContext(patternsSource, sandbox);
  }

  if (breakDetector) {
    sandbox.__privacyVisionPii.detectPiiMatches = () => {
      throw new Error("simulated detector failure");
    };
  }

  vm.runInContext(contentSource, sandbox);
  // Objects created inside the vm context carry that realm's prototypes, which
  // strict deepEqual rejects. The scan result is plain data, so normalise it.
  const result = JSON.parse(JSON.stringify(sandbox.__privacyVisionAgent.scanPiiFields()));
  return { result, textNodes };
}

const textRegions = (result) => result.fields.filter((field) => field.reason.startsWith("text-pii:"));
const reasons = (result) => textRegions(result).map((field) => field.reason);

const DEMO = "Contact Ujjwal Jha in Kolkata at 123 Park Street, Kolkata. He works at Example Technologies. " +
  "Appointment: 15 September 2026 at 3:30 PM. Age: 21. Username: ujjwal_jha.";

test("visible entity text becomes redaction regions through the existing pipeline", () => {
  const { result } = scanPage((body) => [textNode(element("p", body), DEMO)]);

  assert.deepEqual(reasons(result), [
    "text-pii:person",
    "text-pii:location",
    "text-pii:address",
    "text-pii:organization",
    "text-pii:date",
    "text-pii:time",
    "text-pii:age",
    "text-pii:username"
  ]);
  assert.equal(result.textPiiCount, 8);
  assert.equal(result.textPiiError, undefined);

  // Rects come from Range.getClientRects() on the matched span (padded by 1px).
  const person = textRegions(result)[0];
  assert.equal(person.rect.x, 10 + DEMO.indexOf("Ujjwal Jha") * CHAR_WIDTH - 1);
});

test("scan output never contains raw entity values", () => {
  const { result } = scanPage((body) => [textNode(element("p", body), DEMO)]);
  const serialized = JSON.stringify(result);

  for (const raw of ["Ujjwal Jha", "Kolkata", "Park Street", "Example Technologies", "15 September 2026", "3:30 PM", "ujjwal_jha"]) {
    assert.equal(serialized.includes(raw), false, `scan result leaked: ${raw}`);
  }
});

test("hidden, script, style, template and noscript text is ignored", () => {
  const { result } = scanPage((body) => [
    textNode(element("script", body), "const name = 'Name: Priya Sharma';"),
    textNode(element("style", body), "/* Contact Ujjwal Jha */"),
    textNode(element("template", body), "Username: ujjwal_jha"),
    textNode(element("noscript", body), "Appointment: 15 September 2026"),
    textNode(element("div", body, { style: { display: "none" } }), "Name: Priya Sharma"),
    textNode(element("div", body, { style: { visibility: "hidden" } }), "Lives in Siliguri"),
    textNode(element("div", body, { style: { opacity: "0" } }), "Employer: Acme Robotics"),
    textNode(element("div", body, { rect: { x: 0, y: 0, width: 0, height: 0 } }), "Age: 21")
  ]);

  assert.deepEqual(reasons(result), []);
  assert.equal(result.textPiiCount, 0);
});

test("a label in a sibling node gives context to the value node", () => {
  const { result } = scanPage((body) => {
    const list = element("dl", body);
    return [textNode(element("dt", list), "Name:"), textNode(element("dd", list), "Priya Sharma")];
  });

  assert.deepEqual(reasons(result), ["text-pii:person"]);
});

test("unrelated distant nodes do not lend context", () => {
  const { result } = scanPage((body) => {
    const header = element("header", body);
    const main = element("main", body);
    const article = element("article", main);
    return [textNode(element("span", header), "Name:"), textNode(element("p", article), "Priya Sharma")];
  });

  assert.deepEqual(reasons(result), []);
});

test("benign page text produces no regions", () => {
  const { result } = scanPage((body) => [
    "Home", "Settings", "Login", "Submit", "Log out", "Contact Us", "Welcome back!",
    "Kolkata", "Published on 15 September 2026", "Today 3:30 PM", "Samsung Galaxy S24 Ultra"
  ].map((text) => textNode(element("span", body), text)));

  assert.deepEqual(reasons(result), []);
});

test("regression: existing deterministic text PII still produces regions", () => {
  const { result } = scanPage((body) => [
    textNode(element("p", body), "Email: person@example.com"),
    textNode(element("p", body), "Card 4111 1111 1111 1111"),
    textNode(element("p", body), "PAN ABCDE1234F")
  ]);

  assert.deepEqual(reasons(result), ["text-pii:email", "text-pii:payment-card", "text-pii:pan"]);
});

test("fail-closed: a detector failure is reported, not swallowed", () => {
  const { result } = scanPage(
    (body) => [textNode(element("p", body), DEMO)],
    { breakDetector: true }
  );

  // popup.js turns textPiiError into scanFailed and refuses to transmit.
  assert.match(result.textPiiError, /simulated detector failure/);
  assert.equal(textRegions(result).length, 0);
});

test("fail-closed: content.js refuses to register without pii-patterns.js", () => {
  assert.throws(
    () => scanPage((body) => [textNode(element("p", body), DEMO)], { skipPatterns: true }),
    /pii-patterns\.js was not loaded/
  );
});
