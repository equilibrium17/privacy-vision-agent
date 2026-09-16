// Tests for the structured final-verification evidence builder.
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../verification-evidence.js", import.meta.url), "utf8");
new Function(source)();

const { buildVerificationEvidence } = globalThis.__privacyVisionVerificationEvidence;

const kinds = (evidence) => evidence.items.map((item) => item.kind);
const texts = (evidence) => evidence.items.map((item) => item.text).join("\n");

const listingSnapshot = {
  url: "https://shop.test/products",
  title: "Products",
  overlays: [],
  numericControls: [],
  actionControls: [
    { tag: "button", selector: "#add-to-cart-1", label: "Add to cart" },
    { tag: "button", selector: "#open-cart", label: "Cart" }
  ]
};

test("1: cart drawer open with quantity 2 exposes strong, citable evidence", () => {
  const evidence = buildVerificationEvidence({
    url: "https://shop.test/products",
    title: "Products",
    overlays: [{ selector: "#cart-drawer", role: "dialog", label: "Your cart" }],
    numericControls: [{ selector: "#quantity-1", label: "Quantity", type: "number", value: "2" }],
    actionControls: [
      { tag: "button", selector: "#remove-from-cart-1", label: "Remove" },
      { tag: "button", selector: "#proceed-to-buy", label: "Proceed to buy" }
    ]
  }, { completedSteps: ["Open cart", "Set quantity to 2"] });

  assert.equal(evidence.items[0].text, "Current URL: https://shop.test/products");
  assert.ok(kinds(evidence).includes("open-overlay"));
  assert.equal(kinds(evidence).includes("overlay-state"), false);
  assert.match(texts(evidence), /Open overlay #cart-drawer role=dialog "Your cart"/);
  assert.match(texts(evidence), /Input #quantity-1 "Quantity" = 2/);
  assert.match(texts(evidence), /Visible button #remove-from-cart-1 "Remove"/);
  assert.match(texts(evidence), /Visible button #proceed-to-buy "Proceed to buy"/);

  // Ids are unique and sequential so the verifier can cite them.
  assert.deepEqual(evidence.items.map((item) => item.id), evidence.items.map((_item, index) => `E${index + 1}`));
});

test("2: cart not open states the closed state explicitly and has no open-overlay evidence", () => {
  const evidence = buildVerificationEvidence(listingSnapshot);

  assert.ok(kinds(evidence).includes("overlay-state"));
  assert.match(texts(evidence), /No open dialog, drawer or modal is visible/);
  assert.equal(kinds(evidence).includes("open-overlay"), false);
  assert.equal(kinds(evidence).includes("numeric-control"), false);
});

test("3: history claiming the cart was opened never becomes evidence when the DOM shows it closed", () => {
  const claims = ["Open the cart", "Increase quantity to 2"];
  const evidence = buildVerificationEvidence(listingSnapshot, { completedSteps: claims });

  // Claims are returned separately and are not citable evidence items.
  assert.deepEqual(evidence.claims, claims);
  for (const claim of claims) {
    assert.equal(texts(evidence).includes(claim), false, `claim leaked into evidence: ${claim}`);
  }

  // The current state still says closed.
  assert.match(texts(evidence), /No open dialog, drawer or modal is visible/);
  assert.equal(kinds(evidence).includes("open-overlay"), false);
});

test("only safe short numeric values are exposed; passwords and free text are not", () => {
  const evidence = buildVerificationEvidence({
    url: "https://shop.test/cart",
    numericControls: [
      { selector: "#quantity-1", type: "number", value: "2" },
      { selector: "#note", type: "text", value: "leave at door" },
      { selector: "#zip", type: "text", value: "700016" },
      { selector: "#pin", type: "password", value: "1234" }
    ]
  });

  assert.match(texts(evidence), /#quantity-1 = 2/);
  assert.equal(texts(evidence).includes("leave at door"), false);
  assert.equal(texts(evidence).includes("700016"), false);
  assert.equal(texts(evidence).includes("#pin"), false);
});

test("a control seen by both the DOM read and the element scan is listed once", () => {
  const evidence = buildVerificationEvidence(
    { actionControls: [{ tag: "button", selector: "#proceed-to-buy", label: "Proceed to buy" }] },
    { elements: [{ tag: "button", selector: "#proceed-to-buy", label: "Proceed to buy" }, { tag: "a", selector: "#home", label: "Home" }] }
  );

  assert.equal(evidence.items.filter((item) => item.text.includes("#proceed-to-buy")).length, 1);
  assert.ok(texts(evidence).includes("#home"));
});
