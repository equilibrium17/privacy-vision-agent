// Tests for the popup's provider selector (Groq / NVIDIA / Ollama).
// Run with: npm test
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (file) => readFileSync(new URL(`../${file}`, import.meta.url), "utf8");
const html = read("popup.html");
const popup = read("popup.js");

function providerOptions() {
  const select = html.match(/<select[^>]*id="providerSelect"[^>]*>([\s\S]*?)<\/select>/);
  assert.ok(select, "providerSelect exists");
  return {
    markup: select[1],
    values: [...select[1].matchAll(/<option value="([^"]+)"/g)].map((match) => match[1])
  };
}

test("the selector offers Groq, NVIDIA and Ollama, with Groq still the default", () => {
  const { markup, values } = providerOptions();

  assert.deepEqual(values, ["groq", "nvidia", "ollama"]);
  assert.match(markup, /<option value="groq" selected>/);
});

test("every selectable provider is a value the server accepts", () => {
  const literal = read("server/main.py").match(/provider: Literal\[([^\]]+)\]/);
  assert.ok(literal, "AnalyzeRequest.provider literal found");

  for (const value of providerOptions().values) {
    assert.ok(literal[1].includes(`"${value}"`), `server accepts ${value}`);
  }
});

test("the selection reaches the server only through the existing provider field", () => {
  // Plan, action and verify requests each send the selector value unchanged.
  assert.equal((popup.match(/provider: providerSelect\.value/g) ?? []).length, 3);
  // Routing stays server-side: no provider-specific logic or endpoints in the popup.
  assert.equal(/nvidia/i.test(popup), false);
  // Migrated-away providers must not linger anywhere in the popup.
  assert.equal(/gemini|openrouter/i.test(popup), false);
});

test("no provider credential or provider endpoint is present in the extension", () => {
  const sources = ["popup.html", "popup.js", "popup-ui.js", "agent-loop.js", "background.js", "manifest.json"];

  for (const file of sources) {
    assert.equal(
      /NVIDIA_API_KEY|GROQ_API_KEY|integrate\.api\.nvidia\.com|api\.groq\.com|nvapi-|gsk_/i.test(read(file)),
      false,
      `${file} must not reference provider credentials or endpoints`
    );
  }

  // The extension only talks to the local server.
  assert.deepEqual(JSON.parse(read("manifest.json")).host_permissions, ["http://127.0.0.1:8000/*"]);
});
