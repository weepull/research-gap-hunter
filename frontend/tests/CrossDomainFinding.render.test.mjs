// T2: the component renders a finding as a stated result — title, statement, evidence —
// not as the dashed, sunken EmptyState panel.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { importTsx } from "./_tsx.mjs";

const finding = {
  eyebrow: "Finding",
  title: "No corroborated Medical imaging gap has a Computer vision match above the noise floor",
  statement: "2 gaps were compared against 41 future directions at the current corpus size.",
  evidence: [
    { label: "Gaps that met the evidence gate", value: "2 of 56" },
    { label: "Future directions searched", value: "41" },
  ],
};

test("renders the title, the statement and every evidence row", async () => {
  const { default: CrossDomainFinding } = await importTsx("components/CrossDomainFinding.tsx");
  const html = renderToStaticMarkup(createElement(CrossDomainFinding, { finding }));
  assert.match(html, /No corroborated Medical imaging gap/);
  assert.match(html, /at the current corpus size/);
  assert.match(html, /Gaps that met the evidence gate/);
  assert.match(html, /2 of 56/);
  assert.match(html, /Future directions searched/);
  assert.match(html, />Finding</);
});

test("is announced as a status and is not styled as an empty panel", async () => {
  const { default: CrossDomainFinding } = await importTsx("components/CrossDomainFinding.tsx");
  const html = renderToStaticMarkup(createElement(CrossDomainFinding, { finding }));
  assert.match(html, /role="status"/);
  assert.doesNotMatch(html, /border-dashed/);
});
