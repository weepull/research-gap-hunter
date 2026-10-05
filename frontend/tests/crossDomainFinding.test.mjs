// T2: a cross-domain result with no matches must be stated as a finding, with the
// reason and the evidence, never rendered as an empty panel that reads "none exist".
import { test } from "node:test";
import assert from "node:assert/strict";
import { describeCrossDomainFinding } from "../lib/crossDomainFinding.ts";

const label = (d) => ({ computer_vision: "Computer vision", medical_imaging: "Medical imaging" })[d] ?? d;

function report(overrides = {}) {
  return {
    source_domain: "medical_imaging", target_domain: "computer_vision",
    status: "none_above_threshold", message: "backend message",
    matches: [], total_matches: 0, similarity_threshold: 0.8764, min_seed_papers: 2,
    source_gaps: 56, seed_gaps: 2, target_future_directions: 41,
    ...overrides,
  };
}

test("matches_found is not a finding: the matches are the result", () => {
  assert.equal(describeCrossDomainFinding(report({ status: "matches_found", total_matches: 4 }), label), null);
});

test("none_above_threshold states it as a corpus-size result, with the evidence", () => {
  const f = describeCrossDomainFinding(report(), label);
  assert.equal(f.eyebrow, "Finding");
  assert.match(f.title, /Medical imaging/);
  assert.match(f.title, /Computer vision/);
  assert.match(f.statement, /current corpus size/);
  assert.match(f.statement, /\b2\b.*\b41\b/s, "says how many seeds were compared against how many FDs");
  assert.match(f.statement, /0\.8764/);
  const evidence = Object.fromEntries(f.evidence.map((e) => [e.label, e.value]));
  assert.equal(evidence["Gaps that met the evidence gate"], "2 of 56");
  assert.equal(evidence["Future directions searched"], "41");
});

test("no_corroborated_gaps says no matching was attempted, and why", () => {
  const f = describeCrossDomainFinding(report({ status: "no_corroborated_gaps", seed_gaps: 0 }), label);
  assert.match(f.title, /evidence gate/);
  assert.match(f.statement, /no matching was attempted/i);
  assert.match(f.statement, /at least 2 papers/);
});

test("no_data is distinguished from 'nothing above threshold' and uses the backend's reason", () => {
  const f = describeCrossDomainFinding(report({ status: "no_data", message: "No data: the x corpus has no limitations to match from." }), label);
  assert.equal(f.eyebrow, "No data");
  assert.match(f.statement, /no limitations to match from/);
  const above = describeCrossDomainFinding(report(), label);
  assert.notEqual(f.title, above.title);
});

test("every non-match status produces a stated finding, never an empty one", () => {
  for (const status of ["none_above_threshold", "no_corroborated_gaps", "no_data"]) {
    const f = describeCrossDomainFinding(report({ status }), label);
    assert.ok(f.title.length > 0 && f.statement.length > 0, status);
    assert.ok(f.evidence.length > 0, `${status} shows its evidence`);
  }
});

test("an unknown status fails loudly instead of rendering nothing", () => {
  assert.throws(() => describeCrossDomainFinding(report({ status: "mystery" }), label));
});
