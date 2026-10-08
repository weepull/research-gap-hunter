# PLAN.md — Remediation plan for 7 audited defects

**Status: PLAN ONLY. No code changed. Items 1–5 are gated on an advisor decision.**

Date: 2026-09-28 · Base commit: `0eb58bc` · Working tree clean at time of writing
Source: adversarial audit of 2026-09-27, with every file/line reference re-verified
against the current tree before this plan was written.

---

## Line references: corrections to the audit

Two references in the prior audit were wrong and are corrected here. Anything
downstream that quoted them should be updated.

| Claim | Audit said | Actually | Note |
|---|---|---|---|
| `extract_paper` hardcodes domain | `extractor.py:455` | **`extractor.py:452`** | `PROJECT_HARDENING_PLAN.md` item A8 says `:388`, also stale |
| everything else | — | confirmed | see per-item "Root cause" below |

Verified unchanged and correct: `gap_scorer.py:33` (`_CLUSTER_THRESHOLD`),
`:161-163` (`seed_order`), `:236` (frequency divisor), `:354` (stable sort),
`:408` (`most_common`), `:470-481` (`_count_papers_in_domain`),
`api/main.py:243-266` (`/health`), `:254-259` (swallows), `:262` (`status="ok"`),
`:284-290` (`/corpus` swallow), `:410-447` (`/explain`), `:438`
(`similarity_score=0.0`). Test suite re-run: **327 passed in 0.78s**.

---

## Dependency order

The fixes are **not** independent, and landing them in the wrong order destroys
the ability to measure whether any of them worked.

```
PHASE 0  ── #7 integration harness ──┐
            #6 health/corpus honesty ┘   (no advisor gate; both are instrumentation)
                    │
                    ▼
PHASE 1  ── #1 domain labelling          (must precede #3 and #4)
                    │
                    ▼
PHASE 2  ── #3 cluster threshold         (changes what a gap IS)
                    │
                    ▼
PHASE 3  ── #4 frequency denominator     (changes score resolution)
                    │
                    ▼
PHASE 4  ── #2 tie-breaking              (scope depends on #4's outcome)

PHASE 5  ── #5 /explain grounding        (parallel to 2-4; needs #1 only)
```

**Why this order:**

- **#7 and #6 first.** #7 is the measuring instrument for everything else — without
  an integration test that runs the real pipeline, no before/after claim about #1–#4
  can be trusted, and the project's own rule ("tests must fail against pre-fix code")
  is unenforceable for defects that only appear on real data. #6 must land alongside
  it because an integration test currently cannot distinguish a healthy stack from a
  dead one: `/health` returns `status="ok"` with Neo4j unreachable.
- **#1 before #3 and #4.** Domain labels determine corpus membership. #3 re-derives a
  threshold from a *per-domain* null distribution; #4 re-derives a *per-domain*
  denominator. Both are computed over sets that are currently wrong — the CV corpus
  contains analytic number theory (`2405.00658`), atomic physics (`2401.00920`),
  control theory (`2501.03894`), LLM inference (`2502.01618`, `2310.06825`), copyright
  law (`2406.14526`), and at least one medical-imaging paper (`2305.17456`). Deriving
  a "CV noise floor" from a set that is not CV produces a number that will have to be
  thrown away and re-derived after #1. Any before/after ranking comparison across #1
  is also meaningless, because #1 moves papers between domains.
- **#3 before #4.** Cluster membership is the *numerator* of `frequency_score`. #4
  changes the *denominator*. Changing both at once makes it impossible to attribute a
  ranking shift to either. #3 also changes how many gaps exist at all (today: 27 CV /
  24 MI), so it must settle first.
- **#4 before #2.** This matters and is easy to miss: **#4 may substantially dissolve
  the problem #2 exists to solve.** Ties happen because frequency has almost no
  dynamic range (measured effective influence 4.9–7.4% against a documented 40%). If
  #4 restores frequency resolution, many of the current 18-of-27 tied CV gaps separate
  on their own. #2's scope — whether an explicit tie-breaker is even needed, and what
  it should be — cannot be decided until #4's effect is measured.
- **#5 anywhere after #1.** Independent of scoring, but a grounding check that
  validates a pairing against the corpus needs correct domain labels to validate
  against.

---

# PHASE 0 — Instrumentation (no advisor gate)

These two do not change scoring, ranking, or any displayed number. They change
what the system *admits* about itself. A single recommended approach is given for
each, as instructed.

---

## #7 · The 327-test suite is fully mocked — zero integration coverage

### Root cause (confirmed)

Re-verified: **327 tests, 0.78s**. Collected per file:

```
test_api.py 41 · test_gap_scorer.py 57 · test_extractor.py 38 · test_vectors.py 34
test_batch.py 25 · test_rate_limit.py 23 · test_graph.py 20 · test_cross_domain.py 19
test_selfheal.py 15 · test_config.py 9
```

0.78s is arithmetically incompatible with loading Specter2 (~440MB of weights),
opening a Bolt connection, or issuing a Qdrant query. Confirmed: the only real
`get_neo4j_driver()` references in the suite are `tests/test_graph.py:82` and
`:96`, both inside `monkeypatch`/`MagicMock` scopes. There is no
`@pytest.mark.integration`, no `skipif`-on-service-unavailable, and no
`[tool.pytest.ini_options] markers` entry in `pyproject.toml` (confirmed — the
section contains only `testpaths`).

This is not a missing-tests problem so much as a **category** problem: the suite
tests that each function calls its collaborators correctly, and mocks the
collaborators' return values. Every defect in this plan lives in the *interaction
between real components and real data*, which is precisely the seam the suite
does not cross. All seven defects pass green today.

### Recommended approach

Add a **third test tier** rather than converting existing tests. Keep all 327 unit
tests exactly as they are — they are good unit tests and they are fast, which is
worth preserving.

1. **Register a marker** in `pyproject.toml`:
   `markers = ["integration: requires live Neo4j, Qdrant and Ollama"]`, and set
   `addopts = "-m 'not integration'"` so the default `pytest -q` stays at 0.78s and
   nothing in anyone's habit changes.
2. **Add `tests/integration/conftest.py`** with session-scoped fixtures that probe
   each service once (Bolt connect, `GET /collections`, `GET /api/tags`) and
   `pytest.skip` the whole directory with a named reason if any is down — so a
   contributor without Docker sees `skipped`, not `failed`.
3. **Add a fixture corpus**, not the live one. A ~12-paper JSON fixture committed
   under `tests/integration/fixtures/` with *known, hand-labelled* content: a few
   genuine CV papers, a couple of genuine MI papers, one deliberately off-topic
   paper, and one whose extracted limitations are the boilerplate strings
   (`"remains challenging"`). Loaded into a **dedicated** Neo4j database and
   Qdrant collections (`limitations_test`, `future_directions_test`) so a test run
   can never touch the real corpus. This is the load-bearing design decision:
   integration tests against the *live* corpus would be non-deterministic and
   would change meaning every ingestion.
4. **Assert the invariants, not the numbers.** `score_gaps()` on the fixture must
   return a stable gap count; no gap may cite a paper absent from the fixture; no
   CV gap may cite an MI paper; the same input must rank identically twice
   (determinism, which `_corpus_reference_year` explicitly promises).

### Which tests must fail first

The new integration tests are the proof harness for #1–#5, so each is written to
fail against pre-fix code — that is their point. Specifically, on the fixture
corpus and before any other fix lands:

- a test asserting the off-topic fixture paper is not labelled `computer_vision`
  → **must fail** (proves #1)
- a test asserting no cluster exceeds some fraction of the domain's limitations
  → **must fail** for MI (proves #3; today 52 of 104 in one cluster)
- a test asserting `score_gaps()` produces no exact-score ties → **must fail**
  (proves #2/#4; today 18 of 27 CV gaps are tied)
- a test asserting `/health` reports failure when Neo4j is stopped → **must fail**
  (proves #6)

No existing test needs to change. **Effort: medium** (the fixture corpus and the
isolated-database plumbing are most of it, not the assertions).

### Risk

Low to behaviour, real to CI ergonomics. The named risks: (a) if the default
`addopts` is set wrong, CI silently stops running integration tests and reports
green — mitigate with one meta-test asserting the marker is registered and the
integration directory is non-empty; (b) tests that write to Neo4j need teardown
that cannot leak into the real database — mitigate by requiring an explicit
`NEO4J_TEST_DATABASE` env var and skipping (not defaulting) when it is unset.

---

## #6 · `/health` and `/corpus` swallow real failures

### Root cause (confirmed)

`/health` (`api/main.py:243-266`):

- `:254-255` and `:258-259` — `except Exception: lim_count = 0` / `fd_count = 0`.
  A dead Qdrant, a missing collection, and a genuinely empty collection are one
  indistinguishable outcome.
- `:262` — `status="ok"` is a **literal**, computed from nothing.
- **Neo4j is never contacted at all.** `lifespan` (`:146`) opens a driver and puts
  it on `app.state`, and the handler ignores it. Since Neo4j is the scoring source
  of truth (`_count_papers_in_domain` at `:470-481`), the readiness probe omits
  the one service whose absence breaks `/gaps` entirely.

Net effect, confirmed by reading the handler: a deployment with Neo4j down and
Qdrant down returns `200 {"status":"ok","papers":127,"limitations":0,"future_directions":0}`.

`/corpus` (`:284-290`): `def _count(...)` → `except Exception: return 0`. The
comment says "a missing collection is not an error here", which is a fair
intention, but the handler catches *every* exception, so connection failure and
auth failure render as "0 limitations" in `CorpusBanner`.

### Recommended approach

Distinguish **"absent"** from **"unreachable"**, and report per-dependency status
rather than one boolean.

- Extend `HealthResponse` with a `services: dict[str, str]` field mapping each
  dependency to `"ok"` / `"unreachable"` / `"absent"`, and derive the top-level
  `status` from it: `"ok"` only when every required dependency is `ok`,
  `"degraded"` otherwise.
- **Probe Neo4j** using the existing `driver.verify_connectivity()` (already used
  in `graph/populate.py:30`) plus the same domain count the scorer uses, so the
  probe exercises the real query path rather than just the socket.
- **Narrow the excepts.** Catch `UnexpectedResponse` (already imported in
  `vectors/embed.py:15`) for a genuinely missing collection → `"absent"`, count 0.
  Let connection/timeout errors map to `"unreachable"`.
- Return **HTTP 503** when a required dependency is unreachable. This is the one
  judgment call in #6: a 503 is correct for a readiness probe and is what makes a
  platform health check actually restart a broken container, but it will change
  behaviour for any external monitor currently treating 200 as liveness.
  Recommended: 503 on `degraded`, and document it in `DEPLOYMENT.md`.
- Same treatment for `/corpus._count`: narrow to the missing-collection case, and
  surface an explicit `"unavailable"` rather than `0` when the store cannot be
  reached, so `CorpusBanner` can say "corpus size unavailable" instead of
  asserting a false zero.

### What a user would see

- A broken deploy fails visibly: `/health` returns 503 naming which service is
  down, instead of `"ok"`.
- `CorpusBanner` stops printing "0 limitations" over a working results page when
  Qdrant is merely unreachable — it says the figure is unavailable.
- **No change whatsoever on a healthy stack**, which is the common case.

### Tests: which must fail first

Two currently-passing tests **pin the defect** and must be changed, and changing
them is the proof:

- `tests/test_api.py:133` `test_health_missing_collection_returns_zero` — asserts
  `status_code == 200` with a `get_collection` that raises `Exception("collection
  not found")`. Under the fix this must **split into two tests**: a missing
  collection (→ still 200, counts 0, `services["qdrant"] == "absent"`) and an
  unreachable Qdrant (→ 503, `"unreachable"`). The second assertion fails against
  current code, which is the required proof.
- `tests/test_api.py:225` `test_corpus_survives_missing_collections` — same split.
  Its docstring intention ("rather than failing the page") stays valid for the
  *missing* case and is what the narrowed except preserves.

New tests: Neo4j unreachable → 503 and `services["neo4j"] == "unreachable"`
(**must fail today**, since Neo4j is never probed); all-healthy → 200 with every
service `"ok"`; `/corpus` with unreachable Qdrant → explicit unavailable marker.
Plus the integration test from #7 that stops the real Neo4j.

`tests/test_api.py:109` `test_health_returns_ok` needs its mock extended with a
Neo4j probe but its assertion holds.

**Effort: small.** **Risk:** moderate and entirely in the 503 decision — any
external uptime monitor or platform health check pointed at `/health` will start
seeing failures it previously did not. That is the intended outcome, but it is a
behaviour change on deployed infrastructure and belongs in the commit message and
`DEPLOYMENT.md`. The narrowed excepts carry a second risk: an exception type not
anticipated (e.g. a Qdrant client version raising something new) would now
propagate as a 500 instead of being absorbed — mitigate by keeping a final
`except Exception` that maps to `"unreachable"` rather than re-raising.

---

# PHASE 1 — Data labelling

## #1 · `extract_paper()` hardcodes `domain="computer_vision"`

> ### ⚠ REQUIRES ADVISOR DECISION BEFORE IMPLEMENTATION
> Domain drives the frequency denominator, every Qdrant filter, cross-domain
> routing, and which null distribution #3 is derived from. Per CLAUDE.md's
> advisor–executor protocol this is a data-labelling and therefore scoring change.
> **Existing plan item: `PROJECT_HARDENING_PLAN.md` A8, status NEEDS ADVISOR
> DECISION.** This plan supersedes A8's line reference (`:388` → `:452`) and
> escalates its priority from "lower" to first-in-order.

### Root cause (confirmed)

`pipeline/extractor.py:452` — inside `PaperExtract(...)`, the literal
`domain="computer_vision"`. `PaperExtract.domain` also *defaults* to
`"computer_vision"` (`:130`), so the field is doubly hardcoded: the model default
and the construction site.

Confirmed the override situation: exactly two call sites patch it afterwards —
`pipeline/cross_domain.py:110` and `api/main.py:377`, both
`paper.model_copy(update={"domain": domain})`. The third path,
`pipeline/batch.py:163-164` (`ingest_from_query`), calls `extract_paper(arxiv_id)`
and passes the result straight to `_paper_to_row(paper)` with **no domain
handling anywhere in the function** (grep for `domain` in `batch.py` returns only
comments and the `_LIST_FIELDS` machinery). So query-based ingestion stamps every
paper `computer_vision` regardless of content.

Nothing downstream validates the label against the paper. Confirmed consequence
from the live corpus: 5 of the top 7 CV gaps originate in papers that are not
computer vision, and the flagship cross-domain match is driven by a
medical-imaging sentence from `2305.17456` ("Trustworthy Deep Learning for
Medical Image Segmentation") labelled `computer_vision`.

### Options

**1A — `domain` becomes a required, validated parameter with no default.**
`extract_paper(arxiv_id, domain)` positional-or-keyword required; remove the
default from `PaperExtract.domain`; validate against an explicit enum/registry of
known domains and reject anything else.
- *For:* Eliminates the failure mode by construction — a new call site cannot
  silently mislabel, it fails to run. Deterministic, zero inference risk, no model
  dependency, trivially testable. Makes the current "works by convention" contract
  explicit.
- *Against:* Does not fix the **existing** 127 mislabelled rows — that needs a
  separate relabelling pass (see "Backfill" below). Pushes the judgment onto the
  operator, who is exactly who got it wrong: `ingest_from_query("computer vision
  object detection")` returning a number-theory paper would still be stamped CV,
  because the *caller* said CV. Guarantees label *provenance*, not label *truth*.

**1B — content-based classification during extraction.**
Ask the LLM for the domain as part of the extraction JSON (a 7th key), constrained
to a closed list, or classify from the abstract with a separate call.
- *For:* Attacks the actual problem — a paper's domain comes from the paper.
  Catches the off-topic-search-result case that 1A cannot. Cheap to add: one more
  key in an existing call, no extra round trip.
- *Against:* Introduces an LLM-generated value into the **scoring** path, which
  the audit specifically credited this project for avoiding (no score is currently
  LLM-derived). `llama3.1:8b` on a truncated conclusion section will be noisy, and
  a misclassification is now silent *and* non-deterministic — re-extraction could
  move a paper between domains, breaking the determinism guarantee
  `_corpus_reference_year` relies on. Also needs a confidence/abstain path, or it
  will confidently classify the number-theory paper as *something*.

**1C — keyword/heuristic matching over title + abstract.**
Curated term lists per domain, score, assign, abstain below a margin.
- *For:* Deterministic and auditable — you can read why a paper landed where it
  did. No LLM in the scoring path. Cheap.
- *Against:* Brittle in exactly this corpus. "Segment Anything" (`2304.02643`) is
  a CV paper full of medical-sounding segmentation vocabulary; `2305.17456` is a
  medical paper full of CV vocabulary. Keyword lists become a maintained artifact
  that silently rots, and tuning them is unprincipled in the same way the 0.86
  threshold is (#3).

**Combination worth the advisor's attention:** 1A **plus** an abstain-capable 1B
or 1C used only as a *verifier* — the caller must declare the domain (1A), and a
classifier that disagrees strongly flags the paper for review rather than
overriding it. This keeps the LLM out of the scoring path while still catching
off-topic search results. Costs more than either alone.

### Backfill — needed under every option

None of the options repair the 127 existing rows, and this is the harder half.
Re-ingestion is **not** available: `graph/populate.py:83-141`
(`_upsert_paper_counting`) only MERGEs and never removes relationships, and
`Limitation` is `UNIQUE` on `text` (`:17`), so re-extracting a paper that already
has limitations attaches new nodes alongside the old ones — the documented hazard
in CLAUDE.md and `selfheal.py`'s docstring. A relabelling pass must therefore
update `Paper.domain` in place (SQLite + Neo4j) and then **re-run
`embed_limitations()` / `embed_future_directions()`**, because the Qdrant `domain`
payload is copied from `p.domain` at embed time (`vectors/embed.py:177`, `:186`).
Whether the off-topic papers are *relabelled* or *removed from the corpus* is
itself an advisor decision — removal changes the frequency denominator (46 → ~40
for CV) and therefore every score.

### What a user would see

Depends on the option, but in all cases: off-topic entries leave the CV gap list.
Under a relabel-or-remove backfill, the current #1 ("Fair use doctrine…"), #2
(control theory), #3 (LLM latency), #4–#7 (number theory, physics) all disappear
from or move within `/gaps?domain=computer_vision`, and the top of the list
becomes genuine CV limitations — today's #8 ("Loses track of or confuses objects
in crowded scenes…", 3 papers) and #16 are the plausible new leaders.
`/cross-domain` output changes substantially, and the top match
(`2305.17456`-driven, 0.9039) should vanish as medical→medical once its source
paper is relabelled `medical_imaging`. **Corpus counts in `CorpusBanner` change**,
which is user-visible and should be expected.

### Tests

**Must fail against pre-fix code:**
- `extract_paper` called without a domain raises (1A), or returns a
  content-appropriate domain for a known non-CV fixture paper (1B/1C). Fails today
  — currently returns `"computer_vision"` unconditionally.
- `ingest_from_query` does not produce `computer_vision` rows for a non-CV
  fixture. Fails today.
- Integration: no gap in `/gaps?domain=computer_vision` cites a paper whose
  fixture-declared true domain is not CV. Fails today.

**Existing tests that must change:** `tests/test_extractor.py:590`
`test_paper_extract_model_defaults_domain` asserts
`paper.domain == "computer_vision"` — under 1A this default is removed, so the
test must invert to assert a `ValidationError` when `domain` is omitted.
`tests/test_extractor.py:491` also asserts `result.domain == "computer_vision"`
and needs updating. Any test constructing `PaperExtract` without `domain` will
start erroring under 1A — expect a broad but mechanical sweep across
`test_extractor.py`, `test_batch.py`, `test_graph.py`, `test_selfheal.py`.

**Effort: medium for the code, large including backfill + re-embed + verification.**

### Risk

High, and the highest in this plan. Removing the `PaperExtract.domain` default
(1A) is a breaking API change across four test files and `selfheal.py`'s
reconstruction path (`_paper_to_row` round-trips the field). The backfill touches
both stores *and* requires a full re-embed, which rewrites both Qdrant
collections — and `vectors/embed.py:236` assigns point IDs positionally over an
unordered Cypher result with no deletion path, so a re-embed that produces fewer
records leaves orphaned points behind. That orphan hazard should be fixed (or at
minimum the collections dropped and rebuilt) as part of this work, or #1's
backfill will quietly corrupt the vector store.

---

# PHASE 2 — What counts as a gap

## #3 · `_CLUSTER_THRESHOLD = 0.86` was never derived and sits below its noise floor

> ### ⚠ REQUIRES ADVISOR DECISION BEFORE IMPLEMENTATION
> CLAUDE.md rule 3 names similarity thresholds and the clustering approach
> explicitly. This is the single highest-impact number in the system — it defines
> what a "gap" is. **No item exists for it in `PROJECT_HARDENING_PLAN.md`**; it is
> mentioned once at `:58` only to argue it is not comparable to `0.85`. It is a
> genuine blind spot in the hardening plan, not a deferred item.

### Root cause (confirmed)

`pipeline/gap_scorer.py:30-33`, verbatim:

```
# Similarity threshold for grouping two limitation statements into one cluster.
# Specter2-base pairwise similarities on this corpus are compressed (median ~0.82),
# so the threshold sits well above the median to keep clusters tight.
_CLUSTER_THRESHOLD = 0.86
```

The justification is **"above the median"**. That is the exact reasoning error the
project already identified and corrected for the cross-domain threshold —
`pipeline/cross_domain.py:160-165` states that 0.82 "sat *below the median of pure
noise*" and that a compressed space needs a *higher* bar. The correct bar is the
null distribution's upper percentile, not its median. That correction was applied
to `_SOLUTION_THRESHOLDS` and to `find_cross_domain_matches`, and **not** to the
threshold that decides what a gap is.

Measured from the live stored vectors (limitation × limitation, the population
this threshold actually governs):

| population | n pairs | mean | median | **p95** | p99 | max | share clearing 0.86 |
|---|---:|---:|---:|---:|---:|---:|---:|
| computer_vision | 2,016 | 0.8180 | 0.8177 | **0.8736** | 0.8974 | 0.9320 | **10.9%** |
| medical_imaging | 5,356 | 0.8400 | 0.8403 | **0.8957** | 0.9138 | 0.9599 | **28.3%** |

Consequence, measured: MI clustering collapses **52 of 104 limitations into one
cluster** (33 papers, `frequency=0.284`, ranked #11), labelled by one arbitrary
member about convolution kernel sizes and containing unrelated statements about
stain normalisation, augmentation training time, and FC-layer matrix sizes. MI
cluster sizes: `[52, 18, 6, 3, 3, 2, 2, 2, 1×16]` — two clusters hold 70 of 104
limitations. CV: `[9, 6, 5, 5, 4, 4, 3, 3, 2×6, 1×13]`.

The `cluster_limitations` docstring (`:120-137`) claims seed-anchoring "prevents
transitive A→B→C chains from collapsing unrelated limitations into one giant
cluster." Seed-anchoring does remove the transitive-chain mechanism, but the giant
cluster persists by a different route: at a 28%-noise threshold a long, generic
seed absorbs a quarter of the corpus *directly*. The docstring's claim is
therefore false as stated and must be corrected whichever option is chosen.

**Note for the advisor:** the two thresholds that *were* derived hold up on a
corpus that has since doubled from 63 to 127 papers — CV solution threshold
re-derives to **0.8773 exactly**, cross-domain to **0.8789** vs coded 0.8792. The
methodology works. It simply was not applied here.

### Options

**3A — derive per-domain from the limitation×limitation null p95, as A1/A2 did.**
CV → ~0.8736, MI → ~0.8957. Two constants, same pattern as
`_SOLUTION_THRESHOLDS`, with the same "re-derive after significant ingestion"
note.
- *For:* Methodologically identical to the accepted A1/A2 decision, so it needs no
  new justification. Demonstrably reproducible. Directly attacks the mega-cluster.
- *Against:* p95 still admits 5% noise *per pair*, and clustering evaluates
  O(n) pairs per seed — so a seed with 100 candidates still expects ~5 chance
  members. **p95 is the wrong operating point for a transitive-ish operation even
  though it is right for a single accept/reject.** MI's 0.8957 is very close to its
  p99 (0.9138) and its max (0.9599), so the usable band is narrow and MI may
  fragment to near-all-singletons. Raising CV from 0.86 to 0.8736 is a *smaller*
  change than it looks and may not break the CV 9-member cluster.

**3B — derive at a stricter percentile (p99), accepting fragmentation.**
CV → ~0.8974, MI → ~0.9138.
- *For:* Accounts for the multiple-comparisons reality above: ~1% per-pair noise is
  a defensible operating point when each seed tests many candidates. Almost
  certainly eliminates both mega-clusters.
- *Against:* Will push both domains heavily toward singletons. That collides with
  #4 and with A9 — singleton clusters are exactly where `frequency_score` is
  minimal and `solution_deficit_score` saturates at 1.0, which is the mechanism
  producing today's meaningless ~0.60 tie block. **3B could make #2 and #4
  markedly worse.** Also erodes the premise of the product: if nothing clusters,
  there are no corroborated gaps, only 168 individual sentences.

**3C — keep a similarity threshold but add a structural guard.**
Leave the threshold near its derived value and additionally cap cluster size (as a
fraction of domain limitations), or require a candidate to exceed the threshold
against the seed *and* the current members' mean, or re-seed when a cluster grows
past a bound.
- *For:* Directly targets the observed failure (one cluster swallowing half a
  domain) rather than approaching it through a scalar. Preserves genuine
  multi-paper clusters that both 3A and 3B risk destroying.
- *Against:* Introduces a new mechanism and therefore a new unprincipled constant
  (what cap? why?) — the same class of defect this item is fixing. Harder to
  justify and to test. A cap also makes clustering order-dependent in a new way:
  which members make the cut depends on Qdrant hit order.

**Orthogonal defect worth folding in (advisor should be told, and may treat it as
a separate item):** the clustering neighbour query is the only filtered query in
the project that does **not** pass a `domain` filter. `_query_neighbours`
(`:187-210`) is called with `limit=len(limitations)` (`:158`) against a collection
holding all domains. Measured for CV: **mean 42.2 of every 64 hits are
other-domain** and discarded at `:177`, and **3 genuine above-threshold CV pairs
were lost to window truncation**. This contaminates any null derivation done
naively against the live collection and will worsen as MI grows. Compare
`search.py:30`, `gap_scorer.py:447`, `cross_domain.py:193`, which all filter
server-side. Fixing it is small and low-risk, but it *changes clustering output*,
so it cannot be slipped in as a "mechanical fix" — it needs to be part of this
decision.

### What a user would see

Under 3A/3B: the MI gap list changes fundamentally. The 33-paper "convolution
kernel sizes" gap at #11 breaks apart; MI gap count moves off 24; `frequency_score`
values for MI drop sharply as the mega-cluster's 0.284 dissolves into many small
numbers. `SupportBadge`'s "*N papers independently report this limitation*" stops
being wildly false for MI. Under 3B specifically, expect many more amber
single-source badges — honest, but visibly thinner output.

### Tests

**Must fail against pre-fix code:**
- Integration, on the #7 fixture corpus: no cluster contains more than a defined
  fraction of a domain's limitations. **Fails today** (52/104 MI).
- A derivation test that recomputes the null p95 from stored vectors and asserts
  the coded constant matches within tolerance — mirroring whatever guard exists
  for A1/A2. Fails today (0.86 vs 0.8736/0.8957).
- If the domain filter is included: a test asserting the neighbour query carries a
  `domain` filter, and an integration test that an MI limitation never appears in a
  CV cluster. Fails today.

**Existing tests that must change:** `tests/test_gap_scorer.py:601`
`test_cluster_limitations_groups_similar` feeds a hit at **0.90** and expects
clustering; `:622` `test_cluster_limitations_singletons_below_threshold` feeds
0.40 and expects singletons. The 0.40 case is safe under any option. **The 0.90
case breaks under 3B** (CV p99 ≈ 0.8974 is fine, but MI p99 ≈ 0.9138 > 0.90) and
under any 3C variant that adds a second condition. Both docstrings hardcode "0.86"
in prose and must be updated. Any test that patches `_CLUSTER_THRESHOLD` or relies
on its value needs a sweep.

The `cluster_limitations` docstring claim about giant clusters must be corrected
regardless of option chosen.

**Effort: medium** for 3A/3B (constants + derivation script + test sweep),
**large** for 3C (new mechanism, new tests, new justification).

### Risk

High. This changes every gap in both domains and therefore every number the
product displays. Two specific hazards: (a) deriving the null from the live
collection **without** a domain filter would produce a contaminated number — the
derivation script must filter by domain, or it reproduces the very bug described
above; (b) 3B's fragmentation actively worsens #2 and #4, so #3 and #4 should be
decided with each other's outcomes in view rather than sequentially in isolation.

---

# PHASE 3 — Score composition

## #4 · Documented 40/35/25 weighting behaves as ~7/54/39

> ### ⚠ REQUIRES ADVISOR DECISION BEFORE IMPLEMENTATION
> This is the scoring formula and its inputs. CLAUDE.md names the 0.40/0.35/0.25
> weights as "core IP" and forbids changing them unilaterally. Related existing
> items: `PROJECT_HARDENING_PLAN.md` **A9** (deficit dimensionally incoherent,
> DEFERRED) and **A4** (tier weights unvalidated) — see the A4 note below, which
> is now stale.

### Root cause (confirmed)

Two compounding causes, both re-verified.

**(i) The denominator includes papers that can never contribute a numerator.**
`compute_frequency_score` (`:213-236`) ends `return min(weighted_sum / total_papers, 1.0)`,
where `total_papers` is `_count_papers_in_domain()` (`:470-481`) —
`MATCH (p:Paper {domain: $domain}) RETURN count(p)`, i.e. **every** Paper node in
the domain. Measured:

| domain | denominator (all Paper nodes) | papers that extracted ≥1 limitation | papers that only dilute |
|---|---:|---:|---:|
| computer_vision | 46 | 26 | **20 (43%)** |
| medical_imaging | 81 | 50 | **31 (38%)** |

So `frequency_score` is not "share of papers reporting this limitation" as the
docstring (`:214`) and the UI hint in `frontend/components/GapCard.tsx:84` both
state — it is that share **multiplied by the extraction success rate**. A
singleton gap is structurally capped at `1.0/46 = 0.0217`.

**(ii) Consequently the term has no dynamic range.** Measured over the 27 live CV
gaps:

| term | nominal weight | value range | contribution span | **effective influence** (span / sd) |
|---|---:|---|---:|---|
| frequency | 0.40 | 0.0109–0.1304 | 0.0478 | **7.4% / 4.9%** |
| recency | 0.35 | 0.0–1.0 | 0.3500 | **54.0% / 56.3%** |
| deficit | 0.25 | 0.0–1.0 | 0.2500 | **38.6% / 38.7%** |

```
Pearson(full score, score with the frequency term DELETED) = 0.99784
top-10 order identical with frequency deleted: True
```

The 40%-weighted term is inert: deleting it entirely does not move the top 10.
Recency and deficit span the full unit interval; frequency spans 0.12 of it.

**Stale plan item to flag to the advisor:** `PROJECT_HARDENING_PLAN.md:128-137`
(A4) deprioritises the tier weights because "the whole corpus currently carries
`tier="explicit"`, so the weighting is inert today." **That is no longer true** —
the live corpus is `explicit 28 / conclusion 55 / inferred 44`, only 22%
explicit. The tier weights are live and load-bearing: they are precisely what
separates today's rank 1–3 (`0.75/46 = 0.0163`) from rank 4–7 (`0.5/46 = 0.0109`).
A4 should be re-opened alongside #4.

### Options

**4A — fix the denominator only; leave the weights at 0.40/0.35/0.25.**
Divide by papers-in-domain-with-at-least-one-extracted-limitation (26 CV / 50 MI),
not all papers.
- *For:* Smallest principled change. Makes the term mean what the docstring and the
  UI already claim it means, which is a correctness fix rather than a redesign, and
  needs no new constant. Roughly 1.77× (CV) / 1.62× (MI) more dynamic range.
- *Against:* Does **not** close the gap to 40%. Frequency would still span only
  ~0.02–0.23 against recency/deficit's 0–1, so effective influence lands around
  12–13%, not 40%. The headline discrepancy survives. Also changes the meaning of a
  published number without changing its name — a stored `GapResult` becomes
  incomparable to a fresh one, and the corpus figure in `CorpusBanner` would no
  longer be the divisor, contradicting `api/main.py:112-118` and
  `frontend/components/CorpusBanner.tsx`'s documented promise that the displayed
  count *is* the one that divides `frequency_score`. That promise would need
  updating, or a second figure shown.

**4B — normalise each term before weighting.**
Rescale frequency (e.g. by the max observed in the domain, or rank-normalise all
three terms) so each spans [0,1] across the result set, then apply 40/35/25.
- *For:* Makes the documented weights actually *be* the weights — the honest
  reading of "40% frequency". Directly closes the stated defect. `GapCard` already
  scales the frequency *bar* to the max on screen (`:82,84`) while showing the true
  number, so the UI concept exists.
- *Against:* Breaks the absolute interpretability of `score` — it becomes a
  within-result-set relative measure, so adding one paper can change every score
  with no change to the underlying evidence. That **destroys the determinism
  property** `_corpus_reference_year` (`:368-400`) was specifically designed to
  protect, and makes a stored `GapResult` incomparable across runs. It also invites
  the criticism that the ranking is normalised until it produces a desired spread.
  If chosen, the normalisation basis must be corpus-derived and fixed, not
  result-set-derived.

**4C — accept the measured behaviour and re-document the weights honestly.**
Leave the formula, fix the docstrings, CLAUDE.md and the UI hint to state that
frequency is a low-resolution corroboration signal at small corpus size and that
ranking is recency- and deficit-dominated; optionally reweight to something closer
to observed influence, or drop frequency to a displayed-but-unweighted qualifier.
- *For:* Cheapest, zero ranking churn, zero risk of breaking a working path. There
  is a real argument that at n=46 a frequency term *should* be near-inert — a
  1-vs-3 paper difference is not statistically meaningful (Wilson 95% CIs [0.004,
  0.113] and [0.022, 0.175] overlap heavily), so a formula that largely ignores it
  is arguably correct and the documentation is what is wrong.
- *Against:* Concedes that "frequency of reporting" — the product's core claim to
  be finding *corroborated* gaps rather than individual complaints — is not
  meaningfully in the ranking. Leaves the ~0.60 tie block untouched, so #2 must
  then carry the whole fix.

**These interact with #3 and with A9.** 3B-style fragmentation pushes everything
toward singletons, where frequency is minimal and deficit saturates at 1.0 — so
4A/4B's benefit partly depends on #3's outcome, and A9's saturating-deficit
options (3B/3C in that item) would change the recency/deficit balance that
currently dominates. The advisor may wish to decide #3, #4 and A9 as one package.

### What a user would see

4A: all `frequency_score` values rise ~1.6–1.8×; modest reordering where gaps
differ mainly in paper count; the `CorpusBanner` figure stops being the divisor.
4B: scores change globally and become relative; likely the largest reordering of
any fix here, and repeated runs over a growing corpus would show scores drifting
without new evidence. 4C: no output change at all — only labels, hints and docs
change, which is itself user-visible on `GapCard`'s explanatory text.

### Tests

**Must fail against pre-fix code:**
- 4A: `compute_frequency_score` given a domain where some papers have no
  limitations returns the value computed against the *contributing* count. Fails
  today. **Note the signature problem:** `compute_frequency_score(cluster,
  total_papers)` takes the denominator as a plain int, so 4A requires either a new
  parameter or a changed caller contract at `:326` — a test should pin whichever is
  chosen.
- 4B: a test asserting each term's post-normalisation contribution to score
  variance is within tolerance of its nominal weight. Fails today (4.9/56.3/38.7).
- 4C: a documentation-consistency test, or none — 4C's "test" is that all 327 keep
  passing while docstrings change, which proves nothing about behaviour and should
  be stated as such.

**Existing tests that must change:** the eight `compute_frequency_score` tests at
`tests/test_gap_scorer.py:71,77,87,93,99,106,112,126` all pass `total_papers`
directly and will need their expectations recomputed under 4A/4B.
`tests/test_api.py:194` `test_corpus_paper_count_matches_the_scoring_denominator`
**directly pins the invariant 4A breaks** and must be changed — it is the cleanest
available proof that 4A landed. `:87` `test_compute_frequency_score_caps_at_one`
should be re-examined: under 4A the cap becomes reachable in normal operation
rather than being a defensive clamp.

**Effort: small (4C), medium (4A), medium–large (4B).**

### Risk

4A: moderate — changes every frequency value and breaks a documented
corpus-figure ↔ denominator invariant that the API and frontend both assert in
comments. 4B: high — sacrifices determinism and cross-run comparability, both of
which are explicitly-designed properties of this system; should not be chosen
without a decision to abandon them. 4C: low to code, high to the product's claim.

---

# PHASE 4 — Ordering

## #2 · Ranking ties are broken by description string length

> ### ⚠ REQUIRES ADVISOR DECISION BEFORE IMPLEMENTATION
> Tie-breaking is ranking policy — what "more important" means when scores are
> equal. **Decide this after #4 is measured**, because #4 may remove most ties.

### Root cause (confirmed)

Three verified links in a chain:

1. `pipeline/gap_scorer.py:161-163` —
   `seed_order = sorted(range(len(limitations)), key=lambda i: len(texts[i]), reverse=True)`.
   Clusters are created in descending seed-text-length order, so `clusters` is
   ordered by string length.
2. `:403-408` — `_cluster_centroid_text` returns `Counter(...).most_common(1)[0][0]`.
   Because `Limitation` is `UNIQUE` on `text` (`graph/populate.py:17`), every count
   is 1, so the "most frequent" tie always resolves to insertion order, which is
   the seed. Measured across all 27 CV clusters: **description == seed 27/27,
   description == longest member 27/27, every-count-is-1 27/27.** The function
   never computes a centroid; the docstring at `:404` describes an unreachable path.
3. `:354` — `results.sort(key=lambda gap: gap.score, reverse=True)`. Python's sort
   is stable, so equal scores retain `clusters` order, i.e. descending description
   length.

Observed, exactly monotonic within each tie block:

```
rank 1  0.6065  len=72  Fair use doctrine in the United States might not fully…
rank 2  0.6065  len=70  Robust stability of fast moving-horizon estimators…
rank 3  0.6065  len=64  hosting and running a reward model often introduces high latency
rank 4  0.6043  len=50  Assumes specific growth rates for the partial sums
rank 5  0.6043  len=44  Assumes a specific micro-gravity environment
rank 6  0.6043  len=41  Only applies to divisor-bounded functions
rank 7  0.6043  len=34  Limited to two-dimensional systems
```

CV tie groups: `{0.6065: 3, 0.6043: 4, 0.4424: 2, 0.4402: 4, 0.2587: 3, 0.2565: 2}`
— **18 of 27 gaps sit in a tie**. 7 gaps saturate at `recency == 1.0 AND deficit ==
1.0` and therefore land at essentially one score, so the top of the list is a
tie-block ordered by character count.

Note the two distinct defects here: **(a)** ties are broken arbitrarily, and
**(b)** `gap_description` is the longest cluster member rather than a
representative one — which also determines the deficit score and
`proposed_solutions`, since both embed that single string (`:291-296`, `:338`).
(b) is arguably the more damaging and is what makes a 6-member cluster display a
medical sentence written by one of its papers.

### Options

**2A — add explicit, meaningful secondary sort keys.**
e.g. `(score, supporting_paper_count, newest_paper_year, gap_description)` — a
final lexical key guarantees total determinism.
- *For:* Cheap, deterministic, and states a defensible policy: among equally-scored
  gaps, better-corroborated ones rank higher. Aligns ranking with the qualifier
  `SupportBadge` already foregrounds, and with `SupportBadge.tsx:5-7`'s own
  admission that a one-paper gap can outrank a well-attested one. Does not touch
  the score.
- *Against:* Treats the symptom. A 3-way tie at 0.6065 means the *score* failed to
  discriminate; ordering it by paper count is a second ranking criterion outside
  the documented formula, which arguably belongs *in* the formula (that is #4).
  Also: with `supporting_paper_count` as the tiebreak, frequency now influences
  ranking twice — once (negligibly) in the score and once as a tiebreak.

**2B — eliminate ties at source by increasing score resolution.**
Rely on #4 (and possibly A9's saturating deficit) to give the terms real dynamic
range, so exact ties become rare, then add only a lexical final key for
determinism.
- *For:* Fixes the cause. If frequency and deficit genuinely discriminate, ties
  are a non-issue and no policy decision is needed. Keeps one documented formula as
  the sole ranking authority.
- *Against:* Depends entirely on #4's outcome and may not suffice — ties at
  `recency == 1.0, deficit == 1.0` arise from *saturation*, and 4A alone leaves
  those two terms saturating. Under a 3B-fragmented corpus there would be *more*
  such singletons, not fewer. Leaves the product shipping arbitrary order until #4
  and A9 both land.

**2C — surface ties as ties instead of ordering them.**
Rank with explicit ties (1, 1, 1, 4, …) or group equal-scoring gaps visually, plus
a lexical key for stable output.
- *For:* The most honest option — it tells the reader the system cannot separate
  these, which is true. Removes the false precision of "#1" entirely. Cheap
  backend-side.
- *Against:* Requires frontend work (`GapCard`'s rank chip, `app/gaps/page.tsx`)
  and makes "top gap" ill-defined for anything that consumes `/gaps` expecting a
  strict order. A demo audience may read grouped ranks as indecision rather than
  rigour.

**Separate sub-decision the advisor should rule on explicitly:** whether
`_cluster_centroid_text` should be replaced by an actual representative — the
member nearest the cluster's vector centroid, or the member with the most
supporting papers. This is worth deciding independently of tie-breaking because it
changes `gap_description`, `solution_deficit_score` **and** `proposed_solutions`
for every multi-member cluster, and because the current name and docstring are
simply wrong about what the code does. It is also the fix that stops a 6-member
cluster from being labelled with a sentence only one of its papers wrote.

### What a user would see

2A: the ~0.60 block reorders so multi-paper gaps lead; "Fair use doctrine" (1
paper) drops below better-corroborated ties. 2B: nothing until #4 lands, then
fewer ties. 2C: visible rank grouping, no "#1" among tied gaps. If the
centroid-text sub-decision is taken, **gap descriptions change for every
multi-member cluster** — the largest visible text change in this plan, and it
would alter the top cross-domain match, whose source string is a longest-member
artifact.

### Tests

**Must fail against pre-fix code:**
- `score_gaps()` on a fixture producing two equal scores orders them by the chosen
  policy, not by description length. Fails today.
- A determinism test: same input, two calls, identical order — and critically, a
  test that ordering does **not** change when a gap description is padded with
  extra characters at equal score. Fails today (padding changes rank).
- If the centroid sub-decision is taken: `_cluster_centroid_text` (renamed) returns
  the representative member, not the longest. Fails today, 27/27.

**Existing tests that must change:** any `test_gap_scorer.py` test asserting
`score_gaps` output order, and any asserting `_cluster_centroid_text`'s
"most frequent" semantics. The `:404` docstring must be corrected regardless.

**Effort: small (2A), small backend + medium frontend (2C), none-of-its-own (2B).**
The centroid-text sub-decision is **medium** and carries its own risk.

### Risk

2A/2B: low — ordering only, no score changes. 2C: moderate, entirely in the
frontend contract. The centroid sub-decision is **high**: it changes
`gap_description`, the deficit score and `proposed_solutions` for every
multi-member cluster, and it feeds `cross_domain.py:191`, so cross-domain output
changes too.

---

# PHASE 5 — Explanation grounding

## #5 · `/explain` cannot decline an ungrounded pairing

> ### ⚠ REQUIRES ADVISOR DECISION BEFORE IMPLEMENTATION
> Adding a grounding/refusal rule defines when the system asserts a scientific
> connection exists. That is a product-integrity decision, and the threshold or
> corroboration rule it needs is a threshold decision under CLAUDE.md rule 3.

### Root cause (confirmed)

`api/main.py:410-447`, verified line by line:

- `:412-415` — `source_gap` and `target_solution` arrive as **arbitrary client
  query strings** (`Query(..., min_length=1)`). Nothing checks they exist in the
  corpus or correspond to any computed match.
- `:435-443` — a `CrossDomainMatch` is fabricated with `similarity_score=0.0`
  (`:438`, commented "not used by the explanation prompt"), `source_papers=[]`,
  `target_papers=[]`.
- `pipeline/cross_domain.py:43-55` (`_EXPLAIN_PROMPT`) — never interpolates
  `similarity_score` or the paper lists. `:52-54` instructs: *"explain why applying
  this … is scientifically interesting. **Be specific about the shared structure**
  between the problem and the solution."*

So the model receives two strings and an instruction that presupposes a shared
structure exists, with no evidence of strength and no permission to dissent. The
only compliant output is a structural analogy. Demonstrated against live
`llama3.1:8b` with a deliberately meaningless pairing (the number-theory
limitation vs a histopathology future direction):

> "Applying the federated learning approach from medical imaging to the
> divisor-bounded functions problem **in computer vision** is scientifically
> interesting because both domains involve distributed data and computational
> resources… The shared structure lies in the potential for decentralized
> processing and aggregation of local models…"

Fluent, confident, invented — and it repeats the false `computer_vision` label
back as fact, which is #1 surfacing through #5.

### Options

**5A — server-side grounding: verify the pairing before explaining.**
Recompute the pair's similarity from the stored vectors (or require the client to
pass an id from a real `/cross-domain` result) and refuse below threshold with a
422/409 explaining that the pairing does not clear the noise floor.
- *For:* Makes the refusal **empirical rather than rhetorical** — the same null-
  distribution logic already accepted for A1/A2, reused. Deterministic and
  testable, no reliance on model judgment. Closes the arbitrary-string hole
  completely. Also lets the real similarity be passed into the prompt.
- *Against:* Costs an embed + lookup per call (mitigable: `/explain` is already the
  slowest endpoint and is rate-limited to the `llm` tier at 10/min). Constrains
  exploratory use — a user could no longer ask about a hypothesis they thought of
  themselves, which is arguably a legitimate use. Needs a decision on *which*
  threshold and whether it is the same 0.8792.

**5B — prompt-level permission to decline, plus passing the evidence in.**
Interpolate the real similarity and paper counts into the prompt, and rewrite
`:52-54` to allow "these are not meaningfully related" as a valid answer.
- *For:* Cheap, no new lookup, preserves exploratory use. Removes the
  presupposition, which is the specific defect.
- *Against:* Relies on an 8B model to refuse a leading question — weak, and
  untestable in the strict sense (a probabilistic refusal cannot be asserted
  deterministically). Puts a judgment call back in the LLM's hands, which is the
  thing the architecture otherwise carefully avoids. On its own this is a partial
  fix at best.

**5C — reframe the endpoint's contract in the response.**
Keep generation unconditional but return the explanation wrapped with its
measured similarity, whether it cleared the noise floor, and an explicit
"hypothesis, not a finding" marker that the frontend renders prominently.
- *For:* Honest without restricting use; puts the evidence next to the prose so a
  reader can discount it. Consistent with how the project already handles corpus
  transparency (`CorpusBanner`, `SupportBadge`) and with
  `app/cross-domain/page.tsx:118`'s existing "it would be noise presented as a
  discovery" language.
- *Against:* Still emits confident fabricated prose; relies on the reader
  noticing the caveat. Does not stop the output being screenshotted without it.

5A and 5C compose well (verify, and label what survives), and either would
benefit from 5B's prompt change. Worth presenting as a combination.

### What a user would see

5A: `/explain` refuses pairings not present in or not clearing the corpus
threshold, with a message saying so. Explanations of real matches are unchanged
except for being grounded. 5C: every explanation carries its similarity and a
hypothesis marker. Either way, the demonstrated number-theory→histopathology
output becomes either a refusal or a visibly-unsupported hypothesis.

### Tests

**Must fail against pre-fix code:**
- `/explain` with a pairing absent from the corpus returns a refusal (5A). Fails
  today — returns 200 with invented prose.
- `/explain` with a pairing below threshold returns a refusal / carries an
  unsupported marker. Fails today.
- The prompt interpolates the real similarity (5B/5C): assert `_EXPLAIN_PROMPT`
  renders it. Fails today — the template has no such field.
- 5C: the response model includes `similarity_score` and a grounded flag. Fails
  today (`ExplainResponse` is `{explanation: str}` only, `api/main.py:132-133`).

**Existing tests that must change:** `tests/test_api.py:735`
`test_explain_returns_explanation` and `:569`
`test_explain_works_when_not_in_demo_mode` both drive the endpoint with arbitrary
strings and expect 200 — under 5A they must supply a corpus-resident pairing or be
rewritten to assert refusal. `tests/test_cross_domain.py:424`
`test_explain_match_prompt_contains_gap_and_solution` pins the current prompt
contents and must be extended. `:768` `test_explain_missing_params_returns_422`
and `:774` `test_explain_ollama_failure_returns_500` are unaffected.

**Effort: medium (5A), small (5B), small–medium (5C incl. frontend).**

### Risk

Moderate. 5A changes `/explain` from always-succeeds to conditionally-refusing,
which the frontend must handle — `frontend/lib/api.ts` already has
`FeatureDisabledError` for an expected-refusal pattern, so there is a precedent to
follow rather than a new concept. `ExplainResponse` changing shape (5C) is a
breaking API change for any consumer. 5A's added embed call lands on the only
endpoint that already performs synchronous LLM generation, so latency impact is
proportionally small.

---

# Cross-cutting notes

**Re-derivation must be domain-filtered.** Any null-distribution work for #3 must
filter by domain, or it reproduces the unfiltered-neighbour-query bug and produces
a contaminated threshold. The A1/A2 figures reproduce cleanly (CV solution 0.8773
exact, cross-domain 0.8789 vs 0.8792), so the existing methodology is sound and
should be reused verbatim — with the filter.

**Re-embedding hazard, triggered by #1's backfill.** `vectors/embed.py:236`
assigns Qdrant point IDs positionally (`id=i`) over a Cypher result with no
`ORDER BY` (`:172-178`), and `_upsert_records` (`:248`) never deletes. Today 168
points == 168 limitation strings, so no orphans exist. A re-embed producing fewer
records would leave stale points with stale payloads that still match queries.
Any fix that triggers a re-embed should drop and rebuild the collections, or this
should be fixed first. **Also confirmed:** both live collections have
`payload_schema []` — no `domain` index — meaning they have not been re-embedded
since commit `0eb58bc` added `ensure_payload_indexes`. Harmless locally, fatal on
Qdrant Cloud per `embed.py:100-110`.

**Documentation debt each fix creates.** CLAUDE.md's decision log must be appended
for every advisor decision (#1–#5), and these specific passages become wrong once
the corresponding fix lands: the Gap Scoring Formula section (#4), the Similarity
thresholds table (#3), `gap_scorer.py:30-33` (#3), `:120-137`'s giant-cluster
claim (#3), `:404`'s "most frequently occurring" (#2), `CorpusInfo`'s
denominator promise at `api/main.py:112-118` and `CorpusBanner.tsx` (#4),
`GapCard.tsx:84`'s frequency hint (#4), and `SupportBadge.tsx:25`'s "*N papers
independently report this limitation*" — which is false for every multi-member
cluster today and should be corrected regardless of which fixes land.

**Stale hardening-plan items surfaced by this work:** A8's line reference
(`:388` → `:452`) and its "lower priority" ranking; A4's premise that tiers are
uniformly `explicit` (now 28/55/44); and the absence of any item for
`_CLUSTER_THRESHOLD`.

---

# Summary

| # | Item | Phase | Advisor gate | Effort | Risk |
|---|---|---|---|---|---|
| 7 | Integration test tier | 0 | no | medium | low |
| 6 | `/health`, `/corpus` honesty | 0 | no | small | moderate (503) |
| 1 | Domain labelling | 1 | **yes** | medium (large w/ backfill) | **high** |
| 3 | Cluster threshold derivation | 2 | **yes** | medium (large for 3C) | **high** |
| 4 | Frequency denominator / weighting | 3 | **yes** | small–large by option | moderate–high |
| 2 | Tie-breaking (+ centroid text) | 4 | **yes** | small (medium w/ centroid) | low (high w/ centroid) |
| 5 | `/explain` grounding | 5 | **yes** | small–medium | moderate |

**Recommended first action:** raise an ADVISOR QUERY covering #3 and #4 together
(they interact, and A9 belongs in the same conversation), and #1 separately as the
prerequisite whose outcome both depend on. Phase 0 can begin immediately without a
decision, and should — nothing else is verifiable until it exists.
