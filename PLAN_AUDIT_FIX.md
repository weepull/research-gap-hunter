# PLAN_AUDIT_FIX.md — status check and plan for the 7 audit problems

**Plan only. No code changed, nothing committed.** Written 2026-10-04 at HEAD `60d7cfb`,
working tree clean. Every claim below was checked in this run unless it says it was not.

**Services during this run:** Neo4j (`:7687`) and Qdrant (`:6333`) were **down**, with
connection refused on both. That means nothing live was measured: no tie counts, no null
re-derivation, no influence table, no `/health` probe. Each section says where this applies.

---

## Phase 1 — Verified state

### Test results (this run)

```
pytest -q                → 8 failed, 481 passed, 40 deselected, 1 warning in 1.75s
pytest -m integration -q → 40 skipped, 489 deselected, 2 warnings in 0.49s
```

All 8 unit failures are in `tests/test_gap_scorer.py`, and all of them are
`neo4j.exceptions.ServiceUnavailable: Couldn't connect to localhost:7687`:

```
test_score_gaps_anchors_recency_to_corpus_not_wall_clock
test_score_gaps_recency_baseline_is_corpus_wide_not_per_cluster
test_score_gaps_display_solutions_match_scored_solutions
test_score_gaps_threads_domain_into_solution_search
test_score_gaps_formula_weights
test_score_gaps_returns_sorted_list
test_score_gaps_respects_top_n
test_score_gaps_collects_proposed_solutions
```

`RUN_REPORT.md:66` records `489 passed`. That is only true while the production Neo4j is
running. See item (g) below.

### Status of the seven audit problems

| | problem | status | evidence |
|---|---|---|---|
| (a) | domain mislabelling | **partial** | Fixed: `PaperExtract.domain` has no default (`pipeline/extractor.py:155`), `extract_paper` validates it (`:490`), `IngestRequest.domain` is required (`api/main.py:123`, validated at `:138`), and all 4 `extract_paper` call sites pass `domain=` (`api/main.py:598`, `pipeline/batch.py:177`, `pipeline/cross_domain.py:142`, `scripts/ingest_tranche.py:273`). **Still open:** the arXiv primary-category rule (`pipeline/corpus_rules.py:47-55`) is only enforced in `scripts/ingest_tranche.py:198`. `/ingest`, `ingest_from_query` and `ingest_domain_papers` skip it. The read endpoints (`/gaps`, `/search`, `/cross-domain`, `/corpus`) take `domain` as a plain `Query(default="computer_vision")` and never call `validate_domain`. |
| (b) | tie-breaking by string length | **landed** | `scored.sort(key=_ranking_key)` at `pipeline/gap_scorer.py:629`. The key at `:653-690` is `(tier, -score, -papers, -newest_year, description)`, and `_cluster_representative_text` replaces the old `most_common`. Leftover: the `score_gaps` docstring at `:566` still says "The most frequent limitation text in a cluster becomes its gap_description", which Phase 4 made false. Tie counts were **not re-measured** this run because the services were down. |
| (c) | un-derived cluster threshold | **landed** | `_CLUSTER_THRESHOLDS` at `pipeline/gap_scorer.py:71-76` (CV 0.8744 n=6,216, MI 0.8954). There are no `0.86` literals left in code, only in comments. The derivation tool is `scripts/derive_thresholds.py` and the guard is `tests/integration/test_threshold_derivation.py`. The guard was **skipped** this run, so drift against the live corpus is unverified. |
| (d) | 0.40/0.35/0.25 weighting mismatch | **landed (4A + 4C; 4B rejected by the advisor)** | Coefficients are unchanged at `pipeline/gap_scorer.py:592`. The denominator is `_count_contributing_papers` (`:575`, `:912`). The real influence is documented in the docstring (`:450`) and measured by `scripts/measure_weight_influence.py`. Per the advisor decision this stays documented rather than "fixed". The influence table was **not re-measured** this run. |
| (e) | ungrounded `/explain` | **landed** | `verify_pairing` runs before any generation (`api/main.py:669-680`, defined at `pipeline/cross_domain.py:302`). It raises `UngroundedPairingError` (`:249`), which becomes a 422. The response carries the score, threshold, `grounding` and `is_hypothesis` (`api/main.py:694-700`). `tests/integration/test_explain_grounding.py` has 5 tests, all **skipped** this run. |
| (f) | dishonest `/health` | **landed** | Per-dependency probe returning 503 plus `status="degraded"` (`api/main.py:406-441`), and `graph_available` / `vectors_available` on `/corpus` (`:179-180`). It was **not exercised live** this run, even though the services being down would have been the ideal case to test it. |
| (g) | mocked-only test suite | **partial: there is a new defect pointing the other way** | The integration tier exists: opt-in marker, `addopts = "-m 'not integration'"` (`pyproject.toml:40`), 30 tests across 6 files, and probes that skip rather than fail. **But the unit tier is no longer hermetic.** `score_gaps` calls `_count_contributing_papers` (`pipeline/gap_scorer.py:575`), which opens a real driver (`:912` → `graph/populate.py:30`). The 8 tests listed above stub the old `_count_papers_in_domain`, not the new function, so they query the **production** Neo4j when it is up and fail when it is down. Only 1 reference to `_count_contributing_papers` exists in `tests/test_gap_scorer.py`. The cause is commit `ad34741` (PLAN.md #4), which changed the denominator without updating these stubs. |

### Documentation drift found while verifying (not one of the seven, but it blocks trusting them)

- `PROJECT_HARDENING_PLAN.md`, which CLAUDE.md says to read first, is stale:
  - A3 (`:97`) and A9 (`:178`) still say DEFERRED.
  - A8 (`:168`) still says NEEDS ADVISOR DECISION and cites `extractor.py:388`.
  - H1 (`:507`) says `papers` is "the Neo4j count that divides frequency_score", which has been false since Phase 3.
  - The decision log (`:607`) says "_(No decisions recorded yet)_".

  CLAUDE.md records about ten advisor decisions from 2026-09-28/29 that close or change these items, and none of them reached that file.
- `RUN_REPORT.md:66`, `489 passed`: see (g).

---

## Phase 2 — Plan for the items still open

Items (b) through (f) are landed. The only work proposed for them is the doc fix in P4 and
live re-verification (P5). **Nothing here changes a threshold or a weight.** The two places
that come close are marked **ADVISOR**.

### P1 · (g) The unit tier queries production Neo4j

> **Status: DONE 2026-10-04 (advisor approved P1 only).** Guard in `tests/conftest.py`;
> the 8 tests now stub `_count_contributing_papers`. Measured: before, with Neo4j up,
> 8 passed. With the guard and Neo4j up, 8 failed, all `LiveServiceInUnitTest`. After the
> stubs: `489 passed, 40 deselected, 1 warning` with Neo4j stopped **and** with it running.
> Integration tier with the guard: `37 passed, 3 skipped`. P2–P5 are untouched.
>
> **Correction, 2026-10-05: P1's claim that the unit tier no longer touched Neo4j was false
> for two tests.**
> - The tests are `test_corpus_survives_missing_collections` and
>   `test_corpus_reports_unreachable_vectors_as_unknown_not_zero` in `tests/test_api.py`.
> - They stubbed `_count_papers_in_domain` but not `_count_contributing_papers`, so `/corpus`
>   still built a driver.
> - The P1 guard raised, but `/corpus` swallowed the error in its `except Exception`
>   (`api/main.py:463`) and reported `graph_available=False`.
> - Neither test asserted `graph_available`, so both passed. Before P1, while Neo4j was up,
>   they read the production graph.
>
> Fixed in P1b below.
>
> **Step-7 findings, reported and not fixed:**
> - The unit tier still **opens the real `data/papers.db`**: 62 opens, **56 from
>   `tests/test_api.py` and 6 from `tests/test_rate_limit.py`**. *Corrected 2026-10-05:
>   this originally said "all from `tests/test_api.py`", which was read off a truncated
>   excerpt of the log. The count of 62 was accurate.* The `TestClient` lifespan runs startup self-heal
>   (`api/main.py:271` → `pipeline/selfheal.py:204` → `pipeline/batch.py:57`) because the
>   tests mock the Neo4j driver but leave `SELFHEAL_ON_STARTUP` on and `PAPERS_DB_PATH`
>   unset.
>   - Today it only reads. The mocked driver yields no paper ids, so nothing is missing and
>     nothing is written. `papers.db`'s mtime was unchanged after the run.
>   - Two latent hazards remain. A mock that yields ids would write rows into the real
>     corpus DB. On a fresh checkout with no `data/papers.db`, `sqlite_utils` would
>     *create* it.
>   - The SQLite store is not covered by the new guard.
> - Network: with every non-Unix socket connect blocked and logged for the whole unit run,
>   there were **0** attempts. So no unit test reaches Semantic Scholar, arXiv, the PDF
>   host or Ollama.

- **Problem.** 8 unit tests depend on a live, mutable production database. When Neo4j is up
  they pass while reading whatever the real CV corpus holds. When it is down they fail. Either
  way, "hermetic unit suite" is false, and a reported pass count depends on which services
  happened to be running.
- **Root cause.** Commit `ad34741` introduced `_count_contributing_papers` as the frequency
  denominator in `score_gaps` (`pipeline/gap_scorer.py:575`). The tests still monkeypatch
  `_count_papers_in_domain`, which `score_gaps` no longer calls. There is also no suite-wide
  guard that stops a unit test from opening a real connection, so the leak was silent.
- **Proposed fix.**
  1. In the 8 tests, stub `gs._count_contributing_papers`. Where it matters, keep the
     `_count_papers_in_domain` stub as well, so the tests can tell the two apart.
  2. Add a `tests/conftest.py` autouse fixture that applies to non-`integration` tests only.
     It makes `graph.populate.get_neo4j_driver`, `vectors` Qdrant client construction and
     `load_embedding_model` raise `RuntimeError("unit test touched a real service")` unless
     the test patched them itself. This turns any future leak into a clear error regardless of
     service state. The fixture must not break `tests/integration/conftest.py`'s isolation;
     commit `e4549b9` fixed a leak in exactly this area.
- **Files touched.** `tests/test_gap_scorer.py` and a new `tests/conftest.py`. No production code.
- **Verification (fail-first).** With Neo4j **up**, add the guard alone: the 8 tests must fail
  with the guard's message. That proves the guard detects the leak, because today those tests
  pass in that state. Then add the stubs and confirm `pytest -q` gives 489 passed with Neo4j
  **down** and again with it up. Also confirm `pytest -m integration` is unaffected.
- **Threshold/weight change.** None.
- **Also correct** `RUN_REPORT.md:66` to state the condition it was measured under. That
  file is generated by `scripts/write_run_report.py` from `collect_run_facts.py`, so the fix
  may belong in the collector: record service state next to the pytest counts.

### P1b · Extend the guard to SQLite — DONE 2026-10-05

Measured in this order:

| step | state | result |
|---|---|---|
| 1 | Neo4j and Qdrant up | `489 passed, 40 deselected, 1 warning in 1.05s`; 62 real `papers.db` opens (56 `tests/test_api.py`, 6 `tests/test_rate_limit.py`) |
| 3–4 | SQLite guard that only raises | `489 passed, 40 deselected, 1 warning in 0.98s`. Nothing failed: the lifespan's self-heal `except Exception` (`api/main.py:272`) swallowed the refusal. The guard was wrong. |
| 4 | guard records each refusal and fails at teardown | `489 passed, 40 deselected, 1 warning, 62 errors in 0.89s` |
| 2 | self-heal off **and** `pipeline.batch._DB_PATH` set to `tmp_path` | `489 passed, 40 deselected, 1 warning, 2 errors in 1.02s`. The remaining 2 are the `/corpus` Neo4j leak described under P1. |
| + | `graph_available is True` asserted, stub not yet added | 2 failed (`assert False is True`), plus 2 teardown errors |
| + | `_count_contributing_papers` stubbed | the 2 tests pass |
| 5a | Neo4j, Qdrant and Ollama all **stopped** | `489 passed, 40 deselected, 1 warning in 1.18s` |
| 5b | Neo4j, Qdrant and Ollama all **up** | `489 passed, 40 deselected, 1 warning in 1.17s` |
| 6 | `data/papers.db` mtime | `29 Sep 07:03:40 2026` before and after; unchanged |
| + | file-watch and socket-block plugins, all services up | 0 real `data/` accesses, 0 socket attempts |
| + | `pytest -m integration`, all services up | `37 passed, 3 skipped` (2 fixture has no cross-domain match, 1 no MI multi-member cluster) |

**Broad `except` handlers that wrap a live-store call.** Reported only, not fixed. Each of
these can swallow a guard refusal the same way. The teardown check now catches that in the
unit tier, but the handlers stay. The list comes from an AST walk over every
`except Exception` in non-test code (27 in total). None of the non-test code catches
`RuntimeError` more narrowly.

| handler | wraps | store |
|---|---|---|
| `api/main.py:235` | `_warm` → `factory()` (`load_embedding_model`, `get_qdrant_client`, `get_neo4j_driver`) | model, Qdrant, Neo4j |
| `api/main.py:272` | `reconcile_sqlite_from_graph` | Neo4j, SQLite |
| `api/main.py:362` | `_get_db`, `db.execute` | SQLite |
| `api/main.py:381` | `get_neo4j_driver`, `verify_connectivity`, `session.run` | Neo4j |
| `api/main.py:388` | `driver.close` | Neo4j |
| `api/main.py:399` | `client.get_collection` | Qdrant |
| `api/main.py:418` | `get_qdrant_client` | Qdrant |
| `api/main.py:463` | `_count_papers_in_domain`, `_count_contributing_papers` | Neo4j (**the one that hid the P1 miss**) |
| `api/main.py:487` | `client.count` | Qdrant |
| `api/main.py:494` | `get_qdrant_client` | Qdrant |
| `api/main.py:511` | `_get_db`, `db.execute` | SQLite |
| `api/main.py:627` | `/ingest` body: `extract_paper`, `_get_db`, `get_neo4j_driver`, `embed_*` | SQLite, Neo4j, Qdrant, model |
| `pipeline/batch.py:184` | `extract_paper`, `table.insert` | SQLite |
| `pipeline/cross_domain.py:152` | `extract_paper`, `table.insert`, `driver.session` | SQLite, Neo4j |
| `pipeline/selfheal.py:252` | `table.insert` | SQLite |
| `scripts/ingest_tranche.py:306` | `insert`, `driver.session` | SQLite, Neo4j |
| `scripts/domain_backfill.py:265` | `client.delete_collection` | Qdrant |
| `scripts/filter_backfill.py:153` | `client.delete_collection` | Qdrant |
| `scripts/measure_footprint.py:64` | `client.get_collection` | Qdrant |

Not a live store: `api/main.py:692` (Ollama via `explain_match`), `eval/pretriage.py:59`
(Ollama), `pipeline/extractor.py:345` and `eval/extraction_audit.py:90` (PDF HTTP),
`scripts/ingest_tranche.py:80` (log record), `scripts/ingest_tranche.py:279`
(`extract_paper`: network and LLM), and `scripts/measure_footprint.py:32` and `:42` (OS
calls). No unit test imports `scripts/`. `tests/test_eval_harness.py` imports `eval/`.

### P2 · (a) The corpus-admission rule only guards one of four ingestion paths

> **Status: DONE 2026-10-05.** The advisor chose the **one-validator variant**: no path
> retired, no code path deleted, and no change to what the rule accepts.
> - **`corpus_rules.admit(domain, ids, known=None, allowlist=None)`** is the single
>   validator. It wraps `accept`, which is unchanged.
>   - For ids not in `known`, it looks up the primary category in one batched
>     `arxiv_source.fetch_by_ids` call.
>   - An id arXiv has no record of is rejected.
>   - If arXiv can't be reached, it raises `AdmissionUnverifiable`. It fails closed.
> - **Callers:** `/ingest` (422 on rejection, 503 when arXiv can't be asked),
>   `ingest_from_query`, `ingest_domain_papers`, and `scripts/ingest_tranche.py`. The tranche
>   passes its listing candidate as `known`, so it makes no extra request.
> - **Before any extraction or write:** in every path the check runs first. Rejections are
>   logged and counted (`"rejected"` in the return dicts).
> - **Parser fix found along the way:** `parse_feed` cut old-style ids down to their last
>   path segment (`math/0309136` → `0309136`). A by-id lookup could never find them, so
>   `/ingest` would have rejected them for a parsing reason rather than a rule. It now keeps
>   everything after `/abs/`.
> - **Pre-fix: 24 failed, 491 passed.**
>   - Every ungated-path test failed on behaviour: `extract_paper` was called
>     (`assert 1 == 0`), or the call `DID NOT RAISE`.
>   - The two old-style id parses failed (`'math/0309136' in ['2501.12345', '0309136']`).
>   - The `admit` unit tests failed with `AttributeError`.
>   - The tranche-path control and the new-style-id parse case passed, as intended.
> - **Post-fix: 515 passed.**
>   - 5 existing tests compared the return dict exactly; they gained `"rejected": 0`.
>   - `test_admit_decides_exactly_as_accept_does` checks verdict and reason against `accept`
>     over 9 rule cases.
>   - The unit run made 0 socket attempts and opened no real `data/` store.
>   - Integration tier: 37 passed, 3 skipped.

- **Problem.** All 14 papers removed in the two curation passes got in through
  search-style ingestion. The primary-category rule that would have refused them is only
  applied by `scripts/ingest_tranche.py`. `/ingest` (`api/main.py:598`), `ingest_from_query`
  (`pipeline/batch.py:177`) and `ingest_domain_papers` (`pipeline/cross_domain.py:142`)
  validate that the domain *string* is legal, but not that the paper belongs to it.
- **Root cause.** The rule was added in Phase 3b/3c as part of the new tranche driver rather
  than at the shared `extract_paper` / graph-write boundary.
- **Proposed fix: ADVISOR.** This decides what enters the corpus, which feeds every
  frequency denominator and every null. Under CLAUDE.md's protocol it is not mine to settle.
  The options:
  - **P2-A.** Enforce `corpus_rules.accept` on all paths. The `/ingest` and
    `ingest_from_query` paths currently get metadata from Semantic Scholar, which has no arXiv
    primary category, so this needs one arXiv metadata request per paper (3 s pacing) on
    those paths.
  - **P2-B.** Retire `ingest_from_query` and `ingest_domain_papers` as corpus-building paths
    (CLAUDE.md already says to prefer direct-ID ingestion), and gate only `/ingest`.
  - **P2-C.** Leave the paths ungated, but have `verify_declared_domain`-style logging record
    the primary category, so a violation shows up loudly in the logs. Rejecting paths would
    stay manual.

  Lean: P2-A on `/ingest`, P2-B for the two batch paths.
- **Files touched (P2-A/B).** `pipeline/extractor.py` or a thin wrapper in
  `pipeline/corpus_rules.py`, `api/main.py`, `pipeline/batch.py`, `pipeline/cross_domain.py`,
  and the tests. Under the one-module-per-session rule, this spans layers and must be split
  across sessions.
- **Verification (fail-first).** A unit test per path that feeds a mocked `cs.CL`-primary
  paper (Kosmos-2's real category profile) with `domain="computer_vision"` and asserts it is
  refused before any graph or SQLite write. Against current code it must **fail** on
  `/ingest`, `ingest_from_query` and `ingest_domain_papers`, and **pass** on the tranche path,
  which serves as the control.
- **Threshold/weight change.** None directly. Changing admission changes future corpus
  composition, so the nulls will need the usual re-derivation after the next ingestion.

### P3 · (a) The read endpoints accept any domain string

> **Status: DONE 2026-10-05.** `api/main.py:_known_domain` wraps `validate_domain` and
> returns **422** with `expected one of ['computer_vision', 'medical_imaging']`.
> - **Where it runs:** `/gaps`, `/search`, `/corpus` (`domain`), `/cross-domain` (`source`,
>   `target`) and `/explain` (`source`, `target`; after the demo-mode 403, before grounding).
> - **Pre-fix: 7 failed, 515 passed, 1 error.**
>   - Five cases gave `assert 200 == 422`.
>   - Both `/explain` cases failed with `verify_pairing reached with an unknown domain`.
>   - The 1 error was the guard: before the fix, `/corpus` with a bad domain still reached
>     Neo4j.
> - **Post-fix: 522 passed.**
> - **One existing test changed.** `test_gaps_empty_domain_returns_empty_list` used
>   `domain=unknown_domain` to stand in for "a domain with no papers" and asserted 200, which
>   is exactly the conflation P3 removes. It now uses `medical_imaging` and keeps its stated
>   premise.
> - **Frontend:** not touched. `frontend/lib/api.ts` surfaces a 422 as a generic
>   `API 422: …` error, and the frontend only sends the two known domains.

- **Problem.** `/gaps?domain=computer_visoin` presumably returns `200 []`, which reads as
  "no gaps" rather than "bad request". By project doctrine, an empty result is meaningful
  output, so a typo becomes a false finding. I have **not** observed this live this run;
  services were down. It is inferred from the `Query(default=...)` signatures with no
  `validate_domain` call.
- **Root cause.** Phase 1 (1A) made `domain` validated on writes only.
- **Proposed fix.** Call `validate_domain` on `domain`, `source` and `target` in `/gaps`,
  `/search`, `/cross-domain`, `/corpus` and `/explain`, and return a 422 on failure. Keep
  the existing defaults; this is a contract tightening, not a formula change.
- **Files touched.** `api/main.py`, `tests/test_api.py`. Check `frontend/lib/api.ts` handles
  422 on these routes; it already does for `/explain`.
- **Verification (fail-first).** A `TestClient` test for `GET /gaps?domain=nonsense`
  expecting 422 must fail against current code, which returns 200 because `score_gaps` is
  mocked or empty. The same applies to each of the other endpoints.
- **Threshold/weight change.** None. Note that `_cluster_threshold` and `_solution_threshold`
  fall back to the strictest known value for unknown domains; once validation is in, that
  fallback becomes unreachable from the API.

### P4 · Documentation drift (affects (a), (b), (d) and every future session)

> **Status: DONE 2026-10-05. Docs and one docstring only.**
> - **Status lines updated in `PROJECT_HARDENING_PLAN.md`:** A3 (DONE, Phase 1b), A8 (DONE,
>   Phase 1; it also notes `:388` should have been `:452`), A9 (DONE, Option F) and H1 (a
>   correction: `papers` is no longer the divisor). The original audit text under each is
>   kept and marked superseded.
> - **Decision log:** it now states that CLAUDE.md is the single log and holds a pointer
>   table instead of the false "no decisions recorded yet". CLAUDE.md's header was changed
>   to match.
> - **`score_gaps` docstring** now describes two-tier ordering and the centroid-nearest
>   representative.
>
> **Noted, not changed (outside the four approved items):**
> - CLAUDE.md's header still says `_SOLUTION_THRESHOLD = 0.85` and cross-domain `0.82`
>   "sit below their measured noise floors". Both were replaced on 2026-08-23.
> - `PROJECT_HARDENING_PLAN.md` A5 still says the floor "acts as a binary switch". That
>   stopped being true when Option F made deficits continuous.

- **Problem.** The file CLAUDE.md says to read first disagrees with CLAUDE.md about the
  status of A3, A8, A9 and H1, and says no decisions have been recorded.
  `pipeline/gap_scorer.py:566` describes a representative rule that no longer exists.
- **Root cause.** The 2026-09-28/29 runs recorded decisions in CLAUDE.md and `PLAN.md` but
  did not append them to `PROJECT_HARDENING_PLAN.md`'s decision log, even though CLAUDE.md's
  own header requires both.
- **Proposed fix.**
  - Update the Status lines for A3 (closed by Phase 1b), A8 (closed by Phase 1, 1A + 1C),
    A9 (closed by Option F) and H1 (the divisor is now `papers_reporting_limitations`).
  - Mark A5 as un-deferred: `_UNRESOLVED_DEFICIT_FLOOR` is now load-bearing, per CLAUDE.md
    Phase 2.
  - Append decision-log entries pointing to the CLAUDE.md sections rather than duplicating
    them.
  - Fix the `score_gaps` docstring.
- **Files touched.** `PROJECT_HARDENING_PLAN.md` and the `pipeline/gap_scorer.py` docstring
  only.
- **Verification.** Doc-only, so there is no fail-first test. Verify by reading the statuses
  against the CLAUDE.md phase sections. `git diff` on the `.py` file should be comment-only.
- **Threshold/weight change.** None.

### P5 · Live re-verification of (b)–(f), with no code changes

> **Status: MEASURED 2026-10-05. Report only: nothing was changed and nothing is acted on.**
> All runs used Neo4j, Qdrant and Ollama up, at HEAD `4c4c5ad`. `data/papers.db` mtime
> stayed `29 Sep 07:03:40 2026`, and the working tree was clean after every script.

**1 · Integration tier:** `37 passed, 3 skipped, 522 deselected, 1 warning in 39.07s`. The
skips: 2 because the fixture has no cross-domain match above the noise floor (correct
output), and 1 because the MI fixture has no multi-member cluster. The unit tier was
`522 passed`.

**2 · `scripts/derive_thresholds.py` (no `--apply`):** "All constants within tolerance.
Nothing to do."

| constant | coded | derived | drift |
|---|---:|---:|---:|
| `_CLUSTER_THRESHOLDS["computer_vision"]` | 0.8744 | 0.8744 | 0.0000 |
| `_CLUSTER_THRESHOLDS["medical_imaging"]` | 0.8954 | 0.8959 | 0.0005 |
| `_CROSS_DOMAIN_THRESHOLD` | 0.8764 | 0.8764 | 0.0000 |
| `_DEFICIT_RESCALE_ANCHORS` (4 values) | — | — | 0.0000 each |
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.8733 | 0.8733 | 0.0000 |
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.8915 | 0.8917 | 0.0002 |

Null sample sizes: cluster CV n=6,216 and MI n=5,253; solution CV n=5,264 and MI n=6,386;
cross-domain n=11,785.

**3 · `scripts/measure_weight_influence.py`** (CV 57 gaps, MI 53 gaps):

| term | coefficient | CV by spread / by sd | MI by spread / by sd |
|---|---:|---:|---:|
| frequency | 0.40 | 8.3% / 4.4% | 10.3% / 4.3% |
| recency | 0.35 | 53.5% / 66.4% | 52.8% / 68.5% |
| solution_deficit | 0.25 | 38.2% / 29.2% | 37.0% / 27.2% |

- Deleting the frequency term changes the top-10 order in both domains.
- **Docs drift, not fixed:** the table in CLAUDE.md's *Gap Scoring Formula* section still shows
  the pre-Option-F figures (CV frequency 9.2% / 5.7%, and so on).

**4 · Exact-score ties,** counted directly from `score_gaps(top_n=all)`:
- **CV:** 4 of 57 gaps, in 2 groups. Only 2 of those gaps share a score *and* a tier: 0.4391,
  both single-source. The other group, 0.4949, spans the tier boundary, so the tier orders it.
- **MI:** 3 of 53 gaps in 1 group (0.362, all corroborated).
- This matches RUN_REPORT (4 and 3).
- **Tool defect, not fixed:** `measure_weight_influence.py` prints "2 of 57" and "2 of 53
  gaps share a score". It appears to count adjacent equal pairs while labelling them gaps.

**5 · `/health` with one service stopped.** The API was run on port 8765 with
`SELFHEAL_ON_STARTUP=false`.

| state | `/health` | `/corpus?domain=computer_vision` |
|---|---|---|
| all up | 200 `ok`; papers 149, limitations 215, future_directions 109 | 200; papers 64, papers_reporting_limitations 48, limitations 112, future_directions 47 |
| Qdrant stopped | **503** `degraded`, `qdrant: unreachable`, limitations/future_directions `null` | 200, `vectors_available: false`, vector counts `null` |
| Neo4j stopped | **503** `degraded`, `neo4j: unreachable` | 200, `graph_available: false`, papers/papers_reporting `null` |
| each restarted | 200 `ok` | 200, both available |

With Neo4j down, `/health` still reports `papers: 149` because it counts SQLite. That is
existing item B5, not new.

**6 · The three underived constants. Numbers only, no recommendation.**

- `_CORROBORATED_MIN_PAPERS = 2`:

  | | supporting-paper counts | corroborated / single-source | corroborated at ≥3 instead |
  |---|---|---|---|
  | CV | `{1: 36, 2: 9, 3: 7, 4: 3, 5: 1, 8: 1}` | 21 / 36 | 12 |
  | MI | `{1: 37, 2: 8, 3: 4, 4: 2, 5: 1, 13: 1}` | 16 / 37 | 8 |

- `_MAX_CLUSTER_SHARE = 0.20` (cap floor `_MIN_CLUSTER_CAP = 2`). **The cap does not bind in
  either domain.** No split was logged, and clustering with the cap disabled in memory is
  identical.

  | | limitations | cap | clusters | largest | size distribution |
  |---|---:|---:|---:|---|---|
  | CV | 112 | 23 | 57 | 8 (7.1%) | `{1: 34, 2: 11, 3: 5, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1}` |
  | MI | 103 | 21 | 53 | 16 (15.5%) | `{1: 33, 2: 10, 3: 5, 4: 2, 5: 1, 6: 1, 16: 1}` |

- `_UNRESOLVED_DEFICIT_FLOOR = 0.3`. It is applied in `get_unresolved_gaps`
  (`pipeline/cross_domain.py:196`) to **only the top 20 gaps**, since
  `score_gaps(top_n=20)` is the default there.
  - Deficit distribution: CV has 52 distinct values, 5 at exactly 0.0 and 2 at exactly 1.0.
    MI has 46 distinct values, 7 at 0.0 and none at 1.0.
  - Gaps above each floor:

    | floor | CV all | CV top-20 | MI all | MI top-20 |
    |---:|---:|---:|---:|---:|
    | 0.0 | 52/57 | 16 | 46/53 | 13 |
    | 0.2 | 38 | 8 | 37 | 10 |
    | **0.3** | **32** | **7** | **27** | **7** |
    | 0.4 | 20 | 3 | 19 | 6 |
    | 0.5 | 15 | 3 | 13 | 6 |

  - Cross-domain match count, with the floor patched in memory only:

    | floor | CV→MI | MI→CV |
    |---:|---:|---:|
    | 0.0 | 36 | 22 |
    | 0.2 | 9 | 14 |
    | **0.3** | **9** | **8** |
    | 0.4 | 1 | 0 |
    | 0.5 | 1 | 0 |

### A · Dry-run: would `admit()` accept the papers already in the corpus? (report only)

**The corpus is 149 papers, not 127.** 127 was the 2026-08-23 size, before curation passes 1
and 2 (127 → 116 → 113) and the Phase 3d CV tranche (→ 149). SQLite and Neo4j both hold 149:
64 CV and 85 MI. All 149 were audited.

The audit used SQLite read-only, `corpus_rules.admit` per domain with live arXiv lookups, and
the shipped allowlist. Nothing was written.

| | papers | accepted | rejected |
|---|---:|---:|---:|
| computer_vision | 64 | **64** | 0 |
| medical_imaging | 85 | **81** (6 via the allowlist) | **4** |
| total | 149 | **145** | **4** |

All 4 rejections are MI papers with primary `cs.CV` where the medical keyword verifier did not
fire. **None comes from judgement call 3** (no arXiv record), and none from a failed lookup:

| arXiv id | title | verifier |
|---|---|---|
| 2303.08446 | Task-Specific Fine-Tuning via Variational Information Bottleneck for Weakly-Supervised Pathology… | cv=7 mi=6, not confident |
| 2406.11026 | Boosting Medical Image Classification with Segmentation Foundation Model | cv=7 mi=3, confident CV |
| 2408.08058 | Navigating Data Scarcity using Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning… | cv=6 mi=6, not confident |
| 2501.16469 | Object Detection for Medical Image Analysis: Insights from the RT-DETR Model | cv=7 mi=6, not confident |

`2501.16469` is one of the four papers relabelled CV → MI in curation pass 1.

### B · Can one arXiv timeout abort a whole tranche run? (report only)

**Yes, at candidate discovery. Per-paper extraction failures are isolated.**
- **Discovery is not isolated.** `_gather_candidates` calls `search_category` with no handler
  (`scripts/ingest_tranche.py:188`). `arxiv_source._get` retries `_MAX_ATTEMPTS = 4` times,
  backing off from `_BACKOFF_BASE = 5` s (`pipeline/arxiv_source.py:29-30`, loop at `:69`),
  then raises `RuntimeError` (`:85`).
  - `run()` calls `_gather_candidates` at `:240`, outside the `try` that starts at `:257`.
    `main()` (`:345`) has no handler either, so the process exits non-zero.
  - `scripts/run_tranches.py:144-149` then prints "ingest failed; stopping" and **breaks out of
    the whole multi-tranche loop**.
  - It takes four consecutive failures of one page request; a single transient timeout is
    absorbed by the retries.
  - The resume state survives: the cursor is checkpointed after every page (`:192`).
- **Per-paper extraction failures are isolated.** Extraction retries and then logs and
  continues (`:272-295`). Store failures are logged and the loop continues (`:302-314`).
- **P2 did not change this.** The tranche passes its listing candidate as `known`
  (`scripts/ingest_tranche.py:200`), so `admit()` makes no arXiv request on this path. That
  is pinned by `test_admit_with_known_candidates_makes_no_request`.

None of these could be checked live this run. With Neo4j, Qdrant and Ollama up:

1. `pytest -m integration -q`. `RUN_REPORT.md` claims 37 passed and 3 skipped, which has not
   been verified this run.
2. `python scripts/derive_thresholds.py` without `--apply`, to report drift only. Any constant
   past `DRIFT_TOLERANCE` gets **flagged to the ADVISOR**, not applied in this plan's scope.
3. `python scripts/measure_weight_influence.py` to refresh the (d) table.
4. Exact-score tie counts on `/gaps` for both domains, compared with CLAUDE.md's Option F
   table (CV 0, MI 3).
5. `/health` with one service stopped, expecting 503, `status="degraded"` and a `null` count.

### Flagged for the advisor, not decided here

- **P2:** which ingestion paths the corpus-admission rule must cover.
- **Not touched:** `_UNRESOLVED_DEFICIT_FLOOR = 0.3`, `_MAX_CLUSTER_SHARE = 0.20` and
  `_CORROBORATED_MIN_PAPERS = 2` are not derived from any measurement. CLAUDE.md already
  flags the first as load-bearing since Option F. Any change is an advisor decision.
- **Not touched:** any threshold drift that P5 step 2 reports.

---

## Investigations, 2026-10-05: report only, nothing changed

### INV-1 · Why the MI keyword check rejected four medical papers

**The check.** `corpus_rules.accept` (`pipeline/corpus_rules.py:99-116`) applies it only to a
`cs.CV`-primary paper declared `medical_imaging`. It calls
`classify_text(" ".join((title, abstract)))` (`pipeline/domains.py`) and accepts **only if
`verdict.best == "medical_imaging"`**. For `admit()` the text is arXiv's title and abstract.
- **Fields read:** title and abstract, in full. There is no truncation.
- **Normalisation (`_normalise`):** NFKD accent stripping, lowercasing, whitespace
  collapsing, and padding with one space at each end.
  - Hyphens are not normalised: `"x-ray"` and `"x ray"` are different strings.
  - No stemming beyond the term stems themselves (`"radiolog"`, `"diagnos"`).
- **Matching:** plain **substring** containment. Only `" ct "` is space-delimited.
  - Each term counts **once**, however often it occurs.
  - Weights are summed per domain.
- **Lists:** `_MEDICAL_TERMS` (46 entries) and `_VISION_TERMS` (40 entries), quoted in full
  in `pipeline/domains.py` (the two dicts at the top of the module).
- **Verdict:**
  - `best` is the higher-scoring domain, from a stable sort over
    `RESEARCH_DOMAINS = ("computer_vision", "medical_imaging")`, so **a tie goes to
    computer_vision**.
  - `best` is `None` below `_MIN_EVIDENCE = 3`.
  - The verdict is "confident" if the margin is at least `_MIN_MARGIN = 2`.

**Trace.** Text is arXiv title + abstract, exactly what `admit()` saw.

| arXiv id | medical hits | vision hits | result |
|---|---|---|---|
| 2303.08446 (pathology WSI classification) | `patholog` 3, `whole slide` 3 → **6** | `frame` 1 (from "**frame**work"), `image classification` 3 (inside "whole slide **image classification**"), `imagenet` 3 → **7** | 6 < 7 → best=CV |
| 2406.11026 (Boosting **Medical Image** Classification…) | `medical imag` 3 → **3** (3 occurrences, counted once) | `frame` 1 ("framework"), `image classification` 3 (inside "medical **image classification**", ×3), `natural image` 3 (in a contrast sentence) → **7** | 3 < 7 → best=CV, confident |
| 2408.08058 (…Learning Approaches in **Medical Imaging**) | `clinical` 3, `medical imag` 3 → **6** | `imagenet` 3, `laion` 3 (pretraining sources) → **6** | **tie 6 = 6 → CV by tuple order** |
| 2501.16469 (Object Detection for **Medical Image** Analysis) | `lesion` 3, `medical imag` 3 → **6** | `frame` 1 ("framework"), `map` 1 (the "mAP50" metric), `object detection` 3, `visual` 2 → **7** | 6 < 7 → best=CV |

**Classification: (i), a bug.** Precisely, it's a mismatch between the specification and the
implementation, plus a scoring design biased against medical papers that use CV task
vocabulary. It is not (ii) and not (iii).
- **Spec ≠ code.** The rule is documented in four places as accepting a `cs.CV` paper "if the
  medical keyword verifier fires" / "with a medical keyword match"
  (`pipeline/corpus_rules.py:20`, `:56`, `:217`; CLAUDE.md Phase 3b/3c).
  - The code requires medical to *outscore* vision.
  - Under the documented reading, all four pass: medical evidence is 6, 3, 6 and 6, all
    ≥ `_MIN_EVIDENCE`.
  - The tests (`test_mi_accepts_cs_cv_when_the_keyword_verifier_fires` and
    `test_mi_rejects_cs_cv_when_the_verifier_does_not_fire`) use an all-clinical text and a
    zero-medical text, so they don't tell the two readings apart.
  - Both arrived together in `3ca9814` (2026-09-29).
- **Not (ii).** The medical list did fire on all four. What sinks them is the comparison,
  and these contributing mechanics:
  - Vision *task* terms match inside medical phrases ("medical **image classification**").
  - Presence-only counting: three occurrences of "medical imag" are worth 3.
  - Substring false hits: `frame` in "framework" (in 3 of the 4), `map` in "mAP50".
  - The tie rule hands 2408.08058 to CV for no chosen reason.
- **Not (iii).** All four titles name medical imaging. 2303.08446 is computational pathology.
  2501.16469 was **relabelled CV → MI by human review** in curation pass 1. All four are in
  the human-curated MI corpus.

**The documented reading would not be safe either**, so "make the code match the docs" is
not the fix. Over the 64 `cs.CV`-primary papers in the CV corpus:

| reading | CV-corpus papers that would pass the MI conditional | MI `cs.CV` papers accepted |
|---|---:|---:|
| current (mi > cv, ties → CV) | 6 / 64 | 40 / 44 |
| documented "fires" (mi ≥ 3) | 12 / 64 | 44 / 44 |
| current with whole-word matching | 6 / 64 | 40 / 44 |

"Fires" would admit general CV papers to MI on a single clinical substring: FCOS, SAM-Adapter,
Paxion, TrafficImag, and The Shape of Events. Whole-word matching changes nothing in
aggregate, because ties still go to CV.

**HIGH-PRIORITY question: has this check rejected papers during tranche ingestion?**
**No, zero, because it has never run live.**
- The logs *do* retain rejections. `data/ingest_log.jsonl` (56 entries, 2026-09-29
  01:12–01:33 UTC, gitignored, local only) records every `rule_rejected` entry with its
  reason. All 17 are `computer_vision`, by primary: cs.RO 6, eess.IV 4, cs.CL 3, cs.LG 2,
  eess.AS 1, eess.SP 1.
- **All 56 entries are `computer_vision`.** The CV rule never consults the keyword check.
  `data/ingest_checkpoint.json` holds only a `computer_vision` cursor (cs.CV: 300).
  `data/tranche_progress.jsonl`, which `scripts/run_tranches.py:31` would write, does not
  exist.
- **No medical-imaging tranche has ever run**, so the check had no live effect before P2.
  Since P2 (2026-10-05) it also gates `/ingest`, `ingest_from_query` and
  `ingest_domain_papers` for medical_imaging. None of those has been called on the live
  corpus since.

**Found while investigating (out of INV-1's scope, report only): medical papers inside the CV
corpus.**
- The CV rule (`cs.CV` primary only) cannot separate CV from MI, because 44 of the 85
  curated MI papers also have `cs.CV` primaries.
- Phase 3d's CV tranche admitted clinically medical papers into `computer_vision`. The
  keyword verifier confidently calls each of these medical:

| arXiv id | title | medical / vision |
|---|---|---|
| `2609.30708` | Brain MR image segmentation (multiple sclerosis lesions) | 14 / 1 |
| `2609.31788` | 3D CT report generation | 5 / 0 |
| `2609.30566` | population atlases from diffusion models (brain MRI, chest X-ray) | 14 / 3 |
| `2609.30223` | lesion-aware segmentation loss ("clinically critical lesions") | 8 / 0 |
| `2609.30613` | dermoscopic lesion classification | 6 / 7, which the current rule would *reject* from MI |

`2609.30682` (gigapixel scientific images, with whole-slide and X-ray terms) is borderline.
These papers feed CV gaps, the CV frequency denominator and the CV nulls.

**Proposed fix: not applied, needs an advisor decision.** Any rule should be measured on both
populations above: the 44 MI `cs.CV` papers and the 64 CV `cs.CV` papers.
1. **Stop treating a non-confident verdict as a rejection.** Make the MI conditional
   three-way:
   - accept on a confident medical win;
   - reject on a confident vision win, or when medical evidence is below
     `_MIN_EVIDENCE`;
   - otherwise log `rule_review` for a human or the allowlist.

   Measured: the MI corpus becomes 40 accept, 3 review, 1 reject (2406.11026). The CV corpus
   becomes 52 no-evidence, 4 vision-confident, 3 review, 5 accept.
2. **Stop CV task terms matching inside medical phrases.** For example, don't score
   `image classification` or `object detection` when they're preceded by "medical". This is
   what keeps 2406.11026 out. It would need its own measurement on both populations.
3. **Fix the substring hits** (`frame` and `framework`, `map` and `mAP`, `gan` and
   `organ`) with word boundaries. That's cosmetic in aggregate (measured: no count
   changes) but removes misleading evidence.
4. **Separately, a policy question:** should the CV rule reject `cs.CV` papers the verifier
   confidently calls medical? That's the mirror image of the MI conditional, and it would
   catch the five papers above. It changes what CV admits, so it needs the advisor.
5. **For the four existing papers:** adding them to `data/mi_allowlist.txt` would make
   `admit()` agree with the human curation without touching the rule. That's a data
   decision, also for the advisor.

### INV-2 · Sensitivity of the cross-domain output

Measured with constants patched in memory only (restored and asserted), live stores, HEAD
`0a11a50`. Counts use `find_cross_domain_matches(top_n=1000)`; the API's default `top_n`
is 10. "Unresolved" is the number of the top 20 gaps above the floor.

**`_UNRESOLVED_DEFICIT_FLOOR`** (`_CORROBORATED_MIN_PAPERS = 2`):

| floor | CV→MI | MI→CV | unresolved CV (of top 20) | unresolved MI (of top 20) | Jaccard vs previous step (CV→MI, MI→CV) |
|---:|---:|---:|---:|---:|---|
| 0.15 | 10 | 14 | 9 | 10 | — |
| 0.20 | 9 | 14 | 8 | 10 | 0.90, 1.00 |
| 0.25 | 9 | 13 | 7 | 8 | 1.00, 0.93 |
| **0.30** | **9** | **8** | **7** | **7** | 1.00, **0.62** |
| 0.35 | 9 | 0 | 6 | 6 | 1.00, **0.00** |
| 0.40 | 1 | 0 | 3 | 6 | **0.11**, 1.00 |
| 0.45 | 1 | 0 | 3 | 6 | 1.00, 1.00 |
| 0.50 | 1 | 0 | 3 | 6 | 1.00, 1.00 |

**Is there a principled basis? Partly. The coded 0.3 is arbitrary in origin.**
- **A null-derived value exists.** Since Option F,
  `deficit = 1 − (s − p50)/(p99 − p50)`, where `s` is the nearest eligible future
  direction's similarity. So a floor is a similarity cut in disguise.
  - The floor that means exactly "no eligible future direction clears the solution noise
    floor (null p95)" is `d* = 1 − (p95 − p50)/(p99 − p50)`.
  - From the coded constants: **d\* = 0.3122 for CV and 0.2857 for MI.** The floor acts on
    the *source* domain's deficits, so CV→MI uses CV's value and MI→CV uses MI's.
  - The coded 0.3 corresponds to a nearest-FD similarity below 0.8742 (CV) and 0.8904 (MI),
    against solution thresholds of 0.8733 and 0.8915.
- **0.3 is not derived from this.** It was introduced on 2026-07-03 (`2bb9b58`), when the
  deficit was still a count ratio, so its closeness to d\* is a coincidence.
- **Plateaus:**
  - CV→MI has a stable plateau from 0.20 to 0.35 (the same 9 matches, Jaccard 1.00), with a
    cliff at 0.40. 0.3 sits inside it.
  - **MI→CV has no plateau around 0.3.** It changes at every step from 0.25 to 0.35
    (13 → 8 → 0), and 0.3 sits on that cliff. The MI→CV output is therefore highly sensitive
    to this constant. The MI d\* (0.2857) falls between two measured steps whose counts
    differ by 5.
- **Another underived constant is involved.** `get_unresolved_gaps` applies the floor only
  to `score_gaps(top_n=20)` (`pipeline/cross_domain.py:195`). So `20` also gates the
  cross-domain output, and so does the ranking, including `_CORROBORATED_MIN_PAPERS` (below).

**`_CORROBORATED_MIN_PAPERS`** (floor 0.3):

| min papers | CV→MI | MI→CV | corroborated CV (in top 20) | corroborated MI (in top 20) |
|---:|---:|---:|---|---|
| **2** | **9** | **8** | 21/57 (20) | 16/53 (16) |
| 3 | 8 | 18 | 12/57 (12) | 8/53 (8) |
| 4 | 9 | 19 | 5/57 (5) | 4/53 (4) |

- MI→CV more than doubles at 3 or 4.
- The mechanism is the top-20 coupling: fewer corroborated gaps lets more single-source gaps,
  which have higher deficits, into the top 20 that `get_unresolved_gaps` sees.
- At 2, the CV top 20 is entirely corroborated (20 of 20).
- No null or plateau applies to this constant. It is a count threshold whose effect on
  cross-domain is indirect. There is no measured basis for 2, 3 or 4. (A Wilson-interval
  argument about a one-paper versus multi-paper gap is the rationale recorded under Option F.)

### INV-3.10 · `_MAX_CLUSTER_SHARE` is inactive at the current corpus size

- **Mechanism.** `cap = max(_MIN_CLUSTER_CAP, ceil(0.20 × limitations_in_domain))`, counting
  the seed. A cluster splits only when the seed plus its eligible candidates exceed `cap`
  (`pipeline/gap_scorer.py:353`).
- **Today it doesn't bind, so it is inactive rather than dead code.** No split was logged,
  and clustering with the cap disabled in memory is identical.
  - CV: 112 limitations, cap 23, largest cluster 8 (7.1%).
  - MI: 103 limitations, cap 21, largest cluster 16 (15.5%).
- **When it would begin to bind, at today's cluster structure:**
  - MI (largest cluster 16) binds once `cap ≤ 15`, i.e. **≤ 75 MI limitations**
    (`_cluster_cap(75) = 15`, `_cluster_cap(76) = 16`).
  - CV (largest cluster 8) binds at **≤ 35 CV limitations** (`_cluster_cap(35) = 7`,
    `_cluster_cap(36) = 8`).
- **As the corpus grows, the cap grows with it.** It binds only if a domain's largest
  cluster grows faster than 20% of that domain's limitations: from 15.5% (MI) or 7.1% (CV)
  to above 20%.
- **For comparison:** at the old underived 0.86 threshold, MI's largest cluster was 52 of
  104 (50%). The cap would have split that.

---

## Investigations, 2026-10-05 (2): CV-corpus contamination. Report only, nothing changed

Read-only throughout: SQLite opened `mode=ro`, live Qdrant and Neo4j read, constants patched
in memory only and asserted restored, `derive_thresholds.derive_all()` called unmodified, and
`--apply` never used. `data/papers.db` mtime stayed `29 Sep 07:03:40 2026`; the working tree
stayed clean.

**Premise correction.** The CV solution threshold is **0.8733**, not 0.8773. 0.8773 was its
value from 2026-08-23 until Phase 3d re-derived it on 2026-09-29. The CV cluster threshold is
0.8744.

### INV-4 · How contaminated is the CV corpus?

**Method.** The keyword scores are mechanical: `classify_text` over arXiv title and abstract.
The three-way classification is a **human-judgement reading of each title and full
abstract**, deliberately *not* the keyword verifier, because that is what is under audit.
- **Clearly medical:** the paper's subject is clinical or biomedical imaging (patients,
  lesions, a clinical modality, clinical reports).
- **Ambiguous:** a general method evaluated on medical *and* non-medical data, or
  scientific imaging that includes a medical dataset.
- **Clearly CV:** everything else. Every single-hit medical term in a clearly-CV paper was
  checked in context and is a false hit: "**Retina**Net", "diagnostic benchmarks",
  "orche**st**ration", "su**stain**ability".

**Count:**
- **Clearly medical: 4 of 64 (6.3%).** Including the ambiguous papers: **7 of 64 (10.9%).**
- Clearly CV: 57.
- All 4 clearly-medical papers and 2 of the 3 ambiguous ones are `2609.*`, so they entered
  through the Phase 3d `cs.CV` tranche.

| | arXiv id | title | why |
|---|---|---|---|
| clearly medical | 2609.30708 | Combining General and Domain-Specific Pretext Tasks for Brain MR Image Segmentation | brain MRI, MS lesions, "medical image analysis" |
| clearly medical | 2609.31788 | SelfCue: Making a 3D CT Report Generator Say What It Already Knows | 3D CT radiology report generation (CT-RATE abnormalities) |
| clearly medical | 2609.30613 | MedTokenBudget: Lesion-Preserving Token Routing for Dermoscopic Image Classification | dermoscopy lesion classification |
| clearly medical | 2609.30223 | BiCC: Bidirectional Connected-Component Loss for Instance-Aware Segmentation | lesion-wise segmentation loss, nnU-Net, "clinically critical lesions", computer-assisted review |
| ambiguous | 2609.30566 | Atlases Are Already Inside: Recovering Population Templates from Pretrained Diffusion Models | "applies to multiple domains, such as brain MRI, chest X-ray, faces, and 3D shapes", though the evaluation is mostly medical |
| ambiguous | 2609.30682 | Structure-Guided Masked Autoencoders for Ultra-High Resolution Scientific Image Understanding | electron microscopy and X-ray CT of materials, plus one pathology WSI dataset (PAIP) |
| ambiguous | 2304.09148 | SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More | camouflage and shadow are the main tasks; polyp segmentation is secondary |

**Correction to INV-1.** INV-1 listed 2609.30566 among "clinically medical" papers in the CV
corpus. The full abstract makes it ambiguous, so INV-1's "at least five" is four clearly
medical plus three ambiguous.

**The two readings against the judgement:**

| | (a) documented, med ≥ 3 | (b) implemented, med > vis |
|---|---:|---:|
| verdict "MI" | 12 / 64 | 6 / 64 |
| … of the 4 clearly medical | 4 | 3 (misses 2609.30613 dermoscopy, 6 vs 7) |
| … of the 3 ambiguous | 3 | 2 |
| … of the 57 clearly CV (false MI calls) | 5 (FCOS, Paxion, Shape of Events, TrafficImag, CPSS) | 1 (CPSS: "used as diagnostics") |

**All 64:**

| # | arXiv id | title | med | vis | (a) documented: med ≥ 3 | (b) implemented: med > vis | judgement |
|---:|---|---|---:|---:|---|---|---|
| 1 | 1505.04870 | Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Ri… | 0 | 3 | — | — | clearly CV |
| 2 | 1904.01355 | FCOS: Fully Convolutional One-Stage Object Detection | 3 | 8 | MI | — | clearly CV |
| 3 | 1904.08980 | Exploring the Limitations of Behavior Cloning for Autonomous Driving | 0 | 3 | — | — | clearly CV |
| 4 | 2005.12872 | End-to-End Object Detection with Transformers | 0 | 10 | — | — | clearly CV |
| 5 | 2101.09744 | Classic versus deep learning approaches to address computer vision cha… | 0 | 3 | — | — | clearly CV |
| 6 | 2207.10077 | Discover and Mitigate Unknown Biases with Debiasing Alternate Networks | 0 | 0 | — | — | clearly CV |
| 7 | 2304.02643 | Segment Anything | 0 | 3 | — | — | clearly CV |
| 8 | 2304.09148 | SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underpe… | 3 | 7 | MI | — | **ambiguous** |
| 9 | 2305.10683 | Paxion: Patching Action Knowledge in Video-Language Foundation Models | 3 | 6 | MI | — | clearly CV |
| 10 | 2305.11175 | VisionLLM: Large Language Model is also an Open-Ended Decoder for Visi… | 0 | 8 | — | — | clearly CV |
| 11 | 2307.01952 | SDXL: Improving Latent Diffusion Models for High-Resolution Image Synt… | 0 | 10 | — | — | clearly CV |
| 12 | 2308.12966 | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localiza… | 0 | 9 | — | — | clearly CV |
| 13 | 2310.03744 | Improved Baselines with Visual Instruction Tuning | 0 | 6 | — | — | clearly CV |
| 14 | 2311.07575 | SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for … | 0 | 8 | — | — | clearly CV |
| 15 | 2312.02145 | Repurposing Diffusion-Based Image Generators for Monocular Depth Estim… | 0 | 12 | — | — | clearly CV |
| 16 | 2404.01197 | Getting it Right: Improving Spatial Consistency in Text-to-Image Model… | 0 | 6 | — | — | clearly CV |
| 17 | 2405.14458 | YOLOv10: Real-Time End-to-End Object Detection | 0 | 6 | — | — | clearly CV |
| 18 | 2405.14874 | Open-Vocabulary Object Detectors: Robustness Challenges under Distribu… | 0 | 10 | — | — | clearly CV |
| 19 | 2405.16009 | Streaming Long Video Understanding with Large Language Models | 0 | 6 | — | — | clearly CV |
| 20 | 2407.07726 | PaliGemma: A versatile 3B VLM for transfer | 0 | 3 | — | — | clearly CV |
| 21 | 2408.00714 | SAM 2: Segment Anything in Images and Videos | 0 | 4 | — | — | clearly CV |
| 22 | 2409.13112 | Analyzing mixed construction and demolition waste in material recovery… | 2 | 7 | — | — | clearly CV |
| 23 | 2410.02730 | DivScene: Towards Open-Vocabulary Object Navigation with Large Vision … | 0 | 7 | — | — | clearly CV |
| 24 | 2411.03511 | Beyond Complete Shapes: A Benchmark for Quantitative Evaluation of 3D … | 0 | 4 | — | — | clearly CV |
| 25 | 2412.09082 | Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark a… | 0 | 4 | — | — | clearly CV |
| 26 | 2502.13071 | RobuRCDet: Enhancing Robustness of Radar-Camera Fusion in Bird's Eye V… | 0 | 8 | — | — | clearly CV |
| 27 | 2504.03164 | NuScenes-SpatialQA: A Spatial Understanding and Reasoning Benchmark fo… | 0 | 8 | — | — | clearly CV |
| 28 | 2510.16295 | OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membershi… | 0 | 3 | — | — | clearly CV |
| 29 | 2609.30222 | TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Sce… | 0 | 7 | — | — | clearly CV |
| 30 | 2609.30223 | BiCC: Bidirectional Connected-Component Loss for Instance-Aware Segmen… | 8 | 0 | MI | MI | **clearly medical** |
| 31 | 2609.30234 | OmniFabric: Coherent UV Space Texture Synthesis for 3D Garment Reconst… | 0 | 4 | — | — | clearly CV |
| 32 | 2609.30245 | Towards Practical Compression of 3D Gaussian Splatting | 0 | 1 | — | — | clearly CV |
| 33 | 2609.30393 | LiTe-GS: Oracle-Efficient Next Best View Selection for 3D Gaussian Spl… | 0 | 2 | — | — | clearly CV |
| 34 | 2609.30395 | CSCWD: Cross-Scale Channel-wise Knowledge Distillation for Lightweight… | 0 | 5 | — | — | clearly CV |
| 35 | 2609.30402 | What Improves Multimodal Misinformation Detection? Answers from a Larg… | 0 | 1 | — | — | clearly CV |
| 36 | 2609.30434 | ProCAP: Probabilistic Cross-Attentive Prompt Learning for Vision-Langu… | 0 | 10 | — | — | clearly CV |
| 37 | 2609.30450 | LensDesigner: A Self-Improving Agent for Optical Lens Design | 2 | 1 | — | — | clearly CV |
| 38 | 2609.30478 | The Shape of Events: Edge-Based Inductive Biases via Cross-Domain Dist… | 3 | 9 | MI | — | clearly CV |
| 39 | 2609.30566 | Atlases Are Already Inside: Recovering Population Templates from Pretr… | 14 | 3 | MI | MI | **ambiguous** |
| 40 | 2609.30595 | Action Forcing: Training World Models on Unsupervised Video by Recover… | 0 | 3 | — | — | clearly CV |
| 41 | 2609.30609 | MVAgent: Multi-Agent Video Generation via Consistent Condition Constru… | 2 | 4 | — | — | clearly CV |
| 42 | 2609.30613 | MedTokenBudget: Lesion-Preserving Token Routing for Dermoscopic Image … | 6 | 7 | MI | — | **clearly medical** |
| 43 | 2609.30647 | Conditional Predictive Sufficient Statistics for Visual Representation… | 3 | 2 | MI | MI | clearly CV |
| 44 | 2609.30667 | StarWM: Self-Supervised Trained Attention Routing for Robust World Mod… | 0 | 5 | — | — | clearly CV |
| 45 | 2609.30682 | Structure-Guided Masked Autoencoders for Ultra-High Resolution Scienti… | 9 | 4 | MI | MI | **ambiguous** |
| 46 | 2609.30698 | MM-VeriAgent: Learning to Use Extensive Tools to Verify Multimodal Mis… | 0 | 3 | — | — | clearly CV |
| 47 | 2609.30703 | SAGE: Source-Anchored Guidance via Frequency Equalization for Hierarch… | 0 | 1 | — | — | clearly CV |
| 48 | 2609.30708 | Combining General and Domain-Specific Pretext Tasks for Brain MR Image… | 14 | 1 | MI | MI | **clearly medical** |
| 49 | 2609.30709 | VLALight: Lightweight Vision-Language-Action Models for Emergency-Awar… | 0 | 11 | — | — | clearly CV |
| 50 | 2609.30722 | TrafficImag: A Benchmark for Counterfactual Roadside Traffic Video Gen… | 3 | 4 | MI | — | clearly CV |
| 51 | 2609.30724 | EviDETR: Preserving Query-Relevant Temporal Evidence for Moment Retrie… | 0 | 4 | — | — | clearly CV |
| 52 | 2609.30728 | Learning Polarization Image Restoration with General Restoration Prior… | 0 | 1 | — | — | clearly CV |
| 53 | 2609.30733 | Amplify What You Gaze At: Target Saliency Boosting in Text-to-Image Ge… | 0 | 11 | — | — | clearly CV |
| 54 | 2609.30741 | From Mono to Stereo: Accelerating Binocular Gaussian Splatting via Rep… | 0 | 2 | — | — | clearly CV |
| 55 | 2609.30755 | Training-Free Bottleneck Width Planning for Convolutional Autoencoders | 0 | 2 | — | — | clearly CV |
| 56 | 2609.30758 | LLPR: Location-aware learning and physics-based reconstruction for rai… | 0 | 7 | — | — | clearly CV |
| 57 | 2609.30761 | Timo: $\textbf{T}$aming Mult$\textbf{i}$modal Diffusion Transformer fo… | 0 | 4 | — | — | clearly CV |
| 58 | 2609.30769 | Query-Conditioned Prototype Adaptation for Cross-Domain Few-Shot Learn… | 0 | 5 | — | — | clearly CV |
| 59 | 2609.30783 | Skip the Talk, Re-Focus on Vision: Latent Reasoning for Reasoning Segm… | 0 | 3 | — | — | clearly CV |
| 60 | 2609.31780 | Panoptic Scene Program Diffusion Transformer | 0 | 10 | — | — | clearly CV |
| 61 | 2609.31788 | SelfCue: Making a 3D CT Report Generator Say What It Already Knows | 5 | 0 | MI | MI | **clearly medical** |
| 62 | 2609.32013 | TriO: Tri-Modal Unsupervised Occupancy World Model for Anything Percep… | 0 | 2 | — | — | clearly CV |
| 63 | 2609.32027 | Depth Any Seen: Which Surfaces and How Far? | 0 | 0 | — | — | clearly CV |
| 64 | 2609.32036 | ScreenHaystack: Finding Blind Zones in GUI Grounding | 0 | 0 | — | — | clearly CV |

### INV-5 · What the contamination touches

**Artefacts downstream of CV-corpus composition.** Nothing was re-derived or applied.

| artefact | where | how it depends on the CV corpus |
|---|---|---|
| Paper/Limitation/FutureDirection `domain` | Neo4j; `graph/populate.py` | the stored label itself |
| Qdrant `domain` payload | `limitations` and `future_directions` collections; `vectors/embed.py` | every domain-filtered query |
| `_CLUSTER_THRESHOLDS["computer_vision"]` = 0.8744 | `pipeline/gap_scorer.py:75` | CV limitation × limitation null p95 |
| `_SOLUTION_THRESHOLDS["computer_vision"]` = 0.8733 | `pipeline/gap_scorer.py:126` | CV limitation × CV future-direction null p95 |
| `_DEFICIT_RESCALE_ANCHORS["computer_vision"]` = (0.8235, 0.8959) | `pipeline/gap_scorer.py:159` | p50 and p99 of the same null |
| `_CROSS_DOMAIN_THRESHOLD` = 0.8764 | `pipeline/cross_domain.py:71` | pooled CV-lim × MI-FD and MI-lim × CV-FD null |
| CV frequency denominator (48 contributing papers) | `_count_contributing_papers`, `pipeline/gap_scorer.py` | counts Paper nodes with domain = computer_vision |
| CV cluster cap (23) | `_cluster_cap`, `pipeline/gap_scorer.py:243` | ceil(20% × CV limitation count) |
| CV corpus reference year (recency baseline) | `_corpus_reference_year` | newest CV paper year; the 2609.* papers are 2026 |
| CV gap clusters, representatives, scores, tiers, ranking | `score_gaps`; `/gaps` | membership, and centroid-nearest labels |
| Cross-domain matches, both directions | `find_cross_domain_matches`; `/cross-domain` | CV gaps are CV→MI sources; CV future directions are MI→CV targets |
| `/explain` grounding | `verify_pairing`, `pipeline/cross_domain.py:302` | texts must exist in the declared domain; thresholds above |
| `/corpus`, `/health`, CorpusBanner | `api/main.py` | CV paper, limitation and FD counts |
| MI nulls (counterfactually) | `_SOLUTION_THRESHOLDS["medical_imaging"]` etc. | the 4 medical papers are absent from MI; not measured here |
| README figures | `README.md:21` (149 = 64 CV + 85 MI), `:48-51` (64 / 48 / 112 / 57), `:62` (57 gaps), `:80-83` (CV thresholds) | quoted CV-corpus figures |
| RUN_REPORT | `RUN_REPORT.md:9-` corpus table, `:51-56` CV drift rows, `:98` CV top-15 verbatim, `:198` CV→MI 9 matches verbatim, `:239` MI→CV 8 matches verbatim | as above |
| CLAUDE.md | Phase 3d table (64 CV), *Gap Scoring Formula* influence table (57 CV gaps, measured 2026-10-05) | as above |
| **Evaluation label sheet** | `eval/label_sheet.csv` (60 CV rows, **0 labelled**) | **8 rows cite a clearly-medical paper and 6 more cite only an ambiguous one** |
| Threshold guard | `tests/integration/test_threshold_derivation.py` | measures the production collections |
| `PLAN_AUDIT_FIX.md` P5 / INV-2 numbers | this file | measured on this corpus |

**5 · Dry-run re-derivation without the contaminating papers.** This calls
`derive_thresholds.derive_all()` unmodified, through a read-only Qdrant wrapper that drops
points whose `paper_ids` are *all* excluded. A limitation shared with a CV paper is kept.
The control (no exclusion) reproduces every coded value exactly.

| constant | coded | excluding 4 clearly medical | excluding 4 + 3 ambiguous |
|---|---:|---:|---:|
| CV cluster p95 | 0.8744 (n=6,216) | **0.8745** (n=5,671) | 0.8751 (n=5,253) |
| CV solution p95 | 0.8733 (n=5,264) | **0.8738** (n=4,387) | 0.8738 (n=3,914) |
| CV deficit anchors p50/p99 | (0.8235, 0.8959) | (0.8242, 0.8962) | (0.8239, 0.8962) |
| cross-domain p95 | 0.8764 (n=11,785) | **0.8752** (n=10,857) | 0.8747 (n=10,300) |
| MI cluster / solution p95 | 0.8959 / 0.8917 (derived) | unchanged | unchanged |

**Every shift is inside `DRIFT_TOLERANCE` = 0.002.** The largest is cross-domain at −0.0012,
or −0.0017 with the ambiguous papers. **The thresholds are robust to this contamination.**
The damage is in what gets *clustered, labelled and matched*, not in the noise floors.
Nothing was applied.

**6 · CV gaps with a clearly-medical supporting paper: 4 of 57.** Another 4 have only an
ambiguous paper.

| rank | tier, papers | score | gap description | contaminating paper |
|---:|---|---:|---|---|
| 6 | corroborated, 2 | 0.4138 | The best linear readout of a next-embedding model is not its output. | 2609.31788 |
| **7** | corroborated, 5 | 0.4085 | **Non-dermoscopic generalization remains open because auxiliary datasets…** | 2609.30613, *which also supplies the label* |
| 15 | corroborated, 8 | 0.2771 | Scaling model capacity and incorporating broader pretraining data is l… | 2609.30708 |
| 45 | single source | 0.3774 | What SelfCue writes into the condition is limited to the 18 abnormalit… | 2609.31788 |
| 8 | corroborated, 2 | 0.3922 | Deployment in high-stakes settings may amplify errors or unequal perfo… | ambiguous: 2609.30566 |
| 12 | corroborated, 3 | 0.2996 | Conditioning is required to resolve multiple templates or none | ambiguous: 2609.30566 |
| 31 | single source | 0.4536 | Random masking is poorly matched to the structured, multi-scale morpho… | ambiguous: 2609.30682 |
| 35 | single source | 0.4391 | Uniform tokenization produces prohibitively long sequences… | ambiguous: 2609.30682 |

**Contamination dominates the cross-domain output in both directions:**
- **CV→MI: 7 of 9 matches** (similarities 0.8773–0.8878) start from rank 7. Its label is the
  dermoscopy paper's own sentence, which became a CV gap through centroid-nearest labelling
  of a 5-paper cluster. The other 2 come from genuine CV gaps (crowded-scene tracking, and
  trajectory behaviour labels).
- **MI→CV: 4 of 8 target future directions** come from the medical or ambiguous papers in the
  CV corpus: 2609.30613 (×2), 2609.31788 and 2304.09148.

### INV-6 · Is MI→CV real?

**7 · Counts at the principled floors.** The floor acts on the *source* domain's deficits.

| floor | CV→MI | MI→CV |
|---|---:|---:|
| 0.312 in both directions | 9 | **0** |
| 0.286 in both directions | 9 | 8 |
| per source domain (CV 0.312, MI 0.286) | 9 | 8 |
| coded 0.3 (control) | 9 | 8 |

**8 · MI→CV at the principled MI floor 0.286: 8 matches, all from ONE MI gap.** The gap is
"Simply enlarging convolution kernel sizes doesn't invariably enhance segmentation;
performance …", corroborated by 13 papers, **with deficit 0.3064.**

| sim | CV future direction | from (1 paper each) |
|---:|---|---|
| 0.9181 | extend the SAM-Adapter to tackle even more challenging image segmentation tasks… | 2304.09148 (ambiguous) |
| 0.8998 | Iterative training can be used to add more abnormalities to the condition… | 2609.31788 (**clearly medical**) |
| 0.8937 | Larger mask-annotated evaluations, lesion-size stratification, patient/lesion-grouped splits… | 2609.30613 (**clearly medical**) |
| 0.8932 | Scaling model capacity and incorporating broader pretraining data | 2609.30222 |
| 0.8911 | Remaining controls—a mask-oracle upper bound, fully epoch-matched baseline retraining… | 2609.30613 (**clearly medical**) |
| 0.8898 | Future research includes clearer separation of base models and fine-tunes in VLM research | 2407.07726 |
| 0.8824 | Investigating ways to provide a single stage of equal or better quality | 2307.01952 |
| 0.8816 | Incorporating more explicit motion modeling into SAM 2… | 2408.00714 |

**Plainly: MI→CV does not rest on evidence. It rests on one gap's deficit sitting 0.02 above
the floor, and half its targets are misfiled medical papers.**
- **One source gap.** The whole direction is that gap fanned out to 8 future directions.
  Its deficit, 0.3064, lies between the MI principled floor (0.286) and the CV one (0.312).
  So MI→CV is 8 under the per-source principled floor or the coded 0.3, and **0** under a
  single floor of 0.312.
- **Half the targets are not cross-domain.** 3 of the 8 come from clearly-medical papers
  filed under CV, so those pairings are medical to medical. A 4th comes from an ambiguous
  paper (SAM-Adapter, which itself tests medical segmentation).
- **The rest are generic.** "Scaling model capacity…", "clearer separation of base models…",
  "a single stage of equal or better quality", and "explicit motion modeling into SAM 2".
  Their similarities (0.8816–0.8932) sit 0.005–0.017 above the cross-domain threshold of
  0.8764.

It does survive the principled MI floor. But nothing in it is a CV-specific solution matched
to a clinical gap on more than a single gap's worth of evidence.

**9 · The `top_n=20` in `get_unresolved_gaps`.**
- **Origin:** `pipeline/cross_domain.py:188`, `def get_unresolved_gaps(domain, top_n=20)`.
  It was introduced in `2bb9b58` (2026-07-03) in the same commit as the 0.3 floor, and it
  mirrors `score_gaps`'s own default of 20.
- **Never derived.** There is no comment, decision-log entry or measurement for it, and
  `find_cross_domain_matches` always calls it with the default.

| top_n | CV→MI | MI→CV |
|---:|---:|---:|
| 10 | 9 | 8 |
| **20** | **9** | **8** |
| 30 | 9 | 19 |
| 50 | 10 | 26 |

Beyond rank 20 the ranking is almost entirely single-source gaps: MI has 16 corroborated
gaps, so ranks 17 and below are single-source. So widening `top_n` mostly adds single-source
MI gaps as cross-domain sources.
