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
