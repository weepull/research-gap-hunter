# Research Gap Hunter — Project Hardening Plan

**Audit date:** 2026-08-22 · **Commit audited:** `c1f4c1a` · **Tests at audit:** 234 passing
**Corpus at audit:** 63 papers (46 computer_vision, 17 medical_imaging), 89 limitations, 44 future directions, 27 CV gap clusters

Read-only audit. **No code or data was changed while producing this document.**

> **How to use this file.** Every item has a **Status**. `SAFE TO AUTO-FIX` items are pure
> cleanup with no effect on scoring, ranking or displayed output — an executor may implement
> them directly. `NEEDS ADVISOR DECISION` items change what the system reports and must go
> through the Advisor–Executor Protocol in CLAUDE.md. Append decisions to the log at the bottom.

---

## Executive summary

The pipeline is well-built and genuinely works end to end. The problems are not crashes —
they are **silent correctness issues that make output look more authoritative than it is**.

Three findings dominate, and all three are the *same family* as the two bugs fixed this
session (recency baseline, solution-deficit self-matching): a plausible-looking number that
was never validated against a baseline.

1. **`_SOLUTION_THRESHOLD = 0.85` is below the noise floor** (new finding — not previously
   known). 20% of *random* same-domain pairs clear it in CV, 45% in MI. Solution-deficit is
   25% of the composite score.
2. **Cross-domain `0.82` is far below the noise floor** — 62% of random pairs clear it.
3. **Half the extracted future directions are contentless boilerplate**, and they are
   actively zeroing out real gaps.

Together these mean the **solution-deficit term (0.25 weight) and the entire cross-domain
feature are substantially noise-driven today.** Frequency and recency are sound.

**Hard blocker for a public demo:** Ollama has no managed hosting equivalent, so `/ingest`
and `/explain` cannot work on Railway as planned. That is an architecture decision, not a
config fix.

---

## A. Correctness / scoring integrity

### A1 · `_SOLUTION_THRESHOLD = 0.85` is below the noise floor — BLOCKING
**Status: DONE 2026-08-23** — advisor chose **A1-a** (per-domain p95). Implemented as `_SOLUTION_THRESHOLDS` = {computer_vision: 0.8773, medical_imaging: 0.8987} with a conservative fallback (strictest known floor) for unmeasured domains.

**Current state.** `pipeline/gap_scorer.py:35` treats any future direction scoring ≥ 0.85
against a cluster centroid as "addressing" that limitation. Measured null distribution over
all random same-domain (limitation, future-direction) pairs, using the live stored vectors:

| Domain | n pairs | p50 | p90 | p95 | **% of random pairs ≥ 0.85** |
|---|---:|---:|---:|---:|---:|
| computer_vision | 2,176 | 0.8193 | 0.8652 | 0.8773 | **20.1%** |
| medical_imaging | 250 | 0.8442 | 0.8859 | 0.8987 | **44.8%** |

**Why it matters.** One in five random CV pairs — and nearly half of random MI pairs — is
counted as a *solution*. Every false "addressing" match pushes `solution_deficit_score`
toward 0, demoting a gap that may be entirely unsolved. This is the same defect class as the
self-matching bug fixed this session, but it was never measured because 0.85 *looked*
conservative next to the 0.86 cluster threshold. It is not: those two numbers are measuring
different populations and are not comparable.

**Options.**
- **A1-a — raise to the same-domain p95** (~0.877 CV / ~0.899 MI). Principled and consistent
  with whatever is decided for cross-domain. Will push many gaps toward deficit 1.0, which
  compounds the saturation problem in A9. Per-domain values mean two constants, not one.
- **A1-b — single global threshold at the higher of the two** (~0.90). Simpler, safely above
  both nulls, but over-strict for CV and will likely zero out almost all "addressed" matches.
- **A1-c — keep 0.85 but require corroboration** (e.g. ≥2 independent papers proposing a
  matching direction before deficit drops). Attacks the precision problem without re-deriving
  a threshold; adds a new rule that itself needs justification.

### A2 · Cross-domain threshold `0.82` is below the noise floor — BLOCKING
**Status: DONE 2026-08-23** — advisor chose **A2-a** (pooled p95 = 0.8792). Empty cross-domain output accepted as correct behaviour; verified `/cross-domain` and `/gaps` return a clean `[]` with HTTP 200.

**Current state.** `pipeline/cross_domain.py:176`. Null over all 1,490 random cross-domain
pairs, both directions: mean 0.8267, **median 0.8294**, p95 **0.8792**, max 0.9272.

- **61.6% of random pairs (918/1,490) clear 0.82.** The threshold sits *below the median of
  pure noise*.
- At p95 = 0.8792: **0 of 16** CV→MI matches survive (the default API path); 6 of 83 MI→CV survive.
- 5 of those 6 survivors come from just 2 source gaps, and the top survivor's target is the
  single most promiscuous future direction in the corpus (mean similarity 0.8630 to *all*
  opposite-domain limitations).

**Why it matters.** The cross-domain matcher is the project's headline differentiator and is
currently surfacing pairs indistinguishable from chance, e.g. `0.8543` "Assumes specific
growth rates for the partial sums" ↔ "Propose future research directions in the field".

**Options.**
- **A2-a — adopt p95 (0.8792).** Statistically defensible; CV→MI returns empty. The frontend
  already renders a clean "No connections found" empty state, so this degrades gracefully.
- **A2-b — adopt p90 (0.8693)** for a 10%-noise operating point; still eliminates all current
  CV→MI matches.
- **A2-c — keep a lower threshold but label output** as "below noise floor / exploratory".
  Preserves the demo at the cost of presenting noise as a finding.

### A3 · Contentless future directions corrupt solution-deficit scoring — BLOCKING
**Status: DEFERRED 2026-08-23 (advisor decision).** Not an oversight. The real fix (A3-a) needs re-extraction, which is **blocked by B4** — re-ingesting a paper that already has graph relationships duplicates its limitations. Raising thresholds does not substitute: generic text scores *high*, not low.

This directly answers "does the contentless-future-directions issue affect anything beyond
cross-domain?" — **yes, measurably.**

**Current state.** Of 44 extracted future directions, **22 (50%) are short and non-specific**
("Developing new methods that can generalize beyond the suggested benchmark", "improve the
text-to-mask task with more effort"). 20 of 44 begin with a generic verb; median length is
10 words. Downstream:

- **26 of 79 (33%) surfaced `proposed_solutions` are contentless.**
- **4 gaps are scored fully solved (`solution_deficit_score == 0.0`) where every "solution"
  is boilerplate** — including "Insufficient advanced sensor technologies" and "Limited to
  two-dimensional systems".

**Why it matters.** A vacuous sentence semantically matches almost anything, so it
preferentially wins matches and suppresses real gaps. It also feeds the same promiscuity into
cross-domain. Raising thresholds (A1/A2) does **not** fix this — generic text scores *high*,
not low, because it is topically bland and close to the corpus centroid.

**Options.**
- **A3-a — filter at extraction time.** Tighten `_EXTRACTION_PROMPT` to demand concrete,
  technically specific future directions and drop generic ones. Fixes the root; requires
  re-extraction of the corpus, which collides with the no-detach problem (B4).
- **A3-b — filter at scoring time.** Reject future directions failing a specificity test
  (length, presence of a technical term/acronym) before they count. No re-ingestion; the
  heuristic is arbitrary and needs its own validation.
- **A3-c — down-weight by promiscuity.** Compute each future direction's mean similarity to
  all limitations and discount ones that match everything. Principled and self-calibrating;
  most complex, and adds a corpus-dependent quantity to the scoring path.

### A4 · Extraction tier weights are unvalidated — IMPORTANT
**Status: NEEDS ADVISOR DECISION**

`_TIER_WEIGHTS = {"explicit": 1.0, "conclusion": 0.75, "inferred": 0.5}`
(`pipeline/gap_scorer.py:39`) feeds `frequency_score` directly. The values have no derivation
— they are plausible round numbers. CLAUDE.md already forbids changing them without updating
tests, which has preserved them but never justified them. Note the whole corpus currently
carries `tier="explicit"`, so **the weighting is inert today** and only bites as tiers
diversify. Options: derive empirically (do conclusion-tier limitations actually predict
reported gaps less well?); collapse to binary explicit/other; or document as a deliberate
prior and leave.

### A5 · `_UNRESOLVED_DEFICIT_FLOOR = 0.3` is arbitrary — IMPORTANT
**Status: DEFERRED 2026-08-23 (advisor decision).** A quality refinement, not an integrity risk. With deficits saturated at 0.0/1.0 this floor acts as a binary switch, so re-deriving it changes little until A9 is settled.

`pipeline/cross_domain.py:37` decides which gaps are "genuinely unresolved" and therefore
eligible for cross-domain matching. Undocumented derivation. Interacts with A1/A3: since
deficit scores are heavily saturated at 0.0 and 1.0, this floor is effectively a binary
switch, not a tunable dial.

### A6 · `_MAX_FD_CANDIDATES = 20` silently truncates — IMPORTANT
**Status: NEEDS ADVISOR DECISION**

`pipeline/cross_domain.py:41` caps candidates per gap at 20, but there are 34 CV future
directions. Anything ranked 21+ is never considered. Harmless today because the cut is by
descending similarity, but it is a **silent** cap with no logging, and it will bite as the
corpus grows. Options: raise to cover the collection; make it a function of collection size;
or keep and log what was dropped.

### A7 · `min_cluster_size` parameter is declared but never used — MINOR
**Status: DONE 2026-08-23** — commit `1d080c7 / e3c6324`

`cluster_limitations(limitations, min_cluster_size=2)` (`pipeline/gap_scorer.py:91`) never
references `min_cluster_size` in its body. It implies singletons are filtered; they are not —
13 of 27 clusters are singletons. Actively misleading to a reader.
**Recommended fix:** delete the parameter and update the callers/test stubs that pass it.
(Deleting the *parameter* changes no behaviour; actually *implementing* filtering would change
rankings and would be an advisor decision.)

### A8 · `extract_paper()` hardcodes `domain="computer_vision"` — IMPORTANT
**Status: NEEDS ADVISOR DECISION** (affects data labelling, hence scoring)

`pipeline/extractor.py:388` tags every paper CV regardless of content. Both current callers
patch it afterwards (`cross_domain.ingest_domain_papers`, `api/main.py:/ingest`), so it works
by convention only. Any new call site silently mislabels papers, and domain drives the
frequency denominator, the domain filters, and cross-domain routing. Options: make `domain` a
required parameter; infer it during extraction; or keep the default and add a loud docstring
warning plus a test asserting callers override it.

### A9 · Solution-deficit metric is dimensionally incoherent and saturates — IMPORTANT
**Status: DEFERRED 2026-08-23 (advisor decision).** A quality refinement, not an integrity risk — the term still orders gaps sensibly, it just has poor resolution.

Divides a corpus-wide count of future directions by a cluster-local count of papers, so it is
not a proportion. 17 of 27 clusters sit at exactly 0.0 with only 5 distinct values across the
corpus. Carried forward here so it is not lost.

---

## B. Data integrity

### B1 · `ingest_from_query()` has no pacing between papers — IMPORTANT
**Status: DONE 2026-08-23** — commit `8ccdb00`

`pipeline/batch.py:129-145` loops over papers with **no `time.sleep`**, while
`cross_domain.ingest_domain_papers` correctly sleeps `_FETCH_SLEEP_SECONDS = 2`. CLAUDE.md
mandates the sleep. **This is not hypothetical:** all 6 entries in
`data/failed_extractions.log` are 429s written by this path (the `batch` prefix identifies
it), losing 6 papers.
**Recommended fix:** add the same `time.sleep(2)` per iteration, matching the sibling function.

> Note: the "missing rate limiting" item in the four-session hardening plan was closed by
> adding *inbound* API limiting. This *outbound* gap was never addressed and is arguably what
> the original review meant.

### B2 · Semantic Scholar backoff budget is far too short — IMPORTANT
**Status: DONE 2026-08-23** — commit `8ccdb00`

`extractor.fetch_paper_text` and `batch.search_papers` retry 3 times with `2**attempt`
seconds — waits of 1s then 2s, so they give up **~3 seconds** after the first 429. CLAUDE.md
states the correct remedy is to wait ~60s for the rate-limit window to reset. The log confirms
failures despite requests being ~13s apart.
**Recommended fix:** lengthen the backoff schedule (or honour a `Retry-After` header when
present) so a rate-limited fetch actually survives the window.

### B3 · `ingest_from_query()` will crash on schema drift — IMPORTANT
**Status: DONE 2026-08-23** — commit `8ccdb00`

`pipeline/batch.py:140` calls `table.insert(row, pk="arxiv_id", replace=False)` **without
`alter=True`**, while every other write path passes it. CLAUDE.md documents that new
`PaperExtract` fields require `add_column()` on existing DBs — this is exactly how
`extraction_tier` broke before.
**Recommended fix:** add `alter=True` for consistency with the other two insert sites.

### B4 · Re-ingesting a paper accumulates relationships — IMPORTANT
**Status: NEEDS ADVISOR DECISION** (deleting graph data affects scoring inputs)

`graph/populate.py:_upsert_paper_counting` only ever `MERGE`s and never removes
relationships; `Limitation` nodes are keyed on exact `text`. Because LLM extraction is
non-deterministic, re-extracting a paper creates *new* Limitation nodes while the old ones
stay attached, so the paper reports both wordings. Discovered this session; documented in
CLAUDE.md but **the code is unchanged**, so the trap is still live. This currently blocks any
re-extraction remedy, including A3-a.

### B5 · `/health` and the scorer count papers from different stores — MINOR
**Status: NEEDS ADVISOR DECISION** (changes a displayed number)

`/health` reports `papers` from **SQLite**; `_count_papers_in_domain()` — the frequency
denominator — counts **Neo4j** Paper nodes. They agree right now (63/63) only because the
drift was just repaired. A future divergence would leave `/health` reporting "ok" while
scoring used a different corpus size. Options: report both counts; report the Neo4j count
since that is what scoring uses; or add an explicit consistency check that fails loudly.

### B6 · Neo4j/SQLite paper drift — RESOLVED 2026-08-22
Fixed in `c1f4c1a`; root cause and prevention documented in CLAUDE.md. Listed for completeness.

### B7 · `populate_graph()` leaks the driver on exception — MINOR
**Status: DONE 2026-08-23** — commit `1d080c7`
`graph/populate.py:134-154` calls `driver.close()` only on the success path.
**Recommended fix:** wrap in `try/finally`.

---

## C. Test coverage gaps

### C1 · ~17 tests validate dead code — IMPORTANT
**Status: DONE 2026-08-23** — commit `1d080c7`

Two clusters of tests exercise functions **production never calls**:

| Dead function | Tests | Production actually uses |
|---|---:|---|
| `extractor._extract_section` | 9 | `_select_section_from_pages` |
| `populate.upsert_paper` / `upsert_limitation` / `upsert_future_direction` / `upsert_method` / `upsert_dataset` | 8 | `_upsert_paper_counting` |

**Why it matters.** This is the failure mode caught earlier this session — tests that pass
regardless of whether the shipped code works — in its most literal form. The suite's headline
number overstates real coverage. Worse, `_extract_section` searches for headings in a joined
full-text string, which is **precisely the anti-pattern CLAUDE.md forbids** ("Never search for
section headings in joined full-text string"), and it sits there fully tested and ready for a
new contributor to wire back in.
**Recommended fix:** delete both the dead functions and their tests. Behaviour is unchanged by
construction — nothing calls them.

### C2 · The real graph-write path is barely tested — IMPORTANT
**Status: DONE 2026-08-23** — commit `1d080c7`

`_upsert_paper_counting` — which creates every Paper, Limitation, FutureDirection, Method and
Dataset in production — has **one** direct test (tier tagging), while its unused twin has
eight. Coverage is inverted.
**Recommended fix:** port the eight `upsert_*` tests onto `_upsert_paper_counting` as part of
the C1 deletion, so total coverage rises rather than falls.

### C3 · No test would catch B1/B2/B3 — MINOR
**Status: DONE 2026-08-23** — commit `8ccdb00`
No test asserts pacing between ingestion iterations, backoff duration, or `alter=True`. Add
assertions when fixing them, and confirm each fails against current code first.

### C4 · Frontend has no tests — MINOR
**Status: OPEN — accepted, documented 2026-08-23.** Standing up a JS test runner is new infrastructure rather than a bounded fix, so it was not improvised. Stated explicitly here so a reviewer does not have to discover it: `frontend/` has no test runner configured and zero tests.
No test runner configured in `frontend/`. Acceptable for an MVP; worth stating explicitly
rather than leaving a reviewer to discover it.

---

## D. Code quality / maintainability

**All DONE 2026-08-23** — commits `1d080c7` (D1, D4, D5, D6), `e3c6324` (D2, D3), `59d2bab` (D7). None touched scoring or output.

| # | Issue | Location | Recommended fix |
|---|---|---|---|
| D1 | Graph upsert logic duplicated between the five `upsert_*` helpers and `_upsert_paper_counting`; fixing one would not fix the other | `graph/populate.py:41-118` vs `158-252` | Delete the helpers (see C1/C2) |
| D2 | Two different `_log_failure` functions with **different signatures** — `extractor` takes `(id, reason, raw)`, `batch` takes `(id, reason)` | `extractor.py:343`, `batch.py:39` | Consolidate into one, or rename so the difference is visible at the call site |
| D3 | `logging.basicConfig(level=INFO)` called at **import time** in two library modules — hijacks root logging for anything importing them, including the API and tests | `extractor.py:102`, `batch.py:29` | Remove; configure logging in the app entrypoint only |
| D4 | `populate_graph(batch_size=50)` — parameter never used | `graph/populate.py:121` | Delete the parameter |
| D5 | Cypher `CASE WHEN p.arxiv_id = $arxiv_id THEN 1 ELSE 0 END AS created` is always 1 and the result is discarded | `graph/populate.py:170` | Delete the expression |
| D6 | `_extract_section` dead code implementing a documented anti-pattern | `extractor.py:176-207` | Delete (see C1) |
| D7 | `frontend/AGENTS.md` warns that this Next.js version differs from training data, but root `CLAUDE.md` never mentions it | `frontend/AGENTS.md` | Cross-reference it from root CLAUDE.md |

---

## E. Security

### E1 · `/ingest` is unauthenticated — BLOCKING for public deploy
**Status: DONE 2026-08-23** — commit `9b17a45`. `/ingest` returns 403 in demo mode with a message stating the policy and how to ingest locally. Not an auth layer: authenticated ingestion remains planned, and is listed as such in the README roadmap.

`api/main.py:/ingest` accepts an arXiv ID from anyone and triggers a Semantic Scholar fetch, a
PDF download, a 30–60s LLM run, database writes, and a **full re-embed of both Qdrant
collections**. The new rate limiter caps it at 3/min per IP, which bounds but does not prevent
abuse: corpus poisoning, quota exhaustion of your Semantic Scholar key, and trivially
sustained CPU load. Options: require an API key/admin token; disable `/ingest` entirely in a
read-only demo deployment; or move ingestion to an out-of-band job.

### E2 · Error responses leak internal details — BLOCKING for public deploy
**Status: DONE 2026-08-23** — commit `b74d130`

`api/main.py` returns `HTTPException(status_code=500, detail=str(exc))` in both `/ingest` and
`/explain`. `str(exc)` on a driver or HTTP error routinely contains connection URIs, file
paths, and library internals — and the frontend surfaces the raw body in its error string
(`frontend/lib/api.ts:65`).
**Recommended fix:** log the exception server-side, return a generic message and a correlation
id to the client.

### E3 · No arXiv ID validation → path injection into outbound URLs — IMPORTANT
**Status: DONE 2026-08-23** — commit `b74d130`

`arxiv_id` flows unvalidated from the request body/path into f-string URLs:
`f"{_ARXIV_PDF_BASE}/{arxiv_id}"` and `f"{_SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{arxiv_id}"`.
A crafted value containing `../` or a query string can redirect the fetch to other paths on
those hosts. The host is fixed, so this is not full SSRF, but it is unvalidated user input
reaching an outbound request.
**Recommended fix:** validate against the arXiv ID format (`^\d{4}\.\d{4,5}(v\d+)?$` plus the
older `archive/YYMMNNN` form) and reject otherwise, in the Pydantic model and the path param.

### E4 · CORS is fully open — IMPORTANT
**Status: DONE 2026-08-23** — commit `9b17a45`. CORS open locally, restricted to `ALLOWED_ORIGINS` in demo mode, and fails **closed** (no origins) if demo mode is on but `ALLOWED_ORIGINS` is unset. Supersedes the earlier "withheld by operator" note.
`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` (`api/main.py:110-115`).
Fine locally; should be restricted to the Vercel origin before public deploy.

### E5 · Unbounded PDF download into memory — MINOR
**Status: DONE 2026-08-23** — commit `b74d130`
`extractor.fetch_full_text` reads `response.content` with no size cap. A very large PDF
inflates memory. **Recommended fix:** stream with a max-bytes guard.

### E6 · What is actually clean — no action
Verified good: every Cypher query and every SQLite query is **parameterised** (no injection
found); `.env` is gitignored and **not tracked by git**; no credentials or keys appear in
committed files; the rate limiter correctly ignores spoofable `X-Forwarded-For`.

---

## F. Documentation accuracy

**F1, F3–F7 DONE 2026-08-23** in commit `59d2bab`. F2 remains advisor-owned but is now flagged in the README as disputed with the null-distribution numbers, so it no longer reads as settled fact.

| # | Claim | Reality |
|---|---|---|
| F1 | `README.md:109` — "176 tests, 0 failures" | **234 tests.** Stale since the frontend commit |
| F2 | `README.md:276` — "0.82 yields 10 meaningful matches vs. 1 at 0.84" | **Refuted by A2.** 62% of *random* pairs clear 0.82; those matches are not meaningful. **NEEDS ADVISOR DECISION** — rewrite together with the A2 threshold decision |
| F3 | `README.md:272` — Neo4j justified by "papers citing the same dataset" and "adding citation graphs" | **No `CITES` relationships exist** in the graph (verified: 0). The justification cites unimplemented capability |
| F4 | `CLAUDE.md` "Extraction Prompt — always use this exact prompt structure. Do not modify without updating this file" | The real prompt in `extractor.py:56-74` has **two extra instruction lines** plus per-tier guidance not in CLAUDE.md. The rule was broken without updating the doc |
| F5 | `CLAUDE.md` `PaperExtract` model | Missing the `extraction_tier` field the code has |
| F6 | `README.md:326-331` — "Phase 5 / Phase 9 / Phase 10" | CLAUDE.md explicitly removed phase language as stale; README still uses it. The two docs contradict each other |
| F7 | `README.md:110` — "Python 3.11 Required" | Venv runs **3.14.4**; `pyproject.toml` says `>=3.11`. "Required" overstates it |

---

## G. Deployment readiness (local → Render / Vercel / AuraDB / Qdrant Cloud)

> Target changed from Railway to Render on 2026-08-24: Render's free tier is permanent
> and needs no card. Items written before that date name Railway; the analysis is
> unchanged, since every blocker was about ephemeral disks, managed Qdrant and build
> configuration rather than about Railway specifically.

### G1 · Ollama has no managed equivalent — HARD BLOCKER
**Status: DONE 2026-08-23 (revised)** — commits `578c782` then `REVISION`. Resolved by *removing* the dependency rather than replacing it: `DEMO_MODE=true` refuses both `/ingest` and `/explain` with a 403, so a public deployment never needs Ollama. An interim hosted-LLM path (claude-opus-5) was implemented and then deleted at the advisor's direction — a per-request paid API call for anonymous callers is an open-ended cost. `explain_match()` now has a single backend (Ollama) and a test asserts no hosted branch is re-added.

`/ingest` (extraction) and `/explain` (hypothesis explanation) both hard-depend on a local
`llama3.1:8b` at `OLLAMA_BASE_URL`. Railway offers no Ollama service, and an 8B model needs
GPU or a large always-on instance. **The planned deployment cannot run these two endpoints
as written.** Options: swap to a hosted inference API for deployment only (changes the "local,
free" property in the README and the stack table); ship a read-only demo with `/ingest` and
`/explain` disabled; or self-host Ollama on a GPU box and point the deployed API at it.

### G2 · `get_qdrant_client()` cannot reach Qdrant Cloud — BLOCKER
**Status: DONE 2026-08-23** — commit `2cbf248`
`vectors/embed.py:25-29` constructs `QdrantClient(host=..., port=...)` only. Qdrant Cloud
requires an HTTPS URL **and an API key**, neither of which is plumbed through.
**Recommended fix:** support `QDRANT_URL` + `QDRANT_API_KEY` env vars, falling back to
host/port for local use.

### G3 · SQLite on an ephemeral container filesystem — BLOCKER
**Status: DONE 2026-08-24** — commit `4b37611` (advisor decision: implement both layers)

`data/papers.db` was a repo-relative file. Railway container disks are ephemeral, so every
redeploy would silently reset the corpus — the exact failure that caused the drift repaired in
B6, recurring on every deploy.

The advisor chose **both** offered remedies rather than either alone, on the reasoning that a
volume is the correct fix but its failure mode is silent, and a rebuild-on-boot is a correct
safety net but a poor primary because it loses fields the graph never stored.

**Layer 1 — persistence.** `PAPERS_DB_PATH` overrides the SQLite location, so the file can live
on a mounted volume. Resolved once at import (after `load_dotenv()`) so tests that patch
`pipeline.batch._DB_PATH` are unaffected; a blank value falls back to the default rather than
resolving to path `""`. `_get_db()` creates the parent directory, so a freshly mounted empty
volume needs no provisioning step.

**Layer 2 — startup self-heal.** `pipeline/selfheal.py`. The API lifespan compares Neo4j Paper
nodes to SQLite rows and rebuilds any rows SQLite is missing, using the reconstruction path
proven in `c1f4c1a`: read `title`/`year`/`domain` from the Paper node and
limitations/future-directions/methods/datasets from its relationships. **No LLM call, no network
call, deterministic.**

Why reconstruction and not re-ingestion — this is the same constraint recorded for B6 and it has
not gone away: `graph/populate.py:_upsert_paper_counting()` only ever MERGEs and never removes
relationships, and `Limitation` nodes are keyed on exact `text`. Ollama extraction is
non-deterministic, so re-extracting a paper that already has limitations attaches *new*
Limitation nodes while the old ones stay, leaving that paper reporting both wordings and
inflating the corpus with near-duplicates that then corrupt clustering.

Deliberate properties:

- **Additive only.** Existing rows are never overwritten — an extracted row is richer than
  anything the graph can rebuild. Rows SQLite has that the graph does not are counted and
  reported (`sqlite_only`) but never deleted.
- **Loud when it fires.** A rebuild logs at WARNING naming the volume as the likely culprit.
  Reaching that branch means layer 1 did not work, which is worth investigating even though the
  app recovers. The no-op path logs at INFO.
- **Never fatal.** A failure is caught and logged; the API still serves traffic. A broken safety
  net is better than an API that will not boot.
- **Kill switch.** `SELFHEAL_ON_STARTUP=false` disables it without a code change.
- **Known lossy fields.** Neo4j never stored `objectives`, `evaluation_metrics` or `raw_json`, so
  those come back empty. `raw_json` records `{"source": "reconstructed_from_neo4j", ...}` so a
  rebuilt row is visibly reconstructed rather than looking like a failed extraction.

**Verification.** 25 new tests (297 → 322) pin both directions — a real mismatch repaired with
correct field mapping, `json.dumps`'d list fields, tier recovered from the `REPORTS_LIMITATION`
relationship and provenance recorded; and matching stores producing byte-identical rows with no
writes. Live against the real stack the no-op path reported 127/127. The repair path was
exercised by deleting 5 rows from a **copy** of `papers.db` and reconciling: all 5 were rebuilt
with identical `title`, `year`, `domain` and identical limitation sets. The real database was
never opened for writing.

**Scoring impact: none.** `_count_papers_in_domain()` counts Neo4j Paper nodes, which this never
writes to. As established in B6, repairing SQLite cannot move the frequency denominator.

### G4 · Specter2 loads at startup — IMPORTANT
**Status: STILL NEEDS ADVISOR DECISION** (affects cost/instance sizing)

> **Flag, 2026-08-24 — read this before assuming G4 is settled.** The
> `Dockerfile` added in commit `57611a5` bakes the Specter2 weights into the
> image, which is *one of the three options listed below*. That was a build
> artifact decision taken while preparing deployment, **not** an advisor
> decision on this item, and it is deliberately recorded here rather than left
> for someone to discover in the Dockerfile.
>
> What it does settle: cold boot is a disk read rather than a 440MB download.
> What it does **not** settle: the memory footprint at runtime (still ~440MB
> resident before the app serves traffic) or the instance size that implies —
> which is the part this item is actually about. The image cost is ~421MB of
> its 2.85GB.
>
> Reversing it is one `RUN` instruction: delete the pre-cache block and the
> weights download at first boot instead. Nothing else depends on it.
The lifespan handler calls `load_embedding_model()`, downloading/loading ~440MB of weights
before the app serves traffic. Expect slow cold starts and real memory pressure on a small
instance; `HF_HUB_OFFLINE` is explicitly discouraged by CLAUDE.md, so the download path stays
live. Options: bake weights into the image; lazy-load on first use and accept a slow first
request; or move embedding to a separate service.

### G5 · Rate limiter state is per-process — IMPORTANT
**Status: OPEN — needs a deployment choice.** Both remedies (pin to one worker, or move buckets to Redis) are deploy-time decisions, not code changes; Redis would be new infrastructure. Already documented in CLAUDE.md and in `api/rate_limit.py`.
In-memory token buckets do not coordinate across workers, so limits multiply by worker count.
**Recommended fix:** pin to one worker for the demo, or move buckets to Redis.

### G6 · `NEXT_PUBLIC_API_URL` is inlined at build time — IMPORTANT
**Status: DONE 2026-08-23** — commit `271657f`
`frontend/lib/api.ts:6` defaults to `http://localhost:8000`. Next.js inlines `NEXT_PUBLIC_*`
at **build** time, so if it is not set in the Vercel build environment the deployed site will
call localhost and fail with no server-side error.
**Recommended fix:** set it in Vercel and fail the build loudly when it is missing.

### G7 · Frontend does not handle 429 — MINOR
**Status: DONE 2026-08-23** — commit `271657f`
No 429 handling anywhere in the frontend; a rate-limited user sees a raw `API 429: {...}`
string. `/explain` is capped at 10/min and has a visible "Generate Explanation" button, so
this is reachable by ordinary clicking. **Recommended fix:** detect 429, read `Retry-After`,
show a friendly message.

### G8 · Neo4j URI scheme default — MINOR
**Status: DONE 2026-08-23** — commit `59d2bab`
Default is `bolt://localhost:7687`; AuraDB requires `neo4j+s://`. Env-only change, but worth
documenting in the README deploy section so it is not discovered at deploy time.

---

## H. Corpus / scale honesty

### H1 · Nothing anywhere states the corpus size — BLOCKING for public demo
**Status: DONE 2026-08-23** — commit `8f1c8fe` (new `/corpus` endpoint) + `c11d17c` (CorpusBanner on all three results pages). Banner states domain paper count, limitation/future-direction counts and last-updated date; `papers` is the Neo4j count that divides frequency_score, pinned by a test.

The UI presents `0.6065` in a green "good" badge with three sub-score bars, in the visual
language of an authoritative metric. Nothing on the page says the ranking is computed over
**46 CV papers**. A reviewer who assumes a large corpus will read the numbers very
differently from one who knows the denominator. Options: show corpus size and date on every
results page; add a persistent "research preview — N papers" banner; or gate the public demo
behind a larger corpus.

### H2 · Frequency scores are structurally tiny and the colour scale hides it — IMPORTANT
**Status: DONE 2026-08-23** — commit `c11d17c`. Frequency bar is now scaled to the largest frequency on screen and relabelled "Frequency (relative)", with the true score shown to 3 decimals. It no longer reads as empty/broken beside recency and deficit.
`frequency_score = weighted_papers / 46`, so real values run ~0.01–0.06 and the frequency bar
is always visually empty, while the composite score can still show green because recency and
deficit dominate. The displayed sub-scores are not on comparable scales, which is misleading
even though each is individually correct.

### H3 · Rankings are dominated by single-paper clusters — IMPORTANT
**Status: DONE 2026-08-23** — commit `c11d17c`. Every gap carries a SupportBadge directly under its title; single-paper gaps get an amber ⚠ variant and an explanatory tooltip. Nothing is filtered — showing the evidence is the fix. Note the underlying shape is unchanged: 13/27 CV and 17/24 MI gaps are still single-source, they are now just impossible to miss.
**13 of 27 clusters are singletons.** The #2 and #3 ranked gaps ("Assumes specific growth
rates for the partial sums", "Assumes a specific micro-gravity environment") are each backed
by **one paper**, and neither is a computer-vision research gap in any meaningful sense —
they are artefacts of one paper's phrasing. Presenting these as "the most urgent open problems
in the domain" is the single most reputationally risky thing in the project. Options: require
≥2 supporting papers to surface; show support count prominently next to each gap; or keep
singletons but label them "single-source".

### H4 · README overclaims scope — IMPORTANT
**Status: DONE 2026-08-23** — commit `c11d17c`. README states the real 127-paper sample, frames the project as a working prototype rather than a survey, and warns that many gaps are single-source. Supersedes the earlier "withheld by operator" note.
`README.md:18` — "ranks research gaps … the most urgent, underserved open problems in the
domain" and "over the entire corpus". At 63 papers, "the domain" is not covered.
**Recommended fix:** state the corpus size and frame as a working prototype over a curated
sample.

### H5 · The medical-imaging future-direction pool is 10 items — IMPORTANT
**Status: NEEDS ADVISOR DECISION**
Cross-domain CV→MI matches against **10** future directions total. No threshold choice can
make a 10-item target pool statistically meaningful; this is a corpus problem, not a
parameter problem, and it bounds what A2 can achieve.

---

## Suggested sequencing

**Stage 0 — free wins, no decisions needed** (do first; nothing here can change rankings)
B1, B2, B3, B7 (ingestion robustness) · C1, C2, D1–D7 (delete dead code, port its tests) ·
E2, E3, E5 (error leakage, input validation) · F1, F3, F4, F5, F6, F7 (doc corrections) ·
A7 (unused parameter).
*Rationale: shrinks the surface area and makes the coverage number honest before any scoring
work begins.*

**Stage 1 — the scoring-integrity block** (single advisor session; decide together)
A1, A2, A3 are one interlocking decision — thresholds and extraction quality trade against
each other, and fixing thresholds alone will not remove boilerplate matches. A9 (saturation)
and A5 (deficit floor) should be settled in the same sitting since they consume the same
numbers. **B4 must be resolved first if A3-a (re-extraction) is chosen**, because
re-extraction is currently unsafe.

**Stage 2 — honesty before exposure** (hard blocker for public demo)
H1, H2, H3, H4. None of these require the Stage 1 decisions to land; they are about not
overstating what the numbers mean. **H3 is the highest-risk item in the audit** — a technical
reviewer will notice single-paper gaps immediately.

**Stage 3 — deployment**
G1 first: it is an architecture decision that may reshape everything else. Then G2, G3, G6
(config/persistence blockers), then E1, E4 (public-exposure security), then G4, G5, G7, G8.

*Status as of 2026-08-24: G1, G2, G3, G6 are done. G4, G5, G7, G8 and E1/E4 remain.*

**Stage 4 — deferred**
A4 (tier weights, inert until tiers diversify), A6, C4, H5 (needs corpus growth, not code).

### Hard blockers for the "public demo" goal
1. **G1 — Ollama hosting.** `/explain` and `/ingest` cannot run as designed.
2. **H3 + H1 — single-paper gaps presented without corpus context.** The credibility risk.
3. **E1 + E2 — unauthenticated write endpoint and leaking errors.**
4. ~~**G3 — ephemeral SQLite** silently resetting the corpus on redeploy.~~ **Resolved 2026-08-24** (`4b37611`) — persistent volume plus startup self-heal.
5. **A2/A1 — thresholds below the noise floor.** Shipping a discovery tool whose headline
   feature surfaces chance pairings is the worst outcome of everything in this document.

### Explicitly *not* blockers
A4, A6, C3, C4, G4, G7, G8, and every item in section D. These are quality-of-life and can
follow the demo.

---

## Decision log

> Append one entry per advisor decision, newest last. Format mirrors the advisor-decision
> entries in CLAUDE.md: what was decided, why, measured impact, and what was deliberately
> deferred. Once an entry lands here, update the corresponding item's **Status** above and
> record the rationale in CLAUDE.md so it is not re-litigated next session.

```
### <ITEM-ID> · <short title> — decided <YYYY-MM-DD> (advisor: <name>)

**Decision:** <option chosen>
**Rationale:** <why, including numbers that drove it>
**Measured impact:** <ranking delta / test delta / what was verified live>
**Deferred:** <anything explicitly not settled by this decision>
```

_(No decisions recorded yet — audit delivered 2026-08-22.)_

---

## Session summary — 2026-08-23 (autonomous safe-fix pass)

**Branch:** `hardening/safe-auto-fixes` · 8 commits · **not pushed**, not merged
**Base:** `c1f4c1a` (= `main`, = `origin/main`) · **Tests: 234 → 255 passing**
**Scope honoured:** only items marked SAFE TO AUTO-FIX. No threshold, weight, formula
or ranking-affecting change. Nothing pushed. No data deleted.

### Completed

| Commit | Items | What changed |
|---|---|---|
| `3493d30` | — | The audit itself (this file) + CLAUDE.md pointer |
| `1d080c7` | C1, C2, D1, D4, D5, D6, B7 | Deleted dead code; ported its tests onto the real write path |
| `8ccdb00` | B1, B2, B3, C3 | Ingestion pacing, 60s retry budget, `alter=True` |
| `b74d130` | E2, E3, E5 | Error leakage, arXiv id validation, PDF size cap |
| `e3c6324` | D2, D3, A7 | One failure logger, no library logging config, dead param |
| `2cbf248` | G2 | Managed Qdrant via `QDRANT_URL` / `QDRANT_API_KEY` |
| `271657f` | G6, G7 | Fail build on missing API URL; typed 429 handling |
| `59d2bab` | F1, F3–F7, D7, G8 | Documentation corrected to match the code |

**Test count 234 → 255.** Net of removing 17 dead-code tests and adding 38 real ones.
Coverage of `_upsert_paper_counting`, the sole production graph-write path, went from
**1 test to 10**. Every new behavioural test was confirmed to fail against the pre-fix
code before being committed.

### Three things worth knowing

1. **A latent bug was fixed that the audit had not found.** Consolidating the duplicate
   `_log_failure` (D2) initially broke log redirection: `batch`'s tests redirect by
   patching that module's `_LOG_PATH`, and the new delegate resolved the *extractor's*
   path instead — so the suite wrote to the real `data/failed_extractions.log`. Caught by
   an existing test, fixed with an explicit `log_path` parameter, and verified by asserting
   the real log's line count is unchanged across a full run.
   **Residue: three stray `arxiv_id=bad-id` entries were appended to
   `data/failed_extractions.log` (lines 13–15) before the fix.** They were left in place
   rather than deleted, per the no-deletion constraint. Safe to remove by hand — they are
   test artefacts, not real ingestion failures.
2. **Some tests were making real network calls.** The new malformed-arXiv-id cases reached
   the live handler and hit Semantic Scholar and arxiv.org — one run took 97 seconds. They
   now stub the extraction path, so the suite cannot touch the network even if validation
   regresses. An autouse fixture also stops the suite really sleeping through the new
   ingestion pacing (~20s per run otherwise).
3. **Two items marked SAFE in this plan were deliberately skipped**, because the operator's
   instruction excluded them and an explicit instruction outranks this file's status field:
   **E4** (open CORS, excluded with E1) and **H4** (README scope wording, excluded with the
   H1–H5 block). Both are now marked withheld above rather than left looking available.

Not implemented, and why: **C4** (frontend tests) — standing up a JS test runner is new
infrastructure, not a bounded fix; documented instead. **G5** (per-process rate limiter) —
both remedies are deployment choices rather than code changes.

### What is next — advisor decisions, in priority order

**1 · The scoring-integrity block (A1 + A2 + A3) — decide together.**
These interlock: thresholds and extraction quality trade against each other, and raising
thresholds will *not* remove boilerplate matches, because generic text scores high. Settle
A5 and A9 in the same sitting since they consume the same numbers. **If A3-a
(re-extraction) is chosen, B4 must be fixed first** — re-ingesting a paper that already has
graph relationships still duplicates its limitations.

**2 · Corpus honesty (H1, H2, H3, H4) — blocks a public demo independently.**
H3 is the highest-risk item in the whole audit: 13 of 27 clusters are single-paper, and the
#2 and #3 ranked "research gaps" are one-paper artefacts that are not CV research gaps in
any meaningful sense. A technical reviewer will notice immediately.

**3 · Deployment (G1 first).** Ollama has no managed equivalent, so `/ingest` and
`/explain` cannot run on Railway as designed — an architecture decision that may reshape
G3 (ephemeral SQLite) and G4 (Specter2 startup cost). G2, G6, G7, G8 are already done.

**4 · Public-exposure security (E1, E4).** `/ingest` is an unauthenticated write endpoint
that burns the Semantic Scholar quota; CORS is fully open.

**5 · Lower priority.** A4 (tier weights — inert today, the whole corpus is `explicit`),
A6, A8, B5, C4, G5, H5 (needs corpus growth, not code).

### Verification state

`uv run pytest -q` → **255 passed**. Local Qdrant reachable after the client change (both
collections listed). `npx tsc --noEmit` clean. No live-stack scoring run was performed
this session because nothing touching scoring was modified — gap rankings are unchanged by
construction.

---

## Session summary — 2026-08-23 (scoring integrity + corpus growth)

**Branch:** `hardening/scoring-integrity-a1-a2` · **not pushed** · base `a6718c1`
**Tests: 255 → 263** · **Corpus: 63 → 127 papers**

### Phase 1 — A1/A2 implemented, A3/A5/A9 deferred (commit `ef7991d`)

Both thresholds moved to the 95th percentile of their measured null distribution.
All 8 new tests were confirmed to fail against the old values first.

| | Was | Now |
|---|---|---|
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.85 | 0.8773 |
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.85 | 0.8987 |
| cross-domain `similarity_threshold` | 0.82 | 0.8792 |

**One result differed from what the decision anticipated.** CV→MI returned **1**
match (0.8829), not zero: raising the A1 threshold left more gaps unresolved, so
more of them entered cross-domain matching and one cleared the floor. The A1 and
A2 changes interact — worth remembering when reasoning about either alone. Empty
handling was verified independently: an unknown domain returns a clean `[]` with
HTTP 200 from both `/cross-domain` and `/gaps`.

A3, A5 and A9 were documented as deliberate deferrals in CLAUDE.md rather than
fixed, so a future session does not mistake them for oversights.

### Phase 2 — corpus grown from 63 to 127 papers

**Composition: 46 computer_vision, 81 medical_imaging** (was 46/17). Years:
2023 → 44, 2024 → 48, 2025 → 28, 2026 → 1, plus 6 older.
Neo4j: 127 Paper, 166 Limitation, 94 FutureDirection, 353 Method, 197 Dataset.
Qdrant: 168 limitation vectors, 94 future-direction vectors. **Store drift: zero
in both directions.**

**The medical-imaging future-direction pool went from 10 to 60** — the specific
bottleneck behind empty cross-domain output.

Sourcing, per the no-recalled-IDs rule: arXiv's export API rate-limited hard
after an initial burst, so candidates came from the project's own Semantic
Scholar search (8 topic queries → 131 unique arXiv IDs). **Every candidate was
then verified individually against `arxiv.org/abs/<id>`** — confirming the id
resolves, that its live `citation_title` matches what search returned (≥60% token
overlap, to catch an id pointing at a different paper), and that the title is
topically medical imaging. 113 passed; 16 rejected as off-topic, 2 as duplicates.
The 2023+ subset (65) was queued, since older papers frequently have no
limitations section. 64 ingested, 1 lost to a Semantic Scholar 429 that outlasted
even the new 60s retry budget (`2305.00678`).

Ingestion ran at ~26s/paper through `ingest_domain_papers`, honouring the B1/B2
pacing and backoff. Those fixes proved themselves in production: the run
recovered from several 429s that would previously have dropped the papers.

### Threshold drift — diagnostic only, no action taken

Same read-only method, re-run against the enlarged corpus:

| Threshold | Shipped | New p95 | Drift | Null n | Verdict |
|---|---|---|---|---|---|
| A1 computer_vision | 0.8773 | 0.8773 | +0.0000 | 2,176 | **HOLDS** |
| A1 medical_imaging | 0.8987 | 0.8916 | **−0.0071** | 6,240 (was 250) | **MINOR DRIFT** |
| A2 pooled cross-domain | 0.8792 | 0.8789 | −0.0003 | 7,376 (was 1,490) | **HOLDS** |

Two things worth the advisor's attention:

1. **The MI floor moved because the original estimate was thin, not because the
   data changed character.** It was derived from **250** pairs (25 limitations ×
   10 future directions); it now rests on **6,240**. The shipped 0.8987 is
   therefore slightly *conservative* — it admits 3.0% of random pairs where the
   target is 5%. That errs safe (under-counting solutions, so gaps look more open
   than they are) and needs no urgent change, but 0.8916 is the better-supported
   value. **Adjusting it is a new advisor decision, not implemented.**
2. **A2 barely moved despite the null growing 5×.** 0.8792 → 0.8789 across 7,376
   pairs is strong evidence the cross-domain floor is a real property of the
   embedding space in this corpus rather than an artefact of the small sample.

### Live effect of the larger corpus

**CV→MI went from 1 match to 21** (range 0.8821–0.9039), MI→CV from 6 to 13
(0.8794–0.9181) — all still above the unchanged noise floor. Growing the
future-direction pool, not loosening the threshold, is what produced them, and
the pairings are substantively related rather than boilerplate, e.g. "Previous
methods require labels of protected attributes" ↔ "Apply proposed method to
semi-supervised learning (pseudo labels)". This is the clearest evidence so far
that the cross-domain feature works when the corpus can support it.

Medical-imaging gaps rose from 10 to 24. **Singleton clusters remain the dominant
shape — 13/27 CV and 17/24 MI — so H3 is unchanged and still the top demo
blocker.**

### Still open — advisor decisions

1. **H1–H4 corpus honesty** — now the highest priority. H3 (single-paper gaps
   presented as ranked research gaps) is untouched and the corpus growth did not
   dilute it.
2. **A3 contentless future directions** — still blocked by **B4** (re-ingestion
   duplicates limitations rather than replacing them). Worth re-measuring: the
   boilerplate ratio on the new 94-item pool has not been recharacterised.
3. **MI threshold refinement** — adopt 0.8916, or leave the conservative 0.8987.
4. **G1 Ollama hosting** (done), then ~~G3~~ (done 2026-08-24, `4b37611`) / **G4**; **E1/E4** public-exposure security.
5. **A4, A6, A8, B5, C4, G5, H5** — lower priority. Note **A8** now matters more:
   `extract_paper()` hardcodes `domain="computer_vision"` and the medical-imaging
   papers are correct only because `ingest_domain_papers` patches it afterwards.

### Not done, deliberately

No threshold was adjusted in response to the drift finding — that is the
advisor's call. Nothing was pushed. No data deleted. No other
NEEDS-ADVISOR-DECISION item was touched.

---

## Session summary — 2026-08-23 (corpus honesty, H1–H4)

**Branch:** `hardening/corpus-honesty-h1-h4` · **not pushed** · base `f26acc6`
**Tests: 263 → 267** · commits `8f1c8fe` (API), `c11d17c` (frontend + README)

All four items implemented per the advisor decision. Nothing else was touched.

**H1** — new `GET /corpus?domain=` returns paper / limitation / future-direction
counts and a last-updated timestamp; `CorpusBanner` renders it on the gaps,
search and cross-domain pages. `papers` deliberately comes from
`_count_papers_in_domain` (the Neo4j count that divides `frequency_score`) rather
than the SQLite total `/health` reports, so the number beside a score is the one
the score was computed against. A test pins that source.

**H2** — the frequency bar is scaled to the largest frequency on screen and
relabelled "Frequency (relative)"; the numeric readout shows the true score at 3
decimals. On an absolute scale real values (0.01–0.06) rendered as an empty bar
beside recency and deficit at 1.0, which read as broken rather than as low.

**H3** — `SupportBadge` sits directly under every gap title, amber with a ⚠ for
single-paper gaps. Nothing is filtered.

**H4** — README scope note added; "most urgent, underserved open problems in the
domain" and "the entire corpus" are gone.

### Verification

`/corpus` live: CV 46 papers / 64 limitations / 34 future directions (updated
Aug 22), MI 81 / 104 / 60 (updated Aug 23). `npx tsc --noEmit` clean; production
build compiles; the new strings and the `/corpus` call are present in the built
bundle (3 chunks — one per page using the banner).

No browser tooling was available this session, so instead of a screenshot the
rendering was confirmed by driving live API data through the component logic. Top
four CV gaps render "⚠ Supported by 1 paper"; the top MI gap renders "Supported
by 3 papers"; frequency bars differentiate (0.022 full-width vs 0.009 at ~45%).

### What this does and does not fix

It fixes the *presentation* problem: a reader can no longer see a top-ranked gap
without also seeing that one paper backs it and that the corpus is 46 papers.

It does not change the underlying distribution. **13/27 CV and 17/24 MI gaps are
still single-source**, and single-paper gaps still occupy the top CV ranks because
recency and solution-deficit both saturate at 1.0 for them. Whether ranking should
account for support count is a scoring question (adjacent to A9) and was not
touched.

### Still open

Unchanged from the previous summary: **A3** (blocked by **B4**), the optional MI
threshold refinement to 0.8916, ~~**G3**~~ (done 2026-08-24), **G4**, **E1/E4**
public-exposure security, and **A4, A6, A8, B5, C4, G5, H5**. **H5** (medical-imaging
future-direction pool) is materially improved — 10 → 60 — but remains formally open.

---

## Session summary — 2026-08-24 (G3, deployment preparation)

**Branch:** `main` · **not pushed** · base `8b1363c`
**Tests: 297 → 322** · commits `4b37611`, `4a49715`, `57611a5`, `cd5e2e9`

### G3 — resolved

Implemented as two layers per the advisor decision. Full detail is in the G3
entry above; the short version is `PAPERS_DB_PATH` for a mounted volume, plus
`pipeline/selfheal.py` rebuilding any rows SQLite is missing from graph
relationships on startup. Additive only, never fatal, loud when it fires.

Verified live rather than only in unit tests: the no-op path reported 127/127;
deleting 5 rows from a **copy** of `papers.db` rebuilt all 5 with identical
title, year, domain and limitation sets; and a simulated fresh volume rebuilt
all 127 rows with `/health` reporting them. The real database was never written
to in any of these.

### Deployment preparation — done, nothing deployed

`Dockerfile`, `.dockerignore`, `render.yaml`, `frontend/vercel.json`,
`.env.example`, `frontend/.env.example`, `DEPLOYMENT.md`,
`scripts/prune_hf_cache.py`.

No accounts were created, no credentials obtained, nothing deployed. Every step
requiring a real credential is written up in `DEPLOYMENT.md` for the owner.

Three things worth knowing:

- **A latent logging bug was fixed.** `pipeline/extractor.py` states that a
  library module must not call `basicConfig` and that "the app entrypoint owns
  that" — but no entrypoint ever did. Every logger under `pipeline/`, `graph/`
  and `vectors/` was emitting into a handler-less root logger while uvicorn
  configured only its own. The startup self-heal was therefore completely
  silent, including the WARNING that is its entire reason for existing. Found by
  reading a running server's log, not the code.
- **Image size: 3.7GB → 2.85GB.** CPU-only torch (the default PyPI wheel pulls
  the whole CUDA runtime onto a GPU-less container) and pruning the duplicate
  model revision the HuggingFace pre-cache fetches. The prune must share a layer
  with the download; doing it in a following instruction left the image at
  exactly 3.7GB.
- **G4 is *not* resolved** despite the Dockerfile baking in weights. See the flag
  on that item.

### Deliberately not done

`A4`, `A6`, `A8`, `B5` and `H5` were all offered as optional cleanup but every
one of them is marked **NEEDS ADVISOR DECISION**, which was explicitly out of
scope for this session. `C4` (frontend tests) is not advisor-gated but is
already recorded here as new infrastructure rather than a bounded fix. Nothing
was improvised.
