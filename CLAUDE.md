# Research Gap Hunter — CLAUDE.md

> **READ `PROJECT_HARDENING_PLAN.md` FIRST — before writing code or giving advice.**
> It is the full 2026-08-22 audit and the single source of truth for known open issues,
> their severity, and which ones are blocked on an advisor decision. This applies to
> **every future session and every advisor reviewing the project.**
>
> Read it first because several things that look wrong here are already diagnosed there
> (with measured numbers), and several things that look *fine* are known to be unsound —
> most importantly, **`_SOLUTION_THRESHOLD = 0.85` and the cross-domain `0.82` both sit
> below their measured noise floors**, so solution-deficit and cross-domain output are
> substantially noise-driven right now. Do not propose changes to scoring, thresholds,
> clustering, or displayed results without checking whether that item already has an entry
> and a status there. When a decision is made, append it to that file's decision log **and**
> record the rationale here.

## What This Project Is

Research Gap Hunter is an AI-powered scientific discovery platform. It does NOT function as a search engine or retrieval tool. Its purpose is to answer "what should be done next?" — not "what has been done?"

The system ingests academic papers, extracts structured information using a local LLM, builds a knowledge graph, embeds limitation statements as vectors, and runs a discovery engine that surfaces ranked research gaps and cross-domain hypothesis matches.

MVP domain: Computer Vision, extended to Computer Vision ↔ Medical Imaging for cross-domain matching.

---

## Architecture Overview

```
Paper (arXiv ID / PDF)
        ↓
[Pipeline: extractor.py]
Ollama llama3.1:8b → structured JSON
        ↓
[SQLite] raw storage (pipeline/db.py)
        ↓
[Graph: graph/populate.py]
Neo4j rgh-mvp → nodes + relationships
        ↓
[Vectors: vectors/embed.py]
Specter2 embeddings → Qdrant collection: "limitations"
        ↓
[Discovery: pipeline/gap_scorer.py]
Gap scoring + seed-anchored union-find clustering → ranked GapResult list
        ↓
[Cross-domain: pipeline/cross_domain.py]
CV ↔ Medical Imaging structural matching
        ↓
[API: api/main.py]
FastAPI REST layer
        ↓
[Frontend: frontend/]
Next.js + Tailwind
```

---

## Stack — Never Change These Without Asking

| Layer | Tool | Notes |
|---|---|---|
| Extraction LLM | `ollama` / `llama3.1:8b` | Local, free, runs on M5 |
| Graph DB | Neo4j Desktop, instance: `rgh-mvp` | Bolt: `bolt://localhost:7687` |
| Vector Store | Qdrant | `http://localhost:6333`, collection: `limitations` |
| Embeddings | `sentence-transformers` / `allenai/specter2_base` | Paper-level + limitation-level |
| Paper source | Semantic Scholar API | Base URL: `https://api.semanticscholar.org/graph/v1` |
| Raw storage | SQLite via `sqlite-utils` | File: `data/papers.db` |
| API | FastAPI | Port 8000 |
| Frontend | Next.js + Tailwind CSS | In `frontend/` |
| Tests | pytest | In `tests/` |

---

## Frontend Agent Instructions

`frontend/` has its own `AGENTS.md` (surfaced as `frontend/CLAUDE.md`) warning
that the installed Next.js version has breaking changes versus training data,
and that the guides in `node_modules/next/dist/docs/` must be read before
writing frontend code. Read it before touching anything under `frontend/`.

---

## Folder Structure

```
research-gap-hunter/
├── CLAUDE.md                  ← this file
├── .env                       ← secrets, never commit
├── .gitignore
├── pyproject.toml
├── data/
│   └── papers.db              ← SQLite database
├── pipeline/
│   ├── extractor.py           ← extract_paper(arxiv_id) → PaperExtract
│   ├── batch.py               ← batch ingestion runner + SQLite layer
│   ├── config.py              ← DEMO_MODE / CORS flags
│   ├── selfheal.py            ← startup Neo4j → SQLite reconciliation (G3)
│   ├── gap_scorer.py          ← score_gaps() → ranked GapResult list
│   └── cross_domain.py        ← find_cross_domain_matches()
├── graph/
│   └── populate.py            ← SQLite → Neo4j population
├── vectors/
│   ├── embed.py               ← embed_limitations() → Qdrant upsert
│   └── search.py              ← find_similar_limitations(query)
├── api/
│   └── main.py                ← FastAPI app
├── frontend/                  ← Next.js app
└── tests/
    ├── test_extractor.py
    ├── test_graph.py
    ├── test_vectors.py
    └── test_gap_scorer.py
```

---

## Environment Variables (.env)

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# Deployment-only. See .env.example and DEPLOYMENT.md for the full set.
PAPERS_DB_PATH=data/papers.db     # point at a mounted volume in production
SELFHEAL_ON_STARTUP=true          # kill switch for the G3 safety net
```

---

## Core Data Models (Pydantic)

### PaperExtract
```python
class PaperExtract(BaseModel):
    arxiv_id: str
    title: str
    year: int
    domain: str = "computer_vision"
    objectives: list[str]        # what the paper sets out to do
    methods: list[str]           # algorithms / architectures used
    datasets: list[str]          # datasets used for evaluation
    evaluation_metrics: list[str]
    limitations: list[str]       # MOST IMPORTANT — explicit limitation statements
    future_directions: list[str] # what the authors suggest as next steps
    raw_json: str                # full LLM output stored as blob
    ingested_at: str             # ISO timestamp
    extraction_tier: str = "explicit"  # "explicit" | "conclusion" | "inferred"
```

`extraction_tier` records where the limitations text came from and drives
`_TIER_WEIGHTS` in gap scoring. It was added after this model was first written
and was missing from this file until 2026-08-23.

### GapResult
```python
class GapResult(BaseModel):
    gap_description: str
    score: float                 # weighted composite 0–1
    frequency_score: float       # how many papers report it
    recency_score: float         # ratio of last-2yr papers vs all-time
    solution_deficit_score: float # how few future_directions address it
    supporting_papers: list[str] # arxiv_ids
    proposed_solutions: list[str]
```

### CrossDomainMatch
```python
class CrossDomainMatch(BaseModel):
    source_gap: str              # unresolved limitation in source domain
    target_solution: str         # future_direction from target domain
    similarity_score: float      # cosine similarity; cross-domain default 0.82 (lower than cluster threshold 0.86 — different field vocabularies compress cross-domain scores)
    source_papers: list[str]
    target_papers: list[str]
    source_domain: str
    target_domain: str
```

---

## Neo4j Graph Schema

### Nodes
- `Paper` — arxiv_id, title, year, domain
- `Limitation` — text, domain, cluster_id
- `FutureDirection` — text, domain
- `Method` — name
- `Dataset` — name

### Relationships
- `(Paper)-[:REPORTS_LIMITATION]->(Limitation)`
- `(Paper)-[:SUGGESTS_FUTURE]->(FutureDirection)`
- `(Paper)-[:USES_METHOD]->(Method)`
- `(Paper)-[:USES_DATASET]->(Dataset)`
- `(Paper)-[:CITES]->(Paper)`

---

## Qdrant Collections

### `limitations`
- Vector size: 768 (Specter2 output)
- Distance: Cosine
- Payload fields: `paper_id`, `limitation_text`, `year`, `domain`, `cluster_id`

### `future_directions`
- Same structure as limitations
- Used for cross-domain matching

---

## Gap Scoring Formula

```
score = (0.40 × frequency_score) + (0.35 × recency_score) + (0.25 × solution_deficit_score)
```

- `frequency_score` = tier-weighted papers_reporting_limitation /
  **papers_in_domain_reporting_any_limitation** (`_count_contributing_papers`).
  Not `total_papers_in_domain` — changed 2026-09-28, see Phase 3 below.
- `recency_score` = papers_last_2yr_reporting / papers_all_time_reporting
- `solution_deficit_score` = 1 - addressedness, where
  `addressedness = clamp((nearest_similarity - null_p50) / (null_p99 - null_p50), 0, 1)`
  over the *eligible* future directions (same-domain **and** not authored by a paper
  that reports the limitation). **Changed 2026-09-29 (A9 Option F)** from the previous
  `1 - (count_addressing / papers_reporting)`, which was dimensionally incoherent and
  saturated to four distinct values per domain. See Phase 2 below.

> ### The coefficients are not the influence — read this before citing "40/35/25"
>
> 0.40/0.35/0.25 are the coefficients. They are **not** how much each term moves
> the ranking, and the gap is large enough that quoting the weights as if they
> were is misleading:
>
> | term | coefficient | measured influence (CV / MI) |
> |---|---:|---:|
> | frequency | 0.40 | **9.2% / 10.0%** (by spread) · 5.7% / 3.7% (by sd) |
> | recency | 0.35 | 52.9% / 52.5% · 51.9% / 60.9% |
> | solution_deficit | 0.25 | 37.8% / 37.5% · 42.5% / 35.4% |
>
> The reason is range, not weighting: recency and deficit both span the full
> [0, 1] on real data, while frequency spans ~0.04–0.19 (CV) because at this
> corpus size a gap is reported by one to a handful of papers out of a few dozen.
> **Deleting the frequency term entirely leaves the top 10 of both domains
> unchanged.**
>
> This is documented rather than normalised away, by advisor decision (PLAN.md #4
> rejected option 4B): rescaling terms to make coefficients match influence would
> make every score relative to whatever else is in the result set, so adding one
> paper would move every number with no change in evidence — destroying the
> cross-run comparability the recency baseline below exists to guarantee.
>
> Refresh these figures with `python scripts/measure_weight_influence.py`.
> Do not "fix" the discrepancy by reweighting; there has been no advisor decision
> to change the coefficients.

Seed-anchored clustering groups similar limitation statements before scoring, at a
**per-domain threshold derived from the measured null distribution**, with a
structural cap on cluster size (HDBSCAN was the original design but was replaced —
do not reintroduce it without an explicit advisor decision). The cluster
representative used as `gap_description` is the member **nearest the cluster's
vector centroid**. Both of these changed on 2026-09-28; see Phases 2 and 4.

### Recency baseline — decided 2026-08-20

`recency_score` counts papers with `year >= baseline - 1`. The baseline is the
**newest publication year present in the corpus**, computed once per `score_gaps()`
call by `_corpus_reference_year()` — *not* the wall-clock year, and *not* per
cluster. Rationale:

- **The term must keep discriminating.** Ingestion lags publication and the lag
  grows as a corpus sits. On the current corpus (newest CV paper 2025, wall clock
  2026) a wall-clock baseline leaves only 6 of 64 limitations with any recency
  signal, mean 0.094 — the 0.35-weighted term is effectively dead and separates
  nothing. Corpus-anchored gives 28 of 64, mean 0.438.
- **Scoring must be deterministic.** `score_gaps()` is a ranking function; the
  same corpus has to rank the same way whenever it runs. A wall-clock baseline
  makes rankings drift with no data change and makes a stored `GapResult`
  incomparable to a freshly computed one.
- **Corpus-wide, not per-cluster.** A cluster's own newest paper always falls
  inside its own window, so per-cluster baselines would score an all-2019 cluster
  as recent as an all-2025 one.
- Clamped to the current year so a single future-dated paper (bad metadata)
  cannot push the window past every real paper. For a sane corpus the clamp never
  binds, so determinism holds.

Do not reintroduce a hardcoded year. `tests/test_gap_scorer.py` guards both
failure modes (frozen constant, and wall-clock baseline); the guards were
verified to fail against each regression before being committed.

### Similarity thresholds — decided 2026-08-23 (advisor: Fable 5)

Both similarity thresholds were re-derived from **measured null distributions** —
the score that random, unrelated pairs reach by chance, computed over the live
corpus using the stored Specter2 vectors. Below a null's 95th percentile a
"match" is statistically indistinguishable from a random pairing.

| Constant | Was | Now | Null p95 | % of random pairs the OLD value admitted |
|---|---|---|---|---|
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.85 | **0.8773** | 0.8773 (n=2,176) | 20.1% |
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.85 | **0.8987** | 0.8987 (n=250) | 44.8% |
| `find_cross_domain_matches(similarity_threshold=)` | 0.82 | **0.8792** | 0.8792 (n=1,490) | 61.6% |

Two things this corrects:

- **0.85 was never comparable to the 0.86 cluster threshold.** They measure
  different populations (limitation↔future-direction vs limitation↔limitation),
  so 0.85 only *looked* conservative.
- **The old cross-domain rationale was backwards.** It argued that vocabulary
  divergence compresses cross-domain scores, so the threshold should sit *below*
  the within-domain one. Compression raises the noise floor as well as the
  signal, so a compressed space needs a **higher** bar. 0.82 sat below the median
  of pure noise (0.8294).

An **empty cross-domain result is correct output, not a bug** — it means no pair
in the data clears the noise floor. `/cross-domain` and `/gaps` both return a
clean `[]` with HTTP 200, and the frontend renders "No connections found".

**These values are corpus-dependent.** Re-derive the nulls after any significant
ingestion; the read-only analysis is described in PROJECT_HARDENING_PLAN.md
items A1/A2. Do not hand-tune them.

## Post-audit hardening run — decided 2026-09-28 (advisor: Fable 5)

An adversarial audit on 2026-09-27 found seven defects; `PLAN.md` at the repo
root holds the full remediation plan with the options that were put to the
advisor. All seven decisions below are **final** — do not re-open the options.
Phases land in dependency order, because bad domain labels contaminate any
before/after measurement of the scoring fixes.

**Two line references in circulation are wrong.** `extract_paper` hardcodes the
domain at `pipeline/extractor.py:452`, not `:455` (the audit) and not `:388`
(`PROJECT_HARDENING_PLAN.md` item A8).

### Phase 0 · #7 integration tier + #6 health honesty — DONE 2026-09-28

**#7 (commit `8e8b121`).** The 327-test unit suite runs in 0.78s with every
service mocked, so it is structurally unable to observe any of the seven
defects — they all live in the interaction between real components and real
data. Added an opt-in `integration` marker (excluded by default via `addopts`,
so `pytest -q` stays a fast hermetic run), session-scoped probes that **skip
rather than fail** when a service is down, and full store isolation: a separate
Neo4j database (`rghintegration`), separate Qdrant collections
(`*_integration`), and a `tmp_path` SQLite.

Note for future sessions: the collection-name constants must be patched in **all
four** namespaces that hold a by-value binding (`vectors.embed`,
`vectors.search`, `pipeline.gap_scorer`, `pipeline.cross_domain`). Patching only
`vectors.embed` leaves the other three pointed at the production collections.

The fixture corpus (`tests/integration/fixtures/corpus.json`, 13 synthetic
papers with deliberately impossible `99xx` arXiv ids) carries both a
`true_domain` and the wrong `declared_domain` the pre-fix pipeline would store,
reproducing the live defects in miniature. **Fail-first baseline at `0eb58bc`:
18 failed, 12 passed.** Those 18 are the proof harness for #1–#6.

**#6 (commit `df8f7e1`).** `/health` returned `200 {"status":"ok"}`
unconditionally — a literal status, Qdrant errors swallowed to `0`, and Neo4j
never contacted at all. It now probes each dependency and derives the status,
distinguishing `absent` (reachable, collection not created yet — still `ok`)
from `unreachable` (→ `degraded`, **HTTP 503**). Counts are `null` when the
backing store is unreachable. `/corpus` gets the same treatment plus
`graph_available` / `vectors_available`.

**The lifespan is now resilient, and this is load-bearing rather than
incidental.** Every backend was constructed unguarded at startup and
`get_neo4j_driver()` calls `verify_connectivity()`, so an unreachable store made
the process die before it could serve `/health` — a crash loop instead of a
diagnosis, and the 503 path unreachable in exactly the case it exists for. A
failed backend now logs at ERROR and is set to `None`; selfheal is skipped
without a driver. The 503 behaviour change is documented in `DEPLOYMENT.md`
because it affects the Render health check.

### Phase 1 · #1 domain labelling — DONE 2026-09-28

**Decision: option 1A + 1C-as-verifier. LLM classification (1B) was rejected.**

**1A — `domain` is required and validated.** The default is gone from
`PaperExtract.domain` *and* from the construction site in `extract_paper()`
(`pipeline/extractor.py:452`), which were two halves of one mechanism: a call
site that omitted the field produced a plausible-looking `computer_vision` row
instead of failing. `extract_paper(arxiv_id, domain)`,
`ingest_from_query(query, domain, limit=100)` and `IngestRequest.domain` are all
required now, validated by `pipeline.domains.validate_domain`. `selfheal.py`
validates rather than defaulting, so a graph row with a missing domain surfaces
as a per-row failure instead of silently becoming CV.

**1C — the keyword heuristic is a reporter, never an authority.**
`verify_declared_domain()` logs a WARNING when a paper's own title/abstract
confidently disagrees with the declaration and **never changes the stored
value**. This restraint is the whole point and must not be "improved" later: the
heuristic flagged 19 of 127 papers as matching neither domain, and several
(Flickr30k Entities, SPHINX, PaliGemma, streaming video understanding, 3D shape
matching) are perfectly good computer vision whose vocabulary the term lists do
not cover. Letting it auto-assign would have deleted real papers — a quiet
failure mode replacing a loud one. Medical terms are deliberately *clinical*
(modality, subject, reader) rather than task-shaped, because "segmentation" says
nothing about domain: SAM is a CV paper thick with segmentation vocabulary.

`curate_declared_domain()` is a deliberately separate verb that **may** decide,
used only for one-off curation and the integration fixture loader. It is wired
into no ingestion path. `verify_*` reports; `curate_*` decides.

**Backfill (commit below, `scripts/domain_backfill.py`).** The determinations
were reviewed paper by paper against real titles and metadata, and recorded as an
explicit manifest rather than recomputed, so the historical curation is auditable
and cannot silently change if the term lists are edited.

| | before | after |
|---|---:|---:|
| papers | 127 | **116** |
| computer_vision | 46 | **31** |
| medical_imaging | 81 | **85** |
| Qdrant limitations | 168 | **154** |
| Qdrant future_directions | 94 | **86** |

- **11 removed** as bad search-ingestion hits (neither domain): two language
  models, atomic physics, survey astronomy, analytic number theory, copyright
  law, domain-specific NLP, music generation, vehicle motion planning, control
  theory, LLM inference scaling. Removal rather than a third label, because a
  paper kept under any label keeps diluting its domain's frequency denominator.
- **4 relabelled** CV → MI: `2305.17456`, `2307.15872`, `2409.03367`,
  `2501.16469`.
- **10 borderline papers examined and deliberately kept** as CV — listed in the
  script's `KEPT_AFTER_REVIEW` so a future session does not re-litigate them or
  read the classifier's flags as unactioned findings.
- **57 orphaned nodes deleted.** `Limitation`/`FutureDirection` are keyed on
  exact text and `Method`/`Dataset` on name, so they survive their last Paper.

**Qdrant was dropped and rebuilt, not patched.** `vectors/embed.py:236` assigns
point ids positionally over a Cypher result with no `ORDER BY` and only ever
upserts, so a shrinking corpus leaves orphaned points holding stale text and a
stale `domain` payload that still match queries. Side benefit: the rebuild
created the `domain` payload index, which the live collections had been missing
since commit `0eb58bc` added `ensure_payload_indexes`.

**Measured effect on output.** CV gaps 27 → 19, and the entire off-topic top of
the list is gone. Ranks 1–7 were previously copyright law, control theory, LLM
latency, analytic number theory (×2) and atomic physics (×2); they are now
object-tracking failures under occlusion, VLM hallucination, and attribute
bleeding in image synthesis. MI is essentially unchanged, as expected — nothing
was removed from it. The 34-paper mega-cluster and the tie blocks remain, and are
Phase 2 and Phase 4 work respectively.

Note the incidental knock-on: the CV corpus reference year moved 2025 → 2024,
because the removed control-theory paper was the newest CV paper reporting a
limitation. That is `_corpus_reference_year` behaving as designed.

### Phase 2 · #3 cluster threshold + domain filter — DONE 2026-09-28

**Decision: option 3A (per-domain null p95) + the structural cap from 3C.**

`_CLUSTER_THRESHOLD = 0.86` was never derived. Its comment justified it by
sitting "well above the median" of corpus similarities — but the median of a
*noise* distribution is not a bar. That is the identical error this project had
already found and corrected for the cross-domain threshold ("0.82 sat below the
median of pure noise"), and the correction was applied to `_SOLUTION_THRESHOLDS`
and `find_cross_domain_matches` and **never to the threshold that decides what a
gap is**. There was no item for it in `PROJECT_HARDENING_PLAN.md` either.

Derived **after** the domain filter landed and **after** Phase 1's curation, as
required — deriving earlier would have measured a contaminated population:

| | measured null p95 | n pairs | was |
|---|---:|---:|---:|
| `_CLUSTER_THRESHOLDS["computer_vision"]` | **0.8769** | 1,128 | 0.86 |
| `_CLUSTER_THRESHOLDS["medical_imaging"]` | **0.8954** | 5,565 | 0.86 |

Against the old 0.86: 10.9% of arbitrary CV pairs and **28.3% of arbitrary MI
pairs** cleared it. At a ~28% per-pair noise rate a long generic seed absorbs a
quarter of its domain directly.

`scripts/derive_thresholds.py` is new and is now the canonical re-derivation tool
— the accepted A1/A2 method had no script, only prose. It reports all three
populations and **filters by domain**, because deriving a within-domain null from
an unfiltered scan mixes in cross-domain pairs.

**Measured effect — the mega-cluster is gone.**

| | before | after |
|---|---|---|
| MI largest cluster | 52 of 104 (**50%**) | 16 of 106 (**15%**) |
| MI clusters | 24 | 54 |
| CV largest cluster | 9 | 5 (10%) |
| CV clusters | 27 | 28 |

Clusters still partition their input exactly (48 = 48, 106 = 106).

**Be honest about which half did the work: the cap never fires on this corpus.**
CV's largest cluster is 5 against a cap of 10; MI's is 16 against a cap of 22. The
derived threshold alone destroyed the mega-cluster. The cap is a structural guard
against regrowth as the corpus scales — a percentile threshold bounds the error
per *pair*, and the expected number of chance members in a cluster grows with
corpus size — not an active fix today. Do not conclude from the numbers above that
the cap is load-bearing right now, and do not remove it on the grounds that it
never triggers.

**Cap semantics.** `_MAX_CLUSTER_SHARE = 0.20` with `_MIN_CLUSTER_CAP = 2` (20% of
8 limitations is 2, and a cap of 1 would forbid clustering entirely — a degeneracy
guard, not a dial). An over-cap cluster **splits**: candidates are admitted in
descending similarity so the tightest subgroup stays with the seed, and the
remainder is left unassigned to re-seed its own cluster in the same pass. Nothing
is ever dropped.

**The neighbour query is now domain-filtered.** `_query_neighbours` was the only
filtered query in the project that sent no filter, while asking for
`limit=len(limitations)` from a collection holding every domain. Measured
pre-fix: a mean **42 of every 64** CV hits were other-domain points, fetched and
discarded client-side, evicting genuine same-domain neighbours from the window —
three above-threshold CV pairs were lost that way, and the loss grew with the
other domain's size.

**Docstring correction.** `cluster_limitations` claimed seed-anchoring "prevents
transitive A→B→C chains from collapsing unrelated limitations into one giant
cluster". Seed-anchoring does remove the *chain* mechanism, but the giant cluster
happened anyway by a different route — a generic seed absorbing a quarter of the
domain directly. The claim was false as written and now says so.

**Noted, not changed** (outside this phase's mandate): re-derived on the curated
corpus, `_SOLUTION_THRESHOLDS["medical_imaging"]` would now be 0.8915 rather than
the coded 0.8987, so the coded value is currently *stricter* than its null — safe,
but drifting. CV re-derives to 0.8784 against a coded 0.8773, and the cross-domain
default to 0.8793 against a coded 0.8792. Both are effectively unchanged. Any
revision needs its own advisor decision.

### Phase 3 · #4 frequency denominator + honesty pass — DONE 2026-09-28

**Decision: option 4A (fix the denominator) + 4C (document the real behaviour).
Option 4B — normalisation — was rejected.**

**4A.** `frequency_score` divided by `_count_papers_in_domain()`, every Paper node
in the domain. But a paper that extracted no limitations cannot appear in any
numerator, so the metric was the product of two unrelated things: how widely a
limitation is reported, and how often extraction happened to succeed. On the
curated corpus **18 of 31 CV papers and 51 of 85 MI papers contribute**, so 42% of
the CV denominator and 40% of the MI denominator was papers that could never appear
in a numerator — understating every CV frequency by a factor of 1.72 and every MI
frequency by 1.67.

`_count_contributing_papers()` is now the denominator. It is kept as a separate
function from `_count_papers_in_domain()` deliberately: a reader asking "how big
is this corpus" wants every paper, a reader asking "what divides these scores"
wants this one, and conflating them is exactly why the banner used to describe a
number that was not the divisor.

**4B was rejected and must not be reintroduced without a new decision.**
Normalising the terms so their influence matches their coefficients would make
each score relative to the result set: adding a single paper would move every
number with no change in evidence. That destroys cross-run comparability, which
is a deliberately-engineered property here (see the recency-baseline decision).

**4C — what the measurement actually shows.** 4A helped but did not close the gap,
exactly as the plan predicted. Frequency's influence went from 7.4% to **9.2%**
(CV, by spread) against a nominal 40%. Deleting the term entirely still leaves the
top 10 of both domains unchanged. The full table is in the Gap Scoring Formula
section above and is refreshed by `scripts/measure_weight_influence.py`.

**Do not read the residual discrepancy as an unfinished fix.** At n=31 (CV) a
one-paper versus three-paper difference is not statistically meaningful — the
Wilson 95% intervals overlap heavily — so a formula that largely ignores it is
defensible, and the documentation was the thing that was wrong. Forcing frequency
toward 40% of the influence would mean asserting a precision the corpus cannot
support.

**Honesty pass, everywhere the old claim appeared:**

- `gap_scorer` module docstring and `compute_frequency_score` now state what the
  term measures and that it is a weak corroboration signal.
- `/corpus` reports **both** `papers` (corpus size) and
  `papers_reporting_limitations` (the actual divisor).
- `CorpusBanner` shows both, and renders "unavailable" rather than a confident
  zero when Qdrant is unreachable.
- `GapCard`'s meter is relabelled "Frequency (corroboration)" and its hint
  describes the real denominator and the real influence.
- `GapDetailSheet` said **"Frequency — 40% of the score"**, which was the
  misleading claim in its most direct form. It now says the coefficient is 0.40
  while the term accounts for roughly a tenth of what separates the gaps, and
  notes that recency does most of the ordering and that deficit separates in
  blocks because it saturates at 0 and 1.
- `frontend/lib/api.ts` types now match the API, including the nullable counts
  from Phase 0b.

Related, still open and **not** addressed here: `PROJECT_HARDENING_PLAN.md` A9
(the deficit metric is dimensionally incoherent — it divides a corpus-wide count
of future directions by a cluster-local count of papers) and A4 (tier weights are
undrived round numbers). A4's premise is now stale, incidentally: it says the
whole corpus is `explicit` tier, but the live corpus is 28 explicit / 55
conclusion / 44 inferred, so those weights are load-bearing rather than inert.

### Phase 4 · #2 tie-breaking + cluster representative — DONE 2026-09-28

**Decision: option 2A for ordering, plus the independent centroid-representative fix.**

**The representative text never computed a centroid.** `_cluster_centroid_text`
returned `Counter(...).most_common(1)[0][0]` and its docstring called that "the
most frequently occurring limitation text". `Limitation` is `UNIQUE` on `text` and
`get_all_limitations` groups by `l.text`, so **within a cluster every count is
always exactly 1** — the "most frequent" branch could never fire, and
`most_common` always resolved a total tie by insertion order, which is the seed,
which `cluster_limitations` chooses as the longest string. Measured pre-fix: the
description equalled the longest member in **27 of 27** CV clusters. A six-member
cluster was therefore labelled with whichever member had the most characters; in
one live case a sentence about imaged anatomy written by one of its five papers,
which then drove the top cross-domain match.

Renamed to `_cluster_representative_text` and now returns the member nearest the
cluster's own vector centroid (mean of L2-normalised member vectors, renormalised;
cosine, the same metric clustering uses). Memoised per cluster, because the
representative is needed twice — once for the deficit score, once for the
displayed description and solutions.

**Non-obvious property, worth knowing before reading a two-member gap's label:**
both members of a pair are *exactly* equidistant from their own centroid by
symmetry, so a two-member cluster's label is always decided by the lexicographic
tie-break, never by similarity. There is no principled "more central" member of a
pair, so this is not a defect — but the label of a two-member gap carries no claim
to being the better summary of the two. A test asserting this function must mirror
the lexicographic tie-break; using `argmax` silently asserts member ordering
instead (this caught a bug in the integration test itself).

**Ordering (2A).** `_ranking_key` is now
`(-score, -supporting_paper_count, -newest_year, gap_description)`. Previously
`results.sort(key=score)` left ties to the stable sort, which preserved cluster
creation order — and clusters are created longest-seed-first, so **rank inside a
tie block tracked description length**: ranks 1–3 tied at 0.6065 had descriptions
of 72, 70 and 64 characters, ranks 4–7 tied at 0.6043 had 50, 44, 41 and 34,
monotonically descending. A copyright-law paper held rank 1 for having the longest
sentence in its tie group. The final lexicographic key is not meaningful in itself;
it guarantees a *total* order so output is reproducible across runs and processes.

**Report this honestly: ties got more common, not less.**

| | before Phase 4 | after |
|---|---|---|
| CV gaps tied with another | 15 of 28 | **20 of 28** |
| MI gaps tied with another | 27 of 54 | **36 of 54** |

This is the interaction PLAN.md flagged. Phase 2's fragmentation produced many
more singleton clusters, and a singleton with one recent paper and no addressing
future direction scores **exactly** 1.0 on both recency and solution-deficit, so
large numbers of gaps land on identical composites. The tie-break makes the order
deterministic and defensible; it does not make the scores discriminate.

The visible consequence is that the top of both lists is now almost entirely
single-paper gaps, and a 4-paper CV gap (`f=0.1944`) sits at rank 12 beneath eleven
1-paper gaps, purely because its recency is 0.75 rather than 1.0. **That is
`PROJECT_HARDENING_PLAN.md` A9 — saturation of the deficit term — in full view, and
it is out of scope here.** Do not "fix" it by promoting paper count above score in
the sort, or by reweighting: both are formula changes needing their own advisor
decision. The UI does flag single-source gaps in amber, which is the mitigation
that exists today.

### Phase 5 · #5 /explain grounding — DONE 2026-09-28

**Decision: 5A + 5C combined. 5B — prompt wording — was explicitly rejected.**

`/explain` took `source_gap` and `target_solution` as two **arbitrary client query
strings**, built a `CrossDomainMatch` with `similarity_score=0.0` hardcoded
(commented "not used by the explanation prompt"), and called the LLM. The prompt
never interpolated the score and instructed the model to "be specific about the
shared structure", so the only compliant output was a structural analogy — the
model had no way to decline. Fed an analytic-number-theory limitation and a
histopathology future direction, `llama3.1:8b` explained that "both domains involve
distributed data and computational resources", and repeated a false
`computer_vision` label back as fact.

**5A — the gate is code, not wording.** `verify_pairing()` runs before any
generation and raises `UngroundedPairingError` on two independent conditions:

1. **Both texts must exist in the corpus** — the gap as a stored `Limitation` in
   the source domain, the solution as a stored `FutureDirection` in the target. No
   similarity makes a sentence that appears in no paper a finding about the
   literature, so this closes the arbitrary-string hole outright.
2. **The pair must clear the same noise floor the matcher uses.** Similarity is
   recomputed from the two **stored** vectors, so it is exactly the number
   `find_cross_domain_matches` would produce rather than a fresh embedding that
   could drift. Cross-domain pairs use `_CROSS_DOMAIN_THRESHOLD`; same-domain pairs
   use `_solution_threshold(domain)` — reusing the derived nulls rather than
   inventing a third constant.

`_CROSS_DOMAIN_THRESHOLD` is new **only as a name**: 0.8792 was an inline default on
`find_cross_domain_matches` and is now a module constant so the gate and the matcher
cannot drift apart. The value is unchanged. (Re-derived on the curated corpus it
comes to 0.8793; left alone, since changing it is an advisor decision.)

Why 5B was rejected: asking an 8B model to refuse a leading question is weak, and a
probabilistic refusal cannot be asserted by a test. A test now proves the gate is
unreachable-by-LLM by making `_call_ollama_text` raise if it is ever called.

**5C — the evidence ships with the prose.** `ExplainResponse` now carries
`similarity_score`, `threshold`, `grounding` ("corpus_match"), `is_hypothesis`
(always True) and both paper lists. `ExplanationPanel` heads the block
"**Hypothesis** — why this connection might matter", states that it was generated
from the two statements rather than from evidence the connection holds, and prints
the measured similarity against the noise floor. A refusal returns **422** with the
reason, which the frontend can render as information.

Note on the text lookup: it scrolls the **domain-filtered** slice and matches text
client-side rather than filtering on the text field server-side. `domain` is the only
indexed payload field, and Qdrant Cloud *refuses* a filter on an unindexed field
rather than scanning — so a server-side text filter would work locally and fail in
production, the exact trap `ensure_payload_indexes` documents. Indexing a
multi-sentence text field to avoid one scan of a few hundred points is a poor trade.

**Integration tier is now fully green: 27 passed, 3 skipped** (from 18 failing at
the start of the run). The three skips are honest: the 13-paper fixture produces no
cross-domain pair above the noise floor, which by this project's own doctrine is
*correct output rather than a bug*, and its MI slice yields no multi-member cluster
at 0.8954. Two happy-path assertions are therefore exercised against the live
corpus in the run summary rather than by the fixture.

### Corpus curation pass 2 · the arXiv-primary-category rule — DONE 2026-09-28

**Decision: a paper stays in `computer_vision` only if its arXiv PRIMARY category
is `cs.CV`. Cross-listing to `cs.CV` is not sufficient. Applied to CV only; MI was
checked and reported without action.**

**Why a second pass was needed.** Pass 1 (see Phase 1 above) reviewed the 33 papers
the keyword classifier flagged, but accepted the 94 it scored "ok" on a title scan —
so **21 of the 31 papers kept as CV had no individual review note**. Read-only
verification found `2404.07922` (LaVy, a Vietnamese multimodal LLM) had passed that
way, and it was not cosmetic: it supplied CV rank 2 and four of the ten cross-domain
matches. The rule adopted here is external and checkable rather than judgemental,
which is the point — it does not depend on anyone's reading of a title.

| | before | after |
|---|---:|---:|
| papers | 116 | **113** |
| computer_vision | 31 | **28** |
| medical_imaging | 85 | 85 (untouched) |
| CV contributing (frequency denominator) | 18 | **16** |
| Qdrant limitations | 154 | **150** |
| Qdrant future_directions | 86 | **84** |

Removed (`REMOVE_PASS2` in `scripts/domain_backfill.py`), 3 of 31, all arXiv
primary `cs.CL`:

- **`2306.14824` Kosmos-2** — cross-listed `cs.CV`; contributes 0 limitations, so
  removal moved the denominator only.
- **`2401.13601` MM-LLMs survey** — **not cross-listed to `cs.CV` at all**. Its two
  "limitations" were the survey's own hedging ("certain aspects may have eluded our
  scrutiny").
- **`2404.07922` LaVy** — cross-listed `cs.CV`; the contribution is language
  coverage, vision is the modality it operates over.

15 orphaned nodes deleted; both Qdrant collections dropped and rebuilt, for the same
positional-point-id reason as pass 1.

**Medical imaging: reported, deliberately not acted on.** Its primary categories are
`cs.CV` (44) and `eess.IV` (35), plus six papers under `cs.CY`, `cs.LG`,
`physics.med-ph`, `cs.AI` and `cs.CL` that are unambiguously clinical on inspection
(demographic bias in medical vision-language models, MRI reconstruction ×2,
multimodal medical data generation, radiomics automation, a generalist medical
foundation model). **A single-category rule does not transfer to a field that
legitimately spans two**, so applying this rule to MI would delete real papers. If MI
ever needs one, it has to be `{cs.CV, eess.IV}` plus a manual exception list.

`ALL_REMOVALS` is the union of both manifests and is what the script operates on;
removal is idempotent, so re-running applies both passes safely. The two manifests
stay separate so each pass's reasoning remains auditable. `KEPT_AFTER_REVIEW` is left
unedited as the historical record of what pass 1 decided, with a note that pass 2
later removed one of its entries.

**Nulls re-derived, and no constant was changed** (explicitly out of scope):

| constant | coded | re-derived now | drift |
|---|---:|---:|---|
| `_CLUSTER_THRESHOLDS["computer_vision"]` | 0.8769 | 0.8771 (n=946) | +0.0002 |
| `_CLUSTER_THRESHOLDS["medical_imaging"]` | 0.8954 | 0.8954 (n=5,565) | exact |
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.8773 | 0.8789 (n=968) | +0.0016 |
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.8987 | 0.8915 (n=6,572) | −0.0072 |
| `find_cross_domain_matches` default | 0.8792 | 0.8794 (n=5,060) | +0.0002 |

All three cluster/cross-domain values are stable to ~0.0002 across two curation
passes, which is reassuring about the derivation method. The MI solution threshold
remains the one outlier: the coded 0.8987 is **stricter** than its measured null p95
of 0.8915, so it under-counts addressing solutions and makes MI gaps look more open
than they are. Safe, but drifting, and still awaiting an advisor decision.

### Phase 1a · MI solution threshold re-derived — DONE 2026-09-28

| constant | old | new | drift | action |
|---|---:|---:|---:|---|
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.8987 | **0.8915** | 0.0072 | **updated** |
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.8773 | — | 0.0016 | kept |
| `_CLUSTER_THRESHOLDS["computer_vision"]` | 0.8769 | — | 0.0002 | kept |
| `_CLUSTER_THRESHOLDS["medical_imaging"]` | 0.8954 | — | 0.0000 | kept |
| `_CROSS_DOMAIN_THRESHOLD` | 0.8792 | — | 0.0002 | kept |

The old 0.8987 was **stricter than its own noise floor** (measured null p95 0.8915
over n=6,572 pairs), so it under-counted addressing future directions and made
medical-imaging gaps look more open than they are.

**`DRIFT_TOLERANCE = 0.002` and the rule it enforces.** A constant is rewritten only
when its drift from the freshly derived value exceeds 0.002. Below that, drift is not
distinguishable from resampling the same corpus — bootstrap 95% CIs on these p95s span
roughly ±0.002 — so chasing the third decimal would move rankings for no defensible
reason.

**Constants are now only writable through the derivation script.**
`scripts/derive_thresholds.py --apply` rewrites stale values in place and refreshes the
`# n=` provenance comment beside each one. Hand-editing is how the unmeasured 0.86
cluster threshold got in and survived, so that path is closed by convention and by the
guard below.

Two things this exposed, both fixed:

- **Stale provenance comments were claiming unmeasured sample sizes.** The CV solution
  line said `# n=2,176 random pairs`, a figure from the 127-paper corpus; the current
  null is n=968. The *value* stays (drift 0.0016, inside tolerance) but the comment now
  records when it was derived, what the current re-derivation says, and why it was not
  rewritten. Same for both cluster entries.
- **The unit tests pinned exact literals** (`== 0.8987`), which turns every legitimate
  re-derivation into a test failure whose only "fix" is copying whatever the code now
  says — proving nothing. Those three tests now assert *structure* (per-domain lookup,
  above the superseded 0.85, strictest-known fallback, plausible cosine range) and the
  exact values are guarded against the live null distribution instead.

**New guard: `tests/integration/test_threshold_derivation.py`.** It imports
`scripts/derive_thresholds.py` rather than reimplementing the derivation — a guard
computed a second way can pass while the real derivation is wrong — and fails when any
coded constant drifts beyond tolerance. Confirmed to fail against pre-1a code, naming
the MI drift of 0.0072. It measures the **production** collections, not the integration
fixture corpus, because a 13-paper fixture cannot produce a meaningful null. A companion
unit test asserts the *inventory* of threshold families matches the derivation script, so
a new threshold cannot be added without becoming checkable.

### Phase 1b · deterministic extraction-quality gates — DONE 2026-09-29

Closes `PROJECT_HARDENING_PLAN.md` item **A3** without re-extraction, which stays
blocked by B4. New module `pipeline/extraction_filter.py`, applied in
`extract_paper` before anything reaches SQLite or the graph.

**Four gates, all deterministic, no LLM anywhere in them.** Domain assignment,
scoring and thresholds are all LLM-free in this project; a model deciding what counts
as a real limitation would put model judgment directly upstream of every score.

- **(a) prompt echo** — `PROMPT_ECHO_PHRASES`, matched against the **whole**
  normalised string.
- **(b) hedging** — a short `HEDGING_SUBSTRINGS` list for phrases that are
  contentless wherever they appear, plus `HEDGING_WHOLE` for framings that are only
  boilerplate when they *are* the whole string.
- **(c) minimum informative length** — `MIN_WORDS`, the **5th percentile of the
  measured word-count distribution** per field (both come out at 3 words;
  `scripts/derive_length_cutoffs.py` reports the distribution). p5 rather than p10
  because p10 for future directions is 5 words, which would discard terse but
  specific solutions like "contrast-agnostic, pathology-encoded representations".
- **(d) duplicates** — exact-after-normalisation and containment, **parameter-free on
  purpose**. No Jaccard or cosine cutoff, because that would mean inventing an
  unmeasured constant; a parameter-free rule was available so it is used.

**Whole-string matching is load-bearing, not a detail.** `"remains challenging"` alone
is worthless; `"remains challenging to process high-resolution images"` is a genuine
two-paper gap in the live corpus. A substring rule deletes both.

**The echo list is an explicit constant and must stay one.** It was briefly written to
parse the quoted phrases out of the prompt at runtime, which is circular: the
accompanying prompt fix removes those quotes, so a parsed list silently goes empty and
the gate stops rejecting anything while still appearing to work. Three tests hold it
in place — the list is non-empty and contains the known echoes; any phrase quoted in
the prompt is covered by the list; and the prompt quotes no cue phrases at all.

**A dry run caught my own over-reach, which is why the two hedging lists are split.**
The first `HEDGING_SUBSTRINGS` included "further research is needed", "beyond the scope
of this" and "left for future work". Those routinely appear in sentences that *do* name
something — "Further research is needed to improve robustness and generalizability" —
and were being silently discarded. They moved to `HEDGING_WHOLE`. Deleting signal
quietly is worse than leaving a little boilerplate in.

**Backfill (`scripts/filter_backfill.py`) re-filters stored strings and never
re-extracts**, for the CLAUDE.md reason: re-extraction on a paper that already has
graph relationships attaches new `Limitation` nodes while the old ones remain.

| | before | after |
|---|---:|---:|
| extracted strings | 234 | **224** |
| Limitation nodes | 148 | **141** |
| FutureDirection nodes | 84 | **83** |
| CV contributing papers | 16 | **15** |
| MI contributing papers | 51 | **50** |

**10 of 234 strings rejected (4.3%)** — 5 `too_short` (`Overfitting`, `Dataset bias`,
`Training instabilities`, `domain shift`, `hallucinations`), 4 `prompt_echo`
(`remains challenging` and `future work includes`, from `2405.14458` and `2309.17264` —
both papers lost *every* limitation they had), 1 `hedging`
(`overcome current limitations`). Every rejected string is logged to
`data/rejected_extractions.jsonl` (gitignored, runtime artifact) so filtering is
auditable — a silent filter is indistinguishable from an extraction failure.
`raw_json` still stores the **unfiltered** model output, so the filter is reversible.

**Thresholds re-derived afterwards; all five within `DRIFT_TOLERANCE`, none changed**
(max drift 0.0010 on the cross-domain default).

### Phase 2 · A9 resolved — Option F — DONE 2026-09-29

Closes `PROJECT_HARDENING_PLAN.md` item **A9**. Full measurements in
`A9_F_MEASURED.md`. Two independent changes; **the 0.40/0.35/0.25 weights and the
recency term are untouched.**

**1 · `solution_deficit` is continuous.** Was
`1 - (count_addressing / papers_reporting)` — a corpus-wide count over a cluster-local
count, so not a proportion, with only **4 distinct values per domain** and 18 of 28 CV
/ 38 of 54 MI gaps at exactly 1.0. Now `1 - addressedness`, rescaling the **nearest**
eligible future direction's similarity between the domain null's p50 and p99.
Dimensionally coherent (a cosine rescaled by two cosine percentiles) and still
deterministic (the anchors are corpus-derived constants, not per-result-set values).

**2 · Two-tier ranking.** Gaps with ≥2 supporting papers rank above *all*
single-source gaps; each tier then orders by (score desc, papers desc, newest year
desc, description asc). A tier rather than a formula change because recency and
deficit are both maximal in the *absence* of evidence, and no reweighting fixes that
without asserting a precision the corpus cannot support — at 15 contributing CV papers
the Wilson intervals for a one- and a three-paper gap overlap heavily.

| | CV before | CV after | MI before | MI after |
|---|---:|---:|---:|---:|
| exact-score ties | 17 of 24 | **0** | 38 of 53 | **3** |
| distinct scores | 11 | **24 of 24** | 25 | **51** |
| distinct deficit values | 4 | **24** | 4 | **46** |
| deficit at exactly 1.0 | 18 | **1** | 38 | **0** |
| Spearman vs previous | — | 0.7287 | — | 0.5939 |
| corroborated / single-source | — | 5 / 19 | — | 16 / 37 |

Frequency's measured influence is still ~8–10% against a nominal 40% — unchanged
conclusion, and still documented rather than normalised away. But **deleting the
frequency term now changes the top-10 order in both domains**, where before it did
not: with ties gone, the term finally discriminates.

**Anchor choice, per the stated rule.** Keep p50/p99 unless p50/p95 gives strictly
fewer ties *and* higher Spearman. It gives neither, in either domain — CV ties 3 vs 0,
MI ties 15 vs 3, Spearman lower in both. p95 sits only ~0.05 above p50, so the narrow
span re-saturates at the *other* end (MI deficits at exactly 0.0 go from 7 to 24) and
collapses `/cross-domain` to zero matches both ways. **p50/p99 kept.**

**`_UNRESOLVED_DEFICIT_FLOOR = 0.3` is now load-bearing, and that is new.** Item A5
called it "effectively a binary switch, not a tunable dial" — true when deficits took
four values. Now that they are continuous it genuinely selects: gaps above the floor
went 18→16 (CV) and 33→27 (MI), and `/cross-domain` went 10→7 (CV→MI) and 10→4
(MI→CV). It has never been derived from anything. **Flagged, not changed** — out of
Phase 2's scope.

**One policy genuinely changed.** The 2026-08-21 rule was "the solutions shown are
exactly the ones counted against the score". The score is now set by the *nearest*
eligible future direction. When any hit clears `_solution_threshold` that nearest one
is the first item of `proposed_solutions`, so the score is still set by something the
reader can see — but when none clears it, the list is empty and the deficit may still
be below 1.0, because the nearest hit can sit above the null median without reaching
p95. Intentional: a weak-but-real neighbour should not read as "nobody proposed
anything". An empty solution list no longer implies deficit exactly 1.0.

**Frontend.** `GapResult.tier` is sent by the backend rather than derived client-side,
so ranking and label cannot disagree. `/gaps` shows a heading at each tier boundary —
necessary, because scores are deliberately **non-monotonic across it** and without the
divider a higher score below a lower one reads as a broken sort. `SupportBadge` now
says "Corroborated — N papers report similar limitations"; the previous "N papers
report this limitation" was **false for every multi-member cluster**, since a gap is a
cluster of similar statements labelled by its centroid-nearest member, not a sentence
all N papers wrote. Frequency hints in `GapCard` and `GapDetailSheet` now name the real
denominator (papers that contributed ≥1 limitation), and the deficit explanation
describes the continuous measure.

### Phase 3a · stable Qdrant point ids — DONE 2026-09-29

Point ids were `id=i`, the enumeration index of a Cypher result with **no `ORDER BY`**,
and `_upsert_records` only ever upserted. Two consequences: ids were not stable across
re-embeds, and a corpus that *shrank* left orphaned points holding stale text and a
stale `domain` payload that still matched queries. Every curation pass so far has had
to drop and rebuild both collections wholesale to work around it.

Ids are now `uuid5(namespace, f"{domain}\0{text}")` via `vectors.embed.point_id`, and
`_upsert_records` prunes any point whose content is no longer in the graph.

- **`domain` is part of the key** because the same limitation text can legitimately be
  reported in both domains, and those are two points with different payloads.
- **`_POINT_ID_NAMESPACE` is fixed forever.** Changing it re-ids every point in both
  collections and orphans everything already stored.
- **`prune=False`** exists for incremental tranche ingestion, where `records` is only
  the new slice and pruning would delete the rest of the corpus.

Collections rebuilt once so every point carries a content-derived id: 141 limitations,
83 future directions, unchanged counts. **Re-embedding twice more left both counts
identical**, which is the property that was missing — a re-embed is now idempotent
rather than additive.

### Phase 3b/3c · rule-gated resumable ingestion — DONE 2026-09-29

**Corpus rules are now enforced at ingest time** (`pipeline/corpus_rules.py`), based on
arXiv's own **primary category** rather than on anyone's reading of a title. All
fourteen papers removed across the two curation passes got in because ingestion accepted
whatever a keyword search returned.

```
computer_vision: primary in {cs.CV}
medical_imaging: primary in {eess.IV, physics.med-ph}, or primary in {cs.CV}
                 with a medical keyword match
```

Anything else is `rule_rejected` unless its id is in `data/mi_allowlist.txt`.

- **Cross-listing to `cs.CV` is not sufficient for CV.** Every paper removed in
  curation pass 2 was cross-listed `cs.CV` with a `cs.CL` primary.
- **The CV rule is deliberately not applied to MI.** Medical imaging legitimately spans
  two primaries (44 `cs.CV`, 35 `eess.IV`), so a single-category rule would delete real
  papers.
- **The keyword list gates, it never assigns.** It only decides whether a `cs.CV`-primary
  paper may enter medical imaging; the caller always declares the domain. Same restraint
  as `verify_declared_domain`, same reason.
- **`data/mi_allowlist.txt`** is an explicit file of ids, not a broadened rule, so each
  exemption is visible and attributable. Seeded with the six MI papers whose primaries
  are `cs.CY` / `cs.LG` / `physics.med-ph` / `cs.AI` / `cs.CL`.

`pipeline/arxiv_source.py` queries `cat:<category>` rather than keywords — a category is
a claim arXiv itself makes about the paper. One request every three seconds per arXiv's
terms, exponential backoff, version suffixes stripped so ids match what is stored.

`scripts/ingest_tranche.py` requires `--domain`, skips already-ingested ids,
**checkpoints after every paper**, retries twice then logs and continues, and records one
line per paper in `data/ingest_log.jsonl` with outcome `ok` / `no_limitations` /
`pdf_failed` / `extraction_failed` / `rule_rejected`. One paper at a time, model loaded
once, no parallel extraction. Skipping already-ingested papers is a **correctness**
requirement, not an optimisation: `Limitation` is UNIQUE on `text` and
`_upsert_paper_counting` only MERGEs, so re-ingesting attaches a second set of nodes.

**Two things the first live tranche taught, both fixed:**

- **Semantic Scholar 404s on brand-new arXiv papers.** All three papers in the first
  trial failed extraction, because `extract_paper` fetched metadata from Semantic
  Scholar that arXiv had *already provided*. `extract_paper` now takes an optional
  `metadata` dict; the ingest job passes arXiv's title/abstract/year and skips the
  lookup. That also removed the 1 req/sec limit and its up-to-60s 429 retries from the
  bulk path. Single-paper `/ingest` still uses Semantic Scholar, where the metadata is
  richer and the quota cost is one request.
- **PDF failure was observable but not recorded.** The extractor already logs a
  distinctive warning on PDF fallback, so the ingest job listens for it with a log
  handler rather than changing `fetch_full_text`'s signature and its tests.

Measured on the trial tranche: **~44 s per paper** end to end.

`pdf_failed` is common and is not an error — papers over the 30MB cap, or with no
limitations section, fall back to the abstract and still yield limitations at the
`inferred` tier.

### Deliberately deferred, 2026-08-23 — do not treat as oversights

Three known scoring limitations were reviewed at the same time and consciously
left in place. They are documented so a future session does not "discover" them
and fix them unilaterally:

- **A3 — half the extracted future directions are contentless boilerplate.**
  22 of 44 are short and non-specific; 4 gaps score fully solved on nothing but
  boilerplate. The real fix is a tighter extraction prompt, which requires
  re-extracting the corpus — and that is **blocked by B4**, since re-ingesting a
  paper that already has graph relationships duplicates its limitations rather
  than replacing them. Raising thresholds does not help here: generic text scores
  *high*, not low, because it sits near the corpus centroid.
- **A9 — the solution-deficit metric is dimensionally incoherent and saturates.**
  It divides a corpus-wide count of future directions by a cluster-local count of
  papers, so it is not a proportion. A quality refinement, not an integrity risk:
  the term still orders gaps sensibly, it just has poor resolution.
- **A5 — `_UNRESOLVED_DEFICIT_FLOOR = 0.3` is an undocumented constant.** With
  deficits heavily saturated at 0.0 and 1.0 it acts as a binary switch rather
  than a tunable dial, so re-deriving it would change little until A9 is settled.

---

### Solution-deficit scoring — decided 2026-08-21 (advisor: Fable 5)

Three defects from the Fable 5 review, all re-verified against the live stack
before the decision. Advisor chose options 1A / 2A / 3A:

1. **Domain filter (1A).** `_find_addressing_solutions()` now applies a
   server-side Qdrant filter on `domain`. Previously unfiltered, so a
   medical-imaging future direction could mark a CV gap as solved — 15/27
   clusters were contaminated. This also settles a conflict: `cross_domain.py`
   treats a CV-gap ↔ MI-solution pairing as a *discovery*, while the deficit term
   was reading the same signal as *already solved*, demoting exactly the gaps the
   cross-domain feature exists to surface. Same-domain-only is now canonical.
2. **Self-match exclusion (2A).** A future direction is ignored if any of its
   papers also reports the limitation. A paper restating its own open problem as
   future work is the definition of an unsolved gap; counting it inverted the
   signal. 12/27 clusters were affected.
3. **Dead cap removed (3A).** `min(matches, papers_reporting)` was verified to be
   a **no-op** — the outer `max(0.0, ...)` clamp already floors any ratio above
   1.0 at zero, producing identical scores on all 27 clusters. Removing it changed
   no ranking. It was deleted as misleading code, not as a scoring fix.

Both call sites share one policy by explicit advisor decision: the
`proposed_solutions` shown to users are exactly the ones counted against the
score, never a looser set.

**Deferred — do not treat as settled.** The deficit metric is dimensionally
incoherent: it divides a corpus-wide count of future directions by a
cluster-local count of papers, so it is not a proportion and it saturates. Even
after these fixes 17/27 clusters sit at exactly 0.0 with only 5 distinct values
across the corpus, and 13/27 clusters are singletons where the metric is strictly
binary. Options 3B (redefine the denominator) and 3C (saturating function) were
raised and **explicitly deferred to a future session** — they change the formula
and need their own advisor decision.

Measured impact when implemented: 17/27 clusters (63%) reordered; deficit
distribution moved from `{0.0: 21, 0.5: 2, 0.667: 1, 1.0: 3}` to
`{0.0: 17, 0.2: 1, 0.286: 1, 0.5: 1, 1.0: 7}`. Live output was verified to
contain zero self-match and zero cross-domain leaks across 79 displayed
solutions. The 9 regression tests were confirmed to fail against the
pre-fix code before landing.

---

## Extraction Prompt (Ollama)

Always use this exact prompt structure. Do not modify without updating this file.
**This section drifted once already** — the two "If … not explicitly stated" lines
and the whole tier-instruction mechanism below existed in code for a long time
while this file still showed the older prompt. `pipeline/extractor.py` is the
source of truth; if they disagree, the code is right and this section is stale.

Base prompt (`_EXTRACTION_PROMPT`):

```
You are a scientific paper analyst. Extract structured information from the following paper.

Return ONLY valid JSON with these exact keys. No explanation, no markdown, no preamble.

{
  "objectives": ["<list of research objectives>"],
  "methods": ["<list of algorithms or architectures used>"],
  "datasets": ["<list of datasets mentioned>"],
  "evaluation_metrics": ["<list of metrics used>"],
  "limitations": ["<list of explicit limitation statements — be granular, one limitation per item>"],
  "future_directions": ["<list of future work suggestions from the authors>"]
}

If limitations are not explicitly stated, return an empty list [] — do not invent limitations.
If future_directions are not explicitly stated, return an empty list [] — do not invent future_directions.

Paper text:
{paper_text}
```

The base prompt also carries an explicit anti-boilerplate constraint (added
2026-09-29): every item must be a self-contained statement naming a specific problem
or step, and section headings, connective words and hedging about the paper's own
completeness are excluded.

Tier guidance (`_TIER_INSTRUCTIONS`, injected by `_build_prompt()` immediately
before `Paper text:`). The `explicit` tier renders the base prompt unchanged.

> **These deliberately quote NO example phrases, and must not start doing so again.**
> The previous version listed cue words verbatim — "look for phrases like 'however',
> 'despite', 'remains challenging', 'future work includes'" — and `llama3.1:8b` echoed
> them straight back. The corpus ended up with `Limitation` nodes whose entire text was
> one of those cues, and because `Limitation` is UNIQUE on `text` a single boilerplate
> node was shared by every paper that emitted the same string, then became a cluster
> seed, a gap description and a cross-domain match source.
>
> The instructions now describe the *kind* of clause to look for (where authors qualify
> a result, contrast it with what they did not achieve, or defer work) and state that
> the connective or hedging wording itself must never be reported. A test asserts no
> cue phrase is quoted in the prompt, and a second asserts that any quoted phrase which
> *is* present is covered by the filter's echo list.

Prompt quality was `PROJECT_HARDENING_PLAN.md` item **A3** and is now addressed by the
deterministic gates in Phase 1b below rather than by re-extraction — which remains
blocked by B4.

---

## Semantic Scholar API Usage

```python
# Paper search
GET https://api.semanticscholar.org/graph/v1/paper/search
  ?query=computer+vision+object+detection
  &fields=paperId,title,year,abstract,externalIds
  &limit=100

# Paper detail
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}
  ?fields=title,year,abstract,tldr,openAccessPdf

# Rate limit: 1 req/sec unauthenticated, 10 req/sec with API key
# Always use exponential backoff on 429 responses
# API key goes in header: x-api-key: YOUR_KEY
```

---

## API Rate Limiting

`api/rate_limit.py` — in-process token bucket, registered as HTTP middleware in
`api/main.py`. No new dependency (deliberately not `slowapi`; see the module
docstring for why). Buckets are keyed by peer address + cost tier:

| Tier | Endpoints | Sustained rate | Burst |
|---|---|---|---|
| `ingest` | `/ingest` | 3/min | 3 |
| `llm` | `/explain` | 10/min | 10 |
| `heavy` | `/gaps`, `/cross-domain` | 20/min | 20 |
| `search` | `/search` | 30/min | 30 |
| `light` | `/health`, `/paper/*`, unmapped | 120/min | 120 |

Over-quota requests get `429` with a `Retry-After` header. Set
`RATE_LIMIT_ENABLED=false` to disable. Middleware registration order matters:
the limiter is registered **before** CORS so CORS ends up outermost and a 429
still carries CORS headers — otherwise the frontend sees an opaque network error.

State is per-process and in-memory. A multi-worker or deployed setup would need
a shared store (Redis); revisit if deployment happens.

---

## Advisor–Executor Protocol

- ADVISOR = Fable 5 (external chat, relayed by Vipul). Owns architecture,
  algorithm/threshold design, and any stack decision.
- EXECUTOR = Claude Code, whichever model the session happens to run on. Owns
  implementation, tests, and mechanical fixes. Does not make architectural calls
  on its own. The constraint is the role, not the model — a more capable
  executor model does not earn it architectural authority.
- Executor stops and raises an ADVISOR QUERY (context, decision needed,
  options, optional lean) instead of proceeding whenever a change touches:
  gap-scoring weights/formula, clustering approach, cross-domain threshold
  or matching logic, or any stack component swap.
- A recommendation from a prior external review (e.g. a Fable 5 static
  review) is not standing authorization — it must go through this loop
  again in the current session before being implemented.
- Once the advisor decides, the executor implements and updates this file
  with the decision + rationale before moving on, so it isn't re-litigated
  next session.

---

## Rules for Claude Code Sessions

1. **One module per session.** Never work on multiple files across layers simultaneously.
2. **Always read this CLAUDE.md at the start of every session** before writing any code.
3. **Never change the stack unilaterally.** No swapping Neo4j for another DB, no changing Qdrant collection names, no switching embedding models, no changing the clustering algorithm or similarity thresholds — these go through the advisor–executor protocol (the advisor decides, the executor implements) and get logged here with rationale once decided. An external review recommending a stack change is input to that decision, not authorization to make it.
4. **Always write pytest tests** for every function you implement.
5. **Never hardcode secrets.** All credentials come from `.env` via `python-dotenv`.
6. **Validate Pydantic models strictly.** If LLM output fails validation, log the failure to `data/failed_extractions.log` and continue — do not crash.
7. **Use exponential backoff** on all external API calls (Semantic Scholar).
8. **Do not install packages** not listed in `pyproject.toml` without updating it first.

---

## Current Phase

**MVP is built, not Phase 1.** The full pipeline described in the architecture
diagram above (extraction → graph → vectors → gap scoring → cross-domain
matching → API → frontend) exists end to end, including the CV ↔ Medical
Imaging cross-domain matcher. The "Phase 1 — Extraction Pipeline" label
previously in this file was stale and has been removed — do not reintroduce
phase language that implies only the extractor exists.

**Do not trust a specific test-pass count or "everything works" claim from
this file, from memory, or from a prior session's summary without verifying
it yourself at the start of the session:**
```
pytest -q
git log --oneline -10
git status
```
Treat the actual output of these commands as ground truth for "what phase
we're really in," not any number written in prose (here or anywhere else).
If the test count or git state doesn't match what a prior note claims,
say so explicitly before doing any new work — do not silently assume the
higher (or lower) number is correct.

**Active work is post-MVP hardening**, structured as a four-session plan
from an external Fable 5 review:
- Session 1: three cheap bug fixes (hardcoded year, pyproject.toml typo,
  missing rate limiting)
- Session 2: cross-domain similarity threshold re-derivation, possible
  LLM verification stage
- Session 3: deliberate go/no-go decision on keeping Neo4j (requires the
  advisor–executor loop below — not a unilateral change)
- Session 4: remaining improvements

Confirm which of these sessions is actually done (via git log / tests),
not which one this file last said was done.

Regarding the test arXiv IDs previously listed here (`2301.00234`,
`2303.05499`, `2212.09748`): arXiv IDs generated or recalled from an LLM's
memory are unreliable and have previously pointed to unrelated papers.
Verify any arXiv ID against the actual arXiv listing or Semantic Scholar
before using it as a validation case — do not trust an ID just because
it appears in this file or in chat.

---

## MVP Scope

The original MVP scope (CV-only corpus, then CV + Medical Imaging for
cross-domain, ~50→500 papers, 3-page frontend, no auth, no cloud deploy)
has been built. Scope expansion (100-paper corpus growth, deployment to
Render/Vercel/AuraDB/Qdrant Cloud, Kaggle-related work) is tracked
separately and is not gated behind a "Phase 5" that no longer exists in
this plan — it's gated behind finishing the four-session hardening plan
above.

## Known Pitfalls — Do Not Repeat

### PDF Extraction
- pdfplumber fails on two-column academic PDFs — use PyMuPDF (fitz) only, never switch back
- Never search for section headings in joined full-text string — always use page-by-page _select_section_from_pages()
- Section heading regex must match at line start, not mid-sentence — _SECTION_HEADING_RE handles this
- Search last 40% of pages first (tail_start = int(total * 0.6)) — limitations sections are always near the end
- Cap at 20 pages (_MAX_PDF_PAGES = 20), 4000 chars per section (_MAX_SECTION_CHARS = 4000)
- Always use browser-like User-Agent header for arXiv PDF requests to avoid 403s
- Pre-2022 papers often have no limitations section — returning [] is correct behaviour, not a bug
- MuPDF "cannot find ExtGState resource" errors are harmless — PyMuPDF still extracts text, ignore these warnings
- Papers with corrupted PDF resources still get processed via fallback — do not add special handling

### SQLite Serialisation
- NEVER use str(list) to store list fields — produces Python repr format that json.loads cannot parse
- ALWAYS use json.dumps(v) for list fields when writing to SQLite
- ALWAYS use json.loads() with try/except fallback to [] when reading from SQLite
- New fields added to PaperExtract need manual db['papers'].add_column() on existing DBs — SQLite does not auto-migrate
- The extraction_tier column was added manually after initial schema creation — always check for missing columns before ingestion

### Neo4j
- Instance name (rgh-mvp) is NOT the database name — actual database name is 'neo4j', set NEO4J_DATABASE=neo4j in .env
- nodes_created=0 on re-runs is normal — MERGE finds existing nodes, only counts new ones
- Always verify green Active status in Neo4j Desktop before running any pipeline commands
- Constraint "already exists" INFO logs are normal and harmless — not errors

### RESOLVED — Neo4j/SQLite paper drift (diagnosed + fixed 2026-08-22)

**Fixed.** SQLite now holds 63 papers matching Neo4j's 63 Paper nodes; drift is
zero in both directions. The 19 missing rows were rebuilt directly from the graph
— no LLM re-extraction, no network calls, no change to Neo4j. Kept below because
the root cause can recur and the remedy has non-obvious constraints.

Neo4j held **63 Paper nodes to SQLite's 44**. The 19 extra were all
`computer_vision`.

**What they are:** real papers, not test data or failed ingestions. Each has a
genuine title, year, domain, and live relationships (methods/datasets, and 7 of
the 19 have limitations). Their property keys are identical to healthy nodes.
None appear in `data/failed_extractions.log` (which holds only 6 entries, all
429s, none matching).

**Why they diverged:** SQLite `papers` has contiguous rowids 1–44 with no gaps,
so nothing was ever deleted from it — the 19 were never in *this* file. CV rows
were ingested 2026-06-29→07-01 and medical imaging 2026-07-03, and the MI counts
match Neo4j exactly (17 = 17) while only CV diverges (46 vs 27). Best explanation:
`data/papers.db` is gitignored and disposable, so it was recreated at some point
after the first CV ingestion run; Neo4j persisted across that reset and kept the
earlier papers. This is a **restore/rebuild artifact, not an ingestion bug** — no
current code path writes Neo4j without also writing SQLite.

**Why it mattered — user-facing, not cosmetic.** Neo4j is the scoring source of
truth, so these papers fully participate in discovery: they contributed 17 of the
64 limitation nodes feeding the engine. The visible symptom was that **13 of 27
gaps cited at least one paper whose `/paper/{arxiv_id}` returned 404**, because
that endpoint reads SQLite. Confirmed unresolvable and cited at the time:
1505.04870, 1904.08980, 2101.09744, 2207.10077, 2305.10683, 2305.17456, 2409.13112.
All seven now return 200.

**The frequency denominator was NOT affected — an earlier draft of this note got
that wrong.** `_count_papers_in_domain()` counts *Neo4j Paper nodes*, which were
always 46 for CV and stayed 46. The drift was SQLite missing rows, never the graph
missing papers, so no scoring input ever changed. Verified empirically: gap
rankings before and after the fix are byte-identical — 27 gaps, zero reordered,
zero score changes. Deleting the 19 would have moved the denominator (46 → 27) and
genuinely changed rankings; repairing SQLite could not.

**How it was fixed — rebuild from the graph, do not re-extract.** The 19 rows were
reconstructed directly from Neo4j relationships (title, year, domain, methods,
datasets, limitations, future_directions, tier) via the project's own
`_paper_to_row()` so list fields are `json.dumps`'d per the SQLite rule above.

Re-running the normal ingestion path was considered and **rejected**, for a reason
worth remembering: `graph/populate.py:_upsert_paper_counting()` only ever MERGEs
and never removes existing relationships, and `Limitation` nodes are keyed on exact
`text`. Ollama extraction is non-deterministic, so re-extracting a paper that
already has limitations in the graph creates *new* Limitation nodes while the old
ones stay attached — the paper then reports both wordings, inflating the corpus
with near-duplicates and corrupting clustering. **Never re-ingest a paper that
already has graph relationships without detaching them first.**

Trade-off accepted: Neo4j never stored `objectives`, `evaluation_metrics`, or
`raw_json`, so those are empty on the 19 rebuilt rows. `raw_json` instead records
`{"source": "reconstructed_from_neo4j", ...}` so the provenance is visible rather
than looking like a failed extraction. Anyone wanting those fields populated must
re-extract, which requires solving the detach problem above first.

**Prevention — now automated (G3, 2026-08-24, commit `4b37611`).** `data/papers.db`
is gitignored and disposable while Neo4j persists independently, so this can recur
any time the DB is rebuilt — and on an ephemeral container filesystem it would
recur on *every* redeploy. Two layers now address it:

- `PAPERS_DB_PATH` points the store at a persistent volume in deployment.
- `pipeline/selfheal.py` runs from the API lifespan, compares Neo4j Paper nodes to
  SQLite rows, and rebuilds any missing rows by the same reconstruction path used
  above — from graph relationships, no LLM, no network, deterministic. It is
  additive only: existing rows are never overwritten and SQLite-only rows are
  never deleted. A rebuild logs at WARNING, because it means the volume failed.

The manual check is still the right thing to run after any papers.db reset:
`MATCH (p:Paper) RETURN count(p)` against `SELECT count(*) FROM papers`. The
self-heal makes the app recover on its own; it does not make the drift
uninteresting.

### Semantic Scholar API
- Free tier = 1 req/sec — always time.sleep(2) between individual paper fetches, time.sleep(5) between batch queries
- API returns abstracts only — full paper text must come from arXiv PDF via fetch_full_text()
- 429 errors after 3 retries log to data/failed_extractions.log and continue — correct behaviour, do not crash
- data/failed_extractions.log must stay in .gitignore — never commit runtime logs
- 3 failed papers per batch of 20 is normal — do not retry immediately, wait 60s for rate limit reset

### HuggingFace / Specter2
- Model re-downloads on every process unless cache_dir is explicitly set in model_kwargs, processor_kwargs, config_kwargs
- Do not set HF_HUB_OFFLINE=1 in your shell or .env — breaks fresh installs on new machines.
  The deployment `Dockerfile` is the one exception: it bakes the weights in and fails the
  build if they cannot be loaded offline, so a built image provably has them. Measured
  there with the network unreachable: 3s and 0 network log lines with the flag, 141s and
  66 lines of DNS-retry without it.
- The HF cache directory is `HF_CACHE_DIR` (see `vectors/embed.hf_cache_dir()`), defaulting
  to ~/.cache/huggingface/hub. Anything that pre-caches weights must set it to the same
  directory it wrote to — the Dockerfile bakes via the app's own `load_embedding_model()`
  precisely so the two cannot diverge
- Deprecation warnings from sentence-transformers are harmless — do not attempt to fix them
- "No modules.json found" warning is harmless — Specter2 base does not have a modules.json

### Gap Scorer
- Only 1 gap returned at small corpus size is correct — frequency scores are low when denominator is small
- Never lower scoring weights to make demo look better — the formula (0.40/0.35/0.25) is core IP
- Need minimum ~50 papers with explicit limitations for meaningful ranked output
- solution_deficit_score near 0 with fewer than 20 papers is expected — future_directions too sparse to be selective
- Tier weights: explicit=1.0, conclusion=0.75, inferred=0.5 — never change these without updating tests

### Ollama
- Must be running before any extract_paper() call — ConnectionError otherwise
- "address already in use" on ollama serve means it is already running in background — not an error, proceed normally
- Cold start 500 error on first request is handled by retry logic — not a bug
- Always keep ollama serve running in a dedicated terminal tab throughout the session

### Ingestion Script
- Always use the direct arXiv ID ingestion script (not ingest_from_query) for curated paper lists — more reliable
- Always check existing papers first to avoid re-ingesting: existing = set(r['arxiv_id'] for r in db['papers'].rows)
- Always serialize with json.dumps inline: row[k] = json.dumps(v) for list fields before upsert

### Services Startup Order (Every Session)
1. Ollama — run ollama serve (ignore "address already in use" — means already running)
2. Docker Desktop — open app, wait for whale icon to stop animating
3. Qdrant — docker run -p 6333:6333 -v ~/research-gap-hunter/qdrant_storage:/qdrant/storage qdrant/qdrant
4. Neo4j Desktop — open app, start rgh-mvp instance, wait for green Active status
5. Working terminal — cd research-gap-hunter

### Git Hygiene
- Always commit before every Claude Code session — clean working tree = safe rollback
- Runtime artifacts that must stay in .gitignore: data/failed_extractions.log, data/papers.db, qdrant_storage/
- Never commit with misleading messages — Opus caught a commit that said "PDF extraction working" but only contained a failure log
- data/ directory should only have .gitkeep tracked, never actual database files
- All commits must be under vipulparmar3018@gmail.com — the local machine's
  default git email created a duplicate contributor entry on GitHub in the
  past. Check `git config user.email` at the start of a session if commits
  ever show up under an unexpected identity.

### ArXiv ID Verification
- Never trust an arXiv ID as correct just because it was generated or recalled
  by an LLM (including this file's own history, or Claude chat output) — IDs
  have previously pointed to unrelated or nonexistent papers.
- Before using any arXiv ID for ingestion, testing, or validation, confirm it
  independently against arxiv.org or the Semantic Scholar API — don't chain
  trust from one AI-generated list to the next.

### Verifying Claimed Project State
- Any statement in this file, in chat memory, or in a prior session summary
  about "N tests passing," "phase X is done," or "feature Y is working" is a
  claim, not a fact, until re-verified in the current session (`pytest -q`,
  `git log`, actually exercising the code path). Do not build new work on top
  of an unverified claim of prior completion.
