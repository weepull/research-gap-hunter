# Research Gap Hunter — CLAUDE.md

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
│   ├── batch.py               ← batch ingestion runner
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
```

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

- `frequency_score` = papers_reporting_limitation / total_papers_in_domain
- `recency_score` = papers_last_2yr_reporting / papers_all_time_reporting
- `solution_deficit_score` = 1 - (future_directions_addressing / papers_reporting),
  where "addressing" means same-domain **and** not authored by a paper that reports
  the limitation — see the decision below

Seed-anchored union-find clustering groups similar limitation statements before scoring (HDBSCAN was the original design but was replaced — do not reintroduce it without an explicit advisor decision). Cluster representative text is used as `gap_description`.

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

Always use this exact prompt structure. Do not modify without updating this file:

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

Paper text:
{paper_text}
```

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
Railway/Vercel/AuraDB/Qdrant Cloud, Kaggle-related work) is tracked
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

### Semantic Scholar API
- Free tier = 1 req/sec — always time.sleep(2) between individual paper fetches, time.sleep(5) between batch queries
- API returns abstracts only — full paper text must come from arXiv PDF via fetch_full_text()
- 429 errors after 3 retries log to data/failed_extractions.log and continue — correct behaviour, do not crash
- data/failed_extractions.log must stay in .gitignore — never commit runtime logs
- 3 failed papers per batch of 20 is normal — do not retry immediately, wait 60s for rate limit reset

### HuggingFace / Specter2
- Model re-downloads on every process unless cache_dir is explicitly set in model_kwargs, processor_kwargs, config_kwargs
- Do not set HF_HUB_OFFLINE=1 — breaks fresh installs on new machines
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
