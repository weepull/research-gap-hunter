# Research Gap Hunter

**AI-powered scientific discovery — surfaces what should be researched next, not what has been.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![Tests](https://img.shields.io/badge/tests-176_passed-22c55e?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-6366f1?style=flat-square)

---

## What It Does

Academic literature is growing at a rate no researcher can track. A computer vision researcher today must sift through thousands of papers per year just to understand which problems remain unsolved — and that work is manual, biased toward papers the researcher already knows about, and blind to solutions that already exist in adjacent fields.

Research Gap Hunter inverts this. It ingests papers from arXiv, extracts structured limitation statements using a local LLM (Llama 3.1 8B via Ollama), and builds a graph + vector index over the ingested corpus. A scoring engine then ranks research gaps by three independent signals: how frequently a limitation appears across papers, how recently it has been reported, and how few future-work suggestions from those same papers address it. The result is a ranked list of candidate open problems, each shown with the number of papers supporting it.

> **Scope — read this before interpreting any output.** This is a working research
> prototype over a **curated sample of 149 papers** (64 computer vision, 85 medical
> imaging), not a survey of either field. Treat the rankings as a demonstration of the
> method, not as findings about the state of either field.
>
> **The quality of the output has not been measured.** The mechanism is extensively
> tested — 515 unit tests and 37 integration tests against live Neo4j, Qdrant and Ollama —
> but no one has yet judged whether the gaps it surfaces are real. `eval/` contains the
> apparatus for that (a blinded label sheet, two baselines, precision@k with Wilson
> intervals) and it is **awaiting human labels**. Running `eval/score.py` today prints
> "NO LABELS PRESENT" rather than a number, deliberately.
>
> All figures in this README come from scripts you can re-run; see
> [Measured state](#measured-state).

The third dimension is the most novel: cross-domain hypothesis generation. Specter2 embeddings are used to match unresolved limitations in computer vision against proposed solutions in medical imaging. When a CV paper reports "our method fails under domain shift" and a medical imaging paper proposes "domain-adaptive registration via learned deformation fields," Research Gap Hunter surfaces that connection and uses Ollama to generate a natural-language explanation of why the transfer is scientifically plausible. Researchers get concrete, cited hypotheses — not keyword lists.

---

## Measured state

Every number here is produced by a script in `scripts/` or `eval/`, not written by hand.
Regenerate with `bash scripts/nightly_eval.sh`. Last measured 2026-09-29.

### Corpus

| | computer_vision | medical_imaging |
|---|---:|---:|
| papers | 64 | 85 |
| papers contributing ≥1 limitation | **48** | **50** |
| limitation nodes | 112 | 103 |
| gap clusters | 57 | 53 |
| largest cluster | 8 | 16 |

`contributing` is the divisor of `frequency_score`. It is **not** the corpus size: a paper
that extracted no limitations cannot corroborate a gap, and counting it made frequency the
product of two unrelated things. `/corpus` reports both.

### Ranking tiers and ties

| | computer_vision | medical_imaging |
|---|---:|---:|
| gaps | 57 | 53 |
| corroborated (≥2 papers) | 21 | 16 |
| single source | 36 | 37 |
| gaps sharing a score | **4** | **3** |
| distinct scores | 55 | 51 |

Corroborated gaps rank above all single-source gaps, so **scores are non-monotonic across
the tier boundary** — the UI shows a heading at the boundary for that reason. Before the
A9 Option F work, 17 of 24 CV gaps and 38 of 53 MI gaps shared a score.

### Thresholds — all derived, none hand-picked

Derived from measured null distributions by `scripts/derive_thresholds.py`; rewritten only
when drift exceeds `DRIFT_TOLERANCE = 0.002`, and only by that script.
`tests/integration/test_threshold_derivation.py` fails if any constant drifts further.

| constant | computer_vision | medical_imaging |
|---|---:|---:|
| `_CLUSTER_THRESHOLDS` (limitation × limitation p95) | 0.8744 | 0.8954 |
| `_SOLUTION_THRESHOLDS` (limitation × future-direction p95) | 0.8733 | 0.8915 |
| `_DEFICIT_RESCALE_ANCHORS` (p50, p99) | (0.8235, 0.8959) | (0.8385, 0.9127) |
| `_CROSS_DOMAIN_THRESHOLD` (pooled cross-domain p95) | 0.8764 | — |

`_UNRESOLVED_DEFICIT_FLOOR = 0.3` is the **one threshold still not derived from anything**.
It was harmless while the deficit term took four discrete values; now that the deficit is
continuous it genuinely selects which gaps reach cross-domain matching, and it needs a
decision. Recorded rather than quietly left.

### Extraction quality

| | value |
|---|---:|
| filter rejection rate on the pre-fix corpus (one-off backfill) | 4.3% (10 of 234) |
| **filter rejection rate on papers ingested with the fixed prompt** | **0.0%** |
| ingestion outcomes | 26 ok, 7 pdf_failed, 3 no_limitations, 3 extraction_failed |
| candidates refused by the corpus rules | 17 |

The 0% is the point: the extraction prompt used to quote its own cue phrases
(`'remains challenging'`, `'future work includes'`) and the model echoed them back as
findings. With those examples removed from the prompt, newly ingested papers produced no
boilerplate for the filter to catch. The filter remains as a second line of defence.

### Footprint

`scripts/measure_footprint.py`: **472.3 MB** peak RSS with Specter2 resident, against a
512 MB tier — **+39.7 MB headroom, fits without margin**. The corpus vectors are 0.66 MB
and live in Qdrant, not in the API process, so corpus growth barely moves this. The
binding constraint is the model, a fixed cost.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │         Paper Ingestion              │
                         │  arXiv ID / PDF                     │
                         └───────────────┬─────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  PyMuPDF Extraction  │
                              │  Page-aware, last    │
                              │  40% of PDF first    │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Llama 3.1 8B        │
                              │  (Ollama, local)     │
                              │  3-tier extraction:  │
                              │  explicit / concl. / │
                              │  inferred            │
                              └──────────┬──────────┘
                                         │
                         ┌───────────────▼──────────────────┐
                         │           SQLite                  │
                         │  papers.db — raw structured store │
                         └──────┬─────────────┬─────────────┘
                                │             │
               ┌────────────────▼──┐   ┌──────▼──────────────────┐
               │     Neo4j Graph    │   │  Specter2 Embeddings     │
               │  Paper, Limitation,│   │  (allenai/specter2_base) │
               │  FutureDirection,  │   │  768-dim, local          │
               │  Method, Dataset   │   └──────┬──────────────────┘
               │  nodes +           │          │
               │  relationships     │   ┌──────▼──────────────────┐
               └────────────────────┘   │        Qdrant            │
                                        │  collections:            │
                                        │  • limitations           │
                                        │  • future_directions     │
                                        └──────┬──────────────────┘
                                               │
                              ┌────────────────▼──────────────────┐
                              │           Gap Scorer               │
                              │  Seed-anchored clustering          │
                              │  score = 0.40×freq + 0.35×recency │
                              │         + 0.25×solution_deficit    │
                              └──────────┬────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Cross-Domain Matcher│
                              │  CV gaps ↔ MI        │
                              │  solutions via cosine│
                              │  similarity ≥ 0.82   │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │              FastAPI (port 8000)          │
                    │  /health /gaps /search /cross-domain      │
                    │  /ingest /paper/{id} /explain             │
                    └────────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │         Next.js 16 Frontend (port 3000)  │
                    │  Gap Explorer · Semantic Search ·         │
                    │  Cross-Domain Discovery                   │
                    └──────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF extraction | PyMuPDF (`fitz`) | Fast, page-aware text extraction; scans last 40% of PDF first to find conclusions and future-work sections |
| Extraction LLM | Llama 3.1 8B via Ollama | Local, free inference; structured JSON output with 3-tier confidence weighting |
| Raw storage | SQLite via `sqlite-utils` | Lightweight structured store for all paper fields; zero infrastructure |
| Knowledge graph | Neo4j Desktop (`rgh-mvp`) | Relationship-aware queries across Paper → Limitation → FutureDirection → Method → Dataset |
| Embeddings | `allenai/specter2_base` via `sentence-transformers` | Scientific paper embeddings; 768 dimensions; loaded from local HuggingFace cache |
| Vector store | Qdrant | ANN search over `limitations` and `future_directions` collections; cosine distance |
| Gap clustering | Seed-anchored grouping | Membership anchored to seed similarity (not transitive chains); threshold 0.86 within-domain |
| Cross-domain matching | Specter2 + Qdrant | CV gap descriptions queried against MI future-direction vectors; threshold 0.82 |
| Explanation LLM | Llama 3.1 8B via Ollama | Generates natural-language hypothesis explanations; temperature 0.2 |
| API | FastAPI + Uvicorn | 7 endpoints; Pydantic response models; lifespan model warming; CORS |
| Frontend | Next.js 16 + TypeScript + Tailwind CSS v4 | 3 pages; dark theme; server + client components; Inter font |
| Paper source | Semantic Scholar API | arXiv metadata, PDF URLs, open access links |
| Tests | pytest | 515 unit tests (hermetic, enforced by a guard in `tests/conftest.py`) + 40 opt-in integration tests against real services |
| Python version | 3.11+ | `requires-python = ">=3.11"`; developed on 3.14; type hints throughout |

---

## Screenshots

### Gap Explorer
> _Ranked research gaps with frequency, recency, and solution-deficit sub-scores. Color-coded score badges (green > 0.6, yellow 0.4–0.6, red < 0.4)._

`[screenshot: gap-explorer.png]`

### Semantic Search
> _500 ms debounced vector search over the extracted limitation statements in the ingested corpus. Results include cosine similarity score and clickable arXiv links._

`[screenshot: semantic-search.png]`

### Cross-Domain Discovery
> _The hero page. Computer vision open problems matched to medical imaging proposed solutions. Per-card LLM explanation generated on demand._

`[screenshot: cross-domain.png]`

---

## Quick Start

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Neo4j Desktop | 5.x | [neo4j.com/download](https://neo4j.com/download/) |
| Ollama | latest | [ollama.com](https://ollama.com) |
| Docker | 24+ | [docker.com](https://www.docker.com) — for Qdrant |

### 1. Clone and install

```bash
git clone https://github.com/weepull/research-gap-hunter.git
cd research-gap-hunter

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cd frontend && npm install && cd ..
```

### 2. Environment setup

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```env
SEMANTIC_SCHOLAR_API_KEY=your_key_here      # get free key at semanticscholar.org
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# Optional — managed deployments
# QDRANT_URL=https://xxx.cloud.qdrant.io:6333   # takes precedence over HOST/PORT
# QDRANT_API_KEY=your_qdrant_cloud_key
# NEO4J_URI=neo4j+s://xxx.databases.neo4j.io    # AuraDB requires neo4j+s://, not bolt://
# NEO4J_DATABASE=neo4j
# RATE_LIMIT_ENABLED=true                        # set false to disable API rate limiting

# Optional — public demo mode. Leave unset for local development.
# DEMO_MODE=true                                 # disables /ingest and /explain (403)
# ALLOWED_ORIGINS=https://your-frontend.vercel.app   # required when DEMO_MODE=true; CORS fails closed without it
```

**Demo mode.** `DEMO_MODE=true` marks a deployment as publicly reachable and
changes three things: `/ingest` returns 403, `/explain` returns 403, and CORS is
restricted to `ALLOWED_ORIGINS` (refusing everything if that is unset). Both
endpoints are refused for the same reason — each costs real money per anonymous
request, ingestion through the Semantic Scholar quota and a 30-60s LLM job,
explanations through an LLM call. There is deliberately **no hosted-LLM
fallback**: the demo shows the pre-computed discovery output, not live
generation. Leave `DEMO_MODE` unset locally and nothing changes — ingestion and
explanations both work against Ollama, no API key required anywhere.

### 3. Start services

**Ollama** (pull the model once):
```bash
ollama pull llama3.1:8b
ollama serve          # runs at localhost:11434
```

**Qdrant** via Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Neo4j Desktop**: open the app, create a project named `rgh-mvp`, and start the database. Set the password to match your `.env`.

### 4. Ingest papers

```bash
# Ingest a tranche of candidates discovered through the arXiv API.
# --domain is REQUIRED and validated; there is deliberately no default.
python -m scripts.ingest_tranche --domain computer_vision --size 50
python -m scripts.ingest_tranche --domain medical_imaging --size 50

# Quality report for what was ingested
python scripts/ingest_tranche.py --domain computer_vision --report-only

# Many tranches with threshold re-derivation after each
python scripts/run_tranches.py --hours 3 --size 50 --target-per-domain 300
```

Candidates are gated at ingest by `pipeline/corpus_rules.py`, on arXiv's own **primary
category**:

- `computer_vision` — primary must be `cs.CV`. Cross-listing to `cs.CV` is not enough.
- `medical_imaging` — primary in `{eess.IV, physics.med-ph}`, **or** primary `cs.CV` with a
  medical keyword match.
- anything else is logged as `rule_rejected`, unless the id is in `data/mi_allowlist.txt`.

Measured acceptance: of 50 `eess.IV` candidates, 34 are accepted for medical imaging; of 50
`cs.CV` candidates, 6 are (the medical ones). Ingestion runs at **~160 s per paper** —
arXiv paging, PDF download, Ollama extraction — so a 50-paper tranche is roughly 2 hours.
It checkpoints after every paper and skips already-ingested ids, so it resumes safely.

> The previous version of this section documented
> `python3.11 -m pipeline.batch --ids ... --domain ...`. That command **silently did
> nothing**: `pipeline/batch.py` has no `__main__` block, so `-m` merely imported it and
> exited. The arXiv ids it listed were also never verified against arXiv.

### 5. Start the API

```bash
PYTHONPATH=. python3.11 api/main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 6. Start the frontend

```bash
cd frontend
npm run dev
# → http://localhost:3000
```

---

## Project Structure

```
research-gap-hunter/
├── api/
│   └── main.py                  # FastAPI app — 7 endpoints, lifespan model warming
├── pipeline/
│   ├── extractor.py             # extract_paper(arxiv_id) → PaperExtract via Ollama
│   ├── batch.py                 # bulk ingestion: fetch → extract → SQLite → graph → vectors
│   ├── gap_scorer.py            # score_gaps() → ranked GapResult list; seed-anchored clustering
│   └── cross_domain.py          # find_cross_domain_matches(); ingest_domain_papers(); explain_match()
├── graph/
│   └── populate.py              # SQLite → Neo4j; Paper/Limitation/FutureDirection nodes
├── vectors/
│   ├── embed.py                 # embed_limitations(), embed_future_directions() → Qdrant upsert
│   └── search.py                # find_similar_limitations(query) → vector search
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Gap Explorer — ranked cards, sub-score bars, domain/topN controls
│   │   ├── search/page.tsx      # Semantic Search — debounced input, similarity-ranked results
│   │   └── cross-domain/page.tsx# Cross-Domain Discovery — connection cards, on-demand LLM explain
│   ├── components/
│   │   └── Nav.tsx              # Sticky nav, active-state highlighting
│   └── lib/
│       └── api.ts               # Typed fetch client for all API endpoints
├── tests/
│   ├── test_extractor.py        # LLM extraction + 3-tier confidence weighting
│   ├── test_batch.py            # Batch ingestion, retry logic, failure logging
│   ├── test_graph.py            # Neo4j node/relationship population
│   ├── test_vectors.py          # Qdrant upsert, search, collection creation
│   ├── test_gap_scorer.py       # Scoring formula, seed-anchored clustering, thresholds
│   ├── test_cross_domain.py     # Domain ingestion, gap retrieval, cross-domain matching
│   └── test_api.py              # All 7 endpoints; lifespan; CORS; error states
├── data/
│   ├── papers.db                # SQLite — gitignored, created at runtime
│   └── failed_extractions.log   # Extraction failures — gitignored, created at runtime
├── pyproject.toml               # Dependencies + pytest config
└── CLAUDE.md                    # Architecture contract for AI-assisted development
```

---

## Key Technical Decisions

**PyMuPDF over pdfplumber** — PyMuPDF (`fitz`) is 3–5× faster and handles the column-layout PDFs common in CV conferences (CVPR, ECCV, ICCV) without splitting mid-sentence. It also exposes page numbers, which the page-aware extraction strategy depends on: the extractor searches the last 40% of the document first (where limitations and future work live), cutting extraction time and improving yield.

**Neo4j over a relational database** — The core query patterns are graph traversals: "find all limitations reported by papers that use the same method" or "find all future directions from papers sharing a dataset." These are `MATCH` paths in Cypher; they are multi-join `GROUP BY` nightmares in SQL. Neo4j also lets the discovery layer evolve — adding citation graphs, co-author networks, or dataset lineage would require new relationship types rather than schema migrations.

> Note: citation traversal is **not implemented**. The schema defines a `CITES` relationship but the graph currently contains zero of them, so the citation-graph argument above is a statement of intent, not of current capability. Whether Neo4j earns its place at this corpus size is an open question — see `PROJECT_HARDENING_PLAN.md`.

**Seed-anchored clustering over HDBSCAN** — HDBSCAN produced one giant cluster with 64 limitations because transitive similarity (A≈B, B≈C → A,B,C merged even when A and C score 0.72). Seed-anchored grouping fixes this by anchoring every membership decision to the seed's similarity, not a transitive chain. It also batches all Specter2 embeddings in a single `model.encode()` call and uses Qdrant's `query_batch_points` for one-round-trip neighbour fetches — 27 clean clusters from 64 limitations at threshold 0.86.

**Separate thresholds for within-domain and cross-domain matching** — Specter2-base similarity scores on this corpus have a median of ~0.82 and a range of 0.68–0.93. Within-domain clustering needs a threshold of 0.86 to sit well above the median and avoid over-merging. Cross-domain matching uses 0.82 because different field vocabularies (CV vs. medical imaging) compress Specter2 scores further — the best CV↔MI pairs peak around 0.84.

> **Disputed — do not cite this number.** A null-distribution analysis (2026-08-22) found 0.82 sits *below the median of random cross-domain pairs*: 61.6% of 1,490 randomly paired CV/MI items clear it, and the 95th percentile of pure noise is 0.8792. The "10 meaningful matches" claim above has not survived that test. The threshold is pending an advisor decision — see item A2 in `PROJECT_HARDENING_PLAN.md`.

**Local LLM (Ollama) over API calls** — Zero cost, zero latency variance, no rate limits, and the extracted data stays local. On an M-series Mac, Llama 3.1 8B processes a full paper (8–12K tokens) in ~15 seconds. The 3-tier extraction strategy (explicit→conclusion→inferred, confidence weights 1.0/0.75/0.5) compensates for the weaker instruction-following of a 8B model relative to GPT-4.

**Specter2 over general-purpose embeddings** — Specter2 is trained on citation graphs and scientific text. General-purpose models (e.g., `text-embedding-ada-002`) treat "attention mechanism" and "transformer architecture" as distant; Specter2 places them adjacently because they co-appear in millions of scientific citations. This makes cross-domain similarity scores meaningful rather than incidentally high.

---

## Test Suite

```bash
pytest                  # unit tier: hermetic, no services needed
pytest -m integration   # integration tier: real Neo4j, Qdrant, Ollama
pytest -m ''            # both
```

```
515 passed, 40 deselected, 1 warning in 1.10s
37 passed, 3 skipped, 515 deselected, 1 warning in 37.47s
```

Both lines were measured with Neo4j, Qdrant and Ollama all running. The unit figure is
identical with all three stopped (as measured at 489 tests, before P2 added 26), which is
the point. An autouse guard in `tests/conftest.py` refuses, in the unit tier, any real
Neo4j driver, Qdrant client, embedding-model load, or SQLite file outside pytest's temp
directory. It also fails the test at teardown when the code under test swallowed the refusal.

Before the guard existed (2026-10-04), the earlier `489 passed` depended on which services
were running:
- eight `score_gaps` tests and two `/corpus` tests queried the production Neo4j;
- 62 API tests opened the real `data/papers.db` through startup self-heal.

With Neo4j down, the same suite gave 8 failed, 481 passed.

**Two tiers, and the split matters.** The unit tier mocks every external dependency and
runs in under a second, which is what makes it usable on every change. It is also
*structurally incapable* of catching the defects that mattered most here — a cluster
swallowing half a domain, a threshold below its own noise floor, a `/health` endpoint
reporting `ok` with Neo4j down all look fine to a mock. The integration tier exists for
those, runs against live services, and **skips rather than fails** when they are absent.

Integration tests use an isolated Neo4j database (`rghintegration`), separate Qdrant
collections, and a `tmp_path` SQLite, so a run can never touch the real corpus.

| Test file | Coverage |
|---|---|
| `test_extractor.py` | LLM prompt, JSON parsing, 3-tier fallback, Pydantic validation, failure logging |
| `test_batch.py` | Full ingestion pipeline, exponential backoff, deduplication, domain tagging |
| `test_graph.py` | Neo4j node creation, relationship upsert, duplicate handling |
| `test_vectors.py` | Qdrant collection init, batch upsert, cosine search, payload filtering |
| `test_gap_scorer.py` | Scoring formula, seed-anchored clustering (non-transitive, batch-embedded), threshold edge cases |
| `test_cross_domain.py` | Domain ingestion, gap filtering, cross-domain match ranking, Ollama explanation |
| `test_api.py` | All 7 endpoints, per-service health probes, 503-on-degraded, lifespan resilience, CORS, 422/404/500 paths |
| `test_domains.py` | Required-domain validation, the keyword verifier, and that it never assigns a domain |
| `test_extraction_filter.py` | Prompt-echo / hedging / length / duplicate gates, the echo-list guards, audit logging |
| `test_corpus_rules.py` | arXiv feed parsing, the CV primary-category rule, the MI two-primary rule, the allowlist |
| `test_eval_harness.py` | Wilson intervals, sheet round-trip, and the scorer's refusal to read LLM triage |
| `tests/integration/` | Domain ground truth, cluster caps, frequency denominator, tier ordering, `/explain` grounding, per-service health, threshold derivation |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + paper/vector counts |
| `GET` | `/gaps?domain&top_n` | Ranked research gaps with sub-scores |
| `GET` | `/search?q&top_k&domain` | Vector search over limitation statements |
| `GET` | `/cross-domain?source&target&top_n` | Cross-domain hypothesis matches |
| `GET` | `/explain?source_gap&target_solution` | LLM explanation for a gap↔solution pair |
| `POST` | `/ingest` | Ingest a single paper by arXiv ID |
| `GET` | `/paper/{arxiv_id}` | Raw extracted fields for a paper |

Interactive docs at `http://localhost:8000/docs` when the API is running.

---

## Roadmap

> **Everything in this section is planned, not built.** None of it is live today.
> Phase numbering was retired — it implied a linear plan that no longer matches
> reality. Current priorities and blockers live in `PROJECT_HARDENING_PLAN.md`;
> the items below are directional, not scheduled, and carry no delivery date.

### Shipped (for contrast)

- **Demo mode** — `DEMO_MODE=true` disables `/ingest` and `/explain` (403, with
  a message explaining why) and restricts CORS to `ALLOWED_ORIGINS`. Per-endpoint
  rate limiting is active in both modes. The project calls no paid API.

### Planned — corpus and extraction

- **Scale to 500 papers** using the Semantic Scholar bulk API; parallelize ingestion with async workers
- **GROBID integration** for structured section extraction (methods, results, limitations) replacing the LLM prompt — faster, cheaper, more consistent
- **Additional domains** (NLP, robotics, materials science); domain auto-detection from paper abstract
- **Citation graph overlay** in Neo4j; weight gap scores by citing-paper age to surface problems being abandoned vs. gaining attention. The schema defines a `CITES` relationship but the graph currently contains **zero** of them
- **Patent corpus** integration; cross-reference academic limitations against granted patents to find commercially-solved but academically-unacknowledged gaps

### Planned — deployment and access

- **Public deployment** on Render (API) + Vercel (frontend), running in demo mode
- **Authenticated ingestion** so trusted users can add papers to a deployed instance — today `/ingest` is simply disabled in demo mode, with no auth layer of any kind
- **Usage-based pricing via Razorpay** for high-volume programmatic access.
  **Not implemented — there is no billing code, no payment integration, no
  metering, and no paid tier in this repository.** It is a direction under
  consideration, recorded here so the intent is on the record rather than
  implied. The project is MIT-licensed and free to self-host; any future paid
  tier would apply only to a hosted instance.

---

## License

MIT © 2026 Vipul Parmar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
