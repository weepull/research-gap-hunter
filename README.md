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

Research Gap Hunter inverts this. It ingests papers from arXiv, extracts structured limitation statements using a local LLM (Llama 3.1 8B via Ollama), and builds a graph + vector index over the entire corpus. A scoring engine then ranks research gaps by three independent signals: how frequently a limitation appears across papers, how recently it has been reported, and how few future-work suggestions from those same papers address it. The result is a ranked list of the most urgent, underserved open problems in the domain.

The third dimension is the most novel: cross-domain hypothesis generation. Specter2 embeddings are used to match unresolved limitations in computer vision against proposed solutions in medical imaging. When a CV paper reports "our method fails under domain shift" and a medical imaging paper proposes "domain-adaptive registration via learned deformation fields," Research Gap Hunter surfaces that connection and uses Ollama to generate a natural-language explanation of why the transfer is scientifically plausible. Researchers get concrete, cited hypotheses — not keyword lists.

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
| Tests | pytest | 176 tests, 0 failures; all backends mocked at module level |
| Python version | 3.11 | Required; type hints throughout |

---

## Screenshots

### Gap Explorer
> _Ranked research gaps with frequency, recency, and solution-deficit sub-scores. Color-coded score badges (green > 0.6, yellow 0.4–0.6, red < 0.4)._

`[screenshot: gap-explorer.png]`

### Semantic Search
> _500 ms debounced vector search over all extracted limitation statements. Results include cosine similarity score and clickable arXiv links._

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
```

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
# Ingest a batch of computer vision papers (arXiv IDs)
python3.11 -m pipeline.batch \
  --ids 2301.00234 2303.05499 2212.09748 \
  --domain computer_vision

# Or ingest medical imaging papers
python3.11 -m pipeline.batch \
  --ids 2106.08589 2206.07890 \
  --domain medical_imaging
```

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

**Neo4j over a relational database** — The core query patterns are graph traversals: "find all limitations reported by papers that use the same method" or "find future directions from papers citing the same dataset." These are `MATCH` paths in Cypher; they are multi-join `GROUP BY` nightmares in SQL. Neo4j also lets the discovery layer evolve — adding citation graphs, co-author networks, or dataset lineage requires new relationship types, not schema migrations.

**Seed-anchored clustering over HDBSCAN** — HDBSCAN produced one giant cluster with 64 limitations because transitive similarity (A≈B, B≈C → A,B,C merged even when A and C score 0.72). Seed-anchored grouping fixes this by anchoring every membership decision to the seed's similarity, not a transitive chain. It also batches all Specter2 embeddings in a single `model.encode()` call and uses Qdrant's `query_batch_points` for one-round-trip neighbour fetches — 27 clean clusters from 64 limitations at threshold 0.86.

**Separate thresholds for within-domain and cross-domain matching** — Specter2-base similarity scores on this corpus have a median of ~0.82 and a range of 0.68–0.93. Within-domain clustering needs a threshold of 0.86 to sit well above the median and avoid over-merging. Cross-domain matching uses 0.82 because different field vocabularies (CV vs. medical imaging) compress Specter2 scores further — the best CV↔MI pairs peak around 0.84, and 0.82 yields 10 meaningful matches vs. 1 at 0.84.

**Local LLM (Ollama) over API calls** — Zero cost, zero latency variance, no rate limits, and the extracted data stays local. On an M-series Mac, Llama 3.1 8B processes a full paper (8–12K tokens) in ~15 seconds. The 3-tier extraction strategy (explicit→conclusion→inferred, confidence weights 1.0/0.75/0.5) compensates for the weaker instruction-following of a 8B model relative to GPT-4.

**Specter2 over general-purpose embeddings** — Specter2 is trained on citation graphs and scientific text. General-purpose models (e.g., `text-embedding-ada-002`) treat "attention mechanism" and "transformer architecture" as distant; Specter2 places them adjacently because they co-appear in millions of scientific citations. This makes cross-domain similarity scores meaningful rather than incidentally high.

---

## Test Suite

```bash
python3.11 -m pytest tests/ -v
```

```
176 passed, 0 failed, 1 warning
```

All external dependencies (Ollama, Neo4j, Qdrant, SQLite, Semantic Scholar) are mocked at the module level. The suite runs in < 1 second on first invocation and has no network or service dependencies.

| Test file | Coverage |
|---|---|
| `test_extractor.py` | LLM prompt, JSON parsing, 3-tier fallback, Pydantic validation, failure logging |
| `test_batch.py` | Full ingestion pipeline, exponential backoff, deduplication, domain tagging |
| `test_graph.py` | Neo4j node creation, relationship upsert, duplicate handling |
| `test_vectors.py` | Qdrant collection init, batch upsert, cosine search, payload filtering |
| `test_gap_scorer.py` | Scoring formula, seed-anchored clustering (non-transitive, batch-embedded), threshold edge cases |
| `test_cross_domain.py` | Domain ingestion, gap filtering, cross-domain match ranking, Ollama explanation |
| `test_api.py` | All 7 endpoints, lifespan warming, CORS headers, 422/404/500 error paths |

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

- **Phase 5 — Scale to 500 papers** using Semantic Scholar bulk API; parallelize ingestion with async workers
- **Phase 6 — GROBID integration** for structured section extraction (methods, results, limitations) replacing the LLM prompt — faster, cheaper, more consistent
- **Phase 7 — Additional domains** (NLP, robotics, materials science); domain auto-detection from paper abstract
- **Phase 8 — Citation graph overlay** in Neo4j; weight gap scores by citing-paper age to surface problems that are being abandoned vs. gaining attention
- **Phase 9 — Patent corpus** integration; cross-reference academic limitations against granted patents to find commercially-solved but academically-unacknowledged gaps
- **Phase 10 — Public deployment** on Railway (API) + Vercel (frontend) with read-only demo mode and authenticated ingestion

---

## License

MIT © 2026 Vipul Parmar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
