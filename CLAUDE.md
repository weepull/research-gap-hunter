# Research Gap Hunter — CLAUDE.md

## What This Project Is

Research Gap Hunter is an AI-powered scientific discovery platform. It does NOT function as a search engine or retrieval tool. Its purpose is to answer "what should be done next?" — not "what has been done?"

The system ingests academic papers, extracts structured information using a local LLM, builds a knowledge graph, embeds limitation statements as vectors, and runs a discovery engine that surfaces ranked research gaps and cross-domain hypothesis matches.

MVP domain: Computer Vision (CV papers only for Phase 1–4)

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
Gap scoring + HDBSCAN clustering → ranked GapResult list
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
    similarity_score: float      # cosine similarity, threshold 0.78
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

### `future_directions` (Phase 4+)
- Same structure as limitations
- Used for cross-domain matching

---

## Gap Scoring Formula

```
score = (0.40 × frequency_score) + (0.35 × recency_score) + (0.25 × solution_deficit_score)
```

- `frequency_score` = papers_reporting_limitation / total_papers_in_domain
- `recency_score` = papers_last_2yr_reporting / papers_all_time_reporting
- `solution_deficit_score` = 1 - (future_directions_addressing / papers_reporting)

HDBSCAN clusters similar limitation statements before scoring. Cluster centroid text is used as `gap_description`.

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

## Rules for Claude Code Sessions

1. **One module per session.** Never work on multiple files across layers simultaneously.
2. **Always read this CLAUDE.md at the start of every session** before writing any code.
3. **Never change the stack.** No swapping Neo4j for another DB, no changing Qdrant collection names, no switching embedding models without explicit instruction.
4. **Always write pytest tests** for every function you implement.
5. **Never hardcode secrets.** All credentials come from `.env` via `python-dotenv`.
6. **Validate Pydantic models strictly.** If LLM output fails validation, log the failure to `data/failed_extractions.log` and continue — do not crash.
7. **Use exponential backoff** on all external API calls (Semantic Scholar).
8. **Do not install packages** not listed in `pyproject.toml` without updating it first.

---

## Current Phase

**Phase 1 — Extraction Pipeline**

Goal: `extract_paper(arxiv_id: str) -> PaperExtract` working on real CV papers.

Test paper IDs to validate against:
- `2301.00234` — object detection
- `2303.05499` — image segmentation  
- `2212.09748` — vision transformers

---

## MVP Scope (Do Not Expand Until Phase 5 Is Complete)

- Domains: Computer Vision only (Phase 1–3), Computer Vision + Medical Imaging (Phase 4)
- Paper volume: 50 papers (Phase 1), 500 papers (Phase 3 scale-up)
- Frontend: 3 pages only — gaps, search, cross-domain
- No user auth, no cloud deployment of backend — demo mode only
