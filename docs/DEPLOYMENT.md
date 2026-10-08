# Deployment

How to take Research Gap Hunter live on Render (backend) + Vercel (frontend) +
Neo4j AuraDB (graph) + Qdrant Cloud (vectors).

All four have a genuinely free tier and none requires a credit card.

**Nothing here has been deployed.** No accounts were created and no credentials
were obtained on your behalf — every step below that needs a real credential is
yours to do. This document is the runbook for doing it.

Read the whole thing once before starting. The order matters: the two databases
must exist and hold data before the API can start, and the API must have a
public URL before the frontend can be built.

---

## 0. What you are deploying, and what it will cost

| Piece | Where | Free tier | Notes |
|---|---|---|---|
| FastAPI backend | Render | Yes, permanent, no card | Spins down when idle; **no disk**; 512MB RAM |
| Next.js frontend | Vercel | Yes, Hobby | Personal projects only on Hobby |
| Neo4j graph | Neo4j AuraDB | Yes, AuraDB Free | **Pauses after 3 days idle, deleted after 30** |
| Qdrant vectors | Qdrant Cloud | Yes, 1GB cluster | Enough for this corpus by a wide margin |

Four limits worth knowing before you start. None of them blocks a demo, but all
four will surprise you if you meet them cold:

- **Render free services spin down after 15 minutes of inactivity.** The next
  request wakes the container, and that takes roughly **30–60 seconds**. The
  first visitor after a quiet period waits; everyone after them does not.
- **Render free has no persistent disk, and this deployment does not ask for
  one.** SQLite is rebuilt from Neo4j on every boot by the startup self-heal.
  That is by design here — see §4.
- **Render free gives 512MB of RAM, and this app measured 476MB at peak.** It
  fits, with about 36MB to spare. See the memory note in §4 before you assume
  headroom.
- **AuraDB Free auto-pauses after 3 days of inactivity** and is **deleted after
  30 days** of inactivity. A demo nobody visits for a month is gone. If this
  needs to stay up, budget for AuraDB Professional or plan to re-seed.

The deployed app runs in **demo mode**: `/ingest` and `/explain` return 403.
That is deliberate and load-bearing — both need a local `llama3.1:8b`, which has
no managed equivalent, and there is deliberately no hosted-LLM fallback because
a per-request paid API call for anonymous callers is an open-ended cost. The
frontend already renders both refusals as information rather than as errors.

**Everything else works in demo mode**: ranked gaps, semantic search,
cross-domain matching, the corpus banner, and every paper detail.

---

## 1. Accounts you need to create

Create these first. All four have a free tier; none needs a card up front except
Render.

1. **GitHub** — you have this; the repo is already at `weepull/research-gap-hunter`.
2. **Neo4j Aura** — <https://console.neo4j.io>
3. **Qdrant Cloud** — <https://cloud.qdrant.io>
4. **Render** — <https://render.com> (sign in with GitHub; no card required)
5. **Vercel** — <https://vercel.com> (sign in with GitHub)

---

## 2. Neo4j AuraDB

### Create the instance

1. <https://console.neo4j.io> → **New Instance** → **AuraDB Free**.
2. Name it `rgh-mvp` (the name is cosmetic; it is **not** the database name).
3. **Download the credentials file when it is offered. This is the only time
   the password is shown.** It contains:
   - `NEO4J_URI` — looks like `neo4j+s://a1b2c3d4.databases.neo4j.io`
   - `NEO4J_USERNAME` — `neo4j`
   - `NEO4J_PASSWORD` — a generated string
4. Wait for the instance to show **Running**.

> The `+s` in `neo4j+s://` is TLS and is required. A plain `bolt://` URI will
> fail to connect to Aura.

### Find the database name — do not assume it is `neo4j`

Older Aura Free instances name the database `neo4j`. Instances created more
recently name it **after the instance id** instead, e.g. `ca34cbce`. This was hit
on the real instance: authentication succeeds and the connection opens, then
every query fails with

```
Unable to get a routing table for database 'neo4j' because this database does not exist
```

which reads like a credentials problem and is not one. Read the real name:

```bash
python - <<'PY'
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URI"],
                         auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with d.session(database="system") as s:
    for r in s.run("SHOW DATABASES").data():
        if r.get("type") == "standard":
            print("NEO4J_DATABASE =", r["name"])
d.close()
PY
```

Use that value for `NEO4J_DATABASE` everywhere — locally when loading data, and
in Render's environment. `render.yaml` prompts for it rather than presetting it,
for exactly this reason.

### Load your data into it

Your graph currently lives in local Neo4j Desktop with **127 Paper nodes**. Aura
starts empty. Two options:

**Option A — re-run the population step against Aura (simplest).** Your local
`data/papers.db` already holds all 127 papers, and `graph/populate.py` builds
the graph from SQLite. Point it at Aura and run it:

```bash
cd research-gap-hunter
NEO4J_URI="neo4j+s://<your-id>.databases.neo4j.io" \
NEO4J_USER="neo4j" \
NEO4J_PASSWORD="<your-password>" \
NEO4J_DATABASE="<the name you just read, NOT necessarily neo4j>" \
uv run python -c "from graph.populate import populate_graph; print(populate_graph())"
```

This reads SQLite and writes Neo4j. It makes **no LLM and no Semantic Scholar
calls**, so it is safe to re-run. Expect it to report several hundred nodes and
relationships created.

**Option B — dump and restore.** Aura Free does not accept `neo4j-admin`
database dumps. Use Option A.

### Footprint on a 512MB tier — measured 2026-09-29

Run `python scripts/measure_footprint.py` to reproduce. These are measured values, not
estimates:

| | measured |
|---|---:|
| process RSS before loading the model | 92.8 MB |
| process RSS with Specter2 resident | **472.3 MB** |
| attributable to the model | +379.5 MB |
| headroom against a 512MB cap | **+39.7 MB** |
| Qdrant vectors (224 points × 768 × float32) | 0.66 MB |
| HuggingFace model cache on disk | 948.6 MB |

**Verdict: it fits, but without margin.** Under 100MB of headroom a concurrent request or
a larger encode batch can still push it over, and the process is killed rather than
degraded.

**The binding constraint is a fixed cost, which changes what questions are worth asking.**
The embedding model is ~380MB resident for the life of the process, because every query
embeds text. The corpus vectors are **0.66 MB** and live in Qdrant, not in the API
process. So:

- **"Will a bigger corpus fit?" is the wrong question.** Corpus growth barely moves the
  API's memory at all — it moves Qdrant's, which is a separate service and three orders of
  magnitude below the cap. Ten thousand limitations would be ~30MB of vectors.
- **What would actually change the answer** is the model: a smaller embedding model, or
  moving embedding out of the API process behind a queue. Neither is a corpus decision.
- **The 948MB HuggingFace cache is a disk and image-size concern, not a memory one.** It is
  baked into the image (see the offline-load section above); only the loaded weights are
  resident.

If the tier is ever exceeded, the symptom will be an OOM kill on a request that happened to
arrive during an encode, not a gradual slowdown — so treat the 39.7MB as a real operating
limit rather than a comfortable buffer.

### Verify

In the Aura console's **Query** tab:

```cypher
MATCH (p:Paper) RETURN count(p)
```

You want **127**. This load has been performed and verified against the live
instance; a correct result looks exactly like this:

| Node | Count | | Relationship | Count |
|---|---|---|---|---|
| Paper | 127 | | REPORTS_LIMITATION | 168 |
| Limitation | 166 | | SUGGESTS_FUTURE | 94 |
| FutureDirection | 94 | | USES_DATASET | 199 |
| Method | 353 | | USES_METHOD | 355 |
| Dataset | 197 | | | |

937 nodes and 816 relationships in total, with Paper splitting 46 computer
vision / 81 medical imaging — the same split `/corpus` reports.

Note `Limitation` is **166** while `REPORTS_LIMITATION` is **168**: two
limitation texts are reported by more than one paper, and `MERGE` on exact text
dedupes the node while keeping both relationships. That is correct, not a
shortfall.

`populate_graph()` reads SQLite and writes Neo4j; it makes no LLM or Semantic
Scholar calls, so re-running it is safe and idempotent.

---

## 3. Qdrant Cloud

### Create the cluster

1. <https://cloud.qdrant.io> → **Create Cluster** → **Free tier**.
2. Pick the region nearest your Render region (the blueprint defaults to
   `oregon`; change it in `render.yaml` if you pick differently).
3. When it is running, open **API Keys** → **Create API Key**. **Copy it now —
   it is shown once.**
4. From the cluster overview copy the **endpoint URL**. It looks like
   `https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333`.

You now have `QDRANT_URL` and `QDRANT_API_KEY`.

### Load your vectors into it

The cluster starts empty. Re-embed from your local SQLite:

```bash
cd research-gap-hunter
QDRANT_URL="https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333" \
QDRANT_API_KEY="<your-key>" \
uv run python -c "
from vectors.embed import embed_limitations, embed_future_directions
print('limitations:', embed_limitations())
print('future_directions:', embed_future_directions())
"
```

This runs Specter2 locally and uploads the vectors. It takes a few minutes and
downloads ~440MB of model weights the first time. **No LLM calls, no paid API.**

### Verify

```bash
curl -H "api-key: <your-key>" "<your-qdrant-url>/collections/limitations" | head -c 400
```

You want `points_count` to be **168** (the total across both domains), and the
`future_directions` collection to be **94**. This load has been performed and
verified against the live cluster:

| Collection | Points | Dim | Distance | Domain split |
|---|---|---|---|---|
| `limitations` | 168 | 768 | Cosine | 64 CV / 104 MI |
| `future_directions` | 94 | 768 | Cosine | 34 CV / 60 MI |

Those match `REPORTS_LIMITATION` (168) and `SUGGESTS_FUTURE` (94) in the graph
exactly.

> **Point counts are not enough — check the payload index.** A managed Qdrant
> refuses to filter on an unindexed payload field:
>
> ```
> 400 Bad request: Index required but not found for "domain"
>     of one of the following types: [keyword]
> ```
>
> A local Qdrant allows it, so a cluster can look perfectly healthy — right
> counts, right dimensions, GREEN status — while **every filtered query fails**.
> That is `/search`, `/gaps`, `/cross-domain` and the `/corpus` counts: in other
> words, everything except `/health`. This was hit on the live cluster.
>
> `ensure_collection()` now creates the index, and does so for existing
> collections too, so any load run from this repo fixes it. To confirm:
>
> ```bash
> curl -H "api-key: <your-key>" "<your-qdrant-url>/collections/limitations" \
>   | python3 -c "import json,sys; print('indexed:', list(json.load(sys.stdin)['result']['payload_schema']))"
> ```
>
> You want `indexed: ['domain']` on **both** collections.

---

## 4. Render — the backend

### Create the service

1. <https://render.com> → **New** → **Blueprint**, and point it at
   `weepull/research-gap-hunter`.
2. Render reads `render.yaml` from the repo root, which declares a single Docker
   web service on the free plan with a `/health` health check. **Leave the root
   directory as the repo root** — the Dockerfile deliberately excludes
   `frontend/`.
3. Render prompts for every variable marked `sync: false` in the blueprint: the
   Neo4j and Qdrant credentials, and `ALLOWED_ORIGINS`. Nothing secret is stored
   in the repo.

If you would rather not use a blueprint: **New** → **Web Service** → connect the
repo → set **Runtime: Docker**, **Plan: Free**, **Health Check Path: `/health`**,
then add the variables from the list below by hand. The blueprint just does this
for you.

The first build is slow — torch plus ~440MB of model weights baked into the
image. Ten to twenty minutes is normal.

The image is **~2.85GB**. That is measured, not estimated — it was built and run
locally against the live databases before this document was written. Two things
keep it from being much larger: torch is installed from PyTorch's CPU index
rather than PyPI (the default wheel drags in the whole CUDA runtime, several GB
of it, on a container that has no GPU), and the duplicate model revision the
HuggingFace pre-cache pulls is pruned in the same layer that creates it
(`scripts/prune_hf_cache.py`, 440MB reclaimed).

### There is no disk, and that is the design

**Do not add a `disk:` block to `render.yaml`.** Render disks are a paid feature;
adding one to a free service makes the blueprint fail to apply.

Without a disk, `/data/papers.db` is container-local and resets whenever the
container restarts — which on a free service means every time it wakes from
being idle. That is fine, because Neo4j is already the source of truth for
scoring, and the startup self-heal rebuilds SQLite from the graph on every boot:
no LLM call, no network call, deterministic.

So **on Render the self-heal is the primary persistence mechanism, not a safety
net**, and this line in your logs is routine rather than alarming:

```
WARNING  pipeline.selfheal: Paper store drift detected: Neo4j has 127 papers,
SQLite has 0. Rebuilding 127 missing row(s) from graph relationships.
```

It says "check that the volume is mounted" because that is the actionable case
on a platform that *has* volumes. Here there is no volume by choice, so read it
as "the container restarted". Measured locally: rebuilding all 127 rows takes
about **200ms**.

The one thing this costs you: `objectives`, `evaluation_metrics` and `raw_json`
are not stored in Neo4j and cannot be reconstructed, so those come back empty on
every rebuilt row. Nothing in the UI displays them today.

### Memory — read this before assuming headroom

Render's free plan gives **512MB**. Measured on the built image with the limit
actually applied (`docker run --memory=512m`), against the live databases:

| State | Memory |
|---|---|
| After startup, idle | 418MB (82%) |
| Serving `/gaps?top_n=20` | 468MB |
| Peak, during `/cross-domain` | **476MB (93%)** |
| After 10 consecutive searches | 457MB |

It fits, and nothing was OOM-killed across the whole endpoint sweep. But **36MB
of headroom is not much**, and the figures above are from a single-user test on
a 127-paper corpus. Concurrent traffic or a materially larger corpus could cross
the line.

If the service starts dying with exit code 137, that is the OOM killer. The
levers, cheapest first: drop the baked-in weights so they are not resident at
boot (see the image-size note above — three things must come out together), or
move to a paid plan with more memory. Note that Render's Starter plan does not
add RAM over free; you need a larger instance type, not just a paid one.

### Set the environment variables

The blueprint sets the non-secret ones for you. These are the values it will
prompt for, plus what the blueprint already contains:

```
DEMO_MODE=true                      # set by render.yaml
ALLOWED_ORIGINS=https://<your-vercel-domain>.vercel.app

NEO4J_URI=neo4j+s://<your-aura-id>.databases.neo4j.io
NEO4J_USER=neo4j                    # set by render.yaml
NEO4J_PASSWORD=<from the Aura credentials file>
NEO4J_DATABASE=<from SHOW DATABASES, see §2 — often NOT "neo4j">

QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<from Qdrant Cloud>

PAPERS_DB_PATH=/data/papers.db      # set by render.yaml
SELFHEAL_ON_STARTUP=true            # set by render.yaml
RATE_LIMIT_ENABLED=true             # set by render.yaml
LOG_LEVEL=INFO                      # set by render.yaml
```

Three notes:

- **`DEMO_MODE=true` is mandatory in production.** With it false, `/ingest` and
  `/explain` would try to reach an Ollama that does not exist, and CORS would be
  wide open.
- **`ALLOWED_ORIGINS` has a chicken-and-egg problem** with Vercel: you do not
  know the domain until §5. Deploy the frontend first if you prefer, or set this
  afterwards and let Render redeploy. **If you leave it empty while
  `DEMO_MODE=true`, the API allows no origins at all and the frontend will fail
  with a CORS error in the browser.** That is deliberate — a closed door rather
  than a silent fallback to `*`.
- **`SELFHEAL_ON_STARTUP` must stay `true` here.** Without a disk it is the only
  thing that puts papers into SQLite, so turning it off leaves
  `/paper/{arxiv_id}` returning 404 for every paper the gap list cites.

You do **not** need `OLLAMA_*` or `SEMANTIC_SCHOLAR_API_KEY` — both are only
used by paths that demo mode refuses.

### Get the public URL

Render assigns one automatically, shown at the top of the service page. It looks
like `https://research-gap-hunter-api.onrender.com`. There is no separate step to
generate it.

### Cold starts — expect them

A free service **spins down after 15 minutes without traffic**. The next request
wakes it, and that takes roughly **30–60 seconds**: Render has to start the
container, and the app loads the embedding model before it will answer. Locally
that load is about 4 seconds on a warm machine; Render's free CPU is slower and
the container start is on top.

Practically: the first person to open the site after a quiet period will wait,
possibly long enough that the frontend's fetch looks stuck. Everyone after them
gets a normal response until it idles again. If that matters for a demo, hit the
URL yourself a minute beforehand.

### Verify

```bash
curl https://<your-service>.onrender.com/health
```

Expect something like:

```json
{
  "status": "ok",
  "papers": 119,
  "limitations": 152,
  "future_directions": 90,
  "services": {"sqlite": "ok", "neo4j": "ok", "qdrant": "ok"}
}
```

Allow up to a minute for the first call if the service was idle.

> ### `/health` now returns 503 when a store is unreachable
>
> **This is a behaviour change (PLAN.md item #6) and it affects the Render health
> check and any external uptime monitor pointed at this path.**
>
> `/health` previously returned `200 {"status":"ok", …}` unconditionally: the
> status string was a literal, Qdrant errors were swallowed into `0` counts, and
> Neo4j — the scoring source of truth — was never contacted at all. An instance
> with both stores down therefore reported itself healthy, and the Render health
> check kept it in the load balancer.
>
> It now probes each dependency and derives the status:
>
> | `services.<name>` | meaning | effect on status |
> |---|---|---|
> | `ok` | reachable, and the thing asked for is there | — |
> | `absent` | reachable, but the collection/table does not exist yet (cold start) | still `ok` |
> | `unreachable` | the store cannot be reached at all | `degraded` → **HTTP 503** |
>
> Counts are `null` rather than `0` when the store behind them is unreachable, so
> "nothing there" is distinguishable from "could not look".
>
> **What to expect operationally:** a deploy whose Neo4j credentials or Qdrant URL
> are wrong will now fail its health check and Render will not route to it,
> instead of serving an empty-looking but "ok" API. That is the intended
> behaviour. If you would rather the service stay in the load balancer while
> degraded, point the Render health check at a path that always returns 200 —
> do not soften `/health` itself.
>
> **Startup is now resilient on purpose.** The lifespan no longer aborts when a
> backend is unavailable; it logs at ERROR and continues with that backend set to
> `None`. Without this the 503 above would be unreachable in exactly the case it
> exists for: `get_neo4j_driver()` calls `verify_connectivity()`, so an
> unreachable graph used to make the process die before it could serve `/health`,
> producing a crash loop instead of a diagnosis. A process that starts and
> truthfully says `"neo4j": "unreachable"` is strictly more diagnosable than one
> that never answers. Startup reconciliation is skipped (with a WARNING) when the
> graph is unavailable, since it reads from the graph.

`/corpus` follows the same rule: a missing collection still reports `0`, but an
unreachable Qdrant reports `null` with `"vectors_available": false`, so the
corpus banner says the figure is unavailable instead of asserting an empty
corpus underneath a working results list.

The pre-change response was verified locally from the built container running
against the live Neo4j and Qdrant, with no disk and a 512MB memory cap — the
self-heal rebuilt all 127 rows on boot and `/health` reported them.

```bash
curl "https://<your-service>.onrender.com/corpus?domain=computer_vision"
```

Expect 46 papers / 64 limitations / 34 future directions.

Then check the Render **Logs** tab for the self-heal line:

```
WARNING pipeline.selfheal: Paper store drift detected: Neo4j has 127 papers, SQLite has 0.
WARNING pipeline.selfheal: Paper store repair complete: 127 rebuilt, 0 failed.
```

`LOG_LEVEL=INFO` is what makes the surrounding startup lines visible too.

## 5. Vercel — the frontend

1. <https://vercel.com> → **Add New** → **Project** → import
   `weepull/research-gap-hunter`.
2. **Root Directory: `frontend`.** This is essential — the repo root is a Python
   project and Vercel will fail to build it.
3. Framework preset: **Next.js** (auto-detected).
4. **Environment Variables**, before the first build:
   ```
   NEXT_PUBLIC_API_URL=https://<your-service>.onrender.com
   ```
   No trailing slash. Add it for **Production**, **Preview** and **Development**.
5. **Deploy.**

> `NEXT_PUBLIC_*` is inlined at **build** time, not read at runtime. If it is
> missing the production build **fails on purpose** — otherwise the deployed
> bundle would ship `http://localhost:8000` baked in and every request would fail
> in the browser with nothing in any server log. If your build fails with
> *"NEXT_PUBLIC_API_URL is not set"*, that is this guard working correctly.
>
> Changing this variable later requires a **redeploy**, not just a restart.

### Close the CORS loop

Copy your Vercel production domain back into Render's `ALLOWED_ORIGINS`
(service → **Environment**) and let it redeploy. CORS matching is on the origin
string and is platform-agnostic — verified against a
`https://<name>.vercel.app` origin with a Render-style backend: the allowed
origins get an `access-control-allow-origin` header on both the preflight and
the actual request, and an unlisted origin gets none, so the browser blocks
it. If you have a custom domain, include both, comma-separated:

```
ALLOWED_ORIGINS=https://your-app.vercel.app,https://www.yourdomain.com
```

---

## 6. Verify the whole thing

Open your Vercel URL and check each of these. **If the backend has been idle,
give the first request up to a minute** — Render is waking the container.

| Check | Expected |
|---|---|
| Landing page corpus figures | CV 46 / 64 / 34, MI 81 / 104 / 60 |
| `/gaps` | 10 ranked gaps, amber "⚠ Supported by 1 paper" on single-source ones |
| Corpus banner | "Computed over 46 Computer Vision papers…" |
| `/search` for "small objects" | ~10 semantically ranked results |
| `/cross-domain` → Find connections | 10 connections above threshold |
| "Explain this connection" | A calm "Explanations are off in this demo" notice, **not** an error |
| An arXiv chip | Opens the paper on arxiv.org |
| Browser console | **No CORS errors** |

If the pages load but every request fails, it is almost always `ALLOWED_ORIGINS`
not matching your Vercel domain exactly — scheme included, no trailing slash.

---

## 7. Credentials checklist

Everything you personally have to obtain, and where each value goes:

| Value | Where you get it | Where it goes |
|---|---|---|
| `NEO4J_URI` | Aura credentials file | Render env vars (blueprint prompts) |
| `NEO4J_PASSWORD` | Aura credentials file (**shown once**) | Render env vars (blueprint prompts) |
| `QDRANT_URL` | Qdrant Cloud cluster overview | Render env vars (blueprint prompts) |
| `QDRANT_API_KEY` | Qdrant Cloud → API Keys (**shown once**) | Render env vars (blueprint prompts) |
| Render service URL | Shown on the Render service page | Vercel `NEXT_PUBLIC_API_URL` |
| Vercel production domain | Vercel project overview | Render `ALLOWED_ORIGINS` |

`SEMANTIC_SCHOLAR_API_KEY` is optional and only affects local ingestion.

**Never commit any of these.** `.env` is gitignored; `.env.example` holds the
shape of the file with no values.

---

## 8. Known operational limits

These are real and already documented in `PROJECT_HARDENING_PLAN.md`; none is a
bug introduced by deploying.

- **Single worker, on purpose.** The rate limiter keeps per-process in-memory
  token buckets, so N workers would multiply every limit by N. Scaling out needs
  a shared store such as Redis (G5).
- **Cold start is fast; the memory footprint is the real cost.** Specter2 is
  baked into the image and `HF_HUB_OFFLINE=1` keeps startup off the network, so
  boot is a disk read — measured at **4 seconds from container start to a
  healthy response**. What has not changed is that ~440MB of weights is resident
  before the app serves traffic, which is what bounds how small an instance can
  be. That memory question is G4 and is still open.
- **Render free spins down after 15 minutes idle.** The next request pays a
  30–60 second cold start. Not a bug, and not fixable on the free plan.
- **Render free has 512MB and this app peaks at 476MB.** Measured under an
  enforced limit. It fits with ~36MB spare; concurrent traffic or a much larger
  corpus could not. Exit code 137 in the logs is the OOM killer.
- **SQLite is rebuilt from Neo4j on every boot**, because there is no disk. This
  is the intended arrangement on this platform, not a degraded mode — but it
  does mean `objectives`, `evaluation_metrics` and `raw_json` are empty on every
  row, since Neo4j never stored them.
- **AuraDB Free pauses after 3 days idle.** The first request after a pause will
  fail or hang while it resumes.
- **`/ingest` and `/explain` are refused in demo mode.** To grow the corpus, run
  ingestion locally against the cloud Neo4j and Qdrant, exactly as in §2 and §3.

---

## 9. Growing the corpus after deployment

Ingestion always runs locally, because it needs Ollama. Point it at the cloud
services and the deployed app picks up the new data with no redeploy:

```bash
export NEO4J_URI="neo4j+s://<your-aura-id>.databases.neo4j.io"
export NEO4J_USER=neo4j
export NEO4J_PASSWORD="<your-password>"
export QDRANT_URL="https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="<your-key>"
# DEMO_MODE stays false locally so ingestion is allowed.

ollama serve                      # in another terminal
uv run python -c "from pipeline.batch import ingest_from_query; print(ingest_from_query('computer vision object detection', limit=20))"
uv run python -c "from graph.populate import populate_graph; print(populate_graph())"
uv run python -c "from vectors.embed import embed_limitations, embed_future_directions; print(embed_limitations(), embed_future_directions())"
```

> **Never re-ingest a paper that already has graph relationships.**
> `_upsert_paper_counting()` only MERGEs and never removes relationships, and
> `Limitation` nodes are keyed on exact text. Ollama is non-deterministic, so
> re-extracting an existing paper attaches a second set of near-duplicate
> Limitation nodes and corrupts clustering. Ingestion skips papers already in
> SQLite; keep it that way.

After growing the corpus, re-derive the similarity thresholds — they are
corpus-dependent and currently set from measured null distributions. See items
A1/A2 in `PROJECT_HARDENING_PLAN.md`. **Do not hand-tune them.**
