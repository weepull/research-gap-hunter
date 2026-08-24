# Deployment

How to take Research Gap Hunter live on Railway (backend) + Vercel (frontend) +
Neo4j AuraDB (graph) + Qdrant Cloud (vectors).

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
| FastAPI backend | Railway | Trial credit, then ~$5/mo | Needs a **volume**; see §4 |
| Next.js frontend | Vercel | Yes, Hobby | Personal projects only on Hobby |
| Neo4j graph | Neo4j AuraDB | Yes, AuraDB Free | **Pauses after 3 days idle, deleted after 30** |
| Qdrant vectors | Qdrant Cloud | Yes, 1GB cluster | Enough for this corpus by a wide margin |

Two limits worth knowing before you start:

- **AuraDB Free auto-pauses after 3 days of inactivity** and is **deleted after
  30 days** of inactivity. A demo nobody visits for a month is gone. If this
  needs to stay up, budget for AuraDB Professional or plan to re-seed.
- **Railway's free trial is credit-based, not perpetual.** The backend is a
  long-running container with ~1.5GB of image; expect to move to the paid Hobby
  plan.

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
Railway past the trial.

1. **GitHub** — you have this; the repo is already at `weepull/research-gap-hunter`.
2. **Neo4j Aura** — <https://console.neo4j.io>
3. **Qdrant Cloud** — <https://cloud.qdrant.io>
4. **Railway** — <https://railway.app> (sign in with GitHub)
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
NEO4J_DATABASE="neo4j" \
uv run python -c "from graph.populate import populate_graph; print(populate_graph())"
```

This reads SQLite and writes Neo4j. It makes **no LLM and no Semantic Scholar
calls**, so it is safe to re-run. Expect it to report several hundred nodes and
relationships created.

**Option B — dump and restore.** Aura Free does not accept `neo4j-admin`
database dumps. Use Option A.

### Verify

In the Aura console's **Query** tab:

```cypher
MATCH (p:Paper) RETURN count(p)
```

You want **127**. Also check `MATCH (l:Limitation) RETURN count(l)`.

---

## 3. Qdrant Cloud

### Create the cluster

1. <https://cloud.qdrant.io> → **Create Cluster** → **Free tier**.
2. Pick the region nearest your Railway region.
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
`future_directions` collection to be **94**.

---

## 4. Railway — the backend

### Create the service

1. <https://railway.app> → **New Project** → **Deploy from GitHub repo** →
   pick `weepull/research-gap-hunter`.
2. Railway detects `railway.json` and builds with the `Dockerfile`. **Leave the
   root directory as the repo root** — the Dockerfile deliberately excludes
   `frontend/`.
3. The first build is slow (torch plus ~440MB of model weights baked into the
   image). Ten to twenty minutes is normal.

The image is **~2.85GB**. That is measured, not estimated — it was built and run
locally against the live databases before this document was written. Two things
keep it from being much larger: torch is installed from PyTorch's CPU index
rather than PyPI (the default wheel drags in the whole CUDA runtime, several GB
of it, on a container that has no GPU), and the duplicate model revision the
HuggingFace pre-cache pulls is pruned in the same layer that creates it
(`scripts/prune_hf_cache.py`, 440MB reclaimed).

If Railway rejects the image for size, the lever to pull is the baked-in
weights — but **three things have to come out together**, because they depend on
each other:

1. the `RUN` that pre-caches the model (`vectors.embed.load_embedding_model()`),
2. the `RUN` immediately after it that asserts the model loads with
   `HF_HUB_OFFLINE=1` — this one FAILS THE BUILD if the weights are absent, so
   removing only step 1 gives you a failed build, not a smaller image,
3. `ENV HF_HUB_OFFLINE=1` further down — leave it and the runtime is forbidden
   from downloading the weights it no longer has, so the app cannot start at all.

Removing all three trades ~421MB of image for a cold start that downloads the
weights before serving traffic. Removing any subset is broken.

The image sets `HF_HUB_OFFLINE=1`, because it contains the weights and the build
fails if they cannot be loaded offline. Measured on this image with the network
unreachable: **4s to first healthy response and zero network calls**, against a
container that previously hung for over half an hour retrying DNS. You do not
need to set anything for this; it is baked in.

### Add the volume — do not skip this

This is the single most important step, and skipping it causes silent data loss.

1. In the service → **Variables/Settings** → **Volumes** → **New Volume**.
2. Mount path: **`/data`**
3. Size: 1GB is plenty.

Railway container filesystems are ephemeral. Without this volume, `papers.db` is
recreated empty on every redeploy and `/paper/{arxiv_id}` starts 404ing for
papers that the gap list is actively citing.

> The API has a safety net for exactly this: on startup it compares Neo4j Paper
> nodes to SQLite rows and rebuilds any missing rows from the graph. If it fires
> it logs at **WARNING**, naming the volume. **Treat that warning as a bug
> report about your volume configuration, not as a routine message** — the
> rebuild cannot recover `objectives`, `evaluation_metrics` or `raw_json`,
> because Neo4j never stored them.

### Set the environment variables

Service → **Variables** → paste each of these:

```
DEMO_MODE=true
ALLOWED_ORIGINS=https://<your-vercel-domain>.vercel.app

NEO4J_URI=neo4j+s://<your-aura-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<from the Aura credentials file>
NEO4J_DATABASE=neo4j

QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<from Qdrant Cloud>

PAPERS_DB_PATH=/data/papers.db
SELFHEAL_ON_STARTUP=true
RATE_LIMIT_ENABLED=true
LOG_LEVEL=INFO
```

Notes on three of these:

- **`DEMO_MODE=true` is mandatory in production.** With it false, `/ingest` and
  `/explain` would try to reach an Ollama that does not exist, and CORS would be
  wide open.
- **`ALLOWED_ORIGINS` has a chicken-and-egg problem** with Vercel: you do not
  know the domain until §5. Deploy the frontend first if you prefer, or set this
  afterwards and let Railway redeploy. **If you leave it empty while
  `DEMO_MODE=true`, the API allows no origins at all and the frontend will fail
  with a CORS error in the browser.** That is deliberate — a closed door rather
  than a silent fallback to `*`.
- **`PAPERS_DB_PATH` must match the volume mount path** from the previous step.

You do **not** need `OLLAMA_*` or `SEMANTIC_SCHOLAR_API_KEY` — both are only
used by paths that demo mode refuses.

### Seed the volume

The volume starts empty, so SQLite has no papers even though Aura does. You have
two options:

- **Do nothing.** On first boot the self-heal rebuilds all 127 rows from Aura
  automatically. It will log a WARNING saying so, which in this one case is
  expected rather than alarming. You lose `objectives`, `evaluation_metrics` and
  `raw_json` on every row.
- **Upload the real file** (preferred, keeps all fields). With the Railway CLI:
  ```bash
  npm i -g @railway/cli
  railway login
  railway link           # pick your project
  railway run --service <service-name> -- ls /data   # confirm the mount
  ```
  Railway has no direct file-copy command; the practical route is to add a
  one-off command that pulls the file from somewhere you control, or accept the
  self-heal. **For a demo, the self-heal is fine.**

### Get the public URL

Service → **Settings** → **Networking** → **Generate Domain**. You get something
like `https://research-gap-hunter-production.up.railway.app`.

### Verify

```bash
curl https://<your-railway-domain>/health
```

Expect `{"status":"ok","papers":127,"limitations":168,"future_directions":94}`.

This exact response was verified locally from the built container running against
the live Neo4j and Qdrant, with an empty volume — the self-heal rebuilt all 127
rows on boot and `/health` reported them.

```bash
curl "https://<your-railway-domain>/corpus?domain=computer_vision"
```

Expect 46 papers / 64 limitations / 34 future directions.

Then check the Railway **Deploy Logs** for the self-heal line — either

```
INFO  pipeline.selfheal: Paper stores agree: 127 in Neo4j, 127 in SQLite. No repair needed.
```

or the WARNING that it rebuilt rows. `LOG_LEVEL=INFO` is what makes both
visible; at WARNING you would still see the drift message but lose the
confirmation that the stores agree.

---

## 5. Vercel — the frontend

1. <https://vercel.com> → **Add New** → **Project** → import
   `weepull/research-gap-hunter`.
2. **Root Directory: `frontend`.** This is essential — the repo root is a Python
   project and Vercel will fail to build it.
3. Framework preset: **Next.js** (auto-detected).
4. **Environment Variables**, before the first build:
   ```
   NEXT_PUBLIC_API_URL=https://<your-railway-domain>
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

Copy your Vercel production domain back into Railway's `ALLOWED_ORIGINS` and let
it redeploy. If you have a custom domain, include both, comma-separated:

```
ALLOWED_ORIGINS=https://your-app.vercel.app,https://www.yourdomain.com
```

---

## 6. Verify the whole thing

Open your Vercel URL and check each of these:

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
| `NEO4J_URI` | Aura credentials file | Railway variables |
| `NEO4J_PASSWORD` | Aura credentials file (**shown once**) | Railway variables |
| `QDRANT_URL` | Qdrant Cloud cluster overview | Railway variables |
| `QDRANT_API_KEY` | Qdrant Cloud → API Keys (**shown once**) | Railway variables |
| Railway public domain | Railway → Settings → Networking | Vercel `NEXT_PUBLIC_API_URL` |
| Vercel production domain | Vercel project overview | Railway `ALLOWED_ORIGINS` |

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
