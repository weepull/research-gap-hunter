# Backend image for Railway.
#
# The frontend is NOT in this image — it deploys separately to Vercel.
#
# Python 3.12 rather than the 3.14 the local venv happens to run: 3.12 is what
# the scientific wheel ecosystem (torch, sentence-transformers) reliably ships
# binaries for, and pyproject only requires >=3.11. A version without wheels
# would force a source build and blow past any sane build timeout.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the HuggingFace cache on a known path so the weights baked in below
    # are the ones found at runtime.
    HF_HOME=/opt/hf-cache

WORKDIR /app

# curl is only here for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed from pyproject BEFORE any source is copied, so a
# source-only change does not re-resolve the whole dependency tree. The project
# itself is deliberately not installed at this stage — installing it here would
# put package skeletons in site-packages that then shadow the real source
# copied in below.
COPY pyproject.toml ./

# CPU-only torch, installed FIRST and from PyTorch's CPU index.
#
# sentence-transformers pulls torch, and the default PyPI wheel on Linux drags
# in the whole CUDA runtime — nvidia-cudnn alone is ~445MB and the full stack is
# several GB. Railway runs this on CPU, so every byte of that is dead weight in
# the image and in the deploy time. Installing the CPU build first means the
# dependency resolution below finds torch already satisfied and leaves it alone.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && cat /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Bake the Specter2 weights into the image, then prune what is not used.
#
# This is the G4 trade-off, taken deliberately: without it every cold start
# downloads the weights before the app can serve traffic, which on a small
# instance means a slow, failure-prone boot on every redeploy. Boot becomes a
# disk read instead.
#
# Download and prune MUST stay in one RUN. Docker layers are additive, so
# pruning in a later instruction leaves the bytes in the earlier layer and the
# image does not shrink at all — measured at 3.7GB either way before the two
# were merged. See scripts/prune_hf_cache.py for what is dropped and why.
#
# HF_HUB_OFFLINE is deliberately NOT set — CLAUDE.md warns it breaks fresh
# installs, and the runtime should still be able to fall back to a download.
COPY scripts/prune_hf_cache.py /tmp/prune_hf_cache.py
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('allenai/specter2_base', cache_folder='/opt/hf-cache')" \
      || echo 'WARNING: could not pre-cache Specter2; it will download at first boot'; \
    python /tmp/prune_hf_cache.py /opt/hf-cache; \
    rm -f /tmp/prune_hf_cache.py; \
    du -sh /opt/hf-cache

COPY . .

# Editable, no-deps: the real source in /app stays authoritative and nothing is
# re-resolved. Without this the package would either be missing from sys.path or
# shadowed by a stale copy.
RUN pip install --no-deps -e .

# The volume mount point. Railway mounts over this; creating it means the image
# also runs correctly with no volume attached.
RUN mkdir -p /data

EXPOSE 8000

# Railway injects $PORT and it is not always 8000. Single worker on purpose:
# the rate limiter holds per-process in-memory token buckets, so N workers
# would multiply every limit by N (G5 in PROJECT_HARDENING_PLAN.md).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
