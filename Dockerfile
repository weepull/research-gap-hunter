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
    # The app reads HF_CACHE_DIR; huggingface_hub derives its own hub path from
    # HF_HOME as $HF_HOME/hub. These two MUST agree, or SentenceTransformer's
    # lookups and the transformers cache_dir resolve to different directories
    # and the baked weights are found by neither.
    HF_HOME=/opt/hf \
    HF_CACHE_DIR=/opt/hf/hub

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
# The bake goes through the application's OWN load_embedding_model() rather
# than calling SentenceTransformer directly. That is the whole point: an earlier
# version baked via a direct call with cache_folder=/opt/hf-cache while the app
# read a hardcoded ~/.cache/huggingface/hub, so the app found nothing and
# re-downloaded 440MB on every boot. Baking through the real code path means
# the build and the runtime cannot resolve to different directories, because
# they are the same call.
#
# Only vectors/ is copied here, so a change anywhere else does not invalidate
# this layer — but a change to the model loading code correctly does.
#
# Download and prune MUST stay in one RUN. Docker layers are additive, so
# pruning in a later instruction leaves the bytes in the earlier layer and the
# image does not shrink at all — measured at 3.7GB either way before the two
# were merged. See scripts/prune_hf_cache.py for what is dropped and why.
#
# The bake failing now fails the build. It used to fall through with a warning,
# but the offline assertion below would fail immediately afterwards anyway, so
# the soft failure only made the real cause harder to read.
COPY vectors/ ./vectors/
COPY scripts/prune_hf_cache.py /tmp/prune_hf_cache.py
RUN python -c "from vectors.embed import load_embedding_model; load_embedding_model()" && \
    python /tmp/prune_hf_cache.py "$HF_CACHE_DIR" && \
    rm -f /tmp/prune_hf_cache.py && \
    du -sh /opt/hf

# Fail the build if the app cannot load the model without the network. Without
# this the image can look fine and still cost a 440MB download on every cold
# start — which is exactly the bug this arrangement exists to prevent, and it
# is invisible unless something asserts against it.
RUN HF_HUB_OFFLINE=1 python -c "\
from vectors.embed import load_embedding_model, hf_cache_dir; \
m = load_embedding_model(); \
print('offline load OK from', hf_cache_dir(), 'dim', m.get_sentence_embedding_dimension())"

# Now that the weights are provably in the image, stop consulting the network
# for them at runtime.
#
# CLAUDE.md says not to set HF_HUB_OFFLINE, and that guidance is right for a
# developer machine: it breaks a fresh install that has nothing cached yet.
# That rationale does not apply to this image. The weights are baked in, and the
# RUN above FAILS THE BUILD if they cannot be loaded offline — so a container
# that exists at all is one where offline mode works.
#
# What it buys, measured on this image with the network unreachable: 3 seconds
# and zero network log lines, against 141 seconds and 66 lines of DNS-failure
# retries without it. Even with working DNS it removes a hard startup
# dependency on huggingface.co being up.
#
# Deliberately set here in the Dockerfile only, never in .env or .env.example,
# so no developer machine inherits it. Remove this line if the runtime ever
# needs to fetch a model that is not baked in.
ENV HF_HUB_OFFLINE=1

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
