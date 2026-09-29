"""Measure the deployed footprint. READ-ONLY, measures rather than estimates.

DEPLOYMENT.md targets Render's free tier at 512MB. Whether the corpus fits is a question
about *measured* resident memory, not about vector arithmetic, and the dominant term is
not the corpus at all — it is the Specter2 weights held in the process.

Usage:
    python scripts/measure_footprint.py
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

_VECTOR_DIM = 768
_BYTES_PER_FLOAT = 4  # Qdrant stores float32 by default


def _rss_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except Exception:  # noqa: BLE001
        return float("nan")


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    try:
        out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
        return int(out.stdout.split()[0]) / 1024
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> int:
    from vectors.embed import (_COLLECTION_FUTURE_DIRECTIONS, _COLLECTION_LIMITATIONS,
                               get_qdrant_client, hf_cache_dir, load_embedding_model)

    print("=" * 92)
    print("FOOTPRINT — measured")
    print("=" * 92)

    baseline = _rss_mb()
    print(f"  process RSS before loading the model : {baseline:7.1f} MB")

    client = get_qdrant_client()
    total_points = 0
    print("\n  Qdrant collections:")
    for collection in (_COLLECTION_LIMITATIONS, _COLLECTION_FUTURE_DIRECTIONS):
        try:
            info = client.get_collection(collection)
            points = info.points_count or 0
        except Exception as exc:  # noqa: BLE001
            print(f"    {collection}: unreachable ({exc.__class__.__name__})")
            continue
        total_points += points
        raw_mb = points * _VECTOR_DIM * _BYTES_PER_FLOAT / (1024 * 1024)
        print(f"    {collection:<20} {points:>6} points  "
              f"{raw_mb:6.2f} MB of raw float32 vectors")
    print(f"    {'TOTAL':<20} {total_points:>6} points  "
          f"{total_points * _VECTOR_DIM * _BYTES_PER_FLOAT / (1024*1024):6.2f} MB")

    print("\n  on-disk stores (local dev, not what a managed tier bills):")
    for label, path in (("qdrant_storage", Path("qdrant_storage")),
                        ("data/papers.db", Path("data/papers.db"))):
        size = (_dir_size_mb(path) if path.is_dir()
                else (path.stat().st_size / (1024 * 1024) if path.exists() else float("nan")))
        print(f"    {label:<20} {size:7.2f} MB")
    print(f"    {'HF model cache':<20} {_dir_size_mb(Path(hf_cache_dir())):7.2f} MB")

    print("\n  loading Specter2…")
    load_embedding_model()
    loaded = _rss_mb()
    print(f"  process RSS after loading the model  : {loaded:7.1f} MB "
          f"(+{loaded - baseline:.1f} MB)")

    print("\n" + "=" * 92)
    print("VERDICT for a 512MB tier")
    print("=" * 92)
    headroom = 512 - loaded
    print(f"  measured peak RSS with the model resident: {loaded:.1f} MB")
    print(f"  headroom against a 512MB cap            : {headroom:+.1f} MB")
    if headroom < 0:
        print("\n  DOES NOT FIT. The Specter2 weights alone exceed the tier, and they are")
        print("  resident for the life of the process because every query embeds text.")
    elif headroom < 100:
        print("\n  FITS, BUT WITHOUT MARGIN. Under 100MB of headroom, a concurrent request")
        print("  or a larger batch can still push it over.")
    else:
        print("\n  FITS with margin.")
    print("\n  Note what does and does not scale with the corpus. The vectors are "
          f"{total_points * _VECTOR_DIM * _BYTES_PER_FLOAT / (1024*1024):.2f} MB and")
    print("  live in Qdrant, not in this process. Corpus growth therefore barely moves "
          "the")
    print("  API's memory; the model does not move at all. The binding constraint is a")
    print("  fixed cost, so 'will a bigger corpus fit' is the wrong question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
