"""Phase 3d driver: ingest in tranches, re-deriving thresholds after each.

Per the run rules, a tranche must never reach scoring before its derivation passes. So
each cycle is:

    ingest one tranche  ->  re-embed  ->  re-derive nulls  ->  apply stale constants
                        ->  run the derivation guard  ->  quality report

If the guard fails after a tranche, the loop stops rather than continuing to ingest on
top of thresholds that no longer describe the corpus.

Deadline-aware: stops cleanly at a tranche boundary when the budget is spent, never
mid-tranche. Ingestion itself checkpoints per paper, so even a hard kill resumes.

Usage:
    python scripts/run_tranches.py --hours 3.5 --size 50 --target-per-domain 300
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv" / "bin" / "python")
PROGRESS = ROOT / "data" / "tranche_progress.jsonl"
DOMAINS = ("computer_vision", "medical_imaging")


def _run(args: list[str], label: str) -> tuple[int, str]:
    print(f"\n----- {label} -----", flush=True)
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    print(tail, flush=True)
    if proc.returncode != 0:
        print(f"[{label}] exit {proc.returncode}\n{(proc.stderr or '')[-1500:]}", flush=True)
    return proc.returncode, proc.stdout or ""


def _counts() -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from pipeline.gap_scorer import _count_contributing_papers, _count_papers_in_domain
    return {
        d: {"papers": _count_papers_in_domain(d),
            "contributing": _count_contributing_papers(d)}
        for d in DOMAINS
    }


def _note(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--target-per-domain", type=int, default=300)
    args = parser.parse_args()

    deadline = time.monotonic() + args.hours * 3600
    print(f"=== PHASE 3d — budget {args.hours}h, tranche size {args.size}, "
          f"target {args.target_per_domain}/domain ===", flush=True)
    print(f"start {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"initial counts: {_counts()}", flush=True)

    cycle = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"\n=== budget spent; stopping cleanly at a tranche boundary "
                  f"after {cycle} tranche(s) ===", flush=True)
            break

        counts = _counts()
        # Ingest into whichever domain is furthest from target, so growth stays balanced.
        behind = sorted(DOMAINS, key=lambda d: counts[d]["papers"])
        domain = next((d for d in behind
                       if counts[d]["papers"] < args.target_per_domain), None)
        if domain is None:
            print("\n=== both domains at target; stopping ===", flush=True)
            break

        # A tranche is roughly size * 45s. Do not start one we cannot finish.
        projected = args.size * 45
        if remaining < projected * 0.6:
            print(f"\n=== {remaining/60:.0f} min left, a tranche needs about "
                  f"{projected/60:.0f} min; stopping cleanly rather than starting one "
                  f"we cannot finish ===", flush=True)
            break

        cycle += 1
        print(f"\n########## TRANCHE {cycle} — {domain} "
              f"({counts[domain]['papers']} papers now, "
              f"{remaining/3600:.2f}h left) ##########", flush=True)
        started = time.monotonic()

        rc, _ = _run([PY, "scripts/ingest_tranche.py", "--domain", domain,
                      "--size", str(args.size)], f"ingest {domain}")
        if rc != 0:
            print("ingest failed; stopping", flush=True)
            _note({"cycle": cycle, "domain": domain, "status": "ingest_failed"})
            break

        # Re-embed from the graph. Full rebuild with pruning, which is now idempotent
        # thanks to deterministic point ids (Phase 3a), so this stays correct as the
        # corpus grows and is simpler than tracking which points are new.
        _run([PY, "-c",
              "from dotenv import load_dotenv; load_dotenv();"
              "from vectors.embed import embed_limitations, embed_future_directions;"
              "print(embed_limitations()); print(embed_future_directions())"],
             "re-embed")

        # Re-derive and apply anything past DRIFT_TOLERANCE.
        _run([PY, "scripts/derive_thresholds.py", "--apply"], "re-derive thresholds")

        # The guard must pass before this tranche counts as part of scoring.
        rc, _ = _run([PY, "-m", "pytest",
                      "tests/integration/test_threshold_derivation.py",
                      "-m", "integration", "-q"], "derivation guard")
        if rc != 0:
            print("DERIVATION GUARD FAILED — stopping before this tranche reaches "
                  "scoring", flush=True)
            _note({"cycle": cycle, "domain": domain, "status": "guard_failed"})
            break

        after = _counts()
        took = (time.monotonic() - started) / 60
        print(f"\ntranche {cycle} complete in {took:.1f} min: {after}", flush=True)
        _note({
            "cycle": cycle, "domain": domain, "status": "ok",
            "minutes": round(took, 1), "counts": after,
            "at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"\n=== FINAL COUNTS: {_counts()} ===", flush=True)
    _run([PY, "scripts/ingest_tranche.py", "--domain", "computer_vision",
          "--report-only"], "quality report CV")
    _run([PY, "scripts/ingest_tranche.py", "--domain", "medical_imaging",
          "--report-only"], "quality report MI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
