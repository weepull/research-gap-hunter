"""Collect every measurable fact the run report needs, as JSON. READ-ONLY.

The run rule is "never claim a number you did not measure". This script is how that is
enforced: RUN_REPORT.md is written from its output, so any figure in the report traces to
a value read out of the live system rather than to recollection.

Usage:
    python scripts/collect_run_facts.py > /tmp/run_facts.json
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv()

PY = str(ROOT / ".venv" / "bin" / "python")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _pytest(args: list[str]) -> str:
    proc = subprocess.run([PY, "-m", "pytest", *args], cwd=ROOT,
                          capture_output=True, text=True)
    lines = [l for l in (proc.stdout or "").splitlines() if "passed" in l or "failed" in l]
    return lines[-1].strip() if lines else "(no summary line)"


def main() -> int:
    from pipeline.cross_domain import (_CROSS_DOMAIN_THRESHOLD, find_cross_domain_matches,
                                       _UNRESOLVED_DEFICIT_FLOORS, _unresolved_deficit_floor)
    from pipeline.domains import RESEARCH_DOMAINS
    from pipeline.extraction_filter import MIN_WORDS
    from pipeline.gap_scorer import (_CLUSTER_THRESHOLDS, _DEFICIT_RESCALE_ANCHORS,
                                     _SOLUTION_THRESHOLDS, _count_contributing_papers,
                                     _count_papers_in_domain, cluster_limitations,
                                     get_all_limitations, score_gaps)
    import derive_thresholds as dt

    facts: dict = {}

    # --- corpus -----------------------------------------------------------
    corpus = {}
    for domain in RESEARCH_DOMAINS:
        lims = get_all_limitations(domain)
        clusters = cluster_limitations(lims, domain=domain)
        corpus[domain] = {
            "papers": _count_papers_in_domain(domain),
            "contributing_papers": _count_contributing_papers(domain),
            "limitation_nodes": len(lims),
            "clusters": len(clusters),
            "largest_cluster": max((len(c) for c in clusters), default=0),
        }
    facts["corpus"] = corpus

    # --- gaps, ties, tiers -------------------------------------------------
    gaps_by_domain = {}
    for domain in RESEARCH_DOMAINS:
        gaps = score_gaps(domain=domain, top_n=500)
        scores = [round(g.score, 4) for g in gaps]
        counts = Counter(scores)
        gaps_by_domain[domain] = {
            "n": len(gaps),
            "tied_gaps": sum(v for v in counts.values() if v > 1),
            "distinct_scores": len(counts),
            "corroborated": sum(1 for g in gaps if g.tier == "corroborated"),
            "single_source": sum(1 for g in gaps if g.tier == "single_source"),
            "above_unresolved_floor": sum(
                1 for g in gaps if g.solution_deficit_score > _unresolved_deficit_floor(domain)
            ),
            "top15": [
                {"rank": i, "tier": g.tier, "score": g.score,
                 "frequency": g.frequency_score, "recency": g.recency_score,
                 "deficit": g.solution_deficit_score,
                 "papers": g.supporting_papers, "solutions": len(g.proposed_solutions),
                 "text": g.gap_description}
                for i, g in enumerate(gaps[:15], 1)
            ],
        }
    facts["gaps"] = gaps_by_domain

    # --- cross-domain ------------------------------------------------------
    xd = {}
    for source, target in (("computer_vision", "medical_imaging"),
                           ("medical_imaging", "computer_vision")):
        matches = find_cross_domain_matches(source_domain=source, target_domain=target,
                                            top_n=10)
        xd[f"{source}->{target}"] = [
            {"similarity": m.similarity_score, "source_gap": m.source_gap,
             "target_solution": m.target_solution,
             "source_papers": m.source_papers, "target_papers": m.target_papers}
            for m in matches
        ]
    facts["cross_domain"] = xd

    # --- thresholds: coded vs derived --------------------------------------
    facts["thresholds"] = {
        "coded": {
            "cluster": dict(_CLUSTER_THRESHOLDS),
            "solution": dict(_SOLUTION_THRESHOLDS),
            "cross_domain": _CROSS_DOMAIN_THRESHOLD,
            "deficit_anchors": {k: list(v) for k, v in _DEFICIT_RESCALE_ANCHORS.items()},
            "unresolved_deficit_floor": dict(_UNRESOLVED_DEFICIT_FLOORS),
            "min_words": dict(MIN_WORDS),
        },
        "drift": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
            for row in dt.drift_report()
        ],
    }

    # --- ingestion ---------------------------------------------------------
    log = _jsonl(ROOT / "data" / "ingest_log.jsonl")
    by_domain_outcomes = {}
    for domain in RESEARCH_DOMAINS:
        rows = [r for r in log if r.get("domain") == domain]
        by_domain_outcomes[domain] = dict(Counter(r["outcome"] for r in rows))
    raw = sum(r.get("limitations_raw", 0) for r in log)
    kept = sum(r.get("limitations_kept", 0) for r in log)
    facts["ingestion"] = {
        "log_entries": len(log),
        "outcomes_total": dict(Counter(r["outcome"] for r in log)),
        "outcomes_by_domain": by_domain_outcomes,
        "limitations_raw": raw,
        "limitations_kept": kept,
        "filter_rejection_rate": (raw - kept) / raw if raw else None,
        "rule_rejected": [
            {"arxiv_id": r["arxiv_id"], "primary_category": r.get("primary_category", ""),
             "title": r.get("title", ""), "reason": r.get("reason", "")}
            for r in log if r["outcome"] == "rule_rejected"
        ],
    }
    facts["tranches"] = _jsonl(ROOT / "data" / "tranche_progress.jsonl")

    # --- the extraction filter's own audit log -----------------------------
    rejected = _jsonl(ROOT / "data" / "rejected_extractions.jsonl")
    facts["extraction_filter"] = {
        "total_rejected": len(rejected),
        "by_reason": dict(Counter(r["reason"] for r in rejected)),
        "by_field": dict(Counter(r["field"] for r in rejected)),
        "samples": [
            {"arxiv_id": r["arxiv_id"], "field": r["field"], "reason": r["reason"],
             "text": r["text"][:120]}
            for r in rejected[:40]
        ],
    }

    # --- tests -------------------------------------------------------------
    facts["tests"] = {
        "unit": _pytest(["-q"]),
        "integration": _pytest(["-m", "integration", "-q"]),
    }

    json.dump(facts, sys.stdout, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
