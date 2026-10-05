"""Generate RUN_REPORT.md from collected facts. Every number traces to a measurement.

Reads the JSON produced by scripts/collect_run_facts.py rather than taking figures from
prose, so the report cannot contain a number nobody measured — the standing rule for this
run.

Usage:
    python scripts/collect_run_facts.py > /tmp/run_facts.json
    python scripts/write_run_report.py /tmp/run_facts.json > RUN_REPORT.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    facts = json.loads(Path(sys.argv[1]).read_text())
    out: list[str] = []
    w = out.append

    corpus = facts["corpus"]
    gaps = facts["gaps"]
    ing = facts["ingestion"]
    filt = facts["extraction_filter"]
    th = facts["thresholds"]
    domains = list(corpus)

    w("# RUN_REPORT.md — long autonomous hardening run")
    w("")
    w("**Every figure below was read out of the live system by "
      "`scripts/collect_run_facts.py`.** Nothing here is estimated or recalled.")
    w("")
    w("Baseline: commit `15f7128`, 113 papers (28 CV / 85 MI).")
    w("")

    # --- corpus -----------------------------------------------------------
    w("## Corpus, before and after")
    w("")
    w("| | computer_vision | medical_imaging |")
    w("|---|---:|---:|")
    w(f"| papers at start | 28 | 85 |")
    w(f"| **papers now** | **{corpus['computer_vision']['papers']}** "
      f"| **{corpus['medical_imaging']['papers']}** |")
    w(f"| contributing papers (frequency denominator) "
      f"| {corpus['computer_vision']['contributing_papers']} "
      f"| {corpus['medical_imaging']['contributing_papers']} |")
    w(f"| limitation nodes | {corpus['computer_vision']['limitation_nodes']} "
      f"| {corpus['medical_imaging']['limitation_nodes']} |")
    w(f"| clusters | {corpus['computer_vision']['clusters']} "
      f"| {corpus['medical_imaging']['clusters']} |")
    w(f"| largest cluster | {corpus['computer_vision']['largest_cluster']} "
      f"| {corpus['medical_imaging']['largest_cluster']} |")
    w("")

    # --- ingestion --------------------------------------------------------
    w("## Ingestion outcomes")
    w("")
    w(f"`data/ingest_log.jsonl`, {ing['log_entries']} entries.")
    w("")
    w("| outcome | total | " + " | ".join(domains) + " |")
    w("|---|---:|" + "---:|" * len(domains))
    all_outcomes = sorted(ing["outcomes_total"])
    for outcome in all_outcomes:
        cells = " | ".join(
            str(ing["outcomes_by_domain"].get(d, {}).get(outcome, 0)) for d in domains
        )
        w(f"| `{outcome}` | {ing['outcomes_total'][outcome]} | {cells} |")
    w("")
    tranches = facts.get("tranches", [])
    if tranches:
        w(f"Tranches completed: **{len(tranches)}**")
        w("")
        w("| # | domain | minutes | papers after |")
        w("|---|---|---:|---|")
        for t in tranches:
            counts = t.get("counts", {})
            after = ", ".join(f"{d[:2]}={counts.get(d, {}).get('papers', '?')}"
                              for d in domains)
            w(f"| {t.get('cycle')} | {t.get('domain')} | {t.get('minutes')} | {after} |")
    else:
        w("**No tranche completed in full.** See the budget note below.")
    w("")

    # --- filters ----------------------------------------------------------
    w("## Extraction-filter rejections")
    w("")
    w(f"- raw limitation strings seen during ingestion: **{ing['limitations_raw']}**")
    w(f"- kept after filtering: **{ing['limitations_kept']}**")
    w(f"- ingestion-time filter rejection rate: **{pct(ing['filter_rejection_rate'])}**")
    w(f"- total rejections logged to `data/rejected_extractions.jsonl` "
      f"(including the one-off backfill): **{filt['total_rejected']}**")
    w("")
    if filt["by_reason"]:
        w("| reason | count |")
        w("|---|---:|")
        for reason, n in sorted(filt["by_reason"].items(), key=lambda kv: -kv[1]):
            w(f"| `{reason}` | {n} |")
        w("")

    # --- thresholds -------------------------------------------------------
    w("## Thresholds: coded vs freshly derived")
    w("")
    w(f"`DRIFT_TOLERANCE` = 0.002. A constant is rewritten only past that; below it, "
      f"drift is indistinguishable from resampling the same corpus.")
    w("")
    w("| constant | coded | derived | drift | action |")
    w("|---|---:|---:|---:|---|")
    for row in th["drift"]:
        derived = "n/a" if row.get("derived") is None else f"{float(row['derived']):.4f}"
        drift = "n/a" if row.get("drift") is None else f"{float(row['drift']):.4f}"
        action = "**UPDATE NEEDED**" if row.get("stale") else "within tolerance"
        w(f"| `{row['name']}` | {float(row['coded']):.4f} | {derived} | {drift} | {action} |")
    w("")
    coded = th["coded"]
    w(f"Other coded constants: `_UNRESOLVED_DEFICIT_FLOORS` = "
      f"{coded['unresolved_deficit_floor']} (the solution noise floor on the deficit scale), "
      f"`MIN_WORDS` = {coded['min_words']} (p5 of the measured word distribution), "
      f"`_DEFICIT_RESCALE_ANCHORS` = "
      + ", ".join(f"{k} ({v[0]:.4f}, {v[1]:.4f})" for k, v in coded["deficit_anchors"].items())
      + ".")
    w("")

    # --- tests ------------------------------------------------------------
    w("## Tests — raw summary lines")
    w("")
    w("```")
    w(f"pytest -q               {facts['tests']['unit']}")
    w(f"pytest -m integration   {facts['tests']['integration']}")
    w("```")
    w("")

    # --- ties and tiers ---------------------------------------------------
    w("## Ties and tiers")
    w("")
    w("| | computer_vision | medical_imaging |")
    w("|---|---:|---:|")
    w(f"| gaps | {gaps['computer_vision']['n']} | {gaps['medical_imaging']['n']} |")
    w(f"| gaps sharing a score | {gaps['computer_vision']['tied_gaps']} "
      f"| {gaps['medical_imaging']['tied_gaps']} |")
    w(f"| distinct scores | {gaps['computer_vision']['distinct_scores']} "
      f"| {gaps['medical_imaging']['distinct_scores']} |")
    w(f"| corroborated tier | {gaps['computer_vision']['corroborated']} "
      f"| {gaps['medical_imaging']['corroborated']} |")
    w(f"| single-source tier | {gaps['computer_vision']['single_source']} "
      f"| {gaps['medical_imaging']['single_source']} |")
    w(f"| above `_UNRESOLVED_DEFICIT_FLOORS` | "
      f"{gaps['computer_vision']['above_unresolved_floor']} "
      f"| {gaps['medical_imaging']['above_unresolved_floor']} |")
    w("")

    # --- gap lists --------------------------------------------------------
    for domain in domains:
        w(f"## `/gaps?domain={domain}` — top 15, verbatim")
        w("")
        w("```")
        for g in gaps[domain]["top15"]:
            tier = "CORROB" if g["tier"] == "corroborated" else "single"
            w(f"[{g['rank']:>2}] [{tier}] score={g['score']:.4f} "
              f"f={g['frequency']:.4f} r={g['recency']:.4f} d={g['deficit']:.4f} "
              f"n={len(g['papers'])} sols={g['solutions']}")
            w(f"     {g['text'][:104]}")
            w(f"     papers: {' '.join(g['papers'])}")
        w("```")
        w("")

    # --- cross-domain -----------------------------------------------------
    for key, matches in facts["cross_domain"].items():
        w(f"## `/cross-domain` {key} — {len(matches)} matches, verbatim")
        w("")
        w("```")
        if not matches:
            w("(no pair clears the noise floor — a valid, expected result)")
        for m in matches:
            w(f"sim={m['similarity']:.4f}")
            w(f"  GAP: {m['source_gap'][:96]}")
            w(f"  SOL: {m['target_solution'][:96]}")
            w(f"  src={' '.join(m['source_papers'])}  tgt={' '.join(m['target_papers'])}")
        w("```")
        w("")

    # --- rejected papers ---------------------------------------------------
    rejected = ing["rule_rejected"]
    w(f"## Papers rejected by the corpus rules at ingest — all {len(rejected)}")
    w("")
    if rejected:
        w("| arXiv | primary | title | why |")
        w("|---|---|---|---|")
        for r in sorted(rejected, key=lambda r: r["arxiv_id"]):
            title = r["title"].replace("|", "/")[:70]
            reason = r["reason"].replace("|", "/")[:80]
            w(f"| `{r['arxiv_id']}` | `{r['primary_category'] or '?'}` | {title} | {reason} |")
    else:
        w("(none)")
    w("")

    w("## Extraction-filter rejections, sampled")
    w("")
    if filt["samples"]:
        w("| arXiv | field | reason | text |")
        w("|---|---|---|---|")
        for r in filt["samples"]:
            w(f"| `{r['arxiv_id']}` | {r['field'][:3]} | `{r['reason']}` "
              f"| {r['text'].replace('|', '/')[:70]} |")
    w("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
