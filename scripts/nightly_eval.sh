#!/usr/bin/env bash
# Re-run the evaluation harness and store timestamped results. Phase 5.
#
# Read-only with respect to the corpus: it regenerates the label sheet and the extraction
# audit, scores whatever labels currently exist, and snapshots the derived thresholds and
# the live gap lists. It never ingests, never re-embeds and never writes a constant, so it
# is safe to run unattended.
#
# Results land in eval/runs/<UTC timestamp>/ so successive runs can be diffed — the point
# is to notice when output quality moves, which a single run cannot show.
#
# Usage:  bash scripts/nightly_eval.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT="$ROOT/eval/runs/$STAMP"
mkdir -p "$OUT"

echo "=== nightly eval $STAMP ==="
echo "results -> $OUT"

# Deliberately not `set -e`: a step failing (Ollama down, say) must not prevent the
# remaining steps from producing their artifacts. Each exit code is recorded instead.
status=0
run_step() {
  local name="$1"; shift
  echo; echo "--- $name ---"
  if "$@" > "$OUT/$name.txt" 2>&1; then
    echo "ok -> $name.txt"
  else
    local code=$?
    echo "FAILED (exit $code) -> $name.txt"
    echo "$name exit $code" >> "$OUT/FAILURES.txt"
    status=1
  fi
}

run_step corpus_state          "$PY" -c "
from dotenv import load_dotenv; load_dotenv()
from pipeline.gap_scorer import _count_papers_in_domain, _count_contributing_papers
from pipeline.domains import RESEARCH_DOMAINS
for d in RESEARCH_DOMAINS:
    print(d, 'papers', _count_papers_in_domain(d), 'contributing', _count_contributing_papers(d))
"
run_step thresholds            "$PY" scripts/derive_thresholds.py
run_step weight_influence      "$PY" scripts/measure_weight_influence.py
run_step length_cutoffs        "$PY" scripts/derive_length_cutoffs.py
run_step ingest_quality_cv     "$PY" scripts/ingest_tranche.py --domain computer_vision --report-only
run_step ingest_quality_mi     "$PY" scripts/ingest_tranche.py --domain medical_imaging --report-only
run_step label_sheet           "$PY" eval/make_label_sheet.py
run_step precision             "$PY" eval/score.py
run_step footprint             "$PY" scripts/measure_footprint.py
run_step unit_tests            "$PY" -m pytest -q
run_step integration_tests     "$PY" -m pytest -m integration -q

# Snapshot the sheet itself, so a later run can be compared against the labels that
# existed at the time rather than only against the summary numbers.
cp -f "$ROOT/eval/label_sheet.csv" "$OUT/label_sheet.csv" 2>/dev/null || true

echo
echo "=== done: $OUT ==="
if [ -f "$OUT/FAILURES.txt" ]; then
  echo "some steps failed:"; cat "$OUT/FAILURES.txt"
fi
exit $status
