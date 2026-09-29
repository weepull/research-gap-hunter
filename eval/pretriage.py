"""Optional LLM pre-labelling. Phase 4(e). NOT GROUND TRUTH, and structurally unusable
as such.

Writes to `eval/pretriage_NOT_GROUND_TRUTH.csv`, and `eval/score.py` refuses by filename
to read any sheet with `NOT_GROUND_TRUTH` in it. The point is to let a human skim
suggestions and label faster, never to produce a metric.

This is the one place in the project where an LLM touches evaluation, and it is walled
off deliberately. Scoring, ranking, domain assignment, thresholds and the extraction
filter are all LLM-free; a model grading its own pipeline's output would be circular, and
a number produced that way would be worthless in exactly the way that is hardest to see.

Usage:
    python eval/pretriage.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from eval.common import EVAL_DIR, LABEL_HELP, LABELS, read_sheet  # noqa: E402

_OUT = EVAL_DIR / "pretriage_NOT_GROUND_TRUTH.csv"

_PROMPT = """\
You are triaging candidate research gaps extracted from academic papers, to help a human
label them faster. Classify the statement below into exactly one category.

{options}

Statement: "{text}"

Reply with only the category name, nothing else."""


def main() -> int:
    from pipeline.cross_domain import _call_ollama_text

    sheet = EVAL_DIR / "label_sheet.csv"
    if not sheet.exists():
        print(f"No sheet at {sheet}. Run eval/make_label_sheet.py first.")
        return 1

    rows = read_sheet(sheet)
    unique = sorted({(r["domain"], r["gap_text"]) for r in rows})
    options = "\n".join(f"- {name}: {help_text}" for name, help_text in LABEL_HELP.items())

    print(f"triaging {len(unique)} unique gaps with a local LLM — NOT GROUND TRUTH")
    out_rows = []
    for i, (domain, text) in enumerate(unique, 1):
        try:
            reply = _call_ollama_text(_PROMPT.format(options=options, text=text))
        except Exception as exc:  # noqa: BLE001 — triage is optional, never fatal
            reply = f"<error: {exc.__class__.__name__}>"
        guess = next((label for label in LABELS if label in reply.lower()), "")
        out_rows.append({"domain": domain, "gap_text": text,
                         "llm_suggestion": guess, "llm_raw": reply[:200]})
        if i % 10 == 0:
            print(f"  {i}/{len(unique)}", flush=True)

    with _OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# NOT GROUND TRUTH — LLM triage only.",
                         "Never feeds any reported metric.",
                         "eval/score.py refuses this file by filename.", "", ""])
        writer.writerow(["domain", "gap_text", "llm_suggestion", "llm_raw", ""])
        for row in out_rows:
            writer.writerow([row["domain"], row["gap_text"],
                             row["llm_suggestion"], row["llm_raw"], ""])
    print(f"\nwrote {len(out_rows)} suggestions to {_OUT}")
    print("These are SUGGESTIONS for a human to accept or override. They are not data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
