"""Shared helpers for the evaluation harness.

The harness exists because every quality claim this project has made so far has been
about *mechanism* ("the threshold is derived", "the cluster no longer swallows half the
corpus") and none has been about *output*. Nobody has yet looked at a ranked gap list and
said whether the gaps are real. Until the label sheet is filled in by hand, that is still
true, and the harness is careful to say so rather than manufacturing a number.
"""

import csv
import math
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

# The only labels the scorer accepts. Deliberately three, not a 1-5 scale: a coarse
# judgement one annotator can apply consistently beats a fine one they cannot.
LABELS = ("real_gap", "generic_filler", "off_topic")

LABEL_HELP = {
    "real_gap": "a genuine open problem a researcher could act on",
    "generic_filler": "true but contentless — could be said of almost any paper",
    "off_topic": "not a limitation, or not in this domain at all",
}

SHEET_COLUMNS = (
    "domain", "rank", "system", "gap_text", "tier", "score",
    "frequency_score", "recency_score", "solution_deficit_score",
    "supporting_paper_ids", "supporting_paper_titles",
    "label", "notes",
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because n here is 20 and the proportion
    is often near 0 or 1, where the normal interval produces bounds outside [0, 1] and
    badly wrong coverage.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def write_sheet(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})
    return len(rows)


def read_sheet(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paper_titles(arxiv_ids: list[str]) -> dict[str, str]:
    """Titles from SQLite, so the annotator can see what they are judging."""
    from pipeline.batch import _get_db

    db = _get_db()
    if "papers" not in db.table_names():
        return {}
    out = {}
    for arxiv_id in arxiv_ids:
        rows = list(db["papers"].rows_where("arxiv_id = ?", [arxiv_id]))
        if rows:
            out[arxiv_id] = rows[0].get("title", "")
    return out
