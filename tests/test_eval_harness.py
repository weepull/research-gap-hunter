"""Tests for the evaluation harness (Phase 4).

Fail against pre-fix code because eval/ did not exist.
"""

import csv

import pytest

from eval.common import LABELS, SHEET_COLUMNS, read_sheet, wilson_interval, write_sheet


# ---------------------------------------------------------------------------
# Wilson intervals
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(7, 20)
    assert low < 7 / 20 < high


@pytest.mark.parametrize("successes,n", [(0, 20), (20, 20), (1, 20), (19, 20)])
def test_wilson_interval_stays_inside_the_unit_interval(successes, n):
    """The reason for Wilson over the normal approximation: n is small and p is extreme."""
    low, high = wilson_interval(successes, n)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_of_zero_n_is_degenerate_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_as_n_grows():
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_at_n_20_is_wide_enough_to_matter():
    """Sanity check on the headline caveat: 14/20 cannot be distinguished from 10/20."""
    a_low, a_high = wilson_interval(14, 20)
    b_low, b_high = wilson_interval(10, 20)
    assert a_low < b_high, "intervals at n=20 must overlap for a 4-hit difference"


# ---------------------------------------------------------------------------
# Sheet round-trip
# ---------------------------------------------------------------------------


def test_sheet_round_trips(tmp_path):
    rows = [{"domain": "computer_vision", "rank": 1, "system": "current",
             "gap_text": "detection fails on small objects", "tier": "corroborated",
             "score": "0.4751", "label": "", "notes": ""}]
    path = tmp_path / "sheet.csv"
    assert write_sheet(rows, path) == 1
    back = read_sheet(path)
    assert back[0]["gap_text"] == "detection fails on small objects"
    assert back[0]["label"] == ""


def test_sheet_has_a_blank_label_column_for_every_row(tmp_path):
    """The sheet is for a human to fill; a pre-filled label would defeat the point."""
    rows = [{"domain": "d", "rank": i, "system": "current", "gap_text": f"g{i}"}
            for i in range(5)]
    path = tmp_path / "sheet.csv"
    write_sheet(rows, path)
    for row in read_sheet(path):
        assert row["label"] == ""


def test_sheet_columns_include_evidence_for_the_annotator():
    """A label is only meaningful if the annotator can see what backs the gap."""
    for column in ("supporting_paper_ids", "supporting_paper_titles", "tier", "score"):
        assert column in SHEET_COLUMNS


def test_labels_are_the_three_documented_values():
    assert LABELS == ("real_gap", "generic_filler", "off_topic")


# ---------------------------------------------------------------------------
# The scorer refuses LLM triage output
# ---------------------------------------------------------------------------


def test_scorer_refuses_a_not_ground_truth_sheet(tmp_path, capsys):
    """Phase 4(e): LLM triage must never feed a reported metric.

    Enforced by filename so it cannot be bypassed by pointing --sheet at the triage
    file, which is the mistake most likely to happen under time pressure.
    """
    import eval.score as scorer

    path = tmp_path / "pretriage_NOT_GROUND_TRUTH.csv"
    path.write_text("domain,gap_text,label\nx,y,real_gap\n")
    rc = scorer.main.__wrapped__ if hasattr(scorer.main, "__wrapped__") else None
    import sys
    argv = sys.argv
    try:
        sys.argv = ["score.py", "--sheet", str(path)]
        code = scorer.main()
    finally:
        sys.argv = argv
    assert code == 2
    assert "REFUSING" in capsys.readouterr().out


def test_scorer_reports_nothing_without_labels(tmp_path, capsys):
    """An unlabelled sheet must produce no metric, not a zero that reads as one."""
    import sys

    import eval.score as scorer

    rows = [{"domain": "computer_vision", "rank": 1, "system": "current",
             "gap_text": "a gap", "label": ""}]
    path = tmp_path / "label_sheet.csv"
    write_sheet(rows, path)

    argv = sys.argv
    try:
        sys.argv = ["score.py", "--sheet", str(path)]
        assert scorer.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "NO LABELS PRESENT" in out
    assert "has not been measured" in out


def test_scorer_propagates_a_label_across_systems(tmp_path, capsys):
    """The same gap appears under several rankings; labelling it once must suffice."""
    import sys

    import eval.score as scorer

    rows = []
    for system in ("current", "frequency", "random"):
        rows.append({"domain": "computer_vision", "rank": 1, "system": system,
                     "gap_text": "shared gap text", "label": ""})
    rows[0]["label"] = "real_gap"
    path = tmp_path / "label_sheet.csv"
    write_sheet(rows, path)

    argv = sys.argv
    try:
        sys.argv = ["score.py", "--sheet", str(path), "--k", "1"]
        assert scorer.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    # all three systems should be scored from the single label
    for system in ("current", "frequency", "random"):
        assert system in out


def test_scorer_rejects_an_unrecognised_label(tmp_path, capsys):
    import sys

    import eval.score as scorer

    rows = [{"domain": "computer_vision", "rank": 1, "system": "current",
             "gap_text": "a gap", "label": "probably_fine"}]
    path = tmp_path / "label_sheet.csv"
    write_sheet(rows, path)

    argv = sys.argv
    try:
        sys.argv = ["score.py", "--sheet", str(path)]
        assert scorer.main() == 1
    finally:
        sys.argv = argv
    assert "unrecognised labels" in capsys.readouterr().out
