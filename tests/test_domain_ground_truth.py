"""eval/domain_ground_truth.csv — the manual domain classification (U1, 2026-10-06).

Pins what the file must encode, from PLAN_AUDIT_FIX.md: INV-4's paper-by-paper reading of
the 64 papers then in computer_vision, and the R1 relabels. The 85 medical-imaging papers
were never manually classified, so they carry an EMPTY manual_class — the file must not
pretend otherwise, and nothing may fill that gap by running a classifier.
"""

import csv
from collections import Counter
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "eval" / "domain_ground_truth.csv"
AMBIGUOUS = {"2609.30566", "2609.30682", "2304.09148"}
RELABELLED_R1 = {"2609.30708", "2609.31788", "2609.30613", "2609.30223"}


def _rows():
    with PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["arxiv_id", "assigned_domain", "manual_class"]
        return list(reader)


def test_one_row_per_paper_in_the_corpus():
    rows = _rows()
    assert len(rows) == 149
    assert len({r["arxiv_id"] for r in rows}) == 149
    assert Counter(r["assigned_domain"] for r in rows) == {"computer_vision": 60, "medical_imaging": 89}


def test_only_the_three_manual_classes_or_empty():
    assert {r["manual_class"] for r in _rows()} <= {"clearly_cv", "clearly_medical", "ambiguous", ""}


def test_coverage_is_the_64_inv4_papers_and_no_more():
    """Partial coverage is stated, not filled: the 85 MI papers are unclassified."""
    classes = Counter(r["manual_class"] for r in _rows())
    assert classes == {"clearly_cv": 57, "clearly_medical": 4, "ambiguous": 3, "": 85}
    assert all(r["assigned_domain"] == "medical_imaging" for r in _rows() if r["manual_class"] == "")


def test_the_three_ambiguous_papers_are_marked_explicitly():
    assert {r["arxiv_id"] for r in _rows() if r["manual_class"] == "ambiguous"} == AMBIGUOUS
    assert all(r["assigned_domain"] == "computer_vision" for r in _rows() if r["arxiv_id"] in AMBIGUOUS)


def test_the_r1_relabels_are_clearly_medical_and_assigned_medical():
    rows = {r["arxiv_id"]: r for r in _rows()}
    for pid in RELABELLED_R1:
        assert rows[pid]["manual_class"] == "clearly_medical"
        assert rows[pid]["assigned_domain"] == "medical_imaging"


def test_every_clearly_cv_paper_is_assigned_computer_vision():
    assert all(r["assigned_domain"] == "computer_vision" for r in _rows() if r["manual_class"] == "clearly_cv")
