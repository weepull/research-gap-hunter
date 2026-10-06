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
# U1b: human-reviewed and moved CV -> MI in curation pass 1 (scripts/domain_backfill.py RELABEL).
CURATION_PASS_1 = {"2305.17456", "2307.15872", "2409.03367", "2501.16469"}
# INV-1 read these as medical from the TITLE only — deliberately not used as labels.
INV1_TITLE_ONLY = {"2303.08446", "2406.11026", "2408.08058"}
# Every label must trace to a human source. No model-generated label may enter this file.
HUMAN_SOURCES = {"inv4", "curation_pass_1"}


def _rows():
    with PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["arxiv_id", "assigned_domain", "manual_class", "source", "note"]
        return list(reader)


def test_one_row_per_paper_in_the_corpus():
    rows = _rows()
    assert len(rows) == 149
    assert len({r["arxiv_id"] for r in rows}) == 149
    assert Counter(r["assigned_domain"] for r in rows) == {"computer_vision": 60, "medical_imaging": 89}


def test_only_the_three_manual_classes_or_empty():
    assert {r["manual_class"] for r in _rows()} <= {"clearly_cv", "clearly_medical", "ambiguous", ""}


def test_coverage_is_inv4_plus_curation_pass_1_and_no_more():
    """Partial coverage is stated, not filled: 81 MI papers remain unclassified."""
    classes = Counter(r["manual_class"] for r in _rows())
    assert classes == {"clearly_cv": 57, "clearly_medical": 8, "ambiguous": 3, "": 81}
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


def test_every_label_traces_to_a_human_source():
    """U1b: no label without a recorded human source; no source without a label."""
    for r in _rows():
        if r["manual_class"]:
            assert r["source"] in HUMAN_SOURCES, r
        else:
            assert r["source"] == "", r
    assert Counter(r["source"] for r in _rows() if r["source"]) == {"inv4": 64, "curation_pass_1": 4}


def test_curation_pass_1_papers_are_clearly_medical_from_that_source():
    rows = {r["arxiv_id"]: r for r in _rows()}
    for pid in CURATION_PASS_1:
        assert rows[pid]["manual_class"] == "clearly_medical"
        assert rows[pid]["source"] == "curation_pass_1"
        assert rows[pid]["assigned_domain"] == "medical_imaging"


def test_inv1_title_only_reads_are_not_labels_and_say_why():
    rows = {r["arxiv_id"]: r for r in _rows()}
    for pid in INV1_TITLE_ONLY:
        assert rows[pid]["manual_class"] == "" and rows[pid]["source"] == ""
        assert "title" in rows[pid]["note"].lower()
