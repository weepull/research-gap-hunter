"""eval/mi_sample_review.md — the 40-paper MI sample for hand labelling (U1b, 2026-10-06).

The sample must be reproducible from the recorded seed and population, every label field
must be blank or one of the three human classes, and the file must carry no
model-suggested label of any kind.
"""

import random
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "eval" / "mi_sample_review.md"
CLASSES = {"clearly_cv", "clearly_medical", "ambiguous"}


def _text():
    return PATH.read_text()


def _seed_population_sample():
    text = _text()
    seed = int(re.search(r"^- \*\*seed:\*\* `(\d+)`$", text, re.M).group(1))
    population = re.search(r"^- \*\*population \(\d+ ids, sorted\):\*\* (.+)$", text, re.M).group(1)
    population = [x.strip(" `") for x in population.split(",")]
    sample = re.findall(r"^### \d+\. `([^`]+)`$", text, re.M)
    return seed, population, sample


def test_the_sample_is_reproducible_from_the_recorded_seed_and_population():
    seed, population, sample = _seed_population_sample()
    assert len(population) == 81 and population == sorted(population)
    assert len(sample) == 40
    assert random.Random(seed).sample(population, 40) == sample


def test_every_label_field_is_blank_or_a_human_class():
    fields = re.findall(r"^- \*\*manual_class:\*\*[ \t]*(.*)$", _text(), re.M)
    assert len(fields) == 40
    assert all(f.strip() in CLASSES | {""} for f in fields)


def test_the_file_suggests_no_label():
    """No keyword score, verifier verdict, prediction or suggestion may appear.

    Checked outside the quoted abstracts (lines starting "> "), which are the papers' own
    words and may legitimately contain any of these terms.
    """
    lowered = "\n".join(l for l in _text().splitlines() if not l.startswith(">")).lower()
    for forbidden in ("predicted", "suggest", "verdict", "classify_text", "keyword score",
                      "mi=", "cv=", "confidence"):
        assert forbidden not in lowered, forbidden
