"""Deterministic quality gates on extracted limitations and future directions.

## Why

Roughly half the extracted future directions were contentless boilerplate
(`PROJECT_HARDENING_PLAN.md` item A3), and the corpus contained `Limitation` nodes
whose entire text was `"remains challenging"` and `"future work includes"` — the
extraction prompt's own example cue phrases, echoed back by the model. Those
strings then became cluster seeds, gap descriptions, and cross-domain match
sources. One survey's self-hedging (`"certain aspects may have eluded our
scrutiny"`) produced a cross-domain "discovery" at similarity 0.8871.

No threshold-tuning fixes that, because generic text scores *high*, not low: it
sits near the corpus centroid. It has to be filtered before it enters the graph.

## Design constraints

**Every gate here is deterministic and contains no LLM.** Domain assignment,
scoring and thresholds are all LLM-free in this project and this must stay that
way — a model deciding what counts as a real limitation would put model judgment
directly upstream of every score.

**Whole-string matching for echo and hedging, not substring.** `"remains
challenging"` alone is worthless; `"remains challenging to process high-resolution
images"` is a genuine, well-attested gap in the live corpus. A substring rule would
delete the second along with the first. Only phrases that are contentless *wherever*
they appear (a survey hedging about its own coverage) are matched as substrings.

**Duplicate removal is parameter-free on purpose.** Exact-after-normalisation and
containment are used; there is deliberately no similarity threshold. Introducing a
Jaccard or cosine cutoff would mean inventing an unmeasured constant, which is the
failure mode this project has spent two sessions removing. A parameter-free rule was
available, so it is used.

**The length cutoff is a stated percentile**, not a round number:
`scripts/derive_length_cutoffs.py` reports the distribution and the candidates.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path("data/rejected_extractions.jsonl")

FIELDS = ("limitations", "future_directions")


# ---------------------------------------------------------------------------
# Gate (a) — prompt-echo phrases
# ---------------------------------------------------------------------------

# Cue phrases that have appeared as *examples* in the extraction prompt, which the
# model echoed back as if they were findings.
#
# This is an EXPLICIT constant and must stay one. It was briefly implemented by
# parsing the quoted phrases out of the prompt at runtime, which is circular: the
# accompanying fix removes the quotable examples from the prompt, so a parsed list
# would silently become empty and the gate would stop rejecting anything while still
# appearing to work. The list is therefore maintained by hand, and two tests keep it
# honest — one asserts it still contains the known echoes, the other asserts that any
# quoted phrase present in the prompt is covered here, so adding a new example to the
# prompt without updating this list fails.
#
# Compared against the *whole* normalised string, never as a substring.
PROMPT_ECHO_PHRASES: frozenset[str] = frozenset({
    "however",
    "despite",
    "remains challenging",
    "future work includes",
    "we leave x for future",
    "we leave for future",
    "limitations",
    "future work",
    "future directions",
})


# ---------------------------------------------------------------------------
# Gate (b) — hedging boilerplate
# ---------------------------------------------------------------------------

# Contentless WHEREVER they appear. The bar for this list is deliberately high: the
# phrase must be a statement about the paper's own completeness, which can never name
# a research gap no matter what surrounds it.
#
# Several plausible-looking candidates were deliberately NOT put here after the
# backfill dry run showed them discarding real content. "further research is needed",
# "beyond the scope of this" and "left for future work" all routinely appear in
# sentences that DO name something specific — "Further research is needed to improve
# robustness and generalizability", "Multi-modal fusion is beyond the scope of this
# work" — so they live in HEDGING_WHOLE instead, where they are only rejected when
# they constitute the entire string. Getting this wrong deletes signal silently, which
# is worse than leaving a little boilerplate in.
HEDGING_SUBSTRINGS: tuple[str, ...] = (
    "may have eluded",
    "might not be entirely encapsulated",
    "not entirely exhaustive",
    "we hope this paper",
    "we hope this work",
)

# Rejected only when they constitute essentially the whole string.
HEDGING_WHOLE: tuple[str, ...] = (
    "remains challenging",
    "future work includes",
    "needs further investigation",
    "requires further study",
    "further research is needed",
    "more research is required",
    "beyond the scope of this work",
    "left for future work",
    "overcome current limitations",
    "various limitations",
    "several limitations",
    "some limitations",
    "other limitations",
    "additional experiments",
    "more experiments",
)


# ---------------------------------------------------------------------------
# Gate (c) — minimum informative length
# ---------------------------------------------------------------------------

# The 5th percentile of the observed word-count distribution, per field, measured by
# scripts/derive_length_cutoffs.py on the live corpus (2026-09-28: limitations n=150,
# future_directions n=84 — p5 = 3 words for both).
#
# p5 rather than p10: p10 for future_directions is 5 words, which would discard terse
# but genuinely specific solutions such as "contrast-agnostic, pathology-encoded
# representations". p5 removes only the extreme tail — the one- and two-word fragments
# that cannot describe a gap ("Overfitting", "Dataset bias", "domain shift") — and
# leaves anything with a subject and a predicate. The echo and hedging gates catch the
# remaining short boilerplate independently, so this gate does not carry that alone.
MIN_WORDS: dict[str, int] = {"limitations": 3, "future_directions": 3}
_DEFAULT_MIN_WORDS = 3

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Used for every comparison so that "Remains challenging." and "remains
    challenging" are the same string.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _PUNCT_RE.sub(" ", folded.lower())
    return _WS_RE.sub(" ", folded).strip()


@dataclass(frozen=True)
class Rejection:
    """One discarded string and the gate that discarded it."""

    arxiv_id: str
    field: str
    text: str
    reason: str
    detail: str = ""

    def as_record(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "field": self.field,
            "text": self.text,
            "reason": self.reason,
            "detail": self.detail,
            "at": datetime.now(timezone.utc).isoformat(),
        }


def _min_words(field: str) -> int:
    return MIN_WORDS.get(field, _DEFAULT_MIN_WORDS)


def classify(text: str, field: str) -> tuple[str, str] | None:
    """Return (reason, detail) if `text` should be rejected, else None.

    Order matters only for which reason is reported; the gates are independent.
    """
    raw = (text or "").strip()
    if not raw:
        return ("empty", "")

    norm = normalise(raw)
    if not norm:
        return ("empty", "no alphanumeric content")

    # (a) prompt echo — whole string only
    if norm in PROMPT_ECHO_PHRASES:
        return ("prompt_echo", f"whole string is the prompt cue {norm!r}")

    # (b) hedging
    for phrase in HEDGING_SUBSTRINGS:
        if phrase in norm:
            return ("hedging", f"contains {phrase!r}")
    if norm in {normalise(p) for p in HEDGING_WHOLE}:
        return ("hedging", f"whole string is boilerplate {norm!r}")

    # (c) below the derived minimum informative length
    words = len(norm.split())
    minimum = _min_words(field)
    if words < minimum:
        return ("too_short", f"{words} words < p5 cutoff of {minimum} for {field}")

    return None


def _is_duplicate(candidate: str, kept: list[str]) -> tuple[bool, str]:
    """Exact-after-normalisation or containment against already-kept strings.

    Parameter-free by design — see the module docstring. Containment is directional:
    the longer string is the one already kept, because `filter_extractions` processes
    longest-first.
    """
    norm = normalise(candidate)
    for existing in kept:
        existing_norm = normalise(existing)
        if norm == existing_norm:
            return (True, f"exact duplicate of {existing[:60]!r}")
        if norm and norm in existing_norm:
            return (True, f"contained in {existing[:60]!r}")
    return (False, "")


def filter_extractions(
    items: list[str], field: str, arxiv_id: str = ""
) -> tuple[list[str], list[Rejection]]:
    """Apply every gate. Returns (kept, rejections) preserving useful order.

    Candidates are considered longest-first so that containment keeps the more
    specific string; the kept list is then returned in its original relative order,
    so downstream output does not silently reorder.
    """
    if not items:
        return ([], [])

    rejections: list[Rejection] = []
    surviving: list[str] = []

    order = sorted(range(len(items)), key=lambda i: len(normalise(items[i])), reverse=True)
    kept_idx: list[int] = []
    for i in order:
        text = (items[i] or "").strip()
        verdict = classify(text, field)
        if verdict is not None:
            reason, detail = verdict
            rejections.append(Rejection(arxiv_id, field, items[i], reason, detail))
            continue
        duplicate, detail = _is_duplicate(text, [items[j].strip() for j in kept_idx])
        if duplicate:
            rejections.append(Rejection(arxiv_id, field, items[i], "duplicate", detail))
            continue
        kept_idx.append(i)

    for i in sorted(kept_idx):
        surviving.append(items[i].strip())
    return (surviving, rejections)


def log_rejections(rejections: list[Rejection], log_path: Path | None = None) -> int:
    """Append rejections to data/rejected_extractions.jsonl. Returns the count.

    Every discarded string is recorded so filtering is auditable — a silent filter is
    indistinguishable from an extraction failure, and this project has already been
    bitten by a store whose contents nobody could account for.
    """
    if not rejections:
        return 0
    destination = log_path if log_path is not None else _LOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for rejection in rejections:
            handle.write(json.dumps(rejection.as_record()) + "\n")
    return len(rejections)


def filter_and_log(
    items: list[str], field: str, arxiv_id: str = "", log_path: Path | None = None
) -> list[str]:
    """filter_extractions + log_rejections, for callers that only want the survivors."""
    kept, rejected = filter_extractions(items, field, arxiv_id)
    if rejected:
        log_rejections(rejected, log_path)
        logger.info(
            "Filtered %d of %d %s for %s: %s",
            len(rejected), len(items), field, arxiv_id or "<unknown>",
            ", ".join(sorted({r.reason for r in rejected})),
        )
    return kept
