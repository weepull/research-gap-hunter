"""Domain registry, validation (PLAN.md #1 option 1A) and a keyword verifier (1C).

## Why this module exists

`extract_paper()` hardcoded `domain="computer_vision"` at
`pipeline/extractor.py:452`, and `ingest_from_query()` never overrode it. So a
paper's domain recorded *which script ingested it*, never its content. The live
consequence: five of the top seven "computer vision research gaps" came from
papers on analytic number theory, atomic physics, control theory and LLM
inference, and the flagship cross-domain match was driven by a medical-imaging
sentence from a medical paper stamped `computer_vision`.

Domain is not cosmetic. It divides `frequency_score`, filters every Qdrant
query, routes cross-domain matching, and selects which null distribution the
similarity thresholds are derived from.

## The two mechanisms here, and why they are kept apart

**1A — `validate_domain()`.** The caller must declare a domain and it must be a
known one. This makes mislabelling impossible *by construction* at new call
sites: a caller that forgets fails loudly instead of silently producing CV rows.

**1C — `classify_text()` / `verify_declared_domain()`.** A deterministic keyword
heuristic over title and abstract. At ingestion time it is a **non-blocking
verifier only**: it flags a disagreement for human review and *never*
auto-assigns or overrides what the caller declared. This restraint is deliberate
and is the advisor's decision. The heuristic is genuinely brittle on this corpus
— "Segment Anything" is a CV paper dense with segmentation vocabulary, and
"Trustworthy Deep Learning for Medical Image Segmentation" is a medical paper
dense with CV vocabulary — so letting it silently overrule a human would trade a
loud failure mode for a quiet one.

There is deliberately **no LLM classification** (option 1B was rejected). No
score, label or threshold in this system is LLM-derived, and domain feeds
scoring directly; an 8B model classifying from a truncated conclusion section
would be noisy *and* non-deterministic, which would break the determinism
guarantee `_corpus_reference_year` exists to protect.

## Curation is a third, separate thing

`curate_declared_domain()` **may** override or exclude, unlike the ingestion
verifier. It exists for one-off corpus curation — the 2026-09-28 backfill and
the integration fixture loader — where correcting historical labels is the
explicit goal. It is not wired into any ingestion path. Keeping the two verbs
distinct is the whole point: `verify_*` reports, `curate_*` decides.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# The research domains this project covers. Anything else is not a domain label
# — it is a bad search-ingestion hit, and belongs out of the corpus rather than
# in it under some third name.
RESEARCH_DOMAINS: tuple[str, ...] = ("computer_vision", "medical_imaging")

# Weighted term evidence. Weight 3 terms are close to decisive on their own;
# weight 1 terms are only meaningful in aggregate. Medical evidence is
# deliberately *clinical* rather than task-shaped: "segmentation" alone says
# nothing about domain, because both fields segment things. What distinguishes
# medical imaging is the modality (CT, MRI), the subject (patient, anatomy) and
# the reader (radiologist, pathologist).
_MEDICAL_TERMS: dict[str, int] = {
    "medical imag": 3, "clinical": 3, "radiolog": 3, "patholog": 3,
    "diagnos": 3, "patient": 3, "tumour": 3, "tumor": 3, "lesion": 3,
    "anatom": 3, "histopatholog": 3, "whole-slide": 3, "whole slide": 3,
    "mammograph": 3, "radiograph": 3, "endoscop": 3, "ultrasound": 3,
    "biomedical": 3, "healthcare": 3, "disease": 3, "cancer": 3,
    "mri": 3, "ct scan": 3, " ct ": 2, "pet scan": 3, "x-ray": 2, "xray": 2,
    "organ": 2, "cardiac": 3, "brain": 2, "lung": 3, "chest": 2,
    "abdominal": 3, "prostate": 3, "retina": 3, "dermato": 3, "surgical": 3,
    "cohort": 2, "scanner": 2, "modality": 1, "dice": 2, "hospital": 3,
    "nodule": 3, "pneumonia": 3, "covid": 2, "sclerosis": 3, "stain": 2,
}

_VISION_TERMS: dict[str, int] = {
    "object detection": 3, "image classification": 3, "semantic segmentation": 2,
    "instance segmentation": 3, "panoptic": 3, "optical flow": 3,
    "pose estimation": 3, "image synthesis": 3, "text-to-image": 3,
    "image generation": 3, "video": 2, "natural image": 3, "scene": 2,
    "vision transformer": 3, "convolutional": 2, "bounding box": 3,
    "imagenet": 3, "coco": 3, "cityscapes": 3, "kitti": 3, "laion": 3,
    "visual": 2, "image captioning": 3, "tracking": 2, "re-identification": 3,
    "depth estimation": 3, "point cloud": 3, "3d reconstruction": 3,
    "diffusion model": 2, "gan": 1, "vision-language": 3, "multimodal": 1,
    "occlusion": 2, "pedestrian": 3, "autonomous driving": 3, "map": 1,
    "computer vision": 3, "image recognition": 3, "frame": 1, "camera": 2,
}

# A paper must reach this much weighted evidence for *some* domain before the
# verifier will express any opinion at all. Below it the text simply does not
# look like either field, which is the off-topic signal.
_MIN_EVIDENCE = 3
# How much the leading domain must beat the runner-up by to be called confident.
# Both fields share a great deal of machine-learning vocabulary, so a narrow win
# is not evidence of anything.
_MIN_MARGIN = 2

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, and pad for ' ct ' matching."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " " + _WS_RE.sub(" ", folded.lower()) + " "


@dataclass(frozen=True)
class DomainVerdict:
    """What the keyword heuristic thinks, and the evidence it used.

    `best` is None when nothing reached _MIN_EVIDENCE — read that as "this does
    not look like either field", not as "medical imaging by default".
    """

    best: str | None
    scores: dict[str, int]
    matched: dict[str, list[str]] = field(default_factory=dict)
    confident: bool = False

    @property
    def looks_off_topic(self) -> bool:
        return self.best is None

    def describe(self) -> str:
        parts = [f"{d}={self.scores.get(d, 0)}" for d in RESEARCH_DOMAINS]
        return f"best={self.best} confident={self.confident} " + " ".join(parts)


def validate_domain(domain: str) -> str:
    """Return the domain if it is a known research domain, else raise ValueError.

    This is option 1A: no default, no guessing. A call site that does not know
    which domain it is ingesting must fail rather than quietly produce
    computer_vision rows, which is precisely how six off-topic papers and one
    medical paper entered the corpus as CV.
    """
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(
            f"domain is required and must be one of {RESEARCH_DOMAINS}; got {domain!r}"
        )
    candidate = domain.strip()
    if candidate not in RESEARCH_DOMAINS:
        raise ValueError(
            f"unknown domain {candidate!r}; expected one of {RESEARCH_DOMAINS}. "
            "A paper that belongs to neither is a bad ingestion hit and must be "
            "excluded from the corpus, not stored under a third label."
        )
    return candidate


def classify_text(text: str) -> DomainVerdict:
    """Score free text against each domain's term evidence. Deterministic."""
    haystack = _normalise(text)
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for domain, terms in (
        ("medical_imaging", _MEDICAL_TERMS),
        ("computer_vision", _VISION_TERMS),
    ):
        total = 0
        hits: list[str] = []
        for term, weight in terms.items():
            if term in haystack:
                total += weight
                hits.append(term)
        scores[domain] = total
        matched[domain] = sorted(hits)

    ranked = sorted(RESEARCH_DOMAINS, key=lambda d: scores[d], reverse=True)
    leader, runner_up = ranked[0], ranked[1]
    if scores[leader] < _MIN_EVIDENCE:
        return DomainVerdict(best=None, scores=scores, matched=matched, confident=False)
    return DomainVerdict(
        best=leader,
        scores=scores,
        matched=matched,
        confident=(scores[leader] - scores[runner_up]) >= _MIN_MARGIN,
    )


def verify_declared_domain(
    declared: str, title: str = "", abstract: str = "", extra: str = ""
) -> str | None:
    """Non-blocking check of a declared domain against the paper's own text.

    Returns a human-readable warning when the heuristic *confidently* disagrees,
    or None when it agrees, is unsure, or has too little evidence to speak.

    **This never changes the stored domain.** It is a review signal, by advisor
    decision: the heuristic is brittle enough on this corpus that letting it
    overrule a human declaration would replace a loud failure mode with a quiet
    one. Callers log the warning; nothing acts on it automatically.
    """
    verdict = classify_text(" ".join(filter(None, (title, abstract, extra))))
    if verdict.looks_off_topic:
        return (
            f"declared domain {declared!r} but the title/abstract match neither "
            f"{RESEARCH_DOMAINS} strongly enough to corroborate it "
            f"({verdict.describe()}) — review whether this paper belongs in the corpus"
        )
    if verdict.confident and verdict.best != declared:
        return (
            f"declared domain {declared!r} but the text looks like "
            f"{verdict.best!r} ({verdict.describe()}) — review the label"
        )
    return None


def curate_declared_domain(
    declared: str, title: str = "", abstract: str = "", extra: str = ""
) -> str | None:
    """Decide a paper's domain for one-off corpus curation, or None to exclude it.

    **Unlike `verify_declared_domain`, this one decides.** It is used by corpus
    curation — the 2026-09-28 backfill and the integration fixture loader — where
    correcting historical labels is the explicit goal. It is deliberately not
    wired into any ingestion path.

    Returns None when the text corroborates neither domain, meaning the paper is
    a bad search-ingestion hit and belongs out of the corpus entirely rather than
    relabelled into a third category.

    For the real 127-paper corpus the automated verdict was reviewed
    paper-by-paper against titles before being applied, and the reviewed outcome
    is recorded in `scripts/domain_backfill.py` as an explicit manifest rather
    than recomputed — so the historical curation is auditable and cannot silently
    change if these term lists are ever edited.
    """
    verdict = classify_text(" ".join(filter(None, (title, abstract, extra))))
    if verdict.looks_off_topic:
        return None
    if verdict.confident:
        return verdict.best
    # Unsure: keep what the caller declared, provided it is a real domain.
    return declared if declared in RESEARCH_DOMAINS else verdict.best
