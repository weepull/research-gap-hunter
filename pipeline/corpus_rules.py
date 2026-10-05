"""Which candidate papers may enter which domain. Final rules, 2026-09-29.

Every paper that has ever been wrongly in this corpus got there because ingestion
accepted whatever a keyword search returned and stamped it with whatever domain the
calling script had in mind. Fourteen papers were removed across two curation passes:
analytic number theory, atomic physics, survey astronomy, control theory, music
generation, copyright law, domain-specific NLP, vehicle motion planning and four
language models.

These rules move the check to *ingest time* and base it on arXiv's own primary category
rather than on anyone's reading of a title.

## The rules

**computer_vision** — accept only if the arXiv primary category is `cs.CV`.
Cross-listing to `cs.CV` is not sufficient: every paper removed in curation pass 2 was
cross-listed `cs.CV` with a `cs.CL` primary.

**medical_imaging** — accept if the primary is `eess.IV` or `physics.med-ph`, **or** if
the primary is `cs.CV` *and* the medical keyword verifier fires. Medical imaging
legitimately spans two primary categories (44 `cs.CV` and 35 `eess.IV` in the current
corpus), so a single-category rule cannot work for it — applying the CV rule to MI would
delete real papers.

**Anything else is rejected**, logged as `rule_rejected`, unless the arXiv id is in
`data/mi_allowlist.txt`.

## The keyword verifier is a verifier

It gates `cs.CV`-primary papers into medical imaging; it **never assigns a domain**. The
caller always declares the domain and the rule either accepts or rejects that
declaration. This is the same restraint as `pipeline/domains.verify_declared_domain`, for
the same reason: the heuristic is brittle enough that letting it decide would trade a
loud failure mode for a quiet one.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline import arxiv_source
from pipeline.domains import RESEARCH_DOMAINS, classify_text, validate_domain

logger = logging.getLogger(__name__)

_ALLOWLIST_PATH = Path("data/mi_allowlist.txt")

# Primary categories accepted outright for each domain.
PRIMARY_RULES: dict[str, frozenset[str]] = {
    "computer_vision": frozenset({"cs.CV"}),
    "medical_imaging": frozenset({"eess.IV", "physics.med-ph"}),
}

# Primary categories that medical imaging accepts *conditionally*, subject to the
# keyword verifier firing on the title and abstract.
CONDITIONAL_PRIMARY: dict[str, frozenset[str]] = {
    "medical_imaging": frozenset({"cs.CV"}),
}


def load_allowlist(path: Path | None = None) -> set[str]:
    """arXiv ids exempt from the primary-category rule.

    An escape hatch for papers a human has judged in-scope despite an unusual primary
    category — deliberately an explicit file of ids rather than a broadened rule, so
    each exemption is visible and attributable. Blank lines and `#` comments ignored.
    """
    destination = path if path is not None else _ALLOWLIST_PATH
    if not destination.exists():
        return set()
    ids = set()
    for line in destination.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            ids.add(entry)
    return ids


def accept(
    domain: str,
    arxiv_id: str,
    primary_category: str,
    title: str = "",
    abstract: str = "",
    allowlist: set[str] | None = None,
) -> tuple[bool, str]:
    """Should this candidate be ingested into `domain`? Returns (accepted, reason).

    `reason` is always populated, including on acceptance, so `data/ingest_log.jsonl`
    records *why* every paper was taken or skipped rather than only the ones that failed.
    """
    domain = validate_domain(domain)
    allowed = allowlist if allowlist is not None else load_allowlist()

    if arxiv_id in allowed:
        return (True, f"allowlisted (primary {primary_category or 'unknown'})")

    if primary_category in PRIMARY_RULES[domain]:
        return (True, f"primary category {primary_category} accepted for {domain}")

    conditional = CONDITIONAL_PRIMARY.get(domain, frozenset())
    if primary_category in conditional:
        verdict = classify_text(" ".join(filter(None, (title, abstract))))
        if verdict.best == "medical_imaging":
            return (
                True,
                f"primary {primary_category} plus medical keyword match "
                f"(mi={verdict.scores.get('medical_imaging', 0)}, "
                f"cv={verdict.scores.get('computer_vision', 0)})",
            )
        return (
            False,
            f"primary {primary_category} requires a medical keyword match for "
            f"{domain}, but the verifier reports {verdict.describe()}",
        )

    expected = sorted(PRIMARY_RULES[domain] | conditional)
    return (
        False,
        f"primary category {primary_category or 'unknown'} is not accepted for "
        f"{domain} (expected one of {expected}); add the id to data/mi_allowlist.txt "
        f"to override",
    )


class AdmissionUnverifiable(RuntimeError):
    """arXiv could not be asked, so the rule could not be applied.

    Fails closed: a paper whose primary category is unknown is not admitted, because
    "could not check" is exactly how the fourteen curated-out papers got in.
    """


@dataclass(frozen=True)
class Admission:
    """The verdict for one paper. `reason` is always populated, accepted or not."""

    arxiv_id: str
    accepted: bool
    reason: str
    primary_category: str


def _unversioned(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def admit(
    domain: str,
    arxiv_ids: list[str],
    known: dict[str, arxiv_source.ArxivCandidate] | None = None,
    allowlist: set[str] | None = None,
) -> dict[str, Admission]:
    """The single admission check every ingestion path runs before extraction (P2).

    Applies `accept` — unchanged, so this is a coverage fix and not a policy change —
    to each id. The rule needs arXiv's *primary category*, which only arXiv reports:
    the tranche path already has it from the category listing and passes it in
    `known`; `/ingest`, `ingest_from_query` and `ingest_domain_papers` start from a bare
    id, so the missing ones are looked up in one batched `fetch_by_ids` call.

    Returns verdicts keyed by the caller's id exactly as given (a version suffix is
    stripped only for the lookup and the allowlist). An id arXiv has no record of is
    rejected. Raises `AdmissionUnverifiable` if arXiv cannot be asked at all.
    """
    domain = validate_domain(domain)
    allowed = allowlist if allowlist is not None else load_allowlist()
    candidates = {_unversioned(k): v for k, v in (known or {}).items()}

    to_fetch = []
    for arxiv_id in arxiv_ids:
        base = _unversioned(arxiv_id)
        if base not in allowed and base not in candidates and base not in to_fetch:
            to_fetch.append(base)
    if to_fetch:
        # Called through the module so a test can answer it offline.
        try:
            candidates.update(arxiv_source.fetch_by_ids(to_fetch))
        except Exception as exc:  # noqa: BLE001 — any failure means "unverified"
            raise AdmissionUnverifiable(
                f"cannot apply the corpus rules: arXiv lookup failed ({exc})"
            ) from exc

    verdicts: dict[str, Admission] = {}
    for arxiv_id in arxiv_ids:
        base = _unversioned(arxiv_id)
        candidate = candidates.get(base)
        if candidate is None and base not in allowed:
            verdicts[arxiv_id] = Admission(
                arxiv_id, False,
                f"arXiv has no record of {base}, so its primary category cannot be "
                f"checked against the {domain} rule",
                "",
            )
            continue
        primary = candidate.primary_category if candidate else ""
        ok, reason = accept(
            domain, base, primary,
            candidate.title if candidate else "",
            candidate.abstract if candidate else "",
            allowlist=allowed,
        )
        verdicts[arxiv_id] = Admission(arxiv_id, ok, reason, primary)
    return verdicts


def describe_rules() -> str:
    """Human-readable summary, for the ingest log header and the README."""
    lines = []
    for domain in RESEARCH_DOMAINS:
        outright = ", ".join(sorted(PRIMARY_RULES[domain]))
        conditional = CONDITIONAL_PRIMARY.get(domain, frozenset())
        text = f"  {domain}: primary in {{{outright}}}"
        if conditional:
            text += (f", or primary in {{{', '.join(sorted(conditional))}}} with a "
                     f"medical keyword match")
        lines.append(text)
    return "\n".join(lines)
