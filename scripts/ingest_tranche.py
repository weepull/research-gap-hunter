"""Resumable, rule-gated tranche ingestion. Phase 3b/3c.

One paper at a time. The embedding model is loaded once. No parallel extraction: Ollama
is a single local process and concurrent requests make it slower, not faster, while
making failures harder to attribute.

**Resumability is the point.** A tranche is tens of minutes of LLM work and network I/O,
and something will fail partway through. State is checkpointed to disk after *every*
paper, so a re-run skips what is already done rather than starting over — and, because
`Limitation` is UNIQUE on `text` and `_upsert_paper_counting` only MERGEs, re-ingesting
an already-ingested paper would attach a *second* set of limitation nodes rather than
replacing the first. Skipping is a correctness requirement, not an optimisation.

Every paper gets one line in `data/ingest_log.jsonl` with an outcome:

    ok                 ingested with at least one surviving limitation
    no_limitations     ingested, but the filter left nothing to score
    pdf_failed         the arXiv PDF could not be fetched; fell back to the abstract
    extraction_failed  Semantic Scholar, Ollama or Neo4j failed after retries
    rule_rejected      the corpus rules in pipeline/corpus_rules.py refused it

Usage:
    python scripts/ingest_tranche.py --domain computer_vision --size 50
    python scripts/ingest_tranche.py --domain medical_imaging --size 50 --report-only
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline import corpus_rules  # noqa: E402
from pipeline.arxiv_source import search_category  # noqa: E402
from pipeline.batch import _get_db, _paper_to_row  # noqa: E402
from pipeline.domains import validate_domain  # noqa: E402

_CHECKPOINT = Path("data/ingest_checkpoint.json")
_LOG = Path("data/ingest_log.jsonl")

# arXiv categories searched per domain. Medical imaging draws on both of its natural
# primary categories; the corpus rules then decide each candidate individually.
_SEARCH_CATEGORIES = {
    "computer_vision": ("cs.CV",),
    "medical_imaging": ("eess.IV", "cs.CV"),
}
# Candidates fetched per arXiv request. Many will be skipped (already ingested) or
# rejected, so the pool has to be several times the tranche size.
_PAGE = 100
_MAX_PAGES_PER_CATEGORY = 12
_RETRIES = 2


class _PdfFailureWatcher(logging.Handler):
    """Observes the extractor's own PDF-failure warning.

    The extractor signals PDF fallback by logging, and `fetch_full_text` returns the
    same ("", "inferred") shape whether the PDF failed or simply had no limitations
    section. Rather than change that signature and its tests, this listens for the
    existing warning — the information is already being emitted, it just was not being
    collected.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.pdf_failed = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — a broken log record must not break ingestion
            return
        if "PDF fetch failed" in message and "falling back" in message:
            self.pdf_failed = True

    def reset(self) -> None:
        self.pdf_failed = False


def _load_checkpoint() -> dict:
    if not _CHECKPOINT.exists():
        return {}
    try:
        return json.loads(_CHECKPOINT.read_text())
    except (json.JSONDecodeError, OSError):
        logging.warning("Checkpoint unreadable; starting a fresh one")
        return {}


def _save_checkpoint(state: dict) -> None:
    _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(_CHECKPOINT)


def _log_outcome(record: dict) -> None:
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _existing_ids() -> set[str]:
    db = _get_db()
    if "papers" not in db.table_names():
        return set()
    return {row[0] for row in db.execute("SELECT arxiv_id FROM papers").fetchall()}


def quality_report(domain: str | None = None) -> dict:
    """Per-tranche quality figures, computed from the ingest log and the live corpus."""
    if not _LOG.exists():
        return {}
    records = []
    for line in _LOG.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if domain is None or entry.get("domain") == domain:
            records.append(entry)
    if not records:
        return {}

    outcomes = Counter(r["outcome"] for r in records)
    attempted = sum(v for k, v in outcomes.items() if k != "rule_rejected")
    ingested = outcomes["ok"] + outcomes["no_limitations"]
    limitation_counts = [r.get("limitations_kept", 0) for r in records if r["outcome"] in ("ok", "no_limitations")]
    filtered = sum(r.get("limitations_filtered", 0) for r in records)
    raw_total = sum(r.get("limitations_raw", 0) for r in records)

    return {
        "domain": domain or "all",
        "log_entries": len(records),
        "outcomes": dict(outcomes),
        "rule_rejection_rate": outcomes["rule_rejected"] / max(1, len(records)),
        "extraction_failure_rate": outcomes["extraction_failed"] / max(1, attempted),
        "pdf_failure_rate": outcomes["pdf_failed"] / max(1, attempted),
        "pct_ingested_yielding_a_limitation": (outcomes["ok"] / max(1, ingested)),
        "mean_limitations_per_ingested_paper": (
            sum(limitation_counts) / max(1, len(limitation_counts))
        ),
        "filter_rejection_rate": filtered / max(1, raw_total),
    }


def print_quality_report(domain: str | None = None) -> None:
    report = quality_report(domain)
    print("\n" + "=" * 92)
    print(f"QUALITY REPORT — {report.get('domain', domain or 'all')}")
    print("=" * 92)
    if not report:
        print("  no ingest log entries yet")
        return
    print(f"  log entries                        : {report['log_entries']}")
    print(f"  outcomes                           : {report['outcomes']}")
    print(f"  rule rejection rate                : {report['rule_rejection_rate']*100:.1f}%")
    print(f"  extraction failure rate (attempted): {report['extraction_failure_rate']*100:.1f}%")
    print(f"  PDF failure rate (attempted)       : {report['pdf_failure_rate']*100:.1f}%")
    print(f"  %% ingested yielding >=1 limitation : "
          f"{report['pct_ingested_yielding_a_limitation']*100:.1f}%")
    print(f"  mean limitations per ingested paper: "
          f"{report['mean_limitations_per_ingested_paper']:.2f}")
    print(f"  extraction-filter rejection rate   : {report['filter_rejection_rate']*100:.1f}%")


def _gather_candidates(domain: str, needed: int, skip: set[str], state: dict) -> list:
    """Pull arXiv candidates until `needed` acceptable, unseen ones are found."""
    accepted, seen_now = [], set()
    allowlist = corpus_rules.load_allowlist()
    per_domain = state.setdefault(domain, {"cursors": {}, "rejected": []})
    cursors = per_domain.setdefault("cursors", {})

    for category in _SEARCH_CATEGORIES[domain]:
        start = int(cursors.get(category, 0))
        pages = 0
        while len(accepted) < needed and pages < _MAX_PAGES_PER_CATEGORY:
            print(f"    arXiv {category} start={start} …", flush=True)
            batch = search_category(category, start=start, max_results=_PAGE)
            start += _PAGE
            pages += 1
            cursors[category] = start
            _save_checkpoint(state)
            if not batch:
                break
            for candidate in batch:
                if candidate.arxiv_id in skip or candidate.arxiv_id in seen_now:
                    continue
                # The shared validator every ingestion path uses (P2). The listing
                # already carries the primary category, so no lookup is made.
                verdict = corpus_rules.admit(
                    domain, [candidate.arxiv_id],
                    known={candidate.arxiv_id: candidate}, allowlist=allowlist,
                )[candidate.arxiv_id]
                ok, reason = verdict.accepted, verdict.reason
                seen_now.add(candidate.arxiv_id)
                if not ok:
                    _log_outcome({
                        "arxiv_id": candidate.arxiv_id, "domain": domain,
                        "outcome": "rule_rejected", "reason": reason,
                        "primary_category": candidate.primary_category,
                        "title": candidate.title[:200],
                        "at": datetime.now(timezone.utc).isoformat(),
                    })
                    continue
                accepted.append((candidate, reason))
                if len(accepted) >= needed:
                    break
        if len(accepted) >= needed:
            break
    return accepted


def run(domain: str, size: int) -> int:
    from graph.populate import _upsert_paper_counting, create_constraints, get_neo4j_driver
    from pipeline.extractor import extract_paper
    from vectors.embed import load_embedding_model

    domain = validate_domain(domain)
    state = _load_checkpoint()
    existing = _existing_ids()

    print("=" * 92)
    print(f"TRANCHE — domain={domain} size={size}")
    print("=" * 92)
    print("corpus rules:")
    print(corpus_rules.describe_rules())
    print(f"already in corpus: {len(existing)} papers")

    print("\ngathering candidates…")
    candidates = _gather_candidates(domain, size, existing, state)
    print(f"  {len(candidates)} candidate(s) passed the corpus rules")
    if not candidates:
        print("nothing to ingest")
        return 0

    # Load the model once, before the loop.
    load_embedding_model()
    watcher = _PdfFailureWatcher()
    logging.getLogger("pipeline.extractor").addHandler(watcher)

    driver = get_neo4j_driver()
    create_constraints(driver)
    db = _get_db()
    outcomes = Counter()
    started = time.monotonic()

    try:
        for i, (candidate, why) in enumerate(candidates, 1):
            pid = candidate.arxiv_id
            print(f"[{i}/{len(candidates)}] {pid}  {candidate.title[:64]}", flush=True)
            watcher.reset()
            record = {
                "arxiv_id": pid, "domain": domain,
                "primary_category": candidate.primary_category,
                "title": candidate.title[:200], "accept_reason": why,
                "at": datetime.now(timezone.utc).isoformat(),
            }

            paper = None
            last_error = None
            for attempt in range(_RETRIES + 1):
                try:
                    # arXiv already gave us title/abstract/year, so skip the
                    # Semantic Scholar lookup: it 404s on brand-new papers and its
                    # 1 req/sec limit dominated the first trial tranche.
                    paper = extract_paper(pid, domain=domain, metadata={
                        "title": candidate.title,
                        "year": candidate.year,
                        "abstract": candidate.abstract,
                    })
                    break
                except Exception as exc:  # noqa: BLE001 — retry, then log and continue
                    last_error = exc
                    if attempt < _RETRIES:
                        wait = 10 * (attempt + 1)
                        print(f"     extraction failed ({exc.__class__.__name__}); "
                              f"retry in {wait}s", flush=True)
                        time.sleep(wait)

            if paper is None:
                record.update({"outcome": "extraction_failed",
                               "error": f"{last_error.__class__.__name__}: {last_error}"[:300]})
                outcomes["extraction_failed"] += 1
                _log_outcome(record)
                _save_checkpoint(state)
                continue

            raw = json.loads(paper.raw_json)
            raw_lims = len(raw.get("limitations", []) or [])
            kept = len(paper.limitations)

            try:
                db["papers"].insert(_paper_to_row(paper), pk="arxiv_id",
                                    replace=False, alter=True)
                with driver.session() as session:
                    session.execute_write(
                        lambda tx, p=paper.model_dump(): _upsert_paper_counting(tx, p)
                    )
            except Exception as exc:  # noqa: BLE001
                record.update({"outcome": "extraction_failed",
                               "error": f"store failed: {exc.__class__.__name__}: {exc}"[:300]})
                outcomes["extraction_failed"] += 1
                _log_outcome(record)
                _save_checkpoint(state)
                continue

            if watcher.pdf_failed:
                outcome = "pdf_failed"
            elif kept == 0:
                outcome = "no_limitations"
            else:
                outcome = "ok"
            record.update({
                "outcome": outcome, "tier": paper.extraction_tier,
                "limitations_raw": raw_lims, "limitations_kept": kept,
                "limitations_filtered": raw_lims - kept,
                "future_directions_kept": len(paper.future_directions),
            })
            outcomes[outcome] += 1
            _log_outcome(record)
            existing.add(pid)
            _save_checkpoint(state)
            print(f"     {outcome}  tier={paper.extraction_tier} "
                  f"limitations {raw_lims}->{kept}", flush=True)
    finally:
        logging.getLogger("pipeline.extractor").removeHandler(watcher)
        driver.close()

    elapsed = time.monotonic() - started
    print(f"\ntranche done in {elapsed/60:.1f} min: {dict(outcomes)}")
    print_quality_report(domain)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True,
                        help="required; no default, per PLAN.md #1 option 1A")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--report-only", action="store_true",
                        help="print the quality report and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.report_only:
        print_quality_report(validate_domain(args.domain))
        return 0
    return run(args.domain, args.size)


if __name__ == "__main__":
    raise SystemExit(main())
