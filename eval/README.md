# eval/ — evaluation harness

Closes Phase 4. **Nothing here reports a quality number yet, by design.**

Every claim this project has made so far is about *mechanism* — the threshold is derived,
the cluster no longer swallows half the corpus, the domain label matches arXiv's primary
category. None is about *output*: nobody has yet looked at a ranked gap list and judged
whether the gaps are real. This harness is the apparatus for finding out, and it is
deliberately built so that running it without human labels produces an explicit "not
measured" rather than a zero.

## Workflow

```bash
python eval/make_label_sheet.py      # -> eval/label_sheet.csv, awaiting labels
# ... a human fills the `label` column ...
python eval/score.py                 # precision@k per domain and ranking, Wilson 95% CIs

python eval/extraction_audit.py      # -> eval/extraction_audit.csv, awaiting verdicts
python eval/pretriage.py             # OPTIONAL, LLM triage, NOT ground truth
```

## The label sheet

Top 20 gaps per domain under **three rankings**, all drawn from the same scored pool so a
precision difference is attributable to the ranking rather than to different candidates:

| system | ranking |
|---|---|
| `current` | the live system (two-tier, 0.40/0.35/0.25 composite) |
| `frequency` | `frequency_score` only — the obvious naive baseline |
| `random` | fixed-seed shuffle (seed 20260929, reproducible) |

Rows are **interleaved and sorted by gap text**, so the annotator cannot tell which
system proposed a row while labelling. A gap appearing under several rankings needs
labelling once; `score.py` propagates the label by `(domain, gap_text)`.

Labels, deliberately three rather than a 1–5 scale — a coarse judgement one annotator can
apply consistently beats a fine one they cannot:

- `real_gap` — a genuine open problem a researcher could act on
- `generic_filler` — true but contentless; could be said of almost any paper
- `off_topic` — not a limitation, or not in this domain

`real_gap` is a hit. The two miss categories are reported separately because they mean
different things: filler is an extraction-quality problem, off-topic is a corpus-rules
problem.

## The extraction audit

30 papers sampled with a fixed seed, each extracted limitation shown beside the **source
snippet the model actually read** — re-fetched through `fetch_full_text`, the same code
path extraction used, so the auditor sees the same text rather than a different view of
the paper. Strings the Phase 1b filter removed are included with their rejection reason,
so the filter's precision is auditable too.

Verdicts: `stated` / `implied` / `not_supported`.

## Why the LLM triage is walled off

`eval/pretriage.py` can suggest labels, into a file named
`pretriage_NOT_GROUND_TRUTH.csv`. `eval/score.py` **refuses by filename** to read any
sheet containing `NOT_GROUND_TRUTH` — enforced in code, not by convention, because
pointing `--sheet` at the triage file is exactly the mistake that happens under time
pressure.

Scoring, ranking, domain assignment, thresholds and the extraction filter are all
LLM-free. A model grading its own pipeline's output would be circular, and the resulting
number would be worthless in the way that is hardest to notice.

## What this harness cannot tell you

- **Nothing, until the sheet is filled in.** That is the current state.
- **Single annotator**, so there is no inter-annotator agreement and no way to separate a
  labelling error from a system error.
- **n = 20 per domain per ranking.** Wilson intervals at that size are wide: 14/20 and
  10/20 have overlapping 95% intervals, so a four-hit difference between systems is *not*
  established. Overlapping intervals mean "not shown", not "no difference".
- **The corpus is a curated arXiv CV/MI sample**, not the field. precision@k here says
  nothing about coverage — a system could rank its 20 best gaps perfectly and still miss
  every important open problem in computer vision.

## `domain_ground_truth.csv` — manual domain classification (U1, 2026-10-06)

One row per corpus paper: `arxiv_id, assigned_domain, manual_class`.

- `assigned_domain`: the paper's stored domain at 2026-10-06, after the R1 relabel (60 CV,
  89 MI).
- `manual_class`: a human reading of title and full abstract, `clearly_cv`,
  `clearly_medical` or `ambiguous`. It is taken verbatim from the INV-4 table in
  `PLAN_AUDIT_FIX.md`.
- **Coverage is partial: 64 of 149.**
  - INV-4 reviewed the 64 papers then in computer vision: 57 clearly CV, 4 clearly medical
    (relabelled to MI in R1) and 3 ambiguous.
  - **The 85 medical-imaging papers have never been manually classified**, so their
    `manual_class` is empty. That gap has deliberately not been filled with any classifier.
- **The 3 `ambiguous` papers** (2609.30566, 2609.30682, 2304.09148) are excluded from any
  accuracy figure and reported separately.
