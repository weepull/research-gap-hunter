# A9_OPTIONS.md — options for the saturating recency / solution-deficit terms

**Status: ANALYSIS ONLY. No code, threshold or constant was changed to produce this.**
**No recommendation is made.**

Date: 2026-09-28 · Corpus: 116 papers (31 CV / 85 MI) at commit `e4549b9`
Related: `PROJECT_HARDENING_PLAN.md` item **A9**, `PLAN.md` items #2 and #4, and the
Phase 4 entry in `CLAUDE.md` which recorded this as out of scope and needing its own
decision.

Measurements come from a throwaway script run against the live Neo4j and Qdrant,
importing the real `pipeline.gap_scorer`. It was deliberately **not committed**;
it lives at `/tmp/claude-501/rgh/a9/` (`core.py`, `experiment.py`, `diag2.py`,
`sens.py`) and will vanish with the session. Every number below is reproducible
from the current corpus by re-running it.

---

## The problem, quantified

| | computer_vision | medical_imaging |
|---|---:|---:|
| gaps | 28 | 54 |
| gaps sharing a score with another gap | **20 (71%)** | **36 (67%)** |
| distinct score values | 13 | 26 |
| `solution_deficit` at exactly 1.0 | 18 of 28 | 38 of 54 |
| `solution_deficit` at exactly 0.0 | 7 | 14 |
| distinct `solution_deficit` values | **4** | **4** |
| distinct `recency` values | 4 (`0, 0.5, 0.75, 1`) | 7 |
| rank of the best-corroborated gap | **12 of 28** (n=4) | **21 of 54** (n=13) |
| single-paper gaps above it | 11 | 15 |
| single-paper gaps in the top 10 | **10 of 10** | 9 of 10 |

The mechanism: a cluster of one paper, published in the reference window, with no
future direction above the noise floor, scores **exactly** `recency = 1.0` and
**exactly** `solution_deficit = 1.0`. Both maxima come from the *absence* of
evidence rather than its presence, and `frequency` contributes at most 0.078 of
spread, so such gaps pile up at one composite value near 0.62.

Two structural facts worth holding onto while reading the options:

- **`solution_deficit` has only 4 distinct values in each domain.** It is
  effectively a 2-bit signal being weighted at 0.25.
- **`recency` is coarse because the corpora are shallow.** CV spans 2015–2024 but
  only six distinct years; MI spans **2023–2025, three years**. There is very little
  temporal signal in MI to smooth.

---

## Summary of all five options

**computer_vision (28 gaps)**

| Option | exact ties | distinct scores | Spearman vs current | rank of n=4 gap | 1-paper in top 10 |
|---|---:|---:|---:|---:|---:|
| **current** | 20 | 13 | 1.0000 | 12 | 10/10 |
| **A** continuous deficit | **0** | **28** | 0.8916 | 17 ↓ | 10/10 |
| **B** smooth recency | 17 | 16 | 0.8380 | **4** ↑ | 7/10 |
| **C** A + B | **0** | **28** | 0.8462 | 10 ↑ | 7/10 |
| **D** rank by support | 20 | 13 | **0.4745** | **1** | 2/10 |
| **E** shrinkage (k=1) | 20 | 13 | 0.9212 | **1** | 8/10 |

**medical_imaging (54 gaps)**

| Option | exact ties | distinct scores | Spearman vs current | rank of n=13 gap | 1-paper in top 10 |
|---|---:|---:|---:|---:|---:|
| **current** | 36 | 26 | 1.0000 | 21 | 9/10 |
| **A** continuous deficit | **5** | **51** | 0.9065 | 28 ↓ | 9/10 |
| **B** smooth recency | 29 | 31 | 0.8842 | **9** ↑ | 6/10 |
| **C** A + B | **2** | **53** | 0.8737 | 18 ↑ | 9/10 |
| **D** rank by support | 36 | 26 | **0.5237** | **1** | 0/10 |
| **E** shrinkage (k=1) | 36 | **25** | 0.9807 | 18 ↑ | 8/10 |

**The headline is that no single option does both jobs.** Ties and
under-ranked-corroboration are separate problems with separate causes: A attacks
tie count and *worsens* corroboration; B and E attack corroboration and barely
touch tie count. Only C improves both, and only partly.

---

## Option A · Deficit as a continuous value

**Definition used.** Replace `1 − (matches / papers_reporting)` with
`1 − addressedness`, where `addressedness` rescales the **maximum** similarity
between the cluster representative and any same-domain future direction — keeping
the existing self-authorship exclusion — between two anchors taken from the measured
null distribution (limitation × future-direction, same domain):

```
addressedness = clamp( (max_sim − null_p50) / (null_p99 − null_p50), 0, 1 )
deficit       = 1 − addressedness
```

Measured anchors: CV `p50 = 0.8248`, `p99 = 0.9018` (n = 1,152) · MI
`p50 = 0.8379`, `p99 = 0.9126` (n = 6,572).

**Effect on the term.** Deficit goes from 4 distinct values to **27 of 28** (CV) and
**47 of 54** (MI). Papers stuck at exactly 1.0 drop from 18→1 (CV) and 38→0 (MI).

**Top 10, computer_vision:**
```
 1. 0.6167  n=1  d=1.0000  hallucinations
 2. 0.5696  n=1  d=0.7896  Hallucination, where it generates irrelevant information…
 3. 0.5448  n=1  d=0.6901  Loses track of or confuses objects in crowded scenes…
 4. 0.5426  n=1  d=0.6813  Challenges with nearby objects with similar appearance…
 5. 0.5392  n=1  d=0.6900  hallucinates small disconnected components at times
 6. 0.5145  n=1  d=0.5693  Two concepts bleeding into one another
 7. 0.5009  n=1  d=0.5371  not exempt from producing hallucinations…
 8. 0.4817  n=1  d=0.4602  limited problem solving capabilities in certain fields
 9. 0.4722  n=1  d=0.4219  can miss fine structures
10. 0.4682  n=1  d=0.4062  does not produce boundaries as crisply as…
```
**Top 10, medical_imaging:**
```
 1. 0.5988  n=1  d=0.9719  Poor similarity metric
 2. 0.5474  n=1  d=0.7660  Incorrectly underdiagnosing specific subgroups…
 3. 0.5421  n=1  d=0.7448  development of the rehearsal buffer
 4. 0.5353  n=2  d=0.7021  High computational demand
 5. 0.5322  n=1  d=0.7052  The model exhibits significant fairness gaps…
 6. 0.5223  n=1  d=0.6658  The long-tailed distribution problem remains challenging
 7. 0.5105  n=1  d=0.6263  Extensive hyperparameter tuning
 8. 0.4871  n=1  d=0.5171  The study assumes that the generated reports are accurate…
 9. 0.4664  n=1  d=0.4421  Potential for divergent expert opinions…
10. 0.4564  n=1  d=0.4022  Fixed ability to process positional hints
```

**Honestly:**

- **It solves the tie problem outright** (0 and 5 ties) and is the only option that
  does. Scores become almost fully distinct.
- **It does not solve — it worsens — the corroboration problem.** The n=4 CV gap
  falls from rank 12 to **17**, and the top 10 stays **10/10 single-paper**. That is
  not a surprise on reflection: A gives finer resolution to a term that is *already*
  maximal for exactly the gaps in question, so it spreads the singletons out among
  themselves without moving multi-paper gaps up.
- **Interpretability: it changes what the term means.** Today deficit answers *"how
  many distinct future directions address this?"*. Under A it answers *"how close is
  the single closest one?"* — a max, not a proportion. That arguably fixes A9's
  stated defect (the metric is dimensionally incoherent because it divides a
  corpus-wide count by a cluster-local count); a max-similarity is at least
  dimensionally coherent. But "1 − similarity, rescaled" is much harder to explain
  to a reader than "1 − (2 of 3 papers have a proposed solution)", and the current
  `proposed_solutions` list would no longer be the thing the score is computed from,
  breaking the "shown solutions are exactly the counted ones" policy decided
  2026-08-21.
- **Determinism: preserved in the strict sense**, but the anchors are corpus-derived
  and so will drift with ingestion, exactly as `_SOLUTION_THRESHOLDS` already does.
  Bootstrap over the null (200 resamples): CV `p50` 95% CI [0.8220, 0.8269], `p99`
  CI [0.8971, 0.9123]. The anchors are stable, **but the denominator is narrow
  (0.0770) so it amplifies anchor noise ~13×** — a 0.005 shift in either anchor moves
  a mid-range deficit by ~0.065, which is a quarter of the term's weighted range.
- **New unprincipled constant: partly.** The two anchor *values* are measured, not
  chosen. The choice of **p50 and p99 as the anchors** is a judgment, and a
  defensible alternative (p95 as the lower anchor, matching the existing threshold)
  would produce materially different numbers. So: no magic number, but a new
  modelling decision that needs its own justification.

---

## Option B · Recency as smooth decay

**Definition used.** Replace the hard "within 2 years of the reference year" count
with a per-paper linear decay normalised by the corpus's own span, averaged over the
cluster's papers:

```
span    = reference_year − oldest_year_in_domain
recency = mean over papers of  max(0, 1 − (reference_year − year) / span)
```

Chosen over exponential decay specifically because exponential needs a half-life
constant and this does not. Measured spans: **CV 9 years** (2024 back to 2015),
**MI 2 years** (2025 back to 2023).

**Effect on the term.** Distinct recency values 4→6 (CV) and 7→11 (MI).

**Top 10, computer_vision:**
```
 1. 0.6222  n=1  r=1.0000  Challenges with nearby objects with similar appearance…
 2. 0.6222  n=1  r=1.0000  Hallucination, where it generates irrelevant information…
 3. 0.6222  n=1  r=1.0000  Loses track of or confuses objects in crowded scenes…
 4. 0.6194  n=4  r=0.8333  Inadequate robust algorithmic frameworks          <- n=4 reaches rank 4
 5. 0.6056  n=2  r=0.8889  Lack of high-quality annotated data for certain tasks
 6. 0.6056  n=2  r=0.8889  certain aspects may have eluded our scrutiny
 7. 0.5833  n=1  r=0.8889  Two concepts bleeding into one another
 8. 0.5778  n=1  r=0.8889  can miss fine structures
 9. 0.5778  n=1  r=0.8889  does not produce boundaries as crisply as…
10. 0.5778  n=1  r=0.8889  hallucinates small disconnected components at times
```
**Top 10, medical_imaging:**
```
 1. 0.6078  n=1   r=1.0000  The study assumes that the generated reports are accurate…
 2. 0.6059  n=1   r=1.0000  The long-tailed distribution problem remains challenging
 3. 0.6059  n=1   r=1.0000  challenges of explicit anatomical guidance…
 4. 0.6039  n=1   r=1.0000  Extensive hyperparameter tuning
 5. 0.6039  n=1   r=1.0000  Heightened susceptibility to hallucinations…
 6. 0.6039  n=1   r=1.0000  Lack of reasoning capabilities tailored for complex medical…
 7. 0.5223  n=2   r=0.7500  High computational demand
 8. 0.5029  n=3   r=0.6667  The effectiveness of the proposed prompt depends on…
 9. 0.4552  n=13  r=0.3846  Simply enlarging convolution kernel sizes doesn't invariably…
10. 0.4407  n=3   r=0.5000  The simplicity, lightweight, and efficiency of TFA-LT…
```

**Honestly:**

- **It is the best of the five at the corroboration problem**, and it achieves that
  without any special-casing of paper count: the n=4 CV gap goes 12→**4** and the
  n=13 MI gap goes 21→**9**. The mechanism is that a multi-paper cluster almost
  always contains at least one older paper, so averaging a decay over its members
  separates it from a lone new paper — whereas the hard window rounded both to 1.0.
- **It barely dents the tie count**: 20→17 (CV), 36→29 (MI). Six MI gaps still tie at
  the top because every one of their papers is from the reference year, giving
  `r = 1.0` exactly.
- **Determinism: this is the serious problem, and it is worse than it looks.** The
  normaliser is the corpus span, so *every* recency value depends on the single
  oldest paper in the domain. Measured: removing just the oldest CV paper moves the
  span 9→5 years and **changes 22 of 28 recency values, by up to 0.4444** (mean
  0.1259). In MI, removing the oldest moves span 2→1 and changes **25 of 54 values,
  by up to 0.5000** (mean 0.1715). A stored `GapResult` would become incomparable to
  a fresh one after any ingestion that touched the corpus's oldest edge — which is
  precisely the property the 2026-08-20 `_corpus_reference_year` decision was written
  to protect. A fixed-horizon variant (e.g. decay over a stated 5 years) removes this
  hazard but reintroduces a constant.
- **MI has almost nothing to smooth.** A 3-year corpus gives a 2-year span, so the
  decay can only take values in `{1, 0.5, 0}` per paper. The "smoothness" is an
  artifact of averaging across cluster members, not of temporal resolution.
- **Interpretability: mildly worse but still explainable** — "average freshness of
  the papers reporting it, relative to the corpus's own age range". The relative-to-
  corpus part is the confusing bit, and is also the determinism hazard.
- **New unprincipled constant: none** as defined here. That is the whole reason this
  formulation was chosen, and it is bought at the price of the span dependency above.

---

## Option C · A + B together

**Top 10, computer_vision:**
```
 1. 0.5778  n=1  r=0.8889 d=1.0000  hallucinations
 2. 0.5696  n=1  r=1.0000 d=0.7896  Hallucination, where it generates irrelevant…
 3. 0.5448  n=1  r=1.0000 d=0.6901  Loses track of or confuses objects in crowded scenes…
 4. 0.5426  n=1  r=1.0000 d=0.6813  Challenges with nearby objects with similar appearance…
 5. 0.5003  n=1  r=0.8889 d=0.6900  hallucinates small disconnected components at times
 6. 0.4757  n=1  r=0.8889 d=0.5693  Two concepts bleeding into one another
 7. 0.4737  n=2  r=0.8889 d=0.4726  Lack of high-quality annotated data for certain tasks
 8. 0.4683  n=2  r=0.8889 d=0.4509  certain aspects may have eluded our scrutiny
 9. 0.4621  n=1  r=0.8889 d=0.5371  not exempt from producing hallucinations…
10. 0.4494  n=4  r=0.8333 d=0.3200  Inadequate robust algorithmic frameworks
```
**Top 10, medical_imaging:**
```
 1. 0.5223  n=1  r=1.0000 d=0.6658  The long-tailed distribution problem remains challenging
 2. 0.5105  n=1  r=1.0000 d=0.6263  Extensive hyperparameter tuning
 3. 0.4871  n=1  r=1.0000 d=0.5171  The study assumes that the generated reports are accurate…
 4. 0.4478  n=2  r=0.7500 d=0.7021  High computational demand
 5. 0.4343  n=1  r=1.0000 d=0.3215  Heightened susceptibility to hallucinations…
 6. 0.4238  n=1  r=0.5000 d=0.9719  Poor similarity metric
 7. 0.4206  n=1  r=1.0000 d=0.2587  challenges of explicit anatomical guidance…
 8. 0.4180  n=1  r=1.0000 d=0.2563  Lack of reasoning capabilities tailored for complex medical…
 9. 0.3724  n=1  r=0.5000 d=0.7660  Incorrectly underdiagnosing specific subgroups…
10. 0.3671  n=1  r=0.5000 d=0.7448  development of the rehearsal buffer
```

**Honestly:**

- **The only option that improves both axes**: ties 20→**0** (CV) and 36→**2** (MI),
  while the n=4 CV gap moves 12→10 and the n=13 MI gap 21→18.
- **The improvements do not compound the way you might hope.** B alone put the n=4
  gap at rank 4; adding A pushes it back to 10, because A's finer deficit resolution
  penalises that gap (`d = 0.3200` — it has a close future direction) more than the
  singletons around it. The two changes partly cancel on the corroboration axis.
- **MI's top 10 is still 9/10 single-paper.** Ties are gone; the composition barely
  changed.
- **It inherits both downsides**: A's anchor-choice decision and narrow denominator,
  and B's span dependency (22/28 and 25/54 values move when the oldest paper leaves).
- **Interpretability: the weakest of the five.** Both terms become relative-to-corpus
  quantities, so neither score nor sub-score can be read without knowing the corpus's
  age range and its null distribution. Every number on the card becomes conditional
  on the corpus rather than on the paper.
- **New unprincipled constant: none numerically**, but two new modelling decisions.

---

## Option D · Leave the formula; rank primarily by supporting-paper count

**Definition used.** Formula and all three terms untouched. Sort key becomes
`(−supporting_paper_count, −score, −newest_year, description)`; score is displayed as
a secondary figure.

**Top 10, computer_vision** (note the score column is no longer monotonic):
```
 1. 0.5903  n=4  Inadequate robust algorithmic frameworks
 2. 0.5778  n=3  Difficulty in achieving accurate representations consistently due…
 3. 0.5139  n=2  Insufficient advanced sensor technologies
 4. 0.5083  n=2  exploratory and not entirely robust in text-to-mask task
 5. 0.4694  n=2  Lack of high-quality annotated data for certain tasks
 6. 0.4694  n=2  certain aspects may have eluded our scrutiny
 7. 0.3833  n=2  remains challenging to process high-resolution images
 8. 0.2194  n=2  Current video-language models rely on object recognition abilities…
 9. 0.6222  n=1  Challenges with nearby objects with similar appearance…
10. 0.6222  n=1  Hallucination, where it generates irrelevant information…
```
**Top 10, medical_imaging:**
```
 1. 0.4821  n=13  Simply enlarging convolution kernel sizes doesn't invariably enhance…
 2. 0.3055  n=5   Relying on additional supervised information, namely, expert knowledge…
 3. 0.3716  n=4   remains challenging to expand PCLT20K dataset to support additional…
 4. 0.1071  n=4   Performance varies significantly across different medical modalities
 5. 0.5343  n=3   Registration between multimodal medical images with large deformation…
 6. 0.5029  n=3   The effectiveness of the proposed prompt depends on the capabilities…
 7. 0.4990  n=3   The simplicity, lightweight, and efficiency of TFA-LT…
 8. 0.3618  n=3   Limited generalizability to other imaging modalities
 9. 0.6098  n=2   High computational demand
10. 0.4868  n=2   Limited performance in other modalities (e.g., MRI or Ultrasound)
```

**Honestly:**

- **It fixes the corroboration complaint completely and by construction** — the
  best-corroborated gap is rank 1 in both domains, and MI's top 10 contains **zero**
  single-paper gaps.
- **It does nothing about ties.** Tie counts and distinct-score counts are identical
  to current (20/13 and 36/26), because the formula is untouched. The ties simply stop
  determining rank.
- **It is the largest departure from current output by far**: Spearman **0.4745** (CV)
  and **0.5237** (MI). Roughly half the rank information is different.
- **The visible cost is a non-monotonic score column.** MI rank 2 scores 0.3055 while
  rank 4 scores 0.1071 and rank 9 scores 0.6098. A reader seeing a list ordered 1, 2,
  3 with scores 0.48, 0.31, 0.37 will reasonably conclude the score is broken. This
  needs a real UI change — probably grouping by support count with explicit headers —
  not just a re-sort, or it reads as a bug.
- **It rewards the largest cluster, which is not the same as rewarding the
  best-attested gap.** MI rank 1 is the 16-limitation / 13-paper cluster — the residue
  of the mega-cluster Phase 2 broke up. Under D, cluster size becomes the primary
  ranking signal, so any future clustering regression promotes itself to the top of
  the page. That is a coupling worth weighing: the cap exists precisely because
  cluster size is not fully trustworthy.
- **Interpretability: the sub-scores stay exactly as they are**, which is a genuine
  advantage — nothing about what `recency` or `deficit` *means* changes. What changes
  is the claim the page makes: from "ranked by how promising" to "grouped by how
  well-attested".
- **Determinism: unaffected.** No term changes; the sort key is total.
- **New unprincipled constant: none.** It is the only option with no new modelling
  decision at all.

---

## Option E · Shrink recency and deficit toward the corpus mean by evidence weight

**My own suggestion, offered as a fifth option, not a recommendation.**

**Rationale.** A, B and C all treat the symptom (coarse terms) rather than the cause.
The cause is that `recency = 1.0` and `deficit = 1.0` for a one-paper cluster are not
*measurements* — they are what a single observation always produces when it falls in
the window and has no near neighbour. A single observation should not be allowed to
express an extreme value with full confidence. Standard empirical-Bayes shrinkage
says so directly: pull each cluster's term toward the corpus prior in proportion to
how little evidence it has.

**Definition used.** Terms and weights unchanged in meaning; each is shrunk before
weighting, with one pseudo-observation (`k = 1`, i.e. Laplace/add-one):

```
prior_t = paper-weighted mean of term t across the domain's clusters
shrunk  = (n · term + 1 · prior_t) / (n + 1)        where n = supporting papers
```

A one-paper cluster lands halfway to the prior; a ten-paper cluster moves ~9%.

**Top 10, computer_vision:**
```
 1. 0.5762  n=4  r=0.7538 d=0.9385  Inadequate robust algorithmic frameworks     <- n=4 to rank 1
 2. 0.5592  n=3  r=0.9423 d=0.6731  Difficulty in achieving accurate representations…
 3. 0.5434  n=1  r=0.8846 d=0.8462  Challenges with nearby objects with similar appearance…
 4. 0.5434  n=1  r=0.8846 d=0.8462  Hallucination, where it generates irrelevant information…
 5. 0.5434  n=1  r=0.8846 d=0.8462  Loses track of or confuses objects in crowded scenes…
 6. 0.5434  n=1  r=0.8846 d=0.8462  Two concepts bleeding into one another
 7. 0.5378  n=1  r=0.8846 d=0.8462  can miss fine structures
 8. 0.5378  n=1  r=0.8846 d=0.8462  does not produce boundaries as crisply as…
 9. 0.5378  n=1  r=0.8846 d=0.8462  hallucinates small disconnected components at times
10. 0.5378  n=1  r=0.8846 d=0.8462  hallucinations
```
**Top 10, medical_imaging:**
```
 1. 0.5343  n=2  r=0.8652 d=0.8865  High computational demand
 2. 0.4985  n=3  r=0.8989 d=0.6649  Registration between multimodal medical images…
 3. 0.4945  n=1  r=0.7979 d=0.8298  The study assumes that the generated reports are accurate…
 4. 0.4926  n=1  r=0.7979 d=0.8298  The long-tailed distribution problem remains challenging
 5. 0.4926  n=1  r=0.7979 d=0.8298  challenges of explicit anatomical guidance…
 6. 0.4926  n=1  r=0.7979 d=0.8298  Fixed ability to process positional hints
 7. 0.4926  n=1  r=0.7979 d=0.8298  Incorrectly underdiagnosing specific subgroups…
 8. 0.4926  n=1  r=0.7979 d=0.8298  Poor similarity metric
 9. 0.4926  n=1  r=0.7979 d=0.8298  Potential for divergent expert opinions…
10. 0.4926  n=1  r=0.7979 d=0.8298  The model exhibits significant fairness gaps…
```

**Honestly — and this option's headline result is a failure:**

- **It does not reduce ties at all.** 20 of 28 and 36 of 54, identical to current; MI's
  distinct-score count actually *drops* from 26 to **25**. The reason is structural and
  I should have predicted it: two singletons with identical inputs receive identical
  shrinkage, so a transform that is a function of `(term, n)` can never separate gaps
  that already agree on both. **Shrinkage is the wrong tool for the tie problem.**
- **It is very effective on the CV corroboration axis and weak on MI.** The n=4 CV gap
  goes 12→**1**. But the n=13 MI gap only goes 21→**18**, because its recency is
  genuinely low (0.4615) and shrinkage pulls it *up* toward the prior by much less than
  it pulls the singletons *down*.
- **It stays closest to current output of any option that changes the formula:**
  Spearman **0.9212** (CV) and **0.9807** (MI).
- **Determinism: weaker than current, in a way that is easy to miss.** The prior is the
  corpus mean of the term, so *every* gap's score depends on *every other* gap's terms.
  Adding one paper shifts the prior and therefore moves all scores — the same class of
  defect that disqualified option 4B in PLAN.md #4. Unlike 4B it is not relative to the
  *result set* (top_n has no effect), which makes it less severe, but a stored
  `GapResult` is still not comparable across ingestions.
- **Interpretability: the sub-scores stop being directly readable.** A card showing
  `recency 0.8846` for a paper published in the reference year is confusing — the true
  measurement is 1.0 and the displayed figure is a posterior. Either the card shows both
  (raw and shrunk), or it shows a number no reader can reconstruct.
- **New unprincipled constant: arguably none, but `k = 1` is doing real work.** Add-one
  is a conventional default rather than a tuned value, which is the defence. It is still
  a choice that sets exactly how hard singletons are penalised, and `k = 2` or `k = 0.5`
  would reorder the list.

---

## Cross-cutting notes

**The two problems are not one problem.** Tie count is driven by the *granularity* of
`deficit` (4 distinct values) — only A and C fix it. Under-ranked corroboration is
driven by both terms being *maximal in the absence of evidence* — B, D and E fix that,
A makes it worse. Any decision that treats "20 of 28 gaps are tied" and "a 4-paper gap
ranks 12th" as the same issue will end up disappointed by whichever option it picks.

**Every formula option costs cross-run comparability, to differing degrees.** Current
scoring is comparable across runs because all three terms are computed from the cluster
alone plus one corpus-derived year. A adds dependence on the null distribution, B on the
corpus's oldest paper, E on the corpus mean of each term. D is the only option that
keeps comparability fully intact, because it changes no term.

**Extraction quality bounds what any of this can achieve.** Several gaps in every top 10
above are contentless: `"hallucinations"`, `"can miss fine structures"`, `"certain
aspects may have eluded our scrutiny"` (from a survey's own hedging), `"development of
the rehearsal buffer"`. No reweighting makes those better research leads. That is item
**A3** in `PROJECT_HARDENING_PLAN.md`, still blocked on B4 (re-ingesting a paper that
already has graph relationships duplicates its limitations). It is plausible that A9 is
not the binding constraint on output quality right now.

**Option D interacts with the cluster-size cap.** Under D, cluster size becomes the
primary ranking signal, so `_MAX_CLUSTER_SHARE` moves from a quality guard to a
load-bearing ranking parameter. Worth deciding together rather than separately.

**Not measured here, and possibly relevant:** none of the five changes the
`_UNRESOLVED_DEFICIT_FLOOR = 0.3` gate in `cross_domain.get_unresolved_gaps`, which
selects which gaps are eligible for cross-domain matching at all. Under A, C or E the
deficit distribution shifts substantially, so that floor would admit a different set of
gaps and `/cross-domain` output would change even though nothing about the matcher did.
`PROJECT_HARDENING_PLAN.md` A5 already flags that constant as undrived.

---

## Reproducing this

```bash
# services must be up: neo4j (bolt 7687), qdrant (6333)
python /tmp/claude-501/rgh/a9/experiment.py   # tie counts, distinct scores, Spearman, top 10
python /tmp/claude-501/rgh/a9/diag2.py        # term saturation, rank of best-corroborated gap
python /tmp/claude-501/rgh/a9/sens.py         # span sensitivity (B), anchor bootstrap (A)
```

These scripts are throwaway and uncommitted; they will not survive the session. If any
option is chosen, the variant's implementation should be written fresh against the real
`gap_scorer` with tests, not lifted from them.
