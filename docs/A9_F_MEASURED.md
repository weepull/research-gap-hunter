# A9_F_MEASURED.md — Option F, measured on the live corpus

**Status: IMPLEMENTED (Phase 2, 2026-09-29). Every number here came from a script
whose output was read; none is estimated.**

Corpus: 113 papers (28 CV / 85 MI), 15 CV and 50 MI contributing papers, after the
Phase 1b extraction filter. Baseline for all comparisons is the ranking immediately
before Option F, captured to `/tmp/.../before_F.json` at commit `51b8144`.

Option F has two independent parts:

1. **`solution_deficit` becomes continuous** — `1 − addressedness`, where
   `addressedness` rescales the similarity of the **nearest eligible** future direction
   between the domain's null p50 and p99, clamped to [0, 1].
2. **Two-tier ranking** — gaps with ≥2 supporting papers rank above all single-source
   gaps; each tier ordered by (score desc, supporting papers desc, newest year desc,
   description asc).

Recency is unchanged. Smooth decay was rejected: its normaliser is the corpus span, so
removing one old paper moved 22 of 28 CV recency values by up to 0.4444 — it breaks the
determinism `_corpus_reference_year` exists to guarantee.

---

## Headline: the tie problem is solved

| | CV before | CV after | MI before | MI after |
|---|---:|---:|---:|---:|
| gaps | 24 | 24 | 53 | 53 |
| **exact-score ties** | **17 (71%)** | **0** | **38 (72%)** | **3 (6%)** |
| distinct scores | 11 | **24 of 24** | 25 | **51 of 53** |
| distinct `solution_deficit` values | **4** | **24** | **4** | **46** |
| `deficit` at exactly 1.0 | 18 | **1** | 38 | **0** |
| Spearman vs previous ranking | — | **0.7287** | — | **0.5939** |
| corroborated tier (≥2 papers) | — | 5 | — | 16 |
| single-source tier | — | 19 | — | 37 |

The deficit term went from a 2-bit signal weighted at 0.25 to a genuinely continuous
one. CV scores are now **entirely distinct** — 24 of 24.

Spearman of 0.73 (CV) and 0.59 (MI) means this is a substantial reordering, not a
cosmetic one. That is expected: two thirds of gaps previously shared a score, so their
relative order was arbitrary and any real signal necessarily moves them.

---

## Anchor choice: p50/p99 vs p50/p95

The stated rule was to keep p50/p99 **unless p50/p95 gives strictly fewer ties AND
higher Spearman**. Measured:

| | p50/p99 | p50/p95 | p95 better? |
|---|---:|---:|---|
| CV ties | **0** | 3 | no |
| CV Spearman | **0.7287** | 0.7139 | no |
| MI ties | **3** | 15 | no |
| MI Spearman | **0.5939** | 0.5898 | no |

**p50/p95 loses on both criteria in both domains, so p50/p99 is kept.** Not a close
call, and the mechanism is clear: p95 is only ~0.05 above p50, so the narrower span
saturates `addressedness` at 1.0 much sooner. Deficits at exactly 0.0 go from 1 to 7
(CV) and from 7 to **24** (MI) — p50/p95 reintroduces exactly the saturation Option F
exists to remove, just at the other end of the range.

| anchors | CV span | MI span | CV deficits at 0.0 | MI deficits at 0.0 |
|---|---:|---:|---:|---:|
| p50/p99 | 0.0715 | 0.0742 | 1 | 7 |
| p50/p95 | 0.0502 | 0.0533 | 7 | 24 |

p50/p95 also collapses `/cross-domain` to **zero matches in both directions**, because
the deficit floor then admits too few gaps (see below). That alone would disqualify it.

---

## Effect on `_UNRESOLVED_DEFICIT_FLOOR` and `/cross-domain`

`cross_domain.get_unresolved_gaps` keeps only gaps with `solution_deficit > 0.3`, so
changing the deficit distribution changes which gaps are eligible for cross-domain
matching at all — even though the matcher itself is untouched.

| | before | p50/p99 | p50/p95 |
|---|---:|---:|---:|
| CV gaps above the 0.3 floor | 18 of 24 | **16** | 11 |
| MI gaps above the 0.3 floor | 33 of 53 | **27** | 13 |
| `/cross-domain` CV→MI matches | 10 | **7** | 0 |
| `/cross-domain` MI→CV matches | 10 | **4** | 0 |

Fewer cross-domain matches is the expected consequence and is not in itself a
regression: under the old binary deficit, a gap with a *fairly close* future direction
still scored 1.0 and sailed over the floor. It now scores proportionally and some of
those gaps fall below it — which is the floor doing what it was meant to do.

**`_UNRESOLVED_DEFICIT_FLOOR = 0.3` is now the load-bearing undrived constant in this
system, and that is a new situation.** It was previously described (`A5`) as
"effectively a binary switch, not a tunable dial" precisely because deficits saturated
at 0.0 and 1.0 — with only four distinct values, the floor's exact position barely
mattered. Now that deficits are continuous and spread across the full range, 0.3
genuinely selects, and it has never been derived from anything. **This is flagged, not
fixed: changing it is a threshold decision and was not in Phase 2's scope.**

---

## Top 10, computer_vision

Note the score column is **non-monotonic across the tier boundary** — this is the tier
rule working as specified, and it is why the frontend has to show the tier.

```
 1. [CORROBORATED] 0.4751  n=2  d=0.3136  Inadequate robust algorithmic frameworks
 2. [CORROBORATED] 0.4743  n=2  d=0.3370  exploratory and not entirely robust in text-to-mask task
 3. [CORROBORATED] 0.4462  n=3  d=0.0916  Difficulty in achieving accurate representations…human anatomy
 4. [CORROBORATED] 0.4267  n=2  d=0.1469  remains challenging to process high-resolution images
 5. [CORROBORATED] 0.2791  n=2  d=0.2031  Current video-language models rely on object recognition…
 6. [single source] 0.6267 n=1  d=1.0000  Two concepts bleeding into one another
 7. [single source] 0.6000 n=1  d=0.9201  not exempt from producing hallucinations…
 8. [single source] 0.5674 n=1  d=0.7628  Loses track of or confuses objects in crowded scenes…
 9. [single source] 0.5523 n=1  d=0.7026  Challenges with nearby objects with similar appearance…
10. [single source] 0.5480 n=1  d=0.7118  hallucinates small disconnected components at times
```

Before Option F, ranks 1–11 were **all single-paper gaps** and the best-corroborated
four-paper gap sat at rank 12. All five corroborated CV gaps now lead.

## Top 10, medical_imaging

```
 1. [CORROBORATED] 0.5370  n=2   d=0.7080  High computational demand
 2. [CORROBORATED] 0.3982  n=3   d=0.1207  Registration between multimodal medical images…
 3. [CORROBORATED] 0.3720  n=4   d=0.0000  remains challenging to expand PCLT20K dataset…
 4. [CORROBORATED] 0.3620  n=3   d=0.0000  Limited generalizability to other imaging modalities
 5. [CORROBORATED] 0.3620  n=2   d=0.0000  Future work includes addressing the limitations…
 6. [CORROBORATED] 0.3620  n=2   d=0.0000  Limited performance in other modalities (MRI/Ultrasound)
 7. [CORROBORATED] 0.3603  n=2   d=0.0013  Lack of public datasets
 8. [CORROBORATED] 0.3167  n=3   d=0.2695  The simplicity, lightweight, and efficiency of TFA-LT…
 9. [CORROBORATED] 0.3100  n=13  d=0.3060  Simply enlarging convolution kernel sizes doesn't…
10. [CORROBORATED] 0.3066  n=3   d=0.2129  The effectiveness of the proposed prompt depends…
```

MI's top 10 is now entirely corroborated, where before it was 9 of 10 single-source.

---

## What Option F does not fix

- **Three MI gaps still tie** (at 0.3620), all with `deficit` exactly 0.0 — the clamp
  still collapses anything at or above p99. Unavoidable with a clamped rescaling.
- **`_UNRESOLVED_DEFICIT_FLOOR = 0.3` is now genuinely load-bearing and still
  undrived.** See above.
- **Extraction quality bounds the ceiling.** MI rank 5 is "Future work includes
  addressing the limitations of the proposed method" — a future direction stored as a
  limitation, and contentless. The Phase 1b filter rejects the bare cue but not this
  longer form, correctly, since substring matching would delete real gaps. Rank 8's
  "The simplicity, lightweight, and efficiency of TFA-LT shed light on…" is a *strength*
  claim mis-extracted as a limitation. No scoring change repairs mis-extraction.
- **Tier is not a quality judgment.** Two papers reporting similar limitations is
  better evidence than one; it is not evidence the gap matters. At n=15 contributing CV
  papers, "corroborated" means two.

## Reproducing

```bash
python scripts/derive_thresholds.py          # anchors and all five thresholds
python scripts/measure_weight_influence.py   # term influence under the new deficit
```
