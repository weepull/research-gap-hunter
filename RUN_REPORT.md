# RUN_REPORT.md — long autonomous hardening run

**Every figure below was read out of the live system by `scripts/collect_run_facts.py`.** Nothing here is estimated or recalled.

Baseline: commit `15f7128`, 113 papers (28 CV / 85 MI).

## Corpus, before and after

| | computer_vision | medical_imaging |
|---|---:|---:|
| papers at start | 28 | 85 |
| **papers now** | **64** | **85** |
| contributing papers (frequency denominator) | 48 | 50 |
| limitation nodes | 112 | 103 |
| clusters | 57 | 53 |
| largest cluster | 8 | 16 |

## Ingestion outcomes

`data/ingest_log.jsonl`, 56 entries.

| outcome | total | computer_vision | medical_imaging |
|---|---:|---:|---:|
| `extraction_failed` | 3 | 3 | 0 |
| `no_limitations` | 3 | 3 | 0 |
| `ok` | 26 | 26 | 0 |
| `pdf_failed` | 7 | 7 | 0 |
| `rule_rejected` | 17 | 17 | 0 |

**No tranche completed in full.** See the budget note below.

## Extraction-filter rejections

- raw limitation strings seen during ingestion: **74**
- kept after filtering: **74**
- ingestion-time filter rejection rate: **0.0%**
- total rejections logged to `data/rejected_extractions.jsonl` (including the one-off backfill): **10**

| reason | count |
|---|---:|
| `too_short` | 5 |
| `prompt_echo` | 4 |
| `hedging` | 1 |

## Thresholds: coded vs freshly derived

`DRIFT_TOLERANCE` = 0.002. A constant is rewritten only past that; below it, drift is indistinguishable from resampling the same corpus.

| constant | coded | derived | drift | action |
|---|---:|---:|---:|---|
| `_CLUSTER_THRESHOLDS["computer_vision"]` | 0.8744 | 0.8744 | 0.0000 | within tolerance |
| `_CLUSTER_THRESHOLDS["medical_imaging"]` | 0.8954 | 0.8959 | 0.0005 | within tolerance |
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.8733 | 0.8733 | 0.0000 | within tolerance |
| `_SOLUTION_THRESHOLDS["medical_imaging"]` | 0.8915 | 0.8917 | 0.0002 | within tolerance |
| `_DEFICIT_RESCALE_ANCHORS["computer_vision"][p50]` | 0.8235 | 0.8235 | 0.0000 | within tolerance |
| `_DEFICIT_RESCALE_ANCHORS["computer_vision"][p99]` | 0.8959 | 0.8959 | 0.0000 | within tolerance |
| `_DEFICIT_RESCALE_ANCHORS["medical_imaging"][p50]` | 0.8385 | 0.8385 | 0.0000 | within tolerance |
| `_DEFICIT_RESCALE_ANCHORS["medical_imaging"][p99]` | 0.9127 | 0.9127 | 0.0000 | within tolerance |
| `_CROSS_DOMAIN_THRESHOLD` | 0.8764 | 0.8764 | 0.0000 | within tolerance |

Other coded constants: `_UNRESOLVED_DEFICIT_FLOOR` = 0.3 (**still undrived — see caveats**), `MIN_WORDS` = {'limitations': 3, 'future_directions': 3} (p5 of the measured word distribution), `_DEFICIT_RESCALE_ANCHORS` = computer_vision (0.8235, 0.8959), medical_imaging (0.8385, 0.9127).

## Tests — raw summary lines

```
pytest -q               489 passed, 40 deselected, 1 warning in 1.17s
pytest -m integration   37 passed, 3 skipped, 489 deselected, 1 warning in 39.90s
```

**Corrected by hand on 2026-10-04 and 2026-10-05, not by `collect_run_facts.py`.**

The figures originally recorded here (`489 passed … in 0.73s` and
`37 passed, 3 skipped … in 39.23s`) were measured while the unit tier was leaking live
stores:
- eight `score_gaps` tests and two `/corpus` tests read the production Neo4j;
- 62 API tests (56 in `tests/test_api.py`, 6 in `tests/test_rate_limit.py`) opened the real
  `data/papers.db` through startup self-heal.

The unit count passed only because the production database was running. With it stopped,
the result was 8 failed, 481 passed.

The lines above were measured with Neo4j, Qdrant and Ollama all running, after
`tests/conftest.py` began refusing live services and real SQLite files in the unit tier. The
unit line is identical with all three stopped (`… in 1.18s`).

## Ties and tiers

| | computer_vision | medical_imaging |
|---|---:|---:|
| gaps | 57 | 53 |
| gaps sharing a score | 4 | 3 |
| distinct scores | 55 | 51 |
| corroborated tier | 21 | 16 |
| single-source tier | 36 | 37 |
| above `_UNRESOLVED_DEFICIT_FLOOR` | 32 | 27 |

## `/gaps?domain=computer_vision` — top 15, verbatim

```
[ 1] [CORROB] score=0.4949 f=0.0260 r=1.0000 d=0.5378 n=2 sols=0
     The method's approximation quality may degrade when the trade-off between oracle efficiency and approxim
     papers: 2609.30393 2609.30724
[ 2] [CORROB] score=0.4674 f=0.0677 r=1.0000 d=0.3612 n=4 sols=0
     Tool-based verification is more expensive than answer-only classification, so the tool budget should sca
     papers: 2609.30395 2609.30698 2609.30709 2609.30733
[ 3] [CORROB] score=0.4607 f=0.0312 r=1.0000 d=0.3929 n=2 sols=0
     Behavior labels derived from short trajectories remain ambiguous near the boundary between stationary, s
     papers: 2609.30709 2609.30722
[ 4] [CORROB] score=0.4370 f=0.0208 r=1.0000 d=0.3146 n=2 sols=0
     Existing methods may not be able to handle complex raindrop patterns
     papers: 2609.30393 2609.30758
[ 5] [CORROB] score=0.4267 f=0.0521 r=0.6667 d=0.6901 n=3 sols=0
     Loses track of or confuses objects in crowded scenes, after long occlusions or in extended videos
     papers: 2408.00714 2609.30724 2609.30733
[ 6] [CORROB] score=0.4138 f=0.0312 r=1.0000 d=0.2051 n=2 sols=1
     The best linear readout of a next-embedding model is not its output.
     papers: 2609.30647 2609.31788
[ 7] [CORROB] score=0.4085 f=0.0885 r=0.8000 d=0.3724 n=5 sols=0
     Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cannot test 
     papers: 2207.10077 2609.30402 2609.30595 2609.30613 2609.30698
[ 8] [CORROB] score=0.3922 f=0.0312 r=1.0000 d=0.1189 n=2 sols=1
     Deployment in high-stakes settings may amplify errors or unequal performance across populations
     papers: 2609.30434 2609.30566
[ 9] [CORROB] score=0.3646 f=0.0365 r=1.0000 d=0.0000 n=2 sols=5
     Performance remains coupled to underlying pointmap fidelity due to reliance on external geometry
     papers: 2609.30222 2609.30722
[10] [CORROB] score=0.3604 f=0.0260 r=1.0000 d=0.0000 n=2 sols=1
     Generated motion could also support misleading synthetic footage.
     papers: 2609.30741 2609.30761
[11] [CORROB] score=0.3060 f=0.0729 r=0.7500 d=0.0574 n=4 sols=2
     Tracking errors can cause distinct physical surfaces to permanently collapse into a single canonical tra
     papers: 2410.02730 2609.30222 2609.30703 2609.30761
[12] [CORROB] score=0.2996 f=0.0469 r=0.6667 d=0.1902 n=3 sols=3
     Conditioning is required to resolve multiple templates or none
     papers: 2409.13112 2609.30566 2609.32027
[13] [CORROB] score=0.2961 f=0.0469 r=0.3333 d=0.6426 n=3 sols=0
     Challenges with nearby objects with similar appearance (e.g., multiple identical juggling balls)
     papers: 2312.02145 2408.00714 2609.32036
[14] [CORROB] score=0.2913 f=0.0521 r=0.6667 d=0.1486 n=3 sols=2
     ensuring that similar inputs yield consistent outputs despite the model’s generative nature
     papers: 2312.02145 2609.30245 2609.30698
[15] [CORROB] score=0.2771 f=0.1458 r=0.6250 d=0.0000 n=8 sols=12
     Scaling model capacity and incorporating broader pretraining data is likely to yield further gains
     papers: 1505.04870 2101.09744 2305.10683 2609.30222 2609.30434 2609.30667 2609.30708 2609.30733
```

## `/gaps?domain=medical_imaging` — top 15, verbatim

```
[ 1] [CORROB] score=0.5371 f=0.0250 r=1.0000 d=0.7084 n=2 sols=0
     High computational demand
     papers: 2401.17593 2501.09049
[ 2] [CORROB] score=0.3983 f=0.0450 r=1.0000 d=0.1211 n=3 sols=1
     Registration between multimodal medical images with large deformations remains challenging
     papers: 2403.16502 2405.09446 2409.05040
[ 3] [CORROB] score=0.3720 f=0.0550 r=1.0000 d=0.0000 n=4 sols=10
     remains challenging to expand PCLT20K dataset to support additional tasks
     papers: 2408.14770 2503.17261 2503.19005 2509.10620
[ 4] [CORROB] score=0.3620 f=0.0300 r=1.0000 d=0.0000 n=3 sols=9
     Limited generalizability to other imaging modalities
     papers: 2404.15692 2506.09095 2507.01291
[ 5] [CORROB] score=0.3620 f=0.0300 r=1.0000 d=0.0000 n=2 sols=12
     Future work includes addressing the limitations of the proposed method
     papers: 2405.07988 2409.05040
[ 6] [CORROB] score=0.3620 f=0.0300 r=1.0000 d=0.0000 n=2 sols=1
     Limited performance in other modalities (e.g., MRI or Ultrasound)
     papers: 2405.07988 2410.12831
[ 7] [CORROB] score=0.3604 f=0.0250 r=1.0000 d=0.0018 n=2 sols=3
     Lack of public datasets
     papers: 2403.16502 2506.09095
[ 8] [CORROB] score=0.3168 f=0.0400 r=0.6667 d=0.2699 n=3 sols=1
     The simplicity, lightweight, and efficiency of TFA-LT shed light on a new potential approach for leverag
     papers: 2310.07781 2408.14770 2503.00915
[ 9] [CORROB] score=0.3101 f=0.1800 r=0.4615 d=0.3064 n=13 sols=0
     Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance might pla
     papers: 2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620
[10] [CORROB] score=0.3067 f=0.0500 r=0.6667 d=0.2133 n=3 sols=1
     The effectiveness of the proposed prompt depends on the capabilities of the large model. If the large mo
     papers: 2305.17891 2507.19004 2509.24739
[11] [CORROB] score=0.3060 f=0.0650 r=0.8000 d=0.0000 n=5 sols=7
     Relying on additional supervised information, namely, expert knowledge induced by GPT-4, which may be in
     papers: 2303.10323 2411.18101 2506.07044 2507.00185 2508.04044
[12] [CORROB] score=0.2394 f=0.0200 r=0.5000 d=0.2255 n=2 sols=1
     Lack of image-related inductive bias
     papers: 2312.00634 2503.19005
[13] [CORROB] score=0.2105 f=0.0250 r=0.5000 d=0.1020 n=2 sols=2
     Does not address the issue of anatomical structure variability across different individuals
     papers: 2309.15358 2507.00185
[14] [CORROB] score=0.1850 f=0.0250 r=0.5000 d=0.0000 n=2 sols=8
     Limited access to diverse medical modalities containing pathological manifestations
     papers: 2304.12637 2404.11428
[15] [CORROB] score=0.1620 f=0.0250 r=0.0000 d=0.6078 n=2 sols=0
     Ground truth localization information is not readily available
     papers: 2305.06244 2310.07781
```

## `/cross-domain` computer_vision->medical_imaging — 9 matches, verbatim

```
sim=0.8878
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Scaling size and diversity of pretraining datasets for pathology feature extractors and foundati
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2311.11772
sim=0.8841
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Understanding the impact on computational pathology tasks other than WSI classification, such as
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2311.11772
sim=0.8824
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Further investigation of the proposed modules in other computational pathology tasks
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2402.05373
sim=0.8821
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Tailoring SSL methods to the pathology domain to effectively leverage data.
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2311.11772
sim=0.8782
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: advancement of foundation models for computational pathology
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2410.10260
sim=0.8774
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Expanding datasets to include diverse modalities and scenarios for better generalizability
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2507.19004
sim=0.8773
  GAP: Non-dermoscopic generalization remains open because auxiliary datasets without lesion masks cann
  SOL: Investigate if training a large medical vision model from scratch using only medical data can le
  src=2207.10077 2609.30402 2609.30595 2609.30613 2609.30698  tgt=2304.12637
sim=0.8770
  GAP: Loses track of or confuses objects in crowded scenes, after long occlusions or in extended video
  SOL: Improving zero-shot experiences by developing smarter methods to better utilize datasets and mod
  src=2408.00714 2609.30724 2609.30733  tgt=2406.05285
sim=0.8766
  GAP: Behavior labels derived from short trajectories remain ambiguous near the boundary between stati
  SOL: Improving zero-shot experiences by developing smarter methods to better utilize datasets and mod
  src=2609.30709 2609.30722  tgt=2406.05285
```

## `/cross-domain` medical_imaging->computer_vision — 8 matches, verbatim

```
sim=0.9181
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: extend the SAM-Adapter to tackle even more challenging image segmentation tasks and broaden its 
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2304.09148
sim=0.8998
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Iterative training can be used to add more abnormalities to the condition and improve performanc
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2609.31788
sim=0.8937
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Larger mask-annotated evaluations, lesion-size stratification, and patient/lesion-grouped splits
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2609.30613
sim=0.8932
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Scaling model capacity and incorporating broader pretraining data
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2609.30222
sim=0.8911
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Remaining controls—a mask-oracle upper bound, fully epoch-matched baseline retraining, attributi
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2609.30613
sim=0.8898
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Future research includes clearer separation of base models and fine-tunes in VLM research
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2407.07726
sim=0.8824
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Investigating ways to provide a single stage of equal or better quality
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2307.01952
sim=0.8816
  GAP: Simply enlarging convolution kernel sizes doesn’t invariably enhance segmentation; performance m
  SOL: Incorporating more explicit motion modeling into SAM 2 to mitigate errors in challenging cases
  src=2304.12637 2305.17937 2306.13528 2309.15358 2310.00199 2311.11772 2312.10892 2404.15692 2405.07338 2507.19004 2508.04044 2509.00549 2509.10620  tgt=2408.00714
```

## Papers rejected by the corpus rules at ingest — all 17

| arXiv | primary | title | why |
|---|---|---|---|
| `2609.30056` | `cs.RO` | M3GD: Multi-Modal Multi-View Geometric Diffusion for Camera--LiDAR Nov | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30092` | `cs.RO` | Self-Adaptive VLA for Robust Robot Deployment | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30121` | `cs.CL` | What, When, and How: Audio Description as Constrained Global Optimizat | primary category cs.CL is not accepted for computer_vision (expected one of ['cs |
| `2609.30226` | `cs.LG` | PoEM: Predicting RL Outcomes from Existing Policies | primary category cs.LG is not accepted for computer_vision (expected one of ['cs |
| `2609.30238` | `cs.CL` | SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis wit | primary category cs.CL is not accepted for computer_vision (expected one of ['cs |
| `2609.30247` | `cs.RO` | Rolling-WAM: World Action Models with Rolling Imagination | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30249` | `cs.RO` | RAPID: Robot Agentic Programming from Demonstrations | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30436` | `cs.RO` | WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous  | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30459` | `cs.RO` | VkVIO: Cross-platform GPU Acceleration for Visual-Inertial Odometry wi | primary category cs.RO is not accepted for computer_vision (expected one of ['cs |
| `2609.30592` | `cs.LG` | QSV: Quat-Sphere-Vision for Coupled Quaternion Attention on Spherical  | primary category cs.LG is not accepted for computer_vision (expected one of ['cs |
| `2609.30629` | `eess.SP` | FRESHLATENT: Channel-Aware Latent Adaptation for Resource-Constrained  | primary category eess.SP is not accepted for computer_vision (expected one of [' |
| `2609.30631` | `eess.AS` | Adapting Personalized Speech Enhancement for Low-Latency Audio-Visual  | primary category eess.AS is not accepted for computer_vision (expected one of [' |
| `2609.30659` | `eess.IV` | Image Reconstruction from Phase with Untrained Neural Priors | primary category eess.IV is not accepted for computer_vision (expected one of [' |
| `2609.30670` | `cs.CL` | TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Vide | primary category cs.CL is not accepted for computer_vision (expected one of ['cs |
| `2609.31777` | `eess.IV` | Beyond MSE: Rician Likelihood Denoising for Self-Supervised Cardiac $T | primary category eess.IV is not accepted for computer_vision (expected one of [' |
| `2609.31789` | `eess.IV` | MammoClaw: Towards Skill-Evolving Agent Harness for Breast Cancer Mamm | primary category eess.IV is not accepted for computer_vision (expected one of [' |
| `2609.32844` | `eess.IV` | Mask2Restore: Self-Supervised Ultrasound Despeckling via Inpainting | primary category eess.IV is not accepted for computer_vision (expected one of [' |

## Extraction-filter rejections, sampled

| arXiv | field | reason | text |
|---|---|---|---|
| `2405.14458` | lim | `prompt_echo` | future work includes |
| `2405.14458` | lim | `prompt_echo` | remains challenging |
| `2310.03744` | lim | `too_short` | hallucinations |
| `2312.02145` | fut | `hedging` | overcome current limitations |
| `2303.08446` | lim | `too_short` | domain shift |
| `1904.08980` | lim | `too_short` | Training instabilities |
| `1904.08980` | lim | `too_short` | Dataset bias |
| `1904.08980` | lim | `too_short` | Overfitting |
| `2309.17264` | lim | `prompt_echo` | future work includes |
| `2309.17264` | lim | `prompt_echo` | remains challenging |


---

## Phases: what landed

| phase | item | commit | status |
|---|---|---|---|
| 1a | MI solution threshold re-derived + derivation guard | `01301ad` | done |
| 1b | Deterministic extraction-quality gates + backfill | `51b8144` | done |
| 2 | A9 Option F — continuous deficit + two-tier ranking | `2224402` | done |
| 3a | Deterministic Qdrant point ids + stale-point pruning | `07fb2f7` | done |
| 3b/3c | Rule-gated resumable ingestion + quality report | `3ca9814` | done |
| 3d | Corpus growth by tranche | `efce530` | **partial — stopped on budget** |
| 4 | Evaluation harness with baselines | `8b885a8` | done, **awaiting labels** |
| 5 | Footprint, nightly runner, README, DEPLOYMENT | `630e94e`, `01a44c5` | done |

Supporting commits: `ed71da9` (tranche driver), `fa39f89` (driver rate fix), `a9d6d7e`
(report generation).

## Budget: why 300+300 was not reached

The target was ~300 papers per domain. **149 total were reached (64 CV / 85 MI), and the
measured rate explains why the target was never achievable.**

End-to-end ingestion is **~160 s per paper**: arXiv candidate paging at 3 s per request
(arXiv's stated limit), a full PDF download, Ollama extraction on a local 8B model, then
SQLite and Neo4j writes. 600 papers is therefore **~27 hours**, not the ~7 that the initial
45 s/paper estimate implied.

That 45 s came from a 3-paper smoke test with already-warm arXiv pagination that happened
to draw small PDFs — 3.6× optimistic. The driver now projects from the rate the current run
has actually observed, and `fa39f89` records the correction. **This was an estimation error
on my part, not a change in the work.**

One CV tranche ran, stopped at 36 of 50 papers. No MI tranche ran, so every medical-imaging
figure in this report is unchanged from the baseline and serves as a control: no MI
threshold drifted.

## Threshold drift as the corpus grew — counter-intuitive

Growing CV from 28 to 64 papers moved four CV constants past `DRIFT_TOLERANCE`, **all
downward**:

| constant | before | after | drift |
|---|---:|---:|---:|
| `_CLUSTER_THRESHOLDS["computer_vision"]` | 0.8769 | 0.8744 | 0.0025 |
| `_SOLUTION_THRESHOLDS["computer_vision"]` | 0.8773 | 0.8733 | 0.0040 |
| `_CROSS_DOMAIN_THRESHOLD` | 0.8792 | 0.8764 | 0.0028 |
| `_DEFICIT_RESCALE_ANCHORS["computer_vision"]` | (0.8281, 0.8996) | (0.8235, 0.8959) | 0.0046 / 0.0037 |

**A larger corpus lowered the noise floor.** More papers means more diverse limitation
statements, so the similarity an arbitrary pair reaches by chance falls, and p95 falls with
it. Thresholds become slightly *less* strict as the corpus grows. Anyone assuming the
opposite will misread this table. Every MI constant was unchanged — the control.

## The extraction prompt fix worked, measurably

| | rejection rate |
|---|---:|
| pre-fix corpus (one-off backfill) | **4.3%** (10 of 234) |
| papers ingested with the fixed prompt | **0.0%** |

The prompt used to quote its own cue phrases, and the model echoed them back as findings —
the corpus contained `Limitation` nodes whose entire text was `remains challenging` or
`future work includes`. With the examples removed from the prompt, newly ingested papers
produced nothing for the filter to catch. The filter stays as a second line of defence,
because a future prompt edit could reintroduce the problem.

## BLOCKERS.md

**Empty — no file was written.** Nothing in this run required a decision I was not given.
Four things went wrong and were fixed in-run rather than escalated, because each had one
correct answer that touched no scoring, threshold or domain decision:

1. **Semantic Scholar 404s on brand-new arXiv papers**, failing every paper in the first
   tranche. Fixed by passing arXiv's own metadata through (`extract_paper(..., metadata=)`),
   which also removed a 1 req/sec bottleneck.
2. **`derive_thresholds.py --apply` silently failed on the deficit anchors** — its locator
   did not tolerate the dict's type annotation, so the guard stayed red for a misleading
   reason. All locators now tolerate annotations.
3. **The driver's rate estimate was 3.6× optimistic**, so it would start tranches it could
   not finish. Now projects from observed rate.
4. **My own hedging list was too aggressive**, discarding `"Further research is needed to
   improve robustness and generalizability"` — which names what needs improving. Caught by
   the backfill dry run before it was applied; split into substring and whole-string lists.

Two documentation defects were also found and corrected: the README's ingest command
silently did nothing, and I had recorded the contributing-paper counts as 22/54 when the
measured values were 18/51 (corrected in `8531d7f`).

## What this run cannot tell you

**Output quality is unmeasured.** This is the most important limitation and it is not a
detail. Everything verified here is *mechanism*: thresholds are derived from measured nulls,
clusters are capped, domains match arXiv's primary category, ties are broken deterministically,
`/explain` refuses ungrounded pairings. **No one has judged whether the gaps this system
surfaces are real.** `eval/label_sheet.csv` holds 120 rows over 64 unique gaps, blinded
across three rankings, and `eval/score.py` prints "NO LABELS PRESENT" rather than a number
until a human fills it in.

**Evaluation labels are pending, and the design limits what they will show even then.**

- **Single annotator.** No inter-annotator agreement figure, so a labelling error and a
  system error cannot be separated.
- **n = 20 per domain per ranking.** Wilson intervals at that size are wide enough that
  14/20 and 10/20 overlap — a four-hit difference between the current system and a baseline
  would **not** be established. Overlapping intervals mean "not shown", not "no difference".
- **precision@k says nothing about coverage.** A system could rank its 20 best gaps
  perfectly and still miss every important open problem in computer vision.

**The corpus is a curated arXiv CV/MI sample, not the field.** 149 papers, and the CV half
is now dominated by very recent submissions because ingestion sorts newest-first. 48 CV and
50 MI papers contribute any limitation at all, so every frequency is a fraction of a few
dozen. Wilson intervals on a one-paper versus three-paper gap overlap heavily at this size.

**Extraction is unaudited.** `eval/extraction_audit.csv` has 57 rows over 30 papers with the
source snippet beside each extracted limitation, and every `verdict` column is blank. 13 of
57 rows have no snippet because the PDF could not be re-fetched. Scoring quality is bounded
by extraction quality, and extraction has still only been checked for *parseability*.

**Several gaps in the live top-15 are visibly contentless** — statements like "Poor
similarity metric" clear the length gate and are not boilerplate by any rule here, but they
are not research leads either. No scoring change repairs that; it is an extraction-prompt
problem.

**`_UNRESOLVED_DEFICIT_FLOOR = 0.3` is now load-bearing and still undrived.** It was
harmless when the deficit took four discrete values. Now that the deficit is continuous it
genuinely selects which gaps reach cross-domain matching, and it has never been derived from
anything. Flagged in `A9_F_MEASURED.md`, deliberately not changed here.

**A mid-ingestion `/gaps` call returns partially stale results.** New papers reach Neo4j
immediately but Qdrant only at the re-embed, so clustering cannot find them as neighbours
until then. Harmless if the tranche cycle completes; worth knowing if ingestion is
interrupted.

**Nothing was deployed and no cloud store was touched**, as instructed. The 512 MB footprint
figure is measured locally; a managed tier may differ.
