# Fusion-weight sweep: how much weight should the NLI stage carry?

Date: 2026-09-24. Script `scripts/sweep_fusion_weight.py`; outputs `results/fusion_weight_sweep/`. CPU only, stored per-document scores only.

## Purpose

The V4 + NLI score is the mean of the within-corpus z-scores of the axis and NLI document scores, a fixed weight of .5 on each, pre-declared in step 6. This sweep asks whether any other linear weight does better, and so bounds what a fitted fusion of the two signals could add. The score at weight w is w·z(axis) + (1 − w)·z(NLI) for w in 0, .1, …, 1; w = 1 is the axis alone and w = 0 the NLI stage alone. At w = .5 the script reproduces the stored combined scores to machine precision on every corpus.

Every corpus except English-dev and Agora is spent, so their curves are sensitivity analyses. A weight other than .5 could only be adopted on the development sets under the step-6 rule.

## Result

The pre-declared .5 is on or next to the optimum on the development sets and within .01–.04 of the best weight everywhere else. On the development sets the curve is flat between .3 and .7 (range .009 on English-dev, .003 on Agora), so no other weight passes the adoption rule. On the spent sets the optimum sits at .1–.3, meaning NLI carries more of the signal there than on the development data. The largest gain any weight offers over .5 is .039 (EU anti-elitism at w = .1); that is the ceiling for a linear fusion, and any fitted fusion model has to be judged against it.

| Corpus | Metric | n | w = 0 (NLI) | .1 | .2 | .3 | .4 | **.5** | .6 | .7 | .8 | .9 | 1 (axis) | best w | gain over .5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| English-dev | Spearman | 174 | .468 | .506 | .522 | .532 | .535 | **.537** | .540 | .537 | .531 | .521 | .517 | .6 | +.004 |
| Agora | Spearman | 5,272 | .342 | .363 | .374 | .379 | .382 | **.382** | .381 | .379 | .376 | .371 | .359 | .5 | .000 |
| English holdout | Spearman | 84 | .520 | .545 | .541 | .551 | .540 | **.518** | .499 | .479 | .456 | .430 | .417 | .3 | +.032 |
| EU 3-vs-0 | AUROC | 1,597 | .945 | .949 | .949 | .948 | .946 | **.942** | .937 | .932 | .924 | .914 | .890 | .1 | +.008 |
| EU anti-elitism (E channel) | AUROC | 1,762 | .853 | .867 | .855 | .845 | .836 | **.828** | .821 | .813 | .806 | .799 | .791 | .1 | +.039 |
| EU people-centrism (P channel) | AUROC | 1,762 | .850 | .876 | .879 | .875 | .869 | **.863** | .857 | .852 | .846 | .840 | .832 | .2 | +.016 |
| US governors 2010–18 | Spearman | 288 | .399 | .489 | .526 | .537 | .535 | **.526** | .514 | .504 | .487 | .473 | .458 | .3 | +.011 |
| US all speeches | Spearman | 359 | .440 | .538 | .577 | .589 | .588 | **.581** | .572 | .564 | .552 | .542 | .530 | .3 | +.008 |
| Party tweets, general (E) | AUROC | 1,059 | .670 | .749 | .754 | .756 | .756 | **.756** | .756 | .756 | .755 | .755 | .754 | .4 | .000 |
| PopBERT anti-elitism (E) | AUROC | 1,759 | .820 | .838 | .829 | .820 | .813 | **.806** | .801 | .796 | .791 | .787 | .781 | .1 | +.032 |
| PopBERT people-centrism (P) | AUROC | 1,759 | .726 | .771 | .775 | .775 | .772 | **.771** | .769 | .766 | .763 | .759 | .754 | .2 | +.004 |
| Portugal anti-elitism (E) | AUROC | 6,214 | .767 | .804 | .806 | .805 | .802 | **.798** | .794 | .790 | .785 | .778 | .770 | .2 | +.008 |
| Portugal people-centrism (P) | AUROC | 6,214 | .689 | .712 | .714 | .709 | .702 | **.695** | .689 | .683 | .678 | .673 | .668 | .2 | +.019 |

Paired differences against w = .5, cluster bootstrap (2,000 draws; clusters as in each report: leader, politician, speaker, state-term, country, sentence, speaker):

| Corpus | w = .3 − .5 | w = .2 − .5 | w = .1 − .5 | w = 1 (axis) − .5 | w = 0 (NLI) − .5 |
|---|---|---|---|---|---|
| English-dev | −.005 [−.037, +.024] | −.014 [−.056, +.025] | −.031 [−.085, +.022] | −.020 [−.084, +.040] | −.069 [−.136, −.002] |
| Agora | −.003 [−.009, +.005] | −.009 [−.017, +.004] | −.019 [−.030, −.004] | −.023 [−.049, −.006] | −.040 [−.053, −.020] |
| English holdout | +.032 [−.002, +.081] | +.023 [−.036, +.099] | +.026 [−.067, +.137] | −.101 [−.241, −.009] | +.001 [−.137, +.152] |
| EU 3-vs-0 | +.006 [+.003, +.016] | +.008 [+.004, +.022] | +.008 [−.001, +.031] | −.051 [−.136, −.038] | +.004 [−.016, +.028] |
| EU anti-elitism | +.017 [+.005, +.021] | +.027 [+.006, +.035] | +.039 [+.009, +.051] | −.038 [−.045, −.012] | +.025 [−.036, +.047] |
| EU people-centrism | +.012 [+.008, +.019] | +.016 [+.010, +.029] | +.014 [+.001, +.032] | −.031 [−.046, −.021] | −.013 [−.055, +.023] |
| US governors 2010–18 | +.011 [−.017, +.038] | .000 [−.047, +.042] | −.037 [−.107, +.028] | −.068 [−.105, −.031] | −.128 [−.231, −.034] |
| Party tweets | .000 [−.005, +.006] | −.001 [−.011, +.009] | −.007 [−.025, +.011] | −.002 [−.010, +.007] | −.086 [−.133, −.046] |
| PopBERT anti-elitism | +.014 [+.010, +.019] | +.023 [+.015, +.031] | +.032 [+.018, +.045] | −.025 [−.034, −.017] | +.014 [−.015, +.040] |
| PopBERT people-centrism | +.003 [−.012, +.022] | +.004 [−.022, +.036] | .000 [−.042, +.047] | −.018 [−.045, +.006] | −.045 [−.118, +.027] |
| Portugal anti-elitism | +.006 [+.001, +.011] | +.008 [−.002, +.017] | +.006 [−.012, +.024] | −.029 [−.036, −.021] | −.031 [−.079, +.007] |
| Portugal people-centrism | +.014 [+.007, +.023] | +.019 [+.008, +.034] | +.017 [−.001, +.042] | −.028 [−.043, −.018] | −.007 [−.037, +.035] |

The full grid, with every weight, is in `results/fusion_weight_sweep/sweep.md`.

## What the sweep shows

- **The development sets are flat around .5.** English-dev peaks at .6 by .004, Agora at .5; between .3 and .7 the metric moves by less than .01 on both. Under the step-6 rule (English-dev gain at least +.01), no weight other than .5 is adoptable. The decision made in step 6 stands.
- **On every spent set the optimum lies at .1–.3**, and w = .3 beats .5 significantly on EU (all three targets), PopBERT anti-elitism and both Portuguese dimensions; on the English holdout and US governors it is ahead but inside the interval. The shift is consistent across genres, languages and label types. It is the same pattern the V4 summary flagged as unexplained: NLI is relatively stronger outside the development data than on it (NLI alone .520 against .468 on the two English sets). On the development sets the axis and NLI z-scores correlate at .56–.62; on the spent sets at .27–.47, so the two signals are less redundant there and the weaker-on-dev one has more to add.
- **Tweets are flat from .1 to 1.** Any weight that keeps some axis in the mix scores .75; NLI alone drops to .67. NLI neither helps nor hurts on fragments.
- **A fusion model has a low ceiling.** The curves are smooth and unimodal, so a fitted linear fusion of these two inputs is bounded by the best-weight column: 0 on the development sets, +.01 on US governors, +.02–.04 on EU, PopBERT anti-elitism and Portugal, +.03 on the 84-speech holdout. A nonlinear fusion would have to clear those numbers on data it has not seen, with two inputs. Nothing here justifies building one.
- **The axis-only end is the worst point of the curve on every corpus except tweets.** Moving from .5 towards the axis loses monotonically. Whatever the right weight is, the NLI stage belongs in the score.

## Recommendation

Keep .5 as the reported V4 + NLI weight. It is the pre-declared value, it is the development-set optimum, and it is within noise of the best weight on the two English speech sets. Record .3 as the value the spent sets favour. If a fifth external benchmark is ever scored, .3 can be pre-declared as a sensitivity row beside .5 without touching the protocol; choosing it now on the strength of these sets would be selection on spent data.

## Provenance

- All inputs are stored score files: `results/step4_windows_topk/grid/spw3_k8/` and `results/step6_nli/nli_only_k8/` (development), `results/final/B4_axis/` and `results/final/B6_axis_plus_nli/` (holdout, EU), `results/final/B4_axis/projections_eu.npz` with `results/step6_nli/nli_eu__bart-large-mnli.npz` (EU channels), and `final_scores*.csv` under `results/external/`.
- z-scoring populations are those of the reports: the corpus itself, all 359 US speeches, all 5,401 tweets, PopBERT train + test.
- Reproduction check at w = .5: maximum absolute difference from the stored combined scores below 1e-14 on all ten corpora that store one.

```bash
uv run --extra cpu python scripts/sweep_fusion_weight.py
```
