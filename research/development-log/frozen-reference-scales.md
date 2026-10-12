# Frozen reference scales: is the score corpus-dependent?

Date: 2026-09-24. Script `scripts/frozen_reference.py`; outputs `results/frozen_reference/`. CPU only, from stored embeddings and NLI outputs.

## Purpose

Every V4 score is transductive. Each corpus supplies its own mean embedding for centering, its own per-channel projection mean and SD, its own softplus activation scale (the nonzero median), and its own document-level mean and SD for the axis and NLI z-scores that the combination averages. A deployed scorer cannot do this one document at a time, and a reviewer can ask whether the reported numbers depend on it. This step freezes all four statistics on the English development set and re-scores every corpus with the fixed values.

## Method

Reference statistics from the 174 English-dev speeches (20,284 three-sentence windows, carried B4 configuration): the mean window embedding, the P and E projection mean and SD, the P and E activation scales, and the document-level mean and SD of six quantities (PE, P and E axis scores; PE, P and E NLI scores) under both pooling rules the reports use, k:8 for speeches and paragraphs and k:1 for sentences and tweets. Each corpus is scored under its report's rule with the reference statistics of the same quantity under the same rule. The NLI premises stay the stored transductive top-K windows, since NLI was run only on those; freezing changes the NLI score's scale, not its premises.

The transductive path was recomputed in the same code and matched the stored score files to below 1e-6 on every corpus before the frozen path was evaluated. English-dev scored against itself is the identity, as it should be.

## Result

The axis is not corpus-dependent in any way that matters. The combination is, through one channel: the document-level scales that set the implicit fusion weight.

| Corpus | System | Target | ρ(transductive, frozen) | transductive | frozen | Δ [95% CI] |
|---|---|---|---:|---:|---:|---|
| Agora | PE axis | Spearman | .989 | .359 | .364 | +.005 [−.001, +.013] |
| Agora | PE axis + NLI | Spearman | .987 | .382 | .393 | +.010 [+.006, +.014] |
| English holdout | PE axis | Spearman | .999 | .417 | .415 | −.002 [−.014, +.007] |
| English holdout | PE axis + NLI | Spearman | .988 | .518 | .547 | +.029 [−.007, +.074] |
| EU 3-vs-0 | PE axis | AUROC | .998 | .890 | .901 | +.011 [+.002, +.014] |
| EU 3-vs-0 | PE axis + NLI | AUROC | .935 | .942 | .940 | −.001 [−.025, +.002] |
| EU anti-elitism | E axis + NLI | AUROC | .994 | .828 | .815 | −.014 [−.019, −.002] |
| EU people-centrism | P axis + NLI | AUROC | .990 | .863 | .851 | −.012 [−.020, −.006] |
| US governors 2010–18 | PE axis | Spearman | .999 | .458 | .461 | +.003 [−.003, +.010] |
| US governors 2010–18 | PE axis + NLI | Spearman | .994 | .526 | .540 | +.014 [−.001, +.028] |
| US all speeches | PE axis + NLI | Spearman | .995 | .581 | .591 | +.010 [+.000, +.022] |
| Party tweets | E axis | AUROC | .999 | .754 | .755 | +.001 [−.001, +.002] |
| Party tweets | E axis + NLI | AUROC | .998 | .756 | .756 | +.001 [−.002, +.004] |
| PopBERT anti-elitism | E axis | AUROC | .999 | .781 | .786 | +.005 [+.003, +.007] |
| PopBERT anti-elitism | E axis + NLI | AUROC | .998 | .806 | .804 | −.002 [−.005, +.000] |
| PopBERT people-centrism | P axis | AUROC | 1.000 | .754 | .754 | .000 [−.003, +.003] |
| PopBERT people-centrism | P axis + NLI | AUROC | .976 | .771 | .760 | −.011 [−.031, +.006] |
| Portugal anti-elitism | E axis | AUROC | 1.000 | .770 | .772 | +.002 [+.000, +.004] |
| Portugal anti-elitism | E axis + NLI | AUROC | 1.000 | .798 | .799 | +.001 [−.000, +.002] |
| Portugal people-centrism | P axis | AUROC | 1.000 | .668 | .667 | −.001 [−.002, −.000] |
| Portugal people-centrism | P axis + NLI | AUROC | .996 | .695 | .688 | −.008 [−.012, −.005] |

### The axis: nothing moves

Freezing the centering mean, the projection scales and the activation scale leaves the axis-only ranking almost untouched: rank agreement is at least .989 on every corpus and at least .998 on all but Agora, and no axis-only metric moves by more than .011. The expected shifts on tweets and Portuguese did not appear (+.001 and +.002 on anti-elitism). The reason is in the corpus statistics: every corpus mean embedding sits within cosine .977 of the English-dev mean, and the per-channel projection SDs are within about 25% of the reference across genres and languages (P .080–.104, E .108–.144 against .104 and .118). The e5 space is anisotropic, but the dominant common component is shared across corpora, so subtracting the wrong corpus's mean costs little. For a single channel, a frozen z-score is monotone in the projection, so its ranking can change only through centering, and that change is negligible here.

| Corpus | ‖μ − μ_ref‖ | cos(μ, μ_ref) | P proj SD | E proj SD | P act. scale | E act. scale |
|---|---:|---:|---:|---:|---:|---:|
| English-dev (reference) | 0 | 1 | .104 | .118 | .495 | .311 |
| English holdout | 1.55 | .997 | .102 | .116 | .438 | .327 |
| Agora | 3.14 | .988 | .093 | .119 | .392 | .435 |
| US governors | 2.97 | .989 | .096 | .120 | .455 | .259 |
| Portugal (translation) | 3.39 | .986 | .096 | .118 | .300 | .312 |
| PopBERT (translation) | 3.75 | .982 | .102 | .144 | .559 | .654 |
| Party tweets | 4.09 | .979 | .080 | .108 | .504 | .400 |
| EU-OOD | 4.26 | .977 | .103 | .128 | .546 | .315 |

### The combination: the document-level scales are an implicit fusion weight

Where freezing does act is on the axis + NLI score, and it acts through the document-level SDs. The combination averages z(axis) and z(NLI); with frozen scales, a corpus whose NLI scores are more spread than the reference's gets more effective weight on NLI, and vice versa. The implied axis weight, in the terms of the fusion-weight sweep (`fusion-weight-sweep.md`), is w = a / (a + b) with a and b the corpus-to-reference SD ratios of the axis and NLI document scores:

| Corpus | System | axis SD ratio | NLI SD ratio | implied w on axis | Δ under freezing |
|---|---|---:|---:|---:|---:|
| Agora | PE, k:8 | .37 | 1.48 | .20 | +.010 |
| English holdout | PE, k:8 | 1.23 | 2.96 | .29 | +.029 |
| US governors | PE, k:8 | 1.29 | 1.89 | .40 | +.014 |
| EU 3-vs-0 | PE, k:1 | .33 | .35 | .49 | −.001 |
| EU anti-elitism | E, k:1 | 1.24 | .41 | .75 | −.014 |
| EU people-centrism | P, k:1 | 1.25 | .31 | .80 | −.012 |
| Party tweets | E, k:1 | 1.25 | .58 | .68 | +.001 |
| PopBERT anti-elitism | E, k:1 | 1.24 | .90 | .58 | −.002 |
| PopBERT people-centrism | P, k:1 | 1.25 | .18 | .87 | −.011 |
| Portugal anti-elitism | E, k:8 | 1.29 | 1.12 | .53 | +.001 |
| Portugal people-centrism | P, k:8 | 1.35 | .78 | .63 | −.008 |

This reproduces the sweep from the other side. On the speech and paragraph corpora, freezing tilts the weight towards NLI (implied w .2–.4), and the metric rises by the amount the sweep predicts for that weight: the holdout at w ≈ .3 gains +.03, US governors +.01, Agora +.01. On the sentence corpora the reference for k:1 is the English-dev speech scored under k:1, that is, the maximum over about a hundred windows, whose NLI spread is far larger than a single sentence's; freezing therefore tilts the weight towards the axis (w .6–.9), and the metric falls by what the sweep predicts for those weights, .01 on EU and PopBERT people-centrism. The document-level mean shifts of about −2.3 reference SDs on every sentence corpus show the same unit mismatch: a sentence is not a speech maximum. Tweets are flat because they are flat at every weight.

## What this establishes

- **The V4 axis is portable.** A frozen scorer with English-dev centering, projection scales and activation scales reproduces the transductive axis ranking to ρ ≥ .99 on eight corpora in four languages and five genres. Transductive standardization is not doing any work that the reported numbers depend on.
- **The only corpus-dependent component is the fusion scale**, and its effect is bounded by the sweep: at most about ±.03, and in the direction the sweep already showed.
- **A frozen scorer needs a reference per unit type.** Freezing document-level scales from speeches and applying them to sentences mis-sets the fusion weight by a factor of two to three. A deployed scorer should carry one set of document-level scales for speech-length documents (k:8) and one fitted on sentence-length units, for which the English-dev *windows* rather than speech maxima are the natural reference. That is a follow-up; the axis-level statistics need no such split.
- **The reported transductive numbers stand**, with the caveat that their fusion weight is corpus-fitted in a label-free way. Nothing here changes a decision.

## Provenance

- Inputs: `results/embeddings/e5-large-v2/spw3_cap60_clean/` (English, Agora), `results/embeddings_eu/e5-large-v2/spw3_cap60/`, `results/embeddings_external/*/e5-large-v2/spw3_cap60/`, the frozen anchor embeddings, and the stored BART-large-MNLI outputs under `results/step6_nli/` and `results/external/*/`.
- Outputs: `reference_stats.json` and `reference_mean_embedding.npy` (the frozen statistics), `results.json`, `results.md`, one `scores_<corpus>.csv` with transductive and frozen columns per system, and `run_manifest.json`.
- Reproduction check: transductive recomputation against the stored score files, maximum absolute difference below 1e-12 on every checked column (PE axis and PE axis + NLI on the five speech, paragraph and EU sets; per-channel axis and axis + NLI on tweets, PopBERT and Portugal). The EU per-channel rows have no stored per-channel column; the final script standardized them after pooling rather than before, which is rank-identical to the order used here.

```bash
uv run --extra cpu python scripts/frozen_reference.py
```
