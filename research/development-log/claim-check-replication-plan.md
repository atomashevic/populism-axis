# Claim-check replication plan: smaller entailment models

Date: 2026-09-24, written before any candidate was run. This is a replication of the step-6 NLI stage with cheaper models, not a re-selection. The carried V4 + NLI score keeps BART-large-MNLI unless a candidate passes the rule below on the development sets.

## Candidates, chosen on cost before any result

| Candidate | Parameters | Cost relative to BART-large-MNLI (407M) | NLI training data |
|---|---:|---:|---|
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 184M | about 4× cheaper | MNLI, FEVER-NLI, ANLI |
| `cross-encoder/nli-deberta-v3-small` | 44M backbone | about 10× cheaper | SNLI, MNLI |
| `cross-encoder/nli-deberta-v3-xsmall` | 22M backbone | about 20× cheaper | SNLI, MNLI |

The base candidate differs from the small and xsmall ones in training data as well as size; that confound is accepted and noted. All three share the DeBERTa-v3 architecture.

## What is held fixed

Everything except the entailment model: the ten hypotheses (`results/step6_nli/hypotheses.json`), the premises (the stored transductive top-8 windows per document, top window per sentence), the per-window rule (max entailment over a channel's five claims, √(P·E) for the conjunction), the document pooling (mean over the stored windows), and the combination (mean of within-corpus z-scores of the axis and NLI document scores, weight .5). The scoring script `scripts/nli_stage.py` takes the model as one flag and remaps the output columns from each model's own label order.

## Rule, pre-declared

Selection is on English-dev and Agora only. For each candidate, the combined score (B4 axis + candidate NLI) is compared with B6 (B4 axis + BART NLI) by the step-6 paired bootstrap (4,000 draws, seed 20260923, leaders and politicians resampled).

- A candidate is **non-inferior** if its paired Δ Spearman against B6 is at least −.01 on English-dev and at least −.01 on Agora.
- The NLI-only rows and the low-tail calibration diagnostics (median and 99th-percentile entailment, share of windows above .5, rank agreement with BART at window level) are reported for every candidate and decide nothing.
- Every candidate that is non-inferior on both development sets is then scored **once** on the spent sets (English holdout, EU-OOD, US governors, party tweets, PopBERT-EN, Portuguese parliament) under each report's primary system, beside B6. Candidates that fail on the development sets are reported on the development sets only.
- Nothing is chosen on the spent sets. If more than one candidate passes, the cheapest is the replication reported; the others are shown.
- The carried V4 + NLI score stays BART-large-MNLI in every V4 report. A passing candidate is recorded as an equivalent cheaper implementation, not as a new configuration.

## Expected failure mode

The step-6 signal is the ordering of small entailment probabilities (median .001; fewer than 0.3% of windows above .5). A model with equal MNLI accuracy may calibrate that tail differently and rank windows worse. The rule above tests exactly that on the development sets.

## Runs

1. pvgpu1: NLI for the three candidates on `english_dev` and `agora` in one job (`nli_stage.py --model <candidate> --names english_dev,agora`).
2. Local: `scripts/evaluate_nli_replication.py dev`, which applies the rule and writes `results/nli_replication/`.
3. pvgpu1: for passing candidates only, NLI on `english_holdout`, `eu` and the four external request files.
4. Local: `scripts/evaluate_nli_replication.py spent`, one scoring per candidate, then the report.
