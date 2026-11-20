# Claim-check arms plan: ModernBERT NLI models and decision models at BART's size

Date: 2026-11-19. Results and the commands as run: `claim-check-arms.md`. Follow-up to `claim-check-replication.md`, where three smaller DeBERTa-v3 models failed against BART-large-MNLI. This comparison holds model size near 400M parameters and varies the backbone, the training mix and the head. It is not a re-selection: BART-large-MNLI stays the claim-check model of the frozen configuration whatever the result. Nothing has been run; the GPU work goes to `pvgpu2`.

## Question

The DeBERTa run changed size, backbone and training data together, so it could not say why the candidates ranked windows worse than BART. Four candidates on one backbone separate two explanations: a difference that comes from the backbone, and a difference that comes from how the model was trained to decide and which head reads the answer.

## Arms

| Arm | Model | Backbone | Objective and head | Parameters | Sequences encoded per window |
|---|---|---|---|---:|---:|
| A (B6 reference) | `facebook/bart-large-mnli` | BART-large | 3-class NLI cross-encoder | 407M | 10 |
| B | `tasksource/ModernBERT-large-nli` | ModernBERT-large | multi-task NLI fine-tuning (MNLI, ANLI, SICK, WANLI, doc-nli, LingNLI and others), 3-class | 395M | 10 |
| C | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | ModernBERT-large | binary entailment on a zero-shot classification mix | 395M | 10 |
| D | `wfzyx/von` (Von 1.3 engine, 1.2 weights) | ModernBERT-large | Option-Marker decision head, cross-entropy + Brier | 395M | 10, plus 10 state-free sequences in total |
| E | `convaiinnovations/laya` (English checkpoint) | ModernBERT-large | decision head, RLCD | 421M | 10 |

Hub commits are recorded in `scripts/claim_check_arms.py` and `docs/model-revisions.md`.

The decision models do not save encoder passes on this task. The design note that motivated this plan assumed one pass per window for arms D and E. Both SDKs encode one sequence per question (`von.backends.option_marker_backend.evaluate_noul`; `laya.Agent._encode_state`), so ten separate yes/no questions cost ten sequences, the same as a cross-encoder. Laya collates them into forward calls (six windows, 60 sequences per call here, to match the batch of 64 used for arms A to C) and Von runs them one at a time. A single pass would need one ten-option question, which changes the task from ten independent claim checks to a choice among claims and is not run here. The cost table therefore reports sequences encoded, forward calls and wall time as measured.

## Contrasts

| Contrast | What it isolates |
|---|---|
| B − A | The backbone, with the 3-class NLI task held fixed. |
| C − B | Training on a broad zero-shot mix against NLI data only, and a binary against a 3-class head. |
| D − B, D − C, E − B, E − C | The decision head and its training objective on the same backbone. |

If D and E fall within the range of B and C, a decision model is here a ModernBERT cross-encoder with another head. If they differ from both, the head or its objective changes how windows are ranked.

## Held fixed

The stored top-8 windows of the PE axis (`results/step6_nli/requests_{english_dev,agora}.parquet`), the ten claims (`hypotheses.json`), the per-window rule (maximum over a channel's five claims, √(P·E) for the conjunction), the document mean over the stored windows, and the combination with the axis (mean of within-corpus z-scores, weight .5). Only English-dev and Agora are scored. The English holdout, EU-OOD and the external benchmarks are not loaded by the evaluator and no candidate is scored on them.

## Settings per arm

- **A.** The stored step-6 probabilities define B6. BART is run again on `pvgpu2` for wall time and memory on the same card; that output is compared with the stored file and is not used for scores.
- **B.** P(entailment) from the 3-class softmax, as `scripts/nli_stage.py` computes it for BART; batch 64, 320 tokens, premise truncated first.
- **C.** P(entailment) from the 2-class softmax. `nli_stage.py` writes it into the entailment column of the same file layout.
- **D and E.** Each claim becomes one yes/no question, `Does this passage assert the following claim? {claim}`, asked of the window without criteria. The score is P(yes). The wording is used as written and is not tuned.
- **Storage.** Arms B to E are stored in float32. The stored BART file is float16.

### Unrounded probabilities for D and E

The claim check ranks windows by small probabilities: BART's median entailment is .001 and fewer than 0.3% of windows exceed .5. Both SDKs round the probability they return to four decimals, and Von's default further maps it outside a .2–.8 band. Read through the public calls, most windows would tie.

`scripts/decision_model_stage.py` therefore runs each SDK's own steps (sequence packing, forward pass, Von's correction for criteria-less questions, the shipped temperature) and takes the probability before rounding. For the first 16 windows of each corpus it also calls the public SDK function and stops if the two disagree beyond the rounding grid. Von's banding is switched off (`VON_NOUL_DECISION=raw`), its chain library is off, and a window that would be truncated stops the run for either model. The file also stores the raw yes-minus-no logit.

The script depends on internal functions of `von-sdk` 1.3.7 and `laya` 0.4.37 and was written from their source without running them. The probe in step 2 below is its first execution.

### Cautions from the model cards

- Both cards name a yes/no question without criteria as the weakest way to use the model, and Laya's reports answers that follow the option labels more than the text. The plan keeps the fixed criteria-less question. If the probe shows a constant or near-constant P(yes), the arm is reported with that finding. A version with criteria would be a separate run under its own label.
- Von is archived and its calibration was fitted to its own benchmark. Calibration changes the scale of P(yes); the ranking is what is evaluated here.
- Von's temperature depends on the input, so P(yes) is not a monotone function of the logit margin across windows. The raw margin is stored beside P(yes) for a later check; the evaluator reads P(yes) only.

## Readings

1. **Non-inferiority.** Paired Δ Spearman of the combined score against B6 at least −.01 on English-dev and on Agora (4,000 draws, seed 20260923, leaders and politicians resampled), as in the DeBERTa replication.
2. **Incremental information.** On Agora, the partial Spearman correlation of the candidate's document claim score with the human score given the B6 combined score, with a politician-resampled interval. Beside it, an equal-weight score of three inputs (z of the axis, of BART and of the candidate, 1/3 each) compared with B6 by the same paired bootstrap. A candidate is read as adding information when both intervals lie above zero on Agora. The fusion-weight sweep puts the ceiling of a linear fusion near +.04, so gains will be small. English-dev is reported and decides nothing here; its intervals are about ±.06 wide. A candidate that is non-inferior and adds nothing is an equivalent implementation.
3. **Diagnostics.** Window-level Spearman with BART on P, E and the PE conjunction; median, 99th percentile and share above .5; counts of exact zeros, exact ones and distinct stored values. For each candidate, the 50 Agora windows whose rank on the PE conjunction differs most from BART's are written with their text and the coder score of their paragraph to `results/claim_check_arms/disagreement/` (not tracked) and read by hand. The tracked results carry the mean coder score of the windows the candidate ranks higher and of those BART ranks higher.
4. **Cost.** Sequences encoded, forward calls, wall time and peak VRAM for the Agora run of each arm on the same card.

## Runs

The scripts and this plan are committed and pushed before step 1, since cited results must come from a clean tree. Downloads to the `pvgpu2` Hugging Face cache need approval: about 1.6 GB each for B and C, about 3 GB for Von and about 1.7 GB for the English Laya checkpoint (more if its loader fetches the two other checkpoints in the same repository).

1. `pvgpu2`, arms A to C in one job:

   ```bash
   HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_nli -- bash -c 'O=results/claim_check_arms/nli; python scripts/nli_stage.py --model facebook/bart-large-mnli --out-dir $O && python scripts/nli_stage.py --model tasksource/ModernBERT-large-nli --revision ca476cb923a8637073d4ceb0f19f7fc236e260d4 --probs-dtype float32 --out-dir $O && python scripts/nli_stage.py --model MoritzLaurer/ModernBERT-large-zeroshot-v2.0 --revision a51e07b524299e309dd2b88d48b0cfa2bd9ec598 --probs-dtype float32 --out-dir $O'
   ```

2. `pvgpu2`, probe of each decision model on the first 200 English-dev windows. Read `probe__von.json` and `probe__laya.json`: the SDK agreement, the share of stored values on the four-decimal grid (near zero when unrounded), the spread of P(yes), the device after the run (Laya falls back to CPU when memory runs out), and the torch and transformers versions, which should equal those in the arm A to C manifests.

   ```bash
   UV_WITH=von-sdk==1.3.7 HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_probe_von -- python scripts/decision_model_stage.py --arm von --probe 200
   ```

   ```bash
   UV_WITH=laya==0.4.37 HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_probe_laya -- python scripts/decision_model_stage.py --arm laya --probe 200
   ```

3. `pvgpu2`, the full runs, one job per model. Von runs 240,600 single-sequence calls and is expected to take about an hour; the others take minutes.

   ```bash
   UV_WITH=von-sdk==1.3.7 scripts/remote_job.sh arms_von -- python scripts/decision_model_stage.py --arm von
   ```

   ```bash
   UV_WITH=laya==0.4.37 scripts/remote_job.sh arms_laya -- python scripts/decision_model_stage.py --arm laya
   ```

4. Copy `results/claim_check_arms/nli/` to the workstation through a staging directory (`docs/environment.md`), then:

   ```bash
   uv run --extra cpu python scripts/evaluate_claim_check_arms.py
   ```

5. Report in `research/development-log/claim-check-arms.md`, with the manifests and job logs as evidence.

`von-sdk` needs Python 3.12 or later. Neither SDK is in `uv.lock`; `UV_WITH` adds it for one job, and the manifest records the SDK version beside the torch and transformers versions, which should match those of arms A to C.
