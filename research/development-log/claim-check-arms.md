# Claim-check arms: ModernBERT NLI models and decision models do not replicate BART

Date: 2026-11-19. Plan: `claim-check-arms-plan.md`. Scripts `scripts/nli_stage.py`, `scripts/decision_model_stage.py`, `scripts/evaluate_claim_check_arms.py`; arms in `scripts/claim_check_arms.py`. Outputs `results/claim_check_arms/`. GPU jobs on `pvgpu2` at commit `e25a2a0`, clean tree; evaluation on the workstation at `7d8a666`, clean tree. English-dev and Agora only.

## Result

None of the four candidates is non-inferior to BART-large-MNLI. All four lose on Agora with intervals entirely below zero, from −.042 to −.147. On English-dev no difference is significant in either direction. Holding size near 400M parameters does not change the outcome of the DeBERTa replication (`claim-check-replication.md`), and BART-large-MNLI stays the claim-check model.

| Arm | Model | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev claim only | Agora claim only | Non-inferior |
|---|---|---:|---|---:|---|---:|---:|---|
| A (B6) | BART-large-MNLI | .537 | – | .382 | – | .468 | .342 | – |
| B | ModernBERT-large-nli | .500 | −.037 [−.103, +.020] | .341 | −.042 [−.059, −.026] | .329 | .261 | no |
| C | ModernBERT-large-zeroshot-v2.0 | .489 | −.048 [−.104, +.013] | .278 | −.104 [−.139, −.072] | .287 | .172 | no |
| D | Von | .452 | −.085 [−.197, +.022] | .235 | −.147 [−.180, −.110] | .237 | .080 | no |
| E | Laya | .560 | +.024 [−.045, +.087] | .305 | −.077 [−.111, −.045] | .469 | .211 | no |

Paired bootstrap, 4,000 draws, seed 20260923, leaders (English-dev) and politicians (Agora) resampled. Non-inferior: Δ of the combined score against B6 at least −.01 on both sets.

## No candidate adds information to B6

Every candidate except C carries some signal that B6 lacks on Agora: the partial Spearman correlation of its claim score with the coder score, given the B6 combined score, has an interval above zero for B, D and E. Adding that signal as a third equal-weight input lowers agreement in every case, so none meets the plan's reading of "adds information" (both intervals above zero on Agora).

| Arm | Agora partial ρ [95% CI] | Agora three-input | Δ vs B6 [95% CI] | English-dev partial ρ [95% CI] | English-dev three-input | Δ vs B6 [95% CI] |
|---|---|---:|---|---|---:|---|
| B | +.070 [+.030, +.105] | .371 | −.011 [−.022, −.003] | −.005 [−.167, +.151] | .513 | −.024 [−.081, +.025] |
| C | +.010 [−.049, +.060] | .331 | −.052 [−.075, −.035] | +.021 [−.108, +.143] | .501 | −.036 [−.079, +.009] |
| D | +.066 [+.006, +.096] | .321 | −.061 [−.086, −.038] | +.060 [−.089, +.216] | .497 | −.040 [−.120, +.033] |
| E | +.059 [+.019, +.104] | .349 | −.033 [−.053, −.014] | +.249 [+.098, +.390] | .566 | +.029 [−.026, +.081] |

Laya on English-dev is the one case with a sizable partial correlation (+.249), but its three-input gain there is not significant and it reverses on Agora. This is the pattern of the DeBERTa-v3-xsmall model: a gain on 174 speeches from 57 leaders that does not hold on 5,272 paragraphs.

## Backbone, training mix and head

| Contrast | Isolates | English-dev combined Δ [95% CI] | Agora combined Δ [95% CI] | Agora claim-only Δ [95% CI] | Agora window ρ (PE) |
|---|---|---|---|---|---:|
| B − A | backbone, NLI task held fixed | −.037 [−.103, +.020] | −.042 [−.059, −.026] | −.081 [−.121, −.037] | .398 |
| C − B | zero-shot mix and binary head against NLI-only training | −.011 [−.073, +.060] | −.062 [−.086, −.045] | −.089 [−.145, −.057] | .327 |
| D − B | Von decision head against the 3-class NLI head | −.048 [−.172, +.076] | −.106 [−.143, −.068] | −.181 [−.239, −.129] | −.044 |
| D − C | Von decision head against the binary entailment head | −.037 [−.156, +.074] | −.043 [−.084, +.015] | −.092 [−.151, −.007] | .079 |
| E − B | Laya decision head against the 3-class NLI head | +.060 [−.008, +.129] | −.035 [−.067, −.000] | −.050 [−.102, +.010] | .226 |
| E − C | Laya decision head against the binary entailment head | +.071 [+.000, +.133] | +.027 [−.005, +.073] | +.039 [−.014, +.130] | .146 |

- **Backbone.** With the 3-class NLI task held fixed, ModernBERT-large loses .042 against BART on Agora. The loss is larger for the claim check alone (−.081), which the axis partly absorbs in the combination.
- **Training mix.** Broad zero-shot training with a binary head loses a further .062 against NLI-only training on the same backbone.
- **Decision heads.** Laya falls between the two ModernBERT cross-encoders on Agora: below the 3-class NLI model and not distinguishable from the binary one. Von is below both. Neither decision head improves on a ModernBERT cross-encoder for this task; Von's is clearly worse.

## Why: the candidates rank windows differently, and promote content-poor windows

| Arm | PE median | PE p99 | PE share > .5 | window ρ vs BART (P / E / PE) | doc ρ vs BART | distinct stored values | 50 most-disagreeing windows: candidate higher / BART higher, mean coder score |
|---|---:|---:|---:|---|---:|---:|---|
| A, BART (step 6) | .001 | .064 | < .003 | – | – | – | – |
| B | .016 | .076 | .000 | .27 / .56 / .40 | .54 | 225,260 | 20 at .08 / 30 at .13 |
| C | .011 | .093 | .000 | .20 / .46 / .24 | .39 | 7,976 | 50 at .06 / 0 |
| D | .737 | .820 | .988 | .07 / −.06 / −.09 | −.05 | 221,403 | 44 at .10 / 6 at .18 |
| E | .297 | .859 | .194 | .27 / .43 / .31 | .39 | 24,189 | 48 at .04 / 2 at .10 |

Agora windows (22,668; 226,680 window–claim pairs). The mean coder score over all Agora windows is .09.

- **Von does not rank by the claims.** Its P(yes) sits between .29 and .81 for almost every pair, with 99% of windows above .5 on the conjunction. Its window ranking is uncorrelated with BART's (ρ −.09) and its claim check alone reaches .080 on Agora. Asked without criteria, Von answers yes to nearly any claim, which matches its model card's warning about this form of question.
- **Laya separates windows but mostly by something else.** It is spread across the scale (19% of windows above .5 against under 0.3% for BART) and agrees with BART at ρ .31. Among the 50 windows where it differs most from BART it ranks 48 higher, with a mean coder score of .04, below the corpus mean.
- **Reading the disagreement windows.** The windows the candidates promote and BART leaves near zero are mostly not populist by the coders' scores: 9 of Laya's 48, 12 of Von's 44 and 11 of arm C's 50 have a coder score above zero. Laya's are largely government achievements and national optimism ("It is a country that is optimistic…"); Von's and arm C's are short fragments and rhetorical questions ("Why PASOK? Why PASOK."). Arm B's disagreements run in both directions and include windows with populist content that BART misses, which fits its positive partial correlation. The tables with window text are in `results/claim_check_arms/disagreement/` (not tracked).

As in the DeBERTa replication, a model's quality on its own tasks did not predict how well it orders the low-probability windows that carry the claim-check signal.

## Cost

| Arm | Sequences encoded | Forward calls | Wall time (s) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|
| A | 226,680 | 3,542 | 727 | 3,193 |
| B | 226,680 | 3,542 | 878 | 1,387 |
| C | 226,680 | 3,542 | 876 | 1,387 |
| D | 226,680 | 226,680 | 9,127 | 1,540 |
| E | 226,680 | 3,778 | 1,274 | 4,430 |

Agora, NVIDIA A30. The card was shared with another user's job at 100% utilisation throughout, so wall times are not comparable across arms or with earlier runs; BART took 299 s on an idle card in step 6. The decision models save no encoder passes here: both SDKs encode one sequence per yes/no question, so ten claims cost ten sequences per window, as for a cross-encoder. Von also runs them one at a time.

## Checks and departures from the plan

- **Arm A.** BART run again on `pvgpu2` reproduces the stored step-6 probabilities exactly (largest absolute difference 0 on both sets).
- **Unrounded decision-model probabilities.** On the first 16 windows of each corpus the values computed before rounding match the SDKs' public answers to the rounding grid (gap 0). No window was truncated. No stored value is exactly 0 or 1.
- **Ties.** Arms C and E store far fewer distinct values than pairs (7,976 and 24,189 of 226,680). Arm C runs in float16 like BART, and two-class logits in half precision take few distinct values. Laya runs in bfloat16, its SDK's default. Ties reduce resolution in the low tail; a rerun at full precision was not made.
- **SDK installation.** The plan added the SDKs per job with `uv run --with`. On `pvgpu2` this resolved a torch build that the host's driver cannot use, and the first Von probe failed without CUDA (`arms_probe_von_job_log.txt`). The SDKs were installed into the checkout's environment instead; every decision-model manifest records torch 2.14.0+cu126 and transformers 5.6.2, the same as arms A to C. The `UV_WITH` option was removed from `remote_job.sh`.
- **Batching.** Laya ran six windows (60 sequences) per forward call, close to the batch of 64 pairs of the NLI arms.
- **Laya warning.** The checkpoint ships one temperature outside the range its SDK accepts, for choice questions with eleven or more options. The SDK clamps it and warns; yes/no questions are unaffected.

## Decision

BART-large-MNLI remains the claim-check model. No candidate was scored on the held-out or external sets, and none will be. The only candidate that comes close on English-dev, Laya, loses on Agora. If the claim check is ever replaced, the evidence from both comparisons says to test the replacement's low-tail ordering against BART on Agora first; neither newer NLI training nor a decision head reproduced it here.

## Provenance

- `pvgpu2`, fresh clone of `analysis/claim-check-arms` at `e25a2a0` in `~/populism-axis-claim-check-arms/code`, clean tree; `uv sync --extra cu126`; request tables copied from the workstation with matching SHA-256.
- Jobs (logs as `results/claim_check_arms/nli/*_job_log.txt`, manifests as `nli__<label>.manifest.json`, probe summaries as `probe__<label>.json`):

  ```bash
  GPU_MEM_LIMIT_MB=16000 HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_nli -- bash -c 'O=results/claim_check_arms/nli; python scripts/nli_stage.py --model facebook/bart-large-mnli --out-dir $O && python scripts/nli_stage.py --model tasksource/ModernBERT-large-nli --revision ca476cb923a8637073d4ceb0f19f7fc236e260d4 --probs-dtype float32 --out-dir $O && python scripts/nli_stage.py --model MoritzLaurer/ModernBERT-large-zeroshot-v2.0 --revision a51e07b524299e309dd2b88d48b0cfa2bd9ec598 --probs-dtype float32 --out-dir $O'
  ```

  ```bash
  uv pip install --python .venv/bin/python von-sdk==1.3.7 laya==0.4.37
  ```

  ```bash
  GPU_MEM_LIMIT_MB=16000 HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_probe_von2 -- python scripts/decision_model_stage.py --arm von --probe 200
  ```

  ```bash
  GPU_MEM_LIMIT_MB=16000 HF_HUB_OFFLINE=0 scripts/remote_job.sh arms_probe_laya -- python scripts/decision_model_stage.py --arm laya --probe 200
  ```

  ```bash
  GPU_MEM_LIMIT_MB=16000 scripts/remote_job.sh arms_decision -- bash -c "python scripts/decision_model_stage.py --arm laya && python scripts/decision_model_stage.py --arm von"
  ```

  `GPU_MEM_LIMIT_MB=16000` let the jobs share the card with the other user's job (10.8 GB in use); peak use by our jobs was 4.4 GB.
- Model downloads to the `pvgpu2` Hugging Face cache: `tasksource/ModernBERT-large-nli`, `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`, `wfzyx/von` and `convaiinnovations/laya`. Weight hashes for D and E are in their manifests. All four can be deleted.
- Local evaluation from a clean tree at `7d8a666`:

  ```bash
  uv run --extra cpu python scripts/evaluate_claim_check_arms.py
  ```

  Outputs `results/claim_check_arms/{results.json, results.md, run_manifest.json}` and per-arm document scores `scores_<corpus>__<label>.csv`. The probability files (`nli/*.npz`) and the disagreement tables with window text are not tracked.
