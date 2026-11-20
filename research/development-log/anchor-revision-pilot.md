# Anchor-revision pilot: revised P and E propositions on the development sets

Date: 2026-11-20. Plan: `anchor-revision-pilot-plan.md`. Script: `scripts/pilot_anchor_revision.py`. Outputs: `results/anchor_revision_pilot/`. Anchor encoding (`prepare`), GPU NLI inference (`pvgpu2`, NVIDIA A30, BART-large-MNLI) and evaluation all ran at commit `457b249`, clean tree (`dirty: false`). English-dev and Agora only; no held-out or external benchmark was scored.

## Decision

**Keep the frozen V4 anchors (`anchors/proposition_anchors.json`).** None of the three candidate arms passes the adoption rule of `docs/development-protocol.md` (English-dev combined Δ ≥ +.01, Agora combined Δ ≥ −.01, neither 95% interval entirely below zero). The pre-declared primary arm `v5_green` misses on both sets and loses significantly on Agora (`−.008 [−.018, −.002]`). `v5_red` posts a small significant combined gain on Agora (`+.006 [+.001, +.012]`) and lowers both P–E overlap and negativity correlation, but is flat on English-dev (`+.001`). `v5_green_red` posts the largest axis-only gain on English-dev (`+.017`) but drops below V4 once its revised claim hypotheses enter Stage 2 (`−.007`).

| Arm | Revised slots | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | Mean Δ | Passes |
|---|---:|---:|---|---:|---|---:|---|
| `v4` (B6 reference) | 0 | .537 [.421, .642] | – | .382 [.295, .397] | – | – | – |
| `v5_green` *(primary)* | 9 | .543 [.424, .650] | +.006 [−.037, +.047] | .374 [.277, .389] | −.008 [−.018, −.002] | −.001 | no |
| `v5_red` | 6 | .538 [.419, .638] | +.001 [−.024, +.027] | .389 [.298, .402] | +.006 [+.001, +.012] | +.004 | no |
| `v5_green_red` | 15 | .530 [.407, .636] | −.007 [−.051, +.035] | .382 [.281, .397] | −.000 [−.013, +.011] | −.004 | no |

Paired bootstrap, 4,000 draws, seed 20260923, leaders (English-dev, 174 speeches) and politicians (Agora, 5,272 paragraphs) resampled.

## Decomposition: axis gains are undone by tighter Stage-2 hypotheses

All four arms select the exact same ten claim-hypothesis slots as V4 (`P00, P01, P05, P06, P08` and `E00, E02, E04, E05, E06`), so the V4-hypothesis control cleanly separates Stage-1 direction/window changes from the revised wording of the four edited top-5 claims (`P05, P06, E04, E06`).

| Arm | Stage | English-dev ρ | Δ vs V4 [95% CI] | Agora ρ | Δ vs V4 [95% CI] |
|---|---|---:|---|---:|---|
| `v5_green` | Axis only | .525 | +.009 [−.002, +.022] | .354 | −.004 [−.012, +.002] |
| `v5_green` | Claim check, V4 hypotheses | .469 | +.002 [−.024, +.024] | .343 | +.001 [+.000, +.004] |
| `v5_green` | Claim check, arm's hypotheses | .430 | −.037 [−.128, +.041] | .338 | −.004 [−.028, +.019] |
| `v5_green` | Combined, V4 hypotheses | .544 | +.008 [−.006, +.025] | .380 | −.003 [−.008, +.002] |
| `v5_green` | Combined, arm's hypotheses | .543 | +.006 [−.037, +.047] | .374 | −.008 [−.018, −.002] |
| `v5_red` | Axis only | .520 | +.003 [−.021, +.028] | .365 | +.006 [−.001, +.016] |
| `v5_red` | Claim check (same as V4 hyp.) | .460 | −.008 [−.042, +.027] | .342 | −.000 [−.000, +.001] |
| `v5_red` | Combined | .538 | +.001 [−.024, +.027] | .389 | +.006 [+.001, +.012] |
| `v5_green_red` | Axis only | .533 | +.017 [−.011, +.045] | .361 | +.002 [−.004, +.006] |
| `v5_green_red` | Claim check, V4 hypotheses | .460 | −.008 [−.043, +.031] | .343 | +.001 [−.000, +.004] |
| `v5_green_red` | Claim check, arm's hypotheses | .393 | −.075 [−.171, +.009] | .338 | −.004 [−.028, +.018] |
| `v5_green_red` | Combined, V4 hypotheses | .542 | +.005 [−.024, +.037] | .385 | +.003 [+.000, +.005] |
| `v5_green_red` | Combined, arm's hypotheses | .530 | −.007 [−.051, +.035] | .382 | −.000 [−.013, +.011] |

Contrasts between candidate arms (left minus right):

| Contrast | Score | English-dev Δ [95% CI] | Agora Δ [95% CI] |
|---|---|---|---|
| `v5_red − v5_green` | Axis | −.006 [−.033, +.019] | +.010 [−.002, +.027] |
| `v5_green_red − v5_green` | Axis | +.008 [−.016, +.031] | +.006 [−.001, +.016] |
| `v5_green_red − v5_red` | Axis | +.014 [−.001, +.030] | −.004 [−.012, +.002] |
| `v5_red − v5_green` | Combined, arm's hypotheses | −.005 [−.057, +.053] | +.015 [+.008, +.023] |
| `v5_green_red − v5_green` | Combined, arm's hypotheses | −.013 [−.041, +.016] | +.008 [+.003, +.014] |
| `v5_green_red − v5_red` | Combined, arm's hypotheses | −.008 [−.057, +.037] | −.007 [−.017, −.000] |

Two mechanisms explain the entire 2×2 table:

1. **Tightening `P+05` and `E+04` suppresses Stage-2 entailment.** In V4, the two broader *Conditional* hypotheses supply roughly two-thirds of all window maxima: `P05` wins 886/1,392 (64%) English-dev windows and 16,598/22,668 (73%) Agora windows; `E04` wins 944/1,392 (68%) English-dev and 15,509/22,668 (68%) Agora windows. Replacing them with the narrower exclusive forms (`"We are the only true voice of the ordinary people."` and `"The entire political class has betrayed the people it was elected to serve."`) drops `P05` to 15% on both sets (`P08` takes over at 57%/51%) and drops `E04` to 7%/5% (`E06` takes over at 80%/81%). Joint entailment probabilities compress roughly in half (`PE` median `.0043 → .0027`, `p99` `.061 → .032` on English-dev; `.0055 → .0031`, `p99` `.063 → .041` on Agora). With V4's hypotheses held fixed, both `v5_green` and `v5_green_red` match or slightly exceed V4 (`+.008 / −.003` and `+.005 / +.003 [+.000, +.005]`); switching to the revised hypotheses drives the entire loss (`−.037` and `−.075` on English-dev claim-only Spearman).
2. **The negation-framed `red` contrasts help Agora and reduce negativity correlation despite encoder polarity flips.** Adding `red` significantly raises Agora combined Spearman both alone (`v5_red − v4: +.006 [+.001, +.012]`) and on top of `green` (`v5_green_red − v5_green: +.008 [+.003, +.014]`). It also reduces P–E axis overlap (`.503 → .478` in `v5_red`; `.527 → .510` in `v5_green_red`) and lowers window-level VADER negativity correlation (`+.262/+.269 → +.224/+.240` in `v5_red`; `+.279/+.288 → +.244/+.262` in `v5_green_red`). At the same time, leave-one-out projections onto the V4 own-channel axis show that three of the six negation-framed contrast anchors actually project onto the *populist* side of zero (`P−07: −.042 → +.060`, `E−04: −.180 → +.049`, `E−08: −.002 → +.012`), reflecting dense-encoder sensitivity to negation tokens rather than clean contrast geometry.

## Label-free audit and spot checks

| Arm | cos(P, V4) | cos(E, V4) | cos(P, E) | VADER neg. ρ (English-dev / Agora) | E balance gates | P balance gates |
|---|---:|---:|---:|---|---|---|
| `v4` | 1.0000 | 1.0000 | .5029 | +.262 / +.269 | FAIL: `pronoun_they` | FAIL: `words, pronoun_we` |
| `v5_green` | .9822 | .9817 | .5267 | +.279 / +.288 | pass | FAIL: `words, pronoun_we` |
| `v5_red` | .9739 | .9778 | .4775 | +.224 / +.240 | FAIL: `pronoun_they` | FAIL: `words, pronoun_we` |
| `v5_green_red` | .9599 | .9600 | .5101 | +.244 / +.262 | pass | FAIL: `words, pronoun_we` |

* **Spot-check speeches (English-dev combined percentile; 1.0 is top speech).** Tightening the propositions did not resolve the low-GPD false positives: `Philippines_Arroyo_Famous_2` (GPD .05) stays at `1.000` in every arm; `UK_Brown_Campaign_1` (GPD .00) moves only from `.994` to `.971` (`v5_green`) and `.983` (`v5_green_red`); several low-scoring Mbeki speeches move upward under `v5_green` and `v5_green_red` (e.g. `Campaign_2`: `.845 → .937 / .914`; `Famous_2`: `.276 → .483 / .494`; `Ribbon_2`: `.603 → .667 / .764`).

## Checks and provenance

* **V4 checks (all passed):** recomputed V4 axis Spearman `.5166` (reported `.517`) on English-dev and `.3588` (reported `.359`) on Agora; top-8 windows match step-6 requests row-for-row; claim hypotheses match `results/step6_nli/hypotheses.json`. Rerun V4 combined Spearman on `pvgpu2` reproduces committed B6 (`.5368` vs `.5367` on English-dev; `.3823` vs `.3823` on Agora).
* **Workstation (CPU, `laptop-omarchy`, commit `457b249`, `dirty: false`):** anchor encoding (`--repeat 2`, byte-identical across repeats for all three arms), balance report (`results/anchor_revision_pilot/balance_report.{md,json}`), `prepare` (`25,307` unique premises × `14` union hypotheses = `354,298` pairs, `requests_sha256` `ee07d79a…`, `hypotheses_sha256` `d804f250…`), and `evaluate`.
* **GPU (`pvgpu2`, NVIDIA A30, commit `457b249`, `dirty: false`, `torch==2.14.0+cu126`, `transformers==5.6.2`):** `facebook/bart-large-mnli` (`d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`), fp16, batch 64, 5,536 forward calls, 491.8 s, peak VRAM 3,192.4 MB; 0 exact zeros, 0 exact ones, 12,865 distinct values. Job log: `results/anchor_revision_pilot/nli_out/anchor_pilot_nli_job_log.txt`.
* **Tracked evidence:** `results/anchor_revision_pilot/{stage1_summary.json,results.json,REPORT.md,prepare.manifest.json,evaluate.manifest.json,arm_hypotheses.json,balance_report.md,balance_report.json}`, `results/anchor_revision_pilot/nli_in/hypotheses.json`, `results/anchor_revision_pilot/nli_out/{nli__bart-large-mnli.manifest.json,anchor_pilot_nli_job_log.txt}`, and the three anchor-embedding manifests under `results/anchor_revision_pilot/embeddings/e5-large-v2/sentences/`. All `.npz`, `.parquet` and `disagreement/` files stay untracked.
