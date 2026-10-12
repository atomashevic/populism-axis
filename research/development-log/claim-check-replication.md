# Claim-check replication: smaller entailment models fail the non-inferiority rule

Date: 2026-09-24. Plan and rule: `claim-check-replication-plan.md`, committed before any candidate ran. Script `scripts/evaluate_nli_replication.py`; outputs `results/nli_replication/`. NLI job on pvgpu1 at commit `f6a84ed`, clean tree.

## Result

None of the three pre-declared DeBERTa-v3 candidates is non-inferior to BART-large-MNLI on the development sets, so none was scored on the spent sets. Every candidate loses significantly on Agora (Δ −.022 to −.041 against B6, intervals entirely below zero), and two of the three lose on English-dev as well. BART-large-MNLI stays the V4 + NLI entailment model.

| Candidate | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev NLI only | Agora NLI only | Non-inferior |
|---|---:|---|---:|---|---:|---:|---|
| B6 reference, BART-large-MNLI | .537 | – | .382 | – | .468 | .342 | – |
| DeBERTa-v3-base-mnli-fever-anli | .497 | −.040 [−.108, +.025] | .360 | −.022 [−.039, −.008] | .288 | .293 | no |
| nli-deberta-v3-small | .499 | −.038 [−.105, +.024] | .341 | −.041 [−.075, −.020] | .347 | .241 | no |
| nli-deberta-v3-xsmall | .562 | +.025 [−.038, +.089] | .341 | −.041 [−.058, −.024] | .483 | .275 | no |

Rule: paired Δ of the combined score against B6 at least −.01 on both development sets (4,000 draws, seed 20260923, leaders and politicians resampled).

## Why they fail: the low tail is calibrated differently

The step-6 signal is the ordering of small entailment probabilities. The candidates rank that tail differently from BART, and from each other:

| Candidate | P median | P p99 | P > .5 | E median | E p99 | E > .5 | window ρ vs BART (P / E / PE) | doc NLI ρ vs BART (Eng / Agora) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BART-large-MNLI (step 6) | .001 | .064 | < .003 | .001 | .064 | < .003 | – | – |
| DeBERTa-v3-base-mnli-fever-anli | .022 | .756 | .030 | .010 | .535 | .012 | .49 / .40 / .44 | .55 / .63 |
| nli-deberta-v3-small | .002 | .117 | .004 | .001 | .105 | .003 | .43 / .24 / .38 | .57 / .46 |
| nli-deberta-v3-xsmall | .005 | .214 | .003 | .003 | .075 | .000 | .40 / .38 / .45 | .65 / .57 |

- **Window-level agreement with BART is only .38–.45** on the PE conjunction, and document-level agreement .46–.65. These models are not noisy copies of BART; they order the windows differently.
- **The best MNLI model is the worst here.** The base model trained on MNLI, FEVER and ANLI is the most confident (3% of windows above .5 on P, 99th percentile .76) and the worst ranker on English-dev (NLI only .288 against BART's .468). Its adversarial training makes it assert entailment of the broad anchor claims for windows that coders did not score as populist; confidence in the tail is not what this stage needs.
- **The xsmall model's English-dev gain does not transfer.** It reaches .562 combined on English-dev (+.025, not significant) and .341 on Agora (−.041, significant). With 57 leader clusters the English-dev interval is ±.06 wide; the Agora result, on 5,272 paragraphs, is the informative one. This is the pattern the plan warned about: a model that is nearly as accurate on MNLI can still rank these windows worse.
- **Cost saving is modest at this batch size.** On the same pairs BART took 25 s and 299 s; the base model 19 s and 186 s, small and xsmall about 9 s and 117 s each. The 10–20× parameter reduction buys 2.6× wall time, because tokenization and padding dominate at batch 64. Peak VRAM: 1.8 GB, 1.5 GB, 0.9 GB against BART's 3.2 GB.

## Decision

BART-large-MNLI remains the entailment model of the V4 + NLI score. No spent set was scored with any candidate. If a cheaper stage is ever needed, the evidence says to look for a model whose low-tail ordering agrees with BART rather than one with higher MNLI accuracy, and to test it on Agora first.

## Provenance

- pvgpu1 job `nli_replication_dev`, commit `f6a84ed`, clean tree: three models in sequence on `english_dev` (13,920 pairs) and `agora` (226,680 pairs). GPU idle before and after. Manifests `results/step6_nli/nli__<model>.manifest.json`; the `.npz` probability files are gitignored like BART's.
- Downloads to the pvgpu1 Hugging Face cache, approved in advance: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (369 MB weights), `cross-encoder/nli-deberta-v3-small` (568 MB), `cross-encoder/nli-deberta-v3-xsmall` (283 MB). All three can be deleted.
- Local evaluation: `results/nli_replication/dev_results.{json,md}`, `passing.json` (empty), per-candidate score CSVs, `dev.manifest.json`.

```bash
# pvgpu1
scripts/remote_job.sh nli_replication_dev -- bash -c 'for m in MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli cross-encoder/nli-deberta-v3-small cross-encoder/nli-deberta-v3-xsmall; do python scripts/nli_stage.py --names english_dev,agora --model $m --batch-size 64 || exit 1; done'
# local
uv run --extra cpu python scripts/evaluate_nli_replication.py dev
```
