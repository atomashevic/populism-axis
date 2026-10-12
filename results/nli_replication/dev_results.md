B6 reference (BART-large-MNLI): English-dev combined 0.537, NLI only 0.468; Agora combined 0.382, NLI only 0.342.

| Candidate | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev NLI only | Agora NLI only | Non-inferior |
|---|---:|---|---:|---|---:|---:|---|
| DeBERTa-v3-base-mnli-fever-anli | 0.497 | -0.040 [-0.108, +0.025] | 0.360 | -0.022 [-0.039, -0.008] | 0.288 | 0.293 | no |
| nli-deberta-v3-small | 0.499 | -0.038 [-0.105, +0.024] | 0.341 | -0.041 [-0.075, -0.020] | 0.347 | 0.241 | no |
| nli-deberta-v3-xsmall | 0.562 | +0.025 [-0.038, +0.089] | 0.341 | -0.041 [-0.058, -0.024] | 0.483 | 0.275 | no |

Low-tail calibration on English-dev windows (P channel / E channel): median entailment, 99th percentile, share above .5, window-level Spearman with BART.

| Candidate | P median | P p99 | P > .5 | P ρ vs BART | E median | E p99 | E > .5 | E ρ vs BART | PE window ρ vs BART | doc NLI ρ vs BART (Eng / Agora) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DeBERTa-v3-base-mnli-fever-anli | 0.0216 | 0.756 | 0.0302 | 0.492 | 0.0098 | 0.535 | 0.0115 | 0.403 | 0.444 | 0.552 / 0.630 |
| nli-deberta-v3-small | 0.0017 | 0.117 | 0.0043 | 0.426 | 0.0008 | 0.105 | 0.0029 | 0.237 | 0.382 | 0.571 / 0.456 |
| nli-deberta-v3-xsmall | 0.0054 | 0.214 | 0.0029 | 0.403 | 0.0028 | 0.075 | 0.0000 | 0.384 | 0.447 | 0.647 / 0.566 |
