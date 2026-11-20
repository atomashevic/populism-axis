B6 reference (arm A, stored BART-large-MNLI): English-dev combined 0.537, claim check only 0.468; Agora combined 0.382, claim check only 0.342.

Paired differences: 4000 draws, seed 20260923, leaders (English-dev) and politicians (Agora) resampled. Non-inferior: Δ of the combined score against B6 at least -0.01 on both sets.

| Arm | Model | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev claim only | Agora claim only | Non-inferior |
|---|---|---:|---|---:|---|---:|---:|---|
| B | ModernBERT-large-nli | 0.500 | -0.037 [-0.103, +0.020] | 0.341 | -0.042 [-0.059, -0.026] | 0.329 | 0.261 | no |
| C | ModernBERT-large-zeroshot-v2.0 | 0.489 | -0.048 [-0.104, +0.013] | 0.278 | -0.104 [-0.139, -0.072] | 0.287 | 0.172 | no |
| D | von | 0.452 | -0.085 [-0.197, +0.022] | 0.235 | -0.147 [-0.180, -0.110] | 0.237 | 0.080 | no |
| E | laya | 0.560 | +0.024 [-0.045, +0.087] | 0.305 | -0.077 [-0.111, -0.045] | 0.469 | 0.211 | no |

Incremental information. Partial ρ: candidate claim score with the human score given the B6 combined score. Three-input: mean of z(axis), z(BART), z(candidate).

| Arm | Agora partial ρ [95% CI] | Agora three-input | Δ vs B6 [95% CI] | English-dev partial ρ [95% CI] | English-dev three-input | Δ vs B6 [95% CI] | Adds information (Agora) |
|---|---|---:|---|---|---:|---|---|
| B | +0.070 [+0.030, +0.105] | 0.371 | -0.011 [-0.022, -0.003] | -0.005 [-0.167, +0.151] | 0.513 | -0.024 [-0.081, +0.025] | no |
| C | +0.010 [-0.049, +0.060] | 0.331 | -0.052 [-0.075, -0.035] | +0.021 [-0.108, +0.143] | 0.501 | -0.036 [-0.079, +0.009] | no |
| D | +0.066 [+0.006, +0.096] | 0.321 | -0.061 [-0.086, -0.038] | +0.060 [-0.089, +0.216] | 0.497 | -0.040 [-0.120, +0.033] | no |
| E | +0.059 [+0.019, +0.104] | 0.349 | -0.033 [-0.053, -0.014] | +0.249 [+0.098, +0.390] | 0.566 | +0.029 [-0.026, +0.081] | no |

Contrasts between arms (left minus right).

| Contrast | Isolates | English-dev combined Δ [95% CI] | Agora combined Δ [95% CI] | Agora claim-only Δ [95% CI] | Agora window ρ (PE) |
|---|---|---|---|---|---:|
| B − A | backbone, NLI task held fixed | -0.037 [-0.103, +0.020] | -0.042 [-0.059, -0.026] | -0.081 [-0.121, -0.037] | 0.398 |
| C − B | broad zero-shot mix against NLI-only training | -0.011 [-0.073, +0.060] | -0.062 [-0.086, -0.045] | -0.089 [-0.145, -0.057] | 0.327 |
| D − B | Von decision head against the 3-class NLI head | -0.048 [-0.172, +0.076] | -0.106 [-0.143, -0.068] | -0.181 [-0.239, -0.129] | -0.044 |
| D − C | Von decision head against the binary entailment head | -0.037 [-0.156, +0.074] | -0.043 [-0.084, +0.015] | -0.092 [-0.151, -0.007] | 0.079 |
| E − B | Laya decision head against the 3-class NLI head | +0.060 [-0.008, +0.129] | -0.035 [-0.067, -0.000] | -0.050 [-0.102, +0.010] | 0.226 |
| E − C | Laya decision head against the binary entailment head | +0.071 [+0.000, +0.133] | +0.027 [-0.005, +0.073] | +0.039 [-0.014, +0.130] | 0.146 |

Agora windows, PE conjunction: tail, rank agreement with BART, ties in the stored values, and the 50 windows that differ most from BART in rank.

| Arm | median | p99 | share > .5 | window ρ vs BART | doc ρ vs BART | exact 0 / exact 1 / distinct | mean coder score: candidate higher / BART higher (n) |
|---|---:|---:|---:|---:|---:|---|---|
| B | 0.0164 | 0.076 | 0.0000 | 0.398 | 0.539 | 0 / 0 / 225260 | 0.08 / 0.13 (20 / 30) |
| C | 0.0110 | 0.093 | 0.0000 | 0.242 | 0.386 | 0 / 0 / 7976 | 0.06 / – (50 / 0) |
| D | 0.7365 | 0.820 | 0.9882 | -0.085 | -0.048 | 0 / 0 / 221403 | 0.10 / 0.18 (44 / 6) |
| E | 0.2967 | 0.859 | 0.1943 | 0.312 | 0.394 | 0 / 0 / 24189 | 0.04 / 0.10 (48 / 2) |

Cost on Agora (22,668 windows × 10 claims), from the GPU manifests.

| Arm | Sequences encoded | Forward calls | Wall time (s) | Peak VRAM (MB) | GPU |
|---|---:|---:|---:|---:|---|
| A | 226680 | 3542 | 727.0 | 3193.0 | NVIDIA A30 |
| B | 226680 | 3542 | 878.4 | 1387.2 | NVIDIA A30 |
| C | 226680 | 3542 | 875.7 | 1387.2 | NVIDIA A30 |
| D | 226680 | 226680 | 9126.5 | 1539.7 | NVIDIA A30 |
| E | 226680 | 3778 | 1273.8 | 4430.3 | NVIDIA A30 |
