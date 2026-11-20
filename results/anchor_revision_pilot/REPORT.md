# Anchor-revision pilot: development sets

English-dev (174 speeches, leaders resampled) and Agora (5,272 paragraphs, politicians resampled). Paired differences: 4000 draws, seed 20260923. No held-out or external set is scored.

## Checks

- english_dev: V4 axis Spearman 0.5166 (reported 0.517); top windows equal the step-6 requests: True; V4 claim-check rerun combined Spearman 0.5368 (committed 0.5367, max score difference 5.28e-03).
- agora: V4 axis Spearman 0.3588 (reported 0.359); top windows equal the step-6 requests: True; V4 claim-check rerun combined Spearman 0.3823 (committed 0.3823, max score difference 1.75e-02).

## Label-free geometry

| Arm | cos(P, V4) | cos(E, V4) | cos(P, E) | VADER neg. ρ, PE windows (English-dev / Agora) | Claim hypotheses added / dropped |
|---|---:|---:|---:|---|---|
| `v4` | 1.0000 | 1.0000 | 0.5029 | +0.262 / +0.269 | P: +– −–; E: +– −– |
| `v5_green` | 0.9822 | 0.9817 | 0.5267 | +0.279 / +0.288 | P: +– −–; E: +– −– |
| `v5_red` | 0.9739 | 0.9778 | 0.4775 | +0.224 / +0.240 | P: +– −–; E: +– −– |
| `v5_green_red` | 0.9599 | 0.9600 | 0.5101 | +0.244 / +0.262 | P: +– −–; E: +– −– |

Revised sentences on the V4 axis of their own channel, built without the slot (V4 sentence → revised sentence). Contrast slots (−) belong on the negative side.

| Arm | Slot | Own axis | Other channel's axis |
|---|---|---|---|
| `v5_green` | P+02 | +0.150 → +0.165 | +0.061 → +0.061 |
| `v5_green` | P+05 | +0.200 → +0.203 | +0.095 → +0.064 |
| `v5_green` | P+06 | +0.199 → +0.180 | +0.123 → +0.101 |
| `v5_green` | P+07 | +0.055 → +0.151 | -0.024 → +0.064 |
| `v5_green` | P+09 | +0.066 → +0.129 | +0.023 → +0.126 |
| `v5_green` | E+04 | +0.200 → +0.245 | +0.157 → +0.175 |
| `v5_green` | E+06 | +0.184 → +0.223 | +0.096 → +0.121 |
| `v5_green` | E+09 | +0.151 → +0.209 | +0.095 → +0.149 |
| `v5_green` | E-00 | -0.171 → -0.025 | -0.090 → -0.009 |
| `v5_red` | P-04 | -0.077 → -0.035 | -0.068 → -0.064 |
| `v5_red` | P-07 | -0.042 → +0.060 | +0.039 → +0.123 |
| `v5_red` | P-08 | -0.169 → -0.050 | -0.172 → -0.105 |
| `v5_red` | E-04 | -0.180 → +0.049 | -0.183 → -0.021 |
| `v5_red` | E-07 | -0.066 → -0.068 | -0.053 → -0.035 |
| `v5_red` | E-08 | -0.002 → +0.012 | -0.034 → +0.002 |
| `v5_green_red` | P+02 | +0.150 → +0.165 | +0.061 → +0.061 |
| `v5_green_red` | P+05 | +0.200 → +0.203 | +0.095 → +0.064 |
| `v5_green_red` | P+06 | +0.199 → +0.180 | +0.123 → +0.101 |
| `v5_green_red` | P+07 | +0.055 → +0.151 | -0.024 → +0.064 |
| `v5_green_red` | P+09 | +0.066 → +0.129 | +0.023 → +0.126 |
| `v5_green_red` | P-04 | -0.077 → -0.035 | -0.068 → -0.064 |
| `v5_green_red` | P-07 | -0.042 → +0.060 | +0.039 → +0.123 |
| `v5_green_red` | P-08 | -0.169 → -0.050 | -0.172 → -0.105 |
| `v5_green_red` | E+04 | +0.200 → +0.245 | +0.157 → +0.175 |
| `v5_green_red` | E+06 | +0.184 → +0.223 | +0.096 → +0.121 |
| `v5_green_red` | E+09 | +0.151 → +0.209 | +0.095 → +0.149 |
| `v5_green_red` | E-00 | -0.171 → -0.025 | -0.090 → -0.009 |
| `v5_green_red` | E-04 | -0.180 → +0.049 | -0.183 → -0.021 |
| `v5_green_red` | E-07 | -0.066 → -0.068 | -0.053 → -0.035 |
| `v5_green_red` | E-08 | -0.002 → +0.012 | -0.034 → +0.002 |

## Spearman with the human score

| Arm | Score | English-dev [95% CI] | Agora [95% CI] |
|---|---|---|---|
| `v4` | Axis | 0.517 [0.375, 0.645] | 0.359 [0.245, 0.380] |
| `v4` | Claim check, arm's hypotheses | 0.468 [0.352, 0.576] | 0.342 [0.272, 0.359] |
| `v4` | Combined, arm's hypotheses | 0.537 [0.421, 0.642] | 0.382 [0.295, 0.397] |
| `v5_green` | Axis | 0.525 [0.388, 0.653] | 0.354 [0.239, 0.379] |
| `v5_green` | Claim check, arm's hypotheses | 0.430 [0.283, 0.557] | 0.338 [0.259, 0.351] |
| `v5_green` | Combined, arm's hypotheses | 0.543 [0.424, 0.650] | 0.374 [0.277, 0.389] |
| `v5_green` | Claim check, V4 hypotheses | 0.469 [0.362, 0.573] | 0.343 [0.276, 0.361] |
| `v5_green` | Combined, V4 hypotheses | 0.544 [0.432, 0.644] | 0.380 [0.292, 0.396] |
| `v5_red` | Axis | 0.520 [0.381, 0.642] | 0.365 [0.249, 0.386] |
| `v5_red` | Claim check, arm's hypotheses | 0.460 [0.341, 0.573] | 0.342 [0.271, 0.359] |
| `v5_red` | Combined, arm's hypotheses | 0.538 [0.419, 0.638] | 0.389 [0.298, 0.402] |
| `v5_red` | Claim check, V4 hypotheses | 0.460 [0.341, 0.573] | 0.342 [0.271, 0.359] |
| `v5_red` | Combined, V4 hypotheses | 0.538 [0.419, 0.638] | 0.389 [0.298, 0.402] |
| `v5_green_red` | Axis | 0.533 [0.401, 0.649] | 0.361 [0.243, 0.383] |
| `v5_green_red` | Claim check, arm's hypotheses | 0.393 [0.247, 0.526] | 0.338 [0.260, 0.350] |
| `v5_green_red` | Combined, arm's hypotheses | 0.530 [0.407, 0.636] | 0.382 [0.281, 0.397] |
| `v5_green_red` | Claim check, V4 hypotheses | 0.460 [0.346, 0.565] | 0.343 [0.276, 0.360] |
| `v5_green_red` | Combined, V4 hypotheses | 0.542 [0.435, 0.640] | 0.385 [0.294, 0.400] |

## Paired difference against the committed V4 scores

| Arm | Score | English-dev Δ [95% CI] | Agora Δ [95% CI] |
|---|---|---|---|
| `v5_green` | Axis | +0.009 [-0.002, +0.022] | -0.004 [-0.012, +0.002] |
| `v5_green` | Claim check, arm's hypotheses | -0.037 [-0.128, +0.041] | -0.004 [-0.028, +0.019] |
| `v5_green` | Combined, arm's hypotheses | +0.006 [-0.037, +0.047] | -0.008 [-0.018, -0.002] |
| `v5_green` | Claim check, V4 hypotheses | +0.002 [-0.024, +0.024] | +0.001 [+0.000, +0.004] |
| `v5_green` | Combined, V4 hypotheses | +0.008 [-0.006, +0.025] | -0.003 [-0.008, +0.002] |
| `v5_red` | Axis | +0.003 [-0.021, +0.028] | +0.006 [-0.001, +0.016] |
| `v5_red` | Claim check, arm's hypotheses | -0.008 [-0.042, +0.027] | -0.000 [-0.000, +0.001] |
| `v5_red` | Combined, arm's hypotheses | +0.001 [-0.024, +0.027] | +0.006 [+0.001, +0.012] |
| `v5_red` | Claim check, V4 hypotheses | -0.008 [-0.042, +0.027] | -0.000 [-0.000, +0.001] |
| `v5_red` | Combined, V4 hypotheses | +0.001 [-0.024, +0.027] | +0.006 [+0.001, +0.012] |
| `v5_green_red` | Axis | +0.017 [-0.011, +0.045] | +0.002 [-0.004, +0.006] |
| `v5_green_red` | Claim check, arm's hypotheses | -0.075 [-0.171, +0.009] | -0.004 [-0.028, +0.018] |
| `v5_green_red` | Combined, arm's hypotheses | -0.007 [-0.051, +0.035] | -0.000 [-0.013, +0.011] |
| `v5_green_red` | Claim check, V4 hypotheses | -0.008 [-0.043, +0.031] | +0.001 [-0.000, +0.004] |
| `v5_green_red` | Combined, V4 hypotheses | +0.005 [-0.024, +0.037] | +0.003 [+0.000, +0.005] |

## Adoption rule (combined score, arm's hypotheses, against B6)

English-dev Δ at least +0.01, Agora Δ at least -0.01, neither interval entirely below 0.

| Arm | English-dev Δ | Agora Δ | Mean Δ | Passes |
|---|---:|---:|---:|---|
| `v5_green` | +0.006 | -0.008 | -0.001 | no |
| `v5_red` | +0.001 | +0.006 | +0.004 | no |
| `v5_green_red` | -0.007 | -0.000 | -0.004 | no |

## Contrasts between candidate arms (left minus right)

| Contrast | Score | English-dev Δ [95% CI] | Agora Δ [95% CI] |
|---|---|---|---|
| v5_red − v5_green | Axis | -0.006 [-0.033, +0.019] | +0.010 [-0.002, +0.027] |
| v5_green_red − v5_green | Axis | +0.008 [-0.016, +0.031] | +0.006 [-0.001, +0.016] |
| v5_green_red − v5_red | Axis | +0.014 [-0.001, +0.030] | -0.004 [-0.012, +0.002] |
| v5_red − v5_green | Combined, arm's hypotheses | -0.005 [-0.057, +0.053] | +0.015 [+0.008, +0.023] |
| v5_green_red − v5_green | Combined, arm's hypotheses | -0.013 [-0.041, +0.016] | +0.008 [+0.003, +0.014] |
| v5_green_red − v5_red | Combined, arm's hypotheses | -0.008 [-0.057, +0.037] | -0.007 [-0.017, -0.000] |

## Claim-check diagnostics (arm's windows and hypotheses)

| Arm | Set | windows | PE median | PE p99 | share P > .5 | share E > .5 | window ρ axis vs claim check |
|---|---|---:|---:|---:|---:|---:|---:|
| `v4` | english_dev | 1392 | 0.0043 | 0.061 | 0.0029 | 0.0007 | 0.386 |
| `v4` | agora | 22668 | 0.0055 | 0.063 | 0.0015 | 0.0019 | 0.417 |
| `v5_green` | english_dev | 1392 | 0.0027 | 0.032 | 0.0022 | 0.0029 | 0.325 |
| `v5_green` | agora | 22668 | 0.0031 | 0.041 | 0.0012 | 0.0048 | 0.441 |
| `v5_red` | english_dev | 1392 | 0.0043 | 0.061 | 0.0036 | 0.0007 | 0.400 |
| `v5_red` | agora | 22668 | 0.0055 | 0.063 | 0.0015 | 0.0019 | 0.412 |
| `v5_green_red` | english_dev | 1392 | 0.0027 | 0.032 | 0.0036 | 0.0036 | 0.326 |
| `v5_green_red` | agora | 22668 | 0.0031 | 0.041 | 0.0012 | 0.0048 | 0.426 |

## Spot-check speeches (English-dev, percentile of the combined score; 1.0 is the top speech)

| Speech | GPD score | `v4` | `v5_green` | `v5_red` | `v5_green_red` |
|---|---:|---:|---:|---:|---:|
| Philippines_Arroyo_Famous_2 | 0.05 | 1.000 | 1.000 | 1.000 | 1.000 |
| South Africa_Mbeki_Campaign_1 | 0.03 | 0.408 | 0.270 | 0.466 | 0.374 |
| South Africa_Mbeki_Campaign_2 | 0.00 | 0.845 | 0.937 | 0.828 | 0.914 |
| South Africa_Mbeki_Famous_1 | 0.00 | 0.575 | 0.575 | 0.609 | 0.626 |
| South Africa_Mbeki_Famous_2 | 0.00 | 0.276 | 0.483 | 0.276 | 0.494 |
| South Africa_Mbeki_International_1 | 0.13 | 0.638 | 0.649 | 0.626 | 0.684 |
| South Africa_Mbeki_International_2 | 0.50 | 0.828 | 0.816 | 0.839 | 0.845 |
| South Africa_Mbeki_Ribbon_1 | 0.00 | 0.201 | 0.230 | 0.195 | 0.184 |
| South Africa_Mbeki_Ribbon_2 | 0.10 | 0.603 | 0.667 | 0.592 | 0.764 |
| UK_Brown_Campaign_1 | 0.00 | 0.994 | 0.971 | 0.994 | 0.983 |
