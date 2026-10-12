> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, `results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. Axis and claim-check rows are unchanged.

### elite, majority labels (test n=1759, positives=278)

| System | AUROC [95% CI] | AP | F1 @ train-prevalence cut | F1 @ train-optimal cut | F1 @ test-oracle cut |
|---|---|---:|---:|---:|---:|
| E_axis | 0.781 [0.754, 0.808] | 0.379 | 0.393 | 0.430 | 0.450 |
| E_axis_plus_nli | 0.806 [0.781, 0.832] | 0.435 | 0.449 | 0.468 | 0.483 |
| E_nli | 0.820 [0.796, 0.844] | 0.442 | 0.445 | 0.488 | 0.501 |
| E_roberta | 0.669 [0.632, 0.704] | 0.332 | 0.357 | 0.360 | 0.372 |
| E_roberta (superseded softmax) | 0.650 [0.613, 0.686] | 0.325 | 0.342 | 0.328 | 0.353 |
| PE_axis | 0.731 [0.700, 0.761] | 0.299 | 0.331 | 0.385 | 0.405 |
| PE_axis_plus_nli | 0.723 [0.693, 0.753] | 0.316 | 0.338 | 0.372 | 0.387 |
| roberta_populist_total | 0.633 [0.596, 0.670] | 0.293 | 0.324 | 0.325 | 0.330 |
| roberta_populist_total (superseded softmax) | 0.650 [0.612, 0.686] | 0.296 | 0.334 | 0.333 | 0.337 |

### elite, any_coder labels (test n=1759, positives=648)

| System | AUROC [95% CI] | AP | F1 @ train-prevalence cut | F1 @ train-optimal cut | F1 @ test-oracle cut |
|---|---|---:|---:|---:|---:|
| E_axis | 0.789 [0.767, 0.811] | 0.653 | 0.629 | 0.655 | 0.665 |
| E_axis_plus_nli | 0.808 [0.787, 0.829] | 0.695 | 0.652 | 0.668 | 0.680 |
| E_nli | 0.820 [0.800, 0.841] | 0.713 | 0.666 | 0.697 | 0.701 |
| E_roberta | 0.637 [0.607, 0.665] | 0.548 | 0.502 | 0.533 | 0.547 |
| E_roberta (superseded softmax) | 0.620 [0.592, 0.649] | 0.538 | 0.493 | 0.515 | 0.540 |
| PE_axis | 0.738 [0.715, 0.762] | 0.579 | 0.575 | 0.623 | 0.633 |
| PE_axis_plus_nli | 0.717 [0.694, 0.742] | 0.578 | 0.544 | 0.608 | 0.612 |
| roberta_populist_total | 0.601 [0.574, 0.630] | 0.504 | 0.486 | 0.540 | 0.541 |
| roberta_populist_total (superseded softmax) | 0.613 [0.585, 0.641] | 0.505 | 0.510 | 0.538 | 0.538 |

### centr, majority labels (test n=1759, positives=60)

| System | AUROC [95% CI] | AP | F1 @ train-prevalence cut | F1 @ train-optimal cut | F1 @ test-oracle cut |
|---|---|---:|---:|---:|---:|
| P_axis | 0.754 [0.682, 0.816] | 0.152 | 0.207 | 0.177 | 0.255 |
| P_axis_plus_nli | 0.771 [0.706, 0.831] | 0.148 | 0.207 | 0.236 | 0.254 |
| P_nli | 0.726 [0.655, 0.792] | 0.114 | 0.190 | 0.173 | 0.196 |
| P_roberta | 0.690 [0.617, 0.757] | 0.077 | 0.103 | 0.137 | 0.158 |
| P_roberta (superseded softmax) | 0.643 [0.566, 0.713] | 0.062 | 0.103 | 0.111 | 0.143 |
| PE_axis | 0.655 [0.580, 0.726] | 0.068 | 0.121 | 0.105 | 0.135 |
| PE_axis_plus_nli | 0.651 [0.576, 0.722] | 0.067 | 0.103 | 0.106 | 0.126 |
| roberta_populist_total | 0.572 [0.493, 0.642] | 0.057 | 0.086 | 0.082 | 0.122 |
| roberta_populist_total (superseded softmax) | 0.575 [0.501, 0.643] | 0.056 | 0.086 | 0.072 | 0.119 |

### centr, any_coder labels (test n=1759, positives=322)

| System | AUROC [95% CI] | AP | F1 @ train-prevalence cut | F1 @ train-optimal cut | F1 @ test-oracle cut |
|---|---|---:|---:|---:|---:|
| P_axis | 0.705 [0.675, 0.736] | 0.361 | 0.382 | 0.388 | 0.405 |
| P_axis_plus_nli | 0.740 [0.710, 0.769] | 0.420 | 0.425 | 0.429 | 0.442 |
| P_nli | 0.721 [0.689, 0.753] | 0.401 | 0.413 | 0.440 | 0.444 |
| P_roberta | 0.639 [0.605, 0.673] | 0.307 | 0.360 | 0.352 | 0.370 |
| P_roberta (superseded softmax) | 0.613 [0.577, 0.648] | 0.293 | 0.298 | 0.342 | 0.345 |
| PE_axis | 0.602 [0.568, 0.636] | 0.260 | 0.286 | 0.323 | 0.340 |
| PE_axis_plus_nli | 0.628 [0.592, 0.661] | 0.275 | 0.301 | 0.343 | 0.370 |
| roberta_populist_total | 0.555 [0.520, 0.592] | 0.238 | 0.236 | 0.298 | 0.319 |
| roberta_populist_total (superseded softmax) | 0.551 [0.514, 0.587] | 0.234 | 0.235 | 0.303 | 0.316 |

### Paired AUROC differences on the test split, majority labels (sentence bootstrap)

| Comparison | Δ AUROC | 95% CI | p |
|---|---:|---|---:|
| elite: E_axis_plus_nli - E_axis | +0.025 | [+0.017, +0.034] | 0.000 |
| elite: E_nli - E_axis_plus_nli | +0.014 | [-0.015, +0.040] | 0.311 |
| elite: E_axis_plus_nli - E_roberta | +0.137 | [+0.095, +0.181] | 0.000 |
| elite: E_axis_plus_nli - E_roberta (superseded softmax) | +0.156 | [+0.114, +0.199] | 0.000 |
| elite: E_axis_plus_nli - PE_axis_plus_nli | +0.084 | [+0.061, +0.106] | 0.000 |
| centr: P_axis_plus_nli - P_axis | +0.018 | [-0.006, +0.045] | 0.149 |
| centr: P_nli - P_axis_plus_nli | -0.045 | [-0.118, +0.027] | 0.231 |
| centr: P_axis_plus_nli - P_roberta | +0.081 | [+0.005, +0.151] | 0.030 |
| centr: P_axis_plus_nli - P_roberta (superseded softmax) | +0.128 | [+0.052, +0.206] | 0.001 |
| centr: P_axis_plus_nli - PE_axis_plus_nli | +0.121 | [+0.071, +0.169] | 0.000 |
