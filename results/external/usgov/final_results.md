> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, `results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. Axis and claim-check rows are unchanged.

### Speeches: governors_2010_2018 (n=288, groups=73, populist=60)

| System | Spearman [95% CI] | Pearson r² | AUROC | AP | In-sample cutoff: acc / F1 / bal.acc |
|---|---|---:|---:|---:|---|
| B0_v3_config | 0.345 [0.229, 0.456] | 0.163 | 0.701 | 0.439 | 0.72 / 0.49 / 0.69 |
| B4_axis | 0.458 [0.365, 0.545] | 0.237 | 0.767 | 0.460 | 0.80 / 0.54 / 0.71 |
| B6_axis_plus_nli | 0.526 [0.438, 0.605] | 0.268 | 0.813 | 0.488 | 0.77 / 0.57 / 0.76 |
| NLI_only | 0.399 [0.282, 0.505] | 0.118 | 0.728 | 0.454 | 0.77 / 0.51 / 0.70 |
| RoBERTa_anti_elitism_top3 | 0.326 [0.197, 0.447] | 0.147 | 0.678 | 0.455 | 0.71 / 0.47 / 0.68 |
| RoBERTa_anti_elitism_top3 (superseded softmax) | 0.376 [0.252, 0.490] | 0.144 | 0.709 | 0.439 | 0.67 / 0.48 / 0.69 |
| RoBERTa_populist_total_max | 0.367 [0.254, 0.479] | 0.103 | 0.699 | 0.376 | 0.73 / 0.48 / 0.69 |
| RoBERTa_populist_total_max (superseded softmax) | 0.366 [0.252, 0.480] | 0.099 | 0.701 | 0.408 | 0.76 / 0.48 / 0.67 |
| SetFit_share_published | 0.514 [0.403, 0.614] | 0.326 | 0.839 | 0.578 | 0.85 / 0.64 / 0.77 |
| P_only_k8 | 0.415 [0.317, 0.508] | 0.150 | 0.727 | 0.443 | 0.64 / 0.46 / 0.67 |
| E_only_k8 | 0.430 [0.328, 0.525] | 0.178 | 0.746 | 0.426 | 0.78 / 0.47 / 0.66 |

| Paired Δ Spearman | Δ | 95% CI | p |
|---|---:|---|---:|
| B4_axis - B0_v3_config | +0.113 | [+0.039, +0.191] | 0.003 |
| B6_axis_plus_nli - B0_v3_config | +0.181 | [+0.084, +0.282] | 0.001 |
| B6_axis_plus_nli - B4_axis | +0.068 | [+0.031, +0.106] | 0.001 |
| B6_axis_plus_nli - SetFit_share_published | +0.006 | [-0.078, +0.093] | 0.890 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 | +0.200 | [+0.086, +0.318] | 0.000 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 (superseded softmax) | +0.150 | [+0.045, +0.255] | 0.004 |
| B4_axis - SetFit_share_published | -0.055 | [-0.141, +0.034] | 0.224 |

### Speeches: governors_2018_2022 (n=26, groups=6, populist=10)

| System | Spearman [95% CI] | Pearson r² | AUROC | AP | In-sample cutoff: acc / F1 / bal.acc |
|---|---|---:|---:|---:|---|
| B0_v3_config | 0.604 [0.344, 0.778] | 0.262 | 0.838 | 0.797 | 0.81 / 0.74 / 0.79 |
| B4_axis | 0.609 [0.197, 0.811] | 0.376 | 0.856 | 0.811 | 0.81 / 0.78 / 0.82 |
| B6_axis_plus_nli | 0.629 [0.342, 0.789] | 0.338 | 0.863 | 0.828 | 0.81 / 0.78 / 0.82 |
| NLI_only | 0.280 [-0.038, 0.622] | 0.021 | 0.581 | 0.546 | 0.54 / 0.60 / 0.61 |
| RoBERTa_anti_elitism_top3 | 0.732 [0.322, 0.898] | 0.352 | 0.837 | 0.827 | 0.77 / 0.75 / 0.79 |
| RoBERTa_anti_elitism_top3 (superseded softmax) | 0.650 [0.112, 0.893] | 0.376 | 0.787 | 0.791 | 0.81 / 0.74 / 0.79 |
| RoBERTa_populist_total_max | 0.640 [0.166, 0.790] | 0.245 | 0.816 | 0.812 | 0.81 / 0.76 / 0.81 |
| RoBERTa_populist_total_max (superseded softmax) | 0.653 [0.213, 0.803] | 0.224 | 0.822 | 0.820 | 0.81 / 0.76 / 0.81 |
| SetFit_share_published | 0.587 [0.214, 0.835] | 0.448 | 0.938 | 0.840 | 0.92 / 0.91 / 0.94 |
| P_only_k8 | 0.529 [0.113, 0.832] | 0.259 | 0.825 | 0.778 | 0.81 / 0.74 / 0.79 |
| E_only_k8 | 0.611 [0.232, 0.772] | 0.343 | 0.850 | 0.826 | 0.73 / 0.74 / 0.78 |

| Paired Δ Spearman | Δ | 95% CI | p |
|---|---:|---|---:|
| B4_axis - B0_v3_config | +0.005 | [-0.259, +0.138] | 0.923 |
| B6_axis_plus_nli - B0_v3_config | +0.024 | [-0.148, +0.138] | 0.709 |
| B6_axis_plus_nli - B4_axis | +0.020 | [-0.063, +0.164] | 0.578 |
| B6_axis_plus_nli - SetFit_share_published | +0.041 | [-0.119, +0.259] | 0.582 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 | -0.104 | [-0.329, +0.222] | 0.590 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 (superseded softmax) | -0.022 | [-0.308, +0.414] | 0.886 |
| B4_axis - SetFit_share_published | +0.021 | [-0.203, +0.254] | 0.873 |

### Speeches: presidential_2016 (n=45, groups=7, populist=20)

| System | Spearman [95% CI] | Pearson r² | AUROC | AP | In-sample cutoff: acc / F1 / bal.acc |
|---|---|---:|---:|---:|---|
| B0_v3_config | 0.397 [0.089, 0.593] | 0.150 | 0.686 | 0.609 | 0.71 / 0.68 / 0.71 |
| B4_axis | 0.568 [0.287, 0.657] | 0.265 | 0.746 | 0.699 | 0.73 / 0.74 / 0.74 |
| B6_axis_plus_nli | 0.686 [0.390, 0.746] | 0.366 | 0.846 | 0.839 | 0.82 / 0.81 / 0.82 |
| NLI_only | 0.611 [0.190, 0.720] | 0.262 | 0.848 | 0.880 | 0.84 / 0.81 / 0.83 |
| RoBERTa_anti_elitism_top3 | 0.696 [0.397, 0.803] | 0.334 | 0.814 | 0.767 | 0.80 / 0.79 / 0.80 |
| RoBERTa_anti_elitism_top3 (superseded softmax) | 0.706 [0.359, 0.778] | 0.355 | 0.848 | 0.797 | 0.84 / 0.84 / 0.85 |
| RoBERTa_populist_total_max | 0.557 [0.246, 0.790] | 0.080 | 0.719 | 0.572 | 0.76 / 0.76 / 0.77 |
| RoBERTa_populist_total_max (superseded softmax) | 0.614 [0.386, 0.788] | 0.070 | 0.750 | 0.618 | 0.73 / 0.75 / 0.75 |
| SetFit_share_published | 0.723 [0.238, 0.802] | 0.585 | 0.924 | 0.908 | 0.89 / 0.88 / 0.89 |
| P_only_k8 | 0.480 [0.245, 0.550] | 0.189 | 0.724 | 0.678 | 0.76 / 0.72 / 0.75 |
| E_only_k8 | 0.464 [0.094, 0.697] | 0.160 | 0.668 | 0.630 | 0.56 / 0.67 / 0.60 |

| Paired Δ Spearman | Δ | 95% CI | p |
|---|---:|---|---:|
| B4_axis - B0_v3_config | +0.171 | [-0.170, +0.473] | 0.441 |
| B6_axis_plus_nli - B0_v3_config | +0.289 | [+0.056, +0.510] | 0.013 |
| B6_axis_plus_nli - B4_axis | +0.117 | [-0.077, +0.301] | 0.188 |
| B6_axis_plus_nli - SetFit_share_published | -0.038 | [-0.189, +0.227] | 0.566 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 | -0.011 | [-0.229, +0.168] | 0.430 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 (superseded softmax) | -0.021 | [-0.179, +0.160] | 0.539 |
| B4_axis - SetFit_share_published | -0.155 | [-0.410, +0.241] | 0.294 |

### Speeches: all_pooled (n=359, groups=86, populist=90)

| System | Spearman [95% CI] | Pearson r² | AUROC | AP | In-sample cutoff: acc / F1 / bal.acc |
|---|---|---:|---:|---:|---|
| B0_v3_config | 0.448 [0.319, 0.567] | 0.216 | 0.740 | 0.504 | 0.68 / 0.54 / 0.71 |
| B4_axis | 0.530 [0.418, 0.626] | 0.286 | 0.785 | 0.546 | 0.77 / 0.60 / 0.74 |
| B6_axis_plus_nli | 0.581 [0.479, 0.669] | 0.288 | 0.818 | 0.597 | 0.74 / 0.61 / 0.76 |
| NLI_only | 0.440 [0.316, 0.547] | 0.130 | 0.739 | 0.569 | 0.76 / 0.56 / 0.71 |
| RoBERTa_anti_elitism_top3 | 0.445 [0.303, 0.577] | 0.235 | 0.732 | 0.562 | 0.74 / 0.54 / 0.70 |
| RoBERTa_anti_elitism_top3 (superseded softmax) | 0.476 [0.341, 0.596] | 0.239 | 0.751 | 0.575 | 0.71 / 0.54 / 0.70 |
| RoBERTa_populist_total_max | 0.463 [0.342, 0.572] | 0.136 | 0.740 | 0.477 | 0.72 / 0.56 / 0.72 |
| RoBERTa_populist_total_max (superseded softmax) | 0.467 [0.343, 0.578] | 0.130 | 0.745 | 0.499 | 0.75 / 0.56 / 0.71 |
| SetFit_share_published | 0.548 [0.423, 0.651] | 0.372 | 0.865 | 0.656 | 0.84 / 0.71 / 0.82 |
| P_only_k8 | 0.481 [0.364, 0.585] | 0.194 | 0.747 | 0.508 | 0.70 / 0.52 / 0.68 |
| E_only_k8 | 0.508 [0.397, 0.609] | 0.233 | 0.770 | 0.516 | 0.70 / 0.54 / 0.70 |

| Paired Δ Spearman | Δ | 95% CI | p |
|---|---:|---|---:|
| B4_axis - B0_v3_config | +0.082 | [+0.026, +0.144] | 0.004 |
| B6_axis_plus_nli - B0_v3_config | +0.133 | [+0.060, +0.213] | 0.001 |
| B6_axis_plus_nli - B4_axis | +0.051 | [+0.023, +0.083] | 0.000 |
| B6_axis_plus_nli - SetFit_share_published | +0.031 | [-0.045, +0.105] | 0.427 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 | +0.136 | [+0.043, +0.239] | 0.001 |
| B6_axis_plus_nli - RoBERTa_anti_elitism_top3 (superseded softmax) | +0.105 | [+0.023, +0.197] | 0.014 |
| B4_axis - SetFit_share_published | -0.015 | [-0.092, +0.061] | 0.695 |

### Sentences: governors_2010_2018 (n=3539, {'populist': 1411, 'neutral': 1120, 'pluralist': 1008})

| System | AUROC populist vs rest | AP | AUROC vs pluralist | AUROC vs neutral |
|---|---:|---:|---:|---:|
| B4_axis | 0.731 | 0.629 | 0.736 | 0.726 |
| B6_axis_plus_nli | 0.744 | 0.650 | 0.759 | 0.731 |
| NLI_only | 0.696 | 0.620 | 0.726 | 0.669 |
| RoBERTa_populist_total_max | 0.625 | 0.595 | 0.619 | 0.630 |
| RoBERTa_populist_total_max (superseded softmax) | 0.630 | 0.591 | 0.626 | 0.633 |
| P_only | 0.697 | 0.584 | 0.698 | 0.696 |
| E_only | 0.718 | 0.638 | 0.728 | 0.709 |

### Sentences: presidential_2016 (n=81, {'populist': 43, 'pluralist': 37, 'neutral': 1})

| System | AUROC populist vs rest | AP | AUROC vs pluralist | AUROC vs neutral |
|---|---:|---:|---:|---:|
| B4_axis | 0.921 | 0.938 | 0.928 | 0.651 |
| B6_axis_plus_nli | 0.936 | 0.950 | 0.941 | 0.744 |
| NLI_only | 0.823 | 0.858 | 0.819 | 0.977 |
| RoBERTa_populist_total_max | 0.860 | 0.826 | 0.869 | 0.512 |
| RoBERTa_populist_total_max (superseded softmax) | 0.863 | 0.831 | 0.873 | 0.512 |
| P_only | 0.813 | 0.832 | 0.819 | 0.605 |
| E_only | 0.920 | 0.934 | 0.934 | 0.419 |

### Sentences: all_pooled (n=3620, {'populist': 1454, 'neutral': 1121, 'pluralist': 1045})

| System | AUROC populist vs rest | AP | AUROC vs pluralist | AUROC vs neutral |
|---|---:|---:|---:|---:|
| B4_axis | 0.732 | 0.640 | 0.731 | 0.733 |
| B6_axis_plus_nli | 0.744 | 0.659 | 0.751 | 0.738 |
| NLI_only | 0.698 | 0.625 | 0.720 | 0.678 |
| RoBERTa_populist_total_max | 0.630 | 0.602 | 0.621 | 0.639 |
| RoBERTa_populist_total_max (superseded softmax) | 0.635 | 0.598 | 0.628 | 0.642 |
| P_only | 0.698 | 0.594 | 0.692 | 0.703 |
| E_only | 0.721 | 0.650 | 0.727 | 0.716 |
