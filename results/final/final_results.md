> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, `results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. Axis and claim-check rows are unchanged.

| Configuration | English holdout ρ [95% CI] | Extreme AUROC | Leader-term ρ | EU AUROC 3-vs-0 | EU within-speaker AUROC | EU macro-F1 (LOSO) |
|---|---|---:|---:|---:|---:|---:|
| B0_v3_config | 0.376 [0.121, 0.598] | 0.856 | 0.289 | 0.890 | 0.861 | 0.528 |
| B4_axis | 0.417 [0.162, 0.618] | 0.936 | 0.251 | 0.890 | 0.860 | 0.533 |
| B6_axis_plus_nli | 0.518 [0.283, 0.700] | 0.984 | 0.342 | 0.942 | 0.927 | 0.545 |
| RoBERTa_comparator | 0.440 [0.142, 0.678] | 0.883 | 0.060 | 0.900 | 0.854 | 0.555 |
| RoBERTa_comparator (superseded softmax) | 0.464 [0.238, 0.670] | 0.856 | 0.147 | 0.902 | 0.862 | 0.512 |
| NLI_only | 0.520 [0.309, 0.699] | 1.000 | 0.458 | 0.945 | 0.929 | – |

| English holdout paired Δρ | Δ | 95% CI | p |
|---|---:|---|---:|
| B4_axis - B0_v3_config | +0.041 | [-0.044, +0.129] | 0.340 |
| B6_axis_plus_nli - B0_v3_config | +0.142 | [+0.004, +0.332] | 0.043 |
| B6_axis_plus_nli - B4_axis | +0.101 | [+0.009, +0.241] | 0.021 |
| B6_axis_plus_nli - RoBERTa_comparator | +0.079 | [-0.080, +0.251] | 0.345 |
| B6_axis_plus_nli - RoBERTa_comparator (superseded softmax) | +0.055 | [-0.094, +0.205] | 0.485 |
| B4_axis - RoBERTa_comparator | -0.023 | [-0.175, +0.130] | 0.737 |
| B4_axis - RoBERTa_comparator (superseded softmax) | -0.047 | [-0.203, +0.081] | 0.492 |
