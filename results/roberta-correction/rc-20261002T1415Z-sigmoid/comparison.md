| Row | Axis + claim check | RoBERTa softmax (old) | RoBERTa sigmoid (corrected) | Corrected − old | Ours better: old / corrected |
|---|---:|---:|---:|---:|---|
| GPD English holdout, Spearman | 0.518 | 0.464 [0.238, 0.670] | 0.440 [0.142, 0.678] | -0.024 | yes / yes |
| EU-OOD AUROC 3-vs-0 | 0.942 | 0.902 [0.625, 0.978] | 0.900 [0.729, 0.978] | -0.002 | yes / yes |
| EU-OOD LOSO macro-F1 | 0.545 | 0.512 | 0.555 | +0.044 | yes / no |
| Agora paragraphs, Spearman | 0.382 | 0.367 [0.279, 0.387] | 0.372 [0.287, 0.390] | +0.004 | yes / yes |
| US governors 2010-18, Spearman | 0.526 | 0.376 [0.252, 0.490] | 0.326 [0.197, 0.447] | -0.050 | yes / yes |
| Anti-elite tweets AUROC (E) | 0.756 | 0.573 [0.536, 0.603] | 0.571 [0.530, 0.604] | -0.002 | yes / yes |
| PopBERT anti-elitism AUROC (E) | 0.806 | 0.650 [0.613, 0.686] | 0.669 [0.632, 0.704] | +0.019 | yes / yes |
| PopBERT people-centrism AUROC (P) | 0.771 | 0.643 [0.566, 0.713] | 0.690 [0.617, 0.757] | +0.047 | yes / yes |
| Portugal anti-elitism AUROC (E) | 0.798 | 0.868 [0.829, 0.912] | 0.867 [0.818, 0.918] | -0.000 | no / no |
| Portugal people-centrism AUROC (P) | 0.695 | 0.604 [0.560, 0.642] | 0.582 [0.523, 0.635] | -0.021 | yes / yes |

Rows favouring axis + claim check: softmax 9/10, sigmoid 8/10.

| Paired comparison (ours − RoBERTa) | Old Δ [95% CI] | Corrected Δ [95% CI] |
|---|---|---|
| GPD English holdout, Spearman: B6_axis_plus_nli - RoBERTa | +0.055 [-0.094, +0.205] | +0.079 [-0.080, +0.251] |
| GPD English holdout, Spearman: B4_axis - RoBERTa | -0.047 [-0.203, +0.081] | -0.023 [-0.175, +0.130] |
| US governors 2010-18, Spearman: B6_axis_plus_nli - RoBERTa | +0.150 [+0.045, +0.255] | +0.200 [+0.086, +0.318] |
| Anti-elite tweets AUROC (E): E_axis_plus_nli - RoBERTa_anti_elitism | +0.183 [+0.128, +0.247] | +0.185 [+0.122, +0.257] |
| Anti-elite tweets AUROC (E): E_axis - RoBERTa_anti_elitism | +0.181 [+0.127, +0.246] | +0.183 [+0.122, +0.253] |
| PopBERT anti-elitism AUROC (E): E_axis_plus_nli - E_roberta | +0.156 [+0.114, +0.199] | +0.137 [+0.095, +0.181] |
| PopBERT people-centrism AUROC (P): P_axis_plus_nli - P_roberta | +0.128 [+0.052, +0.206] | +0.081 [+0.005, +0.151] |
| Portugal anti-elitism AUROC (E): E_axis_plus_nli - E_roberta | -0.069 [-0.110, -0.012] | -0.069 [-0.116, -0.016] |
| Portugal people-centrism AUROC (P): P_axis_plus_nli - P_roberta | +0.092 [+0.045, +0.126] | +0.113 [+0.064, +0.143] |
| Agora (supplementary, not previously reported), softmax: combined − RoBERTa | +0.015 [-0.034, +0.076] | |
| Agora (supplementary, not previously reported), sigmoid: combined − RoBERTa | +0.011 [-0.036, +0.067] | |
