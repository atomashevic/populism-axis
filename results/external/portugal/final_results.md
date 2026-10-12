> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, `results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. Axis and claim-check rows are unchanged.

### centr (n=6214, positives=302, speakers=144)

| System | AUROC [95% CI] | AP | F1 @ prevalence cut | F1 @ oracle cut |
|---|---|---:|---:|---:|
| P_axis_plus_nli | 0.695 [0.641, 0.735] | 0.136 | 0.222 | 0.222 |
| P_axis | 0.668 [0.611, 0.708] | 0.130 | 0.175 | 0.194 |
| P_nli | 0.689 [0.652, 0.721] | 0.126 | 0.162 | 0.190 |
| P_axis_me5_plain | 0.758 [0.711, 0.790] | 0.150 | 0.215 | 0.221 |
| P_axis_me5_instr | 0.729 [0.685, 0.762] | 0.125 | 0.162 | 0.190 |
| P_axis_me5_plain_plus_nli | 0.773 [0.731, 0.804] | 0.161 | 0.202 | 0.229 |
| PE_axis_plus_nli | 0.701 [0.650, 0.729] | 0.132 | 0.189 | 0.194 |
| PE_axis | 0.636 [0.571, 0.676] | 0.098 | 0.132 | 0.144 |
| P_roberta | 0.582 [0.523, 0.635] | 0.094 | 0.129 | 0.155 |
| P_roberta (superseded softmax) | 0.604 [0.560, 0.642] | 0.092 | 0.132 | 0.148 |
| PE_roberta_total | 0.599 [0.562, 0.628] | 0.088 | 0.129 | 0.143 |
| PE_roberta_total (superseded softmax) | 0.601 [0.562, 0.630] | 0.085 | 0.126 | 0.144 |
| supervised_lr_centr | 0.798 [0.773, 0.825] | 0.239 | 0.325 | 0.334 |

| Paired Δ AUROC (speaker bootstrap) | Δ | 95% CI | p |
|---|---:|---|---:|
| P_axis_plus_nli - P_axis | +0.028 | [+0.018, +0.043] | 0.000 |
| P_axis_plus_nli - P_nli | +0.007 | [-0.035, +0.037] | 0.774 |
| P_axis_plus_nli - P_axis_me5_plain | -0.063 | [-0.094, -0.030] | 0.000 |
| P_axis_plus_nli - P_axis_me5_plain_plus_nli | -0.078 | [-0.106, -0.054] | 0.000 |
| P_axis_plus_nli - PE_axis_plus_nli | -0.006 | [-0.048, +0.041] | 0.810 |
| P_axis_plus_nli - P_roberta | +0.113 | [+0.064, +0.143] | 0.001 |
| P_axis_plus_nli - P_roberta (superseded softmax) | +0.092 | [+0.045, +0.126] | 0.001 |
| P_axis_plus_nli - supervised_lr_centr | -0.103 | [-0.174, -0.051] | 0.000 |

### elite (n=6214, positives=116, speakers=144)

| System | AUROC [95% CI] | AP | F1 @ prevalence cut | F1 @ oracle cut |
|---|---|---:|---:|---:|
| E_axis_plus_nli | 0.798 [0.745, 0.867] | 0.130 | 0.207 | 0.243 |
| E_axis | 0.770 [0.718, 0.837] | 0.081 | 0.103 | 0.136 |
| E_nli | 0.767 [0.718, 0.826] | 0.118 | 0.181 | 0.207 |
| E_axis_me5_plain | 0.838 [0.797, 0.884] | 0.125 | 0.216 | 0.234 |
| E_axis_me5_instr | 0.807 [0.761, 0.862] | 0.118 | 0.190 | 0.202 |
| E_axis_me5_plain_plus_nli | 0.845 [0.802, 0.894] | 0.153 | 0.241 | 0.252 |
| PE_axis_plus_nli | 0.763 [0.692, 0.841] | 0.087 | 0.172 | 0.189 |
| PE_axis | 0.763 [0.707, 0.831] | 0.069 | 0.129 | 0.143 |
| E_roberta | 0.867 [0.818, 0.918] | 0.173 | 0.250 | 0.280 |
| E_roberta (superseded softmax) | 0.868 [0.829, 0.912] | 0.172 | 0.250 | 0.271 |
| PE_roberta_total | 0.841 [0.801, 0.888] | 0.142 | 0.224 | 0.240 |
| PE_roberta_total (superseded softmax) | 0.849 [0.814, 0.891] | 0.144 | 0.224 | 0.240 |
| supervised_lr_elite | 0.823 [0.780, 0.876] | 0.125 | 0.216 | 0.234 |

| Paired Δ AUROC (speaker bootstrap) | Δ | 95% CI | p |
|---|---:|---|---:|
| E_axis_plus_nli - E_axis | +0.029 | [+0.021, +0.036] | 0.000 |
| E_axis_plus_nli - E_nli | +0.031 | [-0.007, +0.079] | 0.123 |
| E_axis_plus_nli - E_axis_me5_plain | -0.039 | [-0.065, -0.006] | 0.024 |
| E_axis_plus_nli - E_axis_me5_plain_plus_nli | -0.046 | [-0.067, -0.020] | 0.002 |
| E_axis_plus_nli - PE_axis_plus_nli | +0.035 | [-0.007, +0.079] | 0.105 |
| E_axis_plus_nli - E_roberta | -0.069 | [-0.116, -0.016] | 0.016 |
| E_axis_plus_nli - E_roberta (superseded softmax) | -0.069 | [-0.110, -0.012] | 0.020 |
| E_axis_plus_nli - supervised_lr_elite | -0.024 | [-0.063, +0.023] | 0.293 |

### Discriminant AUROC: every channel against every dimension

| Dimension (positives) | P axis | E axis | PE axis | P axis+NLI | E axis+NLI | P NLI | E NLI |
|---|---:|---:|---:|---:|---:|---:|---:|
| centr (302) | 0.668 | 0.585 | 0.636 | 0.695 | 0.605 | 0.689 | 0.608 |
| elite (116) | 0.687 | 0.770 | 0.763 | 0.672 | 0.798 | 0.507 | 0.767 |
| sovereignty (23) | 0.822 | 0.551 | 0.678 | 0.832 | 0.550 | 0.744 | 0.535 |
| exclusion (50) | 0.819 | 0.825 | 0.856 | 0.791 | 0.849 | 0.546 | 0.765 |
