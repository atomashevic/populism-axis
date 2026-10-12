### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B1c spw1 k3 (carried) | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |
| spw1 k5 | 0.512 | [+0.369, +0.642] | 0.944 | 0.354 | 0.360 | [+0.270, +0.380] | 0.882 |
| spw1 k8 | 0.516 | [+0.376, +0.643] | 0.939 | 0.327 | 0.355 | [+0.267, +0.375] | 0.860 |
| spw3 k8 (selected) | 0.517 | [+0.375, +0.645] | 0.921 | 0.308 | 0.359 | [+0.245, +0.380] | 0.874 |
| spw3 frac .10 | 0.524 | [+0.408, +0.635] | 0.913 | 0.354 | 0.359 | [+0.244, +0.381] | 0.883 |

All systems for spw3 frac .10 (diagnostic; only the carried system decides):

| System | English ρ | Agora ρ |
|---|---:|---:|
| `EM__frozen__soft` | 0.464 | 0.311 |
| `E__frozen__soft` | 0.497 | 0.322 |
| `M__frozen__soft` | 0.412 | 0.279 |
| `PEM__frozen__soft` | 0.511 | 0.346 |
| `PE__frozen__hard` | 0.520 | 0.327 |
| `PE__frozen__soft` | 0.524 | 0.359 |
| `PM__frozen__soft` | 0.508 | 0.348 |
| `P__frozen__soft` | 0.474 | 0.333 |

### Paired difference vs previous step

Candidate `/home/aleksandar/work/populism-axis/results/v4/step4_windows_topk/grid/spw3_k8` minus baseline `/home/aleksandar/work/populism-axis/results/v4/step1b_text_cleaning/B1c_cap60_clean`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | -0.000 | [-0.023, +0.013] | 0.894 | 5272 | 6 |
| english (leader-clustered) | +0.034 | [-0.029, +0.098] | 0.296 | 174 | 57 |

Rule checks (standard):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: pass
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: pass

Outcome: **ADOPT**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B1c spw1 k3 (carried) | `be0cd6c` | False | `1ec489c0c163` | laptop-omarchy |
| spw1 k5 | `2be6215` | False | `df4d350fa079` | laptop-omarchy |
| spw1 k8 | `2be6215` | False | `45d1d4cea500` | laptop-omarchy |
| spw3 k8 (selected) | `2be6215` | False | `76544d5f4493` | laptop-omarchy |
| spw3 frac .10 | `2be6215` | False | `8c9e44ba0dcd` | laptop-omarchy |
| step4_windows_topk/grid.manifest.json | `2be6215` | False | `f7b0109180a8` | laptop-omarchy |
