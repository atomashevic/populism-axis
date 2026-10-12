### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B1c carried, V3 anchors | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |
| V4 minimal pairs (primary) | 0.444 | [+0.306, +0.575] | 0.917 | 0.388 | 0.347 | [+0.258, +0.362] | 0.862 |
| V3 + V4 pooled | 0.449 | [+0.316, +0.578] | 0.931 | 0.387 | 0.358 | [+0.265, +0.374] | 0.876 |

### Paired difference vs previous step

Candidate `/home/aleksandar/work/populism-axis/results/v4/step3_pair_anchors/pairs_v4` minus baseline `/home/aleksandar/work/populism-axis/results/v4/step1b_text_cleaning/B1c_cap60_clean`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | -0.012 | [-0.033, +0.013] | 0.371 | 5272 | 6 |
| english (leader-clustered) | -0.039 | [-0.123, +0.042] | 0.328 | 174 | 57 |

Rule checks (standard):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: fail
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: fail

Outcome: **KEEP PREVIOUS**

### Paired difference vs previous step

Candidate `/home/aleksandar/work/populism-axis/results/v4/step3_pair_anchors/pooled_v3_v4` minus baseline `/home/aleksandar/work/populism-axis/results/v4/step1b_text_cleaning/B1c_cap60_clean`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | -0.001 | [-0.015, +0.015] | 0.998 | 5272 | 6 |
| english (leader-clustered) | -0.033 | [-0.098, +0.030] | 0.281 | 174 | 57 |

Rule checks (standard):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: fail
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: pass

Outcome: **KEEP PREVIOUS**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B1c carried, V3 anchors | `be0cd6c` | False | `1ec489c0c163` | laptop-omarchy |
| V4 minimal pairs (primary) | `454f1bb` | False | `2cfa1ee970b9` | laptop-omarchy |
| V3 + V4 pooled | `454f1bb` | False | `ab00df8edf2b` | laptop-omarchy |
| step3_pair_anchors/step3.manifest.json | `454f1bb` | False | `252b8df9321a` | laptop-omarchy |
