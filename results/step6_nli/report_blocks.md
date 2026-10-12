### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B4 axis only (carried) | 0.517 | [+0.375, +0.645] | 0.921 | 0.308 | 0.359 | [+0.245, +0.380] | 0.874 |
| NLI only, top 8 | 0.468 | [+0.352, +0.576] | 0.939 | 0.398 | 0.342 | [+0.272, +0.359] | 0.752 |
| NLI only, top 3 | 0.333 | [+0.183, +0.478] | 0.724 | 0.258 | 0.332 | [+0.253, +0.346] | 0.753 |
| Axis + NLI combined, top 8 | 0.537 | [+0.421, +0.642] | 0.951 | 0.409 | 0.382 | [+0.295, +0.397] | 0.859 |

### Paired difference vs previous step

Candidate `/home/aleksandar/work/populism-axis/results/v4/step6_nli/combined_k8` minus baseline `/home/aleksandar/work/populism-axis/results/v4/step4_windows_topk/grid/spw3_k8`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | +0.023 | [+0.006, +0.049] | 0.007 | 5272 | 6 |
| english (leader-clustered) | +0.020 | [-0.041, +0.085] | 0.516 | 174 | 57 |

Rule checks (standard):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: pass
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: pass

Outcome: **ADOPT**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B4 axis only (carried) | `2be6215` | False | `76544d5f4493` | laptop-omarchy |
| NLI only, top 8 | `9cc1780` | False | `2ba0ac4b47b6` | laptop-omarchy |
| NLI only, top 3 | `9cc1780` | False | `8e4cb9bf4fc7` | laptop-omarchy |
| Axis + NLI combined, top 8 | `9cc1780` | False | `bf037884826a` | laptop-omarchy |
