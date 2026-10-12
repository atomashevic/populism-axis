### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B0 V3 splitter | 0.462 | [+0.309, +0.602] | 0.939 | 0.369 | 0.357 | [+0.260, +0.378] | 0.891 |
| B1 cap 60 words | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |

All systems for B1 cap 60 words (diagnostic; only the carried system decides):

| System | English ρ | Agora ρ |
|---|---:|---:|
| `EM__frozen__soft` | 0.430 | 0.320 |
| `E__frozen__soft` | 0.453 | 0.331 |
| `M__frozen__soft` | 0.397 | 0.296 |
| `PEM__frozen__soft` | 0.472 | 0.347 |
| `PE__frozen__hard` | 0.482 | 0.352 |
| `PE__frozen__soft` | 0.483 | 0.359 |
| `PM__frozen__soft` | 0.469 | 0.348 |
| `P__frozen__soft` | 0.493 | 0.341 |

### Paired difference vs previous step

Candidate `results/v4/step1_splitter/B1_cap60` minus baseline `results/v4/step1_splitter/B0_v3_splitter`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | +0.002 | [-0.001, +0.005] | 0.248 | 5272 | 6 |
| english (leader-clustered) | +0.021 | [-0.003, +0.055] | 0.102 | 174 | 57 |

Rule checks (defect_fix):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: pass
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: pass

Outcome: **ADOPT (defect fix, adopted regardless of the rule)**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B0 V3 splitter | `ff9af06` | False | `86aeb112e5b3` | laptop-omarchy |
| B1 cap 60 words | `ff9af06` | False | `4879864ba631` | laptop-omarchy |
| step1_splitter/paired_vs_previous.manifest.json | `ff9af06` | False | `c7c6e1c73846` | laptop-omarchy |
