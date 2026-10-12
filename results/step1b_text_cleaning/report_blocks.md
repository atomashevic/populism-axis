### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B1 cap 60 | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |
| B1c cap 60 + cleaning | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |

All systems for B1c cap 60 + cleaning (diagnostic; only the carried system decides):

| System | English ρ | Agora ρ |
|---|---:|---:|
| `EM__frozen__soft` | 0.430 | 0.320 |
| `E__frozen__soft` | 0.453 | 0.331 |
| `M__frozen__soft` | 0.397 | 0.296 |
| `PEM__frozen__soft` | 0.471 | 0.347 |
| `PE__frozen__hard` | 0.483 | 0.352 |
| `PE__frozen__soft` | 0.483 | 0.359 |
| `PM__frozen__soft` | 0.468 | 0.348 |
| `P__frozen__soft` | 0.495 | 0.341 |

### Paired difference vs previous step

Candidate `results/v4/step1b_text_cleaning/B1c_cap60_clean` minus baseline `results/v4/step1_splitter/B1_cap60`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | +0.000 | [+0.000, +0.000] | 1.000 | 5272 | 6 |
| english (leader-clustered) | -0.000 | [-0.003, +0.001] | 1.000 | 174 | 57 |

Rule checks (defect_fix):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: fail
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: pass

Outcome: **ADOPT (defect fix, adopted regardless of the rule)**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B1 cap 60 | `ff9af06` | False | `4879864ba631` | laptop-omarchy |
| B1c cap 60 + cleaning | `be0cd6c` | False | `1ec489c0c163` | laptop-omarchy |
| step1b_text_cleaning/paired_vs_previous.manifest.json | `6f6acfe` | False | `d330f19ad219` | laptop-omarchy |
| encode_manifests/spw1_cap60_clean.run_manifest.json | `7331bba` | False | `879e616a5937` | pvgpu1 |
