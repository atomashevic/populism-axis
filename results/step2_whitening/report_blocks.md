### Results on development sets

System `PE__frozen__soft`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B1c carried (V3 rule) | 0.483 | [+0.339, +0.618] | 0.940 | 0.359 | 0.359 | [+0.266, +0.378] | 0.889 |
| d0 no residualization | 0.482 | [+0.336, +0.620] | 0.942 | 0.360 | 0.362 | [+0.269, +0.380] | 0.895 |
| d1 no residualization | 0.488 | [+0.346, +0.624] | 0.941 | 0.359 | 0.341 | [+0.235, +0.367] | 0.885 |
| d2 + residualization (primary) | 0.410 | [+0.253, +0.554] | 0.883 | 0.150 | 0.296 | [+0.217, +0.330] | 0.953 |

### Paired difference vs previous step

Candidate `/home/aleksandar/work/populism-axis/results/v4/step2_whitening/grid/d2_res_nowhite` minus baseline `/home/aleksandar/work/populism-axis/results/v4/step1b_text_cleaning/B1c_cap60_clean`, system `PE__frozen__soft`, 4000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| agora (politician-clustered) | -0.063 | [-0.114, +0.001] | 0.057 | 5272 | 6 |
| english (leader-clustered) | -0.073 | [-0.165, +0.021] | 0.119 | 174 | 57 |

Rule checks (standard):

- primary_ci_not_entirely_below_0: pass
- primary_delta_at_least_+0.01: fail
- secondary_ci_not_entirely_below_0: pass
- secondary_delta_at_least_-0.01: fail

Outcome: **KEEP PREVIOUS**

### Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| B1c carried (V3 rule) | `be0cd6c` | False | `1ec489c0c163` | laptop-omarchy |
| d0 no residualization | `b1d7e06` | False | `c442dafc462f` | laptop-omarchy |
| d1 no residualization | `b1d7e06` | False | `393836c9d63c` | laptop-omarchy |
| d2 + residualization (primary) | `b1d7e06` | False | `a6ac2debfcc5` | laptop-omarchy |
| step2_whitening/grid.manifest.json | `b1d7e06` | False | `18ef2aedb0cc` | laptop-omarchy |
