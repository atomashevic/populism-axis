Frozen reference scales: every transductive statistic replaced by its English-dev value. ρ(t, f) is the Spearman rank agreement between the transductive and frozen scores on the evaluated rows; Δ is the frozen minus transductive metric with a cluster-bootstrap 95% CI.

| Corpus | System | Target | Metric | n | ρ(t, f) | transductive | frozen | Δ [95% CI] |
|---|---|---|---|---:|---:|---:|---:|---|
| english_dev | PE_axis |  | spearman | 174 | 1.000 | 0.517 | 0.517 | +0.000 [+0.000, +0.000] |
| english_dev | PE_axis_plus_nli |  | spearman | 174 | 1.000 | 0.537 | 0.537 | +0.000 [+0.000, +0.000] |
| agora | PE_axis |  | spearman | 5272 | 0.989 | 0.359 | 0.364 | +0.005 [-0.001, +0.013] |
| agora | PE_axis_plus_nli |  | spearman | 5272 | 0.987 | 0.382 | 0.393 | +0.010 [+0.006, +0.014] |
| english_holdout | PE_axis |  | spearman | 84 | 0.999 | 0.417 | 0.415 | -0.002 [-0.014, +0.007] |
| english_holdout | PE_axis_plus_nli |  | spearman | 84 | 0.988 | 0.518 | 0.547 | +0.029 [-0.007, +0.074] |
| eu | PE_axis | 3v0 | auroc | 1597 | 0.998 | 0.890 | 0.901 | +0.011 [+0.002, +0.014] |
| eu | PE_axis_plus_nli | 3v0 | auroc | 1597 | 0.935 | 0.942 | 0.940 | -0.001 [-0.025, +0.002] |
| eu | E_axis_plus_nli | AE | auroc | 1762 | 0.994 | 0.828 | 0.815 | -0.014 [-0.019, -0.002] |
| eu | P_axis_plus_nli | PC | auroc | 1762 | 0.990 | 0.863 | 0.851 | -0.012 [-0.020, -0.006] |
| usgov | PE_axis | gov2010_18 | spearman | 288 | 0.999 | 0.458 | 0.461 | +0.003 [-0.003, +0.010] |
| usgov | PE_axis_plus_nli | gov2010_18 | spearman | 288 | 0.994 | 0.526 | 0.540 | +0.014 [-0.001, +0.028] |
| usgov | PE_axis_plus_nli | pooled | spearman | 359 | 0.995 | 0.581 | 0.591 | +0.010 [+0.000, +0.022] |
| tweets | E_axis | general | auroc | 1059 | 0.999 | 0.754 | 0.755 | +0.001 [-0.001, +0.002] |
| tweets | E_axis_plus_nli | general | auroc | 1059 | 0.998 | 0.756 | 0.756 | +0.001 [-0.002, +0.004] |
| popbert | E_axis | AE | auroc | 1759 | 0.999 | 0.781 | 0.786 | +0.005 [+0.003, +0.007] |
| popbert | E_axis_plus_nli | AE | auroc | 1759 | 0.998 | 0.806 | 0.804 | -0.002 [-0.005, +0.000] |
| popbert | P_axis | PC | auroc | 1759 | 1.000 | 0.754 | 0.754 | +0.000 [-0.003, +0.003] |
| popbert | P_axis_plus_nli | PC | auroc | 1759 | 0.976 | 0.771 | 0.760 | -0.011 [-0.031, +0.006] |
| portugal | E_axis | AE | auroc | 6214 | 1.000 | 0.770 | 0.772 | +0.002 [+0.000, +0.004] |
| portugal | E_axis_plus_nli | AE | auroc | 6214 | 1.000 | 0.798 | 0.799 | +0.001 [-0.000, +0.002] |
| portugal | P_axis | PC | auroc | 6214 | 1.000 | 0.668 | 0.667 | -0.001 [-0.002, -0.000] |
| portugal | P_axis_plus_nli | PC | auroc | 6214 | 0.996 | 0.695 | 0.688 | -0.008 [-0.012, -0.005] |

Corpus statistics against the reference (English-dev). Projection mean / SD per channel on the corpus's own centering; ‖μ − μ_ref‖ is the distance between the corpus mean embedding and the reference mean.

| Corpus | windows | ‖μ − μ_ref‖ | cos(μ, μ_ref) | P proj mean / SD | E proj mean / SD | P act. scale | E act. scale |
|---|---:|---:|---:|---|---|---:|---:|
| english_dev | 20284 | 0.000 | 1.0000 | -0.0007 / 0.1036 | -0.0029 / 0.1176 | 0.495 | 0.311 |
| agora | 32498 | 3.142 | 0.9875 | +0.0020 / 0.0928 | +0.0003 / 0.1187 | 0.392 | 0.435 |
| english_holdout | 13373 | 1.552 | 0.9969 | +0.0000 / 0.1022 | -0.0017 / 0.1161 | 0.438 | 0.327 |
| eu | 1763 | 4.256 | 0.9768 | +0.0013 / 0.1032 | -0.0005 / 0.1278 | 0.546 | 0.315 |
| usgov | 51787 | 2.967 | 0.9889 | -0.0004 / 0.0956 | -0.0030 / 0.1199 | 0.455 | 0.259 |
| tweets | 5812 | 4.087 | 0.9788 | +0.0011 / 0.0803 | +0.0007 / 0.1080 | 0.504 | 0.400 |
| popbert | 8814 | 3.751 | 0.9820 | +0.0013 / 0.1023 | +0.0020 / 0.1437 | 0.559 | 0.654 |
| portugal | 9535 | 3.385 | 0.9855 | -0.0003 / 0.0957 | -0.0004 / 0.1179 | 0.300 | 0.312 |
