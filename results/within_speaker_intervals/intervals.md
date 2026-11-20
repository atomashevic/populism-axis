# Bootstrap intervals: texts resampled within speakers

4000 resamples, seed 20260923. The cluster column resamples whole clusters, as the stored intervals did; largest gap between the recomputed and stored cluster intervals 0.011.

| Benchmark | Statistic | Estimate | Within strata | Whole clusters | Strata |
|---|---|---:|---|---|---|
| GPD English evaluation speeches | B6_axis_plus_nli | +0.518 | [+0.377, +0.642] | [+0.282, +0.701] | 27 (leader) |
| GPD English evaluation speeches | B4_axis | +0.417 | [+0.263, +0.557] | [+0.158, +0.621] | 27 (leader) |
| GPD English evaluation speeches | NLI_only | +0.520 | [+0.364, +0.660] | [+0.305, +0.699] | 27 (leader) |
| GPD English evaluation speeches | B0_v3_config | +0.376 | [+0.226, +0.511] | [+0.111, +0.604] | 27 (leader) |
| GPD English evaluation speeches | RoBERTa | +0.440 | [+0.310, +0.556] | [+0.140, +0.685] | 27 (leader) |
| GPD English evaluation speeches | B6_axis_plus_nli - B0_v3_config | +0.142 | [+0.070, +0.215] | [+0.004, +0.332] | 27 (leader) |
| GPD English evaluation speeches | B6_axis_plus_nli - B4_axis | +0.101 | [+0.045, +0.164] | [+0.009, +0.241] | 27 (leader) |
| GPD English evaluation speeches | B6_axis_plus_nli - RoBERTa | +0.079 | [-0.034, +0.191] | [-0.080, +0.251] | 27 (leader) |
| GPD English evaluation speeches | B4_axis - RoBERTa | -0.023 | [-0.145, +0.095] | [-0.175, +0.130] | 27 (leader) |
| GPD English evaluation speeches | B4_axis - B0_v3_config | +0.041 | [-0.013, +0.099] | [-0.044, +0.129] | 27 (leader) |
| EU-OOD, both claims vs neither | B6_axis_plus_nli | +0.942 | [+0.905, +0.969] | [+0.902, +0.951] | 5 (speaker) |
| EU-OOD, both claims vs neither | RoBERTa | +0.900 | [+0.838, +0.952] | [+0.729, +0.978] | 5 (speaker) |
| EU-OOD, both claims vs neither | B6_axis_plus_nli - RoBERTa | +0.041 | [-0.020, +0.108] | [-0.062, +0.179] | 5 (speaker) |
| AgoraSpeech paragraphs (development) | combined | +0.382 | [+0.359, +0.404] | [+0.295, +0.397] | 6 (politician) |
| AgoraSpeech paragraphs (development) | RoBERTa | +0.372 | [+0.347, +0.396] | [+0.289, +0.391] | 6 (politician) |
| AgoraSpeech paragraphs (development) | combined - RoBERTa | +0.011 | [-0.012, +0.033] | [-0.036, +0.067] | 6 (politician) |
| US governor speeches 2010-18 | B6_axis_plus_nli | +0.526 | [+0.442, +0.602] | [+0.438, +0.605] | 73 (state-term) |
| US governor speeches 2010-18 | B4_axis | +0.458 | [+0.372, +0.540] | [+0.365, +0.545] | 73 (state-term) |
| US governor speeches 2010-18 | NLI_only | +0.399 | [+0.306, +0.492] | [+0.282, +0.505] | 73 (state-term) |
| US governor speeches 2010-18 | B0_v3_config | +0.345 | [+0.252, +0.436] | [+0.229, +0.456] | 73 (state-term) |
| US governor speeches 2010-18 | RoBERTa | +0.326 | [+0.231, +0.415] | [+0.197, +0.447] | 73 (state-term) |
| US governor speeches 2010-18 | B6_axis_plus_nli - RoBERTa | +0.200 | [+0.109, +0.290] | [+0.086, +0.318] | 73 (state-term) |
| US governor speeches 2010-18 with SetFit predictions | SetFit_share_published | +0.514 | [+0.430, +0.593] | [+0.403, +0.614] | 72 (state-term) |
| US governor speeches 2010-18 with SetFit predictions | B6_axis_plus_nli - SetFit_share_published | +0.006 | [-0.069, +0.079] | [-0.078, +0.093] | 72 (state-term) |
| Party tweets, general elite criticism | E_axis_plus_nli | +0.756 | [+0.724, +0.786] | [+0.714, +0.801] | 20 (country) |
| Party tweets, general elite criticism | RoBERTa_anti_elitism | +0.571 | [+0.529, +0.613] | [+0.532, +0.605] | 20 (country) |
| Party tweets, general elite criticism | E_axis_plus_nli - XLMT_released_classifier | -0.052 | [-0.091, -0.015] | [-0.101, -0.001] | 20 (country) |
| Party tweets, general elite criticism | E_axis_plus_nli - RoBERTa_anti_elitism | +0.185 | [+0.136, +0.232] | [+0.124, +0.257] | 20 (country) |
| PopBERT test sentences, anti-elitism | E_axis_plus_nli | +0.806 | [+0.780, +0.832] | [+0.781, +0.832] | 1 (none (one stratum)) |
| PopBERT test sentences, anti-elitism | E_roberta | +0.669 | [+0.631, +0.706] | [+0.631, +0.705] | 1 (none (one stratum)) |
| PopBERT test sentences, anti-elitism | E_axis_plus_nli - E_roberta | +0.137 | [+0.093, +0.181] | [+0.095, +0.180] | 1 (none (one stratum)) |
| Portuguese parliamentary paragraphs, anti-elitism | E_axis_plus_nli | +0.798 | [+0.757, +0.839] | [+0.744, +0.869] | 144 (speaker) |
| Portuguese parliamentary paragraphs, anti-elitism | E_roberta | +0.867 | [+0.828, +0.902] | [+0.817, +0.917] | 144 (speaker) |
| Portuguese parliamentary paragraphs, anti-elitism | E_axis_plus_nli - supervised_lr_elite | -0.024 | [-0.076, +0.026] | [-0.063, +0.025] | 144 (speaker) |
| Portuguese parliamentary paragraphs, anti-elitism | E_axis_plus_nli - E_roberta | -0.069 | [-0.118, -0.020] | [-0.116, -0.015] | 144 (speaker) |
| PopBERT test sentences, people-centrism | P_axis_plus_nli | +0.771 | [+0.709, +0.830] | [+0.709, +0.832] | 1 (none (one stratum)) |
| PopBERT test sentences, people-centrism | P_roberta | +0.690 | [+0.618, +0.760] | [+0.621, +0.759] | 1 (none (one stratum)) |
| PopBERT test sentences, people-centrism | P_axis_plus_nli - P_roberta | +0.081 | [+0.008, +0.154] | [+0.008, +0.156] | 1 (none (one stratum)) |
| Portuguese parliamentary paragraphs, people-centrism | P_axis_plus_nli | +0.695 | [+0.666, +0.725] | [+0.642, +0.736] | 144 (speaker) |
| Portuguese parliamentary paragraphs, people-centrism | P_roberta | +0.582 | [+0.546, +0.618] | [+0.525, +0.636] | 144 (speaker) |
| Portuguese parliamentary paragraphs, people-centrism | P_axis_plus_nli - supervised_lr_centr | -0.103 | [-0.140, -0.066] | [-0.173, -0.048] | 144 (speaker) |
| Portuguese parliamentary paragraphs, people-centrism | P_axis_plus_nli - P_roberta | +0.113 | [+0.076, +0.150] | [+0.062, +0.143] | 144 (speaker) |
| PopBERT test sentences, anti-elitism | E_nli - E_axis_plus_nli | +0.014 | [-0.014, +0.041] | [-0.013, +0.042] | 1 (none (one stratum)) |
