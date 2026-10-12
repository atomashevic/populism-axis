# External validation: US governors and 2016 presidential candidates

Date: 2026-09-24. Frozen V4 configurations scored once on a benchmark that no V4 decision has seen. Benchmark and scoring code: `scripts/external/`.

> **Correction, 2026-10-02.** The RoBERTa comparator was first scored with a softmax across its three labels; the checkpoint is a multi-label model with independent sigmoid outputs. Rows and figures marked *sigmoid* are the corrected comparator (run `results/roberta-correction/rc-20261002T1415Z-sigmoid/`: same model, windows, pooling, splits and seeds; `secondary_tables.json` there). Rows marked *softmax (superseded)* are kept as the historical record. V4 axis and NLI numbers are unchanged. The sentence-level RoBERTa rows were re-encoded separately in run `results/roberta-correction/rc-20261002T1530Z-sigmoid-usgov-sentences/` (same checkpoint, windows and code path, from a clean commit).

## Result

On the largest subset, 288 governor speeches from 2010–2018, V4 axis + NLI reaches Spearman .526 against .345 for the V3 configuration (paired Δ +.181 [+.084, +.282]). It is indistinguishable from the paper's own supervised SetFit classifier on ranking (.514; Δ +.006 [−.078, +.093]) and ahead of the fine-tuned RoBERTa comparator (sigmoid .326; Δ +.200 [+.086, +.318]; softmax .376, Δ +.150 [+.045, +.255]). Pooled over all 359 speeches it is the best-ranking system (.581). The supervised SetFit model remains clearly better at the binary classification task (pooled AUROC .865 against .818), and on the 45 presidential speeches both supervised comparators rank slightly ahead of V4.

## The benchmark

Source: van der Veen, Dzebo, Littvay, Hawkins & Dar (2024), *Classifying Populist Language in American Presidential and Governor Speeches Using Automatic Text Analysis* (arXiv 2408.15213), replication package on Harvard Dataverse (doi:10.7910/DVN/ZOHMVO, CC0). Speeches are graded on the same holistic 0–2 rubric as the Global Populism Database, by student coders; the binary task uses the paper's cut at .5.

| Subset | Speeches | Speakers (clusters) | Score ≥ .5 | Median words |
|---|---:|---:|---:|---:|
| Governors 2010–2018 | 288 | 73 state-terms (38 states) | 60 | 1,794 |
| Governors 2019–2023 | 26 | 6 | 10 | 2,659 |
| Presidential candidates 2016 | 45 | 7 | 20 | 2,965 |

Sentences: 3,539 governor sentences (1,411 populist, 1,120 neutral, 1,008 pluralist) and 81 presidential sentences, extracted by the paper from coder rubrics of these same speeches.

Preparation (`scripts/external/build_usgov_benchmark.py`): each speech was re-extracted from its original Word file with paragraph breaks, passed through the V4 text cleaner and a dataset-specific rule that removes metadata lines ("State:", "Term:", "Word Counts:", "Note --") and leading title, byline, URL and date paragraphs before the first paragraph of 30+ words. 46 governor speeches have no Word file in the package and use the package's single-line CSV text, where only leading URLs could be stripped. In total, 650 lines and 0.3% of words were removed; the audit is in `results/external/usgov/cleaning_audit.json`. Four speeches are exact duplicates in the source (the same South Dakota speeches in two coder folders, with identical scores); they are kept, as in the paper.

Two overlaps with data the project already uses: one presidential speech (Trump, 22 June 2016) is the same text as `US_Trump_Campaign_1` in the GPD English holdout, and the RoBERTa comparator was fine-tuned on Trump 2016 campaign sentences, so it is in-domain for the presidential subset.

## Configurations

All scored once, label-free until the metrics, with the e5-large-v2 encoder and the V3 anchors. Centering and z-scoring use all 359 speeches pooled (one external dataset), and all sentences pooled.

- **B0, V3 configuration:** one-sentence windows, no cap, PE soft conjunction, mean of the top 3 windows.
- **B4, V4 axis:** cleaned text, three-sentence windows capped at 60 words per sentence, PE soft, mean of the top 8.
- **B6, V4 axis + NLI:** mean of the z-scores of B4 and the NLI score (BART-large-MNLI entailment of the ten frozen anchor claims on the top-8 windows; top window for sentences).
- **RoBERTa comparator:** `coastalcph/roberta-large-ft-trump-populism`, anti-elitism probability pooled over the top 3 one-sentence windows (the V3 comparator rule); populist-total max also reported.
- **SetFit (published):** the paper's own classifier output, the share of a speech's sentences it labels populist, taken from the replication package (274 of 288 governor speeches could be matched to the package's analysis files by text; all others matched).

Two comparators are not zero-shot: SetFit was trained on sentences drawn from the coder rubrics of these same speeches, and both its threshold and the paper's are chosen on the evaluation data. V4 has never seen this benchmark.

## Speech-level results

Spearman against the holistic score with 95% cluster-bootstrap CIs (clusters = state-term, governor or candidate). AUROC and AP are for the binary task. The last column mirrors the paper's protocol: an F1-optimal cutoff chosen on the same data, then accuracy, F1 and balanced accuracy.

### Governors 2010–2018 (n = 288, 73 clusters, 60 populist)

| System | Spearman [95% CI] | r² | AUROC | AP | In-sample cutoff: acc / F1 / bal. acc |
|---|---|---:|---:|---:|---|
| V3 configuration (B0) | .345 [.229, .456] | .16 | .701 | .439 | .72 / .49 / .69 |
| V4 axis (B4) | .458 [.365, .545] | .24 | .767 | .460 | .80 / .54 / .71 |
| **V4 axis + NLI (B6)** | **.526 [.438, .605]** | .27 | .813 | .488 | .77 / .57 / .76 |
| NLI only | .399 [.282, .505] | .12 | .728 | .454 | .77 / .51 / .70 |
| RoBERTa, anti-elitism top 3, sigmoid | .326 [.197, .447] | .15 | .678 | .455 | .71 / .47 / .68 |
| RoBERTa, anti-elitism top 3, softmax (superseded) | .376 [.252, .490] | .14 | .709 | .439 | .67 / .48 / .69 |
| SetFit, published shares (n = 274) | .514 [.403, .614] | .33 | **.839** | **.578** | .85 / .64 / .77 |
| Paper, Table 2 (same SetFit, its own run) | – | .39 | "AuROC" .77 | – | .85 / .69 / .77 |

### Governors 2019–2023 (n = 26, 6 clusters, 10 populist)

| System | Spearman [95% CI] | AUROC | AP | In-sample cutoff: acc / F1 / bal. acc |
|---|---|---:|---:|---|
| V3 configuration | .604 [.344, .778] | .838 | .797 | .81 / .74 / .79 |
| V4 axis | .609 [.197, .811] | .856 | .811 | .81 / .78 / .82 |
| **V4 axis + NLI** | .629 [.342, .789] | .863 | .828 | .81 / .78 / .82 |
| NLI only | .280 [−.038, .622] | .581 | .546 | .54 / .60 / .61 |
| RoBERTa, anti-elitism top 3, sigmoid | .732 [.322, .898] | .838 | .827 | .77 / .75 / .79 |
| RoBERTa, anti-elitism top 3, softmax (superseded) | .650 [.112, .893] | .787 | .791 | .81 / .74 / .79 |
| SetFit, published shares | .587 [.214, .835] | **.938** | **.840** | .92 / .91 / .94 |
| Paper, Table 2 | – | "AuROC" .91 | – | .92 / .90 / .91 |

### Presidential candidates 2016 (n = 45, 7 clusters, 20 populist)

| System | Spearman [95% CI] | AUROC | AP | In-sample cutoff: acc / F1 / bal. acc |
|---|---|---:|---:|---|
| V3 configuration | .397 [.089, .593] | .686 | .609 | .71 / .68 / .71 |
| V4 axis | .568 [.287, .657] | .746 | .699 | .73 / .74 / .74 |
| **V4 axis + NLI** | .686 [.390, .746] | .846 | .839 | .82 / .81 / .82 |
| NLI only | .611 [.190, .720] | .848 | .880 | .84 / .81 / .83 |
| RoBERTa, anti-elitism top 3 (in-domain), sigmoid | .696 [.397, .803] | .814 | .767 | .80 / .79 / .81 |
| RoBERTa, anti-elitism top 3 (in-domain), softmax (superseded) | .706 [.359, .778] | .848 | .797 | .84 / .84 / .85 |
| SetFit, published shares | **.723 [.238, .802]** | **.924** | **.908** | .89 / .88 / .89 |
| Paper, Table 2 | – | "AuROC" .89 | – | .89 / .87 / .89 |

### All speeches pooled (n = 359, 86 clusters, 90 populist)

| System | Spearman [95% CI] | AUROC | AP |
|---|---|---:|---:|
| V3 configuration | .448 [.319, .567] | .740 | .504 |
| V4 axis | .530 [.418, .626] | .785 | .546 |
| **V4 axis + NLI** | **.581 [.479, .669]** | .818 | .597 |
| RoBERTa, anti-elitism top 3, sigmoid | .445 [.303, .577] | .732 | .562 |
| RoBERTa, anti-elitism top 3, softmax (superseded) | .476 [.341, .596] | .751 | .575 |
| SetFit, published shares | .548 [.423, .651] | **.865** | **.656** |

### Paired differences in Spearman (cluster bootstrap, 4,000 draws)

| Comparison | Governors 2010–18 | Governors 2019–23 | Presidential | Pooled |
|---|---|---|---|---|
| V4 axis − V3 | **+.113 [+.039, +.191]** | +.005 [−.259, +.138] | +.171 [−.170, +.473] | **+.082 [+.026, +.144]** |
| V4 axis + NLI − V3 | **+.181 [+.084, +.282]** | +.024 [−.148, +.138] | **+.289 [+.056, +.510]** | **+.133 [+.060, +.213]** |
| V4 axis + NLI − V4 axis | **+.068 [+.031, +.106]** | +.020 [−.063, +.164] | +.117 [−.077, +.301] | **+.051 [+.023, +.083]** |
| V4 axis + NLI − SetFit | +.006 [−.078, +.093] | +.041 [−.119, +.259] | −.038 [−.189, +.227] | +.031 [−.045, +.105] |
| V4 axis + NLI − RoBERTa, sigmoid | **+.200 [+.086, +.318]** | −.104 [−.329, +.222] | −.011 [−.229, +.168] | **+.136 [+.043, +.239]** |
| V4 axis + NLI − RoBERTa, softmax (superseded) | +.150 [+.045, +.255] | −.022 [−.308, +.414] | −.021 [−.179, +.160] | +.105 [+.023, +.197] |

## Sentence-level results

Zero-shot ranking of the paper's rubric sentences. The paper trains SetFit on 75% of these sentences and reports 3-class accuracy of .66 (governors) and .70 (presidential) on the remaining 25%; a supervised accuracy is not comparable with a zero-shot AUROC, so both are given as they are.

| Subset | System | AUROC populist vs rest | AP | AUROC vs pluralist | AUROC vs neutral |
|---|---|---:|---:|---:|---:|
| Governors (n = 3,539) | V4 axis | .731 | .629 | .736 | .726 |
| | **V4 axis + NLI** | **.744** | **.650** | **.759** | **.731** |
| | NLI only | .696 | .620 | .726 | .669 |
| | RoBERTa, populist total, sigmoid | .625 | .595 | .619 | .630 |
| | RoBERTa, populist total, softmax (superseded) | .630 | .591 | .626 | .633 |
| Presidential (n = 81) | V4 axis | .921 | .938 | .928 | – |
| | **V4 axis + NLI** | **.936** | **.950** | **.941** | – |
| | RoBERTa, populist total, sigmoid | .860 | .826 | .869 | – |
| | RoBERTa, populist total, softmax (superseded) | .863 | .831 | .873 | – |

## Reading

- **The V4 gains transfer.** Every V4 change that was adopted on the development data helps here too, in the same order: the axis changes add +.11 on the main subset and the NLI stage another +.07, both with intervals above zero. This is the third dataset, after the GPD holdout and EU-OOD, on which axis + NLI beats the V3 configuration and the fine-tuned RoBERTa.
- **Ranking parity with a supervised in-domain model.** The paper's SetFit classifier was trained on sentences from these speeches and tuned on the same data. On ranking speeches by their populism score, V4 axis + NLI matches it on every subset within the intervals. Where SetFit clearly wins is the binary decision (AUROC .865 against .818 pooled; AP .656 against .597): its sentence classifier is sharper at the top of the distribution. That is the same shape as on Agora, where RoBERTa separated the extreme paragraphs better.
- **Presidential speeches are where supervised models are strongest.** Both comparators were trained on Trump-era campaign language (RoBERTa on Trump 2016 sentences, SetFit on presidential rubric sentences), and both edge out V4 there, though not significantly with 7 clusters.
- **NLI alone is unstable on small sets.** It ranks the 26 recent governor speeches poorly (.28) but the presidential speeches well (.61). Combined with the axis it is never worse than the axis alone.
- **Sentence-level, the axis reads the rubric sentences better than the fine-tuned RoBERTa** (AUROC .744 against .625 on governors, .936 against .860 on candidates; .630 and .863 under the superseded softmax). These sentences were selected by coders as clear examples of populist, pluralist or neutral language, which suits a claim-level measure.

## Caveats

- **This is one external dataset in one country.** Governors' rhetoric is close to GPD's English speeches in genre, and the rubric is the same. It tests transfer within the construct and language, not across them.
- **Comparator numbers are not all like-for-like.** SetFit shares are the paper's own outputs on data it was trained on, matched by text for 274 of 288 governor speeches. The paper's "AuROC" column is the balanced accuracy of a thresholded prediction, not a ranking AUROC; the ranking AUROC of its released shares (.839 on governors 2010–18) is reported instead.
- **Small subsets.** The 2019–2023 and presidential subsets have 6 and 7 clusters; their intervals span .3 to .5.
- **Text quality.** 46 speeches come from single-line CSV text with residual titles inline, and several transcripts contain timestamps and speaker tags. The published comparators saw the same text.
- **Duplicates and overlap.** Four governor speeches are duplicated in the source with identical scores; one presidential speech is also in the GPD holdout.
- **Pooled standardization.** Centering and z-scoring were done over all 359 speeches, so each subset's scores depend on the other subsets' text; this is label-free and is the same transductive rule V3 and V4 use.

## Provenance

| Artifact | Commit | Host | Time / VRAM |
|---|---|---|---|
| Benchmark build (`data/benchmark_usgov/`, `results/external/usgov/benchmark_freeze.json`) | `dd9e073`, `3814010` | local | – |
| e5-large-v2 encodes: speeches one-sentence (51,000 windows), speeches three-sentence (51,787), sentences (4,226) | `dd9e073` | pvgpu1 | about 6 min total, 0.8 GB |
| RoBERTa comparator, speeches and sentences | `dd9e073` | pvgpu1 | about 3 min |
| NLI, 28,590 + 36,200 pairs | `dd9e073` | pvgpu1 | 69 s, 2.5 GB |
| Axis stage and final metrics (`final_results.json`, `final_results.md`, `final_scores_*.csv`) | `3814010` | local | – |

The GPU was idle before each job and returned to idle afterwards. Two launches failed before any GPU work because the remote checkout had not been updated; they used no GPU time.

```bash
uv run --extra cpu python scripts/external/build_usgov_benchmark.py
# pvgpu1: encode_windows.py --custom-csv ... (spw1 cap0; spw3 cap60 256 tokens; sentences), encode_roberta_custom.py
uv run --extra cpu python scripts/external/score_usgov.py axis
# pvgpu1: nli_stage.py --in-dir results/external/usgov --names usgov_speeches,usgov_sentences
uv run --extra cpu python scripts/external/score_usgov.py final
```
