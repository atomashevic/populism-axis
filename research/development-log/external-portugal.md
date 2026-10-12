# External validation: Portuguese parliament, populist rhetoric during COVID-19

Date: 2026-09-24. Frozen V4 configurations scored once on a hand-coded Portuguese-language paragraph benchmark that no V4 decision has seen. Benchmark and scoring code: `scripts/external/`.

> **Correction, 2026-10-02.** The RoBERTa comparator was first scored with a softmax across its three labels; the checkpoint is a multi-label model with independent sigmoid outputs. Rows and figures marked *sigmoid* are the corrected comparator (run `results/roberta-correction/rc-20261002T1415Z-sigmoid/`: same model, windows, pooling, splits and seeds; `secondary_tables.json` there). Rows marked *softmax (superseded)* are kept as the historical record. V4 axis and NLI numbers are unchanged.

## Result

On 6,214 paragraphs from Assembleia da República debates, the pre-declared V4 route (Marian machine translation to English, e5-large-v2 axis plus the NLI stage) ranks hand-coded anti-elitism at AUROC .798 and people-centrism at .695. Reading the Portuguese original with a multilingual encoder instead (multilingual-e5-large-instruct, plain template, same anchors) is better on both: .838 and .758, and .845 / .773 with the NLI stage on the translation added. This is the first benchmark on which the fine-tuned RoBERTa comparator beats V4: its anti-elitism score reaches .867 with sigmoid scores (paired Δ against V4 E + NLI −.069 [−.116, −.016]; .868 and −.069 [−.110, −.012] under the superseded softmax), though it fails on people-centrism (.582; softmax .604). No model baseline is published for this dataset; a supervised logistic regression fitted here on the same e5 embeddings with speaker-grouped cross-validation reaches .823 for anti-elitism (V4 Δ −.024, n.s.) and .798 for people-centrism (Δ −.103). NLI adds +.03 to both channels. Speaker-clustered intervals are wide: 116 anti-elitism and 302 people-centrism positives.

## The benchmark

Source: Moniz, Brás, Santana Pereira, Belchior & Martins (2025), *Dataset on Parties' Use of Populist Rhetoric in Portugal During the COVID-19 Pandemic* (Zenodo doi:10.5281/zenodo.14626861, CC BY 4.0), analysed in Moniz et al. (2024), *Análise Social* (doi:10.31447/2022112). Paragraphs from 58 plenary sessions (October 2019 – April 2021) on populism-prone issues (banking and finance, law and order, the political system, morality politics, welfare, corruption, immigration, ethnic minorities), each hand-coded as a binary indicator of people-centrism, anti-elitism, popular sovereignty and exclusionism following Manucci & Weber (2017). Krippendorff's α on a 10% double-coded sample: .66, .77, .80, .73. The paper reports party-level and period-level rates, not a classifier.

| | Count |
|---|---:|
| Paragraphs (144 speakers, 9 parties) | 6,214 |
| People-centrism | 302 (4.9%) |
| Anti-elitism | 116 (1.9%) |
| Popular sovereignty | 23 |
| Exclusionism | 50 |
| Median words (Portuguese / English) | 59 / 60 |

`scripts/external/build_portugal_benchmark.py` writes `data/benchmark_portugal/paragraphs.csv` (tracked under CC BY; line breaks inside paragraphs joined, otherwise verbatim). `scripts/external/translate_marian.py` adds `text_en` in `paragraphs_en.csv`: sentence-level beam-search translation with `Helsinki-NLP/opus-mt-ROMANCE-en` (13,951 sentences, deterministic).

## Configurations

All label-free until thresholds and metrics; V3 anchors throughout. Paragraphs are documents: three-sentence windows with stride 1 and a 60-word cap, PE soft conjunction, mean of the top 8 windows (most paragraphs yield one or two windows), per-channel top-8 means of z-scored projections. Centering and z-scoring use all 6,214 paragraphs.

Pre-declared mapping: **P channel → people-centrism**, **E channel → anti-elitism**; the carried PE conjunction is secondary. Pre-declared primary route: e5-large-v2 on the English translation, with the NLI stage (BART-large-MNLI, top-8 windows, five P and five E anchor claims, z-mean with the axis). Sensitivity routes, both fixed before scoring: multilingual-e5-large-instruct on the Portuguese original with the plain template (`me5_plain`, the V4 step-5 finding that instructions hurt) and with its instruction (`me5_instr`, the V3 PopBERT configuration). The `me5_plain` axis combined with the NLI scores of the translation is a post-hoc hybrid and marked as such.

Comparators:

- **RoBERTa** (`coastalcph/roberta-large-ft-trump-populism`) on the English translation: anti-elitism and people-centrism class probabilities, maximum over one-sentence windows.
- **Supervised in-domain reference, fitted here**: logistic regression (C = 1, balanced class weights) on the mean raw e5-large-v2 window embedding of each paragraph, out-of-fold scores from 5-fold cross-validation grouped by speaker. No hyperparameter search. Not a published baseline.

No training split exists, so F1 is reported at the prevalence-matched threshold (uses only the positive count) and at the test-oracle threshold. CIs and paired differences are 95% speaker-clustered bootstrap (144 clusters, 2,000 draws).

## Results

### Anti-elitism (116 of 6,214 positive)

| System | AUROC [95% CI] | AP | F1, prevalence cut | F1, oracle cut |
|---|---|---:|---:|---:|
| **V4 E axis + NLI (translation; pre-declared primary)** | **.798 [.745, .867]** | .130 | .207 | .243 |
| V4 E axis (translation) | .770 [.718, .837] | .081 | .103 | .136 |
| NLI only, E claims (translation) | .767 [.718, .826] | .118 | .181 | .207 |
| E axis, multilingual-e5 plain on the original | .838 [.797, .884] | .125 | .216 | .234 |
| E axis, multilingual-e5 with instruction on the original | .807 [.761, .862] | .118 | .190 | .202 |
| E axis multilingual-e5 plain + NLI on translation (post hoc) | .845 [.802, .894] | .153 | .241 | .252 |
| V4 PE axis + NLI (carried conjunction) | .763 [.692, .841] | .087 | .172 | .189 |
| RoBERTa, anti-elitism, sigmoid | .867 [.818, .918] | .173 | .250 | .280 |
| RoBERTa, anti-elitism, softmax (superseded) | .868 [.829, .912] | .172 | .250 | .271 |
| RoBERTa, any populism (1 − no populism), sigmoid | .841 [.801, .888] | .142 | .224 | .240 |
| RoBERTa, any populism, softmax (superseded) | .849 [.814, .891] | .144 | .224 | .240 |
| Supervised LR on e5 embeddings (fitted here, grouped CV) | .823 [.780, .876] | .125 | .216 | .234 |

### People-centrism (302 of 6,214 positive)

| System | AUROC [95% CI] | AP | F1, prevalence cut | F1, oracle cut |
|---|---|---:|---:|---:|
| **V4 P axis + NLI (translation; pre-declared primary)** | **.695 [.641, .735]** | .136 | .222 | .222 |
| V4 P axis (translation) | .668 [.611, .708] | .130 | .175 | .194 |
| NLI only, P claims (translation) | .689 [.652, .721] | .126 | .162 | .190 |
| P axis, multilingual-e5 plain on the original | .758 [.711, .790] | .150 | .215 | .221 |
| P axis, multilingual-e5 with instruction on the original | .729 [.685, .762] | .125 | .162 | .190 |
| P axis multilingual-e5 plain + NLI on translation (post hoc) | .773 [.731, .804] | .161 | .202 | .229 |
| V4 PE axis + NLI (carried conjunction) | .701 [.650, .729] | .132 | .189 | .194 |
| RoBERTa, people-centrism, sigmoid | .582 [.523, .635] | .094 | .129 | .155 |
| RoBERTa, people-centrism, softmax (superseded) | .604 [.560, .642] | .092 | .132 | .148 |
| Supervised LR on e5 embeddings (fitted here, grouped CV) | .798 [.773, .825] | .239 | .325 | .334 |

### Paired AUROC differences (speaker-clustered bootstrap)

| Comparison | Anti-elitism Δ [95% CI] | p | People-centrism Δ [95% CI] | p |
|---|---:|---:|---:|---:|
| V4 axis + NLI − V4 axis | +.029 [+.021, +.036] | < .001 | +.028 [+.018, +.043] | < .001 |
| V4 axis + NLI − NLI only | +.031 [−.007, +.079] | .12 | +.007 [−.035, +.037] | .77 |
| V4 axis + NLI − multilingual-e5 plain axis | −.039 [−.065, −.006] | .02 | −.063 [−.094, −.030] | < .001 |
| V4 axis + NLI − PE axis + NLI | +.035 [−.007, +.079] | .11 | −.006 [−.048, +.041] | .81 |
| V4 axis + NLI − RoBERTa, sigmoid | −.069 [−.116, −.016] | .016 | +.113 [+.064, +.143] | .001 |
| V4 axis + NLI − RoBERTa, softmax (superseded) | −.069 [−.110, −.012] | .02 | +.092 [+.045, +.126] | .001 |
| V4 axis + NLI − supervised LR (fitted here) | −.024 [−.063, +.023] | .29 | −.103 [−.174, −.051] | < .001 |

### Discriminant check: every channel against every coded dimension (AUROC, translation route)

| Dimension (positives) | P axis | E axis | PE axis | P axis + NLI | E axis + NLI |
|---|---:|---:|---:|---:|---:|
| People-centrism (302) | .668 | .585 | .636 | .695 | .605 |
| Anti-elitism (116) | .687 | .770 | .763 | .672 | .798 |
| Popular sovereignty (23) | .822 | .551 | .678 | .832 | .550 |
| Exclusionism (50) | .819 | .825 | .856 | .791 | .849 |

### Breakdowns

- **Party ordering matches the paper.** Mean V4 anti-elitism z-score by party: Chega +.44, Left Bloc +.13, Liberal Initiative +.10, the other opposition parties near 0, PSD −.03, the governing PS −.25 (cabinet members −.41). The paper singles out Chega as the party that most mobilises people-centric and anti-elite claims.
- **By issue** (anti-elitism AUROC, V4 E + NLI / multilingual + NLI / RoBERTa): banking and finance .80 / .82 / .84; law and order .84 / .85 / .86; political system .82 / .86 / .90; corruption .68 / .79 / .70 (RoBERTa sigmoid; softmax .84 / .89 / .90 / .67).
- **RoBERTa and V4 rank different things.** Their anti-elitism scores correlate at Spearman .17 across all paragraphs (sigmoid; .22 under softmax); among the 116 positives, V4 places 24 and RoBERTa 29 inside the top 116, with only 9 in common (softmax: 24, 29, 8). RoBERTa's extra hits are institutional-capture claims: judges "appointed by PS and PSD", an Attorney-General "held hostage to the political power installed", parties as "the true arc of the system at the service of the great transnational capital".
- **V4's highest-scored non-anti-elite paragraphs are inequality and exclusion claims:** "To those who have money we extend a gold carpet … and to those who do not, we impose martyrdom" (coded exclusionist), "fill the pockets of those who live at the expense of the suffering of those who cannot defend themselves", "taking away those who work and giving away those who don't want to do anything". The coders' anti-elitism requires a people-versus-elite cleavage; the E axis also fires on rich-versus-poor and workers-versus-idle framings.
- **The lowest-scored people-centrism positives** are administrative sentences with "the Portuguese" as the beneficiary ("losses of the New Bank charged to the Resolution Fund, that is to say to all Portuguese"; a bill "in the interest of a higher income for the Portuguese after their contributory career"). The coding manual counts "close to the people" claims; the P anchors describe the people as a virtuous, unified actor.
- **The translation route trails the original** by .04–.06 AUROC on both dimensions, uniformly across parties. On PopBERT, where the translation was GPT-4o, English e5 matched the multilingual reading of the German; here the Marian model is the weaker link. The two routes' paragraph scores correlate at Spearman .80–.83.

## Reading

- **The measure transfers to a third language and a third genre** with the anchors unchanged: .80 anti-elitism AUROC through a small open-source translator, .84 when the encoder reads Portuguese directly. The NLI stage helps on paragraphs, as on sentences and speeches, and unlike on tweets.
- **This is the first benchmark on which the Trump-fine-tuned RoBERTa is the better anti-elitism detector.** Its training material (2016 campaign speeches) is closer to the coders' construct here, elite capture of institutions and the political system, than the V4 anchor claims, which foreground corruption and betrayal of the people. The same model is useless for people-centrism (.58), where V4 leads by .09.
- **People-centrism remains the weak channel** on every benchmark: .70 here, against .80 for a supervised model on the same embeddings, which shows that the embeddings carry the label and the P direction does not capture it as coded.
- **The PE axis also responds to exclusionism** (.86 on 50 positives) and the P axis to popular sovereignty (.82 on 23). The conjunction reads out-group antagonism as populist antagonism; it does not distinguish exclusionist from anti-elite claims.
- **Encoder choice for non-English material is open.** A multilingual encoder on the original beat English e5 on a Marian translation by a significant margin; on GPT-4o translations of German it did not. The carried V4 is an English system, and its extension to other languages needs either a strong translator or a re-run of the encoder step on multilingual models.

## Caveats

- **No published model baseline.** The supervised reference was fitted here, with no tuning, on the same embeddings the axis uses; it bounds what a linear reader of these embeddings can learn from labels, not what a fine-tuned transformer would reach.
- **Small positive counts.** 116 and 302 positives across 144 speakers; the anti-elitism intervals span about .12.
- **Single-coded labels.** Reliability is reported on a 10% sample (α .66 people-centrism), and the rest of the corpus is single-coded.
- **Machine translation, not audited.** Spot checks read well; one known glitch is the loss of a space at a sentence boundary the source lacked. The multilingual rows show the cost, not its cause.
- **Route choice.** The multilingual rows and the hybrid were not the pre-declared primary; they are reported as sensitivity, and the primary row stays the translation route.
- **Downloads.** `Helsinki-NLP/opus-mt-ROMANCE-en` (599 MB) was fetched to the pvgpu1 Hugging Face cache for this run; it was not pre-approved and can be deleted.

## Provenance

| Artifact | Commit | Host | Time / VRAM |
|---|---|---|---|
| Benchmark build (`data/benchmark_portugal/paragraphs.csv`, `results/external/portugal/benchmark_freeze.json`) | `244c25e` | local | – |
| Marian translation (`paragraphs_en.csv`, 13,951 sentences) | `244c25e` | pvgpu1 | 144 s, 1.4 GB |
| e5-large-v2 encode (9,535 windows), multilingual-e5 encodes (2 × 9,458), RoBERTa (15,607 windows) | `244c25e` | pvgpu1 | about 1.5 min total, ≤ 1.3 GB |
| NLI, 87,150 pairs | `244c25e` | pvgpu1 | 121 s, 2.5 GB |
| Axis stage, supervised reference and final metrics (`axis_scores.csv`, `final_scores.csv`, `final_results.json`, `final_results.md`) | this commit | local | – |

The GPU was idle before each job and returned to idle afterwards; all run manifests record a clean checkout.

```bash
uv run --extra cpu python scripts/external/build_portugal_benchmark.py
# pvgpu1: translate_marian.py --csv data/benchmark_portugal/paragraphs.csv --text-column text_pt --out-csv data/benchmark_portugal/paragraphs_en.csv
#         encode_windows.py --encoder e5-large-v2 --custom-csv data/benchmark_portugal/paragraphs_en.csv --text-column text_en --sentences-per-window 3 --stride 1 --max-words 60 --max-length 256 --out-root results/embeddings_external/portugal
#         encode_windows.py --encoder me5-large-instruct-noinstr|me5-large-instruct --custom-csv data/benchmark_portugal/paragraphs.csv --text-column text_pt ... (same windows)
#         encode_roberta_custom.py --csv data/benchmark_portugal/paragraphs_en.csv --text-column text_en --out-dir results/external/portugal/roberta
uv run --extra cpu python scripts/external/score_portugal.py axis
# pvgpu1: nli_stage.py --in-dir results/external/portugal --names portugal --model facebook/bart-large-mnli
uv run --extra cpu python scripts/external/score_portugal.py final
```
