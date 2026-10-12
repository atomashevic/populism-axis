# External validation: PopBERT in English translation

Date: 2026-09-24. Frozen V4 configurations scored once on the frozen PopBERT split, read through the GPT-4o English translations released by Wang et al. (2026). Benchmark and scoring code: `scripts/external/`.

> **Correction, 2026-10-02.** The RoBERTa comparator was first scored with a softmax across its three labels; the checkpoint is a multi-label model with independent sigmoid outputs. Rows and figures marked *sigmoid* are the corrected comparator (run `results/roberta-correction/rc-20261002T1415Z-sigmoid/`: same model, windows, pooling, splits and seeds; `secondary_tables.json` there). Rows marked *softmax (superseded)* are kept as the historical record. V4 axis and NLI numbers are unchanged.

## Result

On the 1,759 test sentences of the frozen PopBERT split, the V4 anti-elitism channel with the NLI stage ranks majority-coded anti-elitism at AUROC .806 (axis alone .781), and the people-centrism channel ranks people-centrism at .771 (axis alone .754). These are the first PopBERT numbers obtained with the English-only V4 encoder: the V3 transfer check, which read the German sentences with a multilingual encoder, reached .794 and .756 with the axis alone, so translating the corpus costs nothing and the NLI stage adds .025 on anti-elitism (paired [+.017, +.034]). At thresholds fitted on the training split, positive-class F1 is .468 for anti-elitism and .236 for people-centrism, against a single-coder ceiling of .576 and .327 (each coder against the majority of the other four) and the published in-domain PopBERT classifier at .847 and .696. The fine-tuned RoBERTa comparator reaches .669 and .690 AUROC with its sigmoid scores (.650 and .643 under the superseded softmax). Anti-elitism is the first task on which the NLI signal alone (.820) is at least as good as the axis + NLI combination (Δ +.014 [−.015, +.040]).

## The benchmark

Source sentences and labels: PopBERT (Erhard, Hanke, Remer, Falenska & Heiberger 2025, *Political Analysis*, doi:10.1017/pan.2024.12), 8,795 Bundestag sentences 2013–2021, each coded by five trained coders for anti-elitism, people-centrism, left-wing and right-wing host ideology. The repository's frozen split (`pa_popbert_benchmark_freeze_r1`, `data/benchmark_popbert/{train,test}.csv`) uses the majority of the five coders as gold; the V3 transfer check was evaluated on it.

English text: Wang, Yu, Wang & He (2025, arXiv:2505.07874), *The Sound of Populism*, whose accompanying dataset (Wang 2026, Harvard Dataverse doi:10.7910/DVN/VJGS7H, CC0) releases a GPT-4o translation of every PopBERT sentence (`gpt_translations.csv`). The file joins the frozen split 1:1: all 8,795 German sentences are identical to the repository's copy, and the per-coder labels reproduce the frozen majority gold exactly. `scripts/external/build_popbert_en_benchmark.py` attaches the translation as `text_en` and writes `data/benchmark_popbert/{train,test}_en.csv` and `results/external/popbert_en/benchmark_freeze.json`.

| | Train | Test |
|---|---:|---:|
| Sentences | 7,036 | 1,759 |
| Anti-elitism, majority gold | 1,062 | 278 |
| People-centrism, majority gold | 221 | 60 |
| Anti-elitism, at least one coder | 2,551 | 648 |
| People-centrism, at least one coder | 1,293 | 322 |
| Median words (English) | 21 | 21 |

## Configurations

All scored once, label-free until the thresholds and metrics, with e5-large-v2 and the frozen V3 anchors. Each sentence is one document; the carried three-sentence windowing reduces to a single window for all but a few long sentences (8,814 windows for 8,795 sentences), so the top-8 pooling is the sentence's own projection. Centering and z-scoring use all 8,795 sentences, as the V3 PopBERT run did.

The benchmark has one label per component, so the pre-declared mapping is **E channel → anti-elitism** and **P channel → people-centrism**. For each, the V4 + NLI score is the mean of the z-scored axis projection and the z-scored maximum entailment (BART-large-MNLI) of that channel's five anchor claims; NLI alone and the carried PE conjunction are reported as secondary.

Comparators:

- **V3 transfer check** (`results/popbert/evaluation.json`): the same anchors and axis rule, multilingual-e5-large-instruct reading the German sentences, evaluated on the same split. No per-sentence scores were stored, so the comparison is unpaired.
- **RoBERTa** (`coastalcph/roberta-large-ft-trump-populism`) on the English translation: anti-elitism and people-centrism class probabilities.
- **PopBERT** (published): GBERT-large fine-tuned on these sentences, 5-fold cross-validation F1, in-domain and supervised. The V3 freeze notes that its F1 exceeds every single coder's agreement with the majority.
- **Wang et al. RoBERTa-large** (published): fine-tuned on the same GPT-4o translations, F1 .85 anti-elitism / .71 people-centrism. Their notebook defines a positive as *any* of the five coders voting yes, draws its own random train/dev/test split (three seeds), and reports the maximum F1 along the test precision–recall curve. None of these choices matches the frozen benchmark, so their numbers are shown next to the "any coder" rows at the test-oracle threshold, the nearest reading, and are not a paired comparison.

The F1-optimal threshold is fitted on the training split only. The prevalence-matched threshold takes only the positive rate from the training labels and places the cut at that rate within the test split's own scores, so it uses the unlabeled test-score distribution. An oracle threshold on the test split is shown as an upper bound.

## Results on the test split, majority gold

CIs are 95% sentence bootstrap (2,000 draws; the frozen split carries no speaker ids).

### Anti-elitism (278 of 1,759 positive)

| System | AUROC [95% CI] | AP | F1, train-prevalence cut | F1, train-optimal cut | F1, oracle cut |
|---|---|---:|---:|---:|---:|
| **V4 E axis + NLI** | **.806 [.781, .832]** | .435 | .449 | .468 | .483 |
| NLI only, E claims | .820 [.796, .844] | .442 | .445 | .488 | .501 |
| V4 E axis | .781 [.754, .808] | .379 | .393 | .430 | .450 |
| V3 axis, multilingual-e5 on German (unpaired) | .794 | .404 | .397 | .431 | – |
| V4 PE axis (carried conjunction) | .731 [.700, .761] | .299 | .331 | .385 | .405 |
| V4 PE axis + NLI | .723 [.693, .753] | .316 | .338 | .372 | .387 |
| RoBERTa, anti-elitism, sigmoid | .669 [.632, .704] | .332 | .357 | .360 | .372 |
| RoBERTa, anti-elitism, softmax (superseded) | .650 [.613, .686] | .325 | .342 | .328 | .353 |
| Single coder vs majority of the other four | – | – | | .576 | |
| PopBERT GBERT, 5-fold CV (supervised, in-domain) | – | – | | .847 | |

### People-centrism (60 of 1,759 positive)

| System | AUROC [95% CI] | AP | F1, train-prevalence cut | F1, train-optimal cut | F1, oracle cut |
|---|---|---:|---:|---:|---:|
| **V4 P axis + NLI** | **.771 [.706, .831]** | .148 | .207 | .236 | .254 |
| V4 P axis | .754 [.682, .816] | .152 | .207 | .177 | .255 |
| V3 axis, multilingual-e5 on German (unpaired) | .756 | .138 | .155 | .204 | – |
| NLI only, P claims | .726 [.655, .792] | .114 | .190 | .173 | .196 |
| V4 PE axis (carried conjunction) | .655 [.580, .726] | .068 | .121 | .105 | .135 |
| RoBERTa, people-centrism, sigmoid | .690 [.617, .757] | .077 | .103 | .137 | .158 |
| RoBERTa, people-centrism, softmax (superseded) | .643 [.566, .713] | .062 | .103 | .111 | .143 |
| Single coder vs majority of the other four | – | – | | .327 | |
| PopBERT GBERT, 5-fold CV (supervised, in-domain) | – | – | | .696 | |

### Paired AUROC differences (sentence bootstrap, 2,000 draws)

| Comparison | Δ AUROC | 95% CI | p |
|---|---:|---|---:|
| Anti-elitism: E axis + NLI − E axis | +.025 | [+.017, +.034] | < .001 |
| Anti-elitism: NLI only − E axis + NLI | +.014 | [−.015, +.040] | .31 |
| Anti-elitism: E axis + NLI − RoBERTa, sigmoid | +.137 | [+.095, +.181] | < .001 |
| Anti-elitism: E axis + NLI − RoBERTa, softmax (superseded) | +.156 | [+.114, +.199] | < .001 |
| Anti-elitism: E axis + NLI − PE axis + NLI | +.084 | [+.061, +.106] | < .001 |
| People-centrism: P axis + NLI − P axis | +.018 | [−.006, +.045] | .15 |
| People-centrism: NLI only − P axis + NLI | −.045 | [−.118, +.027] | .23 |
| People-centrism: P axis + NLI − RoBERTa, sigmoid | +.081 | [+.005, +.151] | .030 |
| People-centrism: P axis + NLI − RoBERTa, softmax (superseded) | +.128 | [+.052, +.206] | .001 |
| People-centrism: P axis + NLI − PE axis + NLI | +.121 | [+.071, +.169] | < .001 |

### Secondary target: at least one coder (the Wang et al. definition)

| System | Anti-elitism AUROC | AP | F1, train-optimal | F1, oracle | People-centrism AUROC | AP | F1, train-optimal | F1, oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 axis + NLI | .808 | .695 | .668 | .680 | .740 | .420 | .429 | .442 |
| NLI only | .820 | .713 | .697 | .701 | .721 | .401 | .440 | .444 |
| V4 axis | .789 | .653 | .655 | .665 | .705 | .361 | .388 | .405 |
| RoBERTa, sigmoid | .637 | .548 | .533 | .547 | .639 | .307 | .352 | .370 |
| RoBERTa, softmax (superseded) | .620 | .538 | .515 | .540 | .613 | .293 | .342 | .345 |
| Wang et al. RoBERTa-large (published; own split, oracle cut, mean of 3 seeds) | – | – | | .85 | – | – | | .71 |

### Breakdowns

- **The scores track coder agreement, not just the majority.** Spearman between the anti-elitism score and the share of coders voting yes is .53 for axis + NLI (.55 for NLI, .50 for the axis). Of the 153 majority-negative sentences that axis + NLI ranks inside the top 278, 58% had at least one coder voting for anti-elitism.
- **The highest-scored "false positives" are anti-elite claims by any reading:** "These war profiteers live in fine villas … while their people starve", "This is class warfare from above … in the interest of the wealthy and property owners against the majority of taxpayers", "the private fortune of Amazon CEO Bezos has grown by 75 billion euros" while cinemas lose 70 percent. Each was coded populist by one or two of the five coders.
- **The lowest-scored positives are elite criticism carried by the German context rather than by the sentence:** "It is your damn duty as a representative to respond to legitimate criticism", "in the best CSU self-service manner", "Madam Chancellor, please finally listen to these experts". The anchors describe a corrupt, self-serving elite betraying the people; addressing a minister or party by name is not enough for either the axis or the entailment model.
- **The NLI stage works on parliamentary sentences and not on tweets.** Here NLI alone reaches .820 on anti-elitism, .04 above the axis; on the party tweets it was .08 below the axis. Bundestag sentences are complete propositions of about 21 words that BART-large-MNLI can match to the anchor claims; tweets are fragments. NLI scores correlate with sentence length more than the axis does (Spearman .21 against .10 with word count).
- **People-centrism is the weaker channel everywhere.** With 60 test positives, the intervals are wide, AP is .15, and the F1 ceiling for a single coder is .327; the V4 + NLI F1 of .236 is 72% of that ceiling, against 81% (.468 of .576) for anti-elitism.

## Reading

- **Translation plus the English V4 encoder matches the multilingual V3 reading of the German original** (.781 against .794 axis-only, unpaired), and the NLI stage moves anti-elitism to .806. The claim that the anchors transfer to German parliamentary speech now holds under two different encoders and two languages of the same text.
- **The training-free measure sits at four fifths of a single coder's agreement with the majority** on both components, and well below the supervised in-domain classifiers, whose F1 exceeds any single coder. That gap is the price of using no labels from the corpus: PopBERT and the Wang et al. RoBERTa learn the coding manual's operationalization, including the German context and speaker conventions that the anchor claims do not encode.
- **The carried PE conjunction is again the wrong instrument for a single-component label**, costing .08 (anti-elitism) and .12 (people-centrism) AUROC against the matching channel, as on the party tweets. Scoring a benchmark's component with that component's channel is the rule, not an exception.
- **NLI alone leads on anti-elitism.** The combination remains the pre-declared V4 + NLI score and is reported as such; the NLI-only row is a finding for the summary, not a substitution.

## Caveats

- **Machine translation, not audited.** Every sentence reaches V4 through a GPT-4o translation that was not checked against the German; the parity with the multilingual-on-German result is indirect evidence of its adequacy.
- **Published numbers are not on this protocol.** PopBERT's F1 is 5-fold cross-validation on all 8,795 sentences; Wang et al.'s is on their own random split, with an any-coder positive rule and a test-optimal threshold, averaged over three seeds. Neither is a paired comparison with the frozen split, and both are supervised in-domain.
- **Labels enter through thresholds only** (train split) and through the z-scoring population (all sentences, label-free). AUROC and AP use no labels.
- **No speaker clustering.** Sentences from the same speech or speaker are treated as independent in the bootstrap; the frozen split does not carry speaker ids.
- **Sixty people-centrism positives.** Point estimates on that dimension move by several hundredths across bootstrap draws.

## Provenance

| Artifact | Commit | Host | Time / VRAM |
|---|---|---|---|
| Benchmark build (`data/benchmark_popbert/{train,test}_en.csv`, `results/external/popbert_en/benchmark_freeze.json`) | `aaa4efd` | local | – |
| e5-large-v2 encode (8,814 windows) and RoBERTa comparator (9,295 windows) | `aaa4efd` | pvgpu1 | under 1 min total, 0.8 GB |
| NLI, 87,950 pairs | `aaa4efd` | pvgpu1 | 71 s, 2.1 GB |
| Axis stage and final metrics (`axis_scores.csv`, `final_scores.csv`, `final_results.json`, `final_results.md`) | this commit | local | – |

The GPU was idle before each job and returned to idle afterwards; the run manifests record a clean checkout at `aaa4efd`.

```bash
uv run --extra cpu python scripts/external/build_popbert_en_benchmark.py
# pvgpu1: encode_windows.py --encoder e5-large-v2 --custom-csv data/benchmark_popbert/{test,train}_en.csv --text-column text_en \
#           --sentences-per-window 3 --stride 1 --max-words 60 --max-length 128 --out-root results/embeddings_external/popbert_en
#         encode_roberta_custom.py --csv data/benchmark_popbert/{test,train}_en.csv --text-column text_en --out-dir results/external/popbert_en/roberta
uv run --extra cpu python scripts/external/score_popbert_en.py axis
# pvgpu1: nli_stage.py --in-dir results/external/popbert_en --names popbert_en --model facebook/bart-large-mnli
uv run --extra cpu python scripts/external/score_popbert_en.py final
```
