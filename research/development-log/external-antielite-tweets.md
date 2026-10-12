# External validation: party anti-elitism tweets

Date: 2026-09-24. Frozen V4 configurations scored once on a benchmark that no V4 decision has seen. Benchmark and scoring code: `scripts/external/`.

> **Correction, 2026-10-02.** The RoBERTa comparator was first scored with a softmax across its three labels; the checkpoint is a multi-label model with independent sigmoid outputs. Rows and figures marked *sigmoid* are the corrected comparator (run `results/roberta-correction/rc-20261002T1415Z-sigmoid/`: same model, windows, pooling, splits and seeds; `secondary_tables.json` there). Rows marked *softmax (superseded)* are kept as the historical record. V4 axis and NLI numbers are unchanged.

## Result

On the paper's fixed test split of 1,059 party tweets, the V4 anti-elitism channel ranks general elite criticism at AUROC .756, against .808 for the paper's own XLM-T classifier, which was fine-tuned on 4,342 tweets from the same collection (paired Δ −.052 [−.102, +.002]). The fine-tuned RoBERTa comparator from the V3 and V4 evaluations fails on this material (AUROC .571 with sigmoid scores, .573 under the superseded softmax; Δ +.185 for V4). At a threshold set on the training split, V4 reaches a positive-class F1 of .54, between the paper's bag-of-words (.46) and XLM-R-embedding (.56) classifiers and below XLM-T (.61). The NLI stage adds nothing on this task (+.002), and the carried PE conjunction is worse than the E channel alone (.687 against .756), because the labels ask for elite criticism without any people-centrism requirement.

## The benchmark

Source: Licht, Abou-Chadi, Barberá & Hua (2025), *Measuring and understanding parties' anti-elite strategies*, Journal of Politics (doi:10.1086/730711), replication package on Harvard Dataverse (doi:10.7910/DVN/T0UKCN, CC0). Tweets by parliamentary parties in 20 countries, 2008–2021, crowd-coded and aggregated with a Dawid-Skene model into `no`, `yes-general` (criticism of the elite in general), `yes-specific` (criticism of specific elite actors) and `yes-unsure` (dropped by the paper).

| | Train | Test |
|---|---:|---:|
| Tweets | 4,342 | 1,059 |
| `yes-general` (the paper's positive class) | 1,024 | 265 |
| `yes-general` or `yes-specific` | 1,641 | 416 |
| Countries / parties | 20 / 177 | 20 / 172 |
| Tweets from English-language countries (untranslated) | 29% | 27% |

The package ships the original-language text and an English machine translation (`text_en`) of every labelled tweet, the paper's train/test flag, and its fine-tuned XLM-T classifier. The shipped file repeats 48 rows verbatim, 4 of them in the test split; the paper's metrics were computed on the rows as shipped, so they are kept. Tweet texts stay out of git; `data/benchmark_antielite_tweets/tweets_labels.csv` holds the ids, split and labels, and `scripts/external/build_antielite_tweets_benchmark.py` rebuilds the text file from the package.

V4 reads the English translation with URLs removed and @handles replaced by `@user`, which mirrors the paper's own preprocessing for XLM-T.

## Configurations

All scored once, label-free until the metrics, with e5-large-v2 and the V3 anchors. Each tweet is one document, so the carried top-8 pooling reduces to the tweet's single window (a few tweets split into two or three sentence windows and take the maximum). Centering and z-scoring use all 5,401 tweets.

The benchmark labels elite criticism only, so the pre-declared primary V4 score is the **E channel** (projection onto the elite-obstruction axis), with **E + NLI** as the V4 + NLI variant: the mean of z-scores of the E projection and the maximum entailment of the five E anchor claims. The carried PE conjunction and PE + NLI are reported as secondary.

Comparators:

- **XLM-T, released** (`cardiffnlp/twitter-xlm-roberta-base` fine-tuned by the authors on the original-language training tweets), re-run on the same tweets. At its 0.5 threshold it reproduces the published test metrics: positive-class F1 .604 against .611 published, macro-F1 .741 against .745, accuracy .812 against .815.
- **RoBERTa** (`coastalcph/roberta-large-ft-trump-populism`), anti-elitism probability on the English translation, the comparator used throughout V3 and V4.
- **Published baselines** from the package's `best_classifier_results.tab`: bag-of-words perceptron and XLM-R-embedding perceptron.

The F1-optimal threshold is fitted on the training split only. The prevalence-matched threshold takes only the positive rate from the training labels (.236) and places the cut at that rate within the evaluated split's own scores, so it uses the unlabeled test-score distribution. An oracle threshold on the test split is shown as an upper bound.

## Results on the test split (n = 1,059, 265 positive)

Target: `yes-general`. CIs are 95% cluster-bootstrap by country (20 clusters).

| System | AUROC [95% CI] | AP | F1 pos / macro, train-prevalence cut | F1 pos / macro, train-F1-optimal cut | F1 pos / macro, oracle cut |
|---|---|---:|---|---|---|
| **XLM-T, released (supervised, in-domain)** | **.808 [.771, .842]** | **.632** | .610 / .742 | .559 / .720 | .614 / .742 |
| XLM-T, published (threshold .5) | – | – | .611 / .745 | | |
| **V4 E axis + NLI** | .756 [.714, .801] | .456 | .462 / .645 | .540 / .655 | .544 / .673 |
| V4 E axis | .754 [.714, .799] | .467 | .458 / .642 | .534 / .670 | .542 / .657 |
| V4 PE axis (carried system) | .711 [.674, .750] | .410 | .408 / .609 | .505 / .615 | .505 / .613 |
| V4 PE axis + NLI (carried V4 + NLI) | .687 [.654, .717] | .396 | .388 / .596 | .480 / .628 | .493 / .609 |
| NLI only, E claims | .670 [.632, .706] | .380 | .384 / .593 | .468 / .550 | .479 / .577 |
| V4 P axis | .621 [.589, .652] | .315 | .326 / .555 | .434 / .509 | .444 / .504 |
| RoBERTa, anti-elitism, sigmoid | .571 [.530, .604] | .347 | .352 / .571 | .383 / .411 | .385 / .430 |
| RoBERTa, anti-elitism, softmax (superseded) | .573 [.536, .603] | .352 | .346 / .568 | .386 / .432 | .388 / .417 |
| XLM-R embeddings + perceptron (paper) | – | – | .557 / .685 | | |
| Bag-of-words perceptron (paper) | – | – | .455 / .611 | | |

Paired AUROC differences on the test split (country-clustered bootstrap, 2,000 draws):

| Comparison | Δ AUROC | 95% CI | p |
|---|---:|---|---:|
| V4 E + NLI − XLM-T released | −.052 | [−.102, +.002] | .06 |
| V4 E + NLI − V4 E axis | +.002 | [−.007, +.010] | .70 |
| V4 E + NLI − V4 PE + NLI | +.068 | [+.037, +.104] | < .001 |
| V4 E + NLI − RoBERTa, sigmoid | +.185 | [+.122, +.257] | < .001 |
| V4 E + NLI − RoBERTa, softmax (superseded) | +.183 | [+.128, +.247] | < .001 |
| RoBERTa, sigmoid − XLM-T released | −.237 | [−.289, −.185] | < .001 |
| RoBERTa, softmax (superseded) − XLM-T released | −.235 | [−.281, −.189] | < .001 |

### Secondary target: any elite criticism (`yes-general` or `yes-specific`, 416 positive)

| System | AUROC | AP | F1 pos, train-prevalence cut |
|---|---:|---:|---:|
| V4 E axis + NLI | **.818** | **.756** | .691 |
| V4 E axis | .797 | .709 | .674 |
| V4 PE axis + NLI | .784 | .704 | .669 |
| XLM-T, released | .694 | .659 | .586 |
| RoBERTa, anti-elitism, sigmoid | .556 | .476 | .434 |
| RoBERTa, anti-elitism, softmax (superseded) | .558 | .475 | .428 |

### Breakdowns

- **Translation costs nothing measurable.** V4 E + NLI reaches AUROC .754 on the 774 machine-translated test tweets and .759 on the 285 untranslated English ones.
- **Country variation is wide.** V4 E + NLI ranges from .92 (Switzerland) and .91 (Greece) to .56 (France) and .62 (Italy); RoBERTa is below .65 in every country but Germany under the superseded softmax; with sigmoid scores it reaches .65 or more in Canada, France, Germany and Ireland.
- **General versus specific.** The paper's key distinction is between criticism of the elite in general and of specific elite actors. The E axis does not make it: mean z-scores are +.68 for `yes-general`, +.59 for `yes-specific` and −.38 for `no`, and the AUROC for general against specific is .52. The highest-scored negatives are all `yes-specific` tweets ("You can't trust Boris Johnson…", "The hypocrisy of the SPÖ can not be beaten…"). XLM-T learned that distinction from the training labels; on the broader target it falls to .694 while V4 rises to .818.

## Reading

- **The V4 anti-elitism channel transfers to a new genre, language mix and construct definition** with no adaptation: .756 AUROC on tweets in 20 countries, mostly machine-translated, is within .05 of a classifier trained on 4,342 tweets from the same collection and scored on the same split.
- **This is the first benchmark where the NLI stage does not help.** Tweets are short and rarely state the anchor claims in a form an entailment model accepts; the NLI signal alone is much weaker (.670) than the axis (.754), and the combination is flat. On speeches, paragraphs and rubric sentences, NLI added .02–.10.
- **The carried PE system is the wrong instrument for this label.** Requiring people-centrism as well as anti-elitism costs .07 AUROC. A benchmark defined on one component should be scored with that component's channel, as the EU-OOD protocol already does.
- **The V3/V4 RoBERTa comparator fails here** (.571 sigmoid, .573 softmax), which qualifies its role as a reference: it was fine-tuned on Trump 2016 speech sentences and does not transfer to short multilingual party tweets. On this benchmark the released XLM-T is the meaningful supervised reference.
- **What separates V4 from XLM-T is the general-versus-specific distinction**, not detection of elite criticism as such. On the broader "any elite criticism" target V4 leads by .12 AUROC.

## Caveats

- **Construct mismatch by design.** V4's E axis measures anti-elite claims of any target; the paper's positive class excludes criticism of specific actors. The primary comparison therefore understates V4 as a measure of elite criticism and overstates it as a measure of the paper's construct.
- **The supervised comparator is in-domain.** XLM-T saw 4,342 labelled tweets from these parties and years. Its AUROC on the pooled 5,401 tweets (.971) is in-sample and not reported as a result.
- **Machine translation.** 72% of the tweets reach V4 through the package's English translation; the per-source breakdown suggests no penalty, but the translations were not audited.
- **Duplicates.** 48 repeated rows, 4 in the test split, are kept to match the paper's evaluation.
- **Cluster count.** Country-clustered intervals rest on 20 clusters; party-level dependence (172 parties in the test split) is finer but not modelled.

## Provenance

| Artifact | Commit | Host | Time / VRAM |
|---|---|---|---|
| Benchmark build (`data/benchmark_antielite_tweets/`, `results/external/antielite_tweets/benchmark_freeze.json`) | `9bbfb33` | local | – |
| e5-large-v2 encode (5,812 windows) and RoBERTa comparator (9,635 windows) | `3ccfb0d` | pvgpu1 | about 1 min, 0.8 GB |
| NLI, 54,010 pairs | `3ccfb0d` | pvgpu1 | 46 s, 1.6 GB |
| XLM-T released classifier | `3ccfb0d` | pvgpu1 | about 1 min |
| Axis stage and final metrics (`final_results.json`, `final_results.md`, `final_scores.csv`) | `61f3cbf` | local | – |

- The XLM-T weights (1.1 GB, Dataverse file 7521995) were downloaded to the local machine and copied to pvgpu1 (`~/models/xlmt-elitecriticism-classifier`), because the direct download to pvgpu1 ran at about 20 KB/s; the SHA-256 of the copy was verified. This download was not pre-approved; it is a 1.1 GB file on the shared host's home directory and can be deleted.
- One launch ran on a stale checkout and failed before any GPU work; `scripts/remote_pull.sh` now prevents the untracked-file pull conflicts that caused it.
- The GPU was idle before each job and returned to idle afterwards.

```bash
uv run --extra cpu python scripts/external/build_antielite_tweets_benchmark.py
# pvgpu1: encode_windows.py --custom-csv data/external/licht_antielite/tweets_text.csv --text-column text_en_clean ...; encode_roberta_custom.py ...
uv run --extra cpu python scripts/external/score_antielite_tweets.py axis
# pvgpu1: nli_stage.py --in-dir results/external/antielite_tweets --names antielite_tweets; score_xlmt_classifier.py ...
uv run --extra cpu python scripts/external/score_antielite_tweets.py final
```
