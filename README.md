# populism-axis

A training-free measure of populist rhetoric. Texts are placed on a **populism axis** built from written anchor propositions in a pretrained sentence-embedding space, and the passages that score highest are passed to a **claim check**, in which an entailment model scores their support for the written populist claims. No model is trained on populism labels; the construct is twenty sentences per dimension that anyone can read and edit.

- Method: `docs/method.md`
- Score your own texts: the `populism-axis` command (below)
- Define your own scale: [Python and CSV tutorial](docs/custom-scales.md), with a reusable reference and explicit threshold labels
- R interface: [installation and custom-scale tutorial](r/populismaxis/README.md), with automatic uv backend setup
- Every number's provenance: `research/development-log/`
- Development protocol: `docs/development-protocol.md`
- Reproduce every reported number: [docs/reproducing.md](docs/reproducing.md); data sources and licences: [data/README.md](data/README.md)

## The measure in one paragraph

Two dimensions define the construct: people-centrism, written as popular-sovereignty claims (P), and anti-elitism, written as elite-obstruction claims (E), each as ten populist and ten mainstream anchor propositions (`anchors/proposition_anchors.json`). A text is split into three-sentence windows, each window is encoded with `intfloat/e5-large-v2`, centred and projected on the P and E axes, and the two projections are combined as a soft conjunction. The document's axis score is the mean of its top 8 windows (top window for sentences and tweets). Those same windows are then scored by `facebook/bart-large-mnli` for entailment of ten written claims (`anchors/claim_hypotheses.json`), and the final score is the mean of the z-scored axis and claim-check scores. Every score decomposes into named windows, their P and E projections and the best-matching claims.

## Benchmarks

All rows use the same anchors and the same frozen configuration, selected on English-dev and AgoraSpeech under `docs/development-protocol.md`; the Agora row is therefore a development-set result. The English holdout and EU-OOD were scored once at the end of the stepwise development, but neither was blind to the whole project: the holdout was visible when the initial configuration (including the retained anchors and encoder) was built, and EU-OOD was scored during that earlier phase. The four external benchmarks were each scored once and had no part in any decision. Comparators are the strongest published or replicated system on each benchmark. Bold marks the best score in the row. RoBERTa scores are the checkpoint's independent per-label sigmoid outputs; an earlier softmax implementation was corrected on 2026-10-02 (`docs/roberta-correction.md`, `results/roberta-correction/rc-20261002T1415Z-sigmoid/REPORT.md`).

| Benchmark | Unit, n, metric | Populism axis | Axis + claim check | Fine-tuned RoBERTa-large | Strongest other comparator |
|---|---|---:|---:|---:|---|
| Global Populism Database, English holdout | 84 speeches, Spearman vs rubric | .417 | **.518** | .440 | human coder pair .748 |
| EU-OOD sentences (Chalkidis et al. 2025) | 1,597 of 1,762 sentences, AUROC 3-vs-0 | .890 | **.942** | .900 | |
| EU-OOD sentences, paper protocol | macro-F1, leave-one-speaker-out | .533 | .545 | .555 | Qwen3-14B zero-shot **.613**, Gemini 2.5 Flash .590 |
| AgoraSpeech paragraphs (development set) | 5,272 paragraphs, Spearman vs human score | .359 | **.382** | .372 | |
| US governors 2010–18 (van der Veen et al. 2024) | 288 speeches, Spearman vs rubric | .458 | **.526** | .326 | SetFit, supervised in-domain .514 |
| Party anti-elitism tweets (Licht et al. 2025), E dimension | 1,059 tweets, AUROC | .754 | .756 | .571 | XLM-T, supervised in-domain **.808** |
| PopBERT Bundestag sentences, anti-elitism, E dimension | 1,759 sentences, AUROC | .781 | **.806** | .669 | |
| PopBERT Bundestag sentences, people-centrism, P dimension | 1,759 sentences, AUROC | .754 | **.771** | .690 | |
| Portuguese parliament (Moniz et al. 2025), anti-elitism, E dimension | 6,214 paragraphs, AUROC | .770 | .798 | **.867** | supervised LR on same embeddings .823 |
| Portuguese parliament, people-centrism, P dimension | 6,214 paragraphs, AUROC | .668 | **.695** | .582 | supervised LR on same embeddings .798 |

What the table shows:

- **The claim check improves the axis on every benchmark** except tweets, where it is neutral. On the English holdout the paired gain over the axis alone is +.10 (95% CI +.01 to +.24).
- **It scores higher than the fine-tuned RoBERTa on eight of ten rows; five of those leads have paired 95% intervals that exclude zero.** These are US governors, tweets, both PopBERT rows and Portuguese people-centrism. The other three are numeric leads only: the English holdout (+.08, CI −.08 to +.25), EU-OOD AUROC (no paired test) and Agora, a development set. RoBERTa was fine-tuned on Trump 2016 campaign sentences and loses about a quarter of its accuracy out of domain; the measure here is not trained on populism labels and is applied unchanged to speeches, paragraphs, sentences and tweets in English and, through English translation, German, Portuguese and other languages. The exceptions are Portuguese anti-elitism (−.07, CI −.12 to −.02), where the coders' construct (institutional capture) matches RoBERTa's training material, and EU-OOD macro-F1 under the paper protocol (.555 against .545; no interval is available for this row).
- **Against supervised in-domain systems** it ties SetFit on ranking US governors, trails XLM-T on tweets by .05 (paired Δ −.052, 95% interval −.102 to +.002), and trails the in-domain PopBERT and Portuguese references. Against zero-shot LLMs, the only measured comparison, it trails on EU-OOD by .05 to .07.
- **The axis alone is the interpretable core**; the claim check adds an entailment score for the written claims. Cheaper entailment models do not replicate it (`research/development-log/claim-check-replication.md`).

PopBERT and Portuguese texts were read in English translation (GPT-4o and Marian); reading the Portuguese original with a multilingual encoder and the same anchors reaches .838 / .758. The EU-OOD and PopBERT rows for the axis and the claim check are per-dimension scores, as each benchmark's protocol requires. Full tables, confidence intervals and paired tests are in the development log.

## Score your own texts

```bash
uv sync --extra cpu
uv run --extra cpu populism-axis verify-assets
uv run --extra cpu populism-axis score --input my_texts.csv --text-column text --id-column id \
    --unit speech --reference frozen --out-dir scored/my_run
```

- `--unit speech` pools the top 8 windows (speeches, paragraphs); `--unit sentence` takes the top window (sentences, tweets).
- `--reference frozen` standardises with the stored English development-set statistics and works for any number of texts, including one. `--reference corpus` standardises on the input itself, as the benchmark evaluations do, and needs a corpus of at least 30 texts.
- `--clean` applies the speech-file cleaner (metadata headers, URL lines). `--no-claim-check` gives the axis alone. `--device cuda` uses a GPU.
- Outputs: `scores.csv` (axis, claim check, combined score, and per-dimension columns), `evidence.json` (each text's top windows with P and E projections and the best-matching claims with their entailment probabilities), `run_manifest.json`.
- The research script `scripts/score_texts.py` takes the same options; the fresh-inference check below used it.

`verify-assets` (or `python3 scripts/verify_frozen_assets.py`, which needs only the Python standard library) checks the bundled reference files and scorer source; it does not check installed dependencies, download models or run inference. A clean checkout with a cold model download reproduced the stored English-dev scores, 7 of 7 checks passed (`docs/fresh-inference-check.md`). See [the frozen-scorer quickstart](docs/frozen-scorer.md) for offline cache verification, exact observed checkpoint hashes and release limitations. To define a scale of your own, see the [custom-scale tutorial](docs/custom-scales.md).

The scorer loads cached models by default. Add `--allow-download` to permit downloads from the Hugging Face Hub at the exact commits pinned in `src/populism_axis/model_revisions.py`. The two models total about 3 GB. On a laptop CPU expect one to two minutes per speech and a few seconds per sentence; on a GPU a few seconds per speech.

## Reproducing the benchmarks

To recompute every number in the table from the stored per-item scores on a CPU, run `scripts/reproduce_table.py` ([docs/reproducing.md](docs/reproducing.md)). Rebuilding the scores from the texts is described below. Encoding and the claim check need a CUDA GPU (`uv sync --extra cu126`); every other step runs on a CPU. A fresh clone contains the redistributable benchmark tables under `data/` (`data/README.md`) and the compact results under `results/`, but not the window embeddings or entailment probabilities, which the first steps below rebuild. The four external benchmarks also need their source corpora, which each builder under `scripts/external/` downloads or documents.

```bash
# GPU: encode windows and anchors
uv run --extra cu126 python scripts/encode_windows.py --encoder e5-large-v2 --sentences-per-window 3 --max-words 60 --max-length 256 --clean-text
uv run --extra cu126 python scripts/encode_sentences.py --sentences anchors/proposition_anchors.json --set-id v3_proposition
# CPU: axis scores and claim-check requests
uv run --extra cpu python scripts/run_config.py --label axis --embeddings-dir results/embeddings/e5-large-v2/spw3_cap60_clean --anchors results/embeddings/e5-large-v2/sentences/v3_proposition__ea98e5b2.npz --top-rule k:8 --out-dir results/axis
uv run --extra cpu python scripts/prepare_nli_requests.py
# GPU: entailment
uv run --extra cu126 python scripts/nli_stage.py --names english_dev,agora
# CPU: evaluation
uv run --extra cpu python scripts/evaluate_nli.py
```

The development runs used `scripts/remote_job.sh`, a wrapper that runs the same commands on the project's shared GPU host with a busy-GPU guard and a log file. The final holdout scoring (`scripts/final_holdout.py --final`) and the four external benchmarks (`scripts/external/`) each have a report in the development log with the exact commands, commits and run times. The holdout and EU-OOD are spent: no further configuration choice may be made on them.

## Repository map

| Path | Role |
|---|---|
| `anchors/` | The written construct: proposition anchors, claim hypotheses, and the matched-pair and nuisance sentence sets tested during development. |
| `src/populism_axis/` | Axis construction and scoring (`contextual_axis.py`, `geometry.py`), encoders, text cleaning, the country-level split, run manifests. |
| `scripts/` | `score_texts.py` for new texts; encoding, configuration, evaluation and sweep scripts; `external/` builders and scorers for the four external benchmarks. |
| `data/benchmarks/` | Frozen English GPD, EU-OOD and AgoraSpeech tables and the development/holdout split. `data/benchmark_*/` hold the external benchmarks; raw sources are documented in each builder. |
| `results/` | Compact evidence per development stage and per benchmark: scores, evaluations, manifests. Embeddings and entailment probabilities are rebuildable caches and not tracked. |
| `research/development-log/` | The stage reports, external validations and follow-up analyses that produced every number above, kept as written. |
| `docs/` | Method, development protocol, environments, results index. |
| `tests/` | Unit tests for the axis, geometry, windowing, cleaning and split. |

## Data

The English speeches come from the Global Populism Database (Hawkins et al.), the EU-OOD sentences from Chalkidis et al. (2025, CC BY-NC-ND 4.0), the AgoraSpeech paragraphs from Sermpezis et al. (2024, CC BY 4.0), the US governor speeches from van der Veen et al. (2024, CC0), the party tweets from Licht et al. (2025, CC0), the PopBERT sentences from Erhard et al. (2025) with GPT-4o translations from Wang et al. (2026, CC0), and the Portuguese paragraphs from Moniz et al. (2025, CC BY 4.0). The GPD speech texts, the EU-OOD sentences, the PopBERT German sentences and the tweet texts are not redistributed here; [data/README.md](data/README.md) lists what is included and how to obtain the rest. Each development-log report cites the source and licence.

## License

The project's code, anchors, claim hypotheses and documentation are released under the [MIT License](LICENSE). MIT is the common choice for a research Python package and is accepted on CRAN for the R package (`License: MIT + file LICENSE`). Third-party data under `data/` keep their sources' terms, and the models keep their own licences; see [docs/publication-rights.md](docs/publication-rights.md).

## Development

```bash
uv sync --extra cpu
uv run --extra cpu python -m unittest discover -s tests
```

`AGENTS.md` records the frozen configuration, which data are spent, and the verification steps.
