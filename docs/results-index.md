# Results index

What is under `results/`, where it came from, and which development-log report reads it. Embeddings, projections, entailment probabilities and window text are rebuildable caches and are not tracked. The small `results/frozen_reference/reference_mean_embedding.npy` is instead a required frozen-scorer runtime asset. Its existing bytes, `reference_stats.json`, anchors and scorer source are covered by `runtime_manifest.json` and `SHA256SUMS`; verify with `python3 scripts/verify_frozen_assets.py`. See `docs/frozen-scorer.md`. `scripts/reproduce_table.py` recomputes the README benchmark table from the score tables listed here (`docs/reproducing.md`).

## Development stages (development data only: English-dev and Agora)

| Directory | Stage | Report |
|---|---|---|
| `results/step0_env_holdout/` | Holdout freeze, country-level split, null baselines | `research/development-log/stage-0-environment-holdout.md` |
| `results/step1_splitter/` | Sentence-splitter cap, re-baseline of the initial configuration | `stage-1-splitter.md` |
| `results/step1b_text_cleaning/` | Speech-file cleaning audit | `stage-1b-text-cleaning.md` |
| `results/step2_whitening/` | Label-free whitening and nuisance residualisation grid (rejected) | `stage-2-whitening-residualization.md` |
| `results/step3_pair_anchors/` | Matched-pair anchors (rejected) | `stage-3-minimal-pairs.md` |
| `results/step4_windows_topk/` | Window size and pooling depth grid; carried cell `grid/spw3_k8` | `stage-4-windows-topk.md` |
| `results/step5_encoders/` | Encoder comparison (e5-large-v2 carried) | `stage-5-encoders.md` |
| `results/step6_nli/` | Claim check: hypotheses, requests, evaluation rows; carried configuration | `stage-6-claim-check.md` |

Each stage directory holds a `carried_config.json` when it changed the carried configuration, `evaluation.json` and `scores_*.csv` per row, `paired_vs_previous.json`, and run manifests.

## Held-out and external evaluations (each scored once)

| Directory | Content | Report |
|---|---|---|
| `results/final/` | English holdout and EU-OOD: axis, axis + claim check, RoBERTa comparator; `final_results.json`, `eu_paper_protocol.json`; per-item RoBERTa scores (`roberta/scores_*.csv`) and the compact AgoraSpeech table (`scores_agora.csv`), exported by `scripts/export_comparator_scores.py` | `summary.md` |
| `results/holdout_access_log.json` | Every access to held-out data | `stage-0-environment-holdout.md` |
| `results/external/usgov/` | US governors and 2016 candidates | `external-usgov.md` |
| `results/external/antielite_tweets/` | Party anti-elitism tweets | `external-antielite-tweets.md` |
| `results/external/popbert_en/` | PopBERT in English translation | `external-popbert-en.md` |
| `results/external/portugal/` | Portuguese parliament paragraphs | `external-portugal.md` |
| `results/popbert/` | Earlier multilingual-encoder reading of the German PopBERT sentences, used as an unpaired reference | `external-popbert-en.md` |

## Follow-up analyses over stored scores

| Directory | Content | Report |
|---|---|---|
| `results/fusion_weight_sweep/` | Axis / claim-check weight sweep on every corpus | `fusion-weight-sweep.md` |
| `results/frozen_reference/` | English-dev reference statistics (`reference_stats.json`, `reference_mean_embedding.npy`) and re-scoring under frozen scales; used by `scripts/score_texts.py --reference frozen` | `frozen-reference-scales.md` |
| `results/nli_replication/` | Smaller entailment models against BART-large-MNLI on the development sets | `claim-check-replication.md` |

## Reading the development log

The reports are kept as they were written during development and refer to the configurations by their working labels: the initial configuration (the axis before the stepwise changes) and the stepwise configurations B0 to B6, where B4 is the populism axis and B6 the populism axis with the claim check. `00-initial-axis-method-and-results.md` describes the initial axis and its benchmark results; the result files it cites for its own comparators were not carried into this tree.
