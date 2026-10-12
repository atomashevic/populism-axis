# Reproducing the results

There are two levels. The first recomputes every reported number from the per-item scores in this repository, on a CPU, in about two minutes. The second rebuilds those scores from the texts and needs a GPU and the source corpora.

## 1. Recompute the reported numbers from the stored scores (CPU)

```bash
uv sync --extra cpu
uv run --extra cpu python scripts/build_eu_sentences.py   # EU-OOD table; downloads the dataset from the Hugging Face Hub
uv run --extra cpu python scripts/reproduce_table.py
```

`reproduce_table.py` reads the score tables under `results/` and the gold labels under `data/`. For each cell of the README benchmark table, it recomputes the metric and compares it with the reported value. It also recomputes the paired bootstrap intervals with the seeds and draw counts of the scripts that produced them. It loads no model and exits with status 1 if any recomputed cell differs from the reported value.

The EU-OOD sentences are licensed CC BY-NC-ND and are not in this repository (`data/README.md`). Without `build_eu_sentences.py`, the two EU-OOD rows are reported as skipped.

## 2. Rebuild the scores from the texts (GPU)

The pipeline runs in five steps:

1. Split each text into windows.
2. Encode the windows with `intfloat/e5-large-v2` (GPU).
3. Project the window embeddings on the anchor axes (CPU).
4. Score the top windows against the claims with `facebook/bart-large-mnli` (GPU).
5. Combine the scores and evaluate them (CPU).

The README section "Reproducing the benchmarks" gives the commands for the development sets. Each development-log report under `research/development-log/` gives the exact commands, commits and run times for its stage or benchmark. The two models are loaded at the commits pinned in `src/populism_axis/model_revisions.py`.

This level needs the texts:

- **Global Populism Database speeches.** Obtain them from Team Populism and place them under `data/speeches_20251120/`.
- **EU-OOD.** Run `scripts/build_eu_sentences.py`.
- **Tweets.** Run `scripts/external/build_antielite_tweets_benchmark.py` with the Licht et al. replication package.
- **Other corpora.** The remaining tables are in `data/`.

The scoring scripts write into `results/`. Run them in a separate clone and compare the output with the committed tables.

## Limits

- **Model revisions.** The historical runs recorded model ids but not revisions. That they used the pinned revisions is inferred (`docs/model-revisions.md`). A fresh checkout with a cold model download did reproduce the stored English-dev scores (`docs/fresh-inference-check.md`).
- **Uncommitted changes.** The final holdout and EU-OOD runs were made with uncommitted changes in the working tree. They are documented in `results/final/PROVENANCE.md`.
- **Spent test sets.** The English holdout and EU-OOD are spent: no configuration choice may be made on them (`docs/development-protocol.md`).
