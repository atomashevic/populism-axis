# Data

The project's own code, anchors and documentation are MIT-licensed. The benchmark data below keep their sources' terms. This file lists what the repository includes, what it leaves out, and how to obtain what is missing. The full licence review is in `docs/publication-rights.md`.

## Included

| Benchmark | Files | Source | Licence |
|---|---|---|---|
| US governor and presidential speeches, sentences | `benchmark_usgov/` | van der Veen et al. (2024), Harvard Dataverse doi:10.7910/DVN/ZOHMVO | CC0 |
| Party anti-elitism tweets: ids, splits, labels (no tweet text) | `benchmark_antielite_tweets/tweets_labels.csv` | Licht et al. (2025), Harvard Dataverse doi:10.7910/DVN/T0UKCN | CC0 |
| Portuguese parliament paragraphs, with a Marian machine translation to English | `benchmark_portugal/` | Moniz et al. (2025), Zenodo doi:10.5281/zenodo.14626861; re-shaped and translated by this project | CC BY 4.0 |
| AgoraSpeech | `zenodo/agoraspeech/AgoraSpeech.csv`, `benchmarks/agora_paragraphs.csv` | Sermpezis et al. (2024), Zenodo doi:10.5281/zenodo.13957177 (v1); paragraph table re-shaped by this project | CC BY 4.0 |
| PopBERT sentences: ids, labels and English translations | `benchmark_popbert/{train,test}_en.csv` | Labels: Erhard et al. (2025); translations: Wang (2026), Harvard Dataverse doi:10.7910/DVN/VJGS7H | Translations CC0; see below for the German sentences |
| English Global Populism Database speeches: ids, metadata, rubric scores (no text) | `benchmarks/english_gpd_speeches.csv`, `benchmarks/english_leader_terms.csv`, `benchmarks/english_split.csv` | Hawkins et al., Team Populism, Global Populism Database v2 (2022); split by this project | Scores from the GPD; no text |

`benchmarks/freeze.json` records the SHA-256 and row counts of the frozen benchmark tables.

## Not included, and how to obtain it

- **Global Populism Database speech texts and codebook.** The 258 English speeches used here (`speeches_20251120/<merging_variable>`) and the GPD v2 table come from Team Populism (https://populism.byu.edu). Their redistribution terms are not established, so they are not in this repository. Place the speech files under `data/speeches_20251120/` to re-encode the English benchmark.
- **EU-OOD sentences** (Chalkidis et al. 2025; `coastalcph/populism-eu-speeches` on the Hugging Face Hub) are CC BY-NC-ND 4.0. Rebuild the project's table with `uv run --extra cpu python scripts/build_eu_sentences.py`, which downloads the dataset and checks the result against the frozen hash.
- **PopBERT German sentences.** The licence of the annotated sentences (Erhard et al. 2025) is not established, so the German text column is removed from the `_en` tables. The scorer uses only the English translation (`text_en`).
- **Tweet texts** are not redistributed under the platform's terms. `scripts/external/build_antielite_tweets_benchmark.py` rebuilds them from the Licht et al. replication package.

The committed score tables under `results/` are keyed by these ids, so every reported number can be recomputed without the texts (`scripts/reproduce_table.py`); the EU-OOD rows need the rebuilt EU table.
