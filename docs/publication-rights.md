# Publication rights: code, data and models

Reviewed 2026-10-02 from what the repository records (builders, development-log
source paragraphs, dataset cards and license files in `data/`). This is an
inventory for a release decision, not legal advice. Licences that the repository
does not record are marked **not recorded** and must be checked at the source
before anything is published. The project's own work is MIT-licensed. The public
repository is built from the private research archive and leaves out the data
whose terms do not allow redistribution; `data/README.md` lists what it
includes and how to obtain the rest.

Updated 2026-10-12: AgoraSpeech is CC BY 4.0 (Zenodo record, v1).

## Summary: licence and what needs a decision before a public release

1. **Project licence: MIT** (`LICENSE`, chosen 2026-10-02). It covers the
   project's own code, anchors, claim hypotheses and docs. It does not relicense
   anything under `data/` or any model; those keep their sources' terms below.
2. **EU-OOD speeches (CC BY-NC-ND 4.0).** `data/huggingface/…` is a verbatim copy
   of the dataset; `data/benchmarks/eu_sentences.csv` is a re-shaped derivative
   (sentences with context windows and project columns). NoDerivatives allows
   sharing the unmodified dataset with attribution, non-commercially; sharing an
   adapted version is the risky part. Options: keep both out of a public release
   and rebuild them from the Hub with a script, or ask the authors (Chalkidis et
   al.) for permission.
3. **Global Populism Database texts and tables.** 1,310 speech files
   (`data/speeches_20251120/`, 27 MB), the GPD v2 table and the GPD codebook PDF
   are in Git. Their licence and redistribution terms are **not recorded**. The
   speeches themselves may carry third-party copyright. Check the GPD
   distribution terms; if redistribution is not clearly allowed, release ids and
   scores only, plus a download-and-rebuild script.
4. **AgoraSpeech (Zenodo doi:10.5281/zenodo.13957177).** The full CSV
   (`data/zenodo/`, 15 MB) and the derived paragraph table are in Git. The Zenodo
   record (v1, 2024-10-24) is **CC BY 4.0**: shareable with attribution and a note
   that the paragraph table is re-shaped.
5. **PopBERT German sentences.** `data/popbert/LICENSE_PopBERT` is the MIT licence
   of the PopBERT code repository (© 2023 Lukas Erhard). Whether it also covers
   the annotated sentences is **not recorded**. The sentences are Bundestag
   plenary text, which German law generally treats as official works (inferred,
   to be confirmed). Check the dataset's own terms.
6. **Text quoted inside results.** Some committed result files quote short source
   passages (for example `results/step2_whitening/**/evidence_english.json` and
   `results/step3_pair_anchors/attribution_english_dev.json`, GPD speeches). They
   follow whatever the source terms in items 3 to 5 allow.

## Data in the repository

| Data | Where | Source and licence as recorded | Status |
|---|---|---|---|
| US governor and presidential speeches, sentences | `data/benchmark_usgov/` | van der Veen et al. (2024), Dataverse doi:10.7910/DVN/ZOHMVO, **CC0** (builder, benchmark freeze) | Clear. Cite. |
| Party tweets: ids, split, labels | `data/benchmark_antielite_tweets/tweets_labels.csv` | Licht et al. (2025), Dataverse doi:10.7910/DVN/T0UKCN, **CC0** | Clear. Tweet texts are kept out of Git (Twitter terms) and rebuilt from the package. |
| PopBERT English translations | `data/benchmark_popbert/*_en.csv` (`text_en`) | Wang (2026), Dataverse doi:10.7910/DVN/VJGS7H, **CC0** | Clear for the translations. The German sentences and labels in the same files: item 5. |
| Portuguese parliament paragraphs and Marian English translation | `data/benchmark_portugal/` | Moniz et al. (2025), Zenodo doi:10.5281/zenodo.14626861, **CC BY 4.0** | Clear with attribution and a note that the text was re-shaped and machine-translated. |
| EU-OOD sentences | `data/huggingface/coastalcph_populism_eu_speeches/`, `data/benchmarks/eu_sentences.csv` | Chalkidis et al. (2025), Hub dataset card: **CC BY-NC-ND 4.0** | Needs a decision: item 2. |
| English GPD speeches, GPD v2 table, codebook | `data/speeches_20251120/`, `data/gpd_v2_20220427.csv`, `data/GPD Codebook_v2.pdf`, `data/benchmarks/english_*.csv` | Hawkins et al., Team Populism; licence **not recorded** | Needs checking: item 3. |
| AgoraSpeech paragraphs | `data/zenodo/agoraspeech/`, `data/benchmarks/agora_paragraphs.csv` | Sermpezis et al., Zenodo, **CC BY 4.0** (record v1) | Clear with attribution: item 4. |
| PopBERT German sentences and labels | `data/benchmark_popbert/{train,test}.csv` | Erhard et al. (2025); `LICENSE_PopBERT` is the code repository's MIT licence | Needs checking: item 5. |

## Project-made assets

| Asset | Status |
|---|---|
| Source code, tests, docs, anchors (`anchors/`), claim hypotheses | Project's own work, MIT (item 1). |
| `results/frozen_reference/reference_mean_embedding.npy`, `reference_stats.json` | Summary statistics computed from English-dev GPD speeches (a 1,024-value mean vector and scales); no text can be recovered from them. Low risk, but they ship with the source terms of item 3 in mind. |
| Score tables and evaluations under `results/` | Numbers keyed by document ids. Low risk, except the quoted passages in item 6. |

## Models

None of the model weights are in Git. The code downloads them; only outputs
(embeddings, probabilities, scores) are produced here.

| Model | Role | Licence | Status |
|---|---|---|---|
| `intfloat/e5-large-v2` | Axis encoder | **MIT** on the current model card (recorded in `runtime_manifest.json`) | Clear for use; keep the notice if weights are ever redistributed. |
| `facebook/bart-large-mnli` | Claim check | **MIT** on the current model card (same record) | Clear for use; same note. |
| `coastalcph/roberta-large-ft-trump-populism` | Comparator only | **not recorded** | Used, not redistributed. Check the card before quoting it as open. |
| XLM-T anti-elite classifier | Comparator (tweets) | Ships in the Licht et al. package, **CC0** | Used, not redistributed. |
| `Helsinki-NLP/opus-mt-ROMANCE-en` | Portuguese translation | **not recorded** | Used to produce `text_en`; check the card for terms on outputs. |
| Sweep-only encoders and NLI models (Qwen3, multilingual-e5, gte, DeBERTa) | Development sweeps | **not recorded** | Not part of the released method. |
