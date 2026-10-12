# Refresh of the stored RoBERTa entries (softmax → sigmoid)

Run `rc-20261002T1600Z-refresh-result-files`, 2026-10-02. This run is comparator-only. It brings the stored result tables in `results/final/` and `results/external/*/` in line with the corrected comparator of `rc-20261002T1415Z-sigmoid` (eight inputs) and `rc-20261002T1530Z-sigmoid-usgov-sentences` (the sentence diagnostic). Axis, claim-check, NLI, SetFit, XLM-T and supervised-reference entries are unchanged. No GPU work and no axis scoring ran.

## Method

`refresh_result_files.py` executes the committed final-stage code unchanged: `main()` of `scripts/final_holdout.py`, and the `final` stage of `scripts/external/score_{usgov,antielite_tweets,popbert_en,portugal}.py`. It runs that code in a temporary sandbox that holds copies of each stage's stored inputs: axis scores, NLI caches and requests, and the XLM-T outputs. Four redirections keep the run comparator-only:

- `final_holdout.run_axis` returns the stored B0/B4 run directories instead of starting an axis run.
- `log_access` is disabled. The run reads the spent holdout labels for the comparator only, as `compare_roberta.py` did, and makes no configuration choice.
- `load_scores` is the only input that changes between the two passes.
- Sandbox manifests are discarded.

Each stage runs twice:

1. **Replay.** The legacy `probabilities_*.npz` softmax caches are handed to the unchanged code. Every one of the 16 output files must equal the stored file byte for byte. All 16 did.
2. **Sigmoid.** The corrected caches go through the strict `load_scores` contract. Every value that differs from the stored file must sit under a JSON key, CSV column or Markdown row that names RoBERTa. The JSON structure, CSV layout and table layout must be unchanged.

The staged files take the sigmoid values in place and keep the old ones:

- **JSON:** a top-level `superseded_softmax_roberta` object maps the path of each replaced subtree (for example `/configurations/RoBERTa_comparator`) to its old value. A `roberta_correction` note names the runs.
- **CSV:** each changed column is followed, at the end of the row, by `<column>__superseded_softmax`.
- **Markdown:** a row marked `(superseded softmax)` follows each corrected row, under a one-line note.

The build ran from the clean commit `05f976e` (`refresh.manifest.json`: `dirty: false`; 2026-10-02 17:23 UTC). `--apply` then moved the staged files into place only after checking that each target still had the sha256 the build verified (`refresh.manifest.json`, `inputs.stored::*`) and that each staged file matched `extra.outputs_sha256`.

## Files changed

| File | Replay identical | Changed |
|---|---|---|
| `results/final/final_results.json` | yes | 25 values |
| `results/final/eu_paper_protocol.json` | yes | 17 values |
| `results/final/final_results.md` | yes | 3 rows |
| `results/external/usgov/final_results.json` | yes | 98 values |
| `results/external/usgov/final_results.md` | yes | 15 rows |
| `results/external/usgov/final_scores_speeches.csv` | yes | `RoBERTa_anti_elitism_top3`, `RoBERTa_populist_total_max` |
| `results/external/usgov/final_scores_sentences.csv` | yes | `RoBERTa_populist_total_max` |
| `results/external/antielite_tweets/final_results.json` | yes | 149 values |
| `results/external/antielite_tweets/final_results.md` | yes | 11 rows |
| `results/external/antielite_tweets/final_scores.csv` | yes | `RoBERTa_anti_elitism`, `RoBERTa_populist_total` |
| `results/external/popbert_en/final_results.json` | yes | 117 values |
| `results/external/popbert_en/final_results.md` | yes | 10 rows |
| `results/external/popbert_en/final_scores.csv` | yes | `E_roberta`, `P_roberta`, `roberta_populist_total` |
| `results/external/portugal/final_results.json` | yes | 40 values |
| `results/external/portugal/final_results.md` | yes | 6 rows |
| `results/external/portugal/final_scores.csv` | yes | `E_roberta`, `P_roberta`, `PE_roberta_total` |

Per-file detail, including the list of superseded subtrees: `verification.json`.

## Consistency with the comparison run

`check_consistency.py` reads the applied files and compares them with `rc-20261002T1415Z-sigmoid`. The README rows, their paired intervals and the English/EU secondary cells match `comparison.json` exactly. The development-log secondary cells for US governors, tweets, PopBERT and Portugal match `secondary_tables.json` to its six decimal places. Both checks cover the sigmoid values in place and the softmax values under `superseded_softmax_roberta`. All 102 checks are equal (`consistency.json`). The Agora row has no entry in these files; it exists only in `comparison.json`.

## Not changed

- The historical `probabilities_*.npz` caches.
- The external `final.manifest.json`, `axis.manifest.json` and `build.manifest.json` files, which document the original axis and NLI runs. This run's own manifest is `refresh.manifest.json`.
- `benchmark_freeze.json` files, whose RoBERTa mentions are published reference numbers from the source papers.
- `results/final/eu_paper_protocol.json` → `paper_reference`, the paper's own RoBERTa F1 values.

## Note for re-running `compare_roberta.py`

`rc-20261002T1415Z-sigmoid/compare_roberta.py` checks its softmax column against the stored files. After this refresh, those files hold the sigmoid values in place, so that check now compares against the corrected numbers. To reproduce its `reproduction_checks` as recorded, run it at `bd58ecc`.
