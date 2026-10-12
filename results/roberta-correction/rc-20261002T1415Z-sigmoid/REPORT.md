# RoBERTa comparator correction: softmax → independent sigmoid

Run `rc-20261002T1415Z-sigmoid`, 2026-10-02. Comparator-only recheck. The axis, claim check, datasets, splits, pooling rules, bootstrap settings and seeds are unchanged; axis and claim-check scores were read from their stored result files.

## Verdict

**"Nine of ten" does not hold after the correction. The corrected count is eight of ten.** The row that changes is EU-OOD leave-one-speaker-out macro-F1: corrected RoBERTa scores .555 against .545 for axis + claim check (softmax RoBERTa had .512). Portuguese anti-elitism remains a RoBERTa win (.867 against .798). The other eight rows still favour axis + claim check, and every paired interval that excluded zero before still excludes zero.

## Ten rows

| Row | Axis + claim check | RoBERTa softmax (old) [95% CI] | RoBERTa sigmoid (corrected) [95% CI] | Corrected − old | Ours better: old / corrected |
|---|---:|---:|---:|---:|---|
| GPD English holdout, Spearman (84) | .518 | .464 [.238, .670] | .440 [.142, .678] | −.024 | yes / yes |
| EU-OOD AUROC 3-vs-0 | .942 | .902 [.625, .978] | .900 [.729, .978] | −.002 | yes / yes |
| EU-OOD LOSO macro-F1 | .545 | .512 | .555 | +.044 | yes / **no** |
| Agora paragraphs, Spearman | .382 | .367 [.279, .387] | .372 [.287, .390] | +.004 | yes / yes |
| US governors 2010–18, Spearman (288) | .526 | .376 [.252, .490] | .326 [.197, .447] | −.050 | yes / yes |
| Anti-elite tweets AUROC, E (1,059) | .756 | .573 [.536, .603] | .571 [.530, .604] | −.002 | yes / yes |
| PopBERT anti-elitism AUROC, E | .806 | .650 [.613, .686] | .669 [.632, .704] | +.019 | yes / yes |
| PopBERT people-centrism AUROC, P | .771 | .643 [.566, .713] | .690 [.617, .757] | +.047 | yes / yes |
| Portugal anti-elitism AUROC, E | .798 | .868 [.829, .912] | .867 [.818, .918] | −.000 | no / no |
| Portugal people-centrism AUROC, P | .695 | .604 [.560, .642] | .582 [.523, .635] | −.021 | yes / yes |

Rows favouring axis + claim check: **softmax 9/10, sigmoid 8/10.** The CIs are the existing single-system intervals: leader-, speaker-, country- or actor-clustered where the original scripts cluster, and sentence bootstrap for PopBERT. EU LOSO macro-F1 never had an interval.

## Existing paired comparisons (ours − RoBERTa)

| Comparison | Old Δ [95% CI] | Corrected Δ [95% CI] |
|---|---|---|
| English holdout: axis + claim check − RoBERTa | +.055 [−.094, +.205] | +.079 [−.080, +.251] |
| English holdout: axis − RoBERTa | −.047 [−.203, +.081] | −.023 [−.175, +.130] |
| US governors 2010–18: axis + claim check − RoBERTa | +.150 [+.045, +.255] | +.200 [+.086, +.318] |
| Tweets: E axis + claim check − RoBERTa | +.183 [+.128, +.247] | +.185 [+.122, +.257] |
| Tweets: E axis − RoBERTa | +.181 [+.127, +.246] | +.183 [+.122, +.253] |
| PopBERT anti-elitism: E axis + claim check − RoBERTa | +.156 [+.114, +.199] | +.137 [+.095, +.181] |
| PopBERT people-centrism: P axis + claim check − RoBERTa | +.128 [+.052, +.206] | +.081 [+.005, +.151] |
| Portugal anti-elitism: E axis + claim check − RoBERTa | −.069 [−.110, −.012] | −.069 [−.116, −.016] |
| Portugal people-centrism: P axis + claim check − RoBERTa | +.092 [+.045, +.126] | +.113 [+.064, +.143] |

EU-OOD and Agora had no paired test against RoBERTa. As a supplementary check that was not previously reported, a politician-clustered paired Δ on Agora is +.015 [−.034, +.076] for softmax and +.011 [−.036, +.067] for sigmoid. Neither excludes zero.

## Verification

- **Code.** Executed code is base `bada4a8` plus the uncommitted working-tree patch. The tree is dirty, and `git diff HEAD --binary` has sha256 `1843c572…a392`, identical before and after the run; that tree is committed as `48ee5bd`. The encoder (`scripts/external/encode_roberta_custom.py`) writes raw logits and independent sigmoid scores and requires pinned revisions. Readers use the strict `load_scores` contract, and every corrected cache passed it.
- **Inputs.** There are 8 inputs with 162,201 windows. Every ordered-window sha256 and doc_index sequence matches the prepared plan and the legacy caches. The legacy total of 166,560 includes the US-governor sentence diagnostic (4,359 windows), which is out of scope.
- **Checkpoint.** The model is `coastalcph/roberta-large-ft-trump-populism@9b5badfa…` (safetensors sha256 = Hub LFS id `f5ed0ae3…`). The tokenizer is `FacebookAI/roberta-large@722cf37b…`. The Hub history shows a single weight upload. The numerical check is stronger: softmax of the new logits reproduces every legacy `probabilities_*.npz` to max |Δ| 1.2e-7. The rerun therefore matches the historical model, windows and tokenization, and differs only in the activation.
- **Old numbers.** Recomputing the old column through the same code path reproduces every stored RoBERTa metric, CI and paired Δ exactly (`reproduction_checks` in `comparison.json`).
- **Our scores.** The axis + claim check values are the stored ones and were not recomputed.

## Unresolved issues

1. **EU LOSO macro-F1 has no uncertainty estimate, and the margin is small (.555 against .545).** The per-class breakdown differs: corrected RoBERTa has AE .332 / PC .410 / N .924, while axis + claim check has AE .285 / PC .455 / N .896. RoBERTa's thresholds are fitted with the existing LOSO best-F1 procedure. As a supplementary check, the paper's native zero-logit rule (sigmoid > .5) gives corrected RoBERTa .550, which still exceeds .545.
2. **The Agora row is a development-set comparison.** Its RoBERTa harness existed only in git history (`scripts/evaluate_roberta_comparator_v3.py` at `d22cbb6^`). I reproduced it here with its historical seed 20260727.
3. **The remote job log header is wrong about the tree.** It reads `dirty=no` because the executed snapshot is not a git checkout. `provenance/provenance.json` records the true dirty state.
4. **Release status (resolved).** The encoder ran from a dirty tree; commit `48ee5bd` is byte-identical to the executed snapshot (verified against `provenance/snapshot_files.sha256`; the snapshot omitted 12 tracked speech files with non-ASCII names, none of them encoder inputs). The CPU comparison and secondary tables were re-run from the clean commit `bd58ecc` and reproduced every output byte for byte; `compare.manifest.json` records `dirty: false`.
5. **Two groups of RoBERTa figures are still softmax.** The secondary cells in the development-log tables were recomputed afterwards (`secondary_tables.json`, plus English extreme AUROC/AP and leader-term ρ and EU per-class F1/AP in summary.md). Two groups were not:
   - the US-governor sentence diagnostics, which were not re-encoded;
   - the development-era rows in `00-initial-axis-method-and-results.md`, which now carry a correction note instead.

## Proposed changes (applied 2026-10-02 after approval; committed in `bd58ecc`)

Applied to README.md, abstract.md, docs/theory.md (8.4), docs/roberta-correction.md, research/development-log/summary.md, the four external-* pages and a note in 00-initial-axis-method-and-results.md. In the development log the softmax rows are kept and labelled superseded, next to the corrected sigmoid rows from `secondary_tables.json` (computed by `secondary_tables.py` with the scripts' own evaluation functions; its softmax column reproduces every stored cell). Stored result files under `results/final/` and `results/external/` were not regenerated.

- **README "What the table shows".** Replace "It beats the fine-tuned RoBERTa on nine of ten rows" with "It beats the fine-tuned RoBERTa on eight of ten rows". The exceptions become Portuguese anti-elitism and the EU-OOD paper-protocol macro-F1 (.555 against .545, no interval).
- **README benchmark table, RoBERTa column.** Change to .440, .900, .555 (bold stays on Qwen3-14B .613 in that row), .372, .326, .571, .669, .690, .867, .582. Add a note that the comparator uses the checkpoint's independent sigmoid outputs, and that an earlier softmax implementation was corrected on 2026-10-02.
- **Development log.** Update the RoBERTa numbers and paired intervals in `external-usgov.md` (.376 → .326; Δ +.150 [+.045, +.255] → +.200 [+.086, +.318]), `external-antielite-tweets.md` (.573 → .571; Δ +.183 → +.185), `external-popbert-en.md` (.650/.643 → .669/.690) and `external-portugal.md` (.868 → .867; Δ −.069 [−.116, −.016]; P .604 → .582). Update `results/final/final_results.md` and the stage/summary pages wherever the English holdout (.464 → .440, Δ +.079 [−.080, +.251]) or EU numbers appear.
- **`abstract.md` (untracked draft).**
  - Both abstract versions (lines 5 and 9): change "nine of ten comparisons" to "eight of ten".
  - Line 5: change RoBERTa ".46" to ".44" and ".38" to ".33". German ".65" becomes ".67", and EU ".90" is unchanged.
  - Line 15: the losses are now Portuguese anti-elitism and EU-OOD paper-protocol macro-F1.
  - Lines 22–27 evidence table: .464 → .440, .376 → .326, .902 → .900, .650 → .669, and the "Nine of ten" row becomes "Eight of ten" with both losses listed.
  - `docs/results-index.md` has no comparator figures.
- **Result files.** Regenerate the RoBERTa entries in `results/final/` and `results/external/*/` only through a reviewed comparator-only path from a clean commit, keeping the legacy softmax files.

## Output paths

All paths are under `results/roberta-correction/rc-20261002T1415Z-sigmoid/`:

- `sigmoid_<name>.npz` ×8: raw logits, sigmoid probabilities, doc_index and metadata. `roberta_sigmoid_<name>.manifest.json` ×8 are the encoder manifests.
- `compare_roberta.py`, `comparison.json`, `comparison.md`, `compare.manifest.json`: the comparison.
- `secondary_tables.py`, `secondary_tables.json`: corrected secondary RoBERTa cells for the development-log tables.
- `provenance/`: `provenance.json`, `uncommitted.patch`, `untracked_files.txt`, `git_status.txt`, `snapshot_files.sha256`, `rerun-inputs-recheck.json`, `run_encode.sh` and `roberta_sigmoid.log`.
- Remote copy of the executed snapshot and outputs: `pvgpu2:~/populism-axis-roberta-correction/rc-20261002T1415Z-sigmoid/`.
- Historical files are untouched: `results/final/roberta/probabilities_*.npz`, `results/external/*/roberta/probabilities_*.npz` and all stored results.
