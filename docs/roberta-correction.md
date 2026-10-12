# RoBERTa comparator activation correction

The released `coastalcph/roberta-large-ft-trump-populism` checkpoint uses three independent labels trained with BCE: no populism, anti-elitism, people-centrism. The [authors’ training code](https://github.com/coastalcph/populism-llms/blob/8b45c61738bafbc5c20fd07b812d433c65a19fd9/finetune_models/finetune_plms.py) assigns annotation 3 to both positive labels and evaluates independent logits at zero. Historical local encoders applied softmax across the three logits. Their retained `probabilities_*.npz` files cannot recover the independent sigmoid scores because they discard the per-window logit offset.

The corrected encoder retains float32 raw logits and computes sigmoid separately for each label. It checks the checkpoint problem type and exact label order. Windows remain one sentence, uncapped, and truncated at 128 tokens by default, with the existing tokenizer, translations and CSV order. GPU half precision, batch size 64 and all pooling/evaluation rules are unchanged. The existing total comparator remains the complement of the no-populism sigmoid score; it is not the sum of the two independent positive probabilities.

New caches are `sigmoid_<name>.npz` with contract `populism_axis.roberta.sigmoid.v1`, label mapping, explicit model/tokenizer commit SHAs, ordered window checksum and raw logits. Manifest names start `roberta_sigmoid_`; manifests hash both input CSV and output cache. The encoder requires exact 40-character revisions and refuses existing corrected cache/manifest names. Keep old files intact. Choose a fresh run directory, for example `results/roberta-correction/<run-id>/`.

Current comparator readers require the new contract and verify the stored probabilities against sigmoid(logits). Renaming or adding a metadata marker to an old softmax cache is insufficient. Missing corrected files produce an explicit error; the historical `probabilities_*.npz` caches are not a fallback. English/EU and US-governor final scripts preflight all comparator caches before result writes. Historical archived scripts remain historical; do not use them for corrected evaluation.

A separately authorized comparator-only recheck should consume existing axis/NLI scores and write new comparison artifacts, rather than invoke the broad final pipelines that also write axis/NLI results or refit supervised references. Preserve spent-set status: this is an implementation correction, not model/threshold/pooling selection. Retain existing LOSO thresholds procedure for the local EU comparison and distinguish it from the paper’s native zero-logit decision rule.

Example encoding command after separate GPU approval and checkpoint/input verification:

```bash
uv run --extra cu126 python scripts/external/encode_roberta_custom.py \
  --csv data/benchmarks/english_gpd_speeches.csv --text-path-column path --name english \
  --out-dir results/roberta-correction/<run-id> \
  --model-revision <exact-model-sha> --tokenizer-revision <exact-tokenizer-sha>
```

## Corrected run (2026-10-02)

Run `results/roberta-correction/rc-20261002T1415Z-sigmoid/` encoded the eight existing inputs (162,201 windows) on `pvgpu2` with model revision `9b5badfaab879d33fb69a9a22615175bc9b7e6c8` (safetensors sha256 `f5ed0ae3…7589`) and tokenizer revision `722cf37b1afa9454edce342e7895e588b6ff1d59`. Softmax over the new logits reproduces every legacy `probabilities_*.npz` to 1.2e-7, so the windows, tokenizer and weights match the historical runs and only the activation differs.

Corrected comparator scores on the ten README rows, old softmax value in parentheses: English holdout .440 (.464), EU-OOD AUROC .900 (.902), EU-OOD LOSO macro-F1 .555 (.512), Agora .372 (.367), US governors 2010–18 .326 (.376), tweets .571 (.573), PopBERT anti-elitism .669 (.650) and people-centrism .690 (.643), Portugal anti-elitism .867 (.868) and people-centrism .582 (.604). Axis + claim check beats the corrected comparator on eight of ten rows; the losses are Portuguese anti-elitism and EU-OOD LOSO macro-F1 (.545 against .555, no interval). Comparison, secondary tables and provenance: `REPORT.md`, `comparison.json`, `secondary_tables.json` and `provenance/` in the run directory.

The encoder ran from an uncommitted working tree (base `bada4a8` plus `provenance/uncommitted.patch`); commit `48ee5bd` is byte-identical to that executed snapshot. The comparison and secondary tables were re-run from the clean commit `bd58ecc` and reproduced `comparison.json`, `comparison.md` and `secondary_tables.json` byte for byte (`compare.manifest.json` records the clean tree).

The US-governor sentence diagnostic (3,620 sentences, 4,359 windows), which is cited in `research/development-log/external-usgov.md`, was re-encoded on 2026-10-02 in run `results/roberta-correction/rc-20261002T1530Z-sigmoid-usgov-sentences/`. That run used the same checkpoint and tokenizer revisions and the same encoder, comparator module and job script (byte-identical files), executed from a clean git checkout of `4655a38`. Softmax over its logits reproduces the legacy cache to 1.2e-7. Corrected populist-total AUROC is .625 on governors (.630 softmax) and .860 on candidates (.863).

## Stored result files (2026-10-02)

The RoBERTa entries of `results/final/{final_results.json, final_results.md, eu_paper_protocol.json}` and of `results/external/*/final_results.{json,md}` and `final_scores*.csv` now hold the corrected comparator. Run `results/roberta-correction/rc-20261002T1600Z-refresh-result-files/` executed the committed final-stage code unchanged in a sandbox. It first checked that the legacy softmax caches reproduce every stored file byte for byte, then reran the same code with the sigmoid caches and verified that only keys, columns and rows naming RoBERTa changed. The old values are kept and labelled:

- JSON: `superseded_softmax_roberta`, keyed by the path of each replaced subtree, plus a `roberta_correction` note.
- CSV: `<column>__superseded_softmax` columns.
- Markdown: a `(superseded softmax)` row under each corrected row.

Axis, claim-check, NLI, SetFit, XLM-T and supervised-reference entries are unchanged. The historical `probabilities_*.npz` caches and the external `final.manifest.json` files (which document the axis/NLI runs) are untouched.
