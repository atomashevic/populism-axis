# RoBERTa sigmoid encode: US-governor sentence diagnostic

Run `rc-20261002T1530Z-sigmoid-usgov-sentences`, 2026-10-02. This run completes the comparator correction of `rc-20261002T1415Z-sigmoid` for the one input that run left out: the 3,620 rubric sentences of the US-governor benchmark (4,359 one-sentence windows). Their RoBERTa rows are cited in `research/development-log/external-usgov.md`, so the old softmax values were re-encoded rather than only marked superseded.

## What ran

- **Code.** Commit `4655a38` (origin/main at launch), executed from a clean git clone of a bundle of that commit on pvgpu2. The encoder, comparator module, window splitter and job script are byte-identical to those of `rc-20261002T1415Z-sigmoid` (`provenance/provenance.json`, `key_files_sha256`). Both the manifest and the job log record commit `4655a38` with `dirty: false`, and both records are valid here because the executed directory is a git checkout.
- **Checkpoint.** `coastalcph/roberta-large-ft-trump-populism@9b5badfaab879d33fb69a9a22615175bc9b7e6c8` and tokenizer `FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`, read from the existing pvgpu2 cache with the Hub offline.
- **Input.** `data/benchmark_usgov/sentences.csv`, column `text`, sha256 `2280a118…bc3a7`, the same file as the 2026-09-24 softmax encode.
- **Job.** `provenance/run_encode.sh` through `scripts/remote_job.sh`. The encode took 37 s on the A30.

Job log header (the log itself is local only):

```
# job usgov_sentences_sigmoid started 2026-10-02T15:28:48Z on pvgpu2
# commit 4655a38595c89faac81a65513b27f18e6df0924a dirty=no
# gpu before: 1 MiB used, load 0.16
# env 3.11.11 2.14.0+cu126 12.6 5.6.2 NVIDIA A30
[roberta] usgov_sentences: 3620 docs -> 4359 windows
```

## Checks

- The cache passes the strict `load_scores` contract.
- The `doc_index` sequence equals the legacy `results/external/usgov/roberta/probabilities_usgov_sentences.npz`.
- Softmax of the new logits reproduces the legacy probabilities to max |Δ| 1.2e-7. The windows, tokenization and weights match the historical encode, and only the activation differs.

## Result

The sentence-level populist-total comparator (1 − sigmoid of the no-populism logit, top window):

| Subset | AUROC populist vs rest | AP | vs pluralist | vs neutral |
|---|---:|---:|---:|---:|
| Governors 2010–18 (3,539), sigmoid | .625 | .595 | .619 | .630 |
| Governors 2010–18, softmax (superseded) | .630 | .591 | .626 | .633 |
| Presidential 2016 (81), sigmoid | .860 | .826 | .869 | – |
| Presidential 2016, softmax (superseded) | .863 | .831 | .873 | – |

V4 axis + claim check is .744 and .936 on the same rows. The finding in the development log still holds. These values are computed by `results/roberta-correction/rc-20261002T1600Z-refresh-result-files/`, which writes them into `results/external/usgov/final_results.{json,md}`.

The cache `sigmoid_usgov_sentences.npz` (sha256 `f1473ca4…0506d`) is local only, like the other corrected caches. A remote copy is in the executed checkout, `pvgpu2:~/populism-axis-roberta-correction/rc-20261002T1530Z-sigmoid-usgov-sentences/code/`.
