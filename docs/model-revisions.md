# Model revisions: pinned and historical

## What the code loads now

`src/populism_axis/model_revisions.py` pins the two models of the frozen
configuration to exact Hugging Face commits:

| Model | Pinned commit | Commit date (Hub) |
|---|---|---|
| `intfloat/e5-large-v2` | `f169b11e22de13617baa190a028a32f3493550b6` | 2025-02-17 |
| `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | 2023-09-05 |

`encoders.py` (and through it `encode_windows.py`, `encode_sentences.py` and
`score_texts.py`), `nli_stage.py` and `score_texts.py`'s claim check pass these
commits to `from_pretrained`. New run manifests record the requested commit and
the one transformers resolved. `score_texts.py` and `nli_stage.py` refuse moving
refs and models without an exact commit. The sweep-only encoders (Qwen3,
multilingual-e5, gte, e5-small) are not part of the frozen configuration and
still follow `main`; their manifests now record the resolved commit. The RoBERTa
comparator was already pinned (`docs/roberta-correction.md`).

## What the historical runs loaded

**Recorded.** All committed e5 and BART encoding manifests (9 for e5, 5 for BART)
come from `pvgpu1`, 2026-09-23 to 2026-09-24, with transformers 5.6.2 and torch
2.14.0+cu126; the development logs place the external-benchmark encodings on
pvgpu1 too. The manifests record the model ID only, not a commit.
`scripts/v4/remote_job.sh` at those commits (`c4fef72`, `c9bfdce`, `dd9e073`)
set `HF_HUB_OFFLINE=1` by default, so those runs loaded whatever snapshot the
pvgpu1 cache held and downloaded nothing. That cache was not inspected or
recorded. The pinned commits above are the laptop snapshots whose files are
hashed in `results/frozen_reference/runtime_manifest.json`.

**Hub history (read 2026-10-02 from the Hub commits API).**

- e5-large-v2: weights uploaded in `bdaaca6` (2023-05-19); `model.safetensors`
  added in `b303079` (2023-06-20); later commits add sentence-transformers files,
  a handler, README edits (2023) and ONNX/OpenVINO exports (`ef82eb1`…`f169b11`,
  2025-02-17). `f169b11` is still `main`.
- bart-large-mnli: `model.safetensors` added in `9fc9c4e` (2022-11-16); the only
  later commit is a README update, `d7645e1` (2023-09-05), still `main`.

**Inferred, not verified.**

1. Any snapshot downloaded after 2025-02-17 (e5) or 2023-09-05 (BART) is exactly
   the pinned commit. The project's first commit is 2026-04-30, so a download made
   for this project resolved to the pinned commits. Whether pvgpu1's cache was
   filled by this project or earlier is not recorded.
2. Even an older snapshot would most likely load the same weights, config and
   tokenizer: the commit titles after the weight uploads touch only exports,
   README, metadata or format conversions. A per-file blob comparison across
   revisions was not done here; a pre-safetensors e5 snapshot (before
   2023-06-20) would load `pytorch_model.bin` instead of `model.safetensors`.

So the historical runs very probably used the pinned bytes, but no record shows
it. Two checks can settle it, and neither needs the historical runs to be
re-presented as something they were not:

- **Cache inspection, if pvgpu1 is still reachable** (read-only):
  ```bash
  ssh pvgpu1 'for m in intfloat--e5-large-v2 facebook--bart-large-mnli; do
    d=~/.cache/huggingface/hub/models--$m; echo "$m main=$(cat $d/refs/main)";
    ls -l --time-style=full-iso $d/snapshots; done'
  ```
  A single snapshot equal to the pinned commit closes the question for the
  bytes. Adjust the path if `HF_HOME`/`HF_HUB_CACHE` was set there.
- **Numerical reproduction** on `pvgpu2` from a fresh checkout:
  [fresh-inference-check.md](fresh-inference-check.md). Agreement within its
  thresholds shows the pinned models reproduce the stored English-dev scores,
  which is what a reader of the results needs, whatever revision string pvgpu1
  held.

Until one of them is run, the reports should say the historical revisions are
inferred, as above.
