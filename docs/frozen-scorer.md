# Frozen scorer quickstart

The frozen scorer needs source, Python dependencies, four small assets and the
two model checkpoints. It does not need raw benchmark corpora or research caches.
The existing runtime assets total 13,525 bytes; the mean is a required 4,224-byte
reference vector, not a rebuildable embedding cache.

## Check the source/reference package

From the checkout root, run:

```bash
python3 scripts/verify_frozen_assets.py
```

The check uses only the Python standard library. It verifies file sizes/SHA-256,
source/dependency-file presence and identity, the float32 reference mean shape,
and reference scales for k:8/k:1. It makes no network calls, writes no outputs and
does not regenerate assets, import scientific libraries or run inference.

The four runtime assets are:
- `anchors/proposition_anchors.json`
- `anchors/claim_hypotheses.json`
- `results/frozen_reference/reference_stats.json`
- `results/frozen_reference/reference_mean_embedding.npy`

`results/frozen_reference/runtime_manifest.json` records the asset and source
hashes plus observed model snapshots. `SHA256SUMS` gives the same asset/source
checksums using paths relative to the checkout. Restore a missing/corrupt asset
from the matching package; do not run frozen_reference.py to repair it. That
research script recomputes and writes results.

## Install dependencies and score new text

These commands are a usage recipe, not a claim that cold inference was tested:

```bash
uv sync --extra cpu
uv run --extra cpu python scripts/score_texts.py --input my_texts.csv \
  --text-column text --id-column id --unit speech --reference frozen \
  --device cpu --out-dir scored/my_run
```

The input must have the named text/id columns. Output is scores.csv, evidence.json
and run_manifest.json in the selected output directory. Use `--unit sentence`
for sentences/tweets (top window); speech/paragraph units use top 8. Pass
`--reference frozen` explicitly: the scorer's parser default is corpus.
Other top-k overrides have no matching frozen document scales and fall back to
corpus scales; keep k:8/k:1 for this runtime package.

`uv.lock` and `pyproject.toml` specify dependencies. Always select CPU or CUDA
extras; a bare uv run can remove torch. Without uv, install torch
explicitly, since it is not a base dependency: `pip install . torch`. The asset check does not verify installed
package versions. Installing dependencies needs network access if not cached.

## Models and optional offline cache check

The scorer loads `intfloat/e5-large-v2` and `facebook/bart-large-mnli` at exact
Hub commits (`src/populism_axis/model_revisions.py`). It uses cached models by
default and fails if the required files are absent. Add `--allow-download` to
permit downloads. Their weight/config/tokenizer files total
2,973,709,905 bytes (about 2.974 GB), outside Git.

Pinned revisions (the inspected laptop snapshots, and the current Hub `main`):
- e5: `f169b11e22de13617baa190a028a32f3493550b6`
- BART: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`

`score_texts.py` refuses a moving ref (`main`, a tag, a short SHA) and refuses a
model without a pinned revision unless `--encoder-revision`/`--nli-revision` gives
an exact 40-character commit. Its `run_manifest.json` records the requested and
the resolved revisions.

To check an existing HF hub cache without downloading anything:

```bash
python3 scripts/verify_frozen_assets.py --model-cache ~/.cache/huggingface/hub
```

This hashes every declared file in the pinned snapshot directories. Where the
cache's moving `main` ref points does not matter, because loading is pinned. It
does not load weights or prove inference results. Use the actual hub-cache path
when HF_HOME/HF_HUB_CACHE is customized.

Pinning fixes what new runs load. It does not establish which revisions produced
the historical pvgpu1 runs, which recorded model IDs only; see
[model-revisions.md](model-revisions.md). The end-to-end inference check from a
fresh checkout is in [fresh-inference-check.md](fresh-inference-check.md).

Both current upstream model cards declare MIT:
[e5](https://huggingface.co/intfloat/e5-large-v2) and
[BART](https://huggingface.co/facebook/bart-large-mnli). Exact historical notices
were not archived. Preserve notices/attribution if later redistributing weights.

## What a fresh checkout establishes

A fresh source checkout passes the stdlib completeness check without NumPy,
torch, model weights or research caches. It can score new text after dependencies
and the pinned models are installed. That is distinct from a tested inference
reproduction: [fresh-inference-check.md](fresh-inference-check.md) re-scores
English-dev from a clean checkout and compares it with the stored frozen scores.
It passed all seven checks on `main` at `003f00ca` (`results/fresh_inference_check/`).

A fresh checkout cannot recreate the benchmark reports from this small package:
research embeddings, NLI arrays and premises are not in Git. It can recompute
every reported number from the stored per-item scores; see
[reproducing.md](reproducing.md).

Publication rights for the code, the reference assets, the benchmark tables and
the models are reviewed in [publication-rights.md](publication-rights.md); what
the public repository includes is listed in `data/README.md`. No model weights
are in Git.
