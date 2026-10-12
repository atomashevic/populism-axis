# Fresh-checkout inference check

This check answers one question: does a clean checkout, with the pinned e5 and
BART commits, reproduce the stored English-dev scores? It re-scores the 174
English-dev speeches with the public scorer (`scripts/score_texts.py --reference
frozen`) and compares them with `results/frozen_reference/scores_english_dev.csv`.
On English-dev the frozen statistics are the corpus's own, so those stored scores
equal the transductive scores of the development log (ρ = .517 for the axis,
.537 for axis + claim check).

English-dev is development data. The check reads no English-holdout or EU-OOD
text and makes no configuration choice.

**Status: run on 2026-10-02, PASS (7 of 7 checks).** Commit `003f00ca` (main
after PR #4), clean detached checkout on `pvgpu2` (NVIDIA A30, torch
2.14.0+cu126, transformers 5.6.2). The e5 and BART files came from a cold
download: an empty cache folder, `HF_HUB_OFFLINE=0`, at 15:59 UTC, during the
same check of the parent commit `586b01f`
([model_download_job_log.txt](../results/fresh_inference_check/model_download_job_log.txt)).
`verify_frozen_assets.py --model-cache` on that folder matched all 12 pinned
files. The scoring job read the folder offline. It gave 174 documents and 20,284
windows, ρ(axis, rubric) = .5166 (stored .5166) and ρ(axis + claim check,
rubric) = .5376 (stored .5367). Spearman between fresh and stored scores was at
least .99999 for the axis columns and at least .9996 for the claim-check
columns. Every document was within 0.05 SD on every column; the largest
difference was 0.031 SD (`claim_check`). A first scoring of the same commit with
the same cache, at 16:57 UTC, gave bit-identical scores (largest difference 0.0);
its output stayed on `pvgpu2` and is not part of the committed evidence. The
report, the manifests and the fresh scores are in
[results/fresh_inference_check/](../results/fresh_inference_check/). The input
CSV (raw speech text, its hash in `scorer_output/run_manifest.json`) and the scorer's
`evidence.json` (window text) are not committed.

## Run it

Launch from the connected workstation (currently `laptop-omarchy`; check
`hostname` and its SSH configuration as `docs/environment.md` says). Use a new
folder on `pvgpu2` and leave the existing GPU checkout untouched.

1. **A fresh clone of the commit under test**, sent as a Git bundle so pvgpu2
   needs no GitHub access. The bundle carries the branch that contains the
   commit (`main` once merged, or a PR branch); the clone on pvgpu2 fetches that
   ref explicitly, because `git clone` of a bundle takes only `refs/heads/*`.
   On the workstation:
   ```bash
   cd ~/work/populism-axis
   BRANCH=main                                  # or the PR branch under review
   git fetch origin "$BRANCH"
   SHA=$(git rev-parse "origin/$BRANCH")
   git bundle create /tmp/pa-$SHA.bundle "refs/remotes/origin/$BRANCH"
   sha256sum /tmp/pa-$SHA.bundle
   scp /tmp/pa-$SHA.bundle pvgpu2:/tmp/
   ssh pvgpu2 "git init -q ~/populism-axis-freshcheck/$SHA && cd ~/populism-axis-freshcheck/$SHA &&
     git fetch -q /tmp/pa-$SHA.bundle refs/remotes/origin/$BRANCH:refs/heads/under-test &&
     git checkout -q --detach $SHA && git rev-parse HEAD && git status --short"
   ```
   `git rev-parse HEAD` must print `$SHA` and `git status --short` nothing.

2. **Environment and package check**, inside that folder on pvgpu2:
   ```bash
   uv sync --extra cu126 --frozen
   python3 scripts/verify_frozen_assets.py
   python3 scripts/verify_frozen_assets.py --model-cache ~/.cache/huggingface/hub
   ```
   The second check reports whether pvgpu2's cache holds the pinned snapshots
   with the recorded bytes. Then pick one model source:
   - **Cold download (the stronger test, needs approval: about 3 GB from the
     Hub).** Use an empty cache inside the check folder and allow downloads for
     this job only: `export HF_HUB_CACHE=$PWD/hf-cache HF_HUB_OFFLINE=0`.
   - **Existing cache, offline.** Only if the second check passed. Keep the
     default `HF_HUB_OFFLINE=1`.

3. **Prepare the input** (CPU):
   ```bash
   uv run --extra cu126 python scripts/check_fresh_inference.py prepare scored/fresh_check
   ```

4. **Score on the GPU** (one job; `scripts/remote_job.sh` applies the host
   etiquette and logs to `logs/fresh_check.log`):
   ```bash
   scripts/remote_job.sh fresh_check -- python scripts/score_texts.py \
     --input scored/fresh_check/english_dev_input.csv --text-column text --id-column id \
     --unit speech --reference frozen --clean --device cuda --threads 2 \
     --nli-batch-size 64 --out-dir scored/fresh_check/scored
   ```
   `--clean` and `--nli-batch-size 64` match the historical English windows and
   NLI stage.

5. **Compare** (CPU), after the job ends:
   ```bash
   uv run --extra cu126 python scripts/check_fresh_inference.py compare scored/fresh_check
   ```
   It prints PASS/FAIL per check, exits 0 only if all pass, and writes
   `scored/fresh_check/fresh_inference_check.json` with a manifest.

6. Copy `scored/fresh_check/` (without the model cache) back to a new staging
   folder on the workstation. Whether and where to commit it is a separate
   decision.

## Expected output

| Check | Passes when |
|---|---|
| `clean_checkout` | the scorer's run manifest has `git.dirty = false` and the same commit as the checkout running the comparison, which must be clean too |
| `frozen_reference_mode` | frozen reference, `k:8`, e5-large-v2 and bart-large-mnli |
| `pinned_revisions` | the requested commits are `f169b11e…50b6` (e5) and `d7645e12…ce` (BART), and the resolved commits, where transformers reports them, are the same |
| `window_count` | 174 documents and exactly 20,284 windows |
| `score_agreement` | Spearman between fresh and stored scores ≥ .999 for `axis`, `axis_P`, `axis_E` and ≥ .99 for the claim-check and combined columns; and every document's axis scores, and at least 95% of documents' claim-check and combined scores, within 0.05 SD (of the stored column) of the stored value |
| `dev_metric_PE_axis` | ρ(axis, GPD rubric) within .005 of the stored .517 |
| `dev_metric_PE_axis_plus_nli` | ρ(axis + claim check, rubric) within .010 of the stored .537 |

These thresholds were set before any run. Exact equality is not expected: the
historical window embeddings and entailment probabilities were stored as float16,
the GPU differs (pvgpu1 then, pvgpu2 now), and a near-tie can swap one of a
speech's top 8 windows, which changes the claim check's premises. The report
also gives the largest absolute, relative and SD-scaled difference per column. The
value tolerance keeps a rescaled or shifted score from passing on rank agreement
alone; the claim-check columns need only 95% of documents within it, because one
swapped top window can move a speech's mean entailment by many SDs.

## Reading a failure

- `window_count` failing points at text cleaning or windowing, not the models.
- `pinned_revisions` failing means the cache or the code did not load the pinned
  commits; nothing else in the report is evidence until that is fixed.
- Axis columns agreeing while claim-check columns fail `score_agreement`
  points at BART or the top-window premises.
- Every score column off by a large margin, with the right window count and
  revisions, would mean the historical runs did not use these bytes (see
  [model-revisions.md](model-revisions.md)). Report that as it is; do not
  regenerate the stored reference assets to make the check pass.
