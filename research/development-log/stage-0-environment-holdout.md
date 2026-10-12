# Stage 0: environment, provenance and holdout freeze

Date: 2026-09-23. Commits `ce9f3f6`, `1c2e595` on `feature/v2-v3-benchmark-pipeline`.

## Purpose

Prepare V4 so that each later step can be run and assessed in isolation.
This means a working GPU environment on pvgpu1, provenance for every result,
and a held-out English split that no V4 decision will see. Step 0 produces no
benchmark result and takes no metric decision.

## Change

- **GPU environment.** torch moved from the base dependencies into two extras, `cpu` and `cu126`, both pinned to `torch==2.14.0` from the PyTorch indexes (`pyproject.toml`, `uv.lock`). The PyPI torch wheel used by V3's lockfile is built for CUDA 13 and cannot run on pvgpu1's driver (560.35, CUDA 12.6). `accelerate` moved with torch because nothing imports it. `pyarrow` was added because V3 and V4 scripts write parquet.
- **Provenance.** `src/populism_axis/run_manifest.py` writes `run_manifest.json` with commit and dirty flag, command, host, library versions, input SHA-256 and a run hash.
- **Holdout split.** `src/populism_axis/v4_split.py` and `scripts/freeze_holdout.py` freeze a country-level English split.
- **Encoder scaffolding.** `src/populism_axis/v4_encoders.py` gives each model an explicit input template. Qwen3-Embedding now gets the instruction prefix from its model card, which V3 omitted. `src/populism_axis/v4_sentences.py` loads V3 anchors and the planned V4 anchor, nuisance and plain sentence files. `scripts/encode_sentences.py` encodes them.
- **pvgpu1 launcher.** `scripts/remote_job.sh` runs one V4 job at a time. It refuses to start when the GPU has more than 4 GB in use or the 1-minute load exceeds 2.5. Jobs run under `nice -n 10` with two CPU threads and offline Hugging Face mode, and log to `logs/`.
- **Documentation.** `docs/development-protocol.md` (step order, data access, adoption rule, report template), `docs/environment.md`, and a `docs/results-index.md` entry.
- **Tests.** `tests/test_v4_step0.py` has 10 tests covering the manifest, the split rule, encoder templates and the sentence loader.

## Config

Split rule (`SplitRule`, seed 20260923):

- Countries are the sampling unit. 5,000 candidate partitions are drawn, each country included with probability .30.
- A candidate is kept if the holdout holds 27–33% of speeches, at least 4 of the 13 extreme positives (dev at least 8), at least 20 speeches with a score above 0, and at least 20 leaders.
- Among kept candidates, the one minimizing |mean score dev − holdout| + KS distance is chosen.
- Labels are used only for this balance and for no method choice.

## Results

**Holdout freeze.** 306 of 5,000 candidates met every constraint. The chosen partition has a balance distance of .035.

| | Dev | Holdout |
|---|---:|---:|
| Speeches | 174 | 84 |
| Leaders | 57 | 27 |
| Countries | 24 | 14 |
| Leader-terms | 74 | 37 |
| Extreme positives (score ≥ 1) | 9 | 4 |
| Extreme negatives (score = 0) | 98 | 47 |
| Speeches with score > 0 | 76 | 37 |
| Mean GPD score | .165 | .168 |
| Word-count Spearman (null) | .236 | .261 |
| Human coder Spearman (one coder pair each) | .699 | .748 |

The holdout countries are Armenia, Croatia, El Salvador, Georgia, Guatemala, India, Indonesia, Italy, Lithuania, Malaysia, Netherlands, Singapore, Sweden and the United States. EU-OOD is held out in full.

**pvgpu1 smoke encode.** The V3 proposition anchors (60 sentences) were encoded twice with `intfloat/e5-large-v2`.

| Check | Result |
|---|---|
| torch on pvgpu1 | 2.14.0+cu126, CUDA available, NVIDIA A30 |
| Time, including model load | 7.7 s |
| Peak VRAM | 676 MB |
| Two passes byte-identical | yes |
| GPU memory after the job | 651 MiB (the idle baseline) |

Axis cosines rebuilt locally from these embeddings match V3's recorded geometry exactly. The V4 encoder therefore reproduces the V3 anchor axes despite the torch upgrade from 2.11 to 2.14.

| Channel pair | V4 smoke | V3 recorded |
|---|---:|---:|
| P·E | +.5029 | +.5029 |
| E·M | +.5587 | +.5587 |
| P·M | +.2661 | +.2661 |

**Tests.** The 10 new tests pass. The full suite has one error that predates V4: `alternative-pairs/benchmark_large_samples.py` hard-codes `/home/atomasevic/populism-axis`, which exists only on pvgpu1.

## Decision

No metric decision. Step 0's success criteria are met: CUDA works on pvgpu1, the smoke encode is deterministic and matches V3, and the holdout is frozen.

## Provenance

- Holdout freeze: `results/step0_env_holdout/holdout_freeze.json`, canonical SHA-256 `f66766cc…eec2fa`. Split file `data/benchmarks/english_split.csv`, SHA-256 `fe6bea10…7824cb`. Its `run_manifest.json` records `dirty: true` because the freeze ran before the step-0 code was committed; the split file hash is the binding record.
- Smoke encode: `results/step0_env_holdout/smoke/e5-large-v2/sentences/v3_proposition_smoke__ea98e5b2.manifest.json`, commit `1c2e595`, clean tree, pvgpu1 log `logs/step0_smoke.log`.
- GPU time: under one minute.

## Deviations from the plan and notes

- The CPU evaluation harness is deferred to step 1, because it can only be validated against encoded windows. This covers `v4_geometry.py`, `build_and_project.py`, `evaluate.py`, `compare_steps.py` and `render_step_report.py`. Step 1 builds and tests it first, and checks it by reproducing V3's .433 on all 258 speeches.
- pvgpu1 has no `rsync`, so results come back with `tar` over ssh (`docs/environment.md`).
- `uv sync --extra cu126` on pvgpu1 removed a few packages that were installed but not declared in `pyproject.toml`, for example `yarl`, `xxhash` and `wrapt`. Any old pvgpu1 workflow that relied on them, such as a `datasets` import, needs them added as dependencies.
- The torch indexes had cu126 builds up to 2.14.0, so neither the cu128 fallback nor an older pin was needed.
- Each split's human-coder ceiling rests on a single coder pair with at least 30 shared speeches. It is a rough reference, not an estimate.
- During the smoke run, another user's two CPU-bound jobs held pvgpu1's 1-minute load at about 2.2–2.3 of 4 cores. The guard threshold of 2.5 leaves little room, so longer jobs in later steps may have to wait.
