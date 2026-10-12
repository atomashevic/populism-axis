# Stage 4: window size and document pooling

Date: 2026-09-23. Code commit `2be6215`.

## Purpose

V3 scores a document as the mean of its three highest-scoring one-sentence windows. The value k = 3 was chosen for the V1 static measure and never tested for V3, and multi-sentence windows were supported but never run. The GPD rubric grades how pervasive populism is across a whole speech, so wider pooling may fit it better. A claim split across sentences may also need wider windows. This step sweeps both.

## Change

- `src/populism_axis/contextual_axis.py`: `aggregate_from_window_scores` pools window scores with one of four rules:
  - `k:<n>`: mean of the n highest windows;
  - `frac:<f>`: mean of the top ⌈f·n_windows⌉;
  - `softcount:<q>`: share of a document's windows above the dataset-wide q-quantile, a label-free pervasiveness readout;
  - `mean`: mean of all windows.
  `aggregate_scores` gains `top_rule`. The default `k:3` path is bit-identical to B1c.
- `scripts/run_config.py --top-rule`, and `scripts/sweep_windows_topk.py`.
- Two- and three-sentence windows (stride 1, 60-word cap per sentence) were encoded on pvgpu1 with 192- and 256-token limits. English uses cleaned text.
- Tests: three pooling tests in `tests/test_v4_step1.py`.

## Config

Carried configuration B1c: e5-large-v2, V3 anchors, V3 rule, PE soft.

Grid: sentences per window {1, 2, 3} × pooling {k:1, k:2, k:3, k:5, k:8, frac:0.10, softcount:0.9}, which gives 21 cells. The pre-declared primary cell is spw 1, k:3 (the carried configuration); the question is whether any other cell beats it under the adoption rule.

## Results on development sets

Full grid: `results/step4_windows_topk/grid.md`. System PE soft. Δ is against B1c; CIs are 95% paired cluster-bootstrap.

| Cell | English-dev ρ | Δ English [95% CI] | Agora ρ | Δ Agora [95% CI] | Rule |
|---|---:|---|---:|---|---|
| spw1 k:1 | .422 | −.061 [−.100, −.023] | .344 | −.015 | fail |
| spw1 k:3 (carried) | .483 | 0 | .359 | 0 | – |
| **spw1 k:5** | .512 | **+.030 [+.009, +.054]** | .360 | +.001 [−.002, +.005] | pass |
| **spw1 k:8** | .516 | **+.034 [+.004, +.068]** | .355 | −.004 [−.011, +.006] | pass |
| spw1 frac .10 | .521 | +.039 [−.024, +.109] | .336 | −.023 [−.030, −.016] | fail |
| spw1 softcount .9 | .511 | +.028 [−.069, +.130] | .283 | −.076 | fail |
| spw2 k:3 | .458 | −.025 [−.079, +.026] | .358 | −.001 | fail |
| spw2 k:8 | .505 | +.022 [−.032, +.077] | .356 | −.003 | pass |
| spw3 k:3 | .484 | +.001 [−.064, +.066] | .362 | +.003 | fail |
| spw3 k:5 | .499 | +.017 [−.050, +.082] | .362 | +.003 | pass |
| **spw3 k:8 (selected)** | .517 | +.034 [−.029, +.098] | .359 | −.000 [−.023, +.013] | pass |
| spw3 frac .10 | .524 | +.041 [−.025, +.110] | .359 | −.001 [−.026, +.012] | pass |

Other metrics for the leading cells:

| Cell | Extreme AUROC | Leader-term ρ | Agora high-vs-low AUROC |
|---|---:|---:|---:|
| B1c (spw1 k:3) | .940 | .359 | .889 |
| spw1 k:5 | .944 | .354 | .882 |
| spw1 k:8 | .939 | .327 | .860 |
| spw3 k:8 | .921 | .308 | .874 |
| spw3 frac .10 | .913 | .354 | .883 |

### What the grid shows

- **Pooling depth, not window size, drives the gain.** At every window size, English-dev ρ rises steadily from k = 1 to k = 8, by .06–.10 in total. Window size on its own does not help: at k = 3, spw 1, 2 and 3 score .483, .458 and .484. The GPD rubric rewards a score that sees more of the speech.
- **The one-sentence gains are the only statistically significant ones.** spw1 k:5 and spw1 k:8 have English intervals entirely above zero, [+.009, +.054] and [+.004, +.068]. They are the first significant improvements in V4. The three-sentence gains are the same size or larger, but their intervals are wider and include zero.
- **Multi-sentence windows change which channel carries the signal.** With one-sentence windows, P alone matches or beats PE (spw1 k:8: P .516, PE .516). With three-sentence windows, PE clearly beats P (.517 against .492 at k:8; .524 against .474 at frac .10). Wider windows let the P and E claims co-occur inside one window, which is what the conjunction needs.
- **Pervasiveness readout.** `softcount:0.9` correlates .79–.88 with the top-3 score on English. It is competitive on English but collapses on Agora (Δ −.08 to −.10), because most Agora paragraphs have only a handful of windows, so a share above a threshold is too coarse there.
- **Leader-term ρ falls with deeper pooling** (.359 → .308 at spw3 k:8). It is not a decision metric and aggregates only 74 development leader-terms, but it runs against the speech-level gain.

## Paired difference vs previous step

Selected cell spw3 k:8 minus B1c, 4,000 draws.

| Dataset | Δρ | 95% CI | p | Clusters |
|---|---:|---|---:|---:|
| English-dev (leader) | +.034 | [−.029, +.098] | .30 | 57 |
| Agora (politician) | −.000 | [−.023, +.013] | .89 | 6 |

## Decision

**Adopt spw3 k:8** under the pre-declared grid rule. The rule takes the passing cell with the largest mean of the two differences, which is spw3 frac .10 at .0203. It then treats cells within .005 as tied and picks the simplest of them, which is spw3 k:8 (.0169, since k is simpler than a fraction).

This choice is fragile and the report records it as such. spw1 k:5 (mean difference .01524) misses the tie band by .00005. spw1 k:5 would give nearly the same English-dev ρ with a statistically significant interval, one-sentence windows (cheaper to encode in step 5), and no loss in leader-term ρ. The rule was fixed in advance, so it stands. Two sensitivity rows are carried into step 5 alongside the adopted cell: spw1 k:5 and spw1 k:8.

Carried configuration: `results/step4_windows_topk/carried_config.json`. It uses e5-large-v2, V3 anchors, three-sentence windows (stride 1, 60 words per sentence, 256 tokens), V3 rule and PE soft, with the mean of the top 8 windows.

## Provenance

| Artifact | Commit | Dirty | Host |
|---|---|---|---|
| spw2 / spw3 window encodes | `3dc07b2` | no | pvgpu1 |
| Grid (21 cells) | `2be6215` | no | local |

- GPU: one job, 240 s in total, peak 842 MB; pvgpu1 returned to idle.
- CPU: about 25 minutes.
- Truncated windows: English 41 (spw2) and 69 (spw3); Agora 0.

```bash
# pvgpu1
scripts/remote_job.sh step4_encode_windows -- bash -c "python scripts/encode_windows.py --encoder e5-large-v2 --datasets english,agora --sentences-per-window 2 --stride 1 --max-words 60 --max-length 192 --batch-size 32 --clean-text && python scripts/encode_windows.py ... --sentences-per-window 3 --max-length 256 ..."
# local
uv run --extra cpu python scripts/sweep_windows_topk.py
```
