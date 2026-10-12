# Stage 2: label-free whitening and nuisance residualization

Date: 2026-09-23. Code commits `b240bef`, `b1d7e06` (grid), `c59709f` (single-set diagnostics).

## Purpose

The V3 axes are built in the embedding space after mean-centering only, and the E and M anchors carry a hostility confound (V3 E populist pole: three times the negative-word mass of its mainstream pole). This step tests two label-free corrections on the carried configuration:

1. **Reference transform.** Remove the top principal directions of the scored dataset's windows and, optionally, apply shrinkage (ZCA) whitening, to windows and anchors alike.
2. **Nuisance residualization.** Build one direction per written nuisance set (hostility, partisan attack, foreign enemy, generic grievance, incivility) and project their span out of the channel axes, as V2.1 did for its word-pair controls.

## Change

- `src/populism_axis/v4_geometry.py`:
  - `fit_reference_transform` and `apply_transform` fit and apply mean, top-d PCs and a whitener Σ_α^−1/2 with Σ_α = (1−α)Σ + α·tr(Σ)/D·I.
  - `transformed_pole_axes` builds the axes after transforming the anchors.
  - `nuisance_directions` builds one direction per nuisance set.
  - `residualize_axes` removes the nuisance span by QR and re-normalizes, mirroring `v21_geometry.residualize_against_controls`.
- `scripts/run_config.py` gains `--transform ref`, `--n-pcs`, `--whiten-shrinkage`, `--nuisance`, `--nuisance-sets` and `--save-projections`. The `v3` path was re-run and matches B1c bit for bit.
- `scripts/sweep_whitening.py` runs the grid.
- Nuisance sets: `anchors/nuisance_sentences.json` (`pa_v4_nuisance_r1`, 5 × 10 matched pairs), approved by the project lead without changes, SHA-256 `a23d574b…`.
- Tests: five new geometry tests in `tests/test_v4_step1.py` (PC removal, identity whitening, residualized orthogonality, pair-difference equivalence). All V4 tests pass.

## Config

- Baseline B1c: e5-large-v2, V3 anchors, capped and cleaned windows, V3 rule, PE soft top-3.
- Grid: d ∈ {0, 1, 2, 4, 8} removed PCs × residualization {off, on} × whitening {none, α = .5}, which gives 20 cells. The transform is fitted separately on English-dev windows and on Agora windows, using no labels.
- Pre-declared primary cell: d = 2, residualization on, no whitening.

## Results on development sets

Full table: `results/step2_whitening/grid.md`. System PE soft. Δ is the cell minus B1c, with 95% paired cluster-bootstrap CIs. The last column is the window-level Spearman correlation between the E projection and VADER negativity on English-dev (label-free); in B1c it is +.337.

| Cell | English-dev ρ | Δ English | Agora ρ | Δ Agora | E vs negativity |
|---|---:|---|---:|---|---:|
| B1c (carried) | .483 | – | .359 | – | +.337 |
| d0, no residualization | .482 | −.001 [−.009, +.006] | .362 | +.003 [+.002, +.004] | +.336 |
| d1, no residualization | .488 | +.005 [−.009, +.022] | .341 | −.018 [−.035, −.007] | +.333 |
| d2, no residualization | .411 | −.072 [−.164, +.026] | .317 | −.042 [−.083, +.009] | −.011 |
| d0, residualized | .418 | −.064 [−.159, +.028] | .316 | −.044 [−.084, +.008] | +.023 |
| **d2, residualized (primary)** | .410 | −.073 [−.165, +.021] | .296 | −.063 [−.114, +.001] | −.121 |
| d8, no residualization | .282 | −.201 [−.321, −.082] | .220 | −.139 [−.217, −.058] | −.107 |
| d0, whitened | .332 | −.150 [−.254, −.054] | .297 | −.062 [−.112, −.011] | +.072 |

The remaining whitened cells also lose .15–.28 on English.

Primary-cell paired comparison (4,000 draws):

| Dataset | Δρ | 95% CI | p | Clusters |
|---|---:|---|---:|---:|
| English-dev (leader) | −.073 | [−.165, +.021] | .12 | 57 |
| Agora (politician) | −.063 | [−.114, +.001] | .06 | 6 |

### What the geometry shows

- **Principal components.** After mean-centering, e5 windows are not dominated by a few directions. On English-dev the top PC explains 5.6% of the variance and the second 3.7%. Removing one PC leaves English unchanged and costs Agora .018. Removing two or more removes channel signal.
- **Whitening.** Shrinkage whitening lowers every cell by .15–.28 on English. Equalizing variance amplifies low-variance directions that carry no construct information.
- **Nuisance directions.** The five nuisance directions are strongly inter-correlated (cosines .48–.66): they share a common negative-affect component. Before residualization, the E axis has cosine .35–.55 with each nuisance direction, and M has .46–.57. Residualizing on all five removes 39% of E's squared norm, 42% of M's and 15% of P's (at d = 0).
- **Residualization works as intended geometrically, and that is what costs accuracy.** It removes the E axis's coupling to VADER negativity (+.34 → +.02), and the ranking of speeches by populism rubric drops by about .06.

### Post-hoc diagnostic: one nuisance set at a time (d = 0)

This analysis was not pre-declared and informs no decision.

| Residualized on | English-dev ρ | Δ English [95% CI] | Agora ρ | Δ Agora | E norm removed |
|---|---:|---|---:|---:|---:|
| Hostility / negative affect | .413 | −.070 [−.159, +.016] | .334 | −.025 | 30% |
| Foreign enemy | .439 | −.043 [−.099, +.013] | .340 | −.019 | 22% |
| Generic grievance | .467 | −.016 [−.084, +.055] | .341 | −.018 | 27% |
| Incivility | .476 | −.007 [−.074, +.061] | .365 | +.006 | 21% |
| Partisan attack | .485 | +.002 [−.047, +.049] | .369 | +.010 | 12% |

Hostility and foreign-enemy hostility carry the signal the benchmarks reward; partisan attack and incivility do not. In these corpora, the coders' populism scores are correlated with hostile affect toward outsiders and with outrage, not only with the people-versus-elite claim. Removing that affect geometrically therefore lowers agreement with the labels. The data cannot separate two explanations: (a) hostility is an integral part of how populism is expressed in these speeches, or (b) coders partly rate tone. A measure that separates the claim from its tone would score lower on these benchmarks under either explanation.

## Decision

**Keep B1c.** No grid cell passes the adoption rule (English Δ ≥ +.01, Agora Δ ≥ −.01, no CI entirely below 0), so the selection rule selects no cell. The pre-declared primary cell loses .073 on English-dev and .063 on Agora. The carried configuration is unchanged (`results/step2_whitening/carried_config.json`).

Implication for step 3: the minimal-pair anchors match affect within each pair, so they attack the same confound at the anchor level rather than by projection. Step 3 will report the same VADER-coupling diagnostic, so that "less tone" and "better benchmark agreement" can be read separately.

## Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| Nuisance sentence encode (`v4_nuisance_r1__a23d574b`) | `b1d7e06` | no | see manifest | pvgpu1 |
| Grid, all 20 cells and the reference row | `b1d7e06` | no | `18ef2aedb0cc` (grid manifest) | local |
| Primary cell `d2_res_nowhite` | `b1d7e06` | no | `a6ac2debfcc5` | local |
| Single-set diagnostics | `c59709f` | no | see manifests | local |

- GPU use: one job, under 30 s, 684 MB peak. Both sentence sets were encoded twice and the passes were byte-identical.
- CPU: about 25 minutes for the grid.

```bash
# pvgpu1
scripts/remote_job.sh step2_encode_sentences -- bash -c "python scripts/encode_sentences.py --encoder e5-large-v2 --sentences anchors/nuisance_sentences.json --set-id v4_nuisance_r1 --repeat 2 && python scripts/encode_sentences.py --encoder e5-large-v2 --sentences anchors/pair_anchors.json --set-id v4_pairs_r1 --repeat 2"
# local
uv run --extra cpu python scripts/sweep_whitening.py --nuisance results/embeddings/e5-large-v2/sentences/v4_nuisance_r1__a23d574b.npz
uv run --extra cpu python scripts/run_config.py --label d0_res_<set> ... --transform ref --nuisance ... --nuisance-sets <set>
```
