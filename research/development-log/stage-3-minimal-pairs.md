# Stage 3: minimal-pair anchors

Date: 2026-09-23. Code commit `454f1bb`. Anchors `pa_v4_pair_anchors_r1` (SHA-256 `24f70345…`), approved by the project lead without changes.

## Purpose

The V3 anchors are 10 populist and 10 mainstream sentences per channel. They were written independently and differ in length, pronouns and sentiment (see `anchors/anchor_balance_report.md`). Step 3 replaces them with 40 minimal pairs per channel. In each pair, the mainstream sentence keeps the topic, actors, grammatical person and length, and replaces only the populist claim. The aim is an axis that encodes the claim rather than surface style or tone.

## Change

- `anchors/pair_anchors.json`, built by `anchors/build_pair_anchors_draft.py`: 3 × 40 pairs. For P and E the pairs pass all balance gates, with identical they/we/you rates on both poles and matched length and sentiment. M is not gated.
- `scripts/run_config.py` accepts several `--anchors` files and pools them.
- `scripts/run_step3_anchors.py` runs the rows, the paired comparisons, the tone diagnostic, the anchor QA and the per-anchor attribution.

## Config

Carried configuration from step 2 (B1c): e5-large-v2, cleaned and capped windows, V3 axis rule, PE soft top-3. Only the anchor set changes.

- **Primary:** the V4 minimal pairs.
- **Sensitivity:** V3 and V4 anchors pooled.

## Results on development sets

System PE soft. CIs are 95% cluster-bootstrap.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---:|
| B1c, V3 anchors (carried) | .483 | [.339, .618] | .940 | .359 | .359 | .889 |
| V4 minimal pairs (primary) | .444 | [.306, .575] | .917 | .388 | .347 | .862 |
| V3 + V4 pooled | .449 | [.316, .578] | .931 | .387 | .358 | .876 |

Single channels (English-dev / Agora):

| Channel | V3 anchors | V4 pairs | Pooled |
|---|---|---|---|
| P | .495 / .341 | .444 / .311 | .471 / .324 |
| E | .453 / .331 | .434 / .346 | .437 / .347 |
| M | .397 / .296 | .374 / .293 | .380 / .298 |
| PEM | .471 / .347 | .437 / .331 | .446 / .341 |

The V4 pairs lower P on both datasets. E is slightly lower on English (−.019) and slightly higher on Agora (+.015). Leader-term ρ rises for both V4 rows (.359 → .388), but it is not a decision metric and its interval is wide.

## Paired difference vs previous step

4,000 paired cluster-bootstrap draws against B1c.

| Row | Dataset | Δρ | 95% CI | p |
|---|---|---:|---|---:|
| V4 pairs | English-dev (57 leaders) | −.039 | [−.123, +.042] | .33 |
| V4 pairs | Agora (6 politicians) | −.012 | [−.033, +.013] | .37 |
| Pooled | English-dev | −.033 | [−.098, +.030] | .28 |
| Pooled | Agora | −.001 | [−.015, +.015] | 1.00 |

## Tone diagnostic

Window-level Spearman correlation between channel projections and VADER negativity (label-free).

| Row | P (English) | E (English) | M (English) | E (Agora) | PE window score (English) |
|---|---:|---:|---:|---:|---:|
| V3 anchors | +.171 | +.337 | +.306 | +.344 | +.249 |
| V4 pairs | +.149 | +.284 | +.343 | +.310 | +.203 |
| Pooled | +.161 | +.315 | +.344 | +.333 | +.224 |

The anchors are matched on sentiment within each pair, yet the E axis built from them still tracks negative tone almost as strongly (+.28 vs +.34). In e5-large-v2, the populist claim and its hostile register lie in largely the same direction. Balancing the anchor sentences removes only a small part of that coupling, and balancing costs accuracy, consistent with the step 2 finding that the benchmark labels reward the tone.

## Anchor QA

Label-free. It informs no pruning.

| Channel | Pair parallelism | Cosine of pairs to axis (min / median / max) | Min leave-one-out axis cosine | Pairs pointing away from the axis |
|---|---:|---|---:|---|
| P | .157 | .12 / .43 / .61 | .997 | none |
| E | .181 | .20 / .45 / .63 | .998 | none |
| M | .208 | .25 / .46 / .67 | .999 | none |

- **Weakest pairs** (lowest cosine to their axis). P: P30 (protest), P12 (families), P10 (direct democracy). E: E37 (revolving door), E05 (salaries), E16 (regulators).
- **Cosine between the V3 and V4 axes.** P .75, E .68, M .78.
- **Channel correlations.** P·E is unchanged (.50). E·M rises from .56 (V3) to .71 (V4), so the V4 E and M channels are more redundant.

### Per-anchor attribution

`results/step3_pair_anchors/attribution_english_dev.json` lists, for the five highest-scoring development speeches, their top windows and the pairs that contribute most to each window's P and E projection. The decomposition is exact: the projection onto a mean-difference axis is the sum of per-pair terms. Example, South Africa_Zuma_Famous_2 (GPD .40), top window: "We need to use our political power to gain our economic power, and it is us nobody will do it for us, only us."

- P is driven by P06 ("Our people alone must decide our laws, not anyone in a distant capital.") and P38 ("My only master is the people…").
- E is driven by E04 ("They hand every contract to their cronies…") and E09 (establishment media).

Two of the five top speeches have high GPD scores (Duterte 1.15, Ahmadinejad 1.67, May 1.50). Two have low ones (Mbeki campaign 0.0, Zuma famous 0.40). The Mbeki window is about national self-determination against outside instruction, which the axis reads as populist.

## Decision

**Keep the V3 anchors (B1c).** Neither the V4 pairs nor the pooled set passes the adoption rule; the English-dev change is negative for both. The carried configuration is unchanged (`results/step3_pair_anchors/carried_config.json`). The approved V4 pairs are kept in the repository for the attribution tooling and for re-testing with a different encoder in step 5.

## Provenance

| Artifact | Commit | Dirty | Host |
|---|---|---|---|
| Pair-anchor encode (`v4_pairs_r1__24f70345`) | `b1d7e06` | no | pvgpu1 (encoded during step 2) |
| V4 pairs, pooled, B1c reference, diagnostics | `454f1bb` | no | local |

No GPU time was used in this step. CPU time was about 5 minutes.

```bash
uv run --extra cpu python scripts/run_step3_anchors.py
```
