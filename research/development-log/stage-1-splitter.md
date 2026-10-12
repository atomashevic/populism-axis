# Stage 1: sentence-splitter cap and re-baseline

Date: 2026-09-23. Code commits `c4fef72` and `ff9af06`.

## Purpose

V3 split documents into windows only at `.`, `!` or `?` followed by whitespace, then truncated every window at 96 encoder tokens. A speech without sentence punctuation became a single window, and only its first ~70 words were scored. This step caps windows at 60 words and re-measures the V3 configuration with and without the cap on the V4 development data. The result is the baseline for every later step.

## Change

- `split_windows(..., max_words=None)` in `src/populism_axis/contextual_axis.py`. `max_words=None` reproduces V3 exactly. With a cap, any sentence longer than the cap is split by `_chunk_long_window`: clauses at `;`, `:`, `,` or line breaks are packed greedily up to 60 words, and a clause still over the cap is cut into 60-word spans. Shorter sentences are left unchanged, and the words of every document are preserved in order.
- `scripts/encode_windows.py` (pvgpu1) stores raw fp16 embeddings, window text, word counts and untruncated token counts, so later steps need no re-encoding.
- CPU harness:
  - `src/populism_axis/v4_geometry.py`: the V3 axis and centering rules on stored embeddings.
  - `scripts/run_config.py`: build, project, aggregate and evaluate one configuration. It includes the holdout lock and access log.
  - `scripts/compare_steps.py`: paired comparison and the adoption rule.
  - `scripts/render_step_report.py`: generates the tables below.
- Tests: `tests/test_v4_step1.py` (13 tests). All 23 V4 tests pass.

## Config

| | B0 (V3 splitter) | B1 (cap) |
|---|---|---|
| Encoder | `intfloat/e5-large-v2`, `query: ` prefix, mean pooling, fp16, 96 tokens | same |
| Anchors | `anchors/proposition_anchors.json` (SHA-256 `ea98e5b2…`) | same |
| Axis rule | V3: normalize(mean populist − mean mainstream), anchors not centered | same |
| Window transform | center on the scored documents' window mean, L2-normalize | same |
| Windows | one sentence, no cap | one sentence, cap 60 words |
| Aggregation | PE, soft conjunction, frozen signs, top-3 windows | same |
| English data | development split, 174 speeches (57 leaders) | same |

## Harness check: V3 reproduction

As declared in `docs/development-protocol.md`, B0 was scored once on all 258 English speeches, with the access logged. Every system matches V3's committed table within .0006. The storage format and the torch upgrade from 2.11 to 2.14 therefore change nothing material.

| System | V3 English ρ | V4 English ρ | V3 Agora ρ | V4 Agora ρ |
|---|---:|---:|---:|---:|
| PE soft (V3 headline) | .4330 | .4328 | .3574 | .3574 |
| P soft | .4516 | .4518 | .3393 | .3392 |
| E soft | .3868 | .3873 | .3291 | .3292 |
| PEM soft | .4269 | .4270 | .3461 | .3461 |
| PE hard | .4330 | .4329 | .3502 | .3502 |

## Results on development sets

System `PE__frozen__soft`. CIs are 95%, leader-clustered for English and politician-clustered for Agora.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B0 V3 splitter | .462 | [.309, .602] | .939 | .369 | .357 | [.260, .378] | .891 |
| B1 cap 60 words | .483 | [.339, .618] | .940 | .359 | .359 | [.266, .378] | .889 |

English-dev ρ is higher than V3's full-set .433 for both configurations. This reflects the composition of the development split, not a method change. Only differences within the development split are informative.

All systems for B1 (diagnostic only):

| System | English ρ | Agora ρ |
|---|---:|---:|
| P soft | .493 | .341 |
| PE soft (carried) | .483 | .359 |
| PE hard | .482 | .352 |
| PEM soft | .472 | .347 |
| PM soft | .469 | .348 |
| E soft | .453 | .331 |
| EM soft | .430 | .320 |
| M soft | .397 | .296 |

As in V3, P alone ranks English speeches slightly better than PE, while PE is better on Agora. The carried system stays PE as pre-declared.

### Window diagnostics

| | English, V3 splitter | English, cap 60 | Agora, V3 splitter | Agora, cap 60 |
|---|---:|---:|---:|---:|
| Windows | 33,549 | 34,237 | 41,308 | 41,912 |
| Longest window (words) | 1,210 | 60 | 185 | 60 |
| Windows over 96 tokens | 352 | 197 | 177 | 0 |
| Tokens lost to truncation | 4.2% | 2.5% | 0.36% | 0 |

The cap eliminates truncation on Agora and on every English speech except those described under data-quality findings.

Development speeches most affected by the cap. The last two columns give each speech's percentile rank among the 174 development speeches under the carried score.

| Speech | GPD score | Words in windows over 60 words | Windows before → after | Rank B0 | Rank B1 |
|---|---:|---:|---|---:|---:|
| UK_Brown_Famous_1 | .10 | 100% | 1 → 10 | .13 | .43 |
| Canada_Harper_Ribbon_2 | .25 | 100% | 1 → 21 | .03 | .24 |
| Canada_Harper_Campaign_2 | .30 | 100% | 1 → 10 | .08 | .19 |
| Canada_Trudeau_Ribbon_1 | .00 | 88% | 4 → 19 | .02 | .11 |
| Canada_Chretien_Famous_3 | .10 | 83% | 35 → 71 | .16 | .70 |
| Australia_Howard_Ribbon_3 | .00 | 71% | 26 → 39 | .06 | .05 |
| Ireland_Kenny_Famous_1 | .40 | 66% | 31 → 45 | .53 | .55 |
| Ireland_Cowen_Famous_1 | .00 | 63% | 58 → 93 | .12 | .20 |
| UK_Brown_Campaign_1 | .00 | 60% | 138 → 202 | .96 | .96 |

Speeches that were a single truncated window sat near the bottom of the ranking only because most of their text was never scored. With the cap they move toward the middle. Across all development speeches, the B0 and B1 rankings correlate at .98. Full list: `results/step1_splitter/affected_speeches.csv`.

## Paired difference vs previous step

B1 minus B0, system `PE__frozen__soft`, 4,000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| English-dev (leader-clustered) | +.021 | [−.003, +.055] | .10 | 174 | 57 |
| Agora (politician-clustered) | +.002 | [−.001, +.005] | .25 | 5,272 | 6 |

Agora has only six politicians, so its clustered interval rests on six clusters and understates uncertainty. The same caveat applies to every later step.

## Decision

**Adopt B1** as a defect fix, as `docs/development-protocol.md` pre-declares. B1 would also pass the standard adoption rule: English-dev Δ is at least +.01, Agora Δ is at least −.01, and neither interval lies entirely below zero. The carried configuration is `results/step1_splitter/carried_config.json` (e5-large-v2, V3 anchors, one-sentence windows capped at 60 words, V3 transform, PE soft top-3).

## Data-quality findings (not changed in this step)

- **Bilingual speech files.** Fourteen English speeches contain Chinese passages: ten Taiwan speeches, all in the development split, and four Singapore speeches in the holdout. These produce 1–9 CJK windows per speech, which the English encoder cannot read. They account for most of the English tokens still lost to truncation. Only one such window reaches a development speech's top 3 (Taiwan_Ying-jeou_Famous_2, GPD score 0).
- **URL headers.** Three Estonia speeches start with long URL strings, which cause the remaining non-CJK truncation.
- **Metadata headers.** 37 speech files start with a metadata header such as "Name of politician: … Title of Speech: …". Across development speeches, 11 windows contain header fields. They lean slightly toward the P pole (mean z .65) but reach no development speech's top 3.

Removing CJK passages and header lines is a benchmark text change, so it would alter V3 comparability and the frozen dataset hashes. It is not part of this step.

## Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| Anchor encode (`v3_proposition__ea98e5b2`) | `c4fef72` | no | see manifest | pvgpu1 |
| Windows, V3 splitter (`spw1_cap0`) | `c4fef72` | no | see manifest | pvgpu1 |
| Windows, cap 60 (`spw1_cap60`) | `ff9af06` | no | see manifest | pvgpu1 |
| B0 V3 splitter (dev) | `ff9af06` | no | `86aeb112e5b3` | local |
| B1 cap 60 (dev) | `ff9af06` | no | `4879864ba631` | local |
| Paired comparison | `ff9af06` | no | `c7c6e1c73846` | local |
| B0 on all 258 (reproduction) | `ff9af06` | no | see manifest | local |

- The V3-splitter windows were encoded at `c4fef72`. That commit differs from `ff9af06` only in how capped runs treat short sentences, so the uncapped path is identical.
- The cap-60 windows were first encoded at `c4fef72`, which collapsed line breaks inside short sentences to spaces. They were re-encoded at `ff9af06` so that the stored text matches committed code. Only the `ff9af06` encoding is used.
- GPU use: about 4.5 minutes in total over four short jobs. Peak VRAM was 794 MB. Each job exited and the GPU returned to its 651 MiB idle baseline. One launch was refused by the load guard (1-minute load 2.53) and succeeded on retry.
- Encode manifests and window statistics: `results/step1_splitter/encode_manifests/`. Holdout access log: `results/holdout_access_log.json` has one entry, the declared reproduction check.

Commands:

```bash
# pvgpu1
scripts/remote_job.sh step1_encode -- bash -c "python scripts/encode_sentences.py --encoder e5-large-v2 --sentences anchors/proposition_anchors.json --set-id v3_proposition && python scripts/encode_windows.py --encoder e5-large-v2 --datasets english,agora --max-words 0 && ..."
scripts/remote_job.sh step1_encode_cap60 -- python scripts/encode_windows.py --encoder e5-large-v2 --datasets english,agora --max-words 60
# local
uv run --extra cpu python scripts/run_config.py --label B0_v3_repro_all258 --embeddings-dir results/embeddings/e5-large-v2/spw1_cap0 --anchors results/embeddings/e5-large-v2/sentences/v3_proposition__ea98e5b2.npz --out-dir results/step1_splitter/B0_v3_repro_all258 --english-split all --access-reason "declared V3 reproduction check"
uv run --extra cpu python scripts/run_config.py --label B0_v3_splitter --embeddings-dir results/embeddings/e5-large-v2/spw1_cap0 --anchors ... --out-dir results/step1_splitter/B0_v3_splitter
uv run --extra cpu python scripts/run_config.py --label B1_cap60 --embeddings-dir results/embeddings/e5-large-v2/spw1_cap60 --anchors ... --out-dir results/step1_splitter/B1_cap60
uv run --extra cpu python scripts/compare_steps.py --baseline-dir results/step1_splitter/B0_v3_splitter --candidate-dir results/step1_splitter/B1_cap60 --rule defect_fix --out results/step1_splitter/paired_vs_previous.json
```
