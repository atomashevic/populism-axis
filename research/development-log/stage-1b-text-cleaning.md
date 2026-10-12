# Stage 1b: speech-text cleaning

Date: 2026-09-23. Code commits `7331bba`, `be0cd6c`, `6f6acfe`.

## Purpose

Step 1 found that some English GPD speech files contain text that is not part of the speech. Ten Taiwan files bundle the original Chinese text, and many files start or end with metadata ("Name of politician: …", "Title of Speech: …", "Source: http…"). At the user's request, this step removes both before windowing and re-measures the carried configuration. The step was not in the original plan and was added between steps 1 and 2.

## Change

- `src/populism_axis/v4_text.py`, `clean_speech_text` (version `v4_clean_r1`). It is applied at load time, and the raw files are not modified. It is label-free and applies the same rule to development and holdout files. It removes:
  - lines whose non-space characters are at least 20% CJK ideographs, and stray ideographs in other lines;
  - metadata lines with a known key (`Name of politician`, `Title of Speech`, `Date`, `Location`, `Source`, `Category`, …) within the first 12 and last 8 non-empty lines of a file;
  - a stricter set of unambiguous metadata keys anywhere, on lines of at most 200 characters, and `Source`/`Link` lines that contain a URL at any length;
  - lines that consist only of a URL, and the byte-order mark.
- Speaker labels in transcripts ("THE PRESIDENT:", "PRIME MINISTER:"), crowd lines ("AUDIENCE: Yes we can!") and prose with a colon are kept. Every ambiguous-key removal longer than 45 characters was checked by hand, and all were metadata once `audience` was dropped from the key list.
- `scripts/audit_text_cleaning.py` writes the audit. `scripts/encode_windows.py` gains `--clean-text`. `scripts/run_config.py` gains per-dataset embedding directories. Agora text has none of these problems, so it keeps its step 1 embeddings.
- `scripts/compare_steps.py` now caps the paired bootstrap p-value at 1. The V3 formula, 2 × min(share of draws ≤ 0, share ≥ 0), counts draws with a difference of exactly 0 in both tails and gave values above 1 here. The step 1 p-values (.10 and .25) are unaffected.
- `tests/test_v4_step1.py` adds five cleaning tests. All 28 V4 tests pass.

## Config

The carried configuration from step 1 is unchanged except for English text cleaning: e5-large-v2, V3 anchors, one-sentence windows capped at 60 words, V3 transform, and PE soft top-3.

## Cleaning audit

| | Count |
|---|---:|
| Speeches changed | 147 of 258 (95 dev, 52 holdout) |
| Words removed | 2,813 of 647,755 (0.43%) |
| Metadata lines removed | 425 |
| URL-only lines removed | 75 |
| Chinese lines removed | 237 (22,809 characters) in 10 Taiwan files and 1 Singapore file |
| Stray ideographs stripped | 59 |
| English windows | 34,237 → 34,173 |
| Windows over 96 tokens | 197 → 126 |

Step 1 attributed four Singapore holdout files to Chinese text. Three of them contain only full-width punctuation inside English lines; the looser character range used in step 1 overcounted them. Every removed metadata and URL line is listed in `results/step1b_text_cleaning/removed_lines.csv`, and per-speech counts with cleaned-text hashes are in `cleaning_audit.json`.

## Results on development sets

System `PE__frozen__soft`. CIs are 95%, leader-clustered for English and politician-clustered for Agora.

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---:|
| B1 cap 60 | .4828 | [.339, .618] | .940 | .359 | .359 | .889 |
| B1c cap 60 + cleaning | .4826 | [.339, .618] | .940 | .359 | .359 | .889 |

The other systems move by at most .002 (P soft .493 → .495, PEM soft .472 → .471).

## Paired difference vs previous step

B1c minus B1, 4,000 paired cluster-bootstrap draws, seed 20260923.

| Dataset | Δρ | 95% CI | p | Documents | Clusters |
|---|---:|---|---:|---:|---:|
| English-dev (leader-clustered) | −.0002 | [−.003, +.001] | 1.00 | 174 | 57 |
| Agora (politician-clustered) | .0000 | [.000, .000] | 1.00 | 5,272 | 6 |

No development speech moved by as much as .02 in percentile rank. The largest shift was −.012, for Austria_Gusenbauer_International_1, where five metadata lines were removed. The Taiwan speech whose top 3 contained a Chinese window in step 1 (Taiwan_Ying-jeou_Famous_2, GPD score 0) moved from the 6.3rd to the 5.7th percentile. Per-speech shifts: `rank_shifts_dev.csv`.

## Decision

**Adopt B1c** as a data-quality fix, as the user directed. It does not meet the standard rule's +.01 threshold, and it has no measurable effect on ranking. It is carried forward because the scored text should be the speech alone, and later steps add encoders and anchors that may read Chinese or metadata differently from e5-large-v2. The carried configuration is `results/step1b_text_cleaning/carried_config.json`.

This cleaning is a benchmark text change. V4 English numbers from here on are computed on cleaned text, while V3's were not; the step 1 V3 reproduction remains on raw text.

## Provenance

| Artifact | Commit | Dirty | Run hash | Host |
|---|---|---|---|---|
| Cleaning audit | `7331bba` | see manifest | see manifest | local |
| English windows, cleaned (`spw1_cap60_clean`) | `7331bba` | no | see manifest | pvgpu1 |
| B1c cap 60 + cleaning (dev) | `be0cd6c` | no | `1ec489c0c163` | local |
| Paired comparison | `6f6acfe` | no | see manifest | local |

GPU use: one job, 65 s, 793 MB peak. The GPU returned to its idle baseline afterwards.

```bash
uv run --extra cpu python scripts/audit_text_cleaning.py
# pvgpu1
scripts/remote_job.sh step1b_encode_clean -- python scripts/encode_windows.py --encoder e5-large-v2 --datasets english --max-words 60 --clean-text
# local
uv run --extra cpu python scripts/run_config.py --label B1c_cap60_clean --embeddings-dir results/embeddings/e5-large-v2/spw1_cap60_clean --dataset-dir agora=results/embeddings/e5-large-v2/spw1_cap60 --anchors results/embeddings/e5-large-v2/sentences/v3_proposition__ea98e5b2.npz --out-dir results/step1b_text_cleaning/B1c_cap60_clean
uv run --extra cpu python scripts/compare_steps.py --baseline-dir results/step1_splitter/B1_cap60 --candidate-dir results/step1b_text_cleaning/B1c_cap60_clean --rule defect_fix --out results/step1b_text_cleaning/paired_vs_previous.json
```
