# Anchor-revision pilot plan: revised P and E propositions on the development sets

Date: 2026-11-19. Status: nothing in this plan has been run from a clean tree. The pilot compares the frozen proposition anchors (`anchors/proposition_anchors.json`, V4) with three candidate sets on English-dev and Agora. It does not change the frozen configuration. The English holdout, EU-OOD and the external benchmarks are not scored. `scripts/pilot_anchor_revision.py` does not read EU-OOD or the external sets; the English benchmark and window files hold all 258 speeches, and the holdout rows are dropped on load. `tests/test_pilot_anchor_revision.py` fails if the script names a held-out or external path.

## Question

A co-author review of the forty P and E propositions rated each one for theoretical fit and proposed two kinds of revision. Green revisions sharpen eight populist propositions (P+02, P+05, P+06, P+07, P+09, E+04, E+06, E+09), seven rated conditional or extension and one, E+09, rated direct but marked for change, and replace one contrast proposition (E−00). Red revisions replace six contrast propositions with negation-framed commitments (P−04, P−07, P−08, E−04, E−07, E−08). The review document, "Populism by Proposition: the forty propositions", is kept by the authors outside the repository.

The pilot asks whether either kind of revision changes how well the axis and the combined score track the human scores on the development sets, and through which stage: the axis direction, the windows it selects, or the claim hypotheses derived from it.

## Arms

| Arm | Populist propositions | Contrast propositions | Revised slots |
|---|---|---|---:|
| V4 (reference) | frozen | frozen | 0 |
| `v5_green` | green | frozen, with E−00 green | 9 |
| `v5_red` | frozen | red | 6 |
| `v5_green_red` | green | red, with E−00 green | 15 |

The M anchors are those of V4 in every arm. `v5_green_red` is exactly `v5_green` plus `v5_red`, and no slot is revised by both; the test suite checks this. The two-by-two design lets each kind of revision be read with and without the other.

## Settled before the first run (2026-11-20)

1. **Wording.** Settled on 2026-11-20 using the co-author review document and label-free geometric diagnostics (cross-channel cosine, nuisance-direction loading, leave-one-out V4 projection and balance gates):
   - **P+02** (green): review doc option 2 — *"We must reclaim this country for its people."* (removes `"ruling elite"` from the P channel and keeps P+02 out of the sovereignty claim hypotheses).
   - **E+04** (green): review doc — *"The entire political class has betrayed the people it was elected to serve."*
   - **E+06** (green): review doc option 1 — *"The powerful enrich themselves while the rest of the country is told to sacrifice."*
   - **E+09** (green): retained candidate wording — *"Wealthy elites have bought the political influence that belongs to ordinary citizens."* (preserves the purchased-influence capture facet and avoids crisis/greed nuisance loading).
   - **E−04** (red): review doc — *"The regulators have not acted beyond the mandate and powers granted to them by law."*
   - **P+05, E−00, P−08**: retained first review-doc alternatives.
   - **E−07** (red): retained second review-doc alternative (*"The strain of recent years has not eroded the accountability of our institutions."*), chosen on 2026-11-19.

   The candidate files (`v5_green.json`, `v5_red.json`, `v5_green_red.json`) and their `rationale` fields have been updated accordingly.

2. **Primary arm.** `v5_green` (pre-declared before encoding and `prepare` on 2026-11-20). `v5_green` directly repairs the populist propositions rated *Conditional* or *Extension* in the construct review plus the `E−00` contrast defect without introducing negation-framed contrast anchors that dense encoders can misread; `v5_red` and `v5_green_red` remain part of the two-by-two design and are evaluated under the same grid rule.

3. **Earlier outputs.** The untracked outputs of the 2026-11-19 all-benchmark axis run (`subset_ids.json`, `arm_hypotheses.json`, `arm_window_map.parquet`, `cache/`, `embeddings/`, `nli_in/`, `prepare.manifest.json`, `stage1_scores.parquet`, `stage1_summary.json`) were archived outside the repository before the clean run, and that exploratory access was entered in `results/holdout_access_log.json`.

## What was seen before this plan

- **All-benchmark run.** The first version of the pilot script (commit `9e0201c`, run on 2026-11-19 from an uncommitted tree at `bb1ebdf`) scored `v5_green` and `v5_green_red` with the axis alone on English-dev, Agora, the English holdout, EU-OOD and the five external sets, and wrote paired differences against V4 for all of them. Its outputs were untracked and have been archived outside `results/anchor_revision_pilot/`; the access is recorded in `results/holdout_access_log.json`. The held-out and external differences are not used in this plan and do not inform the choice of wording, of the primary arm or of any arm.
- **Dry run of this script.** On 2026-11-19 `prepare` and `evaluate` were run from an uncommitted tree into a scratch directory, with the candidate files as they stand. This produced axis-only differences against V4 on English-dev and Agora for all three arms. `evaluate` ran on a file that combined the stored V4 entailment probabilities with random values for every other premise and hypothesis, so its candidate rows are not results; the Agora rows of the V4-hypothesis control are close to real, because 97% or more of each arm's Agora windows are V4's. None of these numbers is citable and none is recorded here.

The rule in the Readings section is fixed in this file so that the outcome follows from the development-set numbers of the clean run.

## Held fixed

e5-large-v2 and the stored window embeddings (`results/embeddings/e5-large-v2/spw3_cap60_clean`), three-sentence windows, the PE soft conjunction, the mean of the top 8 windows, BART-large-MNLI, the per-window claim rule (maximum over a channel's five hypotheses, √(P·E) for the conjunction), and the combined score (mean of within-corpus z-scores, weight .5). Each arm's claim hypotheses follow the frozen rule: the five populist propositions per channel with the largest cosine to the arm's own axis.

V4's anchor embeddings are the stored ones. Candidate anchors are encoded on the workstation CPU; sentences shared with V4 differ from the stored embeddings by at most 1.4e-4 per coordinate.

## Readings

1. **Combined score against B6.** Paired Δ Spearman of each arm's combined score against the committed B6 scores (`results/step6_nli/combined_k8`), 4,000 draws, seed 20260923, leaders resampled on English-dev and politicians on Agora. The adoption rule of `docs/development-protocol.md` is applied as written: English-dev Δ at least +.01, Agora Δ at least −.01, and neither 95% interval entirely below 0. Among arms that pass, the protocol's grid rule selects the one with the largest mean of the two differences; ties within .005 go to the simpler arm, read here as the one with fewer revised slots.
2. **Decomposition.** The same paired difference for the axis alone, for the claim check alone, and for a control in which the arm's windows are checked against V4's ten hypotheses. Axis-only isolates the direction. The control adds the change in selected windows. The arm's own hypotheses add the change in claims. A difference that appears only with the arm's own hypotheses comes from the hypothesis rotation. When an arm's ten hypotheses are V4's, as for `v5_red` with the candidate files of 2026-11-19, the control repeats the arm's own rows.
3. **Contrasts between arms.** `v5_green_red − v5_green` (red given green), `v5_green_red − v5_red` (green given red) and `v5_red − v5_green`, for the axis and the combined score.
4. **Label-free audit.** Cosine of each arm's P and E axes with V4's and with each other; for each revised sentence, its projection and the replaced sentence's projection on the V4 axis of its channel built without that slot, where a contrast sentence belongs on the negative side; Spearman of window scores with VADER negativity; the hypothesis ranking; the balance gates of `scripts/check_anchor_balance.py`. `v5_red` keeps V4's populist E sentences and inherits V4's failed they/them gate; all arms inherit V4's failed P gates.
5. **Claim-check diagnostics and spot checks.** Median, 99th percentile and shares above .5 of the window entailment scores per arm; the percentile under each arm of ten English-dev speeches that the coders scored at .5 or below, nine of them at .13 or below (Brown, Arroyo, Mbeki). Window text for the spot checks is written to `results/anchor_revision_pilot/disagreement/`, which is not tracked.

With 174 speeches from 57 leaders the English-dev interval of a paired difference is several hundredths wide, so the pilot can show a large change and cannot confirm a small one. Agora's intervals are about ±.01.

### What a passing arm means

Passing the rule makes an arm a candidate for a revision step under the development protocol. It is not adopted by this pilot. The held-out and external sets were scored under V4 and are spent, so a revised anchor set cannot be confirmed on them as V4 was; what evidence a revision would need is a separate decision for the authors.

## Checks in the script

- `prepare` stops unless V4, recomputed, gives the reported axis Spearman (.517 English-dev, .359 Agora) within .001, the committed step-4 document scores within 1e-8, the step-6 request windows row for row, and the step-6 claim hypotheses.
- A candidate's embeddings are located by the hash of its current JSON file, so an edited sentence cannot be scored with stale embeddings.
- V4's windows and hypotheses are part of the GPU request. `evaluate` stops unless the rerun reproduces the committed B6 combined Spearman (.537, .382) within .001. The rerun scores each distinct premise once; English-dev has 20 request rows with a repeated premise whose stored probabilities differ by up to 1.8e-4, so small score differences from the committed file are expected and are reported.

## Runs

Steps 1 to 6 follow the open decisions above and a commit and push of the settled candidate files, since cited results must come from a clean tree.

1. Workstation, encode the candidates:

   ```bash
   for arm in v5_green v5_red v5_green_red; do uv run --extra cpu python scripts/encode_sentences.py --encoder e5-large-v2 --sentences anchors/candidates/$arm.json --set-id $arm --out-root results/anchor_revision_pilot/embeddings --device cpu --repeat 2; done
   ```

2. Workstation, balance report:

   ```bash
   uv run --extra cpu python scripts/check_anchor_balance.py anchors/candidates/v5_green.json anchors/candidates/v5_red.json anchors/candidates/v5_green_red.json --out-md results/anchor_revision_pilot/balance_report.md --out-json results/anchor_revision_pilot/balance_report.json
   ```

3. Workstation, audit, axis scores and the request file:

   ```bash
   uv run --extra cpu python scripts/pilot_anchor_revision.py prepare
   ```

   Read `stage1_summary.json` for the hypothesis ranking before going on. With the candidate files of 2026-11-19, before the E−07 change, the request held 25,367 distinct premises and 15 hypotheses, 380,505 pairs; `prepare` prints the current count. BART-large-MNLI took 727 s for 226,680 pairs on the `pvgpu2` A30, which puts this job near 20 minutes.

4. Copy `results/anchor_revision_pilot/nli_in/` to the GPU checkout and compare the SHA-256 of `requests_dev_dedup.parquet` there with `nli_export.requests_sha256` in `stage1_summary.json`. Then on `pvgpu2`:

   ```bash
   scripts/remote_job.sh anchor_pilot_nli -- python scripts/nli_stage.py --in-dir results/anchor_revision_pilot/nli_in --out-dir results/anchor_revision_pilot/nli_out --names dev_dedup
   ```

5. Copy `results/anchor_revision_pilot/nli_out/` to the workstation through a staging directory (`docs/environment.md`), then:

   ```bash
   uv run --extra cpu python scripts/pilot_anchor_revision.py evaluate
   ```

6. Report in `research/development-log/anchor-revision-pilot.md`. Tracked evidence: `stage1_summary.json`, `results.json`, `REPORT.md`, the three manifests, the anchor-embedding manifests, `arm_hypotheses.json`, `nli_in/hypotheses.json`, the balance report and the job log. The request file, the window map, the score tables in parquet, the probability file and `disagreement/` stay untracked.
