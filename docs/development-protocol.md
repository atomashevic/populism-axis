# Development protocol

The development protocol changed the initial proposition-anchor axis one step at a time and assesses each
step before the next begins. The plan is fixed in advance so that each step's
effect is isolated and no decision is taken on held-out data.

## Step order

| Step | Change | GPU work |
|---|---|---|
| 0 | Environment, provenance, holdout freeze, harness | smoke encode |
| 1 | Sentence-splitter cap and re-baseline | encode windows twice |
| 1b | Speech-text cleaning (metadata headers, URL lines, bundled Chinese text); added at the user's request | encode English windows |
| 2 | Label-free whitening and nuisance residualization | encode nuisance sentences |
| 3 | Minimal-pair anchor set (user review before encoding) | encode anchors |
| 4 | Window size and top-k sweep, pervasiveness readout | encode 2- and 3-sentence windows |
| 5 | Encoder comparison with corrected instruction templates | encode per encoder |
| 6 | NLI entailment second stage on the top-k windows | NLI on top windows |
| Final | English holdout and EU-OOD, scored once | encode EU |

## Data access

- Development: English GPD rows with `split == dev` in `data/benchmarks/english_split.csv` (174 speeches, 24 countries) and all of Agora.
- Held out: English GPD rows with `split == holdout` (84 speeches, 14 countries) and all of EU-OOD.
- Held-out data are scored once, after step 6, through the `--final` flag. Every access is appended to `results/holdout_access_log.json`.
- One exception is declared in advance: in step 1 the initial configuration is scored once on all 258 English speeches to confirm that the harness reproduces its previously reported Spearman (.433). This reproduces an already published number and informs no decision.
- The English holdout was visible to the initial configuration's choices. It is unseen by the stepwise decisions only.

## Carried system

PE channels, soft conjunction, frozen signs, top-3 windows, until a step changes one of these. PEM and single channels are always reported and never decide.

## Adoption rule

After each step, the candidate configuration is compared with the previous step's carried configuration by a paired bootstrap of the Spearman difference (4,000 draws, seed 20260923): resampling leaders on English-dev and politicians on Agora.

- Adopt if the English-dev difference is at least +.01, the Agora difference is at least −.01, and neither 95% interval lies entirely below 0.
- Otherwise keep the previous configuration.
- Defect fixes are adopted regardless of the numbers, and their effect is reported: the splitter cap (step 1), the speech-text cleaning (step 1b) and the Qwen3 instruction template (step 5).
- Grid steps (2, 4, 5) pre-declare a primary cell and publish the whole grid. The selected cell maximizes the mean of the two differences among cells that pass the rule. Ties within .005 go to the simpler configuration: fewer removed components, no residualization, one-sentence windows, k = 3, the smaller encoder.

With 174 development speeches from 57 leaders, a Spearman interval is roughly ±.1 wide. Interval exclusion of zero is therefore reported as "significant" but not required for adoption.

## Step report

`research/development-log/stage-<N>-<slug>.md`, headings in this order: Purpose, Change, Config, Results on development sets, Paired difference vs previous step, Decision, Provenance. Tables in Results, Paired difference and Provenance are generated from JSON by `scripts/render_step_report.py`.

## Provenance

Every script writes `run_manifest.json` (`src/populism_axis/run_manifest.py`) with the git commit and dirty flag, command, host, library versions, input file SHA-256 and a run hash. Results cited in a report must come from a clean working tree.
