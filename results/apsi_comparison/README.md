# APSI comparison run

Status: Completed. All 126 paired comparisons, 4,000-draw intervals, numeric scores and the standalone SVG have been collected and verified. See comparisons.csv and paired_comparisons.svg.

The existing axis+BART measure is compared with the unchanged pipeline using Political DEBATE large, and the pinned public APSI implementation (large primary, base sensitivity; native truncation and chunk averaging).

## Verified before launch

- 117 tests passed; all frozen assets and 23 source/dependency files match.
- Every Table 3 point estimate reproduced; BART score reconstruction discrepancy at most 3.55e-15.
- APSI arithmetic and abstentions matched the upstream class on 1,004 synthetic/boundary cases.
- No exact normalized PolNLI training matches among 28,061 prepared documents or 52,265 selected-window requests. This does not rule out near-duplicates, translations or underlying model pretraining exposure.
- APSI uses float32 because the synthetic float16 check exceeded the prespecified maximum probability discrepancy for both models. See precision/precision_check.json.

## Execution

Remote checkout: `/home/atomasevic/populism-axis-apsi-20261017`.
Inference commit: `604c9b750f110d13adefa03b5865b40a5bf7bfa2`.
Plot/evaluation follow-up commit: `6ebf2b4` (only adds the plotting script).
The manuscript branch has no new commits or pushes. A clean inference-source bundle is retained in ignored `cache/run_source.bundle`.

One GPU job at a time, batch size 32, two CPU threads, nice priority 10:

1. Political DEBATE large on the existing selected passages.
2. APSI large, native and chunk modes.
3. APSI base, native and chunk modes.

A separate waiting process starts the 4,000-draw paired evaluation and SVG generation after successful inference. It stops if inference fails. Logs on pvgpu2: `logs/apsi_comparison_20261017.log` and `logs/apsi_followup_20261017.log`. The completion marker is `COMPARISON_COMPLETE` in the follow-up log.

Expected final outputs: `comparisons.csv`, `comparisons.json`, numeric `scores.csv`, `paired_comparisons.svg`, and manifests in this directory. Full probability caches remain ignored. APSI component evaluations use its overall score, with coverage and matched-sample intervals. EU macro-F1 intervals are new and conditional on fixed out-of-fold predictions.

Protocol: `research/development-log/apsi-comparison-plan.md`.

Automatic local collection runs as user service `populism-apsi-collect-20261017`. It retrieves outputs after the remote completion marker, checks manifests, the 126-comparison grid and SVG, and updates this status. The laptop must be awake and connected for collection; remote computation continues independently.
