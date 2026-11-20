# APSI and Political DEBATE comparison

Declared 2026-10-17 before new benchmark inference. This comparison is authorized on all seven benchmarks, including previously accessed sets. It makes no configuration choice for the existing measure. Accesses and input hashes are recorded by preparation, inference and evaluation manifests. Existing manuscript and frozen runtime assets remain unchanged.

## Comparisons

1. Existing axis plus BART claim check, using retained scores.
2. The same pipeline with Political DEBATE large v1.0 replacing BART. The proposition anchors, axis-selected request windows, ten claim hypotheses, max_length=320 with only_first truncation, channel pooling and equal within-corpus z-score combination remain unchanged. Native binary entailment probabilities are used, with no artificial neutral category. GPU inference uses float16 weights and float32 softmax, as the original NLI runner.
3. APSI public implementation at `9a7fa79b858c5cce4357f0af57bb6aec8c4de97c`: large primary, base sensitivity. The exact hypothesis list and weights are in `anchors/apsi_comparator.json`, canonical hypothesis SHA-256 `6ce24cc7c0e36b98c870201c2fd937d47910292dea493b7c2caf7e5a50165d73`. This is a comparison with the public implementation, whose weighted adaptive aggregation and abstention gates differ from the working paper's equal-weight equations. It is not a replication of the reported 20-text validation.

Political DEBATE model revisions: large `1a3aff1ecb97ad93a6de598f7414b1707de7e7c3`; base `55ec647c8b708ad8dee650e3843577c26a7dc32e`. No training, benchmark-based selection, hypothesis editing or parameter tuning is permitted.

## APSI scoring and sensitivities

Native scoring uses full text with published tokenizer truncation=True (longest_first), max_length=512 and two-way float32 softmax entailment. To limit shared GPU time, float16 model weights may be used only after the synthetic-only numerical smoke check: maximum probability difference <= .01, identical abstentions and maximum score difference <= .02 versus float32. This computational check is fixed before benchmark inference; if it fails, use float32. Record precision and discrepancies. Apply all pinned topic and weak-evidence abstention gates, probability floor, weights and adaptive top-k arithmetic unchanged.

Chunk sensitivity uses nonoverlapping original-text spans split at tokenizer offsets. The common token budget is 512 minus the longest of all 48 hypotheses minus pair special tokens. Apply the same APSI scorer independently to every chunk. Average finite chunk scores equally, including the final shorter chunk; all-abstained documents remain NA. A one-chunk document reuses its native inference. Record total and scored chunks. Tokenizer normalization can change counts when a span is retokenized; unchanged native max512 truncation still applies.

APSI has one overall populism score. Use it unchanged against anti-elitism and people-centrism labels, explicitly identifying the construct mismatch. No component hypothesis subsets are introduced. Preserve abstentions and show coverage overall and by label. Compare metrics and paired differences on the same finite document subset; show the existing method's full-sample results separately.

## Evaluation

Seven benchmarks: GPD English (dev and holdout reported separately), Agora, EU-OOD, US governors, party tweets, PopBERT, and Portugal. Retain original evaluation samples, corpus standardization scopes, metrics and resampling units used for manuscript Table 3. Use 4,000 paired bootstrap replicates with the existing seed where applicable. Report the number of valid bootstrap draws. Do not compare separate marginal confidence intervals as a test of difference.

For EU three-class macro-F1, retain the original leave-one-speaker-out component threshold procedure. APSI's one score supplies both component threshold fits, with separate targets. This is an explicitly declared evaluation adapter and does not produce distinct APSI component measurements. On APSI covered subsets, both methods' thresholds must be fitted within the same covered training folds. New EU intervals, absent from Table 3, are conditional speaker-bootstrap intervals of the fixed out-of-fold predictions. The threshold procedure is unchanged and the held-out speaker never sets its own threshold.

Exact normalized text overlap against the public PolNLI training data will be audited. Absence of exact matches does not establish absence of source, paraphrase or earlier pretraining exposure.

## Execution and evidence

Use pvgpu2 only for GPU inference, one GPU job at a time, two CPU threads for inference where possible, with resource checks through scripts/remote_job.sh. Preserve its older checkout and jobs. Run from an isolated clean snapshot of the approved code and this protocol; no branch push is required. Keep raw texts, entailment caches and large artifacts ignored. Retain compact text-free score tables, manifests and job summaries under results/apsi_comparison. Numerical tests and independent implementation review precede full runs; verify frozen assets before reporting.
