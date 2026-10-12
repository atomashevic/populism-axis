# Define and score a custom scale

Describe the high and low poles with propositions, fit a reference once, and reuse it to score new texts. You can then attach two labels using a threshold you supply.

This tutorial uses institutional trust as an illustration. Its propositions, reference texts, and threshold are invented examples. They do not constitute a validated measure or a representative reference population.

## Start from a proposition CSV

From the repository, install the CPU dependencies:

```bash
uv sync --extra cpu
uv run --extra cpu populism-axis template --out trust.csv
```

Edit `trust.csv` in a spreadsheet or text editor. Keep these columns:

```csv
component,pole,proposition
trust,positive,Public institutions act in the public interest.
trust,contrast,Public institutions routinely betray the public.
trust,positive,Public institutions respond fairly to citizens.
trust,contrast,Public institutions privilege powerful insiders.
```

`positive` describes the high pole. `contrast` describes the low pole. Both poles need at least one proposition. More propositions do not automatically improve validity. Write propositions that express the intended construct and evaluate the resulting scores on suitable development data.

Validate and save the definition without loading a model:

```bash
uv run --extra cpu populism-axis define \
  --input trust.csv --name institutional_trust --out trust.json
```

Names start with a letter and contain letters, digits, or single underscores. Double underscores separate output fields and are reserved. Duplicate propositions or overlapping poles within one component are rejected after Unicode and whitespace normalization.

## Fit a reusable reference

Prepare a CSV with `id,text` columns. Select reference texts appropriate to your intended population and text unit. The package requires at least 30 valid documents. This minimum permits fitting; it does not establish scientific adequacy.

For an executable illustration, use the 30 synthetic texts in [custom-scale-reference.csv](../examples/custom-scale-reference.csv):

```bash
uv run --extra cpu populism-axis fit-reference \
  --input examples/custom-scale-reference.csv --id-column id \
  --scale trust.json --unit sentence --out trust-reference.json
```

The default is offline model loading. If the pinned embedding model is absent, the command fails without downloading it. To permit the initial download, add `--allow-download`. The model can take substantial time and memory on a CPU. The custom scale uses no claim-check model by default.

The saved reference contains the definition, fitted statistics, and scoring settings. Reuse this file for subsequent datasets. Scores express position relative to this reference, so reference selection affects their interpretation. Reference windows receive equal weight when fitting window statistics; document scores receive equal weight when fitting document statistics.

## Score new texts

```bash
uv run --extra cpu populism-axis score \
  --input examples/custom-scale-texts.csv --id-column id \
  --reference trust-reference.json --out-dir trust-scores
```

The reference supplies the definition and settings. Explicit settings that disagree with it are rejected. The command writes `scores.csv`, `evidence.json`, and `run_manifest.json`. Existing outputs require explicit `--overwrite`. IDs such as `001` and `NA` remain literal strings.

The `institutional_trust` column is the continuous score. It equals `institutional_trust__axis_z`, the document axis score standardized against the saved reference. A larger value indicates stronger orientation toward the positive propositions. It is neither a probability nor a validated estimate of the author's attitude.

The default custom pooling is `mean`, which averages all window scores. For speech input, windows overlap, so this is not a sentence average. Explicit `--pooling top` selects the highest-scoring windows: up to eight for speech input, or one for sentence input. Fit a separate reference for a different pooling rule or unit.

## Add threshold labels

Choose a threshold using a substantive decision rule or separate validation data. Zero is used here only to demonstrate the API. It is not a recommended classifier threshold.

```bash
uv run --extra cpu populism-axis make-classifier \
  --reference trust-reference.json --threshold 0 \
  --below below_example_cutoff --at-or-above at_or_above_example_cutoff \
  --out trust-classifier.json

uv run --extra cpu populism-axis apply-classifier \
  --classifier trust-classifier.json --input-dir trust-scores \
  --out-dir trust-labeled
```

Applying labels requires no model inference. Values equal to the threshold receive the `at-or-above` label. The added column is `institutional_trust__label`; numerical scores are retained. The rule is bound to the saved reference, definition, and score field. Reusable classifiers require a fitted reference; corpus-standardized scores cannot supply one.

## The same workflow in Python

```python
from populism_axis import (
    FittedReference, define_scale, fit_reference_csv, make_classifier,
    score_texts, write_scores,
)

scale = define_scale(
    "institutional_trust",
    positive=[
        "Public institutions act in the public interest.",
        "Public institutions respond fairly to citizens.",
    ],
    contrast=[
        "Public institutions routinely betray the public.",
        "Public institutions privilege powerful insiders.",
    ],
)
print(scale.summary())
reference = fit_reference_csv(
    "examples/custom-scale-reference.csv", scale=scale,
    id_column="id", unit="sentence",
)
reference.save("python-trust-reference.json")
reference = FittedReference.load("python-trust-reference.json")
result = score_texts(
    ["Our public institutions serve citizens fairly."],
    ids=["001"], reference=reference,
)
print(result.scores[["id", "institutional_trust"]])
rule = make_classifier(
    reference, threshold=0,
    below="below_example_cutoff", at_or_above="at_or_above_example_cutoff",
)
labeled = rule.apply(result)
write_scores(labeled, "python-trust-scores")
```

Python also defaults to offline loading. Set `local_files_only=False` on fitting or scoring only when you want to permit model downloads.

## Multiple components and optional claim checking

For multiple components, supply an ordered mapping with `positive` and `contrast` propositions for each component. Specify `combine="mean"` to average signed component scores, or `combine="conjunction"` to require positive activation across components. These rules have different meanings; the package does not select one automatically.

Custom claim checking is optional. Enable it during reference fitting with `claim_check=True` in Python or `--claim-check` in the CLI. It evaluates entailment of positive propositions using the pinned natural language inference model. Under mean pooling it evaluates every window, which increases computation. Under top pooling it evaluates the overall selected windows. Contrast propositions define the axis but do not become low-pole entailment hypotheses.

The primary continuous score retains its meaning when claim checking is enabled. Additional `__claim_check`, `__claim_check_z`, and `__combined` fields report the optional stage. The combined field is the equal-weight mean of standardized axis and claim-check scores. Interpret it as high-pole evidence, not as a symmetric bipolar score or a calibrated probability.

The bundled populism preset retains its frozen scoring rules. To use it, pass `reference="frozen"` in Python or `--reference frozen` in the CLI. The package verifies its four bundled assets with `populism-axis verify-assets` without loading models.
