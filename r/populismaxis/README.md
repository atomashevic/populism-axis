# populismaxis

Define a continuous text scale with propositions, fit a reusable reference, and optionally assign labels using an explicit threshold. The R package calls the same Python engine as the Python API.

## Install from this repository

The package includes its Python wheel and locked dependency profiles. In R, install the small R dependencies if needed:

```r
install.packages(c("reticulate", "jsonlite", "digest"))
```

From the repository shell:

```sh
R CMD INSTALL r/populismaxis
```

No manual Python installation or executable path is required. The first backend call lets reticulate provision a cached Python 3.12 environment through uv. Loading the R package alone downloads nothing. First setup may take several minutes; model files require several GB when the optional claim check is included.

## Define a scale

This institutional-trust example illustrates the interface. Its propositions and threshold are not a validated instrument.

```r
library(populismaxis)

# Optional preparation before analysis. TRUE explicitly permits model downloads.
setup_backend(download_models = TRUE)

scale <- define_scale(
  "institutional_trust",
  positive = c(
    "Public institutions act in the public interest.",
    "Public institutions respond fairly to citizens."
  ),
  contrast = c(
    "Public institutions routinely betray the public.",
    "Public institutions privilege powerful insiders."
  )
)
summary(scale)

# Use at least 30 relevant reference documents. The repository contains a
# synthetic illustration; replace it with a suitable reference for research.
reference <- fit_reference_csv(
  "examples/custom-scale-reference.csv", scale = scale,
  id_column = "id", unit = "sentence"
)
save_reference(reference, "trust-reference.json")
reference <- load_reference("trust-reference.json")

result <- score_texts(
  "Our public institutions serve citizens fairly.",
  reference = reference, ids = "001"
)
result$scores

# Zero is an illustrative cutoff, not a recommended classification threshold.
rule <- make_classifier(
  reference, threshold = 0,
  below = "below_example_cutoff",
  at_or_above = "at_or_above_example_cutoff"
)
labeled <- apply_classifier(rule, result)
write_scores(labeled, "trust-scores")
```

The primary score is relative to the saved reference. Custom scales default to mean pooling and no claim check. Use `reference = "frozen"` for the unchanged populism preset. A fitted reference supplies its scoring settings; conflicting settings produce an error.

You can also write a `component,pole,proposition` CSV and call `read_scale_csv(path, name)`. The `positive` pole defines higher scores; `contrast` defines lower scores. Multiple components require an explicit `combine = "mean"` or `combine = "conjunction"` rule.

## Save and inspect results

Scale, reference and classifier JSON files work in Python and R. R objects also support `saveRDS()` and `readRDS()` across sessions. The objects retain canonical Python JSON rather than live Python pointers.

`result$scores` is an ordinary R data frame. `result$evidence` and `result$provenance` expose inspection values. Results are read-only snapshots; copy the data frame before modifying it for analysis. Threshold application creates a new result. Python's zero-based window indices and generated document IDs are retained. Character IDs such as `"001"` and `"NA"` remain literal; missing values are rejected. Output files use UTF-8. When reading scores with base R, use `read.csv(path, encoding = "UTF-8", colClasses = c(id = "character"), na.strings = NULL)` to preserve Unicode and literal IDs across system locales.

## Prepared environments, GPU and offline use

To use an existing environment, set `POPULISM_AXIS_PYTHON` before loading the package or call `configure_python(path)` before any backend operation. The package verifies the matching engine and dependency versions without installing into that environment. Conflicting selections produce a restart instruction.

CPU is the default. In a fresh R session, `setup_backend(gpu = TRUE)` selects the pinned CUDA 12.6 dependencies; pass `device = "cuda"` when fitting or scoring. CUDA requires a compatible NVIDIA driver. No automatic torch downgrade is performed on unsupported systems.

Set cache locations before Python starts:

```r
Sys.setenv(
  UV_CACHE_DIR = "/path/to/cache/uv",
  UV_PYTHON_INSTALL_DIR = "/path/to/cache/python",
  HF_HOME = "/path/to/cache/huggingface"
)
```

After preparing dependencies and model files, use `UV_OFFLINE=1` and `HF_HUB_OFFLINE=1` for a fully offline session. `local_files_only=TRUE`, the scoring default, prevents model downloads but does not prevent uv from provisioning missing dependencies.

## Developer verification

The Python source under `src/populism_axis` is authoritative. Rebuild the bundled wheel after changing it:

```sh
uv run --extra cpu python scripts/build_r_backend.py
R CMD build r/populismaxis
POPULISM_AXIS_TEST_PYTHON="$PWD/.venv/bin/python" \
  RETICULATE_PYTHON="$PWD/.venv/bin/python" \
  UV_OFFLINE=1 HF_HUB_OFFLINE=1 \
  R CMD check --no-manual populismaxis_0.1.0.tar.gz
```

The builder verifies frozen assets, exports the existing lockfile without resolving it, and creates a deterministic wheel in a directory named by its SHA256 prefix. It never downloads models. The bundle records its source hashes and lockfile identity. Published artifacts must not change under an existing version.

The R tests use deterministic model substitutes. Set `POPULISM_AXIS_TEST_PYTHON` to a matching prepared environment to run the API and equivalence tests; without it, those tests skip rather than provision dependencies. The equivalence matrix covers 36 custom and preset configurations, complete score tables, evidence, provenance, reference identities and threshold labels.

The environment lifecycle follows [transforEmotion's uv backend](https://github.com/atomashevic/transforEmotion/blob/ff51d1cf26d9ce0968f7d2b483e5f365eee0f501/R/reticulate_env.R), using an independent implementation and this project's own dependency profile.
