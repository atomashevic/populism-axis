\name{define_scale}
\alias{define_scale}
\alias{read_scale_csv}
\alias{fit_reference}
\alias{fit_reference_csv}
\alias{score_texts}
\alias{score_csv}
\alias{make_classifier}
\alias{apply_classifier}
\alias{save_scale}
\alias{load_scale}
\alias{save_reference}
\alias{load_reference}
\alias{save_classifier}
\alias{load_classifier}
\alias{write_scores}
\title{Define Continuous Text Scales, Reuse References and Attach Threshold Labels}
\usage{
define_scale(name, positive = NULL, contrast = NULL,
  components = NULL, combine = NULL)
read_scale_csv(path, name, combine = NULL)
fit_reference(texts, scale, ids = NULL, unit = NULL, pooling = NULL,
  clean = NULL, claim_check = NULL, device = "cpu", threads = 4L,
  encoder_batch_size = NULL, nli_batch_size = 16L,
  local_files_only = TRUE)
fit_reference_csv(path, scale, text_column = "text", id_column = NULL, ...)
score_texts(texts, reference, scale = NULL, ids = NULL, unit = NULL,
  pooling = NULL, clean = NULL, claim_check = NULL, device = "cpu",
  threads = 4L, encoder_batch_size = NULL, nli_batch_size = 16L,
  local_files_only = TRUE)
score_csv(path, reference, scale = NULL, text_column = "text",
  id_column = NULL, ...)
make_classifier(reference, threshold, below, at_or_above, score = NULL)
apply_classifier(classifier, result)
save_scale(scale, path, overwrite = FALSE)
load_scale(path)
save_reference(reference, path, overwrite = FALSE)
load_reference(path)
save_classifier(classifier, path, overwrite = FALSE)
load_classifier(path)
write_scores(result, out_dir, overwrite = FALSE)
}
\arguments{
  \item{name}{A scale name beginning with a letter. Letters, digits and single underscores are supported; double underscores and reserved column names are prohibited.}
  \item{positive, contrast}{Character vectors of high-pole and contrasting propositions. Each requires at least one nonempty proposition. Factors are converted to their labels after missing-value checks.}
  \item{components}{An ordered named list. Each component contains exactly \code{positive} and \code{contrast} vectors. Do not mix this argument with the single-component arguments.}
  \item{combine}{For multiple components, explicitly choose \code{"mean"} or \code{"conjunction"}. Mean permits compensation; conjunction measures joint positive activation within a window.}
  \item{path}{A file path. Proposition CSVs have columns \code{component,pole,proposition}. Corpus CSVs are read literally by Python, preserving IDs such as \code{"001"} and \code{"NA"}.}
  \item{texts}{A nonempty character vector of documents. Length one is supported. Missing values, empty documents and numeric coercion are rejected.}
  \item{scale}{A scale object, or \code{"populism"}. Omitted scoring scale inherits a fitted reference, otherwise selects the preset.}
  \item{ids}{Optional unique nonempty character IDs, one per document. Factors are supported. Omitted IDs retain Python's zero-based string IDs.}
  \item{unit}{\code{"speech"}, \code{"paragraph"}, \code{"sentence"}, or \code{"tweet"}. \code{NULL} inherits a fitted reference or resolves to speech.}
  \item{pooling}{\code{"mean"} or \code{"top"}. \code{NULL} inherits a fitted reference, otherwise custom scales use mean and the preset uses top.}
  \item{clean, claim_check}{Logical flags. \code{NULL} inherits fitted settings. Custom entailment defaults off, while the preset defaults on.}
  \item{device}{Computation device supported by the Python engine. CPU is the default. Backend provisioning and computation-device selection are separate.}
  \item{threads, encoder_batch_size, nli_batch_size}{Positive whole-number resource options, explicitly converted to Python integers. \code{NULL} encoder batch size uses the engine default.}
  \item{local_files_only}{Use cached pinned models only. \code{FALSE} explicitly permits model downloads; this option does not govern backend environment provisioning.}
  \item{text_column, id_column}{CSV column names. An omitted ID column uses zero-based string row IDs.}
  \item{...}{For CSV adapters, named scoring options matching the text functions.}
  \item{reference}{A fitted reference object, \code{"frozen"}, or \code{"corpus"}. Custom reference fitting and corpus scoring require at least 30 valid documents.}
  \item{threshold}{One finite user-supplied numeric cutoff. No cutoff is estimated automatically.}
  \item{below, at_or_above}{Distinct nonempty labels. Equality receives \code{at_or_above}.}
  \item{score}{A numeric score field. Defaults to a custom scale's stable primary field; the preset requires an explicit field.}
  \item{classifier}{An immutable threshold-rule object bound to a fitted definition, reference and score field.}
  \item{result}{An immutable result snapshot.}
  \item{overwrite}{Explicitly replace the specified saved file or score outputs. Defaults to \code{FALSE}.}
  \item{out_dir}{Directory for \code{scores.csv}, \code{evidence.json} and \code{run_manifest.json}. All three paths are checked before writing. Unrelated files are preserved.}
}
\description{Calls the canonical Python engine through reticulate. Definitions, references, classifiers and results contain durable canonical JSON, preserving identities across R sessions without storing live Python pointers.}
\details{
Scoring does not create files. Use \code{write_scores()} to save outputs; evidence contains input passages. Custom scales do not inherit validation from the populism preset. A larger signed score indicates orientation toward positive propositions relative to the reference; it does not establish speaker endorsement or truth.

The scale-named custom column always equals its document-standardized axis score. Optional entailment adds separate support and combined fields. Combined evidence is neither a symmetric bipolar score nor a calibrated probability.

Fitted references contain their definition and effective settings. Explicitly conflicting settings fail. Corpus standardization is batch-dependent and cannot supply a reusable threshold rule. Threshold application does not run either model.

Saved Python JSON files retain the same format as Python's save/load methods. Native objects also support \code{saveRDS()} and \code{readRDS()}. Environment provisioning is described in \code{\link{setup_backend}}.
}
\value{Definition functions return a \code{populismaxis_scale}; fitting returns a \code{populismaxis_reference}; scoring and classification return a \code{populismaxis_result}; \code{make_classifier()} returns a \code{populismaxis_classifier}. Save/write functions invisibly return their input object.}
\seealso{\code{\link{scores}}, \code{\link{setup_backend}}, \code{\link{configure_python}}}
\examples{
\dontrun{
scale <- define_scale("institutional_trust",
  positive = "Public institutions act in the public interest.",
  contrast = "Public institutions routinely betray the public.")
print(scale)
# Fit on at least 30 varied, suitable reference documents.
reference <- fit_reference(reference_texts, scale)
result <- score_texts("Our institutions serve citizens fairly.", reference)
result$scores
rule <- make_classifier(reference, threshold = 0,
  below = "below_example_cutoff", at_or_above = "at_or_above_example_cutoff")
# Zero above is only an illustration, not a recommended cutoff.
labeled <- apply_classifier(rule, result)
}
}
