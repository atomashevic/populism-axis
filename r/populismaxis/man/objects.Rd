\name{scores}
\alias{scores}
\alias{evidence}
\alias{provenance}
\alias{print.populismaxis_object}
\alias{summary.populismaxis_object}
\alias{populismaxis_scale}
\alias{populismaxis_reference}
\alias{populismaxis_classifier}
\alias{populismaxis_result}
\title{Inspect Immutable Durable Scale Objects and Score Results}
\usage{
scores(result)
evidence(result)
provenance(result)
\method{print}{populismaxis_object}(x, ...)
\method{summary}{populismaxis_object}(object, ...)
}
\arguments{
  \item{result}{A result returned by scoring or threshold application.}
  \item{x, object}{A scale, reference, threshold rule or result object.}
  \item{...}{Reserved for method compatibility.}
}
\description{Returns pure R inspection values from immutable canonical JSON snapshots. Inspection requires no Python interpreter.}
\details{
\code{result$scores}, \code{result$evidence}, and \code{result$provenance} are shortcuts for these accessors. \code{object$identity} returns its exact identity string. \code{object$json} exposes its canonical payload for inspection.

Objects contain JSON and identity strings, with no live Python pointers. Normal replacement operations fail explicitly. Extracted views are independent: editing a data frame from \code{scores(result)} does not change the result used by writing or classification. Create a new object through the public API to change a definition, reference or scoring configuration.

Scores preserve column names, input ordering, Unicode and literal string IDs. The returned frame carries no pandas index attribute. Numeric scores use R doubles; exact Python float64 payloads remain authoritative for save/write and threshold application.

Evidence arrays remain lists, including singleton arrays. Window indices remain zero-based. Inspection counts beyond R's integer range use doubles when exactly representable; larger JSON integers remain decimal strings. No silent integer wrapping occurs.

Custom mean-pooling evidence contains all contributing windows. Top-pooling evidence follows overall selection; separately retrieved component diagnostic indices identify their additional contributing windows. The frozen preset preserves its historical evidence schema.
}
\value{\code{scores()} returns a data frame; \code{evidence()} and \code{provenance()} return nested R lists. \code{summary()} returns a list. \code{print()} invisibly returns the object.}
\seealso{\code{\link{score_texts}}, \code{\link{apply_classifier}}, \code{\link{write_scores}}}
