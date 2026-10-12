.pa_json <- function(text) {
  jsonlite::fromJSON(text, simplifyVector = FALSE, bigint_as_char = TRUE)
}

.pa_kind <- function(x) {
  kinds <- c("scale", "reference", "classifier", "result")
  for (kind in kinds) if (inherits(x, paste0("populismaxis_", kind))) return(kind)
  stop("Expected a populismaxis object.", call. = FALSE)
}

.pa_payload <- function(x) {
  text <- .subset2(unclass(x), "json")
  expected <- .subset2(unclass(x), "identity")
  if (!is.character(text) || length(text) != 1L || is.na(text) ||
      !identical(digest::digest(enc2utf8(text), algo = "sha256", serialize = FALSE), expected))
    stop("Object identity does not match its canonical JSON payload.", call. = FALSE)
  text
}
.pa_identity <- function(x) .subset2(unclass(x), "identity")

.pa_wrap <- function(obj, kind) {
  bridge <- .pa_bridge()
  text <- reticulate::py_to_r(bridge$encode_object(obj))
  id <- reticulate::py_to_r(bridge$object_identity(kind, text))
  description <- reticulate::py_to_r(bridge$object_summary(kind, text))
  structure(list(json = text, identity = id, summary_json = description),
            class = c(paste0("populismaxis_", kind), "populismaxis_object"))
}

.pa_unwrap <- function(x, kind) {
  if (!inherits(x, paste0("populismaxis_", kind)))
    stop("Expected a populismaxis ", kind, " object.", call. = FALSE)
  .pa_bridge()$decode_object(kind, .pa_payload(x), .pa_identity(x))
}

.pa_immutable <- function() {
  stop("populismaxis objects are immutable snapshots. Extract a view for analysis; create a new object to change its definition or settings.", call. = FALSE)
}

`$<-.populismaxis_object` <- function(x, name, value) .pa_immutable()
`[[<-.populismaxis_object` <- function(x, ..., value) .pa_immutable()
`[<-.populismaxis_object` <- function(x, ..., value) .pa_immutable()
`names<-.populismaxis_object` <- function(x, value) .pa_immutable()

`$.populismaxis_object` <- function(x, name) {
  if (identical(name, "identity")) return(.pa_identity(x))
  if (identical(name, "json")) return(.pa_payload(x))
  .pa_json(.pa_payload(x))[[name]]
}

`$.populismaxis_result` <- function(x, name) {
  if (identical(name, "scores")) return(scores(x))
  if (identical(name, "evidence")) return(evidence(x))
  if (identical(name, "provenance")) return(provenance(x))
  if (identical(name, "identity")) return(.pa_identity(x))
  if (identical(name, "json")) return(.pa_payload(x))
  NULL
}

scores <- function(result) {
  if (!inherits(result, "populismaxis_result")) stop("Expected a populismaxis result.", call. = FALSE)
  payload <- .pa_json(.pa_payload(result))$scores
  values <- vector("list", length(payload$columns))
  names(values) <- unlist(payload$columns, use.names = FALSE)
  for (i in seq_along(values)) {
    column <- lapply(payload$rows, function(row) row[[i]])
    dtype <- payload$dtypes[[i]]
    if (dtype %in% c("object", "str", "string", "string[python]", "string[pyarrow]")) {
      values[[i]] <- vapply(column, as.character, character(1))
    } else if (grepl("^(u?int|float|Float|Int|UInt)", dtype)) {
      # Integers beyond exact double range stay decimal strings for inspection.
      if (any(vapply(column, is.character, logical(1))))
        values[[i]] <- vapply(column, as.character, character(1))
      else values[[i]] <- vapply(column, as.numeric, numeric(1))
    } else if (dtype %in% c("bool", "boolean")) {
      values[[i]] <- vapply(column, as.logical, logical(1))
    } else stop("Unsupported score-column dtype: ", dtype, call. = FALSE)
  }
  frame <- as.data.frame(values, check.names = FALSE, stringsAsFactors = FALSE, optional = TRUE)
  rownames(frame) <- NULL
  attr(frame, "pandas.index") <- NULL
  frame
}

evidence <- function(result) {
  if (!inherits(result, "populismaxis_result")) stop("Expected a populismaxis result.", call. = FALSE)
  .pa_json(.pa_payload(result))$evidence
}

provenance <- function(result) {
  if (!inherits(result, "populismaxis_result")) stop("Expected a populismaxis result.", call. = FALSE)
  .pa_json(.pa_payload(result))$provenance
}

summary.populismaxis_object <- function(object, ...) {
  kind <- .pa_kind(object)
  value <- .pa_json(.pa_payload(object))
  if (kind == "result") return(list(identity = .pa_identity(object), n_documents = length(value$scores$rows), columns = unlist(value$scores$columns, use.names = FALSE), provenance = value$provenance))
  description <- .pa_json(.subset2(unclass(object), "summary_json"))
  if (!identical(description$identity, .pa_identity(object)))
    stop("Summary identity does not match this object.", call. = FALSE)
  description
}

print.populismaxis_object <- function(x, ...) {
  kind <- .pa_kind(x)
  value <- summary(x)
  label <- switch(kind, scale = value$name, reference = value$scale$name,
                  classifier = value$score, result = paste(value$n_documents, "documents"))
  cat("<populismaxis_", kind, ": ", label, ">\n", sep = "")
  cat("Identity: ", .pa_identity(x), "\n", sep = "")
  invisible(x)
}
