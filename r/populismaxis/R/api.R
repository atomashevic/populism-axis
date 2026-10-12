.pa_text <- function(x, name, nullable = FALSE) {
  if (is.null(x) && nullable) return(NULL)
  if (anyNA(x)) stop(name, " must not contain missing values.", call. = FALSE)
  if (is.factor(x)) x <- as.character(x)
  if (!is.character(x) || !length(x) || any(!nzchar(trimws(x))))
    stop(name, " must be a nonempty character vector with nonempty values.", call. = FALSE)
  unname(x)
}

.pa_scalar <- function(x, name, nullable = FALSE) {
  x <- .pa_text(x, name, nullable)
  if (!is.null(x) && length(x) != 1L) stop(name, " must be one string.", call. = FALSE)
  x
}

.pa_list <- function(x, name, nullable = FALSE) {
  x <- .pa_text(x, name, nullable)
  if (is.null(x)) NULL else unname(as.list(x))
}

.pa_bool <- function(x, name, nullable = FALSE) {
  if (is.null(x) && nullable) return(NULL)
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE", if (nullable) " or NULL" else "", ".", call. = FALSE)
  x
}

.pa_integer <- function(x, name, nullable = FALSE) {
  if (is.null(x) && nullable) return(NULL)
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x <= 0 || x != floor(x) || x > .Machine$integer.max)
    stop(name, " must be a positive whole number no larger than ", .Machine$integer.max, ".", call. = FALSE)
  as.integer(x)
}

.pa_scale <- function(scale, nullable = FALSE) {
  if (is.null(scale) && nullable) return(NULL)
  if (inherits(scale, "populismaxis_scale")) return(.pa_unwrap(scale, "scale"))
  if (is.character(scale) && length(scale) == 1L && !is.na(scale) && identical(scale, "populism")) return(scale)
  stop("scale must be a scale object or 'populism'", if (nullable) " or NULL" else "", ".", call. = FALSE)
}

.pa_reference <- function(reference) {
  if (inherits(reference, "populismaxis_reference")) return(.pa_unwrap(reference, "reference"))
  if (is.character(reference) && length(reference) == 1L && !is.na(reference) && reference %in% c("frozen", "corpus")) return(reference)
  stop("reference must be a fitted reference object, 'frozen', or 'corpus'.", call. = FALSE)
}

.pa_options <- function(unit, pooling, clean, claim_check, device, threads,
                        encoder_batch_size, nli_batch_size, local_files_only) {
  list(unit = .pa_scalar(unit, "unit", TRUE), pooling = .pa_scalar(pooling, "pooling", TRUE),
       clean = .pa_bool(clean, "clean", TRUE), claim_check = .pa_bool(claim_check, "claim_check", TRUE),
       device = .pa_scalar(device, "device"), threads = .pa_integer(threads, "threads"),
       encoder_batch_size = .pa_integer(encoder_batch_size, "encoder_batch_size", TRUE),
       nli_batch_size = .pa_integer(nli_batch_size, "nli_batch_size"),
       local_files_only = .pa_bool(local_files_only, "local_files_only"))
}

.pa_input <- function(texts, ids) {
  texts <- .pa_text(texts, "texts")
  ids <- .pa_text(ids, "ids", TRUE)
  if (!is.null(ids) && (length(ids) != length(texts) || anyDuplicated(ids)))
    stop("ids must be unique, with one string per text.", call. = FALSE)
  list(texts = unname(as.list(texts)), ids = if (is.null(ids)) NULL else unname(as.list(ids)))
}

define_scale <- function(name, positive = NULL, contrast = NULL, components = NULL, combine = NULL) {
  name <- .pa_scalar(name, "name")
  combine <- .pa_scalar(combine, "combine", TRUE)
  if (!is.null(components)) {
    if (!is.null(positive) || !is.null(contrast)) stop("Provide positive and contrast, or components; do not mix them.", call. = FALSE)
    if (!is.list(components) || !length(components) || is.null(names(components)) ||
        anyNA(names(components)) || any(!nzchar(names(components))) || anyDuplicated(names(components)))
      stop("components must be a nonempty named list with unique component names.", call. = FALSE)
    components <- lapply(components, function(component) {
      if (!is.list(component) || length(component) != 2L ||
          !setequal(names(component), c("positive", "contrast")))
        stop("Each component requires exactly positive and contrast vectors.", call. = FALSE)
      list(positive = .pa_list(component$positive, "component positive"),
           contrast = .pa_list(component$contrast, "component contrast"))
    })
  } else {
    positive <- .pa_list(positive, "positive")
    contrast <- .pa_list(contrast, "contrast")
  }
  .pa_wrap(.pa_module()$define_scale(name, positive = positive, contrast = contrast,
                                   components = components, combine = combine), "scale")
}

read_scale_csv <- function(path, name, combine = NULL) {
  path <- .pa_scalar(path, "path")
  name <- .pa_scalar(name, "name")
  combine <- .pa_scalar(combine, "combine", TRUE)
  .pa_wrap(.pa_module()$read_scale_csv(path, name = name, combine = combine), "scale")
}

fit_reference <- function(texts, scale, ids = NULL, unit = NULL, pooling = NULL,
                          clean = NULL, claim_check = NULL, device = "cpu", threads = 4L,
                          encoder_batch_size = NULL, nli_batch_size = 16L, local_files_only = TRUE) {
  input <- .pa_input(texts, ids)
  options <- .pa_options(unit, pooling, clean, claim_check, device, threads, encoder_batch_size, nli_batch_size, local_files_only)
  object <- do.call(.pa_module()$fit_reference, c(input, list(scale = .pa_scale(scale)), options))
  .pa_wrap(object, "reference")
}

fit_reference_csv <- function(path, scale, text_column = "text", id_column = NULL, ...) {
  path <- .pa_scalar(path, "path")
  text_column <- .pa_scalar(text_column, "text_column")
  id_column <- .pa_scalar(id_column, "id_column", TRUE)
  options <- .pa_csv_options(list(...))
  object <- do.call(.pa_module()$fit_reference_csv,
                   c(list(path, scale = .pa_scale(scale), text_column = text_column, id_column = id_column), options))
  .pa_wrap(object, "reference")
}

score_texts <- function(texts, reference, scale = NULL, ids = NULL, unit = NULL, pooling = NULL,
                        clean = NULL, claim_check = NULL, device = "cpu", threads = 4L,
                        encoder_batch_size = NULL, nli_batch_size = 16L, local_files_only = TRUE) {
  input <- .pa_input(texts, ids)
  options <- .pa_options(unit, pooling, clean, claim_check, device, threads, encoder_batch_size, nli_batch_size, local_files_only)
  object <- do.call(.pa_module()$score_texts,
                   c(input, list(reference = .pa_reference(reference), scale = .pa_scale(scale, TRUE)), options))
  .pa_wrap(object, "result")
}

score_csv <- function(path, reference, scale = NULL, text_column = "text", id_column = NULL, ...) {
  path <- .pa_scalar(path, "path")
  text_column <- .pa_scalar(text_column, "text_column")
  id_column <- .pa_scalar(id_column, "id_column", TRUE)
  options <- .pa_csv_options(list(...))
  object <- do.call(.pa_module()$score_csv,
                   c(list(path, reference = .pa_reference(reference), scale = .pa_scale(scale, TRUE), text_column = text_column, id_column = id_column), options))
  .pa_wrap(object, "result")
}

.pa_csv_options <- function(options) {
  defaults <- list(unit = NULL, pooling = NULL, clean = NULL, claim_check = NULL,
                   device = "cpu", threads = 4L, encoder_batch_size = NULL,
                   nli_batch_size = 16L, local_files_only = TRUE)
  if (length(options) && (is.null(names(options)) || any(!nzchar(names(options))) || anyDuplicated(names(options)) || any(!names(options) %in% names(defaults))))
    stop("CSV scoring options must have unique supported argument names.", call. = FALSE)
  defaults[names(options)] <- options
  do.call(.pa_options, defaults)
}

make_classifier <- function(reference, threshold, below, at_or_above, score = NULL) {
  if (!is.numeric(threshold) || length(threshold) != 1L || is.na(threshold) || !is.finite(threshold))
    stop("threshold must be one finite number.", call. = FALSE)
  below <- .pa_scalar(below, "below")
  at_or_above <- .pa_scalar(at_or_above, "at_or_above")
  score <- .pa_scalar(score, "score", TRUE)
  if (identical(below, at_or_above)) stop("below and at_or_above must be distinct labels.", call. = FALSE)
  .pa_wrap(.pa_module()$make_classifier(.pa_unwrap(reference, "reference"), threshold = as.numeric(threshold), below = below, at_or_above = at_or_above, score = score), "classifier")
}

apply_classifier <- function(classifier, result) {
  object <- .pa_unwrap(classifier, "classifier")$apply(.pa_unwrap(result, "result"))
  .pa_wrap(object, "result")
}

.pa_save <- function(object, kind, path, overwrite) {
  path <- .pa_scalar(path, "path")
  overwrite <- .pa_bool(overwrite, "overwrite")
  .pa_unwrap(object, kind)$save(path, overwrite = overwrite)
  invisible(object)
}

.pa_load <- function(path, kind, python_class) {
  path <- .pa_scalar(path, "path")
  object <- .pa_module()[[python_class]]$load(path)
  .pa_wrap(object, kind)
}

save_scale <- function(scale, path, overwrite = FALSE) .pa_save(scale, "scale", path, overwrite)
load_scale <- function(path) .pa_load(path, "scale", "ScaleDefinition")
save_reference <- function(reference, path, overwrite = FALSE) .pa_save(reference, "reference", path, overwrite)
load_reference <- function(path) .pa_load(path, "reference", "FittedReference")
save_classifier <- function(classifier, path, overwrite = FALSE) .pa_save(classifier, "classifier", path, overwrite)
load_classifier <- function(path) .pa_load(path, "classifier", "ThresholdRule")

write_scores <- function(result, out_dir, overwrite = FALSE) {
  out_dir <- .pa_scalar(out_dir, "out_dir")
  overwrite <- .pa_bool(overwrite, "overwrite")
  .pa_module()$write_scores(.pa_unwrap(result, "result"), out_dir, overwrite = overwrite)
  invisible(result)
}
