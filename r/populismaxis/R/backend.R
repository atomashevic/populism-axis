.pa_state <- new.env(parent = emptyenv())
.pa_state$requirements <- character()
.pa_state$gpu <- FALSE
.pa_state$python <- NULL
.pa_state$module <- NULL
.pa_state$bridge <- NULL

.pa_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE.", call. = FALSE)
  x
}

.pa_bundle <- function() {
  root <- system.file("backend", package = "populismaxis", mustWork = TRUE)
  manifest <- jsonlite::fromJSON(file.path(root, "bundle.json"), simplifyVector = FALSE)
  if (!identical(manifest$schema, "populismaxis.backend.v1"))
    stop("Unsupported bundled backend schema.", call. = FALSE)
  wheel <- file.path(root, manifest$wheel)
  if (!file.exists(wheel) || !identical(digest::digest(file = wheel, algo = "sha256"), manifest$wheel_sha256))
    stop("Bundled Python wheel failed SHA256 verification. Reinstall the matching R package.", call. = FALSE)
  list(root = root, manifest = manifest, wheel = wheel)
}

.pa_platform <- function(sysname = Sys.info()[["sysname"]], machine = Sys.info()[["machine"]]) {
  machine <- tolower(machine)
  if (machine %in% c("amd64", "x86-64")) machine <- "x86_64"
  if (machine %in% c("arm64", "aarch64")) machine <- "aarch64"
  paste(sysname, machine, sep = "-")
}

#' @export
python_requirements <- function(gpu = FALSE) {
  gpu <- .pa_flag(gpu, "gpu")
  bundle <- .pa_bundle()
  flavor <- if (gpu) "cu126" else "cpu"
  torch <- bundle$manifest$torch_wheels[[flavor]][[.pa_platform()]]
  if (is.null(torch))
    stop("No pinned torch wheel for ", .pa_platform(), " / ", flavor,
         ". Use a supported platform; torch will not be downgraded.", call. = FALSE)
  profile <- file.path(bundle$root, paste0("requirements-", flavor, ".txt"))
  if (!identical(digest::digest(file = profile, algo = "sha256"), bundle$manifest$requirements_sha256[[flavor]]))
    stop("Bundled dependency profile failed SHA256 verification.", call. = FALSE)
  wheel <- normalizePath(bundle$wheel, winslash = "/", mustWork = TRUE)
  uri <- paste0(if (startsWith(wheel, "/")) "file://" else "file:///",
                utils::URLencode(wheel, reserved = TRUE))
  # URLencode must preserve path separators and a Windows drive colon.
  uri <- gsub("%2F", "/", uri, fixed = TRUE)
  uri <- gsub("%3A", ":", uri, fixed = TRUE)
  c(paste0("populism-axis @ ", uri, "#sha256=", bundle$manifest$wheel_sha256),
    paste0("torch @ ", torch), readLines(profile, warn = FALSE))
}

.pa_python_path <- function(path) {
  path <- path.expand(path)
  paste0(normalizePath(dirname(path), winslash = "/", mustWork = TRUE), "/", basename(path))
}

.pa_fixed_python <- function() {
  local <- .pa_state$python
  package <- Sys.getenv("POPULISM_AXIS_PYTHON", "")
  general <- Sys.getenv("RETICULATE_PYTHON", "")
  named_env <- Sys.getenv("RETICULATE_PYTHON_ENV", "")
  if (nzchar(named_env) && (!is.null(local) || nzchar(package)))
    stop("RETICULATE_PYTHON_ENV already selects an environment. Use that selection alone, or unset it before configure_python()/POPULISM_AXIS_PYTHON.", call. = FALSE)
  if (identical(general, "managed") && (!is.null(local) || nzchar(package)))
    stop("RETICULATE_PYTHON=managed conflicts with the fixed Python request. Choose one mode before initialization.", call. = FALSE)
  if (identical(general, "managed")) general <- ""
  choices <- Filter(function(x) !is.null(x) && nzchar(x), list(local, package, general))
  if (!length(choices)) return(NULL)
  for (path in choices) if (!file.exists(path) || dir.exists(path))
    stop("Configured Python does not exist: ", path, ". Correct the path or remove the override for uv setup.", call. = FALSE)
  normalized <- vapply(choices, .pa_python_path, character(1))
  if (length(unique(normalized)) != 1L)
    stop("Conflicting Python overrides. Align configure_python(), POPULISM_AXIS_PYTHON and RETICULATE_PYTHON, then restart R.", call. = FALSE)
  normalized[[1L]]
}

.pa_use_python <- function(path) {
  # reticulate warns before replacing a prior required interpreter. Turn that
  # warning into an error so the earlier choice remains intact.
  withCallingHandlers(reticulate::use_python(path, required = TRUE),
    warning = function(w) stop("Conflicting Python selection: ", conditionMessage(w),
                              " Align the choices and restart R.", call. = FALSE))
}

.pa_declare <- function(gpu = .pa_state$gpu) {
  gpu <- .pa_flag(gpu, "gpu")
  if (!is.null(.pa_fixed_python())) return(invisible(NULL))
  if (reticulate::py_available(initialize = FALSE)) {
    if (!identical(gpu, .pa_state$gpu))
      stop("Python has started. Restart R before changing the CPU/CUDA build.", call. = FALSE)
    return(invisible(NULL))
  }
  requirements <- python_requirements(gpu)
  if (identical(requirements, .pa_state$requirements)) return(invisible(NULL))
  old_only <- setdiff(.pa_state$requirements, requirements)
  if (length(old_only)) reticulate::py_require(old_only, action = "remove")
  reticulate::py_require(requirements, python_version = .pa_bundle()$manifest$python_version)
  .pa_state$requirements <- requirements
  .pa_state$gpu <- gpu
  invisible(NULL)
}

#' @export
configure_python <- function(python) {
  if (!is.character(python) || length(python) != 1L || is.na(python) || !nzchar(python) || !file.exists(python) || dir.exists(python))
    stop("python must be an existing Python executable path.", call. = FALSE)
  old <- .pa_state$python
  .pa_state$python <- .pa_python_path(python)
  tryCatch({
    requested <- .pa_fixed_python()
    if (reticulate::py_available(initialize = FALSE) &&
        !identical(.pa_python_path(reticulate::py_config()$python), requested))
      stop("Another Python interpreter is already running. Restart R before selecting this environment.", call. = FALSE)
    .pa_use_python(requested)
    invisible(requested)
  }, error = function(e) { .pa_state$python <- old; stop(e) })
}

.pa_verify_engine <- function(module, bundle) {
  version <- reticulate::py_to_r(module$`__version__`)
  if (!identical(version, bundle$manifest$package_version))
    stop("Python package version ", version, " does not match R backend ", bundle$manifest$package_version,
         ". Prepare the matching package and restart R.", call. = FALSE)
  root <- dirname(reticulate::py_to_r(module$`__file__`))
  for (name in names(bundle$manifest$engine_files)) {
    path <- file.path(root, name)
    if (!file.exists(path) || !identical(digest::digest(file = path, algo = "sha256"), bundle$manifest$engine_files[[name]]))
      stop("Python engine differs from the bundled version: ", name,
           ". Install the matching bundled wheel or use automatic uv setup in a new R session.", call. = FALSE)
  }
  metadata <- reticulate::import("importlib.metadata", convert = FALSE)
  torch_version <- tryCatch(reticulate::py_to_r(metadata$version("torch")), error = function(e) "missing")
  if (!identical(strsplit(torch_version, "+", fixed = TRUE)[[1L]][[1L]], bundle$manifest$torch_version))
    stop("Expected torch ", bundle$manifest$torch_version, "; found ", torch_version,
         ". Prepare matching dependencies and restart R.", call. = FALSE)
  checker <- reticulate::import_from_path("populismaxis_backend",
    path = system.file("python", package = "populismaxis", mustWork = TRUE), convert = FALSE)
  # Validate the common locked CPU profile even for a prepared CUDA environment;
  # CUDA-specific dependencies are checked by its resolver and torch import.
  profile <- file.path(bundle$root, "requirements-cpu.txt")
  if (!identical(digest::digest(file = profile, algo = "sha256"), bundle$manifest$requirements_sha256$cpu))
    stop("Bundled dependency profile failed SHA256 verification.", call. = FALSE)
  checker$validate_environment(as.list(readLines(profile, warn = FALSE)),
                               bundle$manifest$torch_version)
  invisible(TRUE)
}

.pa_module <- function() {
  fixed <- .pa_fixed_python()
  if (!is.null(.pa_state$module)) {
    if (!is.null(fixed) && !identical(.pa_python_path(reticulate::py_config()$python), fixed))
      stop("Python override changed after initialization. Restart R.", call. = FALSE)
    return(.pa_state$module)
  }
  .pa_declare()
  if (!is.null(fixed)) .pa_use_python(fixed)
  managed <- Sys.getenv("RETICULATE_USE_MANAGED_VENV", unset = NA_character_)
  # Force the uv route ahead of an incidental working-directory .venv, while
  # preserving explicit reticulate interpreter choices and managed-env opt-out.
  if (is.null(fixed) && is.na(managed) && !reticulate::py_available(initialize = FALSE)) {
    Sys.setenv(RETICULATE_USE_MANAGED_VENV = "yes")
    on.exit(Sys.unsetenv("RETICULATE_USE_MANAGED_VENV"), add = TRUE)
  }
  module <- tryCatch(reticulate::import("populism_axis", convert = FALSE), error = function(e) {
    stop("Could not start the populism-axis Python backend. ", conditionMessage(e),
         "\nFor automatic uv setup, remove conflicting Python overrides and restart R. ",
         "For a fixed environment, install the matching wheel and python_requirements() there.", call. = FALSE)
  })
  if (!is.null(fixed) && !identical(.pa_python_path(reticulate::py_config()$python), fixed))
    stop("Reticulate selected a different interpreter. Align the Python overrides and restart R.", call. = FALSE)
  .pa_verify_engine(module, .pa_bundle())
  .pa_state$module <- module
  module
}

.pa_bridge <- function() {
  .pa_module()
  if (is.null(.pa_state$bridge))
    .pa_state$bridge <- reticulate::import_from_path("populismaxis_bridge",
      path = system.file("python", package = "populismaxis", mustWork = TRUE), convert = FALSE)
  .pa_state$bridge
}

#' @export
setup_backend <- function(gpu = FALSE, download_models = FALSE) {
  gpu <- .pa_flag(gpu, "gpu")
  download_models <- .pa_flag(download_models, "download_models")
  .pa_declare(gpu)
  .pa_module()
  torch <- reticulate::import("torch", convert = FALSE)
  if (gpu && (!grepl("\\+cu126$", reticulate::py_to_r(torch$`__version__`)) ||
              !isTRUE(reticulate::py_to_r(torch$cuda$is_available()))))
    stop("CUDA 12.6 was requested but the loaded torch build or NVIDIA driver is unavailable. Restart R after preparing the CUDA environment.", call. = FALSE)
  if (download_models) {
    offline <- tolower(Sys.getenv("HF_HUB_OFFLINE", "")) %in% c("1", "true", "yes", "on")
    if (offline) stop("Model download requested while HF_HUB_OFFLINE is enabled.", call. = FALSE)
    revisions <- reticulate::import("populism_axis.model_revisions", convert = FALSE)
    hub <- reticulate::import("huggingface_hub", convert = FALSE)
    ids <- c(reticulate::py_to_r(revisions$E5_LARGE_V2), reticulate::py_to_r(revisions$BART_LARGE_MNLI))
    for (id in ids) {
      message("Preparing pinned model: ", id)
      hub$snapshot_download(repo_id = id, revision = revisions$PINNED_REVISIONS[[id]],
        allow_patterns = as.list(c("*.json", "*.txt", "*.model", "*.safetensors")), local_files_only = FALSE)
    }
  }
  invisible(list(python = reticulate::py_config()$python,
                 torch = reticulate::py_to_r(torch$`__version__`), gpu = gpu))
}

.onLoad <- function(libname, pkgname) {
  if (!reticulate::py_available(initialize = FALSE))
    try(.pa_declare(), silent = TRUE)
}
