.pa_test_internal <- function(name) {
  if (isNamespaceLoaded("populismaxis")) getFromNamespace(name, "populismaxis")
  else get(name, envir = .GlobalEnv, inherits = TRUE)
}

.pa_fixture_module <- function() {
  python <- Sys.getenv("POPULISM_AXIS_TEST_PYTHON", unset = "")
  if (!nzchar(python)) testthat::skip("Set POPULISM_AXIS_TEST_PYTHON to a prepared Python environment; tests never provision dependencies.")
  if (exists("configure_python", mode = "function")) configure_python(python)
  else reticulate::use_python(python, required = TRUE)
  .pa_test_internal(".pa_module")()
  reticulate::import_from_path("populismaxis_test_fixtures", path = testthat::test_path("fixtures"), convert = FALSE)
}

.pa_fixture_reference <- function(fixture) .pa_test_internal(".pa_wrap")(fixture$reference(), "reference")
.pa_fixture_result <- function(fixture) .pa_test_internal(".pa_wrap")(fixture$result(), "result")
