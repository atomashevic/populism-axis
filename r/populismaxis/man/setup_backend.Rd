\name{setup_backend}
\alias{setup_backend}
\alias{configure_python}
\alias{python_requirements}
\title{Prepare or Select the Python Backend}
\description{Uses reticulate and uv to prepare a cached Python environment on first backend use. An existing prepared interpreter can be selected instead.}
\usage{
setup_backend(gpu = FALSE, download_models = FALSE)
configure_python(python)
python_requirements(gpu = FALSE)
}
\arguments{
\item{gpu}{Logical. Select the pinned CUDA 12.6 build before Python starts. The default is CPU. Scoring still uses its explicit \code{device} argument.}
\item{download_models}{Logical. Explicitly download the two pinned encoder and claim-check model snapshots. The default prepares dependencies only.}
\item{python}{Path to an existing Python executable containing the matching backend wheel and dependency profile.}
}
\details{
Package loading declares requirements without initializing Python. Managed setup uses Python 3.12 and torch 2.14.0. Supported wheel targets are recorded in the bundled backend manifest; unsupported platforms produce an error. The R distribution carries its matching Python wheel, so no published Python package or source checkout is required.

\code{POPULISM_AXIS_PYTHON}, \code{configure_python()}, and explicit reticulate configuration support prepared environments. These environments are checked but never changed. Conflicting selections or a change after Python initialization require a new R session. Prepared Python must be version 3.11 or newer and satisfy the bundled dependency profile.

Set \code{UV_CACHE_DIR}, \code{UV_PYTHON_INSTALL_DIR}, and \code{HF_HOME} before initialization to relocate caches. Set \code{UV_OFFLINE=1} and \code{HF_HUB_OFFLINE=1} to require prepared caches. A scorer's \code{local_files_only=TRUE} controls model access; it does not itself prevent uv from downloading missing dependencies. \code{RETICULATE_USE_MANAGED_VENV=no} disables the automatic managed route.

Environment provisioning and model downloads are separate. \code{setup_backend(download_models=TRUE)} obtains only the model revisions pinned by the Python engine. No inference is run during setup. A CUDA environment requires a compatible NVIDIA driver.
}
\value{\code{python_requirements()} returns a character vector of requirement specifications. \code{configure_python()} invisibly returns the selected path. \code{setup_backend()} invisibly returns Python path, torch version and requested GPU setting.}
\examples{
# Inspect requirements without installing anything or starting Python:
python_requirements()
\dontrun{
setup_backend(download_models = TRUE)
# A prepared environment is optional:
configure_python("/path/to/environment/bin/python")
}
}
