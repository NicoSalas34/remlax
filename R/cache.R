# ==============================================================================
# cache.R - the solver's cache and the path of the compiled kernel
# ==============================================================================
# The Python solver keeps the programs compiled by XLA in a persistent cache, so
# that a second fit of the same shape does not compile again. The cache lives in
# tools::R_user_dir("remlax", "cache"), as the CRAN policy allows, and its
# content is managed: files unused for more than 30 days are removed once per
# session, and rx_cache_clear() empties it. During R CMD check, or with
# options(remlax.cache = FALSE), a folder of tempdir() is used instead.
# REMLAX_CACHE set by the user takes precedence.
# ==============================================================================

.rx_cache_state <- new.env(parent = emptyenv())

# Folder of the solver's cache for this call.
.rx_cache_dir <- function() {
  if (nzchar(Sys.getenv("REMLAX_CACHE"))) return(Sys.getenv("REMLAX_CACHE"))
  if (.rx_in_check() || isFALSE(getOption("remlax.cache", TRUE)))
    return(file.path(tempdir(), "remlax_cache"))
  d <- tools::R_user_dir("remlax", "cache")
  if (is.null(.rx_cache_state$pruned)) {
    .rx_cache_state$pruned <- TRUE
    .rx_cache_prune(d, days = 30)
  }
  d
}

# Is R CMD check running?
.rx_in_check <- function() {
  nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_")) ||
    (nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")) &&
       tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_")) != "false")
}

# Removes the files of `d` not modified for `days` days.
.rx_cache_prune <- function(d, days) {
  if (!dir.exists(d)) return(invisible(0))
  f <- list.files(d, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
  old <- f[file.mtime(f) < Sys.time() - days * 86400]
  if (length(old)) unlink(old)
  invisible(length(old))
}

# Path of the compiled library that holds the selected inversion kernel
# (src/selinv_c.c), or "" when the package library is not loaded.
.rx_selinv_lib <- function() {
  dll <- getLoadedDLLs()[["remlax"]]
  if (is.null(dll)) "" else dll[["path"]]
}

rx_cache_clear <- function() {
  d <- tools::R_user_dir("remlax", "cache")
  f <- list.files(d, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
  size <- sum(file.size(f), na.rm = TRUE)
  unlink(d, recursive = TRUE)
  message(sprintf("remlax: cache %s emptied (%.1f MB freed)", d, size / 2^20))
  invisible(size)
}
