# ==============================================================================
# launch.R - starting the solver
# ==============================================================================
# Every call to the solver goes through .rx_system2(), for three reasons held
# by the CRAN policy:
#   1. Python writes its .pyc files next to the sources, in the R library where
#      the package is installed; PYTHONDONTWRITEBYTECODE=1 prevents it.
#   2. JAX takes every core on CPU. During R CMD check (_R_CHECK_LIMIT_CORES_
#      set, as --as-cran does) a package may use at most two; one computation
#      thread with Eigen threading off is used then. The option remlax.threads
#      sets the number otherwise.
#   3. system2(env =) is ignored on Windows, so the variables are set with
#      Sys.setenv() for the duration of the call, then restored.
# ==============================================================================

# Number of computation threads to impose, or NULL for no limit.
.rx_threads <- function() {
  o <- getOption("remlax.threads")
  if (!is.null(o)) return(as.integer(o))
  lim <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_"))
  if (nzchar(lim) && lim != "false") return(1L)
  NULL
}

# system2() with the solver's environment variables set for the call.
.rx_system2 <- function(command, args = character(), ..., env = character()) {
  v <- c(PYTHONDONTWRITEBYTECODE = "1")
  n <- .rx_threads()
  if (!is.null(n)) {
    v <- c(v, OMP_NUM_THREADS = n, OPENBLAS_NUM_THREADS = n, MKL_NUM_THREADS = n)
    if (n == 1L)
      v <- c(v, XLA_FLAGS = trimws(paste(Sys.getenv("XLA_FLAGS"),
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")))
  }
  v <- c(v, env)
  previous <- Sys.getenv(names(v), unset = NA, names = TRUE)
  on.exit({
    for (k in names(previous))
      if (is.na(previous[[k]])) Sys.unsetenv(k)
      else do.call(Sys.setenv, stats::setNames(list(previous[[k]]), k))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(v))
  system2(command, args, ...)
}

# Runs the solver with the command line `a` and stops with its output if it
# fails. The persistent server is tried first (server.R); when it is not
# usable the solver runs as a subprocess.
.rx_run_solver <- function(py, a, verbose, what) {
  entry <- rx_solver_args(); n0 <- length(entry)
  r <- if (identical(as.character(a[seq_len(n0)]), as.character(entry)))
    .rx_server_run(py, a[-seq_len(n0)], verbose) else NULL
  if (!is.null(r)) {
    if (!identical(r$code, 0L)) .rx_solver_error(what, r$sortie)
    return(invisible(r$sortie))
  }
  .rx_run_subprocess(py, a, verbose, what)
}

# The solver as a subprocess. In verbose mode the progress (stdout) is shown
# live and stderr goes to a temporary file: capturing stderr in R would also
# capture stdout. With backend = "cpu", JAX_PLATFORMS=cpu keeps JAX away from
# the GPU, which it would otherwise initialise and try to reserve.
# RX_T_LANCEMENT dates the launch for the solver's timing of its start-up.
.rx_run_subprocess <- function(py, a, verbose, what) {
  err <- tempfile("rx_err_"); on.exit(unlink(err), add = TRUE)
  ib <- match("--backend", a)
  env <- if (!is.na(ib) && identical(as.character(a[ib + 1L]), "cpu"))
    c(JAX_PLATFORMS = "cpu") else character()
  env <- c(env, RX_T_LANCEMENT = format(as.numeric(Sys.time()), digits = 17))
  st <- suppressWarnings(.rx_system2(py[1], shQuote(c(py[-1], as.character(a))),
                                     stdout = if (verbose) "" else TRUE,
                                     stderr = if (verbose) err else TRUE, env = env))
  code <- if (verbose) as.integer(st) else as.integer(attr(st, "status") %||% 0L)
  output <- if (verbose) { if (file.exists(err)) readLines(err, warn = FALSE) else character() } else st
  if (!identical(code, 0L)) .rx_solver_error(what, output)
  if (verbose && length(output)) message(paste(output, collapse = "\n"))
  invisible(output)
}

# Error with the last 40 lines of the solver's output.
.rx_solver_error <- function(what, output) {
  stop(paste(c(paste0(what, ". Sortie du solveur :"), utils::tail(output, 40)),
             collapse = "\n"), call. = FALSE)
}
