# ==============================================================================
# python.R - locating and checking the Python solver
# ==============================================================================

# Arguments that select the solver entry point, placed before the bundle
# directory on the command line: $RX_CLI when set, otherwise the launcher of the
# copy shipped with this package followed by "cli". The shipped copy is used
# even when a Python package `remlax` is importable, so that the R interface
# and the solver always come from the same version.
rx_solver_args <- function() {
  if (nzchar(Sys.getenv("RX_CLI"))) return(strsplit(Sys.getenv("RX_CLI"), " +")[[1]])
  c(.rx_launcher(), "cli")
}

# Path of inst/python/remlax_run.py. The launcher runs one module of the shipped
# `remlax` Python package, imported as a package, without searching the current
# directory of the R session.
.rx_launcher <- function() {
  f <- system.file("python", "remlax_run.py", package = "remlax")
  if (!nzchar(f))
    stop("solveur Python introuvable dans le paquet installe (inst/python). ",
         "Reinstaller remlax, ou definir RX_CLI.", call. = FALSE)
  f
}

# Command that starts Python, split on spaces: $RX_PY, else $IGE_JAX_CMD, else
# an Apptainer image $IGE_JAX_SIF, else python3.
rx_python_cmd <- function() {
  if (nzchar(Sys.getenv("RX_PY"))) {
    py <- strsplit(Sys.getenv("RX_PY"), " +")[[1]]
    # RX_PY naming the directory of a virtual environment (the one made by
    # rx_install_python()) rather than its interpreter: take the interpreter.
    if (length(py) == 1L && dir.exists(py)) {
      cand <- file.path(py, c("bin/python", "bin/python3", "Scripts/python.exe"))
      cand <- cand[file.exists(cand)]
      if (length(cand)) return(cand[1])
    }
    return(py)
  }
  if (nzchar(Sys.getenv("IGE_JAX_CMD"))) return(strsplit(Sys.getenv("IGE_JAX_CMD"), " +")[[1]])
  sif <- Sys.getenv("IGE_JAX_SIF")
  if (nzchar(sif)) return(c("apptainer", "exec", "--nv", sif, "python3"))
  "python3"
}

# Checks of an interpreter, cached by command for the session: `import jax`
# takes one to two seconds.
.rx_python_cache <- new.env(parent = emptyenv())

# Does the interpreter carry jax, numpy and scipy? Returns list(ok, python,
# versions, message). The message names the interpreter tried, the RX_PY
# variable and how to install one.
rx_python_check <- function(py = rx_python_cmd(), quiet = FALSE) {
  key <- paste(py, collapse = " ")
  res <- .rx_python_cache[[key]]
  if (is.null(res)) res <- .rx_python_probe(py, key)
  if (!quiet) {
    if (res$ok) message("remlax : ", res$versions[["executable"]], " : jax ", res$versions[["jax"]],
                        ", numpy ", res$versions[["numpy"]], ", scipy ", res$versions[["scipy"]])
    else message(res$message)
  }
  invisible(res)
}

# Runs the interpreter once and caches the result.
.rx_python_probe <- function(py, key) {
  code <- paste0("import sys\n",
                 "try:\n    import jax, numpy, scipy\n",
                 "except Exception as e:\n    print('MANQUE', type(e).__name__, e); sys.exit(3)\n",
                 "print('OK', jax.__version__, numpy.__version__, scipy.__version__, sys.executable)\n")
  out <- suppressWarnings(tryCatch(
    .rx_system2(py[1], shQuote(c(py[-1], "-c", code)), stdout = TRUE, stderr = TRUE),
    error = function(e) structure(conditionMessage(e), status = 127L)))
  st <- attr(out, "status") %||% 0L
  last <- if (length(out)) out[length(out)] else ""
  res <- list(ok = identical(as.integer(st), 0L) && startsWith(last, "OK "),
              python = key, versions = NULL, message = "")
  if (res$ok) {
    v <- strsplit(last, " ")[[1]]
    res$versions <- c(jax = v[2], numpy = v[3], scipy = v[4], executable = v[5])
  } else {
    res$message <- paste0(
      "remlax : l'interpreteur Python \"", key, "\" ne porte pas jax, numpy et scipy",
      if (nzchar(last)) paste0(" (", last, ")") else "", ".\n",
      "  Designer un Python qui les a : Sys.setenv(RX_PY = \"/chemin/vers/python\")\n",
      "  ou dans ~/.Renviron : RX_PY=/chemin/vers/python\n",
      "  Pour en creer un : remlax::rx_install_python() (ou pip install jax numpy scipy).")
  }
  .rx_python_cache[[key]] <- res
  res
}

# Stops before launching the solver when the interpreter does not fit.
.rx_python_require <- function(py) {
  r <- rx_python_check(py, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  invisible(r)
}

# Creates a virtual environment `dir` with `python -m venv`, installs jax, numpy
# and scipy with pip (jax[cuda12] when cuda = TRUE; blackjax when bayes =
# TRUE), checks the import, and prints the RX_PY line to set. It modifies
# neither ~/.Renviron nor the session.
rx_install_python <- function(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"), cuda = FALSE,
                              python = "python3", upgrade = FALSE, ask = interactive(),
                              bayes = FALSE) {
  # An absolute path: a relative RX_PY would resolve against the working
  # directory of each session.
  dir <- normalizePath(dir, mustWork = FALSE)
  exe <- file.path(dir, if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python")
  if (!file.exists(exe) || upgrade) {
    # CRAN policy: writing outside the temporary directory and downloading
    # several hundred MB need a confirmation in an interactive session.
    if (isTRUE(ask)) {
      ok <- utils::askYesNo(paste0(
        "remlax : creer un environnement Python dans ", dir,
        " et y telecharger jax, numpy et scipy depuis PyPI (plusieurs centaines de Mo) ?"))
      if (!isTRUE(ok)) stop("remlax : installation annulee.", call. = FALSE)
    }
    if (!file.exists(exe)) {
      message("remlax : creation du venv ", dir)
      st <- system2(python, c("-m", "venv", shQuote(dir)))
      if (!identical(as.integer(st), 0L))
        stop("remlax : `", python, " -m venv` a echoue (code ", st, "). Python >= 3.10 requis.",
             call. = FALSE)
    }
    packages <- c(if (cuda) "jax[cuda12]>=0.4.30" else "jax>=0.4.30", "numpy>=1.24", "scipy>=1.10",
                  if (bayes) "blackjax>=1.7")
    message("remlax : pip install ", paste(packages, collapse = " "))
    st <- system2(exe, c("-m", "pip", "install", "--upgrade", "--quiet", shQuote(packages)))
    if (!identical(as.integer(st), 0L))
      stop("remlax : pip install a echoue (code ", st, ").", call. = FALSE)
  }
  rm(list = ls(.rx_python_cache), envir = .rx_python_cache)
  r <- rx_python_check(exe, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  message("remlax : interpreteur pret : ", exe, "\n",
          "  jax ", r$versions[["jax"]], ", numpy ", r$versions[["numpy"]],
          ", scipy ", r$versions[["scipy"]], "\n",
          "  Pour cette session : Sys.setenv(RX_PY = \"", exe, "\")\n",
          "  Pour toujours, dans ~/.Renviron : RX_PY=", exe, "\n",
          "  Pour le supprimer : remlax::rx_remove_python()")
  invisible(exe)
}

# Removes the environment made by rx_install_python(), and the package data
# directory when it is left empty. CRAN accepts data in tools::R_user_dir()
# when their content is managed; this function is the means.
rx_remove_python <- function(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"),
                             ask = interactive()) {
  dir <- normalizePath(dir, mustWork = FALSE)
  if (!dir.exists(dir)) {
    message("remlax : rien a supprimer, ", dir, " n'existe pas.")
    return(invisible(FALSE))
  }
  if (isTRUE(ask) && !isTRUE(utils::askYesNo(paste0("remlax : supprimer ", dir, " ?"))))
    return(invisible(FALSE))
  unlink(dir, recursive = TRUE)
  parent <- dirname(dir)
  if (identical(parent, normalizePath(tools::R_user_dir("remlax", "data"), mustWork = FALSE)) &&
      dir.exists(parent) && !length(list.files(parent, all.files = TRUE, no.. = TRUE)))
    unlink(parent, recursive = TRUE)
  rm(list = ls(.rx_python_cache), envir = .rx_python_cache)
  message("remlax : ", dir, " supprime.")
  invisible(!dir.exists(dir))
}
