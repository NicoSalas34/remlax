# ==============================================================================
# fit.R - rx_fit(): fit an rx_model with the Python solver
# ==============================================================================
# rx_fit() writes the model to a bundle, writes the optional inputs (starting
# theta, prediction matrices), builds the solver's command line, runs it
# (persistent server or subprocess), and reads the result back. The options of
# the R interface are translated once, in .rx_fit_options(), into the names of
# the solver's command line.
# ==============================================================================

rx_fit <- function(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
                   maxiter = 3000L, polish = 25L, n_restarts = 0L,
                   hessian = TRUE, blups = TRUE, pev = FALSE,
                   vpredict = NULL, wald = FALSE,
                   kenward_roger = FALSE, predict = NULL, theta_init = NULL,
                   fixed_theta = NULL, floor = -12, ceil = 12,
                   verbose = TRUE, keep = FALSE,
                   optimizer = getOption("remlax.optimizer", Sys.getenv("RX_OPTIMIZER", "lbfgsb")),
                   engine = getOption("remlax.engine", Sys.getenv("RX_ENGINE", "dense")),
                   inversion = c("auto", "dense", "sparse", "hybrid"), bayes = NULL,
                   memory = getOption("remlax.memory", Sys.getenv("RX_MEMORY", "standard")),
                   finish = getOption("remlax.finish", Sys.getenv("RX_FINISH", "newton"))) {
  opt <- .rx_fit_options(optimizer, engine, inversion, memory, finish)
  bounds <- .rx_check_bounds(floor, ceil)
  .rx_check_kinv_engine(model, engine)
  backend <- match.arg(backend)
  # Kenward-Roger and predictions assume y linear in beta and u at given V;
  # with a spatial lag only y* = (I - M) y is, and M depends on a parameter.
  if (!is.null(model$lag) && (isTRUE(kenward_roger) || !is.null(predict)))
    stop("rx_fit : Kenward-Roger et predict ne sont pas disponibles avec un decalage ",
         "spatial (rx_lag).", call. = FALSE)
  if (is.null(dir)) { dir <- tempfile("rx_"); on.exit(if (!keep) unlink(dir, recursive = TRUE)) }

  t_R0 <- Sys.time()
  rx_export(model, dir)
  .rx_write_fit_inputs(dir, predict, theta_init)
  py <- rx_python_cmd()
  a <- c(rx_solver_args(), dir, "--backend", backend, "--maxiter", maxiter, "--polish", polish,
         "--optimiseur", opt$optimizer, "--moteur", opt$engine_cli,
         "--inversion", opt$inversion_cli,
         "--memoire", opt$memory_cli, "--finition", opt$finish_cli,
         .rx_bayes_args(bayes, dir),
         "--restarts", n_restarts,
         "--floor", format(bounds[1], digits = 15), "--ceil", format(bounds[2], digits = 15),
         .rx_output_flags(vpredict, wald, kenward_roger, predict, theta_init, fixed_theta,
                          hessian, blups, pev, verbose))
  t_R1 <- Sys.time()
  # The interpreter is checked only after a failure, to explain it.
  tryCatch(.rx_run_solver(py, a, verbose, "solveur REML : echec"), error = function(e) {
    .rx_python_require(py)
    stop(e)
  })
  t_R2 <- Sys.time()
  res <- rx_read_result(dir, model = model, fixed_theta = fixed_theta)
  t_R3 <- Sys.time()
  res[["chrono"]] <- .rx_fit_chrono(res[["chrono"]], t_R0, t_R1, t_R2, t_R3)
  res[["inversion"]] <- .rx_inversion_api_name(res[["inversion"]])
  res
}

# --- options -------------------------------------------------------------------

# The options of rx_fit() and their names on the solver's command line:
#   optimizer  "lbfgsb", "ai" or "auto"
#   engine     "dense", "mme", "blocks" (solver: "blocs") or "auto"
#   inversion  "auto", "dense", "sparse" (solver: "creuse"), "hybrid" ("hybride")
#   memory     "standard" or "low" (solver: "econome")
#   finish     "newton" or "trust" (solver: "rc")
.rx_fit_options <- function(optimizer, engine, inversion, memory, finish) {
  optimizer <- match.arg(optimizer, c("lbfgsb", "ai", "auto"))
  engine <- match.arg(engine, c("dense", "mme", "blocks", "auto"))
  inversion <- .rx_match_inversion(inversion)
  memory <- match.arg(memory, c("standard", "low"))
  finish <- match.arg(finish, c("newton", "trust"))
  list(optimizer = optimizer,
       engine_cli = if (engine == "blocks") "blocs" else engine,
       inversion_cli = switch(inversion, sparse = "creuse", hybrid = "hybride", inversion),
       memory_cli = if (memory == "low") "econome" else "standard",
       finish_cli = if (finish == "trust") "rc" else "newton")
}

# "creuse", the original name of the sparse inversion, is still accepted.
.rx_match_inversion <- function(inversion) {
  if (identical(inversion, "creuse")) inversion <- "sparse"
  match.arg(inversion, c("auto", "dense", "sparse", "hybrid"))
}

# The solver names the sparse and hybrid inversions "creuse" and "hybride".
.rx_inversion_api_name <- function(inversion) {
  if (identical(inversion, "creuse")) return("sparse")
  if (identical(inversion, "hybride")) return("hybrid")
  inversion
}

# Bounds on every theta. theta is a log standard deviation, var = exp(2 theta),
# so floor = -12 is var = 3.8e-11. The bounds actually used come back in the
# result (par_floor, par_ceil).
.rx_check_bounds <- function(floor, ceil) {
  floor <- as.numeric(floor); ceil <- as.numeric(ceil)
  if (length(floor) != 1L || length(ceil) != 1L || !is.finite(floor) || !is.finite(ceil) ||
      floor >= ceil)
    stop("rx_fit : floor et ceil doivent etre deux scalaires finis avec floor < ceil.",
         call. = FALSE)
  c(floor, ceil)
}

# A term given only by its precision Kinv cannot be fitted by the dense engine,
# which would silently treat it as independent.
.rx_check_kinv_engine <- function(model, engine) {
  if (isTRUE(attr(model$terms, "creux_seulement")) && !(as.character(engine)[1] %in% c("mme", "auto")))
    stop("rx_fit : ce modele contient un ou plusieurs termes declares par leur ",
         "PRECISION (Kinv) sans facteur de K. Le moteur dense a besoin de K et ",
         "ignorerait silencieusement la parente, en estimant les effets comme ",
         "INDEPENDANTS. Fournir K a rx_term(), ou ajuster avec engine = \"mme\".",
         call. = FALSE)
}

# --- inputs and command line ---------------------------------------------------

# Writes a numeric vector or matrix as a raw float64 file, column-major.
.rx_write_bin <- function(path, x) {
  con <- file(path, "wb")
  writeBin(as.double(as.vector(as.matrix(x))), con, size = 8); close(con)
}

# Optional inputs: the prediction matrices L (l x p) and, per random term, M
# (l x t*q); and the starting theta. With maxiter = 0 and polish = 0, a
# starting theta gives a plain evaluation of the likelihood.
.rx_write_fit_inputs <- function(dir, predict, theta_init) {
  if (!is.null(predict)) {
    .rx_write_bin(file.path(dir, "pred_L.bin"), predict$L)
    for (nm in names(predict$M %||% list()))
      .rx_write_bin(file.path(dir, paste0("pred_M_", nm, ".bin")), predict$M[[nm]])
  }
  if (!is.null(theta_init)) {
    con <- file(file.path(dir, "in_theta.bin"), "wb")
    writeBin(as.double(theta_init), con, size = 8); close(con)
  }
}

# Flags of the outputs requested from the solver. The prediction error
# variances are off by default: they cost a (t*q) x n matrix per term. `pev`
# takes TRUE (every term) or a vector of term names.
.rx_output_flags <- function(vpredict, wald, kenward_roger, predict, theta_init, fixed_theta,
                             hessian, blups, pev, verbose) {
  c(if (!is.null(vpredict)) c("--vpredict",
      paste(sprintf("%s=%s", names(vpredict), unlist(vpredict)), collapse = ";")) else NULL,
    if (isTRUE(wald)) "--wald" else NULL,
    if (isTRUE(kenward_roger)) "--kenward-roger" else NULL,
    if (!is.null(predict)) "--predict" else NULL,
    if (!is.null(theta_init)) "--theta-in" else NULL,
    if (!is.null(fixed_theta)) c("--fixed-theta",
      paste(as.integer(fixed_theta), collapse = ",")) else NULL,
    if (!hessian) "--no-hessian" else NULL, if (!blups) "--no-blups" else NULL,
    if (isTRUE(pev)) "--pev" else
      if (is.character(pev) && length(pev)) c("--pev-termes", paste(pev, collapse = ",")) else NULL,
    if (!verbose) "--quiet" else NULL)
}

# Timing by stage: the solver's stages (chrono.py) framed by those of R.
# solveur_total is the wall time from launch to return; demarrage_serveur is
# what it has beyond the solver's stages (start of the persistent server at the
# first fit of a session, or the end of a subprocess).
.rx_fit_chrono <- function(solver_chrono, t_R0, t_R1, t_R2, t_R3) {
  dt <- function(a, b) as.numeric(difftime(b, a, units = "secs"))
  ch_py <- unlist(solver_chrono)
  c(export_R = dt(t_R0, t_R1), ch_py,
    demarrage_serveur = max(dt(t_R1, t_R2) - sum(ch_py), 0),
    solveur_total = dt(t_R1, t_R2), lecture_R = dt(t_R2, t_R3))
}

# --- standard errors -------------------------------------------------------------

# Standard errors of theta. The Hessian is that of -2 logL, so cov(theta) =
# 2 H^-1 on the free subspace: neither at a bound (read from par_floor and
# par_ceil) nor fixed by fixed_theta. Elsewhere NA. A negative eigenvalue of
# the free Hessian marks a saddle point, where every standard error is NA.
rx_se_theta <- function(theta, hessian, par_floor = -12, par_ceil = 12,
                        fixed_theta = NULL, tol_bound = 1e-7) {
  p <- length(theta)
  se <- rep(NA_real_, p)
  if (is.null(hessian) || !p) return(se)
  H <- as.matrix(hessian)
  if (nrow(H) != p || ncol(H) != p) return(se)
  free <- !(theta <= par_floor + tol_bound | theta >= par_ceil - tol_bound)
  if (length(fixed_theta)) free[as.integer(fixed_theta)] <- FALSE
  if (!any(free)) return(se)
  Hf <- (H[free, free, drop = FALSE] + t(H[free, free, drop = FALSE])) / 2
  if (!all(is.finite(Hf))) return(se)
  ev <- eigen(Hf, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-12 * max(abs(ev), 1)) return(se)
  Vi <- tryCatch(solve(Hf), error = function(e) NULL)
  if (is.null(Vi)) return(se)
  d <- diag(Vi)
  se[free] <- ifelse(d > 0, sqrt(2 * d), NA_real_)
  se
}

# --- mixed model equations ---------------------------------------------------------

#' Fit an rx_model through the mixed model equations
#'
#' A shortcut for `rx_fit(model, engine = "mme", ...)`, with the average
#' information optimiser by default. The restricted likelihood is computed from
#' Henderson's mixed model equations, with `W = [X, Z]`,
#' \deqn{C = W' R^{-1} W + \mathrm{diag}(0, G^{-1}),}{C = W' R^-1 W + diag(0, G^-1),}
#' instead of the marginal covariance `V`. The cost then depends on `m`, the
#' number of fixed effects and random levels, and on the fill-in of the factor
#' of `C`, rather than on `n`.
#'
#' The model, the parameters `theta`, the result and its diagnostics are those
#' of [rx_fit()]: both engines maximise the same likelihood and agree to
#' rounding. A model outside the scope of the mixed model equations engine
#' (residual `"none"`, an `rr` structure, a spatial lag, Gaussian metric kernels)
#' is fitted by the dense engine, with a message.
#'
#' @param model an [rx_model] object.
#' @param inversion how the selected entries of `C^-1` are obtained.
#'   `"dense"` inverts `C` with LAPACK on the CPU or with JAX on the GPU.
#'   `"sparse"` factorises `C` with SuperLU and computes the selected inverse
#'   by the supernodal Takahashi recurrence, on the CPU, from a small C kernel
#'   compiled at first use (a NumPy fallback runs without a compiler).
#'   `"auto"` (default) takes the dense route when `m <= 1000`, when the
#'   pattern of `C` fills more than 30 % of its lower triangle, or when its
#'   factor fills more than half of it.
#'   `"hybrid"` splits the unknowns into a dense separator (fixed effects and
#'   the terms that link otherwise independent groups of observations, found
#'   automatically) and independent blocks. The blocks are factorised as dense
#'   matrices in batches and the dense Schur complement of the separator is
#'   factorised last, on the GPU when one is available. When a block is too
#'   large, the route falls back to `"auto"` with a message.
#' @param optimizer `"ai"` (default) or `"lbfgsb"`; see [rx_fit()].
#' @param ... passed to [rx_fit()]: `backend`, `maxiter`, `polish`,
#'   `hessian`, `blups`, `vpredict`, `verbose`, and the others.
#' @return An object of class `rx_fit`, with `moteur = "mme"` and the
#'   inversion actually used in `inversion`. Prediction error variances
#'   (`pev`) are the diagonal of the random-effect block of `C^-1`, read from
#'   the same inversion. A fit that ends with a large relative gradient is
#'   finished by the dense engine from the point reached, and `moteur` then
#'   reads `"mme+dense"`.
#' @seealso [rx_fit()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' \donttest{
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' f <- rx_fit_mme(m, backend = "cpu", verbose = FALSE)
#' f$logLik
#' }
#' @name rx_fit_mme
rx_fit_mme <- function(model, inversion = c("auto", "dense", "sparse", "hybrid"), optimizer = "ai", ...) {
  inversion <- .rx_match_inversion(inversion)
  rx_fit(model, engine = "mme", inversion = inversion, optimizer = optimizer, ...)
}
