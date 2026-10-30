# ==============================================================================
# validate.R - check remlax on the user's machine against lme4, nlme and
#              sommer, and read the published validation campaign
# ==============================================================================
# rx_validate() reruns a subset of the comparisons of the validation campaign
# (repository remlax-validation) against packages available on CRAN. Each case
# fits the same model on the same simulated data with remlax and with the
# reference package, and compares -2 logL, the fixed effects and the variance
# parameters. asreml is absent: CRAN does not accept a dependency on a
# commercial package. Its results ship precomputed in inst/extdata and are read
# by rx_validation_results().
# ==============================================================================

#' Check remlax against lme4, nlme and sommer on this machine
#'
#' Fits a set of models with remlax and with a reference package on the same
#' simulated data, and compares the REML log-likelihood at the optimum, the
#' fixed effects and the variance parameters. The cases are a subset of the
#' published validation campaign (see [rx_validation_results()] and the
#' vignette `vignette("validation", package = "remlax")`); asreml is not
#' included because CRAN packages cannot depend on it.
#'
#' The log-likelihood is compared in each package's own convention: lme4 and
#' nlme report the REML log-likelihood with its constant, which is
#' `fit$logLik`; sommer reports a likelihood shifted by a data-dependent
#' constant, so only its estimates are compared.
#'
#' @param which reference packages to use; those not installed are skipped
#'   with a message.
#' @param backend passed to [rx_fit()].
#' @param verbose print one line per check.
#' @return A data frame with one row per check: `reference`, `case`,
#'   `quantity`, `remlax`, `reference_value`, `gap` (absolute for the
#'   log-likelihood, largest relative gap otherwise), `tolerance` and `pass`.
#' @examplesIf rx_python_check(quiet = TRUE)$ok && requireNamespace("lme4", quietly = TRUE)
#' \donttest{
#' v <- rx_validate("lme4", verbose = FALSE)
#' table(v$case, v$pass)
#' }
#' @export
rx_validate <- function(which = c("lme4", "nlme", "sommer"), backend = "cpu",
                        verbose = TRUE) {
  which <- match.arg(which, several.ok = TRUE)
  chk <- rx_python_check(quiet = TRUE)
  if (!isTRUE(chk$ok))
    stop("rx_validate() needs the Python solver: ", chk$message, call. = FALSE)
  # The simulated data sets fix their seed; the user's generator state is
  # restored on exit, as CRAN requires.
  restore_seed <- .rx_save_seed()
  on.exit(restore_seed(), add = TRUE)
  log <- .rx_validation_log(verbose)
  if ("lme4" %in% which && .rx_has_package("lme4")) .rx_validate_lme4(log, backend)
  if ("nlme" %in% which && .rx_has_package("nlme")) .rx_validate_nlme(log, backend)
  if ("sommer" %in% which && .rx_has_package("sommer")) .rx_validate_sommer(log, backend)
  out <- do.call(rbind, log$rows)
  rownames(out) <- NULL
  attr(out, "versions") <- c(remlax = as.character(utils::packageVersion("remlax")),
                             jax = chk$versions[["jax"]],
                             vapply(intersect(which, c("lme4", "nlme", "sommer")), function(p)
                               if (requireNamespace(p, quietly = TRUE)) as.character(utils::packageVersion(p)) else NA_character_,
                               character(1)))
  out
}

# Saves the state of the global random generator and returns the function
# that restores it, for on.exit() in the caller.
.rx_save_seed <- function() {
  env <- globalenv()
  if (exists(".Random.seed", envir = env, inherits = FALSE)) {
    seed <- get(".Random.seed", envir = env, inherits = FALSE)
    function() assign(".Random.seed", seed, envir = env)
  } else {
    function() if (exists(".Random.seed", envir = env, inherits = FALSE))
      rm(".Random.seed", envir = env)
  }
}

# Is a reference package installed? A missing one is skipped with a message.
.rx_has_package <- function(p) {
  ok <- requireNamespace(p, quietly = TRUE)
  if (!ok) message("rx_validate: package '", p, "' not installed, skipped")
  ok
}

# Largest relative gap between two vectors.
.rx_rel_gap <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)) / pmax(abs(as.numeric(b)), 1e-8))
}

# Indicator incidence (n x levels) of a factor.
.rx_indicator <- function(f) {
  Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1,
                       dims = c(length(f), nlevels(f)))
}

# The log of the checks: one row per check in log$rows, printed as it is added
# when verbose.
.rx_validation_log <- function(verbose) {
  log <- new.env(parent = emptyenv())
  log$rows <- list()
  log$verbose <- verbose
  log
}

.rx_validation_add <- function(log, reference, case, quantity, remlax, ref, gap, tol) {
  r <- data.frame(reference = reference, case = case, quantity = quantity,
                  remlax = remlax, reference_value = ref, gap = gap, tolerance = tol,
                  pass = is.finite(gap) && gap <= tol, stringsAsFactors = FALSE)
  log$rows[[length(log$rows) + 1L]] <- r
  if (log$verbose)
    message(sprintf("  %-7s %-32s %-22s %-4s gap %.2e (tol %.0e)", reference, case, quantity,
                    if (r$pass) "ok" else "FAIL", gap, tol))
}

# The checks shared by every case: -2 logL (when the reference reports it in
# the same convention), fixed effects and variance parameters.
.rx_validation_compare <- function(log, reference, case, fit, n2l_ref, beta_ref, comp_rx, comp_ref) {
  if (!is.null(n2l_ref))
    .rx_validation_add(log, reference, case, "-2 logL", -2 * fit$logLik, n2l_ref,
                       abs(-2 * fit$logLik - n2l_ref), 1e-6)
  .rx_validation_add(log, reference, case, "fixed effects", NA_real_, NA_real_,
                     .rx_rel_gap(fit$beta, beta_ref), 1e-6)
  .rx_validation_add(log, reference, case, "variance parameters", NA_real_, NA_real_,
                     .rx_rel_gap(comp_rx, comp_ref), 1e-4)
}

# Runs one case; an error is recorded as a failed "run" check.
.rx_validation_case <- function(log, reference, case, expr) {
  r <- tryCatch(expr, error = function(e) e)
  if (inherits(r, "error"))
    .rx_validation_add(log, reference, case, "run", NA_real_, NA_real_, NA_real_, 0)
}

# -2 logL of a fitted lme4 or nlme model.
.rx_n2l <- function(m) -2 * as.numeric(stats::logLik(m))

.rx_validate_lme4 <- function(log, backend) {
  ctl <- lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6),
                           check.conv.singular = "ignore")
  .rx_validation_case(log, "lme4", "one random factor", {
    set.seed(11); q <- 40; n <- q * 5
    g <- factor(rep(seq_len(q), each = 5)); x <- stats::rnorm(n)
    d <- data.frame(y = 2 + 0.5 * x + stats::rnorm(q, 0, 1.1)[g] + stats::rnorm(n), x = x, g = g)
    m <- lme4::lmer(y ~ x + (1 | g), d, REML = TRUE, control = ctl)
    f <- rx_reml(y ~ x, random = ~ g, data = d, backend = backend, verbose = FALSE)
    vc <- as.data.frame(lme4::VarCorr(m))$vcov
    .rx_validation_compare(log, "lme4", "one random factor", f, .rx_n2l(m), lme4::fixef(m),
                           c(f$sigmas$g[1, 1], f$sigma_res[1, 1]), vc)
  })
  .rx_validation_case(log, "lme4", "two crossed factors", {
    set.seed(12); a <- factor(rep(1:25, times = 8)); b <- factor(rep(1:8, each = 25)); n <- 200
    x <- stats::rnorm(n)
    d <- data.frame(y = 1 + 0.3 * x + stats::rnorm(25, 0, 1)[a] + stats::rnorm(8, 0, 0.7)[b] + stats::rnorm(n),
                    x = x, a = a, b = b)
    m <- lme4::lmer(y ~ x + (1 | a) + (1 | b), d, REML = TRUE, control = ctl)
    f <- rx_reml(y ~ x, random = ~ a + b, data = d, backend = backend, verbose = FALSE)
    vc <- as.data.frame(lme4::VarCorr(m))
    .rx_validation_compare(log, "lme4", "two crossed factors", f, .rx_n2l(m), lme4::fixef(m),
                           c(f$sigmas$a[1, 1], f$sigmas$b[1, 1], f$sigma_res[1, 1]),
                           c(vc$vcov[vc$grp == "a"], vc$vcov[vc$grp == "b"], vc$vcov[vc$grp == "Residual"]))
  })
  .rx_validation_case(log, "lme4", "correlated random slopes", {
    set.seed(41); q <- 30; mrep <- 8; n <- q * mrep
    g <- factor(rep(seq_len(q), each = mrep)); x <- stats::rnorm(n)
    U <- matrix(stats::rnorm(q * 2), q, 2) %*% chol(matrix(c(1, 0.3, 0.3, 0.5), 2))
    y <- 1 + 0.5 * x + U[g, 1] + U[g, 2] * x + stats::rnorm(n, 0, 0.8)
    d <- data.frame(y = y, x = x, g = g)
    m <- lme4::lmer(y ~ x + (1 + x | g), d, REML = TRUE, control = ctl)
    Z0 <- .rx_indicator(g)
    mod <- rx_model(y, stats::model.matrix(~ x, d),
                    list(rx_term("g", list(int = Z0, slope = Z0 * x), struct = "us", levels = levels(g))))
    f <- rx_fit(mod, backend = backend, verbose = FALSE)
    G4 <- as.matrix(lme4::VarCorr(m)$g)[1:2, 1:2]
    .rx_validation_compare(log, "lme4", "correlated random slopes", f, .rx_n2l(m), lme4::fixef(m),
                           c(f$sigmas$g[c(1, 2, 4)], f$sigma_res[1, 1]), c(G4[c(1, 2, 4)], stats::sigma(m)^2))
  })
}

# Simulated series: G groups of Tn successive times, with residuals drawn by
# noise(Tn) within each group.
.rx_validation_series <- function(G, Tn, seed, noise) {
  set.seed(seed); n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- stats::rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) noise(Tn)))
  data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
}

.rx_validate_nlme <- function(log, backend) {
  ctl <- nlme::glsControl(tolerance = 1e-10, msTol = 1e-10, msMaxIter = 500, opt = "nlminb")
  cor_par <- function(m) stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)
  .rx_validation_case(log, "nlme", "AR1 residual within groups", {
    d <- .rx_validation_series(12, 10, 1, function(Tn) as.numeric(stats::arima.sim(list(ar = 0.6), Tn)))
    m <- nlme::gls(y ~ x, d, correlation = nlme::corAR1(form = ~ t | g), method = "REML", control = ctl)
    f <- rx_reml(y ~ x, residual = ~ id(g):ar1(tf), data = d, trait = "g", backend = backend, verbose = FALSE)
    .rx_validation_compare(log, "nlme", "AR1 residual within groups", f, .rx_n2l(m), stats::coef(m),
                           c(f$sigma_res[1, 1], f$rho[["residual"]]), c(m$sigma^2, cor_par(m)))
  })
  .rx_validation_case(log, "nlme", "exponential in continuous time", {
    set.seed(2); G <- 10; Tn <- 12; n <- G * Tn
    g <- factor(rep(seq_len(G), each = Tn))
    pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(stats::runif(Tn, 0, 15)))))
    x <- stats::rnorm(n)
    e <- unlist(lapply(seq_len(G), function(i) {
      D <- as.matrix(stats::dist(pos[g == i])); as.numeric(t(chol(0.7 ^ D)) %*% stats::rnorm(Tn)) }))
    d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
    m <- nlme::gls(y ~ x, d, correlation = nlme::corCAR1(form = ~ pos | g), method = "REML", control = ctl)
    f <- rx_reml(y ~ x, residual = ~ id(g):exp(pos), data = d, trait = "g", backend = backend, verbose = FALSE)
    .rx_validation_compare(log, "nlme", "exponential in continuous time", f, .rx_n2l(m), stats::coef(m),
                           c(f$sigma_res[1, 1], f$rho[["residual"]]), c(m$sigma^2, cor_par(m)))
  })
}

# sommer reports a likelihood shifted by a data-dependent constant: only its
# estimates are compared.
.rx_validate_sommer <- function(log, backend) {
  .rx_validation_case(log, "sommer", "genomic relationship matrix", {
    set.seed(61); q <- 50; n <- q * 4
    M <- matrix(stats::rbinom(q * 300, 2, 0.3), q, 300)
    K <- sommer::A.mat(M - 1) + diag(1e-4, q)
    dimnames(K) <- list(paste0("g", 1:q), paste0("g", 1:q))
    set.seed(62)
    gid <- factor(rep(paste0("g", 1:q), each = 4), levels = paste0("g", 1:q))
    x <- stats::rnorm(n)
    u <- as.numeric(t(chol(K)) %*% stats::rnorm(q)) * 1.2
    d <- data.frame(y = 5 + 0.4 * x + u[as.integer(gid)] + stats::rnorm(n), x = x, gid = gid)
    so <- suppressWarnings(sommer::mmer(y ~ x, random = ~ sommer::vsr(gid, Gu = K), rcov = ~ units,
                                        data = d, verbose = FALSE, tolParInv = 1e-8,
                                        tolParConvLL = 1e-12, nIters = 300))
    f <- rx_reml(y ~ x, random = ~ vm(gid, K = K), data = d, backend = backend, verbose = FALSE)
    .rx_validation_compare(log, "sommer", "genomic relationship matrix", f, NULL, so$Beta$Estimate,
                           c(f$sigmas$gid[1, 1], f$sigma_res[1, 1]), c(so$sigma[[1]][1, 1], so$sigma[[2]][1, 1]))
  })
}

#' Results of the published validation campaign
#'
#' Reads the tables shipped with the package: every check of the validation
#' campaign against asreml, lme4, sommer, nlme, pbkrtest and closed forms,
#' with its verdict, and the paired estimates behind them. They are copies of
#' the tables of the repository remlax-validation, which holds the scripts
#' that produce them and needs an asreml licence and a cluster for part of
#' them.
#'
#' @param what `"checks"` (one row per check: `suite`, `section`, `check`,
#'   `reference`, `verdict`, `detail`), `"pairs"` (one row per compared
#'   quantity: `kind`, `software`, `check`, `remlax`, `reference`) or
#'   `"benchmark"` (one row per timed fit of the speed benchmark).
#' @return A data frame, with attribute `versions` (the software versions and
#'   the remlax revision tested) for `"checks"` and `"pairs"`.
#' @examples
#' v <- rx_validation_results()
#' table(v$reference, v$verdict)
#' @export
rx_validation_results <- function(what = c("checks", "pairs", "benchmark")) {
  what <- match.arg(what)
  file <- switch(what, checks = "validation_checks.csv", pairs = "validation_pairs.csv",
                 benchmark = "benchmark_table.csv")
  p <- system.file("extdata", file, package = "remlax")
  if (!nzchar(p)) stop("remlax: ", file, " not found in the installed package", call. = FALSE)
  out <- utils::read.csv(p, stringsAsFactors = FALSE, check.names = FALSE)
  v <- system.file("extdata", "validation_versions.json", package = "remlax")
  if (what != "benchmark" && nzchar(v))
    attr(out, "versions") <- jsonlite::fromJSON(v)
  out
}