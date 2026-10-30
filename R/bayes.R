# ==============================================================================
# bayes.R - the Bayesian route (R side)
# ==============================================================================
# The computation is in Python (inst/python/remlax/bayes.py): Hamiltonian Monte
# Carlo adapted by the ChEES criterion (BlackJAX) on the restricted likelihood,
# with beta and u integrated out analytically, and many short chains run
# together. The R side holds the two constructors (rx_prior(), rx_bayes()), the
# prior written to the bundle (prior.json), the command-line flags, and the
# reading and printing of the draws.
# ==============================================================================

#' Prior distributions for the Bayesian route
#'
#' Priors are placed on the natural scale of each parameter and carried to the
#' internal parameters `theta` by the Jacobian of the change of variable.
#'
#' @param sd family of the prior on every standard deviation (iid and diag
#'   terms, diagonal of a `us` block, specific variances of an `fa` block,
#'   residual): `"half_t"` (default), `"half_normal"`, `"half_cauchy"`,
#'   `"inv_gamma"` (on the variance, with `shape` and `rate`) or `"uniform"`
#'   (improper, flat on the standard deviation).
#' @param df degrees of freedom of the half-t (default 3).
#' @param scale scale of the half-t, half-normal or half-Cauchy prior. `NULL`
#'   (default) takes the residual standard deviation of the ordinary least
#'   squares fit of `y` on `X`.
#' @param shape,rate parameters of the inverse gamma prior on a variance.
#' @param lkj shape `eta` of the LKJ prior on the correlation matrix of each
#'   `us` block (default 2; 1 is uniform over correlation matrices, larger
#'   values pull towards independence).
#' @param loading_scale standard deviation of the centred normal prior on the
#'   loadings of `fa` and `rr` blocks; `NULL` takes the same reference scale as
#'   `scale`.
#' @param level prior on correlation parameters between levels that are a
#'   hyperbolic tangent of `theta` (`ar1`, `ar1(row):ar1(col)`, `exp`, `gau`,
#'   `iexp`, `cor`, ...): `"uniform"` on the correlation (default) or
#'   `"normal"`, a normal prior on `theta` with standard deviation `theta_sd`.
#' @param theta_sd standard deviation of the normal prior placed on `theta`
#'   for the parameters without a natural prior (ranges, Matern parameters,
#'   rare covariance structures). A message lists them.
#' @param terms named list of per-term overrides, for instance
#'   `list(gid = list(sd = "half_normal", scale = 0.5), residual = list(scale = 2))`.
#'   Each element may set `sd`, `df`, `scale`, `shape`, `rate`, `lkj` and
#'   `loading_scale`. The residual is named `"residual"`.
#' @return An object of class `rx_prior`.
#' @seealso [rx_bayes()], [rx_fit()]
#' @examples
#' rx_prior()
#' rx_prior(sd = "half_normal", scale = 1, lkj = 4,
#'          terms = list(gid = list(scale = 0.5)))
#' @section Experimental:
#' The Bayesian route is experimental: it has been used on few data sets so far, and its
#' interface may change in a future version without a deprecation period.
#' @name rx_prior
rx_prior <- function(sd = c("half_t", "half_normal", "half_cauchy", "inv_gamma", "uniform"),
                     df = 3, scale = NULL, shape = 0.001, rate = 0.001, lkj = 2,
                     loading_scale = NULL, level = c("uniform", "normal"), theta_sd = 2,
                     terms = list()) {
  sd <- match.arg(sd); level <- match.arg(level)
  stopifnot(is.numeric(df), df > 0, is.numeric(lkj), lkj > 0, is.numeric(theta_sd), theta_sd > 0)
  if (!is.null(scale)) stopifnot(is.numeric(scale), scale > 0)
  if (length(terms) && (is.null(names(terms)) || any(!nzchar(names(terms)))))
    stop("rx_prior: `terms` must be a list NAMED by term", call. = FALSE)
  structure(list(sd = sd, df = df, scale = scale, shape = shape, rate = rate, lkj = lkj,
                 loading_scale = loading_scale, level = level, theta_sd = theta_sd,
                 terms = terms), class = "rx_prior")
}

print.rx_prior <- function(x, ...) {
  cat(sprintf("rx_prior: standard deviations %s%s, scale %s | LKJ(%g) | loadings N(0, %s) | levels %s\n",
              x$sd, if (x$sd == "half_t") sprintf("(df = %g)", x$df) else "",
              if (is.null(x$scale)) "OLS residual standard deviation" else format(x$scale),
              x$lkj, if (is.null(x$loading_scale)) "same scale" else format(x$loading_scale),
              x$level))
  if (length(x$terms)) cat("  overrides:", paste(names(x$terms), collapse = ", "), "\n")
  invisible(x)
}

#' Settings of the Bayesian route
#'
#' Passed as `bayes =` to [rx_fit()] or [rx_reml()]. By default no REML fit is
#' run: the posterior mode of `theta` is found first, and the Hessian of the
#' negative log posterior at the mode gives the starting point and the
#' whitening of the sampler (`start = "map"`). With `start = "reml"`, the model
#' is first fitted by REML, and the Laplace covariance `2 H^-1` at the REML
#' optimum is used instead; when that Hessian is not positive definite (an
#' optimum at a boundary), the posterior mode takes over. The posterior of the
#' variance parameters is then sampled by
#' Hamiltonian Monte Carlo with jittered trajectories, whose step size and
#' trajectory length are adapted by the ChEES criterion across all chains
#' (BlackJAX). All chains share the same trajectory length, so many short
#' chains advance together in the same vectorised calls, on CPU or GPU. The
#' sampler works on whitened parameters `z`, with `theta = theta_0 + L z` and
#' `L L'` the Laplace covariance at the starting point `theta_0`. Fixed
#' effects and random effects are integrated out analytically during sampling;
#' their posterior means and standard deviations are recovered afterwards, by
#' averaging over a subset of draws (Rao-Blackwellisation).
#'
#' @param start `"map"` (default): start from the posterior mode, without a
#'   REML fit, which matters when the REML fit alone takes hours. The fitted
#'   object then holds the variance components, BLUPs and fixed effects
#'   evaluated at the posterior mode, without REML standard errors.
#'   `"reml"`: fit by REML first, then sample; both results are returned.
#' @param chains number of chains, a multiple of `superchains`. `NULL`
#'   (default) takes 8 on the CPU and 32 on the GPU.
#' @param superchains number of groups of chains sharing an overdispersed
#'   starting point (default 4). The nested R-hat compares these groups, and
#'   stays valid when each chain is short.
#' @param warmup,draws adaptation iterations and kept draws per chain (default
#'   150 each).
#' @param seed random seed.
#' @param target_accept target acceptance rate of the step size adaptation
#'   (default 0.651, the optimum for HMC; raise it when divergences appear).
#' @param max_leapfrog cap on the number of leapfrog steps per trajectory
#'   (default 1000).
#' @param effects number of draws, spread over all chains, used for the
#'   posterior moments of the fixed effects and of the random effects (default
#'   100; 0 skips them). The posterior standard deviations of the random effects
#'   are computed for the terms named in `pev` of [rx_fit()].
#' @param prior an [rx_prior()] object.
#' @return An object of class `rx_bayes`.
#' @details Requires the Python package blackjax (`rx_install_python(bayes =
#'   TRUE)`). The posterior is evaluated by the dense engine: with `engine =
#'   "mme"` only the REML fit uses the mixed model equations. A spatial lag of
#'   the response is not supported yet. The result carries the draws of
#'   `theta`, of the variance components and of the `vpredict` expressions,
#'   a summary with posterior means, medians, 95 % credible intervals, split
#'   R-hat, nested R-hat and effective sample sizes, and sampler diagnostics
#'   (divergences, acceptance rate, step size, leapfrog steps).
#' @seealso [rx_prior()], [rx_fit()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok && rx_bayes_available()
#' \donttest{
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:40, each = 4)))
#' d$y <- rnorm(40)[d$gid] + rnorm(160)
#' fit <- rx_reml(y ~ 1, random = ~ gid, data = d, backend = "cpu",
#'                vpredict = c(h2 = "V1/(V1+V2)"),
#'                bayes = rx_bayes(chains = 4, superchains = 2, warmup = 100, draws = 100),
#'                verbose = FALSE)
#' fit$bayes$summary
#' }
#' @section Experimental:
#' The Bayesian route is experimental: it has been used on few data sets so far, and its
#' interface may change in a future version without a deprecation period.
#' @name rx_bayes
rx_bayes <- function(start = c("map", "reml"), chains = NULL, superchains = 4L,
                     warmup = 150L, draws = 150L, seed = 0L,
                     target_accept = 0.651, max_leapfrog = 1000L, effects = 100L,
                     prior = rx_prior()) {
  start <- match.arg(start)
  stopifnot(superchains >= 1, warmup >= 1, draws >= 4, target_accept > 0, target_accept < 1,
            max_leapfrog >= 1, effects >= 0, inherits(prior, "rx_prior"))
  if (!is.null(chains) && (chains < superchains || chains %% superchains != 0))
    stop("rx_bayes: `chains` must be a multiple of `superchains`", call. = FALSE)
  structure(list(start = start, chains = if (is.null(chains)) NULL else as.integer(chains),
                 superchains = as.integer(superchains), warmup = as.integer(warmup),
                 draws = as.integer(draws), seed = as.integer(seed),
                 target_accept = target_accept, max_leapfrog = as.integer(max_leapfrog),
                 effects = as.integer(effects), prior = prior), class = "rx_bayes")
}

#' @rdname rx_bayes
#' @return `rx_bayes_available()`: `TRUE` when the Python interpreter of
#'   remlax has blackjax and optax installed (checked without importing them).
rx_bayes_available <- function() {
  py <- rx_python_cmd()
  # find_spec tells whether the modules are installed without importing them
  # (importing blackjax imports JAX, several seconds).
  code <- paste("import importlib.util as u, sys;",
                "sys.exit(0 if u.find_spec('blackjax') and u.find_spec('optax') else 1)")
  out <- suppressWarnings(tryCatch(
    .rx_system2(py[1], shQuote(c(py[-1], "-c", code)), stdout = TRUE, stderr = TRUE),
    error = function(e) structure("", status = 1L)))
  is.null(attr(out, "status"))
}

# The prior as the JSON read by bayes._resolve_prior.
.rx_prior_json <- function(pr) {
  sd_spec <- function(o) {
    s <- list()
    if (!is.null(o$sd)) s$family <- o$sd
    if (!is.null(o$df)) s$df <- o$df
    if (!is.null(o$scale)) s$scale <- o$scale
    if (!is.null(o$shape)) s$shape <- o$shape
    if (!is.null(o$rate)) s$rate <- o$rate
    s
  }
  j <- list(sd = c(list(family = pr$sd, df = pr$df, shape = pr$shape, rate = pr$rate),
                   if (!is.null(pr$scale)) list(scale = pr$scale)),
            lkj = pr$lkj,
            loading = if (!is.null(pr$loading_scale)) list(scale = pr$loading_scale) else
              structure(list(), names = character(0)),
            level = pr$level,
            generic = pr$theta_sd,
            terms = structure(list(), names = character(0)))
  for (nm in names(pr$terms)) {
    o <- pr$terms[[nm]]; t <- list()
    s <- sd_spec(o)
    if (length(s)) t$sd <- s
    if (!is.null(o$lkj)) t$lkj <- o$lkj
    if (!is.null(o$loading_scale)) t$loading <- list(scale = o$loading_scale)
    j$terms[[nm]] <- if (length(t)) t else structure(list(), names = character(0))
  }
  j
}

# Command-line flags of the Bayesian route; writes prior.json to the bundle.
.rx_bayes_args <- function(bayes, dir) {
  if (is.null(bayes)) return(NULL)
  if (!inherits(bayes, "rx_bayes"))
    stop("rx_fit: `bayes` must come from rx_bayes()", call. = FALSE)
  jsonlite::write_json(.rx_prior_json(bayes$prior), file.path(dir, "prior.json"),
                       auto_unbox = TRUE, digits = NA)
  c("--bayes", "--start", bayes$start %||% "map",
    "--chains", bayes$chains %||% 0L, "--superchains", bayes$superchains,
    "--warmup", bayes$warmup, "--draws", bayes$draws, "--seed", bayes$seed,
    "--target-accept", format(bayes$target_accept, digits = 15),
    "--max-leapfrog", bayes$max_leapfrog, "--prior", "prior.json",
    "--bayes-effects", bayes$effects)
}

# Reads the draws, their summary and the
# Rao-Blackwellised means of beta and of the BLUPs; `rd` reads a binary file.
.rx_bayes_read <- function(r, rd) {
  b <- r[["bayes"]]
  if (is.null(b)) return(r)
  dm <- b[["dims"]]; C <- dm[["chains"]]; D <- dm[["draws"]]; N <- C * D
  out <- list(chain = rep(seq_len(C), each = D), draw = rep(seq_len(D), times = C))
  v <- rd("out_bayes_theta")
  if (!is.null(v)) out$theta <- matrix(v, N, dm[["p"]])
  v <- rd("out_bayes_comp")
  if (!is.null(v)) out$components <- matrix(v, N, dm[["k"]],
                                            dimnames = list(NULL, b[["component_names"]]))
  v <- rd("out_bayes_vp")
  if (!is.null(v)) out$functions <- matrix(v, N, dm[["e"]],
                                           dimnames = list(NULL, b[["expression_names"]]))
  s <- b[["summary"]]
  if (is.data.frame(s)) out$summary <- s
  out$diagnostics <- b[["diagnostics"]]
  out$prior <- b[["prior"]]
  ef <- b[["effects"]]
  if (!is.null(ef)) {
    bm <- rd("out_bayes_beta_mean"); p_ <- length(bm)
    out$beta <- list(mean = bm, cov = matrix(rd("out_bayes_beta_cov"), p_, p_), n_draws = ef[["n"]])
    out$blups <- list(mean = list(), sd = list())
    for (nm in names(ef[["blup_dims"]])) {
      d_ <- ef[["blup_dims"]][[nm]]
      v <- rd(paste0("out_bayes_blup_mean_", nm))
      if (!is.null(v)) out$blups$mean[[nm]] <- matrix(v, d_[1], d_[2])
      v <- rd(paste0("out_bayes_blup_sd_", nm))
      if (!is.null(v)) out$blups$sd[[nm]] <- matrix(v, d_[1], d_[2])
    }
  }
  r[["bayes"]] <- out
  r
}

# Printed by print.rx_fit().
.rx_bayes_print <- function(x) {
  b <- x[["bayes"]]
  if (is.null(b) || is.null(b$summary)) return(invisible())
  dg <- b$diagnostics
  cat(sprintf(paste0("  Posterior (%s, %d chains in %d superchains x %d draws): ",
                     "%d divergence(s), acceptance %.2f, max nested R-hat %.3f, min ESS %.0f, %.1f s\n"),
              dg$algorithm, as.integer(dg$chains), as.integer(dg$superchains),
              as.integer(dg$draws), as.integer(dg$divergences), dg$acceptance,
              dg$nested_rhat_max, dg$ess_min, dg$seconds))
  s <- b$summary
  for (i in seq_len(nrow(s)))
    cat(sprintf("    %-22s mean %10.5g  median %10.5g  CI95 [%10.5g, %10.5g]  nR-hat %.3f  ESS %5.0f\n",
                s$name[i], s$mean[i], s$median[i], s$q025[i], s$q975[i], s$nested_rhat[i], s$ess[i]))
  if (isTRUE(dg$divergences > 0))
    cat("    WARNING: divergences occurred. Increase target_accept (rx_bayes) and check the priors.\n")
  if (isTRUE(dg$nested_rhat_max > 1.01))
    cat("    WARNING: nested R-hat > 1.01. The superchains disagree. Lengthen the warmup.\n")
  invisible()
}
