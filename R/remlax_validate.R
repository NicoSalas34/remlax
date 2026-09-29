# ==============================================================================
# remlax_validate.R - verifier remlax chez soi contre lme4, nlme et sommer, et
#                     lire la campagne de validation publiee
# ------------------------------------------------------------------------------
# rx_validate() refait une selection des comparaisons de la campagne de
# validation (depot remlax-validation) contre les paquets disponibles sur CRAN.
# Chaque cas ajuste le MEME modele sur les MEMES donnees simulees avec remlax et
# avec le logiciel de reference, et compare -2logL, effets fixes et composantes.
# asreml n'y figure pas : CRAN n'accepte pas de dependance a un paquet
# commercial. Ses resultats sont livres, precalcules, dans inst/extdata et lus
# par rx_validation_results().
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
  # Les jeux simules fixent leur graine. L'etat du generateur de l'utilisateur
  # est rendu tel qu'il etait a la sortie, comme le demande le CRAN.
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    graine <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit(assign(".Random.seed", graine, envir = globalenv()), add = TRUE)
  } else {
    on.exit(if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv()), add = TRUE)
  }
  rel <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)) / pmax(abs(as.numeric(b)), 1e-8))
  lignes <- list()
  ajouter <- function(reference, case, quantity, remlax, ref, gap, tol) {
    r <- data.frame(reference = reference, case = case, quantity = quantity,
                    remlax = remlax, reference_value = ref, gap = gap, tolerance = tol,
                    pass = is.finite(gap) && gap <= tol, stringsAsFactors = FALSE)
    lignes[[length(lignes) + 1L]] <<- r
    if (verbose)
      message(sprintf("  %-7s %-32s %-22s %-4s gap %.2e (tol %.0e)", reference, case, quantity,
                      if (r$pass) "ok" else "FAIL", gap, tol))
  }
  commun <- function(reference, case, fit, n2l_ref, beta_ref, comp_rx, comp_ref) {
    if (!is.null(n2l_ref))
      ajouter(reference, case, "-2 logL", -2 * fit$logLik, n2l_ref, abs(-2 * fit$logLik - n2l_ref), 1e-6)
    ajouter(reference, case, "fixed effects", NA_real_, NA_real_, rel(fit$beta, beta_ref), 1e-6)
    ajouter(reference, case, "variance parameters", NA_real_, NA_real_, rel(comp_rx, comp_ref), 1e-4)
  }
  inc <- function(f) Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1,
                                          dims = c(length(f), nlevels(f)))
  essai <- function(reference, case, expr) {
    r <- tryCatch(expr, error = function(e) e)
    if (inherits(r, "error"))
      ajouter(reference, case, "run", NA_real_, NA_real_, NA_real_, 0)
  }
  a_le <- function(p) {
    ok <- requireNamespace(p, quietly = TRUE)
    if (!ok) message("rx_validate: package '", p, "' not installed, skipped")
    ok
  }

  if ("lme4" %in% which && a_le("lme4")) {
    ctl <- lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6),
                             check.conv.singular = "ignore")
    n2l <- function(m) -2 * as.numeric(stats::logLik(m))
    essai("lme4", "one random factor", {
      set.seed(11); q <- 40; n <- q * 5
      g <- factor(rep(seq_len(q), each = 5)); x <- stats::rnorm(n)
      d <- data.frame(y = 2 + 0.5 * x + stats::rnorm(q, 0, 1.1)[g] + stats::rnorm(n), x = x, g = g)
      m <- lme4::lmer(y ~ x + (1 | g), d, REML = TRUE, control = ctl)
      f <- rx_reml(y ~ x, random = ~ g, data = d, backend = backend, verbose = FALSE)
      vc <- as.data.frame(lme4::VarCorr(m))$vcov
      commun("lme4", "one random factor", f, n2l(m), lme4::fixef(m),
             c(f$sigmas$g[1, 1], f$sigma_res[1, 1]), vc)
    })
    essai("lme4", "two crossed factors", {
      set.seed(12); a <- factor(rep(1:25, times = 8)); b <- factor(rep(1:8, each = 25)); n <- 200
      x <- stats::rnorm(n)
      d <- data.frame(y = 1 + 0.3 * x + stats::rnorm(25, 0, 1)[a] + stats::rnorm(8, 0, 0.7)[b] + stats::rnorm(n),
                      x = x, a = a, b = b)
      m <- lme4::lmer(y ~ x + (1 | a) + (1 | b), d, REML = TRUE, control = ctl)
      f <- rx_reml(y ~ x, random = ~ a + b, data = d, backend = backend, verbose = FALSE)
      vc <- as.data.frame(lme4::VarCorr(m))
      commun("lme4", "two crossed factors", f, n2l(m), lme4::fixef(m),
             c(f$sigmas$a[1, 1], f$sigmas$b[1, 1], f$sigma_res[1, 1]),
             c(vc$vcov[vc$grp == "a"], vc$vcov[vc$grp == "b"], vc$vcov[vc$grp == "Residual"]))
    })
    essai("lme4", "correlated random slopes", {
      set.seed(41); q <- 30; mrep <- 8; n <- q * mrep
      g <- factor(rep(seq_len(q), each = mrep)); x <- stats::rnorm(n)
      U <- matrix(stats::rnorm(q * 2), q, 2) %*% chol(matrix(c(1, 0.3, 0.3, 0.5), 2))
      y <- 1 + 0.5 * x + U[g, 1] + U[g, 2] * x + stats::rnorm(n, 0, 0.8)
      d <- data.frame(y = y, x = x, g = g)
      m <- lme4::lmer(y ~ x + (1 + x | g), d, REML = TRUE, control = ctl)
      Z0 <- inc(g)
      mod <- rx_model(y, stats::model.matrix(~ x, d),
                      list(rx_term("g", list(int = Z0, pente = Z0 * x), struct = "us", levels = levels(g))))
      f <- rx_fit(mod, backend = backend, verbose = FALSE)
      G4 <- as.matrix(lme4::VarCorr(m)$g)[1:2, 1:2]
      commun("lme4", "correlated random slopes", f, n2l(m), lme4::fixef(m),
             c(f$sigmas$g[c(1, 2, 4)], f$sigma_res[1, 1]), c(G4[c(1, 2, 4)], stats::sigma(m)^2))
    })
  }

  if ("nlme" %in% which && a_le("nlme")) {
    ctl <- nlme::glsControl(tolerance = 1e-10, msTol = 1e-10, msMaxIter = 500, opt = "nlminb")
    n2l <- function(m) -2 * as.numeric(stats::logLik(m))
    cor_par <- function(m) stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)
    serie <- function(G, Tn, seed, bruit) {
      set.seed(seed); n <- G * Tn
      g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- stats::rnorm(n)
      e <- unlist(lapply(seq_len(G), function(i) bruit(Tn)))
      data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
    }
    essai("nlme", "AR1 residual within groups", {
      d <- serie(12, 10, 1, function(Tn) as.numeric(stats::arima.sim(list(ar = 0.6), Tn)))
      m <- nlme::gls(y ~ x, d, correlation = nlme::corAR1(form = ~ t | g), method = "REML", control = ctl)
      f <- rx_reml(y ~ x, residual = ~ id(g):ar1(tf), data = d, trait = "g", backend = backend, verbose = FALSE)
      commun("nlme", "AR1 residual within groups", f, n2l(m), stats::coef(m),
             c(f$sigma_res[1, 1], f$rho[["residuelle"]]), c(m$sigma^2, cor_par(m)))
    })
    essai("nlme", "exponential in continuous time", {
      set.seed(2); G <- 10; Tn <- 12; n <- G * Tn
      g <- factor(rep(seq_len(G), each = Tn))
      pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(stats::runif(Tn, 0, 15)))))
      x <- stats::rnorm(n)
      e <- unlist(lapply(seq_len(G), function(i) {
        D <- as.matrix(stats::dist(pos[g == i])); as.numeric(t(chol(0.7 ^ D)) %*% stats::rnorm(Tn)) }))
      d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
      m <- nlme::gls(y ~ x, d, correlation = nlme::corCAR1(form = ~ pos | g), method = "REML", control = ctl)
      f <- rx_reml(y ~ x, residual = ~ id(g):exp(pos), data = d, trait = "g", backend = backend, verbose = FALSE)
      commun("nlme", "exponential in continuous time", f, n2l(m), stats::coef(m),
             c(f$sigma_res[1, 1], f$rho[["residuelle"]]), c(m$sigma^2, cor_par(m)))
    })
  }

  if ("sommer" %in% which && a_le("sommer")) {
    essai("sommer", "genomic relationship matrix", {
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
      commun("sommer", "genomic relationship matrix", f, NULL, so$Beta$Estimate,
             c(f$sigmas$gid[1, 1], f$sigma_res[1, 1]), c(so$sigma[[1]][1, 1], so$sigma[[2]][1, 1]))
    })
  }
  out <- do.call(rbind, lignes)
  rownames(out) <- NULL
  attr(out, "versions") <- c(remlax = as.character(utils::packageVersion("remlax")),
                             jax = chk$versions[["jax"]],
                             vapply(intersect(which, c("lme4", "nlme", "sommer")), function(p)
                               if (requireNamespace(p, quietly = TRUE)) as.character(utils::packageVersion(p)) else NA_character_,
                               character(1)))
  out
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
  fichier <- switch(what, checks = "validation_checks.csv", pairs = "validation_pairs.csv",
                    benchmark = "benchmark_table.csv")
  p <- system.file("extdata", fichier, package = "remlax")
  if (!nzchar(p)) stop("remlax: ", fichier, " not found in the installed package", call. = FALSE)
  out <- utils::read.csv(p, stringsAsFactors = FALSE, check.names = FALSE)
  v <- system.file("extdata", "validation_versions.json", package = "remlax")
  if (what != "benchmark" && nzchar(v))
    attr(out, "versions") <- jsonlite::fromJSON(v)
  out
}