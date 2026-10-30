# ==============================================================================
# summary.R - summary of a fit, in the layout of summary.asreml
# ==============================================================================
# From summary.asreml: call, loglik, nedf (residual degrees of freedom), aic,
# bic, varcomp (component, std.error, z.ratio, bound), and coef.fixed and
# coef.random on request. Added: the convergence, judged on the slope
# (projected gradient and Newton decrement) rather than on the change of logLik
# between iterations, and the log-likelihood in both conventions (complete, and
# that of asreml without the constant).
#
# The standard errors of the components come from the solver (table
# `varcomp`: delta method, exact Jacobian, covariance 2 H^-1 on the directions
# of positive curvature). Bound codes: F fixed, B at a bound, P variance,
# U unconstrained.

summary.rx_fit <- function(object, coef = FALSE, ...) {
  x <- object
  varcomp <- .rx_varcomp_table(x)
  X <- x$model$X
  p <- if (!is.null(X)) qr(as.matrix(X))$rank else length(x$beta)
  n <- as.integer(x$n_obs %||% NA)
  nedf <- n - p
  n_par <- as.integer(x$n_par %||% nrow(varcomp))
  # Entries of Sigma with the same pattern in V (the residual-IEE block of a
  # plant term [I, W] with a symmetric W) enter only through their sum: they
  # count neither in the AIC and BIC nor in the degrees of freedom of a LRT.
  n_plat <- as.integer(x[["n_flat_expected"]] %||% 0L)
  n_est <- n_par - n_plat - as.integer(x$n_fixed %||% 0)
  m2l <- -2 * x$logLik
  info <- data.frame(
    criterion = c("logLik", "logLik (asreml)", "AIC", "BIC"),
    value = c(x$logLik, x$logLik_asreml %||% NA, m2l + 2 * n_est, m2l + n_est * log(nedf)))

  fixed <- .rx_fixed_table(x, X)
  random <- if (isTRUE(coef) && length(x$blups)) .rx_random_table(x) else NULL
  # Effective dimensions, as in SpATS, whenever the PEV are available.
  dims <- if (length(x$pev) && !is.null(x$model))
    tryCatch(rx_dimensions(x), error = function(e) NULL) else NULL

  conv <- c(decrement = x$conv_decrement, gradient = x$conv_grad_rel, hessian = x$conv_hessian_ok)
  structure(list(
    call = x$call, backend = x$backend, n_obs = n, nedf = nedf, n_par = n_par,
    n_estimated = n_est, n_not_identified = n_plat, n_at_bound = as.integer(x$n_at_bound %||% NA),
    loglik = x$logLik, loglik_asreml = x$logLik_asreml %||% NA,
    aic = info$value[3], bic = info$value[4], criteria = info,
    converged = length(conv) > 0 && all(as.logical(conv)),
    convergence = list(max_grad = x$max_grad, newton_decrement = x$newton_decrement %||% NA,
                       n_iter = x$n_iter %||% NA, n_neg_eig = x$n_neg_eig %||% NA,
                       seconds = x$seconds %||% NA),
    varcomp = varcomp, dimensions = dims, surfaces = .rx_spats_names(x),
    lag = if (!is.null(x[["lag"]])) x[["lag"]][c("criterion", "path", "logdet_A", "logdet_mean",
                                                 "spectral_radius", "admissible", "logLik_ystar",
                                                 "logLik_jacobian", "Psi", "Psi_se")],
    coef.fixed = fixed, coef.random = random,
    vpredict = x$vpredict$predictions, wald = x$wald$tests),
    class = "summary.rx_fit")
}

# Table of the variance components (component, std.error, z.ratio, bound),
# from the solver's table `varcomp`.
.rx_varcomp_table <- function(x) {
  vc <- x$varcomp
  if (is.null(vc) || !NROW(vc)) {
    comp_names <- x$component_names %||% character()
    vc <- data.frame(i = seq_along(comp_names), name = comp_names, value = NA_real_, se = NA_real_,
                     constraint = NA_character_, stringsAsFactors = FALSE)
    if (length(comp_names)) warning("summary: the fit has no table of components ",
                                    "(it may come from an earlier engine). Refit the model.", call. = FALSE)
  }
  varcomp <- data.frame(component = as.numeric(vc$value),
                        std.error = as.numeric(vc$se),
                        z.ratio = as.numeric(vc$value) / as.numeric(vc$se),
                        bound = as.character(vc$constraint),
                        row.names = make.unique(as.character(vc$name)),
                        stringsAsFactors = FALSE)
  varcomp$z.ratio[!is.finite(varcomp$z.ratio)] <- NA_real_
  varcomp
}

# Fixed effects with their standard errors and z ratios.
.rx_fixed_table <- function(x, X) {
  if (!length(x$beta)) return(NULL)
  se <- if (!is.null(x$vbeta)) sqrt(pmax(diag(as.matrix(x$vbeta)), 0)) else NA_real_
  nm <- if (!is.null(X) && !is.null(colnames(X))) colnames(X) else paste0("beta", seq_along(x$beta))
  data.frame(solution = as.numeric(x$beta), std.error = se,
             z.ratio = as.numeric(x$beta) / se, row.names = nm)
}

# BLUPs of every term, labelled term_level (or term_level:column), with the
# square root of their PEV when available.
.rx_random_table <- function(x) {
  do.call(rbind, lapply(names(x$blups), function(tn) {
    B <- as.matrix(x$blups[[tn]])
    P <- x$pev[[tn]]
    lv <- rownames(B) %||% as.character(seq_len(nrow(B)))
    cols <- colnames(B) %||% as.character(seq_len(ncol(B)))
    lab <- if (ncol(B) == 1L) paste0(tn, "_", lv)
           else paste0(tn, "_", rep(lv, ncol(B)), ":", rep(cols, each = nrow(B)))
    se <- if (!is.null(P) && length(P) == length(B)) sqrt(pmax(as.numeric(P), 0)) else NA_real_
    data.frame(solution = as.numeric(B), std.error = se, row.names = lab)
  }))
}

print.summary.rx_fit <- function(x, digits = max(4L, getOption("digits") - 3L), ...) {
  cat("Mixed model fitted by REML (remlax, ", x$backend %||% "?", ")\n", sep = "")
  if (!is.null(x$call)) cat("Call: ", paste(deparse(x$call, width.cutoff = 70L), collapse = "\n  "),
                            "\n", sep = "")
  cat(sprintf("\n%d observations, %d residual df, %d variance parameter(s), %d of them estimated%s%s\n",
              x$n_obs, x$nedf, x$n_par, x$n_estimated,
              if (isTRUE(x$n_not_identified > 0))
                sprintf(" and identified (%d not identified: entries N)", x$n_not_identified) else "",
              if (isTRUE(x$n_at_bound > 0)) sprintf(", %d at a bound", x$n_at_bound) else ""))
  cat(sprintf("logLik %.4f (asreml convention: %.4f)   AIC %.2f   BIC %.2f\n",
              x$loglik, x$loglik_asreml, x$aic, x$bic))
  cv <- x$convergence
  cat(sprintf("Convergence: %s | max|grad| %.2e | Newton decrement %.2e | %s iteration(s) | %.1f s\n",
              if (isTRUE(x$converged)) "yes" else "NO", cv$max_grad, cv$newton_decrement,
              format(cv$n_iter), cv$seconds))
  if (isTRUE(cv$n_neg_eig > 0))
    cat(sprintf("  WARNING: %d negative eigenvalue(s) of the Hessian (possible saddle point)\n",
                cv$n_neg_eig))

  cat("\nVariance components:\n")
  vc <- x$varcomp
  vc$component <- signif(vc$component, digits); vc$std.error <- signif(vc$std.error, digits)
  vc$z.ratio <- round(vc$z.ratio, 2)
  print(vc, na.print = "")
  cat("  bound: P variance, U unconstrained, B at a bound, F fixed, N not identified alone",
      "(its sum \"a+b\" is)\n")

  if (!is.null(x$lag)) {
    Lg <- x$lag
    cat(sprintf("\nSpatial lag (Psi: receiver in rows, emitter in columns, units of W)\n"))
    cat(sprintf("  criterion %s | log|det(I - M)| %.6f | mean term %.6f | spectral radius of M %.4f | path %s%s\n",
                Lg$criterion, Lg$logdet_A, Lg$logdet_mean, Lg$spectral_radius, Lg$path,
                if (isFALSE(Lg$admissible)) " | OUTSIDE THE ADMISSIBLE REGION" else ""))
    cat(sprintf("  logLik of y* alone %.4f | with the Jacobian %.4f\n", Lg$logLik_ystar,
                Lg$logLik_jacobian %||% NA_real_))
    if (!is.null(Lg$Psi)) {
      cat("  Psi:\n"); print(signif(Lg$Psi, digits))
      if (!is.null(Lg$Psi_se)) { cat("  standard errors:\n"); print(signif(Lg$Psi_se, digits)) }
    }
  }
  if (!is.null(x$dimensions)) {
    cat("\nDimensions (in the manner of SpATS):\n")
    dd <- x$dimensions
    print(data.frame(Effective = round(dd$Effective, 1), Model = dd$Model,
                     Nominal = round(dd$Nominal, 1), Ratio = round(dd$Ratio, 3), Type = dd$Type,
                     row.names = rownames(dd)), na.print = "")
    for (ln in x$surfaces) cat("  ", ln, "\n", sep = "")
    cat("  For a genotype without relationship matrix, Ratio = generalised heritability (Oakey et al. 2006).\n")
  }
  if (!is.null(x$vpredict) && NROW(x$vpredict)) {
    cat("\nFunctions of the components (vpredict):\n")
    vp <- data.frame(estimate = signif(x$vpredict$value, digits),
                     std.error = signif(x$vpredict$se, digits),
                     expression = x$vpredict$expression, row.names = x$vpredict$name)
    print(vp, na.print = "")
  }
  if (!is.null(x$coef.fixed)) {
    cat("\nFixed effects:\n")
    cf <- x$coef.fixed
    print(data.frame(solution = signif(cf$solution, digits), std.error = signif(cf$std.error, digits),
                     z.ratio = round(cf$z.ratio, 2), row.names = rownames(cf)))
  }
  if (!is.null(x$wald) && NROW(x$wald)) {
    cat("\nWald tests (conditional):\n")
    print(data.frame(df = x$wald$df, F.value = round(x$wald$F, 3),
                     Pr = format.pval(x$wald$p, digits = 3), row.names = x$wald$term))
  }
  if (!is.null(x$coef.random)) {
    cr <- x$coef.random
    cat(sprintf("\nRandom effects (BLUP): %d level(s)%s\n", nrow(cr),
                if (nrow(cr) > 20L) ", first 20 shown (see $coef.random)" else ""))
    print(utils::head(data.frame(solution = signif(cr$solution, digits),
                                 std.error = signif(cr$std.error, digits),
                                 row.names = rownames(cr)), 20L), na.print = "")
  }
  invisible(x)
}

# The terms of each spl2d() surface with their SpATS names, one line per
# surface: "spl_fx = f(col), spl_fy = f(row), ...".
.rx_spats_names <- function(fit) {
  sf <- fit$spl2d$surfaces
  if (!length(sf)) return(character())
  vapply(sf, function(s) {
    if (is.null(s$basis$terms)) return("")
    x <- s$x_name; y <- s$y_name
    ref <- c(sprintf("f(%s)", x), sprintf("f(%s)", y), sprintf("f(%s):%s", x, y),
             sprintf("%s:f(%s)", x, y), sprintf("f(%s):f(%s)", x, y))
    paste(paste(s$basis$terms, "=", ref), collapse = ", ")
  }, "")
}
