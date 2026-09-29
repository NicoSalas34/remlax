# ==============================================================================
# summary() D'UN AJUSTEMENT, A LA MANIERE DE summary.asreml
# ==============================================================================
#
# Ce que rend summary.asreml, et ce que l'on reprend :
#   call, loglik, nedf (ddl residuels), aic, bic, varcomp (component,
#   std.error, z.ratio, bound), coef.fixed et coef.random sur demande.
# Ce que l'on ajoute : la convergence, jugee sur la pente (gradient projete et
#   decrement de Newton) et non sur la variation de logLik entre iterations,
#   et la vraisemblance dans les deux conventions (complete, et celle
#   d'asreml, sans la constante), pour qu'une comparaison soit directe.
#
# Les erreurs-types des composantes viennent du moteur (table `varcomp`,
# methode delta, jacobienne exacte, covariance 2 H^-1 sur les directions de
# courbure positive). Codes de contrainte : F fixe, B borne, P variance,
# U non contrainte.

summary.rx_fit <- function(object, coef = FALSE, ...) {
  x <- object
  vc <- x$varcomp
  if (is.null(vc) || !NROW(vc)) {
    noms <- x$composantes_noms %||% character()
    vc <- data.frame(i = seq_along(noms), nom = noms, valeur = NA_real_, se = NA_real_,
                     contrainte = NA_character_, stringsAsFactors = FALSE)
    if (length(noms)) warning("summary : ajustement sans table des composantes ",
                              "(moteur anterieur ?) ; refaire l'ajustement.", call. = FALSE)
  }
  varcomp <- data.frame(component = as.numeric(vc$valeur),
                        std.error = as.numeric(vc$se),
                        z.ratio = as.numeric(vc$valeur) / as.numeric(vc$se),
                        bound = as.character(vc$contrainte),
                        row.names = make.unique(as.character(vc$nom)),
                        stringsAsFactors = FALSE)
  varcomp$z.ratio[!is.finite(varcomp$z.ratio)] <- NA_real_

  X <- x$model$X
  p <- if (!is.null(X)) qr(as.matrix(X))$rank else length(x$beta)
  n <- as.integer(x$n_obs %||% NA)
  nedf <- n - p
  n_par <- as.integer(x$n_par %||% nrow(varcomp))
  n_est <- n_par - as.integer(x$n_fixed %||% 0)
  m2l <- -2 * x$logLik
  info <- data.frame(
    criterion = c("logLik", "logLik (asreml)", "AIC", "BIC"),
    value = c(x$logLik, x$logLik_asreml %||% NA, m2l + 2 * n_est, m2l + n_est * log(nedf)))

  fixed <- NULL
  if (length(x$beta)) {
    se <- if (!is.null(x$vbeta)) sqrt(pmax(diag(as.matrix(x$vbeta)), 0)) else NA_real_
    nm <- if (!is.null(X) && !is.null(colnames(X))) colnames(X) else paste0("beta", seq_along(x$beta))
    fixed <- data.frame(solution = as.numeric(x$beta), std.error = se,
                        z.ratio = as.numeric(x$beta) / se, row.names = nm)
  }

  random <- NULL
  if (isTRUE(coef) && length(x$blups)) {
    random <- do.call(rbind, lapply(names(x$blups), function(tn) {
      B <- as.matrix(x$blups[[tn]])
      P <- x$pev[[tn]]
      niv <- rownames(B) %||% as.character(seq_len(nrow(B)))
      car <- colnames(B) %||% as.character(seq_len(ncol(B)))
      lab <- if (ncol(B) == 1L) paste0(tn, "_", niv)
             else paste0(tn, "_", rep(niv, ncol(B)), ":", rep(car, each = nrow(B)))
      se <- if (!is.null(P) && length(P) == length(B)) sqrt(pmax(as.numeric(P), 0)) else NA_real_
      data.frame(solution = as.numeric(B), std.error = se, row.names = lab)
    }))
  }

  # Dimensions effectives a la SpATS, des que des PEV sont disponibles.
  dims <- if (length(x$pev) && !is.null(x$model))
    tryCatch(rx_dimensions(x), error = function(e) NULL) else NULL

  conv <- c(decrement = x$conv_decrement, gradient = x$conv_grad_rel, hessian = x$conv_hessien_ok)
  structure(list(
    call = x$call, backend = x$backend, n_obs = n, nedf = nedf, n_par = n_par,
    n_estimated = n_est, n_at_bound = as.integer(x$n_at_bound %||% NA),
    loglik = x$logLik, loglik_asreml = x$logLik_asreml %||% NA,
    aic = info$value[3], bic = info$value[4], criteria = info,
    converged = length(conv) > 0 && all(as.logical(conv)),
    convergence = list(max_grad = x$max_grad, newton_decrement = x$newton_decrement %||% NA,
                       n_iter = x$n_iter %||% NA, n_neg_eig = x$n_neg_eig %||% NA,
                       seconds = x$secondes %||% NA),
    varcomp = varcomp, dimensions = dims, coef.fixed = fixed, coef.random = random,
    vpredict = x$vpredict$predictions, wald = x$wald$tests),
    class = "summary.rx_fit")
}

print.summary.rx_fit <- function(x, digits = max(4L, getOption("digits") - 3L), ...) {
  cat("Modele mixte ajuste par REML (remlax, ", x$backend %||% "?", ")\n", sep = "")
  if (!is.null(x$call)) cat("Appel : ", paste(deparse(x$call, width.cutoff = 70L), collapse = "\n  "),
                            "\n", sep = "")
  cat(sprintf("\n%d observations, %d ddl residuels, %d parametre(s) de variance dont %d estime(s)%s\n",
              x$n_obs, x$nedf, x$n_par, x$n_estimated,
              if (isTRUE(x$n_at_bound > 0)) sprintf(", %d a une borne", x$n_at_bound) else ""))
  cat(sprintf("logLik %.4f (convention asreml : %.4f)   AIC %.2f   BIC %.2f\n",
              x$loglik, x$loglik_asreml, x$aic, x$bic))
  cv <- x$convergence
  cat(sprintf("Convergence : %s | max|grad| %.2e | decrement de Newton %.2e | %s iteration(s) | %.1f s\n",
              if (isTRUE(x$converged)) "oui" else "NON", cv$max_grad, cv$newton_decrement,
              format(cv$n_iter), cv$seconds))
  if (isTRUE(cv$n_neg_eig > 0))
    cat(sprintf("  ATTENTION : %d valeur(s) propre(s) negative(s) du Hessien (point de selle ?)\n",
                cv$n_neg_eig))

  cat("\nComposantes de variance :\n")
  vc <- x$varcomp
  vc$component <- signif(vc$component, digits); vc$std.error <- signif(vc$std.error, digits)
  vc$z.ratio <- round(vc$z.ratio, 2)
  print(vc, na.print = "")
  cat("  bound : P variance, U non contrainte, B a une borne, F fixee\n")

  if (!is.null(x$dimensions)) {
    cat("\nDimensions (a la maniere de SpATS) :\n")
    dd <- x$dimensions
    print(data.frame(Effective = round(dd$Effective, 1), Model = dd$Model,
                     Nominal = round(dd$Nominal, 1), Ratio = round(dd$Ratio, 3), Type = dd$Type,
                     row.names = rownames(dd)), na.print = "")
    cat("  Pour un genotype sans parente, Ratio = heritabilite generalisee (Oakey et al. 2006).\n")
  }
  if (!is.null(x$vpredict) && NROW(x$vpredict)) {
    cat("\nFonctions des composantes (vpredict) :\n")
    vp <- data.frame(estimate = signif(x$vpredict$valeur, digits),
                     std.error = signif(x$vpredict$se, digits),
                     expression = x$vpredict$expression, row.names = x$vpredict$nom)
    print(vp, na.print = "")
  }
  if (!is.null(x$coef.fixed)) {
    cat("\nEffets fixes :\n")
    cf <- x$coef.fixed
    print(data.frame(solution = signif(cf$solution, digits), std.error = signif(cf$std.error, digits),
                     z.ratio = round(cf$z.ratio, 2), row.names = rownames(cf)))
  }
  if (!is.null(x$wald) && NROW(x$wald)) {
    cat("\nTests de Wald (conditionnels) :\n")
    print(data.frame(df = x$wald$ddl, F.value = round(x$wald$F, 3),
                     Pr = format.pval(x$wald$p, digits = 3), row.names = x$wald$terme))
  }
  if (!is.null(x$coef.random)) {
    cr <- x$coef.random
    cat(sprintf("\nEffets aleatoires (BLUP) : %d niveau(x)%s\n", nrow(cr),
                if (nrow(cr) > 20L) ", 20 premiers affiches (voir $coef.random)" else ""))
    print(utils::head(data.frame(solution = signif(cr$solution, digits),
                                 std.error = signif(cr$std.error, digits),
                                 row.names = rownames(cr)), 20L), na.print = "")
  }
  invisible(x)
}
