# ==============================================================================
# predict.R - rx_predict(): predictions in the manner of asreml's predict()
# ==============================================================================
# A prediction is a linear combination of the fitted effects; the work is to
# build the right combination.
#
#   classify   the variables held at each of their levels
#   the other variables of the fixed model are averaged:
#     - a factor over its levels (equal weights, or proportional to the
#       observed counts with average = "proportional");
#     - a covariate at its mean, as asreml does.
#
# When a classify variable is the factor of a random term, its BLUP enters the
# prediction (a "genotype mean"), and the standard error becomes a prediction
# error, computed by the solver where V^-1 and the projector live.
# include_random = FALSE predicts from the fixed effects only.
#
# A combination outside the row space of X is not estimable: its value would
# depend on the parametrisation. It is flagged and its value is NA.
#
#   pv <- rx_predict(fit, classify = "trt")
#   pv <- rx_predict(fit, classify = "genotype", vcov = "kenward-roger", sed = TRUE)
# ==============================================================================

rx_predict <- function(fit, classify, levels = NULL, at = NULL,
                       average = c("equal", "proportional"), weights = NULL,
                       vcov = c("simple", "kenward-roger"),
                       sed = FALSE, include_random = TRUE,
                       backend = c("auto", "gpu", "cpu"), verbose = FALSE) {
  average <- match.arg(average); vcov <- match.arg(vcov); backend <- match.arg(backend)
  spec <- .rx_predict_check(fit, classify)
  g <- .rx_predict_grid(fit, spec, levels, at)
  Lo <- .rx_predict_L(g, spec, fit$data, average, weights)
  L <- Lo$L; out <- Lo$out
  estimable <- .rx_estimable(L, fit$model$X)
  M <- if (isTRUE(include_random)) .rx_predict_random(fit, spec$cls, out, nrow(L)) else NULL
  pv <- if (is.null(M)) .rx_predict_fixed(fit, L, vcov) else
        .rx_predict_solver(fit, L, M, vcov, backend, verbose)
  out$predicted.value <- ifelse(estimable, pv$val, NA_real_)
  out$std.error <- ifelse(estimable, sqrt(pmax(diag(pv$Cv), 0)), NA_real_)
  out$estimable <- estimable
  attr(out, "vcov") <- pv$Cv
  if (isTRUE(sed)) out <- .rx_add_sed(out, pv$Cv)
  class(out) <- c("rx_predict", "data.frame")
  out
}

# The classify variables, the terms of the fixed formula and its variables. A
# classify variable may name a random term (a genotype mean), which is not in
# the fixed formula: its contribution comes from the BLUP, not from X.
.rx_predict_check <- function(fit, classify) {
  if (is.null(fit$fixed) || is.null(fit$data))
    stop("rx_predict : cet ajustement ne vient pas de rx_reml() ; la formule des ",
         "effets fixes et les donnees ne sont pas connues.")
  if (!is.null(fit$model$lag))
    stop("rx_predict : non disponible avec un decalage spatial (rx_lag).", call. = FALSE)
  cls <- unlist(strsplit(classify, "[:+]"))
  cls <- trimws(cls[nzchar(cls)])
  tt   <- stats::delete.response(stats::terms(fit$fixed, data = fit$data))
  vars <- all.vars(tt)
  random_names <- vapply(fit$model$terms, `[[`, "", "name")
  if (length(missing_vars <- setdiff(cls, c(vars, random_names))))
    stop("rx_predict : ", paste(missing_vars, collapse = ", "), " n'est ni dans le ",
         "modele fixe (", paste(vars, collapse = ", "), ") ni parmi les termes ",
         "aleatoires (", paste(random_names, collapse = ", "), ").")
  list(cls = cls, tt = tt, vars = vars)
}

# The prediction grid: classify variables at their levels, factors to average
# at all their levels, covariates at their mean; and its fixed design. The
# fixed columns of a spl2d() surface are centred, so at zero they average the
# surface over the field.
.rx_predict_grid <- function(fit, spec, levels, at) {
  data <- fit$data; cls <- spec$cls
  vals <- list()
  for (v in union(spec$vars, cls)) {
    if (!v %in% names(data))
      stop("rx_predict : colonne '", v, "' absente des donnees de l'ajustement.")
    x <- data[[v]]
    if (!is.null(at[[v]])) { vals[[v]] <- at[[v]]; next }
    if (is.factor(x) || is.character(x) || is.logical(x)) {
      lv <- if (!is.null(levels[[v]])) as.character(levels[[v]]) else base::levels(factor(x))
      vals[[v]] <- factor(lv, levels = base::levels(factor(x)))
    } else {
      vals[[v]] <- if (v %in% cls) {
        if (!is.null(levels[[v]])) levels[[v]] else sort(unique(x))
      } else mean(x, na.rm = TRUE)
    }
  }
  grid <- expand.grid(vals, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  Xg <- stats::model.matrix(spec$tt, grid, xlev = fit$xlevels)
  if (!is.null(fit$spl2d$columns))
    Xg <- cbind(Xg, matrix(0, nrow(Xg), length(fit$spl2d$columns),
                           dimnames = list(NULL, fit$spl2d$columns)))
  if (ncol(Xg) != length(fit$beta))
    stop("rx_predict : la grille donne ", ncol(Xg), " colonnes pour ",
         length(fit$beta), " coefficients. Un facteur a-t-il des niveaux absents ",
         "de la grille ?")
  list(vals = vals, grid = grid, Xg = Xg)
}

# The averaging matrix L (one row per classify cell) and the table of cells.
# With average = "proportional", the averaged factors are weighted by their
# observed counts.
.rx_predict_L <- function(g, spec, data, average, weights) {
  cls <- spec$cls; grid <- g$grid; Xg <- g$Xg
  key <- if (length(cls)) do.call(paste, c(grid[cls], sep = "\r")) else rep("1", nrow(grid))
  ukey <- unique(key)
  averaged <- setdiff(spec$vars, cls)
  averaged_factors <- averaged[vapply(averaged, function(v) is.factor(g$vals[[v]]), TRUE)]
  row_weights <- rep(1, nrow(grid))
  if (average == "proportional" && length(averaged_factors)) {
    obs <- do.call(paste, c(lapply(averaged_factors, function(v) as.character(data[[v]])), sep = "\r"))
    grl <- do.call(paste, c(lapply(averaged_factors, function(v) as.character(grid[[v]])), sep = "\r"))
    tab <- table(obs)
    row_weights <- as.numeric(tab[grl]); row_weights[is.na(row_weights)] <- 0
  }
  if (!is.null(weights)) row_weights <- row_weights * weights[key]
  L <- t(vapply(ukey, function(k) {
    i <- which(key == k); w <- row_weights[i]
    if (sum(w) <= 0) stop("rx_predict : poids nuls pour la cellule '",
                          gsub("\r", ":", k), "'.")
    colSums(Xg[i, , drop = FALSE] * (w / sum(w)))
  }, numeric(ncol(Xg))))
  labels <- do.call(rbind, lapply(strsplit(ukey, "\r", fixed = TRUE), function(z) z))
  out <- if (length(cls)) as.data.frame(labels, stringsAsFactors = FALSE) else
         data.frame(prediction = seq_along(ukey))
  names(out) <- if (length(cls)) cls else "prediction"
  list(L = L, out = out)
}

# Estimability: each row of L must lie in the row space of X.
.rx_estimable <- function(L, X) {
  qrX <- qr(X)
  Rx <- qr.R(qrX)[seq_len(qrX$rank), , drop = FALSE]
  row_basis <- t(qr.Q(qr(t(Rx))))
  resid_L <- L - L %*% t(row_basis) %*% row_basis
  sqrt(rowSums(resid_L^2)) < 1e-8 * pmax(sqrt(rowSums(L^2)), 1)
}

# Random part: for each single-trait random term named in the classify, the
# indicator matrix M (cells x levels) of its BLUPs.
.rx_predict_random <- function(fit, cls, out, n_cells) {
  M <- NULL
  for (tm in fit$model$terms) {
    v <- tm$name
    if (!v %in% cls || is.null(tm$levels)) next
    if (tm$t != 1L) {
      warning("rx_predict : terme '", v, "' multi-caractere ignore dans la ",
              "part aleatoire (t = ", tm$t, ").")
      next
    }
    Mi <- matrix(0, n_cells, tm$q)
    j <- match(as.character(out[[v]]), tm$levels)
    if (anyNA(j)) next
    Mi[cbind(seq_len(n_cells), j)] <- 1
    M[[v]] <- Mi
  }
  M
}

# Fixed effects only: computed here from beta and its covariance.
.rx_predict_fixed <- function(fit, L, vcov) {
  Vb <- if (vcov == "kenward-roger") fit$vbeta_kr else fit$vbeta
  if (is.null(Vb))
    stop("rx_predict : la covariance des effets fixes n'est pas dans l'ajustement",
         if (vcov == "kenward-roger") " (relancer rx_reml(..., kenward_roger = TRUE))" else "",
         ".")
  list(val = as.numeric(L %*% fit$beta), Cv = L %*% Vb %*% t(L))
}

# With random effects: the solver evaluates the model at the fitted theta and
# computes the predictions and their prediction error covariance.
.rx_predict_solver <- function(fit, L, M, vcov, backend, verbose) {
  if (vcov == "kenward-roger")
    warning("rx_predict : l'ajustement de Kenward-Roger porte sur les effets ",
            "FIXES ; la part aleatoire garde son erreur de prediction usuelle.")
  dir <- tempfile("rkpred_"); on.exit(unlink(dir, recursive = TRUE))
  rx_export(fit$model, dir)
  .rx_write_bin(file.path(dir, "in_theta.bin"), fit$theta)
  .rx_write_bin(file.path(dir, "pred_L.bin"), L)
  for (nm in names(M)) .rx_write_bin(file.path(dir, paste0("pred_M_", nm, ".bin")), M[[nm]])
  py <- rx_python_cmd()
  .rx_python_require(py)
  a <- c(rx_solver_args(), dir, "--backend", backend, "--only-predict",
         if (!verbose) "--quiet" else NULL)
  .rx_run_solver(py, a, verbose, "rx_predict : le solveur a echoue")
  rr <- rx_read_result(dir)
  list(val = as.numeric(rr$predictions$valeur), Cv = rr$predictions$cov)
}

# Standard errors of the differences between predictions, and their mean.
.rx_add_sed <- function(out, Cv) {
  d <- outer(diag(Cv), diag(Cv), "+") - 2 * Cv
  S <- sqrt(pmax(d, 0)); diag(S) <- NA_real_
  attr(out, "sed") <- S
  attr(out, "sed.moyen") <- sqrt(mean(S[upper.tri(S)]^2, na.rm = TRUE))
  out
}

print.rx_predict <- function(x, ...) {
  cat("Predictions\n")
  print(as.data.frame(x), row.names = FALSE, digits = 6)
  if (!is.null(attr(x, "sed.moyen")))
    cat(sprintf("\n  erreur-type moyenne des differences : %.6f\n", attr(x, "sed.moyen")))
  if (any(!x$estimable))
    cat(sprintf("  %d prediction(s) NON estimable(s)\n", sum(!x$estimable)))
  invisible(x)
}
