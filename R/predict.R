# ==============================================================================
# 8. PREDICTIONS (predict d'asreml)
# ------------------------------------------------------------------------------
# Une prediction est une COMBINAISON LINEAIRE des effets ajustes. Tout le travail
# est de construire la bonne combinaison ; l'algebre, elle, tient en deux lignes.
#
#   classify : les variables tenues a chacun de leurs niveaux
#   les autres variables du modele fixe sont MOYENNEES :
#     - un facteur, sur ses niveaux (poids egaux, ou proportionnels aux effectifs
#       observes si average = "proportional") ;
#     - une covariable, a sa MOYENNE - c'est ce que fait asreml, et c'est ce qui
#       explique qu'une prediction "par variete" soit donnee a Column = 5.5.
#
# TERMES ALEATOIRES. Si une variable du classify est le facteur d'un terme
# aleatoire, son BLUP entre dans la prediction (c'est le cas d'une "moyenne de
# genotype"), et l'erreur-type devient une erreur de PREDICTION, calculee cote
# solveur ou vivent V^-1 et le projecteur. include_random = FALSE rend la
# prediction des seuls effets fixes.
#
# ESTIMABILITE. Une combinaison hors de l'espace des lignes de X n'est pas
# estimable : la rendre quand meme donnerait un nombre qui depend de la
# parametrisation choisie. Elle est marquee NON estimable et sa valeur est NA.
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
  if (is.null(fit$fixed) || is.null(fit$data))
    stop("rx_predict : cet ajustement ne vient pas de rx_reml() ; la formule des ",
         "effets fixes et les donnees ne sont pas connues.")
  if (!is.null(fit$model$lag))
    stop("rx_predict : non disponible avec un decalage spatial (rx_lag).", call. = FALSE)
  cls <- unlist(strsplit(classify, "[:+]"))
  cls <- trimws(cls[nzchar(cls)])
  data <- fit$data
  tt   <- stats::delete.response(stats::terms(fit$fixed, data = data))
  vars <- all.vars(tt)
  # Un classify peut designer un terme ALEATOIRE (le cas "moyenne par genotype")
  # : il n'est alors pas dans la formule des effets fixes, et c'est normal. Sa
  # contribution vient du BLUP, pas de X.
  noms_alea <- vapply(fit$model$terms, `[[`, "", "name")
  if (length(manque <- setdiff(cls, c(vars, noms_alea))))
    stop("rx_predict : ", paste(manque, collapse = ", "), " n'est ni dans le ",
         "modele fixe (", paste(vars, collapse = ", "), ") ni parmi les termes ",
         "aleatoires (", paste(noms_alea, collapse = ", "), ").")

  # --- grille : classify x (facteurs moyennes) ; covariables a leur moyenne ----
  vals <- list()
  for (v in union(vars, cls)) {
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
  grille <- expand.grid(vals, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  Xg <- stats::model.matrix(tt, grille, xlev = fit$xlevels)
  # Colonnes fixes d'une spl2d() : centrees, donc a zero elles moyennent la
  # surface sur le champ, comme une covariable tenue a sa moyenne.
  if (!is.null(fit$spl2d$columns))
    Xg <- cbind(Xg, matrix(0, nrow(Xg), length(fit$spl2d$columns),
                           dimnames = list(NULL, fit$spl2d$columns)))
  if (ncol(Xg) != length(fit$beta))
    stop("rx_predict : la grille donne ", ncol(Xg), " colonnes pour ",
         length(fit$beta), " coefficients. Un facteur a-t-il des niveaux absents ",
         "de la grille ?")

  # --- moyennage sur ce qui n'est pas dans le classify -------------------------
  cle <- if (length(cls)) do.call(paste, c(grille[cls], sep = "\r")) else rep("1", nrow(grille))
  ucle <- unique(cle)
  moy <- setdiff(vars, cls)
  moy_fac <- moy[vapply(moy, function(v) is.factor(vals[[v]]), TRUE)]
  poids_ligne <- rep(1, nrow(grille))
  if (average == "proportional" && length(moy_fac)) {
    # Poids PROPORTIONNELS aux effectifs observes de la combinaison moyennee.
    obs <- do.call(paste, c(lapply(moy_fac, function(v) as.character(data[[v]])), sep = "\r"))
    grl <- do.call(paste, c(lapply(moy_fac, function(v) as.character(grille[[v]])), sep = "\r"))
    tab <- table(obs)
    poids_ligne <- as.numeric(tab[grl]); poids_ligne[is.na(poids_ligne)] <- 0
  }
  if (!is.null(weights)) poids_ligne <- poids_ligne * weights[cle]
  L <- t(vapply(ucle, function(k) {
    i <- which(cle == k); w <- poids_ligne[i]
    if (sum(w) <= 0) stop("rx_predict : poids nuls pour la cellule '",
                          gsub("\r", ":", k), "'.")
    colSums(Xg[i, , drop = FALSE] * (w / sum(w)))
  }, numeric(ncol(Xg))))
  etiq <- do.call(rbind, lapply(strsplit(ucle, "\r", fixed = TRUE), function(z) z))
  out <- if (length(cls)) as.data.frame(etiq, stringsAsFactors = FALSE) else
         data.frame(prediction = seq_along(ucle))
  names(out) <- if (length(cls)) cls else "prediction"

  # --- estimabilite -----------------------------------------------------------
  X <- fit$model$X
  qrX <- qr(X)
  # L doit etre dans l'espace des LIGNES de X : on teste via la projection de
  # chaque ligne de L sur cet espace.
  Rx <- qr.R(qrX)[seq_len(qrX$rank), , drop = FALSE]
  base_lignes <- t(qr.Q(qr(t(Rx))))
  resid_L <- L - L %*% t(base_lignes) %*% base_lignes
  estimable <- sqrt(rowSums(resid_L^2)) < 1e-8 * pmax(sqrt(rowSums(L^2)), 1)

  # --- part aleatoire ---------------------------------------------------------
  M <- NULL
  if (isTRUE(include_random)) {
    for (tm in fit$model$terms) {
      v <- tm$name
      if (!v %in% cls || is.null(tm$levels)) next
      if (tm$t != 1L) {
        warning("rx_predict : terme '", v, "' multi-caractere ignore dans la ",
                "part aleatoire (t = ", tm$t, ").")
        next
      }
      Mi <- matrix(0, nrow(L), tm$q)
      j <- match(as.character(out[[v]]), tm$levels)
      if (anyNA(j)) next
      Mi[cbind(seq_len(nrow(L)), j)] <- 1
      M[[v]] <- Mi
    }
  }

  if (is.null(M)) {
    # Effets fixes seuls : tout se calcule ici, sans repasser par le solveur.
    Vb <- if (vcov == "kenward-roger") fit$vbeta_kr else fit$vbeta
    if (is.null(Vb))
      stop("rx_predict : la covariance des effets fixes n'est pas dans l'ajustement",
           if (vcov == "kenward-roger") " (relancer rx_reml(..., kenward_roger = TRUE))" else "",
           ".")
    val <- as.numeric(L %*% fit$beta)
    Cv  <- L %*% Vb %*% t(L)
  } else {
    if (vcov == "kenward-roger")
      warning("rx_predict : l'ajustement de Kenward-Roger porte sur les effets ",
              "FIXES ; la part aleatoire garde son erreur de prediction usuelle.")
    dir <- tempfile("rkpred_"); on.exit(unlink(dir, recursive = TRUE))
    rx_export(fit$model, dir)
    wbin <- function(nm, x) { con <- file(file.path(dir, nm), "wb")
      writeBin(as.double(as.vector(as.matrix(x))), con, size = 8); close(con) }
    wbin("in_theta.bin", fit$theta); wbin("pred_L.bin", L)
    for (nm in names(M)) wbin(paste0("pred_M_", nm, ".bin"), M[[nm]])
    py <- rx_python_cmd()
    .rx_python_require(py)
    a <- c(rx_solver_args(), dir, "--backend", backend, "--only-predict",
           if (!verbose) "--quiet" else NULL)
    .rx_run_solver(py, a, verbose, "rx_predict : le solveur a echoue")
    rr <- rx_read_result(dir)
    val <- as.numeric(rr$predictions$valeur); Cv <- rr$predictions$cov
  }
  out$predicted.value <- ifelse(estimable, val, NA_real_)
  out$std.error <- ifelse(estimable, sqrt(pmax(diag(Cv), 0)), NA_real_)
  out$estimable <- estimable
  attr(out, "vcov") <- Cv
  if (isTRUE(sed)) {
    d <- outer(diag(Cv), diag(Cv), "+") - 2 * Cv
    S <- sqrt(pmax(d, 0)); diag(S) <- NA_real_
    attr(out, "sed") <- S
    attr(out, "sed.moyen") <- sqrt(mean(S[upper.tri(S)]^2, na.rm = TRUE))
  }
  class(out) <- c("rx_predict", "data.frame")
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
