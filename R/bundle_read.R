rx_read_result <- function(dir, model = NULL, fixed_theta = NULL) {
  f <- file.path(dir, "result.json")
  if (!file.exists(f)) stop("resultat introuvable : ", f)
  r <- jsonlite::fromJSON(f, simplifyVector = TRUE)
  rd <- function(nm) { p <- file.path(dir, paste0(nm, ".bin"))
    if (file.exists(p)) readBin(p, "double", n = file.size(p) %/% 8, size = 8) else NULL }
  # ATTENTION : ACCES EXACTS, PAS `$`. `$` fait du APPARIEMENT PARTIEL sur les
  # listes. Depuis que le resultat porte un champ `sigmas_res`, l'expression
  # `r$sigmas` (qui n'existe pas encore a ce stade) s'appariait sur
  # `sigmas_res` : les matrices des termes etaient ajoutees a la liste des
  # residuelles, et `fit$sigmas[[1]]` rendait la RESIDUELLE au lieu du premier
  # terme. Toutes les comparaisons positionnelles des suites basculaient d'un
  # cran, sans aucune erreur - la logLik, elle, restait juste.
  r[["theta"]] <- rd("out_theta"); r[["beta"]] <- rd("out_beta")
  p_ <- length(r[["beta"]])
  v <- rd("out_vbeta");    if (!is.null(v)) r[["vbeta"]]    <- matrix(v, p_, p_)
  v <- rd("out_vbeta_kr"); if (!is.null(v)) r[["vbeta_kr"]] <- matrix(v, p_, p_)
  v <- rd("out_pred_cov")
  if (!is.null(v)) { k_ <- sqrt(length(v))
    r[["predictions"]][["cov"]] <- matrix(v, k_, k_) }
  r[["sigmas"]] <- list(); r[["sigmas_res"]] <- list(); r[["blups"]] <- list()
  for (nm in names(r[["sigma_dims"]])) {
    v <- rd(paste0("out_sigma_", nm))
    if (!is.null(v)) r[["sigmas"]][[nm]] <- matrix(v, r[["sigma_dims"]][[nm]][1],
                                                   r[["sigma_dims"]][[nm]][2])
  }
  for (nm in names(r[["sigma_res_dims"]])) {
    v <- rd(paste0("out_sigmares_", nm))
    if (!is.null(v)) r[["sigmas_res"]][[nm]] <- matrix(v, r[["sigma_res_dims"]][[nm]][1],
                                                       r[["sigma_res_dims"]][[nm]][2])
  }
  v <- rd("out_sigma_res")
  if (!is.null(v)) r[["sigma_res"]] <- matrix(v, sqrt(length(v)), sqrt(length(v)))
  # LOADINGS ET PEV, ajoutes pour le pipeline IGE. Sigma assemblee ne suffit pas
  # a un tableau de loadings : sa decomposition n'est pas unique. Et H2_Cullis
  # n'est pas calculable sans la variance d'erreur de prediction.
  r[["loadings"]] <- list(); r[["pev"]] <- list()
  for (nm in names(r[["loadings_dims"]])) {
    v <- rd(paste0("out_loadings_", nm))
    if (is.null(v)) next
    L <- matrix(v, r[["loadings_dims"]][[nm]][1], r[["loadings_dims"]][[nm]][2])
    ps <- if (nm %in% names(r[["psi_dims"]])) rd(paste0("out_psi_", nm)) else NULL
    # `psi` est la VARIANCE specifique, convention des tableaux d'asreml : la
    # diagonale de Sigma vaut somme(V_k^2) + psi. Une structure `rr` est un rang
    # reduit pur et n'en a pas : NULL, jamais des zeros qui se liraient comme
    # une mesure.
    r[["loadings"]][[nm]] <- list(Lambda = L, psi = ps)
  }
  for (nm in names(r[["pev_dims"]])) {
    v <- rd(paste0("out_pev_", nm))
    if (!is.null(v)) r[["pev"]][[nm]] <- matrix(v, r[["pev_dims"]][[nm]][1],
                                                r[["pev_dims"]][[nm]][2])
  }
  for (nm in names(r[["blup_dims"]])) {
    v <- rd(paste0("out_blup_", nm))
    if (!is.null(v)) r[["blups"]][[nm]] <- matrix(v, r[["blup_dims"]][[nm]][1],
                                                  r[["blup_dims"]][[nm]][2])
  }
  # LE HESSIEN. cli.py l'ECRIT deja en binaire a cote (out_hessian.bin), mais
  # ce lecteur ne le reprenait pas : cote R seul le drapeau derive
  # conv_hessien_ok survivait. Sans la matrice, un verdict "12 directions quasi
  # nulles sur 57" n'est pas actionnable - les vecteurs propres disent QUELS
  # parametres ne sont pas identifies. Mesure sur le modele a 5 caracteres :
  # 0 valeur propre negative, 12 directions quasi nulles, conditionnement 2,4e9.
  v <- rd("out_hessian")
  if (!is.null(v)) {
    q_ <- as.integer(round(sqrt(length(v))))
    if (q_ * q_ == length(v)) r[["hessian"]] <- matrix(v, q_, q_)
    else warning(sprintf("out_hessian a %d valeurs, non carre : ignore", length(v)))
  }
  # DECALAGE SPATIAL : Psi (unites de W), ses erreurs-types (methode delta du
  # moteur), et y* au point final (beta et les BLUP sont ceux du modele de y*).
  if (!is.null(r[["lag"]])) {
    q_ <- as.integer(r[["lag"]][["q"]])
    tn <- as.character(r[["lag"]][["traits"]])
    v <- rd("out_lag_psi")
    if (!is.null(v)) r[["lag"]][["Psi"]] <- matrix(v, q_, q_, dimnames = list(tn, tn))
    v <- rd("out_lag_psi_se")
    if (!is.null(v)) r[["lag"]][["Psi_se"]] <- matrix(v, q_, q_, dimnames = list(tn, tn))
    r[["ystar"]] <- rd("out_ystar")
  }
  # Voie bayesienne : tirages, resume et diagnostics (remlax_bayes.R).
  r <- .rx_bayes_read(r, rd)
  # Erreurs-types sur theta, sur le sous-espace libre ; NA sans Hessien.
  r[["fixed_theta"]] <- if (length(fixed_theta)) as.integer(fixed_theta) else integer(0)
  r[["se_theta"]] <- rx_se_theta(r[["theta"]], r[["hessian"]],
                                 r[["par_floor"]] %||% -12, r[["par_ceil"]] %||% 12,
                                 r[["fixed_theta"]])
  # Noms des colonnes de Sigma et des BLUP, quand les termes en portent.
  if (!is.null(model) && !is.null(model[["terms"]])) {
    for (tm in model[["terms"]]) {
      cn <- tm[["colnames"]]; nm <- tm[["name"]]
      S <- r[["sigmas"]][[nm]]
      if (!is.null(S) && !is.null(cn) && length(cn) == nrow(S))
        dimnames(r[["sigmas"]][[nm]]) <- list(cn, cn)
      U <- r[["blups"]][[nm]]
      if (!is.null(U)) {
        if (!is.null(cn) && length(cn) == ncol(U)) colnames(r[["blups"]][[nm]]) <- cn
        lv <- tm[["levels"]]
        if (!is.null(lv) && length(lv) == nrow(U)) rownames(r[["blups"]][[nm]]) <- lv
      }
    }
  }
  class(r) <- "rx_fit"; r
}
