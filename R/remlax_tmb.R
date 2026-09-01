# =============================================================================
# remlax : moteur CREUX, par RTMB
# =============================================================================
# POURQUOI UN SECOND MOTEUR, ET PAS UN SECOND BACKEND DU PREMIER.
# Le moteur dense forme V (n x n) et la differentie : son gradient est
# d(-2logL)/dV = P - (Py)(Py)', ou P est PLEINE meme quand V est creuse. Rendre V
# creuse ne gagnerait donc rien. Le creux exige l'autre formulation, celle des
# equations du modele mixte, ou le cout depend du REMPLISSAGE de
#
#     C = [ X'R^-1 X   X'R^-1 Z        ]
#         [ Z'R^-1 X   Z'R^-1 Z + G^-1 ]
#
# Les deux moteurs ne sont pas deux implementations d'une meme formule : ce sont
# deux formules equivalentes dont une seule est creuse.
#
# CE QUE TMB APPORTE, ET QUI SERAIT LONG A ECRIRE.
# On declare beta ET u comme effets aleatoires, et on laisse TMB faire son
# approximation de Laplace. Pour un modele lineaire gaussien cette approximation
# est EXACTE, et integrer beta en plus de u donne precisement la vraisemblance
# REML. TMB fournit alors gratuitement les trois primitives qui manquent a JAX :
# la Cholesky creuse avec permutation reduisant le remplissage, le
# log-determinant, et le gradient exact par differentiation automatique a travers
# la factorisation. Il n'y a pas d'inverse partielle a implementer.
#
# PERIMETRE, FIXE ET ETROIT. Le moteur ne couvre que les structures dont
# l'inverse est creux ET connu en forme fermee :
#
#   entre caracteres   iid, diag, us
#   entre niveaux      id, ar1, ar1ar1, et une parente fournie en A^-1 creuse
#   residuelle         iid, diag entre caracteres
#
# Tout le reste — noyaux metriques, mtrn, sph, cor, parente genomique — a un
# inverse PLEIN et ne gagnerait rien : ces modeles restent sur le moteur dense.
# Ce n'est pas une limitation temporaire, c'est le resultat du diagnostic de
# remlax.sparsity : sur la forme reelle d'un modele IGE (parente genomique plus
# champ spatial a q = n), C est 1,5 fois plus grande que V et remplie a 81 %, et
# l'avantage du creux tombe a un facteur 4 en flops — qui ne survit pas aux
# surcouts des formats creux.
#
# AUCUNE INVERSION NUMERIQUE. Les trois structures de niveaux retenues ont un
# inverse et un log-determinant analytiques :
#   id      K^-1 = I,                      log|K| = 0
#   ar1     K^-1 tridiagonale (ci-dessous), log|K| = (q-1) log(1 - phi^2)
#   ar1ar1  K^-1 = K1^-1 (x) K2^-1,         log|K| = q2 log|K1| + q1 log|K2|
#
# RISQUE PRINCIPAL : LA PARAMETRISATION EXISTE MAINTENANT EN DEUX EXEMPLAIRES.
# RTMB enregistre du code R, donc chol_sigma et level_chol doivent etre portes
# ici. Deux copies d'une parametrisation divergent silencieusement. Le garde-fou
# est tests/R/test_remlax_tmb_parity.R, qui compare Sigma(theta) et K(theta)
# entre R et Python sur des theta tires au hasard, et le depart a chaud
# (theta_init avec maxiter = 0), qui impose a un moteur le theta de l'autre et
# compare les log-vraisemblances : c'est le seul test qui distingue « les deux
# calculent une fonction differente » de « les deux s'arretent ailleurs ».
# =============================================================================

rx_tmb_available <- function() {
  requireNamespace("RTMB", quietly = TRUE) && requireNamespace("Matrix", quietly = TRUE)
}

# -----------------------------------------------------------------------------
# Perimetre
# -----------------------------------------------------------------------------
RX_SPARSE_SIGMA <- c("iid", "diag", "us")
RX_SPARSE_LEVEL <- c("id", "iid", "ar1", "ar1ar1", "prec")   # "prec" = A^-1 fournie

#' Le modele est-il dans le perimetre du moteur creux ?
#'
#' Rend une liste (ok, reason). La raison est destinee a l'utilisateur : elle
#' nomme la structure qui sort du perimetre plutot que de dire « non ».
rx_sparse_scope <- function(terms, residual = NULL) {
  for (tm in terms) {
    st <- tm$struct %||% "iid"
    if (!st %in% RX_SPARSE_SIGMA)
      return(list(ok = FALSE, reason = sprintf(
        "terme '%s' : structure entre caracteres '%s' hors perimetre creux (retenues : %s)",
        tm$name %||% "?", st, paste(RX_SPARSE_SIGMA, collapse = ", "))))
    lv <- tm$lvl %||% (if (!is.null(tm$Kinv)) "prec" else if (!is.null(tm$LK)) "fixed" else "id")
    if (!lv %in% RX_SPARSE_LEVEL)
      return(list(ok = FALSE, reason = sprintf(
        "terme '%s' : structure entre niveaux '%s' a un inverse plein, le creux n'y gagne rien%s",
        tm$name %||% "?", lv,
        if (identical(lv, "fixed"))
          " (une parente fournie en LK dense : fournir Kinv creuse pour passer au creux)" else "")))
  }
  rs <- residual$struct %||% "iid"
  if (!rs %in% c("iid", "diag"))
    return(list(ok = FALSE, reason = sprintf(
      "residuelle '%s' hors perimetre creux (retenues : iid, diag)", rs)))
  if (!is.null(residual$sections))
    return(list(ok = FALSE, reason = "residuelle sectionnee : hors perimetre creux"))
  list(ok = TRUE, reason = "dans le perimetre")
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# -----------------------------------------------------------------------------
# Parametrisation entre caracteres : PORT de structures.chol_sigma
# -----------------------------------------------------------------------------
#' Facteur L tel que Sigma = L L'.
#'
#' L'ordre de theta est celui du cote Python et il est LOAD-BEARING : triangle
#' inferieur ligne par ligne (i croissant, puis j <= i), diagonale exponentiee.
#' Toute divergence ici rend les deux moteurs incomparables sans lever d'erreur.
rx_chol_sigma_R <- function(theta, struct, t) {
  t <- as.integer(t)
  if (struct == "iid") return(exp(theta[1]) * diag(t))
  if (struct == "diag") return(diag(exp(theta[seq_len(t)]), nrow = t))
  if (struct == "us") {
    L <- matrix(0, t, t)
    k <- 1L
    for (i in seq_len(t)) for (j in seq_len(i)) { L[i, j] <- theta[k]; k <- k + 1L }
    diag(L) <- exp(diag(L))
    return(L)
  }
  stop("structure entre caracteres hors perimetre creux : ", struct)
}

rx_n_sigma_params <- function(struct, t) {
  t <- as.integer(t)
  switch(struct, iid = 1L, diag = t, us = as.integer(t * (t + 1) / 2),
         stop("structure inconnue : ", struct))
}

# -----------------------------------------------------------------------------
# Precision entre niveaux : K^-1 CREUSE et log|K|, tous deux analytiques
# -----------------------------------------------------------------------------
#' Precision d'un AR(1) de correlation K_ij = phi^|i-j|.
#'
#' Forme close, tridiagonale :
#'   K^-1 = 1/(1-phi^2) * T,  T_11 = T_qq = 1, T_ii = 1+phi^2 sinon,
#'                            T_{i,i+1} = T_{i+1,i} = -phi
#' et det(K) = (1-phi^2)^(q-1). Aucune factorisation n'est faite : c'est
#' precisement pourquoi ar1 est dans le perimetre creux.
rx_ar1_prec <- function(phi, q) {
  q <- as.integer(q)
  s <- 1 - phi * phi
  d <- rep(1 + phi * phi, q); d[1] <- 1; d[q] <- 1
  P <- Matrix::bandSparse(q, k = c(0, 1), symmetric = TRUE,
                          diagonals = list(d / s, rep(-phi / s, q - 1)))
  list(P = P, logdet = (q - 1) * log(s))
}

#' K^-1 et log|K| d'un terme, sur l'echelle non contrainte de theta.
rx_level_prec_R <- function(theta_lv, kind, q, dims = NULL, Kinv = NULL,
                            Kinv_logdet = NULL) {
  q <- as.integer(q)
  if (kind %in% c("id", "iid"))
    return(list(P = Matrix::Diagonal(q), logdet = 0))
  if (kind == "ar1") {
    return(rx_ar1_prec(tanh(theta_lv[1]), q))
  }
  if (kind == "ar1ar1") {
    if (is.null(dims)) stop("ar1ar1 exige dims = c(n_lignes, n_colonnes)")
    a <- rx_ar1_prec(tanh(theta_lv[1]), dims[1])
    b <- rx_ar1_prec(tanh(theta_lv[2]), dims[2])
    # K = K1 (x) K2 donc K^-1 = K1^-1 (x) K2^-1, et
    # log|K1 (x) K2| = q2 log|K1| + q1 log|K2|
    return(list(P = Matrix::kronecker(a$P, b$P),
                logdet = dims[2] * a$logdet + dims[1] * b$logdet))
  }
  if (kind == "prec") {
    if (is.null(Kinv)) stop("structure 'prec' : fournir Kinv creuse")
    ld <- Kinv_logdet
    if (is.null(ld)) {
      # log|K| = -log|K^-1|, par une Cholesky creuse faite UNE FOIS : K^-1 est
      # fournie et ne depend d'aucun parametre, donc son determinant est une
      # constante et sort de l'optimisation.
      ch <- Matrix::Cholesky(Matrix::forceSymmetric(Kinv), LDL = FALSE)
      ld <- -2 * sum(log(Matrix::diag(as(ch, "Matrix"))))
    }
    return(list(P = Kinv, logdet = ld))
  }
  stop("structure entre niveaux hors perimetre creux : ", kind)
}

rx_n_level_params <- function(kind) {
  switch(kind, id = 0L, iid = 0L, ar1 = 1L, ar1ar1 = 2L, prec = 0L,
         stop("structure entre niveaux inconnue : ", kind))
}
