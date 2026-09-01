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

# -----------------------------------------------------------------------------
# Formes quadratiques : ECRITES DIRECTEMENT, sans former K^-1
# -----------------------------------------------------------------------------
# CHOIX DE CONCEPTION. On pourrait construire K^-1 en matrice creuse a entrees
# differentiables et ecrire u' K^-1 u. On ne le fait pas, pour deux raisons :
#
#  1. Cela suppose que la bibliotheque de differentiation sache porter des
#     valeurs AD dans un format creux — une dependance a une API dont le
#     comportement varie selon les versions.
#  2. C'est inutile. Pour les trois structures du perimetre, la forme quadratique
#     a une expression FERMEE en vecteurs, exacte et differentiable sans effort :
#
#       u' K^-1 u = 1/(1-phi^2) [ sum u_i^2 + phi^2 sum_{i=2}^{q-1} u_i^2
#                                 - 2 phi sum_{i=1}^{q-1} u_i u_{i+1} ]
#
# La structure creuse du hessien reste vue par TMB : l'expression ne couple que
# des niveaux voisins, donc la factorisation reste creuse sans qu'on ait a le
# declarer. On gagne la robustesse sans rien perdre.
#
# Une K^-1 FOURNIE (parente genealogique) est un cas different : ses entrees sont
# des constantes, donc un produit matriciel creux ordinaire suffit.

#' u' K^-1 u pour un AR(1), sans former la matrice.
rx_quad_ar1 <- function(u, phi) {
  q <- length(u)
  s <- 1 - phi * phi
  interieur <- if (q > 2) sum(u[2:(q - 1)]^2) else 0
  crois <- if (q > 1) sum(u[1:(q - 1)] * u[2:q]) else 0
  (sum(u * u) + phi * phi * interieur - 2 * phi * crois) / s
}

#' u' (K1^-1 (x) K2^-1) u sur une grille nr x nc.
#'
#' u est indexe niveau = (ligne - 1) * nc + colonne, l'ordre du produit de
#' Kronecker du cote Python (kron(L_lignes, L_colonnes)). On applique donc la
#' precision des colonnes a chaque ligne, puis celle des lignes aux colonnes du
#' resultat — separabilite, jamais de matrice de taille q x q.
rx_quad_ar1ar1 <- function(u, phi_r, phi_c, nr, nc) {
  U <- matrix(u, nrow = nr, ncol = nc, byrow = TRUE)
  sr <- 1 - phi_r * phi_r
  sc <- 1 - phi_c * phi_c
  # P = Pr (x) Pc, donc u'Pu = tr(U' Pr U Pc). On forme Pr U par colonnes et
  # Pc par lignes, en n'ecrivant que des operations tridiagonales.
  tri <- function(M, phi, s, par_ligne) {
    k <- if (par_ligne) ncol(M) else nrow(M)
    d <- rep(1 + phi * phi, k); d[1] <- 1; d[k] <- 1
    if (par_ligne) {
      out <- M * matrix(d, nrow = nrow(M), ncol = k, byrow = TRUE)
      if (k > 1) {
        out[, 1:(k - 1)] <- out[, 1:(k - 1)] - phi * M[, 2:k, drop = FALSE]
        out[, 2:k] <- out[, 2:k] - phi * M[, 1:(k - 1), drop = FALSE]
      }
    } else {
      out <- M * matrix(d, nrow = k, ncol = ncol(M))
      if (k > 1) {
        out[1:(k - 1), ] <- out[1:(k - 1), ] - phi * M[2:k, , drop = FALSE]
        out[2:k, ] <- out[2:k, ] - phi * M[1:(k - 1), , drop = FALSE]
      }
    }
    out / s
  }
  sum(U * tri(tri(U, phi_c, sc, TRUE), phi_r, sr, FALSE))
}

# -----------------------------------------------------------------------------
# L'objectif, et l'ajustement
# -----------------------------------------------------------------------------
#' Ajustement REML par moteur creux.
#'
#' beta ET u sont declares aleatoires : l'approximation de Laplace de TMB est
#' EXACTE pour un modele lineaire gaussien, et integrer beta en plus de u donne
#' la vraisemblance REML plutot que le maximum de vraisemblance.
#'
#' @param terms liste de termes, format du moteur dense, avec zi/zj/zx, t, q,
#'   struct, lvl, dims, et pour une parente creuse `Kinv` (dgCMatrix).
#' @param theta_init depart a chaud. Avec maxiter = 0 l'objectif est simplement
#'   EVALUE au theta fourni : c'est ainsi qu'on compare les deux moteurs sur la
#'   meme fonction plutot que sur leurs points d'arret respectifs.
rx_fit_sparse <- function(terms, residual, y, X, theta_init = NULL,
                          maxiter = 200L, verbose = TRUE) {
  stopifnot(rx_tmb_available())
  sc <- rx_sparse_scope(terms, residual)
  if (!sc$ok) stop("hors perimetre du moteur creux : ", sc$reason)
  n <- length(y)
  X <- as.matrix(X)
  p <- ncol(X)

  # --- decoupage de theta, DANS L'ORDRE DU MOTEUR DENSE
  np <- integer(0)
  for (tm in terms) {
    lv <- tm$lvl %||% (if (!is.null(tm$Kinv)) "prec" else "id")
    np <- c(np, rx_n_sigma_params(tm$struct %||% "iid", tm$t) + rx_n_level_params(lv))
  }
  n_res <- rx_n_sigma_params(residual$struct %||% "iid",
                             if (identical(residual$struct, "diag")) residual$t else 1L)
  n_theta <- sum(np) + n_res

  Zs <- lapply(terms, function(tm)
    Matrix::sparseMatrix(i = tm$zi + 1L, j = tm$zj + 1L, x = tm$zx,
                         dims = c(n, tm$t * tm$q)))
  trait_res <- if (is.null(residual$trait)) rep(0L, n) else as.integer(residual$trait)

  dat <- list(y = as.numeric(y), X = X, Zs = Zs, terms = terms,
              residual = residual, trait_res = trait_res, np = np, n_res = n_res)

  nll <- function(par) {
    theta <- par$theta; beta <- par$beta; u <- par$u
    mu <- as.vector(dat$X %*% beta)
    o_u <- 0L
    q_tot <- 0
    pen <- 0
    off <- 0L
    for (k in seq_along(dat$terms)) {
      tm <- dat$terms[[k]]
      t_ <- as.integer(tm$t); q_ <- as.integer(tm$q)
      lv <- tm$lvl %||% (if (!is.null(tm$Kinv)) "prec" else "id")
      ns <- rx_n_sigma_params(tm$struct %||% "iid", t_)
      th_s <- theta[(off + 1L):(off + ns)]
      th_l <- if (rx_n_level_params(lv) > 0) theta[(off + ns + 1L):(off + np[k])] else numeric(0)
      off <- off + np[k]

      uk <- u[(o_u + 1L):(o_u + t_ * q_)]
      mu <- mu + as.vector(dat$Zs[[k]] %*% uk)
      o_u <- o_u + t_ * q_

      # Sigma^-1 par le facteur L, sans inversion generale
      L <- rx_chol_sigma_R(th_s, tm$struct %||% "iid", t_)
      logdet_sig <- 2 * sum(log(diag(L)))
      U <- matrix(uk, nrow = q_, ncol = t_)          # niveau varie le plus vite
      W <- t(solve(L, t(U)))                          # U Sigma^{-1/2}, par colonnes

      # forme quadratique entre niveaux, appliquee a chaque colonne de W
      if (lv %in% c("id", "iid")) {
        qf <- sum(W * W); logdet_K <- 0
      } else if (lv == "ar1") {
        phi <- tanh(th_l[1])
        qf <- 0
        for (j in seq_len(t_)) qf <- qf + rx_quad_ar1(W[, j], phi)
        logdet_K <- (q_ - 1) * log(1 - phi * phi)
      } else if (lv == "ar1ar1") {
        pr <- tanh(th_l[1]); pc <- tanh(th_l[2])
        nr <- tm$dims[1]; nc <- tm$dims[2]
        qf <- 0
        for (j in seq_len(t_)) qf <- qf + rx_quad_ar1ar1(W[, j], pr, pc, nr, nc)
        logdet_K <- nc * (nr - 1) * log(1 - pr * pr) + nr * (nc - 1) * log(1 - pc * pc)
      } else {                                        # prec : K^-1 constante
        Ki <- tm$Kinv
        qf <- 0
        for (j in seq_len(t_)) qf <- qf + sum(W[, j] * as.vector(Ki %*% W[, j]))
        logdet_K <- tm$Kinv_logdet %||% 0
      }
      # log|G| = q log|Sigma| + t log|K|
      pen <- pen + 0.5 * (q_ * logdet_sig + t_ * logdet_K) + 0.5 * qf
      q_tot <- q_tot + t_ * q_
    }
    # --- residuelle
    th_r <- theta[(off + 1L):(off + dat$n_res)]
    if (identical(dat$residual$struct, "diag")) {
      s2 <- exp(2 * th_r)[dat$trait_res + 1L]
    } else {
      s2 <- rep(exp(2 * th_r[1]), n)
    }
    r <- dat$y - mu
    nll_data <- 0.5 * sum(log(s2)) + 0.5 * sum(r * r / s2)
    # Constantes de 2 pi incluses pour que la valeur rendue soit -logL et non
    # -logL a une constante pres : c'est ce qui permet de la comparer au moteur
    # dense sans facteur d'ajustement cache.
    nll_data + pen + 0.5 * (n + q_tot) * log(2 * pi)
  }

  th0 <- if (!is.null(theta_init)) as.numeric(theta_init) else {
    v <- log(sqrt(max(stats::var(y), 1e-8) / 2))
    rep(v, n_theta)
  }
  par <- list(theta = th0, beta = rep(0, p),
              u = rep(0, sum(vapply(terms, function(tm) tm$t * tm$q, 1L))))
  obj <- RTMB::MakeADFun(nll, par, random = c("beta", "u"), silent = !verbose)

  if (maxiter <= 0L) {
    v <- obj$fn(th0)
    return(list(theta = th0, logLik = -2 * as.numeric(v), n_par = n_theta,
                n_iter = 0L, engine = "sparse", converged = NA,
                message = "evaluation seule (maxiter = 0)"))
  }
  fit <- stats::nlminb(obj$par, obj$fn, obj$gr,
                       control = list(iter.max = maxiter, eval.max = 4L * maxiter,
                                      trace = if (verbose) 1L else 0L))
  sdr <- try(TMB::sdreport(obj), silent = TRUE)
  list(theta = as.numeric(fit$par), logLik = -2 * fit$objective, n_par = n_theta,
       n_iter = fit$iterations, engine = "sparse",
       converged = identical(fit$convergence, 0L), message = fit$message,
       gradient = as.numeric(obj$gr(fit$par)),
       sdreport = if (inherits(sdr, "try-error")) NULL else sdr)
}
