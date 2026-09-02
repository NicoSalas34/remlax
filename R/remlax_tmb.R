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

# -----------------------------------------------------------------------------
# LES SURCHARGES DE RTMB SONT OBLIGATOIRES A L'INTERIEUR DE L'OBJECTIF
# -----------------------------------------------------------------------------
# RTMB exporte ses propres `matrix`, `diag` et `solve`. Elles ne sont visibles
# que si le paquet est ATTACHE (library(RTMB)) ou nommee explicitement. Ce
# fichier n'appelle que RTMB::MakeADFun, donc a l'interieur de l'objectif ces
# trois fonctions resolvaient vers les versions de base — lesquelles rendent un
# objet numerique ordinaire et PERDENT l'attribut de classe differentiable. Le
# symptome etait :
#
#     Invalid argument to 'advector' (lost class attribute?)
#
# leve par t() sur un objet deja degrade par matrix(). Diagnostic contre-intuitif :
# une sonde a montre que les cinq facons de CONSTRUIRE une matrice triangulaire
# differentiable marchent toutes, y compris matrix(0) puis assignation — a
# condition que `matrix` soit celle de RTMB. Le probleme n'etait donc pas
# l'ecriture mais la RESOLUTION DE NOM.
#
# On lie donc les surcharges localement, dans chaque fonction qui en a besoin,
# plutot que d'attacher RTMB globalement : un fichier de paquet ne doit pas
# masquer base::matrix chez son utilisateur. Sur des entrees numeriques ces
# surcharges se comportent comme celles de base, donc les memes fonctions
# restent utilisables hors differentiation — c'est ce que verifie le test de
# concordance.
#
# PIEGE MESURE AU PASSAGE : backsolve() ne leve AUCUNE erreur sur un advector,
# rend NaN et un gradient nul, avec pour seul indice un avertissement sur des
# parties imaginaires ecartees. Ne pas l'utiliser ici.
.rx_ad <- function(env = parent.frame()) {
  if (!requireNamespace("RTMB", quietly = TRUE)) return(invisible(FALSE))
  assign("matrix", RTMB::matrix, envir = env)
  assign("diag",   RTMB::diag,   envir = env)
  assign("solve",  RTMB::solve,  envir = env)
  invisible(TRUE)
}

rx_tmb_available <- function() {
  requireNamespace("RTMB", quietly = TRUE) && requireNamespace("Matrix", quietly = TRUE)
}

# -----------------------------------------------------------------------------
# Perimetre
# -----------------------------------------------------------------------------
#' Longueur du vecteur theta d'un modele
#'
#' Utile des qu'on veut IMPOSER un theta : un banc apparie doit evaluer les deux
#' moteurs au MEME point, sinon chacun part de son theta initial et les
#' vraisemblances rapportees ne sont pas comparables. Mesure du piege : les
#' valeurs divergeaient de 1e-5 a 9e-3 en croissant avec le nombre de
#' caracteres, ce qui ressemblait a un defaut de formulation, alors qu'a theta
#' commun les deux moteurs s'accordent a 1e-12.
#'
#' Le decoupage est celui du moteur dense, et les deux moteurs comptent a
#' l'identique — verifie par la suite de concordance.
#'
#' @param model objet rx_model
#' @return entier
#' @export
rx_n_theta <- function(model) {
  terms <- model$terms
  residual <- model$residual
  np <- integer(0)
  for (tm in terms) {
    np <- c(np, rx_n_sigma_params(tm$struct %||% "iid", tm$t) +
                rx_n_level_params(rx_level_of(tm)))
  }
  t_res <- as.integer(model$t_res %||% 1L)
  n_res <- rx_n_sigma_params(residual$struct %||% "iid",
                             if (identical(residual$struct, "diag")) t_res else 1L)
  sum(np) + n_res
}

RX_SPARSE_SIGMA <- c("iid", "diag", "us")
RX_SPARSE_LEVEL <- c("id", "iid", "ar1", "ar1ar1", "prec")   # "prec" = A^-1 fournie

#' Le modele est-il dans le perimetre du moteur creux ?
#'
#' Rend une liste (ok, reason). La raison est destinee a l'utilisateur : elle
#' nomme la structure qui sort du perimetre plutot que de dire « non ».
#' Structure de niveaux effective d'un terme rx_term.
#'
#' `level = "auto"` est la valeur par defaut de rx_term : elle signifie « pas de
#' structure entre niveaux », sauf si une parente est fournie. Une parente en
#' `LK` est un facteur de Cholesky DENSE, donc hors perimetre creux ; une parente
#' en `Kinv` est une precision creuse, donc dedans. Cette resolution est faite
#' ici et nulle part ailleurs, pour que le perimetre et l'objectif ne puissent
#' pas en avoir deux lectures differentes.
rx_level_of <- function(tm) {
  lv <- tm$level %||% "auto"
  if (!is.null(tm$Kinv)) return("prec")
  if (!is.null(tm$LK)) return("fixed")
  if (lv %in% c("auto", "id", "iid")) return("id")
  lv
}

rx_sparse_scope <- function(terms, residual = NULL) {
  for (tm in terms) {
    st <- tm$struct %||% "iid"
    if (!st %in% RX_SPARSE_SIGMA)
      return(list(ok = FALSE, reason = sprintf(
        "terme '%s' : structure entre caracteres '%s' hors perimetre creux (retenues : %s)",
        tm$name %||% "?", st, paste(RX_SPARSE_SIGMA, collapse = ", "))))
    lv <- rx_level_of(tm)
    if (!lv %in% RX_SPARSE_LEVEL)
      return(list(ok = FALSE, reason = sprintf(
        "terme '%s' : structure entre niveaux '%s' a un inverse plein, le creux n'y gagne rien%s",
        tm$name %||% "?", lv,
        if (identical(lv, "fixed"))
          " (parente fournie en LK, facteur de Cholesky dense : fournir Kinv creuse pour le creux)" else "")))
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
  .rx_ad()                     # matrix/diag/solve : versions de RTMB
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
  # `fixed` = une parente FOURNIE, par son facteur de Cholesky (champ LK). Elle
  # ne porte aucun parametre a estimer, exactement comme `prec` qui la fournit
  # par son inverse. L'omission faisait echouer rx_n_theta() sur tout terme
  # portant une K, alors que le moteur creux n'y touchait pas : il recoit Kinv,
  # donc `prec`, qui etait present. Le defaut n'apparaissait qu'en imposant
  # theta sur un modele a parente dense.
  switch(kind, id = 0L, iid = 0L, ar1 = 1L, ar1ar1 = 2L, prec = 0L, fixed = 0L,
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
  .rx_ad()
  # u est indexe niveau = (ligne - 1) * nc + colonne. Un remplissage
  # COLONNE PAR COLONNE dans une matrice nc x nr donne donc M[colonne, ligne],
  # et U = t(M). On evite ainsi byrow=, dont le support par la surcharge de
  # RTMB n'est pas garanti, au profit de t() qui dispatche sur la classe.
  U <- t(matrix(u, nrow = nc, ncol = nr))
  sr <- 1 - phi_r * phi_r
  sc <- 1 - phi_c * phi_c
  # P = Pr (x) Pc, donc u'Pu = tr(U' Pr U Pc). On forme Pr U par colonnes et
  # Pc par lignes, en n'ecrivant que des operations tridiagonales.
  tri <- function(M, phi, s, par_ligne) {
    k <- if (par_ligne) ncol(M) else nrow(M)
    d <- rep(1 + phi * phi, k); d[1] <- 1; d[k] <- 1
    if (par_ligne) {
      # d doit multiplier les COLONNES de M. rep() sur la longueur voulue donne
      # le meme resultat qu'un byrow= sans dependre de son support.
      out <- M * matrix(rep(d, each = nrow(M)), nrow = nrow(M), ncol = k)
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
#' @param model objet rendu par rx_model ou rx_reml — LE MEME que celui que
#'   prend rx_fit. Les deux moteurs sont ainsi interchangeables sur une meme
#'   specification, ce qui est la condition pour pouvoir les comparer.
#' @param theta_init depart a chaud. Avec maxiter = 0 l'objectif est simplement
#'   EVALUE au theta fourni : c'est ainsi qu'on compare les deux moteurs sur la
#'   meme fonction plutot que sur leurs points d'arret respectifs.
#' @param maxiter plafond d'iterations. IL DOIT EGALER CELUI DE rx_fit (3000).
#'   Un plafond plus bas d'un cote fait passer une TRONCATURE pour une
#'   convergence : mesure sur la grille, les cellules a p = 45 s'arretaient a
#'   exactement 200 iterations, et leurs temps — 12,4 s et 42,8 s — n'etaient pas
#'   des temps d'ajustement mais des temps de plafond. Comparer les deux moteurs
#'   avec des plafonds differents biaise la comparaison en faveur du plus bas.
#' @param sdreport calculer le rapport d'ecarts-types de TMB. Il forme la
#'   covariance de TOUS les effets aleatoires, ce qui est cher des que q_total
#'   grandit, et il est facultatif : un utilisateur qui ne veut que les
#'   composantes de variance n'en a pas besoin. Le mettre a FALSE est aussi la
#'   condition d'une comparaison EQUITABLE avec rx_fit(hessian = FALSE,
#'   blups = FALSE), qui ne calcule aucun equivalent — sinon le moteur creux
#'   fait un travail que le dense ne fait pas, et l'ecart de temps mesure
#'   sous-estime son avantage.
rx_fit_sparse <- function(model, theta_init = NULL, maxiter = 3000L, verbose = TRUE,
                          sdreport = TRUE) {
  stopifnot(rx_tmb_available())
  terms <- model$terms; residual <- model$residual
  y <- model$y; X <- as.matrix(model$X)
  sc <- rx_sparse_scope(terms, residual)
  if (!sc$ok) stop("hors perimetre du moteur creux : ", sc$reason)
  n <- length(y)
  p <- ncol(X)

  # --- decoupage de theta, DANS L'ORDRE DU MOTEUR DENSE
  np <- integer(0)
  for (tm in terms) {
    np <- c(np, rx_n_sigma_params(tm$struct %||% "iid", tm$t) +
                rx_n_level_params(rx_level_of(tm)))
  }
  # rx_residual ne porte PAS de champ $t : le nombre de caracteres residuels est
  # calcule par rx_model, sous $t_res. Lire residual$t rendait NULL, et
  # rx_n_sigma_params aurait recu un t vide.
  t_res <- as.integer(model$t_res %||% 1L)
  n_res <- rx_n_sigma_params(residual$struct %||% "iid",
                             if (identical(residual$struct, "diag")) t_res else 1L)
  n_theta <- sum(np) + n_res

  # rx_term porte Zl : une liste de t incidences creuses n x q, une par
  # caractere. Les concatener par colonnes donne l'ordre colonne =
  # caractere * q + niveau, exactement celui du cote Python (zj = trait*q+lev).
  # C'est cet ordre qui rend G = Sigma (x) K avec le caractere en facteur
  # EXTERIEUR, et il est load-bearing : l'inverser transposerait Sigma.
  Zs <- lapply(terms, function(tm) {
    Zl <- tm$Zl
    if (length(Zl) != tm$t)
      stop("terme '", tm$name, "' : ", length(Zl), " incidence(s) pour t = ", tm$t)
    methods::as(do.call(cbind, Zl), "dgCMatrix")
  })
  # ATTENTION AU DECALAGE. as.integer(factor(x)) rend 1..t cote R, alors que le
  # cote Python numerote les caracteres 0..t-1. On stocke en base 0, comme
  # Python, et l'indexation ajoute 1 la ou R l'exige — une seule convention,
  # explicite. La version precedente stockait 1..t PUIS ajoutait 1 : la
  # residuelle du premier caractere n'etait jamais utilisee et celle du dernier
  # sortait du vecteur.
  # log|K| D'UNE PRECISION FOURNIE EST UNE CONSTANTE : Kinv ne depend d'aucun
  # parametre, donc son determinant sort de l'optimisation. On le calcule ICI,
  # une fois, hors du ruban de derivation — et surtout on le calcule, ce que
  # l'objectif ne faisait pas : il lisait tm$Kinv_logdet et retombait sur ZERO
  # quand l'utilisateur ne l'avait pas fourni. La vraisemblance etait alors
  # decalee de 0,5 * log|K|, soit 21,17 sur un cas a 60 niveaux — un decalage
  # CONSTANT, donc invisible sur les estimations de theta, qui etaient identiques
  # au cinquieme chiffre pres, mais faux sur toute comparaison de modeles.
  ldK_fixe <- vector("list", length(terms))
  for (k in seq_along(terms)) {
    tm <- terms[[k]]
    if (identical(rx_level_of(tm), "prec")) {
      ldK_fixe[[k]] <- if (!is.null(tm$Kinv_logdet)) as.numeric(tm$Kinv_logdet) else
        -as.numeric(Matrix::determinant(tm$Kinv, logarithm = TRUE)$modulus)
      if (!is.finite(ldK_fixe[[k]]))
        stop("terme '", tm$name, "' : log|K| non fini. Kinv est-elle definie positive ?")
    }
  }

  trait_res <- if (is.null(residual$trait)) rep(0L, n)
               else as.integer(factor(residual$trait)) - 1L
  stopifnot(min(trait_res) == 0L, max(trait_res) + 1L <= t_res)

  dat <- list(y = as.numeric(y), X = X, Zs = Zs, terms = terms,
              residual = residual, trait_res = trait_res, np = np, n_res = n_res,
              ldK_fixe = ldK_fixe)

  nll <- function(par) {
    .rx_ad()                   # sans ceci, matrix() degrade u en numerique
    theta <- par$theta; beta <- par$beta; u <- par$u
    mu <- as.vector(dat$X %*% beta)
    o_u <- 0L
    q_tot <- 0
    pen <- 0
    off <- 0L
    for (k in seq_along(dat$terms)) {
      tm <- dat$terms[[k]]
      t_ <- as.integer(tm$t); q_ <- as.integer(tm$q)
      lv <- rx_level_of(tm)
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
        logdet_K <- dat$ldK_fixe[[k]]
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
    # CONVENTION, ET COMMENT ELLE A ETE ETABLIE. obj$fn rend -logL apres
    # integration de (beta, u) : le moteur rend donc logLik = -obj$fn, la
    # log-vraisemblance elle-meme, comme le fait le moteur dense
    # (fit.py : logLik = -0.5 * neg2_reml). La premiere version rendait
    # -2*obj$fn, et le test de depart a chaud a montre un rapport de EXACTEMENT
    # 2,0000000000 sur trois modeles differents — signature d'une convention et
    # non d'une formulation, puisqu'une divergence de formulation ne produirait
    # pas un facteur constant. C'est exactement ce que ce test doit separer.
    #
    # Les constantes de 2 pi sont incluses ici ; TMB en retire sa part lors de
    # l'integration de Laplace, si bien que le net vaut -(n-p)/2 log(2 pi), la
    # constante REML. Tout decalage residuel apres correction du facteur 2 est
    # donc a chercher dans cette constante, et le test l'affiche pour permettre
    # de l'attribuer plutot que de la supposer.
    nll_data + pen + 0.5 * (n + q_tot) * log(2 * pi)
  }

  th0 <- if (!is.null(theta_init)) as.numeric(theta_init) else {
    v <- log(sqrt(max(stats::var(y), 1e-8) / 2))
    rep(v, n_theta)
  }
  q_total <- sum(vapply(terms, function(tm) as.integer(tm$t) * as.integer(tm$q), 1L))
  par <- list(theta = th0, beta = rep(0, p), u = rep(0, q_total))
  # TOLERANCE DU PROBLEME INTERNE. L'approximation de Laplace est EXACTE pour un
  # modele lineaire gaussien, mais seulement si le mode interne en (beta, u) est
  # trouve exactement. TMB le cherche par Newton, avec une tolerance par defaut
  # calibree pour des modeles non gaussiens ou l'approximation elle-meme domine
  # l'erreur. Ici elle ne domine pas, et l'arret precoce se lisait directement :
  # sur le champ ar1 a q = n, evaluer au theta du dense laissait 1,9e-4 d'ecart
  # de log-vraisemblance, alors que le sens inverse — ou le moteur dense n'a
  # aucun probleme interne — n'en laissait que 1,9e-8. Cette ASYMETRIE est la
  # signature d'une convergence interne insuffisante, pas d'une formulation
  # differente. Le probleme interne etant quadratique, le resserrer ne coute
  # qu'une iteration ou deux.
  # CONSTRUCTION DU RUBAN, CHRONOMETREE A PART. Elle est payee UNE fois par
  # appel, et un ajustement l'amortit sur toutes ses iterations. La confondre
  # avec le calcul rendait le cout unitaire du moteur creux inutilisable :
  # mesure du piege, le controle « ajustement >= iterations x cout unitaire »
  # echouait sur les HUIT cellules creuses d'un balayage et sur aucune dense —
  # 38 iterations a 0,757 s auraient demande 28,8 s quand l'ajustement entier
  # prenait 11,25 s. Le rapport separe donc les deux postes, comme le fait le
  # moteur dense depuis qu'un bilan de reconstruction a echoue pour la meme
  # raison.
  ..t0 <- proc.time()[["elapsed"]]
  obj <- RTMB::MakeADFun(nll, par, random = c("beta", "u"), silent = !verbose,
                         inner.control = list(maxit = 200L, tol = 1e-14,
                                              tol10 = 0, smartsearch = FALSE))

  ..construct_s <- proc.time()[["elapsed"]] - ..t0

  if (maxiter <= 0L) {
    # PIEGE DE TMB, MESURE. Appeler obj$fn DEUX FOIS AU MEME theta ne mesure
    # presque rien : le second appel reutilise le mode interne en (beta, u) deja
    # converge par le premier, donc il saute le probleme de Laplace qui est le
    # gros du calcul. Chronometre ainsi, le cout unitaire ressortait a 0,00000 s
    # a quatre tailles — une valeur qui aurait fait passer le moteur creux pour
    # gratuit. La mesure se fait donc a un theta que le ruban n'a pas vu ; le
    # cout ne depend que des dimensions, pas des valeurs, ce qui est la premisse
    # de tout le protocole. La vraisemblance rendue reste celle de th0, seule
    # comparable entre moteurs.
    v <- obj$fn(th0)
    # RESOLUTION. proc.time() quantifie a ~10 ms sur ce systeme, or une
    # evaluation creuse est sous-milliseconde aux petites tailles : mesuree une
    # a une elle ressortait a 0,000 ou 0,001 s, c'est-a-dire au grain de
    # l'horloge et non au cout. On chronometre donc PLUSIEURS evaluations, avec
    # Sys.time() qui descend sous la microseconde, chacune a un theta distinct
    # pour qu'aucune ne reutilise le mode interne de la precedente. Ce sont les
    # petites tailles qui portent le point de croisement entre moteurs : leur
    # resolution decide de la conclusion.
    ..k <- 5L
    ..t1 <- Sys.time()
    for (..i in seq_len(..k)) invisible(obj$fn(th0 + ..i * 1e-3))
    ..eval_s <- as.numeric(difftime(Sys.time(), ..t1, units = "secs")) / ..k
    return(list(theta = th0, logLik = -as.numeric(v), n_par = n_theta,
                n_iter = 0L, engine = "sparse", converged = NA,
                construct_s = ..construct_s, eval_s = ..eval_s,
                secondes = ..construct_s + ..eval_s,
                message = "evaluation seule (maxiter = 0)"))
  }
  fit <- stats::nlminb(obj$par, obj$fn, obj$gr,
                       control = list(iter.max = maxiter, eval.max = 4L * maxiter,
                                      trace = if (verbose) 1L else 0L))
  sdr <- if (isTRUE(sdreport)) try(TMB::sdreport(obj), silent = TRUE) else NULL
  list(theta = as.numeric(fit$par), logLik = -as.numeric(fit$objective), n_par = n_theta,
       n_iter = fit$iterations, engine = "sparse",
       construct_s = ..construct_s,
       n_eval = as.integer(fit$evaluations[["function"]] %||% NA_integer_),
       secondes = proc.time()[["elapsed"]] - ..t0,
       converged = identical(fit$convergence, 0L), message = fit$message,
       gradient = as.numeric(obj$gr(fit$par)),
       sdreport = if (inherits(sdr, "try-error")) NULL else sdr)
}
