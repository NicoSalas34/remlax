# =============================================================================
# Les deux moteurs calculent-ils la MEME fonction ?
# =============================================================================
# LE TEST QUI COMPTE, ET POURQUOI IL A FALLU LE DEPART A CHAUD.
# Comparer deux solveurs par leurs resultats d'ajustement ne distingue pas deux
# causes tres differentes : soit ils calculent des vraisemblances differentes
# (erreur de formulation), soit ils calculent la meme et s'arretent a des
# endroits differents (difference d'optimisation). Les composantes de variance
# diffèrent dans les deux cas.
#
# Le depart a chaud tranche. On impose a un moteur le theta trouve par l'autre,
# avec maxiter = 0 : il n'ajuste rien, il EVALUE. Si les deux -2logL coincident,
# les deux calculent la meme fonction et tout ecart restant est de
# l'optimisation. S'ils diffèrent, c'est la formulation, et l'ecart se lit
# directement en unites de log-vraisemblance.
#
# C'est aussi la seule façon de valider le moteur creux SANS reference externe :
# le moteur dense est deja valide contre asreml, lme4, sommer et pbkrtest
# (154 verifications), donc il fait ici office d'etalon.
#
# CE QUI EST COMPARE, ET DANS QUEL SENS
#   A. dense ajuste -> theta_dense -> creux EVALUE a theta_dense
#   B. creux ajuste -> theta_creux -> dense EVALUE a theta_creux
#   C. les deux ajustent librement : composantes, et laquelle atteint le
#      meilleur optimum (le test A/B ayant deja etabli que la surface est la meme)
# Le sens B importe autant que A : si seul A passait, un moteur pourrait avoir
# une surface correcte sur le chemin de l'autre et fausse ailleurs.
suppressPackageStartupMessages({ library(Matrix) })
ICI <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) "tests/R")
RACINE <- normalizePath(file.path(ICI, "..", ".."), mustWork = FALSE)
source(file.path(RACINE, "R", "remlax.R"))
source(file.path(RACINE, "R", "remlax_tmb.R"))

TOL_LL <- 1e-6          # unites de -2logL : bien en dessous de tout ecart interpretable
ok <- 0L; ko <- 0L
verif <- function(nom, ecart, tol = TOL_LL) {
  bon <- is.finite(ecart) && abs(ecart) <= tol
  cat(sprintf("  [%s] %-56s ecart %.3e\n", if (bon) "OK" else "ECHEC", nom, ecart))
  if (bon) ok <<- ok + 1L else ko <<- ko + 1L
}

set.seed(20260901)

# --- dispositifs, tous dans le perimetre creux -------------------------------
# rx_term prend (name, Z, ...) ou Z est un facteur, une matrice, ou une liste de
# t matrices. On passe donc des facteurs : rx_term construit l'incidence.
inc <- function(f) { f <- factor(f); Matrix::sparseMatrix(
  i = seq_along(f), j = as.integer(f), x = 1, dims = c(length(f), nlevels(f))) }

faire_iid <- function(n = 400L, q = 60L, sg = 1.2, se = 0.9) {
  lev <- factor(sample.int(q, n, replace = TRUE), levels = seq_len(q))
  u <- rnorm(q, 0, sg)
  list(y = 5 + u[as.integer(lev)] + rnorm(n, 0, se), X = matrix(1, n, 1),
       terms = list(rx_term("g", lev, struct = "iid")),
       residual = rx_residual(), verite = c(sg = sg, se = se))
}
#' Champ ar1 entre niveaux, AVEC REPETITION.
#'
#' `rep_par_niveau` compte les observations par niveau et il est load-bearing.
#' A une seule observation par niveau, la variance residuelle n'est pas
#' identifiee : le champ absorbe tout, l'optimiseur envoie sigma_e a sa borne
#' inferieure, et la vraisemblance devient plate dans cette direction — deux
#' theta distants de 4 unites y donnent la meme logLik a 1,5e-7 pres. Ce n'est
#' pas un dispositif sur lequel comparer deux solveurs, et c'est ce qu'une
#' premiere version de ce test faisait : elle mesurait un artefact de
#' conditionnement (cf. le cas degenere ci-dessous) et l'imputait au moteur
#' creux. Deux repetitions suffisent a identifier la residuelle.
faire_ar1 <- function(nr = 20L, nc = 15L, phi = 0.6, sg = 1.0, se = 0.7,
                      rep_par_niveau = 2L) {
  q <- nr * nc
  z <- numeric(q); z[1] <- rnorm(1)
  for (i in 2:q) z[i] <- phi * z[i - 1] + rnorm(1, 0, sqrt(1 - phi^2))
  lev <- factor(rep(seq_len(q), rep_par_niveau), levels = seq_len(q))
  n <- length(lev)
  list(y = 3 + sg * z[as.integer(lev)] + rnorm(n, 0, se), X = matrix(1, n, 1),
       terms = list(rx_term("f", lev, struct = "iid", level = "ar1")),
       residual = rx_residual(), verite = c(phi = phi, sg = sg, se = se))
}

#' Le MEME champ sans repetition : cas degenere, garde deliberement.
#'
#' Il documente une limite STRUCTURELLE de la formulation creuse, pas un defaut
#' d'implementation. Les equations du modele mixte contiennent R^-1, donc quand
#' sigma_e^2 tend vers zero — ici 4e-11, l'optimiseur l'ayant pousse a sa borne —
#' R^-1 depasse 1e10 et C = Z'R^-1 Z + G^-1 devient tres mal conditionnee. Le
#' moteur creux perd alors ~2e-4 sur la log-vraisemblance, la ou le moteur dense,
#' qui travaille sur V = sigma^2 K + sigma_e^2 I, reste exact a 6e-13. Un calcul
#' REML independant en algebre dense arbitre en faveur du moteur dense sur ce
#' point precis. C'est la raison pour laquelle asreml travaille sur le RAPPORT
#' des variances plutot que sur sigma_e^2 lui-meme.
faire_ar1_degenere <- function(...) faire_ar1(..., rep_par_niveau = 1L)
faire_bivarie <- function(n = 300L, q = 50L) {
  lev <- sample.int(q, n, replace = TRUE)
  L <- matrix(c(1.1, 0.5, 0, 0.8), 2, 2)
  U <- matrix(rnorm(q * 2), q, 2) %*% t(L)
  trait <- rep(1:2, each = n)
  Zt <- inc(factor(rep(lev, 2), levels = seq_len(q)))
  # une incidence PAR CARACTERE : nulle hors du caractere concerne
  Z1 <- Zt; Z1[trait != 1L, ] <- 0
  Z2 <- Zt; Z2[trait != 2L, ] <- 0
  y <- c(2 + U[lev, 1], 4 + U[lev, 2]) + rnorm(2 * n, 0, 0.6)
  X <- model.matrix(~ factor(trait))
  list(y = y, X = X,
       terms = list(rx_term("g", list(Z1, Z2), struct = "us")),
       residual = rx_residual(trait = factor(trait), struct = "diag"),
       verite = c(s11 = L[1, 1]^2, s22 = sum(L[2, ]^2)))
}

CAS <- list(
  list(nom = "un facteur iid", f = faire_iid),
  list(nom = "champ ar1 entre niveaux (2 repetitions)", f = faire_ar1),
  list(nom = "champ ar1 sans repetition", f = faire_ar1_degenere, bord = TRUE),
  list(nom = "bivarie us, residuelle diag", f = faire_bivarie)
)

for (cs in CAS) {
  cat(sprintf("\n=== %s ===\n", cs$nom))
  if (!is.null(cs$note)) cat("  ", cs$note, "\n", sep = "")
  d <- cs$f()
  m <- rx_model(d$y, d$X, terms = d$terms, residual = d$residual, name = cs$nom)

  # --- moteur dense : l'etalon, deja valide contre asreml, lme4, sommer et
  # pbkrtest (154 verifications). C'est lui qui fait reference ici.
  dn <- try(rx_fit(m, hessian = FALSE, blups = FALSE, verbose = FALSE), silent = TRUE)
  if (inherits(dn, "try-error")) {
    cat("  moteur dense en echec :", conditionMessage(attr(dn, "condition")), "\n"); next
  }

  # Le logLik RENDU par l'ajustement est-il celui du theta rendu ? Les deux
  # viennent du meme appel, donc ils devraient concorder — mais c'est une
  # hypothese, et un ecart ici se confondrait avec un desaccord entre moteurs.
  dnA <- rx_fit(m, theta_init = dn$theta, maxiter = 0L, polish = 0L,
                hessian = FALSE, blups = FALSE, verbose = FALSE)
  verif("0 : dense reevalue a son propre theta", dnA$logLik - dn$logLik)

  # --- A. le creux EVALUE au theta du dense
  spA <- try(rx_fit_sparse(m, theta_init = dn$theta, maxiter = 0L, verbose = FALSE),
             silent = TRUE)
  if (inherits(spA, "try-error")) {
    cat("  moteur creux en echec :", conditionMessage(attr(spA, "condition")), "\n")
    ko <- ko + 1L; next
  }
  verif("A : creux evalue au theta du dense", spA$logLik - dn$logLik,
        tol = cs$tol %||% TOL_LL)
  # De quoi ATTRIBUER un eventuel decalage residuel plutot que de le supposer :
  # la constante REML vaut -(n-p)/2 log(2 pi), et le moteur dense expose la
  # sienne sous const_2pi.
  n_ <- length(d$y); p_ <- ncol(d$X)
  cat(sprintf("      n = %d, p = %d | constante REML (n-p)/2 log(2pi) = %.6f | const_2pi du dense = %s\n",
              n_, p_, 0.5 * (n_ - p_) * log(2 * pi),
              if (is.null(dn$const_2pi)) "absente" else sprintf("%.6f", dn$const_2pi)))

  # --- le creux ajuste librement
  sp <- rx_fit_sparse(m, maxiter = 300L, verbose = FALSE)

  # --- B. le dense EVALUE au theta du creux
  dnB <- rx_fit(m, theta_init = sp$theta, maxiter = 0L, polish = 0L,
                hessian = FALSE, blups = FALSE, verbose = FALSE)
  verif("B : dense evalue au theta du creux", dnB$logLik - sp$logLik,
        tol = cs$tol %||% TOL_LL)

  # --- C. optima atteints. Le sens du signe est informatif : on ne le cache pas
  # derriere une valeur absolue.
  cat(sprintf("  C : -2logL dense %.8f | creux %.8f | ecart %+.3e\n",
              dn$logLik, sp$logLik, sp$logLik - dn$logLik))
  verif("C : les deux optima a moins de 1e-4", sp$logLik - dn$logLik, tol = 1e-4)
  if (!is.null(d$verite))
    cat("      verite de simulation :",
        paste(sprintf("%s=%.3f", names(d$verite), d$verite), collapse = " "), "\n")
}

# -----------------------------------------------------------------------------
# La limite de conditionnement de la formulation creuse, MESUREE et non supposee
# -----------------------------------------------------------------------------
# Les equations du modele mixte contiennent R^-1. Quand sigma_e^2 tend vers zero,
# R^-1 explose et C = Z'R^-1 Z + G^-1 devient tres mal conditionnee : le moteur
# creux perd de la precision la ou le moteur dense, qui travaille sur
# V = sigma^2 K + sigma_e^2 I, n'a aucun probleme. C'est structurel, et c'est la
# raison pour laquelle asreml parametre le RAPPORT des variances plutot que
# sigma_e^2 lui-meme.
#
# On ne compte pas sur l'optimiseur pour tomber dans ce regime : on l'IMPOSE, en
# evaluant les deux moteurs a un theta dont la composante residuelle est mise a
# la borne inferieure du solveur. La mesure est donc reproductible.
cat("\n=== limite de conditionnement : sigma_e a la borne, theta impose ===\n")
d <- faire_ar1_degenere()
m <- rx_model(d$y, d$X, terms = d$terms, residual = d$residual, name = "bord")
dn <- rx_fit(m, hessian = FALSE, blups = FALSE, verbose = FALSE)
for (bord in c(-4, -6, -8, -10, -12)) {
  th <- dn$theta; th[length(th)] <- bord
  dd <- rx_fit(m, theta_init = th, maxiter = 0L, polish = 0L,
               hessian = FALSE, blups = FALSE, verbose = FALSE)$logLik
  ss <- rx_fit_sparse(m, theta_init = th, maxiter = 0L, verbose = FALSE)$logLik
  cat(sprintf("  theta_residuel = %+5.1f  (sigma_e^2 = %.2e) : dense %.8f | creux %.8f | ecart %+.2e\n",
              bord, exp(2 * bord), dd, ss, ss - dd))
}
cat("  Lecture : l'ecart croit quand sigma_e^2 diminue. C'est la limite a citer\n")
cat("  dans la documentation du moteur creux, pas un defaut a corriger.\n")

cat(sprintf("\n=== %d verifications, %d echec(s) ===\n", ok + ko, ko))
if (ko > 0L) quit(status = 1L)
