# =============================================================================
# Concordance du port R de la parametrisation avec le cote Python
# =============================================================================
# POURQUOI CE TEST EXISTE. Le moteur creux est ecrit en R parce que RTMB
# enregistre du code R. La parametrisation — theta -> Sigma, theta -> K — existe
# donc EN DEUX EXEMPLAIRES, et deux copies divergent silencieusement : rien ne
# leve d'erreur si l'ordre de theta differe d'une case, seuls les resultats
# cessent d'etre comparables. Ce test compare les deux cotes sur des theta tires
# au hasard, ce qui est plus sûr et moins cher qu'une relecture.
#
# CE QU'IL VERIFIE, ET DANS QUEL ORDRE
#   1. Sigma(theta) identique pour iid, diag, us, a plusieurs t
#   2. K^-1(theta) du cote R est bien l'inverse de K(theta) du cote Python
#      (ar1, ar1ar1) — pas « une matrice creuse plausible » mais l'inverse exact
#   3. log|K| analytique du cote R = log-determinant numerique du cote Python
#   4. la forme quadratique ecrite en vecteurs = u' K^-1 u forme explicitement
#
# Le point 2 est le plus important : c'est celui qui casserait sans bruit si la
# forme close de la precision AR(1) etait fausse d'un facteur ou d'un signe.
suppressPackageStartupMessages({ library(Matrix) })
ICI <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) "tests/R")
source(file.path(ICI, "..", "..", "R", "remlax_tmb.R"))

PY <- Sys.getenv("RX_PY", "python3")
SRC <- normalizePath(file.path(ICI, "..", "..", "src"), mustWork = FALSE)
ok_tot <- 0L; ko_tot <- 0L
verif <- function(nom, ecart, tol = 1e-10) {
  bon <- is.finite(ecart) && ecart <= tol
  cat(sprintf("  [%s] %-58s ecart %.3e\n", if (bon) "OK" else "ECHEC", nom, ecart))
  if (bon) ok_tot <<- ok_tot + 1L else ko_tot <<- ko_tot + 1L
}

#' Appelle le cote Python et relit une matrice en texte.
py_mat <- function(code) {
  script <- sprintf("import sys, numpy as np\nsys.path.insert(0, %s)\n%s",
                    shQuote(SRC), code)
  tf <- tempfile(fileext = ".py"); writeLines(script, tf)
  out <- system2(PY, tf, stdout = TRUE, stderr = FALSE)
  unlink(tf)
  as.matrix(read.table(text = paste(out, collapse = "\n")))
}

set.seed(20260901)

# --- 1. Sigma(theta) -------------------------------------------------------
cat("=== 1. Sigma(theta), entre caracteres ===\n")
for (st in c("iid", "diag", "us")) for (t in c(1L, 2L, 3L, 5L)) {
  np <- rx_n_sigma_params(st, t)
  th <- round(rnorm(np, 0, 0.4), 6)
  L_R <- rx_chol_sigma_R(th, st, t)
  S_R <- L_R %*% t(L_R)
  S_py <- py_mat(sprintf(
"from remlax.structures import chol_sigma
th = np.array([%s])
L = np.asarray(chol_sigma(th, '%s', %d))
S = L @ L.T
np.savetxt(sys.stdout, S, fmt='%%.17g')", paste(th, collapse = ", "), st, t))
  verif(sprintf("Sigma : %s, t = %d, %d parametres", st, t, np),
        max(abs(S_R - S_py)))
}

# --- 2. K^-1 du cote R contre l'inverse de K du cote Python -----------------
cat("\n=== 2. K^-1(theta) : forme close R contre inverse numerique Python ===\n")
for (q in c(5L, 12L, 40L)) {
  th <- round(rnorm(1, 0, 0.7), 6)
  pr <- rx_level_prec_R(th, "ar1", q)
  K_py <- py_mat(sprintf(
"from remlax.levels import level_chol
th = np.array([%s])
L = np.asarray(level_chol(th, 'ar1', %d))
np.savetxt(sys.stdout, L @ L.T, fmt='%%.17g')", th, q))
  verif(sprintf("ar1 : q = %d, phi = %.4f : K^-1 K = I", q, tanh(th)),
        max(abs(as.matrix(pr$P %*% K_py) - diag(q))))
  verif(sprintf("ar1 : q = %d : log|K| analytique contre numerique", q),
        abs(pr$logdet - determinant(K_py, logarithm = TRUE)$modulus))
}
for (dims in list(c(3L, 4L), c(5L, 6L))) {
  th <- round(rnorm(2, 0, 0.6), 6); q <- prod(dims)
  pr <- rx_level_prec_R(th, "ar1ar1", q, dims = dims)
  K_py <- py_mat(sprintf(
"from remlax.levels import level_chol
th = np.array([%s, %s])
L = np.asarray(level_chol(th, 'ar1ar1', %d, dims=(%d, %d)))
np.savetxt(sys.stdout, L @ L.T, fmt='%%.17g')", th[1], th[2], q, dims[1], dims[2]))
  verif(sprintf("ar1ar1 : %dx%d, phi = (%.3f, %.3f) : K^-1 K = I",
                dims[1], dims[2], tanh(th[1]), tanh(th[2])),
        max(abs(as.matrix(pr$P %*% K_py) - diag(q))))
  verif(sprintf("ar1ar1 : %dx%d : log|K| analytique contre numerique", dims[1], dims[2]),
        abs(pr$logdet - determinant(K_py, logarithm = TRUE)$modulus))
}

# --- 3. la forme quadratique vectorielle contre la matrice explicite --------
cat("\n=== 3. forme quadratique ecrite en vecteurs contre u' K^-1 u ===\n")
for (q in c(6L, 25L)) {
  th <- round(rnorm(1, 0, 0.5), 6); u <- rnorm(q)
  P <- rx_level_prec_R(th, "ar1", q)$P
  verif(sprintf("ar1 : q = %d", q),
        abs(rx_quad_ar1(u, tanh(th)) - as.numeric(t(u) %*% P %*% u)))
}
for (dims in list(c(4L, 5L), c(7L, 3L))) {
  th <- round(rnorm(2, 0, 0.5), 6); q <- prod(dims); u <- rnorm(q)
  P <- rx_level_prec_R(th, "ar1ar1", q, dims = dims)$P
  verif(sprintf("ar1ar1 : %dx%d", dims[1], dims[2]),
        abs(rx_quad_ar1ar1(u, tanh(th[1]), tanh(th[2]), dims[1], dims[2]) -
            as.numeric(t(u) %*% P %*% u)))
}

# --- 4. le perimetre refuse ce qu'il doit refuser ---------------------------
cat("\n=== 4. perimetre ===\n")
hors <- list(
  list(nom = "noyau metrique iexp", tm = list(name = "s", struct = "iid", t = 1L, q = 9L, lvl = "iexp")),
  list(nom = "parente dense LK", tm = list(name = "g", struct = "iid", t = 1L, q = 9L, LK = diag(9))),
  list(nom = "fa entre caracteres", tm = list(name = "g", struct = "fa", t = 4L, q = 9L, rank = 2L)))
for (h in hors) {
  s <- rx_sparse_scope(list(h$tm), list(struct = "iid", t = 1L))
  cat(sprintf("  [%s] refuse %-24s : %s\n", if (!s$ok) "OK" else "ECHEC", h$nom,
              substr(s$reason, 1, 78)))
  if (!s$ok) ok_tot <- ok_tot + 1L else ko_tot <- ko_tot + 1L
}
dedans <- rx_sparse_scope(list(list(name = "g", struct = "us", t = 2L, q = 9L, lvl = "ar1")),
                          list(struct = "diag", t = 2L))
cat(sprintf("  [%s] accepte us(2) + ar1 + residuelle diag\n", if (dedans$ok) "OK" else "ECHEC"))
if (dedans$ok) ok_tot <- ok_tot + 1L else ko_tot <- ko_tot + 1L

cat(sprintf("\n=== %d verifications, %d echec(s) ===\n", ok_tot + ko_tot, ko_tot))
if (ko_tot > 0L) quit(status = 1L)
