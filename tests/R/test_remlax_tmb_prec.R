# =============================================================================
# La fente Kinv : declarer un terme par sa PRECISION plutot que par sa parente
# =============================================================================
# C'est la voie du moteur creux pour une parente genealogique : les regles de
# Henderson donnent A^-1 directement, creuse, sans jamais former ni factoriser A.
#
# L'ARBITRE est un calcul REML en algebre dense, ecrit ici et independant des
# deux moteurs : V = sigma^2 Z A Z' + sigma_e^2 I, log|V| + log|X'V^-1X| + y'Py.
# Sans lui, un decalage CONSTANT de vraisemblance passe inapercu — les
# estimations de theta restent justes au cinquieme chiffre, seule la valeur de la
# vraisemblance est fausse, donc toute comparaison de modeles l'est aussi. C'est
# exactement le defaut que ce test a trouve : l'objectif lisait tm$Kinv_logdet et
# retombait sur ZERO quand il etait absent, d'ou un ecart de 0,5 * log|K|.
suppressPackageStartupMessages(library(Matrix))
ici <- function(f) file.path(dirname(dirname(dirname(normalizePath(
  sys.frame(1)$ofile %||% "tests/R/x")))), f)
`%||%` <- function(a, b) if (is.null(a)) b else a
source("R/remlax.R"); source("R/remlax_tmb.R")

ok <- 0L; ko <- 0L
verif <- function(nom, ecart, tol = 1e-8) {
  bon <- is.finite(ecart) && abs(ecart) <= tol
  if (bon) ok <<- ok + 1L else ko <<- ko + 1L
  cat(sprintf("  [%s] %-52s ecart %+.3e\n", if (bon) "OK" else "ECHEC", nom, ecart))
}

set.seed(7)
q <- 60L; rep_ <- 5L
# A^-1 creuse et definie positive, du type que produisent les regles de Henderson
Ai <- bandSparse(q, k = c(0, 1), symmetric = TRUE,
                 diagonals = list(rep(2.2, q), rep(-0.6, q - 1)))
A <- solve(as.matrix(Ai))
lev <- factor(rep(seq_len(q), rep_), levels = seq_len(q)); n <- length(lev)
u <- as.numeric(chol(A) %*% rnorm(q))
y <- 3 + u[as.integer(lev)] + rnorm(n, 0, 0.7); X <- matrix(1, n, 1)
Z <- sparseMatrix(i = seq_len(n), j = as.integer(lev), x = 1, dims = c(n, q))

cat("=== 1. la fente accepte une precision creuse, et refuse les melanges ===\n")
tm <- rx_term("g", lev, Kinv = Ai, struct = "iid")
verif("Kinv portee par le terme", nrow(tm$Kinv) - q)
verif("niveau resolu en 'prec'", if (identical(rx_level_of(tm), "prec")) 0 else 1)
verif("dans le perimetre creux", if (rx_sparse_scope(list(tm), rx_residual())$ok) 0 else 1)
verif("refuse K et Kinv ensemble",
      if (inherits(try(rx_term("g", lev, K = A, Kinv = Ai), silent = TRUE),
                   "try-error")) 0 else 1)

cat("=== 2. la MEME parente par deux voies donne la MEME vraisemblance ===\n")
m_s <- rx_model(y, X, terms = list(tm), residual = rx_residual())
m_d <- rx_model(y, X, terms = list(rx_term("g", lev, K = A, struct = "iid")),
                residual = rx_residual())
dn <- rx_fit(m_d, hessian = FALSE, blups = FALSE, verbose = FALSE)
sp <- rx_fit_sparse(m_s, maxiter = 300L, verbose = FALSE)

arbitre <- function(th) {
  V <- exp(th[1])^2 * as.matrix(Z %*% A %*% t(Z)) + exp(th[2])^2 * diag(n)
  Vi <- solve(V); B <- t(X) %*% Vi %*% X
  P <- Vi - Vi %*% X %*% solve(B) %*% t(X) %*% Vi
  -0.5 * as.numeric(determinant(V, TRUE)$modulus + determinant(B, TRUE)$modulus +
                    t(y) %*% P %*% y) - 0.5 * (n - ncol(X)) * log(2 * pi)
}
a_d <- arbitre(dn$theta)
verif("arbitre contre moteur dense, au theta du dense", dn$logLik - a_d, 1e-9)
sA <- rx_fit_sparse(m_s, theta_init = dn$theta, maxiter = 0L, verbose = FALSE)
verif("arbitre contre moteur creux, au theta du dense", sA$logLik - a_d, 1e-6)
verif("les deux moteurs, au theta du dense", sA$logLik - dn$logLik, 1e-6)
verif("optima des deux moteurs", sp$logLik - dn$logLik, 1e-5)
cat(sprintf("      log|A| = %.6f : un oubli de ce terme decalerait de %.3f\n",
            as.numeric(determinant(A, TRUE)$modulus),
            0.5 * as.numeric(determinant(A, TRUE)$modulus)))

cat("=== 3. log|K| fourni par l'utilisateur : meme resultat ===\n")
tm2 <- rx_term("g", lev, Kinv = Ai, struct = "iid",
               Kinv_logdet = as.numeric(determinant(A, TRUE)$modulus))
m_s2 <- rx_model(y, X, terms = list(tm2), residual = rx_residual())
s2 <- rx_fit_sparse(m_s2, theta_init = dn$theta, maxiter = 0L, verbose = FALSE)
verif("log|K| fourni contre log|K| calcule", s2$logLik - sA$logLik, 1e-10)

cat(sprintf("\n=== %d verifications, %d echec(s) ===\n", ok + ko, ko))
if (ko > 0) quit(status = 1)
