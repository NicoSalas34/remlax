# =============================================================================
# UNE cellule de la grille, dans UN processus. Rien d'autre.
# =============================================================================
# POURQUOI UN PROCESSUS PAR CELLULE. Le balayage precedent mesurait toutes les
# cellules dans un meme processus, et sa colonne de compilation s'est revelee
# non interpretable : elle tombait de 79 s a 2,7 s sur les dernieres cellules,
# signe qu'un programme deja compile etait reutilise d'une cellule a l'autre.
# L'ordre d'execution influencait donc le resultat. Un processus neuf par
# cellule supprime la cause plutot que de corriger l'effet, au prix d'un
# demarrage par mesure — qui est justement ce qu'on veut mesurer.
#
# LES TROIS AXES, ET POURQUOI ILS SONT SEPARABLES.
#   n  observations         -> cout de la factorisation dense, en n^3
#   q  effets aleatoires    -> cout de la factorisation creuse, et de l'assemblage
#   p  parametres estimes   -> NOMBRE D'EVALUATIONS, non leur cout unitaire
# p est fait varier par le nombre de caracteres t sous une structure us sur le
# terme genetique, la residuelle restant diag : p = t(t+1)/2 + t. Faire croitre
# p par la RESIDUELLE le confondrait avec la structure de R.
#
# LES QUATRE CELLULES. Le premier jeton est le CAS, le second le MOTEUR.
#   creux-creux   champ ar1                      -> moteur creux      (son terrain)
#   creux-dense   champ ar1                      -> moteur dense
#   dense-dense   parente genomique pleine       -> moteur dense      (son terrain)
#   dense-creux   parente genomique pleine       -> moteur creux      LE PIRE CAS
# La derniere est celle que le perimetre declare du moteur creux exclut par
# principe : on la force par la fente Kinv, pour chiffrer ce que coute le
# mauvais choix plutot que de l'affirmer.
suppressPackageStartupMessages(library(Matrix))
a <- commandArgs(TRUE)
arg <- function(k, d = NULL) { i <- match(paste0("--", k), a); if (is.na(i)) d else a[i + 1L] }
CAS <- arg("cas", "creux"); MOT <- arg("moteur", "dense"); BK <- arg("backend", "cpu")
N   <- as.integer(arg("n", "2000")); Q <- as.integer(arg("q", "500"))
TT  <- as.integer(arg("t", "2")); OUT <- arg("out", "grille.csv"); TAG <- arg("tag", "")
SEED <- as.integer(arg("seed", "20260901"))
REPO <- arg("repo", ".")
source(file.path(REPO, "R/remlax.R"))
if (MOT == "creux") source(file.path(REPO, "R/remlax_tmb.R"))
set.seed(SEED)

inc <- function(f, q) Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f),
                                           x = 1, dims = c(length(f), q))

# --- le dispositif : t caracteres en format long, une incidence par caractere
n_unit <- max(50L, N %/% TT); n <- n_unit * TT
lev <- if (CAS == "creux") rep_len(seq_len(Q), n_unit) else sample.int(Q, n_unit, replace = TRUE)
trait <- rep(seq_len(TT), each = n_unit)
Zt <- inc(factor(rep(lev, TT), levels = seq_len(Q)), Q)
Zl <- lapply(seq_len(TT), function(k) { Z <- Zt; Z[trait != k, ] <- 0; Z })
y <- 2 + rnorm(n, 0, 1.2); X <- model.matrix(~ factor(trait))

K <- Ki <- NULL
if (CAS == "dense") {
  # parente genomique dense selon VanRaden, sur des marqueurs bialleliques
  M <- matrix(rbinom(Q * 500L, 2L, 0.3), Q, 500L)
  pf <- colMeans(M) / 2; W <- sweep(M, 2, 2 * pf)
  K <- (W %*% t(W)) / (2 * sum(pf * (1 - pf)))
  K <- K + diag(Q) * 1e-6 * mean(diag(K))
}

tm <- if (CAS == "creux") {
  rx_term("g", Zl, struct = "us", level = "ar1")
} else if (MOT == "creux") {
  # Kinv est PLEINE : c'est precisement le point de cette cellule.
  rx_term("g", Zl, struct = "us",
          Kinv = methods::as(methods::as(solve(K), "CsparseMatrix"), "generalMatrix"))
} else {
  rx_term("g", Zl, struct = "us", K = K)
}
m <- rx_model(y, X, terms = list(tm),
              residual = rx_residual(trait = factor(trait), struct = "diag"))

t0 <- proc.time()[["elapsed"]]
r <- try(if (MOT == "creux") rx_fit_sparse(m, verbose = FALSE)
         else rx_fit(m, backend = BK, hessian = FALSE, blups = FALSE, verbose = FALSE),
         silent = TRUE)
paroi <- proc.time()[["elapsed"]] - t0

ligne <- data.frame(
  cellule = paste0(CAS, "-", MOT), cas = CAS, moteur = MOT,
  backend = if (MOT == "dense") BK else "cpu",
  n = n, q = Q, t = TT, tag = TAG,
  p = if (inherits(r, "try-error")) NA_integer_ else as.integer(r$n_par),
  # t(t+1)/2 pour la us du terme, t pour la residuelle diag, et UN de plus
  # dans le cas creux : le champ ar1 porte sa correlation.
  p_attendu = as.integer(TT * (TT + 1) / 2 + TT + (CAS == "creux")),
  n_iter = if (inherits(r, "try-error")) NA_integer_ else as.integer(r$n_iter %||% NA),
  # PAROI contre INTERNE : le moteur dense est invoque hors processus et paie un
  # demarrage de python a chaque appel. Les deux sont enregistres ; seul
  # l'interne compare les algebres, seule la paroi dit ce que l'utilisateur subit.
  fit_paroi_s = paroi,
  fit_interne_s = if (inherits(r, "try-error")) NA_real_
                  else as.numeric(r$secondes %||% paroi),
  logLik = if (inherits(r, "try-error")) NA_real_ else as.numeric(r$logLik),
  statut = if (inherits(r, "try-error")) "echec" else "ok",
  erreur = if (inherits(r, "try-error"))
             substr(conditionMessage(attr(r, "condition")), 1, 160) else "",
  stringsAsFactors = FALSE)

write.table(ligne, OUT, sep = ",", row.names = FALSE, append = file.exists(OUT),
            col.names = !file.exists(OUT))
cat(sprintf("[%s-%s/%s] n=%-6d q=%-5d t=%-3d p=%-4s iter=%-5s paroi %9.2f s | interne %9.2f s %s\n",
            CAS, MOT, if (MOT == "dense") BK else "cpu", n, Q, TT,
            ligne$p, ligne$n_iter, ligne$fit_paroi_s, ligne$fit_interne_s, ligne$erreur))
