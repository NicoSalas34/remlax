# =============================================================================
# LE BILAN DE COUT DU MOTEUR CREUX SE REFERME-T-IL ?
# =============================================================================
# POURQUOI CE FICHIER EXISTE. Le protocole de mesure repose sur un modele a
# trois postes : une construction payee une fois, un cout par evaluation, et un
# nombre d'evaluations. Le controle qui le refute est
#
#     ajustement  >=  nombre d'evaluations x cout unitaire
#
# une borne INFERIEURE absolue. Elle a echoue sur les HUIT cellules creuses d'un
# balayage et sur aucune dense : le cout unitaire mesure incluait la
# construction du ruban de derivation, payee a chaque appel de rx_fit_sparse
# mais une seule fois dans un ajustement. A t = 8, 38 iterations a 0,757 s
# auraient demande 28,8 s quand l'ajustement entier prenait 11,25 s.
#
# Deux pieges de mesure ont ete trouves en le corrigeant, et ce fichier les
# garde sous surveillance :
#   1. appeler obj$fn deux fois AU MEME theta ne mesure presque rien, le second
#      appel reutilisant le mode interne en (beta, u) deja converge ;
#   2. proc.time() quantifie a ~10 ms, or une evaluation creuse est
#      sous-milliseconde aux petites tailles — celles qui portent le point de
#      croisement entre moteurs.
suppressMessages({source("R/remlax.R"); source("R/remlax_tmb.R")})
src <- readLines("benchmarks/bench_engines.R")
i <- grep("^cas_traits <- function", src); j <- grep("^lignes <- list", src)[1]
eval(parse(text = paste(src[i:(j - 1L)], collapse = "\n")))

a <- commandArgs(TRUE)
arg <- function(k, d) { i <- match(paste0("--", k), a); if (is.na(i)) d else a[i + 1L] }
NU <- as.integer(arg("n-unit", "600")); QT <- as.integer(arg("q-trait", "80"))
TS <- as.integer(strsplit(arg("ts", "1,2,3,4,5"), ",")[[1]])

ok <- 0L; ko <- 0L
cat(sprintf("%3s %7s %10s %11s %6s %7s %9s %8s\n", "t", "n", "construct",
            "eval", "iter", "n_eval", "ajust", "borne/aj"))
for (t in TS) {
  set.seed(77L)
  d  <- cas_traits(NU, t, QT, K = NULL)
  ms <- rx_model(d$y, d$X, terms = d$terms_s, residual = d$residual, name = "s")
  th <- rep(0, rx_n_theta(ms))
  e <- rx_fit_sparse(ms, theta_init = th, maxiter = 0L, verbose = FALSE)
  f <- rx_fit_sparse(ms, maxiter = 3000L, verbose = FALSE)
  ne <- if (is.na(f$n_eval)) f$n_iter else f$n_eval
  borne <- f$construct_s + ne * e$eval_s
  rap <- borne / f$secondes
  cat(sprintf("%3d %7d %10.4f %11.6f %6d %7d %9.3f %8.2f%s\n",
              t, d$n, e$construct_s, e$eval_s, f$n_iter, ne, f$secondes, rap,
              if (rap > 1.05) "  <-- BORNE DEPASSEE" else ""))
  # La borne peut etre en dessous du total (postes non mesures : gradients,
  # sdreport), jamais AU-DESSUS : cela signifierait un cout unitaire surestime.
  if (rap > 1.05) ko <- ko + 1L else ok <- ok + 1L
}
cat(sprintf("\n%d cellule(s) coherente(s), %d borne(s) depassee(s)\n", ok, ko))
if (ko > 0L) quit(status = 1L)
