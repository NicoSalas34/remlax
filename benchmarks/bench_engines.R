# =============================================================================
# La grille de decision : {cas creux, cas dense} x {dense CPU, dense GPU, creux CPU}
# =============================================================================
# SIX cellules et non huit. Le moteur creux factorise par CHOLMOD, qui est CPU :
# la cellule "creux x GPU" N'EXISTE PAS, et cette absence est un resultat — elle
# rend les choix "creux" et "GPU" exclusifs, donc la decision est un arbitrage et
# non une optimisation a deux axes independants.
#
# Les deux cas sont definis par la STRUCTURE, jamais par la taille :
#   CREUX : facteur iid + champ ar1, dont les inverses sont diagonal et tridiagonal
#   DENSE : parente genomique, dont l'inverse est PLEIN
#
# La cellule la plus instructive est le cas DENSE force sur le moteur creux. Le
# perimetre declare l'exclut par principe ; on l'y force ici, via Kinv, pour
# mesurer l'ampleur du cout. C'est la reponse quantitative a "pourquoi ne pas
# faire du creux partout", et ce qui calibre l'outil de decision au lieu de le
# laisser sur un rapport de flops non calibre.
suppressPackageStartupMessages(library(Matrix))
`%||%` <- function(a, b) if (is.null(a)) b else a
source("R/remlax.R"); source("R/remlax_tmb.R")

a <- commandArgs(TRUE)
arg <- function(k, d) { i <- match(paste0("--", k), a); if (is.na(i)) d else a[i + 1L] }
BK   <- arg("backend", "cpu")            # cpu | gpu, pour le moteur DENSE
CELL <- strsplit(arg("cells", "creux-dense,creux-creux,dense-dense,dense-creux"), ",")[[1]]
NS   <- as.integer(strsplit(arg("ns", "2000,4000"), ",")[[1]])
QS   <- as.integer(strsplit(arg("qs", "1000,2000"), ",")[[1]])
REPS <- as.integer(arg("reps", "3"))
OUT  <- arg("out", "bench_engines.csv")
TAG  <- arg("tag", "")

# rx_fit peut ou non exposer le backend : on le passe s'il existe.
# rx_fit expose backend = c("auto", "gpu", "cpu") : on le fixe explicitement,
# jamais "auto", pour qu'une cellule GPU qui retomberait sur le CPU se voie.
fit_dense <- function(m, ...) rx_fit(m, backend = BK, ...)

# BIAIS A NE PAS MESURER. Le moteur dense est invoque HORS PROCESSUS : chaque
# appel paie le demarrage de Python et la compilation XLA, mesures a ~3 s contre
# 0,07 s pour une evaluation du moteur creux en processus. Comparer ces temps de
# paroi produirait une conclusion fausse sur le cout du calcul. On enregistre
# donc, pour le moteur dense, le temps INTERNE que le solveur rapporte lui-meme
# (champ `secondes`), et le temps de paroi dans une colonne separee : l'ecart
# entre les deux EST le cout d'invocation, qui a son propre interet puisqu'un
# ajustement le paie une fois et une boucle de comparaison de modeles a chaque
# appel.
chrono <- function(f, reps) {
  ts <- numeric(reps)
  for (k in seq_len(reps)) { t0 <- proc.time()[["elapsed"]]; v <- f(); ts[k] <- proc.time()[["elapsed"]] - t0 }
  list(med = median(ts), min = min(ts), max = max(ts), val = v)
}

# --- les deux familles de dispositifs -----------------------------------------
cas_creux <- function(n) {
  q1 <- n %/% 5L; nr <- 40L; nc <- n %/% (2L * nr)
  qf <- nr * nc
  f1 <- factor(sample.int(q1, n, replace = TRUE), levels = seq_len(q1))
  f2 <- factor(rep(seq_len(qf), length.out = n), levels = seq_len(qf))
  y <- 2 + rnorm(n, 0, 1.2); X <- matrix(1, n, 1)
  list(y = y, X = X,
       terms_d = list(rx_term("a", f1, struct = "iid"),
                      rx_term("s", f2, struct = "iid", level = "ar1")),
       terms_s = list(rx_term("a", f1, struct = "iid"),
                      rx_term("s", f2, struct = "iid", level = "ar1")),
       residual = rx_residual(), q = q1 + qf)
}

cas_dense <- function(n, q) {
  # parente genomique dense selon VanRaden, sur des marqueurs bialleliques
  M <- matrix(rbinom(q * 500L, 2L, 0.3), q, 500L)
  p <- colMeans(M) / 2; W <- sweep(M, 2, 2 * p)
  K <- (W %*% t(W)) / (2 * sum(p * (1 - p)))
  K <- K + diag(q) * 1e-6 * mean(diag(K))
  lev <- factor(rep(seq_len(q), length.out = n), levels = seq_len(q))
  y <- 2 + rnorm(n, 0, 1.2); X <- matrix(1, n, 1)
  # Kinv est PLEINE : c'est le point. On la fournit pour forcer le moteur creux.
  Ki <- as(as(solve(K), "CsparseMatrix"), "generalMatrix")
  list(y = y, X = X,
       terms_d = list(rx_term("g", lev, K = K, struct = "iid")),
       terms_s = list(rx_term("g", lev, Kinv = Ki, struct = "iid")),
       residual = rx_residual(), q = q, K = K)
}

lignes <- list()
note <- function(...) cat(..., "\n", sep = "")

for (cl in CELL) {
  parts <- strsplit(cl, "-")[[1]]; cas <- parts[1]; mot <- parts[2]
  grille <- if (cas == "creux") lapply(NS, function(n) list(n = n, q = NA))
            else unlist(lapply(NS, function(n) lapply(QS, function(q) list(n = n, q = q))), recursive = FALSE)
  for (g in grille) {
    set.seed(1000L + g$n + (if (is.na(g$q)) 0L else g$q))
    # q >= n : plus d'effets que d'observations. Le facteur porte alors des
    # niveaux NON OBSERVES, que rx_term ne compte pas — d'ou une K de taille q
    # face a un terme de taille inferieure. Le cas est degenere (la variance
    # n'est pas identifiee) : on le saute en le DISANT, plutot que de le laisser
    # interrompre le balayage.
    if (!is.na(g$q) && g$q >= g$n) {
      note(sprintf("[%s] n=%d q=%d IGNORE : q >= n, cas degenere", cl, g$n, g$q))
      next
    }
    d <- if (cas == "creux") cas_creux(g$n) else cas_dense(g$n, g$q)
    tms <- if (mot == "dense") d$terms_d else d$terms_s
    m <- rx_model(d$y, d$X, terms = tms, residual = d$residual, name = cl)
    r <- tryCatch({
      # theta de depart commun, pour que les deux moteurs evaluent AU MEME point
      if (mot == "dense") {
        ev <- chrono(function() fit_dense(m, maxiter = 0L, polish = 0L,
                                          hessian = FALSE, blups = FALSE,
                                          verbose = FALSE), REPS)
        aj <- chrono(function() fit_dense(m, hessian = FALSE, blups = FALSE,
                                         verbose = FALSE), 1L)
        list(ev = ev$val$secondes %||% NA_real_, ev_paroi = ev$med,
             ev_min = ev$min, ev_max = ev$max, ll = ev$val$logLik,
             fit = aj$val$secondes %||% NA_real_, fit_paroi = aj$med,
             n_iter = aj$val$n_iter %||% NA_integer_, err = "")
      } else {
        ev <- chrono(function() rx_fit_sparse(m, maxiter = 0L, verbose = FALSE)$logLik, REPS)
        aj <- chrono(function() rx_fit_sparse(m, maxiter = 300L, verbose = FALSE), 1L)
        list(ev = ev$med, ev_paroi = ev$med, ev_min = ev$min, ev_max = ev$max,
             ll = ev$val, fit = aj$med, fit_paroi = aj$med,
             n_iter = aj$val$n_iter %||% NA_integer_, err = "")
      }
    }, error = function(e) list(ev = NA, ev_paroi = NA, ev_min = NA, ev_max = NA,
                                ll = NA, fit = NA, fit_paroi = NA, n_iter = NA,
                                err = substr(conditionMessage(e), 1, 200)))
    lignes[[length(lignes) + 1L]] <- data.frame(
      cellule = cl, cas = cas, moteur = mot, backend = if (mot == "dense") BK else "cpu",
      n = g$n, q_gen = g$q, q_total = d$q, reps = REPS,
      eval_s = r$ev, eval_paroi_s = r$ev_paroi,
      eval_min_s = r$ev_min, eval_max_s = r$ev_max,
      fit_s = r$fit, fit_paroi_s = r$fit_paroi, n_iter = r$n_iter, logLik = r$ll,
      rss_mb = as.numeric(gsub("[^0-9]", "", system("grep VmRSS /proc/self/status", intern = TRUE))) / 1024,
      tag = TAG, erreur = r$err, stringsAsFactors = FALSE)
    note(sprintf("[%s] n=%-6d q=%-6s eval %8.4f s (paroi %8.3f) | ajust %8.2f s | iter %-4s %s",
                 cl, g$n, ifelse(is.na(g$q), "-", g$q),
                 r$ev %||% NA_real_, r$ev_paroi %||% NA_real_, r$fit %||% NA_real_,
                 r$n_iter, r$err))
  }
}
df <- do.call(rbind, lignes)
dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)
write.csv(df, OUT, row.names = FALSE)
note(sprintf("%d mesures ecrites dans %s", nrow(df), OUT))
