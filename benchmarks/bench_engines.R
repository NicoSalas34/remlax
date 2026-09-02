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
# --- balayage en caracteres, a n PAR CARACTERE constant (cellules "traits-*")
TS   <- as.integer(strsplit(arg("ts", "1,2,3,4,5,6,7,8"), ",")[[1]])
NU   <- as.integer(arg("n-unit", "1000"))      # observations par caractere
QT   <- as.integer(arg("q-trait", "300"))      # niveaux du terme genetique
KIN  <- identical(arg("kinship", "non"), "oui")
# La parente est construite UNE fois : elle ne depend pas de t, et la
# reconstruire par cellule changerait les donnees entre les points de la courbe.
KMAT <- NULL
if (KIN) {
  set.seed(4242L)
  M <- matrix(rbinom(QT * 1000L, 2L, 0.3), QT, 1000L)
  fr <- colMeans(M) / 2; W <- sweep(M, 2, 2 * fr)
  KMAT <- (W %*% t(W)) / (2 * sum(fr * (1 - fr)))
  KMAT <- KMAT + diag(QT) * 1e-6 * mean(diag(KMAT))
}

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
# ECHAUFFEMENT HORS MESURE. Le premier appel d'une cellule paie des couts de
# construction qui ne sont pas le calcul mesure : compilation C++ de RTMB cote
# creux, compilation XLA cote dense. Mesure du piege : le moteur creux affichait
# 0,209 s a t = 1 puis 0,012 s a t = 2, un facteur 17 a l'envers de la pente
# attendue — la mediane de deux repetitions contenait encore la construction.
# Un appel non chronometre precede donc les repetitions.
chrono <- function(f, reps) {
  invisible(try(f(), silent = TRUE))
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

# --- COMPLEXITE DU MODELE A n PAR CARACTERE CONSTANT --------------------------
# Le dispositif que fait reellement un selectionneur : les MEMES plantes, mesurees
# sur de plus en plus de caracteres. Il couple donc deliberement n et p, la ou le
# balayage precedent tenait n constant — les deux ensemble separent les effets.
#
# us(t) porte t(t+1)/2 parametres (les variances ET les covariances), soit 36 a
# t = 8. La residuelle est diag(t) pour que p soit domine par le terme genetique
# et reste lisible : une residuelle us doublerait le compte sans changer l'axe.
#
# L'incidence est une LISTE de t matrices n x q, une par colonne de Sigma, ce
# qu'attend rx_term. Toutes les observations d'une unite partagent son niveau.
cas_traits <- function(n_unit, t, q, K = NULL) {
  n <- n_unit * t
  lev <- rep(seq_len(q), length.out = n_unit)
  # une matrice par caractere : la ligne (i-1)*t + j porte le caractere j
  Zl <- lapply(seq_len(t), function(j) {
    M <- Matrix::sparseMatrix(i = seq_len(n_unit), j = lev, x = 1,
                              dims = c(n_unit, q))
    # dupliquer chaque unite t fois, mais n'activer que le bloc du caractere j
    idx <- (seq_len(n_unit) - 1L) * t + j
    Z <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                              dims = c(n, q))
    Z[idx, ] <- M
    Z
  })
  trait <- factor(rep(seq_len(t), times = n_unit))
  unit  <- rep(seq_len(n_unit), each = t)
  # signal genetique correle entre caracteres, pour que us ait quelque chose a
  # estimer : sans covariance reelle, l'optimum est sur le bord et le nombre
  # d'iterations mesure une degenerescence, pas une difficulte.
  A <- matrix(rnorm(t * t), t, t) / sqrt(t)
  G <- A %*% t(A) + diag(t) * 0.5
  U <- matrix(rnorm(q * t), q, t) %*% chol(G)
  if (!is.null(K)) U <- chol(K) %*% U
  y <- 2 + U[cbind(lev[unit], as.integer(trait))] + rnorm(n, 0, 0.8)
  # t = 1 est le point de depart demande, et model.matrix refuse un facteur a un
  # seul niveau : la moyenne generale suffit alors.
  X <- if (t == 1L) matrix(1, n, 1) else model.matrix(~ trait)
  Ki <- if (is.null(K)) NULL else
        as(as(solve(K), "CsparseMatrix"), "generalMatrix")
  list(y = y, X = X, q = q, t = t, n = n,
       terms_d = list(rx_term("g", Zl, K = K, struct = "us")),
       terms_s = list(rx_term("g", Zl, Kinv = Ki, struct = "us")),
       # A t = 1 il n'y a qu'une variance residuelle : `diag` n'a rien a
       # diagonaliser et le solveur le refuse, a juste titre.
       residual = if (t == 1L) rx_residual()
                  else rx_residual(struct = "diag", trait = trait, unit = unit))
}


lignes <- list()
note <- function(...) cat(..., "\n", sep = "")

for (cl in CELL) {
  parts <- strsplit(cl, "-")[[1]]; cas <- parts[1]; mot <- parts[2]
  grille <- if (cas == "traits") lapply(TS, function(tt) list(n = NU * tt, q = QT, t = tt))
            else if (cas == "creux") lapply(NS, function(n) list(n = n, q = NA))
            else unlist(lapply(NS, function(n) lapply(QS, function(q) list(n = n, q = q))), recursive = FALSE)
  for (g in grille) {
    set.seed(1000L + g$n + (if (is.na(g$q)) 0L else g$q))
    # Meme garde-fou que le banc python : sous 2q unites des niveaux ne sont pas
    # repliques, le systeme devient quasi singulier, et le surcout se lirait
    # comme un effet de la complexite alors que le dispositif est degenere.
    if (identical(cas, "traits") && 2L * g$q > NU) {
      note(sprintf("[%s] t=%d IGNORE : 2q=%d > n_unit=%d, replication insuffisante",
                   cl, g$t, 2L * g$q, NU)); next
    }
    # q >= n : plus d'effets que d'observations. Le facteur porte alors des
    # niveaux NON OBSERVES, que rx_term ne compte pas — d'ou une K de taille q
    # face a un terme de taille inferieure. Le cas est degenere (la variance
    # n'est pas identifiee) : on le saute en le DISANT, plutot que de le laisser
    # interrompre le balayage.
    if (identical(cas, "traits") && g$q * 2L > NU) {
      note(sprintf("[%s] t=%d IGNORE : q=%d > n_unit/2=%d, replication insuffisante",
                   cl, g$t, g$q, NU %/% 2L)); next
    }
    if (!identical(cas, "traits") && !is.na(g$q) && g$q >= g$n) {
      note(sprintf("[%s] n=%d q=%d IGNORE : q >= n, cas degenere", cl, g$n, g$q))
      next
    }
    d <- if (cas == "traits") cas_traits(NU, g$t, g$q, K = if (KIN) KMAT else NULL)
         else if (cas == "creux") cas_creux(g$n) else cas_dense(g$n, g$q)
    tms <- if (mot == "dense") d$terms_d else d$terms_s
    m <- rx_model(d$y, d$X, terms = tms, residual = d$residual, name = cl)
    # THETA IMPOSE, IDENTIQUE POUR LES DEUX MOTEURS. Sans lui, chaque moteur
    # evalue a SON theta initial et la colonne logLik n'est pas comparable d'un
    # moteur a l'autre — un piege verifie : les valeurs divergeaient de 1e-5 a
    # 9e-3 en croissant avec t, ce qui ressemblait a un defaut de formulation,
    # alors qu'a theta COMMUN les deux moteurs s'accordent a 1e-12. Le zero sur
    # l'echelle transformee (variances a 1, correlations nulles) est le choix
    # naturel : il est bien conditionne, reproductible, et ne depend d'aucun
    # moteur. Les deux comptes de parametres sont identiques, verifie par la
    # suite de concordance.
    th0 <- rep(0, rx_n_theta(m))
    r <- tryCatch({
      if (mot == "dense") {
        ev <- chrono(function() fit_dense(m, theta_init = th0, maxiter = 0L,
                                          polish = 0L, hessian = FALSE,
                                          blups = FALSE, verbose = FALSE), REPS)
        aj <- chrono(function() fit_dense(m, hessian = FALSE, blups = FALSE,
                                         verbose = FALSE), 1L)
        list(ev = ev$val$secondes %||% NA_real_, ev_paroi = ev$med,
             ev_min = ev$min, ev_max = ev$max, ll = ev$val$logLik,
             # LA VRAISEMBLANCE A L'OPTIMUM, distincte de celle a theta impose.
             # Sans elle, comparer des TEMPS d'ajustement entre moteurs n'a pas
             # de sens : un moteur qui s'arrete a un point moins bon n'est pas
             # plus rapide, il est moins bon. La colonne ll est la valeur au
             # theta impose commun et ne repond pas a cette question.
             ll_fit = aj$val$logLik %||% NA_real_,
             fit = aj$val$secondes %||% NA_real_, fit_paroi = aj$med,
             n_iter = aj$val$n_iter %||% NA_integer_,
             # COMPTE et non inference. Diviser le temps d'ajustement par le
             # cout unitaire donne un nombre d'evaluations plausible et faux :
             # c'est precisement l'erreur qui faisait echouer la reconstruction
             # du modele de cout d'un facteur deux. Le solveur les compte.
             n_eval = aj$val$n_eval %||% NA_integer_,
             compile_s = aj$val$compile_s %||% NA_real_, err = "")
      } else {
        # LE COUT UNITAIRE CREUX EST CELUI QUE LE SOLVEUR RAPPORTE, jamais le
        # temps de paroi de l'appel : celui-ci inclut la construction du ruban
        # de derivation, payee a chaque appel de rx_fit_sparse mais une seule
        # fois dans un ajustement. Mesure du piege : le controle
        # « ajustement >= n_eval x cout unitaire » echouait sur les HUIT
        # cellules creuses d'un balayage et sur aucune dense. On ne prend donc
        # plus $logLik ici, pour garder acces aux postes du resultat.
        ev <- chrono(function() rx_fit_sparse(m, theta_init = th0, maxiter = 0L,
                                              verbose = FALSE), REPS)
        # SYMETRIE DE L'APPEL. Le moteur dense est appele avec hessian = FALSE
        # et blups = FALSE ; sans sdreport = FALSE le moteur creux formerait en
        # plus la covariance de TOUS les effets aleatoires, que le dense ne
        # calcule pas. Mesure de l'asymetrie : ce rapport pesait 28 a 58 pour
        # cent du temps d'ajustement creux a q_total de 300 a 900, et sa part
        # croit avec le nombre d'effets — a t = 8 sur le cluster, 93 pour cent
        # du temps d'ajustement echappait au bilan des postes. Le comparer ainsi
        # SOUS-ESTIMAIT l'avantage du moteur creux.
        # Le plafond doit aussi egaler celui du dense : 300 contre 3000 aurait
        # tronque le creux et laisse le dense aller au bout, defaut deja corrige
        # une fois dans ce projet.
        aj <- chrono(function() rx_fit_sparse(m, maxiter = 3000L, verbose = FALSE,
                                              sdreport = FALSE), 1L)
        list(ev = ev$val$eval_s %||% NA_real_, ev_paroi = ev$med,
             ev_min = ev$min, ev_max = ev$max,
             ll = ev$val$logLik %||% NA_real_,
             ll_fit = aj$val$logLik %||% NA_real_,
             fit = aj$val$secondes %||% aj$med, fit_paroi = aj$med,
             n_iter = aj$val$n_iter %||% NA_integer_,
             n_eval = aj$val$n_eval %||% NA_integer_,
             # `compile_s` porte ici la CONSTRUCTION du ruban, poste homologue
             # de la compilation XLA du moteur dense : paye une fois, amorti sur
             # les iterations.
             compile_s = ev$val$construct_s %||% NA_real_, err = "")
      }
    }, error = function(e) list(ev = NA_real_, ev_paroi = NA_real_,
                                ev_min = NA_real_, ev_max = NA_real_,
                                ll = NA_real_, fit = NA_real_, fit_paroi = NA_real_,
                                n_iter = NA_integer_, n_eval = NA_integer_,
                                compile_s = NA_real_, ll_fit = NA_real_,
                                err = substr(conditionMessage(e), 1, 200)))
    lignes[[length(lignes) + 1L]] <- data.frame(
      cellule = cl, cas = cas, moteur = mot, backend = if (mot == "dense") BK else "cpu",
      n = g$n, t = if (is.null(g$t)) 1L else g$t, q_gen = g$q, q_total = d$q,
      p_us = if (is.null(g$t)) NA_integer_ else g$t * (g$t + 1L) / 2L, reps = REPS,
      eval_s = r$ev, eval_paroi_s = r$ev_paroi,
      eval_min_s = r$ev_min, eval_max_s = r$ev_max,
      fit_s = r$fit, fit_paroi_s = r$fit_paroi, n_iter = r$n_iter,
      n_eval = r$n_eval, compile_s = r$compile_s,
      logLik = r$ll, logLik_fit = r$ll_fit,
      rss_mb = as.numeric(gsub("[^0-9]", "", system("grep VmRSS /proc/self/status", intern = TRUE))) / 1024,
      tag = TAG, erreur = r$err, stringsAsFactors = FALSE)
    note(sprintf("[%s] t=%-2s n=%-6d q=%-5s eval %8.4f s (paroi %8.3f) | ajust %8.2f s | iter %-4s %s",
                 cl, ifelse(is.null(g$t), "-", g$t), g$n, ifelse(is.na(g$q), "-", g$q),
                 r$ev %||% NA_real_, r$ev_paroi %||% NA_real_, r$fit %||% NA_real_,
                 r$n_iter, r$err))
  }
}
df <- do.call(rbind, lignes)
dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)
write.csv(df, OUT, row.names = FALSE)
note(sprintf("%d mesures ecrites dans %s", nrow(df), OUT))
