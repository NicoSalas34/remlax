# =============================================================================
# bench_complexite.R — banc de vitesse A MODELE EGAL sur DEUX axes : la taille
#                      (n) et la complexite du modele (nombre de parametres de
#                      variance, structures), pour asreml, remlax CPU, remlax GPU
# =============================================================================
# CE QUI EST MESURE, par ajustement.
#   wall_s      horloge murale de l'appel complet (R -> resultat)
#   startup_s   remlax : wall - compile - solveur (lancement de Python, import
#               de JAX, ecriture/lecture du bundle) ; asreml : NA
#   compile_s   remlax : compilation XLA, payee UNE fois ; asreml : NA
#   solver_s    remlax : optimisation seule (L-BFGS-B + polissage) ; asreml : wall
#   n_iter      remlax : iterations L-BFGS-B ; asreml : iterations AI
#   n_eval      remlax : evaluations de -2logL et gradient (recherche lineaire
#               et polissage compris) ; asreml : = n_iter
#   s_per_iter  solver_s / n_iter
#   s_per_eval  remlax : solver_s / n_eval ; eval_s : une evaluation chronometree
#   logLik      convention asreml (fit$logLik_asreml), comparable a a$loglik
#   converged, n_par, n_obs, components
#
# CAS (n_par croissant a taille fixee) :
#   iid     un facteur aleatoire + covariable                              2
#   grm     idem avec parente VanRaden dense (q = n/4)                     2
#   ar1ar1  champ nr x nc, genotype iid, residuelle ar1(row):ar1(col)      4
#   us3, us6, us9, us12   t caracteres, us genetique et us residuelle      t(t+1)
#   usK3, usK6            idem avec parente dense sur le terme genetique   t(t+1)
#   ige     le modele du chapitre 3 : str(dge + voisinage) us(2) sur GRM=I,
#           IGE inter-specifique iid, champ ar1 x ar1 aleatoire, pepite    8
#
# DELAI PAR AJUSTEMENT (--timeout-fit, s). remlax : prefixe `timeout` sur la
# commande Python (RX_PY = "timeout <s> python3") ; asreml : fork + kill. Un
# ajustement arrete est enregistre en status "timeout" avec le delai en wall_s.
#
#   Rscript benchmarks/bench_complexite.R --sizes 500,2000,8000 \
#       --cases iid,grm,ar1ar1,us3,us6,us9,us12,usK3,usK6,ige \
#       --software asreml,remlax --tag cpu4 --timeout-fit 7200 --out benchmarks/results
#   RX_BACKEND=gpu Rscript benchmarks/bench_complexite.R --software remlax --tag a100
#
# Chaque mesure est imprimee des qu'elle est prise (ligne "BENCH|..."), et le
# CSV est reecrit a chaque ligne : un job tue laisse ses donnees.
# =============================================================================
suppressPackageStartupMessages({ library(here); library(Matrix); library(jsonlite) })
source(here::here("R", "remlax.R"))

args <- commandArgs(trailingOnly = TRUE)
opt <- function(nom, defaut) {
  i <- match(paste0("--", nom), args)
  if (is.na(i) || i == length(args)) defaut else args[i + 1L]
}
SIZES <- as.integer(strsplit(opt("sizes", "500,2000"), ",")[[1]])
CASES <- strsplit(opt("cases", "iid,grm,ar1ar1,us3,us6,us9,us12,usK3,usK6,ige"), ",")[[1]]
SOFT  <- strsplit(opt("software", "asreml,remlax"), ",")[[1]]
TAG   <- opt("tag", "cpu")
OUT   <- opt("out", here::here("benchmarks", "results"))
TIMEOUT <- as.numeric(opt("timeout-fit", "7200"))
# Repetitions par genotype : la DENSITE du probleme. q = n / REPS pour iid, grm,
# ar1ar1 et ige ; q = (n / t) / REPS pour les cas multi-caracteres. REPS = 1
# n'est estimable qu'avec parente (grm, usK) ou voisinage (ige).
REPS <- as.numeric(opt("reps", "4"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
MAXIT_REF <- 100L
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
DATE <- format(Sys.Date(), "%Y-%m-%d")
FICHIER <- file.path(OUT, sprintf("benchc_%s_%s.csv", DATE, TAG))  # TAG doit porter reps si != 4

# remlax : delai par le prefixe timeout sur la commande Python
if ("remlax" %in% SOFT && is.finite(TIMEOUT) && TIMEOUT > 0) {
  py <- rx_python_cmd()
  if (py[1] != "timeout") Sys.setenv(RX_PY = paste(c("timeout", as.integer(TIMEOUT), py), collapse = " "))
}
if ("asreml" %in% SOFT) {
  if (!requireNamespace("asreml", quietly = TRUE)) { cat("asreml absent\n"); SOFT <- setdiff(SOFT, "asreml") }
  else { suppressPackageStartupMessages(library(asreml)); asreml.options(trace = FALSE, workspace = "8gb", pworkspace = "2gb") }
}
sha <- Sys.getenv("RX_GIT_SHA", "")
if (!nzchar(sha)) sha <- tryCatch(suppressWarnings(system2("git", c("-C", shQuote(here::here()), "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE))[1], error = function(e) "?")
jaxv <- tryCatch(system2(rx_python_cmd()[length(rx_python_cmd())], c("-c", shQuote("import jax; print(jax.__version__)")), stdout = TRUE)[1], error = function(e) "?")
versions <- c(remlax = paste0(if (is.na(sha) || !nzchar(sha)) "?" else sha, "/jax", jaxv),
              asreml = if ("asreml" %in% SOFT) as.character(packageVersion("asreml")) else NA)
cat("versions :", paste(names(versions), versions, sep = "=", collapse = " | "), "| backend", BK,
    "| coeurs", Sys.getenv("SLURM_CPUS_PER_TASK", "?"), "| OMP", Sys.getenv("OMP_NUM_THREADS", "?"),
    "| timeout", TIMEOUT, "s\n")

COLS <- c("date", "tag", "backend", "case", "n", "reps", "q", "n_par", "software", "version", "status",
          "wall_s", "startup_s", "compile_s", "solver_s", "n_iter", "n_eval", "s_per_iter", "s_per_eval", "eval_s",
          "converged", "logLik", "components", "error")
RES <- list()
ligne <- function(...) {
  r <- list(...)
  for (k in setdiff(COLS, names(r))) r[[k]] <- NA
  r <- r[COLS]
  RES[[length(RES) + 1L]] <<- as.data.frame(r, stringsAsFactors = FALSE)
  cat("BENCH|", paste(sapply(r, function(v) if (is.null(v) || length(v) == 0) "" else as.character(v)), collapse = "|"), "\n", sep = "")
  write.csv(do.call(rbind, RES), FICHIER, row.names = FALSE)
}
comp_str <- function(v) paste(sprintf("%.6g", as.numeric(v)), collapse = ";")
chrono <- function(expr) { t0 <- proc.time()[["elapsed"]]; v <- expr; list(v = v, s = proc.time()[["elapsed"]] - t0) }
grm <- function(q, seed) {
  set.seed(seed); M <- matrix(rbinom(q * 2 * q, 2, 0.3), q, 2 * q)
  pfr <- colMeans(M) / 2; W <- sweep(M, 2, 2 * pfr)
  K <- tcrossprod(W) / (2 * sum(pfr * (1 - pfr))) + diag(1e-3, q)
  dimnames(K) <- list(paste0("g", 1:q), paste0("g", 1:q)); K
}
n_traits <- function(cas) as.integer(sub("^usK?", "", cas))

# ---- generation des dispositifs ------------------------------------------------
gen_case <- function(cas, n) {
  set.seed(1000 + n)
  if (cas %in% c("iid", "grm")) {
    q <- max(2L, round(n / REPS)); gid <- factor(rep(paste0("g", 1:q), length.out = n), levels = paste0("g", 1:q))
    K <- if (cas == "grm") grm(q, n) else NULL
    u <- if (is.null(K)) rnorm(q, 0, 1.2) else as.numeric(t(chol(K)) %*% rnorm(q)) * 1.2
    x <- rnorm(n); y <- 5 + 0.4 * x + u[as.integer(gid)] + rnorm(n, 0, 1)
    list(d = data.frame(y = y, x = x, gid = gid), K = K, q = q)
  } else if (grepl("^usK?[0-9]+$", cas)) {
    tt <- n_traits(cas); n1 <- n %/% tt; q <- max(2L, round(n1 / REPS))
    gid <- factor(rep(paste0("g", 1:q), length.out = n1), levels = paste0("g", 1:q))
    K <- if (grepl("^usK", cas)) grm(q, n) else NULL
    G <- crossprod(matrix(rnorm(tt * tt), tt)) / tt + diag(0.5, tt)
    Rm <- crossprod(matrix(rnorm(tt * tt), tt)) / tt + diag(0.5, tt)
    Uw <- matrix(rnorm(q * tt), q, tt)
    if (!is.null(K)) Uw <- t(chol(K)) %*% Uw
    U <- Uw %*% chol(G)
    Y <- U[as.integer(gid), ] + matrix(rnorm(n1 * tt), n1, tt) %*% chol(Rm) + matrix(seq_len(tt), n1, tt, byrow = TRUE)
    colnames(Y) <- paste0("y", seq_len(tt))
    dw <- data.frame(gid = gid, Y)
    dl <- data.frame(y = c(Y), gid = rep(gid, tt), trait = factor(rep(colnames(Y), each = n1), levels = colnames(Y)),
                     unit = rep(seq_len(n1), tt))
    list(d = dw, dl = dl, t = tt, K = K, q = q)
  } else if (cas == "ar1ar1") {
    nr <- round(sqrt(n / 1.25)); nc <- n %/% nr; ntot <- nr * nc; q <- max(2L, round(ntot / REPS))
    g <- expand.grid(col = seq_len(nc), row = seq_len(nr))
    Kr <- 0.6 ^ abs(outer(1:nr, 1:nr, "-")); Kc <- 0.4 ^ abs(outer(1:nc, 1:nc, "-"))
    e <- as.numeric(kronecker(t(chol(Kr)), t(chol(Kc))) %*% rnorm(ntot)) * 1.0
    gid <- factor(sample(rep(paste0("g", 1:q), length.out = ntot)), levels = paste0("g", 1:q))
    y <- 4 + rnorm(q, 0, 1.0)[as.integer(gid)] + e + rnorm(ntot, 0, 0.5)
    d <- data.frame(g, gid = gid, y = y); d$row <- factor(d$row); d$col <- factor(d$col)
    d <- d[order(d$row, d$col), ]
    list(d = d, dims = c(nr, nc), q = q)
  } else if (cas == "ige") {
    # Grille NR x NC, deux especes alternant par colonne : A (focale) sur les
    # colonnes impaires, B sur les paires. n parcelles A. Voisinage d'ordre 1
    # (fenetre de Chebyshev), poids 1/d, pas de normalisation.
    NC <- 2L * round(sqrt(n / 2 / 1.25)); NR <- ceiling(2 * n / NC)
    g <- expand.grid(col = seq_len(NC), row = seq_len(NR)); g$id <- seq_len(nrow(g))
    g$esp <- ifelse(g$col %% 2L == 1L, "A", "B")
    QA <- max(2L, round(sum(g$esp == "A") / REPS)); QB <- max(2L, round(sum(g$esp == "B") / REPS))
    gA <- paste0("a", seq_len(QA)); gB <- paste0("b", seq_len(QB))
    g$geno <- NA_character_
    g$geno[g$esp == "A"] <- sample(rep_len(gA, sum(g$esp == "A")))
    g$geno[g$esp == "B"] <- sample(rep_len(gB, sum(g$esp == "B")))
    iA <- which(g$esp == "A"); nA <- length(iA)
    idx <- matrix(NA_integer_, NR, NC); idx[cbind(g$row, g$col)] <- g$id
    mk_Z <- function(voisin_esp, glev) {
      ii <- integer(0); jj <- integer(0); xx <- numeric(0)
      for (dr in -1:1) for (dc in -1:1) {
        if (dr == 0 && dc == 0) next
        r2 <- g$row[iA] + dr; c2 <- g$col[iA] + dc
        ok <- r2 >= 1 & r2 <= NR & c2 >= 1 & c2 <= NC
        j <- rep(NA_integer_, nA); j[ok] <- idx[cbind(r2[ok], c2[ok])]
        ok <- ok & !is.na(j) & g$esp[ifelse(is.na(j), 1L, j)] == voisin_esp
        ii <- c(ii, which(ok)); jj <- c(jj, match(g$geno[j[ok]], glev)); xx <- c(xx, rep(1 / sqrt(dr^2 + dc^2), sum(ok)))
      }
      Matrix::sparseMatrix(i = ii, j = jj, x = xx, dims = c(nA, length(glev)), dimnames = list(NULL, glev))
    }
    ZgA <- Matrix::sparseMatrix(i = seq_len(nA), j = match(g$geno[iA], gA), x = 1, dims = c(nA, QA), dimnames = list(NULL, gA))
    ZnA <- mk_Z("A", gA); ZxA <- mk_Z("B", gB)
    SIG <- matrix(c(0.90, -0.35, -0.35, 0.30), 2, 2); V_INTER <- 0.22; V_AR1 <- 0.45; RHO_R <- 0.65; RHO_C <- 0.40; V_E <- 0.50
    U <- matrix(rnorm(QA * 2), QA, 2) %*% chol(SIG); uX <- rnorm(QB, 0, sqrt(V_INTER))
    col_obs <- as.integer(factor(g$col[iA])); NC_OBS <- max(col_obs)
    Kr <- RHO_R ^ abs(outer(1:NR, 1:NR, "-")); Kc <- RHO_C ^ abs(outer(1:NC_OBS, 1:NC_OBS, "-"))
    champ <- as.numeric(kronecker(t(chol(Kr)), t(chol(Kc))) %*% rnorm(NR * NC_OBS)) * sqrt(V_AR1)
    cellA_obs <- (g$row[iA] - 1) * NC_OBS + col_obs
    y <- 3 + as.numeric(ZgA %*% U[, 1]) + as.numeric(ZnA %*% U[, 2]) + as.numeric(ZxA %*% uX) + champ[cellA_obs] + rnorm(nA, 0, sqrt(V_E))
    d <- data.frame(y = y, geno = factor(g$geno[iA], levels = gA), row = factor(g$row[iA]), col = factor(col_obs))
    list(d = d, ZgA = ZgA, ZnA = ZnA, ZxA = ZxA, gA = gA, gB = gB, cell = cellA_obs, dims = c(NR, NC_OBS), QA = QA, QB = QB, q = QA)
  } else stop("cas inconnu ", cas)
}

# ---- un ajustement par logiciel ---------------------------------------------------
fit_remlax <- function(cas, D) {
  ctl <- list(backend = BK, verbose = FALSE, hessian = FALSE, blups = FALSE)
  if (cas == "iid")  f <- do.call(rx_reml, c(list(y ~ x, random = ~ gid, data = D$d), ctl))
  if (cas == "grm")  f <- do.call(rx_reml, c(list(y ~ x, random = ~ vm(gid, K = D$K), data = D$d), ctl))
  if (grepl("^us[0-9]+$", cas))
    f <- do.call(rx_reml, c(list(y ~ 0 + trait, random = ~ us(gid), residual = ~ us(trait):unit, data = D$dl, trait = "trait", unit = "unit"), ctl))
  if (grepl("^usK[0-9]+$", cas))
    f <- do.call(rx_reml, c(list(y ~ 0 + trait, random = ~ us(gid, K = D$K), residual = ~ us(trait):unit, data = D$dl, trait = "trait", unit = "unit"), ctl))
  if (cas == "ar1ar1") f <- do.call(rx_reml, c(list(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = D$d), ctl))
  if (cas == "ige") {
    nA <- nrow(D$d)
    mod <- rx_model(y = D$d$y, X = matrix(1, nA, 1),
                    terms = list(rx_term("dge_ige", list(D$ZgA, D$ZnA), struct = "us", levels = D$gA),
                                 rx_term("ige_inter", list(D$ZxA), struct = "iid", levels = D$gB),
                                 rx_term("champ", Matrix::sparseMatrix(i = seq_len(nA), j = D$cell, x = 1, dims = c(nA, prod(D$dims))),
                                         t = 1L, struct = "iid", level = "ar1ar1", dims = D$dims)),
                    residual = rx_residual("iid"))
    f <- do.call(rx_fit, c(list(mod), ctl))
  }
  comp <- c(unlist(lapply(f$sigmas, function(S) S[lower.tri(S, diag = TRUE)])), f$sigma_res[lower.tri(f$sigma_res, diag = TRUE)], unlist(f$rho))
  list(logLik = f$logLik_asreml, comp = comp, n_par = f$n_par, solver_s = f[["secondes"]], compile_s = f[["compile_s"]],
       n_eval = f[["n_eval"]], n_iter = f[["n_iter"]], eval_s = f[["eval_s"]],
       converged = isTRUE(as.logical(f$scipy_success)), version = versions[["remlax"]])
}
fit_asreml_brut <- function(cas, D) {
  d <- D$d
  if (cas == "iid") a <- asreml(y ~ x, random = ~ gid, data = d, maxit = MAXIT_REF)
  if (cas == "grm") { K <- D$K; a <- asreml(y ~ x, random = ~ vm(gid, K), data = d, maxit = MAXIT_REF) }
  if (grepl("^usK?[0-9]+$", cas)) {
    fml <- stats::as.formula(paste("cbind(", paste(grep("^y", names(d), value = TRUE), collapse = ","), ") ~ trait"))
    if (grepl("^usK", cas)) { K <- D$K; a <- asreml(fml, random = ~ us(trait):vm(gid, K), residual = ~ id(units):us(trait), data = d, maxit = MAXIT_REF) }
    else a <- asreml(fml, random = ~ us(trait):gid, residual = ~ id(units):us(trait), data = d, maxit = MAXIT_REF)
  }
  if (cas == "ar1ar1") a <- asreml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d, maxit = MAXIT_REF)
  if (cas == "ige") {
    da <- cbind(d, as.data.frame(as.matrix(D$ZnA)), as.data.frame(as.matrix(D$ZxA)))
    names(da) <- c(names(d), paste0("N", seq_len(D$QA)), paste0("X", seq_len(D$QB)))
    # asreml evalue `group` par eval(call$group) hors de cette fonction : les
    # indices doivent etre visibles depuis l'environnement global.
    assign("iN", grep("^N[0-9]+$", names(da)), envir = globalenv())
    assign("iX", grep("^X[0-9]+$", names(da)), envir = globalenv())
    a <- asreml(y ~ 1, random = stats::as.formula(sprintf("~ str(~ geno + grp(nb), ~ us(2):id(%d)) + grp(xb) + ar1(row):ar1(col)", D$QA)),
                group = list(nb = iN, xb = iX), data = da, maxit = MAXIT_REF)
  }
  vc <- summary(a)$varcomp
  it <- tryCatch({ m <- a$monitor; if (is.matrix(m)) ncol(m) - 1L else if (is.matrix(a$trace)) nrow(a$trace) else NA_integer_ }, error = function(e) NA_integer_)
  list(logLik = a$loglik, comp = vc$component, n_par = nrow(vc), n_eval = it, n_iter = it,
       converged = isTRUE(a$converge), version = versions[["asreml"]])
}
fit_asreml <- function(cas, D) {
  # fork + delai : asreml ne s'interrompt pas par setTimeLimit
  if (!is.finite(TIMEOUT) || TIMEOUT <= 0) return(fit_asreml_brut(cas, D))
  p <- parallel::mcparallel(suppressWarnings(fit_asreml_brut(cas, D)))
  r <- parallel::mccollect(p, wait = FALSE, timeout = TIMEOUT)
  if (is.null(r)) { tools::pskill(p$pid, tools::SIGKILL); try(parallel::mccollect(p, wait = FALSE), silent = TRUE); stop("TIMEOUT") }
  r <- r[[1]]
  if (inherits(r, "try-error")) stop(paste(as.character(r), collapse = " "))
  if (inherits(r, "error")) stop(conditionMessage(r))
  r
}
FITS <- list(remlax = fit_remlax, asreml = fit_asreml)

# ---- boucle -----------------------------------------------------------------------
for (n in SIZES) for (cas in CASES) {
  D <- tryCatch(gen_case(cas, n), error = function(e) e)
  if (inherits(D, "error")) { ligne(date = DATE, tag = TAG, backend = BK, case = cas, n = n, software = "generation", status = "error", error = conditionMessage(D)); next }
  n_obs <- if (is.null(D$dl)) nrow(D$d) else nrow(D$dl)
  cat(sprintf("\n### %s n=%d (%d observations) ###\n", cas, n, n_obs)); flush.console()
  for (sw in SOFT) {
    # Le serveur de licences asreml est partage (« All licenses in use ») : on
    # attend et on reessaie, le chrono repartant a zero a chaque essai.
    for (essai in 1:30) {
      t0 <- proc.time()[["elapsed"]]
      r <- tryCatch(chrono(suppressWarnings(FITS[[sw]](cas, D))), error = function(e) e)
      if (!(inherits(r, "error") && grepl("icense", conditionMessage(r)))) break
      cat("  [licence asreml occupee, nouvel essai dans 60 s]\n"); Sys.sleep(60)
    }
    bk <- if (sw == "remlax") BK else "cpu"
    if (inherits(r, "error")) {
      msg <- conditionMessage(r); ecoule <- proc.time()[["elapsed"]] - t0
      # remlax tue par `timeout` : le solveur echoue apres ~TIMEOUT secondes
      to <- grepl("TIMEOUT", msg) || (is.finite(TIMEOUT) && TIMEOUT > 0 && ecoule >= 0.97 * TIMEOUT)
      ligne(date = DATE, tag = TAG, backend = bk, case = cas, n = n_obs, reps = REPS, q = D$q, software = sw, version = versions[[sw]],
            status = if (to) "timeout" else "error", wall_s = round(ecoule, 1),
            error = substr(gsub("[\r\n|]+", " ", msg), 1, 300))
      next
    }
    v <- r$v
    solver <- if (is.null(v$solver_s)) NA else v$solver_s
    compile <- if (is.null(v$compile_s)) NA else v$compile_s
    n_it <- if (length(v$n_iter) == 1L && is.finite(v$n_iter) && v$n_iter > 0) v$n_iter else NA
    n_ev <- if (length(v$n_eval) == 1L && is.finite(v$n_eval) && v$n_eval > 0) v$n_eval else NA
    base <- if (sw == "remlax") solver else r$s
    ligne(date = DATE, tag = TAG, backend = bk, case = cas, n = n_obs, reps = REPS, q = D$q, n_par = v$n_par, software = sw, version = v$version, status = "ok",
          wall_s = round(r$s, 3),
          startup_s = if (sw == "remlax") round(r$s - compile - solver, 3) else NA,
          compile_s = if (is.na(compile)) NA else round(compile, 3),
          solver_s = if (sw == "remlax") round(solver, 3) else round(r$s, 3),
          n_iter = n_it, n_eval = n_ev,
          s_per_iter = if (is.na(n_it)) NA else round(base / n_it, 4),
          s_per_eval = if (is.na(n_ev)) NA else round(base / n_ev, 4),
          eval_s = if (is.null(v$eval_s)) NA else round(v$eval_s, 4),
          converged = v$converged, logLik = sprintf("%.6f", v$logLik), components = comp_str(v$comp), error = "")
  }
}
cat("\necrit :", FICHIER, "(", length(RES), "lignes )\n")
