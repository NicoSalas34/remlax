# =============================================================================
# verif_ecarts_asreml.R — les cas du banc ou remlax rend une logLik plus haute
#                          qu'asreml de plus de 1e-3 : arret d'asreml, ou
#                          fonction differente ?
# -----------------------------------------------------------------------------
# Pour chaque cas (memes donnees que bench_complexite.R, memes graines) :
#   1. asreml avec maxit = 100 (l'appel du banc), puis update() repete jusqu'a
#      ce que la logLik bouge de moins de 1e-6 ou 40 updates ;
#   2. remlax (backend RX_BACKEND) ;
#   3. la trajectoire de logLik d'asreml et l'ecart final a remlax.
# Si asreml rejoint remlax en continuant d'iterer, l'ecart du banc est son
# critere d'arret (variation relative de logLik < 0.002 x iteration et
# composantes < 1 %), pas une difference de vraisemblance.
#   Rscript benchmarks/verif_ecarts_asreml.R [--cases us9:500,us12:500,us12:2000,ige:500,usK6:2000:40]
# =============================================================================
suppressPackageStartupMessages({ library(here); library(Matrix); library(jsonlite); library(asreml) })
asreml.options(trace = FALSE, workspace = "8gb", pworkspace = "2gb")
src <- readLines(here::here("benchmarks", "bench_complexite.R"))
# on reprend les definitions du banc (generateurs, ajustements) sans sa boucle
i0 <- grep("^# ---- boucle", src)
args_sauve <- commandArgs(trailingOnly = TRUE)
commandArgs <- function(trailingOnly = TRUE) c("--software", "asreml,remlax", "--timeout-fit", "0")
eval(parse(text = src[seq_len(i0 - 1)]))
opt2 <- function(nom, defaut) { i <- match(paste0("--", nom), args_sauve); if (is.na(i) || i == length(args_sauve)) defaut else args_sauve[i + 1L] }
# Le 3e champ est le nombre de repetitions par genotype. L'axe taille du banc a
# pose q = n1 %/% 2 pour les cas multi-caracteres (us9 a n = 500 : n1 = 55,
# q = 27) ; round(n1 / reps) du generateur actuel le reproduit avec reps = n1 / q.
CAS <- strsplit(opt2("cases", "us9:500:2.037037,us12:500:2.05,us12:2000:2,ige:500:4,usK6:2000:40"), ",")[[1]]
OUT <- opt2("out", here::here("benchmarks", "results"))
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
res <- list()
for (spec in CAS) tryCatch({
  p <- strsplit(spec, ":")[[1]]; cas <- p[1]; n <- as.integer(p[2]); REPS <<- if (length(p) > 2) as.numeric(p[3]) else 4
  D <- gen_case(cas, n)
  cat(sprintf("\n=== %s n=%d reps=%g q=%d ===\n", cas, n, REPS, D$q))
  # remlax
  t0 <- proc.time()[["elapsed"]]; f <- fit_remlax(cas, D); t_rx <- proc.time()[["elapsed"]] - t0
  cat(sprintf("remlax : logLik %.6f | %d iterations | %d evaluations | %.1f s\n", f$logLik, f$n_iter, f$n_eval, t_rx))
  # asreml : appel du banc puis updates
  d <- D$d
  obj <- NULL
  if (grepl("^us[0-9]+$", cas)) {
    fml <- stats::as.formula(paste("cbind(", paste(grep("^y", names(d), value = TRUE), collapse = ","), ") ~ trait"))
    obj <- asreml(fml, random = ~ us(trait):gid, residual = ~ id(units):us(trait), data = d, maxit = MAXIT_REF)
  } else if (grepl("^usK[0-9]+$", cas)) {
    fml <- stats::as.formula(paste("cbind(", paste(grep("^y", names(d), value = TRUE), collapse = ","), ") ~ trait"))
    K <- D$K; obj <- asreml(fml, random = ~ us(trait):vm(gid, K), residual = ~ id(units):us(trait), data = d, maxit = MAXIT_REF)
  } else if (cas == "ige") {
    da <- cbind(d, as.data.frame(as.matrix(D$ZnA)), as.data.frame(as.matrix(D$ZxA)))
    names(da) <- c(names(d), paste0("N", seq_len(D$QA)), paste0("X", seq_len(D$QB)))
    assign("iN", grep("^N[0-9]+$", names(da)), envir = globalenv()); assign("iX", grep("^X[0-9]+$", names(da)), envir = globalenv())
    obj <- asreml(y ~ 1, random = stats::as.formula(sprintf("~ str(~ geno + grp(nb), ~ us(2):id(%d)) + grp(xb) + ar1(row):ar1(col)", D$QA)),
                  group = list(nb = iN, xb = iX), data = da, maxit = MAXIT_REF)
  } else stop("cas non prevu ici : ", cas)
  traj <- obj$loglik; conv <- obj$converge
  cat(sprintf("asreml, appel du banc (maxit %d) : logLik %.6f | converge %s | ecart a remlax %+.4e\n", MAXIT_REF, obj$loglik, conv, obj$loglik - f$logLik))
  for (k in 1:40) {
    prev <- obj$loglik
    obj <- suppressWarnings(update(obj, trace = FALSE, maxit = MAXIT_REF))
    traj <- c(traj, obj$loglik)
    if (abs(obj$loglik - prev) < 1e-6) break
  }
  cat(sprintf("asreml, apres %d update(s) : logLik %.6f | converge %s | ecart a remlax %+.4e\n", length(traj) - 1, obj$loglik, obj$converge, obj$loglik - f$logLik))
  cat("  trajectoire :", paste(sprintf("%.4f", traj), collapse = " "), "\n")
  res[[spec]] <- list(case = cas, n = n, reps = REPS, remlax_logLik = f$logLik, remlax_backend = BK,
                      asreml_logLik_bench = traj[1], asreml_converge_bench = conv,
                      asreml_logLik_updates = obj$loglik, n_updates = length(traj) - 1, asreml_converge_final = obj$converge,
                      gap_bench = traj[1] - f$logLik, gap_final = obj$loglik - f$logLik, trajectory = traj)
  write_json(res, file.path(OUT, "verif_ecarts_asreml.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
}, error = function(e) {
  cat("ECHEC", spec, ":", conditionMessage(e), "\n")
  res[[spec]] <<- list(spec = spec, error = conditionMessage(e))
  write_json(res, file.path(OUT, "verif_ecarts_asreml.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
})
cat("\necrit :", file.path(OUT, "verif_ecarts_asreml.json"), "\n")