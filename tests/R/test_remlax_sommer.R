# ==============================================================================
# test_remlax_sommer.R — matrices de parente et multi-caractere contre sommer
# ------------------------------------------------------------------------------
# sommer (Covarrubias-Pazaran) est la reference ouverte pour les modeles
# genomiques : matrice de parente sur un terme, plusieurs caracteres avec
# covariance us ou diag entre caracteres, plusieurs matrices de parente sur
# le meme facteur (additif + dominance), multi-environnement. test_remlax.R
# ne couvrait que le cas univarie a une GRM et le cas bivarie us sans GRM.
#
# CONVERGENCE DE sommer. A ses reglages par defaut (tolParConvLL = 1e-3),
# sommer s'arrete a 1e-4 pres des composantes : les ecarts mesures seraient
# ceux de son critere d'arret. On le resserre (tolParConvLL = 1e-12, nIters
# larges), apres quoi les composantes concordent a 1e-6 relatif.
#
# CONSTANTE DE logLik. La vraisemblance rendue par sommer (monitor[1, ]) n'est
# ni celle de remlax ni celle d'asreml : elle depend d'une mise a l'echelle
# interne de la reponse. Seules les DIFFERENCES de -2logL entre deux modeles
# ajustes par le meme logiciel sont comparables, et c'est ce qui est compare
# (section 3, us contre diag). L'ecart de constante est mesure et imprime,
# jamais corrige a la main.
#
#   Rscript tests/R/test_remlax_sommer.R
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite); library(sommer)
})
source(here::here("R", "remlax.R"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
cat("sommer", as.character(packageVersion("sommer")), "| R", R.version$major, R.version$minor, "\n")

ECHECS <- character(0)
verifier <- function(nom, cond, detail = "") {
  cat(sprintf("  %-56s %s %s\n", nom, if (isTRUE(cond)) "OK " else "ECHEC", detail))
  if (!isTRUE(cond)) ECHECS <<- c(ECHECS, nom)
}
rel <- function(a, b) abs(a - b) / pmax(abs(b), 1e-8)
relmax <- function(A, B) max(abs(as.numeric(A) - as.numeric(B))) / max(abs(as.numeric(B)))
section <- function(titre, expr) {
  cat("\n=== ", titre, " ===\n", sep = "")
  tryCatch(expr, error = function(e) verifier(paste0(sub(" .*", "", titre), " : execution"),
                                              FALSE, conditionMessage(e)))
}
MMER <- function(...) suppressWarnings(mmer(..., verbose = FALSE, tolParInv = 1e-8,
                                            tolParConvLL = 1e-12, nIters = 300))
llik_so <- function(so) so$monitor[1, ncol(so$monitor)]
grm <- function(q, nm, seed) {
  set.seed(seed)
  M <- matrix(rbinom(q * nm, 2, 0.3), q, nm)
  K <- A.mat(M - 1) + diag(1e-4, q)
  dimnames(K) <- list(paste0("g", 1:q), paste0("g", 1:q))
  list(K = K, M = M)
}
long <- function(Y, gid) {
  n1 <- nrow(Y)
  data.frame(y = c(Y), gid = rep(gid, ncol(Y)),
             trait = factor(rep(colnames(Y), each = n1), levels = colnames(Y)),
             unit = rep(seq_len(n1), ncol(Y)))
}

# ==============================================================================
section("1. une GRM, univarie : composantes, effets fixes, BLUP, PEV", {
# ==============================================================================
  q <- 50; rep_ <- 4; n <- q * rep_
  K <- grm(q, 300, 61)$K
  set.seed(62)
  gid <- factor(rep(paste0("g", 1:q), each = rep_), levels = paste0("g", 1:q))
  x <- rnorm(n)
  u <- as.numeric(t(chol(K)) %*% rnorm(q)) * 1.2
  y <- 5 + 0.4 * x + u[as.integer(gid)] + rnorm(n, 0, 1.0)
  d <- data.frame(y = y, x = x, gid = gid)
  so <- MMER(y ~ x, random = ~ vsr(gid, Gu = K), rcov = ~ units, data = d)
  fit <- rx_reml(y ~ x, random = ~ vm(gid, K = K), data = d, backend = BK, verbose = FALSE, pev = TRUE)
  s_g <- so$sigma[[1]][1, 1]; s_e <- so$sigma[[2]][1, 1]
  verifier("GRM : composantes de variance", rel(fit$sigmas$gid[1, 1], s_g) < 1e-4 && rel(fit$sigma_res[1, 1], s_e) < 1e-4,
           sprintf("s2g %.6f/%.6f | s2e %.6f/%.6f", fit$sigmas$gid[1, 1], s_g, fit$sigma_res[1, 1], s_e))
  verifier("GRM : effets fixes", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate)))))
  verifier("GRM : erreurs-types des effets fixes",
           max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(as.matrix(so$VarBeta))))) < 1e-4,
           sprintf("ecart rel max %.2e", max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(as.matrix(so$VarBeta)))))))
  b_so <- so$U[[1]][[1]][levels(gid)]
  verifier("GRM : BLUPs", max(abs(as.numeric(fit$blups$gid) - b_so)) < 1e-4,
           sprintf("ecart max %.2e", max(abs(as.numeric(fit$blups$gid) - b_so))))
  pev_so <- diag(as.matrix(so$PevU[[1]][[1]]))[levels(gid)]
  verifier("GRM : PEV des BLUPs", max(rel(as.numeric(fit$pev$gid), pev_so)) < 1e-4,
           sprintf("ecart rel max %.2e | PEV moyenne %.5f", max(rel(as.numeric(fit$pev$gid), pev_so)), mean(pev_so)))
  # Constante : mesuree, pas corrigee. sommer rend une vraisemblance decalee
  # d'une constante qui depend des donnees (mise a l'echelle de la reponse).
  off <- llik_so(so) - fit$logLik_asreml
  verifier("GRM : constante de logLik de sommer (mesuree)", is.finite(off),
           sprintf("sommer %.4f | remlax logLik_asreml %.4f | ecart %.4f ((n-p)/2 log var(y) = %.4f)",
                   llik_so(so), fit$logLik_asreml, off, (n - 2) / 2 * log(var(y))))
})

# ==============================================================================
section("2. trois caracteres, us genetique avec GRM et us residuelle", {
# ==============================================================================
  q <- 60; rep_ <- 3; n1 <- q * rep_
  K <- grm(q, 400, 51)$K
  set.seed(52)
  G <- matrix(c(1.5, 0.8, 0.3, 0.8, 1.0, 0.4, 0.3, 0.4, 0.8), 3)
  Rm <- matrix(c(0.8, 0.2, 0.1, 0.2, 1.2, 0.3, 0.1, 0.3, 0.6), 3)
  U <- t(chol(K)) %*% matrix(rnorm(q * 3), q, 3) %*% chol(G)
  gid <- factor(rep(paste0("g", 1:q), each = rep_), levels = paste0("g", 1:q))
  Y <- U[as.integer(gid), ] + matrix(rnorm(n1 * 3), n1, 3) %*% chol(Rm) +
       matrix(c(3, 1, 2), n1, 3, byrow = TRUE)
  colnames(Y) <- c("y1", "y2", "y3")
  dw <- data.frame(gid = gid, Y)
  so <- MMER(cbind(y1, y2, y3) ~ 1, random = ~ vsr(gid, Gu = K, Gtc = unsm(3)),
             rcov = ~ vsr(units, Gtc = unsm(3)), data = dw)
  dl <- long(Y, gid)
  fit <- rx_reml(y ~ 0 + trait, random = ~ us(gid, K = K), residual = ~ us(trait):unit,
                 data = dl, trait = "trait", unit = "unit", backend = BK, verbose = FALSE)
  verifier("us3 + GRM : les 6 composantes genetiques", relmax(fit$sigmas$gid, so$sigma[[1]]) < 1e-4,
           sprintf("ecart rel max %.2e | diag remlax %s | sommer %s", relmax(fit$sigmas$gid, so$sigma[[1]]),
                   paste(round(diag(fit$sigmas$gid), 4), collapse = " "), paste(round(diag(so$sigma[[1]]), 4), collapse = " ")))
  verifier("us3 + GRM : les 6 composantes residuelles", relmax(fit$sigma_res, so$sigma[[2]]) < 1e-4,
           sprintf("ecart rel max %.2e", relmax(fit$sigma_res, so$sigma[[2]])))
  rg <- cov2cor(fit$sigmas$gid); rs <- cov2cor(so$sigma[[1]])
  verifier("us3 + GRM : les 3 correlations genetiques", max(abs(rg[lower.tri(rg)] - rs[lower.tri(rs)])) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(rg[lower.tri(rg)], 4), collapse = " "),
                   paste(round(rs[lower.tri(rs)], 4), collapse = " ")))
  verifier("us3 + GRM : effets fixes (3 moyennes)", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate)))))
  b_so <- sapply(c("y1", "y2", "y3"), function(tr) so$U[[1]][[tr]][levels(gid)])
  verifier("us3 + GRM : BLUPs des 3 caracteres", max(abs(as.numeric(fit$blups$gid) - as.numeric(b_so))) < 1e-4,
           sprintf("ecart max %.2e sur %d valeurs", max(abs(as.numeric(fit$blups$gid) - as.numeric(b_so))), length(b_so)))
  # Le meme dispositif en diag (section 3) : on garde -2logL pour la difference.
  assign("LL_US", list(rx = -2 * fit$logLik, so = -2 * llik_so(so)), envir = globalenv())
  assign("DW", dw, envir = globalenv()); assign("DL", dl, envir = globalenv()); assign("K3", K, envir = globalenv())
})

# ==============================================================================
section("3. trois caracteres, diag genetique avec GRM et diag residuelle ; LRT us vs diag", {
# ==============================================================================
  so <- MMER(cbind(y1, y2, y3) ~ 1, random = ~ vsr(gid, Gu = K3, Gtc = diag(3)),
             rcov = ~ vsr(units, Gtc = diag(3)), data = DW)
  fit <- rx_reml(y ~ 0 + trait, random = ~ diag(gid, K = K3), residual = ~ diag(trait):unit,
                 data = DL, trait = "trait", unit = "unit", backend = BK, verbose = FALSE)
  verifier("diag3 + GRM : les 3 variances genetiques", max(rel(diag(fit$sigmas$gid), diag(so$sigma[[1]]))) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(diag(fit$sigmas$gid), 4), collapse = " "),
                   paste(round(diag(so$sigma[[1]]), 4), collapse = " ")))
  verifier("diag3 + GRM : les 3 variances residuelles", max(rel(diag(fit$sigma_res), diag(so$sigma[[2]]))) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(diag(fit$sigma_res), 4), collapse = " "),
                   paste(round(diag(so$sigma[[2]]), 4), collapse = " ")))
  d_rx <- -2 * fit$logLik - LL_US$rx; d_so <- -2 * llik_so(so) - LL_US$so
  verifier("LRT us vs diag : meme difference de -2logL", abs(d_rx - d_so) < 1e-3,
           sprintf("remlax %.5f | sommer %.5f (6 ddl)", d_rx, d_so))
})

# ==============================================================================
section("4. additif + dominance : deux matrices de parente sur le meme facteur", {
# ==============================================================================
  q <- 80; rep_ <- 3; n <- q * rep_
  g0 <- grm(q, 500, 71); K <- g0$K
  D <- D.mat(g0$M - 1) + diag(1e-4, q); dimnames(D) <- dimnames(K)
  set.seed(72)
  gid <- factor(rep(paste0("g", 1:q), each = rep_), levels = paste0("g", 1:q))
  ua <- as.numeric(t(chol(K)) %*% rnorm(q)) * 1.0
  ud <- as.numeric(t(chol(D)) %*% rnorm(q)) * 0.7
  y <- 10 + ua[as.integer(gid)] + ud[as.integer(gid)] + rnorm(n, 0, 1.0)
  d <- data.frame(y = y, gid = gid, gid2 = gid)
  so <- MMER(y ~ 1, random = ~ vsr(gid, Gu = K) + vsr(gid2, Gu = D), rcov = ~ units, data = d)
  fit <- rx_reml(y ~ 1, random = ~ vm(gid, K = K) + vm(gid2, K = D), data = d, backend = BK, verbose = FALSE)
  v_rx <- c(fit$sigmas$gid[1, 1], fit$sigmas$gid2[1, 1], fit$sigma_res[1, 1])
  v_so <- c(so$sigma[[1]][1, 1], so$sigma[[2]][1, 1], so$sigma[[3]][1, 1])
  verifier("A + D : les trois composantes", max(rel(v_rx, v_so)) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(v_rx, 5), collapse = " "), paste(round(v_so, 5), collapse = " ")))
  b_rx <- c(as.numeric(fit$blups$gid), as.numeric(fit$blups$gid2))
  b_so <- c(so$U[[1]][[1]][levels(gid)], so$U[[2]][[1]][levels(gid)])
  verifier("A + D : BLUPs additifs et de dominance", max(abs(b_rx - b_so)) < 1e-4,
           sprintf("ecart max %.2e", max(abs(b_rx - b_so))))
})

# ==============================================================================
section("5. multi-environnement : une variance genetique par milieu avec GRM", {
# ==============================================================================
  q <- 50; ne <- 4
  K <- grm(q, 300, 81)$K
  set.seed(82)
  s_e <- c(0.6, 1.0, 1.4, 0.8)
  U <- t(chol(K)) %*% matrix(rnorm(q * ne), q, ne) %*% diag(sqrt(s_e))
  d <- expand.grid(gid = paste0("g", 1:q), env = paste0("E", 1:ne), rep = 1:2, stringsAsFactors = TRUE)
  d <- d[sample(nrow(d), 300), ]                       # dispositif incomplet, 2 repetitions
  d$gid <- factor(d$gid, levels = paste0("g", 1:q)); d$env <- factor(d$env)
  d$y <- 2 + as.numeric(d$env) + U[cbind(as.integer(d$gid), as.integer(d$env))] +
         rnorm(nrow(d), 0, c(0.8, 1.0, 1.2, 0.9)[as.integer(d$env)])
  d$unit <- seq_len(nrow(d))
  so <- MMER(y ~ env, random = ~ vsr(dsr(env), gid, Gu = K), rcov = ~ vsr(dsr(env), units), data = d)
  fit <- rx_reml(y ~ env, random = ~ diag(gid, K = K), residual = ~ diag(env):unit,
                 data = d, trait = "env", unit = "unit", backend = BK, verbose = FALSE)
  v_so_g <- sapply(1:ne, function(k) so$sigma[[k]][1, 1]); v_so_e <- sapply(1:ne, function(k) so$sigma[[ne + k]][1, 1])
  verifier("MET : les 4 variances genetiques", relmax(diag(fit$sigmas$gid), v_so_g) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(diag(fit$sigmas$gid), 4), collapse = " "),
                   paste(round(v_so_g, 4), collapse = " ")))
  verifier("MET : les 4 variances residuelles", relmax(diag(fit$sigma_res), v_so_e) < 1e-4,
           sprintf("remlax %s | sommer %s", paste(round(diag(fit$sigma_res), 4), collapse = " "),
                   paste(round(v_so_e, 4), collapse = " ")))
  verifier("MET : effets fixes", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(so$Beta$Estimate)))))
})

# ==============================================================================
cat("\n==============================================================\n")
if (length(ECHECS)) {
  cat("ECHECS (", length(ECHECS), ") :\n", sep = ""); for (e in ECHECS) cat("  -", e, "\n")
  quit(status = 1)
}
cat("TOUS LES CONTROLES PASSENT : memes composantes, effets fixes, BLUP et PEV\n",
    "que sommer sur les modeles genomiques testes.\n")
