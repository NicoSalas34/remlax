# ==============================================================================
# test_remlkit_asreml.R — validation contre ASREML, la reference du domaine
# ------------------------------------------------------------------------------
# Les autres suites confrontent remlkit a lme4 et sommer. Celle-ci le confronte
# a asreml, sur les structures qu'asreml sait faire et que les deux autres ne
# font pas : AR1, AR1 separable, splines 2D, et str() (covariance PARTAGEE entre
# plusieurs termes).
#
# CONVENTION DE LOG-VRAISEMBLANCE. asreml omet la constante (n-p)/2 * log(2*pi),
# remlkit l'inclut (comme lme4). La comparaison porte donc sur `logLik_asreml`,
# champ prevu pour ca. Sans cette precaution l'ecart vaut plusieurs centaines et
# ressemble a un desaccord de modele : il vaut 219,6 sur un jeu a n=240, p=1.
#
# TOLERANCES. logLik a 1e-5 (les deux optimiseurs s'arretent differemment) ;
# composantes a 1e-3 en relatif, SAUF celles qui sont au plancher des deux cotes,
# ou le relatif n'a aucun sens (comparer 3e-11 a 2e-10 donne 0,85).
#
#   Rscript scripts/tests/test_remlkit_asreml.R
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite); library(splines); library(asreml)
})
source(here::here("R", "remlkit.R"))

ECHECS <- character(0)
verifier <- function(nom, cond, detail = "") {
  cat(sprintf("  %-52s %s %s\n", nom, if (isTRUE(cond)) "OK " else "ECHEC", detail))
  if (!isTRUE(cond)) ECHECS <<- c(ECHECS, nom)
}
relc <- function(a, b, plancher = 1e-6) {
  if (max(abs(a), abs(b)) < plancher) return(0)   # deux zeros numeriques
  abs(a - b) / max(abs(b), plancher)
}
stab <- function(a, k = 5) { for (i in seq_len(k)) a <- update(a, trace = FALSE); a }

# ==============================================================================
cat("\n=== 1. AR1 unidimensionnel ===\n")
# ==============================================================================
set.seed(1); q <- 40; rp <- 6; n <- q * rp
K <- 0.75 ^ abs(outer(1:q, 1:q, "-"))
u <- as.numeric(t(chol(K)) %*% rnorm(q)) * 1.4
lev <- rep(1:q, each = rp)
d <- data.frame(y = 2 + u[lev] + rnorm(n, 0, 0.8), lev = factor(lev, levels = 1:q))
d <- d[order(d$lev), ]
a <- stab(asreml(y ~ 1, random = ~ ar1v(lev), data = d, trace = FALSE,
                 maxit = 100, workspace = "2gb"))
vc <- summary(a)$varcomp
f <- rk_reml(y ~ 1, random = ~ ar1(lev), data = d, backend = "auto",
             n_restarts = 5, verbose = FALSE)
verifier("AR1 : rho",  relc(f$rho[[1]][1], vc["lev!lev!cor", "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", f$rho[[1]][1], vc["lev!lev!cor", "component"]))
verifier("AR1 : variance", relc(f$sigmas[[1]][1, 1], vc["lev!lev!var", "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", f$sigmas[[1]][1, 1], vc["lev!lev!var", "component"]))
verifier("AR1 : residuelle", relc(f$sigma_res[1, 1], vc["units!R", "component"]) < 1e-3)
verifier("AR1 : logLik (convention asreml)", abs(f$logLik_asreml - a$loglik) < 1e-5,
         sprintf("%.9f vs %.9f", f$logLik_asreml, a$loglik))

# ==============================================================================
cat("\n=== 2. AR1 x AR1 separable (champ spatial) ===\n")
# ==============================================================================
set.seed(4); nr <- 12; nc <- 10; K2 <- 2
g <- expand.grid(col = 1:nc, row = 1:nr)
Kr <- 0.7 ^ abs(outer(1:nr, 1:nr, "-")); Kc <- 0.45 ^ abs(outer(1:nc, 1:nc, "-"))
u <- as.numeric(kronecker(t(chol(Kr)), t(chol(Kc))) %*% rnorm(nr * nc)) * 1.2
d <- do.call(rbind, replicate(K2, g, simplify = FALSE))
d$y <- 3 + u[(d$row - 1) * nc + d$col] + rnorm(nrow(d), 0, 0.9)
d$row <- factor(d$row); d$col <- factor(d$col); d <- d[order(d$row, d$col), ]
a <- stab(asreml(y ~ 1, random = ~ ar1(row):ar1(col), data = d, trace = FALSE,
                 maxit = 100, workspace = "2gb"))
vc <- summary(a)$varcomp
f <- rk_reml(y ~ 1, random = ~ ar1(row, col), data = d, backend = "auto",
             n_restarts = 5, verbose = FALSE)
verifier("AR1xAR1 : variance du champ",
         relc(f$sigmas[[1]][1, 1], vc["row:col", "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", f$sigmas[[1]][1, 1], vc["row:col", "component"]))
verifier("AR1xAR1 : rho ligne",
         relc(f$rho[[1]][1], vc["row:col!row!cor", "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", f$rho[[1]][1], vc["row:col!row!cor", "component"]))
verifier("AR1xAR1 : rho colonne",
         relc(f$rho[[1]][2], vc["row:col!col!cor", "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", f$rho[[1]][2], vc["row:col!col!cor", "component"]))
verifier("AR1xAR1 : residuelle", relc(f$sigma_res[1, 1], vc["units!R", "component"]) < 1e-3)
verifier("AR1xAR1 : logLik", abs(f$logLik_asreml - a$loglik) < 1e-5,
         sprintf("%.9f vs %.9f", f$logLik_asreml, a$loglik))

# ==============================================================================
cat("\n=== 3. spline 2D (MEMES matrices de base des deux cotes) ===\n")
# ==============================================================================
set.seed(6); nr <- 20; nc <- 18
g <- expand.grid(col = 1:nc, row = 1:nr)
surf <- 2 * sin(g$row / 4) + 1.5 * cos(g$col / 3) + 0.02 * g$row * g$col
y <- 5 + surf + rnorm(nrow(g), 0, 0.6)
sp <- rk_spl2d(g$row, g$col, nseg = c(6, 6))
Zs <- lapply(sp$terms, function(t) as.matrix(t$Zl[[1]]))
dd <- data.frame(y = y, sp$X, Zs[[1]], Zs[[2]], Zs[[3]])
colnames(dd) <- c("y", paste0("L", seq_len(ncol(sp$X))),
                  paste0("A", seq_len(ncol(Zs[[1]]))), paste0("B", seq_len(ncol(Zs[[2]]))),
                  paste0("C", seq_len(ncol(Zs[[3]]))))
a <- stab(asreml(stats::as.formula(paste("y ~", paste(paste0("L", seq_len(ncol(sp$X))), collapse = " + "))),
                 random = ~ grp(gA) + grp(gB) + grp(gC),
                 group = list(gA = grep("^A", names(dd)), gB = grep("^B", names(dd)),
                              gC = grep("^C", names(dd))),
                 data = dd, trace = FALSE, maxit = 100, workspace = "2gb"))
vc <- summary(a)$varcomp
f <- rk_fit(rk_model(y, cbind(1, sp$X), sp$terms), backend = "auto",
            n_restarts = 3, verbose = FALSE)
for (i in seq_along(sp$terms)) {
  nm <- sp$terms[[i]]$name; ref <- vc[c("grp(gA)", "grp(gB)", "grp(gC)")[i], "component"]
  verifier(sprintf("spline 2D : variance %s", nm),
           relc(f$sigmas[[nm]][1, 1], ref) < 1e-3,
           sprintf("%.6e vs %.6e", f$sigmas[[nm]][1, 1], ref))
}
verifier("spline 2D : residuelle", relc(f$sigma_res[1, 1], vc["units!R", "component"]) < 1e-3)
verifier("spline 2D : logLik", abs(f$logLik_asreml - a$loglik) < 1e-5,
         sprintf("%.9f vs %.9f", f$logLik_asreml, a$loglik))
verifier("spline 2D : composante nulle signalee degeneree",
         "spl_xy" %in% f$composantes_degenerees,
         paste(f$composantes_degenerees, collapse = ", "))

# ==============================================================================
cat("\n=== 4. str() : covariance PARTAGEE entre deux termes ===\n")
# ==============================================================================
set.seed(12); q <- 40; rp <- 6; n <- q * rp
S <- matrix(c(1.4, 0.7, 0.7, 0.9), 2, 2)
U <- matrix(rnorm(q * 2), q, 2) %*% chol(S)
gid <- factor(rep(paste0("g", 1:q), each = rp), levels = paste0("g", 1:q))
w <- rnorm(n)
d <- data.frame(y = 3 + U[as.integer(gid), 1] + w * U[as.integer(gid), 2] + rnorm(n, 0, 0.8),
                gid = gid, w = w)
Zw <- Matrix::sparseMatrix(i = seq_len(n), j = as.integer(gid), x = w, dims = c(n, q),
                           dimnames = list(NULL, levels(gid)))
a <- stab(asreml(y ~ 1 + w, random = ~ str(~ gid + gid:w, ~ us(2):id(40)),
                 data = d, trace = FALSE, maxit = 100, workspace = "2gb"))
vc <- summary(a)$varcomp
f <- rk_reml(y ~ 1 + w, random = ~ str(~ gid + mm(Zw, name = "pente"), struct = "us"),
             data = d, backend = "auto", n_restarts = 3, verbose = FALSE)
G <- f$sigmas[[1]]
verifier("str : var terme 1", relc(G[1, 1], vc[1, "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", G[1, 1], vc[1, "component"]))
verifier("str : covariance", relc(G[1, 2], vc[2, "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", G[1, 2], vc[2, "component"]))
verifier("str : var terme 2", relc(G[2, 2], vc[3, "component"]) < 1e-3,
         sprintf("%.7f vs %.7f", G[2, 2], vc[3, "component"]))
verifier("str : residuelle", relc(f$sigma_res[1, 1], vc["units!R", "component"]) < 1e-3)
verifier("str : logLik", abs(f$logLik_asreml - a$loglik) < 1e-5,
         sprintf("%.9f vs %.9f", f$logLik_asreml, a$loglik))

# ==============================================================================
cat("\n=== 5. vpredict et Wald ===\n")
# ==============================================================================
set.seed(31); q <- 60; rp <- 5; n <- q * rp
dv <- data.frame(gid = factor(rep(paste0("g", 1:q), each = rp)),
                 trt = factor(rep(1:3, length.out = n)))
dv$y <- 4 + rep(rnorm(q, 0, 1.3), each = rp) + rnorm(n, 0, 0.9) + as.numeric(dv$trt) * 0.5
a <- stab(asreml(y ~ trt, random = ~ gid, data = dv, trace = FALSE, maxit = 100), 4)
h_a <- asreml::vpredict(a, h2 ~ V1 / (V1 + V2))
w_a <- asreml::wald(a)
f <- rk_reml(y ~ trt, random = ~ gid, data = dv, backend = "auto", n_restarts = 3,
             verbose = FALSE, vpredict = c(h2 = "V1/(V1+V2)"), wald = TRUE)
pv <- f$vpredict$predictions
verifier("vpredict : estimation de h2",
         abs(pv$valeur[1] - h_a$Estimate) < 1e-5,
         sprintf("%.7f vs %.7f", pv$valeur[1], h_a$Estimate))
verifier("vpredict : ERREUR-TYPE de h2 (delta method)",
         abs(pv$se[1] - h_a$SE) < 1e-5,
         sprintf("%.7f vs %.7f", pv$se[1], h_a$SE))
# asreml donne un Wald SEQUENTIEL (type I) par defaut, remlkit un CONDITIONNEL
# (type III). Ils coincident sur le DERNIER terme, seul cas ou les deux notions
# se rejoignent : c'est donc lui qu'on compare.
tw <- f$wald$tests
i_trt <- which(tw$terme == "trt")
p_asr <- w_a[rownames(w_a) == "trt", "Pr(Chisq)"]
verifier("Wald : ddl du terme trt", tw$ddl[i_trt] == 2L, sprintf("%d", tw$ddl[i_trt]))
verifier("Wald : p-valeur du dernier terme (seq == cond)",
         abs(tw$p[i_trt] - p_asr) < 1e-9,
         sprintf("%.4g vs %.4g", tw$p[i_trt], p_asr))

# ==============================================================================
cat("\n=== 6. parametrisations SATUREES : chol(t-1), ante(t-1), fa(t), rr(t) ===\n")
# ==============================================================================
# Elles decrivent toutes la MEME Sigma non structuree. Une divergence de logLik
# signalerait une erreur de parametrisation, pas un choix de modele.
set.seed(77); qq <- 45; TT <- 4; rr_ <- 3
gg <- factor(rep(paste0("g", 1:qq), each = rr_)); nn <- qq * rr_
GG <- crossprod(matrix(rnorm(TT * TT), TT, TT)) + diag(TT)
UU <- matrix(rnorm(qq * TT), qq, TT) %*% chol(GG)
YY <- sapply(1:TT, function(k) 2 + k * 0.3 + rep(UU[, k], each = rr_) + rnorm(nn, 0, 0.9))
dsat <- data.frame(y = as.vector(YY), gid = rep(gg, TT),
                   trait = factor(rep(paste0("t", 1:TT), each = nn)), unit = rep(1:nn, TT))
fsat <- function(r) rk_reml(y ~ 0 + trait, random = stats::as.formula(paste("~", r)),
                            residual = ~ diag(trait):units, data = dsat, trait = "trait",
                            unit = "unit", backend = "auto", n_restarts = 4, verbose = FALSE)
fu <- fsat("us(gid)")
for (r in c("chol(gid, rank=3)", "ante(gid, rank=3)", "fa(gid, rank=4)", "rr(gid, rank=4)"))
  verifier(sprintf("%-18s == us", r), abs(fsat(r)$logLik - fu$logLik) < 1e-6,
           sprintf("ecart %.2e", abs(fsat(r)$logLik - fu$logLik)))

cat("\n", strrep("=", 68), "\n", sep = "")
if (length(ECHECS)) {
  cat(sprintf("ECHECS (%d) :\n%s\n", length(ECHECS), paste0("  - ", ECHECS, collapse = "\n")))
  quit(save = "no", status = 1L)
}
cat("remlkit reproduit asreml sur AR1, AR1xAR1, splines 2D, str(), vpredict et Wald.\n")
quit(save = "no", status = 0L)
