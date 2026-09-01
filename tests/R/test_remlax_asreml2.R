# ==============================================================================
# test_remlax_asreml2.R — validation des structures et de l'inference AJOUTEES
# le 01/09/2026 : lvr, ilv, mtrn anisotrope, own, dsum, predict, Kenward-Roger.
#
# TROIS SORTES DE TEMOINS, et il faut les trois :
#   asreml     pour ce qu'il sait faire (lvr, mtrn, dsum, predict)
#   pbkrtest   pour Kenward-Roger, dont c'est l'implementation de reference
#   R direct   une reimplementation independante du profil de vraisemblance,
#              qui teste la CHAINE ENTIERE (formule R -> paquet -> solveur) et
#              non seulement le nombre final
#
# Les formules de lvr et de l'anisotropie de mtrn ne sont PAS dans le manuel :
# elles ont ete identifiees en comparant des courbes de logLik a asreml (cf.
# docs/note_remlax.md). Ces tests sont ce qui les fige.
#
#   Rscript scripts/tests/test_remlax_asreml2.R
# ==============================================================================
suppressPackageStartupMessages({ library(Matrix) })
RACINE <- Sys.getenv("IGE_RACINE", ".")
source(file.path(RACINE, "R", "remlax.R"))
BACKEND <- Sys.getenv("RX_BACKEND", "auto")

.n_ok <- 0L; .n_ko <- 0L
ok <- function(lbl, cond, det = "") {
  if (isTRUE(cond)) { .n_ok <<- .n_ok + 1L; cat(sprintf("  %-52s OK   %s\n", lbl, det)) }
  else { .n_ko <<- .n_ko + 1L; cat(sprintf("  %-52s ECHEC %s\n", lbl, det)) }
}
a_asreml <- requireNamespace("asreml", quietly = TRUE) &&
  !inherits(try(suppressMessages(library(asreml)), silent = TRUE), "try-error")
if (a_asreml) asreml.options(trace = FALSE)
a_pbkr <- requireNamespace("pbkrtest", quietly = TRUE) && requireNamespace("lme4", quietly = TRUE)

# --- vraisemblance REML d'une correlation DONNEE, sigma2 profile --------------
# Temoin totalement independant du solveur : c'est lui qui dit si la structure
# construite par remlax est bien celle qu'on croit.
ll_profil <- function(C, y, X) {
  n <- length(y); p <- qr(X)$rank
  ev <- eigen(C, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-12) return(NA_real_)
  Ci <- chol2inv(chol(C)); A <- t(X) %*% Ci %*% X
  b <- solve(A, t(X) %*% Ci %*% y); r <- y - X %*% b
  s2 <- as.numeric(t(r) %*% Ci %*% r / (n - p))
  as.numeric(-0.5 * (sum(log(ev)) + (n - p) * log(s2) + log(det(A)) + (n - p)))
}

set.seed(20260901)

cat("\n=== 1. lvr : tente tronquee max(0, 1 - d/phi) ===\n")
q <- 14
d1 <- data.frame(pos = 1:q)
H <- abs(outer(d1$pos, d1$pos, "-"))
d1$y <- as.numeric(t(chol(pmax(0, 1 - H / 6) + diag(0.25, q))) %*% rnorm(q)) + 5
f1 <- rx_reml(y ~ 1, random = NULL, residual = ~ lvr(pos), data = d1,
              backend = BACKEND, verbose = FALSE, n_restarts = 4)
portee <- f1$rho[["residuelle!portee"]]
# GRILLE FINE, pas `optimize`. La vraisemblance d'une tente tronquee n'est pas
# unimodale : son support change chaque fois que la portee franchit un entier,
# ce qui cree autant d'optima locaux. Une recherche golden-section y reste
# piegee — c'est d'ailleurs ce qui arrive a asreml sur ce meme jeu.
.ll_lvr <- function(p) ll_profil(matrix(pmax(0, 1 - H / p), q, q), matrix(d1$y),
                                 matrix(1, q, 1))
gr_p <- seq(1.02, 60, by = 0.002)
gr_v <- vapply(gr_p, .ll_lvr, 0)
i0 <- which.max(replace(gr_v, !is.finite(gr_v), -Inf))
raf <- optimize(.ll_lvr, gr_p[c(max(i0 - 1, 1), min(i0 + 1, length(gr_p)))],
                maximum = TRUE, tol = 1e-11)
gr <- if (raf$objective > gr_v[i0]) raf else list(maximum = gr_p[i0], objective = gr_v[i0])
ok("lvr : logLik = profil R independant",
   abs(f1$logLik_asreml - gr$objective) < 1e-5,
   sprintf("remlax %.7f | R %.7f | portee %.4f vs %.4f",
           f1$logLik_asreml, gr$objective, portee %||% NA, gr$maximum))
if (a_asreml) {
  d1$posf <- factor(d1$pos)
  fa1 <- try(asreml(y ~ 1, residual = ~ lvr(posf), data = d1, maxit = 60), silent = TRUE)
  ok("lvr : asreml n'atteint pas un meilleur optimum",
     !inherits(fa1, "try-error") && fa1$loglik <= f1$logLik_asreml + 1e-5,
     if (inherits(fa1, "try-error")) "asreml a echoue" else
       sprintf("asreml %.6f <= remlax %.6f", fa1$loglik, f1$logLik_asreml))
}

cat("\n=== 2. mtrn : Matern anisotrope, parametres FIXES des deux cotes ===\n")
nr <- 7; nc <- 6
g2 <- expand.grid(x = 1:nr, y = 1:nc)
dx <- outer(g2$x, g2$x, "-"); dy <- outer(g2$y, g2$y, "-")
matern_ref <- function(phi, nu, delta, alpha, lambda) {
  u <-  dx * cos(alpha) + dy * sin(alpha)
  v <- -dx * sin(alpha) + dy * cos(alpha)
  sd_ <- sqrt(delta)
  h <- (abs(sd_ * u)^lambda + abs(v / sd_)^lambda)^(1 / lambda)
  z <- h / phi
  matrix(ifelse(z < 1e-12, 1, (2^(1 - nu) / gamma(nu)) * z^nu * besselK(pmax(z, 1e-12), nu)),
         nrow(g2), nrow(g2))
}
# Reponse tiree d'un VRAI champ de Matern (portee 2,5, nu 1,5) plus une pepite.
# Sur du bruit pur la portee n'est pas identifiable : la vraisemblance est plate
# et le test 3 passait sans rien verifier.
g2$resp <- as.numeric(t(chol(matern_ref(2.5, 1.5, 1, 0, 2) + diag(0.15, nrow(g2)))) %*%
                      rnorm(nrow(g2))) + 3

for (cas in list(list(lbl = "isotrope nu=0.8", phi = 3, nu = 0.8, de = 1, al = 0, la = 2),
                 list(lbl = "nu=1.5 delta=2",  phi = 3, nu = 1.5, de = 2, al = 0, la = 2),
                 list(lbl = "delta=2 alpha=.6", phi = 3, nu = 0.8, de = 2, al = 0.6, la = 2),
                 list(lbl = "nu=0.5 lambda=1",  phi = 3, nu = 0.5, de = 1, al = 0, la = 1))) {
  # La formule est construite en TEXTE : substitute() rendrait un `call`, que
  # le parseur refuse (a raison : une formule porte son environnement).
  fml <- stats::as.formula(sprintf(
    "~ mtrn(x, y, phi = '%s F', nu = '%s F', delta = '%s F', alpha = '%s F', lambda = %s)",
    cas$phi, cas$nu, cas$de, cas$al, cas$la))
  fm <- rx_reml(resp ~ 1, residual = fml, data = g2, backend = BACKEND, verbose = FALSE)
  ref <- ll_profil(matern_ref(cas$phi, cas$nu, cas$de, cas$al, cas$la),
                   matrix(g2$resp), matrix(1, nrow(g2), 1))
  ok(sprintf("mtrn %s : logLik = reference besselK", cas$lbl),
     is.finite(ref) && abs(fm$logLik_asreml - ref) < 1e-6,
     sprintf("remlax %.7f | R %.7f", fm$logLik_asreml, ref))
  if (a_asreml) {
    fam <- try(asreml(resp ~ 1, residual = fml, data = g2, maxit = 30), silent = TRUE)
    if (!inherits(fam, "try-error") && is.finite(ref))
      ok(sprintf("mtrn %s : logLik = asreml", cas$lbl),
         abs(fam$loglik - fm$logLik_asreml) < 1e-5,
         sprintf("asreml %.7f | remlax %.7f", fam$loglik, fm$logLik_asreml))
  }
}

cat("\n=== 3. mtrn : la portee est bien ESTIMEE quand on la libere ===\n")
fm2 <- rx_reml(resp ~ 1, residual = ~ mtrn(x, y, phi = 2, nu = "1.5 F"),
               data = g2, backend = BACKEND, verbose = FALSE, n_restarts = 4)
ph <- fm2$rho[["residuelle!phi"]]   # mtrn : cles prefixees (phi/nu/delta/alpha)
# Grille fine, pour la meme raison que pour lvr : rien ne garantit que le profil
# d'une Matern soit unimodal en phi, et `optimize` s'arretait sur le bord de
# l'intervalle en annoncant un optimum.
.ll_m <- function(p) ll_profil(matern_ref(p, 1.5, 1, 0, 2), matrix(g2$resp),
                               matrix(1, nrow(g2), 1))
g_p <- exp(seq(log(0.05), log(200), length.out = 3000))
g_v <- vapply(g_p, .ll_m, 0)
j0 <- which.max(replace(g_v, !is.finite(g_v), -Inf))
raf <- optimize(.ll_m, g_p[c(max(j0 - 1, 1), min(j0 + 1, length(g_p)))],
                maximum = TRUE, tol = 1e-11)
prof <- if (raf$objective > g_v[j0]) raf else list(maximum = g_p[j0], objective = g_v[j0])
ok("mtrn : phi estime = argmax du profil R",
   is.finite(prof$objective) && abs(fm2$logLik_asreml - prof$objective) < 1e-5 &&
     abs(log(ph %||% NA) - log(prof$maximum)) < 0.02,
   sprintf("phi %.5f vs %.5f (vrai 2.5) | logLik %.7f vs %.7f",
           ph %||% NA, prof$maximum, fm2$logLik_asreml, prof$objective))

cat("\n=== 4. own : une structure ecrite par l'utilisateur reproduit exp() ===\n")
q4 <- 16; d4 <- data.frame(pos = factor(1:q4), posn = 1:q4)
H4 <- abs(outer(1:q4, 1:q4, "-"))
d4$y <- as.numeric(t(chol(0.7^H4 + diag(0.3, q4))) %*% rnorm(q4)) + 2
fe <- rx_reml(y ~ 1, residual = ~ exp(posn), data = d4, backend = BACKEND,
              verbose = FALSE, n_restarts = 3)
fo <- rx_reml(y ~ 1, residual = ~ own(pos, expr = "exp(-lag*exp(p1))", n_par = 1),
              data = d4, backend = BACKEND, verbose = FALSE, n_restarts = 3)
ok("own : phi^d ecrit en exp(-d*exp(p)) donne la meme logLik",
   abs(fe$logLik - fo$logLik) < 1e-6,
   sprintf("exp() %.8f | own() %.8f", fe$logLik, fo$logLik))

cat("\n=== 5. dsum : une residuelle par section ===\n")
n5 <- 96
d5 <- data.frame(site = factor(rep(c("A", "B"), each = n5 / 2)),
                 gid = factor(rep(sprintf("g%02d", 1:24), each = 4)),
                 col = factor(rep(1:8, length.out = n5)))
u5 <- rnorm(24, 0, 1.1)
d5$y <- 4 + u5[as.integer(d5$gid)] +
  rnorm(n5, 0, ifelse(d5$site == "A", 0.5, 1.4))          # variances CONTRASTEES
f5a <- rx_reml(y ~ 1, random = ~ iid(gid), residual = ~ units, data = d5,
               backend = BACKEND, verbose = FALSE)
f5b <- rx_reml(y ~ 1, random = ~ iid(gid), residual = ~ dsum(~ units | site),
               data = d5, backend = BACKEND, verbose = FALSE)
vr <- vapply(f5b$sigmas_res, function(S) S[1, 1], 0)
ok("dsum : deux sections ameliorent la vraisemblance",
   f5b$logLik > f5a$logLik + 1,
   sprintf("1 section %.4f -> 2 sections %.4f (%+.2f)", f5a$logLik, f5b$logLik,
           f5b$logLik - f5a$logLik))
ok("dsum : les deux variances residuelles sont separees",
   length(vr) == 2L && max(vr) / min(vr) > 3,
   sprintf("%s (vraies 0.25 et 1.96)", paste(sprintf("%.4f", vr), collapse = " / ")))
if (a_asreml) {
  fa5 <- try(asreml(y ~ 1, random = ~ gid, residual = ~ dsum(~ units | site),
                    data = d5, maxit = 60), silent = TRUE)
  if (!inherits(fa5, "try-error"))
    ok("dsum : logLik = asreml", abs(fa5$loglik - f5b$logLik_asreml) < 1e-4,
       sprintf("asreml %.6f | remlax %.6f", fa5$loglik, f5b$logLik_asreml))
}

cat("\n=== 5b. dsum : des structures DIFFERENTES selon la section ===\n")
d5b <- expand.grid(col = 1:8, site = c("A", "B", "C"))
d5b$col <- factor(d5b$col); d5b$site <- factor(d5b$site)
d5b$gid <- factor(rep(1:6, length.out = nrow(d5b)))
d5b$y <- rnorm(nrow(d5b)) + rnorm(6)[as.integer(d5b$gid)]
f5c <- rx_reml(y ~ 1, random = ~ iid(gid),
               residual = ~ dsum(~ ar1(col) + ar1(col) + units | site,
                                 levels = list("A", "B", "C")),
               data = d5b, backend = BACKEND, verbose = FALSE)
ok("dsum : trois sections, deux AR1 et une iid",
   length(f5c$sigmas_res) == 3L && sum(grepl("^[AB]$", names(f5c$rho))) == 2L &&
     f5c$n_par == 6L,
   sprintf("%d sections | %d parametres | rho sur %s", length(f5c$sigmas_res),
           f5c$n_par, paste(setdiff(names(f5c$rho), ""), collapse = "+")))
# Une section AR1 dont les unites sont dupliquees rend R singuliere : le solveur
# doit le REFUSER, pas rendre NaN. C'est le defaut qu'a exhume le balayage de
# parite.
mauvais <- try(rx_reml(y ~ 1, random = ~ iid(gid),
                       residual = ~ dsum(~ ar1(col) + units | site,
                                         levels = list(c("A", "B"), "C")),
                       data = d5b, backend = BACKEND, verbose = FALSE), silent = TRUE)
ok("dsum : unites dupliquees dans une section -> refus explicite",
   inherits(mauvais, "try-error"), "erreur levee plutot que -2logL = NaN")

cat("\n=== 6. predict : moyennes ajustees et leurs erreurs-types ===\n")
n6 <- 90
d6 <- data.frame(trt = factor(rep(c("A", "B", "C"), each = n6 / 3)),
                 bloc = factor(rep(1:6, length.out = n6)),
                 x = rnorm(n6))
ub <- rnorm(6, 0, 0.9)
d6$y <- 5 + c(A = 0, B = 1.2, C = -0.4)[as.character(d6$trt)] + 0.5 * d6$x +
  ub[as.integer(d6$bloc)] + rnorm(n6, 0, 0.7)
f6 <- rx_reml(y ~ trt + x, random = ~ iid(bloc), data = d6, backend = BACKEND,
              verbose = FALSE, wald = TRUE)
pv <- rx_predict(f6, classify = "trt", sed = TRUE)
if (a_asreml) {
  fa6 <- try(asreml(y ~ trt + x, random = ~ bloc, data = d6, maxit = 60), silent = TRUE)
  if (!inherits(fa6, "try-error")) {
    pa <- predict(fa6, classify = "trt")$pvals
    ok("predict : valeurs = asreml",
       max(abs(pv$predicted.value - pa$predicted.value)) < 1e-4,
       sprintf("ecart max %.2e", max(abs(pv$predicted.value - pa$predicted.value))))
    ok("predict : erreurs-types = asreml",
       max(abs(pv$std.error - pa$std.error)) < 1e-4,
       sprintf("ecart max %.2e", max(abs(pv$std.error - pa$std.error))))
  }
}
ok("predict : covariable a sa moyenne, predictions estimables",
   all(pv$estimable) && nrow(pv) == 3L,
   sprintf("%s", paste(sprintf("%s=%.4f(%.4f)", pv$trt, pv$predicted.value,
                               pv$std.error), collapse = " ")))
ok("predict : erreur-type des differences disponible",
   is.finite(attr(pv, "sed.moyen")),
   sprintf("sed moyen %.5f", attr(pv, "sed.moyen")))

cat("\n=== 7. Kenward-Roger vs pbkrtest ===\n")
if (a_pbkr) {
  set.seed(11); nb <- 5
  d7 <- do.call(rbind, lapply(1:nb, function(b) {
    k <- sample(3:7, 1)                                  # DESEQUILIBRE volontaire
    data.frame(bloc = b, trt = sample(c("A", "B", "C"), k, replace = TRUE))
  }))
  d7$bloc <- factor(d7$bloc); d7$trt <- factor(d7$trt); d7$x <- rnorm(nrow(d7))
  u7 <- rnorm(nb, 0, 1.1)
  d7$y <- 2 + c(A = 0, B = 0.7, C = 1.3)[as.character(d7$trt)] + 0.4 * d7$x +
    u7[as.integer(d7$bloc)] + rnorm(nrow(d7), 0, 0.9)
  m1 <- lme4::lmer(y ~ trt + x + (1 | bloc), data = d7, REML = TRUE)
  m0 <- lme4::lmer(y ~ x + (1 | bloc), data = d7, REML = TRUE)
  kr <- pbkrtest::KRmodcomp(m1, m0)$stats
  se_ref <- sqrt(diag(as.matrix(pbkrtest::vcovAdj(m1))))
  f7 <- rx_reml(y ~ trt + x, random = ~ iid(bloc), data = d7, backend = BACKEND,
                verbose = FALSE, kenward_roger = TRUE)
  tw <- f7$kenward_roger$tests
  i <- which(tw$terme == "trt")
  ok("K-R : ddl du denominateur = pbkrtest",
     abs(tw$denDF[i] - kr$ddf) < 1e-3,
     sprintf("remlax %.5f | pbkrtest %.5f", tw$denDF[i], kr$ddf))
  ok("K-R : statistique F = pbkrtest", abs(tw$F[i] - kr$Fstat) < 1e-5,
     sprintf("remlax %.6f | pbkrtest %.6f", tw$F[i], kr$Fstat))
  ok("K-R : erreurs-types ajustees = pbkrtest",
     max(abs(f7$kenward_roger$se_beta - se_ref)) < 1e-5,
     sprintf("ecart max %.2e", max(abs(f7$kenward_roger$se_beta - se_ref))))
  ok("K-R : l'ajustement GONFLE les erreurs-types",
     all(f7$kenward_roger$se_beta >= f7$kenward_roger$se_beta_brut - 1e-12),
     sprintf("rapport moyen %.5f",
             mean(f7$kenward_roger$se_beta / f7$kenward_roger$se_beta_brut)))
} else cat("  (pbkrtest/lme4 absents : section sautee)\n")

cat("\n=== 8. predict avec la part aleatoire (BLUP + erreur de prediction) ===\n")
f8 <- rx_reml(y ~ 1 + x, random = ~ iid(bloc), data = d6, backend = BACKEND, verbose = FALSE)
pb <- rx_predict(f8, classify = "bloc", include_random = TRUE, verbose = FALSE)
pf <- rx_predict(f8, classify = "bloc", include_random = FALSE, verbose = FALSE)
ok("predict : le BLUP entre dans la moyenne par bloc",
   sd(pb$predicted.value) > 1e-6 && sd(pf$predicted.value) < 1e-9,
   sprintf("ecart-type avec BLUP %.5f, sans %.2e",
           sd(pb$predicted.value), sd(pf$predicted.value)))
ok("predict : l'erreur de PREDICTION reste finie et positive",
   all(is.finite(pb$std.error)) && all(pb$std.error > 0),
   sprintf("SE %s", paste(sprintf("%.4f", pb$std.error), collapse = " ")))

cat(sprintf("\n%s\n%d test(s) OK, %d echec(s)\n", strrep("=", 70), .n_ok, .n_ko))
if (.n_ko > 0L) quit(status = 1)
