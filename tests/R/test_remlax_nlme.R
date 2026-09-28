# ==============================================================================
# test_remlax_nlme.R — structures de correlation contre nlme (gls et lme)
# ------------------------------------------------------------------------------
# nlme (Pinheiro & Bates) est la reference ouverte pour les structures de
# correlation residuelle : corAR1, corCAR1, corARMA, corCompSymm, corSymm,
# corExp, corGaus, corLin, corSpher, et varIdent pour l'heteroscedasticite.
# Ces structures n'avaient jusqu'ici de reference qu'asreml (sous licence) ou
# aucune (ar2, arma, gau, cor). Ici chacune est confrontee a nlme sur le MEME
# modele et les MEMES donnees.
#
# COMMENT UN corStruct GROUPE S'ECRIT DANS remlax. nlme ecrit corAR1(form = ~ t | g)
# pour « AR1 dans chaque groupe g, aucune correlation entre groupes ». Dans
# remlax, la residuelle est R = Sigma_trait (x) C_unit : en declarant le GROUPE
# comme « trait » (trait = "g") avec la structure id(g), Sigma = s2 I_g, et
# C_unit porte la correlation entre les temps ou positions. residual =
# ~ id(g):ar1(tf) avec trait = "g" est donc exactement corAR1(form = ~ t | g).
#
# CONSTANTE DE logLik. nlme rend la vraisemblance restreinte COMPLETE, avec le
# terme -(n-p)/2 log(2 pi), comme lme4 ; c'est fit$logLik de remlax qui lui est
# comparable, pas fit$logLik_asreml. Le premier controle le verifie.
#
# TOLERANCES : celles des autres suites (-2logL 1e-6 absolu, variances 1e-4
# relatif, effets fixes 1e-6 relatif), avec nlme resserre par glsControl /
# lmeControl (tolerance 1e-10) : a ses reglages par defaut nlme s'arrete a
# quelques 1e-6 de son optimum et ce serait lui, pas remlax, que la
# tolerance mesurerait.
#
#   Rscript tests/R/test_remlax_nlme.R
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite); library(nlme)
})
source(here::here("R", "remlax.R"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
cat("nlme", as.character(packageVersion("nlme")), "| R", R.version$major, R.version$minor, "\n")

ECHECS <- character(0)
verifier <- function(nom, cond, detail = "") {
  cat(sprintf("  %-56s %s %s\n", nom, if (isTRUE(cond)) "OK " else "ECHEC", detail))
  if (!isTRUE(cond)) ECHECS <<- c(ECHECS, nom)
}
rel <- function(a, b) abs(a - b) / pmax(abs(b), 1e-8)
section <- function(titre, expr) {
  cat("\n=== ", titre, " ===\n", sep = "")
  tryCatch(expr, error = function(e) verifier(paste0(sub(" .*", "", titre), " : execution"),
                                              FALSE, conditionMessage(e)))
}
CTL <- glsControl(tolerance = 1e-10, msTol = 1e-10, msMaxIter = 500, opt = "nlminb")
CTL_LME <- lmeControl(tolerance = 1e-10, msTol = 1e-10, msMaxIter = 500, opt = "nlminb",
                      niterEM = 50)
cor_par <- function(m) coef(m$modelStruct$corStruct, unconstrained = FALSE)
n2l <- function(m) -2 * as.numeric(logLik(m))
se_gls <- function(m) sqrt(diag(vcov(m)))

# Controles communs a toutes les sections : -2logL, sigma2, beta, SE(beta).
controles_communs <- function(tag, m, fit, s2_nlme = m$sigma^2) {
  verifier(paste0(tag, " : -2logL identique"), abs(-2 * fit$logLik - n2l(m)) < 1e-6,
           sprintf("%.9f vs %.9f", -2 * fit$logLik, n2l(m)))
  verifier(paste0(tag, " : variance residuelle"), rel(fit$sigma_res[1, 1], s2_nlme) < 1e-4,
           sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], s2_nlme))
  verifier(paste0(tag, " : effets fixes"),
           max(rel(as.numeric(fit$beta), as.numeric(coef(m)))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(coef(m))))))
  verifier(paste0(tag, " : erreurs-types des effets fixes"),
           max(rel(sqrt(diag(fit$vbeta)), se_gls(m))) < 1e-4,
           sprintf("ecart rel max %.2e", max(rel(sqrt(diag(fit$vbeta)), se_gls(m)))))
}

# Series groupees : G groupes de Tn temps, une covariable, un effet fixe.
serie <- function(G, Tn, seed, bruit) {
  set.seed(seed)
  n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) bruit(Tn)))
  data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
}

# ==============================================================================
section("1. corAR1 : AR1 dans chaque groupe, contre gls", {
# ==============================================================================
  d <- serie(12, 10, 1, function(Tn) as.numeric(arima.sim(list(ar = 0.6), Tn)))
  m <- gls(y ~ x, d, correlation = corAR1(form = ~ t | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):ar1(tf), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  n <- nrow(d); p <- 2
  verifier("constante de logLik : nlme = remlax$logLik",
           abs(fit$logLik - as.numeric(logLik(m))) < 1e-6 &&
             abs(fit$logLik_asreml - fit$logLik - (n - p) / 2 * log(2 * pi)) < 1e-9,
           sprintf("nlme %.6f | remlax %.6f | logLik_asreml %.6f (+(n-p)/2 log 2pi = %.4f)",
                   as.numeric(logLik(m)), fit$logLik, fit$logLik_asreml, (n - p) / 2 * log(2 * pi)))
  controles_communs("corAR1", m, fit)
  verifier("corAR1 : rho", abs(fit$rho[["residuelle"]] - cor_par(m)) < 1e-4,
           sprintf("%.7f vs %.7f", fit$rho[["residuelle"]], cor_par(m)))
})

# ==============================================================================
section("2. corCAR1 : AR1 continu a temps irreguliers, contre gls", {
# ==============================================================================
  set.seed(2); G <- 10; Tn <- 12; n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn))
  # temps distincts d'un groupe a l'autre (une unite = une position)
  pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(runif(Tn, 0, 15)))))
  x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) {
    D <- as.matrix(dist(pos[g == i])); as.numeric(t(chol(0.7 ^ D)) %*% rnorm(Tn)) }))
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
  m <- gls(y ~ x, d, correlation = corCAR1(form = ~ pos | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):exp(pos), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("corCAR1", m, fit)
  verifier("corCAR1 : phi (correlation par unite de temps)",
           abs(fit$rho[["residuelle"]] - cor_par(m)) < 1e-4,
           sprintf("%.7f vs %.7f (vrai 0.7)", fit$rho[["residuelle"]], cor_par(m)))
})

# ==============================================================================
section("3. corARMA(p=2) : AR2, contre gls", {
# ==============================================================================
  d <- serie(10, 12, 3, function(Tn) as.numeric(arima.sim(list(ar = c(0.5, 0.2)), Tn)))
  m <- gls(y ~ x, d, correlation = corARMA(form = ~ t | g, p = 2), method = "REML",
           control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):ar2(tf), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("ar2", m, fit)
  verifier("ar2 : phi1 et phi2", max(abs(fit$rho[["residuelle"]] - cor_par(m))) < 1e-4,
           sprintf("remlax %s | nlme %s", paste(round(fit$rho[["residuelle"]], 6), collapse = " "),
                   paste(round(cor_par(m), 6), collapse = " ")))
})

# ==============================================================================
section("4. corARMA(p=1, q=1) : ARMA(1,1), contre gls", {
# ==============================================================================
  d <- serie(10, 12, 4, function(Tn) as.numeric(arima.sim(list(ar = 0.5, ma = 0.3), Tn)))
  m <- gls(y ~ x, d, correlation = corARMA(form = ~ t | g, p = 1, q = 1), method = "REML",
           control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):arma(tf), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("arma", m, fit)
  # remlax range ses deux parametres dans l'ordre (theta_MA, phi_AR) ; nlme
  # (phi, theta). On compare les deux couples apres reordonnancement.
  r_rx <- fit$rho[["residuelle"]]; r_nl <- cor_par(m)
  verifier("arma : phi et theta (ordre remlax : theta, phi)",
           abs(r_rx[2] - r_nl[1]) < 1e-4 && abs(r_rx[1] - r_nl[2]) < 1e-4,
           sprintf("remlax theta %.6f phi %.6f | nlme phi %.6f theta %.6f",
                   r_rx[1], r_rx[2], r_nl[1], r_nl[2]))
})

# ==============================================================================
section("5. corCompSymm : correlation uniforme, contre gls", {
# ==============================================================================
  d <- serie(10, 12, 5, function(Tn) rnorm(1, 0, 0.8) + rnorm(Tn, 0, 1))
  m <- gls(y ~ x, d, correlation = corCompSymm(form = ~ 1 | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):cor(tf), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("cor", m, fit)
  verifier("cor : rho uniforme", abs(fit$rho[["residuelle"]] - cor_par(m)) < 1e-4,
           sprintf("%.7f vs %.7f", fit$rho[["residuelle"]], cor_par(m)))
})

# ==============================================================================
section("6. corExp : decroissance exponentielle en 1D, contre gls", {
# ==============================================================================
  # POURQUOI CE CAS EST LE PLUS IMPORTANT DE LA SUITE. Il a revele, le
  # 2026-09-28, que les structures en rho^d demarraient a rho = 0 exactement,
  # point ou le gradient est nul par construction : l'optimiseur concluait
  # a la convergence sans bouger, et -2logL restait a 65 unites de l'optimum
  # trouve par gls. Le depart est desormais rho = 0.5 au plus proche voisin.
  set.seed(6); G <- 8; Tn <- 15; n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn))
  pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(runif(Tn, 0, 30)))))
  x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) {
    D <- as.matrix(dist(pos[g == i])); as.numeric(t(chol(exp(-D / 4))) %*% rnorm(Tn)) }))
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
  m <- gls(y ~ x, d, correlation = corExp(form = ~ pos | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):exp(pos), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("exp", m, fit)
  verifier("exp : rho = exp(-1/portee nlme)",
           abs(fit$rho[["residuelle"]] - exp(-1 / cor_par(m))) < 1e-4,
           sprintf("%.7f vs %.7f (portee %.4f, vraie 4)", fit$rho[["residuelle"]],
                   exp(-1 / cor_par(m)), cor_par(m)))
  verifier("exp : l'ajustement a bouge de son point de depart",
           fit$n_iter > 0 && fit$rho[["residuelle"]] > 0.05,
           sprintf("%d iterations, rho %.4f", fit$n_iter, fit$rho[["residuelle"]]))
})

# ==============================================================================
section("7. corGaus : decroissance gaussienne en 1D, contre gls", {
# ==============================================================================
  # Positions ENTIERES a pas >= 1. La correlation gaussienne s'ecrit rho^(d^2)
  # avec rho = exp(-1/portee^2) : a pas < 1 et portee courte, rho optimal
  # tombe sous 1e-6 et la parametrisation en tanh n'a plus de pente. C'est
  # une limite de la parametrisation d'asreml que remlax reprend, pas un
  # defaut d'optimisation : on rescale les coordonnees, comme avec asreml.
  set.seed(7); G <- 8; Tn <- 15; n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn))
  pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(sample(1:40, Tn)))))
  x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) {
    D <- as.matrix(dist(pos[g == i]))
    as.numeric(t(chol(exp(-(D / 3) ^ 2) + diag(1e-8, Tn))) %*% rnorm(Tn)) }))
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
  m <- gls(y ~ x, d, correlation = corGaus(form = ~ pos | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):gau(pos), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("gau", m, fit)
  verifier("gau : rho = exp(-1/portee^2)",
           abs(fit$rho[["residuelle"]] - exp(-1 / cor_par(m) ^ 2)) < 1e-4,
           sprintf("%.7f vs %.7f (portee %.4f, vraie 3)", fit$rho[["residuelle"]],
                   exp(-1 / cor_par(m) ^ 2), cor_par(m)))
})

# ==============================================================================
section("8. corLin : tente lineaire tronquee (lvr), contre gls", {
# ==============================================================================
  set.seed(8); G <- 8; Tn <- 15; n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn))
  pos <- as.numeric(unlist(lapply(seq_len(G), function(i) sort(runif(Tn, 0, 30)))))
  x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) {
    D <- as.matrix(dist(pos[g == i]))
    as.numeric(t(chol(pmax(1 - D / 6, 0) + diag(1e-8, Tn))) %*% rnorm(Tn)) }))
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, pos = pos)
  # La tente tronquee rend la vraisemblance NON LISSE en la portee (un pli a
  # chaque distance observee) et multimodale : gls trouve 200.19 depuis
  # value = 5 et 188.62 depuis value = 2, remlax 193.93 depuis son depart.
  # On compare donc deux choses qui ne dependent pas du mode atteint : la
  # vraisemblance des deux logiciels A LA MEME portee (celle du meilleur gls,
  # imposee a remlax par fixed_theta), et le fait que l'optimum libre de
  # remlax avec redemarrages ne soit pas moins bon que le meilleur gls.
  fits_nl <- lapply(c(2, 3, 4, 6, 8), function(v) tryCatch(
    gls(y ~ x, d, correlation = corLin(form = ~ pos | g, value = v), method = "REML",
        control = CTL), error = function(e) NULL))
  fits_nl <- Filter(Negate(is.null), fits_nl)
  m <- fits_nl[[which.min(sapply(fits_nl, n2l))]]
  fit <- rx_reml(y ~ x, residual = ~ id(g):lvr(pos), data = d, trait = "g",
                 backend = BK, verbose = FALSE, n_restarts = 4)
  fit_fixe <- rx_reml(y ~ x, residual = ~ id(g):lvr(pos), data = d, trait = "g",
                      backend = BK, verbose = FALSE,
                      theta_init = c(log(m$sigma), log(cor_par(m))), fixed_theta = 2)
  controles_communs("lvr (portee gls)", m, fit_fixe)
  verifier("lvr : portee lue = portee imposee",
           rel(fit_fixe$rho[["residuelle!portee"]], cor_par(m)) < 1e-6,
           sprintf("%.5f vs %.5f (vraie 6)", fit_fixe$rho[["residuelle!portee"]], cor_par(m)))
  verifier("lvr : optimum libre de remlax pas moins bon que gls",
           -2 * fit$logLik <= n2l(m) + 1e-6,
           sprintf("remlax %.6f (portee %.4f, %d redemarrages) | meilleur gls sur %d departs %.6f (portee %.4f)",
                   -2 * fit$logLik, fit$rho[["residuelle!portee"]], 4L, length(fits_nl), n2l(m), cor_par(m)))
})

# ==============================================================================
section("9. corSpher : spherique en 2D, contre gls", {
# ==============================================================================
  set.seed(9); G <- 6; Tn <- 20; n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn))
  cx <- runif(n, 0, 20); cy <- runif(n, 0, 20)
  sph <- function(D, r) ifelse(D < r, 1 - 1.5 * D / r + 0.5 * (D / r) ^ 3, 0)
  e <- unlist(lapply(seq_len(G), function(i) {
    D <- as.matrix(dist(cbind(cx, cy)[g == i, ]))
    as.numeric(t(chol(sph(D, 8) + diag(1e-8, Tn))) %*% rnorm(Tn)) }))
  x <- rnorm(n)
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, cx = cx, cy = cy)
  m <- gls(y ~ x, d, correlation = corSpher(form = ~ cx + cy | g, value = 6), method = "REML",
           control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ id(g):sph(cx, cy), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  controles_communs("sph", m, fit)
  verifier("sph : portee", rel(fit$rho[["residuelle!portee"]], cor_par(m)) < 1e-3,
           sprintf("%.5f vs %.5f (vraie 8)", fit$rho[["residuelle!portee"]], cor_par(m)))
})

# ==============================================================================
section("10. corSymm + varIdent : covariance non structuree (us), contre gls", {
# ==============================================================================
  set.seed(10); G <- 40; Tn <- 4; n <- G * Tn
  S <- matrix(c(1.0, 0.5, 0.3, 0.1,
                0.5, 1.5, 0.6, 0.2,
                0.3, 0.6, 2.0, 0.7,
                0.1, 0.2, 0.7, 0.8), 4, 4)
  g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- rnorm(n)
  e <- as.numeric(t(matrix(rnorm(n), G, Tn) %*% chol(S)))
  d <- data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
  m <- gls(y ~ x, d, correlation = corSymm(form = ~ t | g), weights = varIdent(form = ~ 1 | tf),
           method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ us(tf):g, data = d, trait = "tf",
                 backend = BK, verbose = FALSE)
  verifier("us : -2logL identique", abs(-2 * fit$logLik - n2l(m)) < 1e-6,
           sprintf("%.9f vs %.9f", -2 * fit$logLik, n2l(m)))
  # Sigma de gls : sigma^2 * D C D, D = poids de varIdent (1 pour le niveau de reference).
  w <- c(1, coef(m$modelStruct$varStruct, unconstrained = FALSE, allCoef = FALSE))
  w <- w[order(c(levels(d$tf)[1], names(w)[-1]))]
  C <- corMatrix(m$modelStruct$corStruct)[[1]]
  S_nlme <- m$sigma ^ 2 * outer(w, w) * C
  # Ecart ABSOLU rapporte a la plus grande variance : a -2logL egale a 1e-8,
  # une covariance de 0.1 differe de 2.7e-3 en relatif entre deux optimiseurs
  # arretes sur une surface plate ; c'est la surface, pas la vraisemblance.
  ecart_us <- max(abs(as.numeric(fit$sigma_res) - as.numeric(S_nlme))) / max(abs(S_nlme))
  verifier("us : les 10 composantes de Sigma", ecart_us < 1e-3,
           sprintf("ecart abs max / max var %.2e | diag remlax %s | nlme %s", ecart_us,
                   paste(round(diag(fit$sigma_res), 4), collapse = " "),
                   paste(round(diag(S_nlme), 4), collapse = " ")))
  verifier("us : effets fixes", max(rel(as.numeric(fit$beta), as.numeric(coef(m)))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(coef(m))))))
})

# ==============================================================================
section("11. varIdent seul : une variance residuelle par groupe (diag), contre gls", {
# ==============================================================================
  set.seed(11); G <- 5; Tn <- 30; n <- G * Tn
  sds <- c(0.5, 1, 1.5, 2, 3)
  g <- factor(rep(seq_len(G), each = Tn)); x <- rnorm(n)
  d <- data.frame(y = 1 + 0.5 * x + rnorm(n, 0, rep(sds, each = Tn)), x = x, g = g,
                  u = factor(seq_len(n)))
  m <- gls(y ~ x, d, weights = varIdent(form = ~ 1 | g), method = "REML", control = CTL)
  fit <- rx_reml(y ~ x, residual = ~ diag(g):u, data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  verifier("diag : -2logL identique", abs(-2 * fit$logLik - n2l(m)) < 1e-6,
           sprintf("%.9f vs %.9f", -2 * fit$logLik, n2l(m)))
  w <- c(1, coef(m$modelStruct$varStruct, unconstrained = FALSE, allCoef = FALSE))
  v_nlme <- m$sigma ^ 2 * w ^ 2
  verifier("diag : les cinq variances", max(rel(diag(fit$sigma_res), v_nlme)) < 1e-4,
           sprintf("remlax %s | nlme %s", paste(round(diag(fit$sigma_res), 4), collapse = " "),
                   paste(round(v_nlme, 4), collapse = " ")))
  verifier("diag : effets fixes", max(rel(as.numeric(fit$beta), as.numeric(coef(m)))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(coef(m))))))
})

# ==============================================================================
section("12. lme : intercept aleatoire + corAR1 intra-groupe, contre lme", {
# ==============================================================================
  d <- serie(15, 8, 12, function(Tn) rnorm(1, 0, 1.2) + as.numeric(arima.sim(list(ar = 0.5), Tn)))
  m <- lme(y ~ x, random = ~ 1 | g, correlation = corAR1(form = ~ t | g), data = d,
           method = "REML", control = CTL_LME)
  fit <- rx_reml(y ~ x, random = ~ g, residual = ~ id(g):ar1(tf), data = d, trait = "g",
                 backend = BK, verbose = FALSE)
  s2g <- as.numeric(VarCorr(m)[1, 1]); s2e <- m$sigma ^ 2
  verifier("lme AR1 : -2logL identique", abs(-2 * fit$logLik - n2l(m)) < 1e-6,
           sprintf("%.9f vs %.9f", -2 * fit$logLik, n2l(m)))
  verifier("lme AR1 : variance du groupe", rel(fit$sigmas$g[1, 1], s2g) < 1e-4,
           sprintf("%.6f vs %.6f", fit$sigmas$g[1, 1], s2g))
  verifier("lme AR1 : variance residuelle", rel(fit$sigma_res[1, 1], s2e) < 1e-4,
           sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], s2e))
  verifier("lme AR1 : rho", abs(fit$rho[["residuelle"]] - cor_par(m)) < 1e-4,
           sprintf("%.7f vs %.7f", fit$rho[["residuelle"]], cor_par(m)))
  verifier("lme AR1 : effets fixes", max(rel(as.numeric(fit$beta), as.numeric(fixef(m)))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(fixef(m))))))
  verifier("lme AR1 : erreurs-types des effets fixes",
           max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(vcov(m))))) < 1e-4,
           sprintf("ecart rel max %.2e", max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(vcov(m)))))))
  b_rx <- as.numeric(fit$blups$g); b_nl <- ranef(m)[levels(d$g), 1]
  verifier("lme AR1 : BLUPs", max(abs(b_rx - b_nl)) < 1e-4,
           sprintf("ecart max %.2e", max(abs(b_rx - b_nl))))
})

# ==============================================================================
cat("\n==============================================================\n")
if (length(ECHECS)) {
  cat("ECHECS (", length(ECHECS), ") :\n", sep = ""); for (e in ECHECS) cat("  -", e, "\n")
  quit(status = 1)
}
cat("TOUS LES CONTROLES PASSENT : remlax rend la meme vraisemblance restreinte\n",
    "que nlme::gls et nlme::lme sur les structures de correlation testees.\n")
