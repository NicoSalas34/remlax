# ==============================================================================
# test_remlax_lme4.R — termes croises, emboites, pentes aleatoires, BLUP et PEV
#                      contre lme4
# ------------------------------------------------------------------------------
# test_remlax.R couvre deja un facteur, deux facteurs croises equilibres et la
# formule. Ici : les dispositifs que lme4 ajuste et que le chapitre utilise
# sous d'autres noms — une covariance us entre deux colonnes d'incidence du
# meme facteur (pentes aleatoires = le terme dge_ige a deux colonnes), la
# version diag de ce terme, un emboitement, trois facteurs croises
# desequilibres — et, sur chacun, les BLUP et la variance d'erreur de
# prediction.
#
# PEV : DEUX CONVENTIONS. lme4 (ranef(condVar = TRUE)) rend Var(u | y) a beta
# CONNU : sigma^2 (Z'Z + sigma^2 G^-1)^-1. remlax rend la PEV des equations du
# modele mixte, diag(G - G Z' P Z G), qui integre l'incertitude sur beta, comme
# asreml. Les deux sont calculees ici a la main depuis les estimations de lme4
# et chacune est confrontee au logiciel qui la rend.
#
#   Rscript tests/R/test_remlax_lme4.R
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite); library(lme4)
})
source(here::here("R", "remlax.R"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
cat("lme4", as.character(packageVersion("lme4")), "| R", R.version$major, R.version$minor, "\n")

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
CTL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6),
                   check.conv.singular = "ignore")
n2l <- function(m) -2 * as.numeric(logLik(m))
inc <- function(f) Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1,
                                        dims = c(length(f), nlevels(f)))

# PEV a la main, depuis G (bloc-diagonale par terme), sigma2 et Z, X.
#   MME : sigma2 * [C^-1]_uu avec C = [X'X X'Z ; Z'X Z'Z + sigma2 G^-1]
#   lme4 : sigma2 * (Z'Z + sigma2 G^-1)^-1
pev_main <- function(X, Z, Ginv, s2) {
  X <- as.matrix(X); Z <- as.matrix(Z)
  C <- rbind(cbind(crossprod(X), crossprod(X, Z)),
             cbind(crossprod(Z, X), crossprod(Z) + s2 * Ginv))
  Ci <- solve(C); p <- ncol(X)
  list(mme = s2 * diag(Ci)[-(1:p)],
       cond = s2 * diag(solve(crossprod(Z) + s2 * Ginv)))
}
controles_communs <- function(tag, m4, fit) {
  verifier(paste0(tag, " : -2logL identique"), abs(-2 * fit$logLik - n2l(m4)) < 1e-6,
           sprintf("%.9f vs %.9f", -2 * fit$logLik, n2l(m4)))
  verifier(paste0(tag, " : effets fixes"),
           max(rel(as.numeric(fit$beta), as.numeric(fixef(m4)))) < 1e-6,
           sprintf("ecart rel max %.2e", max(rel(as.numeric(fit$beta), as.numeric(fixef(m4))))))
  verifier(paste0(tag, " : erreurs-types des effets fixes"),
           max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(as.matrix(vcov(m4)))))) < 1e-4,
           sprintf("ecart rel max %.2e",
                   max(rel(sqrt(diag(fit$vbeta)), sqrt(diag(as.matrix(vcov(m4))))))))
}

# ==============================================================================
section("1. pentes aleatoires correlees (1 + x | g) : us a deux colonnes", {
# ==============================================================================
  set.seed(41); q <- 30; mrep <- 8; n <- q * mrep
  g <- factor(rep(seq_len(q), each = mrep)); x <- rnorm(n)
  U <- matrix(rnorm(q * 2), q, 2) %*% chol(matrix(c(1, 0.3, 0.3, 0.5), 2))
  y <- 1 + 0.5 * x + U[g, 1] + U[g, 2] * x + rnorm(n, 0, 0.8)
  d <- data.frame(y = y, x = x, g = g)
  m4 <- lmer(y ~ x + (1 + x | g), d, REML = TRUE, control = CTL)
  G4 <- as.matrix(VarCorr(m4)$g)[1:2, 1:2]; s2 <- sigma(m4) ^ 2
  Z0 <- inc(g); Z1 <- Z0 * x
  mod <- rx_model(y, model.matrix(~ x, d),
                  list(rx_term("g", list(int = Z0, pente = Z1), struct = "us", levels = levels(g))))
  fit <- rx_fit(mod, backend = BK, verbose = FALSE, pev = TRUE)
  controles_communs("pentes us", m4, fit)
  verifier("pentes us : les trois composantes de Sigma",
           max(rel(as.numeric(fit$sigmas$g), as.numeric(G4))) < 1e-4,
           sprintf("remlax %s | lme4 %s", paste(round(as.numeric(fit$sigmas$g)[c(1, 2, 4)], 5), collapse = " "),
                   paste(round(as.numeric(G4)[c(1, 2, 4)], 5), collapse = " ")))
  verifier("pentes us : variance residuelle", rel(fit$sigma_res[1, 1], s2) < 1e-4,
           sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], s2))
  b4 <- as.matrix(ranef(m4)$g)[levels(g), ]
  verifier("pentes us : BLUPs (intercepts et pentes)",
           max(abs(as.numeric(fit$blups$g) - as.numeric(b4))) < 1e-4,
           sprintf("ecart max %.2e sur %d valeurs", max(abs(as.numeric(fit$blups$g) - as.numeric(b4))), length(b4)))
  # PEV : G = Sigma (x) I_q, colonnes de Z rangees (colonne de Sigma, niveau)
  Z <- cbind(as.matrix(Z0), as.matrix(Z1))
  pv <- pev_main(model.matrix(~ x, d), Z, kronecker(solve(fit$sigmas$g), diag(q)), fit$sigma_res[1, 1])
  pv4 <- attr(ranef(m4, condVar = TRUE)$g, "postVar")
  cond4 <- c(pv4[1, 1, ], pv4[2, 2, ])
  verifier("pentes us : PEV remlax = formule des MME (avec beta)",
           max(rel(as.numeric(fit$pev$g), pv$mme)) < 1e-4,
           sprintf("ecart rel max %.2e | PEV moyenne %.5f", max(rel(as.numeric(fit$pev$g), pv$mme)), mean(pv$mme)))
  verifier("pentes us : condVar lme4 = formule sans beta",
           max(rel(cond4, pv$cond)) < 1e-3,
           sprintf("ecart rel max %.2e | condVar moyenne %.5f (les deux conventions different)",
                   max(rel(cond4, pv$cond)), mean(cond4)))
})

# ==============================================================================
section("2. pentes aleatoires independantes (1 + x || g) : diag a deux colonnes", {
# ==============================================================================
  set.seed(42); q <- 25; mrep <- 10; n <- q * mrep
  g <- factor(rep(seq_len(q), each = mrep)); x <- rnorm(n)
  y <- 2 + 0.3 * x + rnorm(q, 0, 1.1)[g] + rnorm(q, 0, 0.6)[g] * x + rnorm(n, 0, 0.9)
  d <- data.frame(y = y, x = x, g = g)
  m4 <- lmer(y ~ x + (1 + x || g), d, REML = TRUE, control = CTL)
  vc <- as.data.frame(VarCorr(m4))
  Z0 <- inc(g); Z1 <- Z0 * x
  mod <- rx_model(y, model.matrix(~ x, d),
                  list(rx_term("g", list(int = Z0, pente = Z1), struct = "diag", levels = levels(g))))
  fit <- rx_fit(mod, backend = BK, verbose = FALSE)
  controles_communs("pentes diag", m4, fit)
  v4 <- c(vc$vcov[vc$grp == "g" & vc$var1 == "(Intercept)"], vc$vcov[vc$grp == "g.1" | (vc$grp == "g" & vc$var1 == "x")])
  v4 <- vc$vcov[vc$grp != "Residual"]
  verifier("pentes diag : les deux variances", max(rel(diag(fit$sigmas$g), v4)) < 1e-4,
           sprintf("remlax %s | lme4 %s", paste(round(diag(fit$sigmas$g), 5), collapse = " "),
                   paste(round(v4, 5), collapse = " ")))
  verifier("pentes diag : covariance nulle par construction", fit$sigmas$g[1, 2] == 0,
           sprintf("%.1e", fit$sigmas$g[1, 2]))
})

# ==============================================================================
section("3. emboitement (1 | g/h) : deux termes iid", {
# ==============================================================================
  set.seed(43); a <- 12; b <- 4; r <- 5; n <- a * b * r
  g <- factor(rep(seq_len(a), each = b * r)); h <- factor(rep(rep(seq_len(b), each = r), a))
  gh <- factor(paste(h, g, sep = ":"))     # nommage h:g, comme ranef(lme4)
  y <- 3 + rnorm(a, 0, 1.2)[g] + rnorm(nlevels(gh), 0, 0.7)[gh] + rnorm(n, 0, 1.0)
  d <- data.frame(y = y, g = g, h = h, gh = gh)
  m4 <- lmer(y ~ 1 + (1 | g / h), d, REML = TRUE, control = CTL)
  vc <- as.data.frame(VarCorr(m4))
  fit <- rx_reml(y ~ 1, random = ~ g + gh, data = d, backend = BK, verbose = FALSE)
  controles_communs("emboite", m4, fit)
  verifier("emboite : variances g et h dans g",
           rel(fit$sigmas$g[1, 1], vc$vcov[vc$grp == "g"]) < 1e-4 &&
             rel(fit$sigmas$gh[1, 1], vc$vcov[vc$grp == "h:g"]) < 1e-4 &&
             rel(fit$sigma_res[1, 1], vc$vcov[vc$grp == "Residual"]) < 1e-4,
           sprintf("g %.5f/%.5f | h:g %.5f/%.5f | res %.5f/%.5f", fit$sigmas$g[1, 1],
                   vc$vcov[vc$grp == "g"], fit$sigmas$gh[1, 1], vc$vcov[vc$grp == "h:g"],
                   fit$sigma_res[1, 1], vc$vcov[vc$grp == "Residual"]))
  b_g <- ranef(m4)$g[levels(g), 1]; b_gh <- ranef(m4)$`h:g`[levels(gh), 1]
  verifier("emboite : BLUPs des deux termes",
           max(abs(as.numeric(fit$blups$g) - b_g)) < 1e-4 &&
             max(abs(as.numeric(fit$blups$gh) - b_gh)) < 1e-4,
           sprintf("ecart max %.2e / %.2e", max(abs(as.numeric(fit$blups$g) - b_g)),
                   max(abs(as.numeric(fit$blups$gh) - b_gh))))
})

# ==============================================================================
section("4. trois facteurs croises, dispositif desequilibre", {
# ==============================================================================
  set.seed(44); a <- 20; b <- 6; cc <- 5
  full <- expand.grid(g = seq_len(a), bl = seq_len(b), an = seq_len(cc))
  full <- full[sample(nrow(full), 400), ]                     # 400 des 600 cellules
  g <- factor(full$g); bl <- factor(full$bl); an <- factor(full$an); n <- nrow(full)
  x <- rnorm(n)
  y <- 5 + 0.4 * x + rnorm(a, 0, 1.3)[g] + rnorm(b, 0, 0.8)[bl] + rnorm(cc, 0, 0.5)[an] + rnorm(n, 0, 1.0)
  d <- data.frame(y = y, x = x, g = g, bl = bl, an = an)
  m4 <- lmer(y ~ x + (1 | g) + (1 | bl) + (1 | an), d, REML = TRUE, control = CTL)
  vc <- as.data.frame(VarCorr(m4))
  fit <- rx_reml(y ~ x, random = ~ g + bl + an, data = d, backend = BK, verbose = FALSE, pev = TRUE)
  controles_communs("3 croises", m4, fit)
  v_rx <- c(fit$sigmas$g[1, 1], fit$sigmas$bl[1, 1], fit$sigmas$an[1, 1], fit$sigma_res[1, 1])
  v_4 <- vc$vcov[match(c("g", "bl", "an", "Residual"), vc$grp)]
  verifier("3 croises : les quatre variances", max(rel(v_rx, v_4)) < 1e-4,
           sprintf("remlax %s | lme4 %s", paste(round(v_rx, 5), collapse = " "),
                   paste(round(v_4, 5), collapse = " ")))
  ecart_b <- max(sapply(c("g", "bl", "an"), function(f)
    max(abs(as.numeric(fit$blups[[f]]) - ranef(m4)[[f]][levels(d[[f]]), 1]))))
  verifier("3 croises : BLUPs des trois termes", ecart_b < 1e-4, sprintf("ecart max %.2e", ecart_b))
  # PEV du terme g par les MME, avec les trois termes dans Z
  Z <- cbind(as.matrix(inc(g)), as.matrix(inc(bl)), as.matrix(inc(an)))
  Ginv <- diag(c(rep(1 / v_rx[1], a), rep(1 / v_rx[2], b), rep(1 / v_rx[3], cc)))
  pv <- pev_main(model.matrix(~ x, d), Z, Ginv, v_rx[4])
  pev_rx <- c(as.numeric(fit$pev$g), as.numeric(fit$pev$bl), as.numeric(fit$pev$an))
  verifier("3 croises : PEV des trois termes = formule des MME",
           max(rel(pev_rx, pv$mme)) < 1e-4,
           sprintf("ecart rel max %.2e", max(rel(pev_rx, pv$mme))))
  verifier("3 croises : PEV de g decroit avec la replication",
           cor(as.numeric(fit$pev$g), as.numeric(table(g))) < 0,
           sprintf("cor(PEV, n_rep) = %.3f", cor(as.numeric(fit$pev$g), as.numeric(table(g)))))
})

# ==============================================================================
section("5. un facteur, tailles de groupe tres inegales, deux covariables", {
# ==============================================================================
  set.seed(45); q <- 40
  taille <- sample(1:12, q, replace = TRUE); n <- sum(taille)
  g <- factor(rep(seq_len(q), taille)); x1 <- rnorm(n); x2 <- runif(n)
  y <- 1 + 0.5 * x1 - 1.5 * x2 + rnorm(q, 0, 1.4)[g] + rnorm(n, 0, 1.0)
  d <- data.frame(y = y, x1 = x1, x2 = x2, g = g)
  m4 <- lmer(y ~ x1 + x2 + (1 | g), d, REML = TRUE, control = CTL)
  vc <- as.data.frame(VarCorr(m4))
  fit <- rx_reml(y ~ x1 + x2, random = ~ g, data = d, backend = BK, verbose = FALSE)
  controles_communs("desequilibre", m4, fit)
  verifier("desequilibre : composantes de variance",
           rel(fit$sigmas$g[1, 1], vc$vcov[vc$grp == "g"]) < 1e-4 &&
             rel(fit$sigma_res[1, 1], vc$vcov[vc$grp == "Residual"]) < 1e-4,
           sprintf("s2g %.6f/%.6f | s2e %.6f/%.6f", fit$sigmas$g[1, 1], vc$vcov[vc$grp == "g"],
                   fit$sigma_res[1, 1], vc$vcov[vc$grp == "Residual"]))
  verifier("desequilibre : BLUPs",
           max(abs(as.numeric(fit$blups$g) - ranef(m4)$g[levels(g), 1])) < 1e-4,
           sprintf("ecart max %.2e (tailles de 1 a 12)", max(abs(as.numeric(fit$blups$g) - ranef(m4)$g[levels(g), 1]))))
})

# ==============================================================================
cat("\n==============================================================\n")
if (length(ECHECS)) {
  cat("ECHECS (", length(ECHECS), ") :\n", sep = ""); for (e in ECHECS) cat("  -", e, "\n")
  quit(status = 1)
}
cat("TOUS LES CONTROLES PASSENT : memes -2logL, composantes, effets fixes, BLUP\n",
    "et PEV (formule des MME) que lme4 sur tous les dispositifs testes.\n")
