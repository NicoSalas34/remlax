# Tests de rx_cor_z, rx_sigmas_from_theta, rx_ratios, rx_se_theta et
# rx_grid_summary. Pur R sauf les deux comparaisons au solveur (skip sans jax).
# La validation sur mvC7 vit dans validation/ch3_ratios_vs_ige.R.

# ---- rx_cor_z ------------------------------------------------------------------
test_that("rx_cor_z : symetrie a r = 0, exemples de la table, bornes, Pearson (T1-T5)", {
  z <- rx_cor_z(0, se = 0.1)
  expect_equal(z$ci_high, -z$ci_low)
  expect_equal(z$ci_high, tanh(stats::qnorm(0.975) * 0.1), tolerance = 1e-6)
  expect_equal(z$ci_high, 0.1935, tolerance = 1e-3)
  z3 <- rx_cor_z(c(-0.937, 0.979, -0.153), se = c(0.076, 0.022, 1.093))
  expect_equal(round(z3$ci_low[1:2], 3), c(-0.994, 0.844))
  expect_equal(round(z3$ci_high[1:2], 3), c(-0.456, 0.997))
  expect_gt(z3$width[3], 1.5)
  expect_equal(z3$informative, c(TRUE, TRUE, FALSE))
  expect_equal(z3$z, z3$r / z3$se)
  z1 <- rx_cor_z(0.999999, se = 0.01)
  expect_true(all(is.finite(c(z1$ci_low, z1$ci_high, z1$se_z))))
  zp <- rx_cor_z(0.3, n = 181)
  expect_equal(zp$se_z, 1 / sqrt(178))
  x <- rnorm(181); y <- 0.3 * x + rnorm(181)
  ct <- stats::cor.test(x, y)
  zc <- rx_cor_z(unname(ct$estimate), n = 181)
  expect_equal(c(zc$ci_low, zc$ci_high), as.numeric(ct$conf.int), tolerance = 1e-6)
  expect_equal(zc$method, "pearson")
  zn <- rx_cor_z(0.5, se = 0.2, width_max = NULL)
  expect_true(is.na(zn$informative))
  expect_error(rx_cor_z(0.5), "fournir")
  expect_error(rx_cor_z(0.5, se = 0.1, n = 10), "fournir")
})

# ---- carte theta -> Sigma ------------------------------------------------------
test_that("rx_sigma_of : us ligne par ligne, iid, diag, fa, rr, chol, ante, corh", {
  th <- c(log(1.5), 0.4, log(0.8), -0.3, 0.2, log(2))
  S <- rx_sigma_of(th, "us", 3)
  L <- matrix(0, 3, 3); L[1, 1] <- 1.5; L[2, 1] <- 0.4; L[2, 2] <- 0.8
  L[3, 1] <- -0.3; L[3, 2] <- 0.2; L[3, 3] <- 2
  expect_equal(S, L %*% t(L))
  expect_equal(rx_sigma_of(log(0.7), "iid", 2), 0.49 * diag(2))
  expect_equal(rx_sigma_of(c(log(0.7), log(2)), "diag", 2), diag(c(0.49, 4)))
  # fa rang 1 sur t = 2 : Lambda = (l1, l2)', psi = exp(p)
  thf <- c(0.5, -0.2, log(0.3), log(0.6))
  expect_equal(rx_sigma_of(thf, "fa", 2, 1),
               c(0.5, -0.2) %o% c(0.5, -0.2) + diag(c(0.09, 0.36)))
  expect_equal(rx_sigma_of(c(0.5, -0.2), "rr", 2, 1), c(0.5, -0.2) %o% c(0.5, -0.2))
  # chol bande 1 sur t = 2 : L = [[1,0],[a,1]], D = d^2
  thc <- c(0.3, log(1.2), log(0.5))
  Lc <- matrix(c(1, 0.3, 0, 1), 2)
  expect_equal(rx_sigma_of(thc, "chol", 2, 1), Lc %*% diag(c(1.44, 0.25)) %*% t(Lc))
  # ante : Sigma^-1 = U D U' avec U = Lc' (unitriangulaire superieure), D = d^2
  expect_equal(solve(rx_sigma_of(thc, "ante", 2, 1)), t(Lc) %*% diag(c(1.44, 0.25)) %*% Lc)
  # corh : correlation uniforme dans (-1/(t-1), 1)
  Sc <- rx_sigma_of(c(log(1), log(2), 0), "corh", 2)
  expect_equal(diag(Sc), c(1, 4))
  expect_equal(Sc[1, 2] / 2, 0)   # tanh(0) = 0 -> r = milieu de (-1, 1) = 0
  expect_error(rx_sigma_of(1, "inconnue", 1), "non prise en charge")
})

test_that("rx_sigmas_from_theta : ordre du solveur (termes puis sections) et dimnames", {
  n <- 12
  Zd <- Matrix::sparseMatrix(i = 1:n, j = rep(1:3, 4), x = 1, dims = c(n, 3))
  tm <- rx_term("g", list(D = Zd, W = Zd * 0.5), struct = "us")
  ar <- rx_term("col", factor(rep(1:4, each = 3)), level = "ar1")
  res <- rx_residual("us", trait = factor(rep(c("t1", "t2"), 6)), unit = rep(1:6, each = 2))
  mod <- rx_model(rnorm(n), cbind(1, rep(0:1, 6)), list(tm, ar), res)
  expect_equal(mod$n_par, 3L + 2L + 3L)
  th <- c(log(1.1), 0.2, log(0.7), log(0.4), 0.3, log(0.9), -0.1, log(1.3))
  S <- rx_sigmas_from_theta(th, mod)
  expect_equal(names(S), c("g", "col", "residual"))
  expect_equal(dimnames(S$g), list(c("D", "W"), c("D", "W")))
  expect_equal(S$g, rx_sigma_of(th[1:3], "us", 2), ignore_attr = TRUE)
  expect_equal(unname(S$col), 0.16 * diag(1))       # le rho n'est pas une composante
  expect_equal(dimnames(S$residual), list(c("t1", "t2"), c("t1", "t2")))
  expect_equal(S$residual, rx_sigma_of(th[6:8], "us", 2), ignore_attr = TRUE)
  expect_error(rx_sigmas_from_theta(th[-1], mod), "attend")
})

# ---- rx_ratios sur un modele a un terme et une residuelle, Hessien fourni -----
# Le Hessien est celui de -2 logL : on en fabrique un defini positif pour tester
# l'algebre de la methode delta sans solveur.
faux_fit <- function(theta, H, model, floor = -12, ceil = 12, fixed = integer(0)) {
  structure(list(theta = theta, hessian = H, par_floor = floor, par_ceil = ceil,
                 fixed_theta = fixed, model = model,
                 se_theta = rx_se_theta(theta, H, floor, ceil, fixed),
                 sigmas = rx_sigmas_from_theta(theta, model)[vapply(model$terms, `[[`, "", "name")]),
            class = "rx_fit")
}
modele_jouet <- function() {
  set.seed(11)
  n <- 40; q <- 8
  g <- rep(seq_len(q), 5)
  Zd <- Matrix::sparseMatrix(i = seq_len(n), j = g, x = 1, dims = c(n, q))
  Zn <- Matrix::Matrix(matrix(rpois(n * q, 0.4), n, q), sparse = TRUE)
  tm <- rx_term("gen", list(D = Zd, N = Zn), struct = "us")
  bl <- rx_term("bloc", factor(rep(1:5, each = q)))
  rx_model(rnorm(n), cbind(rep(1, n)), list(tm, bl))
}

test_that("rx_ratios : SE d'une variance = 2 var se_theta ; parts a 1 ; invariance (T4, T5)", {
  mod <- modele_jouet()
  th <- c(log(1.2), 0.3, log(0.5), log(0.4), log(0.9))
  set.seed(2); A <- matrix(rnorm(25), 5); H <- crossprod(A) + diag(5) * 20
  fit <- faux_fit(th, H, mod)
  comp <- data.frame(target = "y", direct = "gen:D", indirect_within = "gen:N", other = "bloc+residual")
  r0 <- rx_ratios(fit, comp)
  expect_s3_class(r0, "rx_ratios")
  expect_false(r0$scaled[1])
  # variance scalaire (bloc) : SE = 2 var se_theta
  vb <- r0[r0$quantity == "var" & r0$component == "bloc", ]
  expect_equal(vb$estimate, exp(2 * th[4]))
  expect_equal(vb$se, 2 * exp(2 * th[4]) * fit$se_theta[4], tolerance = 1e-6)
  # parts qui somment a 1
  expect_equal(sum(r0$estimate[r0$quantity == "share"]), 1, tolerance = 1e-12)
  # h2_ext_total = h2_ext_within + h2_between (pas de between ici)
  h <- function(q, c) r0$estimate[r0$quantity == q & r0$component == c]
  expect_equal(h("h2", "h2_ext_total"), h("h2", "h2_ext_within"))
  # exposure = NULL contre exposure a 1 : memes nombres, colonne scaled differente
  ex1 <- data.frame(target = "y", d = 1, k_within = 1, c = 1, S_within = 1)
  r1 <- rx_ratios(fit, comp, exposure = ex1)
  expect_true(r1$scaled[1])
  expect_equal(r1$estimate, r0$estimate); expect_equal(r1$se, r0$se)
  # invariance de r_direct_indirect a la mise a l'echelle ; le cov prend c
  ex2 <- data.frame(target = "y", d = 1.9, k_within = 0.3, c = -0.05, S_within = 4)
  r2 <- rx_ratios(fit, comp, exposure = ex2)
  g2 <- function(q, c) r2$estimate[r2$quantity == q & r2$component == c]
  expect_equal(g2("h2", "r_direct_indirect"), h("h2", "r_direct_indirect"))
  expect_equal(g2("var", "direct"), 1.9 * h("var", "direct"))
  expect_equal(g2("var", "indirect_within"), 0.3 * h("var", "indirect_within"))
  expect_equal(g2("var", "cov_direct_indirect"), -0.05 * h("var", "cov_direct_indirect"))
  S <- rx_sigmas_from_theta(th, mod)$gen
  expect_equal(g2("tbv_var", "own:y"), 1.9 * (S[1, 1] + 2 * 4 * S[1, 2] + 16 * S[2, 2]))
  expect_equal(g2("tau2", "own:y"), g2("tbv_var", "own:y") / g2("var", "phenotypic"))
  # la constante multiplie la quantite et sa SE a l'identique
  s2 <- function(q, c) r2$se[r2$quantity == q & r2$component == c]
  s0 <- function(q, c) r0$se[r0$quantity == q & r0$component == c]
  expect_equal(s2("var", "direct"), 1.9 * s0("var", "direct"), tolerance = 1e-8)
  expect_equal(attr(r2, "check_se"), 1, tolerance = 1e-12)
  expect_output(print(r2), "rx_ratios")
})

test_that("rx_ratios : controle theta -> Sigma, references illisibles, jacobian solver", {
  mod <- modele_jouet()
  th <- c(log(1.2), 0.3, log(0.5), log(0.4), log(0.9))
  fit <- faux_fit(th, diag(5) * 30, mod)
  comp <- data.frame(target = "y", direct = "gen:D", indirect_within = "gen:N", other = "bloc+residual")
  fit_perm <- fit; fit_perm$theta <- th[c(2, 1, 3, 4, 5)]
  expect_error(rx_ratios(fit_perm, comp), "differe de fit\\$sigmas")
  expect_error(rx_ratios(fit, data.frame(target = "y", direct = "gen:Q")), "absente")
  expect_error(rx_ratios(fit, data.frame(target = "y", direct = "xx")), "aucun terme")
  expect_error(rx_ratios(fit, data.frame(target = "y", direct = "gen")), "en nommer une")
  expect_error(rx_ratios(fit, comp, jacobian = "solver"), "pas implemente")
  expect_error(rx_ratios(fit, data.frame(target = "y", direct = "gen:D", indirect_within = "bloc")),
               "meme terme")
  # references par indice et section residuelle nommee
  r <- rx_ratios(fit, data.frame(target = "y", direct = "gen[1]", indirect_within = "gen[2]",
                                 other = "bloc[1]+residual[1]"), quantities = "variances")
  expect_equal(nrow(r), 7L)
})

test_that("rx_ratios : parametre a la borne -> hors sous-espace, COND_BOUND, se_theta NA (T6)", {
  mod <- modele_jouet()
  th <- c(log(1.2), 0.3, log(0.5), -12, log(0.9))     # bloc au plancher
  set.seed(3); A <- matrix(rnorm(25), 5); H <- crossprod(A) + diag(5) * 20
  fit <- faux_fit(th, H, mod)
  expect_true(is.na(fit$se_theta[4])); expect_true(all(is.finite(fit$se_theta[-4])))
  comp <- data.frame(target = "y", direct = "gen:D", indirect_within = "gen:N", other = "bloc+residual")
  r <- rx_ratios(fit, comp)
  expect_equal(attr(r, "free"), c(1L, 2L, 3L, 5L))
  vb <- r[r$quantity == "var" & r$component == "bloc", ]
  expect_equal(vb$flag, "FLOOR")
  # la part du bloc depend entierement du parametre borne : COND_BOUND
  sb <- r[r$quantity == "share" & r$component == "bloc", ]
  expect_equal(sb$flag, "COND_BOUND"); expect_gt(sb$dep_bound, 0.05)
  # une quantite qui n'en depend presque pas reste OK
  expect_equal(r$flag[r$quantity == "h2" & r$component == "r_direct_indirect"], "OK")
  # fixed_theta sort aussi du sous-espace
  fit2 <- faux_fit(c(log(1.2), 0.3, log(0.5), log(0.4), log(0.9)), H, mod, fixed = 2L)
  expect_true(is.na(fit2$se_theta[2]))
  r2 <- rx_ratios(fit2, comp)
  expect_equal(attr(r2, "free"), c(1L, 3L, 4L, 5L))
})

test_that("rx_ratios : courbure negative -> refuse (NA) ou projette (NOT_IDENTIFIED) (T7)", {
  mod <- modele_jouet()
  th <- c(log(1.2), 0.3, log(0.5), log(0.4), log(0.9))
  H <- diag(c(30, 40, 50, -5, 60))                     # direction 4 (bloc) de courbure negative
  fit <- faux_fit(th, H, mod)
  expect_true(all(is.na(fit$se_theta)))                # rx_se_theta refuse un point de selle
  comp <- data.frame(target = "y", direct = "gen:D", indirect_within = "gen:N", other = "bloc+residual")
  rr <- rx_ratios(fit, comp, curvature = "refuse")
  expect_true(all(is.na(rr$se)))
  expect_true(all(rr$flag %in% c("NO_HESSIAN", "NOT_ESTIMATED")))
  expect_message(rp <- rx_ratios(fit, comp, curvature = "project"), "courbure negative")
  expect_true(all(is.finite(rp$se[rp$quantity == "var" & rp$component == "direct"])))
  vb <- rp[rp$quantity == "var" & rp$component == "bloc", ]
  expect_equal(vb$flag, "NOT_IDENTIFIED"); expect_equal(vb$dep_excluded, 1)
  expect_equal(vb$se, 0)
  # sans Hessien : estimations rendues, SE NA, drapeau NO_HESSIAN
  fit0 <- fit; fit0$hessian <- NULL
  r0 <- rx_ratios(fit0, comp)
  expect_true(all(is.na(r0$se))); expect_true(all(r0$flag == "NO_HESSIAN"))
  expect_equal(r0$estimate, rp$estimate)
})

test_that("rx_ratios : deux groupes, effet croise, tau2 au denominateur de la cible recevante", {
  set.seed(12)
  n <- 30; qa <- 5; qb <- 4
  ZdA <- Matrix::sparseMatrix(i = 1:n, j = rep(1:qa, 6), x = 1, dims = c(n, qa))
  ZnA <- Matrix::Matrix(matrix(rpois(n * qa, 0.5), n, qa), sparse = TRUE)
  ZnB <- Matrix::Matrix(matrix(rpois(n * qb, 0.5), n, qb), sparse = TRUE)
  tA <- rx_term("gA", list(D_y = ZdA, N_y = ZnA), struct = "us")
  tB <- rx_term("gB", list(N_on_y = ZnB), struct = "us")
  mod <- rx_model(rnorm(n), cbind(rep(1, n)), list(tA, tB))
  th <- c(log(1), 0.2, log(0.6), log(0.3), log(0.8))
  fit <- faux_fit(th, diag(5) * 25, mod)
  comp <- data.frame(target = "y", direct = "gA:D_y", indirect_within = "gA:N_y",
                     indirect_between = "gB:N_on_y", other = "residual")
  ex <- data.frame(target = "y", d = 2, k_within = 0.5, k_between = 0.25, c = 0.1,
                   S_within = 3, S_between = 2)
  r <- rx_ratios(fit, comp, exposure = ex)
  g <- function(q, tg, c) r$estimate[r$quantity == q & r$target == tg & r$component == c]
  S <- rx_sigmas_from_theta(th, mod)
  VP <- 2 * S$gA[1, 1] + 0.5 * S$gA[2, 2] + 0.25 * S$gB[1, 1] + S$residual[1, 1] + 2 * 0.1 * S$gA[1, 2]
  expect_equal(g("var", "y", "phenotypic"), VP)
  expect_equal(g("h2", "y", "h2_indirect_between"), 0.25 * S$gB[1, 1] / VP)
  expect_equal(g("h2", "y", "h2_ext_total"), g("h2", "y", "h2_ext_within") + g("h2", "y", "h2_indirect_between"))
  # TBV croise : sqrt(d_emetteur) S_between ; sans cible directe dans gB, d_emetteur
  # est la moyenne des d des emetteurs de gB, ici aucun -> NaN refuse par la carte ?
  # Ici gB n'a pas d'effet direct : d_em = mean(numeric(0)) = NaN. On le verifie.
  expect_true(is.nan(g("tbv_var", "gB", "cross:y")) || is.na(g("tbv_var", "gB", "cross:y")))
  # avec un emetteur declare dans gB, le croise prend son d
  tB2 <- rx_term("gB", list(D_z = ZdA[, 1:4], N_on_y = ZnB), struct = "us")
  mod2 <- rx_model(rnorm(n), cbind(rep(1, n)), list(tA, tB2))
  th2 <- c(th[1:3], log(0.7), 0.1, log(0.3), log(0.8))
  fit2 <- faux_fit(th2, diag(7) * 25, mod2)
  comp2 <- rbind(comp, data.frame(target = "z", direct = "gB:D_z", indirect_within = NA,
                                  indirect_between = NA, other = "residual"))
  ex2 <- rbind(ex, data.frame(target = "z", d = 1.5, k_within = 1, k_between = 1, c = 1,
                              S_within = 1, S_between = 1))
  r2 <- rx_ratios(fit2, comp2, exposure = ex2)
  S2 <- rx_sigmas_from_theta(th2, mod2)
  vx <- r2$estimate[r2$quantity == "tbv_var" & r2$target == "gB" & r2$component == "cross:y"]
  expect_equal(vx, 1.5 * 4 * S2$gB[2, 2])
  VPy <- r2$estimate[r2$quantity == "var" & r2$target == "y" & r2$component == "phenotypic"]
  expect_equal(r2$estimate[r2$quantity == "tau2" & r2$target == "gB" & r2$component == "cross:y"], vx / VPy)
  expect_true(any(r2$quantity == "tbv_cor" & r2$component == "own:z~cross:y"))
})

# ---- contre le solveur (skip sans jax) ------------------------------------------
test_that("rx_ratios et se_theta egalent vpredict du solveur sur un jouet (T1, T8)", {
  skip_si_pas_de_jax()
  set.seed(21)
  n <- 90; q <- 15
  gid <- factor(rep(sprintf("g%02d", 1:q), 6)); bloc <- factor(rep(1:6, each = q))
  y <- 1 + rnorm(q, sd = 1)[gid] + rnorm(6, sd = 0.4)[bloc] + rnorm(n, sd = 0.8)
  mod <- rx_model(y, cbind(rep(1, n)), list(rx_term("gid", gid), rx_term("bloc", bloc)))
  fit <- rx_fit(mod, backend = "cpu", verbose = FALSE,
                vpredict = c(h2 = "V1/(V1+V2+V3)", vg = "V1"))
  vp <- fit[["vpredict"]][["predictions"]]
  expect_equal(length(fit[["se_theta"]]), 3L)
  expect_true(all(is.finite(fit[["se_theta"]])))
  # SE(vg) du solveur = 2 vg se_theta[1] (derivee de exp(2 theta))
  expect_equal(vp$se[vp$nom == "vg"], 2 * exp(2 * fit$theta[1]) * fit$se_theta[1], tolerance = 1e-8)
  r <- rx_ratios(fit, data.frame(target = "y", direct = "gid", other = "bloc+residual"),
                 model = mod, quantities = "h2")
  expect_equal(r$estimate[r$component == "h2"], vp$valeur[vp$nom == "h2"], tolerance = 1e-10)
  expect_equal(r$se[r$component == "h2"], vp$se[vp$nom == "h2"], tolerance = 1e-6)
  expect_equal(attr(r, "check_se"), 1, tolerance = 1e-12)
})

test_that("rx_term(colnames) : dimnames sur fit$sigmas et fit$blups, colnames refuses", {
  skip_si_pas_de_jax()
  set.seed(22)
  n <- 60; q <- 10
  g <- rep(1:q, 6)
  Zd <- Matrix::sparseMatrix(i = 1:n, j = g, x = 1, dims = c(n, q), dimnames = list(NULL, paste0("g", 1:q)))
  Zn <- Matrix::Matrix(matrix(rpois(n * q, 0.3), n, q), dimnames = list(NULL, paste0("g", 1:q)), sparse = TRUE)
  y <- rnorm(q)[g] + rnorm(n)
  tm <- rx_term("gen", list(direct = Zd, neigh = Zn), struct = "us")
  expect_error(rx_term("gen", list(Zd, Zn), struct = "us", colnames = "a"), "noms distincts")
  mod <- rx_model(y, cbind(rep(1, n)), list(tm))
  fit <- rx_fit(mod, backend = "cpu", verbose = FALSE)
  expect_equal(dimnames(fit$sigmas$gen), list(c("direct", "neigh"), c("direct", "neigh")))
  expect_equal(colnames(fit$blups$gen), c("direct", "neigh"))
  expect_equal(rownames(fit$blups$gen), paste0("g", 1:q))
  e <- rx_exposure(mod, direct = "gen:direct", indirect = "gen:neigh", rows = "auto")
  expect_equal(e$d, 1)
})

# ---- rx_grid_summary ------------------------------------------------------------
grille_jouet <- function() {
  g <- expand.grid(r1 = 1:3, r2 = 1:3)
  g$logLik <- -c(10, 9.2, 9.6, 8.9, 8.0, 8.5, 9.9, 8.7, 9.4)
  g$n_par <- 4L; g$AIC <- 2 * g$n_par - 2 * g$logLik
  g$pd_hessian <- c(TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE)
  g$n_at_bound <- c(0, 0, 1, 0, 0, 0, 2, 0, 0)
  g$n_par_free <- 4L - g$n_at_bound
  g
}

test_that("rx_grid_summary : best, supported, ranges, manquants, best_pd (T1)", {
  g <- grille_jouet()
  s <- rx_grid_summary(g, coords = c("r1", "r2"))
  expect_s3_class(s, "rx_grid_summary")
  expect_equal(s$best$r1, 2); expect_equal(s$best$r2, 2)       # logLik -8.0
  # soutenu : AIC - min <= 2  <=> logLik >= -9.0 : (2,2) -8.0, (3,2) -8.5, (2,3) -8.7, (1,2) -8.9
  expect_equal(s$n_supported$n_supported, 4L)
  expect_equal(s$ranges$min[s$ranges$coord == "r1"], 1); expect_equal(s$ranges$max[s$ranges$coord == "r1"], 3)
  expect_equal(s$ranges$min[s$ranges$coord == "r2"], 2); expect_equal(s$ranges$max[s$ranges$coord == "r2"], 3)
  expect_equal(s$n_supported$n_product, 3 * 2)
  expect_false(s$n_supported$n_supported == s$n_supported$n_product)
  expect_equal(s$counts$n_pd_false, 1L); expect_equal(s$counts$n_at_bound_pos, 2L)
  expect_equal(s$counts$n_missing, 0L)
  expect_equal(s$best_pd$r1, 3); expect_equal(s$best_pd$r2, 2)
  expect_equal(s$best_pd$delta_pd, 1, tolerance = 1e-12); expect_false(s$best_pd$same_as_best)
  s2 <- rx_grid_summary(g[-5, ], coords = c("r1", "r2"))
  expect_equal(s2$counts$n_missing, 1L)
  expect_equal(s2$best$r1, 3)
  expect_output(print(s), "rx_grid_summary")
})

test_that("rx_grid_summary : by, AIC recalcule ou controle, effective (T2, T3, T4)", {
  g <- grille_jouet()
  g2 <- rbind(cbind(g, trait = "a", n_obs = 100L),
              cbind(transform(g, logLik = logLik - 50, AIC = AIC + 100), trait = "b", n_obs = 80L))
  s <- rx_grid_summary(g2, coords = c("r1", "r2"), by = "trait")
  expect_equal(nrow(s$best), 2L); expect_equal(s$best$r1, c(2, 2))
  expect_equal(s$n_supported$n_supported, c(4L, 4L))
  expect_error(rx_grid_summary(g2, coords = c("r1", "r2")), "fournir `by`")
  g3 <- g; g3$AIC <- NULL
  s3 <- rx_grid_summary(g3, coords = c("r1", "r2"))
  expect_equal(s3$best$AIC, 8 - 2 * (-8.0))
  g4 <- g; g4$AIC[1] <- g4$AIC[1] + 1
  expect_error(rx_grid_summary(g4, coords = c("r1", "r2")), "differe")
  expect_error(rx_grid_summary(g3[, -3], coords = c("r1", "r2")), "ni")
  # effective : la cellule (1,3) a n_par_free 2 -> AIC_eff = 4 + 19.8 = 23.8 contre (2,2) 8 + 16 = 24
  se <- rx_grid_summary(g, coords = c("r1", "r2"), effective = TRUE)
  expect_equal(se$effective$best$r1, 1); expect_equal(se$effective$best$r2, 3)
  expect_true(se$effective$best_moved$best_moved)
  expect_error(rx_grid_summary(g, coords = c("r1", "zz")), "absente")
})
