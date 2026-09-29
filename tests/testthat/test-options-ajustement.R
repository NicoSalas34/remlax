# Options de rx_fit / rx_reml et rx_predict, sur de petits plans simules.
# Tout s'ignore sans Python porteur de jax (helper-python.R). Les references
# sont en forme close : sur un plan en blocs complets equilibre, le REML
# retrouve l'analyse de variance, le Wald du traitement est le F de l'ANOVA et
# Kenward-Roger rend (t-1)(b-1) degres de liberte.

.rcbd <- function(seed = 1, t = 4L, b = 6L, sb = 0.7) {
  set.seed(seed)
  d <- expand.grid(trt = factor(LETTERS[1:t]), bloc = factor(seq_len(b)))
  d$x <- rnorm(nrow(d))
  d$y <- 5 + 0.5 * as.integer(d$trt) + 0.3 * d$x +
    rnorm(b, 0, sb)[as.integer(d$bloc)] + rnorm(nrow(d))
  d
}
.anova_rcbd <- function(d) {
  y <- d$y; t <- nlevels(d$trt); b <- nlevels(d$bloc)
  mt <- tapply(y, d$trt, mean); mb <- tapply(y, d$bloc, mean)
  sst <- b * sum((mt - mean(y))^2); ssb <- t * sum((mb - mean(y))^2)
  sse <- sum((y - mean(y))^2) - sst - ssb
  mse <- sse / ((t - 1) * (b - 1))
  list(F = (sst / (t - 1)) / mse, mse = mse, sb2 = (ssb / (b - 1) - mse) / t)
}

test_that("theta_init en evaluation seule, fixed_theta, hessian/blups coupes", {
  skip_si_pas_de_jax()
  d <- .rcbd()
  m <- rx_model(d$y, model.matrix(y ~ trt, d), list(rx_term("bloc", d$bloc)))
  f <- rx_fit(m, backend = "cpu", verbose = FALSE)
  expect_equal(rx_n_theta(m), length(f[["theta"]]))
  fe <- rx_fit(m, backend = "cpu", verbose = FALSE, theta_init = f[["theta"]],
               maxiter = 0, polish = 0)
  expect_identical(fe[["theta"]], f[["theta"]])
  expect_equal(fe[["logLik"]], f[["logLik"]], tolerance = 1e-10)
  expect_equal(fe[["n_iter"]], 0)
  expect_match(fe[["scipy_message"]], "evaluation seule")
  th0 <- f[["theta"]]; th0[1] <- th0[1] + 0.5
  ff <- rx_fit(m, backend = "cpu", verbose = FALSE, theta_init = th0, fixed_theta = 1L)
  expect_equal(ff[["theta"]][1], th0[1])
  expect_equal(ff[["n_fixed"]], 1); expect_equal(ff[["n_par_free"]], 1)
  expect_true(ff[["logLik"]] <= f[["logLik"]] + 1e-8)
  fh <- rx_fit(m, backend = "cpu", verbose = FALSE, hessian = FALSE, blups = FALSE)
  expect_null(fh[["hessian"]]); expect_length(fh[["blups"]], 0L); expect_null(fh[["beta"]])
  # NaN sort en null dans result.json, que jsonlite relit en NULL
  expect_true(is.null(fh[["newton_decrement"]]) || is.na(fh[["newton_decrement"]]))
  expect_true(is.finite(fh[["grad_proj_max"]]))
  fr <- rx_fit(m, backend = "cpu", verbose = FALSE, n_restarts = 1L)
  expect_equal(fr[["n_restarts"]], 1)
  expect_true(fr[["restart_gain"]] >= 0)
  expect_equal(fr[["logLik"]], f[["logLik"]], tolerance = 1e-6)
  # keep = TRUE : le repertoire survit, avec result.json
  dir <- tempfile("rx_keep_"); on.exit(unlink(dir, recursive = TRUE))
  rx_fit(m, backend = "cpu", verbose = FALSE, dir = dir, keep = TRUE, hessian = FALSE)
  expect_true(file.exists(file.path(dir, "result.json")))
  expect_s3_class(rx_read_result(dir), "rx_fit")
  expect_output(print(f), "Ajustement REML")
  expect_output(print(f), "Sigma\\[bloc\\]")
})

test_that("wald, kenward_roger et vpredict : formules exactes du plan equilibre", {
  skip_si_pas_de_jax()
  d <- .rcbd(); d$x <- NULL
  a <- .anova_rcbd(d)
  skip_if(a$sb2 <= 0, "variance de bloc ANOVA negative : REML != ANOVA")
  f <- rx_reml(y ~ trt, random = ~ bloc, data = d, backend = "cpu", verbose = FALSE,
               wald = TRUE, kenward_roger = TRUE,
               vpredict = c(icc = "V1/(V1+V2)", sd_e = "sqrt(V2)"))
  expect_equal(f[["sigma_res"]][1, 1], a$mse, tolerance = 1e-6)
  expect_equal(f[["sigmas"]][["bloc"]][1, 1], a$sb2, tolerance = 1e-5)
  w <- f[["wald"]][["tests"]]
  expect_equal(w$terme, c("(Intercept)", "trt"))
  expect_equal(w$ddl[2], 3)
  expect_equal(w$F[2], a$F, tolerance = 1e-6)
  expect_equal(w$p[2], pchisq(3 * a$F, 3, lower.tail = FALSE), tolerance = 1e-6)
  kr <- f[["kenward_roger"]]
  expect_true(isTRUE(kr[["disponible"]] == 1))
  expect_equal(kr[["tests"]]$denDF[2], 15, tolerance = 1e-8)
  expect_equal(kr[["tests"]]$F[2], a$F, tolerance = 1e-6)
  expect_equal(kr[["tests"]]$p[2], pf(a$F, 3, 15, lower.tail = FALSE), tolerance = 1e-6)
  expect_equal(kr[["se_beta"]][2], sqrt(2 * a$mse / 6), tolerance = 1e-6)
  expect_equal(dim(f[["vbeta_kr"]]), c(4L, 4L))
  vp <- f[["vpredict"]]
  expect_equal(vp[["composantes"]]$nom, c("bloc", "residuelle"))
  expect_equal(f[["composantes_noms"]], c("bloc", "residuelle"))
  icc <- vp[["predictions"]]
  v1 <- vp[["composantes"]]$valeur[1]; v2 <- vp[["composantes"]]$valeur[2]
  expect_equal(icc$valeur[icc$nom == "icc"], v1 / (v1 + v2), tolerance = 1e-8)
  expect_equal(icc$valeur[icc$nom == "sd_e"], sqrt(v2), tolerance = 1e-8)
  # delta method a la main : var(theta) = 2 H^-1, J par differences finies
  H <- f[["hessian"]]; th <- f[["theta"]]
  g <- function(t_) exp(2 * t_[1]) / (exp(2 * t_[1]) + exp(2 * t_[2]))
  J <- vapply(1:2, function(j) { e <- c(0, 0); e[j] <- 1e-6
    (g(th + e) - g(th - e)) / 2e-6 }, 1)
  expect_equal(icc$se[icc$nom == "icc"], sqrt(drop(t(J) %*% (2 * solve(H)) %*% J)),
               tolerance = 1e-4)
  expect_output(print(f), "vpredict")
  expect_output(print(f), "Wald")
})

test_that("rx_predict : classify, at, levels, average, sed, part aleatoire, vcov", {
  skip_si_pas_de_jax()
  d <- .rcbd()
  d <- d[-c(2, 7), ]                                    # desequilibre leger
  f <- rx_reml(y ~ trt + x, random = ~ bloc, data = d, backend = "cpu", verbose = FALSE,
               kenward_roger = TRUE)
  X <- f[["model"]]$X; beta <- f[["beta"]]; Vb <- f[["vbeta"]]
  # classify = trt : covariable a sa moyenne, L = [1, e_trt, mean(x)]
  p <- rx_predict(f, classify = "trt", sed = TRUE)
  expect_s3_class(p, "rx_predict")
  expect_equal(nrow(p), 4L); expect_true(all(p$estimable))
  L <- cbind(1, rbind(0, diag(3)), mean(d$x))
  expect_equal(p$predicted.value, as.numeric(L %*% beta), tolerance = 1e-8)
  expect_equal(p$std.error, sqrt(diag(L %*% Vb %*% t(L))), tolerance = 1e-8)
  S <- attr(p, "sed")
  expect_equal(dim(S), c(4L, 4L)); expect_true(all(is.na(diag(S))))
  Cv <- L %*% Vb %*% t(L)
  expect_equal(S[1, 2], sqrt(Cv[1, 1] + Cv[2, 2] - 2 * Cv[1, 2]), tolerance = 1e-8)
  expect_equal(attr(p, "sed.moyen"), sqrt(mean(S[upper.tri(S)]^2)), tolerance = 1e-10)
  expect_output(print(p), "erreur-type moyenne des differences")
  # at : covariable forcee a 0 ; levels : sous-ensemble
  p0 <- rx_predict(f, classify = "trt", at = list(x = 0))
  expect_equal(p0$predicted.value, as.numeric(cbind(1, rbind(0, diag(3)), 0) %*% beta),
               tolerance = 1e-8)
  p2 <- rx_predict(f, classify = "trt", levels = list(trt = c("A", "C")))
  expect_equal(as.character(p2$trt), c("A", "C"))
  expect_equal(p2$predicted.value, p$predicted.value[c(1, 3)], tolerance = 1e-10)
  # classify = x (covariable) : une ligne par valeur distincte, trt moyenne
  px <- rx_predict(f, classify = "x", levels = list(x = c(-1, 1)))
  expect_equal(nrow(px), 2L)
  expect_equal(px$predicted.value, as.numeric(cbind(1, matrix(0.25, 2, 3), c(-1, 1)) %*% beta),
               tolerance = 1e-8)
  # average = "proportional" : poids = effectifs observes des traitements
  pp <- rx_predict(f, classify = "x", levels = list(x = 0), average = "proportional")
  wt <- as.numeric(table(d$trt)); wt <- wt / sum(wt)
  expect_equal(pp$predicted.value, sum(c(1, wt[-1], 0) * beta), tolerance = 1e-8)
  # weights : reponderation explicite des cellules du classify
  pw <- rx_predict(f, classify = "x", levels = list(x = 0), weights = c("0" = 2))
  expect_equal(pw$predicted.value, pp$predicted.value * 0 + sum(c(1, rep(0.25, 3), 0) * beta),
               tolerance = 1e-8)
  # part aleatoire : moyenne par bloc = fixe + BLUP ; erreur de PREDICTION
  pb <- rx_predict(f, classify = "bloc", backend = "cpu")
  pf_ <- rx_predict(f, classify = "bloc", include_random = FALSE)
  expect_equal(nrow(pb), 6L)
  expect_equal(pb$predicted.value - pf_$predicted.value, as.numeric(f[["blups"]][["bloc"]]),
               tolerance = 1e-6)
  expect_false(isTRUE(all.equal(pb$std.error, pf_$std.error)))
  expect_equal(pf_$predicted.value, rep(pf_$predicted.value[1], 6), tolerance = 1e-12)
  # vcov = kenward-roger : covariance ajustee, avertissement avec part aleatoire
  pk <- rx_predict(f, classify = "trt", vcov = "kenward-roger")
  expect_equal(pk$std.error, sqrt(diag(L %*% f[["vbeta_kr"]] %*% t(L))), tolerance = 1e-8)
  expect_warning(rx_predict(f, classify = "bloc", vcov = "kenward-roger", backend = "cpu"),
                 "FIXES")
  f2 <- rx_reml(y ~ trt + x, random = ~ bloc, data = d, backend = "cpu", verbose = FALSE)
  expect_error(rx_predict(f2, classify = "trt", vcov = "kenward-roger"), "kenward_roger")
  # refus : variable inconnue, ajustement sans formule
  expect_error(rx_predict(f, classify = "zzz"), "ni dans le modele")
  fm <- rx_fit(f[["model"]], backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_error(rx_predict(fm, classify = "trt"), "rx_reml")
})

test_that("dsum, produit separable, spline 2D, pev, multi-caractere", {
  skip_si_pas_de_jax()
  # dsum : deux sites, une residuelle chacun ; la section nomme ses sorties
  set.seed(7)
  d <- data.frame(gid = factor(rep(1:15, 6)), site = factor(rep(c("S1", "S2"), each = 45)))
  d$y <- rnorm(15, 0, 1)[d$gid] + rnorm(90, 0, ifelse(d$site == "S1", 0.5, 1.5))
  f <- rx_reml(y ~ site, random = ~ gid, residual = ~ dsum(~ units | site), data = d,
               backend = "cpu", verbose = FALSE, pev = TRUE)
  expect_equal(f[["n_par"]], 3L)
  expect_equal(names(f[["sigmas_res"]]), c("S1", "S2"))
  expect_true(f[["sigmas_res"]][["S1"]][1, 1] < f[["sigmas_res"]][["S2"]][1, 1])
  expect_equal(f[["composantes_noms"]], c("gid", "S1", "S2"))
  expect_equal(dim(f[["pev"]][["gid"]]), c(15L, 1L))
  expect_true(all(f[["pev"]][["gid"]] > 0 & f[["pev"]][["gid"]] < f[["sigmas"]][["gid"]][1, 1]))
  fn <- rx_reml(y ~ site, random = ~ gid, residual = ~ dsum(~ units | site), data = d,
                backend = "cpu", verbose = FALSE, pev = "gid", hessian = FALSE)
  expect_equal(names(fn[["pev"]]), "gid")
  # produit separable id (x) ar1 (x) ar1 : 3 parametres, deux rho nommes
  nb <- 3L; nr <- 4L; nc <- 5L; q <- nb * nr * nc; n <- 2L * q
  Z <- Matrix::sparseMatrix(i = seq_len(n), j = rep(seq_len(q), 2L), x = 1, dims = c(n, q))
  tm <- rx_term("champ", Z, struct = "iid", t = 1L, level = "sep",
                parts = list(list("id", nb), list("ar1", nc), list("ar1", nr)))
  ms <- rx_model(rnorm(n), matrix(1, n, 1), list(tm))
  expect_equal(rx_n_theta(ms), 4L)
  fs <- rx_fit(ms, backend = "cpu", verbose = FALSE, hessian = FALSE, blups = FALSE, maxiter = 60L)
  expect_length(fs[["theta"]], 4L)
  expect_equal(sort(grep("^champ!ar1_", names(fs[["rho"]]), value = TRUE)),
               c("champ!ar1_2_phi", "champ!ar1_3_phi"))
  expect_equal(fs[["rho"]][["champ!ar1_2_phi"]], tanh(fs[["theta"]][2]), tolerance = 1e-8)
  # spline 2D : genotype, cinq composantes PS-ANOVA et la residuelle
  g <- expand.grid(r = 1:10, c = 1:9); g$gid <- factor(rep(1:10, 9))
  g$y <- 0.05 * (g$r - 5)^2 + rnorm(10)[g$gid] + rnorm(90, 0, 0.5)
  sp <- rx_spl2d(g$r, g$c, nseg = c(4, 4))
  msp <- rx_model(g$y, cbind("(Intercept)" = 1, sp$X), c(list(rx_term("gid", g$gid)), sp$terms))
  fsp <- rx_fit(msp, backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_equal(fsp[["n_par"]], 7L)
  expect_equal(names(fsp[["sigmas"]]), c("gid", paste0("spl_", c("fx", "fy", "fx_y", "x_fy", "fx_fy"))))
  expect_true(is.finite(fsp[["logLik"]]))
  # multi-caractere : formule et voie explicite donnent la meme vraisemblance
  set.seed(9)
  dl <- expand.grid(unite = factor(1:40), trait = factor(c("t1", "t2")))
  dl$gid <- factor(rep(1:20, 2))[as.integer(dl$unite)]
  dl$y <- rnorm(80) + rnorm(20)[dl$gid] * ifelse(dl$trait == "t1", 1, 0.5)
  ff <- rx_reml(y ~ trait, random = ~ us(gid), residual = ~ us(trait):units, data = dl,
                trait = "trait", unit = "unite", backend = "cpu", verbose = FALSE,
                hessian = FALSE, blups = FALSE)
  expect_equal(ff[["n_par"]], 6L)
  expect_equal(ff[["composantes_noms"]][1:3], c("gid[1,1]", "gid[2,1]", "gid[2,2]"))
  Zl <- lapply(levels(dl$trait), function(tt) { ix <- which(dl$trait == tt)
    Matrix::sparseMatrix(i = ix, j = as.integer(dl$gid)[ix], x = 1, dims = c(80, 20)) })
  me <- rx_model(dl$y, model.matrix(y ~ trait, dl), list(rx_term("gid", Zl, struct = "us")),
                 rx_residual("us", trait = dl$trait, unit = dl$unite))
  fe <- rx_fit(me, backend = "cpu", verbose = FALSE, hessian = FALSE, blups = FALSE)
  expect_equal(fe[["logLik"]], ff[["logLik"]], tolerance = 1e-8)
  expect_equal(dim(ff[["sigmas"]][["gid"]]), c(2L, 2L))
  # theta_init impose sur ce modele : evaluation exacte, longueur par rx_n_theta
  th <- rep(0, rx_n_theta(me))
  f0 <- rx_fit(me, backend = "cpu", verbose = FALSE, theta_init = th, maxiter = 0, polish = 0,
               hessian = FALSE, blups = FALSE)
  expect_identical(f0[["theta"]], th)
  expect_true(f0[["logLik"]] < fe[["logLik"]])
})
