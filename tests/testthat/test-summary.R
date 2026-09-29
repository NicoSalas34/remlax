# summary() d'un ajustement, a la maniere de summary.asreml.

skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")

test_that("summary : table des composantes, effets fixes, criteres", {
  set.seed(1)
  d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
  d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
  m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
  f <- rx_fit(m, backend = "cpu", verbose = FALSE)
  s <- summary(f)
  expect_s3_class(s, "summary.rx_fit")
  expect_identical(rownames(s$varcomp), c("gid", "residuelle"))
  expect_identical(s$varcomp$bound, c("P", "P"))
  expect_equal(s$varcomp$component, c(f$sigmas$gid[1, 1], f$sigma_res[1, 1]), tolerance = 1e-10)
  # erreurs-types du moteur (jacobienne exacte) contre rx_ratios (differences
  # finies cote R) : deux calculs independants de la meme methode delta.
  comp <- data.frame(target = "y", direct = "gid", indirect_within = NA,
                     indirect_between = NA, other = "residual")
  rr <- rx_ratios(f, comp, model = m, quantities = "variances")
  se_rr <- rr$se[match(c("direct", "residual"), rr$component)]
  expect_equal(s$varcomp$std.error, se_rr, tolerance = 1e-5)
  expect_equal(s$varcomp$z.ratio, s$varcomp$component / s$varcomp$std.error)
  expect_equal(s$coef.fixed$std.error, sqrt(diag(f$vbeta)))
  expect_equal(s$nedf, 118L)
  expect_equal(s$aic, -2 * f$logLik + 2 * 2)
  expect_equal(s$bic, -2 * f$logLik + 2 * log(118))
  expect_true(s$converged)
  expect_output(print(s), "Composantes de variance")
})

test_that("summary : une variance nulle est marquee B sans erreur-type, les autres gardent la leur", {
  set.seed(2026)
  d <- expand.grid(gid = factor(1:60), bloc = factor(1:4))
  d$y <- 12 + rnorm(60)[d$gid] + rnorm(nrow(d))
  f <- rx_reml(y ~ 1, random = ~ gid + bloc, data = d, backend = "cpu", verbose = FALSE)
  vc <- summary(f)$varcomp
  skip_if(vc["bloc", "component"] > 1e-4, "variance de bloc non nulle sur ce tirage")
  expect_identical(vc["bloc", "bound"], "B")
  expect_true(is.na(vc["bloc", "std.error"]))
  expect_true(all(is.finite(vc[c("gid", "residuelle"), "std.error"])))
})

test_that("summary : covariances multi-caracteres codees U, variances P", {
  set.seed(11)
  q <- 40; nrep <- 3
  u <- matrix(rnorm(q * 2), q) %*% chol(matrix(c(1, .5, .5, 2), 2))
  d <- expand.grid(rep = 1:nrep, gid = factor(1:q), trait = factor(c("t1", "t2")))
  d$unite <- factor(paste(d$gid, d$rep))
  d <- d[order(d$trait, d$gid, d$rep), ]
  d$y <- u[cbind(as.integer(d$gid), as.integer(d$trait))] + rnorm(nrow(d))
  f <- rx_reml(y ~ trait, random = ~ us(gid), residual = ~ us(trait):units, data = d,
               trait = "trait", unit = "unite", backend = "cpu", verbose = FALSE)
  vc <- summary(f)$varcomp
  expect_identical(vc[c("gid[1,1]", "gid[2,1]", "gid[2,2]"), "bound"], c("P", "U", "P"))
  expect_equal(vc["gid[2,1]", "component"], f$sigmas$gid[2, 1], tolerance = 1e-10)
})

test_that("summary(coef = TRUE) rend les BLUP avec leur erreur-type si pev = TRUE", {
  set.seed(3)
  d <- data.frame(gid = factor(rep(1:15, each = 4)))
  d$y <- rnorm(15)[d$gid] + rnorm(60)
  f <- rx_reml(y ~ 1, random = ~ gid, data = d, backend = "cpu", verbose = FALSE, pev = TRUE)
  cr <- summary(f, coef = TRUE)$coef.random
  expect_identical(nrow(cr), 15L)
  expect_equal(cr$solution, as.numeric(f$blups$gid))
  expect_equal(cr$std.error, sqrt(as.numeric(f$pev$gid)))
})
