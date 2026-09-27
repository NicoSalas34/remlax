# Ajustement de fumee : un facteur aleatoire, plan equilibre, CPU.
# S'ignore sans Python porteur de jax (helper-python.R).
test_that("rx_reml ajuste un facteur aleatoire et retrouve la variance simulee", {
  skip_si_pas_de_jax()
  set.seed(2026)
  n_gen <- 60; n_bloc <- 4
  d <- expand.grid(gid = factor(seq_len(n_gen)), bloc = factor(seq_len(n_bloc)))
  g <- rnorm(n_gen, 0, sqrt(1.5)); b <- rnorm(n_bloc, 0, sqrt(0.4))
  d$y <- 12 + g[as.integer(d$gid)] + b[as.integer(d$bloc)] + rnorm(nrow(d))
  fit <- rx_reml(fixed = y ~ 1, random = ~ gid + iid(bloc), data = d,
                 vpredict = c(h2 = "V1/(V1+V2+V3)"), backend = "cpu",
                 verbose = FALSE)
  expect_s3_class(fit, "rx_fit")
  expect_equal(fit[["n_par"]], 3L)
  expect_equal(fit[["n_obs"]], 240L)
  # valeurs du README (meme graine, meme plan) : logLik -388.088838
  expect_equal(fit[["logLik"]], -388.088838, tolerance = 1e-6)
  expect_equal(fit[["sigmas"]][["gid"]][1, 1], 1.401, tolerance = 1e-3)
  expect_equal(fit[["sigma_res"]][1, 1], 0.8988, tolerance = 1e-3)
  expect_true(fit[["newton_decrement"]] < 1e-4)
  pr <- fit[["vpredict"]][["predictions"]]
  expect_equal(pr$valeur[pr$nom == "h2"], 0.60735, tolerance = 1e-4)
  expect_equal(pr$se[pr$nom == "h2"], 0.05871, tolerance = 1e-3)
})

test_that("rx_fit refuse un terme Kinv sans facteur de K", {
  set.seed(3)
  gid <- factor(rep(1:6, 3)); y <- rnorm(18); X <- matrix(1, 18, 1)
  Kinv <- Matrix::Diagonal(6)
  tm <- rx_term("gid", gid, Kinv = Kinv)
  m <- rx_model(y, X, terms = list(tm))
  expect_error(rx_fit(m, backend = "cpu", verbose = FALSE), "PRECISION")
})
