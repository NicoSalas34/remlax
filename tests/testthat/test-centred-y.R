# The likelihood is computed on y centred on X (center_y in incidence.py).
# REML is invariant to y -> y - X b; raw, a large mean made y'R^-1y and
# r'C^-1r two numbers of order 1e5 whose difference is the likelihood, and the
# mixed model equations engine lost up to 1e-5 in -2logL on a bivariate model
# with traits of very different scales (milk yield and protein of pedigreemm).

test_that("dense and MME engines agree on a bivariate model with a large mean", {
  skip_si_pas_de_jax()
  set.seed(21)
  q <- 150L; nr <- 3L
  K <- crossprod(matrix(rnorm(q * 40), 40)) / 40 + diag(0.3, q)
  dimnames(K) <- list(paste0("g", 1:q), paste0("g", 1:q))
  L <- t(chol(K))
  u1 <- drop(L %*% rnorm(q)); u2 <- 0.02 * (0.7 * u1 + drop(L %*% rnorm(q)))
  d <- expand.grid(rep = seq_len(nr), g = paste0("g", 1:q))
  d$rec <- factor(seq_len(nrow(d)))
  gi <- match(d$g, rownames(K))
  e1 <- rnorm(nrow(d), 0, 2); e2 <- 0.015 * (0.5 * e1 / 2 + rnorm(nrow(d)))
  # trait 1 near 20 with variance ~5, trait 2 near 0.6 with variance ~5e-4
  dl <- rbind(data.frame(g = d$g, rec = d$rec, car = "a", y = 20 + u1[gi] + e1),
              data.frame(g = d$g, rec = d$rec, car = "b", y = 0.6 + u2[gi] + e2))
  dl$car <- factor(dl$car); dl$g <- factor(dl$g, levels = rownames(K))
  dl <- dl[order(dl$rec, dl$car), ]
  fit <- function(eng, th = NULL) {
    args <- list(y ~ car, random = ~ us(g, K = K), residual = ~ us(car):units, data = dl,
                 trait = "car", unit = "rec", engine = eng, backend = "cpu", verbose = FALSE)
    if (!is.null(th)) args <- c(args, list(theta_init = th, fixed_theta = seq_along(th)))
    do.call(rx_reml, args)
  }
  fd <- fit("dense")
  fm <- fit("mme", fd$theta)
  expect_equal(fm$logLik, fd$logLik, tolerance = 1e-9 / abs(fd$logLik))
  # beta comes back on the raw scale despite the centring
  fmo <- fit("mme")
  expect_equal(unname(fmo$beta), unname(fd$beta), tolerance = 1e-5)
  # shifting y by a combination of X leaves the likelihood unchanged
  dl$y <- dl$y + 1e3 * (dl$car == "a")
  expect_equal(fit("mme", fd$theta)$logLik, fd$logLik, tolerance = 1e-9 / abs(fd$logLik))
})
