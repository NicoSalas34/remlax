# spl2d() dans la formule de rx_reml() : meme ajustement que la voie explicite.

skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")

jeu <- function() {
  set.seed(7)
  g <- expand.grid(r = 1:12, c = 1:10)
  g$gid <- factor(sample(rep(1:30, 4)))
  g$trt <- factor(rep(c("a", "b"), length.out = nrow(g)))
  g$y <- 0.05 * (g$r - 6)^2 + 0.3 * sin(g$c / 2) + 0.4 * (g$trt == "b") +
    rnorm(30)[g$gid] + rnorm(nrow(g), 0, 0.5)
  g
}

test_that("spl2d() dans rx_reml() equivaut a rx_spl2d() + rx_model()", {
  g <- jeu()
  f <- rx_reml(y ~ trt, random = ~ gid + spl2d(r, c, nseg = c(4, 4)), data = g,
               backend = "cpu", verbose = FALSE)
  sp <- rx_spl2d(g$r, g$c, nseg = c(4, 4))
  m <- rx_model(g$y, cbind(model.matrix(~ trt, g), sp$X),
                c(list(rx_term("gid", g$gid)), sp$terms))
  fe <- rx_fit(m, backend = "cpu", verbose = FALSE)
  expect_equal(f$logLik, fe$logLik, tolerance = 1e-8)
  expect_identical(names(f$sigmas), c("gid", "spl_x", "spl_y", "spl_xy"))
  expect_identical(f$spl2d$columns, colnames(sp$X))
  expect_identical(attr(f$model$X, "termes"), c("(Intercept)", "trt", "spl2d(r, c, nseg = c(4, 4))"))
})

test_that("spl2d() : nom, arguments nommes, coordonnees en facteur, erreurs", {
  g <- jeu(); g$rf <- factor(g$r)
  f <- rx_reml(y ~ 1, random = ~ spl2d(x = rf, y = c, nseg = c(3, 3), name = "champ"), data = g,
               backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_identical(names(f$sigmas), c("champ_x", "champ_y", "champ_xy"))
  g$lettre <- factor(letters[g$r])
  expect_error(rx_reml(y ~ 1, random = ~ spl2d(lettre, c), data = g, backend = "cpu"),
               "non numeriques")
  expect_error(rx_reml(y ~ 1, random = ~ spl2d(r), data = g, backend = "cpu"), "y manquante")
})

test_that("spl2d() : predictions, Wald et summary tiennent compte de la partie nulle", {
  g <- jeu()
  f <- rx_reml(y ~ trt, random = ~ gid + spl2d(r, c, nseg = c(4, 4)), data = g,
               backend = "cpu", verbose = FALSE, wald = TRUE)
  p <- rx_predict(f, classify = "trt")
  expect_identical(nrow(p), 2L)
  expect_true(all(p$estimable))
  # l'ecart des deux traitements est exactement le coefficient trtb
  expect_equal(diff(p$predicted.value), unname(f$beta[2]), tolerance = 1e-10)
  expect_true("spl2d(r, c, nseg = c(4, 4))" %in% f$wald$tests$terme)
  s <- summary(f)
  expect_true(all(c("spl_x", "spl_y", "spl_xy") %in% rownames(s$varcomp)))
  expect_true(all(grepl("^spl_lin", tail(rownames(s$coef.fixed), length(f$spl2d$columns)))))
})
