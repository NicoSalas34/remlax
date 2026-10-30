# Defauts corriges apres le nettoyage de 2026-10 : un test par defaut.

test_that("X garde toutes les lignes quelle que soit l'option na.action", {
  d <- data.frame(y = c(1, 2, NA, 4, 5, 6), x = c(1, NA, 3, 4, 5, 6),
                  f = factor(c("a", "b", "a", NA, "b", "a")))
  op <- options(na.action = "na.omit"); on.exit(options(op))
  fx <- remlax:::.rx_fixed_design(y ~ x + f, d)
  expect_equal(nrow(fx$X), length(fx$y))
  cl <- remlax:::.rx_drop_incomplete(fx, d, NULL, NULL, NULL)
  expect_equal(nrow(cl$X), 3L)
  expect_equal(length(cl$y), 3L)
})

test_that("un trait donne par son nom de colonne reste un nom apres les lignes incompletes", {
  d <- data.frame(y = c(1, NA, 3, 4), x = 1:4, car = factor(c("a", "b", "a", "b")))
  fx <- remlax:::.rx_fixed_design(y ~ x, d)
  cl <- suppressMessages(remlax:::.rx_drop_incomplete(fx, d, "car", NULL, NULL))
  expect_identical(cl$trait, "car")
  cl2 <- suppressMessages(remlax:::.rx_drop_incomplete(fx, d, d$car, NULL, NULL))
  expect_equal(as.character(cl2$trait), c("a", "a", "b"))
})

test_that("rx_reml avec trait par nom et lignes incompletes egale l'ajustement sur donnees nettoyees", {
  skip_si_pas_de_jax()
  set.seed(3)
  d <- data.frame(gid = factor(rep(1:25, each = 4)), car = factor(rep(c("a", "b"), 50)))
  d$y <- rnorm(25)[d$gid] + rnorm(100) + (d$car == "b")
  d$y[c(5, 17, 60)] <- NA
  f1 <- suppressMessages(rx_reml(y ~ car, random = ~ gid, residual = ~ units:diag(car),
                                 data = d, trait = "car", backend = "cpu", verbose = FALSE))
  dc <- d[!is.na(d$y), ]
  f2 <- rx_reml(y ~ car, random = ~ gid, residual = ~ units:diag(car),
                data = dc, trait = "car", backend = "cpu", verbose = FALSE)
  expect_equal(f1$logLik, f2$logLik, tolerance = 1e-10)
})
