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

test_that("deux scans successifs n'utilisent pas le meme dossier", {
  skip_si_pas_de_jax()
  set.seed(5)
  q <- 20; n <- q * 4
  gen <- factor(rep(seq_len(q), 4))
  M <- matrix(sample(c(-1, 1), q * 15, replace = TRUE), q, 15,
              dimnames = list(levels(gen), paste0("snp", 1:15)))
  y <- 1 + M[as.integer(gen), 1] + rnorm(q)[gen] + rnorm(n)
  m <- rx_model(y, matrix(1, n, 1), list(rx_term("gen", gen)))
  f <- rx_fit(m, backend = "cpu", verbose = FALSE)
  avant <- list.files(tempdir(), "^rx_scan_")
  for (i in 1:2)
    rx_scan(f, m, M, incidences = c(dir = "gen"), tests = "dir", maf = NULL,
            backend = "cpu", verbose = FALSE, keep = TRUE)
  nouveaux <- setdiff(list.files(tempdir(), "^rx_scan_"), avant)
  expect_length(nouveaux, 2L)
  unlink(file.path(tempdir(), nouveaux), recursive = TRUE)
})

test_that("Wald et Kenward-Roger tiennent compte des poids a priori", {
  skip_si_pas_de_jax()
  set.seed(11)
  q <- 20; n <- 160
  g <- factor(sample(seq_len(q), n, replace = TRUE), levels = seq_len(q))
  x <- rnorm(n); w <- runif(n, 0.3, 3)
  y <- 1 + 0.4 * x + rnorm(q)[g] + rnorm(n) / sqrt(w)
  X <- cbind(1, x)
  m <- rx_model(y, X, list(rx_term("g", g)), weights = w)
  f <- rx_fit(m, backend = "cpu", wald = TRUE, kenward_roger = TRUE, verbose = FALSE)
  Z <- Matrix::sparseMatrix(i = seq_len(n), j = as.integer(g), x = 1, dims = c(n, q))
  V <- as.matrix(f$sigmas$g[1, 1] * Z %*% Matrix::t(Z)) + f$sigma_res[1, 1] * diag(1 / w)
  Vi <- solve(V)
  vb <- solve(t(X) %*% Vi %*% X)
  b <- vb %*% t(X) %*% Vi %*% y
  expect_equal(unname(f$wald$beta), unname(drop(b)), tolerance = 1e-6)
  expect_equal(f$wald$tests$F[2], drop(b[2]^2 / vb[2, 2]), tolerance = 1e-6)
  expect_equal(unname(f$kenward_roger$se_beta_brut), unname(sqrt(diag(vb))), tolerance = 1e-6)
})
