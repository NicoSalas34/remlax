# Pur R, aucun Python : les comptages de parametres doivent reproduire
# structures.py. Les valeurs attendues sont celles du catalogue
# (docs/structures.md) : w(w+1)/2 pour us, n_loadings + w pour fa, etc.
test_that("rx_n_params reproduit le catalogue", {
  expect_equal(rx_n_params("iid", 4L), 1L)
  expect_equal(rx_n_params("diag", 4L), 4L)
  expect_equal(rx_n_params("us", 2L), 3L)
  expect_equal(rx_n_params("us", 6L), 21L)
  expect_equal(rx_n_params("fa", 5L, rank = 1L), rx_n_loadings(5L, 1L) + 5L)
  expect_equal(rx_n_params("rr", 5L, rank = 2L), rx_n_loadings(5L, 2L))
  expect_equal(rx_n_params("corh", 4L), 5L)
  expect_error(rx_n_params("inconnue", 2L))
})

test_that("rx_n_level compte les parametres entre niveaux", {
  expect_equal(rx_n_level("id"), 0L)
  expect_equal(rx_n_level("ar1"), 1L)
  expect_equal(rx_n_level("ar2"), 2L)
  expect_equal(rx_n_level("corb", order = 3L), 3L)
})

test_that("rx_term et rx_model construisent et comptent", {
  set.seed(1)
  n_gen <- 12; n_rep <- 3
  gid <- factor(rep(seq_len(n_gen), n_rep))
  y <- rnorm(n_gen * n_rep)
  X <- matrix(1, n_gen * n_rep, 1)
  tm <- rx_term("gid", gid)
  expect_s3_class(tm, "rx_term")
  expect_equal(tm$t, 1L)
  expect_equal(tm$q, n_gen)
  m <- rx_model(y, X, terms = list(tm), residual = rx_residual())
  expect_s3_class(m, "rx_model")
  expect_equal(rx_n_theta(m), 2L)
  expect_output(print(m), "gid")
  # une X singuliere est refusee : log|X'V^-1X| serait -Inf
  expect_error(rx_model(y, cbind(1, 1), terms = list(tm)))
  # rien a estimer : aucun terme et residuelle iid
  expect_error(rx_model(y, X, terms = list()))
})

test_that("rx_export ecrit un manifeste relisible", {
  set.seed(2)
  gid <- factor(rep(1:5, 4)); y <- rnorm(20); X <- matrix(1, 20, 1)
  m <- rx_model(y, X, terms = list(rx_term("gid", gid)))
  d <- tempfile("rx_export_")
  rx_export(m, d)
  expect_true(file.exists(file.path(d, "manifest.json")))
  man <- jsonlite::fromJSON(file.path(d, "manifest.json"))
  expect_true(file.exists(file.path(d, "y.bin")))
  expect_true(file.exists(file.path(d, "X.bin")))
  # y relu a l'identique : float64 column-major
  yy <- readBin(file.path(d, "y.bin"), what = "double", n = 20, size = 8)
  expect_equal(yy, y)
  unlink(d, recursive = TRUE)
})

test_that("le solveur Python livre avec le paquet est present", {
  cli <- system.file("python", "remlax", "cli.py", package = "remlax")
  expect_true(nzchar(cli))
  expect_true(file.exists(system.file("python", "remlax", "__init__.py", package = "remlax")))
})
