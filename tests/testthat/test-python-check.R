# rx_python_check et le refus de rx_fit sans jax. Pur R : on designe
# volontairement un interpreteur qui n'a pas jax (ou qui n'existe pas).
test_that("un Python sans jax est refuse avant le lancement, avec un message actionnable", {
  ancien <- Sys.getenv("RX_PY", unset = NA)
  on.exit(if (is.na(ancien)) Sys.unsetenv("RX_PY") else Sys.setenv(RX_PY = ancien), add = TRUE)
  Sys.setenv(RX_PY = "/chemin/inexistant/python_sans_jax")
  r <- rx_python_check(quiet = TRUE)
  expect_false(r$ok)
  expect_match(r$message, "RX_PY")
  expect_match(r$message, "rx_install_python")
  expect_match(r$message, "python_sans_jax")
  set.seed(1)
  gid <- factor(rep(1:5, 4)); y <- rnorm(20); X <- matrix(1, 20, 1)
  m <- rx_model(y, X, terms = list(rx_term("gid", gid)))
  expect_error(rx_fit(m, backend = "cpu", verbose = FALSE), "ne porte pas jax")
})

test_that("rx_python_check rend les versions quand jax est present", {
  skip_si_pas_de_jax()
  r <- rx_python_check(quiet = TRUE)
  expect_true(r$ok)
  expect_true(all(c("jax", "numpy", "scipy") %in% names(r$versions)))
})

test_that("le lancement du moteur pose puis restaure ses variables d'environnement", {
  avant <- Sys.getenv(c("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "XLA_FLAGS"), unset = NA)
  withr_opt <- options(remlax.threads = 1L); on.exit(options(withr_opt), add = TRUE)
  expect_identical(remlax:::.rx_threads(), 1L)
  r <- remlax:::.rx_system2(file.path(R.home("bin"), "Rscript"),
    c("-e", shQuote("cat(Sys.getenv(c('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS')))")),
    stdout = TRUE)
  expect_identical(r, "1 1")
  expect_identical(Sys.getenv(c("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "XLA_FLAGS"), unset = NA),
                   avant)
})

test_that("pendant R CMD check, un seul fil de calcul est demande", {
  withr_opt <- options(remlax.threads = NULL); on.exit(options(withr_opt), add = TRUE)
  ancien <- Sys.getenv("_R_CHECK_LIMIT_CORES_", unset = NA)
  on.exit(if (is.na(ancien)) Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
          else Sys.setenv(`_R_CHECK_LIMIT_CORES_` = ancien), add = TRUE)
  Sys.setenv(`_R_CHECK_LIMIT_CORES_` = "TRUE")
  expect_identical(remlax:::.rx_threads(), 1L)
  Sys.setenv(`_R_CHECK_LIMIT_CORES_` = "false")
  expect_null(remlax:::.rx_threads())
})

test_that("une expression vpredict fautive est refusee avant l'ajustement, avec la liste des composantes", {
  skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")
  set.seed(1)
  d <- data.frame(gid = factor(rep(1:20, each = 3)))
  d$y <- rnorm(20)[d$gid] + rnorm(60)
  for (v in c(FALSE, TRUE))
    expect_error(suppressMessages(capture.output(
      rx_reml(y ~ 1, random = ~ gid, data = d, vpredict = c(h2 = "V1/(V1+V2+V3)"),
              backend = "cpu", verbose = v))),
      "V3 inconnu.*V1 = gid, V2 = residuelle")
})

test_that("rx_python_check() confirme l'interpreteur quand il convient", {
  skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")
  expect_message(rx_python_check(), "jax .*numpy .*scipy")
  expect_silent(rx_python_check(quiet = TRUE))
})

test_that("dsum : sections nommees d'apres la variable, toutes imprimees, V numerotes", {
  skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")
  set.seed(3)
  d <- expand.grid(gid = factor(1:20), bloc = factor(1:3))
  d$y <- rnorm(20)[d$gid] + rnorm(nrow(d), sd = c(0.8, 1, 1.3)[d$bloc])
  f <- rx_reml(y ~ bloc, random = ~ gid, residual = ~ dsum(~ units | bloc), data = d,
               vpredict = c(h2 = "V1/(V1+V2)"), backend = "cpu", verbose = FALSE)
  expect_identical(names(f$sigmas_res), c("bloc_1", "bloc_2", "bloc_3"))
  out <- capture.output(print(f))
  expect_length(grep("^  residuelle\\[bloc_", out), 3L)
  expect_true(any(grepl("V1 = gid, V2 = bloc_1, V3 = bloc_2, V4 = bloc_3", out, fixed = TRUE)))
})
