# Moteur creux (RTMB) : perimetre en pur R, puis concordance de la
# vraisemblance avec le moteur dense a theta COMMUN, impose via rx_n_theta.
# S'ignore sans RTMB ; la partie dense s'ignore aussi sans Python porteur de jax.

test_that("rx_sparse_scope nomme la structure qui sort du perimetre", {
  set.seed(2)
  gid <- factor(rep(1:8, 5)); n <- 40
  ok <- rx_sparse_scope(list(rx_term("g", gid)), rx_residual())
  expect_true(ok$ok)
  s <- rx_sparse_scope(list(rx_term("g", list(matrix(1, n, 8), matrix(1, n, 8)),
                                    struct = "fa", rank = 1L)), rx_residual())
  expect_false(s$ok); expect_match(s$reason, "fa")
  K <- diag(8); dimnames(K) <- list(levels(gid), levels(gid))
  s <- rx_sparse_scope(list(rx_term("g", gid, K = K)), rx_residual())
  expect_false(s$ok); expect_match(s$reason, "Kinv")
  s <- rx_sparse_scope(list(rx_term("g", gid, level = "ar2")), rx_residual())
  expect_false(s$ok); expect_match(s$reason, "ar2")
  s <- rx_sparse_scope(list(rx_term("g", gid)), rx_residual("us", trait = gl(2, 20)))
  expect_false(s$ok); expect_match(s$reason, "residuelle")
  s <- rx_sparse_scope(list(rx_term("g", gid)),
                       rx_residual(sections = list(rx_residual(rows = 1:20), rx_residual(rows = 21:40))))
  expect_false(s$ok); expect_match(s$reason, "sectionnee")
  expect_true(rx_sparse_scope(list(rx_term("g", gid, Kinv = Matrix::Diagonal(8))), rx_residual())$ok)
  expect_true(rx_sparse_scope(list(rx_term("g", gid, level = "ar1")), rx_residual())$ok)
  expect_type(rx_tmb_available(), "logical")
})

test_that("rx_fit_sparse : refus hors perimetre, puis concordance avec rx_fit", {
  skip_if_not(rx_tmb_available(), "RTMB non charge")
  set.seed(3)
  n_gen <- 25; n_bloc <- 4
  d <- expand.grid(gid = factor(seq_len(n_gen)), bloc = factor(seq_len(n_bloc)))
  d$y <- 10 + rnorm(n_gen, 0, 1)[d$gid] + rnorm(n_bloc, 0, 0.5)[d$bloc] + rnorm(nrow(d))
  m <- rx_model(d$y, matrix(1, nrow(d), 1),
                list(rx_term("gid", d$gid), rx_term("bloc", d$bloc, level = "ar1")))
  expect_equal(rx_n_theta(m), 4L)
  m_us <- rx_model(d$y, matrix(1, nrow(d), 1), list(rx_term("gid", d$gid)),
                   rx_residual("us", trait = d$bloc, unit = d$gid))
  expect_error(rx_fit_sparse(m_us, verbose = FALSE), "perimetre")
  th <- c(0.1, -0.3, 0.2, 0.05)
  s0 <- rx_fit_sparse(m, theta_init = th, maxiter = 0L, verbose = FALSE, sdreport = FALSE)
  expect_identical(s0$theta, th)
  expect_equal(s0$engine, "sparse")
  expect_true(is.finite(s0$logLik))
  s1 <- rx_fit_sparse(m, verbose = FALSE, sdreport = FALSE)
  expect_length(s1$theta, rx_n_theta(m))
  expect_true(s1$logLik >= s0$logLik)
  skip_si_pas_de_jax()
  d0 <- rx_fit(m, backend = "cpu", verbose = FALSE, theta_init = th, maxiter = 0, polish = 0,
               hessian = FALSE, blups = FALSE)
  expect_equal(d0[["logLik"]], s0$logLik, tolerance = 1e-8)
  d1 <- rx_fit(m, backend = "cpu", verbose = FALSE, theta_init = s1$theta, maxiter = 0,
               polish = 0, hessian = FALSE, blups = FALSE)
  expect_equal(d1[["logLik"]], s1$logLik, tolerance = 1e-6)
  # une parente CREUSE par sa precision : Kinv tridiagonale (AR1 de rho 0.5)
  P <- rx_ar1_prec(0.5, n_gen)
  mk <- rx_model(d$y, matrix(1, nrow(d), 1),
                 list(rx_term("gid", d$gid, Kinv = P$P, Kinv_logdet = P$logdet)))
  sk <- rx_fit_sparse(mk, theta_init = c(0.1, 0.0), maxiter = 0L, verbose = FALSE, sdreport = FALSE)
  K <- 0.5^abs(outer(seq_len(n_gen), seq_len(n_gen), "-"))
  dimnames(K) <- list(levels(d$gid), levels(d$gid))
  mK <- rx_model(d$y, matrix(1, nrow(d), 1), list(rx_term("gid", d$gid, K = K)))
  dk <- rx_fit(mK, backend = "cpu", verbose = FALSE, theta_init = c(0.1, 0.0), maxiter = 0,
               polish = 0, hessian = FALSE, blups = FALSE)
  expect_equal(dk[["logLik"]], sk$logLik, tolerance = 1e-8)
})
