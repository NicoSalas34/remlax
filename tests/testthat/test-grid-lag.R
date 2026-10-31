# Lag structures (ar1, ar2, ..., ar1 x ar1) on a grid with an empty row or
# column, and a fit whose parameters are all fixed. The reference is the REML
# log-likelihood computed densely in R on the full grid, in the asreml
# convention (without the 2 pi constant).

.ar_mat <- function(lv, r) r^abs(outer(lv, lv, "-"))
.reml_dense <- function(y, X, V) {
  Vi <- solve(V); XtViX <- crossprod(X, Vi %*% X)
  P <- Vi - Vi %*% X %*% solve(XtViX, crossprod(X, Vi))
  -0.5 * (as.numeric(determinant(V)$modulus) + as.numeric(determinant(XtViX)$modulus) +
            drop(t(y) %*% P %*% y))
}
# 8 x 10 field, column 5 empty and one more cell missing.
.gappy_field <- function(seed = 3, NR = 8L, NC = 10L) {
  set.seed(seed)
  d <- expand.grid(col = seq_len(NC), row = seq_len(NR))[, c("row", "col")]
  d$gid <- factor(sample(1:20, nrow(d), TRUE))
  L <- t(chol(kronecker(.ar_mat(seq_len(NR), .6), .ar_mat(seq_len(NC), .5))))
  d$y <- 1 + rnorm(20, 0, .8)[d$gid] + drop(L %*% rnorm(NR * NC))
  d <- d[d$col != 5 & !(d$row == 2 & d$col == 7), ]
  attr(d, "NR") <- NR; attr(d, "NC") <- NC
  d
}

test_that("grid levels keep empty rows and columns", {
  g <- remlax:::.rx_grid_factor
  expect_identical(levels(g(c(1, 2, 4), "t")), c("1", "2", "3", "4"))
  # step of the data: odd columns only form a grid of step 2
  expect_identical(levels(g(c(1, 3, 5, 3), "t")), c("1", "3", "5"))
  expect_identical(levels(g(c(1, 3, 5, 8), "t")), as.character(1:8))
  # a factor keeps its unused levels, in their order
  expect_identical(levels(g(factor(c("a", "b"), levels = c("a", "z", "b")), "t")),
                   c("a", "z", "b"))
  # text that reads as numbers is ordered as numbers
  expect_identical(levels(g(c("10", "2", "4"), "t")), as.character(c(2, 4, 6, 8, 10)))
  expect_identical(levels(g(c(0.5, 1, 2), "t")), c("0.5", "1", "1.5", "2"))
  expect_error(g(c(0, 1, 2.5), "t"), "regular grid")
  expect_identical(levels(g(c("b", "a"), "t")), c("a", "b"))
})

test_that("ar1(row):ar1(col) residual keeps an empty column", {
  d <- .gappy_field(); NR <- attr(d, "NR"); NC <- attr(d, "NC")
  r <- remlax:::.rx_parse_residual(~ ar1(row):ar1(col), d)
  expect_equal(r$dims, c(NR, NC))
  expect_equal(r$n_unit, NR * NC)
  expect_equal(r$unit, (d$row - 1L) * NC + d$col)
  d2 <- transform(d, row = factor(row, levels = 1:NR), col = factor(col, levels = 1:NC))
  expect_equal(remlax:::.rx_parse_residual(~ ar1(row):ar1(col), d2)$unit, r$unit)
})

test_that("lag structures on a gappy grid match the dense REML", {
  skip_si_pas_de_jax()
  d <- .gappy_field(); NR <- attr(d, "NR"); NC <- attr(d, "NC")
  X <- matrix(1, nrow(d)); Zg <- model.matrix(~ 0 + gid, d)
  cell <- (d$row - 1L) * NC + d$col
  for (eng in c("dense", "mme")) {
    f <- rx_reml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d,
                 backend = "cpu", verbose = FALSE, engine = eng)
    rr <- f$rho$residual
    V <- f$sigmas$gid[1] * tcrossprod(Zg) + f$sigma_res[1] *
      kronecker(.ar_mat(1:NR, rr[1]), .ar_mat(1:NC, rr[2]))[cell, cell]
    expect_equal(f$logLik_asreml, .reml_dense(d$y, X, V), tolerance = 1e-8)

    f <- rx_reml(y ~ 1, random = ~ gid + ar1(row, col), data = d, backend = "cpu",
                 verbose = FALSE, engine = eng)
    nm <- setdiff(names(f$sigmas), "gid"); rr <- f$rho[[nm]]
    expect_length(f$blups[[nm]], NR * NC)
    V <- f$sigmas$gid[1] * tcrossprod(Zg) + diag(f$sigma_res[1], nrow(d)) + f$sigmas[[nm]][1] *
      kronecker(.ar_mat(1:NR, rr[1]), .ar_mat(1:NC, rr[2]))[cell, cell]
    expect_equal(f$logLik_asreml, .reml_dense(d$y, X, V), tolerance = 1e-8)

    e <- transform(d[d$row %in% 1:3, ], rowf = factor(row))
    f <- rx_reml(y ~ 1, random = ~ rowf + ar1(col), data = e, backend = "cpu",
                 verbose = FALSE, engine = eng)
    expect_length(f$blups$col, NC)
    Zc <- outer(e$col, 1:NC, "==") * 1; Zr <- model.matrix(~ 0 + rowf, e)
    V <- f$sigmas$rowf[1] * tcrossprod(Zr) + diag(f$sigma_res[1], nrow(e)) +
      f$sigmas$col[1] * Zc %*% .ar_mat(1:NC, f$rho$col) %*% t(Zc)
    expect_equal(f$logLik_asreml, .reml_dense(e$y, matrix(1, nrow(e)), V), tolerance = 1e-8)
  }
  set.seed(5); tt <- c(1:6, 9:14, 16:20)
  s <- data.frame(t = tt, y = drop(t(chol(.ar_mat(tt, .7))) %*% rnorm(length(tt))))
  f <- rx_reml(y ~ 1, residual = ~ ar1(t), data = s, backend = "cpu", verbose = FALSE)
  expect_equal(f$logLik_asreml,
               .reml_dense(s$y, matrix(1, nrow(s)), f$sigma_res[1] * .ar_mat(tt, unlist(f$rho))),
               tolerance = 1e-8)
})

test_that("every parameter fixed: evaluation at the given point", {
  skip_si_pas_de_jax()
  d <- .gappy_field()
  f <- rx_reml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d,
               backend = "cpu", verbose = FALSE)
  th <- f$theta; th[1] <- th[1] + 0.3
  fe <- rx_reml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d,
                backend = "cpu", verbose = FALSE, theta_init = th,
                fixed_theta = seq_along(th))
  expect_equal(fe$theta, th)
  expect_equal(fe$n_iter, 0)
  expect_match(fe$scipy_message, "every parameter fixed")
  expect_true(fe$logLik < f$logLik)
  f0 <- rx_reml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d,
                backend = "cpu", verbose = FALSE, theta_init = th, maxiter = 0, polish = 0)
  expect_equal(fe$logLik, f0$logLik, tolerance = 1e-10)
})
