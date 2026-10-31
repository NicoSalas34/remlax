# Multi-trait scan through rx_scan(): a bivariate model with a multi-trait
# genetic term and a neighbour term, each test checked against a GLS written
# in R at the fitted variances.

test_that("rx_scan on a multi-trait model matches a dense GLS at fixed V", {
  skip_si_pas_de_jax()
  set.seed(31)
  q <- 30L; nu <- 150L; p <- 8L
  K <- crossprod(matrix(rnorm(q * 60), 60)) / 60 + diag(0.2, q)
  dimnames(K) <- list(paste0("g", 1:q), paste0("g", 1:q))
  g <- sample(rownames(K), nu, TRUE); nb <- sample(rownames(K), nu, TRUE)
  M <- matrix(sample(c(-1, 1), q * p, TRUE), q, p, dimnames = list(rownames(K), paste0("s", 1:p)))
  Lk <- t(chol(K)); u <- cbind(Lk %*% rnorm(q), Lk %*% rnorm(q))
  u[, 2] <- 0.5 * u[, 1] + u[, 2]
  gi <- match(g, rownames(K)); ni <- match(nb, rownames(K))
  y1 <- 10 + u[gi, 1] + 0.4 * u[ni, 1] + 0.8 * M[gi, 1] + rnorm(nu)
  y2 <- 2 + u[gi, 2] + rnorm(nu, 0, 1.5)
  dl <- rbind(data.frame(rec = seq_len(nu), g = g, nb = nb, car = "a", y = y1),
              data.frame(rec = seq_len(nu), g = g, nb = nb, car = "b", y = y2))
  dl <- dl[-3, ]                                   # one record missing
  dl$car <- factor(dl$car); dl$rec <- factor(dl$rec)
  dl$g <- factor(dl$g, levels = rownames(K)); dl$nb <- factor(dl$nb, levels = rownames(K))
  dl <- dl[order(dl$rec, dl$car), ]
  fit <- rx_reml(y ~ car, random = ~ us(g, K = K) + us(nb, K = K),
                 residual = ~ us(car):units, data = dl, trait = "car", unit = "rec",
                 backend = "cpu", verbose = FALSE)
  tests <- c("dir", "dir[b]", "dir@common", "dir@het", "ind|dir", "dir+ind")
  res <- rx_scan(fit, fit$model, markers = M, incidences = c(dir = "g", ind = "nb"),
                 tests = tests, maf = NULL, verbose = FALSE, backend = "cpu")
  expect_equal(nrow(res), p)
  expect_true(all(c("beta_dir[a]", "beta_dir[b]", "p_dir@het") %in% names(res)))

  # Reference: V from the fitted components, GLS per SNP.
  n <- nrow(dl); tr <- as.integer(dl$car)
  Zg <- outer(as.integer(dl$g), seq_len(q), "==") * 1
  Zn <- outer(as.integer(dl$nb), seq_len(q), "==") * 1
  o <- function(S) outer(tr, tr, function(i, j) S[cbind(i, j)])
  V <- o(fit$sigmas$g) * (Zg %*% K %*% t(Zg)) + o(fit$sigmas$nb) * (Zn %*% K %*% t(Zn)) +
    o(fit$sigma_res) * outer(as.integer(dl$rec), as.integer(dl$rec), "==")
  X <- model.matrix(~ car, dl)
  Vi <- solve(V); P <- Vi - Vi %*% X %*% solve(crossprod(X, Vi %*% X), crossprod(X, Vi))
  D <- sapply(1:2, function(k) as.numeric(tr == k))
  chi2_ref <- function(j, fams, L) {
    S <- do.call(cbind, lapply(fams, function(f) {
      s <- if (f == "dir") Zg %*% M[, j] else Zn %*% M[, j]
      D * as.numeric(s)
    }))
    I <- crossprod(S, P %*% S); b <- solve(I, crossprod(S, P %*% dl$y))
    Lb <- L %*% b; drop(t(Lb) %*% solve(L %*% solve(I, t(L)), Lb))
  }
  for (j in seq_len(p)) {
    expect_equal(res[["chi2_dir"]][j], chi2_ref(j, "dir", diag(2)), tolerance = 1e-6)
    expect_equal(res[["chi2_dir[b]"]][j], chi2_ref(j, "dir", matrix(c(0, 1), 1)), tolerance = 1e-6)
    expect_equal(res[["chi2_dir@het"]][j], chi2_ref(j, "dir", matrix(c(-1, 1), 1)), tolerance = 1e-6)
    expect_equal(res[["chi2_ind|dir"]][j], chi2_ref(j, c("ind", "dir"), diag(4)[1:2, ]), tolerance = 1e-6)
    expect_equal(res[["chi2_dir+ind"]][j], chi2_ref(j, c("dir", "ind"), diag(4)), tolerance = 1e-6)
  }
  expect_equal(res[["chi2_dir"]], res[["chi2_dir@common"]] + res[["chi2_dir@het"]], tolerance = 1e-8)
  expect_equal(unique(res[["df_dir"]]), 2)
  expect_lt(res[["p_dir"]][1], 1e-3)               # the simulated SNP

  # Exact refit of the top SNP for @het and a conditional test: the Wald test of
  # rx_scan_exact equals a refit written by hand with the same columns.
  tr <- dl$car; Dk <- sapply(levels(tr), function(l) as.numeric(tr == l))
  Zgl <- lapply(1:2, function(k) Matrix::Matrix(Zg * Dk[, k], sparse = TRUE))
  Znl <- lapply(1:2, function(k) Matrix::Matrix(Zn * Dk[, k], sparse = TRUE))
  X0 <- model.matrix(~ car, dl)
  build <- function(Kx, X_sup = NULL)
    rx_model(dl$y, cbind(X0, X_sup), list(rx_term("g", Zgl, K = Kx, struct = "us"),
                                          rx_term("nb", Znl, K = Kx, struct = "us")),
             rx_residual("us", trait = dl$car, unit = dl$rec))
  map <- data.frame(snp = colnames(M), chr = rep(1:2, length.out = p))
  for (tst in c("dir@het", "ind|dir")) {
    ex <- rx_scan_exact(res, tst, build, function(Mx) K, M, map, c(dir = "g", ind = "nb"),
                        top = 1L, loco = FALSE, verbose = FALSE, fit_args = list(backend = "cpu"))
    j <- match(ex$snp, colnames(M))
    sd <- as.numeric(Zg %*% M[, j]); si <- as.numeric(Zn %*% M[, j])
    S <- if (tst == "dir@het") sd * Dk else cbind(si * Dk, sd * Dk)
    fh <- rx_fit(build(K, X_sup = S), verbose = FALSE, backend = "cpu")
    ix <- ncol(X0) + seq_len(ncol(S)); b <- fh$beta[ix]; Vb <- fh$vbeta[ix, ix]
    L <- if (tst == "dir@het") matrix(c(-1, 1), 1) else diag(4)[1:2, ]
    Lb <- L %*% b
    expect_equal(ex$chi2_exact, drop(t(Lb) %*% solve(L %*% Vb %*% t(L), Lb)), tolerance = 1e-6)
    expect_equal(ex$df, nrow(L))
    expect_equal(ex$p_fixed_V, res[[paste0("p_", tst)]][j])
  }
})
