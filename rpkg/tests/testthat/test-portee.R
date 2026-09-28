# Noyaux a PORTEE (sph, cir) sur un champ 2D irregulier : le solveur doit
# atteindre le maximum GLOBAL d'un profil REML calcule en R dense, independant
# de remlax. Origine : validation asreml3 du 2026-09-28, section B7 (cir rendu a
# l'iteration 0 pour gradient NaN ; trois maxima locaux en la portee).
noyau_portee <- function(kind, D, r) {
  h <- pmin(D / r, 1)
  if (kind == "sph") 1 - 1.5 * h + 0.5 * h^3
  else 1 - (2 / pi) * (h * sqrt(1 - h^2) + asin(h))
}
profil_reml <- function(y, C) {
  # -2logL REML (avec la constante (n-p) log 2pi) au s2 optimal en forme close
  n <- length(y); X <- matrix(1, n, 1)
  cV <- chol(C + diag(1e-10, n)); Ci <- chol2inv(cV)
  XVX <- t(X) %*% Ci %*% X; beta <- solve(XVX, t(X) %*% Ci %*% y)
  r <- y - X %*% beta; yPy <- as.numeric(t(r) %*% Ci %*% r); s2 <- yPy / (n - 1)
  (n - 1) * log(2 * pi) + 2 * sum(log(diag(cV))) + n * log(s2) + log(det(XVX)) - log(s2) + yPy / s2
}

test_that("sph et cir atteignent le maximum global du profil REML en la portee", {
  skip_si_pas_de_jax()
  set.seed(200); nB <- 90
  co <- data.frame(cx = round(runif(nB, 0, 20), 2), cy = round(runif(nB, 0, 20), 2))
  co <- co[!duplicated(co), ]; nB <- nrow(co)
  De <- sqrt(outer(co$cx, co$cx, "-")^2 + outer(co$cy, co$cy, "-")^2)
  for (kind in c("sph", "cir")) {
    set.seed(if (kind == "sph") 306 else 307)
    e <- as.numeric(t(chol(noyau_portee(kind, De, 8) + diag(1e-6, nB))) %*% rnorm(nB))
    d <- data.frame(y = 2 + e + rnorm(nB, 0, 0.1), co)
    fml <- stats::as.formula(sprintf("~ %s(cx, cy)", kind))
    fit <- rx_reml(y ~ 1, residual = fml, data = d, backend = "cpu", verbose = FALSE)
    expect_gt(fit[["n_iter"]], 0L)
    portees <- seq(3, 14, by = 0.05)
    prof <- vapply(portees, function(r) profil_reml(d$y, noyau_portee(kind, De, r)), numeric(1))
    # jamais au-dessus du meilleur point de la grille, et dans la bonne colline
    expect_lte(-2 * fit[["logLik"]], min(prof) + 1e-6)
    expect_lt(abs(fit[["rho"]][["residuelle!portee"]] - portees[which.min(prof)]), 0.06)
  }
  # le champ cir a trois maxima locaux ; l'optimum global est a la portee 7.41
  # (logLik convention asreml 13.876 ; asreml, parti pres de 5, rend 10.848)
  expect_equal(fit[["rho"]][["residuelle!portee"]], 7.408, tolerance = 2e-3)
  expect_equal(fit[["logLik_asreml"]], 13.8759, tolerance = 1e-5)
})