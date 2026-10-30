# Modele structurel a decalage spatial (rx_lag) et residuelle "none".
#
# Les tests d'ajustement s'ignorent sans Python porteur de jax (helper-python.R).
# La validation croisee contre le moteur independant igepath_reml (programme
# IGE) ne tourne que si REMLAX_IGEPATH designe son repertoire `python/` :
# c'est un autre depot, absent de CRAN et de la CI.

# --- donnees ------------------------------------------------------------------
.reine <- function(nr, nc) {
  co <- expand.grid(c = seq_len(nc), r = seq_len(nr))
  D <- unname(as.matrix(stats::dist(co, method = "maximum")))
  Matrix::Matrix((D == 1) * 1, sparse = TRUE)
}

# q caracteres en format long (caractere lent, plante rapide), DGE us, residuelle
# us, et y = (I - Psi (x) W)^-1 (X b + Z a + e).
.jeu_lag <- function(nr = 6, nc = 5, ng = 10, Psi = diag(0, 2), seed = 1, W = NULL,
                     G = matrix(c(1, .3, .3, 1), 2), R = matrix(c(1, -.2, -.2, 1), 2)) {
  set.seed(seed)
  q <- nrow(Psi); N <- nr * nc
  tn <- paste0("T", seq_len(q)); dimnames(Psi) <- list(tn, tn)
  W <- W %||% .reine(nr, nc)
  g <- sample(rep(seq_len(ng), length.out = N))
  a <- matrix(stats::rnorm(ng * q), ng) %*% chol(G[seq_len(q), seq_len(q), drop = FALSE])
  e <- matrix(stats::rnorm(N * q), N) %*% chol(R[seq_len(q), seq_len(q), drop = FALSE])
  eta <- as.numeric(a[g, , drop = FALSE] + e) + rep(seq_len(q) * 3, each = N)
  y <- as.numeric(solve(diag(N * q) - kronecker(Psi, as.matrix(W)), eta))
  trait <- factor(rep(tn, each = N), levels = tn); unit <- rep(seq_len(N), q)
  X <- if (q == 1) matrix(1, N, 1) else stats::model.matrix(~ trait - 1)
  Zg <- Matrix::sparseMatrix(i = seq_len(N), j = g, x = 1, dims = c(N, ng))
  zero <- Matrix::Matrix(0, N, ng, sparse = TRUE)
  ZD <- stats::setNames(lapply(seq_len(q), function(k)
    do.call(rbind, lapply(seq_len(q), function(j) if (j == k) Zg else zero))), tn)
  list(y = y, X = X, trait = trait, unit = unit, W = W, ZD = ZD, N = N, q = q, g = g, ng = ng,
       Psi = Psi, row = rep(rep(seq_len(nr), each = nc), q), col = rep(rep(seq_len(nc), nr), q))
}

.modele_tb <- function(d, lag = NULL, res = NULL) {
  res <- res %||% rx_residual("us", trait = d$trait, unit = d$unit)
  rx_model(d$y, d$X, list(rx_term("gen", d$ZD, struct = "us")), res, lag = lag)
}

# -2logL dense, ecrit ICI sans le moteur : V a partir des Sigma rendues.
.neg2_dense <- function(fit, d, Wl, crit) {
  n <- length(d$y); q <- d$q; N <- d$N
  Z <- as.matrix(do.call(cbind, d$ZD))
  V <- Z %*% kronecker(fit$sigmas[["gen"]], diag(d$ng)) %*% t(Z) +
    kronecker(as.matrix(fit[["sigma_res"]]), diag(N))
  Psi <- fit$lag$Psi
  M <- matrix(0, n, n)
  for (t in seq_len(q)) for (u in seq_len(q))
    M[(t - 1) * N + seq_len(N), (u - 1) * N + seq_len(N)] <-
      Psi[t, u] * as.matrix(Wl[[if (length(Wl) == 1L) 1L else t]])
  A <- diag(n) - M; ys <- as.numeric(A %*% d$y); X <- d$X; p <- ncol(X)
  Vi <- solve(V); XVX <- t(X) %*% Vi %*% X
  P <- Vi - Vi %*% X %*% solve(XVX, t(X) %*% Vi)
  ld <- function(M) as.numeric(determinant(M, logarithm = TRUE)$modulus)
  v <- if (crit == "ml") n * log(2 * pi) + ld(V) + sum(ys * (P %*% ys))
       else (n - p) * log(2 * pi) + ld(V) + ld(XVX) + sum(ys * (P %*% ys))
  v <- v - 2 * ld(A)
  if (crit == "reml_y") { Xy <- solve(A, X); v <- v - ld(t(Xy) %*% Xy) + ld(t(X) %*% X) }
  v
}

# --- specification (sans Python) ----------------------------------------------
test_that("rx_lag : entrees libres, ordre des parametres, alignement par noms", {
  d <- .jeu_lag()
  L <- rx_lag(d$W, d$trait, d$unit)
  expect_s3_class(L, "rx_lag")
  expect_equal(L$n_free, 4L)
  # receveur par receveur : l'ordre de theta cote moteur (numpy.nonzero)
  expect_equal(L$param_names, c("Psi[T1<-T1]", "Psi[T1<-T2]", "Psi[T2<-T1]", "Psi[T2<-T2]"))
  expect_equal(rx_lag(d$W, d$trait, d$unit, struct = "diag")$n_free, 2L)
  Lr <- rx_lag(d$W, d$trait, d$unit, receivers = "T2", emitters = "T1")
  expect_equal(Lr$param_names, "Psi[T2<-T1]")
  # un masque fourni dans l'autre ordre est realigne par ses dimnames : TRUE en
  # ligne T2, colonne T1, soit T2 <- T1 (lu par position, ce serait T1 <- T2)
  mk <- matrix(c(FALSE, FALSE, TRUE, FALSE), 2, dimnames = list(c("T2", "T1"), c("T2", "T1")))
  expect_equal(rx_lag(d$W, d$trait, d$unit, struct = "mask", mask = mk)$param_names, "Psi[T2<-T1]")
  # unite par etiquette : apparie aux rownames de W
  W2 <- d$W; dimnames(W2) <- list(paste0("p", seq_len(d$N)), paste0("p", seq_len(d$N)))
  expect_equal(rx_lag(W2, d$trait, paste0("p", d$unit))$unit, d$unit)
  m <- .modele_tb(d, lag = L)
  expect_equal(m$n_par, 3L + 3L + 4L)
  expect_equal(rx_n_theta(m), 10L)
})

test_that("rx_lag refuse les specifications ambigues ou fausses", {
  d <- .jeu_lag()
  Wd <- d$W; Wd[1, 1] <- 1
  expect_error(rx_lag(Wd, d$trait, d$unit), "non-zero diagonal")
  u <- d$unit; u[1] <- 2L
  expect_error(rx_lag(d$W, d$trait, u), "duplicated")
  expect_error(rx_lag(d$W, d$trait, d$unit + 1000), "1\\.\\.")
  expect_error(rx_lag(d$W, d$trait, as.character(d$unit)), "rownames")
  expect_error(rx_lag(list(d$W, d$W), d$trait, d$unit), "NAMED")
  expect_error(rx_lag(d$W, d$trait, d$unit, struct = "mask"), "requires `mask`")
  expect_error(rx_lag(d$W, d$trait, d$unit, receivers = "T9"), "unknown")
  expect_error(rx_model(d$y[-1], d$X[-1, ], list(rx_term("g", factor(rep(d$g, 2))[-1])),
                        lag = rx_lag(d$W, d$trait, d$unit)), "rows declared")
})

test_that("decalage et residuelle none : KR refuse, residuelle none comptee", {
  d <- .jeu_lag()
  m <- .modele_tb(d, lag = rx_lag(d$W, d$trait, d$unit))
  expect_error(rx_fit(m, kenward_roger = TRUE, verbose = FALSE), "Kenward-Roger")
  m0 <- .modele_tb(d, res = rx_residual("none"))
  expect_equal(m0$n_par, 3L)
  expect_error(rx_model(d$y, d$X, list(), rx_residual("none")), "V would be zero")
})

# --- ajustement -----------------------------------------------------------------
test_that("Psi tenue a 0 reproduit EXACTEMENT l'ajustement sans decalage", {
  skip_si_pas_de_jax()
  d <- .jeu_lag(Psi = matrix(c(0.05, -0.08, 0, 0), 2), seed = 2)
  f0 <- rx_fit(.modele_tb(d), backend = "cpu", verbose = FALSE)
  for (crit in c("reml_y", "jacobian")) {
    L <- rx_lag(d$W, d$trait, d$unit, struct = "mask", mask = matrix(FALSE, 2, 2), criterion = crit)
    f <- rx_fit(.modele_tb(d, lag = L), backend = "cpu", verbose = FALSE)
    expect_identical(f$logLik, f0$logLik)
    expect_identical(f$theta, f0$theta)
    expect_identical(f$beta, f0$beta)
    expect_identical(f$lag$logdet_A, 0)
  }
  # meme chose par fixed_theta : Psi libre mais tenue a son depart 0
  L <- rx_lag(d$W, d$trait, d$unit)
  f <- rx_fit(.modele_tb(d, lag = L), backend = "cpu", verbose = FALSE, fixed_theta = 7:10)
  expect_equal(f$logLik, f0$logLik, tolerance = 1e-10)
  expect_equal(f$theta[1:6], f0$theta, tolerance = 1e-6)
})

test_that("l'objectif egale un calcul dense independant (3 criteres, chemins fast et general)", {
  skip_si_pas_de_jax()
  d <- .jeu_lag(Psi = matrix(c(0.08, -0.1, 0.03, 0), 2), seed = 3)
  W2 <- d$W * 0.5; W2[1, 2] <- 2
  th <- c(0.1, 0.2, -0.3, 0.0, 0.3, -0.1, 0.4, -0.5, 0.6, 0.2)
  for (Wl in list(list(d$W), list(T1 = d$W, T2 = W2))) for (crit in c("reml_y", "jacobian", "ml")) {
    L <- rx_lag(if (length(Wl) == 1L) Wl[[1]] else Wl, d$trait, d$unit, criterion = crit)
    f <- rx_fit(.modele_tb(d, lag = L), backend = "cpu", verbose = FALSE, theta_init = th,
                maxiter = 0, polish = 0, hessian = FALSE, blups = FALSE)
    expect_identical(f$lag$path, if (length(Wl) == 1L) "fast" else "general")
    expect_equal(-2 * f$logLik, .neg2_dense(f, d, Wl, crit), tolerance = 1e-10)
  }
})

test_that("Psi est retrouvee sur un jeu simule avec retroaction (boucle a l'equilibre)", {
  skip_si_pas_de_jax()
  # 20 x 20 plantes, 100 genotypes ; T1 <- T1 = +0.6 / s et T2 <- T1 = -0.6 / s
  W <- .reine(20, 20); s <- mean(Matrix::rowSums(W))
  Psi <- matrix(c(0.6 / s, -0.6 / s, 0, 0), 2)
  d <- .jeu_lag(nr = 20, nc = 20, ng = 100, Psi = Psi, seed = 11, W = W)
  f <- rx_fit(.modele_tb(d, lag = rx_lag(W, d$trait, d$unit)), backend = "cpu", verbose = FALSE)
  expect_true(f$newton_decrement < 1e-4)
  expect_true(f$lag$spectral_radius > 0.3 && isTRUE(f$lag$admissible))
  z <- (f$lag$Psi - Psi) / f$lag$Psi_se
  expect_true(all(is.finite(z)))
  expect_true(all(abs(z) < 3.5))
  expect_true(f$lag$logdet_A < 0)
})

test_that("decalage + residuelle spatiale us(trait):ar1(row):ar1(col) par rx_reml", {
  skip_si_pas_de_jax()
  d <- .jeu_lag(nr = 12, nc = 10, ng = 30, Psi = matrix(c(0, 0.1, 0, 0), 2), seed = 5)
  dat <- data.frame(y = d$y, trait = d$trait, gen = factor(rep(d$g, 2)),
                    row = factor(d$row), col = factor(d$col))
  L <- rx_lag(d$W, d$trait, d$unit)
  f <- rx_reml(y ~ trait - 1, random = ~ us(gen), residual = ~ us(trait):ar1(row):ar1(col),
               data = dat, trait = "trait", lag = L, backend = "cpu", verbose = FALSE)
  expect_true(isTRUE(f$lag$admissible))
  expect_true(f$newton_decrement < 1e-4)
  expect_equal(dim(f$lag$Psi), c(2L, 2L))
  expect_true(any(grepl("residual", names(f[["rho"]]))))
  # Psi tenue a 0 : meme ajustement que sans decalage, residuelle spatiale comprise
  f0 <- rx_reml(y ~ trait - 1, random = ~ us(gen), residual = ~ us(trait):ar1(row):ar1(col),
                data = dat, trait = "trait", backend = "cpu", verbose = FALSE)
  fz <- rx_reml(y ~ trait - 1, random = ~ us(gen), residual = ~ us(trait):ar1(row):ar1(col),
                data = dat, trait = "trait", backend = "cpu", verbose = FALSE,
                lag = rx_lag(d$W, d$trait, d$unit, struct = "mask", mask = matrix(FALSE, 2, 2)))
  expect_identical(fz$logLik, f0$logLik)
  expect_true(f$logLik >= f0$logLik - 1e-8)
})

test_that("residuelle none : V non definie positive refusee en clair, VP+ ajuste", {
  skip_si_pas_de_jax()
  d <- .jeu_lag(nr = 5, nc = 4, ng = 6, seed = 6)
  expect_error(rx_fit(.modele_tb(d, res = rx_residual("none")), backend = "cpu", verbose = FALSE),
               "residual 'none'")
  N <- d$N; I_N <- Matrix::Diagonal(N) * 1; zN <- Matrix::Matrix(0, N, N, sparse = TRUE)
  Zp <- list(res_T1 = rbind(I_N, zN), res_T2 = rbind(zN, I_N),
             IEE_T1 = rbind(d$W, zN), IEE_T2 = rbind(zN, d$W))
  m <- rx_model(d$y, d$X, list(rx_term("gen", d$ZD, struct = "us"),
                                rx_term("plante", Zp, struct = "us")), rx_residual("none"))
  f <- rx_fit(m, backend = "cpu", verbose = FALSE, maxiter = 500)
  expect_null(f[["sigma_res"]])
  expect_equal(dim(f$sigmas[["plante"]]), c(4L, 4L))
  expect_output(print(f), "residual: none")
  # W symetrique : Cov(res_T1, IEE_T2) et Cov(res_T2, IEE_T1) ne sont identifiees
  # que par leur somme (theorie_v2.md T1b) : ici 3 (DGE us) + 10 (plante us 4)
  # = 13 parametres, dont 12 identifies.
  expect_equal(f$n_par, 13L)
  expect_equal(f$n_flat_expected, 1L)
  expect_equal(f$n_par_identified, 12L)
  s <- summary(f)
  expect_equal(s$n_estimated, 12L)
  expect_equal(s$aic, -2 * f$logLik + 2 * 12)
  expect_equal(s$varcomp[c("plante[3,2]", "plante[4,1]"), "bound"], c("N", "N"))
  expect_true(is.finite(s$varcomp["plante[3,2]+plante[4,1]", "std.error"]))
  expect_output(print(f), "flat direction\\(s\\)")
})

test_that("TB+ avec IEE* et Psi reciproque libre : avertissement ; masque : rien", {
  d <- .jeu_lag(nr = 5, nc = 4, ng = 6, seed = 6)
  N <- d$N; zN <- Matrix::Matrix(0, N, N, sparse = TRUE)
  Ziee <- list(IEE_T1 = rbind(d$W, zN), IEE_T2 = rbind(zN, d$W))
  termes <- list(rx_term("gen", d$ZD, struct = "us"), rx_term("iee", Ziee, struct = "us"))
  res <- rx_residual("us", trait = d$trait, unit = d$unit)
  expect_warning(rx_model(d$y, d$X, termes, res, lag = rx_lag(d$W, d$trait, d$unit)),
                 "weakly")
  expect_no_warning(rx_model(d$y, d$X, termes, res,
                             lag = rx_lag(d$W, d$trait, d$unit, receivers = "T2", emitters = "T1")))
  expect_no_warning(rx_model(d$y, d$X, termes[1], res, lag = rx_lag(d$W, d$trait, d$unit)))
})

# --- validation croisee contre igepath_reml (facultative) -----------------------
test_that("VP+, TB et TB+ : memes logLik et estimations que le moteur igepath_reml", {
  skip_si_pas_de_jax()
  ige <- Sys.getenv("REMLAX_IGEPATH")
  skip_if(!nzchar(ige) || !dir.exists(file.path(ige, "igepath_reml")),
          "REMLAX_IGEPATH ne designe pas le repertoire python/ du depot IGE")
  W <- .reine(10, 10); s <- mean(Matrix::rowSums(W))
  d <- .jeu_lag(nr = 10, nc = 10, ng = 25, seed = 21, W = W,
                Psi = matrix(c(0.5 / s, -1 / s, 0, 0), 2))
  dir <- tempfile("xval_"); dir.create(dir); on.exit(unlink(dir, recursive = TRUE))
  N <- d$N
  utils::write.csv(data.frame(id = sprintf("P%03d", seq_len(N)), bloc = 1, ligne = d$row[1:N],
                              colonne = d$col[1:N], genotype = sprintf("G%03d", d$g),
                              T1 = d$y[1:N], T2 = d$y[N + 1:N]),
                   file.path(dir, "plants.csv"), row.names = FALSE, quote = FALSE)
  sw <- Matrix::summary(methods::as(W, "generalMatrix"))
  utils::write.csv(data.frame(i = sw$i, j = sw$j, x = sw$x), file.path(dir, "W.csv"),
                   row.names = FALSE, quote = FALSE)
  py <- rx_python_cmd()
  out <- file.path(dir, "engine.json")
  old <- Sys.getenv("PYTHONPATH"); Sys.setenv(PYTHONPATH = ige); on.exit(Sys.setenv(PYTHONPATH = old), add = TRUE)
  st <- system2(py[1], c(py[-1], "-m", "igepath_reml.fit", "--data", dir, "--fits",
                         shQuote("VP+:T1,T2;TB:T1,T2;TB+:T1,T2"), "--out", out, "--no-se",
                         "--criterion", "reml_z"), stdout = FALSE, stderr = FALSE)
  expect_equal(as.integer(st), 0L)
  ref <- jsonlite::fromJSON(out, simplifyVector = FALSE)$fits
  names(ref) <- vapply(ref, function(r) r$model, "")
  ref <- ref[c("VP+", "TB", "TB+")]
  val <- function(m, k) { x <- ref[[m]]$estimates[[k]]$values; do.call(rbind, lapply(x, unlist)) }
  rel <- function(a, b) max(abs(a - b)) / max(abs(b))
  # VP+ : terme genetique [Z, WZ] us 4, terme de plante [I, W] us 4, sans residuelle
  zero <- Matrix::Matrix(0, N, d$ng, sparse = TRUE)
  WZ <- W %*% Matrix::sparseMatrix(i = seq_len(N), j = d$g, x = 1, dims = c(N, d$ng))
  Zgen <- c(d$ZD, list(IGE_T1 = rbind(WZ, zero), IGE_T2 = rbind(zero, WZ)))
  I_N <- Matrix::Diagonal(N) * 1; zN <- Matrix::Matrix(0, N, N, sparse = TRUE)
  Zp <- list(res_T1 = rbind(I_N, zN), res_T2 = rbind(zN, I_N), IEE_T1 = rbind(W, zN), IEE_T2 = rbind(zN, W))
  res_us <- rx_residual("us", trait = d$trait, unit = d$unit)
  L <- rx_lag(W, d$trait, d$unit)
  fits <- list(
    rx_fit(rx_model(d$y, d$X, list(rx_term("gen", Zgen, struct = "us"),
                                   rx_term("plante", Zp, struct = "us")), rx_residual("none")),
           backend = "cpu", verbose = FALSE, blups = FALSE),
    rx_fit(.modele_tb(d, lag = L), backend = "cpu", verbose = FALSE, blups = FALSE),
    rx_fit(rx_model(d$y, d$X, list(rx_term("gen", Zgen, struct = "us")), res_us, lag = L),
           backend = "cpu", verbose = FALSE, blups = FALSE))
  for (k in 1:3) {
    expect_equal(fits[[k]]$logLik, ref[[k]]$logLik, tolerance = 1e-5)
    G <- fits[[k]]$sigmas[["gen"]]
    expect_lt(rel(G, val(k, "G")[seq_len(nrow(G)), seq_len(nrow(G))]), 1e-4)
  }
  expect_lt(rel(fits[[2]]$lag$Psi, val(2, "Psi")), 1e-4)
  expect_lt(rel(fits[[3]]$lag$Psi, val(3, "Psi")), 1e-4)
  expect_lt(rel(as.matrix(fits[[3]][["sigma_res"]]), val(3, "residual")), 1e-4)
})

# --- moyennes non nulles et plantes de bordure (rx_border) ----------------------
test_that("rx_border : W 1 par caractere, puissances, etiquettes, W par caractere", {
  d <- .jeu_lag()
  B <- rx_border(d$W, d$trait, d$unit)
  expect_equal(dim(B), c(length(d$y), 2L))
  expect_equal(colnames(B), c("border1:T1", "border1:T2"))
  s <- Matrix::rowSums(d$W); sb <- mean(s)
  expect_equal(B[d$trait == "T1", 1], s[d$unit[d$trait == "T1"]] / sb)
  expect_true(all(B[d$trait == "T2", 1] == 0))
  B2 <- rx_border(d$W, d$trait, d$unit, order = 2, scale = FALSE)
  expect_equal(colnames(B2), c("border1:T1", "border1:T2", "border2:T1", "border2:T2"))
  expect_equal(B2[d$trait == "T2", 4], as.numeric(d$W %*% s)[d$unit[d$trait == "T2"]])
  # unites par etiquette, dans un autre ordre que W
  W2 <- d$W; dimnames(W2) <- list(paste0("p", seq_len(d$N)), paste0("p", seq_len(d$N)))
  o <- sample(length(d$y))
  expect_equal(rx_border(W2, d$trait[o], paste0("p", d$unit[o])), B[o, ], ignore_attr = TRUE)
  # une W par caractere receveur
  Wb <- d$W; Wb[1, 2] <- Wb[2, 1] <- 0
  Bl <- rx_border(list(T1 = d$W, T2 = Wb), d$trait, d$unit, scale = FALSE)
  expect_equal(Bl[d$trait == "T2", 2], Matrix::rowSums(Wb)[d$unit[d$trait == "T2"]])
  expect_error(rx_border(d$W, d$trait, d$unit, order = 0), "order")
})

test_that("rx_border : la moyenne structurelle sans boucle est dans [X, rx_border]", {
  # E y = (I - M)^-1 X mu. Sans boucle (seul T2 <- T1), elle est EXACTEMENT dans
  # l'espace engendre par X et W 1 par caractere ; avec boucle, l'ordre 2 reduit
  # l'ecart d'un facteur important.
  d <- .jeu_lag(nr = 7, nc = 6)
  N <- d$N; mu <- c(10, 4)
  resid <- function(Psi, order) {
    M <- kronecker(Psi, as.matrix(d$W))
    Ey <- solve(diag(2 * N) - M, rep(mu, each = N))
    Xb <- cbind(d$X, rx_border(d$W, d$trait, d$unit, order = order))
    r <- stats::lm.fit(Xb, Ey)$residuals
    sqrt(mean(r^2))
  }
  Ps <- matrix(c(0, -2, 0, 0), 2) / 6.5
  expect_lt(resid(Ps, 1), 1e-10)
  # sans la covariable : l'ecart vaut |psi mu_1| fois l'ecart-type de s_i
  r0 <- sqrt(mean(stats::lm.fit(d$X, solve(diag(2 * N) - kronecker(Ps, as.matrix(d$W)),
                                           rep(mu, each = N)))$residuals^2))
  expect_gt(r0, 1)
  Pl <- matrix(c(-0.6, -0.6, 0, 0), 2) / 6.5
  expect_gt(resid(Pl, 1), 1e-3)
  expect_lt(resid(Pl, 2), resid(Pl, 1) / 4)
})

test_that("rx_border dans X : ajustement TB avec un effet de bordure qui n'est pas une interaction", {
  skip_si_pas_de_jax()
  # moyennes 10 et un effet de bordure -0,3 par voisine en plus (lumiere en rive),
  # sur les deux caracteres, dans x ; vrai Psi : T2 <- T1 seul
  set.seed(5)
  nr <- 10; nc <- 8; N <- nr * nc; ng <- 20
  W <- .reine(nr, nc); s <- Matrix::rowSums(W)
  Psi <- matrix(c(0, -0.25, 0, 0), 2, dimnames = list(c("T1", "T2"), c("T1", "T2")))
  g <- sample(rep(seq_len(ng), length.out = N))
  a <- matrix(stats::rnorm(ng * 2), ng) * sqrt(0.5)
  e <- matrix(stats::rnorm(N * 2), N) * sqrt(0.5)
  x <- as.numeric(a[g, ] + e) + rep(c(10, 10), each = N) - 0.3 * rep(s - mean(s), 2)
  y <- as.numeric(solve(diag(2 * N) - kronecker(Psi, as.matrix(W)), x))
  trait <- factor(rep(c("T1", "T2"), each = N)); unit <- rep(seq_len(N), 2)
  X <- stats::model.matrix(~ trait - 1)
  Zg <- Matrix::sparseMatrix(i = seq_len(N), j = g, x = 1, dims = c(N, ng))
  zero <- Matrix::Matrix(0, N, ng, sparse = TRUE)
  ZD <- list(T1 = rbind(Zg, zero), T2 = rbind(zero, Zg))
  L <- rx_lag(W, trait, unit, receivers = "T2", emitters = "T1")
  ajuste <- function(X) rx_fit(rx_model(y, X, list(rx_term("gen", ZD, struct = "us")),
                                        rx_residual("us", trait = trait, unit = unit), lag = L),
                               backend = "cpu", verbose = FALSE)
  fb <- ajuste(cbind(X, rx_border(W, trait, unit)))
  expect_length(fb$beta, 4L)
  psi_b <- fb$lag$Psi["T2", "T1"]
  se_b <- fb$lag$Psi_se["T2", "T1"]
  expect_lt(abs(psi_b - Psi["T2", "T1"]), 3 * se_b)
  # la pente de bordure est l'effet structurel : -0,3 x s_bar par unite de s_i / s_bar
  expect_lt(abs(fb$beta[3] - (-0.3 * mean(s))), 4 * sqrt(fb$vbeta[3, 3]))
})
