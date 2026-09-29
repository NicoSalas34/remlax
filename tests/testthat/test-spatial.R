# Sorties spatiales a la maniere de SpATS.

skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")

test_that("dimension effective : plan equilibre, ratio = heritabilite sur moyennes, somme = n", {
  set.seed(1); q <- 40; r <- 3
  d <- data.frame(gid = factor(rep(1:q, each = r)))
  d$y <- 10 + rnorm(q, 0, 1.2)[d$gid] + rnorm(q * r)
  f <- rx_reml(y ~ 1, random = ~ gid, data = d, backend = "cpu", verbose = FALSE, pev = TRUE)
  dd <- rx_dimensions(f)
  sg <- f$sigmas$gid[1, 1]; se <- f$sigma_res[1, 1]
  expect_equal(dd["gid", "Ratio"], r * sg / (r * sg + se), tolerance = 1e-8)
  expect_equal(dd["gid", "Nominal"], q - 1)
  expect_equal(sum(dd$Effective), nrow(d), tolerance = 1e-8)
  expect_equal(rx_heritability(f, "gid"), dd["gid", "Ratio"])
  expect_error(rx_heritability(f, "absent"), "n'est pas un terme")
})

jeu <- function() {
  set.seed(7); g <- expand.grid(row = 1:16, col = 1:10)
  g$gid <- factor(sample(rep(1:40, 4)))
  g$vrai <- 0.03 * (g$row - 8)^2 - 0.4 * cos(g$col / 3)
  g$y <- 5 + g$vrai + rnorm(40)[g$gid] + rnorm(nrow(g), 0, 0.5)
  g
}

test_that("spl2d : PEV demandees par defaut, dimensions, tendance, cartes exportees", {
  g <- jeu()
  f <- rx_reml(y ~ 1, random = ~ gid + spl2d(col, row, nseg = c(5, 8)), data = g,
               backend = "cpu", verbose = FALSE)
  sp5 <- paste0("spl_", c("fx", "fy", "fx_y", "x_fy", "fx_fy"))
  expect_true(all(c("gid", sp5) %in% names(f$pev)))
  dd <- summary(f)$dimensions
  expect_identical(rownames(dd), c("(Intercept)", "spl2d(col, row, nseg = c(5, 8))", "gid",
                                   sp5, "Residual"))
  expect_equal(sum(dd$Effective), nrow(g), tolerance = 1e-8)
  expect_true(all(dd$Effective[dd$Type == "random"] <= dd$Nominal[dd$Type == "random"] + 1e-8))
  tg <- rx_spatial_trend(f, grid = c(30, 20))
  expect_identical(dim(tg), c(600L, 3L)); expect_identical(names(tg), c("col", "row", "trend"))
  avant <- file.exists("Rplots.pdf"); nd <- length(grDevices::dev.list())
  png_f <- tempfile(fileext = ".png"); pdf_f <- tempfile(fileext = ".pdf")
  out <- plot(f, file = png_f); plot(f, spaTrend = "percentage", file = pdf_f)
  expect_true(file.exists(png_f) && file.size(png_f) > 1000 && file.exists(pdf_f))
  expect_identical(names(out$plots), c("col", "row", "observed", "fitted", "residual", "trend"))
  expect_equal(out$plots$residual, residuals(f))
  expect_equal(out$plots$fitted + out$plots$residual, g$y)
  expect_gt(cor(out$plots$trend, g$vrai), 0.95)
  # la tendance sur la grille, evaluee aux parcelles, retrouve la tendance aux parcelles
  s <- f$spl2d$surfaces$spl
  expect_equal(remlax:::.rx_trend_at(f, s, g$col, g$row), out$plots$trend)
  expect_error(plot(f, file = tempfile(fileext = ".xyz")), "extension")
  # aucun peripherique laisse ouvert, aucun Rplots.pdf cree
  expect_identical(length(grDevices::dev.list()), nd)
  if (!avant) expect_false(file.exists("Rplots.pdf"))
  expect_error(rx_spatial_trend(f, surface = "autre"), "inconnue")
})

# ------------------------------------------------------------------------------
# Equivalence avec le PSANOVA de SpATS (Rodriguez-Alvarez et al. 2018)
# ------------------------------------------------------------------------------
test_that("spl2d : memes covariances que PSANOVA de SpATS, terme a terme", {
  skip_if_not_installed("SpATS")
  cons <- get0("construct.2d.pspline", envir = asNamespace("SpATS"), inherits = FALSE)
  skip_if(is.null(cons), "construct.2d.pspline absent de cette version de SpATS")
  g <- expand.grid(row = 1:12, col = 1:9)
  for (nd in list(c(1L, 1L), c(2L, 3L))) {
    sp <- rx_spl2d(g$col, g$row, nseg = c(4L, 6L), nest.div = nd)
    ref <- cons(stats::as.formula(sprintf("~ SpATS::PSANOVA(col, row, nseg = c(4, 6), nest.div = c(%d, %d))",
                                          nd[1], nd[2])), g, rep(FALSE, nrow(g)))
    q <- ref$dim$random; fin <- cumsum(q); deb <- fin - q + 1L
    expect_identical(vapply(sp$terms, `[[`, 1L, "q"), as.integer(q))
    for (k in 1:5) {
      Zs <- ref$Z[, deb[k]:fin[k], drop = FALSE]; gk <- ref$g[[k]][deb[k]:fin[k]]
      Cs <- Zs %*% (t(Zs) / gk)
      Cr <- tcrossprod(as.matrix(sp$terms[[k]]$Zl[[1]]))
      expect_lt(max(abs(Cr - Cs)) / max(abs(Cs)), 1e-10)
    }
    expect_identical(qr(cbind(1, sp$X, ref$X))$rank, 4L)
  }
})

test_that("spl2d : meme ajustement que SpATS (variances, dimensions, H2, valeurs ajustees)", {
  skip_if_not_installed("SpATS")
  set.seed(7); g <- expand.grid(row = 1:16, col = 1:10)
  g$geno <- factor(sample(rep(1:40, 4)))
  g$y <- 5 + 0.03 * (g$row - 8)^2 - 0.4 * cos(g$col / 3) + 0.05 * g$row * sin(g$col / 2) +
    rnorm(40)[g$geno] + rnorm(nrow(g), 0, 0.5)
  fs <- SpATS::SpATS("y", genotype = "geno", genotype.as.random = TRUE,
                     spatial = ~ SpATS::PSANOVA(col, row, nseg = c(5, 8)), data = g,
                     control = SpATS::controlSpATS(tolerance = 1e-10, maxit = 5000, monitoring = 0))
  fr <- rx_reml(y ~ 1, random = ~ geno + spl2d(col, row, nseg = c(5, 8)), data = g,
                backend = "cpu", verbose = FALSE)
  termes <- c("geno", fr$spl2d$surfaces$spl$basis$terms)
  vr <- vapply(termes, function(k) fr$sigmas[[k]][1, 1], 0)
  vs <- unname(fs$var.comp)
  pos <- vs > 1e-6 * max(vs)                        # composantes hors de la borne
  expect_equal(unname(vr[pos]), vs[pos], tolerance = 1e-4)
  expect_true(all(vr[!pos] < 1e-6 * max(vs)))
  expect_equal(fr$sigma_res[1, 1], unname(fs$psi[1]), tolerance = 1e-4)
  dd <- rx_dimensions(fr)
  eds <- fs$eff.dim[names(fs$var.comp)]
  expect_equal(unname(dd[termes, "Effective"][pos]), unname(eds[pos]), tolerance = 1e-4)
  expect_equal(fitted(fr), as.numeric(fs$fitted), tolerance = 1e-5)
  expect_equal(rx_heritability(fr, "geno"), unname(eds["geno"]) / 39, tolerance = 1e-4)
})
