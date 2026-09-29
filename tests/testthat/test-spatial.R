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
  expect_true(all(c("gid", "spl_x", "spl_y", "spl_xy") %in% names(f$pev)))
  dd <- summary(f)$dimensions
  expect_identical(rownames(dd), c("(Intercept)", "spl2d(col, row, nseg = c(5, 8))", "gid",
                                   "spl_x", "spl_y", "spl_xy", "Residual"))
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
