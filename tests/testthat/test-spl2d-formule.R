# spl2d() dans la formule de rx_reml() : meme ajustement que la voie explicite.

skip_if_not(rx_python_check(quiet = TRUE)$ok, "pas de Python avec jax")

jeu <- function() {
  set.seed(7)
  g <- expand.grid(r = 1:12, c = 1:10)
  g$gid <- factor(sample(rep(1:30, 4)))
  g$trt <- factor(rep(c("a", "b"), length.out = nrow(g)))
  g$y <- 0.05 * (g$r - 6)^2 + 0.3 * sin(g$c / 2) + 0.4 * (g$trt == "b") +
    rnorm(30)[g$gid] + rnorm(nrow(g), 0, 0.5)
  g
}

test_that("spl2d() dans rx_reml() equivaut a rx_spl2d() + rx_model()", {
  g <- jeu()
  f <- rx_reml(y ~ trt, random = ~ gid + spl2d(r, c, nseg = c(4, 4)), data = g,
               backend = "cpu", verbose = FALSE)
  sp <- rx_spl2d(g$r, g$c, nseg = c(4, 4))
  m <- rx_model(g$y, cbind(model.matrix(~ trt, g), sp$X),
                c(list(rx_term("gid", g$gid)), sp$terms))
  fe <- rx_fit(m, backend = "cpu", verbose = FALSE)
  expect_equal(f$logLik, fe$logLik, tolerance = 1e-8)
  expect_identical(names(f$sigmas), c("gid", paste0("spl_", c("fx", "fy", "fx_y", "x_fy", "fx_fy"))))
  expect_identical(f$spl2d$columns, colnames(sp$X))
  expect_identical(attr(f$model$X, "termes"), c("(Intercept)", "trt", "spl2d(r, c, nseg = c(4, 4))"))
})

test_that("spl2d() : nom, arguments nommes, coordonnees en facteur, erreurs", {
  g <- jeu(); g$rf <- factor(g$r)
  f <- rx_reml(y ~ 1, random = ~ spl2d(x = rf, y = c, nseg = c(3, 3), name = "champ"), data = g,
               backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_identical(names(f$sigmas), paste0("champ_", c("fx", "fy", "fx_y", "x_fy", "fx_fy")))
  g$lettre <- factor(letters[g$r])
  expect_error(rx_reml(y ~ 1, random = ~ spl2d(lettre, c), data = g, backend = "cpu"),
               "non numeriques")
  expect_error(rx_reml(y ~ 1, random = ~ spl2d(r), data = g, backend = "cpu"), "y manquante")
})

test_that("spl2d() : predictions, Wald et summary tiennent compte de la partie nulle", {
  g <- jeu()
  f <- rx_reml(y ~ trt, random = ~ gid + spl2d(r, c, nseg = c(4, 4)), data = g,
               backend = "cpu", verbose = FALSE, wald = TRUE)
  p <- rx_predict(f, classify = "trt")
  expect_identical(nrow(p), 2L)
  expect_true(all(p$estimable))
  # l'ecart des deux traitements est exactement le coefficient trtb
  expect_equal(diff(p$predicted.value), unname(f$beta[2]), tolerance = 1e-10)
  expect_true("spl2d(r, c, nseg = c(4, 4))" %in% f$wald$tests$terme)
  s <- summary(f)
  expect_true(all(paste0("spl_", c("fx", "fy", "fx_y", "x_fy", "fx_fy")) %in% rownames(s$varcomp)))
  expect_true(all(grepl("^spl_lin", tail(rownames(s$coef.fixed), length(f$spl2d$columns)))))
})

# ------------------------------------------------------------------------------
# Une surface par niveau : spl2d(x, y, at = bloc), a la maniere de spl2Dc() de sommer
# ------------------------------------------------------------------------------
jeu_blocs <- function() {
  set.seed(3)
  g <- do.call(rbind, lapply(1:3, function(b) {
    d <- expand.grid(row = 1:10, col = 1:6); d$bloc <- paste0("B", b); d }))
  g$bloc <- factor(g$bloc)
  g$gid <- factor(unlist(lapply(1:3, function(b) sample(1:60))))
  tend <- with(g, ifelse(bloc == "B1", 0.05 * (row - 5)^2,
                         ifelse(bloc == "B2", -0.5 * cos(col / 2), 0.08 * row)))
  g$y <- 10 + c(0, 1, -1)[g$bloc] + tend + rnorm(60, 0, 0.8)[g$gid] + rnorm(nrow(g), 0, 0.5)
  g
}

test_that("spl2d(at = ) : une surface par niveau, egale a la voie explicite", {
  g <- jeu_blocs()
  f <- rx_reml(y ~ bloc, random = ~ gid + spl2d(col, row, nseg = c(3, 5), at = bloc), data = g,
               backend = "cpu", verbose = FALSE)
  sfx <- c("fx", "fy", "fx_y", "x_fy", "fx_fy")
  expect_identical(names(f$sigmas), c("gid", paste0("spl_", rep(c("B1", "B2", "B3"), each = 5), "_", sfx)))
  expect_identical(names(f$spl2d$surfaces), c("spl_B1", "spl_B2", "spl_B3"))
  expect_length(f$spl2d$columns, 9L)
  expect_identical(f$spl2d$surfaces$spl_B2$rows, which(g$bloc == "B2"))
  expect_identical(f$spl2d$surfaces$spl_B2$at, list(name = "bloc", level = "B2"))
  # voie explicite : rx_spl2d() sur les lignes de chaque bloc, completee par des zeros
  Xs <- list(); tms <- list()
  for (b in levels(g$bloc)) {
    r <- which(g$bloc == b)
    sp <- rx_spl2d(g$col[r], g$row[r], nseg = c(3, 5), prefix = paste0("spl_", b))
    plein <- function(M) { P <- matrix(0, nrow(g), ncol(M)); P[r, ] <- as.matrix(M); P }
    Xs[[b]] <- plein(sp$X)
    tms <- c(tms, lapply(sp$terms, function(tm) rx_term(tm$name, list(plein(tm$Zl[[1]])))))
  }
  m <- rx_model(g$y, cbind(model.matrix(~ bloc, g), do.call(cbind, Xs)),
                c(list(rx_term("gid", g$gid)), tms))
  fe <- rx_fit(m, backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_equal(f$logLik, fe$logLik, tolerance = 1e-8)
  # dimensions : la somme vaut n ; chaque surface a sa ligne de correspondance SpATS
  dd <- rx_dimensions(f)
  expect_equal(sum(dd$Effective), nrow(g), tolerance = 1e-8)
  expect_length(summary(f)$surfaces, 3L)
  # une prediction par bloc tient la partie nulle a zero, donc la moyenne de la surface
  p <- rx_predict(f, classify = "bloc")
  expect_identical(nrow(p), 3L); expect_true(all(p$estimable))
})

test_that("spl2d(at = ) : un seul niveau equivaut a une surface sans at", {
  g <- jeu_blocs(); g$tout <- factor("A")
  f0 <- rx_reml(y ~ 1, random = ~ gid + spl2d(col, row, nseg = c(3, 5)), data = g,
                backend = "cpu", verbose = FALSE, hessian = FALSE)
  f1 <- suppressMessages(rx_reml(y ~ 1, random = ~ gid + spl2d(col, row, nseg = c(3, 5), at = tout),
                                 data = g, backend = "cpu", verbose = FALSE, hessian = FALSE))
  expect_equal(f1$logLik, f0$logLik, tolerance = 1e-10)
  expect_identical(names(f1$spl2d$surfaces), "spl_A")
})

test_that("spl2d(at = ) : at.levels, at.var, message et erreurs", {
  g <- jeu_blocs()
  f <- rx_reml(y ~ bloc, random = ~ gid + spl2d(col, row, nseg = c(3, 5), at.var = bloc,
                                                at.levels = c("B1", "B3")),
               data = g, backend = "cpu", verbose = FALSE, hessian = FALSE)
  expect_identical(names(f$spl2d$surfaces), c("spl_B1", "spl_B3"))
  expect_true(all(f$model$X[g$bloc == "B2", f$spl2d$columns] == 0))
  expect_message(rx_reml(y ~ 1, random = ~ gid + spl2d(col, row, nseg = c(3, 5), at = bloc),
                         data = g, backend = "cpu", verbose = FALSE, hessian = FALSE),
                 "ni dans les effets fixes")
  expect_error(rx_reml(y ~ bloc, random = ~ spl2d(col, row, at = bloc, at.levels = "B9"),
                       data = g, backend = "cpu"), "absents")
  g$col[g$bloc == "B3"] <- 1
  expect_error(rx_reml(y ~ bloc, random = ~ spl2d(col, row, at = bloc), data = g, backend = "cpu"),
               "une valeur de x")
})

test_that("spl2d(at = ) : tendance et cartes de toutes les surfaces", {
  g <- jeu_blocs()
  f <- rx_reml(y ~ bloc, random = ~ gid + spl2d(col, row, nseg = c(3, 5), at = bloc), data = g,
               backend = "cpu", verbose = FALSE)
  tr <- rx_spatial_trend(f, surface = "all", grid = c(8, 6))
  expect_identical(names(tr), c("surface", "level", "col", "row", "trend"))
  expect_identical(nrow(tr), 3L * 48L)
  expect_identical(unique(tr$level), c("B1", "B2", "B3"))
  expect_equal(tr$trend[tr$surface == "spl_B2"], rx_spatial_trend(f, "spl_B2", grid = c(8, 6))$trend)
  avant <- file.exists("Rplots.pdf"); nd <- length(grDevices::dev.list())
  pdf_f <- tempfile(fileext = ".pdf")
  out <- plot(f, surface = "all", file = pdf_f)
  expect_identical(names(out), c("spl_B1", "spl_B2", "spl_B3"))
  expect_identical(nrow(out$spl_B2$plots), sum(g$bloc == "B2"))
  expect_equal(out$spl_B2$plots$fitted, fitted(f)[g$bloc == "B2"])
  png_f <- tempfile(fileext = ".png")
  out2 <- plot(f, surface = c("spl_B1", "spl_B3"), file = png_f)
  fich <- vapply(out2, `[[`, "", "file")
  expect_true(all(file.exists(fich)))
  expect_identical(unname(basename(fich)),
                   paste0(sub("\\.png$", "", basename(png_f)), c("_spl_B1.png", "_spl_B3.png")))
  expect_identical(length(grDevices::dev.list()), nd)
  if (!avant) expect_false(file.exists("Rplots.pdf"))
})
