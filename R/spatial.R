# ==============================================================================
# spatial.R - spatial outputs in the manner of SpATS
# ==============================================================================
# Three outputs taken from SpATS (Rodriguez-Alvarez et al. 2018):
#   - the table of dimensions: for each component the effective dimension ED
#     (effective degrees of freedom), the model dimension, the nominal
#     dimension rank[X, Z_k] - rank[X], and their ratio;
#   - the generalised heritability of Oakey et al. (2006), ED_g / nominal_g;
#   - the spatial surface on a fine grid, and maps: data, fitted values,
#     residuals, trend, genotype BLUPs.
#
# Effective dimension of a random term u ~ N(0, s2 I_q):
#   ED = q - tr(PEV) / s2,
# PEV being the prediction error variance of the q BLUPs. Only its diagonal
# enters, returned by the solver with pev = TRUE. The formula assumes K = I:
# with a relationship matrix it would need tr(K^-1 PEV), hence the full PEV,
# and the term is left without ED.

# Fitted values X beta + sum of Z u.
fitted.rx_fit <- function(object, ...) {
  m <- object$model
  if (is.null(m)) stop("fitted : l'ajustement ne porte pas son modele.", call. = FALSE)
  f <- as.numeric(m$X %*% object$beta)
  for (tm in m$terms) {
    B <- object$blups[[tm$name]]
    if (is.null(B)) next
    B <- as.matrix(B)
    if (length(tm$Zl) != ncol(B)) next
    for (j in seq_along(tm$Zl)) f <- f + as.numeric(tm$Zl[[j]] %*% B[, j])
  }
  f
}

# Residuals y - fitted. With a spatial lag the fitted values are those of
# y* = (I - M) y (the model linear in beta and u), and the residuals y* - fitted.
residuals.rx_fit <- function(object, ...) {
  if (is.null(object$model$y)) stop("residuals : l'ajustement ne porte pas son modele.", call. = FALSE)
  as.numeric(object[["ystar"]] %||% object$model$y) - fitted.rx_fit(object)
}

# Names of the requested surfaces: NULL is the first, "all" all of them.
.rx_surface_names <- function(fit, surface) {
  sf <- fit$spl2d$surfaces
  if (!length(sf)) stop("aucune surface spl2d() dans cet ajustement.", call. = FALSE)
  if (is.null(surface)) return(names(sf)[1])
  if (identical(surface, "all")) return(names(sf))
  unknown <- setdiff(surface, names(sf))
  if (length(unknown))
    stop("surface '", unknown[1], "' inconnue ; disponibles : ", paste(names(sf), collapse = ", "),
         call. = FALSE)
  surface
}

# One surface by name (the first when NULL).
.rx_surface <- function(fit, surface) {
  sf <- fit$spl2d$surfaces
  if (!length(sf)) stop("aucune surface spl2d() dans cet ajustement.", call. = FALSE)
  if (is.null(surface)) surface <- names(sf)[1]
  if (!surface %in% names(sf))
    stop("surface '", surface, "' inconnue ; disponibles : ", paste(names(sf), collapse = ", "),
         call. = FALSE)
  sf[[surface]]
}

# Trend of a surface at (x, y): the null part plus the smooth parts.
.rx_trend_at <- function(fit, s, x, y) {
  e <- .rx_spl2d_eval(s$basis, x, y)
  pre <- s$basis$prefix
  nm <- colnames(fit$model$X)
  b <- fit$beta[match(s$basis$columns, nm)]
  tr <- as.numeric(e$X %*% b)
  for (k in names(e$Z)) {
    u <- fit$blups[[paste0(pre, "_", k)]]
    if (!is.null(u)) tr <- tr + as.numeric(e$Z[[k]] %*% as.numeric(u))
  }
  tr
}

rx_spatial_trend <- function(fit, surface = NULL, grid = c(100L, 100L)) {
  surface_names <- .rx_surface_names(fit, surface)
  if (length(surface_names) > 1L) {
    # Several surfaces: one table, with the surface and its level.
    tabs <- lapply(surface_names, function(nm) {
      g <- rx_spatial_trend(fit, surface = nm, grid = grid)
      at <- fit$spl2d$surfaces[[nm]]$at
      names(g)[1:2] <- c("x", "y")
      cbind(surface = nm, level = if (is.null(at)) NA_character_ else at$level, g) })
    out <- do.call(rbind, tabs)
    s1 <- fit$spl2d$surfaces[[surface_names[1]]]
    same <- all(vapply(surface_names, function(nm) identical(c(fit$spl2d$surfaces[[nm]]$x_name,
                                                      fit$spl2d$surfaces[[nm]]$y_name),
                                                    c(s1$x_name, s1$y_name)), TRUE))
    if (same) names(out)[3:4] <- c(s1$x_name, s1$y_name)
    rownames(out) <- NULL
    return(out)
  }
  s <- .rx_surface(fit, surface_names)
  grid <- rep_len(as.integer(grid), 2L)
  gx <- seq(min(s$x), max(s$x), length.out = grid[1])
  gy <- seq(min(s$y), max(s$y), length.out = grid[2])
  g <- expand.grid(x = gx, y = gy)
  g$trend <- .rx_trend_at(fit, s, g$x, g$y)
  names(g)[1:2] <- c(s$x_name, s$y_name)
  attr(g, "surface") <- s$basis$prefix
  g
}

# Table of dimensions, in the manner of summary.SpATS.
rx_dimensions <- function(fit) {
  m <- fit$model
  if (is.null(m)) stop("rx_dimensions : l'ajustement ne porte pas son modele.", call. = FALSE)
  X <- as.matrix(m$X); n <- nrow(X)
  rX <- qr(X, tol = 1e-7)$rank
  rows <- list()
  # Fixed effects: a term counts its independent columns.
  asg <- attr(m$X, "assign"); lab <- attr(m$X, "term_labels")
  if (!is.null(asg) && !is.null(lab)) {
    ua <- sort(unique(asg))
    for (k in seq_along(ua)) {
      d <- qr(X[, asg == ua[k], drop = FALSE], tol = 1e-7)$rank
      rows[[length(rows) + 1L]] <- data.frame(component = lab[k], Effective = d, Model = d,
                                                  Nominal = d, Type = "fixed")
    }
  } else {
    rows[[1]] <- data.frame(component = "fixed", Effective = rX, Model = ncol(X),
                              Nominal = rX, Type = "fixed")
  }
  for (tm in m$terms) {
    S <- fit$sigmas[[tm$name]]; P <- fit$pev[[tm$name]]
    ok <- tm$t == 1L && identical(tm$struct, "iid") && is.null(tm$LK) &&
      (is.null(tm$level) || tm$level %in% c("id", "auto")) && !is.null(S) && !is.null(P)
    Z <- tm$Zl[[1]]
    ed <- if (ok) tm$q - sum(as.numeric(P)) / S[1, 1] else NA_real_
    nominal <- if (tm$t == 1L) qr(cbind(X, as.matrix(Z)), tol = 1e-7)$rank - rX else NA_real_
    rows[[length(rows) + 1L]] <- data.frame(component = tm$name, Effective = ed,
                                            Model = tm$q * tm$t, Nominal = nominal, Type = "random")
  }
  out <- do.call(rbind, rows)
  out$Ratio <- out$Effective / out$Nominal
  tot <- sum(out$Effective)
  res <- data.frame(component = "Residual", Effective = n - tot, Model = n, Nominal = n - rX,
                    Type = "residual", Ratio = (n - tot) / (n - rX))
  out <- rbind(out, res)
  out <- out[, c("component", "Effective", "Model", "Nominal", "Ratio", "Type")]
  rownames(out) <- out$component
  attr(out, "total_ed") <- tot
  out
}

rx_heritability <- function(fit, genotype) {
  d <- rx_dimensions(fit)
  if (!genotype %in% rownames(d) || d[genotype, "Type"] != "random")
    stop("rx_heritability : '", genotype, "' n'est pas un terme aleatoire du modele.", call. = FALSE)
  h <- d[genotype, "Ratio"]
  if (is.na(h)) stop("rx_heritability : dimension effective non calculable pour '", genotype,
                     "' (refaire l'ajustement avec pev = TRUE ; terme iid a un caractere sans ",
                     "matrice de parente).", call. = FALSE)
  h
}

# ---- maps --------------------------------------------------------------------

# A matrix on the grid of unique coordinates; NULL when the grid is not
# regular (more than four cells per observation).
.rx_grid <- function(x, y, z) {
  ux <- sort(unique(x)); uy <- sort(unique(y))
  if (length(ux) * length(uy) > 4 * length(z)) return(NULL)
  M <- matrix(NA_real_, length(ux), length(uy))
  ix <- match(x, ux); iy <- match(y, uy)
  s <- tapply(z, list(ix, iy), mean)
  M[as.integer(rownames(s)), as.integer(colnames(s))] <- s
  list(x = ux, y = uy, z = M)
}

# One map with its colour key.
.rx_map <- function(x, y, z, main, pal, zlim, xlab, ylab) {
  graphics::par(mar = c(3.2, 3.2, 2, 0.4), mgp = c(2, 0.6, 0))
  g <- .rx_grid(x, y, z)
  if (!is.null(g)) {
    graphics::image(g$x, g$y, g$z, col = pal, zlim = zlim, main = main, xlab = xlab, ylab = ylab,
                    useRaster = length(g$x) * length(g$y) > 2e4)
  } else {
    k <- cut(pmin(pmax(z, zlim[1]), zlim[2]), length(pal), labels = FALSE, include.lowest = TRUE)
    graphics::plot(x, y, pch = 15, col = pal[k], main = main, xlab = xlab, ylab = ylab)
  }
  graphics::box()
  graphics::par(mar = c(3.2, 0.3, 2, 2.6))
  v <- seq(zlim[1], zlim[2], length.out = length(pal))
  graphics::image(1, v, matrix(v, 1), col = pal, axes = FALSE, xlab = "", ylab = "")
  graphics::axis(4, las = 1, cex.axis = 0.8); graphics::box()
}

plot.rx_fit <- function(x, genotype = NULL, surface = NULL, spaTrend = c("raw", "percentage"),
                        grid = c(100L, 100L), file = NULL, width = 12, height = 7.5, res = 150, ...) {
  fit <- x; spaTrend <- match.arg(spaTrend)
  surface_names <- .rx_surface_names(fit, surface)
  if (is.null(genotype)) {
    term_names <- vapply(fit$model$terms, `[[`, "", "name")
    others <- setdiff(term_names, unlist(lapply(fit$spl2d$surfaces, function(z) z$basis$terms)))
    genotype <- if (length(others)) others[1] else NA_character_
  }
  ext <- if (!is.null(file)) tolower(tools::file_ext(file)) else ""
  if (!is.null(file) && !ext %in% c("png", "jpg", "jpeg", "tiff", "tif", "pdf", "svg"))
    stop("plot : extension '", ext, "' non geree (png, jpeg, tiff, pdf, svg).", call. = FALSE)
  # Several surfaces in a one-page format: one file per surface, suffixed by
  # the name of the surface.
  if (length(surface_names) > 1L && !is.null(file) && ext != "pdf") {
    out <- lapply(surface_names, function(nm) {
      f <- sub(paste0("\\.", tools::file_ext(file), "$"), paste0("_", nm, ".", tools::file_ext(file)), file)
      r <- plot.rx_fit(fit, genotype = genotype, surface = nm, spaTrend = spaTrend, grid = grid,
                       file = f, width = width, height = height, res = res)
      r$file <- f; r })
    return(invisible(stats::setNames(out, surface_names)))
  }
  if (!is.null(file)) {
    switch(ext,
           png  = grDevices::png(file, width = width, height = height, units = "in", res = res),
           jpg  = , jpeg = grDevices::jpeg(file, width = width, height = height, units = "in", res = res),
           tiff = , tif = grDevices::tiff(file, width = width, height = height, units = "in", res = res),
           pdf  = grDevices::pdf(file, width = width, height = height),
           svg  = grDevices::svg(file, width = width, height = height))
    # The file is closed on exit; par() is not restored afterwards, since
    # without an open device par() would open a default one (Rplots.pdf).
    on.exit(grDevices::dev.off(), add = TRUE)
  } else {
    op <- graphics::par(no.readonly = TRUE); on.exit(graphics::par(op), add = TRUE)
    if (length(surface_names) > 1L && grDevices::dev.interactive()) {
      ask <- grDevices::devAskNewPage(TRUE); on.exit(grDevices::devAskNewPage(ask), add = TRUE)
    }
  }
  out <- lapply(surface_names, function(nm)
    .rx_draw(fit, fit$spl2d$surfaces[[nm]], genotype, spaTrend, grid))
  invisible(if (length(surface_names) == 1L) out[[1]] else stats::setNames(out, surface_names))
}

# The six panels of a surface: data, fitted values, residuals, trend, genotype
# BLUPs, residuals against fitted values. A surface per level (at =) covers
# only the plots of its level.
.rx_draw <- function(fit, s, genotype, spaTrend, grid) {
  idx <- if (is.null(s$rows)) seq_along(fit$model$y) else s$rows
  obs <- as.numeric(fit$model$y)[idx]; aj <- fitted.rx_fit(fit)[idx]; rs <- obs - aj
  tp <- .rx_trend_at(fit, s, s$x, s$y)
  tg <- rx_spatial_trend(fit, surface = s$basis$prefix, grid = grid)
  vt <- tg$trend
  if (spaTrend == "percentage") vt <- 100 * vt / mean(obs)
  graphics::layout(matrix(c(1:6, 7, 8, 9, 9, 10, 10), 2, byrow = TRUE), widths = rep(c(4, 1), 3))
  # Interpolated between fixed anchors of the palettes YlGnBu (reversed) and
  # Blue-Red 2: hcl.colors(101, "Blue-Red 2") failed with "invalid hcl color"
  # on R 4.6.1 for macOS.
  seqp <- grDevices::colorRampPalette(c("#FCFFDD", "#E3F8D4", "#BCE9C5", "#85D5B7", "#18BDB0",
                                        "#00A0AE", "#007EB3", "#004C90", "#26185F"))(100)
  divp <- grDevices::colorRampPalette(c("#4A6FE3", "#788CE1", "#9DA8E2", "#C0C5E3", "#E2E2E2",
                                        "#E6BCC3", "#E495A5", "#DD6D87", "#D33F6A"))(101)
  zl <- range(c(obs, aj), na.rm = TRUE)
  sym <- function(v) { a <- max(abs(v), na.rm = TRUE); c(-a, a) }
  xl <- s$x_name; yl <- s$y_name
  level_label <- if (!is.null(s$at)) paste0(" : ", s$at$name, " = ", s$at$level) else ""
  .rx_map(s$x, s$y, obs, paste0("Donnees", level_label), seqp, zl, xl, yl)
  .rx_map(s$x, s$y, aj, "Valeurs ajustees", seqp, zl, xl, yl)
  .rx_map(s$x, s$y, rs, "Residus", divp, sym(rs), xl, yl)
  graphics::par(mar = c(3.2, 3.2, 2, 0.4), mgp = c(2, 0.6, 0))
  M <- matrix(vt, length(unique(tg[[1]])))
  zt <- sym(vt)
  graphics::image(unique(tg[[1]]), unique(tg[[2]]), M, col = divp, zlim = zt,
                  main = if (spaTrend == "raw") "Tendance spatiale" else "Tendance spatiale (%)",
                  xlab = xl, ylab = yl, useRaster = TRUE)
  graphics::box()
  graphics::par(mar = c(3.2, 0.3, 2, 2.6))
  v <- seq(zt[1], zt[2], length.out = length(divp))
  graphics::image(1, v, matrix(v, 1), col = divp, axes = FALSE, xlab = "", ylab = "")
  graphics::axis(4, las = 1, cex.axis = 0.8); graphics::box()
  graphics::par(mar = c(3.2, 3.2, 2, 1))
  if (!is.na(genotype) && !is.null(fit$blups[[genotype]])) {
    graphics::hist(as.numeric(fit$blups[[genotype]]), breaks = 20, col = "grey75", border = "white",
                   main = paste0("BLUP : ", genotype), xlab = genotype, ylab = "Effectif")
  } else graphics::plot.new()
  graphics::plot(aj, rs, pch = 16, cex = 0.6, col = grDevices::adjustcolor("grey20", 0.6),
                 main = "Residus contre valeurs ajustees", xlab = "Valeurs ajustees", ylab = "Residus")
  graphics::abline(h = 0, lty = 2, col = "grey50")
  list(plots = stats::setNames(data.frame(s$x, s$y, obs, aj, rs, tp),
                               c(xl, yl, "observed", "fitted", "residual", "trend")),
       trend = tg)
}
