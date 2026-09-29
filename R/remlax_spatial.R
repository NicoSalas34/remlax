# ==============================================================================
# SORTIES SPATIALES A LA MANIERE DE SpATS
# ==============================================================================
#
# Trois objets, repris de SpATS (Rodriguez-Alvarez et al. 2018) :
#   1. le tableau des DIMENSIONS : pour chaque composante, la dimension
#      effective ED (ou degres de liberte effectifs), la dimension du modele,
#      la dimension nominale rang[X, Z_k] - rang[X] et leur rapport ;
#   2. l'HERITABILITE GENERALISEE d'Oakey et al. (2006), ED_g / nominale_g ;
#   3. la SURFACE SPATIALE sur une grille fine, et des cartes : donnees,
#      valeurs ajustees, residus, tendance, BLUP des genotypes.
#
# Dimension effective d'un terme aleatoire u ~ N(0, s2 I_q) :
#   ED = q - tr(PEV) / s2,
# PEV etant la variance d'erreur de prediction des q BLUP. Seule la diagonale
# de PEV intervient, et le moteur la rend avec pev = TRUE. La formule suppose
# K = I : avec une matrice de parente, il faudrait tr(K^-1 PEV), donc la PEV
# complete, et le terme est laisse sans ED.

# Valeurs ajustees X beta + somme des Z u, et residus y - ajustees.
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

residuals.rx_fit <- function(object, ...) {
  if (is.null(object$model$y)) stop("residuals : l'ajustement ne porte pas son modele.", call. = FALSE)
  as.numeric(object$model$y) - fitted.rx_fit(object)
}

# Noms des surfaces demandees : NULL = la premiere, "all" = toutes.
.rx_noms_surfaces <- function(fit, surface) {
  sf <- fit$spl2d$surfaces
  if (!length(sf)) stop("aucune surface spl2d() dans cet ajustement.", call. = FALSE)
  if (is.null(surface)) return(names(sf)[1])
  if (identical(surface, "all")) return(names(sf))
  inc <- setdiff(surface, names(sf))
  if (length(inc))
    stop("surface '", inc[1], "' inconnue ; disponibles : ", paste(names(sf), collapse = ", "),
         call. = FALSE)
  surface
}

.rx_surface <- function(fit, surface) {
  sf <- fit$spl2d$surfaces
  if (!length(sf)) stop("aucune surface spl2d() dans cet ajustement.", call. = FALSE)
  if (is.null(surface)) surface <- names(sf)[1]
  if (!surface %in% names(sf))
    stop("surface '", surface, "' inconnue ; disponibles : ", paste(names(sf), collapse = ", "),
         call. = FALSE)
  sf[[surface]]
}

# Tendance d'une surface en (x, y) : partie nulle + trois parties lissees.
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
  noms <- .rx_noms_surfaces(fit, surface)
  if (length(noms) > 1L) {
    # Plusieurs surfaces : un seul tableau, avec la surface et son niveau.
    tabs <- lapply(noms, function(nm) {
      g <- rx_spatial_trend(fit, surface = nm, grid = grid)
      at <- fit$spl2d$surfaces[[nm]]$at
      names(g)[1:2] <- c("x", "y")
      cbind(surface = nm, level = if (is.null(at)) NA_character_ else at$level, g) })
    out <- do.call(rbind, tabs)
    s1 <- fit$spl2d$surfaces[[noms[1]]]
    same <- all(vapply(noms, function(nm) identical(c(fit$spl2d$surfaces[[nm]]$x_name,
                                                      fit$spl2d$surfaces[[nm]]$y_name),
                                                    c(s1$x_name, s1$y_name)), TRUE))
    if (same) names(out)[3:4] <- c(s1$x_name, s1$y_name)
    rownames(out) <- NULL
    return(out)
  }
  s <- .rx_surface(fit, noms)
  grid <- rep_len(as.integer(grid), 2L)
  gx <- seq(min(s$x), max(s$x), length.out = grid[1])
  gy <- seq(min(s$y), max(s$y), length.out = grid[2])
  g <- expand.grid(x = gx, y = gy)
  g$trend <- .rx_trend_at(fit, s, g$x, g$y)
  names(g)[1:2] <- c(s$x_name, s$y_name)
  attr(g, "surface") <- s$basis$prefix
  g
}

# Tableau des dimensions, a la maniere de summary.SpATS.
rx_dimensions <- function(fit) {
  m <- fit$model
  if (is.null(m)) stop("rx_dimensions : l'ajustement ne porte pas son modele.", call. = FALSE)
  X <- as.matrix(m$X); n <- nrow(X)
  rX <- qr(X, tol = 1e-7)$rank
  lignes <- list()
  # Effets fixes : un terme = son nombre de colonnes independantes.
  asg <- attr(m$X, "assign"); lab <- attr(m$X, "termes")
  if (!is.null(asg) && !is.null(lab)) {
    ua <- sort(unique(asg))
    for (k in seq_along(ua)) {
      d <- qr(X[, asg == ua[k], drop = FALSE], tol = 1e-7)$rank
      lignes[[length(lignes) + 1L]] <- data.frame(component = lab[k], Effective = d, Model = d,
                                                  Nominal = d, Type = "fixed")
    }
  } else {
    lignes[[1]] <- data.frame(component = "fixed", Effective = rX, Model = ncol(X),
                              Nominal = rX, Type = "fixed")
  }
  for (tm in m$terms) {
    S <- fit$sigmas[[tm$name]]; P <- fit$pev[[tm$name]]
    ok <- tm$t == 1L && identical(tm$struct, "iid") && is.null(tm$LK) &&
      (is.null(tm$level) || tm$level %in% c("id", "auto")) && !is.null(S) && !is.null(P)
    Z <- tm$Zl[[1]]
    ed <- if (ok) tm$q - sum(as.numeric(P)) / S[1, 1] else NA_real_
    nom <- if (tm$t == 1L) qr(cbind(X, as.matrix(Z)), tol = 1e-7)$rank - rX else NA_real_
    lignes[[length(lignes) + 1L]] <- data.frame(component = tm$name, Effective = ed,
                                                Model = tm$q * tm$t, Nominal = nom, Type = "random")
  }
  out <- do.call(rbind, lignes)
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

# ---- cartes ------------------------------------------------------------------

# Remplit une matrice sur la grille des coordonnees uniques ; NULL si la grille
# n'est pas reguliere (plus de quatre cases par observation).
.rx_grille <- function(x, y, z) {
  ux <- sort(unique(x)); uy <- sort(unique(y))
  if (length(ux) * length(uy) > 4 * length(z)) return(NULL)
  M <- matrix(NA_real_, length(ux), length(uy))
  ix <- match(x, ux); iy <- match(y, uy)
  s <- tapply(z, list(ix, iy), mean)
  M[as.integer(rownames(s)), as.integer(colnames(s))] <- s
  list(x = ux, y = uy, z = M)
}

.rx_carte <- function(x, y, z, main, pal, zlim, xlab, ylab) {
  graphics::par(mar = c(3.2, 3.2, 2, 0.4), mgp = c(2, 0.6, 0))
  g <- .rx_grille(x, y, z)
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
  noms <- .rx_noms_surfaces(fit, surface)
  if (is.null(genotype)) {
    termes <- vapply(fit$model$terms, `[[`, "", "name")
    autres <- setdiff(termes, unlist(lapply(fit$spl2d$surfaces, function(z) z$basis$terms)))
    genotype <- if (length(autres)) autres[1] else NA_character_
  }
  ext <- if (!is.null(file)) tolower(tools::file_ext(file)) else ""
  if (!is.null(file) && !ext %in% c("png", "jpg", "jpeg", "tiff", "tif", "pdf", "svg"))
    stop("plot : extension '", ext, "' non geree (png, jpeg, tiff, pdf, svg).", call. = FALSE)
  # Plusieurs surfaces dans un format a une page : un fichier par surface,
  # suffixe par le nom de la surface.
  if (length(noms) > 1L && !is.null(file) && ext != "pdf") {
    out <- lapply(noms, function(nm) {
      f <- sub(paste0("\\.", tools::file_ext(file), "$"), paste0("_", nm, ".", tools::file_ext(file)), file)
      r <- plot.rx_fit(fit, genotype = genotype, surface = nm, spaTrend = spaTrend, grid = grid,
                       file = f, width = width, height = height, res = res)
      r$file <- f; r })
    return(invisible(stats::setNames(out, noms)))
  }
  if (!is.null(file)) {
    switch(ext,
           png  = grDevices::png(file, width = width, height = height, units = "in", res = res),
           jpg  = , jpeg = grDevices::jpeg(file, width = width, height = height, units = "in", res = res),
           tiff = , tif = grDevices::tiff(file, width = width, height = height, units = "in", res = res),
           pdf  = grDevices::pdf(file, width = width, height = height),
           svg  = grDevices::svg(file, width = width, height = height))
    # Le fichier est ferme a la sortie. Ne PAS restaurer par() ensuite : sans
    # peripherique ouvert, par() en ouvrirait un par defaut (Rplots.pdf).
    on.exit(grDevices::dev.off(), add = TRUE)
  } else {
    op <- graphics::par(no.readonly = TRUE); on.exit(graphics::par(op), add = TRUE)
    if (length(noms) > 1L && grDevices::dev.interactive()) {
      ask <- grDevices::devAskNewPage(TRUE); on.exit(grDevices::devAskNewPage(ask), add = TRUE)
    }
  }
  out <- lapply(noms, function(nm)
    .rx_dessin(fit, fit$spl2d$surfaces[[nm]], genotype, spaTrend, grid))
  invisible(if (length(noms) == 1L) out[[1]] else stats::setNames(out, noms))
}

# Les six panneaux d'une surface : donnees, valeurs ajustees, residus,
# tendance, BLUP du genotype, residus contre valeurs ajustees.
.rx_dessin <- function(fit, s, genotype, spaTrend, grid) {
  # Une surface par niveau (at =) ne couvre que les parcelles de son niveau.
  idx <- if (is.null(s$rows)) seq_along(fit$model$y) else s$rows
  obs <- as.numeric(fit$model$y)[idx]; aj <- fitted.rx_fit(fit)[idx]; rs <- obs - aj
  tp <- .rx_trend_at(fit, s, s$x, s$y)
  tg <- rx_spatial_trend(fit, surface = s$basis$prefix, grid = grid)
  vt <- tg$trend
  if (spaTrend == "percentage") vt <- 100 * vt / mean(obs)
  graphics::layout(matrix(c(1:6, 7, 8, 9, 9, 10, 10), 2, byrow = TRUE), widths = rep(c(4, 1), 3))
  seqp <- grDevices::hcl.colors(100, "YlGnBu", rev = TRUE)
  divp <- grDevices::hcl.colors(101, "Blue-Red 2")
  zl <- range(c(obs, aj), na.rm = TRUE)
  sym <- function(v) { a <- max(abs(v), na.rm = TRUE); c(-a, a) }
  xl <- s$x_name; yl <- s$y_name
  niveau <- if (!is.null(s$at)) paste0(" : ", s$at$name, " = ", s$at$level) else ""
  .rx_carte(s$x, s$y, obs, paste0("Donnees", niveau), seqp, zl, xl, yl)
  .rx_carte(s$x, s$y, aj, "Valeurs ajustees", seqp, zl, xl, yl)
  .rx_carte(s$x, s$y, rs, "Residus", divp, sym(rs), xl, yl)
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
