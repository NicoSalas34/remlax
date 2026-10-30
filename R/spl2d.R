# ==============================================================================
# spl2d.R - two-dimensional P-spline surfaces (PS-ANOVA of SpATS)
# ==============================================================================
# A smooth surface in two dimensions is a random effect with a known
# incidence: nothing is added to the solver, only matrices are built, as
# sommer (spl2Dc) and asreml do.
#
# Construction (Eilers and Marx; PS-ANOVA decomposition of Rodriguez-Alvarez
# et al. 2018, construct.2d.pspline):
#   1. B-spline bases B_x (n x c_x) and B_y (n x c_y), degree 3, `nseg`
#      segments;
#   2. second-order difference penalty D'D, of rank c - 2, with null space
#      {1, x};
#   3. reparametrisation into a null part (constant and slope, which go to the
#      fixed effects) and a penalised part (random, with Z = B U diag(d^-1/2)
#      so that its covariance is sigma^2 I).
#
# Five random blocks, those of the PSANOVA of SpATS, each with its own
# variance:
#   fx      f(x)          Z_x
#   fy      f(y)          Z_y
#   fx_y    f(x):y        Z_x    * linear y
#   x_fy    x:f(y)        linear x * Z_y
#   fx_fy   f(x):f(y)     Z_x (x) Z_y, on nested bases when nest.div > 1
# SpATS writes G_k = s2_k diag(1/d) on Z = B U; scaling Z by d^-1/2 here gives
# an iid term with the same variance s2_k. For f(x):f(y), d is the Kronecker
# sum 1 (x) d_x + d_y (x) 1, as in SpATS. The constant and linear columns are
# those of SpATS (MM.basis, decom = 4): 1 / sqrt(c) and B a, where a is the
# centred and normed coefficient index. Their scale does not change the fit
# but sets the unit of s2 for f(x), f(y), f(x):y and x:f(y), which makes the
# variances equal to those of SpATS.
#
#   sp <- rx_spl2d(d$col, d$row, nseg = c(10, 10))
#   mod <- rx_model(y, cbind(X, sp$X), c(list(...), sp$terms))
# The columns sp$X are the null part (x, y, x*y). They must go to the fixed
# effects; otherwise the surface is penalised down to its linear component and
# the smoothing is biased.
# ==============================================================================

# spl2d(x, y, nseg =, deg =, pord =, name =, at =, at.levels =) terms of a
# random formula. Returns the remaining formula (NULL when nothing remains),
# the random terms of the surfaces, their null part X, the term of origin of
# each column of X, and a description of each surface.
.rx_extract_spl2d <- function(random, data, trait = NULL) {
  none <- list(random = random, terms = list(), X = NULL, groups = character())
  if (is.null(random)) return(none)
  env <- environment(random); if (is.null(env)) env <- parent.frame()
  labels <- attr(terms(random, keep.order = TRUE), "term.labels")
  is_spl <- vapply(labels, function(txt) {
    e <- str2lang(txt); is.call(e) && as.character(e[[1]]) %in% c("spl2d", "spl2D")
  }, logical(1))
  if (!any(is_spl)) return(none)
  if (!is.null(trait) && nlevels(factor(trait)) > 1L)
    stop("spl2d(): the formula does not support a multi-trait surface. ",
         "Build the basis with rx_spl2d() and go through rx_model().", call. = FALSE)
  terms_out <- list(); Xs <- list(); groups <- character(); used_names <- character()
  surfaces <- list()
  for (txt in labels[is_spl]) {
    sp <- .rx_spl2d_call(txt, data, env, used_names)
    used_names <- c(used_names, sp$prefix)
    for (lv in sp$levels) {
      out <- .rx_spl2d_level(lv, sp, data, txt)
      terms_out <- c(terms_out, out$terms); Xs[[length(Xs) + 1L]] <- out$X
      groups <- c(groups, rep(txt, ncol(out$X)))
      surfaces[[lv$prefix]] <- out$surface
    }
  }
  rest <- labels[!is_spl]
  list(random = if (length(rest)) stats::as.formula(paste("~", paste(rest, collapse = " + ")),
                                                     env = env) else NULL,
       terms = terms_out, X = do.call(cbind, Xs), groups = groups, surfaces = surfaces)
}

# Arguments of one spl2d() call: coordinates, basis settings, name, and the
# levels of `at` (one surface per level, as sommer's spl2Dc(): each level has
# its own basis over the range of its own coordinates, its own five variances
# and null part, and is zero on the rows of the other levels).
.rx_spl2d_call <- function(txt, data, env, used_names) {
  e <- str2lang(txt); args <- as.list(e)[-1]
  arg_names <- names(args); if (is.null(arg_names)) arg_names <- rep("", length(args))
  pos <- args[arg_names == ""]; opt <- args[arg_names != ""]
  xa <- opt$x; ya <- opt$y; ip <- 1L
  if (is.null(xa)) { if (length(pos) < ip) stop("spl2d(): the x coordinate is missing.", call. = FALSE)
                     xa <- pos[[ip]]; ip <- ip + 1L }
  if (is.null(ya)) { if (length(pos) < ip) stop("spl2d(): the y coordinate is missing.", call. = FALSE)
                     ya <- pos[[ip]] }
  ev <- function(a, def) if (is.null(a)) def else eval(a, env)
  sp <- list(nseg = as.integer(ev(opt$nseg, c(10L, 10L))))
  sp$prefix <- as.character(ev(opt$name, if (!length(used_names)) "spl" else
                                           paste0("spl", length(used_names) + 1L)))
  if (sp$prefix %in% used_names)
    stop("spl2d(): two surfaces have the name '", sp$prefix, "'.", call. = FALSE)
  sp$x <- .rx_spl2d_coordinate(xa, "x", data, env)
  sp$y <- .rx_spl2d_coordinate(ya, "y", data, env)
  sp$deg <- as.integer(ev(opt$deg, c(3L, 3L))); sp$pord <- as.integer(ev(opt$pord, 2L))
  sp$nest <- as.integer(ev(opt$nest.div, c(1L, 1L)))
  sp$x_name <- paste(deparse(xa), collapse = ""); sp$y_name <- paste(deparse(ya), collapse = "")
  at_expr <- if (!is.null(opt$at)) opt$at else opt$at.var
  if (is.null(at_expr)) {
    sp$levels <- list(list(rows = seq_len(nrow(data)), prefix = sp$prefix, level = NULL))
    sp$at_name <- NULL
    return(sp)
  }
  sp$at_name <- paste(deparse(at_expr), collapse = "")
  av <- eval(at_expr, data, env)
  if (length(av) != nrow(data) || anyNA(av))
    stop("spl2d(): the variable at = ", sp$at_name, " must be complete, of length ",
         nrow(data), ".", call. = FALSE)
  av <- factor(av)
  lv <- as.character(ev(opt$at.levels, levels(av)))
  outside <- setdiff(lv, levels(av))
  if (length(outside))
    stop("spl2d(): at.levels has levels missing from ", sp$at_name, ": ",
         paste(outside, collapse = ", "), ".", call. = FALSE)
  sp$levels <- lapply(lv, function(l) list(rows = which(av == l), level = l,
                                           prefix = paste0(sp$prefix, "_", gsub("[^A-Za-z0-9.]", "_", l))))
  prefixes <- vapply(sp$levels, `[[`, "", "prefix")
  if (anyDuplicated(prefixes))
    stop("spl2d(): two levels of ", sp$at_name, " give the same term name ('",
         prefixes[duplicated(prefixes)][1], "').", call. = FALSE)
  sp
}

# A coordinate of spl2d(): numeric, complete, one value per row. A factor is
# accepted when its levels are numbers.
.rx_spl2d_coordinate <- function(a, which, data, env) {
  v <- eval(a, data, env)
  if (is.factor(v)) {
    v2 <- suppressWarnings(as.numeric(as.character(v)))
    if (anyNA(v2)) stop("spl2d(): the coordinate ", which, " is a factor with non-numeric ",
                        "levels. Give a numeric coordinate.", call. = FALSE)
    v <- v2
  }
  if (!is.numeric(v) || length(v) != nrow(data) || anyNA(v))
    stop("spl2d(): the coordinate ", which, " must be numeric, complete, of length ",
         nrow(data), ".", call. = FALSE)
  v
}

# The surface of one level: its basis on the rows of the level, padded with
# zeros on the other rows when the level does not cover the data.
.rx_spl2d_level <- function(lv, sp, data, txt) {
  r <- lv$rows
  if (length(unique(sp$x[r])) < 2L || length(unique(sp$y[r])) < 2L)
    stop("spl2d(): ", if (is.null(lv$level)) "the surface" else
         paste0("the level '", lv$level, "' of ", sp$at_name),
         " has only one value of x or y. A surface needs two dimensions.", call. = FALSE)
  s <- rx_spl2d(sp$x[r], sp$y[r], nseg = sp$nseg, deg = sp$deg, pord = sp$pord,
                nest.div = sp$nest, prefix = lv$prefix)
  full_height <- function(M) {
    P <- matrix(0, nrow(data), ncol(M), dimnames = list(NULL, colnames(M)))
    P[r, ] <- as.matrix(M); P
  }
  if (length(r) < nrow(data))
    s$terms <- lapply(s$terms, function(tm) {
      rx_term(tm$name, list(full_height(tm$Zl[[1]])), struct = "iid") })
  list(terms = s$terms, X = full_height(s$X),
       surface = list(basis = s$basis, x = sp$x[r], y = sp$y[r], rows = r, label = txt,
                      x_name = sp$x_name, y_name = sp$y_name,
                      at = if (!is.null(lv$level)) list(name = sp$at_name, level = lv$level)))
}

# --- bases ---------------------------------------------------------------------

# B-spline basis of degree `deg` on `nseg` equal segments of [xl, xr].
.rx_bbase <- function(x, nseg = 6L, deg = 3L, xl = min(x), xr = max(x)) {
  dx <- (xr - xl) / nseg
  kn <- seq(xl - deg * dx, xr + deg * dx, by = dx)
  splines::spline.des(kn, x, deg + 1L, 0 * x, outer.ok = TRUE)$design
}

# Decomposition of the penalty of a 1D basis: penalised eigenvectors and their
# eigenvalues, null space, and the linear direction a of SpATS. They are kept in
# the basis to re-evaluate the surface at new points with exactly the same
# reparametrisation.
.rx_pen_decomp <- function(c_, pord) {
  D <- diff(diag(c_), differences = pord); e <- eigen(crossprod(D), symmetric = TRUE)
  keep <- seq_len(c_ - pord)
  a <- seq_len(c_) - (c_ + 1) / 2
  list(U = e$vectors[, keep, drop = FALSE], d = e$values[keep],
       N = e$vectors[, -keep, drop = FALSE], lin = a / sqrt(sum(a^2)))
}

# Row-wise Khatri-Rao product.
.rx_rowk <- function(A, B)
  do.call(cbind, lapply(seq_len(ncol(B)), function(j) A * B[, j]))

# One axis of a basis at the points v (nested basis for f(x):f(y) if asked).
.rx_spl2d_axis <- function(v, ax, deg, nested = FALSE) {
  ns <- if (nested) ax$nseg / ax$nest else ax$nseg
  pen <- if (nested) ax$pen_n else ax$pen
  B <- .rx_bbase(as.numeric(v), ns, deg, ax$xl, ax$xr)
  list(B = B, Z = B %*% pen$U, d = pen$d, N = B %*% pen$N, lin = as.numeric(B %*% pen$lin),
       one = 1 / sqrt(ncol(B)))
}

# Suffixes of the five terms.
.rx_spl2d_suffixes <- c("fx", "fy", "fx_y", "x_fy", "fx_fy")

# Evaluates a spl2d basis at (x, y): the null part (centred, retained columns)
# and the five penalised incidences, already scaled. Outside the fitted domain
# the B-splines are extrapolated (outer.ok), so a map on a grid should stay in
# the rectangle of the data.
.rx_spl2d_eval <- function(basis, x, y) {
  deg <- rep_len(basis$deg, 2L)
  ex <- .rx_spl2d_axis(x, basis$x, deg[1]); ey <- .rx_spl2d_axis(y, basis$y, deg[2])
  exn <- if (basis$x$nest > 1) .rx_spl2d_axis(x, basis$x, deg[1], TRUE) else ex
  eyn <- if (basis$y$nest > 1) .rx_spl2d_axis(y, basis$y, deg[2], TRUE) else ey
  ech <- function(Z, d) sweep(Z, 2L, sqrt(d), "/")
  Zx <- ech(ex$Z, ex$d); Zy <- ech(ey$Z, ey$d)
  Zxy <- ech(.rx_rowk(exn$Z, eyn$Z), as.vector(outer(exn$d, eyn$d, "+")))
  Xn <- .rx_rowk(ex$N, ey$N)
  Xn <- sweep(Xn, 2L, basis$centre)[, basis$keep, drop = FALSE]
  # f(x) carries the normed constant column of y, 1 / sqrt(c_y), as in SpATS:
  # without effect on the fit, this factor sets the unit of s2.
  Z <- list(Zx * ey$one, Zy * ex$one, Zx * ey$lin, ex$lin * Zy, Zxy)
  names(Z) <- .rx_spl2d_suffixes
  list(X = Xn, Z = Z)
}

# 2D P-spline basis (PS-ANOVA of SpATS), ready for rx_model(): the null part X
# (fixed), the five random rx_term objects, and the basis needed to evaluate the
# surface again (.rx_spl2d_eval()).
rx_spl2d <- function(x, y, nseg = c(10L, 10L), deg = c(3L, 3L), pord = 2L,
                     nest.div = c(1L, 1L), prefix = "spl") {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (length(y) != n) stop("rx_spl2d: x and y have different lengths.")
  if (any(pord != 2))
    stop("rx_spl2d: only the penalty of order 2 is supported (as in PSANOVA of SpATS).",
         call. = FALSE)
  ns <- rep_len(as.integer(round(nseg)), 2L); deg <- rep_len(as.integer(deg), 2L)
  nd <- rep_len(as.integer(nest.div), 2L)
  if (any(ns %% nd != 0))
    stop("rx_spl2d: nseg must be a multiple of nest.div.", call. = FALSE)
  basis <- list(deg = deg, pord = 2L, prefix = prefix,
                x = list(xl = min(x), xr = max(x), nseg = ns[1], nest = nd[1]),
                y = list(xl = min(y), xr = max(y), nseg = ns[2], nest = nd[2]))
  basis$x$pen <- .rx_pen_decomp(ns[1] + deg[1], 2L)
  basis$y$pen <- .rx_pen_decomp(ns[2] + deg[2], 2L)
  if (nd[1] > 1) basis$x$pen_n <- .rx_pen_decomp(ns[1] / nd[1] + deg[1], 2L)
  if (nd[2] > 1) basis$y$pen_n <- .rx_pen_decomp(ns[2] / nd[2] + deg[2], 2L)
  # The null part contains the constant direction, so cbind(1, Xn) would be
  # rank deficient and refused by rx_model(). Xn is centred, then a
  # rank-revealing QR keeps an independent basis: X can stand next to an
  # intercept and spans the fixed part of SpATS, x, y and x*y.
  basis$centre <- 0; basis$keep <- TRUE
  brut <- .rx_spl2d_eval(basis, x, y)
  basis$centre <- colMeans(brut$X)
  Xc <- sweep(brut$X, 2L, basis$centre)
  qrX <- qr(Xc, tol = 1e-9)
  basis$keep <- sort(qrX$pivot[seq_len(qrX$rank)])
  Xn <- Xc[, basis$keep, drop = FALSE]
  colnames(Xn) <- paste0(prefix, "_lin", seq_len(ncol(Xn)))
  basis$columns <- colnames(Xn)
  basis$terms <- paste0(prefix, "_", .rx_spl2d_suffixes)
  list(X = Xn,
       terms = lapply(seq_along(brut$Z), function(k)
         rx_term(basis$terms[k], list(brut$Z[[k]]), struct = "iid")),
       basis = basis)
}
