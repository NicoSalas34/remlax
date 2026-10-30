# ==============================================================================
# design.R - neighbourhood incidences and exposure functionals
# ==============================================================================
#   nb <- rx_neighbourhood(coord = d[, c("row", "col")], group = d$species,
#                          block = d$plot, level = d$genotype, id = d$id,
#                          rank = list("A<-A" = 5, "A<-B" = 7), reach = 0,
#                          dilution = 0, spacing = c(5, 5))
#   ex <- rx_exposure(Z_direct, nb$level[["A<-A"]][rows, ], K = K_A)
#
# These functions know nothing of species, traits or indirect genetic effects.
# They know units placed on a grid, groups of units, levels at which the
# weights are summed, and pairs (receiving group, emitting group). The caller
# says that a group is a species and a level a genotype.
#
# Numerical conventions, fixed here and checked by the tests:
#   - the window is taken in grid indices (|d_row| <= r and |d_col| <= r, the
#     unit itself excluded); the weight uses the physical distance
#     delta = sqrt((s1 d_row)^2 + (s2 d_col)^2);
#   - kernel "power": w = delta^(-lambda), so w = 1 everywhere at lambda = 0;
#   - the dilution divides each row by n_i^d, where n_i is the number of
#     neighbours of the unit in the emitting group, not the sum of the weights;
#   - dilution comes before the L2 normalisation, which would otherwise undo
#     it;
#   - level names are returned as they are.
# ==============================================================================

# One parameter per pair: a scalar (every pair), a named G x G matrix (rows:
# receiving group, columns: emitting group), or a vector or list named by pair
# "receiver<-emitter". Returns a numeric vector named by pair, in the order of
# `pairs`. A pair missing from a named form is an error: guessing it would give
# a plausible and wrong design.
.rx_per_pair <- function(x, pairs, groups, what) {
  if (is.matrix(x)) {
    if (is.null(dimnames(x)) || !all(groups %in% rownames(x)) || !all(groups %in% colnames(x)))
      stop("rx_neighbourhood: `", what, "` is a matrix without the dimnames of the groups (",
           paste(groups, collapse = ", "), ").", call. = FALSE)
    a <- sub("<-.*$", "", pairs); b <- sub("^.*<-", "", pairs)
    out <- as.numeric(x[cbind(match(a, rownames(x)), match(b, colnames(x)))])
    return(stats::setNames(out, pairs))
  }
  if (is.list(x) || (length(x) > 1L || !is.null(names(x)))) {
    x <- unlist(x)
    if (is.null(names(x)))
      stop("rx_neighbourhood: `", what, "` has several values but no pair names.",
           call. = FALSE)
    missing_pairs <- setdiff(pairs, names(x))
    if (length(missing_pairs))
      stop("rx_neighbourhood: `", what, "` gives nothing for the pair(s) ",
           paste(missing_pairs, collapse = ", "), ".", call. = FALSE)
    return(stats::setNames(as.numeric(x[pairs]), pairs))
  }
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x))
    stop("rx_neighbourhood: `", what, "` must be a finite scalar, a named matrix ",
         "or a list named by pair.", call. = FALSE)
  stats::setNames(rep(as.numeric(x), length(pairs)), pairs)
}

# Triplets (i, j, x) to a sparse or dense matrix, with dimnames.
.rx_assemble_sparse <- function(i, j, x, dims, dn, sparse) {
  M <- Matrix::sparseMatrix(i = i, j = j, x = x, dims = dims, dimnames = dn)
  if (sparse) M else as.matrix(M)
}

# Divides each row by its L2 norm (rows of zeros are left as they are).
.rx_l2_normalise <- function(M, sparse, dn) {
  nr <- sqrt(Matrix::rowSums(M^2)); nr[nr == 0] <- 1
  M <- if (sparse) Matrix::Diagonal(x = 1 / nr) %*% M else M / nr
  if (sparse) { M <- methods::as(M, "CsparseMatrix"); dimnames(M) <- dn }
  M
}

rx_neighbourhood <- function(coord, group, block = NULL, level = NULL, id = NULL,
                             rank, reach = 0, dilution = 0,
                             kernel = c("power", "exponential", "none"),
                             window = c("chebyshev", "euclidean"),
                             spacing = c(1, 1), normalise = FALSE,
                             pairs = NULL, output = c("level", "unit", "both"),
                             sparse = TRUE) {
  kernel <- match.arg(kernel); window <- match.arg(window); output <- match.arg(output)
  u <- .rx_nb_units(coord, group, block, id, level, spacing, output)
  pairs <- .rx_nb_pairs(pairs, u$groups)
  r_p <- .rx_per_pair(rank, pairs, u$groups, "rank")
  l_p <- .rx_per_pair(reach, pairs, u$groups, "reach")
  d_p <- .rx_per_pair(dilution, pairs, u$groups, "dilution")
  if (any(r_p < 0) || any(r_p != round(r_p)))
    stop("rx_neighbourhood: `rank` must be an integer >= 0 for each pair.", call. = FALSE)
  if (any(l_p < 0) || any(d_p < 0))
    stop("rx_neighbourhood: `reach` and `dilution` must be >= 0.", call. = FALSE)
  # Levels of each emitting group, in the order sort(unique(level)).
  u$levels_of <- if (output != "unit")
    lapply(stats::setNames(u$groups, u$groups), function(g) sort(unique(u$level[u$group == g]))) else NULL
  u$positions <- lapply(stats::setNames(u$groups, u$groups), function(g) which(u$group == g))

  out_level <- list(); out_unit <- list(); n_neighbours <- list()
  for (pr in pairs) {
    p <- .rx_nb_pair(u, pr, r_p[[pr]], l_p[[pr]], d_p[[pr]], kernel, window, output, normalise, sparse)
    if (output != "unit") out_level[[pr]] <- p$level
    if (output != "level") out_unit[[pr]] <- p$unit
    n_neighbours[[pr]] <- p$n_neighbours
  }
  params <- data.frame(pair = pairs, receiver = sub("<-.*$", "", pairs),
                       emitter = sub("^.*<-", "", pairs),
                       rank = as.numeric(r_p), reach = as.numeric(l_p),
                       dilution = as.numeric(d_p), kernel = kernel, window = window,
                       spacing_row = u$spacing[1L], spacing_col = u$spacing[2L],
                       normalise = normalise, stringsAsFactors = FALSE, row.names = NULL)
  structure(list(level = if (output != "unit") out_level else NULL,
                 unit = if (output != "level") out_unit else NULL,
                 n_neighbours = n_neighbours, params = params),
            class = "rx_neighbourhood")
}

# Checked units: integer grid positions, groups, blocks, identifiers, levels
# (needed for the output by level) and the physical spacing.
.rx_nb_units <- function(coord, group, block, id, level, spacing, output) {
  coord <- as.matrix(coord)
  if (ncol(coord) != 2L) stop("rx_neighbourhood: `coord` must have two columns (row, column).",
                              call. = FALSE)
  n <- nrow(coord)
  storage.mode(coord) <- "double"
  if (any(!is.finite(coord))) stop("rx_neighbourhood: `coord` has non-finite values.", call. = FALSE)
  if (any(coord != round(coord)))
    stop("rx_neighbourhood: `coord` must hold INTEGER positions on the grid. ",
         "The physical positions are coord * spacing.", call. = FALSE)
  if (length(group) != n) stop("rx_neighbourhood: `group` does not have ", n, " elements.", call. = FALSE)
  group <- factor(group)
  if (anyNA(group)) stop("rx_neighbourhood: `group` has NAs.", call. = FALSE)
  block <- if (is.null(block)) factor(rep(1L, n)) else factor(block)
  if (length(block) != n) stop("rx_neighbourhood: `block` does not have ", n, " elements.", call. = FALSE)
  id <- if (is.null(id)) as.character(seq_len(n)) else as.character(id)
  if (length(id) != n) stop("rx_neighbourhood: `id` does not have ", n, " elements.", call. = FALSE)
  if (anyDuplicated(id)) stop("rx_neighbourhood: `id` has duplicates.", call. = FALSE)
  if (output != "unit") {
    if (is.null(level)) stop("rx_neighbourhood: `level` is required for the output by level.",
                             call. = FALSE)
    level <- as.character(level)
    if (length(level) != n || anyNA(level))
      stop("rx_neighbourhood: `level` must have ", n, " elements without NA.", call. = FALSE)
  }
  spacing <- as.numeric(spacing)
  if (length(spacing) == 1L) spacing <- rep(spacing, 2L)
  if (length(spacing) != 2L || any(!is.finite(spacing)) || any(spacing <= 0))
    stop("rx_neighbourhood: `spacing` must be two strictly positive steps.", call. = FALSE)
  list(coord = coord, group = group, groups = levels(group), block = block, id = id,
       level = level, spacing = spacing, delta0 = min(spacing))
}

# Pairs "receiver<-emitter": all of them by default, else the given ones.
.rx_nb_pairs <- function(pairs, groups) {
  all_pairs <- as.vector(t(outer(groups, groups, function(a, b) paste0(a, "<-", b))))
  if (is.null(pairs)) return(all_pairs)
  unknown <- setdiff(pairs, all_pairs)
  if (length(unknown))
    stop("rx_neighbourhood: unknown pair(s): ", paste(unknown, collapse = ", "),
         ". Expected form \"receiver<-emitter\" with the groups ",
         paste(groups, collapse = ", "), ".", call. = FALSE)
  pairs
}

# One pair: within each block, the kernel weights between the receiving and
# the emitting units, divided by n_i^d, accumulated as triplets in local
# indices (positions within the groups); then the incidence by level (weights
# summed per emitting level) and by unit, optionally L2-normalised, and the
# neighbour counts n_i.
.rx_nb_pair <- function(u, pr, r, lam, dd, kernel, window, output, normalise, sparse) {
  a <- sub("<-.*$", "", pr); b <- sub("^.*<-", "", pr)
  ia <- u$positions[[a]]; ib <- u$positions[[b]]
  na_ <- length(ia); nb_ <- length(ib)
  coord <- u$coord; block <- u$block
  ui <- integer(0); uj <- integer(0); ux <- numeric(0)
  li <- integer(0); lj <- integer(0); lx <- numeric(0)
  nn <- integer(na_)
  if (r > 0 && na_ && nb_) for (bl in levels(block)) {
    sa <- which(block[ia] == bl); sb <- which(block[ib] == bl)
    if (!length(sa) || !length(sb)) next
    ra <- coord[ia[sa], 1L]; ca <- coord[ia[sa], 2L]
    rb <- coord[ib[sb], 1L]; cb <- coord[ib[sb], 2L]
    dr <- abs(outer(ra, rb, "-")); dc <- abs(outer(ca, cb, "-"))
    # Physical distance, with the arithmetic of dist() on (s1 row, s2 column).
    delta <- sqrt(outer(u$spacing[1L] * ra, u$spacing[1L] * rb, "-")^2 +
                    outer(u$spacing[2L] * ca, u$spacing[2L] * cb, "-")^2)
    m <- if (window == "chebyshev") (dr <= r) & (dc <= r) & (delta > 0)
         else (delta <= r * u$delta0) & (delta > 0)
    W <- matrix(0, length(sa), length(sb))
    if (kernel == "power")            W[m] <- delta[m]^(-lam)
    else if (kernel == "exponential") W[m] <- exp(-lam * (delta[m] - u$delta0))
    else                              W[m] <- 1
    W[is.infinite(W)] <- 0
    ni <- rowSums(m)
    nn[sa] <- ni
    div <- ni; div[div == 0] <- 1
    div <- if (dd == 0) rep(1, length(sa)) else div^dd
    if (output != "level") {
      Wd <- W / div
      nz <- which(Wd != 0, arr.ind = TRUE)
      ui <- c(ui, sa[nz[, 1L]]); uj <- c(uj, sb[nz[, 2L]]); ux <- c(ux, Wd[nz])
    }
    if (output != "unit") {
      lv <- u$levels_of[[b]]
      Gb <- matrix(0, length(sb), length(lv))
      Gb[cbind(seq_along(sb), match(u$level[ib[sb]], lv))] <- 1
      M <- (W %*% Gb) / div
      nz <- which(M != 0, arr.ind = TRUE)
      li <- c(li, sa[nz[, 1L]]); lj <- c(lj, nz[, 2L]); lx <- c(lx, M[nz])
    }
  }
  out <- list(n_neighbours = stats::setNames(nn, u$id[ia]))
  if (output != "unit") {
    lv <- u$levels_of[[b]]
    dn <- list(u$id[ia], lv)
    M <- .rx_assemble_sparse(li, lj, lx, c(na_, length(lv)), dn, sparse)
    out$level <- if (normalise) .rx_l2_normalise(M, sparse, dn) else M
  }
  if (output != "level") {
    dn <- list(u$id[ia], u$id[ib])
    U <- .rx_assemble_sparse(ui, uj, ux, c(na_, nb_), dn, sparse)
    out$unit <- if (normalise) .rx_l2_normalise(U, sparse, dn) else U
  }
  out
}

print.rx_neighbourhood <- function(x, ...) {
  p <- x$params
  cat(sprintf("Neighbourhood: %d pair(s), kernel %s, window %s, spacing (%g, %g)%s\n",
              nrow(p), p$kernel[1], p$window[1], p$spacing_row[1], p$spacing_col[1],
              if (isTRUE(p$normalise[1])) ", L2-normalised" else ""))
  for (i in seq_len(nrow(p))) {
    nn <- x$n_neighbours[[p$pair[i]]]
    dimtxt <- if (!is.null(x$level)) paste0("level ", paste(dim(x$level[[p$pair[i]]]), collapse = "x"))
              else paste0("unit ", paste(dim(x$unit[[p$pair[i]]]), collapse = "x"))
    cat(sprintf("  %-20s rank %2g  reach %g  dilution %g  | %s | neighbours/unit %.2f (max %d)\n",
                p$pair[i], p$rank[i], p$reach[i], p$dilution[i], dimtxt,
                mean(nn), max(nn)))
  }
  invisible(x)
}

# ------------------------------------------------------------------------------
# Exposure functionals: d, k, k_identity, c, S, n_eff
# ------------------------------------------------------------------------------
# The phenotypic variance brought by an indirect component of variance s2_I is
# k s2_I; that of the direct component is d s2_D; the direct-indirect
# covariance enters Var(P) with the coefficient 2 c; the total genetic value of
# a level is sqrt(d) (u_D + S u_I). Every mean is taken over `rows`: in a model
# stacked over several targets the incidences cover every row, and averaging
# over all of them would divide each statistic by the share of rows belonging
# to the target. The model form takes rows = "auto", the rows where the direct
# incidence is not zero.

.rx_kstat <- function(Z, K, rows) {
  rs <- if (is.null(K)) Matrix::rowSums(Z^2) else Matrix::rowSums((Z %*% K) * Z)
  mean(as.numeric(rs)[rows])
}

# Resolves "term:label", "term[i]" or "term" (t = 1) into (term, column).
.rx_column_ref <- function(model, ref, quoi) {
  tm_nom <- sub("[:\\[].*$", "", ref)
  tm <- model[["terms"]][[tm_nom]]
  if (is.null(tm)) stop("rx_exposure: `", quoi, "` names the term '", tm_nom,
                        "', which is not in the model.", call. = FALSE)
  if (grepl("\\[", ref)) {
    j <- as.integer(sub("^.*\\[(\\d+)\\]$", "\\1", ref))
  } else if (grepl(":", ref, fixed = TRUE)) {
    lab <- sub("^[^:]*:", "", ref)
    cn <- tm[["colnames"]]
    if (is.null(cn)) stop("rx_exposure: the term '", tm_nom, "' has no column names. ",
                          "Use '", tm_nom, "[i]'.", call. = FALSE)
    j <- match(lab, cn)
    if (is.na(j)) stop("rx_exposure: column '", lab, "' is missing from the term '", tm_nom,
                       "' (", paste(cn, collapse = ", "), ").", call. = FALSE)
  } else {
    if (tm[["t"]] != 1L) stop("rx_exposure: the term '", tm_nom, "' has t = ", tm[["t"]],
                              " columns. Name one of them.", call. = FALSE)
    j <- 1L
  }
  if (is.na(j) || j < 1L || j > tm[["t"]])
    stop("rx_exposure: column ", j, " outside 1..", tm[["t"]], " for '", tm_nom, "'.",
         call. = FALSE)
  list(term = tm, j = j, Z = tm[["Zl"]][[j]],
       K = if (is.null(tm[["LK"]])) NULL else tm[["LK"]] %*% t(tm[["LK"]]))
}

# Exposure functionals of a pair (direct incidence, indirect incidence), in
# two forms. Matrix form: rx_exposure(Z_direct, Z_indirect, K, rows, K_direct),
# K being the relationship of the emitting group (NULL for the identity).
# Model form: rx_exposure(model, direct = "term:label", indirect =
# "term:label", rows = "auto"), where K is read from the term of the indirect
# incidence and K_direct from that of the direct incidence. Returns d, k,
# k_identity, c, S, n_eff, n_rows and the convention.
rx_exposure <- function(Z_direct = NULL, Z_indirect = NULL, K = NULL, rows = NULL,
                        K_direct = K, direct = NULL, indirect = NULL) {
  meme_groupe <- TRUE
  if (inherits(Z_direct, "rx_fit") || inherits(Z_direct, "rx_model")) {
    model <- if (inherits(Z_direct, "rx_fit")) Z_direct[["model"]] else Z_direct
    if (is.null(model) || is.null(model[["terms"]]))
      stop("rx_exposure: the fit carries no model. Pass the rx_model.",
           call. = FALSE)
    if (is.null(indirect)) stop("rx_exposure: `indirect` is required in the model form.",
                                call. = FALSE)
    ri <- .rx_column_ref(model, indirect, "indirect")
    Z_indirect <- ri$Z
    if (is.null(K)) K <- ri$K
    if (!is.null(direct)) {
      rd <- .rx_column_ref(model, direct, "direct")
      Z_direct <- rd$Z
      if (missing(K_direct) || is.null(K_direct)) K_direct <- rd$K
      meme_groupe <- identical(rd$term[["name"]], ri$term[["name"]])
    } else Z_direct <- NULL
    if (identical(rows, "auto")) {
      if (is.null(Z_direct)) stop("rx_exposure: rows = \"auto\" requires `direct`.", call. = FALSE)
      rows <- which(as.numeric(Matrix::rowSums(Z_direct != 0)) > 0)
    }
  }
  if (is.null(Z_indirect)) stop("rx_exposure: `Z_indirect` is required.", call. = FALSE)
  n <- nrow(Z_indirect)
  if (!is.null(Z_direct) && nrow(Z_direct) != n)
    stop("rx_exposure: Z_direct has ", nrow(Z_direct), " rows, Z_indirect ", n, ".", call. = FALSE)
  if (is.null(rows)) rows <- seq_len(n)
  if (is.logical(rows)) rows <- which(rows)
  rows <- as.integer(rows)
  if (!length(rows)) stop("rx_exposure: no rows to average.", call. = FALSE)
  if (any(rows < 1L | rows > n)) stop("rx_exposure: `rows` outside 1..", n, ".", call. = FALSE)
  if (!is.null(K)) {
    K <- as.matrix(K)
    if (nrow(K) != ncol(Z_indirect) || ncol(K) != ncol(Z_indirect))
      stop("rx_exposure: K is ", nrow(K), "x", ncol(K), " for ", ncol(Z_indirect),
           " columns of Z_indirect.", call. = FALSE)
    # alignment by name when both sides carry names
    if (!is.null(colnames(Z_indirect)) && !is.null(rownames(K))) {
      manque <- setdiff(colnames(Z_indirect), rownames(K))
      if (length(manque)) stop("rx_exposure: ", length(manque), " column(s) of Z_indirect ",
                               "missing from K (e.g. ", paste(utils::head(manque, 3), collapse = ", "),
                               ").", call. = FALSE)
      K <- K[colnames(Z_indirect), colnames(Z_indirect), drop = FALSE]
    }
  }
  if (!is.null(Z_direct) && !is.null(K_direct)) {
    K_direct <- as.matrix(K_direct)
    if (nrow(K_direct) != ncol(Z_direct))
      stop("rx_exposure: K_direct is ", nrow(K_direct), "x", ncol(K_direct), " for ",
           ncol(Z_direct), " columns of Z_direct.", call. = FALSE)
    if (!is.null(colnames(Z_direct)) && !is.null(rownames(K_direct))) {
      manque <- setdiff(colnames(Z_direct), rownames(K_direct))
      if (length(manque)) stop("rx_exposure: ", length(manque), " column(s) of Z_direct ",
                               "missing from K_direct.", call. = FALSE)
      K_direct <- K_direct[colnames(Z_direct), colnames(Z_direct), drop = FALSE]
    }
  }
  k_id <- .rx_kstat(Z_indirect, NULL, rows)
  k <- if (is.null(K)) k_id else .rx_kstat(Z_indirect, K, rows)
  d <- if (is.null(Z_direct)) NA_real_ else .rx_kstat(Z_direct, K_direct, rows)
  # c: mean of (Zd K Zn')_ii; NA without a direct incidence, when the two
  # groups differ (K does not link two groups), or when the dimensions do not
  # allow the product.
  cc <- NA_real_
  if (!is.null(Z_direct) && meme_groupe && ncol(Z_direct) == ncol(Z_indirect)) {
    rs <- if (is.null(K)) Matrix::rowSums(Z_direct * Z_indirect)
          else Matrix::rowSums((Z_direct %*% K) * Z_indirect)
    cc <- mean(as.numeric(rs)[rows])
  }
  S <- mean(as.numeric(Matrix::rowSums(Z_indirect))[rows])
  n_eff <- if (isTRUE(is.finite(d) && k > 0)) S^2 * d / k else NA_real_
  list(d = d, k = k, k_identity = k_id, c = cc, S = S, n_eff = n_eff,
       n_rows = length(rows), convention = if (is.null(K)) "identity" else "K")
}

# ------------------------------------------------------------------------------
# Genomic relationship matrix (VanRaden 1, any ploidy, blending)
# ------------------------------------------------------------------------------
# M holds allele doses (individuals x markers) as a fraction of the ploidy k
# (0, 1/k, ..., 1), or as counts 0..k with coding = "count". blend adds a share
# of the identity: G_b = (1 - blend) G + blend I. Returns a q x q matrix with
# the row names of M and the attribute "denominator".
rx_grm <- function(M, ploidy = 2, blend = 0, coding = c("fraction", "count")) {
  coding <- match.arg(coding)
  M <- as.matrix(M)
  if (anyNA(M)) stop("rx_grm: M has NAs. Impute them first.", call. = FALSE)
  D <- if (coding == "fraction") M * ploidy else M
  if (any(D < 0) || any(D > ploidy))
    stop("rx_grm: some doses fall outside 0..", ploidy, ". Check `coding` and `ploidy`.",
         call. = FALSE)
  p <- colMeans(D) / ploidy
  Z <- sweep(D, 2L, ploidy * p, "-")
  cst <- ploidy * sum(p * (1 - p))
  if (cst <= 0) stop("rx_grm: all markers are monomorphic.", call. = FALSE)
  G <- tcrossprod(Z) / cst
  if (blend > 0) G <- (1 - blend) * G + blend * diag(nrow(G))
  dimnames(G) <- list(rownames(M), rownames(M))
  attr(G, "denominator") <- cst
  G
}
