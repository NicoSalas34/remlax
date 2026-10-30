# ==============================================================================
# lag.R - spatial lag of the response: the structural model
# ==============================================================================
#
#     y = M y + X b + sum_k Z_k u_k + e,      M[(t,i),(u,j)] = Psi[t,u] W_t[i,j]
#
# y is in long format: each row carries a trait t and a unit i (the plant).
# Psi[t, u] is the effect of the emitting trait u of a neighbour on the
# receiving trait t; W_t (N x N, zero diagonal) is shared by all traits or
# given per receiving trait. With A = I - M invertible, y* = A y follows the
# ordinary model of the package: every term and residual applies to y*, and
# only the objective changes (the Jacobian log|det A| and, for the exact REML
# of y, the mean term; see inst/python/remlax/lag.py and
# vignette("structural")).
#
# This file only describes the lag and checks it; the solver (lag.py)
# receives W as triplets.
# ==============================================================================

# A q x q matrix (receiver in rows, emitter in columns) aligned on the trait
# levels. Dimnames, when present, decide the order: a matrix given in another
# order would otherwise be read transposed or permuted without error.
.rx_qxq <- function(x, tn, what, default = 0) {
  q <- length(tn)
  if (is.null(x)) return(matrix(default, q, q, dimnames = list(tn, tn)))
  if (length(x) == 1L && is.null(dim(x))) return(matrix(x, q, q, dimnames = list(tn, tn)))
  x <- as.matrix(x)
  if (!identical(dim(x), c(q, q)))
    stop("rx_lag: `", what, "` must be a ", q, " x ", q,
         " matrix (receiver in rows, emitter in columns). Got ", nrow(x), " x ", ncol(x), ".",
         call. = FALSE)
  if (!is.null(rownames(x)) && !is.null(colnames(x))) {
    if (!setequal(rownames(x), tn) || !setequal(colnames(x), tn))
      stop("rx_lag: the dimnames of `", what, "` are not the traits (",
           paste(tn, collapse = ", "), ").", call. = FALSE)
    x <- x[tn, tn, drop = FALSE]
  }
  dimnames(x) <- list(tn, tn)
  x
}

# A W: sparse, general, numeric, square, with a zero diagonal (a plant is not
# its own neighbour; an effect on itself cannot be told from the fixed effects
# and the residual).
.rx_W <- function(w, k) {
  if (!inherits(w, "Matrix")) w <- Matrix::Matrix(as.matrix(w), sparse = TRUE)
  w <- methods::as(methods::as(w * 1, "CsparseMatrix"), "generalMatrix")
  if (nrow(w) != ncol(w))
    stop("rx_lag: W", if (!is.null(k)) paste0("[", k, "]"), " is not square (",
         nrow(w), " x ", ncol(w), ").", call. = FALSE)
  if (any(Matrix::diag(w) != 0))
    stop("rx_lag: W", if (!is.null(k)) paste0("[", k, "]"), " has a non-zero diagonal. ",
         "A plant is not its own neighbour. An effect on itself cannot be told apart ",
         "from the fixed effects and the residual.", call. = FALSE)
  if (any(!is.finite(w@x)))
    stop("rx_lag: W contains non-finite values.", call. = FALSE)
  w
}

rx_lag <- function(W, trait, unit, struct = c("full", "diag", "mask"), mask = NULL,
                   init = NULL, value = NULL, emitters = NULL, receivers = NULL,
                   criterion = c("reml_y", "jacobian", "ml")) {
  struct <- match.arg(struct)
  criterion <- match.arg(criterion)
  trait <- factor(trait)
  tn <- levels(trait); q <- length(tn); n <- length(trait)
  if (anyNA(trait)) stop("rx_lag: `trait` contains NAs.", call. = FALSE)
  Wl <- .rx_lag_W(W, tn)
  N <- nrow(Wl[[1]])
  idx <- .rx_lag_units(unit, n, N, Wl)
  key <- (as.integer(trait) - 1L) * N + idx
  if (anyDuplicated(key))
    stop("rx_lag: ", sum(duplicated(key)), " duplicated (trait, unit) pair(s). M is only ",
         "defined for one observation per trait and per unit.", call. = FALSE)
  msk <- .rx_lag_mask(struct, mask, tn, receivers, emitters)
  val <- .rx_qxq(value, tn, "value")
  if (any(!is.finite(val))) stop("rx_lag: `value` must be finite.", call. = FALSE)
  val[msk] <- 0
  ini <- .rx_qxq(init, tn, "init")
  if (any(!is.finite(ini[msk]))) stop("rx_lag: `init` must be finite on the free entries.",
                                      call. = FALSE)
  ini[!msk] <- 0
  # The free parameters are ordered receiver by receiver (row by row), as
  # numpy.nonzero in the solver; R's which() runs by columns, hence t(msk).
  ij <- which(t(msk), arr.ind = TRUE)
  param_names <- if (nrow(ij)) sprintf("Psi[%s<-%s]", tn[ij[, 2]], tn[ij[, 1]]) else character()
  structure(list(W = Wl, trait = trait, traits = tn, unit = idx, N = as.integer(N),
                 q = as.integer(q), struct = struct, mask = msk, value = val, init = ini,
                 criterion = criterion, n_free = as.integer(sum(msk)), param_names = param_names),
            class = "rx_lag")
}

# W: one shared matrix, or a list of one per receiving trait, named by the
# traits (the order of a list is not a convention that can be checked).
.rx_lag_W <- function(W, tn) {
  q <- length(tn)
  Wl <- if (is.list(W) && !inherits(W, "Matrix")) W else list(W)
  if (!length(Wl) %in% c(1L, q))
    stop("rx_lag: ", length(Wl), " W matrix(es) for ", q, " trait(s). Give one common W ",
         "or one per receiving trait.", call. = FALSE)
  if (length(Wl) == q && q > 1L) {
    if (is.null(names(Wl)))
      stop("rx_lag: a list of W per trait must be NAMED by the traits (",
           paste(tn, collapse = ", "), "). The order of the list is not a convention ",
           "that can be checked.", call. = FALSE)
    if (!setequal(names(Wl), tn))
      stop("rx_lag: the names of the W list (", paste(names(Wl), collapse = ", "),
           ") differ from the traits (", paste(tn, collapse = ", "), ").", call. = FALSE)
    Wl <- Wl[tn]
  }
  Wl <- lapply(seq_along(Wl), function(k) .rx_W(Wl[[k]], if (length(Wl) > 1L) names(Wl)[k] else NULL))
  N <- nrow(Wl[[1]])
  if (any(vapply(Wl, nrow, 1L) != N))
    stop("rx_lag: the W matrices do not all have ", N, " rows.", call. = FALSE)
  Wl
}

# Row of W of each observation: a numeric unit is a position 1..N; any other
# unit is a label matched to the row names of W. A factor is not converted to
# its codes, whose order has no reason to be that of the rows of W.
.rx_lag_units <- function(unit, n, N, Wl) {
  if (length(unit) != n)
    stop("rx_lag: `unit` has ", length(unit), " elements for ", n, " rows.", call. = FALSE)
  if (is.numeric(unit)) {
    if (any(!is.finite(unit)) || any(unit != round(unit)) || any(unit < 1) || any(unit > N))
      stop("rx_lag: a numeric `unit` must give integer row indices of W in 1..",
           N, ".", call. = FALSE)
    return(as.integer(unit))
  }
  rn <- rownames(Wl[[1]])
  if (is.null(rn))
    stop("rx_lag: `unit` is not numeric and W has no rownames. Give the row index ",
         "of W of each observation, or name the rows of W.", call. = FALSE)
  idx <- match(as.character(unit), rn)
  if (anyNA(idx))
    stop("rx_lag: ", sum(is.na(idx)), " unit(s) missing from the rownames of W (e.g. ",
         paste(utils::head(unique(as.character(unit)[is.na(idx)]), 3), collapse = ", "),
         ").", call. = FALSE)
  idx
}

# Free entries of Psi: all, the diagonal, or a mask, then restricted to the
# given receivers and emitters.
.rx_lag_mask <- function(struct, mask, tn, receivers, emitters) {
  q <- length(tn)
  msk <- switch(struct,
                full = matrix(TRUE, q, q, dimnames = list(tn, tn)),
                diag = { m <- matrix(FALSE, q, q, dimnames = list(tn, tn)); diag(m) <- TRUE; m },
                mask = {
                  if (is.null(mask)) stop("rx_lag: struct = \"mask\" requires `mask`.", call. = FALSE)
                  m <- .rx_qxq(mask, tn, "mask")
                  storage.mode(m) <- "logical"
                  if (anyNA(m)) stop("rx_lag: `mask` contains NAs.", call. = FALSE)
                  m
                })
  if (!is.null(receivers)) {
    if (!all(receivers %in% tn))
      stop("rx_lag: unknown receiver(s): ", paste(setdiff(receivers, tn), collapse = ", "),
           call. = FALSE)
    msk[!tn %in% receivers, ] <- FALSE
  }
  if (!is.null(emitters)) {
    if (!all(emitters %in% tn))
      stop("rx_lag: unknown emitter(s): ", paste(setdiff(emitters, tn), collapse = ", "),
           call. = FALSE)
    msk[, !tn %in% emitters] <- FALSE
  }
  msk
}

# Border covariates. With non-zero trait means the structural model gives
# E y = (I - M)^-1 X b: without loops, E y_(t,i) = mu_t + sum_u Psi[t,u] mu_u
# (W_t 1)_i, a mean that depends on the number of neighbours of plant i
# (borders, missing plants). The structural criteria account for it, but a
# model without lag (DGE/IGE, VP+) does not: it needs W 1 as a covariate, per
# trait (order > 1 adds W^2 1, ...). The same covariate added to the
# structural model also absorbs a border effect that is not an interaction
# (light at the edge), so that Psi is no longer read in the border pattern of
# the mean.
rx_border <- function(W, trait, unit, order = 1L, scale = TRUE) {
  order <- as.integer(order)
  if (length(order) != 1L || is.na(order) || order < 1L)
    stop("rx_border: `order` must be an integer >= 1.", call. = FALSE)
  trait <- factor(trait)
  tn <- levels(trait); q <- length(tn)
  # rx_lag() makes every check (W, matching of units, duplicates).
  L <- rx_lag(W, trait, unit, struct = "mask", mask = matrix(FALSE, q, q, dimnames = list(tn, tn)))
  N <- L$N
  out <- matrix(0, length(trait), q * order)
  nm <- character(q * order)
  for (k in seq_len(q)) {
    Wk <- L$W[[if (length(L$W) == 1L) 1L else k]]
    s <- if (scale) mean(Matrix::rowSums(abs(Wk))) else 1
    if (!(s > 0)) s <- 1
    rows <- which(as.integer(L$trait) == k)
    v <- rep(1, N)
    for (o in seq_len(order)) {
      v <- as.numeric(Wk %*% v) / s
      col <- (o - 1L) * q + k
      out[rows, col] <- v[L$unit[rows]]
      nm[col] <- sprintf("border%d:%s", o, tn[k])
    }
  }
  colnames(out) <- nm
  out
}

print.rx_lag <- function(x, ...) {
  cat(sprintf("Spatial lag: %d trait(s), %d unit(s), %s, %d free entry(ies) of Psi, criterion %s\n",
              x$q, x$N, if (length(x$W) == 1L) "one common W" else "one W per receiving trait",
              x$n_free, x$criterion))
  m <- ifelse(x$mask, "free", format(x$value, digits = 3))
  dimnames(m) <- list(paste0(x$traits, " <-"), x$traits)
  print(noquote(m))
  invisible(x)
}

# Warns for TB+ with IEE* and a reciprocal pair of Psi free. With a plant term
# carrying an incidence W placed on one trait (IEE*), the plant level of the
# structural model is VP+ in other coordinates: with Psi[t,u] and Psi[u,t] both
# free it inherits the q(q-1)/2 weak directions of VP+, so who transmits to
# whom is identified only at second order and the free IEE* absorbs it. It
# warns and does not refuse: without IEE*, TB stays identified with a full Psi
# through the genetic level.
.rx_warn_tb_iee <- function(terms, lag) {
  m <- lag$mask
  recip <- which(m & t(m) & row(m) != col(m), arr.ind = TRUE)
  if (!nrow(recip)) return(invisible(FALSE))
  carries_W <- function(Zc) {
    if (ncol(Zc) != lag$N || nrow(Zc) != length(lag$unit)) return(FALSE)
    nz <- which(Matrix::rowSums(abs(Zc)) > 0)
    if (!length(nz)) return(FALSE)
    tr <- unique(as.integer(lag$trait)[nz])
    if (length(tr) != 1L) return(FALSE)
    rows <- which(as.integer(lag$trait) == tr)
    Wt <- lag$W[[if (length(lag$W) == 1L) 1L else tr]]
    isTRUE(max(abs(Zc[rows, , drop = FALSE] - Wt[lag$unit[rows], , drop = FALSE])) < 1e-12)
  }
  iee <- vapply(terms, function(tm) any(vapply(tm$Zl, carries_W, logical(1))), logical(1))
  if (!any(iee)) return(invisible(FALSE))
  pairs <- unique(apply(recip, 1, function(r) paste(sort(lag$traits[r]), collapse = "/")))
  term_names <- vapply(terms, function(tm) tm$name %||% "?", "")
  warning("rx_model: term(s) ", paste(term_names[iee], collapse = ", "),
          " with an incidence W (IEE*) and Psi free in both directions for ",
          paste(pairs, collapse = ", "), ". The direction of transmission is then only weakly ",
          "identified (weak directions of VP+). Leave at most one of Psi[t,u] and ",
          "Psi[u,t] free (rx_lag(struct = \"mask\") or receivers/emitters).", call. = FALSE)
  invisible(TRUE)
}
