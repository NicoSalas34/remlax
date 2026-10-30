# ==============================================================================
# bundle_write.R - writing a model to a bundle for the Python solver
# ==============================================================================
# A bundle is a directory of raw binary files (float64 or int32, column-major)
# and text files, described by manifest.json: one entry per array with its
# name, type and shape. R needs no Python dependency to write it, and the
# solver needs no R dependency to read it (bundle.py).
# ==============================================================================

rx_export <- function(model, dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  w <- .rx_bundle_writer(dir)
  .rx_export_fixed(w, model)
  .rx_export_terms(w, model$terms)
  .rx_export_residual(w, model)
  .rx_export_lag(w, model$lag)
  .rx_bundle_finish(w)
  invisible(dir)
}

# --- writer ------------------------------------------------------------------

# A writer holds the directory and the manifest entries written so far.
.rx_bundle_writer <- function(dir) {
  w <- new.env(parent = emptyenv())
  w$dir <- dir
  w$manifest <- list()
  w
}

# Writes one array: dtype "f8" (float64), "i4" (int32) or "str" (one line per
# element in a .txt file). I() keeps the shape a JSON array even of length one.
.rx_put <- function(w, name, x, dtype = "f8") {
  if (dtype == "str") {
    writeLines(as.character(x), file.path(w$dir, paste0(name, ".txt")))
    w$manifest[[length(w$manifest) + 1L]] <- list(name = name, dtype = "str", shape = I(length(x)))
    return(invisible())
  }
  shp <- if (is.matrix(x)) dim(x) else length(x)
  con <- file(file.path(w$dir, paste0(name, ".bin")), "wb")
  if (dtype == "f8") writeBin(as.double(as.vector(x)), con, size = 8)
  else               writeBin(as.integer(as.vector(x)), con, size = 4)
  close(con)
  w$manifest[[length(w$manifest) + 1L]] <- list(name = name, dtype = dtype, shape = I(shp))
  invisible()
}

.rx_bundle_finish <- function(w, file = "manifest.json") {
  writeLines(jsonlite::toJSON(w$manifest, auto_unbox = TRUE), file.path(w$dir, file))
}

# --- parts of a model ----------------------------------------------------------

# y, X, the prior weights, and the term of each column of X (attribute
# "assign" of model.matrix), so that a Wald test covers a whole term.
.rx_export_fixed <- function(w, model) {
  .rx_put(w, "y", model$y); .rx_put(w, "X", model$X)
  if (!is.null(model$weights)) .rx_put(w, "weights", model$weights)
  if (!is.null(model$X_assign)) {
    .rx_put(w, "X_assign", as.integer(model$X_assign), "i4")
    .rx_put(w, "X_terms", as.character(model$X_terms), "str")
  }
}

# Each term: Z stacked as n x (t*q) in 0-based COO triplets (column
# (a - 1) * q + level), its structures and their options, the Cholesky factor
# of K, or the sparse precision Kinv with log|K|.
.rx_export_terms <- function(w, terms) {
  .rx_put(w, "term_names", vapply(terms, `[[`, "", "name"), "str")
  for (tm in terms) {
    p <- paste0("term_", tm$name)
    S  <- methods::as(do.call(cbind, tm$Zl), "TsparseMatrix")
    .rx_put(w, paste0(p, "_zi"), S@i, "i4")
    .rx_put(w, paste0(p, "_zj"), S@j, "i4")
    .rx_put(w, paste0(p, "_zx"), S@x)
    .rx_put(w, paste0(p, "_t"), tm$t, "i4")
    .rx_put(w, paste0(p, "_q"), tm$q, "i4")
    .rx_put(w, paste0(p, "_rank"), tm$rank, "i4")
    .rx_put(w, paste0(p, "_struct"), tm$struct, "str")
    .rx_put(w, paste0(p, "_lvl"), tm$level, "str")
    if (!is.null(tm$order) && tm$order > 0L)
      .rx_put(w, paste0(p, "_lvlorder"), as.integer(tm$order), "i4")
    if (!is.null(tm$coord)) .rx_put(w, paste0(p, "_coord"), tm$coord)
    if (!is.null(tm$dims)) .rx_put(w, paste0(p, "_dims"), as.integer(tm$dims), "i4")
    if (!is.null(tm$opts) && length(tm$opts)) {
      .rx_put(w, paste0(p, "_lvloptk"), names(tm$opts), "str")
      .rx_put(w, paste0(p, "_lvloptv"), as.numeric(tm$opts))
    }
    if (!is.null(tm$expr)) .rx_put(w, paste0(p, "_lvlexpr"), tm$expr, "str")
    if (!is.null(tm$parts) && length(tm$parts)) {
      # `parts`, a list of (family, dimension), goes as two parallel vectors.
      .rx_put(w, paste0(p, "_lvlpartk"),
              vapply(tm$parts, function(x) as.character(x[[1]]), character(1)), "str")
      .rx_put(w, paste0(p, "_lvlpartq"),
              vapply(tm$parts, function(x) as.integer(x[[2]]), integer(1)), "i4")
    }
    if (!is.null(tm$LK)) .rx_put(w, paste0(p, "_LK"), tm$LK)
    if (!is.null(tm$Kinv)) {
      Ti <- methods::as(methods::as(tm$Kinv, "CsparseMatrix"), "TsparseMatrix")
      .rx_put(w, paste0(p, "_Kinv_i"), Ti@i, "i4"); .rx_put(w, paste0(p, "_Kinv_j"), Ti@j, "i4")
      .rx_put(w, paste0(p, "_Kinv_x"), Ti@x)
      ld <- tm$Kinv_logdet %||%
        -as.numeric(Matrix::determinant(Matrix::forceSymmetric(tm$Kinv), logarithm = TRUE)$modulus)
      .rx_put(w, paste0(p, "_Kinv_logdet"), ld)
    }
    if (!is.null(tm$levels)) .rx_put(w, paste0(p, "_levels"), tm$levels, "str")
  }
}

# The residual, then its sections (same fields, prefix res_s<k>).
.rx_export_residual <- function(w, model) {
  .rx_export_section(w, "res", model$residual, model$n)
  secs <- model$residual$sections
  .rx_put(w, "res_nsec", as.integer(length(secs)), "i4")
  if (length(secs)) {
    .rx_put(w, "res_secnames", vapply(secs, function(s) s$name %||% "section", ""), "str")
    for (k in seq_along(secs)) {
      s <- secs[[k]]
      .rx_export_section(w, sprintf("res_s%d", k - 1L), s, length(s$rows))
      .rx_put(w, sprintf("res_s%d_rows", k - 1L), as.integer(s$rows) - 1L, "i4")
    }
  }
}

# One residual section. Units are 0-based codes. On an ar1 x ar1 grid the unit
# keeps its cell code, (row - 1) * n_columns + column, which is its position in
# the Kronecker product: renumbering it with factor() would drop the empty
# cells and shift the following ones. Elsewhere the unit is only a label.
.rx_export_section <- function(w, pre, r, n_rows) {
  .rx_put(w, paste0(pre, "_struct"), r$struct, "str")
  .rx_put(w, paste0(pre, "_lvl"), r$level %||% "id", "str")
  if (!is.null(r$order) && r$order > 0L)
    .rx_put(w, paste0(pre, "_lvlorder"), as.integer(r$order), "i4")
  if (!is.null(r$coord)) .rx_put(w, paste0(pre, "_coord"), as.matrix(r$coord))
  if (!is.null(r$dims)) .rx_put(w, paste0(pre, "_dims"), as.integer(r$dims), "i4")
  if (!is.null(r$opts) && length(r$opts)) {
    .rx_put(w, paste0(pre, "_lvloptk"), names(r$opts), "str")
    .rx_put(w, paste0(pre, "_lvloptv"), as.numeric(r$opts))
  }
  if (!is.null(r$expr)) .rx_put(w, paste0(pre, "_lvlexpr"), r$expr, "str")
  .rx_put(w, paste0(pre, "_t"), if (is.null(r$trait)) 1L else nlevels(factor(r$trait)), "i4")
  .rx_put(w, paste0(pre, "_rank"), r$rank, "i4")
  .rx_put(w, paste0(pre, "_trait"),
          if (is.null(r$trait)) rep(0L, n_rows) else as.integer(factor(r$trait)) - 1L, "i4")
  grid <- identical(r$level, "ar1ar1") && !is.null(r$n_unit) && !is.null(r$unit)
  .rx_put(w, paste0(pre, "_unit"),
          if (is.null(r$unit)) seq_len(n_rows) - 1L else
          if (grid) as.integer(r$unit) - 1L else as.integer(factor(r$unit)) - 1L, "i4")
  if (grid) .rx_put(w, paste0(pre, "_nunit"), as.integer(r$n_unit), "i4")
}

# Spatial lag: W in 0-based COO triplets (one shared, or one per receiving
# trait), 0-based trait and unit of each row, and three q x q matrices
# (receiver in rows): free entries, values of the fixed entries, starting
# values of the free ones.
.rx_export_lag <- function(w, L) {
  if (is.null(L)) return(invisible())
  .rx_put(w, "lag_q", L$q, "i4"); .rx_put(w, "lag_N", L$N, "i4")
  .rx_put(w, "lag_nW", length(L$W), "i4")
  for (k in seq_along(L$W)) {
    S <- methods::as(L$W[[k]], "TsparseMatrix")
    .rx_put(w, sprintf("lag_W%d_i", k - 1L), S@i, "i4")
    .rx_put(w, sprintf("lag_W%d_j", k - 1L), S@j, "i4")
    .rx_put(w, sprintf("lag_W%d_x", k - 1L), S@x)
  }
  .rx_put(w, "lag_trait", as.integer(L$trait) - 1L, "i4")
  .rx_put(w, "lag_unit", as.integer(L$unit) - 1L, "i4")
  .rx_put(w, "lag_mask", matrix(as.integer(L$mask), L$q, L$q), "i4")
  .rx_put(w, "lag_value", matrix(as.numeric(L$value), L$q, L$q))
  .rx_put(w, "lag_init", matrix(as.numeric(L$init), L$q, L$q))
  .rx_put(w, "lag_traits", L$traits, "str")
  .rx_put(w, "lag_criterion", L$criterion, "str")
}
