# ==============================================================================
# formula_residual.R - the residual formula of rx_reml(), as in asreml
# ==============================================================================
#   residual = ~ units                          R = sigma^2 I
#   residual = ~ us(trait):units                traits of a unit correlated
#   residual = ~ ar1(row):ar1(col)              separable residual field
#   residual = ~ diag(trait):ar1(row):ar1(col)  both at once
#   residual = ~ exp(pos, coord = x)            metric decay
#
# The formula is a product of factors separated by ":". At most one factor
# acts on the traits (the one whose argument is the column given as `trait`);
# the others act on the units. `units` (or `id(units)`) means no structure
# between units: two observations are correlated only on the same unit.
#
# dsum() gives one residual structure per section (direct sum):
#   residual = ~ dsum(~ ar1(row):ar1(col) | site)
#       the same form on every site, with parameters of its own on each
#   residual = ~ dsum(~ ar1(row):ar1(col) + units | site, levels = list(1:3, 4))
#       different forms on different sites
# The sections partition the observations, so that two trials of different
# sizes or geometries can be fitted together without imposing the spatial
# structure of one on the other.
# ==============================================================================

.rx_parse_residual <- function(residual, data, trait = NULL, unit = NULL) {
  if (inherits(residual, "formula")) {
    labels <- attr(terms(residual, keep.order = TRUE), "term.labels")
    if (length(labels) == 1L && grepl("^dsum\\(", labels))
      return(.rx_parse_dsum(str2lang(labels), data, trait, unit,
                            env = environment(residual) %||% parent.frame()))
  }
  .rx_parse_residual_section(residual, data, trait = trait, unit = unit)
}

# One residual section per group of levels of the section factor. The result
# is the first section, with all sections in `sections`.
.rx_parse_dsum <- function(e, data, trait, unit, env) {
  args <- as.list(e)[-1]
  arg_names <- names(args); if (is.null(arg_names)) arg_names <- rep("", length(args))
  inner <- args[arg_names == ""][[1]]
  # terms() returns the argument as a one-sided formula whose body is
  # `structure | section`: the `~` must be unwrapped before looking for `|`.
  if (is.call(inner) && identical(as.character(inner[[1]]), "~"))
    inner <- inner[[length(inner)]]
  if (!is.call(inner) || !identical(as.character(inner[[1]]), "|"))
    stop("dsum: write dsum(~ <structure> | <section factor>). Got ",
         deparse(inner)[1])
  structures <- .rx_dsum_terms(inner[[2]]); section_var <- as.character(inner[[3]])
  if (!section_var %in% names(data))
    stop("dsum: section column '", section_var, "' is missing from `data`.")
  section_factor <- factor(data[[section_var]])
  # Several structures separated by `+` need one group of levels each;
  # otherwise the same structure applies to every level.
  groups <- if (!is.null(args$levels)) eval(args$levels, envir = data, enclos = env) else NULL
  if (length(structures) > 1L) {
    if (is.null(groups) || length(groups) != length(structures))
      stop("dsum: ", length(structures), " structures separated by '+' but ",
           if (is.null(groups)) "no" else length(groups),
           " group(s) in `levels`. Give levels = list(...) of the same length.")
  } else {
    groups <- as.list(levels(section_factor)); structures <- rep(structures, length(groups))
  }
  sections <- list()
  for (k in seq_along(structures)) {
    group <- as.character(groups[[k]])
    idx <- which(as.character(section_factor) %in% group)
    if (!length(idx))
      stop("dsum: group ", k, " (", paste(group, collapse = ", "),
           ") covers no observation.")
    s <- .rx_parse_residual_section(
      stats::as.formula(call("~", structures[[k]]), env = env),
      data[idx, , drop = FALSE],
      trait = .rx_subset_column(trait, idx, names(data)),
      unit  = .rx_subset_column(unit, idx, names(data)))
    s$rows <- idx
    s$name <- .rx_section_name(group, section_var)
    sections[[k]] <- s
  }
  covered <- unlist(lapply(sections, `[[`, "rows"))
  if (length(covered) != nrow(data) || anyDuplicated(covered))
    stop("dsum: the sections do not partition the ", nrow(data),
         " observations (", length(covered), " rows listed, ",
         length(unique(covered)), " distinct). A missing row would leave V ",
         "with a zero variance. A duplicated row would count twice.")
  r <- sections[[1]]; r$sections <- sections; r$rows <- NULL; r$name <- NULL
  r
}

# Name of a section: the level when it is a valid R name ("A"), otherwise the
# section variable and the level ("site_1", not the "X1" of make.names()).
.rx_section_name <- function(group, section_var) {
  raw <- paste(group, collapse = "_")
  if (identical(make.names(raw), raw)) raw else make.names(paste0(section_var, "_", raw))
}

# Top-level terms of a formula body, without splitting the inner ":".
.rx_dsum_terms <- function(e) {
  if (is.call(e) && identical(as.character(e[[1]]), "+"))
    return(c(.rx_dsum_terms(e[[2]]), list(e[[3]])))
  list(e)
}

# Subset of a `trait` or `unit` given as a column name (kept as is) or as a
# vector (subset).
.rx_subset_column <- function(x, idx, data_names) {
  if (is.null(x)) return(NULL)
  if (is.character(x) && length(x) == 1L && x %in% data_names) return(x)
  x[idx]
}

# Name of the column of `data` holding `x`: `x` itself when it is a column
# name, otherwise the column equal to the vector `x`. The parser needs the
# name to recognise diag(trait) in the formula; without it the trait factor
# was read as a unit factor and gave another model without any error.
.rx_column_name <- function(x, data) {
  if (is.character(x) && length(x) == 1L && x %in% names(data)) return(x)
  if (is.null(x)) return(NULL)
  for (nm in names(data))
    if (length(data[[nm]]) == length(x) &&
        identical(as.character(data[[nm]]), as.character(x))) return(nm)
  NULL
}

# Vector of `x`, given as a column name or as a vector.
.rx_column_vector <- function(x, data) {
  if (is.character(x) && length(x) == 1L && x %in% names(data)) return(data[[x]])
  x
}

# One residual structure (no dsum).
.rx_parse_residual_section <- function(residual, data, trait = NULL, unit = NULL) {
  if (is.character(residual) && length(residual) == 1L &&
      residual %in% c("units", "iid", "diag", "us", "fa", "none")) {
    r <- if (residual == "units") "iid" else residual
    return(rx_residual(r, trait = .rx_column_vector(trait, data),
                       unit = .rx_column_vector(unit, data)))
  }
  if (!inherits(residual, "formula")) stop("`residual`: a string or a formula is expected.")
  env <- environment(residual); if (is.null(env)) env <- parent.frame()
  labels <- attr(terms(residual, keep.order = TRUE), "term.labels")
  if (!length(labels)) stop("`residual` contains no factor.")
  factors <- trimws(unlist(strsplit(labels, ":", fixed = TRUE)))
  trait_name <- .rx_column_name(trait, data)
  unit_name  <- .rx_column_name(unit, data)

  # ar1(row):ar1(col) is recognised before the general loop, where the second
  # ar1() would count as a second structure between units.
  if (sum(grepl("^ar1\\(", factors)) == 2L)
    return(.rx_residual_ar1ar1(factors, data, trait, trait_name))

  p <- .rx_residual_factors(factors, data, trait_name, env)
  unit_factors <- setdiff(p$unit_factors, trait_name)
  if (length(unit_factors) > 1L)
    stop("residual: several unit factors (", paste(unit_factors, collapse = ", "), ").")
  unit_vec <- if (!is.null(p$cell_unit)) p$cell_unit else
              if (length(unit_factors)) data[[unit_factors[1]]] else
              if (!is.null(unit_name)) data[[unit_name]] else .rx_column_vector(unit, data)
  trait_vec <- if (!is.null(trait_name)) data[[trait_name]] else .rx_column_vector(trait, data)
  coord <- p$coord
  if (length(coord) == 1L && is.na(coord))
    coord <- .rx_unit_positions(unit_vec, p$level, unit_factors)
  n_unit <- if (!is.null(p$cell_unit)) length(unique(p$cell_unit)) else NULL
  # A lag structure keeps the empty levels of its grid: the unit becomes its
  # rank in the grid and n_unit counts the levels without observation too.
  if (p$level %in% RX_LAG_1D && length(unit_factors) && is.null(coord)) {
    g <- .rx_grid_factor(unit_vec, paste0("residual = ~ ", p$level, "(", unit_factors[1], ")"))
    unit_vec <- as.integer(g); n_unit <- nlevels(g)
  }
  rx_residual(p$struct, trait = trait_vec, unit = unit_vec, rank = p$rank,
              level = p$level, order = p$order, coord = coord,
              opts = p$opts, expr = p$expr, n_unit = n_unit)
}

# ar1(row):ar1(col), possibly with a structure between traits. The unit is the
# cell: level = (row - 1) * n_columns + column, row slow and column fast, the
# order of the Kronecker product in the solver. With t traits in long format
# each cell appears t times by construction, so duplicates are checked on the
# pair (cell, trait): only the same trait twice in a cell makes R singular.
.rx_residual_ar1ar1 <- function(factors, data, trait, trait_name) {
  is_ar1 <- grepl("^ar1\\(", factors)
  cols <- vapply(factors[is_ar1], function(z) as.character(str2lang(z)[[2]]), "")
  if (!all(cols %in% names(data)))
    stop("residual: column(s) not found: ",
         paste(setdiff(cols, names(data)), collapse = ", "))
  what <- paste0("residual = ~ ar1(", cols[1], "):ar1(", cols[2], ")")
  fr <- .rx_grid_factor(data[[cols[1]]], what)
  fc <- .rx_grid_factor(data[[cols[2]]], what)
  unit <- (as.integer(fr) - 1L) * nlevels(fc) + as.integer(fc)
  key <- if (!is.null(trait_name)) paste(unit, as.integer(factor(data[[trait_name]])),
                                         sep = "\r") else unit
  if (anyDuplicated(key))
    stop("residual = ~ ar1(", cols[1], "):ar1(", cols[2], "): ",
         sum(duplicated(key)), " duplicated (cell, trait) pair(s). ",
         "A structured residual field allows at most one observation per cell ",
         "and per trait.")
  struct <- "iid"
  for (fx in factors[!is_ar1]) {
    if (fx %in% c("units", "id(units)")) next
    e <- str2lang(fx)
    if (is.call(e) && !is.null(trait_name) && identical(as.character(e[[2]]), trait_name))
      struct <- switch(as.character(e[[1]]), iid = , idv = , id = "iid",
                       diag = , idh = "diag", us = , corgh = "us", "iid")
  }
  rx_residual(struct,
              trait = if (is.null(trait_name)) .rx_column_vector(trait, data) else data[[trait_name]],
              unit = unit, level = "ar1ar1", dims = c(nlevels(fr), nlevels(fc)),
              n_unit = nlevels(fr) * nlevels(fc))
}

# Reads the factors of a residual formula other than ar1(row):ar1(col): the
# structure between traits, at most one structure between units with its
# options, and the unit factors. Coordinates are NA when a 1D metric structure
# must take them from the unit levels.
.rx_residual_factors <- function(factors, data, trait_name, env) {
  p <- list(struct = "iid", rank = 0L, level = "id", order = 0L, coord = NULL,
            opts = NULL, expr = NULL, cell_unit = NULL, unit_factors = character(0))
  for (fx in factors) {
    if (fx %in% c("units", "id(units)")) next
    e <- str2lang(fx)
    if (is.name(e)) {
      p$unit_factors <- c(p$unit_factors, as.character(e)); next
    }
    f  <- as.character(e[[1]])
    a1 <- if (length(e) >= 2L) as.character(e[[2]]) else ""
    args <- as.list(e)[-(1:2)]
    if (f %in% c("iid", "idv", "id", "diag", "idh", "us", "corgh", "fa")) {
      if (!is.null(trait_name) && identical(a1, trait_name)) {
        p$struct <- switch(f, iid = , idv = , id = "iid", diag = , idh = "diag",
                           us = , corgh = "us", fa = "fa")
        if (!is.null(args$rank)) p$rank <- as.integer(eval(args$rank, env))
      } else if (!a1 %in% c("units", "")) {
        p$unit_factors <- c(p$unit_factors, a1)
      }
      next
    }
    if (f %in% RX_LEVEL_STRUCTURES) {
      p <- .rx_residual_level(p, f, e, fx, a1, args, data, env)
      next
    }
    stop("residual: unrecognised factor '", fx, "'.\n  Expected: units, ",
         "id/iid/diag/us/fa(<trait>), or a structure between levels (",
         paste(setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed")), collapse = "/"), ").")
  }
  p
}

# A structure between units in the residual formula. Its options are read
# before any early exit, so that mtrn(x, y, phi = ...) keeps them. A 2D metric
# structure written on two columns, mtrn(x, y), takes the cell (x, y) as unit
# and its coordinates from the data.
.rx_residual_level <- function(p, f, e, fx, a1, args, data, env) {
  if (p$level != "id")
    stop("residual: two structures between units ('", p$level, "' and '", f,
         "'). Only one factor structures the units, except ar1(r):ar1(c), ",
         "written ar1(", a1, ", <column>).")
  p$level <- f
  positional <- as.list(e)[-1]
  pos_names <- names(positional); if (is.null(pos_names)) pos_names <- rep("", length(positional))
  positional <- positional[pos_names == ""]
  if (!is.null(args$order)) p$order <- as.integer(eval(args$order, env))
  if (!is.null(args$coord)) p$coord <- as.matrix(eval(args$coord, envir = data, enclos = env))
  if (f == "mtrn") p$opts <- .rx_mtrn_opts(args, env)
  if (f == "own") {
    p$expr <- as.character(eval(args$expr, env))
    p$opts <- c(n_par = as.numeric(eval(args$n_par %||% 1L, env)),
                normalise = as.numeric(eval(args$normalise %||% TRUE, env)))
  }
  if (f %in% RX_METRIC_2D && length(positional) == 2L && is.null(p$coord)) {
    xv <- eval(positional[[1]], envir = data, enclos = env)
    yv <- eval(positional[[2]], envir = data, enclos = env)
    key <- paste(xv, yv, sep = "\r")
    u <- !duplicated(key); ord_u <- order(key[u]); cell_levels <- key[u][ord_u]
    cell <- as.integer(factor(key, levels = cell_levels))
    if (anyDuplicated(cell))
      stop("residual = ~ ", fx, ": ", sum(duplicated(cell)),
           " duplicated cell(s). A structured residual field allows at most ",
           "one observation per cell.")
    p$coord <- cbind(as.numeric(xv[u][ord_u]), as.numeric(yv[u][ord_u]))
    p$cell_unit <- cell
    return(p)
  }
  p$unit_factors <- c(p$unit_factors, a1)
  if (f %in% RX_METRIC_1D && is.null(p$coord)) p$coord <- NA
  p
}

# Positions of a 1D metric structure given without coordinates: the unit
# levels, which must then be numeric.
.rx_unit_positions <- function(unit_vec, level, unit_factors) {
  co <- suppressWarnings(as.numeric(levels(factor(unit_vec))))
  if (any(!is.finite(co)))
    stop("residual: '", level, "' requires numeric positions. The levels ",
         "of '", if (length(unit_factors)) unit_factors[1] else "unit", "' are not numeric. ",
         "Give coord = <positions>.")
  as.matrix(co)
}
