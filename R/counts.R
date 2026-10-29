# ==============================================================================
# counts.R - structure catalogues and parameter counts
# ==============================================================================
# The R side and the Python solver must cut theta at the same places. Every
# count here reproduces structures.py (Sigma structures) or levels.py
# (between-level structures); a divergence would make the two sides read theta
# differently without any error.
# ==============================================================================

# Sigma structures (appendix C of the ASReml-R 4.2 manual). Correspondences:
#   iid = idv   diag = idh   us = corgh   fa(k)   rr(k)   chol(k)   ante(k)   corh
RX_STRUCTURES <- c("iid", "diag", "us", "fa", "rr", "chol", "ante", "corh")

# Number of loadings of a trapezoidal t x r loading matrix.
rx_n_loadings <- function(t, r) sum(pmin(seq_len(t), r))

# Number of parameters of a Sigma structure over t columns.
rx_n_params <- function(struct, t, rank = 0L) {
  t <- as.integer(t); k <- as.integer(rank)
  switch(struct,
         iid  = 1L,
         diag = t,
         us   = as.integer(t * (t + 1) / 2),
         fa   = as.integer(rx_n_loadings(t, k) + t),
         # The manual counts k*omega for rr. Gamma is constrained to be
         # trapezoidal, as for fa: otherwise it is only defined up to a
         # rotation and the Hessian is singular in k(k-1)/2 directions.
         rr   = as.integer(rx_n_loadings(t, k)),
         chol = ,
         ante = as.integer((k + 1) * (t - k / 2)),
         corh = as.integer(t + 1L),
         stop("structure inconnue : ", struct))
}

# Between-level (correlation) structures, aligned on appendix C of the
# ASReml-R 4.2 manual. The variance lives in Sigma: asreml's `ar1v` is
# struct = "iid" with level = "ar1" here, and `ar1h` is struct = "diag" with
# level = "ar1".
RX_LEVEL_STRUCTURES <- c("id", "fixed", "cor", "corb", "corg",
                         "ar1", "ar2", "ar3", "sar", "ma1", "ma2", "arma",
                         "exp", "gau", "lvr", "iexp", "igau", "ieuc",
                         "sph", "cir", "aexp", "agau", "mtrn", "own", "ar1ar1",
                         # Separable product of any number of factors. A
                         # spatial field replicated independently by block is
                         # id (x) ar1 (x) ar1: the id factor adds no parameter,
                         # so the correlations are shared between replicates,
                         # which ar1ar1 cannot express. `parts` carries the
                         # list of (family, dimension).
                         "sep")

# Parameters of the structures whose count is fixed. corb has `order`
# parameters, corg order*(order-1)/2, mtrn those declared (.rx_mtrn_opts) and
# own the n_par given by the user.
RX_LEVEL_NP <- c(id = 0L, fixed = 0L, cor = 1L, ar1 = 1L, ar2 = 2L, ar3 = 3L,
                 sar = 1L, ma1 = 1L, ma2 = 2L, arma = 2L, exp = 1L, gau = 1L,
                 lvr = 1L, iexp = 1L, igau = 1L, ieuc = 1L,
                 sph = 1L, cir = 1L, aexp = 2L, agau = 2L, ar1ar1 = 2L)

# Metric structures need coordinates. 1D and 2D are kept apart: a coordinate
# with one column given to iexp() would otherwise be accepted and give another
# model than the one asked for.
RX_METRIC_1D <- c("exp", "gau", "lvr")
RX_METRIC_2D <- c("iexp", "igau", "ieuc", "sph", "cir",
                  "aexp", "agau", "mtrn")
RX_METRIC <- c(RX_METRIC_1D, RX_METRIC_2D)

# Number of parameters of a between-level structure (as levels.py).
rx_n_level <- function(level, order = 0L, opts = NULL, parts = NULL) {
  if (identical(level, "sep")) return(.rx_n_level_sep(parts))
  if (level == "mtrn") return(.rx_n_level_mtrn(opts))
  if (level == "own")  return(as.integer(opts[["n_par"]] %||% 0L))
  if (level == "corb") return(as.integer(order))
  if (level == "corg") return(as.integer(order * (order - 1L) / 2L))
  RX_LEVEL_NP[[level]]
}

# A separable product counts the parameters of its factors; `id` adds none.
# Families whose count depends on an order or on options, and metric families,
# which need coordinates that `parts` does not carry, are refused: they would
# silently count zero parameters and become an identity.
.rx_n_level_sep <- function(parts) {
  if (is.null(parts) || !length(parts))
    stop("level='sep' exige `parts`, la liste des facteurs (famille, dimension).",
         call. = FALSE)
  families <- vapply(parts, function(x) as.character(x[[1]]), "")
  refused <- intersect(families, c("ar2", "ar3", "ma2", "arma", "corb", "corg",
                                   "mtrn", "own", RX_METRIC))
  if (length(refused))
    stop("level='sep' : les familles ", paste(sort(unique(refused)), collapse = ", "),
         " ne sont pas admises dans `parts` (compte dependant d'un ordre ou ",
         "d'options, ou coordonnees requises).", call. = FALSE)
  sum(vapply(parts, function(x) rx_n_level(as.character(x[[1]]), 0L), integer(1)))
}

# Matern counts the parameters flagged as estimated. Defaults as in levels.py:
# phi is estimated unless stated otherwise, nu, delta and alpha are held fixed.
.rx_n_level_mtrn <- function(opts) {
  sum(vapply(c("phi", "nu", "delta", "alpha"), function(p) {
    v <- if (is.null(opts)) numeric(0) else
      suppressWarnings(as.numeric(unlist(opts[paste0("est_", p)])))
    if (length(v) != 1L || is.na(v)) v <- if (p == "phi") 1 else 0
    isTRUE(v > 0.5)
  }, TRUE))
}

# Effective between-level structure of a term. `level = "auto"`, the default
# of rx_term(), means no structure unless a relationship matrix is given: a
# dense Cholesky factor `LK` gives "fixed", a sparse precision `Kinv` gives
# "prec". This is the only place where that resolution is made.
.rx_level_of <- function(tm) {
  lv <- tm$level %||% "auto"
  if (!is.null(tm$Kinv)) return("prec")
  if (!is.null(tm$LK)) return("fixed")
  if (lv %in% c("auto", "id", "iid")) return("id")
  lv
}

# Number of traits of a residual (or of one of its sections).
.rx_n_traits <- function(r) if (is.null(r$trait)) 1L else nlevels(factor(r$trait))

# Length of the theta vector of a model: the parameters of each term (Sigma,
# then between levels), then of each residual section, then the free entries
# of Psi of a spatial lag. It is what theta_init and a common theta imposed on
# two fits must match.
rx_n_theta <- function(model) {
  n_level <- function(lv, order, opts, parts) {
    if (lv %in% c("id", "fixed", "prec")) return(0L)
    as.integer(rx_n_level(lv, order %||% 0L, opts, parts = parts))
  }
  np <- 0L
  for (tm in model$terms)
    np <- np + as.integer(rx_n_params(tm$struct %||% "iid", tm$t, tm$rank %||% 0L)) +
      n_level(.rx_level_of(tm), tm$order, tm$opts, tm$parts)
  res <- model$residual
  for (s in (res$sections %||% list(res))) {
    if (identical(s$struct, "none")) next
    np <- np + as.integer(rx_n_params(s$struct %||% "iid", .rx_n_traits(s), s$rank %||% 0L)) +
      n_level(s$level %||% "id", s$order, s$opts, NULL)
  }
  np + (if (is.null(model$lag)) 0L else as.integer(model$lag$n_free))
}
