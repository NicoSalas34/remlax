# ==============================================================================
# residual.R - rx_residual(): the residual structure
# ==============================================================================

# Residual structure.
#   struct   "iid", "diag", "us", "fa" between the traits of a unit, or "none"
#   trait    trait factor of length n; NULL means a single trait
#   unit     unit identifier: two observations of the same unit on different
#            traits are correlated under "us"
#   level    structure between units (ar1, ar1ar1, metric kernels, ...)
#   sections a list of rx_residual() objects, one per section of a dsum()
# struct = "none" means no residual at all (R = 0): V must then be positive
# definite through the terms alone, typically a plant term whose incidences
# carry the residual, the indirect environmental effects and their covariance.
# The solver checks V at the starting point and says so when it is not.
rx_residual <- function(struct = "iid", trait = NULL, unit = NULL, rank = 0L,
                        level = "id", order = 0L, coord = NULL, n_unit = NULL,
                        dims = NULL, opts = NULL, expr = NULL, sections = NULL,
                        rows = NULL, name = NULL) {
  struct <- match.arg(struct, c("iid", "diag", "us", "fa", "none"))
  if (struct == "none" && (!identical(level, "id") || !is.null(sections)))
    stop("rx_residual(\"none\") : ni structure entre unites ni sections, il n'y a pas de ",
         "residuelle a structurer.", call. = FALSE)
  level  <- match.arg(level, RX_LEVEL_STRUCTURES)
  # A separable product needs its factors (parts), which a residual section
  # does not carry: ar1ar1 covers the separable residual field.
  if (identical(level, "sep"))
    stop("rx_residual : le produit separable (level = \"sep\") n'est pas disponible pour ",
         "la residuelle. Pour un champ ar1 x ar1, utiliser level = \"ar1ar1\" avec dims.",
         call. = FALSE)
  structure(list(struct = struct, trait = trait, unit = unit, rank = as.integer(rank),
                 level = level, order = as.integer(order), coord = coord,
                 n_unit = n_unit, dims = dims, opts = opts, expr = expr,
                 sections = sections, rows = rows, name = name),
            class = "rx_residual")
}
