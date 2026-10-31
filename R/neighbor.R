# ==============================================================================
# neighbor.R - Neighbor GWAS of Sato et al. (2021), univariate and multi-trait
# ==============================================================================
# Sato, Yamamoto, Shimizu & Nagano (2021, Heredity 126:597-614,
# doi:10.1038/s41437-020-00401-w), as implemented in rNeighborGWAS 1.2.5:
#
#   neighbour covariate  x_i * mean(x_j) over the plants j of the same group
#                        with 0 < distance(i, j) <= scale (+/-1 coding, 0 for
#                        heterozygotes); 0 for a plant without neighbours.
#                        It is the `sim` family of rx_scan(), with neighbour
#                        weights W[i, j] = 1 / |N(i)|.
#   self kinship         K1 = ((q - 1)/2 + X X'/2) / (q - 1)
#   neighbour kinship    K2 = X2 X2' / (q - 1),  X2 the neighbour covariates
#                        of all q markers
#   both projected to the nearest positive definite matrix (Matrix::nearPD).
#   self effect          tested under the null with K1 only (rx_scan "dir")
#   neighbour effect     tested given the self effect, under K1 + K2
#                        (rx_scan "sim|dir"), both at fixed V.
#
# The multi-trait generalisation keeps the same matrices on the plants and
# puts a covariance between traits on each term (us by default):
#   u1 ~ N(0, Sigma_1 (x) K1), u2 ~ N(0, Sigma_2 (x) K2), e ~ N(0, Sigma_e (x) I)
# and tests the SNP effects per trait with the multi-trait grammar of
# rx_scan(): "dir" (any effect, T df), "dir[trait]", "dir@common",
# "dir@het", "sim|dir", "sim@het|dir", ...
# ==============================================================================

#' Neighbour weights of Neighbor GWAS
#'
#' `W[i, j] = 1 / |N(i)|` for the plants `j` of the same group as `i` with
#' `0 < d(i, j) <= scale` (Euclidean distance), zero elsewhere. A plant
#' without neighbours has a zero row. This is the neighbourhood of
#' `rNeighborGWAS::nei_coval()` without distance decay.
#' @param coord matrix or data frame with two columns of coordinates.
#' @param scale neighbourhood radius, in the units of `coord`.
#' @param group optional grouping (plots, trays): neighbours share a group.
#' @return a sparse n x n matrix.
#' @export
rx_neighbor_weights <- function(coord, scale, group = NULL) {
  coord <- as.matrix(coord)
  if (ncol(coord) != 2L) stop("rx_neighbor_weights: `coord` needs two columns.", call. = FALSE)
  n <- nrow(coord)
  if (is.null(group)) group <- rep(1L, n)
  if (length(group) != n) stop("rx_neighbor_weights: `group` has ", length(group),
                               " elements for ", n, " plants.", call. = FALSE)
  ii <- integer(0); jj <- integer(0); xx <- numeric(0)
  for (g in unique(group)) {
    id <- which(group == g)
    d2 <- outer(coord[id, 1], coord[id, 1], "-")^2 + outer(coord[id, 2], coord[id, 2], "-")^2
    nb <- d2 > 0 & d2 <= scale^2
    cnt <- rowSums(nb)
    w <- which(nb, arr.ind = TRUE)
    if (nrow(w)) {
      ii <- c(ii, id[w[, 1]]); jj <- c(jj, id[w[, 2]]); xx <- c(xx, 1 / cnt[w[, 1]])
    }
  }
  Matrix::sparseMatrix(i = ii, j = jj, x = xx, dims = c(n, n))
}

#' Self and neighbour kinships of Neighbor GWAS
#'
#' @param M dose matrix, plants x markers, coded -1/+1 (0 for heterozygotes).
#' @param W neighbour weights from [rx_neighbor_weights()].
#' @param near_pd project both matrices to the nearest positive definite
#'   matrix, as rNeighborGWAS does.
#' @return list(K1, K2, X2) with X2 the neighbour covariates of all markers.
#' @export
rx_neighbor_kinship <- function(M, W, near_pd = TRUE) {
  M <- as.matrix(M); M[is.na(M)] <- 0
  if (nrow(W) != nrow(M) || ncol(W) != nrow(M))
    stop("rx_neighbor_kinship: W must be ", nrow(M), " x ", nrow(M), ".", call. = FALSE)
  q <- ncol(M)
  X2 <- M * as.matrix(W %*% M)
  K1 <- ((q - 1) / 2 + tcrossprod(M) / 2) / (q - 1)
  K2 <- tcrossprod(X2) / (q - 1)
  if (near_pd) {
    K1 <- as.matrix(Matrix::nearPD(K1, maxit = 1e6)$mat)
    K2 <- as.matrix(Matrix::nearPD(K2, maxit = 1e6)$mat)
  }
  ids <- rownames(M)
  if (!is.null(ids)) { dimnames(K1) <- dimnames(K2) <- list(ids, ids); rownames(X2) <- ids }
  list(K1 = K1, K2 = K2, X2 = X2)
}

#' Neighbor GWAS, univariate or multi-trait
#'
#' Fits the null model of Sato et al. (2021) on the plants, with one trait or
#' several, and scans the markers at fixed V.
#'
#' @param data one row per plant, with the traits, the plant identifier and
#'   the fixed covariates.
#' @param traits names of the trait columns (one or more).
#' @param plant name of the plant identifier column (the rows of `markers`).
#' @param markers plants x markers, -1/+1 (0 for heterozygotes), rows named by
#'   plant.
#' @param W neighbour weights between the plants, in the order of
#'   `rownames(markers)` ([rx_neighbor_weights()]).
#' @param fixed right-hand side of the fixed effects, applied per trait
#'   (`~ 1` by default; `~ group` for a grouping covariate as in
#'   rNeighborGWAS `addcovar`).
#' @param struct covariance between traits of each term ("us" by default,
#'   "diag" for independent traits); ignored with one trait.
#' @param tests scan tests. Tests that involve only the `dir` family are run
#'   under the null with K1 alone when `self_null = "K1"`, as in Sato et al.;
#'   the others under K1 + K2.
#' @param self_null "K1" (Sato) or "K1+K2" (one null for every test).
#' @param kin optional list(K1, K2) to reuse matrices already computed.
#' @param fit_args extra arguments of [rx_reml()] (backend, engine...).
#' @return a list: `scan` (one row per marker), `fit` (null with K1 + K2),
#'   `fit_self` (null with K1, or NULL), `pve` (per trait: shares of the self,
#'   neighbour and residual variances in the null with K1 + K2), and the
#'   matrices.
#' @export
rx_neighbor_gwas <- function(data, traits, plant, markers, W, fixed = ~ 1,
                             struct = c("us", "diag"),
                             tests = c("dir", "sim|dir"), self_null = c("K1", "K1+K2"),
                             kin = NULL, maf = NULL, fit_args = list(), verbose = TRUE) {
  struct <- match.arg(struct); self_null <- match.arg(self_null)
  ids <- rownames(markers)
  if (is.null(ids)) stop("rx_neighbor_gwas: `markers` needs row names (plants).", call. = FALSE)
  if (!all(as.character(data[[plant]]) %in% ids))
    stop("rx_neighbor_gwas: plants of `data` missing from `markers`.", call. = FALSE)
  if (is.null(kin)) kin <- rx_neighbor_kinship(markers, W)
  K1 <- kin$K1; K2 <- kin$K2; dimnames(K1) <- dimnames(K2) <- list(ids, ids)
  T <- length(traits)
  long <- .rx_neighbor_long(data, traits, plant, ids)
  fm <- .rx_neighbor_formulas(T, fixed, struct, environment())
  args0 <- c(list(fixed = fm$fixed, residual = fm$residual, data = long, verbose = FALSE),
             if (T > 1L) list(trait = ".trait", unit = ".unit"), fit_args)
  fit <- do.call(rx_reml, c(list(random = fm$full), args0))
  only_dir <- vapply(tests, function(t) all(.rx_scan_test_families(t) == "dir"), TRUE)
  fit_self <- if (self_null == "K1" && any(only_dir))
    do.call(rx_reml, c(list(random = fm$self), args0))
  # Row-level neighbour incidence for the `sim` family: the weights of the
  # plant of each row, over the plants (columns = levels of .plant).
  Zn <- W[as.integer(long$.plant), , drop = FALSE]
  run <- function(f, tt) {
    inc <- if (any(.rx_scan_test_families(tt) != "dir")) list(dir = ".plant", ind = Zn)
           else list(dir = ".plant")
    rx_scan(f, f$model, markers = as.matrix(markers), incidences = inc, tests = tt,
            maf = maf, verbose = verbose, backend = fit_args$backend %||% "auto")
  }
  out <- if (!is.null(fit_self)) {
    o <- run(fit_self, tests[only_dir])
    if (any(!only_dir)) { r2 <- run(fit, tests[!only_dir]); o <- cbind(o, r2[setdiff(names(r2), names(o))]) }
    o
  } else run(fit, tests)
  c(list(scan = out, fit = fit, fit_self = fit_self), .rx_neighbor_shares(fit, traits, K1, K2),
    list(K1 = K1, K2 = K2))
}

# Long format, one row per plant x trait, plants in the order of `ids`.
.rx_neighbor_long <- function(data, traits, plant, ids) {
  long <- do.call(rbind, lapply(traits, function(tr) {
    d <- data; d$.y <- d[[tr]]; d$.trait <- tr; d
  }))
  long$.trait <- factor(long$.trait, levels = traits)
  long$.plant <- factor(as.character(long[[plant]]), levels = ids)
  long$.plantN <- long$.plant
  long$.unit <- long$.plant
  long[order(long$.plant, long$.trait), ]
}

# Fixed, random (K1 + K2, and K1 alone) and residual formulas; K1 and K2 are
# looked up in `env`.
.rx_neighbor_formulas <- function(T, fixed, struct, env) {
  rhs <- paste(deparse(fixed[[length(fixed)]]), collapse = " ")
  fx <- stats::as.formula(
    if (T == 1L) paste(".y ~", rhs)
    else if (rhs == "1") ".y ~ .trait"
    else paste0(".y ~ .trait + .trait:(", rhs, ")"))
  st <- if (T > 1L) struct else "vm"
  full <- stats::as.formula(sprintf("~ %s(.plant, K = K1) + %s(.plantN, K = K2)", st, st), env = env)
  self <- stats::as.formula(sprintf("~ %s(.plant, K = K1)", st), env = env)
  list(fixed = fx, full = full, self = self,
       residual = if (T > 1L) stats::as.formula(sprintf("~ %s(.trait):units", struct)) else "units")
}

# pve: the definition of rNeighborGWAS (variance parameters over their sum).
# share: each parameter times the mean diagonal of its matrix, the expected
# variance contribution per plant; K2 has a small diagonal when the neighbours
# are few, and the raw ratio then overstates the neighbour term. Correlations
# between traits of the self and neighbour terms with several traits.
.rx_neighbor_shares <- function(fit, traits, K1, K2) {
  S1 <- as.matrix(fit$sigmas$.plant); S2 <- as.matrix(fit$sigmas$.plantN)
  Se <- as.matrix(fit$sigma_res)
  tot <- diag(S1) + diag(S2) + diag(Se)
  v1 <- diag(S1) * mean(diag(K1)); v2 <- diag(S2) * mean(diag(K2)); vt <- v1 + v2 + diag(Se)
  list(pve = data.frame(trait = traits, self = diag(S1) / tot, neighbour = diag(S2) / tot,
                        residual = diag(Se) / tot, share_self = v1 / vt,
                        share_neighbour = v2 / vt, share_residual = diag(Se) / vt),
       cor_self = if (length(traits) > 1L) stats::cov2cor(S1),
       cor_neighbour = if (length(traits) > 1L) stats::cov2cor(S2))
}

# Families named by a scan test ("sim@het|dir" -> c("sim", "dir")).
.rx_scan_test_families <- function(test) {
  it <- trimws(unlist(strsplit(test, "[|+]")))
  unique(sub("[\\[@].*$", "", it))
}

#' Choice of the neighbourhood scale (Neighbor GWAS)
#'
#' Fits the null model with K1 + K2 for each scale, univariate or multi-trait,
#' as `rNeighborGWAS::calc_PVEnei()` does for one trait: log-likelihood, the
#' likelihood ratio test of the neighbour term against the null with K1
#' alone (variance at the boundary: mixture 0.5 chi2_0 + 0.5 chi2_1 for one
#' trait; with T traits the reference distribution of a covariance matrix at
#' the boundary is not chi2, and the p-value is given for one trait only), and
#' the shares of variance per trait.
#' @inheritParams rx_neighbor_gwas
#' @param coord plant coordinates (two columns, rows in the order of
#'   `rownames(markers)`).
#' @param scales radii to try.
#' @param group grouping of the plants (neighbours share a group).
#' @return a data frame, one row per scale and trait.
#' @export
rx_neighbor_scale <- function(data, traits, plant, markers, coord, scales, group = NULL,
                              fixed = ~ 1, struct = c("us", "diag"), fit_args = list()) {
  struct <- match.arg(struct)
  M <- as.matrix(markers); M[is.na(M)] <- 0
  q <- ncol(M)
  K1 <- ((q - 1) / 2 + tcrossprod(M) / 2) / (q - 1)
  K1 <- as.matrix(Matrix::nearPD(K1, maxit = 1e6)$mat)
  rows <- list()
  fit0 <- NULL
  for (s in sort(scales)) {
    W <- rx_neighbor_weights(coord, s, group)
    X2 <- M * as.matrix(W %*% M)
    K2 <- as.matrix(Matrix::nearPD(tcrossprod(X2) / (q - 1), maxit = 1e6)$mat)
    nb <- rx_neighbor_gwas(data, traits, plant, M, W, fixed = fixed, struct = struct,
                           tests = "dir", self_null = "K1", kin = list(K1 = K1, K2 = K2),
                           fit_args = fit_args, verbose = FALSE)
    if (is.null(fit0)) fit0 <- nb$fit_self
    lrt <- 2 * (nb$fit$logLik - fit0$logLik)
    p <- if (length(traits) == 1L) 0.5 * stats::pchisq(max(lrt, 0), 1, lower.tail = FALSE) else NA_real_
    rows[[length(rows) + 1L]] <- data.frame(scale = s, nb$pve, logLik = nb$fit$logLik,
                                            logLik_K1 = fit0$logLik, lrt = lrt, p_lrt = p)
  }
  do.call(rbind, rows)
}
