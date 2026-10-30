# ==============================================================================
# bundle_read.R - reading the solver's result into an rx_fit object
# ==============================================================================
# The solver writes result.json (scalars, small tables and the dimensions of
# every array) and raw float64 files out_*.bin, read back here.
#
# Fields are always accessed with [[ ]], never $: $ matches partially on lists,
# so r$sigmas would match sigmas_res before sigmas exists.
# ==============================================================================

rx_read_result <- function(dir, model = NULL, fixed_theta = NULL) {
  f <- file.path(dir, "result.json")
  if (!file.exists(f)) stop("result not found: ", f)
  r <- jsonlite::fromJSON(f, simplifyVector = TRUE)
  rd <- .rx_bin_reader(dir)
  r <- .rx_read_fixed_effects(r, rd)
  r <- .rx_read_covariances(r, rd)
  r <- .rx_read_hessian(r, rd)
  r <- .rx_read_lag(r, rd)
  r <- .rx_bayes_read(r, rd)
  # Standard errors of theta on the free subspace; NA without a Hessian.
  r[["fixed_theta"]] <- if (length(fixed_theta)) as.integer(fixed_theta) else integer(0)
  r[["se_theta"]] <- rx_se_theta(r[["theta"]], r[["hessian"]],
                                 r[["par_floor"]] %||% -12, r[["par_ceil"]] %||% 12,
                                 r[["fixed_theta"]])
  r <- .rx_label_sigmas_blups(r, model)
  class(r) <- "rx_fit"; r
}

# Reader of out_<name>.bin: a float64 vector, or NULL when the file is absent.
.rx_bin_reader <- function(dir) {
  function(nm) {
    p <- file.path(dir, paste0(nm, ".bin"))
    if (file.exists(p)) readBin(p, "double", n = file.size(p) %/% 8, size = 8) else NULL
  }
}

# theta, beta, the covariance of beta (and its Kenward-Roger version), and the
# covariance of the predictions.
.rx_read_fixed_effects <- function(r, rd) {
  r[["theta"]] <- rd("out_theta"); r[["beta"]] <- rd("out_beta")
  p_ <- length(r[["beta"]])
  v <- rd("out_vbeta");    if (!is.null(v)) r[["vbeta"]]    <- matrix(v, p_, p_)
  v <- rd("out_vbeta_kr"); if (!is.null(v)) r[["vbeta_kr"]] <- matrix(v, p_, p_)
  v <- rd("out_pred_cov")
  if (!is.null(v)) { k_ <- sqrt(length(v))
    r[["predictions"]][["cov"]] <- matrix(v, k_, k_) }
  r
}

# Matrices out_<prefix><name>.bin whose dimensions result.json gives in the
# field `dims_field`, as a list named by term or section.
.rx_read_matrices <- function(r, rd, dims_field, prefix) {
  out <- list()
  for (nm in names(r[[dims_field]])) {
    v <- rd(paste0(prefix, nm))
    if (!is.null(v)) out[[nm]] <- matrix(v, r[[dims_field]][[nm]][1], r[[dims_field]][[nm]][2])
  }
  out
}

# Sigma of each term and of each residual section, the residual Sigma, the
# loadings of fa and rr terms, the prediction error variances and the BLUPs.
# `psi` is the specific variance, as in asreml's tables: the diagonal of Sigma
# is sum(Lambda^2) + psi. A pure reduced-rank `rr` has none (NULL, never zeros).
.rx_read_covariances <- function(r, rd) {
  r[["sigmas"]] <- list(); r[["sigmas_res"]] <- list(); r[["blups"]] <- list()
  r[["sigmas"]] <- .rx_read_matrices(r, rd, "sigma_dims", "out_sigma_")
  r[["sigmas_res"]] <- .rx_read_matrices(r, rd, "sigma_res_dims", "out_sigmares_")
  v <- rd("out_sigma_res")
  if (!is.null(v)) r[["sigma_res"]] <- matrix(v, sqrt(length(v)), sqrt(length(v)))
  r[["loadings"]] <- list(); r[["pev"]] <- list()
  for (nm in names(r[["loadings_dims"]])) {
    v <- rd(paste0("out_loadings_", nm))
    if (is.null(v)) next
    L <- matrix(v, r[["loadings_dims"]][[nm]][1], r[["loadings_dims"]][[nm]][2])
    ps <- if (nm %in% names(r[["psi_dims"]])) rd(paste0("out_psi_", nm)) else NULL
    r[["loadings"]][[nm]] <- list(Lambda = L, psi = ps)
  }
  r[["pev"]] <- .rx_read_matrices(r, rd, "pev_dims", "out_pev_")
  r[["blups"]] <- .rx_read_matrices(r, rd, "blup_dims", "out_blup_")
  r
}

# Hessian of -2 logL in theta, which identifies the parameters behind flat
# directions or negative curvature.
.rx_read_hessian <- function(r, rd) {
  v <- rd("out_hessian")
  if (!is.null(v)) {
    q_ <- as.integer(round(sqrt(length(v))))
    if (q_ * q_ == length(v)) r[["hessian"]] <- matrix(v, q_, q_)
    else warning(sprintf("out_hessian has %d values and is not square: ignored", length(v)))
  }
  r
}

# Spatial lag: Psi (in units of W), its standard errors (delta method of the
# solver), and y* at the final point (beta and the BLUPs belong to the model
# of y*).
.rx_read_lag <- function(r, rd) {
  if (is.null(r[["lag"]])) return(r)
  q_ <- as.integer(r[["lag"]][["q"]])
  tn <- as.character(r[["lag"]][["traits"]])
  v <- rd("out_lag_psi")
  if (!is.null(v)) r[["lag"]][["Psi"]] <- matrix(v, q_, q_, dimnames = list(tn, tn))
  v <- rd("out_lag_psi_se")
  if (!is.null(v)) r[["lag"]][["Psi_se"]] <- matrix(v, q_, q_, dimnames = list(tn, tn))
  r[["ystar"]] <- rd("out_ystar")
  r
}

# Names of the columns of Sigma and of the BLUPs, and the levels of the BLUPs,
# when the terms carry them.
.rx_label_sigmas_blups <- function(r, model) {
  if (is.null(model) || is.null(model[["terms"]])) return(r)
  for (tm in model[["terms"]]) {
    cn <- tm[["colnames"]]; nm <- tm[["name"]]
    S <- r[["sigmas"]][[nm]]
    if (!is.null(S) && !is.null(cn) && length(cn) == nrow(S))
      dimnames(r[["sigmas"]][[nm]]) <- list(cn, cn)
    U <- r[["blups"]][[nm]]
    if (!is.null(U)) {
      if (!is.null(cn) && length(cn) == ncol(U)) colnames(r[["blups"]][[nm]]) <- cn
      lv <- tm[["levels"]]
      if (!is.null(lv) && length(lv) == nrow(U)) rownames(r[["blups"]][[nm]]) <- lv
    }
  }
  r
}
