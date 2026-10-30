# ==============================================================================
# model.R - rx_model(): the model object handed to the solver
# ==============================================================================

rx_model <- function(y, X, terms, residual = rx_residual(), name = "model", lag = NULL,
                     weights = NULL) {
  n <- length(y)
  weights <- .rx_check_weights(weights, n)
  if (!is.matrix(X)) X <- matrix(X, nrow = n)
  if (nrow(X) != n) stop("X has ", nrow(X), " rows for ", n, " observations.")
  .rx_check_something_to_estimate(terms, residual, lag)
  if (!is.null(lag)) .rx_check_lag(lag, terms, n)
  terms <- .rx_check_terms(terms, n)
  .rx_check_residual(residual, n)
  t_res <- .rx_n_traits(residual)
  # X must have full column rank: otherwise log|X'V^-1X| is -Inf and REML is
  # not defined.
  r <- qr(X)$rank
  if (r < ncol(X))
    stop("X has rank ", r, " for ", ncol(X), " columns. Remove the redundant columns.")
  if (!is.null(residual$n_unit)) residual$n_unit <- as.integer(residual$n_unit)
  structure(list(name = name, y = as.numeric(y), X = X, terms = terms,
                 X_assign = attr(X, "assign"), X_terms = attr(X, "term_labels"),
                 residual = residual, n = n, t_res = t_res, lag = lag, weights = weights,
                 n_par = .rx_model_n_par(terms, residual, lag)),
            class = "rx_model")
}

# Prior weights: R = W^-1/2 R0 W^-1/2. With an iid residual whose variance is
# fixed at 1 (fixed_theta), R = diag(1/w): known residual variances, as in a
# two-stage analysis. The solver scales the rows by sqrt(w) and corrects the
# likelihood.
.rx_check_weights <- function(weights, n) {
  if (is.null(weights)) return(NULL)
  weights <- as.numeric(weights)
  if (length(weights) != n || any(!is.finite(weights)) || any(weights <= 0))
    stop("weights: ", n, " finite values > 0 expected.", call. = FALSE)
  weights
}

# A model without random terms is valid when the residual is structured (the
# usual spatial model of a field trial), not when it is iid.
.rx_check_something_to_estimate <- function(terms, residual, lag) {
  if (!length(terms) && identical(residual$level, "id") &&
      identical(residual$struct, "iid") && is.null(lag))
    stop("no random term and an iid residual: there is nothing to estimate.")
  if (!length(terms) && identical(residual$struct, "none"))
    stop("residual \"none\" without a random term: V would be zero.", call. = FALSE)
}

.rx_check_lag <- function(lag, terms, n) {
  if (!inherits(lag, "rx_lag")) stop("`lag` must be an rx_lag object (see rx_lag()).",
                                     call. = FALSE)
  if (length(lag$trait) != n)
    stop("rx_lag: ", length(lag$trait), " rows declared for ", n, " observations. ",
         "`trait` and `unit` of rx_lag() describe the rows of y, in the same order.",
         call. = FALSE)
  .rx_warn_tb_iee(terms, lag)
}

# Checks the terms and names the list by term name: rx_exposure() and
# rx_ratios() find terms by model$terms[[name]]. A term given only by its
# precision Kinv cannot be fitted by the dense engine, which needs a factor of
# K; the attribute "kinv_only" tells rx_fit().
.rx_check_terms <- function(terms, n) {
  for (tm in terms) {
    if (!inherits(tm, "rx_term")) stop("terms must contain rx_term objects.")
  }
  term_names <- vapply(terms, `[[`, "", "name")
  if (anyDuplicated(term_names))
    stop("duplicated terms: ", paste(unique(term_names[duplicated(term_names)]), collapse = ", "))
  if (length(terms)) names(terms) <- term_names
  for (tm in terms) {
    if (!is.null(tm$Kinv) && is.null(tm$LK))
      attr(terms, "kinv_only") <- TRUE
    if (nrow(tm$Zl[[1]]) != n)
      stop("term '", tm$name, "': the incidence has ", nrow(tm$Zl[[1]]), " rows for ", n, " obs.")
  }
  terms
}

# Multi-trait residual structures need several traits, and the sections of a
# dsum() must cover every row exactly once.
.rx_check_residual <- function(residual, n) {
  for (s in (residual$sections %||% list(residual))) {
    if (identical(s$struct, "none")) next
    if (s$struct %in% c("diag", "us", "fa") && .rx_n_traits(s) < 2L)
      stop("residual structure '", s$struct, "'",
           if (!is.null(s$name)) paste0(" (section ", s$name, ")") else "",
           " without several traits: give `trait` to rx_residual().")
  }
  if (!is.null(residual$sections)) {
    covered <- unlist(lapply(residual$sections, `[[`, "rows"))
    if (length(covered) != n || anyDuplicated(covered))
      stop("dsum: the sections cover ", length(covered), " row(s) (",
           length(unique(covered)), " distinct) for ", n, " observations.")
  }
}

# Number of parameters: the terms, then the Sigma and between-level parameters
# of each residual section, then the free entries of Psi, as in theta.
.rx_model_n_par <- function(terms, residual, lag) {
  sum(vapply(terms, `[[`, 1L, "n_par")) +
    sum(vapply(residual$sections %||% list(residual), function(s)
      if (identical(s$struct, "none")) 0L else
      rx_n_params(s$struct, .rx_n_traits(s), s$rank) +
        rx_n_level(s$level %||% "id", s$order %||% 0L, s$opts), 1L)) +
    (if (is.null(lag)) 0L else lag$n_free)
}

print.rx_model <- function(x, ...) {
  cat(sprintf("REML model '%s': %d observations, %d fixed effects\n",
              x$name, x$n, ncol(x$X)))
  for (tm in x$terms)
    cat(sprintf("  term  %-14s %-5s t=%-3d q=%-5d %-10s (%d parameter%s)\n",
                tm$name, tm$struct, tm$t, tm$q,
                switch(tm$level, id = "K=I", fixed = "K given",
                       ar1ar1 = "AR1xAR1", toupper(tm$level)), tm$n_par,
                if (tm$n_par > 1) "s" else ""))
  if (identical(x$residual$struct, "none")) {
    cat("  residual       none (V carried by the terms)\n")
  } else
  cat(sprintf("  residual       %-5s t=%d  (%d parameter%s)\n", x$residual$struct, x$t_res,
              rx_n_params(x$residual$struct, x$t_res, x$residual$rank),
              if (rx_n_params(x$residual$struct, x$t_res, x$residual$rank) > 1) "s" else ""))
  if (!is.null(x$lag))
    cat(sprintf("  lag            Psi %dx%d, %d free entry(ies), criterion %s\n",
                x$lag$q, x$lag$q, x$lag$n_free, x$lag$criterion))
  cat(sprintf("  total: %d parameters\n", x$n_par))
  invisible(x)
}
