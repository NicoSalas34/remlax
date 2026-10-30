# ==============================================================================
# model.R - rx_model(): the model object handed to the solver
# ==============================================================================

rx_model <- function(y, X, terms, residual = rx_residual(), name = "model", lag = NULL,
                     weights = NULL) {
  n <- length(y)
  weights <- .rx_check_weights(weights, n)
  if (!is.matrix(X)) X <- matrix(X, nrow = n)
  if (nrow(X) != n) stop("X a ", nrow(X), " lignes pour ", n, " observations.")
  .rx_check_something_to_estimate(terms, residual, lag)
  if (!is.null(lag)) .rx_check_lag(lag, terms, n)
  terms <- .rx_check_terms(terms, n)
  .rx_check_residual(residual, n)
  t_res <- .rx_n_traits(residual)
  # X must have full column rank: otherwise log|X'V^-1X| is -Inf and REML is
  # not defined.
  r <- qr(X)$rank
  if (r < ncol(X))
    stop("X est de rang ", r, " pour ", ncol(X), " colonnes : retirer les colonnes redondantes.")
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
    stop("weights : ", n, " valeurs finies et > 0 attendues.", call. = FALSE)
  weights
}

# A model without random terms is valid when the residual is structured (the
# usual spatial model of a field trial), not when it is iid.
.rx_check_something_to_estimate <- function(terms, residual, lag) {
  if (!length(terms) && identical(residual$level, "id") &&
      identical(residual$struct, "iid") && is.null(lag))
    stop("aucun terme aleatoire et residuelle iid : il n'y a rien a estimer.")
  if (!length(terms) && identical(residual$struct, "none"))
    stop("residuelle \"none\" sans terme aleatoire : V serait nulle.", call. = FALSE)
}

.rx_check_lag <- function(lag, terms, n) {
  if (!inherits(lag, "rx_lag")) stop("`lag` doit etre un objet rx_lag (voir rx_lag()).",
                                     call. = FALSE)
  if (length(lag$trait) != n)
    stop("rx_lag : ", length(lag$trait), " lignes declarees pour ", n, " observations. ",
         "`trait` et `unit` de rx_lag() decrivent les lignes de y, dans le meme ordre.",
         call. = FALSE)
  .rx_warn_tb_iee(terms, lag)
}

# Checks the terms and names the list by term name: rx_exposure() and
# rx_ratios() find terms by model$terms[[name]]. A term given only by its
# precision Kinv cannot be fitted by the dense engine, which needs a factor of
# K; the attribute "kinv_only" tells rx_fit().
.rx_check_terms <- function(terms, n) {
  for (tm in terms) {
    if (!inherits(tm, "rx_term")) stop("terms doit contenir des objets rx_term.")
  }
  term_names <- vapply(terms, `[[`, "", "name")
  if (anyDuplicated(term_names))
    stop("termes en double : ", paste(unique(term_names[duplicated(term_names)]), collapse = ", "))
  if (length(terms)) names(terms) <- term_names
  for (tm in terms) {
    if (!is.null(tm$Kinv) && is.null(tm$LK))
      attr(terms, "kinv_only") <- TRUE
    if (nrow(tm$Zl[[1]]) != n)
      stop("terme '", tm$name, "' : incidence a ", nrow(tm$Zl[[1]]), " lignes pour ", n, " obs.")
  }
  terms
}

# Multi-trait residual structures need several traits, and the sections of a
# dsum() must cover every row exactly once.
.rx_check_residual <- function(residual, n) {
  for (s in (residual$sections %||% list(residual))) {
    if (identical(s$struct, "none")) next
    if (s$struct %in% c("diag", "us", "fa") && .rx_n_traits(s) < 2L)
      stop("structure residuelle '", s$struct, "'",
           if (!is.null(s$name)) paste0(" (section ", s$name, ")") else "",
           " sans caractere multiple : fournir `trait` a rx_residual().")
  }
  if (!is.null(residual$sections)) {
    covered <- unlist(lapply(residual$sections, `[[`, "rows"))
    if (length(covered) != n || anyDuplicated(covered))
      stop("dsum : les sections couvrent ", length(covered), " ligne(s) (",
           length(unique(covered)), " distinctes) pour ", n, " observations.")
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
  cat(sprintf("Modele REML '%s' : %d observations, %d effets fixes\n",
              x$name, x$n, ncol(x$X)))
  for (tm in x$terms)
    cat(sprintf("  terme %-14s %-5s t=%-3d q=%-5d %-10s (%d parametre%s)\n",
                tm$name, tm$struct, tm$t, tm$q,
                switch(tm$level, id = "K=I", fixed = "K fournie",
                       ar1ar1 = "AR1xAR1", toupper(tm$level)), tm$n_par,
                if (tm$n_par > 1) "s" else ""))
  if (identical(x$residual$struct, "none")) {
    cat("  residuelle     aucune (V portee par les termes)\n")
  } else
  cat(sprintf("  residuelle     %-5s t=%d  (%d parametre%s)\n", x$residual$struct, x$t_res,
              rx_n_params(x$residual$struct, x$t_res, x$residual$rank),
              if (rx_n_params(x$residual$struct, x$t_res, x$residual$rank) > 1) "s" else ""))
  if (!is.null(x$lag))
    cat(sprintf("  decalage       Psi %dx%d, %d entree(s) libre(s), critere %s\n",
                x$lag$q, x$lag$q, x$lag$n_free, x$lag$criterion))
  cat(sprintf("  total : %d parametres\n", x$n_par))
  invisible(x)
}
