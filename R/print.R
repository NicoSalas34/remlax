# ==============================================================================
# print.R - print method of an rx_fit object
# ==============================================================================

print.rx_fit <- function(x, ...) {
  .rx_print_header(x)
  .rx_print_sigmas(x)
  .rx_print_flat_directions(x)
  .rx_print_lag(x)
  .rx_print_vpredict(x)
  .rx_print_wald(x)
  .rx_bayes_print(x)
  invisible(x)
}

# First line: the REML fit and its convergence diagnostics or, for the
# Bayesian route without REML, the values at the posterior mode.
.rx_print_header <- function(x) {
  if (identical(x[["bayes_start"]], "map")) {
    mp <- x[["posterior_mode"]]
    cat(sprintf("Mode a posteriori (%s), sans ajustement REML : logLik REML au mode %.6f | %d parametres | %d obs | %.1f s\n",
                x$backend, x$logLik, x$n_par, x$n_obs, (mp$seconds %||% NA)))
    return(invisible())
  }
  cat(sprintf("Ajustement REML (%s) : logLik %.6f | %d parametres | %d obs | %.1f s\n",
              x$backend, x$logLik, x$n_par, x$n_obs, x$seconds))
  cat(sprintf("  max|grad| %.2e | decrement de Newton %.2e | %d valeur(s) propre(s) negative(s)\n",
              x$max_grad, x$newton_decrement %||% NA, x$n_neg_eig %||% NA))
}

# Diagonal of each Sigma, then the residual: one line per dsum() section.
.rx_print_sigmas <- function(x) {
  for (nm in names(x$sigmas)) {
    S <- x$sigmas[[nm]]
    cat(sprintf("  Sigma[%s] %dx%d, diagonale : %s\n", nm, nrow(S), ncol(S),
                paste(format(diag(S), digits = 4), collapse = " ")))
  }
  sr <- x$sigmas_res
  if (length(sr) > 1L) {
    for (nm in names(sr))
      cat(sprintf("  residuelle[%s] : %s\n", nm,
                  paste(format(diag(as.matrix(sr[[nm]])), digits = 4), collapse = " ")))
  } else if (!is.null(x[["sigma_res"]]))
    cat(sprintf("  residuelle : %s\n", paste(format(diag(as.matrix(x[["sigma_res"]])), digits = 4), collapse = " ")))
  else cat("  residuelle : aucune\n")
}

# Entries with the same pattern in V are identified only through their sum;
# the number of identified parameters is the one that counts for a LRT.
.rx_print_flat_directions <- function(x) {
  if (isTRUE(x[["n_flat_expected"]] > 0))
    cat(sprintf("  %d direction(s) plate(s) par construction (entrees de meme motif, seule leur somme est identifiee) : %d parametres identifies\n",
                as.integer(x[["n_flat_expected"]]), as.integer(x[["n_par_identified"]])))
}

# Psi of a spatial lag, receiver in rows, with its standard errors.
.rx_print_lag <- function(x) {
  Lg <- x[["lag"]]
  if (is.null(Lg) || is.null(Lg[["Psi"]])) return(invisible())
  cat(sprintf("  Psi (receveur en ligne, emetteur en colonne ; unites de W) | critere %s | log|det A| %.6f | rayon spectral de M %.4f%s\n",
              Lg[["criterion"]], Lg[["logdet_A"]], Lg[["spectral_radius"]],
              if (isFALSE(Lg[["admissible"]])) " | HORS REGION ADMISSIBLE" else ""))
  P <- Lg[["Psi"]]; S <- Lg[["Psi_se"]]
  m <- matrix(sprintf("%10.5f", P), nrow(P))
  if (!is.null(S)) m[] <- ifelse(is.na(S), paste0(m, "          "), sprintf("%s (%7.5f)", m, S))
  dimnames(m) <- list(paste0("    ", rownames(P), " <-"), colnames(P))
  print(noquote(m))
}

# The components V1, V2, ... of the vpredict expressions, then their values.
.rx_print_vpredict <- function(x) {
  cp <- x$vpredict$components
  if (!is.null(cp) && NROW(cp))
    cat("  composantes (vpredict) : ",
        paste(sprintf("V%d = %s", cp$i, cp$name), collapse = ", "), "\n", sep = "")
  if (!is.null(x$vpredict) && length(x$vpredict$predictions)) {
    cat("  vpredict :\n")
    pr <- x$vpredict$predictions
    for (i in seq_len(nrow(pr)))
      cat(sprintf("    %-10s %10.5f   SE %s   [%s]\n", pr$name[i], pr$value[i],
                  if (is.na(pr$se[i])) "  (n.d.)" else sprintf("%8.5f", pr$se[i]),
                  pr$expression[i]))
  }
}

.rx_print_wald <- function(x) {
  if (is.null(x$wald) || !length(x$wald$tests)) return(invisible())
  cat("  Wald (conditionnel, chi2/ddl) :\n")
  tw <- x$wald$tests
  for (i in seq_len(nrow(tw)))
    cat(sprintf("    %-14s ddl %2d   F %9.3f   p %.4g\n",
                tw$term[i], tw$df[i], tw$F[i], tw$p[i]))
}
