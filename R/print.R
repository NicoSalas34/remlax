print.rx_fit <- function(x, ...) {
  if (identical(x[["depart_bayes"]], "map")) {
    # voie bayesienne sans REML : valeurs au MODE A POSTERIORI, pas d'optimum
    # REML ni de diagnostic de convergence REML a afficher
    mp <- x[["mode_posterieur"]]
    cat(sprintf("Mode a posteriori (%s), sans ajustement REML : logLik REML au mode %.6f | %d parametres | %d obs | %.1f s\n",
                x$backend, x$logLik, x$n_par, x$n_obs, (mp$secondes %||% NA)))
  } else {
  cat(sprintf("Ajustement REML (%s) : logLik %.6f | %d parametres | %d obs | %.1f s\n",
              x$backend, x$logLik, x$n_par, x$n_obs, x$secondes))
  cat(sprintf("  max|grad| %.2e | decrement de Newton %.2e | %d valeur(s) propre(s) negative(s)\n",
              x$max_grad, x$newton_decrement %||% NA, x$n_neg_eig %||% NA))
  }
  for (nm in names(x$sigmas)) {
    S <- x$sigmas[[nm]]
    cat(sprintf("  Sigma[%s] %dx%d, diagonale : %s\n", nm, nrow(S), ncol(S),
                paste(format(diag(S), digits = 4), collapse = " ")))
  }
  # Residuelle : une ligne par section. Avec dsum, n'afficher que la premiere
  # laissait croire a une residuelle unique.
  sr <- x$sigmas_res
  if (length(sr) > 1L) {
    for (nm in names(sr))
      cat(sprintf("  residuelle[%s] : %s\n", nm,
                  paste(format(diag(as.matrix(sr[[nm]])), digits = 4), collapse = " ")))
  } else if (!is.null(x[["sigma_res"]]))
    # [[ et non $ : `x$sigma_res` s'appariait partiellement sur `sigma_res_dims`
    # quand la residuelle est absente ("none"), et imprimait une ligne vide.
    cat(sprintf("  residuelle : %s\n", paste(format(diag(as.matrix(x[["sigma_res"]])), digits = 4), collapse = " ")))
  else cat("  residuelle : aucune\n")
  # Entrees non identifiees separement (meme motif dans V) : on le dit, avec
  # le nombre de parametres identifies, celui qui compte pour un LRT.
  if (isTRUE(x[["n_plat_attendu"]] > 0))
    cat(sprintf("  %d direction(s) plate(s) par construction (entrees de meme motif, seule leur somme est identifiee) : %d parametres identifies\n",
                as.integer(x[["n_plat_attendu"]]), as.integer(x[["n_par_identifies"]])))
  Lg <- x[["lag"]]
  if (!is.null(Lg) && !is.null(Lg[["Psi"]])) {
    cat(sprintf("  Psi (receveur en ligne, emetteur en colonne ; unites de W) | critere %s | log|det A| %.6f | rayon spectral de M %.4f%s\n",
                Lg[["criterion"]], Lg[["logdet_A"]], Lg[["rayon_spectral"]],
                if (isFALSE(Lg[["admissible"]])) " | HORS REGION ADMISSIBLE" else ""))
    P <- Lg[["Psi"]]; S <- Lg[["Psi_se"]]
    m <- matrix(sprintf("%10.5f", P), nrow(P))
    if (!is.null(S)) m[] <- ifelse(is.na(S), paste0(m, "          "), sprintf("%s (%7.5f)", m, S))
    dimnames(m) <- list(paste0("    ", rownames(P), " <-"), colnames(P))
    print(noquote(m))
  }
  # Correspondance V1, V2, ... des expressions vpredict, pour qu'on sache ce
  # que l'on a divise par quoi.
  cp <- x$vpredict$composantes
  if (!is.null(cp) && NROW(cp))
    cat("  composantes (vpredict) : ",
        paste(sprintf("V%d = %s", cp$i, cp$nom), collapse = ", "), "\n", sep = "")
  if (!is.null(x$vpredict) && length(x$vpredict$predictions)) {
    cat("  vpredict :\n")
    pr <- x$vpredict$predictions
    for (i in seq_len(nrow(pr)))
      cat(sprintf("    %-10s %10.5f   SE %s   [%s]\n", pr$nom[i], pr$valeur[i],
                  if (is.na(pr$se[i])) "  (n.d.)" else sprintf("%8.5f", pr$se[i]),
                  pr$expression[i]))
  }
  if (!is.null(x$wald) && length(x$wald$tests)) {
    cat("  Wald (conditionnel, chi2/ddl) :\n")
    tw <- x$wald$tests
    for (i in seq_len(nrow(tw)))
      cat(sprintf("    %-14s ddl %2d   F %9.3f   p %.4g\n",
                  tw$terme[i], tw$ddl[i], tw$F[i], tw$p[i]))
  }
  .rx_bayes_print(x)
  invisible(x)
}
