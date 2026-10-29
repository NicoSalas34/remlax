# ==============================================================================
# 3. MODELE
# ==============================================================================
rx_model <- function(y, X, terms, residual = rx_residual(), name = "modele", lag = NULL,
                     weights = NULL) {
  n <- length(y)
  # POIDS A PRIORI : R = W^-1/2 R0 W^-1/2. Avec une residuelle iid dont la
  # variance est fixee a 1 (fixed_theta), R = diag(1/w) : variances residuelles
  # connues, comme dans une analyse en deux etapes. Le solveur met les lignes a
  # l'echelle sqrt(w) et corrige la vraisemblance (fit.fit_reml).
  if (!is.null(weights)) {
    weights <- as.numeric(weights)
    if (length(weights) != n || any(!is.finite(weights)) || any(weights <= 0))
      stop("weights : ", n, " valeurs finies et > 0 attendues.", call. = FALSE)
  }
  if (!is.matrix(X)) X <- matrix(X, nrow = n)
  if (nrow(X) != n) stop("X a ", nrow(X), " lignes pour ", n, " observations.")
  # Un modele SANS terme aleatoire est licite des lors que la residuelle est
  # structuree (residual = ~ ar1(row):ar1(col)) : c'est le modele spatial le
  # plus courant en essai au champ.
  if (!length(terms) && identical(residual$level, "id") &&
      identical(residual$struct, "iid") && is.null(lag))
    stop("aucun terme aleatoire et residuelle iid : il n'y a rien a estimer.")
  if (!length(terms) && identical(residual$struct, "none"))
    stop("residuelle \"none\" sans terme aleatoire : V serait nulle.", call. = FALSE)
  if (!is.null(lag)) {
    if (!inherits(lag, "rx_lag")) stop("`lag` doit etre un objet rx_lag (voir rx_lag()).",
                                       call. = FALSE)
    if (length(lag$trait) != n)
      stop("rx_lag : ", length(lag$trait), " lignes declarees pour ", n, " observations. ",
           "`trait` et `unit` de rx_lag() decrivent les lignes de y, dans le meme ordre.",
           call. = FALSE)
    .rx_avertir_tb_iee(terms, lag)
  }
  for (tm in terms) {
    if (!inherits(tm, "rx_term")) stop("terms doit contenir des objets rx_term.")
  }
  # La liste des termes est NOMMEE par leurs noms : rx_exposure() et rx_ratios()
  # les retrouvent par model$terms[[nom]], et un nom en double serait deja une
  # collision dans les sorties du solveur.
  noms_t <- vapply(terms, `[[`, "", "name")
  if (anyDuplicated(noms_t)) stop("termes en double : ", paste(unique(noms_t[duplicated(noms_t)]), collapse = ", "))
  if (length(terms)) names(terms) <- noms_t
  for (tm in terms) {
    # Un terme declare par sa PRECISION n'est pas ajustable par le moteur dense :
    # celui-ci a besoin d'un facteur de K, et le retrouver depuis K^-1 demanderait
    # une inversion dense - exactement ce que la voie creuse evite. On le dit ici
    # plutot que de laisser le solveur echouer plus loin sur un LK manquant.
    if (!is.null(tm$Kinv) && is.null(tm$LK))
      attr(terms, "creux_seulement") <- TRUE
    if (nrow(tm$Zl[[1]]) != n)
      stop("terme '", tm$name, "' : incidence a ", nrow(tm$Zl[[1]]), " lignes pour ", n, " obs.")
  }
  .t_de <- function(r) if (is.null(r$trait)) 1L else nlevels(factor(r$trait))
  for (s in (residual$sections %||% list(residual))) {
    if (identical(s$struct, "none")) next
    if (s$struct %in% c("diag", "us", "fa") && .t_de(s) < 2L)
      stop("structure residuelle '", s$struct, "'",
           if (!is.null(s$name)) paste0(" (section ", s$name, ")") else "",
           " sans caractere multiple : fournir `trait` a rx_residual().")
  }
  if (!is.null(residual$sections)) {
    vus <- unlist(lapply(residual$sections, `[[`, "rows"))
    if (length(vus) != n || anyDuplicated(vus))
      stop("dsum : les sections couvrent ", length(vus), " ligne(s) (",
           length(unique(vus)), " distinctes) pour ", n, " observations.")
  }
  t_res <- .t_de(residual)
  # X de rang plein : sinon log|X'V^-1X| est -Inf et la REML n'est pas definie.
  r <- qr(X)$rank
  if (r < ncol(X))
    stop("X est de rang ", r, " pour ", ncol(X), " colonnes : retirer les colonnes redondantes.")
  if (!is.null(residual$n_unit)) residual$n_unit <- as.integer(residual$n_unit)
  X_assign <- attr(X, "assign"); X_termes <- attr(X, "termes")
  structure(list(name = name, y = as.numeric(y), X = X, terms = terms,
                 X_assign = X_assign, X_termes = X_termes,
                 residual = residual, n = n, t_res = t_res, lag = lag, weights = weights,
                 # La residuelle porte aussi des parametres de NIVEAUX (le rho
                 # d'un ar1 residuel), et une residuelle en sections en porte un
                 # jeu par section. Les omettre affichait 2 parametres pour un
                 # champ ar1 x ar1 qui en a 4. Les entrees libres de Psi
                 # viennent en dernier, comme dans theta.
                 n_par = sum(vapply(terms, `[[`, 1L, "n_par")) +
                   sum(vapply(residual$sections %||% list(residual), function(s)
                     if (identical(s$struct, "none")) 0L else
                     rx_n_params(s$struct, .t_de(s), s$rank) +
                       rx_n_level(s$level %||% "id", s$order %||% 0L, s$opts), 1L)) +
                   (if (is.null(lag)) 0L else lag$n_free)),
            class = "rx_model")
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
