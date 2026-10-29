# ==============================================================================
# remlax_scan_plus.R - LOCO et reajustement exact des tetes de liste, AUTOUR de
# rx_scan (R/remlax_scan.R), sans le modifier
# ==============================================================================
# rx_scan teste chaque SNP a V figee, avec UN modele nul dont la parente K
# contient le SNP teste et ses voisins en desequilibre de liaison. Deux defauts en
# decoulent, que ce fichier corrige :
#
#   - CONTAMINATION PROXIMALE. Le terme polygenique absorbe une partie de l'effet
#     du SNP teste, puisque K le contient. rx_scan_loco() estime un nul par
#     chromosome, avec la parente calculee SANS ce chromosome (leave one
#     chromosome out), et scanne ce chromosome sous ce nul.
#   - CONSERVATIVITE DU TEST A V FIGEE. Sur un SNP causal, figer V sous le nul
#     perd environ un facteur 6 sur la p-valeur (mesure contre rrBLUP), sans rien
#     changer sur les SNP nuls. rx_scan_exact() reajuste le modele avec le SNP en
#     effet fixe pour les SNP de tete, et rend un Wald a composantes de variance
#     reestimees. Le test de rapport de vraisemblance REML n'est pas utilisable
#     ici : deux modeles a effets fixes differents n'ont pas la meme vraisemblance
#     restreinte.
#
# LE MODELE EST CONSTRUIT PAR L'APPELANT. Les deux fonctions recoivent
# `construire(K, X_sup = NULL)`, qui rend le rx_model pour une parente K et, le
# cas echeant, des colonnes d'effets fixes ajoutees a X. C'est la seule facon de
# changer K sans deviner ou elle entre dans le modele (DGE seul, DGE et IGE dans
# un meme terme, plusieurs termes de meme parente), et de garder par
# construction le dispositif du modele nul.
# ==============================================================================

# Nombre de degres de liberte d'un test de rx_scan ("dir", "dir+ind", "ind|dir")
.rx_scan_ddl <- function(test) length(strsplit(sub("\\|.*$", "", test), "+", fixed = TRUE)[[1]])

# Incidence designee par "terme" ou "terme:a", meme regle que rx_scan
.rx_scan_incidence <- function(model, spec) {
  spec <- as.character(spec); a <- 1L
  if (grepl(":", spec, fixed = TRUE)) {
    mor <- strsplit(spec, ":", fixed = TRUE)[[1]]; spec <- mor[1]; a <- as.integer(mor[2])
  }
  noms <- vapply(model$terms, `[[`, "", "name")
  k <- match(spec, noms)
  if (is.na(k)) stop("terme '", spec, "' absent du modele (", paste(noms, collapse = ", "), ").",
                     call. = FALSE)
  tm <- model$terms[[k]]
  if (a < 1L || a > tm$t) stop("indice ", a, " hors de 1..", tm$t, " pour '", spec, "'.", call. = FALSE)
  tm$Zl[[a]]
}

# Facteur d'inflation par test, recalcule sur tous les SNP
.rx_scan_lambda <- function(res, tests) {
  vapply(tests, function(t) {
    p <- res[[paste0("p_", t)]]; p <- p[is.finite(p)]
    if (!length(p)) return(NA_real_)
    k <- .rx_scan_ddl(t)
    stats::median(stats::qchisq(p, k, lower.tail = FALSE)) / stats::qchisq(0.5, k)
  }, 0)
}


# GWAS a V figee, un modele nul par chromosome (LOCO)
#
# @param construire fonction (K, X_sup = NULL) -> rx_model.
# @param K_de fonction (doses q x p') -> parente q x q, nommee comme les niveaux.
# @param marqueurs doses genotypiques q x p, lignes dans l'ordre des niveaux du
#   terme genetique. Centrees si la famille `ind` est testee (un decalage des
#   doses se propage par les sommes de lignes de l'incidence de voisinage).
# @param carte data.frame a colonnes `snp`, `chr` (obligatoire) et `pos`.
# @param incidences,tests,maf,codage transmis a rx_scan.
# @param fit_args liste d'arguments de rx_fit (backend, optimizer, ...).
# @param theta_init depart des ajustements par chromosome, par exemple le theta
#   du nul global : les nuls LOCO en sont proches.
# @return data.frame comme rx_scan, avec la colonne `chr_loco`, et l'attribut
#   "loco" : un tableau des nuls par chromosome (logLik, convergence, theta).
rx_scan_loco <- function(construire, K_de, marqueurs, carte, incidences,
                         tests = c("dir", "ind", "dir+ind"), maf = NULL, codage = "pm1",
                         fit_args = list(), theta_init = NULL, verbose = TRUE, ...) {
  stopifnot(is.function(construire), is.function(K_de), is.matrix(marqueurs),
            is.data.frame(carte), !is.null(carte$chr), nrow(carte) == ncol(marqueurs))
  chrs <- unique(as.character(carte$chr))
  if (length(chrs) < 2L) stop("rx_scan_loco : il faut au moins deux chromosomes.", call. = FALSE)
  morceaux <- list(); nuls <- list()
  for (cc in chrs) {
    dedans <- as.character(carte$chr) == cc
    t0 <- Sys.time()
    K_c <- K_de(marqueurs[, !dedans, drop = FALSE])
    m_c <- construire(K_c)
    f_c <- do.call(rx_fit, c(list(m_c, theta_init = theta_init, verbose = FALSE), fit_args))
    r_c <- rx_scan(f_c, m_c, marqueurs[, dedans, drop = FALSE], carte = carte[dedans, , drop = FALSE],
                   incidences = incidences, tests = tests, maf = maf, codage = codage,
                   verbose = FALSE, ...)
    r_c$chr_loco <- cc
    morceaux[[cc]] <- r_c
    nuls[[cc]] <- data.frame(chr = cc, n_snp = sum(dedans), logLik = f_c$logLik,
                             converge = isTRUE(as.logical(f_c$conv_grad_rel)) ||
                               isTRUE(as.logical(f_c$conv_decrement)),
                             secondes = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    nuls[[cc]]$theta <- list(f_c$theta)
    if (verbose) message(sprintf("  rx_scan_loco : chromosome %s, %d SNP, nul logLik %.3f, %.1f s",
                                 cc, sum(dedans), f_c$logLik, nuls[[cc]]$secondes))
  }
  out <- do.call(rbind, morceaux); rownames(out) <- NULL
  attr(out, "loco") <- do.call(rbind, nuls)
  attr(out, "lambda") <- .rx_scan_lambda(out, tests)
  out
}


# Reajustement exact des SNP de tete : Wald a composantes de variance reestimees
#
# @param res sortie de rx_scan ou rx_scan_loco.
# @param test test lineaire a reajuster : "dir", "ind" ou "dir+ind" (la famille
#   bilineaire `sim` et les tests conditionnels ne sont pas pris en charge).
# @param construire,K_de,marqueurs,carte,incidences comme rx_scan_loco.
# @param top nombre de SNP de tete, classes par la p-valeur de `test` dans `res`.
# @param loco parente sans le chromosome du SNP (TRUE) ou parente complete.
# @return data.frame des SNP de tete : p a V figee, chi2 et p exacts, effets et
#   erreurs-types des colonnes ajoutees, logLik et convergence du reajustement.
rx_scan_exact <- function(res, test, construire, K_de, marqueurs, carte, incidences,
                          top = 10L, loco = TRUE, fit_args = list(), theta_init = NULL,
                          verbose = TRUE) {
  fams <- strsplit(test, "+", fixed = TRUE)[[1]]
  if (grepl("|", test, fixed = TRUE) || !all(fams %in% c("dir", "ind")))
    stop("rx_scan_exact : test lineaire attendu (\"dir\", \"ind\" ou \"dir+ind\").", call. = FALSE)
  col <- paste0("p_", test)
  if (is.null(res[[col]])) stop("colonne '", col, "' absente de `res`.", call. = FALSE)
  if (is.null(res$snp)) stop("`res` doit porter la colonne `snp`.", call. = FALSE)
  ord <- order(res[[col]])
  ord <- ord[is.finite(res[[col]][ord])][seq_len(min(top, sum(is.finite(res[[col]]))))]
  K_tout <- if (!loco) K_de(marqueurs) else NULL
  m0 <- construire(if (loco) K_de(marqueurs) else K_tout)
  Zf <- lapply(stats::setNames(fams, fams), function(f) .rx_scan_incidence(m0, incidences[[f]]))
  cache_K <- list()
  lignes <- lapply(ord, function(i) {
    snp <- res$snp[i]; j <- match(snp, carte$snp)
    if (is.na(j)) j <- match(snp, colnames(marqueurs))
    K <- if (loco) {
      cc <- as.character(carte$chr[j])
      if (is.null(cache_K[[cc]])) cache_K[[cc]] <<- K_de(marqueurs[, as.character(carte$chr) != cc, drop = FALSE])
      cache_K[[cc]]
    } else K_tout
    S <- do.call(cbind, lapply(fams, function(f) as.numeric(Zf[[f]] %*% marqueurs[, j])))
    colnames(S) <- paste0("snp_", fams)
    t0 <- Sys.time()
    f <- do.call(rx_fit, c(list(construire(K, X_sup = S), theta_init = theta_init, verbose = FALSE),
                           fit_args))
    k <- ncol(S); ix <- length(f$beta) - k + seq_len(k)
    b <- f$beta[ix]; Vb <- f$vbeta[ix, ix, drop = FALSE]
    chi2 <- as.numeric(t(b) %*% solve(Vb, b))
    out <- data.frame(snp = snp, chr = carte$chr[j], pos = if (!is.null(carte$pos)) carte$pos[j] else NA,
                      p_fige = res[[col]][i], chi2_exact = chi2,
                      p_exact = stats::pchisq(chi2, k, lower.tail = FALSE),
                      logLik = f$logLik,
                      converge = isTRUE(as.logical(f$conv_grad_rel)) || isTRUE(as.logical(f$conv_decrement)),
                      secondes = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    for (a in seq_len(k)) { out[[paste0("beta_", fams[a])]] <- b[a]; out[[paste0("se_", fams[a])]] <- sqrt(Vb[a, a]) }
    if (verbose) message(sprintf("  rx_scan_exact : %s  p fige %.2e  p exact %.2e", snp, out$p_fige, out$p_exact))
    out
  })
  out <- do.call(rbind, lignes); rownames(out) <- NULL
  attr(out, "test") <- test
  out
}
