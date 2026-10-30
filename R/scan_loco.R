# ==============================================================================
# scan_loco.R - one null per chromosome, and exact refit of the top SNPs
# ==============================================================================
# rx_scan() tests each SNP at fixed V under one null model whose relationship
# matrix K contains the tested SNP and its neighbours in linkage
# disequilibrium. Two consequences are corrected here:
#
#   - The polygenic term absorbs part of the effect of the tested SNP.
#     rx_scan_loco() fits one null per chromosome, with K computed without that
#     chromosome (leave one chromosome out), and scans the chromosome under it.
#   - At fixed V, the p-value of a causal SNP is about six times too large.
#     rx_scan_exact() refits the model with the SNP as a fixed effect for the
#     top SNPs and returns a Wald test with re-estimated variance components.
#     A REML likelihood ratio test cannot be used: two models with different
#     fixed effects do not share the same restricted likelihood.
#
# The caller builds the model: both functions receive build(K, X_sup =
# NULL), which returns the rx_model for a relationship matrix K and, when
# given, extra fixed columns X_sup. K can then change without guessing where it
# enters the model, and the design of the null is kept by construction.
# ==============================================================================

rx_scan_loco <- function(build, K_of, markers, map, incidences,
                         tests = c("dir", "ind", "dir+ind"), maf = NULL, coding = "pm1",
                         fit_args = list(), theta_init = NULL, verbose = TRUE, ...) {
  dots <- .rx_renamed_args(environment(), list(...),
                           c(construire = "build", K_de = "K_of", marqueurs = "markers",
                             carte = "map", codage = "coding"),
                           "rx_scan_loco", forward = TRUE)
  stopifnot(is.function(build), is.function(K_of), is.matrix(markers),
            is.data.frame(map), !is.null(map$chr), nrow(map) == ncol(markers))
  chrs <- unique(as.character(map$chr))
  if (length(chrs) < 2L) stop("rx_scan_loco : il faut au moins deux chromosomes.", call. = FALSE)
  pieces <- list(); nulls <- list()
  for (cc in chrs) {
    inside <- as.character(map$chr) == cc
    t0 <- Sys.time()
    K_c <- K_of(markers[, !inside, drop = FALSE])
    m_c <- build(K_c)
    f_c <- do.call(rx_fit, c(list(m_c, theta_init = theta_init, verbose = FALSE), fit_args))
    r_c <- do.call(rx_scan, c(list(f_c, m_c, markers[, inside, drop = FALSE],
                                   map = map[inside, , drop = FALSE],
                                   incidences = incidences, tests = tests, maf = maf,
                                   coding = coding, verbose = FALSE), dots))
    r_c$chr_loco <- cc
    pieces[[cc]] <- r_c
    nulls[[cc]] <- data.frame(chr = cc, n_snp = sum(inside), logLik = f_c$logLik,
                              converge = .rx_fit_converged(f_c),
                              secondes = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    nulls[[cc]]$theta <- list(f_c$theta)
    if (verbose) message(sprintf("  rx_scan_loco : chromosome %s, %d SNP, nul logLik %.3f, %.1f s",
                                 cc, sum(inside), f_c$logLik, nulls[[cc]]$secondes))
  }
  out <- do.call(rbind, pieces); rownames(out) <- NULL
  attr(out, "loco") <- do.call(rbind, nulls)
  attr(out, "lambda") <- .rx_scan_lambda(out, tests)
  out
}

rx_scan_exact <- function(res, test, build, K_of, markers, map, incidences,
                          top = 10L, loco = TRUE, fit_args = list(), theta_init = NULL,
                          verbose = TRUE, ...) {
  .rx_renamed_args(environment(), list(...),
                   c(construire = "build", K_de = "K_of", marqueurs = "markers", carte = "map"),
                   "rx_scan_exact")
  fams <- strsplit(test, "+", fixed = TRUE)[[1]]
  if (grepl("|", test, fixed = TRUE) || !all(fams %in% c("dir", "ind")))
    stop("rx_scan_exact : test lineaire attendu (\"dir\", \"ind\" ou \"dir+ind\").", call. = FALSE)
  col <- paste0("p_", test)
  if (is.null(res[[col]])) stop("colonne '", col, "' absente de `res`.", call. = FALSE)
  if (is.null(res$snp)) stop("`res` doit porter la colonne `snp`.", call. = FALSE)
  ord <- order(res[[col]])
  ord <- ord[is.finite(res[[col]][ord])][seq_len(min(top, sum(is.finite(res[[col]]))))]
  K_all <- if (!loco) K_of(markers) else NULL
  m0 <- build(if (loco) K_of(markers) else K_all)
  Zf <- lapply(stats::setNames(fams, fams), function(f) .rx_scan_incidence(m0, incidences[[f]]))
  # K without a chromosome, computed once per chromosome.
  K_cache <- new.env(parent = emptyenv())
  K_for <- function(cc) {
    if (is.null(K_cache[[cc]]))
      K_cache[[cc]] <- K_of(markers[, as.character(map$chr) != cc, drop = FALSE])
    K_cache[[cc]]
  }
  rows <- lapply(ord, function(i) {
    snp <- res$snp[i]; j <- match(snp, map$snp)
    if (is.na(j)) j <- match(snp, colnames(markers))
    K <- if (loco) K_for(as.character(map$chr[j])) else K_all
    out <- .rx_refit_snp(build, K, Zf, fams, markers[, j], fit_args, theta_init)
    out <- cbind(data.frame(snp = snp, chr = map$chr[j],
                            pos = if (!is.null(map$pos)) map$pos[j] else NA,
                            p_fige = res[[col]][i]), out)
    if (verbose) message(sprintf("  rx_scan_exact : %s  p fige %.2e  p exact %.2e", snp, out$p_fige, out$p_exact))
    out
  })
  out <- do.call(rbind, rows); rownames(out) <- NULL
  attr(out, "test") <- test
  out
}

# Refit with the SNP columns (its doses through each incidence) appended to X:
# the Wald statistic and p-value of those columns, their estimates and
# standard errors, and the log-likelihood and convergence of the refit.
.rx_refit_snp <- function(build, K, Zf, fams, doses, fit_args, theta_init) {
  S <- do.call(cbind, lapply(fams, function(f) as.numeric(Zf[[f]] %*% doses)))
  colnames(S) <- paste0("snp_", fams)
  t0 <- Sys.time()
  f <- do.call(rx_fit, c(list(build(K, X_sup = S), theta_init = theta_init, verbose = FALSE),
                         fit_args))
  k <- ncol(S); ix <- length(f$beta) - k + seq_len(k)
  b <- f$beta[ix]; Vb <- f$vbeta[ix, ix, drop = FALSE]
  chi2 <- as.numeric(t(b) %*% solve(Vb, b))
  out <- data.frame(chi2_exact = chi2,
                    p_exact = stats::pchisq(chi2, k, lower.tail = FALSE),
                    logLik = f$logLik,
                    converge = .rx_fit_converged(f),
                    secondes = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  for (a in seq_len(k)) { out[[paste0("beta_", fams[a])]] <- b[a]; out[[paste0("se_", fams[a])]] <- sqrt(Vb[a, a]) }
  out
}

# Did a fit converge (relative gradient or Newton decrement criterion)?
.rx_fit_converged <- function(f) {
  isTRUE(as.logical(f$conv_grad_rel)) || isTRUE(as.logical(f$conv_decrement))
}

# Degrees of freedom of a scan test ("dir", "dir+ind", "ind|dir").
.rx_scan_df <- function(test) length(strsplit(sub("\\|.*$", "", test), "+", fixed = TRUE)[[1]])

# Incidence named by "term" or "term:a" (no index required when t > 1).
.rx_scan_incidence <- function(model, spec) {
  ref <- .rx_parse_term_ref(spec)
  spec <- ref$name; a <- ref$index
  term_names <- vapply(model$terms, `[[`, "", "name")
  k <- match(spec, term_names)
  if (is.na(k)) stop("terme '", spec, "' absent du modele (", paste(term_names, collapse = ", "), ").",
                     call. = FALSE)
  tm <- model$terms[[k]]
  if (a < 1L || a > tm$t) stop("indice ", a, " hors de 1..", tm$t, " pour '", spec, "'.", call. = FALSE)
  tm$Zl[[a]]
}

# Genomic control lambda per test, recomputed over all SNPs.
.rx_scan_lambda <- function(res, tests) {
  vapply(tests, function(t) {
    p <- res[[paste0("p_", t)]]; p <- p[is.finite(p)]
    if (!length(p)) return(NA_real_)
    k <- .rx_scan_df(t)
    stats::median(stats::qchisq(p, k, lower.tail = FALSE)) / stats::qchisq(0.5, k)
  }, 0)
}
