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
  if (length(chrs) < 2L) stop("rx_scan_loco: at least two chromosomes are needed.", call. = FALSE)
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
                              converged = .rx_fit_converged(f_c),
                              seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    nulls[[cc]]$theta <- list(f_c$theta)
    if (verbose) message(sprintf("  rx_scan_loco: chromosome %s, %d SNPs, null logLik %.3f, %.1f s",
                                 cc, sum(inside), f_c$logLik, nulls[[cc]]$seconds))
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
  col <- paste0("p_", test)
  if (is.null(res[[col]])) stop("column '", col, "' is missing from `res`.", call. = FALSE)
  if (is.null(res$snp)) stop("`res` must have the column `snp`.", call. = FALSE)
  ord <- order(res[[col]])
  ord <- ord[is.finite(res[[col]][ord])][seq_len(min(top, sum(is.finite(res[[col]]))))]
  K_all <- if (!loco) K_of(markers) else NULL
  m0 <- build(if (loco) K_of(markers) else K_all)
  tr <- m0$residual$trait
  traits <- if (!is.null(tr) && nlevels(as.factor(tr)) > 1L) levels(as.factor(tr)) else ""
  trait_codes <- if (length(traits) > 1L) as.integer(as.factor(tr)) else rep(1L, m0$n)
  des <- .rx_scan_design(test, traits)
  fams <- unique(vapply(des$cols, `[[`, "", "fam"))
  lin_fams <- union(intersect(fams, c("dir", "ind")), if ("sim" %in% fams) c("dir", "ind"))
  Zf <- lapply(stats::setNames(lin_fams, lin_fams), function(f) {
    if (!f %in% names(incidences))
      stop("rx_scan_exact: the test needs the incidence '", f, "'.", call. = FALSE)
    .rx_scan_full_incidence(m0, incidences[[f]])
  })
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
    out <- .rx_refit_snp(build, K, Zf, des, trait_codes, markers[, j], fit_args, theta_init)
    out <- cbind(data.frame(snp = snp, chr = map$chr[j],
                            pos = if (!is.null(map$pos)) map$pos[j] else NA,
                            p_fixed_V = res[[col]][i]), out)
    if (verbose) message(sprintf("  rx_scan_exact: %s  p fixed V %.2e  p exact %.2e", snp, out$p_fixed_V, out$p_exact))
    out
  })
  out <- do.call(rbind, rows); rownames(out) <- NULL
  attr(out, "test") <- test
  out
}

# Model columns and tested hypothesis of a scan test, the R mirror of
# scan._design (inst/python/remlax/scan.py). cols: one list(fam, k) per
# column, k = 0 for the common column of a family (sum over traits); L: the
# matrix of the tested linear hypothesis on the column coefficients.
.rx_scan_design <- function(test, traits) {
  T <- length(traits)
  parts <- strsplit(test, "|", fixed = TRUE)[[1]]
  items <- function(x) if (is.na(x) || !nzchar(trimws(x))) list() else
    lapply(strsplit(x, "+", fixed = TRUE)[[1]], function(it) {
      it <- trimws(it)
      m <- regmatches(it, regexec("^(dir|ind|sim)(\\[([^]]*)\\])?(@(common|het))?$", it))[[1]]
      if (!length(m)) stop("test '", test, "': cannot read '", it, "'.", call. = FALSE)
      tr <- if (nzchar(m[3])) trimws(strsplit(m[4], ",", fixed = TRUE)[[1]]) else NULL
      list(fam = m[2], traits = tr, mod = if (nzchar(m[6])) m[6] else NULL)
    })
  tested <- items(parts[1]); cond <- items(parts[2])
  cols <- list(); where <- list()
  for (it in c(tested, cond)) {
    mode <- if (identical(it$mod, "common")) "common" else "all"
    key <- paste(it$fam, mode)
    if (!is.null(where[[key]])) next
    if (mode == "common") {
      where[[key]] <- length(cols) + 1L; cols <- c(cols, list(list(fam = it$fam, k = 0L)))
    } else {
      where[[key]] <- length(cols) + seq_len(T)
      cols <- c(cols, lapply(seq_len(T), function(k) list(fam = it$fam, k = k)))
    }
  }
  m <- length(cols); rows <- list(); labels <- character(0)
  unit <- function(i) { r <- numeric(m); r[i] <- 1; r }
  for (it in tested) {
    if (identical(it$mod, "common")) {
      rows <- c(rows, list(unit(where[[paste(it$fam, "common")]])))
      labels <- c(labels, paste0(it$fam, "@common"))
    } else if (identical(it$mod, "het")) {
      if (T < 2L) stop("test '", test, "': @het needs at least two traits.", call. = FALSE)
      ix <- where[[paste(it$fam, "all")]]
      for (k in 2:T) { r <- unit(ix[k]); r[ix[1]] <- -1; rows <- c(rows, list(r))
        labels <- c(labels, sprintf("%s[%s-%s]", it$fam, traits[k], traits[1])) }
    } else {
      ix <- where[[paste(it$fam, "all")]]
      ks <- if (is.null(it$traits)) seq_len(T) else vapply(it$traits, function(t) {
        k <- match(t, traits)
        if (is.na(k) && grepl("^[0-9]+$", t)) k <- as.integer(t)
        if (is.na(k) || k < 1L || k > T) stop("test '", test, "': unknown trait '", t, "'.", call. = FALSE)
        k }, 1L)
      for (k in ks) { rows <- c(rows, list(unit(ix[k])))
        labels <- c(labels, if (T == 1L) it$fam else sprintf("%s[%s]", it$fam, traits[k])) }
    }
  }
  list(cols = cols, L = do.call(rbind, rows), labels = labels)
}

# Full-height incidence of a family: the matrix of the term, or the sum of the
# per-trait incidences of a multi-trait term.
.rx_scan_full_incidence <- function(model, spec) {
  ref <- .rx_parse_term_ref(spec)
  k <- match(ref$name, vapply(model$terms, `[[`, "", "name"))
  if (is.na(k)) return(.rx_scan_incidence(model, spec))
  tm <- model$terms[[k]]
  if (!ref$indexed && .rx_scan_multitrait_term(model, tm)) return(Reduce(`+`, tm$Zl))
  .rx_scan_incidence(model, spec)
}

# Refit with the SNP columns appended to X: per-trait columns D_k s_f, or the
# common column s_f; the Wald statistic of the tested hypothesis L b = 0 with
# re-estimated variance components, the tested estimates and their standard
# errors, and the log-likelihood and convergence of the refit.
.rx_refit_snp <- function(build, K, Zf, des, trait_codes, doses, fit_args, theta_init) {
  sf <- list()
  for (f in names(Zf)) sf[[f]] <- as.numeric(Zf[[f]] %*% doses)
  if (any(vapply(des$cols, `[[`, "", "fam") == "sim")) sf$sim <- sf$dir * sf$ind
  S <- do.call(cbind, lapply(des$cols, function(cl)
    if (cl$k == 0L) sf[[cl$fam]] else sf[[cl$fam]] * (trait_codes == cl$k)))
  colnames(S) <- paste0("snp_", vapply(des$cols, function(cl) paste0(cl$fam, "_", cl$k), ""))
  t0 <- Sys.time()
  f <- do.call(rx_fit, c(list(build(K, X_sup = S), theta_init = theta_init, verbose = FALSE),
                         fit_args))
  k <- ncol(S); ix <- length(f$beta) - k + seq_len(k)
  b <- f$beta[ix]; Vb <- f$vbeta[ix, ix, drop = FALSE]
  L <- des$L; Lb <- drop(L %*% b); LVL <- L %*% Vb %*% t(L)
  chi2 <- as.numeric(t(Lb) %*% solve(LVL, Lb))
  out <- data.frame(chi2_exact = chi2,
                    df = nrow(L),
                    p_exact = stats::pchisq(chi2, nrow(L), lower.tail = FALSE),
                    logLik = f$logLik,
                    converged = .rx_fit_converged(f),
                    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
                    check.names = FALSE)
  for (a in seq_along(Lb)) {
    out[[paste0("beta_", des$labels[a])]] <- Lb[a]
    out[[paste0("se_", des$labels[a])]] <- sqrt(LVL[a, a])
  }
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
  if (is.na(k)) stop("term '", spec, "' is not in the model (", paste(term_names, collapse = ", "), ").",
                     call. = FALSE)
  tm <- model$terms[[k]]
  if (a < 1L || a > tm$t) stop("index ", a, " outside 1..", tm$t, " for '", spec, "'.", call. = FALSE)
  tm$Zl[[a]]
}

# Genomic control lambda per test, recomputed over all SNPs.
.rx_scan_lambda <- function(res, tests) {
  vapply(tests, function(t) {
    p <- res[[paste0("p_", t)]]; p <- p[is.finite(p)]
    if (!length(p)) return(NA_real_)
    dfc <- res[[paste0("df_", t)]]
    k <- if (!is.null(dfc) && any(is.finite(dfc))) dfc[is.finite(dfc)][1] else .rx_scan_df(t)
    stats::median(stats::qchisq(p, k, lower.tail = FALSE)) / stats::qchisq(0.5, k)
  }, 0)
}
