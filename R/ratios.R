# ==============================================================================
# ratios.R - ratios with delta-method standard errors, correlation intervals,
# and the AIC summary of a grid of fits
# ==============================================================================
#   S   <- rx_sigmas_from_theta(fit[["theta"]], model)      # theta -> Sigma
#   rat <- rx_ratios(fit, components, exposure, model = model)
#   ci  <- rx_cor_z(r, se)
#   g   <- rx_grid_summary(table, coords = c("rank_within", "rank_between"), by = "trait")
#
# Conventions of the solver, checked by the tests:
#   - theta is a log standard deviation: var = exp(2 theta);
#   - us: Sigma = L L', L lower triangular filled row by row, diagonal
#     exp(theta), off-diagonal raw theta;
#   - the Hessian is that of -2 logL, so cov(theta) = 2 H^-1 on the free
#     subspace;
#   - the exposure constants (d, k, c, S) are constants of the design: they
#     multiply a quantity and its standard error alike.
# ==============================================================================

# ------------------------------------------------------------------------------
# Correlation intervals on Fisher's z scale
# ------------------------------------------------------------------------------
# `se` is the standard error of r on the correlation scale (delta method), or
# `n` the size of a Pearson correlation (se_z = 1 / sqrt(n - 3)). An interval
# wider than width_max is declared uninformative (NULL: not judged); r is
# clamped before atanh.
rx_cor_z <- function(r, se = NULL, n = NULL, level = 0.95, width_max = 1.5,
                     clamp = 0.999999) {
  r <- as.numeric(r)
  if (is.null(se) == is.null(n))
    stop("rx_cor_z : fournir `se` (methode delta) OU `n` (Pearson), pas les deux ni aucun.",
         call. = FALSE)
  if (!is.null(se)) {
    se <- as.numeric(se)
    if (length(se) == 1L) se <- rep(se, length(r))
    if (length(se) != length(r)) stop("rx_cor_z : `se` n'a pas la longueur de `r`.", call. = FALSE)
    se_z <- se / pmax(1 - r^2, 1e-8)
    method <- "delta"
  } else {
    n <- as.numeric(n)
    if (length(n) == 1L) n <- rep(n, length(r))
    if (length(n) != length(r)) stop("rx_cor_z : `n` n'a pas la longueur de `r`.", call. = FALSE)
    if (any(n <= 3, na.rm = TRUE)) stop("rx_cor_z : `n` doit depasser 3.", call. = FALSE)
    se <- rep(NA_real_, length(r))
    se_z <- 1 / sqrt(n - 3)
    method <- "pearson"
  }
  rc <- pmin(pmax(r, -clamp), clamp)
  q <- stats::qnorm(1 - (1 - level) / 2)
  zf <- atanh(rc)
  lo <- tanh(zf - q * se_z); hi <- tanh(zf + q * se_z)
  w <- hi - lo
  inf <- if (is.null(width_max)) rep(NA, length(r)) else w <= width_max
  data.frame(r = r, se = se, z = if (method == "delta") r / se else NA_real_,
             z_fisher = zf, se_z = se_z, ci_low = lo, ci_high = hi, width = w,
             informative = inf, method = method, level = level,
             width_max = if (is.null(width_max)) NA_real_ else width_max,
             stringsAsFactors = FALSE)
}

# ------------------------------------------------------------------------------
# The map theta -> Sigma on the R side
# ------------------------------------------------------------------------------
# Same parametrisation as structures.py (chol_sigma). It is checked against
# fit$sigmas to 1e-8 before any Jacobian: a permuted theta fails there.

.rx_L_us <- function(th, t) {
  L <- matrix(0, t, t); k <- 1L
  for (i in seq_len(t)) for (j in seq_len(i)) {
    L[i, j] <- if (i == j) exp(th[k]) else th[k]; k <- k + 1L
  }
  L
}
# Lower trapezoidal loadings (t x rank), filled row by row.
.rx_trapezoid <- function(th, t, rank) {
  Lam <- matrix(0, t, rank); k <- 1L
  for (i in seq_len(t)) for (j in seq_len(min(i, rank))) { Lam[i, j] <- th[k]; k <- k + 1L }
  Lam
}
# Unit lower triangular matrix with k_b sub-diagonals, filled row by row.
.rx_band <- function(th, t, k_b) {
  L <- diag(t); k <- 1L
  for (i in seq_len(t)) {
    js <- if (i > 1L) seq.int(max(1L, i - k_b), i - 1L) else integer(0)
    for (j in js) { L[i, j] <- th[k]; k <- k + 1L }
  }
  L
}

# Sigma (t x t) of a structure from its theta.
rx_sigma_of <- function(th, struct, t, rank = 0L) {
  switch(struct,
    iid  = exp(2 * th[1L]) * diag(t),
    diag = diag(exp(2 * th[seq_len(t)]), t),
    us   = { L <- .rx_L_us(th, t); L %*% t(L) },
    fa   = { nl <- rx_n_loadings(t, rank); Lam <- .rx_trapezoid(th[seq_len(nl)], t, rank)
             psi <- exp(th[nl + seq_len(t)]); Lam %*% t(Lam) + diag(psi^2, t) },
    rr   = { Gam <- .rx_trapezoid(th, t, rank); Gam %*% t(Gam) },
    chol = { nb <- sum(pmin(seq_len(t) - 1L, rank)); L <- .rx_band(th[seq_len(nb)], t, rank)
             d <- exp(th[nb + seq_len(t)]); L %*% diag(d^2, t) %*% t(L) },
    ante = { nb <- sum(pmin(seq_len(t) - 1L, rank)); Ut <- .rx_band(th[seq_len(nb)], t, rank)
             d <- exp(th[nb + seq_len(t)]); A <- solve(Ut); A %*% diag(1 / d^2, t) %*% t(A) },
    corh = { sd <- exp(th[seq_len(t)]); lo <- -1 / max(t - 1L, 1L)
             r <- lo + (tanh(th[t + 1L]) + 1) * 0.5 * (1 - lo)
             C <- (1 - r) * diag(t) + r * matrix(1, t, t); diag(sd, t) %*% C %*% diag(sd, t) },
    stop("rx_sigma_of : structure '", struct, "' non prise en charge.", call. = FALSE))
}

# Layout of theta by term, then by residual section: a list of (name, struct,
# t, rank, idx_sigma, idx_level, labels) in the solver's order.
.rx_theta_layout <- function(model) {
  if (is.null(model) || is.null(model[["terms"]]))
    stop("rx_ratios : le modele est requis (rx_model, ou fit[[\"model\"]]).", call. = FALSE)
  out <- list(); o <- 0L
  for (tm in model[["terms"]]) {
    ns <- rx_n_params(tm[["struct"]], tm[["t"]], tm[["rank"]])
    nl <- rx_n_level(tm[["level"]], tm[["order"]] %||% 0L, tm[["opts"]], parts = tm[["parts"]])
    lab <- tm[["colnames"]] %||% (if (tm[["t"]] == 1L) tm[["name"]] else paste0(tm[["name"]], "[", seq_len(tm[["t"]]), "]"))
    out[[tm[["name"]]]] <- list(name = tm[["name"]], kind = "term", struct = tm[["struct"]],
                                t = tm[["t"]], rank = tm[["rank"]], idx_sigma = o + seq_len(ns),
                                idx_level = o + ns + seq_len(nl), labels = lab)
    o <- o + ns + nl
  }
  res <- model[["residual"]]
  secs <- res[["sections"]] %||% list(res)
  for (s in secs) {
    if (identical(s[["struct"]], "none")) next
    ts <- if (is.null(s[["trait"]])) 1L else nlevels(factor(s[["trait"]]))
    ns <- rx_n_params(s[["struct"]], ts, s[["rank"]] %||% 0L)
    nl <- rx_n_level(s[["level"]] %||% "id", s[["order"]] %||% 0L, s[["opts"]])
    nm <- if (is.null(s[["name"]])) "residual" else paste0("residual:", s[["name"]])
    lab <- if (ts == 1L) nm else levels(factor(s[["trait"]]))
    out[[nm]] <- list(name = nm, kind = "residual", struct = s[["struct"]], t = ts,
                      rank = s[["rank"]] %||% 0L, idx_sigma = o + seq_len(ns),
                      idx_level = o + ns + seq_len(nl), labels = lab)
    o <- o + ns + nl
  }
  # The free entries of Psi (rx_lag) end theta: not a Sigma, but part of its
  # length.
  attr(out, "n_theta") <- o + (if (is.null(model[["lag"]])) 0L else model[["lag"]][["n_free"]])
  out
}

# Every Sigma (terms, then residual sections) from theta, as a named list of
# t x t matrices with dimnames; residual sections are named "residual" or
# "residual:<section>".
rx_sigmas_from_theta <- function(theta, model) {
  lay <- .rx_theta_layout(model)
  if (length(theta) != attr(lay, "n_theta"))
    stop("rx_sigmas_from_theta : theta a ", length(theta), " valeurs, le modele en attend ",
         attr(lay, "n_theta"), ".", call. = FALSE)
  out <- list()
  for (l in lay) {
    S <- rx_sigma_of(theta[l$idx_sigma], l$struct, l$t, l$rank)
    dimnames(S) <- list(l$labels, l$labels)
    out[[l$name]] <- S
  }
  out
}

# Indices in theta of the diagonal entries of each Sigma (NA when the
# parametrisation does not isolate them: fa, rr, chol, ante, corh).
.rx_idx_diag <- function(lay) {
  lapply(lay, function(l) {
    t <- l$t
    switch(l$struct,
      iid  = rep(l$idx_sigma[1L], t),
      diag = l$idx_sigma[seq_len(t)],
      us   = l$idx_sigma[seq_len(t) * (seq_len(t) + 1L) / 2L],
      rep(NA_integer_, t))
  })
}

# Resolves a component reference: "name", "name[i]", "name[i,j]",
# "name:label", "name:label_a~label_b". Returns list(sigma, i, j).
.rx_ref <- function(ref, lay) {
  ref <- as.character(ref)
  noms <- names(lay)
  cand <- noms[vapply(noms, function(k) ref == k || startsWith(ref, paste0(k, "[")) ||
                        startsWith(ref, paste0(k, ":")), logical(1))]
  if (!length(cand)) stop("rx_ratios : composante '", ref, "' : aucun terme ni section '",
                          sub("[:\\[].*$", "", ref), "' dans le modele (connus : ",
                          paste(noms, collapse = ", "), ").", call. = FALSE)
  k <- cand[which.max(nchar(cand))]
  l <- lay[[k]]; reste <- substring(ref, nchar(k) + 1L)
  if (!nzchar(reste)) {
    if (l$t != 1L) stop("rx_ratios : '", ref, "' a t = ", l$t, " colonnes ; en nommer une.",
                        call. = FALSE)
    return(list(sigma = k, i = 1L, j = 1L))
  }
  if (grepl("^\\[\\d+(,\\d+)?\\]$", reste)) {
    ij <- as.integer(strsplit(gsub("[][]", "", reste), ",")[[1]])
    if (length(ij) == 1L) ij <- c(ij, ij)
    if (any(ij < 1L | ij > l$t)) stop("rx_ratios : '", ref, "' hors de 1..", l$t, ".", call. = FALSE)
    return(list(sigma = k, i = ij[1], j = ij[2]))
  }
  if (startsWith(reste, ":")) {
    labs <- strsplit(substring(reste, 2L), "~", fixed = TRUE)[[1]]
    if (length(labs) == 1L) labs <- c(labs, labs)
    ij <- match(labs, l$labels)
    if (anyNA(ij)) stop("rx_ratios : etiquette(s) '", paste(labs[is.na(ij)], collapse = "', '"),
                        "' absente(s) de '", k, "' (", paste(l$labels, collapse = ", "), ").",
                        call. = FALSE)
    return(list(sigma = k, i = ij[1], j = ij[2]))
  }
  stop("rx_ratios : reference '", ref, "' illisible.", call. = FALSE)
}

# ------------------------------------------------------------------------------
# rx_ratios(): the map theta -> quantities, differentiated once
# ------------------------------------------------------------------------------
# `components` names the components of each target; `exposure` carries the
# constants of the design. The map q(theta) is written once and differentiated
# numerically; the standard errors of every quantity follow from
# SE = sqrt(J V J'), V = 2 H^-1 on the free subspace.

.rx_split_refs <- function(x) {
  if (is.null(x) || (length(x) == 1L && (is.na(x) || !nzchar(trimws(x))))) return(character(0))
  if (is.list(x)) x <- unlist(x)
  trimws(unlist(strsplit(as.character(x), "+", fixed = TRUE)))
}
.rx_k_other <- function(x) {
  # "ref=val+ref=val", a named vector or a named list, to a named vector
  if (is.null(x) || (length(x) == 1L && is.na(x))) return(numeric(0))
  if (is.character(x) && length(x) == 1L) {
    if (!nzchar(trimws(x))) return(numeric(0))
    parts <- trimws(strsplit(x, "+", fixed = TRUE)[[1]])
    kv <- strsplit(parts, "=", fixed = TRUE)
    return(stats::setNames(as.numeric(vapply(kv, `[`, "", 2L)), trimws(vapply(kv, `[`, "", 1L))))
  }
  unlist(x)
}

# Ratios, shares, heritabilities, tau2, correlations and their standard
# errors. `components` has one row per target: target, direct,
# indirect_within, indirect_between (NA when absent) and other (references
# separated by "+", additive components of the phenotypic variance).
# `exposure` has one row per target: d, k_within, k_between, c, S_within,
# S_between and k_other ("ref=val+ref=val"); NULL means 1.
rx_ratios <- function(fit, components, exposure = NULL, model = NULL,
                      quantities = c("variances", "shares", "h2", "h2_ext", "tau2",
                                     "correlations", "residual_correlations", "tbv"),
                      scale = TRUE, level = 0.95, width_max = 1.5,
                      jacobian = c("numeric", "solver"), curvature = c("project", "refuse"),
                      bound_tol = 1e-7, dep_bound = 0.05, step = 1e-5) {
  jacobian <- match.arg(jacobian); curvature <- match.arg(curvature)
  if (jacobian == "solver")
    stop("rx_ratios : jacobian = \"solver\" n'est pas implemente ; utiliser \"numeric\".",
         call. = FALSE)
  quantities <- match.arg(quantities, several.ok = TRUE)
  model <- model %||% fit[["model"]]
  lay <- .rx_theta_layout(model)
  theta <- as.numeric(fit[["theta"]])
  if (length(theta) != attr(lay, "n_theta"))
    stop("rx_ratios : theta a ", length(theta), " valeurs, le modele en attend ",
         attr(lay, "n_theta"), ".", call. = FALSE)
  S0 <- rx_sigmas_from_theta(theta, model)
  .rx_check_sigmas(fit, S0)
  targets <- .rx_ratio_targets(components, lay)
  ex <- .rx_exposure_table(exposure, names(targets), scale)
  quantity_map <- .rx_ratio_map(model, lay, targets, ex$E, ex$k_other, quantities)
  q0 <- quantity_map(theta)
  nq <- length(q0)

  cv <- .rx_free_theta_cov(fit, theta, bound_tol, curvature)
  V <- cv$V; free <- cv$free; at_bound <- cv$at_bound; fixed <- cv$fixed
  se_theta <- rep(NA_real_, length(theta))
  if (!is.null(V)) se_theta[free] <- sqrt(pmax(diag(V), 0))
  check_se <- NA_real_
  if (!is.null(fit[["se_theta"]]) && any(is.finite(fit[["se_theta"]]) & is.finite(se_theta)))
    check_se <- stats::median((se_theta / fit[["se_theta"]])[is.finite(se_theta) & is.finite(fit[["se_theta"]])])

  # Standard errors SE = sqrt(J V J') with a central-difference Jacobian, and
  # the share of each quantity's sensitivity carried by parameters at a bound
  # (dep_b) or by directions excluded for negative curvature (dep_x).
  J <- if (!is.null(V)) .rx_numeric_jacobian(quantity_map, theta, step, nq) else
       matrix(NA_real_, nq, length(theta))
  se <- rep(NA_real_, nq); dep_b <- rep(NA_real_, nq); dep_x <- rep(NA_real_, nq)
  if (!is.null(V)) {
    Jf <- J[, free, drop = FALSE]
    Jf[is.na(Jf)] <- 0
    se <- sqrt(pmax(rowSums((Jf %*% V) * Jf), 0))
    Ja <- abs(J); Ja[is.na(Ja)] <- 0
    tot <- rowSums(Ja); tot[tot == 0] <- 1
    dep_b <- rowSums(Ja[, at_bound & !fixed, drop = FALSE]) / tot
    dep_x <- if (is.null(cv$W_excl)) rep(0, nq)
             else rowSums(Ja[, free, drop = FALSE][, cv$W_excl > 0.5, drop = FALSE]) / tot
  }

  # Long table: quantity|target|component names, Wald intervals, and Fisher-z
  # intervals for the correlations.
  parts <- strsplit(names(q0), "|", fixed = TRUE)
  quantity <- vapply(parts, `[`, "", 1L); target <- vapply(parts, `[`, "", 2L)
  component <- vapply(parts, function(p) paste(p[-(1:2)], collapse = "|"), "")
  est <- as.numeric(q0)
  is_cor <- quantity %in% c("cor", "rescor", "tbv_cor") |
            (quantity == "h2" & component == "r_direct_indirect")
  qn <- stats::qnorm(1 - (1 - level) / 2)
  lo <- est - qn * se; hi <- est + qn * se
  informative <- rep(NA, nq)
  if (any(is_cor)) {
    cz <- rx_cor_z(est[is_cor], se = ifelse(is.na(se[is_cor]), NA_real_, se[is_cor]),
                   level = level, width_max = width_max)
    lo[is_cor] <- cz$ci_low; hi[is_cor] <- cz$ci_high; informative[is_cor] <- cz$informative
  }
  flag <- .rx_ratio_flags(quantity, target, component, est, targets, lay, at_bound,
                          cv$no_H, V, dep_x, dep_b, dep_bound, is_cor, informative)
  scaled <- ex$scaled
  out <- data.frame(target = target, quantity = quantity, component = component,
                    estimate = est, se = se, z = est / se, ci_low = lo, ci_high = hi,
                    flag = flag, dep_bound = dep_b, dep_excluded = dep_x,
                    scaled = scaled, convention = if (scaled) "exposure" else "raw",
                    stringsAsFactors = FALSE, row.names = NULL)
  attr(out, "V") <- V; attr(out, "free") <- free; attr(out, "se_theta") <- se_theta
  attr(out, "check_se") <- check_se; attr(out, "sigmas") <- S0; attr(out, "exposure") <- ex$E
  class(out) <- c("rx_ratios", "data.frame")
  out
}

# Sigma rebuilt from theta must equal the solver's: a permuted theta fails here.
.rx_check_sigmas <- function(fit, S0) {
  if (is.null(fit[["sigmas"]])) return(invisible())
  for (nm in names(fit[["sigmas"]])) {
    A <- fit[["sigmas"]][[nm]]; B <- S0[[nm]]
    if (!is.null(A) && !is.null(B) && all(dim(A) == dim(B))) {
      e <- max(abs(as.matrix(A) - B)) / max(1, max(abs(B)))
      if (e > 1e-8)
        stop(sprintf(paste0("rx_ratios : Sigma reconstruite depuis theta differe de fit$sigmas ",
                            "pour '%s' (ecart relatif %.2e). L'ordre de theta n'est pas celui ",
                            "du modele fourni."), nm, e), call. = FALSE)
    }
  }
}

# The components of each target, resolved in the theta layout: direct,
# indirect_within (in the same term as direct), indirect_between and other.
.rx_ratio_targets <- function(components, lay) {
  components <- as.data.frame(components, stringsAsFactors = FALSE)
  req <- c("target", "direct")
  if (!all(req %in% names(components)))
    stop("rx_ratios : `components` doit porter les colonnes target et direct.", call. = FALSE)
  for (c_ in c("indirect_within", "indirect_between", "other"))
    if (!c_ %in% names(components)) components[[c_]] <- NA_character_
  nt <- nrow(components); tg <- as.character(components$target)
  if (anyDuplicated(tg)) stop("rx_ratios : cibles en double.", call. = FALSE)
  .na <- function(x) is.null(x) || (length(x) == 1L && (is.na(x) || !nzchar(x)))
  targets <- lapply(seq_len(nt), function(i) {
    d <- .rx_ref(components$direct[i], lay)
    iw <- if (.na(components$indirect_within[i])) NULL else .rx_ref(components$indirect_within[i], lay)
    ib <- if (.na(components$indirect_between[i])) NULL else .rx_ref(components$indirect_between[i], lay)
    if (!is.null(iw) && iw$sigma != d$sigma)
      stop("rx_ratios : cible '", tg[i], "' : direct et indirect_within doivent vivre dans le ",
           "meme terme (covariance directe-indirecte).", call. = FALSE)
    oth <- lapply(.rx_split_refs(components$other[i]), .rx_ref, lay = lay)
    names(oth) <- .rx_split_refs(components$other[i])
    list(target = tg[i], d = d, iw = iw, ib = ib, other = oth)
  })
  names(targets) <- tg
  targets
}

# Exposure constants of each target (1 when absent or when scale = FALSE).
.rx_exposure_table <- function(exposure, tg, scale) {
  nt <- length(tg)
  scaled <- scale && !is.null(exposure)
  E <- data.frame(target = tg, d = 1, k_within = 1, k_between = 1, c = 1, S_within = 1,
                  S_between = 1, stringsAsFactors = FALSE)
  k_other <- stats::setNames(vector("list", nt), tg)
  if (!is.null(exposure)) {
    exposure <- as.data.frame(exposure, stringsAsFactors = FALSE)
    if (!"target" %in% names(exposure)) stop("rx_ratios : `exposure` sans colonne target.", call. = FALSE)
    m <- match(tg, as.character(exposure$target))
    if (anyNA(m)) stop("rx_ratios : cible(s) sans exposition : ",
                       paste(tg[is.na(m)], collapse = ", "), ".", call. = FALSE)
    for (c_ in c("d", "k_within", "k_between", "c", "S_within", "S_between"))
      if (c_ %in% names(exposure)) {
        v <- as.numeric(exposure[[c_]][m]); v[is.na(v)] <- 1; E[[c_]] <- v
      }
    if ("k_other" %in% names(exposure))
      for (i in seq_len(nt)) k_other[[i]] <- .rx_k_other(exposure[["k_other"]][[m[i]]])
  }
  if (!scaled) { E[, -1] <- 1; k_other <- stats::setNames(vector("list", nt), tg) }
  list(E = E, k_other = k_other, scaled = scaled)
}

# The map theta -> quantities, written once: variances, shares, heritabilities,
# correlations of the genetic terms (those carrying a direct or a cross
# effect) and of the multi-trait residuals, and the variances, covariances and
# correlations of total breeding values (tbv) and tau2.
.rx_ratio_map <- function(model, lay, targets, E, k_other, quantities) {
  cible <- targets; tg <- names(targets); nt <- length(tg)
  gen_terms <- unique(c(vapply(cible, function(z) z$d$sigma, ""),
                        unlist(lapply(cible, function(z) if (is.null(z$ib)) NULL else z$ib$sigma))))
  res_names <- names(lay)[vapply(lay, function(l) l$kind == "residual" && l$t > 1L, logical(1))]
  function(th) {
    S <- rx_sigmas_from_theta(th, model)
    g <- function(r) S[[r$sigma]][r$i, r$j]
    q <- c()
    VP <- stats::setNames(numeric(nt), tg)
    for (i in seq_len(nt)) {
      z <- cible[[i]]; e <- E[i, ]
      s2D <- g(z$d); VD <- e$d * s2D
      s2W <- if (is.null(z$iw)) 0 else g(z$iw); VW <- e$k_within * s2W
      s2B <- if (is.null(z$ib)) 0 else g(z$ib); VB <- e$k_between * s2B
      cDI <- if (is.null(z$iw)) 0 else S[[z$d$sigma]][z$d$i, z$iw$i]
      C <- e$c * cDI
      Vo <- vapply(names(z$other), function(nm) {
        k <- k_other[[i]][nm]; if (is.null(k) || is.na(k)) k <- 1
        as.numeric(k) * g(z$other[[nm]]) }, numeric(1))
      VS <- VD + VW + VB + sum(Vo); VPi <- VS + 2 * C
      VP[i] <- VPi
      tgt <- z$target
      if ("variances" %in% quantities) {
        q[paste("var", tgt, "direct", sep = "|")] <- VD
        if (!is.null(z$iw)) q[paste("var", tgt, "indirect_within", sep = "|")] <- VW
        if (!is.null(z$ib)) q[paste("var", tgt, "indirect_between", sep = "|")] <- VB
        for (nm in names(z$other)) q[paste("var", tgt, nm, sep = "|")] <- Vo[[nm]]
        if (!is.null(z$iw)) q[paste("var", tgt, "cov_direct_indirect", sep = "|")] <- C
        q[paste("var", tgt, "sum", sep = "|")] <- VS
        q[paste("var", tgt, "phenotypic", sep = "|")] <- VPi
      }
      if ("shares" %in% quantities) {
        q[paste("share", tgt, "direct", sep = "|")] <- VD / VS
        if (!is.null(z$iw)) q[paste("share", tgt, "indirect_within", sep = "|")] <- VW / VS
        if (!is.null(z$ib)) q[paste("share", tgt, "indirect_between", sep = "|")] <- VB / VS
        for (nm in names(z$other)) q[paste("share", tgt, nm, sep = "|")] <- Vo[[nm]] / VS
      }
      if ("h2" %in% quantities) {
        q[paste("h2", tgt, "h2", sep = "|")] <- VD / VPi
        if (!is.null(z$iw)) q[paste("h2", tgt, "h2_indirect_within", sep = "|")] <- VW / VPi
        if (!is.null(z$ib)) q[paste("h2", tgt, "h2_indirect_between", sep = "|")] <- VB / VPi
      }
      if ("h2_ext" %in% quantities) {
        hw <- (VD + 2 * C + VW) / VPi
        q[paste("h2", tgt, "h2_ext_within", sep = "|")] <- hw
        q[paste("h2", tgt, "h2_ext_total", sep = "|")] <- hw + VB / VPi
        if (!is.null(z$iw))
          q[paste("h2", tgt, "r_direct_indirect", sep = "|")] <-
            if (isTRUE(s2D > 0 && s2W > 0)) cDI / sqrt(s2D * s2W) else NA_real_
      }
    }
    if ("correlations" %in% quantities) for (k in gen_terms) {
      M <- S[[k]]; t <- nrow(M); dd <- sqrt(diag(M)); lab <- lay[[k]]$labels
      if (t > 1L) for (a in seq_len(t - 1L)) for (b in (a + 1L):t)
        q[paste("cor", k, paste0(lab[a], "~", lab[b]), sep = "|")] <-
          if (isTRUE(dd[a] > 0 && dd[b] > 0)) M[a, b] / (dd[a] * dd[b]) else NA_real_
    }
    if ("residual_correlations" %in% quantities) for (k in res_names) {
      M <- S[[k]]; t <- nrow(M); dd <- sqrt(diag(M)); lab <- lay[[k]]$labels
      for (a in seq_len(t - 1L)) for (b in (a + 1L):t)
        q[paste("rescor", k, paste0(lab[a], "~", lab[b]), sep = "|")] <-
          if (isTRUE(dd[a] > 0 && dd[b] > 0)) M[a, b] / (dd[a] * dd[b]) else NA_real_
    }
    if (any(c("tbv", "tau2") %in% quantities)) for (k in gen_terms) {
      M <- S[[k]]; t <- nrow(M)
      # targets whose direct effect lives in k: the emitters of this term
      em <- which(vapply(cible, function(z) z$d$sigma == k, logical(1)))
      d_em <- mean(E$d[em])
      vecs <- list(); recv <- c()
      for (i in em) {
        z <- cible[[i]]; a <- numeric(t)
        a[z$d$i] <- sqrt(E$d[i])
        if (!is.null(z$iw)) a[z$iw$i] <- sqrt(E$d[i]) * E$S_within[i]
        vecs[[paste0("own:", z$target)]] <- a; recv <- c(recv, z$target)
      }
      for (i in seq_len(nt)) {
        z <- cible[[i]]
        if (!is.null(z$ib) && z$ib$sigma == k) {
          a <- numeric(t); a[z$ib$i] <- sqrt(d_em) * E$S_between[i]
          vecs[[paste0("cross:", z$target)]] <- a; recv <- c(recv, z$target)
        }
      }
      nmv <- names(vecs)
      for (a in seq_along(nmv)) {
        va <- as.numeric(t(vecs[[a]]) %*% M %*% vecs[[a]])
        if ("tbv" %in% quantities) q[paste("tbv_var", k, nmv[a], sep = "|")] <- va
        if ("tau2" %in% quantities) q[paste("tau2", k, nmv[a], sep = "|")] <- va / VP[[recv[a]]]
      }
      if ("tbv" %in% quantities && length(nmv) > 1L)
        for (a in seq_len(length(nmv) - 1L)) for (b in (a + 1L):length(nmv)) {
          va <- as.numeric(t(vecs[[a]]) %*% M %*% vecs[[a]])
          vb <- as.numeric(t(vecs[[b]]) %*% M %*% vecs[[b]])
          cv <- as.numeric(t(vecs[[a]]) %*% M %*% vecs[[b]])
          q[paste("tbv_cov", k, paste0(nmv[a], "~", nmv[b]), sep = "|")] <- cv
          q[paste("tbv_cor", k, paste0(nmv[a], "~", nmv[b]), sep = "|")] <-
            if (isTRUE(va > 0 && vb > 0)) cv / sqrt(va * vb) else NA_real_
        }
    }
    q
  }
}

# Free subspace (neither at a bound nor fixed) and covariance of theta,
# V = 2 H^-1 on it. With a negative eigenvalue, "refuse" gives no V and
# "project" projects on the positive curvature, with the weight of each free
# parameter in the excluded directions (W_excl).
.rx_free_theta_cov <- function(fit, theta, bound_tol, curvature) {
  fl <- fit[["par_floor"]] %||% -12; ce <- fit[["par_ceil"]] %||% 12
  at_bound <- theta <= fl + bound_tol | theta >= ce - bound_tol
  fixed <- rep(FALSE, length(theta))
  if (length(fit[["fixed_theta"]])) fixed[as.integer(fit[["fixed_theta"]])] <- TRUE
  free <- which(!at_bound & !fixed)
  H <- fit[["hessian"]]
  V <- NULL; W_excl <- NULL; no_H <- is.null(H)
  if (!no_H && length(free)) {
    Hf <- as.matrix(H)[free, free, drop = FALSE]; Hf <- (Hf + t(Hf)) / 2
    ed <- eigen(Hf, symmetric = TRUE)
    pos <- ed$values > 0
    if (all(pos)) {
      V <- 2 * solve(Hf)
    } else if (curvature == "refuse") {
      V <- NULL
    } else {
      message(sprintf("rx_ratios : courbure negative, %d direction(s) sur %d ecartee(s) ; ",
                      sum(!pos), length(pos)), "V est projetee sur la courbure positive.")
      Vi <- ed$vectors[, pos, drop = FALSE] %*% diag(1 / ed$values[pos], sum(pos)) %*%
            t(ed$vectors[, pos, drop = FALSE])
      V <- 2 * Vi
      W_excl <- rowSums(ed$vectors[, !pos, drop = FALSE]^2)
    }
  }
  list(V = V, W_excl = W_excl, no_H = no_H, free = free, at_bound = at_bound, fixed = fixed)
}

# Jacobian of the map by central differences, with a step relative to theta.
.rx_numeric_jacobian <- function(quantity_map, theta, step, nq) {
  J <- matrix(0, nq, length(theta))
  for (j in seq_along(theta)) {
    h <- step * max(1, abs(theta[j]))
    tp <- theta; tp[j] <- tp[j] + h; tm <- theta; tm[j] <- tm[j] - h
    J[, j] <- (quantity_map(tp) - quantity_map(tm)) / (2 * h)
  }
  J
}

# Flag of each quantity: FLOOR (variance whose diagonal theta is at the bound),
# NOT_ESTIMATED, NO_HESSIAN, NOT_IDENTIFIED (carried by excluded directions),
# COND_BOUND (sensitive to parameters at a bound), NOT_ESTIMABLE (correlation
# with an uninformative interval), else OK.
.rx_ratio_flags <- function(quantity, target, component, est, cible, lay, at_bound,
                            no_H, V, dep_x, dep_b, dep_bound, is_cor, informative) {
  nq <- length(est)
  idx_diag <- .rx_idx_diag(lay)
  flag <- rep("OK", nq)
  for (r in seq_len(nq)) if (quantity[r] == "var" &&
                             !component[r] %in% c("cov_direct_indirect", "sum", "phenotypic")) {
    z <- cible[[target[r]]]
    ref <- switch(component[r], direct = z$d, indirect_within = z$iw, indirect_between = z$ib,
                  z$other[[component[r]]])
    if (is.null(ref)) next
    jd <- idx_diag[[ref$sigma]][ref$i]
    if (!is.na(jd) && at_bound[jd]) flag[r] <- "FLOOR"
    else if (is.finite(est[r]) && est[r] == 0) flag[r] <- "NOT_ESTIMATED"
  }
  if (no_H || is.null(V)) flag[flag == "OK"] <- "NO_HESSIAN"
  else {
    flag[flag == "OK" & is.finite(dep_x) & dep_x > 0.5] <- "NOT_IDENTIFIED"
    flag[flag == "OK" & is.finite(dep_b) & dep_b > dep_bound] <- "COND_BOUND"
    flag[flag == "OK" & is_cor & !is.na(informative) & !informative] <- "NOT_ESTIMABLE"
  }
  flag[!is.finite(est)] <- "NOT_ESTIMATED"
  flag
}

print.rx_ratios <- function(x, ...) {
  cat(sprintf("rx_ratios : %d quantite(s), %d cible(s), %s ; sous-espace libre %d/%d ; drapeaux : %s\n",
              nrow(x), length(unique(x$target)), if (isTRUE(x$scaled[1])) "mises a l'echelle" else "brutes",
              length(attr(x, "free")), length(attr(x, "se_theta")),
              paste(names(table(x$flag)), table(x$flag), sep = "=", collapse = ", ")))
  if (is.finite(attr(x, "check_se") %||% NA))
    cat(sprintf("  controle se_theta / fit$se_theta (mediane) : %.6f\n", attr(x, "check_se")))
  print.data.frame(utils::head(as.data.frame(x)[, c("target", "quantity", "component", "estimate", "se", "flag")], 12), ...)
  if (nrow(x) > 12) cat("  ...\n")
  invisible(x)
}

# ------------------------------------------------------------------------------
# AIC summary of a grid of fits
# ------------------------------------------------------------------------------
# `table` has one row per fit; `coords` are the columns that define a cell.
# The AIC is recomputed from logLik and n_par when absent. `by` separates grids
# that cannot be compared; `tol` is the width of the supported set in AIC
# units. With effective = TRUE, the summary is also computed under
# AIC_eff = 2 n_par_free - 2 logLik. Returns an rx_grid_summary list: best,
# supported, ranges, n_supported, n_product, counts, best_pd, effective, delta.
rx_grid_summary <- function(table, coords, aic = "AIC", loglik = "logLik", n_par = "n_par",
                            by = NULL, tol = 2, pd = "pd_hessian", n_at_bound = "n_at_bound",
                            n_par_free = "n_par_free", n_obs = "n_obs", effective = FALSE) {
  tb <- as.data.frame(table, stringsAsFactors = FALSE)
  if (!all(coords %in% names(tb)))
    stop("rx_grid_summary : coordonnee(s) absente(s) : ",
         paste(setdiff(coords, names(tb)), collapse = ", "), ".", call. = FALSE)
  if (!is.null(by) && !all(by %in% names(tb)))
    stop("rx_grid_summary : colonne(s) `by` absente(s).", call. = FALSE)
  if (!aic %in% names(tb)) {
    if (!all(c(loglik, n_par) %in% names(tb)))
      stop("rx_grid_summary : ni `", aic, "` ni (`", loglik, "`, `", n_par, "`) dans la table.",
           call. = FALSE)
    tb[[aic]] <- 2 * tb[[n_par]] - 2 * tb[[loglik]]
  } else if (all(c(loglik, n_par) %in% names(tb))) {
    e <- max(abs(tb[[aic]] - (2 * tb[[n_par]] - 2 * tb[[loglik]])), na.rm = TRUE)
    if (is.finite(e) && e > 1e-6)
      stop(sprintf("rx_grid_summary : `%s` fourni differe de 2 %s - 2 %s (ecart max %.3g).",
                   aic, n_par, loglik, e), call. = FALSE)
  }
  if (n_obs %in% names(tb) && is.null(by) && length(unique(tb[[n_obs]])) > 1L)
    stop("rx_grid_summary : `", n_obs, "` varie dans la table ; des grilles d'effectifs ",
         "differents ne se comparent pas, fournir `by`.", call. = FALSE)
  grp <- if (is.null(by)) rep("all", nrow(tb)) else do.call(paste, c(tb[by], sep = "|"))
  tb$.group <- grp
  bilan <- function(sub, col_aic) {
    if (n_obs %in% names(sub) && length(unique(sub[[n_obs]])) > 1L)
      stop("rx_grid_summary : `", n_obs, "` varie a l'interieur du groupe '", sub$.group[1],
           "'.", call. = FALSE)
    a <- sub[[col_aic]]
    ok <- is.finite(a)
    if (!any(ok)) stop("rx_grid_summary : aucun AIC fini dans le groupe '", sub$.group[1], "'.",
                       call. = FALSE)
    amin <- min(a[ok])
    sub$delta_aic <- a - amin
    sub$supported <- is.finite(sub$delta_aic) & sub$delta_aic <= tol
    ib <- which(a == amin)[1]
    best <- sub[ib, c(".group", coords, col_aic, intersect(c(loglik, n_par, pd, n_at_bound, n_par_free), names(sub))), drop = FALSE]
    sup <- sub[sub$supported, , drop = FALSE]
    rg <- do.call(rbind, lapply(coords, function(cc) {
      v <- sup[[cc]]
      data.frame(.group = sub$.group[1], coord = cc, min = min(v), max = max(v),
                 n_distinct = length(unique(v)), constant = length(unique(v)) == 1L,
                 stringsAsFactors = FALSE)
    }))
    # Cells missing from the Cartesian product of the observed values.
    n_cart <- prod(vapply(coords, function(cc) length(unique(sub[[cc]])), numeric(1)))
    cle <- do.call(paste, c(sub[coords], sep = "|"))
    counts <- data.frame(.group = sub$.group[1], n_cells = nrow(sub),
                         n_duplicated = sum(duplicated(cle)),
                         n_missing = n_cart - length(unique(cle)),
                         n_pd_false = if (pd %in% names(sub)) sum(!as.logical(sub[[pd]]), na.rm = TRUE) else NA_integer_,
                         n_at_bound_pos = if (n_at_bound %in% names(sub)) sum(sub[[n_at_bound]] > 0, na.rm = TRUE) else NA_integer_,
                         stringsAsFactors = FALSE)
    best_pd <- NULL
    if (pd %in% names(sub)) {
      okpd <- ok & as.logical(sub[[pd]]) %in% TRUE
      if (any(okpd)) {
        ip <- which(okpd)[which.min(a[okpd])]
        best_pd <- sub[ip, c(".group", coords, col_aic), drop = FALSE]
        best_pd$delta_pd <- a[ip] - amin
        best_pd$same_as_best <- ip == ib
      }
    }
    list(best = best, supported = sup, ranges = rg,
         n_supported = data.frame(.group = sub$.group[1], n_supported = nrow(sup),
                                  n_product = prod(rg$n_distinct), stringsAsFactors = FALSE),
         counts = counts, best_pd = best_pd, delta = sub)
  }
  grs <- split(tb, tb$.group)
  parts <- lapply(grs, bilan, col_aic = aic)
  rb <- function(nm) do.call(rbind, lapply(parts, `[[`, nm))
  out <- list(best = rb("best"), supported = rb("supported"), ranges = rb("ranges"),
              n_supported = rb("n_supported"), counts = rb("counts"),
              best_pd = rb("best_pd"), delta = rb("delta"), tol = tol, coords = coords, by = by)
  rownames(out$best) <- NULL; rownames(out$ranges) <- NULL; rownames(out$counts) <- NULL
  rownames(out$n_supported) <- NULL; if (!is.null(out$best_pd)) rownames(out$best_pd) <- NULL
  if (effective) {
    if (!all(c(n_par_free, loglik) %in% names(tb)))
      stop("rx_grid_summary : effective = TRUE exige `", n_par_free, "` et `", loglik, "`.",
           call. = FALSE)
    tb$AIC_eff <- 2 * tb[[n_par_free]] - 2 * tb[[loglik]]
    pe <- lapply(split(tb, tb$.group), bilan, col_aic = "AIC_eff")
    eff <- list(best = do.call(rbind, lapply(pe, `[[`, "best")),
                supported = do.call(rbind, lapply(pe, `[[`, "supported")),
                ranges = do.call(rbind, lapply(pe, `[[`, "ranges")),
                n_supported = do.call(rbind, lapply(pe, `[[`, "n_supported")))
    rownames(eff$best) <- NULL
    cle <- function(b) do.call(paste, c(b[coords], sep = "|"))
    eff$best_moved <- data.frame(.group = out$best$.group,
                                 best_moved = cle(out$best) != cle(eff$best)[match(out$best$.group, eff$best$.group)],
                                 stringsAsFactors = FALSE)
    out$effective <- eff
  }
  if (is.null(by)) for (nm in c("best", "ranges", "n_supported", "counts", "best_pd"))
    if (!is.null(out[[nm]])) out[[nm]]$.group <- NULL
  class(out) <- "rx_grid_summary"
  out
}

print.rx_grid_summary <- function(x, ...) {
  cat(sprintf("rx_grid_summary : %d groupe(s), coordonnees %s, tol %g\n",
              nrow(x$best), paste(x$coords, collapse = ", "), x$tol))
  b <- x$best
  b$n_supported <- x$n_supported$n_supported
  print.data.frame(b, row.names = FALSE, ...)
  invisible(x)
}
