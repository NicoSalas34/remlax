# ==============================================================================
# remlax_ratios.R - ratios, erreurs-types par methode delta, intervalles de
# correlation, bilan d'une grille d'AIC. POSE A COTE de R/remlax.R
# ==============================================================================
# Ce fichier n'est pas source par R/remlax.R et n'en modifie rien. Il le
# SUPPOSE deja source : il appelle rx_n_params(), rx_n_level(), rx_se_theta()
# et `%||%`, toutes existantes.
#
#   source("R/remlax.R")
#   source("R/remlax_ratios.R")
#   S   <- rx_sigmas_from_theta(fit[["theta"]], model)      # la carte theta -> Sigma
#   rat <- rx_ratios(fit, components, exposure, model = model)
#   ci  <- rx_cor_z(r, se)
#   g   <- rx_grid_summary(table, coords = c("rank_within", "rank_between"), by = "trait")
#
# CONVENTIONS, reprises du solveur et verifiees par les tests :
#   - theta est un log d'ecart-type : var = exp(2 theta) ;
#   - us : Sigma = L L', L triangulaire inferieure remplie LIGNE PAR LIGNE,
#     diagonale exp(theta), hors-diagonale theta brut ;
#   - le Hessien rendu est celui de -2 logL, donc cov(theta) = 2 H^-1 sur le
#     sous-espace libre ;
#   - les constantes d'exposition (d, k, c, S) sont des constantes du
#     dispositif : elles multiplient la quantite et son erreur-type a l'identique.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Intervalles de correlation sur l'echelle z de Fisher
# ------------------------------------------------------------------------------
# @param r correlations
# @param se erreur-type de r sur l'echelle de la correlation (methode delta) ;
#   exclusif avec n
# @param n effectif d'une correlation de Pearson ; alors se_z = 1 / sqrt(n - 3)
# @param level niveau de l'intervalle
# @param width_max largeur au-dela de laquelle l'intervalle est declare non
#   informatif ; NULL pour ne pas juger
# @param clamp borne appliquee a r avant atanh
# @return data.frame : r, se, z, z_fisher, se_z, ci_low, ci_high, width,
#   informative, method
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
# 2. La carte theta -> Sigma, cote R
# ------------------------------------------------------------------------------
# Meme parametrisation que structures.py (chol_sigma). Elle est verifiee contre
# fit$sigmas a 1e-8 avant tout jacobien : une permutation de theta echoue la.

.rx_L_us <- function(th, t) {
  L <- matrix(0, t, t); k <- 1L
  for (i in seq_len(t)) for (j in seq_len(i)) {
    L[i, j] <- if (i == j) exp(th[k]) else th[k]; k <- k + 1L
  }
  L
}
.rx_trapeze <- function(th, t, rank) {
  Lam <- matrix(0, t, rank); k <- 1L
  for (i in seq_len(t)) for (j in seq_len(min(i, rank))) { Lam[i, j] <- th[k]; k <- k + 1L }
  Lam
}
.rx_bande <- function(th, t, k_b) {
  L <- diag(t); k <- 1L
  for (i in seq_len(t)) {
    js <- if (i > 1L) seq.int(max(1L, i - k_b), i - 1L) else integer(0)
    for (j in js) { L[i, j] <- th[k]; k <- k + 1L }
  }
  L
}

# Sigma d'une structure a partir de son theta (t x t)
rx_sigma_of <- function(th, struct, t, rank = 0L) {
  switch(struct,
    iid  = exp(2 * th[1L]) * diag(t),
    diag = diag(exp(2 * th[seq_len(t)]), t),
    us   = { L <- .rx_L_us(th, t); L %*% t(L) },
    fa   = { nl <- rx_n_loadings(t, rank); Lam <- .rx_trapeze(th[seq_len(nl)], t, rank)
             psi <- exp(th[nl + seq_len(t)]); Lam %*% t(Lam) + diag(psi^2, t) },
    rr   = { Gam <- .rx_trapeze(th, t, rank); Gam %*% t(Gam) },
    chol = { nb <- sum(pmin(seq_len(t) - 1L, rank)); L <- .rx_bande(th[seq_len(nb)], t, rank)
             d <- exp(th[nb + seq_len(t)]); L %*% diag(d^2, t) %*% t(L) },
    ante = { nb <- sum(pmin(seq_len(t) - 1L, rank)); Ut <- .rx_bande(th[seq_len(nb)], t, rank)
             d <- exp(th[nb + seq_len(t)]); A <- solve(Ut); A %*% diag(1 / d^2, t) %*% t(A) },
    corh = { sd <- exp(th[seq_len(t)]); lo <- -1 / max(t - 1L, 1L)
             r <- lo + (tanh(th[t + 1L]) + 1) * 0.5 * (1 - lo)
             C <- (1 - r) * diag(t) + r * matrix(1, t, t); diag(sd, t) %*% C %*% diag(sd, t) },
    stop("rx_sigma_of : structure '", struct, "' non prise en charge.", call. = FALSE))
}

# Decoupage de theta par terme puis par section residuelle : rend une liste de
# (nom, struct, t, rank, idx_sigma, idx_level, labels) dans l'ordre du solveur.
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
  attr(out, "n_theta") <- o
  out
}

# Toutes les matrices Sigma (termes, puis sections residuelles) depuis theta
#
# @param theta vecteur theta du solveur
# @param model le rx_model ajuste
# @return liste nommee de matrices t x t avec dimnames ; les sections residuelles
#   sont nommees "residual" ou "residual:<section>"
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

# Indices de theta des elements DIAGONAUX de chaque Sigma (NA quand la
# parametrisation ne les isole pas : fa, rr, chol, ante, corh).
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

# Resout une reference de composante : "nom", "nom[i]", "nom[i,j]",
# "nom:etiquette", "nom:etiquette_a~etiquette_b". Rend list(sigma, i, j).
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
# 3. rx_ratios : la carte theta -> quantites, derivee une fois
# ------------------------------------------------------------------------------
# `components` nomme les composantes de chaque cible ; `exposure` porte les
# constantes du dispositif. La carte q(theta) est ecrite UNE fois et derivee
# numeriquement ; les erreurs-types de toutes les quantites en decoulent par
# SE = sqrt(J V J'), V = 2 H^-1 sur le sous-espace libre.

.rx_split_refs <- function(x) {
  if (is.null(x) || (length(x) == 1L && (is.na(x) || !nzchar(trimws(x))))) return(character(0))
  if (is.list(x)) x <- unlist(x)
  trimws(unlist(strsplit(as.character(x), "+", fixed = TRUE)))
}
.rx_k_other <- function(x) {
  # "ref=val+ref=val", vecteur nomme, ou liste nommee -> vecteur nomme
  if (is.null(x) || (length(x) == 1L && is.na(x))) return(numeric(0))
  if (is.character(x) && length(x) == 1L) {
    if (!nzchar(trimws(x))) return(numeric(0))
    parts <- trimws(strsplit(x, "+", fixed = TRUE)[[1]])
    kv <- strsplit(parts, "=", fixed = TRUE)
    return(stats::setNames(as.numeric(vapply(kv, `[`, "", 2L)), trimws(vapply(kv, `[`, "", 1L))))
  }
  unlist(x)
}

# Ratios, parts, heritabilites, tau2, correlations et leurs erreurs-types
#
# @param fit rx_fit avec theta, hessian, par_floor, par_ceil
# @param components data.frame, une ligne par cible : target, direct,
#   indirect_within (NA si absente), indirect_between (NA si absente), other
#   (references separees par "+", composantes additives de la variance
#   phenotypique), et facultativement group
# @param exposure data.frame, une ligne par cible : target, d, k_within,
#   k_between, c, S_within, S_between, k_other ("ref=val+ref=val") ; NULL = 1
# @param model le rx_model si fit ne le porte pas
# @param quantities sous-ensemble de variances, shares, h2, h2_ext, tau2,
#   correlations, residual_correlations, tbv
# @param scale appliquer exposure
# @param level,width_max passes a rx_cor_z pour les correlations
# @param jacobian "numeric" (differences finies centrees)
# @param curvature "project" ou "refuse" face a une valeur propre negative de H
# @param bound_tol,dep_bound tolerances pour les bornes
# @param step pas relatif des differences finies
# @return data.frame long de classe rx_ratios
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
  # --- controle theta -> Sigma contre le solveur -----------------------------
  S0 <- rx_sigmas_from_theta(theta, model)
  if (!is.null(fit[["sigmas"]])) for (nm in names(fit[["sigmas"]])) {
    A <- fit[["sigmas"]][[nm]]; B <- S0[[nm]]
    if (!is.null(A) && !is.null(B) && all(dim(A) == dim(B))) {
      e <- max(abs(as.matrix(A) - B)) / max(1, max(abs(B)))
      if (e > 1e-8)
        stop(sprintf(paste0("rx_ratios : Sigma reconstruite depuis theta differe de fit$sigmas ",
                            "pour '%s' (ecart relatif %.2e). L'ordre de theta n'est pas celui ",
                            "du modele fourni."), nm, e), call. = FALSE)
    }
  }
  # --- composantes par cible --------------------------------------------------
  components <- as.data.frame(components, stringsAsFactors = FALSE)
  req <- c("target", "direct")
  if (!all(req %in% names(components)))
    stop("rx_ratios : `components` doit porter les colonnes target et direct.", call. = FALSE)
  for (c_ in c("indirect_within", "indirect_between", "other"))
    if (!c_ %in% names(components)) components[[c_]] <- NA_character_
  nt <- nrow(components); tg <- as.character(components$target)
  if (anyDuplicated(tg)) stop("rx_ratios : cibles en double.", call. = FALSE)
  .na <- function(x) is.null(x) || (length(x) == 1L && (is.na(x) || !nzchar(x)))
  cible <- lapply(seq_len(nt), function(i) {
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
  names(cible) <- tg
  # --- expositions --------------------------------------------------------------
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
  term_of <- function(ref) ref$sigma
  # --- la carte theta -> quantites --------------------------------------------
  # les termes genetiques : ceux qui portent un effet direct ou un effet croise
  gen_terms <- unique(c(vapply(cible, function(z) z$d$sigma, ""),
                        unlist(lapply(cible, function(z) if (is.null(z$ib)) NULL else z$ib$sigma))))
  res_names <- names(lay)[vapply(lay, function(l) l$kind == "residual" && l$t > 1L, logical(1))]
  carte <- function(th) {
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
      # cibles dont l'effet direct vit dans k : les emetteurs de ce terme
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
  q0 <- carte(theta)
  nq <- length(q0)
  # --- sous-espace libre et covariance de theta -------------------------------
  fl <- fit[["par_floor"]] %||% -12; ce <- fit[["par_ceil"]] %||% 12
  at_bound <- theta <= fl + bound_tol | theta >= ce - bound_tol
  fixed <- rep(FALSE, length(theta))
  if (length(fit[["fixed_theta"]])) fixed[as.integer(fit[["fixed_theta"]])] <- TRUE
  libre <- which(!at_bound & !fixed)
  H <- fit[["hessian"]]
  V <- NULL; W_excl <- NULL; no_H <- is.null(H)
  if (!no_H && length(libre)) {
    Hf <- as.matrix(H)[libre, libre, drop = FALSE]; Hf <- (Hf + t(Hf)) / 2
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
      W_excl <- rowSums(ed$vectors[, !pos, drop = FALSE]^2)   # poids par parametre libre
    }
  }
  se_theta <- rep(NA_real_, length(theta))
  if (!is.null(V)) se_theta[libre] <- sqrt(pmax(diag(V), 0))
  check_se <- NA_real_
  if (!is.null(fit[["se_theta"]]) && any(is.finite(fit[["se_theta"]]) & is.finite(se_theta)))
    check_se <- stats::median((se_theta / fit[["se_theta"]])[is.finite(se_theta) & is.finite(fit[["se_theta"]])])
  # --- jacobien numerique, differences centrees --------------------------------
  J <- matrix(NA_real_, nq, length(theta))
  if (!is.null(V)) {
    J[] <- 0
    for (j in seq_along(theta)) {
      h <- step * max(1, abs(theta[j]))
      tp <- theta; tp[j] <- tp[j] + h; tm <- theta; tm[j] <- tm[j] - h
      J[, j] <- (carte(tp) - carte(tm)) / (2 * h)
    }
  }
  se <- rep(NA_real_, nq); dep_b <- rep(NA_real_, nq); dep_x <- rep(NA_real_, nq)
  if (!is.null(V)) {
    Jf <- J[, libre, drop = FALSE]
    Jf[is.na(Jf)] <- 0
    se <- sqrt(pmax(rowSums((Jf %*% V) * Jf), 0))
    Ja <- abs(J); Ja[is.na(Ja)] <- 0
    tot <- rowSums(Ja); tot[tot == 0] <- 1
    dep_b <- rowSums(Ja[, at_bound & !fixed, drop = FALSE]) / tot
    dep_x <- if (is.null(W_excl)) rep(0, nq)
             else rowSums(Ja[, libre, drop = FALSE][, W_excl > 0.5, drop = FALSE]) / tot
  }
  # --- table longue -------------------------------------------------------------
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
  # drapeaux
  idx_diag <- .rx_idx_diag(lay)
  flag <- rep("OK", nq)
  # FLOOR : variance brute d'une composante dont le theta diagonal est a la borne
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
  out <- data.frame(target = target, quantity = quantity, component = component,
                    estimate = est, se = se, z = est / se, ci_low = lo, ci_high = hi,
                    flag = flag, dep_bound = dep_b, dep_excluded = dep_x,
                    scaled = scaled, convention = if (scaled) "exposure" else "raw",
                    stringsAsFactors = FALSE, row.names = NULL)
  attr(out, "V") <- V; attr(out, "free") <- libre; attr(out, "se_theta") <- se_theta
  attr(out, "check_se") <- check_se; attr(out, "sigmas") <- S0; attr(out, "exposure") <- E
  class(out) <- c("rx_ratios", "data.frame")
  out
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
# 4. Bilan AIC d'une grille d'ajustements
# ------------------------------------------------------------------------------
# @param table data.frame, une ligne par ajustement
# @param coords noms des colonnes qui definissent une cellule
# @param aic,loglik,n_par noms de colonnes ; aic recalcule si absent
# @param by colonne(s) separant des grilles non comparables
# @param tol largeur de l'ensemble soutenu en unites d'AIC
# @param pd,n_at_bound,n_par_free,n_obs noms de colonnes optionnelles
# @param effective calculer aussi le bilan sous AIC_eff = 2 n_par_free - 2 logLik
# @return liste de classe rx_grid_summary : best, supported, ranges, n_supported,
#   n_product, counts, best_pd, effective, delta
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
    # cellules manquantes par rapport au produit cartesien des valeurs observees
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
