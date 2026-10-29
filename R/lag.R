# ==============================================================================
# DECALAGE SPATIAL (spatial lag) : LE MODELE STRUCTUREL
# ==============================================================================
#
#     y = M y + X b + sum_k Z_k u_k + e,      M[(t,i),(u,j)] = Psi[t,u] W_t[i,j]
#
# y est en format LONG : chaque ligne porte un caractere t et une unite i (la
# plante). Psi[t, u] est l'effet du caractere EMETTEUR u d'une voisine sur le
# caractere RECEVEUR t ; W_t (N x N, diagonale nulle) est commune a tous les
# caracteres ou propre a chaque caractere receveur. Avec A = I - M inversible,
# y* = A y suit le modele ordinaire du paquet : tout terme, toute residuelle
# s'appliquent sans changement a y*, et seul l'objectif change (le jacobien
# log|det A|, et pour la REML exacte de y le terme de moyenne ; voir
# inst/python/remlax/lag.py et vignette("structural")).
#
# Ce fichier ne fait que DECRIRE le decalage et le verifier ; le calcul est
# dans le moteur (lag.py), qui recoit W en triplets.
# ==============================================================================

# Une matrice q x q (receveur en ligne, emetteur en colonne) alignee sur les
# niveaux de caractere. Les dimnames, quand il y en a, decident de l'ordre :
# une matrice fournie dans un autre ordre que les niveaux serait sinon lue
# transposee ou permutee sans erreur.
.rx_qxq <- function(x, tn, quoi, defaut = 0) {
  q <- length(tn)
  if (is.null(x)) return(matrix(defaut, q, q, dimnames = list(tn, tn)))
  if (length(x) == 1L && is.null(dim(x))) return(matrix(x, q, q, dimnames = list(tn, tn)))
  x <- as.matrix(x)
  if (!identical(dim(x), c(q, q)))
    stop("rx_lag : `", quoi, "` doit etre une matrice ", q, " x ", q,
         " (receveur en ligne, emetteur en colonne) ; recu ", nrow(x), " x ", ncol(x), ".",
         call. = FALSE)
  if (!is.null(rownames(x)) && !is.null(colnames(x))) {
    if (!setequal(rownames(x), tn) || !setequal(colnames(x), tn))
      stop("rx_lag : les dimnames de `", quoi, "` ne sont pas les caracteres (",
           paste(tn, collapse = ", "), ").", call. = FALSE)
    x <- x[tn, tn, drop = FALSE]
  }
  dimnames(x) <- list(tn, tn)
  x
}

# Une W : creuse, generale, numerique, carree, diagonale nulle.
.rx_W <- function(w, k) {
  if (!inherits(w, "Matrix")) w <- Matrix::Matrix(as.matrix(w), sparse = TRUE)
  w <- methods::as(methods::as(w * 1, "CsparseMatrix"), "generalMatrix")
  if (nrow(w) != ncol(w))
    stop("rx_lag : W", if (!is.null(k)) paste0("[", k, "]"), " n'est pas carree (",
         nrow(w), " x ", ncol(w), ").", call. = FALSE)
  if (any(Matrix::diag(w) != 0))
    stop("rx_lag : W", if (!is.null(k)) paste0("[", k, "]"), " a une diagonale non nulle. ",
         "Une plante n'est pas sa propre voisine ; un effet sur soi-meme ne se distingue ",
         "pas des effets fixes et de la residuelle.", call. = FALSE)
  if (any(!is.finite(w@x)))
    stop("rx_lag : W contient des valeurs non finies.", call. = FALSE)
  w
}

rx_lag <- function(W, trait, unit, struct = c("full", "diag", "mask"), mask = NULL,
                   init = NULL, value = NULL, emitters = NULL, receivers = NULL,
                   criterion = c("reml_y", "jacobian", "ml")) {
  struct <- match.arg(struct)
  criterion <- match.arg(criterion)
  trait <- factor(trait)
  tn <- levels(trait); q <- length(tn); n <- length(trait)
  if (anyNA(trait)) stop("rx_lag : `trait` contient des NA.", call. = FALSE)

  # --- W : une matrice commune, ou une par caractere RECEVEUR ---------------
  Wl <- if (is.list(W) && !inherits(W, "Matrix")) W else list(W)
  if (!length(Wl) %in% c(1L, q))
    stop("rx_lag : ", length(Wl), " matrice(s) W pour ", q, " caractere(s) ; donner une W ",
         "commune ou une par caractere receveur.", call. = FALSE)
  if (length(Wl) == q && q > 1L) {
    if (is.null(names(Wl)))
      stop("rx_lag : une liste de W par caractere doit etre NOMMEE par les caracteres (",
           paste(tn, collapse = ", "), ") : l'ordre de la liste n'est pas une convention ",
           "que l'on puisse verifier.", call. = FALSE)
    if (!setequal(names(Wl), tn))
      stop("rx_lag : noms de la liste W (", paste(names(Wl), collapse = ", "),
           ") differents des caracteres (", paste(tn, collapse = ", "), ").", call. = FALSE)
    Wl <- Wl[tn]
  }
  Wl <- lapply(seq_along(Wl), function(k) .rx_W(Wl[[k]], if (length(Wl) > 1L) names(Wl)[k] else NULL))
  N <- nrow(Wl[[1]])
  if (any(vapply(Wl, nrow, 1L) != N))
    stop("rx_lag : les matrices W n'ont pas toutes ", N, " lignes.", call. = FALSE)

  # --- unite : indice de ligne de W -----------------------------------------
  # Numerique : position 1..N. Sinon : etiquette, appariee aux rownames de W.
  # Un facteur n'est PAS converti en ses codes : l'ordre de ses niveaux n'a
  # aucune raison d'etre celui des lignes de W.
  if (length(unit) != n)
    stop("rx_lag : `unit` a ", length(unit), " elements pour ", n, " lignes.", call. = FALSE)
  if (is.numeric(unit)) {
    if (any(!is.finite(unit)) || any(unit != round(unit)) || any(unit < 1) || any(unit > N))
      stop("rx_lag : `unit` numerique doit donner des indices de ligne de W entiers dans 1..",
           N, ".", call. = FALSE)
    idx <- as.integer(unit)
  } else {
    rn <- rownames(Wl[[1]])
    if (is.null(rn))
      stop("rx_lag : `unit` n'est pas numerique et W n'a pas de rownames ; donner l'indice ",
           "de ligne de W de chaque observation, ou nommer les lignes de W.", call. = FALSE)
    idx <- match(as.character(unit), rn)
    if (anyNA(idx))
      stop("rx_lag : ", sum(is.na(idx)), " unite(s) absente(s) des rownames de W (ex. ",
           paste(utils::head(unique(as.character(unit)[is.na(idx)]), 3), collapse = ", "),
           ").", call. = FALSE)
  }
  cle <- (as.integer(trait) - 1L) * N + idx
  if (anyDuplicated(cle))
    stop("rx_lag : ", sum(duplicated(cle)), " couple(s) (caractere, unite) en double ; M n'est ",
         "definie que pour une observation par caractere et par unite.", call. = FALSE)

  # --- entrees libres de Psi --------------------------------------------------
  msk <- switch(struct,
                full = matrix(TRUE, q, q, dimnames = list(tn, tn)),
                diag = { m <- matrix(FALSE, q, q, dimnames = list(tn, tn)); diag(m) <- TRUE; m },
                mask = {
                  if (is.null(mask)) stop("rx_lag : struct = \"mask\" exige `mask`.", call. = FALSE)
                  m <- .rx_qxq(mask, tn, "mask")
                  storage.mode(m) <- "logical"
                  if (anyNA(m)) stop("rx_lag : `mask` contient des NA.", call. = FALSE)
                  m
                })
  if (!is.null(receivers)) {
    if (!all(receivers %in% tn))
      stop("rx_lag : receveur(s) inconnu(s) : ", paste(setdiff(receivers, tn), collapse = ", "),
           call. = FALSE)
    msk[!tn %in% receivers, ] <- FALSE
  }
  if (!is.null(emitters)) {
    if (!all(emitters %in% tn))
      stop("rx_lag : emetteur(s) inconnu(s) : ", paste(setdiff(emitters, tn), collapse = ", "),
           call. = FALSE)
    msk[, !tn %in% emitters] <- FALSE
  }
  val <- .rx_qxq(value, tn, "value")
  if (any(!is.finite(val))) stop("rx_lag : `value` doit etre finie.", call. = FALSE)
  val[msk] <- 0
  ini <- .rx_qxq(init, tn, "init")
  if (any(!is.finite(ini[msk]))) stop("rx_lag : `init` doit etre finie sur les entrees libres.",
                                      call. = FALSE)
  ini[!msk] <- 0
  # Ordre des parametres libres dans theta : RECEVEUR PAR RECEVEUR (ligne par
  # ligne), celui de numpy.nonzero cote moteur. which() de R parcourt en
  # colonnes : on ne s'en sert donc pas pour nommer.
  ij <- which(t(msk), arr.ind = TRUE)
  noms <- if (nrow(ij)) sprintf("Psi[%s<-%s]", tn[ij[, 2]], tn[ij[, 1]]) else character()
  structure(list(W = Wl, trait = trait, traits = tn, unit = idx, N = as.integer(N),
                 q = as.integer(q), struct = struct, mask = msk, value = val, init = ini,
                 criterion = criterion, n_free = as.integer(sum(msk)), param_names = noms),
            class = "rx_lag")
}

# COVARIABLES DE BORDURE. Avec des moyennes de caracteres non nulles, le modele
# structurel donne E y = (I - M)^-1 X b : sans boucle, E y_(t,i) = mu_t +
# sum_u Psi[t,u] mu_u (W_t 1)_i, une moyenne qui depend du nombre de voisines de
# la plante i (bordures, manquants). Les criteres structurels (reml_y, jacobian,
# ml) en tiennent compte, mais un modele sans decalage (DGE/IGE, VP+) ne l'a pas :
# il faut lui donner W 1 comme covariable, par caractere. La meme covariable,
# ajoutee au modele structurel, absorbe aussi un effet de bordure qui n'est pas
# une interaction (lumiere en rive) : Psi n'est alors plus lu dans le motif de
# bordure de la moyenne (IGE_trait_paths_sim, revision_moyennes_bordures.md).
rx_border <- function(W, trait, unit, order = 1L, scale = TRUE) {
  order <- as.integer(order)
  if (length(order) != 1L || is.na(order) || order < 1L)
    stop("rx_border : `order` doit etre un entier >= 1.", call. = FALSE)
  trait <- factor(trait)
  tn <- levels(trait); q <- length(tn)
  # rx_lag fait toutes les verifications (W, appariement des unites, doublons)
  L <- rx_lag(W, trait, unit, struct = "mask", mask = matrix(FALSE, q, q, dimnames = list(tn, tn)))
  N <- L$N
  out <- matrix(0, length(trait), q * order)
  nm <- character(q * order)
  for (k in seq_len(q)) {
    Wk <- L$W[[if (length(L$W) == 1L) 1L else k]]
    s <- if (scale) mean(Matrix::rowSums(abs(Wk))) else 1
    if (!(s > 0)) s <- 1
    rows <- which(as.integer(L$trait) == k)
    v <- rep(1, N)
    for (o in seq_len(order)) {
      v <- as.numeric(Wk %*% v) / s
      col <- (o - 1L) * q + k
      out[rows, col] <- v[L$unit[rows]]
      nm[col] <- sprintf("border%d:%s", o, tn[k])
    }
  }
  colnames(out) <- nm
  out
}

print.rx_lag <- function(x, ...) {
  cat(sprintf("Decalage spatial : %d caractere(s), %d unite(s), %s, %d entree(s) libre(s) de Psi, critere %s\n",
              x$q, x$N, if (length(x$W) == 1L) "une W commune" else "une W par caractere receveur",
              x$n_free, x$criterion))
  m <- ifelse(x$mask, "libre", format(x$value, digits = 3))
  dimnames(m) <- list(paste0(x$traits, " <-"), x$traits)
  print(noquote(m))
  invisible(x)
}

# TB+ AVEC IEE* ET PSI RECIPROQUE : AVERTIR. Avec un terme de plante qui porte
# une incidence W placee sur un caractere (IEE*), le niveau plante du modele
# structurel est VP+ dans d'autres coordonnees : avec Psi[t,u] ET Psi[u,t]
# libres, il herite des q(q-1)/2 directions faibles de VP+ (theorie_v2.md, T1b)
# - qui transmet a qui n'est identifie qu'au second ordre, et l'IEE* libre
# l'absorbe. Mesure (theorie_v2.md, etude A) : liberer la reciproque augmente
# l'erreur-type de psi_21 de 8 a 87 % a psi_v = -2, la multiplie par 6,4 a
# psi_v = -0,5, ou rend l'information singuliere. On avertit, on n'interdit
# pas : sans IEE*, TB reste identifie avec Psi pleine (par le niveau genetique).
.rx_warn_tb_iee <- function(terms, lag) {
  m <- lag$mask
  recip <- which(m & t(m) & row(m) != col(m), arr.ind = TRUE)
  if (!nrow(recip)) return(invisible(FALSE))
  porte_W <- function(Zc) {
    if (ncol(Zc) != lag$N || nrow(Zc) != length(lag$unit)) return(FALSE)
    nz <- which(Matrix::rowSums(abs(Zc)) > 0)
    if (!length(nz)) return(FALSE)
    tr <- unique(as.integer(lag$trait)[nz])
    if (length(tr) != 1L) return(FALSE)
    rows <- which(as.integer(lag$trait) == tr)
    Wt <- lag$W[[if (length(lag$W) == 1L) 1L else tr]]
    isTRUE(max(abs(Zc[rows, , drop = FALSE] - Wt[lag$unit[rows], , drop = FALSE])) < 1e-12)
  }
  iee <- vapply(terms, function(tm) any(vapply(tm$Zl, porte_W, logical(1))), logical(1))
  if (!any(iee)) return(invisible(FALSE))
  paires <- unique(apply(recip, 1, function(r) paste(sort(lag$traits[r]), collapse = "/")))
  noms <- vapply(terms, function(tm) tm$name %||% "?", "")
  warning("rx_model : terme(s) ", paste(noms[iee], collapse = ", "),
          " avec une incidence W (IEE*) et Psi libre dans les deux sens pour ",
          paste(paires, collapse = ", "), ". Le sens de la transmission n'y est que faiblement ",
          "identifie (directions faibles de VP+) : laisser libre au plus une de Psi[t,u], ",
          "Psi[u,t] (rx_lag(struct = \"mask\") ou receivers/emitters).", call. = FALSE)
  invisible(TRUE)
}
