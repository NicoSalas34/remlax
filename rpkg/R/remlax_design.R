# ------------------------------------------------------------------------------
# FICHIER GENERE par rpkg/tools/sync_sources.R depuis R/remlax_design.R.
# Ne pas editer ici : editer le script vivant, puis relancer la synchronisation.
# Les transformations appliquees sont enumerees en tete de sync_sources.R.
# ------------------------------------------------------------------------------

# ==============================================================================
# remlax_design.R - incidences de voisinage et fonctionnelles d'exposition,
# POSE A COTE de R/remlax.R
# ==============================================================================
# Ce fichier n'est pas source par R/remlax.R et n'en modifie rien. Il le
# SUPPOSE deja source (il lit des objets rx_model / rx_term par [[ ]]) mais
# n'appelle aucune de ses fonctions : tout ici est du R pur avec Matrix.
#
#   source("R/remlax.R")
#   source("R/remlax_design.R")
#   nb <- rx_neighbourhood(coord = d[, c("row", "col")], group = d$species,
#                          block = d$plot, level = d$genotype, id = d$id,
#                          rank = list("A<-A" = 5, "A<-B" = 7), reach = 0,
#                          dilution = 0, spacing = c(5, 5))
#   ex <- rx_exposure(Z_direct, nb$level[["A<-A"]][rows, ], K = K_A)
#
# CE QUE CES FONCTIONS NE SAVENT PAS. Elles ne connaissent ni espece, ni
# caractere, ni effet genetique indirect. Elles connaissent des UNITES posees
# sur une grille, des GROUPES d'unites, des NIVEAUX auxquels les poids sont
# sommes, et des paires (groupe recevant, groupe emettant). C'est l'appelant
# qui dit que le groupe est une espece et le niveau un genotype.
#
# CONVENTIONS NUMERIQUES, fixees ici et verifiees par les tests :
#   - la fenetre est prise en INDICES de grille (|d_ligne| <= r et
#     |d_colonne| <= r, l'unite elle-meme exclue) ; le poids est pris sur la
#     distance PHYSIQUE delta = sqrt((s1 d_ligne)^2 + (s2 d_colonne)^2) ;
#   - noyau "power" : w = delta^(-lambda), donc w = 1 partout a lambda = 0 ;
#   - la dilution divise chaque ligne par n_i^d ou n_i est le NOMBRE de
#     voisins de l'unite dans le groupe emettant, pas la somme des poids ;
#   - l'ordre est dilution PUIS normalisation L2, l'inverse annulerait la
#     dilution ;
#   - les noms des niveaux sont rendus TELS QUELS.
# ==============================================================================

# Un parametre par paire. Accepte un scalaire (toutes les paires), une matrice
# G x G nommee (lignes = groupe recevant, colonnes = groupe emettant), un
# vecteur ou une liste nommes par paire "recevant<-emettant". Rend un vecteur
# numerique nomme par paire, dans l'ordre de `pairs`. Une paire absente d'une
# forme nommee est une ERREUR : la deviner donnerait un dispositif plausible et
# faux.
.rx_par_paire <- function(x, pairs, groups, nom) {
  if (is.matrix(x)) {
    if (is.null(dimnames(x)) || !all(groups %in% rownames(x)) || !all(groups %in% colnames(x)))
      stop("rx_neighbourhood : `", nom, "` est une matrice sans les dimnames des groupes (",
           paste(groups, collapse = ", "), ").", call. = FALSE)
    a <- sub("<-.*$", "", pairs); b <- sub("^.*<-", "", pairs)
    out <- as.numeric(x[cbind(match(a, rownames(x)), match(b, colnames(x)))])
    return(stats::setNames(out, pairs))
  }
  if (is.list(x) || (length(x) > 1L || !is.null(names(x)))) {
    x <- unlist(x)
    if (is.null(names(x)))
      stop("rx_neighbourhood : `", nom, "` a plusieurs valeurs mais aucun nom de paire.",
           call. = FALSE)
    manque <- setdiff(pairs, names(x))
    if (length(manque))
      stop("rx_neighbourhood : `", nom, "` ne donne rien pour la ou les paires ",
           paste(manque, collapse = ", "), ".", call. = FALSE)
    return(stats::setNames(as.numeric(x[pairs]), pairs))
  }
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x))
    stop("rx_neighbourhood : `", nom, "` doit etre un scalaire fini, une matrice nommee ",
         "ou une liste nommee par paire.", call. = FALSE)
  stats::setNames(rep(as.numeric(x), length(pairs)), pairs)
}

# Triplets (i, j, x) -> matrice creuse ou dense selon `sparse`, avec dimnames.
.rx_assembler <- function(i, j, x, dims, dn, sparse) {
  M <- Matrix::sparseMatrix(i = i, j = j, x = x, dims = dims, dimnames = dn)
  if (sparse) M else as.matrix(M)
}

# Incidences de voisinage ponderees par la distance
#
# @param coord matrice ou data.frame n x 2 de positions ENTIERES (ligne, colonne)
# @param group facteur de longueur n : la classe de chaque unite
# @param block facteur de longueur n ou NULL : deux unites de blocs differents
#   ne sont jamais voisines
# @param level facteur de longueur n : le niveau auquel les poids sont sommes
#   dans la sortie par niveau (requis si output contient "level")
# @param id identifiants des unites (noms de lignes des sorties)
# @param rank rayon en pas de grille : scalaire, matrice G x G nommee, ou liste
#   nommee par paire "recevant<-emettant"
# @param reach exposant lambda du noyau, memes formes
# @param dilution exposant d de la dilution, memes formes
# @param kernel "power" (delta^-lambda), "exponential" (exp(-lambda (delta -
#   delta_0)), delta_0 = plus petit pas physique) ou "none" (w = 1)
# @param window "chebyshev" (carre en indices) ou "euclidean" (delta <= r min(spacing))
# @param spacing c(pas_ligne, pas_colonne) en unite physique
# @param normalise diviser chaque ligne par sa norme L2 APRES la dilution
# @param pairs NULL (toutes les paires) ou vecteur de "recevant<-emettant"
# @param output "level", "unit" ou "both"
# @param sparse rendre des dgCMatrix
# @return liste de classe rx_neighbourhood : level, unit, n_neighbours, params
rx_neighbourhood <- function(coord, group, block = NULL, level = NULL, id = NULL,
                             rank, reach = 0, dilution = 0,
                             kernel = c("power", "exponential", "none"),
                             window = c("chebyshev", "euclidean"),
                             spacing = c(1, 1), normalise = FALSE,
                             pairs = NULL, output = c("level", "unit", "both"),
                             sparse = TRUE) {
  kernel <- match.arg(kernel); window <- match.arg(window); output <- match.arg(output)
  coord <- as.matrix(coord)
  if (ncol(coord) != 2L) stop("rx_neighbourhood : `coord` doit avoir deux colonnes (ligne, colonne).",
                              call. = FALSE)
  n <- nrow(coord)
  storage.mode(coord) <- "double"
  if (any(!is.finite(coord))) stop("rx_neighbourhood : `coord` porte des valeurs non finies.", call. = FALSE)
  if (any(coord != round(coord)))
    stop("rx_neighbourhood : `coord` doit porter des positions ENTIERES sur la grille ; ",
         "les positions physiques sont coord * spacing.", call. = FALSE)
  if (length(group) != n) stop("rx_neighbourhood : `group` n'a pas ", n, " elements.", call. = FALSE)
  group <- factor(group)
  if (anyNA(group)) stop("rx_neighbourhood : `group` porte des NA.", call. = FALSE)
  groups <- levels(group)
  block <- if (is.null(block)) factor(rep(1L, n)) else factor(block)
  if (length(block) != n) stop("rx_neighbourhood : `block` n'a pas ", n, " elements.", call. = FALSE)
  id <- if (is.null(id)) as.character(seq_len(n)) else as.character(id)
  if (length(id) != n) stop("rx_neighbourhood : `id` n'a pas ", n, " elements.", call. = FALSE)
  if (anyDuplicated(id)) stop("rx_neighbourhood : `id` porte des doublons.", call. = FALSE)
  if (output != "unit") {
    if (is.null(level)) stop("rx_neighbourhood : `level` est requis pour la sortie par niveau.",
                             call. = FALSE)
    level <- as.character(level)
    if (length(level) != n || anyNA(level))
      stop("rx_neighbourhood : `level` doit avoir ", n, " elements sans NA.", call. = FALSE)
  }
  spacing <- as.numeric(spacing)
  if (length(spacing) == 1L) spacing <- rep(spacing, 2L)
  if (length(spacing) != 2L || any(!is.finite(spacing)) || any(spacing <= 0))
    stop("rx_neighbourhood : `spacing` doit etre deux pas strictement positifs.", call. = FALSE)

  toutes <- as.vector(t(outer(groups, groups, function(a, b) paste0(a, "<-", b))))
  if (is.null(pairs)) pairs <- toutes
  else {
    inconnues <- setdiff(pairs, toutes)
    if (length(inconnues))
      stop("rx_neighbourhood : paire(s) inconnue(s) : ", paste(inconnues, collapse = ", "),
           ". Forme attendue \"recevant<-emettant\" avec les groupes ",
           paste(groups, collapse = ", "), ".", call. = FALSE)
  }
  r_p <- .rx_par_paire(rank, pairs, groups, "rank")
  l_p <- .rx_par_paire(reach, pairs, groups, "reach")
  d_p <- .rx_par_paire(dilution, pairs, groups, "dilution")
  if (any(r_p < 0) || any(r_p != round(r_p)))
    stop("rx_neighbourhood : `rank` doit etre un entier >= 0 pour chaque paire.", call. = FALSE)
  if (any(l_p < 0) || any(d_p < 0))
    stop("rx_neighbourhood : `reach` et `dilution` doivent etre >= 0.", call. = FALSE)

  delta0 <- min(spacing)
  # niveaux par groupe emettant, dans l'ordre sort(unique(level))
  lev_de <- if (output != "unit")
    lapply(stats::setNames(groups, groups), function(g) sort(unique(level[group == g]))) else NULL
  pos_g <- lapply(stats::setNames(groups, groups), function(g) which(group == g))

  out_level <- list(); out_unit <- list(); n_vois <- list()
  for (pr in pairs) {
    a <- sub("<-.*$", "", pr); b <- sub("^.*<-", "", pr)
    ia <- pos_g[[a]]; ib <- pos_g[[b]]
    na_ <- length(ia); nb_ <- length(ib)
    r <- r_p[[pr]]; lam <- l_p[[pr]]; dd <- d_p[[pr]]
    # accumulation en triplets, en indices LOCAUX (position dans le groupe)
    ui <- integer(0); uj <- integer(0); ux <- numeric(0)
    li <- integer(0); lj <- integer(0); lx <- numeric(0)
    nn <- integer(na_)
    if (r > 0 && na_ && nb_) for (bl in levels(block)) {
      sa <- which(block[ia] == bl); sb <- which(block[ib] == bl)
      if (!length(sa) || !length(sb)) next
      ra <- coord[ia[sa], 1L]; ca <- coord[ia[sa], 2L]
      rb <- coord[ib[sb], 1L]; cb <- coord[ib[sb], 2L]
      dr <- abs(outer(ra, rb, "-")); dc <- abs(outer(ca, cb, "-"))
      # distance physique : meme arithmetique que dist() sur (s1 ligne, s2 colonne)
      delta <- sqrt(outer(spacing[1L] * ra, spacing[1L] * rb, "-")^2 +
                      outer(spacing[2L] * ca, spacing[2L] * cb, "-")^2)
      m <- if (window == "chebyshev") (dr <= r) & (dc <= r) & (delta > 0)
           else (delta <= r * delta0) & (delta > 0)
      W <- matrix(0, length(sa), length(sb))
      if (kernel == "power")            W[m] <- delta[m]^(-lam)
      else if (kernel == "exponential") W[m] <- exp(-lam * (delta[m] - delta0))
      else                              W[m] <- 1
      W[is.infinite(W)] <- 0
      ni <- rowSums(m)
      nn[sa] <- ni
      div <- ni; div[div == 0] <- 1
      div <- if (dd == 0) rep(1, length(sa)) else div^dd
      if (output != "level") {
        Wd <- W / div
        nz <- which(Wd != 0, arr.ind = TRUE)
        ui <- c(ui, sa[nz[, 1L]]); uj <- c(uj, sb[nz[, 2L]]); ux <- c(ux, Wd[nz])
      }
      if (output != "unit") {
        lv <- lev_de[[b]]
        Gb <- matrix(0, length(sb), length(lv))
        Gb[cbind(seq_along(sb), match(level[ib[sb]], lv))] <- 1
        M <- (W %*% Gb) / div
        nz <- which(M != 0, arr.ind = TRUE)
        li <- c(li, sa[nz[, 1L]]); lj <- c(lj, nz[, 2L]); lx <- c(lx, M[nz])
      }
    }
    if (output != "unit") {
      lv <- lev_de[[b]]
      M <- .rx_assembler(li, lj, lx, c(na_, length(lv)), list(id[ia], lv), sparse)
      if (normalise) {
        nr <- sqrt(Matrix::rowSums(M^2)); nr[nr == 0] <- 1
        M <- if (sparse) Matrix::Diagonal(x = 1 / nr) %*% M else M / nr
        if (sparse) { M <- methods::as(M, "CsparseMatrix"); dimnames(M) <- list(id[ia], lv) }
      }
      out_level[[pr]] <- M
    }
    if (output != "level") {
      U <- .rx_assembler(ui, uj, ux, c(na_, nb_), list(id[ia], id[ib]), sparse)
      if (normalise) {
        nr <- sqrt(Matrix::rowSums(U^2)); nr[nr == 0] <- 1
        U <- if (sparse) Matrix::Diagonal(x = 1 / nr) %*% U else U / nr
        if (sparse) { U <- methods::as(U, "CsparseMatrix"); dimnames(U) <- list(id[ia], id[ib]) }
      }
      out_unit[[pr]] <- U
    }
    n_vois[[pr]] <- stats::setNames(nn, id[ia])
  }
  params <- data.frame(pair = pairs, receiver = sub("<-.*$", "", pairs),
                       emitter = sub("^.*<-", "", pairs),
                       rank = as.numeric(r_p), reach = as.numeric(l_p),
                       dilution = as.numeric(d_p), kernel = kernel, window = window,
                       spacing_row = spacing[1L], spacing_col = spacing[2L],
                       normalise = normalise, stringsAsFactors = FALSE, row.names = NULL)
  structure(list(level = if (output != "unit") out_level else NULL,
                 unit = if (output != "level") out_unit else NULL,
                 n_neighbours = n_vois, params = params),
            class = "rx_neighbourhood")
}

print.rx_neighbourhood <- function(x, ...) {
  p <- x$params
  cat(sprintf("Voisinage : %d paire(s), noyau %s, fenetre %s, pas (%g, %g)%s\n",
              nrow(p), p$kernel[1], p$window[1], p$spacing_row[1], p$spacing_col[1],
              if (isTRUE(p$normalise[1])) ", normalise L2" else ""))
  for (i in seq_len(nrow(p))) {
    nn <- x$n_neighbours[[p$pair[i]]]
    dimtxt <- if (!is.null(x$level)) paste0("level ", paste(dim(x$level[[p$pair[i]]]), collapse = "x"))
              else paste0("unit ", paste(dim(x$unit[[p$pair[i]]]), collapse = "x"))
    cat(sprintf("  %-20s rank %2g  reach %g  dilution %g  | %s | voisins/unite %.2f (max %d)\n",
                p$pair[i], p$rank[i], p$reach[i], p$dilution[i], dimtxt,
                mean(nn), max(nn)))
  }
  invisible(x)
}

# ------------------------------------------------------------------------------
# Fonctionnelles d'exposition : d, k, k_identity, c, S, n_eff
# ------------------------------------------------------------------------------
# Ce que ces nombres font : la variance phenotypique apportee par une
# composante indirecte de variance s2_I vaut k s2_I ; celle de la composante
# directe vaut d s2_D ; la covariance directe-indirecte entre dans Var(P) avec
# le coefficient 2 c ; la valeur genetique totale d'un niveau vaut
# sqrt(d) (u_D + S u_I). Toutes les moyennes se prennent sur `rows`.
#
# POURQUOI `rows` EXISTE. Dans un modele empile sur plusieurs cibles, les
# incidences couvrent toutes les lignes ; moyenner sur toutes divise chaque
# statistique par la part des lignes qui appartiennent a la cible. La forme
# modele prend rows = "auto" : les lignes ou l'incidence directe est non nulle.

.rx_kstat <- function(Z, K, rows) {
  rs <- if (is.null(K)) Matrix::rowSums(Z^2) else Matrix::rowSums((Z %*% K) * Z)
  mean(as.numeric(rs)[rows])
}

# Resout "terme:etiquette", "terme[i]" ou "terme" (t = 1) en (terme, colonne).
.rx_ref_colonne <- function(model, ref, quoi) {
  tm_nom <- sub("[:\\[].*$", "", ref)
  tm <- model[["terms"]][[tm_nom]]
  if (is.null(tm)) stop("rx_exposure : `", quoi, "` nomme le terme '", tm_nom,
                        "', absent du modele.", call. = FALSE)
  if (grepl("\\[", ref)) {
    j <- as.integer(sub("^.*\\[(\\d+)\\]$", "\\1", ref))
  } else if (grepl(":", ref, fixed = TRUE)) {
    lab <- sub("^[^:]*:", "", ref)
    cn <- tm[["colnames"]]
    if (is.null(cn)) stop("rx_exposure : le terme '", tm_nom, "' n'a pas de noms de colonnes ; ",
                          "utiliser '", tm_nom, "[i]'.", call. = FALSE)
    j <- match(lab, cn)
    if (is.na(j)) stop("rx_exposure : colonne '", lab, "' absente du terme '", tm_nom,
                       "' (", paste(cn, collapse = ", "), ").", call. = FALSE)
  } else {
    if (tm[["t"]] != 1L) stop("rx_exposure : le terme '", tm_nom, "' a t = ", tm[["t"]],
                              " colonnes ; en nommer une.", call. = FALSE)
    j <- 1L
  }
  if (is.na(j) || j < 1L || j > tm[["t"]])
    stop("rx_exposure : colonne ", j, " hors de 1..", tm[["t"]], " pour '", tm_nom, "'.",
         call. = FALSE)
  list(term = tm, j = j, Z = tm[["Zl"]][[j]],
       K = if (is.null(tm[["LK"]])) NULL else tm[["LK"]] %*% t(tm[["LK"]]))
}

# Fonctionnelles d'exposition d'une paire (incidence directe, incidence indirecte)
#
# Deux formes. Forme matricielle : rx_exposure(Z_direct, Z_indirect, K, rows,
# K_direct). Forme modele : rx_exposure(model, direct = "terme:etiquette",
# indirect = "terme:etiquette", rows = "auto"), ou K est lu dans le terme de
# l'incidence indirecte et K_direct dans celui de l'incidence directe.
#
# @param Z_direct incidence n x q de l'effet direct, ou NULL, ou un rx_model / rx_fit
# @param Z_indirect incidence n x q_e de l'effet indirect
# @param K parente q_e x q_e du groupe EMETTANT, NULL pour l'identite
# @param rows indices des lignes a moyenner ; NULL = toutes ; "auto" = lignes
#   ou Z_direct est non nulle (forme modele)
# @param K_direct parente du groupe de l'effet direct, par defaut K
# @param direct,indirect references "terme:etiquette" ou "terme[i]" (forme modele)
# @return liste : d, k, k_identity, c, S, n_eff, n_rows, convention
rx_exposure <- function(Z_direct = NULL, Z_indirect = NULL, K = NULL, rows = NULL,
                        K_direct = K, direct = NULL, indirect = NULL) {
  meme_groupe <- TRUE
  if (inherits(Z_direct, "rx_fit") || inherits(Z_direct, "rx_model")) {
    model <- if (inherits(Z_direct, "rx_fit")) Z_direct[["model"]] else Z_direct
    if (is.null(model) || is.null(model[["terms"]]))
      stop("rx_exposure : l'ajustement ne porte pas de modele ; passer le rx_model.",
           call. = FALSE)
    if (is.null(indirect)) stop("rx_exposure : `indirect` est requis dans la forme modele.",
                                call. = FALSE)
    ri <- .rx_ref_colonne(model, indirect, "indirect")
    Z_indirect <- ri$Z
    if (is.null(K)) K <- ri$K
    if (!is.null(direct)) {
      rd <- .rx_ref_colonne(model, direct, "direct")
      Z_direct <- rd$Z
      if (missing(K_direct) || is.null(K_direct)) K_direct <- rd$K
      meme_groupe <- identical(rd$term[["name"]], ri$term[["name"]])
    } else Z_direct <- NULL
    if (identical(rows, "auto")) {
      if (is.null(Z_direct)) stop("rx_exposure : rows = \"auto\" exige `direct`.", call. = FALSE)
      rows <- which(as.numeric(Matrix::rowSums(Z_direct != 0)) > 0)
    }
  }
  if (is.null(Z_indirect)) stop("rx_exposure : `Z_indirect` est requis.", call. = FALSE)
  n <- nrow(Z_indirect)
  if (!is.null(Z_direct) && nrow(Z_direct) != n)
    stop("rx_exposure : Z_direct a ", nrow(Z_direct), " lignes, Z_indirect ", n, ".", call. = FALSE)
  if (is.null(rows)) rows <- seq_len(n)
  if (is.logical(rows)) rows <- which(rows)
  rows <- as.integer(rows)
  if (!length(rows)) stop("rx_exposure : aucune ligne a moyenner.", call. = FALSE)
  if (any(rows < 1L | rows > n)) stop("rx_exposure : `rows` hors de 1..", n, ".", call. = FALSE)
  if (!is.null(K)) {
    K <- as.matrix(K)
    if (nrow(K) != ncol(Z_indirect) || ncol(K) != ncol(Z_indirect))
      stop("rx_exposure : K est ", nrow(K), "x", ncol(K), " pour ", ncol(Z_indirect),
           " colonnes de Z_indirect.", call. = FALSE)
    # alignement par nom quand les deux cotes en portent
    if (!is.null(colnames(Z_indirect)) && !is.null(rownames(K))) {
      manque <- setdiff(colnames(Z_indirect), rownames(K))
      if (length(manque)) stop("rx_exposure : ", length(manque), " colonne(s) de Z_indirect ",
                               "absente(s) de K (ex. ", paste(utils::head(manque, 3), collapse = ", "),
                               ").", call. = FALSE)
      K <- K[colnames(Z_indirect), colnames(Z_indirect), drop = FALSE]
    }
  }
  if (!is.null(Z_direct) && !is.null(K_direct)) {
    K_direct <- as.matrix(K_direct)
    if (nrow(K_direct) != ncol(Z_direct))
      stop("rx_exposure : K_direct est ", nrow(K_direct), "x", ncol(K_direct), " pour ",
           ncol(Z_direct), " colonnes de Z_direct.", call. = FALSE)
    if (!is.null(colnames(Z_direct)) && !is.null(rownames(K_direct))) {
      manque <- setdiff(colnames(Z_direct), rownames(K_direct))
      if (length(manque)) stop("rx_exposure : ", length(manque), " colonne(s) de Z_direct ",
                               "absente(s) de K_direct.", call. = FALSE)
      K_direct <- K_direct[colnames(Z_direct), colnames(Z_direct), drop = FALSE]
    }
  }
  k_id <- .rx_kstat(Z_indirect, NULL, rows)
  k <- if (is.null(K)) k_id else .rx_kstat(Z_indirect, K, rows)
  d <- if (is.null(Z_direct)) NA_real_ else .rx_kstat(Z_direct, K_direct, rows)
  # c : moyenne de (Zd K Zn')_ii ; NA si pas d'incidence directe ou si les deux
  # groupes different (K ne relie pas deux groupes) ou si les dimensions ne
  # permettent pas le produit.
  cc <- NA_real_
  if (!is.null(Z_direct) && meme_groupe && ncol(Z_direct) == ncol(Z_indirect)) {
    rs <- if (is.null(K)) Matrix::rowSums(Z_direct * Z_indirect)
          else Matrix::rowSums((Z_direct %*% K) * Z_indirect)
    cc <- mean(as.numeric(rs)[rows])
  }
  S <- mean(as.numeric(Matrix::rowSums(Z_indirect))[rows])
  n_eff <- if (isTRUE(is.finite(d) && k > 0)) S^2 * d / k else NA_real_
  list(d = d, k = k, k_identity = k_id, c = cc, S = S, n_eff = n_eff,
       n_rows = length(rows), convention = if (is.null(K)) "identity" else "K")
}

# ------------------------------------------------------------------------------
# Matrice de parente genomique (VanRaden 1, ploidie quelconque, blending)
# ------------------------------------------------------------------------------
# @param M matrice individus x marqueurs de doses alleliques, codees en
#   FRACTION de la ploidie (0, 1/k, ..., 1) ; avec `coding = "count"` les doses
#   sont en nombre d'alleles (0..k)
# @param ploidy ploidie k
# @param blend part de l'identite ajoutee : G_b = (1 - blend) G + blend I
# @param coding "fraction" ou "count"
# @return matrice q x q avec les rownames de M ; attribut "denominator"
rx_grm <- function(M, ploidy = 2, blend = 0, coding = c("fraction", "count")) {
  coding <- match.arg(coding)
  M <- as.matrix(M)
  if (anyNA(M)) stop("rx_grm : M porte des NA ; imputer avant.", call. = FALSE)
  D <- if (coding == "fraction") M * ploidy else M
  if (any(D < 0) || any(D > ploidy))
    stop("rx_grm : des doses sortent de 0..", ploidy, " ; verifier `coding` et `ploidy`.",
         call. = FALSE)
  p <- colMeans(D) / ploidy
  Z <- sweep(D, 2L, ploidy * p, "-")
  cst <- ploidy * sum(p * (1 - p))
  if (cst <= 0) stop("rx_grm : tous les marqueurs sont monomorphes.", call. = FALSE)
  G <- tcrossprod(Z) / cst
  if (blend > 0) G <- (1 - blend) * G + blend * diag(nrow(G))
  dimnames(G) <- list(rownames(M), rownames(M))
  attr(G, "denominator") <- cst
  G
}
