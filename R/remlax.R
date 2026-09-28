# ==============================================================================
# remlax — INTERFACE R D'UN SOLVEUR REML GENERIQUE
# ==============================================================================
#
# Ce fichier decrit des MODELES ; il n'en ajuste aucun. Il construit les
# incidences, verifie leur coherence, serialise le tout, et delegue le calcul au
# solveur JAX (scripts/gpu/remlax/), qui tourne sur GPU si la machine en a un
# et sur CPU sinon, SANS QUE LE MODELE CHANGE.
#
# MODELE
#     y = X beta + sum_k Z_k u_k + e
#     u_k ~ N(0, Sigma_k (x) K_k)     Sigma_k : t_k x t_k, K_k : q_k x q_k
#     e   ~ N(0, R)
#
# Un TERME est donc : un ensemble de t incidences (n x q) partageant les memes
# q niveaux, plus une structure de covariance Sigma sur ces t colonnes, plus une
# parente K entre niveaux. Ce seul objet couvre :
#   - un facteur aleatoire simple            t = 1, K = I
#   - un effet genetique avec parente        t = 1, K = GRM
#   - un modele multi-caractere us / FA      t = nb de caracteres
#   - le modele IGE de ce projet             t = DGE + IGE intra + IGE inter,
#                                            chaque colonne ayant SA propre
#                                            incidence (voisinage pondere)
#
# INTERFACE, dans l'esprit d'asreml
#     rx_reml(fixed    = y ~ 1 + traitement,
#             random   = ~ vm(genotype, K = Kmat) + iid(bloc),
#             residual = ~ units,
#             data     = df)
# et, pour tout ce qu'une formule ne sait pas dire (incidences ponderees,
# colonnes heterogenes), la voie explicite :
#     rx_model(y, X, terms = list(rx_term(...)), residual = rx_residual(...))
#
# CE QUE ce solveur PARTAGE avec asreml : l'objectif (la vraisemblance
# restreinte), donc l'optimum, et un schema de type Newton pour y finir.
# CE QU'IL NE PARTAGE PAS : asreml utilise l'information moyenne (AI-REML) des
# la premiere iteration ; ici L-BFGS-B fait l'approche et Newton le polissage.
# Les estimes coincident, le chemin non — et le nombre d'iterations non plus.
#
# CONVENTION D'ORDRE DES COLONNES (partagee avec structures.py) :
#     colonne de Z = (col - 1) * q + niveau     -> NIVEAU le plus rapide
# Sigma (x) K suit le meme ordre. En changer d'un cote sans l'autre donne un
# autre modele, sans aucune erreur visible : ne pas y toucher.
# ==============================================================================

suppressPackageStartupMessages({ library(Matrix); library(jsonlite) })

# Repertoire de CE fichier, capture au moment du source(). Sert a retrouver le
# solveur Python dans l'arborescence du depot quand le paquet n'est pas installe.
# `sys.frame(i)$ofile` n'existe que pendant un source() : on parcourt la pile
# d'appels, du plus recent au plus ancien, et on retient le premier trouve.
.RX_FILE_DIR <- local({
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  NA_character_
})

# Structures de Sigma (annexe C du manuel ASReml-R 4.2). Correspondances :
#   iid = idv   diag = idh   us = corgh   fa(k)   rr(k)   chol(k)   ante(k)   corh
RX_STRUCTURES <- c("iid", "diag", "us", "fa", "rr", "chol", "ante", "corh")

# ==============================================================================
# 1. STRUCTURES : nombre de parametres (doit reproduire structures.py)
# ==============================================================================
rx_n_loadings <- function(t, r) sum(pmin(seq_len(t), r))

rx_n_params <- function(struct, t, rank = 0L) {
  t <- as.integer(t); k <- as.integer(rank)
  switch(struct,
         iid  = 1L,
         diag = t,
         us   = as.integer(t * (t + 1) / 2),
         fa   = as.integer(rx_n_loadings(t, k) + t),
         # Le manuel annonce k*omega pour rr. On contraint Gamma a etre
         # TRAPEZOIDALE, comme pour fa : sans cela Gamma n'est definie qu'a une
         # rotation pres et le Hessien est singulier de k(k-1)/2 dimensions.
         # structures.py compte pareil ; deux decomptes differents feraient
         # decouper theta au mauvais endroit.
         rr   = as.integer(rx_n_loadings(t, k)),
         chol = ,
         ante = as.integer((k + 1) * (t - k / 2)),
         corh = as.integer(t + 1L),
         stop("structure inconnue : ", struct))
}

# ==============================================================================
# 2. TERMES
# ==============================================================================
#' Terme aleatoire
#'
#' @param name   nom (sert de cle dans les sorties)
#' @param Z      incidence. Trois formes acceptees :
#'               - un vecteur (facteur/caractere) de longueur n : incidence
#'                 indicatrice, t = 1 ;
#'               - une LISTE de t matrices n x q : une par colonne de Sigma ;
#'               - une matrice n x (t*q) deja empilee, avec `t` fourni.
#' @param K      parente entre niveaux (q x q, dimnames = niveaux) ou NULL = I
#' @param struct "iid" | "diag" | "us" | "fa"
#' @param rank   rang de la structure FA
#' @param levels niveaux, si Z est fourni en matrice (sinon deduits du facteur)
# Catalogue des structures ENTRE NIVEAUX (correlations), aligne sur l'annexe C
# du manuel ASReml-R 4.2. La VARIANCE vit dans Sigma : `ar1v` d'asreml = ici
# struct="iid" + level="ar1" ; `ar1h` = struct="diag" + level="ar1".
RX_LEVEL_STRUCTURES <- c("id", "fixed", "cor", "corb", "corg",
                         "ar1", "ar2", "ar3", "sar", "ma1", "ma2", "arma",
                         "exp", "gau", "lvr", "iexp", "igau", "ieuc",
                         "sph", "cir", "aexp", "agau", "mtrn", "own", "ar1ar1",
                         # PRODUIT SEPARABLE A NOMBRE QUELCONQUE DE FACTEURS.
                         # `ar1ar1` ne couvre que DEUX facteurs. Un champ spatial
                         # replique independamment par bloc s'ecrit
                         # id (x) ar1 (x) ar1 : trois facteurs, dont le premier
                         # n'apporte AUCUN parametre, donc les correlations sont
                         # PARTAGEES entre les repliques. C'est ce que le moteur
                         # d'IGE_analysis estime, et ce que `ar1ar1` ne peut pas
                         # exprimer. `parts` porte la liste (famille, dimension).
                         "sep")
RX_LEVEL_NP <- c(id = 0L, fixed = 0L, cor = 1L, ar1 = 1L, ar2 = 2L, ar3 = 3L,
                 sar = 1L, ma1 = 1L, ma2 = 2L, arma = 2L, exp = 1L, gau = 1L,
                 lvr = 1L, iexp = 1L, igau = 1L, ieuc = 1L,
                 sph = 1L, cir = 1L, aexp = 2L, agau = 2L, ar1ar1 = 2L)
# corb : `order` parametres ; corg : order*(order-1)/2 ; mtrn : ceux qui sont
# declares (cf. .rx_mtrn_opts) ; own : n_par declare par l'utilisateur.
# Structures METRIQUES : elles exigent des coordonnees. 1D et 2D separees, car
# une coordonnee a une seule colonne fournie a iexp() serait acceptee en silence
# et donnerait un modele different de celui demande.
RX_METRIQUES_1D <- c("exp", "gau", "lvr")
RX_METRIQUES_2D <- c("iexp", "igau", "ieuc", "sph", "cir",
                     "aexp", "agau", "mtrn")
RX_METRIQUES <- c(RX_METRIQUES_1D, RX_METRIQUES_2D)

#' Nombre de parametres d'une structure entre niveaux (doit suivre levels.py)
rx_n_level <- function(level, order = 0L, opts = NULL, parts = NULL) {
  # PRODUIT SEPARABLE : la somme des parametres de ses facteurs. `id` en apporte
  # zero, ce qui est precisement ce qui rend les correlations PARTAGEES entre
  # les repliques d'un champ.
  if (identical(level, "sep")) {
    if (is.null(parts) || !length(parts))
      stop("level='sep' exige `parts`, la liste des facteurs (famille, dimension).",
           call. = FALSE)
    # Meme refus que levels.n_level_params : une famille dont le compte depend
    # d'un ordre ou d'options (corb, corg, mtrn, own) tomberait a zero parametre
    # et deviendrait une identite en silence ; les familles metriques exigent
    # des coordonnees que `parts` ne porte pas.
    fam <- vapply(parts, function(x) as.character(x[[1]]), "")
    mauvais <- intersect(fam, c("ar2", "ar3", "ma2", "arma", "corb", "corg",
                                "mtrn", "own", RX_METRIQUES))
    if (length(mauvais))
      stop("level='sep' : les familles ", paste(sort(unique(mauvais)), collapse = ", "),
           " ne sont pas admises dans `parts` (compte dependant d'un ordre ou ",
           "d'options, ou coordonnees requises).", call. = FALSE)
    return(sum(vapply(parts, function(x)
      rx_n_level(as.character(x[[1]]), 0L), integer(1))))
  }
  if (level == "mtrn")
    # Memes defauts que levels.n_level_params : phi est estime sauf mention
    # contraire, les trois autres sont tenus fixes. Un `est_*` absent (rx_term
    # appele directement avec des opts partiels) levait « subscript out of
    # bounds » ; et le compter a zero pour phi aurait decoupe theta
    # autrement que le solveur.
    return(sum(vapply(c("phi", "nu", "delta", "alpha"), function(p) {
      v <- if (is.null(opts)) numeric(0) else
        suppressWarnings(as.numeric(unlist(opts[paste0("est_", p)])))
      if (length(v) != 1L || is.na(v)) v <- if (p == "phi") 1 else 0
      isTRUE(v > 0.5)
    }, TRUE)))
  if (level == "own")  return(as.integer(opts[["n_par"]] %||% 0L))
  if (level == "corb") return(as.integer(order))
  if (level == "corg") return(as.integer(order * (order - 1L) / 2L))
  RX_LEVEL_NP[[level]]
}

#' Argument facon asreml : 3 = estime en partant de 3 ; "3 F" = fixe a 3 ;
#' absent = fixe au defaut. C'est la regle du manuel, reprise telle quelle.
.rx_arg_vs <- function(x, defaut, estime_si_absent = FALSE) {
  if (is.null(x)) return(list(est = estime_si_absent, val = defaut))
  if (is.numeric(x)) return(list(est = TRUE, val = as.numeric(x)))
  s <- trimws(as.character(x))
  code <- toupper(sub("^[-0-9.eE+]*\\s*", "", s))
  v <- suppressWarnings(as.numeric(sub("\\s*[A-Za-z]*$", "", s)))
  if (is.na(v)) stop("valeur de parametre illisible : '", s, "'")
  list(est = !identical(code, "F"), val = v)
}

.rx_mtrn_opts <- function(opt, env = parent.frame()) {
  ev <- function(z) if (is.null(z)) NULL else eval(z, envir = env)
  dfl <- list(phi = NA, nu = 0.5, delta = 1.0, alpha = 0.0)
  o <- c(lambda = as.numeric(ev(opt$lambda) %||% 2))
  if (!o[["lambda"]] %in% c(1, 2))
    stop("mtrn : lambda vaut 1 (city-block) ou 2 (euclidienne), pas ", o[["lambda"]], ".")
  for (p in c("phi", "nu", "delta", "alpha")) {
    a <- .rx_arg_vs(ev(opt[[p]]), dfl[[p]], estime_si_absent = (p == "phi"))
    o[paste0("est_", p)] <- as.numeric(isTRUE(a$est))
    o[p] <- if (is.na(a$val)) dfl[[p]] else a$val
    if (isTRUE(a$est) && !is.na(a$val)) o[paste0("init_", p)] <- a$val
  }
  o
}

#' @param Kinv PRECISION entre niveaux, K^-1, en matrice creuse (dgCMatrix).
#'   Alternative a `K` et destinee au moteur CREUX, qui n'a jamais besoin de K
#'   ni de sa Cholesky : les equations du modele mixte contiennent G^-1, donc
#'   K^-1. C'est ce qui permet de fournir une parente genealogique par les regles
#'   de Henderson, ou A^-1 est creuse — environ cinq non-nuls par individu — sans
#'   jamais former A ni la factoriser. Fournir `K` ne permet pas cela : `LK` est
#'   un facteur DENSE, et l'inverse d'une matrice creuse est generalement plein.
#'
#'   `Kinv_logdet` permet de passer log|K| quand il est connu (Henderson le donne
#'   en somme de termes locaux) ; sinon il est calcule une fois par une Cholesky
#'   creuse. Il ne depend d'aucun parametre, donc c'est une constante de
#'   l'optimisation.
#'
#'   Le moteur DENSE ignore `Kinv` : il lui faudrait inverser pour retrouver K,
#'   ce qui annulerait tout l'interet. Un terme ainsi declare n'est donc ajustable
#'   que par le moteur creux, et rx_model le signale.
rx_term <- function(name, Z, K = NULL, struct = "iid", rank = 0L,
                    t = NULL, levels = NULL, level = "auto",
                    dims = NULL, order = 0L, coord = NULL,
                    opts = NULL, expr = NULL, parts = NULL,
                    Kinv = NULL, Kinv_logdet = NULL, colnames = NULL) {
  # NOMS DES COLONNES DE SIGMA. Sans eux, tout consommateur d'un Sigma t x t
  # apparie ses composantes PAR INDICE, et une etape aval qui reetiquette depuis
  # une liste tenue a part ne peut que verifier la longueur. Les noms voyagent
  # jusqu'a fit$sigmas (dimnames), fit$blups (colonnes) et fit$composantes_noms.
  # Par defaut ce sont les noms de la liste Z quand elle en porte.
  if (is.null(colnames) && is.list(Z) && !is.null(names(Z)) && all(nzchar(names(Z))))
    colnames <- names(Z)
  if (!is.null(Kinv) && !is.null(K))
    stop("terme '", name, "' : fournir K OU Kinv, pas les deux. K sert au moteur ",
         "dense (facteur de Cholesky), Kinv au moteur creux (precision).")
  struct <- match.arg(struct, RX_STRUCTURES)
  level  <- match.arg(level, c("auto", RX_LEVEL_STRUCTURES))
  if (is.factor(Z) || is.character(Z)) {
    f <- factor(Z)
    levels <- levels(f)
    q <- length(levels)
    Zm <- Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f),
                               x = 1, dims = c(length(f), q))
    Zl <- list(Zm); t <- 1L
  } else if (is.list(Z)) {
    t <- length(Z); Zl <- lapply(Z, function(m) methods::as(as.matrix(m), "dgCMatrix"))
    q <- ncol(Zl[[1]])
    if (any(vapply(Zl, ncol, 1L) != q))
      stop("terme '", name, "' : les incidences n'ont pas toutes ", q, " colonnes.")
  } else {
    if (is.null(t)) stop("terme '", name, "' : fournir `t` avec une matrice empilee.")
    Zm <- methods::as(as.matrix(Z), "dgCMatrix")
    q <- ncol(Zm) / t
    if (q != round(q)) stop("terme '", name, "' : ncol(Z) n'est pas un multiple de t.")
    q <- as.integer(q)
    Zl <- lapply(seq_len(t), function(a) Zm[, ((a - 1) * q + 1):(a * q), drop = FALSE])
  }
  # Des incidences fournies en matrices qui portent des noms de colonnes
  # definissent les niveaux : K est alors realigne sur eux par nom, et un
  # niveau absent de K est une erreur. Sans noms, l'appariement reste positionnel.
  if (is.null(levels) && is.list(Z)) {
    cn <- colnames(Zl[[1]])
    if (!is.null(cn) && !anyDuplicated(cn) &&
        all(vapply(Zl, function(m) identical(colnames(m), cn), logical(1))))
      levels <- cn
  }
  if (!is.null(colnames)) {
    colnames <- as.character(colnames)
    if (length(colnames) != t || anyDuplicated(colnames) || anyNA(colnames))
      stop("terme '", name, "' : `colnames` doit donner ", t, " noms distincts (recu ",
           length(colnames), ").")
  }
  if (struct == "iid" && t > 1L)
    message("terme '", name, "' : struct='iid' avec t=", t,
            " -> une seule variance partagee par les ", t, " colonnes.")
  if (struct %in% c("fa", "rr") && (rank < 1L || rank > t))
    stop("terme '", name, "' : rank doit etre dans 1..", t, " pour une structure ", struct, ".")
  if (struct %in% c("chol", "ante") && (rank < 1L || rank > t - 1L))
    stop("terme '", name, "' : pour ", struct, ", rank (l'ordre de la bande) doit ",
         "etre dans 1..", t - 1L, ".")

  LK <- NULL
  if (!is.null(K)) {
    K <- as.matrix(K)
    if (!is.null(levels) && !is.null(rownames(K))) {
      manque <- setdiff(levels, rownames(K))
      if (length(manque))
        stop("terme '", name, "' : ", length(manque),
             " niveau(x) absent(s) de K (ex. ", paste(head(manque, 3), collapse = ", "), ").")
      K <- K[levels, levels, drop = FALSE]
    }
    if (nrow(K) != q) stop("terme '", name, "' : K est ", nrow(K), "x", ncol(K),
                           " mais il y a ", q, " niveaux.")
    # Cholesky avec bending si besoin : une GRM est frequemment semi-definie
    # (genotypes identiques, plus de marqueurs que d'individus), et echouer ici
    # obligerait l'utilisateur a la reparer lui-meme sans savoir de combien.
    ch <- tryCatch(chol(K), error = function(e) NULL)
    if (is.null(ch)) {
      eps <- 1e-8 * mean(diag(K))
      for (i in 1:8) {
        ch <- tryCatch(chol(K + diag(eps, q)), error = function(e) NULL)
        if (!is.null(ch)) { message("terme '", name, "' : K non definie positive, ",
                                    "bending +", format(eps, digits = 2), " applique."); break }
        eps <- eps * 10
      }
      if (is.null(ch)) stop("terme '", name, "' : K reste non factorisable apres bending.")
    }
    LK <- t(ch)                                   # K = LK LK'
  }
  # Structure ENTRE NIVEAUX. "auto" = "fixed" si une K est fournie, "id" sinon.
  # ar1 / ar1ar1 ajoutent des parametres (rho), estimes comme les variances.
  if (level == "auto") level <- if (!is.null(LK)) "fixed" else "id"
  if (level == "ar1ar1") {
    if (is.null(dims) || length(dims) != 2L)
      stop("terme '", name, "' : level='ar1ar1' exige dims = c(n_lignes, n_colonnes).")
    if (prod(dims) != q)
      stop("terme '", name, "' : dims = ", paste(dims, collapse = "x"),
           " ne fait pas ", q, " niveaux.")
  }
  if (!level %in% c("id", "fixed") && !is.null(LK))
    stop("terme '", name, "' : une structure '", level, "' et une matrice K ",
         "fournie sont exclusives (la structure EST la covariance entre niveaux).")
  if (level %in% RX_METRIQUES) {
    if (is.null(coord))
      stop("terme '", name, "' : level='", level, "' exige `coord` ",
           "(coordonnees des ", q, " niveaux, vecteur ou matrice a 2 colonnes).")
    coord <- as.matrix(coord)
    if (nrow(coord) != q)
      stop("terme '", name, "' : `coord` a ", nrow(coord), " lignes pour ", q, " niveaux.")
    if (level %in% RX_METRIQUES_2D && ncol(coord) != 2L)
      stop("terme '", name, "' : '", level, "' est une structure a DEUX ",
           "dimensions ; `coord` a ", ncol(coord), " colonne(s). Une coordonnee ",
           "1D serait acceptee en silence et donnerait un autre modele.")
  }
  if (level == "own" && (is.null(expr) || !nzchar(expr)))
    stop("terme '", name, "' : level='own' exige `expr`, l'expression de la ",
         "correlation (variables : d, dx, dy, lag, I, J, p1..pk).")
  n_lvl <- rx_n_level(level, order, opts, parts = parts)
  # Validation de Kinv : c'est une matrice CREUSE q x q symetrique. On verifie la
  # taille et la symetrie du MOTIF, pas les valeurs — une precision fournie par
  # l'utilisateur peut legitimement etre stockee en triangle.
  if (!is.null(Kinv)) {
    Kinv <- methods::as(methods::as(Kinv, "CsparseMatrix"), "generalMatrix")
    if (nrow(Kinv) != q || ncol(Kinv) != q)
      stop("terme '", name, "' : Kinv est ", nrow(Kinv), "x", ncol(Kinv),
           " mais il y a ", q, " niveaux.")
    dens <- Kinv@x |> length() / (as.numeric(q) * q)
    if (dens > 0.5)
      message("terme '", name, "' : Kinv est remplie a ",
              sprintf("%.0f %%", 100 * dens), ". Le moteur creux n'y gagnera ",
              "rien — c'est le cas d'une parente genomique, dont l'inverse est ",
              "plein. Voir remlax.sparsity pour le diagnostic.")
  }
  structure(list(name = name, Zl = Zl, t = as.integer(t), q = as.integer(q),
                 struct = struct, rank = as.integer(rank), LK = LK,
                 Kinv = Kinv, Kinv_logdet = Kinv_logdet,
                 level = level, dims = dims, order = as.integer(order), coord = coord,
                 opts = opts, expr = expr, parts = parts,
                 levels = levels, colnames = colnames,
                 n_par = rx_n_params(struct, t, rank) + n_lvl),
            class = "rx_term")
}

#' Structure residuelle
#' @param struct "iid" | "diag" | "us"
#' @param trait  facteur de caractere (longueur n) ; NULL = un seul caractere
#' @param unit   identifiant d'unite : deux observations de la MEME unite sur
#'               des caracteres differents sont correlees sous `us`.
rx_residual <- function(struct = "iid", trait = NULL, unit = NULL, rank = 0L,
                        level = "id", order = 0L, coord = NULL, n_unit = NULL,
                        dims = NULL, opts = NULL, expr = NULL, sections = NULL,
                        rows = NULL, name = NULL) {
  struct <- match.arg(struct, c("iid", "diag", "us", "fa"))
  level  <- match.arg(level, RX_LEVEL_STRUCTURES)
  structure(list(struct = struct, trait = trait, unit = unit, rank = as.integer(rank),
                 level = level, order = as.integer(order), coord = coord,
                 n_unit = n_unit, dims = dims, opts = opts, expr = expr,
                 sections = sections, rows = rows, name = name),
            class = "rx_residual")
}

# ------------------------------------------------------------------------------
# SPECIFICATION DE LA RESIDUELLE PAR FORMULE, comme asreml
# ------------------------------------------------------------------------------
#   residual = ~ units                        R = sigma^2 I
#   residual = ~ us(trait):units              couplage entre caracteres d'une
#                                             meme unite
#   residual = ~ ar1(row):ar1(col)            champ residuel separable
#   residual = ~ diag(trait):ar1(row):ar1(col)  les deux a la fois
#   residual = ~ exp(pos, coord = x)          decroissance metrique
#
# La formule se lit comme un produit de facteurs separes par ":". AU PLUS UN
# facteur porte sur les CARACTERES (celui dont l'argument est la colonne passee
# en `trait`) ; les autres portent sur les UNITES. C'est la meme grammaire que
# pour les termes aleatoires, ce qui evite d'avoir deux langages a retenir.
#
# `units` (ou `id(units)`) signifie "aucune structure entre unites" : deux
# observations ne sont alors correlees que si elles sont sur la MEME unite.
# ------------------------------------------------------------------------------
# dsum : UNE STRUCTURE RESIDUELLE PAR SECTION (somme directe)
# ------------------------------------------------------------------------------
#   residual = ~ dsum(~ ar1(row):ar1(col) | site)
#       la MEME forme sur chaque site, mais des parametres PROPRES a chacun
#   residual = ~ dsum(~ ar1(row):ar1(col) + units | site, levels = list(1:3, 4))
#       des formes DIFFERENTES selon les sites
#
# Les sections partitionnent les observations : R est bloc-diagonale a une
# permutation pres. C'est ce qui permet a deux essais de tailles ou de
# geometries differentes de coexister dans un seul ajustement, sans imposer a
# l'un la structure spatiale de l'autre.
.rx_parse_residual <- function(residual, data, trait = NULL, unit = NULL) {
  if (inherits(residual, "formula")) {
    lab1 <- attr(terms(residual, keep.order = TRUE), "term.labels")
    if (length(lab1) == 1L && grepl("^dsum\\(", lab1)) {
      e <- str2lang(lab1)
      args <- as.list(e)[-1]
      nm <- names(args); if (is.null(nm)) nm <- rep("", length(args))
      inner <- args[nm == ""][[1]]
      # `terms()` rend l'argument tel quel : une FORMULE unilaterale dont le
      # corps est `structure | section`. Il faut donc deballer le `~` avant de
      # chercher le `|`, sinon inner[[1]] vaut "~" et le test echoue toujours.
      if (is.call(inner) && identical(as.character(inner[[1]]), "~"))
        inner <- inner[[length(inner)]]
      if (!is.call(inner) || !identical(as.character(inner[[1]]), "|"))
        stop("dsum : ecrire dsum(~ <structure> | <facteur de section>) ; recu ",
             deparse(inner)[1])
      gauche <- inner[[2]]; sec_var <- as.character(inner[[3]])
      if (!sec_var %in% names(data))
        stop("dsum : colonne de section '", sec_var, "' absente de `data`.")
      fsec <- factor(data[[sec_var]])
      # `+` au premier niveau = structures DIFFERENTES ; sinon la meme partout.
      formules <- attr(terms(stats::as.formula(call("~", gauche)),
                             keep.order = TRUE), "term.labels")
      bloc <- .rx_dsum_blocs(gauche)
      lv <- if (!is.null(args$levels)) eval(args$levels, envir = data) else NULL
      if (length(bloc) > 1L) {
        if (is.null(lv) || length(lv) != length(bloc))
          stop("dsum : ", length(bloc), " structures separees par '+' mais ",
               if (is.null(lv)) "aucun" else length(lv),
               " groupe(s) dans `levels`. Donner levels = list(...) de meme longueur.")
      } else {
        lv <- as.list(levels(fsec)); bloc <- rep(bloc, length(lv))
      }
      secs <- list()
      for (k in seq_along(bloc)) {
        niv <- as.character(lv[[k]])
        idx <- which(as.character(fsec) %in% niv)
        if (!length(idx))
          stop("dsum : le groupe ", k, " (", paste(niv, collapse = ", "),
               ") ne couvre aucune observation.")
        s <- .rx_parse_residual_1(stats::as.formula(call("~", bloc[[k]])),
                                  data[idx, , drop = FALSE],
                                  trait = .rx_sub(trait, idx, data, names(data)),
                                  unit  = .rx_sub(unit, idx, data, names(data)))
        s$rows <- idx
        s$name <- make.names(paste(niv, collapse = "_"))
        secs[[k]] <- s
      }
      vus <- unlist(lapply(secs, `[[`, "rows"))
      if (length(vus) != nrow(data) || anyDuplicated(vus))
        stop("dsum : les sections ne partitionnent pas les ", nrow(data),
             " observations (", length(vus), " lignes citees, ",
             length(unique(vus)), " distinctes). Une ligne oubliee sortirait de V ",
             "avec une variance nulle ; une ligne doublee la compterait deux fois.")
      r <- secs[[1]]; r$sections <- secs; r$rows <- NULL; r$name <- NULL
      return(r)
    }
  }
  .rx_parse_residual_1(residual, data, trait = trait, unit = unit)
}

# Termes du premier niveau d'une formule, sans casser les ':' internes.
.rx_dsum_blocs <- function(e) {
  if (is.call(e) && identical(as.character(e[[1]]), "+"))
    return(c(.rx_dsum_blocs(e[[2]]), list(e[[3]])))
  list(e)
}

# Sous-ensemble d'un `trait`/`unit` fourni en NOM de colonne ou en VECTEUR.
.rx_sub <- function(x, idx, data, noms) {
  if (is.null(x)) return(NULL)
  if (is.character(x) && length(x) == 1L && x %in% noms) return(x)
  x[idx]
}

.rx_parse_residual_1 <- function(residual, data, trait = NULL, unit = NULL) {
  # `trait` et `unit` peuvent arriver comme NOM de colonne ou comme VECTEUR.
  # Le parseur a besoin du NOM pour reconnaitre `diag(trait)` dans la formule :
  # sans lui, `trait` etait pris pour un facteur d'UNITE et la residuelle
  # devenait iid avec 4 "unites" au lieu de 180 — un modele completement
  # different, sans aucune erreur visible. On retrouve donc le nom quand seul
  # le vecteur est fourni, en le cherchant parmi les colonnes de `data`.
  .nom_de <- function(x) {
    if (is.character(x) && length(x) == 1L && x %in% names(data)) return(x)
    if (is.null(x)) return(NULL)
    for (nm in names(data))
      if (length(data[[nm]]) == length(x) &&
          identical(as.character(data[[nm]]), as.character(x))) return(nm)
    NULL
  }
  # `trait` / `unit` acceptes en NOM de colonne ou en VECTEUR : on garde les deux
  # formes, le nom pour lire la formule, le vecteur pour construire le modele.
  .vec_de <- function(x) {
    if (is.character(x) && length(x) == 1L && x %in% names(data)) return(data[[x]])
    x
  }
  if (is.character(residual) && length(residual) == 1L &&
      residual %in% c("units", "iid", "diag", "us", "fa")) {
    r <- if (residual == "units") "iid" else residual
    return(rx_residual(r, trait = .vec_de(trait), unit = .vec_de(unit)))
  }
  if (!inherits(residual, "formula")) stop("`residual` : chaine ou formule attendue.")
  env <- environment(residual); if (is.null(env)) env <- parent.frame()
  lab <- attr(terms(residual, keep.order = TRUE), "term.labels")
  if (!length(lab)) stop("`residual` ne contient aucun facteur.")
  facteurs <- unlist(strsplit(lab, ":", fixed = TRUE))
  facteurs <- trimws(facteurs)

  st <- "iid"; rk <- 0L; lvl <- "id"; ordre <- 0L; co <- NULL
  lvl_opts <- NULL; lvl_expr <- NULL; cell_un <- NULL
  nom_trait <- .nom_de(trait)
  nom_unit  <- .nom_de(unit)
  un_fac <- character(0)

  # AR1 x AR1 : reconnu AVANT la boucle, sinon le deuxieme ar1() declenche
  # "deux structures entre unites". C'est la forme la plus courante d'un champ
  # residuel spatial (residual = ~ ar1(row):ar1(col) chez asreml).
  ar1f <- grepl("^ar1\\(", facteurs)
  if (sum(ar1f) == 2L) {
    cols <- vapply(facteurs[ar1f], function(z) as.character(str2lang(z)[[2]]), "")
    if (!all(cols %in% names(data)))
      stop("residual : colonne(s) introuvable(s) : ",
           paste(setdiff(cols, names(data)), collapse = ", "))
    fr <- factor(data[[cols[1]]]); fc <- factor(data[[cols[2]]])
    # niveau = (ligne-1)*n_colonnes + colonne : LIGNE lente, COLONNE rapide,
    # exactement l'ordre du produit de Kronecker cote solveur.
    un <- (as.integer(fr) - 1L) * nlevels(fc) + as.integer(fc)
    # Le controle porte sur le couple (cellule, CARACTERE), pas sur la cellule
    # seule. Sur donnees longues a t caracteres, chaque cellule du champ apparait
    # t fois PAR CONSTRUCTION : R = Sigma_caractere (x) C_cellule est parfaitement
    # reguliere, et le solveur l'accepte (fit.py:_valider_section teste bien le
    # couple). Tester la cellule seule refusait donc us(trait):ar1(row):ar1(col),
    # c'est-a-dire le modele multi-caractere spatial le plus courant, alors que
    # seule la repetition d'un MEME caractere dans une MEME cellule rend R
    # singuliere.
    cle_dup <- if (!is.null(nom_trait)) paste(un, as.integer(factor(data[[nom_trait]])),
                                             sep = "\r") else un
    if (anyDuplicated(cle_dup))
      stop("residual = ~ ar1(", cols[1], "):ar1(", cols[2], ") : ",
           sum(duplicated(cle_dup)), " couple(s) (cellule, caractere) en double. ",
           "Un champ residuel structure exige au plus une observation par cellule ",
           "et par caractere.")
    st2 <- "iid"
    for (fx in facteurs[!ar1f]) {
      if (fx %in% c("units", "id(units)")) next
      e2 <- str2lang(fx)
      if (is.call(e2) && !is.null(nom_trait) &&
          identical(as.character(e2[[2]]), nom_trait))
        st2 <- switch(as.character(e2[[1]]), iid = , idv = , id = "iid",
                      diag = , idh = "diag", us = , corgh = "us", "iid")
    }
    return(rx_residual(st2, trait = if (is.null(nom_trait)) .vec_de(trait) else data[[nom_trait]],
                       unit = un, level = "ar1ar1", dims = c(nlevels(fr), nlevels(fc)),
                       n_unit = nlevels(fr) * nlevels(fc)))
  }
  for (fx in facteurs) {
    if (fx %in% c("units", "id(units)")) next          # aucune structure
    e <- str2lang(fx)
    if (is.name(e)) {                                   # un simple nom de colonne
      un_fac <- c(un_fac, as.character(e)); next
    }
    f  <- as.character(e[[1]])
    a1 <- if (length(e) >= 2L) as.character(e[[2]]) else ""
    args <- as.list(e)[-(1:2)]
    if (f %in% c("iid", "idv", "id", "diag", "idh", "us", "corgh", "fa")) {
      if (!is.null(nom_trait) && identical(a1, nom_trait)) {
        st <- switch(f, iid = , idv = , id = "iid", diag = , idh = "diag",
                     us = , corgh = "us", fa = "fa")
        if (!is.null(args$rank)) rk <- as.integer(eval(args$rank))
      } else if (a1 %in% c("units", "")) {
        next
      } else {
        un_fac <- c(un_fac, a1)
      }
      next
    }
    if (f %in% RX_LEVEL_STRUCTURES) {
      if (lvl != "id")
        stop("residual : deux structures entre unites ('", lvl, "' et '", f,
             "'). Un seul facteur structure les unites, sauf ar1(r):ar1(c), ",
             "ecrit ar1(", a1, ", <colonne>).")
      lvl <- f
      posit <- as.list(e)[-1]
      nposit <- names(posit); if (is.null(nposit)) nposit <- rep("", length(posit))
      posit <- posit[nposit == ""]
      if (!is.null(args$order)) ordre <- as.integer(eval(args$order))
      if (!is.null(args$coord)) co <- as.matrix(eval(args$coord, envir = data, enclos = env))
      # Options de la structure AVANT toute sortie de boucle. Elles etaient
      # lues plus bas, apres le `next` de la branche metrique 2D : une
      # residuelle ~ mtrn(x, y, phi = ..., nu = ...) repartait donc sans aucune
      # option, donc sans aucun parametre estime et avec les valeurs par defaut
      # (phi = 1, nu = 0.5). Quatre jeux de parametres differents donnaient la
      # MEME vraisemblance, a -19.5572007 — le signe qu'aucun n'arrivait.
      if (f == "mtrn") lvl_opts <- .rx_mtrn_opts(args, env)
      if (f == "own") {
        lvl_expr <- as.character(eval(args$expr, env))
        lvl_opts <- c(n_par = as.numeric(eval(args$n_par %||% 1L, env)),
                      normalise = as.numeric(eval(args$normalise %||% TRUE, env)))
      }
      # Metrique 2D en deux colonnes : mtrn(x, y), iexp(x, y)... L'unite est
      # alors la CELLULE (x, y) et les coordonnees en decoulent.
      if (f %in% RX_METRIQUES_2D && length(posit) == 2L && is.null(co)) {
        xv <- eval(posit[[1]], envir = data, enclos = env)
        yv <- eval(posit[[2]], envir = data, enclos = env)
        cle <- paste(xv, yv, sep = "\r")
        u <- !duplicated(cle); ord_u <- order(cle[u]); lev2 <- cle[u][ord_u]
        cell <- as.integer(factor(cle, levels = lev2))
        if (anyDuplicated(cell))
          stop("residual = ~ ", fx, " : ", sum(duplicated(cell)),
               " cellule(s) en double. Un champ residuel structure exige au plus ",
               "une observation par cellule.")
        co <- cbind(as.numeric(xv[u][ord_u]), as.numeric(yv[u][ord_u]))
        cell_un <- cell
        next
      }
      un_fac <- c(un_fac, a1)
      if (f %in% RX_METRIQUES_1D && is.null(co)) co <- NA          # deduit plus bas
      next
    }
    stop("residual : facteur non reconnu '", fx, "'.\n  Attendu : units, ",
         "id/iid/diag/us/fa(<caractere>), ou une structure entre niveaux (",
         paste(setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed")), collapse = "/"), ").")
  }

  un_fac <- setdiff(un_fac, nom_trait)      # le caractere n'est pas une unite
  if (length(un_fac) > 1L)
    stop("residual : plusieurs facteurs d'unite (", paste(un_fac, collapse = ", "), ").")
  un_vec <- if (!is.null(cell_un)) cell_un else
            if (length(un_fac)) data[[un_fac[1]]] else
            if (!is.null(nom_unit)) data[[nom_unit]] else .vec_de(unit)
  tr_vec <- if (!is.null(nom_trait)) data[[nom_trait]] else .vec_de(trait)
  # Metrique 1D sans coordonnees : les niveaux d'unite doivent etre numeriques.
  if (length(co) == 1L && is.na(co)) {
    lev1 <- levels(factor(un_vec))
    co <- suppressWarnings(as.numeric(lev1))
    if (any(!is.finite(co)))
      stop("residual : '", lvl, "' exige des positions numeriques ; les niveaux ",
           "de '", if (length(un_fac)) un_fac[1] else "unit", "' ne le sont pas. ",
           "Fournir coord = <positions>.")
    co <- as.matrix(co)
  }
  rx_residual(st, trait = tr_vec, unit = un_vec, rank = rk,
              level = lvl, order = ordre, coord = co,
              opts = lvl_opts, expr = lvl_expr,
              n_unit = if (!is.null(cell_un)) length(unique(cell_un)) else NULL)
}

# ==============================================================================
# 3. MODELE
# ==============================================================================
rx_model <- function(y, X, terms, residual = rx_residual(), name = "modele") {
  n <- length(y)
  if (!is.matrix(X)) X <- matrix(X, nrow = n)
  if (nrow(X) != n) stop("X a ", nrow(X), " lignes pour ", n, " observations.")
  # Un modele SANS terme aleatoire est licite des lors que la residuelle est
  # structuree (residual = ~ ar1(row):ar1(col)) : c'est le modele spatial le
  # plus courant en essai au champ.
  if (!length(terms) && identical(residual$level, "id") &&
      identical(residual$struct, "iid"))
    stop("aucun terme aleatoire et residuelle iid : il n'y a rien a estimer.")
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
    # une inversion dense — exactement ce que la voie creuse evite. On le dit ici
    # plutot que de laisser le solveur echouer plus loin sur un LK manquant.
    if (!is.null(tm$Kinv) && is.null(tm$LK))
      attr(terms, "creux_seulement") <- TRUE
    if (nrow(tm$Zl[[1]]) != n)
      stop("terme '", tm$name, "' : incidence a ", nrow(tm$Zl[[1]]), " lignes pour ", n, " obs.")
  }
  .t_de <- function(r) if (is.null(r$trait)) 1L else nlevels(factor(r$trait))
  for (s in (residual$sections %||% list(residual))) {
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
  tr <- residual$trait
  t_res <- .t_de(residual)
  # X de rang plein : sinon log|X'V^-1X| est -Inf et la REML n'est pas definie.
  r <- qr(X)$rank
  if (r < ncol(X))
    stop("X est de rang ", r, " pour ", ncol(X), " colonnes : retirer les colonnes redondantes.")
  if (!is.null(residual$n_unit)) residual$n_unit <- as.integer(residual$n_unit)
  X_assign <- attr(X, "assign"); X_termes <- attr(X, "termes")
  structure(list(name = name, y = as.numeric(y), X = X, terms = terms,
                 X_assign = X_assign, X_termes = X_termes,
                 residual = residual, n = n, t_res = t_res,
                 # La residuelle porte aussi des parametres de NIVEAUX (le rho
                 # d'un ar1 residuel), et une residuelle en sections en porte un
                 # jeu par section. Les omettre affichait 2 parametres pour un
                 # champ ar1 x ar1 qui en a 4.
                 n_par = sum(vapply(terms, `[[`, 1L, "n_par")) +
                   sum(vapply(residual$sections %||% list(residual), function(s)
                     rx_n_params(s$struct, .t_de(s), s$rank) +
                       rx_n_level(s$level %||% "id", s$order %||% 0L, s$opts), 1L))),
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
  cat(sprintf("  residuelle     %-5s t=%d  (%d parametre%s)\n", x$residual$struct, x$t_res,
              rx_n_params(x$residual$struct, x$t_res, x$residual$rank),
              if (rx_n_params(x$residual$struct, x$t_res, x$residual$rank) > 1) "s" else ""))
  cat(sprintf("  total : %d parametres de variance\n", x$n_par))
  invisible(x)
}

# ==============================================================================
# 4. SERIALISATION (binaire brut + manifeste ; aucune dependance Python en R)
# ==============================================================================
rx_export <- function(model, dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  man <- list()
  put <- function(name, x, dtype = "f8") {
    if (dtype == "str") {
      writeLines(as.character(x), file.path(dir, paste0(name, ".txt")))
      man[[length(man) + 1L]] <<- list(name = name, dtype = "str", shape = I(length(x)))
      return(invisible())
    }
    shp <- if (is.matrix(x)) dim(x) else length(x)
    con <- file(file.path(dir, paste0(name, ".bin")), "wb")
    if (dtype == "f8") writeBin(as.double(as.vector(x)), con, size = 8)
    else               writeBin(as.integer(as.vector(x)), con, size = 4)
    close(con)
    # I() force un TABLEAU JSON meme pour une dimension unique : sans lui,
    # auto_unbox rend "shape":120 au lieu de "shape":[120].
    man[[length(man) + 1L]] <<- list(name = name, dtype = dtype, shape = I(shp))
    invisible()
  }
  put("y", model$y); put("X", model$X)
  # Groupement des colonnes de X par TERME du modele (attribut `assign` de
  # model.matrix). Sans lui, un test de Wald porterait sur chaque colonne prise
  # isolement, ce qui n'a pas de sens des qu'un facteur a plus de deux niveaux :
  # asreml teste le terme ENTIER, a plusieurs degres de liberte.
  if (!is.null(model$X_assign)) {
    put("X_assign", as.integer(model$X_assign), "i4")
    put("X_termes", as.character(model$X_termes), "str")
  }
  put("term_names", vapply(model$terms, `[[`, "", "name"), "str")
  for (tm in model$terms) {
    p <- tm$name
    # Z empile en n x (t*q), colonne = (a-1)*q + niveau, puis triplet COO.
    Zc <- do.call(cbind, tm$Zl)
    S  <- methods::as(Zc, "TsparseMatrix")
    put(paste0("term_", p, "_zi"), S@i, "i4")      # 0-based, comme attendu en Python
    put(paste0("term_", p, "_zj"), S@j, "i4")
    put(paste0("term_", p, "_zx"), S@x)
    put(paste0("term_", p, "_t"), tm$t, "i4")
    put(paste0("term_", p, "_q"), tm$q, "i4")
    put(paste0("term_", p, "_rank"), tm$rank, "i4")
    put(paste0("term_", p, "_struct"), tm$struct, "str")
    put(paste0("term_", p, "_lvl"), tm$level, "str")
    if (!is.null(tm$order) && tm$order > 0L)
      put(paste0("term_", p, "_lvlorder"), as.integer(tm$order), "i4")
    if (!is.null(tm$coord)) put(paste0("term_", p, "_coord"), tm$coord)
    if (!is.null(tm$dims)) put(paste0("term_", p, "_dims"), as.integer(tm$dims), "i4")
    if (!is.null(tm$opts) && length(tm$opts)) {
      put(paste0("term_", p, "_lvloptk"), names(tm$opts), "str")
      put(paste0("term_", p, "_lvloptv"), as.numeric(tm$opts))
    }
    if (!is.null(tm$expr)) put(paste0("term_", p, "_lvlexpr"), tm$expr, "str")
    if (!is.null(tm$parts) && length(tm$parts)) {
      # `parts` est une liste de (famille, dimension). On la serialise en deux
      # vecteurs paralleles : les noms d'un cote, les dimensions de l'autre.
      put(paste0("term_", p, "_lvlpartk"),
          vapply(tm$parts, function(x) as.character(x[[1]]), character(1)), "str")
      put(paste0("term_", p, "_lvlpartq"),
          vapply(tm$parts, function(x) as.integer(x[[2]]), integer(1)), "i4")
    }
    if (!is.null(tm$LK)) put(paste0("term_", p, "_LK"), tm$LK)
    if (!is.null(tm$levels)) put(paste0("term_", p, "_levels"), tm$levels, "str")
  }
  # Une section (ou la residuelle entiere) : memes champs, meme prefixe.
  put_res <- function(pre, r, nlig) {
    put(paste0(pre, "_struct"), r$struct, "str")
    put(paste0(pre, "_lvl"), r$level %||% "id", "str")
    if (!is.null(r$order) && r$order > 0L)
      put(paste0(pre, "_lvlorder"), as.integer(r$order), "i4")
    if (!is.null(r$coord)) put(paste0(pre, "_coord"), as.matrix(r$coord))
    if (!is.null(r$dims)) put(paste0(pre, "_dims"), as.integer(r$dims), "i4")
    if (!is.null(r$opts) && length(r$opts)) {
      put(paste0(pre, "_lvloptk"), names(r$opts), "str")
      put(paste0(pre, "_lvloptv"), as.numeric(r$opts))
    }
    if (!is.null(r$expr)) put(paste0(pre, "_lvlexpr"), r$expr, "str")
    put(paste0(pre, "_t"), if (is.null(r$trait)) 1L else nlevels(factor(r$trait)), "i4")
    put(paste0(pre, "_rank"), r$rank, "i4")
    put(paste0(pre, "_trait"),
        if (is.null(r$trait)) rep(0L, nlig) else as.integer(factor(r$trait)) - 1L, "i4")
    put(paste0(pre, "_unit"),
        if (is.null(r$unit)) seq_len(nlig) - 1L else as.integer(factor(r$unit)) - 1L, "i4")
  }
  put_res("res", model$residual, model$n)
  secs <- model$residual$sections
  put("res_nsec", as.integer(length(secs)), "i4")
  if (length(secs)) {
    put("res_secnames", vapply(secs, function(s) s$name %||% "section", ""), "str")
    for (k in seq_along(secs)) {
      s <- secs[[k]]
      put_res(sprintf("res_s%d", k - 1L), s, length(s$rows))
      put(sprintf("res_s%d_rows", k - 1L), as.integer(s$rows) - 1L, "i4")
    }
  }
  writeLines(jsonlite::toJSON(man, auto_unbox = TRUE), file.path(dir, "manifest.json"))
  invisible(dir)
}

# ==============================================================================
# 5. AJUSTEMENT : delegation au solveur JAX
# ==============================================================================
#' @param backend "auto" | "gpu" | "cpu" — choix de MACHINE, jamais de modele.
#' @param vpredict expressions sur les composantes, facon asreml :
#'   c(h2 = "V1/(V1+V2)", rg = "V2/sqrt(V1*V3)"). Les Vi sont numerotees dans
#'   l'ordre rendu par `fit$composantes_noms`.
#' @param wald TRUE pour les tests de Wald sur les effets fixes
#' @param fixed_theta indices (1-based) des parametres a FIXER a leur depart
rx_fit <- function(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
                   maxiter = 3000L, polish = 25L, n_restarts = 0L,
                   hessian = TRUE, blups = TRUE, pev = FALSE,
                   vpredict = NULL, wald = FALSE,
                   kenward_roger = FALSE, predict = NULL, theta_init = NULL,
                   fixed_theta = NULL, floor = -12, ceil = 12,
                   verbose = TRUE, keep = FALSE) {
  # LES BORNES DE theta SE DEMANDENT ICI, ET NULLE PART AILLEURS. Le CLI les
  # acceptait (--floor, --ceil) mais aucun argument R ne les transmettait : un
  # appelant qui croyait poser un plancher a -8 obtenait -12 sans un mot, et
  # un pipeline a enregistre PAR_FLOOR=-8 dans ses specs pendant que le solveur
  # travaillait a -12. theta est un LOG D'ECART-TYPE : var = exp(2 theta), donc
  # floor = -12 vaut var = 3,8e-11. Les valeurs effectives reviennent dans le
  # resultat (par_floor, par_ceil) : les lire la, ne jamais les recoder.
  floor <- as.numeric(floor); ceil <- as.numeric(ceil)
  if (length(floor) != 1L || length(ceil) != 1L || !is.finite(floor) || !is.finite(ceil) ||
      floor >= ceil)
    stop("rx_fit : floor et ceil doivent etre deux scalaires finis avec floor < ceil.",
         call. = FALSE)
  # UN MODELE DECLARE PAR SA PRECISION N'EST PAS AJUSTABLE ICI, ET LE TAIRE
  # COUTE CHER. rx_model() pose l'attribut `creux_seulement` des qu'un terme
  # porte Kinv sans LK ; personne ne le lisait, si bien que ce moteur ajustait
  # le modele en traitant le terme comme INDEPENDANT (level "id", ligne "K=I"
  # dans le journal) sans un mot. Cout mesure : la variance genetique de
  # Hauteur.4 sortait a 0,3019 -- exactement celle de l'ajustement univarie
  # SANS parente (0,3013) -- au lieu de 0,1849, celle de l'univarie AVEC
  # parente (0,1839). Toutes les heritabilites en decoulant etaient gonflees
  # d'un facteur 1,63, sans qu'aucun diagnostic ne le signale.
  if (isTRUE(attr(model$terms, "creux_seulement")))
    stop("rx_fit : ce modele contient un ou plusieurs termes declares par leur ",
         "PRECISION (Kinv) sans facteur de K. Le moteur dense a besoin de K et ",
         "ignorerait silencieusement la parente, en estimant les effets comme ",
         "INDEPENDANTS. Fournir K a rx_term(), ou ajuster avec rx_fit_sparse().",
         call. = FALSE)

  backend <- match.arg(backend)
  if (is.null(dir)) { dir <- tempfile("rx_"); on.exit(if (!keep) unlink(dir, recursive = TRUE)) }
  rx_export(model, dir)
  # predict : L (l x p) et, par terme aleatoire, M (l x t*q). Ecrits en
  # column-major comme tout le reste du paquet.
  if (!is.null(predict)) {
    wbin <- function(nm, x) { con <- file(file.path(dir, nm), "wb")
      writeBin(as.double(as.vector(as.matrix(x))), con, size = 8); close(con) }
    wbin("pred_L.bin", predict$L)
    for (nm in names(predict$M %||% list())) wbin(paste0("pred_M_", nm, ".bin"), predict$M[[nm]])
  }
  # Depart a chaud. Avec maxiter = 0 et polish = 0, c'est une simple EVALUATION
  # de la vraisemblance au theta fourni : le seul moyen de savoir si deux
  # solveurs calculent la meme fonction ou s'arretent seulement a des endroits
  # differents.
  if (!is.null(theta_init)) {
    con <- file(file.path(dir, "in_theta.bin"), "wb")
    writeBin(as.double(theta_init), con, size = 8); close(con)
  }
  py <- rx_python_cmd()
  .rx_python_assurer(py)
  a <- c(rx_solver_args(), dir, "--backend", backend, "--maxiter", maxiter, "--polish", polish,
         "--restarts", n_restarts,
         "--floor", format(floor, digits = 15), "--ceil", format(ceil, digits = 15),
         if (!is.null(vpredict)) c("--vpredict",
           paste(sprintf("%s=%s", names(vpredict), unlist(vpredict)), collapse = ";")) else NULL,
         if (isTRUE(wald)) "--wald" else NULL,
         if (isTRUE(kenward_roger)) "--kenward-roger" else NULL,
         if (!is.null(predict)) "--predict" else NULL,
         if (!is.null(theta_init)) "--theta-in" else NULL,
         if (!is.null(fixed_theta)) c("--fixed-theta",
           paste(as.integer(fixed_theta), collapse = ",")) else NULL,
         if (!hessian) "--no-hessian" else NULL, if (!blups) "--no-blups" else NULL,
         # DEFAUT A FALSE, deliberement. La PEV coute une matrice (t*q) x n par
         # terme : gratuite pour un modele univarie a quelques centaines de
         # genotypes, lourde des que t*q monte. Elle se demande donc, elle ne
         # s'impose pas. Sans elle H2_Cullis n'est pas calculable.
         # pev accepte TRUE (tous les termes) ou un VECTEUR DE NOMS. Sur le
         # modele IGE complet, 21 des 23 termes sont des nuisances dont la
         # variance d'erreur de prediction n'interesse personne, et les
         # demander toutes a epuise la memoire lors du premier essai reel.
         if (isTRUE(pev)) "--pev" else
           if (is.character(pev) && length(pev)) c("--pev-termes", paste(pev, collapse = ",")) else NULL,
         if (!verbose) "--quiet" else NULL)
  st <- system2(py[1], shQuote(c(py[-1], as.character(a))),
                stdout = if (verbose) "" else TRUE, stderr = if (verbose) "" else TRUE)
  if (!verbose && !is.null(attr(st, "status")) && attr(st, "status") != 0) {
    cat(st, sep = "\n"); stop("solveur REML : echec")
  }
  if (verbose && !identical(as.integer(st), 0L)) stop("solveur REML : echec")
  rx_read_result(dir, model = model, fixed_theta = fixed_theta)
}

# ERREURS-TYPES SUR theta. Le Hessien rendu est celui de -2 logL, donc
# cov(theta) = 2 H^-1 sur le sous-espace LIBRE : ni a une borne (lue dans
# par_floor / par_ceil, jamais recodee), ni fixe par fixed_theta. Ailleurs NA.
# Une valeur propre negative de H_f signale un point de selle : les erreurs-types
# y sont NA plutot qu'un nombre sans sens. Verifie contre le vpredict du solveur
# (test-ratios.R) et contre 06_theta_se.csv du chapitre 3.
rx_se_theta <- function(theta, hessian, par_floor = -12, par_ceil = 12,
                        fixed_theta = NULL, tol_bound = 1e-7) {
  p <- length(theta)
  se <- rep(NA_real_, p)
  if (is.null(hessian) || !p) return(se)
  H <- as.matrix(hessian)
  if (nrow(H) != p || ncol(H) != p) return(se)
  libre <- !(theta <= par_floor + tol_bound | theta >= par_ceil - tol_bound)
  if (length(fixed_theta)) libre[as.integer(fixed_theta)] <- FALSE
  if (!any(libre)) return(se)
  Hf <- (H[libre, libre, drop = FALSE] + t(H[libre, libre, drop = FALSE])) / 2
  if (!all(is.finite(Hf))) return(se)
  ev <- eigen(Hf, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-12 * max(abs(ev), 1)) return(se)
  Vi <- tryCatch(solve(Hf), error = function(e) NULL)
  if (is.null(Vi)) return(se)
  d <- diag(Vi)
  se[libre] <- ifelse(d > 0, sqrt(2 * d), NA_real_)
  se
}

rx_here <- function() if (requireNamespace("here", quietly = TRUE)) here::here("R") else "R"

#' Repertoire de l'interface R, quelle que soit la facon dont elle a ete chargee.
rx_pkg_dir <- function() {
  if (!is.na(.RX_FILE_DIR)) return(.RX_FILE_DIR)
  rx_here()
}

#' Localisation du solveur Python : les arguments a passer AVANT le repertoire
#' du paquet serialise.
#'
#' Trois voies, dans cet ordre :
#'   1. $RX_CLI            explicite : chemin d'un cli.py, ou "-m remlax.cli"
#'   2. paquet installe    `-m remlax.cli` si `import remlax` reussit — voie a
#'                         preferer, elle ne suppose rien du repertoire courant
#'   3. arborescence       src/remlax/cli.py a cote de R/remlax.R (depot non
#'                         installe), ou l'ancien scripts/gpu/remlax/cli.py
rx_solver_args <- function() {
  if (nzchar(Sys.getenv("RX_CLI"))) return(strsplit(Sys.getenv("RX_CLI"), " +")[[1]])
  py <- rx_python_cmd()
  ok <- suppressWarnings(try(system2(py[1], shQuote(c(py[-1], "-c", "import remlax")),
                                     stdout = FALSE, stderr = FALSE), silent = TRUE))
  if (identical(as.integer(ok), 0L)) return(c("-m", "remlax.cli"))
  base <- rx_pkg_dir()
  cands <- c(file.path(base, "..", "src", "remlax", "cli.py"),
             file.path(base, "..", "scripts", "gpu", "remlax", "cli.py"),
             file.path("src", "remlax", "cli.py"))
  for (p in cands) if (file.exists(p)) return(normalizePath(p))
  stop("solveur Python introuvable. Installer le paquet (`pip install -e .`), ",
       "ou definir $RX_CLI, ou lancer depuis la racine du depot.")
}

#' Commande Python : $RX_PY, sinon $IGE_JAX_CMD, sinon $IGE_JAX_SIF, sinon python3
rx_python_cmd <- function() {
  if (nzchar(Sys.getenv("RX_PY"))) return(strsplit(Sys.getenv("RX_PY"), " +")[[1]])
  if (nzchar(Sys.getenv("IGE_JAX_CMD"))) return(strsplit(Sys.getenv("IGE_JAX_CMD"), " +")[[1]])
  sif <- Sys.getenv("IGE_JAX_SIF")
  if (nzchar(sif)) return(c("apptainer", "exec", "--nv", sif, "python3"))
  "python3"
}

# Cache des verifications d'interpreteur, par commande. Un `import jax` coute
# une a deux secondes : le payer a chaque ajustement d'une grille serait
# absurde, le payer une fois par session est invisible.
.rx_python_cache <- new.env(parent = emptyenv())

#' L'interpreteur choisi porte-t-il jax, numpy et scipy ?
#'
#' Sans ce controle, un Python sans jax donnait un traceback Python de dix
#' lignes finissant par « No module named 'jax' » puis « solveur REML : echec »,
#' sans jamais dire QUEL interpreteur avait ete essaye ni COMMENT en designer
#' un autre. C'est arrive au premier essai du paquet installe : python3 du
#' systeme, pas de jax. Le diagnostic doit nommer l'interpreteur, la variable
#' RX_PY et la commande d'installation.
#'
#' @param py commande Python decoupee (defaut : rx_python_cmd()).
#' @param quiet ne rien imprimer.
#' @return liste(ok, python, versions, message), invisible si ok.
rx_python_check <- function(py = rx_python_cmd(), quiet = FALSE) {
  cle <- paste(py, collapse = " ")
  if (!is.null(.rx_python_cache[[cle]])) return(invisible(.rx_python_cache[[cle]]))
  code <- paste0("import sys\n",
                 "try:\n    import jax, numpy, scipy\n",
                 "except Exception as e:\n    print('MANQUE', type(e).__name__, e); sys.exit(3)\n",
                 "print('OK', jax.__version__, numpy.__version__, scipy.__version__, sys.executable)\n")
  out <- suppressWarnings(tryCatch(
    system2(py[1], shQuote(c(py[-1], "-c", code)), stdout = TRUE, stderr = TRUE),
    error = function(e) structure(conditionMessage(e), status = 127L)))
  st <- attr(out, "status") %||% 0L
  ligne <- if (length(out)) out[length(out)] else ""
  res <- list(ok = identical(as.integer(st), 0L) && startsWith(ligne, "OK "),
              python = cle, versions = NULL, message = "")
  if (res$ok) {
    v <- strsplit(ligne, " ")[[1]]
    res$versions <- c(jax = v[2], numpy = v[3], scipy = v[4], executable = v[5])
  } else {
    res$message <- paste0(
      "remlax : l'interpreteur Python \"", cle, "\" ne porte pas jax, numpy et scipy",
      if (nzchar(ligne)) paste0(" (", ligne, ")") else "", ".\n",
      "  Designer un Python qui les a : Sys.setenv(RX_PY = \"/chemin/vers/python\")\n",
      "  ou dans ~/.Renviron : RX_PY=/chemin/vers/python\n",
      "  Pour en creer un : remlax::rx_install_python() (ou pip install jax numpy scipy).")
  }
  .rx_python_cache[[cle]] <- res
  if (!quiet && !res$ok) message(res$message)
  invisible(res)
}

# Arret net avant tout lancement du solveur si l'interpreteur ne convient pas.
.rx_python_assurer <- function(py) {
  r <- rx_python_check(py, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  invisible(r)
}

#' Creer un environnement virtuel Python pour le solveur
#'
#' Cree `dir` par `python -m venv`, y installe jax, numpy et scipy par pip
#' (`jax[cuda12]` si cuda = TRUE), verifie l'import, et imprime la ligne
#' RX_PY a poser. Ne modifie ni ~/.Renviron ni la session : l'utilisateur
#' choisit ou et comment la conserver. Sans reticulate, par choix.
#'
#' @param dir repertoire du venv (defaut ~/.remlax/venv).
#' @param cuda installer la version CUDA 12 de jax.
#' @param python interpreteur de base (>= 3.10) servant a creer le venv.
#' @param upgrade reinstaller si le venv existe deja.
#' @return chemin de l'interpreteur cree, invisible.
rx_install_python <- function(dir = path.expand("~/.remlax/venv"), cuda = FALSE,
                              python = "python3", upgrade = FALSE) {
  # Chemin ABSOLU : une RX_PY relative se resoudrait contre le repertoire
  # courant de chaque session, donc tantot un venv, tantot rien.
  dir <- normalizePath(dir, mustWork = FALSE)
  exe <- file.path(dir, if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python")
  if (!file.exists(exe) || upgrade) {
    if (!file.exists(exe)) {
      message("remlax : creation du venv ", dir)
      st <- system2(python, c("-m", "venv", shQuote(dir)))
      if (!identical(as.integer(st), 0L))
        stop("remlax : `", python, " -m venv` a echoue (code ", st, "). Python >= 3.10 requis.",
             call. = FALSE)
    }
    paquets <- c(if (cuda) "jax[cuda12]>=0.4.30" else "jax>=0.4.30", "numpy>=1.24", "scipy>=1.10")
    message("remlax : pip install ", paste(paquets, collapse = " "))
    st <- system2(exe, c("-m", "pip", "install", "--upgrade", "--quiet", shQuote(paquets)))
    if (!identical(as.integer(st), 0L))
      stop("remlax : pip install a echoue (code ", st, ").", call. = FALSE)
  }
  rm(list = ls(.rx_python_cache), envir = .rx_python_cache)
  r <- rx_python_check(exe, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  message("remlax : interpreteur pret : ", exe, "\n",
          "  jax ", r$versions[["jax"]], ", numpy ", r$versions[["numpy"]],
          ", scipy ", r$versions[["scipy"]], "\n",
          "  Pour cette session : Sys.setenv(RX_PY = \"", exe, "\")\n",
          "  Pour toujours, dans ~/.Renviron : RX_PY=", exe)
  invisible(exe)
}

rx_read_result <- function(dir, model = NULL, fixed_theta = NULL) {
  f <- file.path(dir, "result.json")
  if (!file.exists(f)) stop("resultat introuvable : ", f)
  r <- jsonlite::fromJSON(f, simplifyVector = TRUE)
  rd <- function(nm) { p <- file.path(dir, paste0(nm, ".bin"))
    if (file.exists(p)) readBin(p, "double", n = file.size(p) %/% 8, size = 8) else NULL }
  # ATTENTION : ACCES EXACTS, PAS `$`. `$` fait du APPARIEMENT PARTIEL sur les
  # listes. Depuis que le resultat porte un champ `sigmas_res`, l'expression
  # `r$sigmas` (qui n'existe pas encore a ce stade) s'appariait sur
  # `sigmas_res` : les matrices des termes etaient ajoutees a la liste des
  # residuelles, et `fit$sigmas[[1]]` rendait la RESIDUELLE au lieu du premier
  # terme. Toutes les comparaisons positionnelles des suites basculaient d'un
  # cran, sans aucune erreur — la logLik, elle, restait juste.
  r[["theta"]] <- rd("out_theta"); r[["beta"]] <- rd("out_beta")
  p_ <- length(r[["beta"]])
  v <- rd("out_vbeta");    if (!is.null(v)) r[["vbeta"]]    <- matrix(v, p_, p_)
  v <- rd("out_vbeta_kr"); if (!is.null(v)) r[["vbeta_kr"]] <- matrix(v, p_, p_)
  v <- rd("out_pred_cov")
  if (!is.null(v)) { k_ <- sqrt(length(v))
    r[["predictions"]][["cov"]] <- matrix(v, k_, k_) }
  r[["sigmas"]] <- list(); r[["sigmas_res"]] <- list(); r[["blups"]] <- list()
  for (nm in names(r[["sigma_dims"]])) {
    v <- rd(paste0("out_sigma_", nm))
    if (!is.null(v)) r[["sigmas"]][[nm]] <- matrix(v, r[["sigma_dims"]][[nm]][1],
                                                   r[["sigma_dims"]][[nm]][2])
  }
  for (nm in names(r[["sigma_res_dims"]])) {
    v <- rd(paste0("out_sigmares_", nm))
    if (!is.null(v)) r[["sigmas_res"]][[nm]] <- matrix(v, r[["sigma_res_dims"]][[nm]][1],
                                                       r[["sigma_res_dims"]][[nm]][2])
  }
  v <- rd("out_sigma_res")
  if (!is.null(v)) r[["sigma_res"]] <- matrix(v, sqrt(length(v)), sqrt(length(v)))
  # LOADINGS ET PEV, ajoutes pour le pipeline IGE. Sigma assemblee ne suffit pas
  # a un tableau de loadings : sa decomposition n'est pas unique. Et H2_Cullis
  # n'est pas calculable sans la variance d'erreur de prediction.
  r[["loadings"]] <- list(); r[["pev"]] <- list()
  for (nm in names(r[["loadings_dims"]])) {
    v <- rd(paste0("out_loadings_", nm))
    if (is.null(v)) next
    L <- matrix(v, r[["loadings_dims"]][[nm]][1], r[["loadings_dims"]][[nm]][2])
    ps <- if (nm %in% names(r[["psi_dims"]])) rd(paste0("out_psi_", nm)) else NULL
    # `psi` est la VARIANCE specifique, convention des tableaux d'asreml : la
    # diagonale de Sigma vaut somme(V_k^2) + psi. Une structure `rr` est un rang
    # reduit pur et n'en a pas : NULL, jamais des zeros qui se liraient comme
    # une mesure.
    r[["loadings"]][[nm]] <- list(Lambda = L, psi = ps)
  }
  for (nm in names(r[["pev_dims"]])) {
    v <- rd(paste0("out_pev_", nm))
    if (!is.null(v)) r[["pev"]][[nm]] <- matrix(v, r[["pev_dims"]][[nm]][1],
                                                r[["pev_dims"]][[nm]][2])
  }
  for (nm in names(r[["blup_dims"]])) {
    v <- rd(paste0("out_blup_", nm))
    if (!is.null(v)) r[["blups"]][[nm]] <- matrix(v, r[["blup_dims"]][[nm]][1],
                                                  r[["blup_dims"]][[nm]][2])
  }
  # LE HESSIEN. cli.py l'ECRIT deja en binaire a cote (out_hessian.bin), mais
  # ce lecteur ne le reprenait pas : cote R seul le drapeau derive
  # conv_hessien_ok survivait. Sans la matrice, un verdict "12 directions quasi
  # nulles sur 57" n'est pas actionnable — les vecteurs propres disent QUELS
  # parametres ne sont pas identifies. Mesure sur le modele a 5 caracteres :
  # 0 valeur propre negative, 12 directions quasi nulles, conditionnement 2,4e9.
  v <- rd("out_hessian")
  if (!is.null(v)) {
    q_ <- as.integer(round(sqrt(length(v))))
    if (q_ * q_ == length(v)) r[["hessian"]] <- matrix(v, q_, q_)
    else warning(sprintf("out_hessian a %d valeurs, non carre : ignore", length(v)))
  }
  # Erreurs-types sur theta, sur le sous-espace libre ; NA sans Hessien.
  r[["fixed_theta"]] <- if (length(fixed_theta)) as.integer(fixed_theta) else integer(0)
  r[["se_theta"]] <- rx_se_theta(r[["theta"]], r[["hessian"]],
                                 r[["par_floor"]] %||% -12, r[["par_ceil"]] %||% 12,
                                 r[["fixed_theta"]])
  # Noms des colonnes de Sigma et des BLUP, quand les termes en portent.
  if (!is.null(model) && !is.null(model[["terms"]])) {
    for (tm in model[["terms"]]) {
      cn <- tm[["colnames"]]; nm <- tm[["name"]]
      S <- r[["sigmas"]][[nm]]
      if (!is.null(S) && !is.null(cn) && length(cn) == nrow(S))
        dimnames(r[["sigmas"]][[nm]]) <- list(cn, cn)
      U <- r[["blups"]][[nm]]
      if (!is.null(U)) {
        if (!is.null(cn) && length(cn) == ncol(U)) colnames(r[["blups"]][[nm]]) <- cn
        lv <- tm[["levels"]]
        if (!is.null(lv) && length(lv) == nrow(U)) rownames(r[["blups"]][[nm]]) <- lv
      }
    }
  }
  class(r) <- "rx_fit"; r
}

print.rx_fit <- function(x, ...) {
  cat(sprintf("Ajustement REML (%s) : logLik %.6f | %d parametres | %d obs | %.1f s\n",
              x$backend, x$logLik, x$n_par, x$n_obs, x$secondes))
  cat(sprintf("  max|grad| %.2e | decrement de Newton %.2e | %d valeur(s) propre(s) negative(s)\n",
              x$max_grad, x$newton_decrement %||% NA, x$n_neg_eig %||% NA))
  for (nm in names(x$sigmas)) {
    S <- x$sigmas[[nm]]
    cat(sprintf("  Sigma[%s] %dx%d, diagonale : %s\n", nm, nrow(S), ncol(S),
                paste(format(diag(S), digits = 4), collapse = " ")))
  }
  if (!is.null(x$sigma_res))
    cat(sprintf("  residuelle : %s\n", paste(format(diag(as.matrix(x$sigma_res)), digits = 4), collapse = " ")))
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
  invisible(x)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# ==============================================================================
# 6. INTERFACE PAR FORMULE, DANS L'ESPRIT D'ASREML
# ------------------------------------------------------------------------------
#     rx_reml(fixed    = y ~ 1 + traitement,
#             random   = ~ vm(genotype, K) + iid(bloc),
#             residual = ~ units,
#             data     = df)
#
# GRAMMAIRE des termes aleatoires. Chaque terme est un appel dont le premier
# argument est la colonne de groupement :
#     iid(f)             variance scalaire                       Sigma = s2
#     vm(f, K)           idem, avec parente K entre niveaux       (comme asreml)
#     diag(f)            une variance par caractere               Sigma diagonale
#     us(f)              covariance libre entre caracteres        Sigma us
#     fa(f, rank = k)    factor-analytic de rang k
#     un nom nu `f`      equivaut a iid(f)
# `K = ` est accepte par toutes : us(f, K = Kmat) est un modele multi-caractere
# avec parente. `rank =` n'a de sens que pour fa().
#
# MULTI-CARACTERE. Les structures diag/us/fa exigent `trait = ` : les donnees
# sont alors en format LONG (une ligne par unite x caractere) et l'incidence de
# chaque colonne de Sigma est construite automatiquement. Ce format est choisi
# plutot qu'un cbind() de reponses parce qu'il gere sans rien de special les
# caracteres mesures sur des sous-ensembles differents d'unites — ce qui est la
# regle des que le phenotypage n'est pas complet.
#
# CE QUE LA FORMULE NE SAIT PAS DIRE : une incidence PONDEREE (voisinages,
# covariables continues par niveau). Passer alors par rx_term(Z = <matrice>)
# et rx_model() : c'est la meme machinerie, sans le sucre.
# ==============================================================================

# --- construction d'une incidence a partir d'une expression -------------------
# Rend list(Z = <matrice n x q ou NULL si facteur>, f = <facteur ou NULL>,
#           levels = <niveaux>, q = <nb niveaux>).
.rx_incidence <- function(e, data, env) {
  if (is.name(e) || is.character(e)) {
    nm <- as.character(e)
    if (!nm %in% names(data)) stop("colonne '", nm, "' absente de `data`.")
    f <- factor(data[[nm]])
    return(list(f = f, Z = NULL, levels = levels(f), q = nlevels(f), nom = nm))
  }
  if (is.call(e) && as.character(e[[1]]) == "mm") {
    Z <- eval(e[[2]], envir = data, enclos = env)
    Z <- if (inherits(Z, "Matrix")) Z else as.matrix(Z)
    lv <- if (!is.null(colnames(Z))) colnames(Z) else as.character(seq_len(ncol(Z)))
    nm <- if (!is.null(e$name)) as.character(e$name) else deparse(e[[2]])[1]
    return(list(f = NULL, Z = Z, levels = lv, q = ncol(Z), nom = make.names(nm)))
  }
  stop("expression d'incidence non reconnue : ", deparse(e),
       "\n  attendu : un nom de colonne, ou mm(<matrice>).")
}

.rx_parse_random <- function(random, data, trait = NULL) {
  if (is.null(random)) return(list())
  env <- environment(random)
  if (is.null(env)) env <- parent.frame()
  tt <- attr(terms(random, keep.order = TRUE, specials = NULL), "term.labels")
  if (!length(tt)) stop("`random` ne contient aucun terme.")
  t_lev <- if (is.null(trait)) NULL else levels(factor(trait))
  n <- nrow(data)

  lapply(tt, function(txt) {
    e <- str2lang(txt)
    fn <- if (is.name(e)) "iid" else as.character(e[[1]])

    # ---- str() : UNE covariance pour PLUSIEURS termes ------------------------
    # Equivalent de str(~ a + b, ~us(2):id(n)) d'asreml. Les termes groupes
    # doivent partager EXACTEMENT les memes niveaux : c'est ce partage qui donne
    # un sens a une covariance entre eux. Chacun garde SON incidence, ce qui
    # permet de correler un effet direct et un effet de voisinage porte par les
    # memes genotypes — le cas qui a motive tout ceci.
    if (fn == "str") {
      inner <- e[[2]]
      if (!inherits(eval(call("~", inner[[2]])), "formula") && !is.call(inner))
        stop("str() : premier argument attendu sous forme ~ a + b")
      lab <- attr(terms(stats::as.formula(inner), keep.order = TRUE), "term.labels")
      if (length(lab) < 2L)
        stop("str() : au moins deux termes attendus, recu ", length(lab), ".")
      args <- as.list(e)[-(1:2)]
      st <- if (!is.null(args$struct)) as.character(eval(args$struct)) else "us"
      K  <- if (!is.null(args$K)) eval(args$K, envir = data, enclos = env) else NULL
      rk <- if (!is.null(args$rank)) as.integer(eval(args$rank)) else 0L
      # Nom lisible : on prend le nom de chaque incidence (colonne ou mm(name=)),
      # pas le texte brut de l'expression, qui donnait "gid_mmZwnamepente".
      nm <- if (!is.null(args$name)) as.character(args$name) else NA_character_
      inc <- lapply(lab, function(l) .rx_incidence(str2lang(l), data, env))
      if (is.na(nm)) nm <- paste(vapply(inc, `[[`, "", "nom"), collapse = "_")
      lv0 <- inc[[1]]$levels
      for (k in seq_along(inc)) if (!identical(inc[[k]]$levels, lv0))
        stop("str() : le terme '", lab[k], "' n'a pas les memes niveaux que '", lab[1],
             "'. Une covariance entre termes n'a de sens qu'a niveaux partages.")
      Zl <- lapply(inc, function(z) if (is.null(z$Z))
        Matrix::sparseMatrix(i = seq_len(n), j = as.integer(z$f), x = 1,
                             dims = c(n, z$q)) else methods::as(as.matrix(z$Z), "dgCMatrix"))
      return(rx_term(nm, Zl, K = K, struct = st, rank = rk, levels = lv0))
    }

    # ---- structures de CORRELATION entre niveaux -----------------------------
    # ar1(f), ar2(f), ar3(f), sar(f), ma1(f), ma2(f), arma(f), cor(f),
    # corb(f, order=), corg(f), et ar1(ligne, colonne) pour le produit
    # separable. Metriques 1D : exp(x), gau(x), lvr(x). Metriques 2D :
    # iexp(x,y), igau(x,y), ieuc(x,y), ilv(x,y), sph(x,y), cir(x,y),
    # aexp(x,y), agau(x,y), mtrn(x,y, phi=, nu=, delta=, alpha=, lambda=).
    # own(f, expr=, n_par=) pour une structure definie par l'utilisateur.
    # La variance vient de Sigma : `struct=` reste disponible pour la rendre
    # heterogene entre caracteres (l'equivalent des suffixes v/h d'asreml).
    #
    # CE TEST ETAIT INATTEIGNABLE. Les metriques 2D etaient testees a
    # l'INTERIEUR d'un bloc qui n'acceptait deja que ar1/ar2/.../gau : iexp() et
    # ses voisines tombaient donc dans "terme non reconnu", alors qu'elles
    # etaient implementees des deux cotes. Une seule liste desormais.
    if (fn %in% setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed", "ar1ar1"))) {
      args <- as.list(e)[-1]
      nommes <- names(args); if (is.null(nommes)) nommes <- rep("", length(args))
      pos <- args[nommes == ""]
      opt <- args[nommes != ""]
      ordre <- if (!is.null(opt$order)) as.integer(eval(opt$order, env)) else 0L
      st    <- if (!is.null(opt$struct)) as.character(eval(opt$struct, env)) else "iid"
      rkk   <- if (!is.null(opt$rank)) as.integer(eval(opt$rank, env)) else 0L
      # `struct =` HETEROGENE ENTRE CARACTERES (le v/h d'asreml). L'incidence
      # etait passee en facteur brut quel que soit `struct`, donc t = 1 et une
      # `diag` ou une `us` a une seule colonne : ar1(col, struct = "diag") se
      # reduisait a ar1(col) en silence, 1 variance au lieu d'une par
      # caractere. On decoupe donc l'indicatrice par caractere, exactement
      # comme pour diag(gid)/us(gid), des que `struct` n'est pas "iid".
      .par_caractere <- function(codes, q, txt) {
        if (identical(st, "iid")) return(NULL)
        if (is.null(trait))
          stop("terme '", txt, "' : struct='", st, "' multi-caractere, mais ",
               "aucun `trait` n'a ete fourni a rx_reml().")
        tf <- factor(trait, levels = t_lev)
        lapply(t_lev, function(tt2) {
          ix <- which(tf == tt2)
          Matrix::sparseMatrix(i = ix, j = codes[ix], x = 1, dims = c(n, q))
        })
      }

      # ar1(ligne, colonne) : produit separable, forme la plus courante.
      if (fn == "ar1" && length(pos) == 2L) {
        fr <- factor(data[[as.character(pos[[1]])]])
        fc <- factor(data[[as.character(pos[[2]])]])
        nr <- nlevels(fr); nc <- nlevels(fc)
        # niveau = (ligne-1)*n_colonnes + colonne : LIGNE lente, COLONNE rapide.
        # levels.py construit L_r (x) L_c dans le meme ordre.
        idx <- (as.integer(fr) - 1L) * nc + as.integer(fc)
        Z <- Matrix::sparseMatrix(i = seq_len(n), j = idx, x = 1, dims = c(n, nr * nc))
        Zl <- .par_caractere(idx, nr * nc, txt)
        return(rx_term(paste0(as.character(pos[[1]]), "_", as.character(pos[[2]])),
                       if (is.null(Zl)) Z else Zl, t = if (is.null(Zl)) 1L else NULL,
                       struct = st, rank = rkk, level = "ar1ar1", dims = c(nr, nc),
                       levels = as.vector(outer(levels(fc), levels(fr),
                                                function(a, b) paste(b, a, sep = ":")))))
      }

      co <- if (!is.null(opt$coord))
        as.matrix(eval(opt$coord, envir = data, enclos = env)) else NULL
      # Metrique a DEUX arguments positionnels : mtrn(x, y) comme chez asreml.
      # Le niveau est alors la CELLULE (x, y), et les coordonnees en decoulent.
      if (fn %in% RX_METRIQUES_2D && length(pos) == 2L && is.null(co)) {
        xv <- eval(pos[[1]], envir = data, enclos = env)
        yv <- eval(pos[[2]], envir = data, enclos = env)
        cle <- paste(xv, yv, sep = "\r")
        u <- !duplicated(cle); ordre_u <- order(cle[u])
        lev <- cle[u][ordre_u]
        fz <- factor(cle, levels = lev)
        co <- cbind(as.numeric(xv[u][ordre_u]), as.numeric(yv[u][ordre_u]))
        if (any(!is.finite(co)))
          stop("terme '", txt, "' : coordonnees non numeriques.")
        nm <- paste0(deparse(pos[[1]])[1], "_", deparse(pos[[2]])[1])
        opts <- if (fn == "mtrn") .rx_mtrn_opts(opt, env) else NULL
        Zl <- .par_caractere(as.integer(fz), length(lev), txt)
        return(rx_term(make.names(nm), if (is.null(Zl)) fz else Zl, struct = st,
                       rank = rkk, level = fn, levels = lev, coord = co, opts = opts))
      }
      if (!length(pos))
        stop(fn, "() : aucun facteur de groupement.")
      z <- .rx_incidence(pos[[1]], data, env)
      # Metrique 1D sans `coord` : les NIVEAUX doivent etre numeriques, sinon la
      # distance n'a pas de sens et le modele serait silencieusement faux.
      if (fn %in% RX_METRIQUES_1D && is.null(co)) {
        co <- suppressWarnings(as.numeric(z$levels))
        if (any(!is.finite(co)))
          stop("terme '", txt, "' : les niveaux ne sont pas numeriques ; ",
               "fournir coord = <positions des ", z$q, " niveaux>.")
        co <- as.matrix(co)
      }
      opts <- NULL; expr <- NULL
      if (fn == "mtrn") opts <- .rx_mtrn_opts(opt, env)
      if (fn == "own") {
        expr <- as.character(eval(opt$expr, env))
        npar <- as.integer(eval(opt$n_par %||% 1L, env))
        opts <- c(n_par = as.numeric(npar),
                  normalise = as.numeric(eval(opt$normalise %||% TRUE, env)))
      }
      if (fn == "corg") ordre <- z$q
      Zl <- .par_caractere(as.integer(z$f), z$q, txt)
      return(rx_term(z$nom, if (is.null(Zl)) z$f else Zl, struct = st, rank = rkk,
                     level = fn, levels = z$levels, order = ordre, coord = co,
                     opts = opts, expr = expr))
    }

    # ---- termes simples ------------------------------------------------------
    if (fn %in% c("rr", "chol", "ante", "corh")) {
      args <- as.list(e)[-(1:2)]
      rk <- if (!is.null(args$rank)) as.integer(eval(args$rank)) else 1L
      K  <- if (!is.null(args$K)) eval(args$K, envir = data, enclos = env) else NULL
      z <- .rx_incidence(e[[2]], data, env)
      if (is.null(trait))
        stop("terme '", txt, "' : structure '", fn, "' multi-caractere, mais ",
             "aucun `trait` n'a ete fourni a rx_reml().")
      tf <- factor(trait, levels = t_lev)
      Zl <- lapply(t_lev, function(tt2) {
        ix <- which(tf == tt2)
        Matrix::sparseMatrix(i = ix, j = as.integer(z$f)[ix], x = 1, dims = c(n, z$q))
      })
      return(rx_term(z$nom, Zl, K = K, struct = fn, rank = rk, levels = z$levels))
    }
    if (!fn %in% c("iid", "vm", "diag", "us", "fa", "mm"))
      stop("terme aleatoire non reconnu : '", txt, "'.\n  Structures de Sigma : ",
           paste(RX_STRUCTURES, collapse = "/"), " (via iid/vm/diag/us/fa/mm)\n",
           "  Structures entre niveaux : ",
           paste(setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed", "ar1ar1")), collapse = "/"),
           "\n  Groupement : str(~ a + b, struct=)   Incidence fournie : mm(Z)")
    args <- if (is.name(e)) list() else as.list(e)[-(1:2)]
    struct <- switch(fn, iid = "iid", vm = "iid", mm = "iid",
                     diag = "diag", us = "us", fa = "fa")
    if (!is.null(args$struct)) struct <- as.character(eval(args$struct))
    K <- if (!is.null(args$K)) eval(args$K, envir = data, enclos = env) else
         if (fn == "vm" && length(args) >= 1L && (is.null(names(args)) || !nzchar(names(args)[1])))
           eval(args[[1]], envir = data, enclos = env) else NULL
    rank <- if (!is.null(args$rank)) as.integer(eval(args$rank)) else 0L

    if (fn == "mm") {                    # incidence PERSONNALISEE
      z <- .rx_incidence(e, data, env)
      if (struct == "iid")
        return(rx_term(z$nom, list(z$Z), K = K, struct = "iid", levels = z$levels))
      if (is.null(trait))
        stop("mm(..., struct='", struct, "') : structure multi-caractere sans `trait`.")
      stop("mm() multi-caractere : passer une LISTE d'incidences a rx_term(), ",
           "une seule matrice ne peut pas porter ", length(t_lev), " colonnes.")
    }

    # `e` peut etre un simple nom (~ g) : il n'a alors pas de e[[2]].
    z <- .rx_incidence(if (is.name(e)) e else e[[2]], data, env)
    if (struct == "iid")
      return(rx_term(z$nom, z$f, K = K, struct = "iid", levels = z$levels))
    if (is.null(trait))
      stop("terme '", txt, "' : structure '", struct,
           "' multi-caractere, mais aucun `trait` n'a ete fourni a rx_reml().")
    tf <- factor(trait, levels = t_lev)
    Zl <- lapply(t_lev, function(tt2) {
      ix <- which(tf == tt2)
      Matrix::sparseMatrix(i = ix, j = as.integer(z$f)[ix], x = 1, dims = c(n, z$q))
    })
    rx_term(z$nom, Zl, K = K, struct = struct, rank = rank, levels = z$levels)
  })
}

#' Ajustement REML par formule
#'
#' @param fixed    formule des effets fixes (reponse a gauche)
#' @param random   formule des termes aleatoires (cf. grammaire ci-dessus)
#' @param residual "units" (iid), "diag" ou "us" ; ou une formule ~units
#' @param trait    nom de la colonne de caractere (format long), ou NULL
#' @param unit     nom de la colonne d'unite ; deux lignes de la meme unite sur
#'                 des caracteres differents sont correlees sous residual="us"
#' @param backend  "auto" | "gpu" | "cpu" — choix de MACHINE, jamais de modele
rx_reml <- function(fixed, random = NULL, residual = "units", data,
                    trait = NULL, unit = NULL,
                    backend = c("auto", "gpu", "cpu"), ...) {
  # `...` transmet vpredict, wald, fixed_theta, n_restarts, verbose... a rx_fit.
  backend <- match.arg(backend)

  mf <- model.frame(fixed, data, na.action = na.pass)
  y  <- model.response(mf)
  X  <- model.matrix(fixed, data)
  .asg <- attr(X, "assign")
  .lab <- c("(Intercept)", attr(terms(fixed), "term.labels"))
  attr(X, "termes") <- .lab[sort(unique(.asg)) + 1L]
  ok <- stats::complete.cases(y, X)
  if (!all(ok)) {
    message("rx_reml : ", sum(!ok), " ligne(s) incompletes retirees.")
    data <- data[ok, , drop = FALSE]
    y <- y[ok]; X <- X[ok, , drop = FALSE]
    # Le sous-ensemblage de X PERD ses attributs : sans eux, wald() testerait
    # chaque colonne isolement au lieu du terme entier. `.keep_attr` n'existait
    # nulle part — toute execution avec une ligne incomplete s'arretait ici sur
    # "objet introuvable".
    attr(X, "assign") <- .asg; attr(X, "termes") <- .lab[sort(unique(.asg)) + 1L]
    if (!is.null(trait)) trait <- trait[ok]
    if (!is.null(unit))  unit  <- unit[ok]
  }
  tr <- if (is.character(trait) && length(trait) == 1L) data[[trait]] else trait
  un <- if (is.character(unit)  && length(unit)  == 1L) data[[unit]]  else unit
  terms_l <- .rx_parse_random(random, data, trait = tr)
  res_obj <- .rx_parse_residual(residual, data,
                                trait = if (!is.null(trait)) trait else tr,
                                unit  = if (!is.null(unit))  unit  else un)
  mod <- rx_model(y, X, terms_l, res_obj,
                  name = deparse(fixed[[2]])[1])
  fit <- rx_fit(mod, backend = backend, ...)
  fit$model <- mod
  # Conserve de quoi predire : formule des effets fixes, donnees nettoyees et
  # niveaux des facteurs. Sans les xlevels, une prediction construite sur une
  # grille reduite n'aurait PAS les memes colonnes que X, et L beta melangerait
  # les coefficients sans rien signaler.
  fit$fixed   <- fixed
  fit$data    <- data
  fit$xlevels <- stats::.getXlevels(terms(fixed, data = data),
                                    model.frame(fixed, data, na.action = stats::na.pass))
  fit$call    <- match.call()
  fit
}

# ==============================================================================
# 7. SPLINES 2D (produit tensoriel de P-splines)
# ------------------------------------------------------------------------------
# Une surface lisse en deux dimensions s'ecrit comme un effet ALEATOIRE a
# incidence connue : il n'y a donc rien a ajouter au solveur, seulement des
# matrices a construire. C'est ainsi que procedent sommer (spl2Da) et asreml.
#
# CONSTRUCTION (Eilers & Marx ; decomposition PS-ANOVA de Rodriguez-Alvarez et al.)
#   1. base de B-splines B_x (n x c_x) et B_y (n x c_y), degre 3, `nseg` segments ;
#   2. penalite de differences d'ordre 2 : D' D, de rang c - 2, noyau = {1, x} ;
#   3. reparametrisation en partie NULLE (le noyau : constante et pente, qui
#      vont dans les effets FIXES) et partie PENALISEE (le reste, aleatoire,
#      avec Z = B U diag(d^-1/2) de sorte que la covariance soit sigma^2 I).
#
# Trois blocs aleatoires en sortie, comme dans PS-ANOVA :
#   f_x    lissage principal selon x        (penalise en x, lineaire en y)
#   f_y    lissage principal selon y
#   f_xy   interaction lisse
# Chacun a SA variance : c'est ce qui rend le lissage anisotrope, une surface
# pouvant etre rugueuse dans un sens et lisse dans l'autre.
#
#   sp <- rx_spl2d(d$row, d$col, nseg = c(6, 6))
#   mod <- rx_model(y, cbind(X, sp$X), c(list(...), sp$terms))
# Les colonnes sp$X sont la partie NULLE : elles DOIVENT aller dans les effets
# fixes, sinon la surface est penalisee jusque dans sa composante lineaire et le
# lissage est biaise.
# ==============================================================================

.rx_bbase <- function(x, nseg = 6L, deg = 3L, xl = min(x), xr = max(x)) {
  dx <- (xr - xl) / nseg
  kn <- seq(xl - deg * dx, xr + deg * dx, by = dx)
  splines::spline.des(kn, x, deg + 1L, 0 * x, outer.ok = TRUE)$design
}

#' Base P-spline 2D, prete pour rx_model()
#' @return list(X = partie nulle (fixe), terms = liste de rx_term aleatoires)
rx_spl2d <- function(x, y, nseg = c(6L, 6L), deg = 3L, pord = 2L, prefix = "spl") {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (length(y) != n) stop("rx_spl2d : x et y de longueurs differentes.")
  Bx <- .rx_bbase(x, nseg[1], deg); By <- .rx_bbase(y, nseg[min(2, length(nseg))], deg)
  # Decomposition de la penalite : U_pen (colonnes penalisees) et noyau
  decomp <- function(B, pord) {
    c_ <- ncol(B); D <- diff(diag(c_), differences = pord); P <- crossprod(D)
    e <- eigen(P, symmetric = TRUE)
    keep <- e$values > 1e-10 * max(e$values)
    list(Zc = B %*% e$vectors[, keep, drop = FALSE] %*%
           diag(1 / sqrt(e$values[keep]), sum(keep)),
         Xn = B %*% e$vectors[, !keep, drop = FALSE])   # noyau : constante + pente
  }
  dx <- decomp(Bx, pord); dy <- decomp(By, pord)
  rowk <- function(A, B) {                    # produit de Khatri-Rao par ligne
    do.call(cbind, lapply(seq_len(ncol(B)), function(j) A * B[, j]))
  }
  Zx  <- rowk(dx$Zc, dy$Xn)                   # lissage en x, lineaire en y
  Zy  <- rowk(dx$Xn, dy$Zc)
  Zxy <- rowk(dx$Zc, dy$Zc)                   # interaction lisse
  Xn  <- rowk(dx$Xn, dy$Xn)                   # partie NULLE -> effets fixes
  # La partie nulle CONTIENT la direction constante : cbind(1, Xn) serait de rang
  # deficient et rx_model() le refuserait (a raison). On projette donc Xn hors de
  # l'intercept, puis on ne garde qu'une base independante par QR revelatrice de
  # rang. `X` est ainsi utilisable tel quel a cote d'un intercept.
  Xn <- Xn - matrix(colMeans(Xn), n, ncol(Xn), byrow = TRUE)
  qrX <- qr(Xn, tol = 1e-9)
  Xn <- Xn[, sort(qrX$pivot[seq_len(qrX$rank)]), drop = FALSE]
  colnames(Xn) <- paste0(prefix, "_lin", seq_len(ncol(Xn)))
  list(X = Xn,
       terms = list(
         rx_term(paste0(prefix, "_x"),  list(Zx),  struct = "iid"),
         rx_term(paste0(prefix, "_y"),  list(Zy),  struct = "iid"),
         rx_term(paste0(prefix, "_xy"), list(Zxy), struct = "iid")))
}

# ==============================================================================
# 8. PREDICTIONS (predict d'asreml)
# ------------------------------------------------------------------------------
# Une prediction est une COMBINAISON LINEAIRE des effets ajustes. Tout le travail
# est de construire la bonne combinaison ; l'algebre, elle, tient en deux lignes.
#
#   classify : les variables tenues a chacun de leurs niveaux
#   les autres variables du modele fixe sont MOYENNEES :
#     - un facteur, sur ses niveaux (poids egaux, ou proportionnels aux effectifs
#       observes si average = "proportional") ;
#     - une covariable, a sa MOYENNE — c'est ce que fait asreml, et c'est ce qui
#       explique qu'une prediction "par variete" soit donnee a Column = 5.5.
#
# TERMES ALEATOIRES. Si une variable du classify est le facteur d'un terme
# aleatoire, son BLUP entre dans la prediction (c'est le cas d'une "moyenne de
# genotype"), et l'erreur-type devient une erreur de PREDICTION, calculee cote
# solveur ou vivent V^-1 et le projecteur. include_random = FALSE rend la
# prediction des seuls effets fixes.
#
# ESTIMABILITE. Une combinaison hors de l'espace des lignes de X n'est pas
# estimable : la rendre quand meme donnerait un nombre qui depend de la
# parametrisation choisie. Elle est marquee NON estimable et sa valeur est NA.
#
#   pv <- rx_predict(fit, classify = "trt")
#   pv <- rx_predict(fit, classify = "genotype", vcov = "kenward-roger", sed = TRUE)
# ==============================================================================
rx_predict <- function(fit, classify, levels = NULL, at = NULL,
                       average = c("equal", "proportional"), weights = NULL,
                       vcov = c("simple", "kenward-roger"),
                       sed = FALSE, include_random = TRUE,
                       backend = c("auto", "gpu", "cpu"), verbose = FALSE) {
  average <- match.arg(average); vcov <- match.arg(vcov); backend <- match.arg(backend)
  if (is.null(fit$fixed) || is.null(fit$data))
    stop("rx_predict : cet ajustement ne vient pas de rx_reml() ; la formule des ",
         "effets fixes et les donnees ne sont pas connues.")
  cls <- unlist(strsplit(classify, "[:+]"))
  cls <- trimws(cls[nzchar(cls)])
  data <- fit$data
  tt   <- stats::delete.response(stats::terms(fit$fixed, data = data))
  vars <- all.vars(tt)
  # Un classify peut designer un terme ALEATOIRE (le cas "moyenne par genotype")
  # : il n'est alors pas dans la formule des effets fixes, et c'est normal. Sa
  # contribution vient du BLUP, pas de X.
  noms_alea <- vapply(fit$model$terms, `[[`, "", "name")
  if (length(manque <- setdiff(cls, c(vars, noms_alea))))
    stop("rx_predict : ", paste(manque, collapse = ", "), " n'est ni dans le ",
         "modele fixe (", paste(vars, collapse = ", "), ") ni parmi les termes ",
         "aleatoires (", paste(noms_alea, collapse = ", "), ").")

  # --- grille : classify x (facteurs moyennes) ; covariables a leur moyenne ----
  vals <- list()
  for (v in union(vars, cls)) {
    if (!v %in% names(data))
      stop("rx_predict : colonne '", v, "' absente des donnees de l'ajustement.")
    x <- data[[v]]
    if (!is.null(at[[v]])) { vals[[v]] <- at[[v]]; next }
    if (is.factor(x) || is.character(x) || is.logical(x)) {
      lv <- if (!is.null(levels[[v]])) as.character(levels[[v]]) else base::levels(factor(x))
      vals[[v]] <- factor(lv, levels = base::levels(factor(x)))
    } else {
      vals[[v]] <- if (v %in% cls) {
        if (!is.null(levels[[v]])) levels[[v]] else sort(unique(x))
      } else mean(x, na.rm = TRUE)
    }
  }
  grille <- expand.grid(vals, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  Xg <- stats::model.matrix(tt, grille, xlev = fit$xlevels)
  if (ncol(Xg) != length(fit$beta))
    stop("rx_predict : la grille donne ", ncol(Xg), " colonnes pour ",
         length(fit$beta), " coefficients. Un facteur a-t-il des niveaux absents ",
         "de la grille ?")

  # --- moyennage sur ce qui n'est pas dans le classify -------------------------
  cle <- if (length(cls)) do.call(paste, c(grille[cls], sep = "\r")) else rep("1", nrow(grille))
  ucle <- unique(cle)
  moy <- setdiff(vars, cls)
  moy_fac <- moy[vapply(moy, function(v) is.factor(vals[[v]]), TRUE)]
  poids_ligne <- rep(1, nrow(grille))
  if (average == "proportional" && length(moy_fac)) {
    # Poids PROPORTIONNELS aux effectifs observes de la combinaison moyennee.
    obs <- do.call(paste, c(lapply(moy_fac, function(v) as.character(data[[v]])), sep = "\r"))
    grl <- do.call(paste, c(lapply(moy_fac, function(v) as.character(grille[[v]])), sep = "\r"))
    tab <- table(obs)
    poids_ligne <- as.numeric(tab[grl]); poids_ligne[is.na(poids_ligne)] <- 0
  }
  if (!is.null(weights)) poids_ligne <- poids_ligne * weights[cle]
  L <- t(vapply(ucle, function(k) {
    i <- which(cle == k); w <- poids_ligne[i]
    if (sum(w) <= 0) stop("rx_predict : poids nuls pour la cellule '",
                          gsub("\r", ":", k), "'.")
    colSums(Xg[i, , drop = FALSE] * (w / sum(w)))
  }, numeric(ncol(Xg))))
  etiq <- do.call(rbind, lapply(strsplit(ucle, "\r", fixed = TRUE), function(z) z))
  out <- if (length(cls)) as.data.frame(etiq, stringsAsFactors = FALSE) else
         data.frame(prediction = seq_along(ucle))
  names(out) <- if (length(cls)) cls else "prediction"

  # --- estimabilite -----------------------------------------------------------
  X <- fit$model$X
  qrX <- qr(X); proj <- qr.Q(qrX)[, seq_len(qrX$rank), drop = FALSE]
  # L doit etre dans l'espace des LIGNES de X : on teste via la projection de
  # chaque ligne de L sur cet espace.
  Rx <- qr.R(qrX)[seq_len(qrX$rank), , drop = FALSE]
  base_lignes <- t(qr.Q(qr(t(Rx))))
  resid_L <- L - L %*% t(base_lignes) %*% base_lignes
  estimable <- sqrt(rowSums(resid_L^2)) < 1e-8 * pmax(sqrt(rowSums(L^2)), 1)

  # --- part aleatoire ---------------------------------------------------------
  M <- NULL
  if (isTRUE(include_random)) {
    for (tm in fit$model$terms) {
      v <- tm$name
      if (!v %in% cls || is.null(tm$levels)) next
      if (tm$t != 1L) {
        warning("rx_predict : terme '", v, "' multi-caractere ignore dans la ",
                "part aleatoire (t = ", tm$t, ").")
        next
      }
      Mi <- matrix(0, nrow(L), tm$q)
      j <- match(as.character(out[[v]]), tm$levels)
      if (anyNA(j)) next
      Mi[cbind(seq_len(nrow(L)), j)] <- 1
      M[[v]] <- Mi
    }
  }

  if (is.null(M)) {
    # Effets fixes seuls : tout se calcule ici, sans repasser par le solveur.
    Vb <- if (vcov == "kenward-roger") fit$vbeta_kr else fit$vbeta
    if (is.null(Vb))
      stop("rx_predict : la covariance des effets fixes n'est pas dans l'ajustement",
           if (vcov == "kenward-roger") " (relancer rx_reml(..., kenward_roger = TRUE))" else "",
           ".")
    val <- as.numeric(L %*% fit$beta)
    Cv  <- L %*% Vb %*% t(L)
  } else {
    if (vcov == "kenward-roger")
      warning("rx_predict : l'ajustement de Kenward-Roger porte sur les effets ",
              "FIXES ; la part aleatoire garde son erreur de prediction usuelle.")
    dir <- tempfile("rkpred_"); on.exit(unlink(dir, recursive = TRUE))
    rx_export(fit$model, dir)
    wbin <- function(nm, x) { con <- file(file.path(dir, nm), "wb")
      writeBin(as.double(as.vector(as.matrix(x))), con, size = 8); close(con) }
    wbin("in_theta.bin", fit$theta); wbin("pred_L.bin", L)
    for (nm in names(M)) wbin(paste0("pred_M_", nm, ".bin"), M[[nm]])
    py <- rx_python_cmd()
    .rx_python_assurer(py)
    a <- c(rx_solver_args(), dir, "--backend", backend, "--only-predict",
           if (!verbose) "--quiet" else NULL)
    st <- system2(py[1], shQuote(c(py[-1], as.character(a))),
                  stdout = if (verbose) "" else TRUE, stderr = if (verbose) "" else TRUE)
    if (!is.null(attr(st, "status")) && attr(st, "status") != 0) {
      cat(st, sep = "\n"); stop("rx_predict : le solveur a echoue") }
    rr <- rx_read_result(dir)
    val <- as.numeric(rr$predictions$valeur); Cv <- rr$predictions$cov
  }
  out$predicted.value <- ifelse(estimable, val, NA_real_)
  out$std.error <- ifelse(estimable, sqrt(pmax(diag(Cv), 0)), NA_real_)
  out$estimable <- estimable
  attr(out, "vcov") <- Cv
  if (isTRUE(sed)) {
    d <- outer(diag(Cv), diag(Cv), "+") - 2 * Cv
    S <- sqrt(pmax(d, 0)); diag(S) <- NA_real_
    attr(out, "sed") <- S
    attr(out, "sed.moyen") <- sqrt(mean(S[upper.tri(S)]^2, na.rm = TRUE))
  }
  class(out) <- c("rx_predict", "data.frame")
  out
}

print.rx_predict <- function(x, ...) {
  cat("Predictions\n")
  print(as.data.frame(x), row.names = FALSE, digits = 6)
  if (!is.null(attr(x, "sed.moyen")))
    cat(sprintf("\n  erreur-type moyenne des differences : %.6f\n", attr(x, "sed.moyen")))
  if (any(!x$estimable))
    cat(sprintf("  %d prediction(s) NON estimable(s)\n", sum(!x$estimable)))
  invisible(x)
}
