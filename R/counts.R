# ==============================================================================
# remlax - INTERFACE R D'UN SOLVEUR REML GENERIQUE
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
# Les estimes coincident, le chemin non - et le nombre d'iterations non plus.
#
# CONVENTION D'ORDRE DES COLONNES (partagee avec structures.py) :
#     colonne de Z = (col - 1) * q + niveau     -> NIVEAU le plus rapide
# Sigma (x) K suit le meme ordre. En changer d'un cote sans l'autre donne un
# autre modele, sans aucune erreur visible : ne pas y toucher.
# ==============================================================================

# (paquet : Matrix et jsonlite sont importes par NAMESPACE)


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
# Terme aleatoire
#
# @param name   nom (sert de cle dans les sorties)
# @param Z      incidence. Trois formes acceptees :
#               - un vecteur (facteur/caractere) de longueur n : incidence
#                 indicatrice, t = 1 ;
#               - une LISTE de t matrices n x q : une par colonne de Sigma ;
#               - une matrice n x (t*q) deja empilee, avec `t` fourni.
# @param K      parente entre niveaux (q x q, dimnames = niveaux) ou NULL = I
# @param struct "iid" | "diag" | "us" | "fa"
# @param rank   rang de la structure FA
# @param levels niveaux, si Z est fourni en matrice (sinon deduits du facteur)
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
                         # PARTAGEES entre les repliques. C'est le champ du modele
                         # de Salas et al. (2026), et ce que `ar1ar1` ne peut pas
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

# Nombre de parametres d'une structure entre niveaux (doit suivre levels.py)
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

# Structure de niveaux effective d'un terme rx_term.
#
# `level = "auto"` est la valeur par defaut de rx_term : elle signifie « pas de
# structure entre niveaux », sauf si une parente est fournie. Une parente en
# `LK` est un facteur de Cholesky dense ("fixed") ; une parente en `Kinv` est
# une precision creuse ("prec"). Cette resolution est faite ici et nulle part
# ailleurs.
rx_level_of <- function(tm) {
  lv <- tm$level %||% "auto"
  if (!is.null(tm$Kinv)) return("prec")
  if (!is.null(tm$LK)) return("fixed")
  if (lv %in% c("auto", "id", "iid")) return("id")
  lv
}

# Longueur du vecteur theta d'un modele
#
# Utile des qu'on veut IMPOSER un theta : un banc apparie doit evaluer deux
# moteurs au MEME point, sinon chacun part de son theta initial et les
# vraisemblances rapportees ne sont pas comparables. Mesure du piege : les
# valeurs divergeaient de 1e-5 a 9e-3 en croissant avec le nombre de
# caracteres, alors qu'a theta commun les moteurs s'accordent a 1e-12.
#
# @param model objet rx_model
# @return entier
# @export
rx_n_theta <- function(model) {
  # COMPTE GENERAL. Une version anterieure passait par des compteurs qui ne
  # connaissaient que iid/diag/us et id/ar1/ar1ar1/prec/fixed : sur un modele
  # fa, corh, cor, ar2, une residuelle us ou sectionnee, la fonction levait
  # « structure inconnue » alors que sa documentation en fait la longueur du
  # theta de N'IMPORTE quel modele, celle qu'il faut pour theta_init. Le
  # decoupage est celui du moteur dense (rx_n_params + rx_n_level par terme,
  # puis par section residuelle), verifie par test-comptages.R.
  n_lv <- function(lv, order, opts, parts) {
    if (lv %in% c("id", "fixed", "prec")) return(0L)
    as.integer(rx_n_level(lv, order %||% 0L, opts, parts = parts))
  }
  np <- 0L
  for (tm in model$terms)
    np <- np + as.integer(rx_n_params(tm$struct %||% "iid", tm$t, tm$rank %||% 0L)) +
      n_lv(rx_level_of(tm), tm$order, tm$opts, tm$parts)
  .t_de <- function(r) if (is.null(r$trait)) 1L else nlevels(factor(r$trait))
  res <- model$residual
  for (s in (res$sections %||% list(res))) {
    if (identical(s$struct, "none")) next
    np <- np + as.integer(rx_n_params(s$struct %||% "iid", .t_de(s), s$rank %||% 0L)) +
      n_lv(s$level %||% "id", s$order, s$opts, NULL)
  }
  # Entrees libres de Psi (rx_lag), en queue de theta comme dans le moteur.
  np + (if (is.null(model$lag)) 0L else as.integer(model$lag$n_free))
}
