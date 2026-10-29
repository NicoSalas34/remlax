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
# caracteres mesures sur des sous-ensembles differents d'unites - ce qui est la
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
    # memes genotypes - le cas qui a motive tout ceci.
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
      if (fn %in% RX_METRIC_2D && length(pos) == 2L && is.null(co)) {
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
      if (fn %in% RX_METRIC_1D && is.null(co)) {
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
