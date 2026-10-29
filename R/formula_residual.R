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
        # Nom de section : le niveau s'il est deja un nom R valide ("A"), sinon
        # prefixe par la variable ("bloc_1" et non "X1", que make.names fabrique
        # a partir d'un niveau numerique et que personne ne reconnait).
        brut <- paste(niv, collapse = "_")
        s$name <- if (identical(make.names(brut), brut)) brut
                  else make.names(paste0(sec_var, "_", brut))
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
  # devenait iid avec 4 "unites" au lieu de 180 - un modele completement
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
      residual %in% c("units", "iid", "diag", "us", "fa", "none")) {
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
      # MEME vraisemblance, a -19.5572007 - le signe qu'aucun n'arrivait.
      if (f == "mtrn") lvl_opts <- .rx_mtrn_opts(args, env)
      if (f == "own") {
        lvl_expr <- as.character(eval(args$expr, env))
        lvl_opts <- c(n_par = as.numeric(eval(args$n_par %||% 1L, env)),
                      normalise = as.numeric(eval(args$normalise %||% TRUE, env)))
      }
      # Metrique 2D en deux colonnes : mtrn(x, y), iexp(x, y)... L'unite est
      # alors la CELLULE (x, y) et les coordonnees en decoulent.
      if (f %in% RX_METRIC_2D && length(posit) == 2L && is.null(co)) {
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
      if (f %in% RX_METRIC_1D && is.null(co)) co <- NA          # deduit plus bas
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
