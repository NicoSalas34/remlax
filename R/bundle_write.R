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
  if (!is.null(model$weights)) put("poids", model$weights)
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
    if (!is.null(tm$Kinv)) {
      # precision creuse (pedigree) : lue par le seul moteur MME
      Ti <- methods::as(methods::as(tm$Kinv, "CsparseMatrix"), "TsparseMatrix")
      put(paste0("term_", p, "_Kinv_i"), Ti@i, "i4"); put(paste0("term_", p, "_Kinv_j"), Ti@j, "i4")
      put(paste0("term_", p, "_Kinv_x"), Ti@x)
      ld <- tm$Kinv_logdet %||%
        -as.numeric(Matrix::determinant(Matrix::forceSymmetric(tm$Kinv), logarithm = TRUE)$modulus)
      put(paste0("term_", p, "_Kinv_logdet"), ld)
    }
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
    # UNE GRILLE AR1 x AR1 GARDE SES CODES DE CELLULE. L'unite y vaut
    # (ligne-1)*n_colonnes + colonne, de 1 a n_unit : c'est la POSITION dans le
    # produit de Kronecker. La renumeroter par factor() supprimait les cellules
    # sans observation et decalait toutes les suivantes, sans aucune erreur : le
    # solveur lisait la correlation aux mauvaises cellules. MESURE (5 % de
    # cellules manquantes) : logLik -854,39 contre -820,77 avec la vraie
    # geometrie, rho des lignes 0,18 pour 0,6 simule. Partout ailleurs l'unite
    # n'est qu'une etiquette, et factor() reste correct.
    grille <- identical(r$level, "ar1ar1") && !is.null(r$n_unit) && !is.null(r$unit)
    put(paste0(pre, "_unit"),
        if (is.null(r$unit)) seq_len(nlig) - 1L else
        if (grille) as.integer(r$unit) - 1L else as.integer(factor(r$unit)) - 1L, "i4")
    if (grille) put(paste0(pre, "_nunit"), as.integer(r$n_unit), "i4")
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
  # DECALAGE SPATIAL. W en triplets COO 0-based (une commune, ou une par
  # caractere receveur), indices de caractere et d'unite 0-based par ligne,
  # et trois matrices q x q (receveur en ligne) : entrees libres, valeurs des
  # entrees non libres, depart des entrees libres.
  if (!is.null(model$lag)) {
    L <- model$lag
    put("lag_q", L$q, "i4"); put("lag_N", L$N, "i4"); put("lag_nW", length(L$W), "i4")
    for (k in seq_along(L$W)) {
      S <- methods::as(L$W[[k]], "TsparseMatrix")
      put(sprintf("lag_W%d_i", k - 1L), S@i, "i4")
      put(sprintf("lag_W%d_j", k - 1L), S@j, "i4")
      put(sprintf("lag_W%d_x", k - 1L), S@x)
    }
    put("lag_trait", as.integer(L$trait) - 1L, "i4")
    put("lag_unit", as.integer(L$unit) - 1L, "i4")
    put("lag_mask", matrix(as.integer(L$mask), L$q, L$q), "i4")
    put("lag_value", matrix(as.numeric(L$value), L$q, L$q))
    put("lag_init", matrix(as.numeric(L$init), L$q, L$q))
    put("lag_traits", L$traits, "str")
    put("lag_criterion", L$criterion, "str")
  }
  writeLines(jsonlite::toJSON(man, auto_unbox = TRUE), file.path(dir, "manifest.json"))
  invisible(dir)
}
