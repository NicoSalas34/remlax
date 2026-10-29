# ==============================================================================
# formula_random.R - the random formula of rx_reml(), as in asreml
# ==============================================================================
#     rx_reml(fixed    = y ~ 1 + treatment,
#             random   = ~ vm(genotype, K) + iid(block),
#             residual = ~ units,
#             data     = df)
#
# Each random term is a call whose first argument is the grouping column:
#     iid(f)             one variance                              Sigma = s2
#     vm(f, K)           the same, with a relationship K between levels
#     diag(f)            one variance per trait                    diagonal Sigma
#     us(f)              free covariance between traits            us Sigma
#     fa(f, rank = k)    factor analytic of rank k
#     rr, chol, ante, corh(f, ...)   the other Sigma structures
#     a bare name `f`    the same as iid(f)
#     ar1(f), ..., mtrn(x, y, ...), own(f, ...)   correlation between levels
#     str(~ a + b, struct = "us")    one covariance for several terms
#     mm(Z)              a supplied incidence matrix
# `K =` is accepted by all of them. diag/us/fa and the other multi-trait
# structures need `trait =`: the data are then in long format (one row per
# unit and trait) and the incidence of each column of Sigma is built from the
# trait factor, which handles traits measured on different subsets of units.
# ==============================================================================

.rx_parse_random <- function(random, data, trait = NULL) {
  if (is.null(random)) return(list())
  env <- environment(random)
  if (is.null(env)) env <- parent.frame()
  labels <- attr(terms(random, keep.order = TRUE, specials = NULL), "term.labels")
  if (!length(labels)) stop("`random` ne contient aucun terme.")
  ctx <- list(data = data, env = env, trait = trait, n = nrow(data),
              trait_levels = if (is.null(trait)) NULL else levels(factor(trait)))
  lapply(labels, .rx_random_term, ctx = ctx)
}

# One random term, dispatched on the function name of its call.
.rx_random_term <- function(txt, ctx) {
  e <- str2lang(txt)
  fn <- if (is.name(e)) "iid" else as.character(e[[1]])
  if (fn == "str") return(.rx_random_str(e, ctx))
  if (fn %in% setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed", "ar1ar1")))
    return(.rx_random_level(e, fn, txt, ctx))
  if (fn %in% c("rr", "chol", "ante", "corh")) return(.rx_random_multitrait(e, fn, txt, ctx))
  if (!fn %in% c("iid", "vm", "diag", "us", "fa", "mm"))
    stop("terme aleatoire non reconnu : '", txt, "'.\n  Structures de Sigma : ",
         paste(RX_STRUCTURES, collapse = "/"), " (via iid/vm/diag/us/fa/mm)\n",
         "  Structures entre niveaux : ",
         paste(setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed", "ar1ar1")), collapse = "/"),
         "\n  Groupement : str(~ a + b, struct=)   Incidence fournie : mm(Z)")
  .rx_random_simple(e, fn, txt, ctx)
}

# Incidence of an expression: a column name (a factor) or mm(<matrix>).
# Returns list(f = factor or NULL, Z = matrix or NULL, levels, q, nom).
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

# One indicator incidence per trait: the rows of trait k point to their level
# in the k-th matrix and are zero elsewhere.
.rx_split_by_trait <- function(codes, q, ctx) {
  tf <- factor(ctx$trait, levels = ctx$trait_levels)
  lapply(ctx$trait_levels, function(level) {
    ix <- which(tf == level)
    Matrix::sparseMatrix(i = ix, j = codes[ix], x = 1, dims = c(ctx$n, q))
  })
}

# str(~ a + b, struct = ): one covariance for several terms, as asreml's
# str(~ a + b, ~us(2):id(n)). The grouped terms must share exactly the same
# levels, which is what gives a covariance between them a meaning; each keeps
# its own incidence, so that a direct effect and a neighbour effect carried by
# the same genotypes can be correlated.
.rx_random_str <- function(e, ctx) {
  inner <- e[[2]]
  if (!is.call(inner) || !identical(as.character(inner[[1]]), "~"))
    stop("str() : premier argument attendu sous forme ~ a + b")
  labels <- attr(terms(stats::as.formula(inner), keep.order = TRUE), "term.labels")
  if (length(labels) < 2L)
    stop("str() : au moins deux termes attendus, recu ", length(labels), ".")
  args <- as.list(e)[-(1:2)]
  st <- if (!is.null(args$struct)) as.character(eval(args$struct, ctx$env)) else "us"
  K  <- if (!is.null(args$K)) eval(args$K, envir = ctx$data, enclos = ctx$env) else NULL
  rk <- if (!is.null(args$rank)) as.integer(eval(args$rank, ctx$env)) else 0L
  inc <- lapply(labels, function(l) .rx_incidence(str2lang(l), ctx$data, ctx$env))
  # Readable name: the names of the incidences (column or mm(name =)).
  nm <- if (!is.null(args$name)) as.character(args$name) else
        paste(vapply(inc, `[[`, "", "nom"), collapse = "_")
  lv0 <- inc[[1]]$levels
  for (k in seq_along(inc)) if (!identical(inc[[k]]$levels, lv0))
    stop("str() : le terme '", labels[k], "' n'a pas les memes niveaux que '", labels[1],
         "'. Une covariance entre termes n'a de sens qu'a niveaux partages.")
  Zl <- lapply(inc, function(z) if (is.null(z$Z))
    Matrix::sparseMatrix(i = seq_len(ctx$n), j = as.integer(z$f), x = 1,
                         dims = c(ctx$n, z$q)) else methods::as(as.matrix(z$Z), "dgCMatrix"))
  rx_term(nm, Zl, K = K, struct = st, rank = rk, levels = lv0)
}

# A correlation structure between levels: ar1(f), ar2, ar3, sar, ma1, ma2,
# arma, cor, corb(f, order =), corg, ar1(row, col) for the separable product,
# the 1D metrics exp(x), gau(x), lvr(x), the 2D metrics iexp(x, y), igau,
# ieuc, sph, cir, aexp, agau, mtrn(x, y, phi =, nu =, delta =, alpha =,
# lambda =), and own(f, expr =, n_par =). The variance comes from Sigma:
# struct = makes it heterogeneous between traits (asreml's v/h suffixes), with
# one indicator incidence per trait.
.rx_random_level <- function(e, fn, txt, ctx) {
  args <- as.list(e)[-1]
  arg_names <- names(args); if (is.null(arg_names)) arg_names <- rep("", length(args))
  pos <- args[arg_names == ""]
  opt <- args[arg_names != ""]
  env <- ctx$env; data <- ctx$data
  lvl_order <- if (!is.null(opt$order)) as.integer(eval(opt$order, env)) else 0L
  st    <- if (!is.null(opt$struct)) as.character(eval(opt$struct, env)) else "iid"
  rk    <- if (!is.null(opt$rank)) as.integer(eval(opt$rank, env)) else 0L
  by_trait <- function(codes, q) {
    if (identical(st, "iid")) return(NULL)
    if (is.null(ctx$trait))
      stop("terme '", txt, "' : struct='", st, "' multi-caractere, mais ",
           "aucun `trait` n'a ete fourni a rx_reml().")
    .rx_split_by_trait(codes, q, ctx)
  }

  # ar1(row, col): level = (row - 1) * n_columns + column, row slow and column
  # fast, the order of L_r (x) L_c in levels.py.
  if (fn == "ar1" && length(pos) == 2L) {
    fr <- factor(data[[as.character(pos[[1]])]])
    fc <- factor(data[[as.character(pos[[2]])]])
    nr <- nlevels(fr); nc <- nlevels(fc)
    idx <- (as.integer(fr) - 1L) * nc + as.integer(fc)
    Z <- Matrix::sparseMatrix(i = seq_len(ctx$n), j = idx, x = 1, dims = c(ctx$n, nr * nc))
    Zl <- by_trait(idx, nr * nc)
    return(rx_term(paste0(as.character(pos[[1]]), "_", as.character(pos[[2]])),
                   if (is.null(Zl)) Z else Zl, t = if (is.null(Zl)) 1L else NULL,
                   struct = st, rank = rk, level = "ar1ar1", dims = c(nr, nc),
                   levels = as.vector(outer(levels(fc), levels(fr),
                                            function(a, b) paste(b, a, sep = ":")))))
  }

  co <- if (!is.null(opt$coord))
    as.matrix(eval(opt$coord, envir = data, enclos = env)) else NULL
  # A 2D metric on two positional arguments, mtrn(x, y): the level is the cell
  # (x, y) and the coordinates follow from the data.
  if (fn %in% RX_METRIC_2D && length(pos) == 2L && is.null(co)) {
    xv <- eval(pos[[1]], envir = data, enclos = env)
    yv <- eval(pos[[2]], envir = data, enclos = env)
    key <- paste(xv, yv, sep = "\r")
    u <- !duplicated(key); ord_u <- order(key[u])
    lev <- key[u][ord_u]
    fz <- factor(key, levels = lev)
    co <- cbind(as.numeric(xv[u][ord_u]), as.numeric(yv[u][ord_u]))
    if (any(!is.finite(co)))
      stop("terme '", txt, "' : coordonnees non numeriques.")
    nm <- paste0(deparse(pos[[1]])[1], "_", deparse(pos[[2]])[1])
    opts <- if (fn == "mtrn") .rx_mtrn_opts(opt, env) else NULL
    Zl <- by_trait(as.integer(fz), length(lev))
    return(rx_term(make.names(nm), if (is.null(Zl)) fz else Zl, struct = st,
                   rank = rk, level = fn, levels = lev, coord = co, opts = opts))
  }
  if (!length(pos))
    stop(fn, "() : aucun facteur de groupement.")
  z <- .rx_incidence(pos[[1]], data, env)
  # A 1D metric without `coord` takes the levels as positions; they must then
  # be numeric, otherwise the distance has no meaning.
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
  if (fn == "corg") lvl_order <- z$q
  Zl <- by_trait(as.integer(z$f), z$q)
  rx_term(z$nom, if (is.null(Zl)) z$f else Zl, struct = st, rank = rk,
          level = fn, levels = z$levels, order = lvl_order, coord = co,
          opts = opts, expr = expr)
}

# rr, chol, ante and corh: multi-trait Sigma structures, which always need
# `trait`. rank defaults to 1.
.rx_random_multitrait <- function(e, fn, txt, ctx) {
  args <- as.list(e)[-(1:2)]
  rk <- if (!is.null(args$rank)) as.integer(eval(args$rank, ctx$env)) else 1L
  K  <- if (!is.null(args$K)) eval(args$K, envir = ctx$data, enclos = ctx$env) else NULL
  z <- .rx_incidence(e[[2]], ctx$data, ctx$env)
  if (is.null(ctx$trait))
    stop("terme '", txt, "' : structure '", fn, "' multi-caractere, mais ",
         "aucun `trait` n'a ete fourni a rx_reml().")
  rx_term(z$nom, .rx_split_by_trait(as.integer(z$f), z$q, ctx), K = K, struct = fn,
          rank = rk, levels = z$levels)
}

# iid, vm, diag, us, fa and mm. vm(f, K) takes K as its second positional
# argument; mm(Z) carries a supplied incidence and only an iid Sigma.
.rx_random_simple <- function(e, fn, txt, ctx) {
  env <- ctx$env; data <- ctx$data
  args <- if (is.name(e)) list() else as.list(e)[-(1:2)]
  struct <- switch(fn, iid = "iid", vm = "iid", mm = "iid",
                   diag = "diag", us = "us", fa = "fa")
  if (!is.null(args$struct)) struct <- as.character(eval(args$struct, env))
  K <- if (!is.null(args$K)) eval(args$K, envir = data, enclos = env) else
       if (fn == "vm" && length(args) >= 1L && (is.null(names(args)) || !nzchar(names(args)[1])))
         eval(args[[1]], envir = data, enclos = env) else NULL
  rank <- if (!is.null(args$rank)) as.integer(eval(args$rank, env)) else 0L

  if (fn == "mm") {
    z <- .rx_incidence(e, data, env)
    if (struct == "iid")
      return(rx_term(z$nom, list(z$Z), K = K, struct = "iid", levels = z$levels))
    if (is.null(ctx$trait))
      stop("mm(..., struct='", struct, "') : structure multi-caractere sans `trait`.")
    stop("mm() multi-caractere : passer une LISTE d'incidences a rx_term(), ",
         "une seule matrice ne peut pas porter ", length(ctx$trait_levels), " colonnes.")
  }

  # A bare name (~ g) has no e[[2]].
  z <- .rx_incidence(if (is.name(e)) e else e[[2]], data, env)
  if (struct == "iid")
    return(rx_term(z$nom, z$f, K = K, struct = "iid", levels = z$levels))
  if (is.null(ctx$trait))
    stop("terme '", txt, "' : structure '", struct,
         "' multi-caractere, mais aucun `trait` n'a ete fourni a rx_reml().")
  rx_term(z$nom, .rx_split_by_trait(as.integer(z$f), z$q, ctx), K = K, struct = struct,
          rank = rank, levels = z$levels)
}
