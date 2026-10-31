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
  if (!length(labels)) stop("`random` contains no term.")
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
    stop("unrecognised random term: '", txt, "'.\n  Structures of Sigma: ",
         paste(RX_STRUCTURES, collapse = "/"), " (via iid/vm/diag/us/fa/mm)\n",
         "  Structures between levels: ",
         paste(setdiff(RX_LEVEL_STRUCTURES, c("id", "fixed", "ar1ar1")), collapse = "/"),
         "\n  Grouping: str(~ a + b, struct=)   Given incidence: mm(Z)")
  .rx_random_simple(e, fn, txt, ctx)
}

# Incidence of an expression: a column name (a factor) or mm(<matrix>).
# Returns list(f = factor or NULL, Z = matrix or NULL, levels, q, nom).
.rx_incidence <- function(e, data, env) {
  if (is.name(e) || is.character(e)) {
    nm <- as.character(e)
    if (!nm %in% names(data)) stop("column '", nm, "' is missing from `data`.")
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
  stop("unrecognised incidence expression: ", deparse(e),
       "\n  expected: a column name, or mm(<matrix>).")
}

# Levels of a coordinate for a lag structure (ar1, ar2, ..., corb, and the two
# factors of ar1 x ar1). The lag between two levels is their difference in
# rank, so a row or column without any record must stay in the grid: dropping
# it brings its neighbours one step closer. A factor keeps all its levels, used
# or not, in their order. Numeric positions, or text that reads
# as numbers, span their range with the step of the data: the greatest common
# divisor of the gaps for whole numbers (1, 3, 5 is a grid of step 2; 1, 2, 4
# includes 3), the smallest gap otherwise. Other text keeps the levels present.
.rx_grid_factor <- function(x, what) {
  if (is.factor(x)) return(x)
  v <- if (is.numeric(x)) as.numeric(x) else suppressWarnings(as.numeric(as.character(x)))
  if (any(is.na(v) & !is.na(x))) return(factor(x))
  u <- sort(unique(v[!is.na(v)]))
  if (length(u) < 2L) return(factor(v, levels = u))
  gaps <- diff(u)
  whole <- all(abs(u - round(u)) < 1e-8)
  step <- if (whole) Reduce(function(a, b) { while (b > 0) { r <- a %% b; a <- b; b <- r }; a },
                            round(gaps)) else min(gaps)
  k <- (u - u[1]) / step
  if (any(abs(k - round(k)) > 1e-6))
    stop(what, ": the positions are not on a regular grid (smallest gap ",
         format(step), "). Give a factor whose levels are the grid.")
  pos <- u[1] + step * seq(0, round(max(k)))
  factor(round((v - u[1]) / step) + 1L, levels = seq_along(pos),
         labels = as.character(pos))
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
    stop("str(): the first argument must have the form ~ a + b")
  labels <- attr(terms(stats::as.formula(inner), keep.order = TRUE), "term.labels")
  if (length(labels) < 2L)
    stop("str(): at least two terms expected, got ", length(labels), ".")
  args <- as.list(e)[-(1:2)]
  .rx_check_term_args(args, c("struct", "K", "Kinv", "Kinv_logdet", "rank", "name"),
                      deparse(e)[1])
  st <- if (!is.null(args[["struct"]])) as.character(eval(args[["struct"]], ctx$env)) else "us"
  K  <- if (!is.null(args[["K"]])) eval(args[["K"]], envir = ctx$data, enclos = ctx$env) else NULL
  Kinv <- if (!is.null(args[["Kinv"]])) eval(args[["Kinv"]], envir = ctx$data, enclos = ctx$env) else NULL
  Kinv_logdet <- if (!is.null(args[["Kinv_logdet"]])) as.numeric(eval(args[["Kinv_logdet"]], ctx$env)) else NULL
  rk <- if (!is.null(args[["rank"]])) as.integer(eval(args[["rank"]], ctx$env)) else 0L
  inc <- lapply(labels, function(l)
    .rx_kinv_levels(Kinv, .rx_incidence(str2lang(l), ctx$data, ctx$env), deparse(e)[1]))
  # Readable name: the names of the incidences (column or mm(name =)).
  nm <- if (!is.null(args[["name"]])) as.character(args[["name"]]) else
        paste(vapply(inc, `[[`, "", "nom"), collapse = "_")
  lv0 <- inc[[1]]$levels
  for (k in seq_along(inc)) if (!identical(inc[[k]]$levels, lv0))
    stop("str(): the term '", labels[k], "' does not have the same levels as '", labels[1],
         "'. A covariance between terms only makes sense on shared levels.")
  Zl <- lapply(inc, function(z) if (is.null(z$Z))
    Matrix::sparseMatrix(i = seq_len(ctx$n), j = as.integer(z$f), x = 1,
                         dims = c(ctx$n, z$q)) else methods::as(as.matrix(z$Z), "dgCMatrix"))
  rx_term(nm, Zl, K = K, struct = st, rank = rk, levels = lv0,
          Kinv = Kinv, Kinv_logdet = Kinv_logdet)
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
  lvl_order <- if (!is.null(opt[["order"]])) as.integer(eval(opt[["order"]], env)) else 0L
  st    <- if (!is.null(opt[["struct"]])) as.character(eval(opt[["struct"]], env)) else "iid"
  rk    <- if (!is.null(opt[["rank"]])) as.integer(eval(opt[["rank"]], env)) else 0L
  by_trait <- function(codes, q) {
    if (identical(st, "iid")) return(NULL)
    if (is.null(ctx$trait))
      stop("term '", txt, "': struct='", st, "' is multi-trait, but ",
           "no `trait` was given to rx_reml().")
    .rx_split_by_trait(codes, q, ctx)
  }

  # ar1(row, col): level = (row - 1) * n_columns + column, row slow and column
  # fast, the order of L_r (x) L_c in levels.py.
  if (fn == "ar1" && length(pos) == 2L) {
    fr <- .rx_grid_factor(data[[as.character(pos[[1]])]], paste0("term '", txt, "'"))
    fc <- .rx_grid_factor(data[[as.character(pos[[2]])]], paste0("term '", txt, "'"))
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

  co <- if (!is.null(opt[["coord"]]))
    as.matrix(eval(opt[["coord"]], envir = data, enclos = env)) else NULL
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
      stop("term '", txt, "': non-numeric coordinates.")
    nm <- paste0(deparse(pos[[1]])[1], "_", deparse(pos[[2]])[1])
    opts <- if (fn == "mtrn") .rx_mtrn_opts(opt, env) else NULL
    Zl <- by_trait(as.integer(fz), length(lev))
    return(rx_term(make.names(nm), if (is.null(Zl)) fz else Zl, struct = st,
                   rank = rk, level = fn, levels = lev, coord = co, opts = opts))
  }
  if (!length(pos))
    stop(fn, "(): no grouping factor.")
  z <- .rx_incidence(pos[[1]], data, env)
  if (fn %in% RX_LAG_1D && !is.null(z$f) && is.null(co)) {
    z$f <- .rx_grid_factor(data[[z$nom]], paste0("term '", txt, "'"))
    z$levels <- levels(z$f); z$q <- nlevels(z$f)
  }
  # A 1D metric without `coord` takes the levels as positions; they must then
  # be numeric, otherwise the distance has no meaning.
  if (fn %in% RX_METRIC_1D && is.null(co)) {
    co <- suppressWarnings(as.numeric(z$levels))
    if (any(!is.finite(co)))
      stop("term '", txt, "': the levels are not numeric. ",
           "Give coord = <positions of the ", z$q, " levels>.")
    co <- as.matrix(co)
  }
  opts <- NULL; expr <- NULL
  if (fn == "mtrn") opts <- .rx_mtrn_opts(opt, env)
  if (fn == "own") {
    expr <- as.character(eval(opt[["expr"]], env))
    npar <- as.integer(eval(opt[["n_par"]] %||% 1L, env))
    opts <- c(n_par = as.numeric(npar),
              normalise = as.numeric(eval(opt[["normalise"]] %||% TRUE, env)))
  }
  if (fn == "corg") lvl_order <- z$q
  Zl <- by_trait(as.integer(z$f), z$q)
  # On a lag grid the incidence is given as a matrix: rx_term() would rebuild
  # the factor and drop the empty levels again.
  if (is.null(Zl) && fn %in% RX_LAG_1D && !is.null(z$f))
    Zl <- list(Matrix::sparseMatrix(i = seq_len(ctx$n), j = as.integer(z$f), x = 1,
                                    dims = c(ctx$n, z$q)))
  rx_term(z$nom, if (is.null(Zl)) z$f else Zl, struct = st, rank = rk,
          level = fn, levels = z$levels, order = lvl_order, coord = co,
          opts = opts, expr = expr)
}

# rr, chol, ante and corh: multi-trait Sigma structures, which always need
# `trait`. rank defaults to 1.
.rx_random_multitrait <- function(e, fn, txt, ctx) {
  args <- as.list(e)[-(1:2)]
  .rx_check_term_args(args, c("rank", "K", "Kinv", "Kinv_logdet"), txt)
  rk <- if (!is.null(args[["rank"]])) as.integer(eval(args[["rank"]], ctx$env)) else 1L
  K  <- if (!is.null(args[["K"]])) eval(args[["K"]], envir = ctx$data, enclos = ctx$env) else NULL
  Kinv <- if (!is.null(args[["Kinv"]])) eval(args[["Kinv"]], envir = ctx$data, enclos = ctx$env) else NULL
  Kinv_logdet <- if (!is.null(args[["Kinv_logdet"]])) as.numeric(eval(args[["Kinv_logdet"]], ctx$env)) else NULL
  z <- .rx_kinv_levels(Kinv, .rx_incidence(e[[2]], ctx$data, ctx$env), txt)
  if (is.null(ctx$trait))
    stop("term '", txt, "': structure '", fn, "' is multi-trait, but ",
         "no `trait` was given to rx_reml().")
  rx_term(z$nom, .rx_split_by_trait(as.integer(z$f), z$q, ctx), K = K, struct = fn,
          rank = rk, levels = z$levels, Kinv = Kinv, Kinv_logdet = Kinv_logdet)
}

# Named arguments of a term, checked against the names it accepts. An unknown
# name (a typo, or Kinv where only K was read) would otherwise be dropped
# without a word, and the term fitted without the matrix it was given.
.rx_check_term_args <- function(args, allowed, txt, n_positional = 0L) {
  nm <- names(args); if (is.null(nm)) nm <- rep("", length(args))
  bad <- setdiff(nm[nzchar(nm)], allowed)
  if (length(bad))
    stop("term '", txt, "': unknown argument(s) ", paste(bad, collapse = ", "),
         ". Accepted: ", paste(allowed, collapse = ", "), ".", call. = FALSE)
  if (sum(!nzchar(nm)) > n_positional)
    stop("term '", txt, "': too many unnamed arguments.", call. = FALSE)
  invisible(NULL)
}

# A precision Kinv between levels (a pedigree inverse, for example). Its rows
# carry every level of the relationship, ancestors without a record included:
# an inverse cannot be cut down to the levels present, so the factor takes the
# levels of Kinv, in its order. Without row names, Kinv must have exactly one
# row per level of the factor.
.rx_kinv_levels <- function(Kinv, z, txt) {
  if (is.null(Kinv)) return(z)
  rn <- rownames(Kinv)
  if (is.null(z$f)) {
    if (nrow(Kinv) != z$q)
      stop("term '", txt, "': Kinv is ", nrow(Kinv), "x", ncol(Kinv), " but the ",
           "incidence has ", z$q, " columns.", call. = FALSE)
    if (!is.null(rn) && !identical(rn, z$levels))
      stop("term '", txt, "': the row names of Kinv differ from the column names ",
           "of the incidence.", call. = FALSE)
    return(z)
  }
  if (is.null(rn)) {
    if (nrow(Kinv) != z$q)
      stop("term '", txt, "': Kinv has ", nrow(Kinv), " rows and no names, but ",
           "the factor has ", z$q, " levels. Name the rows of Kinv.", call. = FALSE)
    return(z)
  }
  if (!is.null(colnames(Kinv)) && !identical(colnames(Kinv), rn))
    stop("term '", txt, "': Kinv must have the same row and column names.", call. = FALSE)
  miss <- setdiff(levels(droplevels(z$f)), rn)
  if (length(miss))
    stop("term '", txt, "': ", length(miss), " level(s) missing from Kinv (e.g. ",
         paste(head(miss, 3), collapse = ", "), ").", call. = FALSE)
  z$f <- factor(as.character(z$f), levels = rn)
  z$levels <- rn; z$q <- length(rn)
  z
}

# iid, vm, diag, us, fa and mm. vm(f, K) takes K as its second positional
# argument; mm(Z) carries a supplied incidence and only an iid Sigma.
.rx_random_simple <- function(e, fn, txt, ctx) {
  env <- ctx$env; data <- ctx$data
  args <- if (is.name(e)) list() else as.list(e)[-(1:2)]
  struct <- switch(fn, iid = "iid", vm = "iid", mm = "iid",
                   diag = "diag", us = "us", fa = "fa")
  if (!is.null(args[["struct"]])) struct <- as.character(eval(args[["struct"]], env))
  .rx_check_term_args(args, if (fn == "mm") c("name", "K", "struct") else
                       c("K", "Kinv", "Kinv_logdet", "struct", "rank"), txt,
                     n_positional = if (fn == "vm") 1L else 0L)
  K <- if (!is.null(args[["K"]])) eval(args[["K"]], envir = data, enclos = env) else
       if (fn == "vm" && length(args) >= 1L && (is.null(names(args)) || !nzchar(names(args)[1])))
         eval(args[[1]], envir = data, enclos = env) else NULL
  Kinv <- if (!is.null(args[["Kinv"]])) eval(args[["Kinv"]], envir = data, enclos = env) else NULL
  Kinv_logdet <- if (!is.null(args[["Kinv_logdet"]])) as.numeric(eval(args[["Kinv_logdet"]], env)) else NULL
  rank <- if (!is.null(args[["rank"]])) as.integer(eval(args[["rank"]], env)) else 0L

  if (fn == "mm") {
    z <- .rx_incidence(e, data, env)
    if (struct == "iid")
      return(rx_term(z$nom, list(z$Z), K = K, struct = "iid", levels = z$levels))
    if (is.null(ctx$trait))
      stop("mm(..., struct='", struct, "'): multi-trait structure without `trait`.")
    stop("multi-trait mm(): pass a LIST of incidences to rx_term(). ",
         "A single matrix cannot carry ", length(ctx$trait_levels), " columns.")
  }

  # A bare name (~ g) has no e[[2]].
  z <- .rx_incidence(if (is.name(e)) e else e[[2]], data, env)
  z <- .rx_kinv_levels(Kinv, z, txt)
  if (struct == "iid")
    # With Kinv the factor keeps levels without records (ancestors): it goes
    # as a matrix, since rx_term() would rebuild it and drop those levels.
    return(rx_term(z$nom, if (is.null(z$f)) list(z$Z) else if (is.null(Kinv)) z$f else
                     list(Matrix::sparseMatrix(i = seq_len(ctx$n), j = as.integer(z$f), x = 1,
                                               dims = c(ctx$n, z$q))),
                   K = K, struct = "iid",
                   levels = z$levels, Kinv = Kinv, Kinv_logdet = Kinv_logdet))
  if (is.null(ctx$trait))
    stop("term '", txt, "': structure '", struct,
         "' is multi-trait, but no `trait` was given to rx_reml().")
  rx_term(z$nom, .rx_split_by_trait(as.integer(z$f), z$q, ctx), K = K, struct = struct,
          rank = rank, levels = z$levels, Kinv = Kinv, Kinv_logdet = Kinv_logdet)
}
