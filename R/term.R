# ==============================================================================
# term.R - rx_term(): one random term of the model
# ==============================================================================
# A term is a set of t incidence matrices (n x q) sharing the same q levels, a
# covariance structure Sigma between those t columns, and a correlation or
# relationship matrix K between the levels:
#
#     u ~ N(0, Sigma (x) K)
#
# Column order of a term, shared with the Python solver (structures.py):
#     column of Z = (column of Sigma - 1) * q + level      (level varies fastest)
# Sigma (x) K follows the same order. Changing it on one side only gives
# another model without any visible error.
# ==============================================================================

# asreml-style parameter value: 3 is estimated from 3, "3 F" is fixed at 3, and
# an absent value is fixed at the default (or estimated, if requested).
.rx_arg_vs <- function(x, default, estimated_if_absent = FALSE) {
  if (is.null(x)) return(list(est = estimated_if_absent, val = default))
  if (is.numeric(x)) return(list(est = TRUE, val = as.numeric(x)))
  s <- trimws(as.character(x))
  code <- toupper(sub("^[-0-9.eE+]*\\s*", "", s))
  v <- suppressWarnings(as.numeric(sub("\\s*[A-Za-z]*$", "", s)))
  if (is.na(v)) stop("valeur de parametre illisible : '", s, "'")
  list(est = !identical(code, "F"), val = v)
}

# Options of a Matern structure, as the named numeric vector read by the
# solver: lambda (1 city-block, 2 Euclidean), then for phi, nu, delta and alpha
# a value, an `est_` flag and, when estimated from a given value, an `init_`.
.rx_mtrn_opts <- function(opt, env = parent.frame()) {
  ev <- function(z) if (is.null(z)) NULL else eval(z, envir = env)
  defaults <- list(phi = NA, nu = 0.5, delta = 1.0, alpha = 0.0)
  o <- c(lambda = as.numeric(ev(opt$lambda) %||% 2))
  if (!o[["lambda"]] %in% c(1, 2))
    stop("mtrn : lambda vaut 1 (city-block) ou 2 (euclidienne), pas ", o[["lambda"]], ".")
  for (p in c("phi", "nu", "delta", "alpha")) {
    a <- .rx_arg_vs(ev(opt[[p]]), defaults[[p]], estimated_if_absent = (p == "phi"))
    o[paste0("est_", p)] <- as.numeric(isTRUE(a$est))
    o[p] <- if (is.na(a$val)) defaults[[p]] else a$val
    if (isTRUE(a$est) && !is.na(a$val)) o[paste0("init_", p)] <- a$val
  }
  o
}

# Sparse dgCMatrix without going through a dense matrix.
.rx_dgc <- function(m) {
  if (!inherits(m, "Matrix")) m <- methods::as(as.matrix(m), "dMatrix")
  methods::as(methods::as(methods::as(m, "dMatrix"), "generalMatrix"), "CsparseMatrix")
}

rx_term <- function(name, Z, K = NULL, struct = "iid", rank = 0L,
                    t = NULL, levels = NULL, level = "auto",
                    dims = NULL, order = 0L, coord = NULL,
                    opts = NULL, expr = NULL, parts = NULL,
                    Kinv = NULL, Kinv_logdet = NULL, colnames = NULL) {
  # Names of the columns of Sigma travel to fit$sigmas, fit$blups and
  # fit$component_names. By default, the names of a list of incidences.
  if (is.null(colnames) && is.list(Z) && !is.null(names(Z)) && all(nzchar(names(Z))))
    colnames <- names(Z)
  if (!is.null(Kinv) && !is.null(K))
    stop("terme '", name, "' : fournir K OU Kinv, pas les deux. K sert au moteur ",
         "dense (facteur de Cholesky), Kinv au moteur creux (precision).")
  struct <- match.arg(struct, RX_STRUCTURES)
  level  <- match.arg(level, c("auto", RX_LEVEL_STRUCTURES))

  inc <- .rx_term_incidences(name, Z, t, levels)
  Zl <- inc$Zl; q <- inc$q; t <- inc$t; levels <- inc$levels
  colnames <- .rx_term_colnames(name, colnames, t)
  .rx_term_check_struct(name, struct, rank, t)
  LK <- .rx_kinship_factor(name, K, levels, q)
  if (level == "auto") level <- if (!is.null(LK)) "fixed" else "id"
  coord <- .rx_term_check_level(name, level, LK, dims, q, coord, expr)
  n_lvl <- rx_n_level(level, order, opts, parts = parts)
  Kinv <- .rx_term_check_kinv(name, Kinv, q)

  structure(list(name = name, Zl = Zl, t = as.integer(t), q = as.integer(q),
                 struct = struct, rank = as.integer(rank), LK = LK,
                 Kinv = Kinv, Kinv_logdet = Kinv_logdet,
                 level = level, dims = dims, order = as.integer(order), coord = coord,
                 opts = opts, expr = expr, parts = parts,
                 levels = levels, colnames = colnames,
                 n_par = rx_n_params(struct, t, rank) + n_lvl),
            class = "rx_term")
}

# The t incidence matrices of a term, from one of three forms: a factor or
# character vector (indicator incidence, t = 1), a list of t n x q matrices,
# or one n x (t*q) matrix with `t` given. A factor defines the levels; a list
# of matrices with identical column names defines them when `levels` is NULL,
# and K is then aligned on them by name.
.rx_term_incidences <- function(name, Z, t, levels) {
  if (is.factor(Z) || is.character(Z)) {
    f <- factor(Z)
    levels <- levels(f)
    q <- length(levels)
    Zm <- Matrix::sparseMatrix(i = seq_along(f), j = as.integer(f),
                               x = 1, dims = c(length(f), q))
    Zl <- list(Zm); t <- 1L
  } else if (is.list(Z)) {
    t <- length(Z); Zl <- lapply(Z, .rx_dgc)
    q <- ncol(Zl[[1]])
    if (any(vapply(Zl, ncol, 1L) != q))
      stop("terme '", name, "' : les incidences n'ont pas toutes ", q, " colonnes.")
  } else {
    if (is.null(t)) stop("terme '", name, "' : fournir `t` avec une matrice empilee.")
    Zm <- .rx_dgc(Z)
    q <- ncol(Zm) / t
    if (q != round(q)) stop("terme '", name, "' : ncol(Z) n'est pas un multiple de t.")
    q <- as.integer(q)
    Zl <- lapply(seq_len(t), function(a) Zm[, ((a - 1) * q + 1):(a * q), drop = FALSE])
  }
  if (is.null(levels) && is.list(Z)) {
    cn <- colnames(Zl[[1]])
    if (!is.null(cn) && !anyDuplicated(cn) &&
        all(vapply(Zl, function(m) identical(colnames(m), cn), logical(1))))
      levels <- cn
  }
  list(Zl = Zl, q = q, t = t, levels = levels)
}

.rx_term_colnames <- function(name, colnames, t) {
  if (is.null(colnames)) return(NULL)
  colnames <- as.character(colnames)
  if (length(colnames) != t || anyDuplicated(colnames) || anyNA(colnames))
    stop("terme '", name, "' : `colnames` doit donner ", t, " noms distincts (recu ",
         length(colnames), ").")
  colnames
}

.rx_term_check_struct <- function(name, struct, rank, t) {
  if (struct == "iid" && t > 1L)
    message("terme '", name, "' : struct='iid' avec t=", t,
            " -> une seule variance partagee par les ", t, " colonnes.")
  if (struct %in% c("fa", "rr") && (rank < 1L || rank > t))
    stop("terme '", name, "' : rank doit etre dans 1..", t, " pour une structure ", struct, ".")
  if (struct %in% c("chol", "ante") && (rank < 1L || rank > t - 1L))
    stop("terme '", name, "' : pour ", struct, ", rank (l'ordre de la bande) doit ",
         "etre dans 1..", t - 1L, ".")
}

# Lower Cholesky factor LK of a relationship matrix K (K = LK LK'), aligned on
# the levels by name when both carry names. A genomic relationship matrix is
# often only semi-definite (identical genotypes, more markers than
# individuals), so K is bent by a growing multiple of its mean diagonal until
# it factorises, with a message saying by how much.
.rx_kinship_factor <- function(name, K, levels, q) {
  if (is.null(K)) return(NULL)
  K <- as.matrix(K)
  if (!is.null(levels) && !is.null(rownames(K))) {
    missing_levels <- setdiff(levels, rownames(K))
    if (length(missing_levels))
      stop("terme '", name, "' : ", length(missing_levels),
           " niveau(x) absent(s) de K (ex. ", paste(head(missing_levels, 3), collapse = ", "), ").")
    K <- K[levels, levels, drop = FALSE]
  }
  if (nrow(K) != q) stop("terme '", name, "' : K est ", nrow(K), "x", ncol(K),
                         " mais il y a ", q, " niveaux.")
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
  t(ch)
}

# Checks of the between-level structure; returns the coordinates as a matrix.
.rx_term_check_level <- function(name, level, LK, dims, q, coord, expr) {
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
  if (level %in% RX_METRIC) {
    if (is.null(coord))
      stop("terme '", name, "' : level='", level, "' exige `coord` ",
           "(coordonnees des ", q, " niveaux, vecteur ou matrice a 2 colonnes).")
    coord <- as.matrix(coord)
    if (nrow(coord) != q)
      stop("terme '", name, "' : `coord` a ", nrow(coord), " lignes pour ", q, " niveaux.")
    if (level %in% RX_METRIC_2D && ncol(coord) != 2L)
      stop("terme '", name, "' : '", level, "' est une structure a DEUX ",
           "dimensions ; `coord` a ", ncol(coord), " colonne(s). Une coordonnee ",
           "1D serait acceptee en silence et donnerait un autre modele.")
  }
  if (level == "own" && (is.null(expr) || !nzchar(expr)))
    stop("terme '", name, "' : level='own' exige `expr`, l'expression de la ",
         "correlation (variables : d, dx, dy, lag, I, J, p1..pk).")
  coord
}

# `Kinv` is the sparse precision K^-1 between levels, an alternative to K for
# the mixed model equations engine, which only needs K^-1: a pedigree
# relationship built by Henderson's rules has a sparse inverse (about five
# non-zeros per individual), whereas K itself and its Cholesky factor are
# dense. `Kinv_logdet` passes log|K| when it is known; otherwise the solver
# computes it once. The dense engine ignores Kinv, and rx_model() says so.
# Only the size is checked here, not the values: a precision may legitimately
# be stored as a triangle.
.rx_term_check_kinv <- function(name, Kinv, q) {
  if (is.null(Kinv)) return(NULL)
  Kinv <- methods::as(methods::as(Kinv, "CsparseMatrix"), "generalMatrix")
  if (nrow(Kinv) != q || ncol(Kinv) != q)
    stop("terme '", name, "' : Kinv est ", nrow(Kinv), "x", ncol(Kinv),
         " mais il y a ", q, " niveaux.")
  density <- Kinv@x |> length() / (as.numeric(q) * q)
  if (density > 0.5)
    message("terme '", name, "' : Kinv est remplie a ",
            sprintf("%.0f %%", 100 * density), ". Le moteur creux n'y gagnera ",
            "rien - c'est le cas d'une parente genomique, dont l'inverse est ",
            "plein.")
  Kinv
}
