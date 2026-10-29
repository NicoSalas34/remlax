# Extrait les termes spl2d(x, y, nseg =, deg =, pord =, name =) d'une formule
# aleatoire. Rend la formule restante (NULL si plus rien), les termes aleatoires
# des surfaces, leur partie nulle et, par colonne, le terme d'origine.
.rx_extraire_spl2d <- function(random, data, trait = NULL) {
  vide <- list(random = random, terms = list(), X = NULL, groups = character())
  if (is.null(random)) return(vide)
  env <- environment(random); if (is.null(env)) env <- parent.frame()
  tt <- attr(terms(random, keep.order = TRUE), "term.labels")
  est <- vapply(tt, function(txt) {
    e <- str2lang(txt); is.call(e) && as.character(e[[1]]) %in% c("spl2d", "spl2D")
  }, logical(1))
  if (!any(est)) return(vide)
  if (!is.null(trait) && nlevels(factor(trait)) > 1L)
    stop("spl2d() : surface multi-caractere non prise en charge par la formule. ",
         "Construire la base par rx_spl2d() et passer par rx_model().", call. = FALSE)
  coord <- function(a, quoi) {
    v <- eval(a, data, env)
    if (is.factor(v)) {
      v2 <- suppressWarnings(as.numeric(as.character(v)))
      if (anyNA(v2)) stop("spl2d() : la coordonnee ", quoi, " est un facteur a niveaux non ",
                          "numeriques ; donner une coordonnee numerique.", call. = FALSE)
      v <- v2
    }
    if (!is.numeric(v) || length(v) != nrow(data) || anyNA(v))
      stop("spl2d() : la coordonnee ", quoi, " doit etre numerique, complete, de longueur ",
           nrow(data), ".", call. = FALSE)
    v
  }
  termes <- list(); Xs <- list(); grp <- character(); vus <- character(); surfaces <- list()
  for (txt in tt[est]) {
    e <- str2lang(txt); args <- as.list(e)[-1]
    nm <- names(args); if (is.null(nm)) nm <- rep("", length(args))
    pos <- args[nm == ""]; opt <- args[nm != ""]
    xa <- opt$x; ya <- opt$y; ip <- 1L
    if (is.null(xa)) { if (length(pos) < ip) stop("spl2d() : coordonnee x manquante.", call. = FALSE)
                       xa <- pos[[ip]]; ip <- ip + 1L }
    if (is.null(ya)) { if (length(pos) < ip) stop("spl2d() : coordonnee y manquante.", call. = FALSE)
                       ya <- pos[[ip]] }
    ev <- function(a, def) if (is.null(a)) def else eval(a, env)
    nseg <- as.integer(ev(opt$nseg, c(10L, 10L)))
    prefix <- as.character(ev(opt$name, if (!length(vus)) "spl" else paste0("spl", length(vus) + 1L)))
    if (prefix %in% vus) stop("spl2d() : deux surfaces portent le nom '", prefix, "'.", call. = FALSE)
    vus <- c(vus, prefix)
    xv <- coord(xa, "x"); yv <- coord(ya, "y")
    deg <- as.integer(ev(opt$deg, c(3L, 3L))); pord <- as.integer(ev(opt$pord, 2L))
    nest <- as.integer(ev(opt$nest.div, c(1L, 1L)))
    x_name <- paste(deparse(xa), collapse = ""); y_name <- paste(deparse(ya), collapse = "")
    # Une surface par niveau (at = , at.levels =), a la maniere de spl2Dc() de
    # sommer : chaque niveau a sa base, construite sur l'etendue de SES
    # coordonnees, ses cinq variances et sa partie nulle ; ses lignes hors du
    # niveau sont nulles.
    aa <- if (!is.null(opt$at)) opt$at else opt$at.var
    if (is.null(aa)) {
      niv <- list(list(rows = seq_len(nrow(data)), prefix = prefix, level = NULL))
      at_name <- NULL
    } else {
      at_name <- paste(deparse(aa), collapse = "")
      av <- eval(aa, data, env)
      if (length(av) != nrow(data) || anyNA(av))
        stop("spl2d() : la variable at = ", at_name, " doit etre complete, de longueur ",
             nrow(data), ".", call. = FALSE)
      av <- factor(av)
      lv <- ev(opt$at.levels, levels(av))
      lv <- as.character(lv)
      hors <- setdiff(lv, levels(av))
      if (length(hors))
        stop("spl2d() : at.levels contient des niveaux absents de ", at_name, " : ",
             paste(hors, collapse = ", "), ".", call. = FALSE)
      niv <- lapply(lv, function(l) list(rows = which(av == l), level = l,
                                         prefix = paste0(prefix, "_", gsub("[^A-Za-z0-9.]", "_", l))))
      pf <- vapply(niv, `[[`, "", "prefix")
      if (anyDuplicated(pf))
        stop("spl2d() : deux niveaux de ", at_name, " donnent le meme nom de terme ('",
             pf[duplicated(pf)][1], "').", call. = FALSE)
    }
    for (nv in niv) {
      r <- nv$rows
      if (length(unique(xv[r])) < 2L || length(unique(yv[r])) < 2L)
        stop("spl2d() : ", if (is.null(nv$level)) "la surface" else
             paste0("le niveau '", nv$level, "' de ", at_name),
             " n'a qu'une valeur de x ou de y ; une surface demande deux dimensions.", call. = FALSE)
      s <- rx_spl2d(xv[r], yv[r], nseg = nseg, deg = deg, pord = pord, nest.div = nest,
                    prefix = nv$prefix)
      plein <- function(M) { P <- matrix(0, nrow(data), ncol(M), dimnames = list(NULL, colnames(M)))
                             P[r, ] <- as.matrix(M); P }
      if (length(r) < nrow(data))
        s$terms <- lapply(s$terms, function(tm) {
          rx_term(tm$name, list(plein(tm$Zl[[1]])), struct = "iid") })
      termes <- c(termes, s$terms); Xs[[length(Xs) + 1L]] <- plein(s$X)
      grp <- c(grp, rep(txt, ncol(s$X)))
      surfaces[[nv$prefix]] <- list(basis = s$basis, x = xv[r], y = yv[r], rows = r, label = txt,
                                    x_name = x_name, y_name = y_name,
                                    at = if (!is.null(nv$level)) list(name = at_name, level = nv$level))
    }
  }
  reste <- tt[!est]
  list(random = if (length(reste)) stats::as.formula(paste("~", paste(reste, collapse = " + ")),
                                                      env = env) else NULL,
       terms = termes, X = do.call(cbind, Xs), groups = grp, surfaces = surfaces)
}

# ==============================================================================
# 7. SPLINES 2D (produit tensoriel de P-splines)
# ------------------------------------------------------------------------------
# Une surface lisse en deux dimensions s'ecrit comme un effet ALEATOIRE a
# incidence connue : il n'y a donc rien a ajouter au solveur, seulement des
# matrices a construire. C'est ainsi que procedent sommer (spl2Dc) et asreml.
#
# CONSTRUCTION (Eilers & Marx ; decomposition PS-ANOVA de Rodriguez-Alvarez et al.)
#   1. base de B-splines B_x (n x c_x) et B_y (n x c_y), degre 3, `nseg` segments ;
#   2. penalite de differences d'ordre 2 : D' D, de rang c - 2, noyau = {1, x} ;
#   3. reparametrisation en partie NULLE (le noyau : constante et pente, qui
#      vont dans les effets FIXES) et partie PENALISEE (le reste, aleatoire,
#      avec Z = B U diag(d^-1/2) de sorte que la covariance soit sigma^2 I).
#
# Cinq blocs aleatoires, exactement ceux du PSANOVA de SpATS
# (Rodriguez-Alvarez et al. 2018, construct.2d.pspline) :
#   fx      f(x)          Z_x
#   fy      f(y)          Z_y
#   fx_y    f(x):y        Z_x    * y lineaire
#   x_fy    x:f(y)        x lineaire * Z_y
#   fx_fy   f(x):f(y)     Z_x (x) Z_y, sur des bases emboitees si nest.div > 1
# Chacun a SA variance. SpATS ecrit G_k = s2_k diag(1/d) sur Z = B U ; on met ici
# Z a l'echelle par d^-1/2, ce qui donne un terme iid de MEME variance s2_k.
# Pour f(x):f(y), d est la somme de Kronecker 1 (x) d_x + d_y (x) 1, comme dans
# SpATS. Les colonnes constante et lineaire sont celles de SpATS (MM.basis,
# decom = 4) : 1 / sqrt(c) et B a, ou a est l'indice des coefficients centre et
# norme. Leur echelle ne change pas l'ajustement, mais elle fixe l'unite de s2
# pour f(x), f(y), f(x):y et x:f(y) : c'est ce qui rend les variances egales a
# celles de SpATS.
#
#   sp <- rx_spl2d(d$col, d$row, nseg = c(10, 10))
#   mod <- rx_model(y, cbind(X, sp$X), c(list(...), sp$terms))
# Les colonnes sp$X sont la partie NULLE (x, y, x*y) : elles DOIVENT aller dans
# les effets fixes, sinon la surface est penalisee jusque dans sa composante
# lineaire et le lissage est biaise.
# ==============================================================================

.rx_bbase <- function(x, nseg = 6L, deg = 3L, xl = min(x), xr = max(x)) {
  dx <- (xr - xl) / nseg
  kn <- seq(xl - deg * dx, xr + deg * dx, by = dx)
  splines::spline.des(kn, x, deg + 1L, 0 * x, outer.ok = TRUE)$design
}

# Decomposition de la penalite d'une base 1D : vecteurs propres PENALISES (et
# leurs valeurs), NOYAU, et direction LINEAIRE a de SpATS. Conserves dans la
# base, pour reevaluer la surface en de nouveaux points avec EXACTEMENT la meme
# reparametrisation.
.rx_pen_decomp <- function(c_, pord) {
  D <- diff(diag(c_), differences = pord); e <- eigen(crossprod(D), symmetric = TRUE)
  keep <- seq_len(c_ - pord)
  a <- seq_len(c_) - (c_ + 1) / 2
  list(U = e$vectors[, keep, drop = FALSE], d = e$values[keep],
       N = e$vectors[, -keep, drop = FALSE], lin = a / sqrt(sum(a^2)))
}

.rx_rowk <- function(A, B)                 # produit de Khatri-Rao par ligne
  do.call(cbind, lapply(seq_len(ncol(B)), function(j) A * B[, j]))

.rx_spl2d_axe <- function(v, ax, deg, nested = FALSE) {
  ns <- if (nested) ax$nseg / ax$nest else ax$nseg
  pen <- if (nested) ax$pen_n else ax$pen
  B <- .rx_bbase(as.numeric(v), ns, deg, ax$xl, ax$xr)
  list(B = B, Z = B %*% pen$U, d = pen$d, N = B %*% pen$N, lin = as.numeric(B %*% pen$lin),
       one = 1 / sqrt(ncol(B)))
}

# Noms des cinq termes, suffixes du prefixe.
.rx_spl2d_suffixes <- c("fx", "fy", "fx_y", "x_fy", "fx_fy")

# Evalue une base spl2d en (x, y). Rend la partie nulle (centree, colonnes
# retenues) et les cinq incidences penalisees, deja mises a l'echelle. Hors du
# domaine d'ajustement, les B-splines sont extrapolees (outer.ok) : la carte sur
# grille reste donc dans le rectangle des donnees.
.rx_spl2d_eval <- function(basis, x, y) {
  deg <- rep_len(basis$deg, 2L)
  ex <- .rx_spl2d_axe(x, basis$x, deg[1]); ey <- .rx_spl2d_axe(y, basis$y, deg[2])
  exn <- if (basis$x$nest > 1) .rx_spl2d_axe(x, basis$x, deg[1], TRUE) else ex
  eyn <- if (basis$y$nest > 1) .rx_spl2d_axe(y, basis$y, deg[2], TRUE) else ey
  ech <- function(Z, d) sweep(Z, 2L, sqrt(d), "/")
  Zx <- ech(ex$Z, ex$d); Zy <- ech(ey$Z, ey$d)
  Zxy <- ech(.rx_rowk(exn$Z, eyn$Z), as.vector(outer(exn$d, eyn$d, "+")))
  Xn <- .rx_rowk(ex$N, ey$N)
  Xn <- sweep(Xn, 2L, basis$centre)[, basis$keep, drop = FALSE]
  # f(x) porte la colonne constante normee de y, 1 / sqrt(c_y), comme dans SpATS :
  # sans effet sur l'ajustement, ce facteur fixe l'unite de s2.
  Z <- list(Zx * ey$one, Zy * ex$one, Zx * ey$lin, ex$lin * Zy, Zxy)
  names(Z) <- .rx_spl2d_suffixes
  list(X = Xn, Z = Z)
}

# Base P-spline 2D, PS-ANOVA de SpATS, prete pour rx_model()
# @return list(X = partie nulle (fixe), terms = liste de rx_term aleatoires,
#   basis = de quoi reevaluer la surface, voir .rx_spl2d_eval())
rx_spl2d <- function(x, y, nseg = c(10L, 10L), deg = c(3L, 3L), pord = 2L,
                     nest.div = c(1L, 1L), prefix = "spl") {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (length(y) != n) stop("rx_spl2d : x et y de longueurs differentes.")
  if (any(pord != 2))
    stop("rx_spl2d : seule la penalite d'ordre 2 est prise en charge (comme PSANOVA de SpATS).",
         call. = FALSE)
  ns <- rep_len(as.integer(round(nseg)), 2L); deg <- rep_len(as.integer(deg), 2L)
  nd <- rep_len(as.integer(nest.div), 2L)
  if (any(ns %% nd != 0))
    stop("rx_spl2d : nseg doit etre un multiple de nest.div.", call. = FALSE)
  basis <- list(deg = deg, pord = 2L, prefix = prefix,
                x = list(xl = min(x), xr = max(x), nseg = ns[1], nest = nd[1]),
                y = list(xl = min(y), xr = max(y), nseg = ns[2], nest = nd[2]))
  basis$x$pen <- .rx_pen_decomp(ns[1] + deg[1], 2L)
  basis$y$pen <- .rx_pen_decomp(ns[2] + deg[2], 2L)
  if (nd[1] > 1) basis$x$pen_n <- .rx_pen_decomp(ns[1] / nd[1] + deg[1], 2L)
  if (nd[2] > 1) basis$y$pen_n <- .rx_pen_decomp(ns[2] / nd[2] + deg[2], 2L)
  # La partie nulle CONTIENT la direction constante : cbind(1, Xn) serait de rang
  # deficient et rx_model() le refuserait (a raison). On projette donc Xn hors de
  # l'intercept, puis on ne garde qu'une base independante par QR revelatrice de
  # rang. `X` est ainsi utilisable tel quel a cote d'un intercept. Il engendre le
  # meme espace que la partie fixe de SpATS : x, y et x*y.
  basis$centre <- 0; basis$keep <- TRUE
  brut <- .rx_spl2d_eval(basis, x, y)
  basis$centre <- colMeans(brut$X)
  Xc <- sweep(brut$X, 2L, basis$centre)
  qrX <- qr(Xc, tol = 1e-9)
  basis$keep <- sort(qrX$pivot[seq_len(qrX$rank)])
  Xn <- Xc[, basis$keep, drop = FALSE]
  colnames(Xn) <- paste0(prefix, "_lin", seq_len(ncol(Xn)))
  basis$columns <- colnames(Xn)
  basis$terms <- paste0(prefix, "_", .rx_spl2d_suffixes)
  list(X = Xn,
       terms = lapply(seq_along(brut$Z), function(k)
         rx_term(basis$terms[k], list(brut$Z[[k]]), struct = "iid")),
       basis = basis)
}
