# Argument facon asreml : 3 = estime en partant de 3 ; "3 F" = fixe a 3 ;
# absent = fixe au defaut. C'est la regle du manuel, reprise telle quelle.
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

# @param Kinv PRECISION entre niveaux, K^-1, en matrice creuse (dgCMatrix).
#   Alternative a `K` et destinee au moteur CREUX, qui n'a jamais besoin de K
#   ni de sa Cholesky : les equations du modele mixte contiennent G^-1, donc
#   K^-1. C'est ce qui permet de fournir une parente genealogique par les regles
#   de Henderson, ou A^-1 est creuse - environ cinq non-nuls par individu - sans
#   jamais former A ni la factoriser. Fournir `K` ne permet pas cela : `LK` est
#   un facteur DENSE, et l'inverse d'une matrice creuse est generalement plein.
#
#   `Kinv_logdet` permet de passer log|K| quand il est connu (Henderson le donne
#   en somme de termes locaux) ; sinon il est calcule une fois par une Cholesky
#   creuse. Il ne depend d'aucun parametre, donc c'est une constante de
#   l'optimisation.
#
#   Le moteur DENSE ignore `Kinv` : il lui faudrait inverser pour retrouver K,
#   ce qui annulerait tout l'interet. Un terme ainsi declare n'est donc ajustable
#   que par le moteur creux, et rx_model le signale.
# Incidence en dgCMatrix SANS passer par une matrice dense : as.matrix()
# sur une incidence creuse de 128 000 x 2000 allouait 1,9 Go pour la
# reconvertir aussitot en creux.
.rx_dgc <- function(m) {
  if (!inherits(m, "Matrix")) m <- methods::as(as.matrix(m), "dMatrix")
  methods::as(methods::as(methods::as(m, "dMatrix"), "generalMatrix"), "CsparseMatrix")
}

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
  # taille et la symetrie du MOTIF, pas les valeurs - une precision fournie par
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
              "rien - c'est le cas d'une parente genomique, dont l'inverse est ",
              "plein.")
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
