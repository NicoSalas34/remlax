# ==============================================================================
# scan.R - rx_scan(): genome-wide scan at fixed V on a fitted model
# ==============================================================================
#   fit  <- rx_fit(model)
#   gwas <- rx_scan(fit, model, markers = M, map = map,
#                   incidences = c(dir = "gen", ind = "voisin"),
#                   tests = c("dir", "ind", "dir+ind+sim", "ind|dir+sim"))
#
# rx_scan() refits nothing: it writes the theta already estimated by rx_fit()
# into the bundle and runs the scan with --theta-in --maxiter 0, so the solver
# rebuilds V and stops. The variance components of the scan are exactly those
# of the fit.
#
# The incidences of the focal genotype and of the neighbours are named by the
# terms of the fitted model that carry them, never supplied as matrices: a scan
# run on an incidence different from that of the null model would be a silent
# error.
# ==============================================================================

rx_scan <- function(fit, model, markers, map = NULL,
                    incidences = c(dir = "gen", ind = "voisin"),
                    tests = c("dir", "ind", "sim", "dir+ind+sim"),
                    maf = 0.05, coding = c("pm1", "012"),
                    backend = c("auto", "gpu", "cpu"), block = 1024L,
                    dir = NULL, verbose = TRUE, keep = FALSE, ...) {
  .rx_renamed_args(environment(), list(...),
                   c(marqueurs = "markers", carte = "map", codage = "coding", bloc = "block"),
                   "rx_scan")
  backend <- match.arg(backend)
  coding  <- match.arg(coding)
  stopifnot(is.list(fit), !is.null(fit$theta), is.list(model))
  if (!is.null(model$lag))
    stop("rx_scan : non disponible avec un decalage spatial (rx_lag) ; le balayage suppose ",
         "y lineaire a V donnee.", call. = FALSE)
  Z <- .rx_scan_incidences(model, incidences)
  markers <- .rx_scan_doses(markers, Z, incidences)
  p <- ncol(markers[[names(Z)[1]]])
  keep_snp <- .rx_scan_maf(markers, Z, maf, coding, verbose, p)

  # The bundle is kept when the solver fails, so that the path in the error
  # message still exists.
  keep_bundle <- FALSE
  if (is.null(dir)) {
    # tempfile(): unique, so two scans started in the same second do not share a folder.
    dir <- tempfile("rx_scan_")
    on.exit(if (!keep && !keep_bundle) unlink(dir, recursive = TRUE), add = TRUE)
  }
  # tempdir() can be relative (when $TMPDIR is); the path would then resolve
  # against the working directory of the Python process.
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir <- normalizePath(dir, mustWork = TRUE)
  .rx_scan_write(dir, model, fit, Z, markers, keep_snp, map, p)

  # Every value is quoted for the shell: a conditional test "ind|dir+sim"
  # would otherwise be read as a pipe, and the bundle path may contain spaces.
  sol <- .rx_scan_solver_args()
  args <- c(shQuote(sol$args), shQuote(dir),
            "--tests", shQuote(paste(tests, collapse = ",")),
            "--backend", backend, "--theta-in", "--maxiter", "0",
            "--bloc", as.character(as.integer(block)))
  if (!verbose) args <- c(args, "--quiet")
  code <- .rx_system2(rx_python_cmd(), args, stdout = "", stderr = "", env = sol$env)
  if (!identical(as.integer(code), 0L)) {
    keep_bundle <- TRUE
    stop("rx_scan : le solveur a rendu le code ", code,
         ". Paquet conserve pour diagnostic dans ", dir,
         "\n  relancer a la main : ", paste(c(rx_python_cmd(), args), collapse = " "),
         call. = FALSE)
  }
  # check.names = FALSE keeps the test names (`p_dir+ind+sim`, `p_sim|dir+ind`)
  # that make.names() would turn into dots.
  out <- utils::read.csv(file.path(dir, "scan_resultats.csv"),
                         stringsAsFactors = FALSE, check.names = FALSE)
  attr(out, "meta") <- jsonlite::fromJSON(file.path(dir, "scan_meta.json"))
  out
}

# Arguments and environment that select the scan entry point: $RX_SCAN_CLI when
# set, otherwise the shipped launcher followed by "scan_cli". It cannot reuse
# rx_solver_args(), which already names the fitting entry point.
.rx_scan_solver_args <- function() {
  if (nzchar(Sys.getenv("RX_SCAN_CLI")))
    return(list(args = strsplit(Sys.getenv("RX_SCAN_CLI"), " +")[[1]], env = character(0)))
  py <- rx_python_cmd()
  .rx_python_require(py)
  list(args = c(.rx_launcher(), "scan_cli"), env = character(0))
}

# "term" or "term:a": the name of a term and the index of one of its
# incidences (NA when the index is not an integer).
.rx_parse_term_ref <- function(spec) {
  spec <- as.character(spec)
  if (!grepl(":", spec, fixed = TRUE)) return(list(name = spec, index = 1L, indexed = FALSE))
  parts <- strsplit(spec, ":", fixed = TRUE)[[1]]
  list(name = parts[1], index = suppressWarnings(as.integer(parts[2])), indexed = TRUE)
}

# The incidences of the scan, taken from the terms of the model. A term may
# carry several incidences (a us structure on direct and indirect effects is
# one term with t = 2), so "term:a" names the a-th; it is required when t > 1,
# since guessing would give the direct effect the neighbour incidence.
.rx_scan_incidences <- function(model, incidences) {
  term_names <- vapply(model$terms, `[[`, "", "name")
  Z <- list()
  for (fam in names(incidences)) {
    if (!fam %in% c("dir", "ind"))
      stop("incidences : nom '", fam, "' inconnu ; attendu 'dir' et/ou 'ind'.",
           call. = FALSE)
    ref <- .rx_parse_term_ref(incidences[[fam]])
    spec <- ref$name; a <- ref$index
    if (is.na(a)) stop("incidences['", fam, "'] : apres ':' il faut un entier, ",
                       "l'indice de la colonne du terme.", call. = FALSE)
    k <- match(spec, term_names)
    if (is.na(k))
      stop("incidences['", fam, "'] = '", spec,
           "' : ce terme n'est pas dans le modele. Termes disponibles : ",
           paste(term_names, collapse = ", "), call. = FALSE)
    tm <- model$terms[[k]]
    if (tm$t > 1L && !ref$indexed)
      stop("terme '", tm$name, "' porte t = ", tm$t, " incidences. Preciser ",
           "laquelle : incidences['", fam, "'] = \"", spec, ":1\" pour la ",
           "premiere. Deviner l'indice attribuerait l'effet direct a ",
           "l'incidence de voisinage en silence.", call. = FALSE)
    if (a < 1L || a > tm$t)
      stop("incidences['", fam, "'] : indice ", a, " hors de 1..", tm$t,
           " pour le terme '", tm$name, "'.", call. = FALSE)
    Z[[fam]] <- tm$Zl[[a]]
  }
  if (!length(Z)) stop("incidences vide.", call. = FALSE)
  Z
}

# Dose matrices by family, indexed by genotype in the order of the levels of
# the term. One matrix serves both families unless a list is given (neighbours
# of another species carry another set of SNPs).
.rx_scan_doses <- function(markers, Z, incidences) {
  if (!is.list(markers)) markers <- list(dir = markers, ind = markers)
  for (fam in names(Z)) {
    if (is.null(markers[[fam]]))
      stop("markers : matrice absente pour la famille '", fam, "'.", call. = FALSE)
    M <- as.matrix(markers[[fam]])
    if (nrow(M) != ncol(Z[[fam]]))
      stop("markers[['", fam, "']] a ", nrow(M), " lignes mais l'incidence '",
           incidences[[fam]], "' a ", ncol(Z[[fam]]), " colonnes. Les doses sont ",
           "indexees par GENOTYPE, dans l'ordre des niveaux du terme.",
           call. = FALSE)
    markers[[fam]] <- M
  }
  if (length(unique(vapply(markers[names(Z)], ncol, 1L))) > 1L)
    stop("les matrices de doses n'ont pas le meme nombre de SNP.", call. = FALSE)
  markers
}

# SNPs kept by the minor allele frequency filter, computed on the direct doses
# with the coding of `coding`; monomorphic SNPs are dropped.
.rx_scan_maf <- function(markers, Z, maf, coding, verbose, p) {
  keep <- rep(TRUE, p)
  if (is.null(maf)) return(keep)
  Md <- markers[[if ("dir" %in% names(Z)) "dir" else names(Z)[1]]]
  f <- if (coding == "pm1") (colMeans(Md) + 1) / 2 else colMeans(Md) / 2
  keep <- pmin(f, 1 - f) >= maf & apply(Md, 2, stats::sd) > 0
  if (verbose)
    message(sprintf("  rx_scan : %d SNP sur %d passent MAF >= %.3g",
                    sum(keep), p, maf))
  keep
}

# The bundle of the fitted model, its theta, and the arrays of the scan
# (incidences in 0-based COO triplets, doses, kept SNPs, map) described by
# scan_manifest.json. Identical doses for both families are written once.
.rx_scan_write <- function(dir, model, fit, Z, markers, keep_snp, map, p) {
  rx_export(model, dir)
  con <- file(file.path(dir, "in_theta.bin"), "wb")
  writeBin(as.double(fit$theta), con, size = 8); close(con)
  w <- .rx_bundle_writer(dir)
  fams <- names(Z)
  shared <- length(fams) < 2 ||
    (identical(dim(markers[[fams[1]]]), dim(markers[[fams[2]]])) &&
       identical(markers[[fams[1]]], markers[[fams[2]]]))
  meta <- list(p = p)
  for (fam in fams) {
    S <- methods::as(methods::as(Z[[fam]], "CsparseMatrix"), "TsparseMatrix")
    pre <- paste0("scan_", fam)
    .rx_put(w, paste0(pre, "_zi"), S@i, "i4")
    .rx_put(w, paste0(pre, "_zj"), S@j, "i4")
    .rx_put(w, paste0(pre, "_zx"), S@x)
    meta[[paste0("q_", fam)]] <- ncol(Z[[fam]])
    if (!shared) .rx_put(w, paste0("scan_M", fam), markers[[fam]])
  }
  if (shared) .rx_put(w, "scan_M", markers[[fams[1]]])
  .rx_put(w, "scan_garder", as.integer(keep_snp), "i4")
  if (!is.null(map)) {
    if (!is.null(map$snp)) .rx_put(w, "scan_snp", map$snp, "str")
    if (!is.null(map$chr)) .rx_put(w, "scan_chr", map$chr, "str")
    if (!is.null(map$pos)) .rx_put(w, "scan_pos", as.double(map$pos))
  } else if (!is.null(colnames(markers[[names(Z)[1]]]))) {
    .rx_put(w, "scan_snp", colnames(markers[[names(Z)[1]]]), "str")
  }
  writeLines(jsonlite::toJSON(list(arrays = w$manifest, meta = meta), auto_unbox = TRUE),
             file.path(dir, "scan_manifest.json"))
}

# Bonferroni threshold over the SNPs actually tested.
rx_scan_threshold <- function(res, test, alpha = 0.05) {
  col <- paste0("p_", test)
  if (is.null(res[[col]]))
    stop("colonne '", col, "' absente ; tests disponibles : ",
         paste(sub("^p_", "", grep("^p_", names(res), value = TRUE)),
               collapse = ", "), call. = FALSE)
  m <- sum(is.finite(res[[col]]))
  alpha / max(m, 1L)
}

# Former name of rx_scan_threshold().
rx_scan_seuil <- function(res, test, alpha = 0.05) {
  .Deprecated("rx_scan_threshold", package = "remlax")
  rx_scan_threshold(res, test, alpha)
}

# Arguments renamed in English: a former name passed through `...` still works,
# with a deprecation warning, and is assigned to its new name in `env`. Returns
# the other arguments of `dots`, which the caller may forward.
.rx_renamed_args <- function(env, dots, renamed, fun, forward = FALSE) {
  if (!length(dots)) return(invisible(dots))
  nm <- names(dots)
  if (is.null(nm)) nm <- rep("", length(dots))
  old <- nm %in% names(renamed)
  if (!forward && any(!old))
    stop(fun, " : argument(s) inconnu(s) : ",
         paste(ifelse(nzchar(nm[!old]), nm[!old], "<sans nom>"), collapse = ", "), call. = FALSE)
  for (o in nm[old]) {
    new <- renamed[[o]]
    warning(sprintf("%s : l'argument `%s` est renomme `%s` ; l'ancien nom sera retire.",
                    fun, o, new), call. = FALSE)
    assign(new, dots[[o]], envir = env)
  }
  invisible(dots[!old])
}
