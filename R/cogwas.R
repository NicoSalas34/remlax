# ==============================================================================
# cogwas.R -- co-GWAS: focal locus x neighbour locus, and LD pruning
# ------------------------------------------------------------------------------
# rx_ld_prune()    greedy pruning by r^2 within a window of SNPs, per chromosome
# ==============================================================================

#' Prune markers by linkage disequilibrium
#'
#' Greedy pruning in the spirit of PLINK `--indep-pairwise`. SNPs are visited
#' in the order of the map, chromosome by chromosome. A SNP is kept when its
#' r^2 with every SNP already kept among the previous `window` SNPs of its
#' chromosome is below `r2`. r^2 is computed across the rows of `markers`
#' (genotypes: lines, families or mothers), so identical profiles, or profiles
#' identical up to the sign of the coding, are always reduced to one.
#' Monomorphic SNPs are dropped.
#'
#' @param markers dose matrix, genotypes x SNPs.
#' @param map optional data frame with `chr` (and `pos`), one row per column of
#'   `markers`. Without it the SNPs form one chromosome, in column order.
#' @param r2 largest r^2 allowed between two kept SNPs of a window.
#' @param window number of preceding SNPs compared with each SNP.
#' @return the indices of the kept columns, in increasing order.
#' @export
rx_ld_prune <- function(markers, map = NULL, r2 = 0.8, window = 500L) {
  M <- as.matrix(markers)
  storage.mode(M) <- "double"
  p <- ncol(M)
  chr <- if (is.null(map)) rep(1L, p) else map$chr
  if (length(chr) != p) stop("rx_ld_prune: `map` must have one row per column of `markers`.",
                             call. = FALSE)
  ord <- if (!is.null(map$pos)) order(chr, map$pos) else order(chr, seq_len(p))
  rank <- integer(p); rank[ord] <- seq_len(p)
  Z <- scale(M, center = TRUE, scale = TRUE) / sqrt(max(nrow(M) - 1, 1))
  ok <- is.finite(colSums(Z))
  kept <- logical(p)
  last <- integer(0)
  prev_chr <- NULL
  for (j in ord) {
    if (!ok[j]) next
    if (!identical(chr[j], prev_chr)) { last <- integer(0); prev_chr <- chr[j] }
    if (length(last)) {
      r <- crossprod(Z[, last, drop = FALSE], Z[, j])
      if (max(r^2) >= r2) { last <- last[rank[last] > rank[j] - window]; next }
    }
    kept[j] <- TRUE
    last <- c(last[rank[last] > rank[j] - window], j)
  }
  which(kept)
}

#' Co-GWAS: every focal locus crossed with every neighbour locus
#'
#' For each pair of a focal locus a and a neighbour locus b, the covariate of
#' a plant is the product of its own dose at a and the dose of its neighbours
#' at b. The neighbours may belong to another species (wheat focal, alfalfa
#' neighbours), each with its own markers, or to the same one; with the same
#' locus on both sides the pair is the `sim` covariate of [rx_scan()].
#' Tests at fixed V, from the null model `fit`:
#' `"pair"`, `"pair|dir"`, `"pair|ind"`, `"pair|dir+ind"` (the pair given the
#' main effects of the two loci).
#'
#' Every term is computed in genotype space (the doses only depend on
#' genotypes), exactly, at a cost of q_ind^2 per pair instead of n^2. The
#' `top` pairs of each test are returned, with the smallest p-value per focal
#' and per neighbour locus and the genomic control lambda over all pairs.
#' Prune the markers first ([rx_ld_prune()]): the number of pairs is the
#' product of the two numbers of SNPs.
#'
#' The null model should hold the polygenic background of the interaction,
#' for instance a term of plants with the kinship
#' `(Z_g K_g Z_g') * (Z_n K_n Z_n')` (Hadamard product), the analogue of the
#' second kinship of Sato et al. (2021). Without it, an interaction between
#' focal and neighbour genotypes that the null does not model inflates the
#' tests of pairs.
#'
#' @param fit fit of the null model ([rx_fit()]).
#' @param model the [rx_model] of `fit`.
#' @param focal dose matrix q_dir x p_a, rows in the order of the levels of the
#'   focal term, centred doses (+/-1 or centred dosages).
#' @param neighbour dose matrix q_ind x p_b, rows in the order of the columns of
#'   the neighbour incidence.
#' @param incidences named vector: `dir` (focal term) and `ind` (neighbour
#'   term), as in [rx_scan()].
#' @param tests tests among `"pair"`, `"pair|dir"`, `"pair|ind"`, `"pair|dir+ind"`.
#' @param top number of pairs returned per test.
#' @param backend `"auto"`, `"gpu"` or `"cpu"`.
#' @param block_a,block_b numbers of focal and neighbour loci per block.
#' @param verbose print progress.
#' @return a list: `top` (data frame: test, rank, focal and neighbour SNP
#'   names and indices, chi2, p, beta, se), `focal` and `neighbour` (smallest
#'   p per locus and test), and `meta` (number of pairs, lambda per test, times).
#' @export
rx_cogwas <- function(fit, model, focal, neighbour, incidences = c(dir = "gen", ind = "voisin"),
                      tests = "pair|dir+ind", top = 1000L, backend = c("auto", "gpu", "cpu"),
                      block_a = 1024L, block_b = 8192L, verbose = TRUE) {
  backend <- match.arg(backend)
  stopifnot(is.list(fit), !is.null(fit$theta), all(c("dir", "ind") %in% names(incidences)))
  if (!is.null(model$lag)) stop("rx_cogwas: not available with a spatial lag.", call. = FALSE)
  Z <- .rx_scan_incidences(model, incidences[c("dir", "ind")])
  if (is.list(Z$dir) || is.list(Z$ind))
    stop("rx_cogwas: multi-trait models are not supported yet.", call. = FALSE)
  focal <- as.matrix(focal); neighbour <- as.matrix(neighbour)
  if (nrow(focal) != ncol(Z$dir) || nrow(neighbour) != ncol(Z$ind))
    stop("rx_cogwas: `focal` needs one row per focal genotype (", ncol(Z$dir),
         ") and `neighbour` one row per neighbour genotype (", ncol(Z$ind), ").", call. = FALSE)
  dir <- tempfile("rx_cogwas_")
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  dir.create(dir, recursive = TRUE)
  dir <- normalizePath(dir, mustWork = TRUE)
  rx_export(model, dir)
  .rx_write_f64(file.path(dir, "in_theta.bin"), fit$theta)
  w <- .rx_bundle_writer(dir)
  meta <- c(.rx_scan_write_incidence(w, "dir", Z$dir), .rx_scan_write_incidence(w, "ind", Z$ind))
  .rx_put(w, "cog_Ma", focal)
  .rx_put(w, "cog_Mb", neighbour)
  writeLines(jsonlite::toJSON(list(arrays = w$manifest, meta = meta), auto_unbox = TRUE),
             file.path(dir, "cogwas_manifest.json"))
  py <- rx_python_cmd()
  .rx_python_require(py)
  args <- c(shQuote(c(.rx_launcher(), "cogwas_cli")), shQuote(dir),
            "--tests", shQuote(paste(tests, collapse = ",")), "--top", as.character(as.integer(top)),
            "--backend", backend, "--block-a", as.character(as.integer(block_a)),
            "--block-b", as.character(as.integer(block_b)))
  if (!verbose) args <- c(args, "--quiet")
  code <- .rx_system2(py, args, stdout = "", stderr = "", env = character(0))
  if (!identical(as.integer(code), 0L))
    stop("rx_cogwas: the solver returned the code ", code, call. = FALSE)
  rd <- function(f) utils::read.csv(file.path(dir, f), stringsAsFactors = FALSE, check.names = FALSE)
  top_tab <- rd("cogwas_top.csv")
  nm_a <- colnames(focal); nm_b <- colnames(neighbour)
  if (!is.null(nm_a)) top_tab$focal_snp <- nm_a[top_tab$focal]
  if (!is.null(nm_b)) top_tab$neighbour_snp <- nm_b[top_tab$neighbour]
  fo <- rd("cogwas_focal.csv"); ne <- rd("cogwas_neighbour.csv")
  if (!is.null(nm_a)) fo <- cbind(snp = nm_a, fo)
  if (!is.null(nm_b)) ne <- cbind(snp = nm_b, ne)
  list(top = top_tab, focal = fo, neighbour = ne,
       meta = jsonlite::fromJSON(file.path(dir, "cogwas_meta.json")))
}
