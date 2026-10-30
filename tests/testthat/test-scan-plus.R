# rx_scan_loco et rx_scan_exact : refus en pur R, puis un jeu simule a trois
# chromosomes (skip sans Python porteur de jax). On verifie la structure des
# sorties (un nul par chromosome, chaque SNP scanne sous le nul de son
# chromosome) et, contre un ajustement direct dont la colonne du SNP est
# construite ici independamment, l'effet et l'erreur-type du reajustement exact.

.jeu_loco <- function(seed = 3, q = 40, rep = 3, n_chr = 3, m_chr = 20, causal = 25) {
  set.seed(seed)
  n <- q * rep
  gen <- factor(sprintf("G%02d", rep(seq_len(q), rep)))
  p <- n_chr * m_chr
  M <- matrix(sample(c(-1, 1), q * p, replace = TRUE), q, p,
              dimnames = list(levels(gen), paste0("snp", seq_len(p))))
  map <- data.frame(snp = colnames(M), chr = rep(seq_len(n_chr), each = m_chr),
                      pos = rep(seq_len(m_chr), n_chr))
  y <- 1 + 1.0 * M[as.integer(gen), causal] + rnorm(q, 0, 0.5)[gen] + rnorm(n)
  K_of <- function(W) {
    W <- scale(W, scale = FALSE)
    K <- tcrossprod(W) / ncol(W) + diag(0.01, nrow(W))
    dimnames(K) <- list(rownames(W), rownames(W)); K
  }
  build <- function(K, X_sup = NULL)
    rx_model(y, cbind(rep(1, length(y)), X_sup), list(rx_term("gen", gen, K = K)), rx_residual())
  list(y = y, gen = gen, M = M, map = map, K_of = K_of, build = build,
       causal = colnames(M)[causal])
}

test_that("rx_scan_loco et rx_scan_exact refusent ce qui ne peut pas s'interpreter", {
  j <- .jeu_loco()
  un_chr <- j$map; un_chr$chr <- 1
  expect_error(rx_scan_loco(j$build, j$K_of, j$M, un_chr, incidences = c(dir = "gen"),
                            tests = "dir", verbose = FALSE), "deux chromosomes")
  res <- data.frame(snp = colnames(j$M), p_dir = runif(ncol(j$M)))
  expect_error(rx_scan_exact(res, "sim", j$build, j$K_of, j$M, j$map, c(dir = "gen"),
                             verbose = FALSE), "test lineaire")
  expect_error(rx_scan_exact(res, "dir|ind", j$build, j$K_of, j$M, j$map, c(dir = "gen"),
                             verbose = FALSE), "test lineaire")
  expect_error(rx_scan_exact(res, "ind", j$build, j$K_of, j$M, j$map, c(dir = "gen"),
                             verbose = FALSE), "absente")
  expect_error(rx_scan_exact(res["p_dir"], "dir", j$build, j$K_of, j$M, j$map,
                             c(dir = "gen"), verbose = FALSE), "snp")
})

test_that("rx_scan_loco : un nul par chromosome, chaque SNP sous le nul de son chromosome", {
  skip_si_pas_de_jax()
  j <- .jeu_loco()
  fa <- list(backend = "cpu", hessian = FALSE)
  lo <- rx_scan_loco(j$build, j$K_of, j$M, j$map, incidences = c(dir = "gen"),
                     tests = "dir", fit_args = fa, verbose = FALSE, backend = "cpu")
  expect_equal(nrow(lo), ncol(j$M))
  expect_setequal(lo$snp, colnames(j$M))
  expect_identical(as.character(lo$chr_loco),
                   as.character(j$map$chr[match(lo$snp, j$map$snp)]))
  nuls <- attr(lo, "loco")
  expect_equal(nrow(nuls), 3L)
  expect_equal(nuls$n_snp, c(20L, 20L, 20L))
  expect_true(all(nuls$converge))
  expect_true(all(is.finite(nuls$logLik)))
  expect_named(attr(lo, "lambda"), "dir")
  expect_identical(lo$snp[which.min(lo$p_dir)], j$causal)
  # le nul du chromosome 1 est bien celui de la parente sans ce chromosome
  K1 <- j$K_of(j$M[, j$map$chr != 1])
  f1 <- do.call(rx_fit, c(list(j$build(K1), verbose = FALSE), fa))
  expect_equal(nuls$logLik[nuls$chr == "1"], f1$logLik, tolerance = 1e-6)
})

test_that("rx_scan_exact : effet et erreur-type egaux a un ajustement direct", {
  skip_si_pas_de_jax()
  j <- .jeu_loco()
  fa <- list(backend = "cpu", hessian = FALSE)
  K <- j$K_of(j$M)
  f0 <- do.call(rx_fit, c(list(j$build(K), verbose = FALSE), fa))
  sc <- rx_scan(f0, j$build(K), j$M, map = j$map, incidences = c(dir = "gen"),
                tests = "dir", maf = NULL, backend = "cpu", verbose = FALSE)
  for (loco in c(FALSE, TRUE)) {
    ex <- rx_scan_exact(sc, "dir", j$build, j$K_of, j$M, j$map, c(dir = "gen"),
                        top = 3, loco = loco, fit_args = fa, verbose = FALSE)
    expect_equal(nrow(ex), 3L, info = loco)
    expect_true(all(c("p_fige", "chi2_exact", "p_exact", "beta_dir", "se_dir", "converge") %in%
                      names(ex)), info = loco)
    expect_true(all(ex$converge), info = loco)
    expect_identical(ex$snp[1], j$causal, info = loco)
    expect_lt(ex$p_exact[1], ex$p_fige[1])
    # colonne du SNP construite ici, a partir des doses et du facteur genotype
    s <- ex$snp[1]
    Ks <- if (loco) j$K_of(j$M[, j$map$chr != j$map$chr[j$map$snp == s]]) else K
    x <- j$M[as.character(j$gen), s]
    fd <- do.call(rx_fit, c(list(j$build(Ks, X_sup = cbind(snp = x)), verbose = FALSE), fa))
    k <- length(fd$beta)
    expect_equal(ex$beta_dir[1], fd$beta[k], tolerance = 1e-6, info = loco)
    expect_equal(ex$se_dir[1], sqrt(fd$vbeta[k, k]), tolerance = 1e-6, info = loco)
    expect_equal(ex$p_exact[1], stats::pchisq(fd$beta[k]^2 / fd$vbeta[k, k], 1, lower.tail = FALSE),
                 tolerance = 1e-6, info = loco)
  }
})
