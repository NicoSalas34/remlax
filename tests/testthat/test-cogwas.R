# rx_cogwas : la diagonale (paire j, j) egale le test sim de rx_scan, qui
# accepte une matrice de doses par famille (sim|dir+ind en intra-espece). Cas inter-especes (focales et
# voisines genotypees a part) et intra-espece (memes doses des deux cotes).

.jeu_cogwas <- function(seed = 2, qg = 25, qf = 15, n = 300, p = 8, same = FALSE) {
  set.seed(seed)
  g <- factor(sprintf("G%02d", sample(qg, n, TRUE)), levels = sprintf("G%02d", seq_len(qg)))
  qn <- if (same) qg else qf
  W <- matrix(0, n, qn)
  for (k in 1:3) W[cbind(seq_len(n), sample(qn, n, TRUE))] <- W[cbind(seq_len(n), sample(qn, n, TRUE))] + 1 / 3
  Ma <- matrix(sample(c(-1, 1), qg * p, TRUE), qg, p, dimnames = list(levels(g), paste0("a", 1:p)))
  Mb <- if (same) Ma else matrix(sample(c(-1, 1), qn * p, TRUE), qn, p, dimnames = list(NULL, paste0("b", 1:p)))
  Kg <- tcrossprod(Ma) / p + diag(0.2, qg)
  Kn <- tcrossprod(Mb) / p + diag(0.2, qn)
  u <- Ma[as.integer(g), 3] * drop(W %*% Mb[, 5])
  y <- 1 + rnorm(qg)[g] + drop(W %*% rnorm(qn)) * 0.5 + rnorm(n) + 0.8 * u
  model <- rx_model(y, matrix(1, n, 1),
                    list(rx_term("gen", g, K = Kg), rx_term("voisin", W, K = Kn, t = 1)), rx_residual())
  list(model = model, Ma = Ma, Mb = Mb)
}

test_that("rx_cogwas : diagonale = sim de rx_scan, inter et intra espece", {
  skip_si_pas_de_jax()
  for (same in c(FALSE, TRUE)) {
    j <- .jeu_cogwas(same = same)
    f <- rx_fit(j$model, verbose = FALSE, backend = "cpu", hessian = FALSE)
    p <- ncol(j$Ma)
    cg <- rx_cogwas(f, j$model, j$Ma, j$Mb, tests = c("pair|dir+ind", "pair"), top = p * p,
                    backend = "cpu", verbose = FALSE)
    expect_equal(cg$meta$n_pairs, p * p)
    # rx_scan refuses joint tests across two marker sets (column j is not the
    # same SNP on both sides): between species only `sim` is compared here;
    # pair|dir+ind between species is checked against a dense GLS in
    # tests-python/test_cogwas.py.
    tt_all <- if (same) c("pair|dir+ind", "pair") else "pair"
    sc <- rx_scan(f, j$model, if (same) j$Ma else list(dir = j$Ma, ind = j$Mb),
                  tests = sub("pair", "sim", tt_all, fixed = TRUE),
                  maf = NULL, backend = "cpu", verbose = FALSE)
    for (tt in tt_all) {
      tp <- cg$top[cg$top$test == tt, ]
      d <- tp[tp$focal == tp$neighbour, ]
      d <- d[order(d$focal), ]
      expect_equal(d$chi2, sc[[paste0("chi2_", sub("pair", "sim", tt, fixed = TRUE))]],
                   tolerance = 1e-7, info = paste(same, tt))
    }
    best <- cg$top[cg$top$test == "pair|dir+ind", ][1, ]
    expect_identical(c(best$focal_snp, best$neighbour_snp),
                     c(colnames(j$Ma)[3], colnames(j$Mb)[5]), info = same)
    expect_equal(nrow(cg$focal), p); expect_equal(nrow(cg$neighbour), p)
  }
})
