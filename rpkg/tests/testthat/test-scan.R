# Scan GWAS a V figee : refus en pur R, puis un scan complet sur un jeu simule
# (skip sans Python porteur de jax). Le SNP causal doit ressortir en tete, la
# table doit porter une colonne par test, et rx_scan_seuil compte les SNP
# effectivement testes.

.jeu_scan <- function(seed = 5, q = 30, rep = 4, p = 40) {
  set.seed(seed)
  n <- q * rep
  gen <- factor(rep(seq_len(q), rep))
  vois <- factor(c(as.integer(gen)[-1], as.integer(gen)[1]), levels = levels(gen))
  M <- matrix(sample(c(-1, 1), q * p, replace = TRUE), q, p,
              dimnames = list(levels(gen), paste0("snp", seq_len(p))))
  y <- 2 + 1.2 * M[as.integer(gen), 1] + rnorm(q, 0, 0.6)[gen] + rnorm(n)
  list(d = data.frame(y = y, gen = gen, vois = vois), M = M, n = n, q = q, p = p)
}

test_that("rx_scan refuse ce qui ne peut pas s'interpreter, avant tout appel Python", {
  j <- .jeu_scan()
  m <- rx_model(j$d$y, matrix(1, j$n, 1),
                list(rx_term("gen", j$d$gen), rx_term("vois", j$d$vois)))
  faux <- list(theta = c(0, 0, 0))
  expect_error(rx_scan(faux, m, j$M, incidences = c(dir = "gen", zzz = "vois"), verbose = FALSE),
               "inconnu")
  expect_error(rx_scan(faux, m, j$M, incidences = c(dir = "absent"), verbose = FALSE),
               "pas dans le modele")
  expect_error(rx_scan(faux, m, j$M[1:10, ], incidences = c(dir = "gen"), verbose = FALSE),
               "lignes")
  expect_error(rx_scan(faux, m, list(dir = j$M), incidences = c(dir = "gen", ind = "vois"),
                       verbose = FALSE), "absente")
  m2 <- rx_model(j$d$y, matrix(1, j$n, 1),
                 list(rx_term("dge_ige", list(as.matrix(rx_term("gen", j$d$gen)$Zl[[1]]),
                                              as.matrix(rx_term("vois", j$d$vois)$Zl[[1]])),
                              struct = "us")))
  expect_error(rx_scan(faux, m2, j$M, incidences = c(dir = "dge_ige"), verbose = FALSE),
               "Preciser")
  expect_error(rx_scan(faux, m2, j$M, incidences = c(dir = "dge_ige:3"), verbose = FALSE),
               "hors de")
  res <- data.frame(p_dir = c(0.1, NA, 0.5, 0.01))
  expect_equal(rx_scan_seuil(res, "dir"), 0.05 / 3)
  expect_equal(rx_scan_seuil(res, "dir", alpha = 0.1), 0.1 / 3)
  expect_error(rx_scan_seuil(res, "ind"), "absente")
})

test_that("rx_scan sur un jeu simule : colonnes, causal en tete, seuil", {
  skip_si_pas_de_jax()
  j <- .jeu_scan()
  m <- rx_model(j$d$y, matrix(1, j$n, 1),
                list(rx_term("gen", j$d$gen), rx_term("vois", j$d$vois)))
  f <- rx_fit(m, backend = "cpu", verbose = FALSE, hessian = FALSE)
  tests <- c("dir", "ind", "dir+ind", "ind|dir")
  sc <- rx_scan(f, m, j$M, incidences = c(dir = "gen", ind = "vois"), tests = tests,
                maf = 0.05, codage = "pm1", backend = "cpu", verbose = FALSE)
  expect_s3_class(sc, "data.frame")
  expect_equal(nrow(sc), j$p)
  expect_equal(sc$snp[1:2], c("snp1", "snp2"))
  for (t in tests) {
    expect_true(all(c(paste0("chi2_", t), paste0("p_", t), paste0("ddl_", t)) %in% names(sc)),
                info = t)
  }
  expect_true(all(c("beta_dir", "se_dir", "r_dir_ind") %in% names(sc)))
  expect_equal(unique(sc[["ddl_dir+ind"]]), 2)
  expect_equal(which.min(sc$p_dir), 1L)
  expect_true(sc$p_dir[1] < rx_scan_seuil(sc, "dir"))
  expect_equal(sc$chi2_dir, (sc$beta_dir / sc$se_dir)^2, tolerance = 1e-8)
  meta <- attr(sc, "meta")
  expect_equal(meta$p, j$p)
  expect_equal(meta$neg2_reml, -2 * f[["logLik"]], tolerance = 1e-8)
  expect_true(is.finite(meta$lambda_gc[["dir"]]))
  # doses 0/1/2 avec la famille sim : le solveur refuse (doses non centrees)
  expect_error(rx_scan(f, m, j$M + 1, incidences = c(dir = "gen", ind = "vois"),
                       tests = c("sim"), codage = "012", backend = "cpu", verbose = FALSE),
               "code")
})
