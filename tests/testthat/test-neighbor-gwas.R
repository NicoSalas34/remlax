# Neighbor GWAS (Sato et al. 2021): the univariate case against rNeighborGWAS,
# and the multi-trait case against univariate fits when the traits are
# independent (diag structures: V block diagonal by trait).

.nei_data <- function(n = 120, q = 150, seed = 7) {
  set.seed(seed)
  g <- matrix(sample(c(-1, 1), n * q, TRUE), n, q, dimnames = list(paste0("p", 1:n), paste0("m", 1:q)))
  smap <- cbind(runif(n, 1, 60), runif(n, 1, 60)); grp <- rep(1:2, each = n / 2)
  W <- rx_neighbor_weights(smap, 15, grp)
  gn <- g * as.matrix(W %*% g)
  y1 <- 0.6 * g[, 1] + 0.5 * gn[, 2] + as.numeric(g %*% rnorm(q, 0, 0.05)) + rnorm(n)
  y2 <- -0.4 * gn[, 2] + as.numeric(gn %*% rnorm(q, 0, 0.1)) + rnorm(n, 0, 1.3)
  list(g = g, smap = smap, grp = grp, W = W,
       d = data.frame(plant = rownames(g), y1 = y1, y2 = y2, grp = factor(grp)))
}

test_that("neighbour covariates and kinships follow rNeighborGWAS", {
  skip_if_not_installed("rNeighborGWAS")
  j <- .nei_data()
  kin <- rx_neighbor_kinship(j$g, j$W)
  expect_equal(unname(kin$X2), unname(rNeighborGWAS::nei_coval(j$g, j$smap, 15, grouping = j$grp)),
               tolerance = 1e-14)
  expect_error(rx_neighbor_weights(j$smap[, 1, drop = FALSE], 15), "two columns")
})

test_that("univariate Neighbor GWAS: same p-values as rNeighborGWAS::nei_lmm", {
  skip_if_not_installed("rNeighborGWAS")
  skip_si_pas_de_jax()
  j <- .nei_data()
  kin <- rx_neighbor_kinship(j$g, j$W)
  ref <- as.data.frame(rNeighborGWAS::nei_lmm(geno = j$g, g_nei = kin$X2, pheno = j$d$y1,
                                               addcovar = as.matrix(j$grp)))
  nb <- rx_neighbor_gwas(j$d, "y1", "plant", j$g, j$W, fixed = ~ grp, kin = kin,
                         fit_args = list(backend = "cpu"), verbose = FALSE)
  expect_equal(-log10(nb$scan$p_dir), -log10(ref$p_self), tolerance = 1e-4)
  expect_equal(-log10(nb$scan[["p_sim|dir"]]), -log10(ref$p_nei), tolerance = 1e-4)
  expect_true(all(c("share_self", "share_neighbour") %in% names(nb$pve)))
})

test_that("multi-trait Neighbor GWAS with independent traits = univariate scans", {
  skip_si_pas_de_jax()
  j <- .nei_data()
  kin <- rx_neighbor_kinship(j$g, j$W)
  tests <- c("dir", "sim|dir", "dir[y2]", "sim@het|dir", "sim@common|dir")
  mt <- rx_neighbor_gwas(j$d, c("y1", "y2"), "plant", j$g, j$W, struct = "diag", kin = kin,
                         tests = tests, fit_args = list(backend = "cpu"), verbose = FALSE)
  u1 <- rx_neighbor_gwas(j$d, "y1", "plant", j$g, j$W, kin = kin,
                         fit_args = list(backend = "cpu"), verbose = FALSE)
  u2 <- rx_neighbor_gwas(j$d, "y2", "plant", j$g, j$W, kin = kin,
                         fit_args = list(backend = "cpu"), verbose = FALSE)
  # V block diagonal by trait: the any-effect statistic is the sum of the two
  expect_equal(mt$scan$chi2_dir, u1$scan$chi2_dir + u2$scan$chi2_dir, tolerance = 1e-4)
  expect_equal(mt$scan[["chi2_sim|dir"]], u1$scan[["chi2_sim|dir"]] + u2$scan[["chi2_sim|dir"]],
               tolerance = 1e-4)
  expect_equal(mt$scan[["chi2_dir[y2]"]], u2$scan$chi2_dir, tolerance = 1e-4)
  expect_equal(mt$scan[["chi2_sim|dir"]],
               mt$scan[["chi2_sim@het|dir"]] + mt$scan[["chi2_sim@common|dir"]], tolerance = 1e-8)
  expect_equal(mt$pve$self, c(u1$pve$self, u2$pve$self), tolerance = 1e-4)
  expect_equal(unique(mt$scan[["df_sim|dir"]]), 2)
})
