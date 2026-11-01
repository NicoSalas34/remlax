# rx_ld_prune : doublons et inversions de codage reduits a un, monomorphes
# ecartes, fenetre remise a zero a chaque chromosome, et aucun couple garde
# dans une fenetre au-dela du seuil (verifie par force brute).

test_that("rx_ld_prune garde un SNP par profil et respecte le seuil dans la fenetre", {
  set.seed(1)
  q <- 40; p <- 60
  M <- matrix(sample(c(-1, 1), q * p, TRUE), q, p)
  M[, 2] <- M[, 1]; M[, 3] <- -M[, 1]          # doublon et inversion
  M[, 10] <- 1                                  # monomorphe
  M[, 20] <- ifelse(runif(q) < 0.9, M[, 19], -M[, 19])   # forte DL avec 19
  keep <- rx_ld_prune(M, r2 = 0.5, window = 100)
  expect_true(1 %in% keep)
  expect_false(any(c(2, 3, 10) %in% keep))
  expect_false(all(c(19, 20) %in% keep))
  r2 <- cor(M[, keep])^2
  expect_true(all(r2[upper.tri(r2)] < 0.5))
  # chromosomes : la fenetre repart de zero
  map <- data.frame(chr = rep(1:2, each = 30), pos = c(1:30, 1:30))
  M2 <- M; M2[, 31] <- M2[, 1]
  k2 <- rx_ld_prune(M2, map, r2 = 0.5, window = 100)
  expect_true(all(c(1, 31) %in% k2))
  # fenetre : deux doublons eloignes de plus d'une fenetre sont gardes
  M3 <- M; M3[, 60] <- M3[, 1]
  expect_true(60 %in% rx_ld_prune(M3, r2 = 0.5, window = 5))
  expect_false(60 %in% rx_ld_prune(M3, r2 = 0.5, window = 100))
})
