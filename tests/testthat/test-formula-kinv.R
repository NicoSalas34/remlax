# A relationship given by its precision in the formula, vm(id, Kinv = Ai), as
# a pedigree inverse is usually given. Before 0.2.1 the formula read only K:
# Kinv was taken for K by partial matching of `$`, and any other unknown
# argument was dropped without a word.

# Small pedigree: 4 founders, 6 offspring, 6 grand-offspring. Records only on
# the 12 non-founders, so that the founders live in Kinv only.
.small_ped <- function() {
  data.frame(id = paste0("i", 1:16),
             sire = c(NA, NA, NA, NA, "i1", "i1", "i1", "i3", "i3", "i3",
                      "i5", "i5", "i8", "i8", "i6", "i9"),
             dam  = c(NA, NA, NA, NA, "i2", "i2", "i4", "i4", "i2", "i4",
                      "i7", "i10", "i9", "i7", "i10", "i7"))
}
.a_matrix <- function(ped) {
  n <- nrow(ped); A <- diag(n); idx <- setNames(seq_len(n), ped$id)
  for (i in seq_len(n)) {
    s <- idx[ped$sire[i]]; d <- idx[ped$dam[i]]
    for (j in seq_len(i - 1L)) {
      v <- 0.5 * ((if (!is.na(s)) A[j, s] else 0) + (if (!is.na(d)) A[j, d] else 0))
      A[i, j] <- A[j, i] <- v
    }
    A[i, i] <- 1 + (if (!is.na(s) && !is.na(d)) 0.5 * A[s, d] else 0)
  }
  dimnames(A) <- list(ped$id, ped$id)
  A
}

test_that("vm(id, Kinv =) matches vm(id, K =), founders without records included", {
  skip_si_pas_de_jax()
  ped <- .small_ped(); A <- .a_matrix(ped)
  Ai <- Matrix::Matrix(solve(A), sparse = TRUE); dimnames(Ai) <- dimnames(A)
  set.seed(11)
  u <- drop(t(chol(A)) %*% rnorm(16))
  d <- data.frame(id = rep(ped$id[5:16], each = 3))
  d$y <- 2 + 0.8 * u[match(d$id, ped$id)] + rnorm(nrow(d), 0, 0.6)
  fk <- rx_reml(y ~ 1, random = ~ vm(id, K = A), data = d, backend = "cpu",
                verbose = FALSE)
  fi <- rx_reml(y ~ 1, random = ~ vm(id, Kinv = Ai), data = d, backend = "cpu",
                verbose = FALSE, engine = "mme")
  expect_equal(fi$logLik, fk$logLik, tolerance = 1e-7)
  expect_equal(fi$sigmas$id[1], fk$sigmas$id[1], tolerance = 1e-4)
  # one BLUP per row of Kinv, in its order; the recorded ones agree with K
  expect_identical(rownames(fi$blups$id), ped$id)
  expect_equal(fi$blups$id[rownames(fk$blups$id), 1], fk$blups$id[, 1], tolerance = 1e-4)
  # rows of Kinv in another order: aligned by name
  o <- c(16:1); Aio <- Ai[o, o]
  fo <- rx_reml(y ~ 1, random = ~ vm(id, Kinv = Aio), data = d, backend = "cpu",
                verbose = FALSE, engine = "mme")
  expect_equal(fo$logLik, fk$logLik, tolerance = 1e-7)
})

test_that("Kinv in the formula: checks", {
  ped <- .small_ped(); A <- .a_matrix(ped)
  Ai <- Matrix::Matrix(solve(A), sparse = TRUE); dimnames(Ai) <- dimnames(A)
  d <- data.frame(id = rep(ped$id[5:16], each = 2), y = rnorm(24))
  p <- function(f) remlax:::.rx_parse_random(f, d)
  # an unknown argument is refused instead of being dropped
  expect_error(p(~ vm(id, Kinverse = Ai)), "unknown argument")
  expect_error(p(~ vm(id, A, Ai)), "too many unnamed")
  # Kinv is read as Kinv, never as K
  tm <- p(~ vm(id, Kinv = Ai))[[1]]
  expect_null(tm$LK); expect_equal(nrow(tm$Kinv), 16L); expect_equal(tm$q, 16L)
  # a level absent from Kinv
  d2 <- rbind(d, data.frame(id = "zz", y = 0))
  expect_error(remlax:::.rx_parse_random(~ vm(id, Kinv = Ai), d2), "missing from Kinv")
  # without names, one row per level is required
  Au <- Ai; dimnames(Au) <- NULL
  expect_error(p(~ vm(id, Kinv = Au)), "no names")
  # the dense engine refuses a term known only by its precision
  expect_error(rx_reml(y ~ 1, random = ~ vm(id, Kinv = Ai), data = d, backend = "cpu",
                       verbose = FALSE, engine = "dense"), "PRECISION")
})
