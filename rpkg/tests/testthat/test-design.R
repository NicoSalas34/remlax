# Tests de rx_neighbourhood, rx_exposure et rx_grm sur un dispositif jouet.
# Pur R : aucun solveur n'est appele. La validation sur les donnees reelles du
# chapitre 3 vit dans validation/ch3_neighbourhood_vs_ige.R (skip sans donnees).

# Grille 6 x 6, deux groupes en colonnes alternees (A en colonnes impaires, B en
# colonnes paires), un bloc, un niveau par paire de lignes.
jouet <- function(blocs = 1L) {
  g <- expand.grid(row = 1:6, col = 1:6)
  d <- do.call(rbind, lapply(seq_len(blocs), function(b) cbind(g, block = b)))
  d$group <- ifelse(d$col %% 2L == 1L, "A", "B")
  d$level <- paste0(d$group, (d$row + 1L) %/% 2L, "_", d$block)
  d$id <- paste0("u", seq_len(nrow(d)))
  d
}

test_that("rx_neighbourhood : comptes de voisins a rank 1 sur la grille alternee (T1)", {
  d <- jouet()
  nb <- rx_neighbourhood(d[, c("row", "col")], d$group, level = d$level, id = d$id,
                         rank = 1, output = "both")
  # unite interieure de A : 2 conspecifiques (dessus, dessous), 6 heterospecifiques
  int <- d$group == "A" & d$row %in% 2:5 & d$col %in% 3:5
  expect_true(all(nb$n_neighbours[["A<-A"]][d$id[int]] == 2L))
  expect_true(all(nb$n_neighbours[["A<-B"]][d$id[int]] == 6L))
  # les sommes de lignes de unit valent n_i ; level %*% 1 = unit %*% 1
  for (pr in names(nb$unit)) {
    expect_equal(as.numeric(Matrix::rowSums(nb$unit[[pr]])), as.numeric(nb$n_neighbours[[pr]]))
    expect_equal(as.numeric(Matrix::rowSums(nb$level[[pr]])),
                 as.numeric(Matrix::rowSums(nb$unit[[pr]])))
  }
  expect_equal(dim(nb$level[["A<-B"]]), c(18L, 3L))
  expect_equal(colnames(nb$level[["A<-B"]]), sort(unique(d$level[d$group == "B"])))
  expect_s3_class(nb, "rx_neighbourhood")
  expect_output(print(nb), "Voisinage")
})

test_that("rx_neighbourhood : poids en distance physique, spacing c(5, 5) (T2)", {
  d <- jouet()
  nb <- rx_neighbourhood(d[, c("row", "col")], d$group, id = d$id, rank = 1, reach = 1,
                         spacing = c(5, 5), output = "unit")
  U <- as.matrix(nb$unit[["A<-A"]])
  i <- which(d$row == 3 & d$col == 3); j <- which(d$row == 4 & d$col == 3)
  expect_equal(U[d$id[i], d$id[j]], 1 / 5)
  V <- as.matrix(nb$unit[["A<-B"]])
  k <- which(d$row == 4 & d$col == 4)
  expect_equal(V[d$id[i], d$id[k]], 1 / sqrt(50))
  nb2 <- rx_neighbourhood(d[, c("row", "col")], d$group, id = d$id, rank = 1, reach = 2,
                          spacing = c(5, 5), output = "unit")
  expect_equal(as.matrix(nb2$unit[["A<-A"]])[d$id[i], d$id[j]], 1 / 25)
  expect_equal(as.matrix(nb2$unit[["A<-B"]])[d$id[i], d$id[k]], 1 / 50)
  # noyau "none" : poids 1 quel que soit reach ; "exponential" : 1 au pas minimal
  nb3 <- rx_neighbourhood(d[, c("row", "col")], d$group, id = d$id, rank = 1, reach = 3,
                          kernel = "none", spacing = c(5, 5), output = "unit")
  expect_equal(as.matrix(nb3$unit[["A<-B"]])[d$id[i], d$id[k]], 1)
  nb4 <- rx_neighbourhood(d[, c("row", "col")], d$group, id = d$id, rank = 1, reach = 0.1,
                          kernel = "exponential", spacing = c(5, 5), output = "unit")
  expect_equal(as.matrix(nb4$unit[["A<-A"]])[d$id[i], d$id[j]], 1)
  expect_equal(as.matrix(nb4$unit[["A<-B"]])[d$id[i], d$id[k]], exp(-0.1 * (sqrt(50) - 5)))
})

test_that("rx_neighbourhood : dilution par le NOMBRE de voisins (T3, T4)", {
  d <- jouet()
  co <- d[, c("row", "col")]
  n1 <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = 2, reach = 0,
                         dilution = 1, output = "both")
  rs <- Matrix::rowSums(n1$level[["A<-A"]])
  expect_true(all(abs(rs[n1$n_neighbours[["A<-A"]] > 0] - 1) < 1e-12))
  n0 <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = 2, reach = 1.5,
                         dilution = 0, spacing = c(5, 5), output = "both")
  nd <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = 2, reach = 1.5,
                         dilution = 0.7, spacing = c(5, 5), output = "both")
  for (pr in c("A<-A", "A<-B")) {
    ni <- pmax(n0$n_neighbours[[pr]], 1)
    expect_equal(as.matrix(nd$level[[pr]]), as.matrix(n0$level[[pr]]) / ni^0.7)
    expect_equal(as.matrix(nd$unit[[pr]]), as.matrix(n0$unit[[pr]]) / ni^0.7)
  }
  # a lambda = 0 la dilution complete est une moyenne au sens strict
  expect_equal(unname(rs[rs > 0]), rep(1, sum(rs > 0)))
})

test_that("rx_neighbourhood : normalisation L2 apres dilution (T5)", {
  d <- jouet()
  nb <- rx_neighbourhood(d[, c("row", "col")], d$group, level = d$level, id = d$id, rank = 2,
                         reach = 1, dilution = 0.5, spacing = c(5, 5), normalise = TRUE,
                         output = "both")
  for (pr in names(nb$level)) {
    n2 <- Matrix::rowSums(nb$level[[pr]]^2)
    expect_true(all(abs(n2 - 1) < 1e-12 | abs(n2) < 1e-12))
    n2u <- Matrix::rowSums(nb$unit[[pr]]^2)
    expect_true(all(abs(n2u - 1) < 1e-12 | abs(n2u) < 1e-12))
  }
})

test_that("rx_neighbourhood : rang par paire = rang scalaire paire a paire (T6)", {
  d <- jouet()
  co <- d[, c("row", "col")]
  nbp <- rx_neighbourhood(co, d$group, level = d$level, id = d$id,
                          rank = list("A<-A" = 1, "A<-B" = 3), pairs = c("A<-A", "A<-B"))
  nb1 <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = 1)
  nb3 <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = 3)
  expect_equal(as.matrix(nbp$level[["A<-A"]]), as.matrix(nb1$level[["A<-A"]]))
  expect_equal(as.matrix(nbp$level[["A<-B"]]), as.matrix(nb3$level[["A<-B"]]))
  # forme matricielle : lignes = recevant, colonnes = emettant
  R <- matrix(c(1, 2, 3, 4), 2, 2, dimnames = list(c("A", "B"), c("A", "B")))
  nbm <- rx_neighbourhood(co, d$group, level = d$level, id = d$id, rank = R)
  expect_equal(nbm$params$rank[nbm$params$pair == "A<-B"], 3)
  expect_equal(nbm$params$rank[nbm$params$pair == "B<-A"], 2)
  expect_equal(nbm$params$pair, c("A<-A", "A<-B", "B<-A", "B<-B"))
  expect_error(rx_neighbourhood(co, d$group, level = d$level, rank = list("A<-A" = 1)),
               "ne donne rien")
  expect_error(rx_neighbourhood(co, d$group, level = d$level, rank = 1, pairs = "A<-C"),
               "inconnue")
})

test_that("rx_neighbourhood : deux blocs ne se voient pas (T7) ; rank 0 = vide", {
  d <- jouet(2L)
  nb <- rx_neighbourhood(d[, c("row", "col")], d$group, block = d$block, level = d$level,
                         id = d$id, rank = 6, output = "both")
  U <- as.matrix(nb$unit[["A<-B"]])
  ia <- d$id[d$group == "A" & d$block == 1]; jb <- d$id[d$group == "B" & d$block == 2]
  expect_true(all(U[ia, jb] == 0))
  expect_true(all(U[ia, d$id[d$group == "B" & d$block == 1]] > 0))
  n0 <- rx_neighbourhood(d[, c("row", "col")], d$group, block = d$block, level = d$level,
                         id = d$id, rank = 0)
  expect_equal(sum(n0$level[["A<-A"]]), 0)
  # refus : coordonnees non entieres, id en double, level absent
  expect_error(rx_neighbourhood(d[, c("row", "col")] + 0.5, d$group, level = d$level, rank = 1),
               "ENTIERES")
  expect_error(rx_neighbourhood(d[, c("row", "col")], d$group, level = d$level,
                                id = rep("a", nrow(d)), rank = 1), "doublons")
  expect_error(rx_neighbourhood(d[, c("row", "col")], d$group, rank = 1), "level")
})

test_that("rx_neighbourhood : sparse = FALSE rend des matrices denses identiques", {
  d <- jouet()
  a <- rx_neighbourhood(d[, c("row", "col")], d$group, level = d$level, id = d$id, rank = 2,
                        reach = 1, output = "both", sparse = TRUE)
  b <- rx_neighbourhood(d[, c("row", "col")], d$group, level = d$level, id = d$id, rank = 2,
                        reach = 1, output = "both", sparse = FALSE)
  expect_true(is.matrix(b$level[["A<-A"]]))
  expect_s4_class(a$level[["A<-A"]], "dgCMatrix")
  expect_equal(as.matrix(a$level[["A<-A"]]), b$level[["A<-A"]])
  expect_equal(as.matrix(a$unit[["B<-A"]]), b$unit[["B<-A"]])
})

# ---- rx_exposure -------------------------------------------------------------
test_that("rx_exposure : K = NULL, K = I, un seul voisin (T1, T2, T3)", {
  set.seed(3)
  n <- 40; q <- 8
  g <- sample(q, n, TRUE)
  Zd <- Matrix::sparseMatrix(i = seq_len(n), j = g, x = 1, dims = c(n, q))
  Zn <- Matrix::Matrix(matrix(rpois(n * q, 0.6), n, q), sparse = TRUE)
  e <- rx_exposure(Zd, Zn)
  expect_equal(e$d, 1)
  expect_equal(e$k, e$k_identity)
  expect_equal(e$k_identity, mean(Matrix::rowSums(Zn^2)))
  expect_equal(e$c, mean(Matrix::rowSums(Zd * Zn)))
  expect_equal(e$S, mean(Matrix::rowSums(Zn)))
  expect_equal(e$convention, "identity")
  e2 <- rx_exposure(Zd, Zn, K = diag(q))
  expect_equal(e2[c("d", "k", "c", "S", "n_eff")], e[c("d", "k", "c", "S", "n_eff")])
  expect_equal(e2$convention, "K")
  # un voisin, poids 1 : k = mean(diag(K)[voisin]), S = 1, n_eff = d / k
  A <- crossprod(matrix(rnorm(q * q), q)); K <- A / mean(diag(A)) + diag(0.1, q)
  v <- sample(q, n, TRUE)
  Z1 <- Matrix::sparseMatrix(i = seq_len(n), j = v, x = 1, dims = c(n, q))
  e3 <- rx_exposure(Zd, Z1, K = K)
  expect_equal(e3$k, mean(diag(K)[v]))
  expect_equal(e3$d, mean(diag(K)[g]))
  expect_equal(e3$S, 1)
  expect_equal(e3$n_eff, e3$d / e3$k)
  expect_equal(e3$c, mean(K[cbind(g, v)]))
})

test_that("rx_exposure : invariance d'echelle et restriction aux lignes (T4, T5)", {
  set.seed(4)
  n <- 30; q <- 6
  Zd <- Matrix::sparseMatrix(i = seq_len(n), j = sample(q, n, TRUE), x = 1, dims = c(n, q))
  Zn <- Matrix::Matrix(matrix(runif(n * q), n, q))
  A <- crossprod(matrix(rnorm(q * q), q)); K <- A / mean(diag(A))
  e <- rx_exposure(Zd, Zn, K = K); ea <- rx_exposure(Zd, 3 * Zn, K = K)
  expect_equal(ea$k, 9 * e$k); expect_equal(ea$c, 3 * e$c); expect_equal(ea$S, 3 * e$S)
  expect_equal(ea$n_eff, e$n_eff); expect_equal(ea$d, e$d)
  # empilement de deux cibles : rows = premiere moitie == sous-matrice ; sans rows,
  # k est divise par la part des lignes quand la seconde moitie est vide
  Z2 <- rbind(Zn, 0 * Zn); D2 <- rbind(Zd, Zd)
  er <- rx_exposure(D2, Z2, K = K, rows = seq_len(n))
  expect_equal(er[c("d", "k", "c", "S")], e[c("d", "k", "c", "S")])
  expect_equal(er$n_rows, n)
  et <- rx_exposure(D2, Z2, K = K)
  expect_equal(et$k, e$k / 2)
  expect_error(rx_exposure(Zd, Zn, K = K[1:3, 1:3]), "K est")
  expect_error(rx_exposure(Zd, Zn, rows = 0L), "hors de")
})

test_that("rx_exposure : forme modele, K lu dans le terme, rows = auto", {
  set.seed(5)
  n <- 24; q <- 5
  g <- rep(seq_len(q), length.out = n)
  lv <- paste0("g", seq_len(q))
  Zd <- Matrix::sparseMatrix(i = seq_len(n), j = g, x = 1, dims = c(n, q), dimnames = list(NULL, lv))
  Zn <- Matrix::Matrix(matrix(rpois(n * q, 0.5), n, q), dimnames = list(NULL, lv), sparse = TRUE)
  A <- crossprod(matrix(rnorm(q * q), q)); K <- A / mean(diag(A)) + diag(0.2, q)
  dimnames(K) <- list(lv, lv)
  # deux cibles empilees : la seconde n'a d'incidence directe que sur ses lignes
  Zd2 <- rbind(Zd, 0 * Zd); Zd2b <- rbind(0 * Zd, Zd); Zn2 <- rbind(Zn, 0 * Zn)
  tm <- rx_term("gen", list(D1 = Zd2, N1 = Zn2, D2 = Zd2b), K = K, struct = "us")
  expect_equal(tm$colnames, c("D1", "N1", "D2"))
  expect_equal(tm$levels, lv)
  mod <- rx_model(rnorm(2 * n), cbind(1, rep(0:1, each = n)), list(tm))
  e <- rx_exposure(mod, direct = "gen:D1", indirect = "gen:N1", rows = "auto")
  e_ref <- rx_exposure(Zd, Zn, K = K)
  expect_equal(e[c("d", "k", "c", "S", "n_rows")], e_ref[c("d", "k", "c", "S", "n_rows")])
  e_idx <- rx_exposure(mod, direct = "gen[1]", indirect = "gen[2]", rows = "auto")
  expect_equal(e_idx$k, e$k)
  expect_error(rx_exposure(mod, direct = "gen:D1", indirect = "gen:X"), "absente")
  expect_error(rx_exposure(mod, direct = "gen:D1", indirect = "autre:N1"), "absent du modele")
})

# ---- rx_grm ------------------------------------------------------------------
test_that("rx_grm : VanRaden 1 a ploidie 2, diagonale moyenne proche de 1, blending", {
  set.seed(6)
  M <- matrix(rbinom(50 * 400, 2, 0.35), 50, 400) / 2
  rownames(M) <- paste0("i", 1:50)
  G <- rx_grm(M, ploidy = 2)
  D <- M * 2; p <- colMeans(D) / 2; Z <- sweep(D, 2, 2 * p)
  expect_equal(unclass(G)[, ], (Z %*% t(Z) / (2 * sum(p * (1 - p))))[, ], ignore_attr = TRUE)
  expect_lt(abs(mean(diag(G)) - 1), 0.1)
  Gb <- rx_grm(M, ploidy = 2, blend = 0.02)
  expect_equal(unclass(Gb)[, ], (0.98 * unclass(G) + 0.02 * diag(50))[, ], ignore_attr = TRUE)
  expect_equal(rownames(Gb), rownames(M))
  Gc <- rx_grm(M * 2, ploidy = 2, coding = "count")
  expect_equal(unclass(Gc)[, ], unclass(G)[, ])
  expect_error(rx_grm(M * 3, ploidy = 2, coding = "count"), "sortent")
})
