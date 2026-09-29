# Grammaire de formule, SANS Python : chaque structure de Sigma et chaque
# structure entre niveaux est construite par .rx_parse_random / rx_model, et
# son nombre de parametres est compare a la formule de l'annexe C (recalculee
# ici, pas relue dans rx_n_params). rx_n_theta doit rendre le meme compte que
# rx_model$n_par sur TOUS les modeles, pas seulement dans le perimetre creux.
#
# Dispositif : 90 lignes, 10 genotypes x 9 colonnes, trois caracteres en
# format long ; partie fixe ~ trait.
.d_gram <- function() {
  set.seed(11)
  d <- data.frame(gid = factor(rep(1:10, 9)), col = factor(rep(1:9, each = 10)),
                  trait = factor(rep(c("A", "B", "C"), 30)), y = rnorm(90))
  d$coln <- as.integer(d$col); d$rown <- as.integer(d$gid)
  d$pos <- d$coln * 1.5
  d
}
.modele_gram <- function(random, d = .d_gram(), residual = "units", trait = d$trait) {
  tl <- .rx_parse_random(random, d, trait = trait)
  rs <- .rx_parse_residual(residual, d, trait = if (is.null(trait)) NULL else "trait",
                           unit = NULL)
  rx_model(d$y, model.matrix(y ~ trait, d), tl, rs)
}
.nload <- function(t, r) sum(pmin(seq_len(t), r))

test_that("structures de Sigma : compte de parametres par formule", {
  d <- .d_gram()
  K <- crossprod(matrix(rnorm(100), 10)) / 10 + diag(10)
  dimnames(K) <- list(levels(d$gid), levels(d$gid))
  Zw <- matrix(rnorm(900), 90, 10, dimnames = list(NULL, levels(d$gid)))
  w <- 3L
  attendu <- c(
    "~ gid" = 1, "~ iid(gid)" = 1, "~ vm(gid, K = K)" = 1, "~ vm(gid, K)" = 1,
    "~ diag(gid)" = w, "~ us(gid)" = w * (w + 1) / 2, "~ us(gid, K = K)" = w * (w + 1) / 2,
    "~ fa(gid, rank = 1)" = .nload(w, 1) + w, "~ fa(gid, rank = 2)" = .nload(w, 2) + w,
    "~ rr(gid, rank = 1)" = .nload(w, 1), "~ rr(gid, rank = 2)" = .nload(w, 2),
    "~ chol(gid, rank = 1)" = 2 * (w - 0.5), "~ chol(gid, rank = 2)" = 3 * (w - 1),
    "~ ante(gid, rank = 1)" = 2 * (w - 0.5), "~ ante(gid, rank = 2)" = 3 * (w - 1),
    "~ corh(gid)" = w + 1, "~ mm(Zw)" = 1,
    "~ str(~ gid + mm(Zw, name = \"w\"), struct = \"us\")" = 3,
    "~ str(~ gid + mm(Zw, name = \"w\"), struct = \"diag\")" = 2,
    "~ str(~ gid + mm(Zw, name = \"w\"), struct = \"fa\", rank = 1)" = 4,
    "~ gid + iid(col)" = 2)
  for (k in seq_along(attendu)) {
    f <- stats::as.formula(names(attendu)[k])
    environment(f) <- environment()
    m <- .modele_gram(f, d)
    expect_equal(m$n_par - 1L, as.integer(attendu[[k]]), info = names(attendu)[k])
    expect_equal(rx_n_theta(m), m$n_par, info = names(attendu)[k])
  }
  # K reordonnee sur les niveaux, niveau absent refuse
  m <- .modele_gram(~ vm(gid, K = K[10:1, 10:1]), d)
  expect_equal(m$terms[[1]]$LK %*% t(m$terms[[1]]$LK), unname(K), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_error(.modele_gram(~ vm(gid, K = K[-1, -1]), d), "absent")
  # bending d'une K semi-definie
  Ks <- K; Ks[1, ] <- Ks[, 1] <- Ks[2, ]; Ks[1, 1] <- Ks[2, 2]
  expect_message(rx_term("g", d$gid, K = Ks), "bending")
})

test_that("structures entre niveaux : compte de parametres par formule", {
  d <- .d_gram()
  q_col <- 9L
  attendu <- c(
    "~ cor(col)" = 1, "~ ar1(col)" = 1, "~ ar2(col)" = 2, "~ ar3(col)" = 3, "~ sar(col)" = 1,
    "~ ma1(col)" = 1, "~ ma2(col)" = 2, "~ arma(col)" = 2, "~ corb(col, order = 2)" = 2,
    "~ corg(col)" = q_col * (q_col - 1) / 2, "~ ar1(rown, coln)" = 2,
    "~ exp(col)" = 1, "~ gau(col)" = 1, "~ lvr(col)" = 1, "~ exp(col, coord = 1:9 * 2)" = 1,
    "~ iexp(coln, rown)" = 1, "~ igau(coln, rown)" = 1, "~ ieuc(coln, rown)" = 1,
    "~ sph(coln, rown)" = 1, "~ cir(coln, rown)" = 1, "~ aexp(coln, rown)" = 2,
    "~ agau(coln, rown)" = 2, "~ mtrn(coln, rown)" = 1,
    "~ mtrn(coln, rown, nu = 1, delta = 2)" = 3, "~ mtrn(coln, rown, nu = \"1 F\")" = 1,
    "~ mtrn(coln, rown, phi = \"2 F\", nu = 1.5, alpha = 0.3, lambda = 1)" = 2,
    "~ own(col, expr = \"exp(-lag*exp(p1))\", n_par = 1)" = 1,
    "~ own(col, expr = \"p1*exp(-lag*exp(p2))\", n_par = 2, normalise = FALSE)" = 2,
    # struct = heterogene entre caracteres : Sigma diag (3) ou us (6) + rho
    "~ ar1(col, struct = \"diag\")" = 3 + 1, "~ ar1(col, struct = \"us\")" = 6 + 1,
    "~ ar1(rown, coln, struct = \"diag\")" = 3 + 2,
    "~ iexp(coln, rown, struct = \"diag\")" = 3 + 1)
  for (k in seq_along(attendu)) {
    f <- stats::as.formula(names(attendu)[k])
    m <- .modele_gram(f, d)
    tm <- m$terms[[1]]
    # + 1 pour la variance de Sigma (iid) sauf struct heterogene
    sig <- if (grepl("struct = \"diag\"", names(attendu)[k])) 0L else
           if (grepl("struct = \"us\"", names(attendu)[k])) 0L else 1L
    expect_equal(tm$n_par, as.integer(attendu[[k]] + sig), info = names(attendu)[k])
    expect_equal(rx_n_theta(m), m$n_par, info = names(attendu)[k])
    expect_equal(m$n_par, tm$n_par + 1L, info = names(attendu)[k])
  }
  # ar1(row, col) : niveaux ligne:colonne, ligne lente
  m <- .modele_gram(~ ar1(rown, coln), d)
  expect_equal(m$terms[[1]]$level, "ar1ar1")
  expect_equal(m$terms[[1]]$dims, c(10L, 9L))
  expect_equal(m$terms[[1]]$levels[1:2], c("1:1", "1:2"))
  # metrique 2D a deux arguments : coordonnees de la cellule
  m <- .modele_gram(~ iexp(coln, rown), d)
  expect_equal(ncol(m$terms[[1]]$coord), 2L)
  expect_equal(nrow(m$terms[[1]]$coord), m$terms[[1]]$q)
  # mtrn : options serialisees
  m <- .modele_gram(~ mtrn(coln, rown, nu = 1, delta = 2), d)
  o <- m$terms[[1]]$opts
  expect_equal(unname(o[c("est_phi", "est_nu", "est_delta", "est_alpha")]), c(1, 1, 1, 0))
  expect_equal(unname(o[["init_nu"]]), 1)
  expect_error(.modele_gram(~ mtrn(coln, rown, lambda = 3), d), "lambda")
  # struct heterogene sans trait : refus explicite
  expect_error(.rx_parse_random(~ ar1(col, struct = "diag"), d, trait = NULL), "trait")
})

test_that("grammaire de la residuelle : formes, comptes et refus", {
  d <- .d_gram()
  d$unite <- factor(rep(1:30, each = 3))
  d$rowf <- factor(d$rown); d$colf <- factor(d$coln)
  cnt <- function(res, trait = "trait", unit = NULL, dd = d) {
    r <- .rx_parse_residual(res, dd, trait = trait, unit = unit)
    Xd <- if (is.null(trait)) matrix(1, nrow(dd), 1) else model.matrix(y ~ trait, dd)
    m <- rx_model(dd$y, Xd, list(rx_term("gid", dd$gid)), r)
    c(m$n_par - 1L, rx_n_theta(m) - 1L)
  }
  expect_equal(cnt("units"), c(1, 1))
  expect_equal(cnt(~ units), c(1, 1))
  expect_equal(cnt(~ id(units)), c(1, 1))
  expect_equal(cnt(~ us(trait):units, unit = "unite"), c(6, 6))
  expect_equal(cnt(~ diag(trait):units, unit = "unite"), c(3, 3))
  expect_equal(cnt(~ fa(trait, rank = 1):units, unit = "unite"), c(.nload(3, 1) + 3, .nload(3, 1) + 3))
  expect_equal(cnt("us", unit = "unite"), c(6, 6))
  expect_equal(cnt(~ exp(pos), trait = NULL, dd = d[d$trait == "A", ]), c(2, 2))
  expect_equal(cnt(~ ar1(colf), trait = NULL, dd = d[d$trait == "A", ]), c(2, 2))
  expect_equal(cnt(~ dsum(~ units | trait)), c(3, 3))
  expect_equal(cnt(~ dsum(~ ar1(colf) | trait)), c(6, 6))
  r <- .rx_parse_residual(~ dsum(~ ar1(colf) + units | trait, levels = list(c("A", "B"), "C")),
                          d, trait = "trait")
  expect_length(r$sections, 2L)
  expect_equal(r$sections[[1]]$level, "ar1")
  expect_equal(r$sections[[2]]$level, "id")
  expect_equal(sort(unlist(lapply(r$sections, `[[`, "rows"))), 1:90)
  # champ separable, avec et sans structure de caractere ; grille 10 x 9
  dA <- d[d$trait == "A", ]
  expect_equal(cnt(~ ar1(rowf):ar1(colf), trait = NULL, dd = dA), c(3, 3))
  r <- .rx_parse_residual(~ ar1(rowf):ar1(colf), dA, trait = NULL, unit = NULL)
  expect_equal(r$level, "ar1ar1"); expect_equal(r$dims, c(10L, 9L))
  dl <- rbind(dA, dA, dA); dl$trait <- factor(rep(c("A", "B", "C"), each = 30))
  expect_equal(cnt(~ us(trait):ar1(rowf):ar1(colf), dd = dl), c(6 + 2, 6 + 2))
  expect_equal(cnt(~ diag(trait):ar1(rowf):ar1(colf), dd = dl), c(3 + 2, 3 + 2))
  # refus
  # meme cellule repetee SANS caractere : refus ; avec us(trait) : accepte (plus haut)
  expect_error(.rx_parse_residual(~ ar1(rowf):ar1(colf), dl, trait = NULL), "en double")
  expect_error(.rx_parse_residual(~ iexp(coln, rown), dl, trait = NULL), "en double")
  expect_error(.rx_parse_residual(~ ar1(colf):ma1(rowf), dA, trait = NULL), "deux structures")
  expect_error(.rx_parse_residual(~ toto(colf), dA, trait = NULL), "non reconnu")
  dA$lettre <- factor(letters[dA$coln])
  expect_error(.rx_parse_residual(~ exp(lettre), dA, trait = NULL), "numeriques")
  expect_error(.rx_parse_residual(~ ar1(colf):ar1(zzz), dA, trait = NULL), "introuvable")
  expect_error(.rx_parse_residual(~ dsum(~ units), d, trait = "trait"), "dsum")
  expect_error(.rx_parse_residual(~ dsum(~ units | zzz), d, trait = "trait"), "absente")
  expect_error(.rx_parse_residual(~ dsum(~ ar1(colf) + units | trait), d, trait = "trait"),
               "levels")
  expect_error(.rx_parse_residual(~ dsum(~ ar1(colf) + units | trait, levels = list("A", "B")),
                                  d, trait = "trait"), "partitionnent")
  expect_error(.rx_parse_residual(3, d), "formule")
  expect_error(rx_model(d$y, matrix(1, 90, 1), list(rx_term("gid", d$gid)),
                        rx_residual("us", trait = NULL)), "caractere multiple")
})

test_that("rx_term : trois formes de Z, et chaque refus", {
  d <- .d_gram()
  n <- nrow(d)
  # facteur, liste de matrices, matrice empilee : meme incidence
  t1 <- rx_term("g", d$gid)
  Zm <- as.matrix(t1$Zl[[1]])
  t2 <- rx_term("g", list(Zm, Zm), struct = "us", levels = levels(d$gid))
  t3 <- rx_term("g", cbind(Zm, Zm), t = 2L, struct = "us", levels = levels(d$gid))
  expect_equal(t1$t, 1L); expect_equal(t2$t, 2L); expect_equal(t3$t, 2L)
  expect_equal(as.matrix(t2$Zl[[2]]), as.matrix(t3$Zl[[2]]), ignore_attr = TRUE)
  expect_equal(t2$n_par, 3L)
  expect_message(rx_term("g", list(Zm, Zm), struct = "iid"), "une seule variance")
  expect_error(rx_term("g", list(Zm, Zm[, 1:5])), "colonnes")
  expect_error(rx_term("g", cbind(Zm, Zm)), "fournir `t`")
  expect_error(rx_term("g", cbind(Zm, Zm[, 1:3]), t = 2L), "multiple")
  expect_error(rx_term("g", d$gid, struct = "fa", rank = 0L), "rank")
  expect_error(rx_term("g", list(Zm, Zm), struct = "chol", rank = 2L), "rank")
  K <- diag(10); dimnames(K) <- list(levels(d$gid), levels(d$gid))
  expect_error(rx_term("g", d$gid, K = K, level = "ar1"), "exclusives")
  expect_error(rx_term("g", d$gid, K = K[1:5, 1:5]), "absent")
  expect_error(rx_term("g", d$gid, K = unname(K)[1:5, 1:5]), "niveaux")
  expect_error(rx_term("g", d$gid, level = "exp"), "coord")
  expect_error(rx_term("g", d$gid, level = "iexp", coord = 1:10), "DEUX")
  expect_error(rx_term("g", d$gid, level = "exp", coord = 1:4), "lignes")
  expect_error(rx_term("g", d$gid, level = "own"), "expr")
  expect_error(rx_term("g", d$gid, level = "ar1ar1"), "dims")
  expect_error(rx_term("g", d$gid, level = "ar1ar1", dims = c(3, 3)), "niveaux")
  expect_error(rx_term("g", d$gid, K = K, Kinv = Matrix::Diagonal(10)), "pas les deux")
  expect_error(rx_term("g", d$gid, Kinv = Matrix::Diagonal(4)), "niveaux")
  expect_error(rx_term("g", d$gid, level = "sep"), "parts")
  expect_error(rx_term("g", d$gid, level = "sep", parts = list(list("corb", 2), list("ar1", 5))),
               "sep")
  tsep <- rx_term("g", d$gid, level = "sep", parts = list(list("id", 2), list("ar1", 5)))
  expect_equal(tsep$n_par, 2L)
  expect_error(rx_term("g", d$gid, struct = "zzz"))
  expect_error(rx_term("g", d$gid, level = "zzz"))
  # rx_model : incidence de mauvaise hauteur, objets non rx_term
  expect_error(rx_model(d$y[1:50], matrix(1, 50, 1), list(t1)), "lignes")
  expect_error(rx_model(d$y, matrix(1, n, 1), list(1)), "rx_term")
  expect_error(rx_model(d$y, matrix(1, 40, 1), list(t1)), "lignes")
  # termes non reconnus, colonne absente
  expect_error(.rx_parse_random(~ toto(gid), d), "non reconnu")
  expect_error(.rx_parse_random(~ iid(zzz), d), "absente")
  expect_error(.rx_parse_random(~ us(gid), d, trait = NULL), "multi-caractere")
  expect_error(.rx_parse_random(~ rr(gid, rank = 1), d, trait = NULL), "multi-caractere")
  expect_error(.rx_parse_random(~ str(~ gid, struct = "us"), d), "deux termes")
  expect_error(.rx_parse_random(~ str(~ gid + col, struct = "us"), d), "niveaux")
  expect_error(.rx_parse_random(~ mm(matrix(1, 90, 3), struct = "us"), d, trait = NULL),
               "trait")
  # sans terme aleatoire : residuelle structuree acceptee, iid refusee
  dA <- d[d$trait == "A", ]; dA$rowf <- factor(dA$rown); dA$colf <- factor(dA$coln)
  m0 <- rx_model(dA$y, matrix(1, 30, 1), list(),
                 .rx_parse_residual(~ ar1(rowf):ar1(colf), dA, trait = NULL))
  expect_equal(m0$n_par, 3L); expect_equal(rx_n_theta(m0), 3L)
  expect_output(print(m0), "total : 3 parametres")
})

test_that("rx_spl2d : partie nulle, cinq termes PS-ANOVA, tailles", {
  set.seed(5)
  x <- rep(1:10, 9); y <- rep(1:9, each = 10)
  sp <- rx_spl2d(x, y, nseg = c(6, 6))
  expect_equal(ncol(sp$X), 3L)
  expect_equal(colnames(sp$X), paste0("spl_lin", 1:3))
  expect_equal(vapply(sp$terms, `[[`, "", "name"),
               paste0("spl_", c("fx", "fy", "fx_y", "x_fy", "fx_fy")))
  expect_equal(vapply(sp$terms, `[[`, 1L, "q"), c(7L, 7L, 7L, 7L, 49L))
  expect_true(all(abs(colMeans(sp$X)) < 1e-10))                # centree
  X <- cbind(1, sp$X)
  expect_equal(qr(X)$rank, 4L)                                  # independante de l'intercept
  m <- rx_model(rnorm(90), X, c(list(rx_term("gid", factor(rep(1:10, 9)))), sp$terms))
  expect_equal(m$n_par, 7L); expect_equal(rx_n_theta(m), 7L)
  sp2 <- rx_spl2d(x, y, nseg = c(4, 5), prefix = "s")
  expect_equal(vapply(sp2$terms, `[[`, 1L, "q"), c(5L, 6L, 5L, 6L, 5L * 6L))
  expect_equal(sp2$terms[[1]]$name, "s_fx")
  # bases emboitees pour f(x):f(y) : nseg / nest.div segments
  sp3 <- rx_spl2d(x, y, nseg = c(4, 6), nest.div = c(2, 3))
  expect_equal(vapply(sp3$terms, `[[`, 1L, "q"), c(5L, 7L, 5L, 7L, 3L * 3L))
  expect_error(rx_spl2d(x, y, nseg = c(5, 6), nest.div = 2), "multiple")
  expect_error(rx_spl2d(x, y, pord = 3), "ordre 2")
  expect_error(rx_spl2d(x, y[1:10]), "longueurs")
})

test_that("rx_export ecrit chaque champ d'un terme structure et le relit", {
  d <- .d_gram()
  coord <- cbind(rnorm(10), rnorm(10))
  tm <- rx_term("g", d$gid, level = "mtrn", coord = coord,
                opts = c(est_phi = 1, est_nu = 1, delta = 2, lambda = 2))
  ts <- rx_term("s", d$gid, level = "sep", parts = list(list("id", 2), list("ar1", 5)))
  to <- rx_term("o", d$gid, level = "own", expr = "exp(-lag*p1)", opts = c(n_par = 1))
  dA <- d[d$trait == "A", ]
  m <- rx_model(dA$y, matrix(1, 30, 1), list(rx_term("g", dA$gid)),
                .rx_parse_residual(~ dsum(~ units | col), dA, trait = NULL))
  dir <- tempfile("rx_exp_"); on.exit(unlink(dir, recursive = TRUE))
  rx_export(rx_model(d$y, matrix(1, 90, 1), list(tm, ts, to)), dir)
  man <- jsonlite::fromJSON(file.path(dir, "manifest.json"))
  noms <- man$name
  for (f in c("term_g_coord", "term_g_lvloptk", "term_g_lvloptv", "term_s_lvlpartk",
              "term_s_lvlpartq", "term_o_lvlexpr", "term_g_lvl", "res_nsec"))
    expect_true(f %in% noms, info = f)
  expect_equal(readLines(file.path(dir, "term_o_lvlexpr.txt")), "exp(-lag*p1)")
  expect_equal(readLines(file.path(dir, "term_s_lvlpartk.txt")), c("id", "ar1"))
  co <- readBin(file.path(dir, "term_g_coord.bin"), "double", n = 20, size = 8)
  expect_equal(matrix(co, 10, 2), coord)
  # dsum : sections avec leurs lignes 0-based
  dir2 <- tempfile("rx_exp2_"); on.exit(unlink(dir2, recursive = TRUE), add = TRUE)
  rx_export(m, dir2)
  man2 <- jsonlite::fromJSON(file.path(dir2, "manifest.json"))
  expect_true(all(c("res_secnames", "res_s0_rows", "res_s8_rows") %in% man2$name))
  expect_equal(readBin(file.path(dir2, "res_nsec.bin"), "integer", n = 1, size = 4), 9L)
})
