# Un produit separable a trois facteurs traverse-t-il jusqu'au solveur ?
# POURQUOI CE TEST. `ar1ar1` ne couvre que DEUX facteurs. Un champ spatial
# replique independamment par bloc s'ecrit id (x) ar1 (x) ar1 : le facteur
# identite n'apporte aucun parametre, donc les deux correlations sont PARTAGEES
# entre les repliques. C'est ce que le moteur d'IGE_analysis estime, et c'est ce
# que l'interface R ne savait pas exprimer.
suppressMessages(source(file.path(Sys.getenv("RX_DIR", "."), "R", "remlax.R")))
set.seed(4)
nb <- 3L; nr <- 4L; nc <- 5L; q <- nb * nr * nc
n <- 2L * q
# une observation par cellule, en double, avec l'index du moteur :
# lin = bac*nr*nc + col*nr + row  -> la LIGNE varie le plus vite
cell <- rep(seq_len(q), 2L)
Z <- Matrix::sparseMatrix(i = seq_len(n), j = cell, x = 1, dims = c(n, q))
y <- rnorm(n); X <- matrix(1, n, 1)
tm <- rx_term("champ", Z, struct = "iid", t = 1L, level = "sep",
              parts = list(list("id", nb), list("ar1", nc), list("ar1", nr)))
cat("n_par du terme :", tm$n_par, " (attendu 3 : une variance + deux correlations)\n")
m <- rx_model(y, X, terms = list(champ = tm),
              residual = rx_residual("iid", unit = seq_len(n)))
f <- rx_fit(m, hessian = FALSE, blups = FALSE, maxiter = 60L, verbose = FALSE)
cat("logLik :", format(f$logLik, digits = 8), "\n")
cat("theta  :", paste(round(f$theta, 4), collapse = " "), "\n")
cat("correlations rapportees :", paste(names(f$rho), signif(unlist(f$rho), 4),
                                         collapse = " | "), "\n")
# LES DEUX CORRELATIONS EN CLAIR : theta[2] est celle des COLONNES et theta[3]
# celle des LIGNES, dans cet ordre, parce que l'index du moteur fait varier la
# ligne le plus vite. On les rapporte a l'echelle utilisee, tanh(theta).
cat(sprintf("rho colonnes = %.4f | rho lignes = %.4f\n",
            tanh(f$theta[2]), tanh(f$theta[3])))
# theta porte 4 entrees : 3 pour le terme (une variance + deux correlations) et
# une pour la residuelle. Mon assertion initiale comptait mal.
ok <- length(f$theta) == 4L && tm$n_par == 3L && is.finite(f$logLik) &&
      # le cote R aplatit les cles en "<terme>!<famille>_<indice>_<param>" :
      # l'indice du facteur est dans le nom, ce qui dit LAQUELLE des deux
      # correlations est celle des lignes.
      length(grep("^champ!ar1_", names(f$rho))) == 2L
cat(if (ok) "\nok : le produit a trois facteurs traverse jusqu'au solveur\n"
    else "\nECHEC\n")
if (!ok) quit(status = 1L)
