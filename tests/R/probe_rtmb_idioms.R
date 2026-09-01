# Quel idiome RTMB permet de construire une matrice triangulaire differentiable,
# et de resoudre avec elle ? On teste, on ne devine pas.
library(RTMB)
cat("RTMB", as.character(packageVersion("RTMB")), "\n")
cat("--- fonctions exportees pertinentes ---\n")
ex <- ls("package:RTMB")
cat(paste(grep("^(AD|advector|matrix|solve|diag|Matrix|sparse)", ex, value = TRUE, ignore.case = TRUE),
          collapse = " "), "\n\n")

essai <- function(nom, f) {
  r <- tryCatch({
    obj <- MakeADFun(function(p) f(p$th), list(th = c(0.1, 0.2, 0.3)), silent = TRUE)
    v <- obj$fn(c(0.1, 0.2, 0.3)); g <- obj$gr(c(0.1, 0.2, 0.3))
    sprintf("OK    valeur %.6f, gradient [%s]", v, paste(sprintf("%.4f", g), collapse = ", "))
  }, error = function(e) paste("ECHEC", conditionMessage(e)))
  cat(sprintf("  %-46s %s\n", nom, r))
}

cat("--- construire L triangulaire 2x2 et rendre sum(L %*% t(L)) ---\n")

essai("matrix(0) puis assignation", function(th) {
  L <- matrix(0, 2, 2); L[1,1] <- exp(th[1]); L[2,1] <- th[2]; L[2,2] <- exp(th[3])
  sum(L %*% t(L))
})
essai("AD(matrix(0)) puis assignation", function(th) {
  L <- AD(matrix(0, 2, 2)); L[1,1] <- exp(th[1]); L[2,1] <- th[2]; L[2,2] <- exp(th[3])
  sum(L %*% t(L))
})
essai("advector puis dim()", function(th) {
  v <- advector(rep(0, 4)); v[1] <- exp(th[1]); v[2] <- th[2]; v[4] <- exp(th[3])
  dim(v) <- c(2L, 2L)
  sum(v %*% t(v))
})
essai("matrix() directement sur un advector", function(th) {
  L <- matrix(c(exp(th[1]), th[2], 0, exp(th[3])), 2, 2)
  sum(L %*% t(L))
})
essai("cbind de colonnes advector", function(th) {
  L <- cbind(c(exp(th[1]), th[2]), c(0 * th[1], exp(th[3])))
  sum(L %*% t(L))
})

cat("\n--- resoudre : solve(L, M) avec L differentiable ---\n")
essai("solve(L, matrice numerique)", function(th) {
  L <- matrix(c(exp(th[1]), th[2], 0, exp(th[3])), 2, 2)
  sum(solve(L, matrix(c(1, 2, 3, 4), 2, 2)))
})
essai("solve(L) puis produit", function(th) {
  L <- matrix(c(exp(th[1]), th[2], 0, exp(th[3])), 2, 2)
  sum(solve(L) %*% matrix(c(1, 2, 3, 4), 2, 2))
})
essai("backsolve", function(th) {
  L <- matrix(c(exp(th[1]), th[2], 0, exp(th[3])), 2, 2)
  sum(backsolve(L, c(1, 2), upper.tri = FALSE))
})

cat("\n--- produit par une matrice creuse CONSTANTE ---\n")
essai("dgCMatrix %*% vecteur advector", function(th) {
  S <- Matrix::sparseMatrix(i = c(1,2,2), j = c(1,1,2), x = c(1,-0.5,1), dims = c(2,2))
  u <- c(th[1], th[2])
  sum(u * as.vector(S %*% u))
})
essai("as.matrix(dgCMatrix) %*% advector", function(th) {
  S <- as.matrix(Matrix::sparseMatrix(i = c(1,2,2), j = c(1,1,2), x = c(1,-0.5,1), dims = c(2,2)))
  u <- c(th[1], th[2])
  sum(u * as.vector(S %*% u))
})

cat("\n--- log-determinant d'une matrice differentiable ---\n")
essai("2*sum(log(diag(L)))", function(th) {
  L <- matrix(c(exp(th[1]), th[2], 0, exp(th[3])), 2, 2)
  2 * sum(log(diag(L)))
})
