# ==============================================================================
# diag_asreml3_ecarts.R — les trois modeles ou asreml et remlax ne rendent pas
#                          la meme logLik (ma2, sph, cir) : cause, pas verdict
# ------------------------------------------------------------------------------
# Memes donnees que test_remlax_asreml3.R (memes graines). Pour chaque modele :
#   1. asreml : message de convergence, codes de borne, trajectoire de logLik,
#      toutes les composantes ; puis 20 update() pour voir s'il oscille ;
#   2. un ARBITRE ecrit en R dense (log|V| + log|X'V^-1 X| + y'Py), evalue aux
#      parametres d'asreml ET a ceux de remlax, sous chaque convention
#      plausible (signe des theta MA ; distance euclidienne ou city-block) :
#      celui qui retrouve la valeur d'asreml dit ce qu'asreml calcule ;
#   3. asreml refit avec ses parametres FIXES aux valeurs de remlax (R.param,
#      con = "F"), pour lire la logLik d'asreml au point de remlax ;
#   4. controle de la constante : X (codage, colonnes, rang) des deux cotes,
#      nedf d'asreml, n - p de remlax, log|X'V^-1 X| a la main.
#   Rscript tests/R/diag_asreml3_ecarts.R
# ==============================================================================
suppressPackageStartupMessages({ library(here); library(Matrix); library(jsonlite); library(asreml) })
source(here::here("R", "remlax.R"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
asreml.options(workspace = "2gb", pworkspace = "1gb")
cat("asreml", as.character(packageVersion("asreml")), "\n")

reml_R <- function(y, X, V) {
  # Convention asreml : -1/2 (log|V| + log|X'V^-1X| + y'Py), sans (n-p)/2 log 2pi
  cV <- tryCatch(chol(V), error = function(e) NULL)
  if (is.null(cV)) return(c(logLik = NA_real_, ldV = NA_real_, ldXVX = NA_real_, yPy = NA_real_))   # V non PD : le noyau n'est pas valide sur cette metrique
  Vi <- chol2inv(cV)
  XVX <- t(X) %*% Vi %*% X
  P <- Vi - Vi %*% X %*% solve(XVX, t(X) %*% Vi)
  ldV <- 2 * sum(log(diag(cV))); ldX <- determinant(XVX, log = TRUE)$modulus[1]; yPy <- as.numeric(t(y) %*% P %*% y)
  c(logLik = -0.5 * (ldV + ldX + yPy), ldV = ldV, ldXVX = ldX, yPy = yPy)
}
imprimer_asreml <- function(a, tag) {
  cat("\n[", tag, "] asreml : converge =", a$converge, "| loglik =", sprintf("%.9f", a$loglik),
      "| nedf =", a$nedf, "| noeff =", paste(a$noeff, collapse = ","), "\n")
  vc <- summary(a)$varcomp; print(vc)
  cat("  vparameters :", paste(names(a$vparameters), sprintf("%.6f", a$vparameters), sep = "=", collapse = " | "), "\n")
  cat("  vparameters.con :", paste(a$vparameters.con, collapse = " "), "\n")
  if (!is.null(a$monitor)) { cat("  monitor (logLik par iteration) :\n"); print(round(a$monitor[1:min(3, nrow(a$monitor)), , drop = FALSE], 6)) }
  cat("  coef fixes :", paste(round(as.numeric(coef(a)$fixed), 6), collapse = " "), "\n")
  invisible(vc)
}
suivre_updates <- function(a, k = 20) {
  ll <- numeric(k)
  for (i in seq_len(k)) { a <- suppressWarnings(update(a, trace = FALSE)); ll[i] <- a$loglik }
  cat("  logLik apres chaque update (", k, ") :", paste(sprintf("%.6f", ll), collapse = " "), "\n")
  a
}
fixer_R_param <- function(a, valeurs) {
  # valeurs : liste chemin EXACT ("/section/composante") -> vecteur de valeurs ;
  # chaque composante nommee est fixee (con = "F") a ces valeurs. Les chemins
  # sont ceux qu'imprime str(a$R.param) : "/g:tf/variance", "/g:tf/tf",
  # "/sph(cx, cy)/variance", "/sph(cx, cy)/sph(cx, cy)".
  rp <- a$R.param
  poser <- function(x, chemin = "") {
    if (is.list(x) && !is.null(x$initial) && !is.null(x$con)) {
      if (chemin %in% names(valeurs)) {
        v <- valeurs[[chemin]]
        if (length(v) != length(x$initial)) stop("longueur ", length(v), " pour ", chemin, " (attendu ", length(x$initial), ")")
        x$initial[] <- v; x$con[] <- rep("F", length(x$con)); cat("    fixe", chemin, "=", paste(v, collapse = " "), "\n")
      }
      return(x)
    }
    if (is.list(x)) for (nm in names(x)) x[[nm]] <- poser(x[[nm]], paste(chemin, nm, sep = "/"))
    x
  }
  poser(rp)
}

# ------------------------------------------------------------------------------
# 1. ma2 : series groupees (section A4 de la suite)
# ------------------------------------------------------------------------------
cat("\n=== ma2 ===\n")
serie <- function(G, Tn, seed, bruit) {
  set.seed(seed); n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) bruit(Tn)))
  data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
}
d <- serie(12L, 12L, 104, function(Tn) as.numeric(arima.sim(list(ma = c(0.5, 0.3)), Tn)))
d <- d[order(d$g, d$t), ]
fit <- rx_reml(y ~ x, residual = ~ id(g):ma2(tf), data = d, trait = "g", backend = BK, verbose = FALSE, n_restarts = 2)
cat("remlax : logLik_asreml", sprintf("%.9f", fit$logLik_asreml), "| s2", fit$sigma_res[1, 1], "| theta MA", paste(round(unlist(fit$rho), 6), collapse = " "),
    "| n_obs", fit$n_obs, "| p", ncol(fit$model$X), "| beta", paste(round(fit$beta, 6), collapse = " "), "| decrement", fit$newton_decrement, "\n")
a <- suppressWarnings(asreml(y ~ x, residual = ~ g:ma2(tf), data = d, trace = FALSE, maxit = 100))
vc <- imprimer_asreml(a, "ma2, premier appel")
a <- suivre_updates(a); vc <- imprimer_asreml(a, "ma2, apres 20 updates")
# arbitre R : correlation MA(2) sous les deux conventions de signe
acf_ma2 <- function(th, Tn, signe = +1) {
  t1 <- signe * th[1]; t2 <- signe * th[2]; den <- 1 + t1^2 + t2^2
  r <- c(1, (t1 + t1 * t2) / den, t2 / den, rep(0, Tn - 3)); toeplitz(r)
}
X <- model.matrix(~ x, d); G <- nlevels(d$g); Tn <- nlevels(d$tf)
s2_as <- vc[grep("!R$", rownames(vc)), "component"]; th_as <- vc[grep("!theta|!cor|!lag|!ma", rownames(vc)), "component"]
cat("asreml lu : s2", s2_as, " theta", paste(th_as, collapse = " "), "\n")
for (signe in c(+1, -1)) {
  V_as <- s2_as * kronecker(diag(G), acf_ma2(th_as, Tn, signe))
  V_rx <- fit$sigma_res[1, 1] * kronecker(diag(G), acf_ma2(unlist(fit$rho)[1:2], Tn, signe))
  r1 <- reml_R(d$y, X, V_as); r2 <- reml_R(d$y, X, V_rx)
  cat(sprintf("  arbitre R (signe %+d) : aux params asreml %.9f [asreml dit %.9f] | aux params remlax %.9f [remlax dit %.9f]\n",
              signe, r1["logLik"], a$loglik, r2["logLik"], fit$logLik_asreml))
  cat(sprintf("     pieces aux params asreml : log|V| %.6f  log|X'V-1X| %.6f  y'Py %.6f\n", r1["ldV"], r1["ldXVX"], r1["yPy"]))
}
# asreml fixe aux parametres de remlax
cat("  structure de R.param :\n"); str(a$R.param, max.level = 4, give.attr = FALSE)
rp <- tryCatch(fixer_R_param(a, list("/g:tf/variance" = fit$sigma_res[1, 1], "/g:tf/tf" = as.numeric(unlist(fit$rho)[1:2]))),
               error = function(e) { cat("R.param :", conditionMessage(e), "\n"); NULL })
if (!is.null(rp)) {
  a2 <- tryCatch(suppressWarnings(asreml(y ~ x, residual = ~ g:ma2(tf), data = d, R.param = rp, trace = FALSE, maxit = 1)),
                 error = function(e) { cat("  asreml fixe :", conditionMessage(e), "\n"); NULL })
  if (!is.null(a2)) { cat(sprintf("  asreml aux params de remlax (fixes) : %.9f  [remlax dit %.9f ; arbitre signe -1 dit %.9f]\n", a2$loglik, fit$logLik_asreml,
                                  reml_R(d$y, X, fit$sigma_res[1, 1] * kronecker(diag(G), acf_ma2(unlist(fit$rho)[1:2], Tn, -1)))["logLik"])); print(summary(a2)$varcomp) }
}
# ------------------------------------------------------------------------------
# 2 et 3. sph et cir : champ 2D irregulier (sections B6, B7)
# ------------------------------------------------------------------------------
set.seed(200); nB <- 90
coordB <- data.frame(cx = round(runif(nB, 0, 20), 2), cy = round(runif(nB, 0, 20), 2))
coordB <- coordB[!duplicated(coordB), ]; nB <- nrow(coordB)
Dx <- abs(outer(coordB$cx, coordB$cx, "-")); Dy <- abs(outer(coordB$cy, coordB$cy, "-")); De <- sqrt(Dx^2 + Dy^2); Dcb <- Dx + Dy
sph_f <- function(D, r) { h <- pmin(D / r, 1); 1 - 1.5 * h + 0.5 * h^3 }
cir_f <- function(D, r) { h <- pmin(D / r, 1); 1 - (2 / pi) * (h * sqrt(1 - h^2) + asin(h)) }
noyaux <- list(list(nom = "sph", k = 6L, f = sph_f, C = sph_f(De, 8)),
               list(nom = "cir", k = 7L, f = cir_f, C = cir_f(De, 8)))
for (s in noyaux) {
  cat("\n===", s$nom, "===\n")
  set.seed(300 + s$k)
  e <- as.numeric(t(chol(s$C + diag(1e-6, nB))) %*% rnorm(nB))
  d <- data.frame(y = 2 + e + rnorm(nB, 0, 0.1), coordB)
  fml <- stats::as.formula(sprintf("~ %s(cx, cy)", s$nom))
  fit <- rx_reml(y ~ 1, residual = fml, data = d, backend = BK, verbose = FALSE, n_restarts = 3)
  rg_rx <- fit$rho[["residuelle!portee"]]
  cat("remlax : logLik_asreml", sprintf("%.9f", fit$logLik_asreml), "| s2", fit$sigma_res[1, 1], "| portee", rg_rx, "| n_iter", fit$n_iter, "| decrement", fit$newton_decrement, "\n")
  a <- suppressWarnings(asreml(y ~ 1, residual = fml, data = d, trace = FALSE, maxit = 100))
  vc <- imprimer_asreml(a, paste(s$nom, "premier appel"))
  a <- suivre_updates(a); vc <- imprimer_asreml(a, paste(s$nom, "apres 20 updates"))
  s2_as <- vc[grep("!R$", rownames(vc)), "component"][1]
  rg_as <- vc[grep("!phi|!range|!pow|!cor", rownames(vc)), "component"][1]
  cat("asreml lu : s2", s2_as, " portee", rg_as, "\n")
  X <- matrix(1, nB, 1)
  # l'ordre des unites : remlax et l'arbitre suivent l'ordre des lignes de d (coordB) ; asreml trie
  # ses unites par ses propres cles, mais la vraisemblance ne depend pas de l'ordre.
  for (dist in list(c("euclidienne", "De"), c("city-block", "Dcb"))) {
    D <- get(dist[2])
    for (pars in list(c("asreml", s2_as, rg_as), c("remlax", fit$sigma_res[1, 1], rg_rx))) {
      V <- as.numeric(pars[2]) * s$f(D, as.numeric(pars[3]))
      r <- reml_R(d$y, X, V + diag(1e-10, nB))
      cat(sprintf("  arbitre R, distance %-11s aux params %-6s : %.9f  (log|V| %.5f, log|X'V-1X| %.5f, y'Py %.5f)\n",
                  dist[1], pars[1], r["logLik"], r["ldV"], r["ldXVX"], r["yPy"]))
    }
  }
  cat(sprintf("  rappel : asreml dit %.9f a ses params ; remlax dit %.9f a ses params\n", a$loglik, fit$logLik_asreml))
  # portee a l'echelle d'asreml : et si sa portee etait definie autrement (phi = portee/3, ou sqrt) ?
  for (fac in c(1, 3, 1 / 3, sqrt(3))) {
    V <- s2_as * s$f(De, rg_as * fac); r <- reml_R(d$y, X, V + diag(1e-10, nB))
    cat(sprintf("  arbitre R euclidien, portee asreml x %.3f : %.9f\n", fac, r["logLik"]))
  }
  # remlax fixe aux params d'asreml, et asreml fixe aux params de remlax
  f2 <- rx_reml(y ~ 1, residual = fml, data = d, backend = BK, verbose = FALSE,
                theta_init = c(log(sqrt(s2_as)), log(rg_as)), maxiter = 0, polish = 0)
  cat(sprintf("  remlax aux params d'asreml : %.9f\n", f2$logLik_asreml))
  cat("  structure de R.param :\n"); str(a$R.param, max.level = 4, give.attr = FALSE)
  sec <- names(a$R.param)[1]
  rp <- tryCatch(fixer_R_param(a, setNames(list(fit$sigma_res[1, 1], rg_rx), c(paste0("/", sec, "/variance"), paste0("/", sec, "/", sec)))),
                 error = function(e) { cat("R.param :", conditionMessage(e), "\n"); NULL })
  if (!is.null(rp)) {
    a2 <- tryCatch(suppressWarnings(asreml(y ~ 1, residual = fml, data = d, R.param = rp, trace = FALSE, maxit = 1)),
                   error = function(e) { cat("  asreml fixe :", conditionMessage(e), "\n"); NULL })
    if (!is.null(a2)) { cat(sprintf("  asreml aux params de remlax (fixes) : %.9f\n", a2$loglik)); print(summary(a2)$varcomp) }
  }
}
cat("\nfin du diagnostic\n")
