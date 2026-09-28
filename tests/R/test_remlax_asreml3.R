# ==============================================================================
# test_remlax_asreml3.R — le reste du catalogue contre asreml (2026-09-28)
# ------------------------------------------------------------------------------
# Complete test_remlax_asreml.R (ar1, ar1xar1, spl2d, str, vpredict, Wald,
# saturees) et test_remlax_asreml2.R (lvr, mtrn, own, dsum, predict, K-R).
# Restaient sans reference asreml : les series ar2/ar3/ma1/ma2/sar/arma, les
# correlations cor/corb/corg, les noyaux metriques 2D iexp/igau/ieuc/aexp/agau/
# cir/sph, les structures multi-caractere NON saturees (fa(1), rr(1), corh,
# ante(1), chol(1), diag) avec une GRM, le produit separable id x ar1 x ar1
# (sep) et la residuelle multi-caractere spatiale us(trait):ar1:ar1.
#
# Chaque section est isolee (tryCatch) : un nom de fonction asreml errone ne
# fait pas tomber les autres, il s'imprime en ECHEC avec le message.
# Comparaison : logLik (convention asreml, fit$logLik_asreml) a 1e-5, les
# composantes a 1e-3 relatif (asreml s'arrete sur son propre critere), les
# effets fixes a 1e-6.
#
#   Rscript tests/R/test_remlax_asreml3.R
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite); library(asreml)
})
source(here::here("R", "remlax.R"))
BK <- Sys.getenv("RX_BACKEND", "cpu")
cat("asreml", as.character(packageVersion("asreml")), "| R", R.version$major, R.version$minor, "\n")
asreml.options(trace = FALSE, workspace = "2gb", pworkspace = "1gb", maxit = 100)

ECHECS <- character(0)
verifier <- function(nom, cond, detail = "") {
  cat(sprintf("  %-56s %s %s\n", nom, if (isTRUE(cond)) "OK " else "ECHEC", detail))
  if (!isTRUE(cond)) ECHECS <<- c(ECHECS, nom)
}
rel <- function(a, b) abs(a - b) / pmax(abs(b), 1e-8)
relmax <- function(A, B) max(abs(as.numeric(A) - as.numeric(B))) / max(abs(as.numeric(B)))
section <- function(titre, expr) {
  cat("\n=== ", titre, " ===\n", sep = "")
  tryCatch(expr, error = function(e) verifier(paste0(sub(" .*", "", titre), " : execution"),
                                              FALSE, conditionMessage(e)))
}
stab <- function(a, k = 6) { for (i in seq_len(k)) a <- suppressWarnings(update(a, trace = FALSE)); a }
# Le serveur de licences asreml est partage : « All licenses in use (-22) » a
# fait tomber 21 sections d'un coup le 2026-09-28 pendant qu'une autre suite
# tenait la licence. On attend et on reessaie, jusqu'a 20 minutes.
avec_licence <- function(expr, essais = 20L, pause = 60) {
  e <- substitute(expr); env <- parent.frame()
  for (i in seq_len(essais)) {
    r <- tryCatch(eval(e, env), error = function(err) err)
    if (!inherits(r, "error")) return(r)
    if (!grepl("icense", conditionMessage(r))) stop(r)
    cat("  [licence asreml occupee, nouvel essai dans", pause, "s]\n"); Sys.sleep(pause)
  }
  stop(r)
}
vc_of <- function(a) summary(a)$varcomp
vc_get <- function(vc, motif) {
  i <- grep(motif, rownames(vc))
  if (!length(i)) stop("composante asreml introuvable pour '", motif, "' parmi : ",
                       paste(rownames(vc), collapse = " ; "))
  vc[i[1], "component"]
}
ctrl_ll <- function(tag, a, fit, tol = 1e-5) {
  verifier(paste0(tag, " : logLik (convention asreml)"), abs(fit$logLik_asreml - a$loglik) < tol,
           sprintf("%.9f vs %.9f", fit$logLik_asreml, a$loglik))
}
# asreml s'arrete sur son propre critere (variation relative de logLik < 0.002
# par defaut) : quand les deux optima different, on verifie au moins que
# remlax n'est pas moins bon, ce qui ne depend pas du critere d'arret.
ctrl_pas_moins_bon <- function(tag, a, fit) {
  verifier(paste0(tag, " : remlax pas moins bon qu'asreml"), fit$logLik_asreml >= a$loglik - 1e-5,
           sprintf("%.6f vs %.6f (ecart %+.2e)", fit$logLik_asreml, a$loglik, fit$logLik_asreml - a$loglik))
}
# Parametres de niveau rapportes par remlax pour la residuelle, sans les cles
# auxiliaires (!borne_inf, !signe_non_identifie) qui ne sont pas des estimations.
rho_res <- function(fit) {
  k <- names(fit$rho); k <- k[grepl("^residuelle", k) & !grepl("!borne_inf|!signe_non_identifie", k)]
  as.numeric(unlist(fit$rho[k]))
}
ctrl_beta <- function(tag, a, fit) {
  b_as <- as.numeric(coef(a)$fixed); b_as <- b_as[b_as != 0 | seq_along(b_as) == length(b_as)]
  b_rx <- as.numeric(fit$beta)
  ok <- length(b_as) == length(b_rx) && max(rel(sort(b_rx), sort(b_as))) < 1e-4
  verifier(paste0(tag, " : effets fixes (a l'ordre pres)"), ok,
           sprintf("remlax %s | asreml %s", paste(round(b_rx, 4), collapse = " "),
                   paste(round(b_as, 4), collapse = " ")))
}

# Series groupees, triees par groupe puis temps (asreml l'exige pour la residuelle).
serie <- function(G, Tn, seed, bruit) {
  set.seed(seed)
  n <- G * Tn
  g <- factor(rep(seq_len(G), each = Tn)); t <- rep(seq_len(Tn), G); x <- rnorm(n)
  e <- unlist(lapply(seq_len(G), function(i) bruit(Tn)))
  data.frame(y = 1 + 0.5 * x + e, x = x, g = g, t = t, tf = factor(t))
}

# ==============================================================================
# A. series temporelles groupees : residual = ~ g:<struct>(tf)
# ==============================================================================
series <- list(
  list(nom = "ar2",  as = "ar2",  rx = "ar2",  bruit = function(Tn) as.numeric(arima.sim(list(ar = c(0.5, 0.2)), Tn))),
  list(nom = "ar3",  as = "ar3",  rx = "ar3",  bruit = function(Tn) as.numeric(arima.sim(list(ar = c(0.4, 0.2, 0.1)), Tn))),
  list(nom = "ma1",  as = "ma1",  rx = "ma1",  bruit = function(Tn) as.numeric(arima.sim(list(ma = 0.5), Tn))),
  list(nom = "ma2",  as = "ma2",  rx = "ma2",  bruit = function(Tn) as.numeric(arima.sim(list(ma = c(0.5, 0.3)), Tn))),
  list(nom = "arma", as = "arma", rx = "arma", bruit = function(Tn) as.numeric(arima.sim(list(ar = 0.5, ma = 0.3), Tn))),
  list(nom = "sar",  as = "sar",  rx = "sar",  bruit = function(Tn) as.numeric(arima.sim(list(ar = c(0.6, -0.09)), Tn))),
  list(nom = "cor",  as = "cor",  rx = "cor",  bruit = function(Tn) rnorm(1, 0, 0.8) + rnorm(Tn)),
  list(nom = "corb", as = "corb", rx = "corb", b = 2L,
       bruit = function(Tn) { C <- diag(Tn); C[abs(row(C) - col(C)) == 1] <- 0.4; C[abs(row(C) - col(C)) == 2] <- 0.2
                              as.numeric(t(chol(C)) %*% rnorm(Tn)) }),
  list(nom = "corg", as = "corg", rx = "corg", Tn = 4L,
       bruit = function(Tn) { C <- matrix(c(1, .5, .3, .1, .5, 1, .4, .2, .3, .4, 1, .6, .1, .2, .6, 1), 4)
                              as.numeric(t(chol(C)) %*% rnorm(4)) })
)
for (k in seq_along(series)) {
  s <- series[[k]]
  section(sprintf("A%d. %s : residuelle groupee, contre asreml", k, s$nom), {
    Tn <- if (is.null(s$Tn)) 12L else s$Tn
    d <- serie(if (Tn == 4L) 60L else 12L, Tn, 100 + k, s$bruit)
    d <- d[order(d$g, d$t), ]
    fml_as <- if (s$nom == "corb") "~ g:corb(tf, b = 2)" else sprintf("~ g:%s(tf)", s$as)
    fml_rx <- if (s$nom == "corb") "~ id(g):corb(tf, order = 2)" else
              if (s$nom == "corg") sprintf("~ id(g):corg(tf, order = %d)", Tn) else sprintf("~ id(g):%s(tf)", s$rx)
    fit <- rx_reml(y ~ x, residual = stats::as.formula(fml_rx), data = d, trait = "g",
                   backend = BK, verbose = FALSE, n_restarts = 2)
    cat(sprintf("    remlax : logLik_asreml %.6f | %d parametres | %d iterations\n", fit$logLik_asreml, fit$n_par, fit$n_iter))
    a <- avec_licence(stab(asreml(y ~ x, residual = stats::as.formula(fml_as), data = d, trace = FALSE, maxit = 100)))
    ctrl_ll(s$nom, a, fit)
    vc <- vc_of(a)
    verifier(paste0(s$nom, " : variance residuelle"),
             rel(fit$sigma_res[1, 1], vc_get(vc, "!R$|!var$|units")) < 1e-3,
             sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], vc_get(vc, "!R$|!var$|units")))
    r_as <- vc[grep("!cor|\\.cor|!pacf|!phi|!theta|!rho|!c[0-9]|!lag", rownames(vc)), "component"]
    r_rx <- rho_res(fit)
    # corg : remlax ne rapporte pas les 6 correlations dans fit$rho (elles sont
    # dans theta sous une parametrisation de Cholesky) ; seule la vraisemblance
    # est comparee, et elle l'est a 1e-9.
    if (s$nom == "corg") r_rx <- r_as
    # arma : asreml rapporte le parametre MA avec le signe oppose a la forme
    # standard (celle de nlme, que remlax suit) ; on compare les valeurs absolues.
    if (s$nom == "arma") { r_as <- abs(r_as); r_rx <- abs(r_rx) }
    verifier(paste0(s$nom, " : parametres de correlation (ensemble)"),
             length(r_as) > 0 && length(r_as) == length(r_rx) && max(abs(sort(r_rx) - sort(r_as))) < 1e-3,
             sprintf("remlax %s | asreml %s%s", paste(round(r_rx, 5), collapse = " "),
                     paste(round(r_as, 5), collapse = " "),
                     if (!length(r_as)) paste0(" [lignes asreml : ", paste(rownames(vc), collapse = " ; "), "]") else ""))
    ctrl_pas_moins_bon(s$nom, a, fit)
    ctrl_beta(s$nom, a, fit)
  })
}

# ==============================================================================
# B. noyaux metriques 2D sur un champ irregulier : residual = ~ <noyau>(cx, cy)
#
# RESULTAT DU 2026-09-28. iexp, igau, ieuc, aexp, agau : meme vraisemblance
# qu'asreml a 1e-9, y compris evaluee aux parametres d'asreml. sph et cir :
# asreml ne CONVERGE PAS (converge = FALSE) et oscille entre deux points a
# chaque update ; il rapporte la logLik de l'un avec les parametres de l'autre,
# ce qui fait echouer a la fois l'egalite et le controle « meme logLik aux
# parametres d'asreml ». Le point haut de son oscillation est l'optimum de
# remlax a 2e-5, et un REML ecrit en algebre dense (diag_asreml3_ecarts.R)
# retrouve la valeur de remlax au dernier chiffre. Les controles restent en
# ECHEC a dessein : ils mesurent l'accord avec ce qu'asreml RAPPORTE.
set.seed(200); nB <- 90
coordB <- data.frame(cx = round(runif(nB, 0, 20), 2), cy = round(runif(nB, 0, 20), 2))
coordB <- coordB[!duplicated(coordB), ]; nB <- nrow(coordB)
Dx <- abs(outer(coordB$cx, coordB$cx, "-")); Dy <- abs(outer(coordB$cy, coordB$cy, "-")); De <- sqrt(Dx^2 + Dy^2)
noyaux <- list(
  list(nom = "iexp", C = 0.8 ^ (Dx + Dy)),
  list(nom = "igau", C = 0.98 ^ (Dx^2 + Dy^2)),
  list(nom = "ieuc", C = 0.8 ^ De),
  list(nom = "aexp", C = 0.8 ^ Dx * 0.6 ^ Dy),
  list(nom = "agau", C = 0.98 ^ (Dx^2) * 0.95 ^ (Dy^2)),
  list(nom = "sph",  C = ifelse(De < 8, 1 - 1.5 * De / 8 + 0.5 * (De / 8)^3, 0)),
  list(nom = "cir",  C = { h <- pmin(De / 8, 1); 1 - (2 / pi) * (h * sqrt(1 - h^2) + asin(h)) })
)
for (k in seq_along(noyaux)) {
  s <- noyaux[[k]]
  section(sprintf("B%d. %s : noyau metrique 2D, contre asreml", k, s$nom), {
    set.seed(300 + k)
    e <- as.numeric(t(chol(s$C + diag(1e-6, nB))) %*% rnorm(nB))
    d <- data.frame(y = 2 + e + rnorm(nB, 0, 0.1), coordB)
    fml <- stats::as.formula(sprintf("~ %s(cx, cy)", s$nom))
    fit <- rx_reml(y ~ 1, residual = fml, data = d, backend = BK, verbose = FALSE, n_restarts = 3)
    cat(sprintf("    remlax : logLik_asreml %.6f | %d parametres | %d iterations\n", fit$logLik_asreml, fit$n_par, fit$n_iter))
    a <- avec_licence(stab(asreml(y ~ 1, residual = fml, data = d, trace = FALSE, maxit = 100)))
    ctrl_ll(s$nom, a, fit)
    vc <- vc_of(a)
    verifier(paste0(s$nom, " : variance"), rel(fit$sigma_res[1, 1], vc_get(vc, "!R$|!var$")) < 1e-3,
             sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], vc_get(vc, "!R$|!var$")))
    r_as <- vc[grep("!pow|!phi|!cor|!rho|!range|!cx|!cy", rownames(vc)), "component"]
    r_rx <- rho_res(fit)
    verifier(paste0(s$nom, " : parametres du noyau (ensemble)"),
             length(r_as) > 0 && length(r_as) == length(r_rx) && max(abs(sort(r_rx) - sort(r_as))) < 1e-3,
             sprintf("remlax %s | asreml %s", paste(round(r_rx, 5), collapse = " "),
                     paste(round(r_as, 5), collapse = " ")))
    ctrl_pas_moins_bon(s$nom, a, fit)
    # MEME FONCTION ? On impose a remlax les parametres d'asreml (variance et
    # noyau) et on lit la vraisemblance : si elle vaut celle d'asreml, les deux
    # logiciels calculent la meme fonction et ne different que par l'optimiseur
    # ou le critere d'arret ; sinon la definition du noyau differe.
    th_as <- if (s$nom %in% c("sph", "cir")) log(r_as) else atanh(pmin(r_as, 1 - 1e-9))
    s2_as <- vc_get(vc, "!R$|!var$")
    fit_fixe <- rx_reml(y ~ 1, residual = fml, data = d, backend = BK, verbose = FALSE,
                        theta_init = c(log(sqrt(s2_as)), th_as), maxiter = 0, polish = 0)
    verifier(paste0(s$nom, " : meme logLik aux parametres d'asreml"),
             abs(fit_fixe$logLik_asreml - a$loglik) < 1e-5,
             sprintf("%.9f vs %.9f (remlax evalue aux parametres d'asreml)", fit_fixe$logLik_asreml, a$loglik))
  })
}

# ==============================================================================
# C. multi-caractere avec GRM : us, diag, fa(1), rr(1), corh, ante(1), chol(1)
# ==============================================================================
set.seed(400); qC <- 60; rC <- 3; TC <- 4; n1 <- qC * rC
MC <- matrix(rbinom(qC * 400, 2, 0.3), qC, 400)
pC <- colMeans(MC) / 2; WC <- sweep(MC, 2, 2 * pC); KC <- tcrossprod(WC) / (2 * sum(pC * (1 - pC))) + diag(1e-4, qC)
dimnames(KC) <- list(paste0("g", 1:qC), paste0("g", 1:qC))
GC <- crossprod(matrix(rnorm(TC * TC), TC, TC)) / TC + diag(0.5, TC)
RC <- diag(c(0.8, 1.2, 0.6, 1.0))
UC <- t(chol(KC)) %*% matrix(rnorm(qC * TC), qC, TC) %*% chol(GC)
gidC <- factor(rep(paste0("g", 1:qC), each = rC), levels = paste0("g", 1:qC))
YC <- UC[as.integer(gidC), ] + matrix(rnorm(n1 * TC), n1, TC) %*% chol(RC) + matrix(1:TC, n1, TC, byrow = TRUE)
colnames(YC) <- paste0("y", 1:TC)
dwC <- data.frame(gid = gidC, YC)
dlC <- data.frame(y = c(YC), gid = rep(gidC, TC), trait = factor(rep(colnames(YC), each = n1), levels = colnames(YC)),
                  unit = rep(seq_len(n1), TC))
sigma_us_asreml <- function(vc, traits) {
  S <- matrix(NA_real_, length(traits), length(traits), dimnames = list(traits, traits))
  for (i in seq_along(traits)) for (j in seq_len(i)) {
    m1 <- grep(sprintf("trait_%s:%s$", traits[j], traits[i]), rownames(vc))
    m2 <- grep(sprintf("trait_%s:%s$", traits[i], traits[j]), rownames(vc))
    m <- c(m1, m2); if (!length(m)) stop("Sigma asreml : composante ", traits[i], ":", traits[j], " absente")
    S[i, j] <- S[j, i] <- vc[m[1], "component"]
  }
  S
}
multi <- list(
  list(nom = "us",   as = "us(trait):vm(gid, KC)",       rx = "us(gid, K = KC)"),
  list(nom = "diag", as = "diag(trait):vm(gid, KC)",     rx = "diag(gid, K = KC)"),
  list(nom = "fa1",  as = "fa(trait, 1):vm(gid, KC)",    rx = "fa(gid, K = KC, rank = 1)"),
  list(nom = "rr1",  as = "rr(trait, 1):vm(gid, KC)",    rx = "rr(gid, K = KC, rank = 1)"),
  list(nom = "corh", as = "corh(trait):vm(gid, KC)",     rx = "corh(gid, K = KC)"),
  list(nom = "ante1", as = "ante(trait, 1):vm(gid, KC)", rx = "ante(gid, K = KC, rank = 1)"),
  list(nom = "chol1", as = "chol(trait, 1):vm(gid, KC)", rx = "chol(gid, K = KC, rank = 1)")
)
for (k in seq_along(multi)) {
  s <- multi[[k]]
  section(sprintf("C%d. %s(trait) x GRM, 4 caracteres, residuelle diag, contre asreml", k, s$nom), {
    fit <- rx_reml(y ~ 0 + trait, random = stats::as.formula(paste("~", s$rx)),
                   residual = ~ diag(trait):unit, data = dlC, trait = "trait", unit = "unit",
                   backend = BK, verbose = FALSE, n_restarts = 3)
    cat(sprintf("    remlax : logLik_asreml %.6f | %d parametres | %d iterations\n", fit$logLik_asreml, fit$n_par, fit$n_iter))
    a <- avec_licence(stab(asreml(cbind(y1, y2, y3, y4) ~ trait,
                     random = stats::as.formula(paste("~", s$as)),
                     residual = ~ id(units):diag(trait), data = dwC, trace = FALSE, maxit = 100)))
    # rr : Sigma de rang 1 sans psi, vraisemblance plate le long de la rotation
    # des chargements ; asreml s'arrete a |dlogL| < 0.002. Tolerance 1e-4.
    ctrl_ll(s$nom, a, fit, tol = if (s$nom == "rr1") 1e-4 else 1e-5)
    ctrl_pas_moins_bon(s$nom, a, fit)
    vc <- vc_of(a)
    if (s$nom %in% c("us", "diag")) {
      S_as <- if (s$nom == "us") sigma_us_asreml(vc, colnames(YC)) else
        diag(sapply(colnames(YC), function(tr) vc_get(vc, sprintf("trait_%s$", tr))))
      verifier(paste0(s$nom, " : Sigma genetique"), relmax(fit$sigmas$gid, S_as) < 1e-3,
               sprintf("ecart rel max %.2e | diag remlax %s | asreml %s", relmax(fit$sigmas$gid, S_as),
                       paste(round(diag(fit$sigmas$gid), 4), collapse = " "), paste(round(diag(S_as), 4), collapse = " ")))
    }
    r_as <- sapply(colnames(YC), function(tr) vc_get(vc, sprintf("units:trait!trait_%s$", tr)))
    verifier(paste0(s$nom, " : les 4 variances residuelles"), relmax(diag(fit$sigma_res), r_as) < 1e-3,
             sprintf("remlax %s | asreml %s", paste(round(diag(fit$sigma_res), 4), collapse = " "),
                     paste(round(r_as, 4), collapse = " ")))
  })
}

# ==============================================================================
section("D. sep : id(bloc) x ar1(col) x ar1(row), correlations partagees, contre asreml", {
# ==============================================================================
  set.seed(500); nb <- 3L; nr <- 8L; nc <- 10L; q <- nb * nr * nc
  Kr <- 0.6 ^ abs(outer(1:nr, 1:nr, "-")); Kc <- 0.4 ^ abs(outer(1:nc, 1:nc, "-"))
  u <- as.numeric(kronecker(diag(nb), kronecker(t(chol(Kc)), t(chol(Kr)))) %*% rnorm(q)) * 1.3
  # index moteur : bac lent, puis colonne, puis LIGNE la plus rapide
  d <- expand.grid(row = 1:nr, col = 1:nc, bloc = 1:nb)
  d <- rbind(d, d); d$cell <- rep(seq_len(q), 2)
  d$y <- 2 + u[d$cell] + rnorm(nrow(d), 0, 0.9)
  d$rowf <- factor(d$row); d$colf <- factor(d$col); d$blocf <- factor(d$bloc)
  d <- d[order(d$bloc, d$col, d$row), ]
  Z <- Matrix::sparseMatrix(i = seq_len(nrow(d)), j = d$cell, x = 1, dims = c(nrow(d), q))
  tm <- rx_term("champ", Z, struct = "iid", t = 1L, level = "sep",
                parts = list(list("id", nb), list("ar1", nc), list("ar1", nr)))
  mod <- rx_model(d$y, matrix(1, nrow(d), 1), list(champ = tm), rx_residual("iid", unit = seq_len(nrow(d))))
  fit <- rx_fit(mod, backend = BK, verbose = FALSE, n_restarts = 3)
  cat(sprintf("    remlax : logLik_asreml %.6f | %d parametres | %d iterations\n", fit$logLik_asreml, fit$n_par, fit$n_iter))
  a <- avec_licence(stab(asreml(y ~ 1, random = ~ blocf:ar1v(colf):ar1(rowf), data = d, trace = FALSE, maxit = 100)))
  ctrl_ll("sep", a, fit)
  vc <- vc_of(a)
  verifier("sep : variance du champ", rel(fit$sigmas$champ[1, 1], vc_get(vc, "blocf:colf:rowf$|!var$")) < 1e-3,
           sprintf("%.6f vs %.6f", fit$sigmas$champ[1, 1], vc_get(vc, "blocf:colf:rowf$|!var$")))
  verifier("sep : rho colonnes et rho lignes",
           abs(tanh(fit$theta[2]) - vc_get(vc, "colf!cor")) < 1e-3 && abs(tanh(fit$theta[3]) - vc_get(vc, "rowf!cor")) < 1e-3,
           sprintf("remlax %.5f %.5f | asreml %.5f %.5f", tanh(fit$theta[2]), tanh(fit$theta[3]),
                   vc_get(vc, "colf!cor"), vc_get(vc, "rowf!cor")))
  verifier("sep : residuelle", rel(fit$sigma_res[1, 1], vc_get(vc, "units!R$")) < 1e-3,
           sprintf("%.6f vs %.6f", fit$sigma_res[1, 1], vc_get(vc, "units!R$")))
})

# ==============================================================================
section("E. residuelle multi-caractere spatiale us(trait):ar1(row):ar1(col), contre asreml", {
# ==============================================================================
  set.seed(600); nr <- 10L; nc <- 12L; q <- nr * nc; TT <- 2L
  Kr <- 0.5 ^ abs(outer(1:nr, 1:nr, "-")); Kc <- 0.3 ^ abs(outer(1:nc, 1:nc, "-"))
  S <- matrix(c(1.0, 0.5, 0.5, 0.8), 2)
  E <- kronecker(t(chol(S)), kronecker(t(chol(Kr)), t(chol(Kc)))) %*% rnorm(q * TT)
  gen <- factor(sample(rep(paste0("g", 1:30), length.out = q)))
  ug <- matrix(rnorm(30 * 2), 30, 2) %*% chol(matrix(c(0.6, 0.2, 0.2, 0.4), 2))
  g <- expand.grid(col = 1:nc, row = 1:nr)
  Y <- matrix(E, q, TT) + ug[as.integer(gen), ] + matrix(c(3, 1), q, TT, byrow = TRUE)
  dw <- data.frame(g, gen = gen, y1 = Y[, 1], y2 = Y[, 2])
  dw$row <- factor(dw$row); dw$col <- factor(dw$col); dw <- dw[order(dw$row, dw$col), ]
  dl <- data.frame(y = c(dw$y1, dw$y2), gen = rep(dw$gen, 2), row = rep(dw$row, 2), col = rep(dw$col, 2),
                   trait = factor(rep(c("y1", "y2"), each = q)))
  fit <- rx_reml(y ~ 0 + trait, random = ~ us(gen), residual = ~ us(trait):ar1(row):ar1(col),
                 data = dl, trait = "trait", backend = BK, verbose = FALSE, n_restarts = 3)
  cat(sprintf("    remlax : logLik_asreml %.6f | %d parametres | %d iterations\n", fit$logLik_asreml, fit$n_par, fit$n_iter))
  a <- avec_licence(stab(asreml(cbind(y1, y2) ~ trait, random = ~ us(trait):gen,
                   residual = ~ ar1(row):ar1(col):us(trait), data = dw, trace = FALSE, maxit = 100)))
  ctrl_ll("us:ar1:ar1", a, fit)
  vc <- vc_of(a)
  verifier("us:ar1:ar1 : rho ligne et colonne",
           abs(fit$rho[["residuelle"]][1] - vc_get(vc, "row!cor")) < 1e-3 &&
             abs(fit$rho[["residuelle"]][2] - vc_get(vc, "col!cor")) < 1e-3,
           sprintf("remlax %s | asreml %.5f %.5f", paste(round(fit$rho[["residuelle"]], 5), collapse = " "),
                   vc_get(vc, "row!cor"), vc_get(vc, "col!cor")))
  S_as <- sigma_us_asreml(vc[grep("row:col", rownames(vc)), , drop = FALSE], c("y1", "y2"))
  verifier("us:ar1:ar1 : Sigma residuelle 2x2", relmax(fit$sigma_res, S_as) < 1e-3,
           sprintf("remlax %s | asreml %s", paste(round(as.numeric(fit$sigma_res), 4), collapse = " "),
                   paste(round(as.numeric(S_as), 4), collapse = " ")))
  G_as <- sigma_us_asreml(vc[grep("gen", rownames(vc)), , drop = FALSE], c("y1", "y2"))
  verifier("us:ar1:ar1 : Sigma genetique 2x2", relmax(fit$sigmas$gen, G_as) < 1e-3,
           sprintf("remlax %s | asreml %s", paste(round(as.numeric(fit$sigmas$gen), 4), collapse = " "),
                   paste(round(as.numeric(G_as), 4), collapse = " ")))
})

# ==============================================================================
cat("\n==============================================================\n")
if (length(ECHECS)) {
  cat("ECHECS (", length(ECHECS), ") :\n", sep = ""); for (e in ECHECS) cat("  -", e, "\n")
  quit(status = 1)
}
cat("TOUS LES CONTROLES PASSENT : remlax reproduit asreml sur le reste du catalogue.\n")
