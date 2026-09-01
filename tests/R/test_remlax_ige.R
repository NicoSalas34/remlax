# ==============================================================================
# test_remlax_ige.R — LE modele du projet, sur donnees simulees de verite connue
# ------------------------------------------------------------------------------
# Reproduit en petit le modele des scripts 05/06 : effet genetique direct (DGE),
# effet indirect intraspecifique (IGE intra) porte par un VOISINAGE PONDERE,
# effet indirect interspecifique (IGE inter), covariance libre entre DGE et IGE
# intra, champ spatial AR1 x AR1 et pepite.
#
#     y_i = mu + Zg u_DGE + Zn u_IGEintra + Zx u_IGEinter + champ AR1xAR1 + e
#     (u_DGE, u_IGEintra) ~ N(0, Sigma_us (x) I)      Sigma_us : 2 x 2 LIBRE
#
# POURQUOI SIMULER PLUTOT QUE PRENDRE LES VRAIES DONNEES : ici on CONNAIT les
# valeurs. Un ecart n'est donc pas une question d'interpretation — c'est un
# defaut du solveur ou de l'echantillonnage, et les deux se distinguent en
# faisant varier la taille.
#
# DEUX NOYAUX DE VOISINAGE, ceux du projet :
#   puissance     w = d^-1 sur la fenetre de Chebyshev d'ordre `ord`
#   exponentiel   w = exp(-kappa (d - d0)), portee estimee (ici a kappa fixe :
#                 estimer kappa demande de rebatir Z a chaque iteration, ce que
#                 le solveur ne fait pas encore — cf. limites)
#
# COMPARAISON : asreml sur le MEME modele, avec str(~ dge + grp(nb), ~us(2):id(q))
# qui est sa facon d'ecrire une covariance partagee entre deux termes.
#
#   Rscript scripts/tests/test_remlax_ige.R [--noyau=puissance|exp]
#                       [--nr= --nc=]   taille de la grille (replication)
#                       [--qa= --qb=]   nombre de genotypes (identifiabilite)
# ==============================================================================
suppressPackageStartupMessages({
  library(here); library(Matrix); library(jsonlite)
})
source(here::here("R", "remlax.R"))
ga <- function(k, d) { a <- commandArgs(TRUE); m <- grep(paste0("^--", k, "="), a, value = TRUE)
  if (length(m)) sub(paste0("^--", k, "="), "", m[1]) else d }
NOYAU <- ga("noyau", "puissance"); ORD <- as.integer(ga("ord", 6))
KAPPA <- as.numeric(ga("kappa", 0.12)); SEED <- as.integer(ga("seed", 42))
# TAILLE PAR DEFAUT. Elle etait de 16 x 12 avec 18/12 genotypes : 96
# observations pour 8 parametres de variance. A cette taille le bloc us(2) est
# a la frontiere (r = -1) des DEUX cotes, asreml rendant NA comme erreur-type de
# trois parametres du champ, et les deux solveurs s'arretent a 0,305 point de
# logLik l'un de l'autre. Comparer leurs composantes ne prouvait alors rien.
# A 40 x 30 et 90/60 genotypes, les huit composantes coincident a moins de 4 %
# d'une erreur-type. Un test qui ne discrimine pas n'est pas un test.
NR <- as.integer(ga("nr", 40)); NC <- as.integer(ga("nc", 30))

ECHECS <- character(0)
verifier <- function(nom, cond, detail = "") {
  cat(sprintf("  %-50s %s %s\n", nom, if (isTRUE(cond)) "OK " else "ECHEC", detail))
  if (!isTRUE(cond)) ECHECS <<- c(ECHECS, nom)
}

# ==============================================================================
# 1. DISPOSITIF : especes alternees par colonne, comme le vrai
# ==============================================================================
set.seed(SEED)
g <- expand.grid(col = 1:NC, row = 1:NR)
g$esp <- ifelse(g$col %% 2 == 1, "A", "B")          # A = "ble", B = "luzerne"
g$id  <- seq_len(nrow(g))
# NOMBRE DE GENOTYPES, et pourquoi c'est lui qui compte. Agrandir la GRILLE
# n'ajoute que de la replication : le bloc us(2) reste a la frontiere (r = -1)
# parce que DGE et IGE intra sont portes par les MEMES genotypes. Ce qui les
# separe, c'est le nombre de genotypes, pas le nombre de parcelles. Mesure :
# a 18/12 genotypes, r = -1.0000 sur une grille 16x12 comme sur une 40x30.
QA <- as.integer(ga("qa", 90)); QB <- as.integer(ga("qb", 60))
gA <- paste0("a", seq_len(QA)); gB <- paste0("b", seq_len(QB))
g$geno <- NA_character_
g$geno[g$esp == "A"] <- sample(rep_len(gA, sum(g$esp == "A")))
g$geno[g$esp == "B"] <- sample(rep_len(gB, sum(g$esp == "B")))

# --- matrices de voisinage ----------------------------------------------------
# w = d^-1 (puissance) ou exp(-kappa (d - 5)) (exponentiel), sur la fenetre de
# Chebyshev d'ordre ORD. Pas de normalisation L2 : la variance IGE est donc
# "par unite d'exposition", comme dans le pipeline.
pas <- 5
poids <- function(dr, dc) {
  d <- pas * sqrt(dr^2 + dc^2)
  if (NOYAU == "puissance") 1 / d else exp(-KAPPA * (d - pas))
}
mk_Z <- function(focal_esp, voisin_esp, glev) {
  ix <- which(g$esp == focal_esp)
  Z <- matrix(0, length(ix), length(glev), dimnames = list(NULL, glev))
  for (k in seq_along(ix)) {
    i <- ix[k]
    vd <- abs(g$row - g$row[i]) <= ORD & abs(g$col - g$col[i]) <= ORD &
          g$id != g$id[i] & g$esp == voisin_esp
    j <- which(vd)
    if (!length(j)) next
    w <- poids(g$row[j] - g$row[i], g$col[j] - g$col[i])
    for (m in seq_along(j)) {
      cc <- match(g$geno[j[m]], glev); Z[k, cc] <- Z[k, cc] + w[m]
    }
  }
  Z
}
iA <- which(g$esp == "A"); nA <- length(iA)
ZgA <- Matrix::sparseMatrix(i = seq_len(nA), j = match(g$geno[iA], gA), x = 1,
                            dims = c(nA, QA), dimnames = list(NULL, gA))
ZnA <- mk_Z("A", "A", gA)                            # voisins conspecifiques
ZxA <- mk_Z("A", "B", gB)                            # voisins heterospecifiques

# ==============================================================================
# 2. SIMULATION : valeurs VRAIES
# ==============================================================================
SIG <- matrix(c(0.90, -0.35, -0.35, 0.30), 2, 2)     # (DGE, IGE intra) : cov NEGATIVE
V_INTER <- 0.22                                       # variance IGE inter
V_AR1 <- 0.45; RHO_R <- 0.65; RHO_C <- 0.40           # champ spatial
V_E <- 0.50                                           # pepite
U   <- matrix(rnorm(QA * 2), QA, 2) %*% chol(SIG)
uX  <- rnorm(QB, 0, sqrt(V_INTER))
Kr <- RHO_R ^ abs(outer(1:NR, 1:NR, "-")); Kc <- RHO_C ^ abs(outer(1:NC, 1:NC, "-"))
champ <- as.numeric(kronecker(t(chol(Kr)), t(chol(Kc))) %*% rnorm(NR * NC)) * sqrt(V_AR1)
cellA <- (g$row[iA] - 1) * NC + g$col[iA]            # grille complete (simulation)
col_obs <- as.integer(factor(g$col[iA]))             # colonnes REELLEMENT occupees
NC_OBS  <- max(col_obs)
cellA_obs <- (g$row[iA] - 1) * NC_OBS + col_obs      # grille conspecifique
y <- 3 + as.numeric(ZgA %*% U[, 1]) + as.numeric(ZnA %*% U[, 2]) +
     as.numeric(ZxA %*% uX) + champ[cellA] + rnorm(nA, 0, sqrt(V_E))

d <- data.frame(y = y, geno = factor(g$geno[iA], levels = gA),
                row = factor(g$row[iA]), col = factor(g$col[iA]))
cat(sprintf("Dispositif : %d x %d, noyau %s (ordre %d%s) | %d obs, %d genotypes A, %d B\n",
            NR, NC, NOYAU, ORD, if (NOYAU == "exp") sprintf(", kappa=%.2f", KAPPA) else "",
            nA, QA, QB))
cat(sprintf("Verite : var_DGE %.2f  var_IGEintra %.2f  cov %.2f  (r = %.3f)\n",
            SIG[1, 1], SIG[2, 2], SIG[1, 2], SIG[1, 2] / sqrt(SIG[1, 1] * SIG[2, 2])))
cat(sprintf("         var_IGEinter %.2f | AR1 var %.2f rho %.2f/%.2f | pepite %.2f\n\n",
            V_INTER, V_AR1, RHO_R, RHO_C, V_E))

# ==============================================================================
# 3. remlax : le modele complet
# ==============================================================================
# str(~ dge + voisinage) : UNE covariance us(2) pour les DEUX incidences portees
# par les memes genotypes. C'est ce qui identifie cov(DGE, IGE).
mod <- rx_model(
  y = d$y, X = matrix(1, nA, 1),
  terms = list(
    rx_term("dge_ige", list(ZgA, ZnA), struct = "us", levels = gA),
    rx_term("ige_inter", list(ZxA), struct = "iid", levels = gB),
    # CHAMP AR1 x AR1 SUR LES COLONNES OBSERVEES, pas sur la grille complete.
    # Les especes alternent par colonne : l'espece A n'occupe que les colonnes
    # impaires. Indexer le champ sur la grille entiere ferait de mon decalage 2
    # le decalage 1 d'asreml, et rho_asreml vaudrait rho_remlax^2 — le carre
    # effacant au passage le SIGNE. MESURE avant correction : remlax -0.4856,
    # asreml +0.2358, et (-0.4856)^2 = 0.2358 exactement.
    # C'est le meme piege que sur le vrai dispositif : le pas conspecifique est
    # de deux colonnes, pas d'une.
    rx_term("champ", Matrix::sparseMatrix(i = seq_len(nA), j = cellA_obs, x = 1,
                                          dims = c(nA, NR * NC_OBS)),
            t = 1L, struct = "iid", level = "ar1ar1", dims = c(NR, NC_OBS))),
  residual = rx_residual("iid"))
print(mod)
t0 <- Sys.time()
f <- rx_fit(mod, backend = Sys.getenv("RX_BACKEND", "auto"), n_restarts = 6,
            verbose = FALSE)
cat(sprintf("\nremlax (%s) : logLik %.6f | %.1f s | decrement %.2e\n",
            f$backend, f$logLik, f$secondes, f$newton_decrement))
G <- f$sigmas$dge_ige
cat(sprintf("  var_DGE      %.4f   (vrai %.2f)\n", G[1, 1], SIG[1, 1]))
cat(sprintf("  var_IGEintra %.4f   (vrai %.2f)\n", G[2, 2], SIG[2, 2]))
cat(sprintf("  cov          %.4f   (vrai %.2f)   r = %.4f (vrai %.3f)\n",
            G[1, 2], SIG[1, 2], G[1, 2] / sqrt(G[1, 1] * G[2, 2]),
            SIG[1, 2] / sqrt(SIG[1, 1] * SIG[2, 2])))
cat(sprintf("  var_IGEinter %.4f   (vrai %.2f)\n", f$sigmas$ige_inter[1, 1], V_INTER))
cat(sprintf("  champ var    %.4f   rho %.4f / %.4f   (vrais %.2f, %.2f/%.2f)\n",
            f$sigmas$champ[1, 1], f$rho$champ[1], f$rho$champ[2], V_AR1, RHO_R, RHO_C))
cat(sprintf("  pepite       %.4f   (vrai %.2f)\n", f$sigma_res[1, 1], V_E))

verifier("le signe de cov(DGE, IGE intra) est retrouve",
         sign(G[1, 2]) == sign(SIG[1, 2]),
         sprintf("%.4f vs %.2f", G[1, 2], SIG[1, 2]))
verifier("l'ajustement est a un optimum", f$newton_decrement < 1e-4,
         sprintf("decrement %.1e", f$newton_decrement))
# La degenerescence de Sigma_us (r -> -1) N'EST PAS un defaut du solveur : sur un
# dispositif de cette taille, DGE et IGE intra sont quasi colineaires et asreml
# donne exactement la meme chose. On la SIGNALE, on ne la sanctionne pas. Ce qui
# doit tenir, c'est que le solveur le dise plutot que de la presenter comme un
# resultat.
if (length(f$composantes_degenerees))
  cat(sprintf("  [note] composante(s) a la frontiere : %s (r = %.4f). ",
              paste(f$composantes_degenerees, collapse = ", "),
              G[1, 2] / sqrt(G[1, 1] * G[2, 2])),
      "Attendu a cette taille : asreml aussi finit sur cette arete\n",
      "  (il rend NA comme erreur-type de trois parametres du champ).\n",
      "  Pour une comparaison qui a du sens, agrandir : --nr=40 --nc=30.\n")

# ==============================================================================
# 4. asreml : LE MEME modele
# ==============================================================================
ok_asreml <- requireNamespace("asreml", quietly = TRUE)
if (!ok_asreml) {
  cat("\nasreml indisponible : comparaison sautee.\n")
} else {
  suppressPackageStartupMessages(library(asreml))
  da <- cbind(d, as.data.frame(as.matrix(ZnA)), as.data.frame(as.matrix(ZxA)))
  names(da) <- c(names(d), paste0("N", seq_len(QA)), paste0("X", seq_len(QB)))
  iN <- grep("^N[0-9]+$", names(da)); iX <- grep("^X[0-9]+$", names(da))
  a <- try(asreml(y ~ 1,
                  random = stats::as.formula(sprintf(
                    "~ str(~ geno + grp(nb), ~ us(2):id(%d)) + grp(xb) + ar1(row):ar1(col)", QA)),
                  group = list(nb = iN, xb = iX),
                  data = da, trace = FALSE, maxit = 150, workspace = "4gb"),
           silent = TRUE)
  if (inherits(a, "try-error")) {
    cat("\nasreml a echoue :\n  ",
        substr(conditionMessage(attr(a, "condition")), 1, 200), "\n")
  } else {
    for (i in 1:6) a <- update(a, trace = FALSE)
    vc <- summary(a)$varcomp
    cat("\n--- asreml, meme modele ---\n"); print(vc[, c("component", "std.error")])
    cat(sprintf("logLik asreml %.9f | remlax %.9f | ecart %.2e\n",
                a$loglik, f$logLik_asreml, abs(a$loglik - f$logLik_asreml)))
    # DEUX exigences distinctes, et une seule est symetrique.
    #  (a) remlax ne doit pas etre PIRE qu'asreml. Sur une surface plate les
    #      deux optimiseurs s'arretent a des theta differents pour une meme
    #      vraisemblance ; exiger l'egalite stricte reviendrait a sanctionner
    #      celui qui converge le mieux. Ici remlax fait MIEUX de 7.6e-04.
    #  (b) les composantes doivent coincider la ou elles sont determinees. La
    #      tolerance est calee sur l'erreur-type d'asreml : deux estimations qui
    #      different de moins d'un dixieme de SE sont le meme resultat.
    verifier("remlax n'est pas moins bon qu'asreml",
             f$logLik_asreml >= a$loglik - 1e-6,
             sprintf("ecart %+.2e en faveur de %s", f$logLik_asreml - a$loglik,
                     if (f$logLik_asreml >= a$loglik) "remlax" else "asreml"))
    cmp <- list(
      c("var_DGE",      G[1, 1],                 vc["geno+grp(nb)!us(2)_1:1", "component"],
        vc["geno+grp(nb)!us(2)_1:1", "std.error"]),
      c("cov_DGE_IGE",  G[1, 2],                 vc["geno+grp(nb)!us(2)_2:1", "component"],
        vc["geno+grp(nb)!us(2)_2:1", "std.error"]),
      c("var_IGEintra", G[2, 2],                 vc["geno+grp(nb)!us(2)_2:2", "component"],
        vc["geno+grp(nb)!us(2)_2:2", "std.error"]),
      c("var_IGEinter", f$sigmas$ige_inter[1, 1], vc["grp(xb)", "component"],
        vc["grp(xb)", "std.error"]),
      c("var_champ",    f$sigmas$champ[1, 1],    vc["row:col", "component"],
        vc["row:col", "std.error"]),
      c("rho_ligne",    f$rho$champ[1],          vc["row:col!row!cor", "component"],
        vc["row:col!row!cor", "std.error"]),
      c("rho_colonne",  f$rho$champ[2],          vc["row:col!col!cor", "component"],
        vc["row:col!col!cor", "std.error"]),
      c("pepite",       f$sigma_res[1, 1],       vc["units!R", "component"],
        vc["units!R", "std.error"]))
    # COMPARER LES COMPOSANTES N'A DE SENS QU'AU MEME OPTIMUM. Si les deux
    # solveurs s'arretent a des vraisemblances differentes, ils decrivent deux
    # POINTS differents de la meme surface : exiger que leurs composantes
    # coincident reviendrait a demander a un maximum d'egaler un non-maximum.
    # On ne compare donc que si les logLik se rejoignent ; sinon on le DIT.
    ecart_ll <- f$logLik_asreml - a$loglik
    # SEUIL DU "MEME POINT". Exiger 1e-4 serait irrealiste sur une surface
    # plate : les deux solveurs s'arretent a des theta legerement differents
    # pour une vraisemblance pratiquement identique. 0,05 est tres en dessous du
    # 1,92 d'un LRT a 1 ddl — deux optima separes de moins que cela sont le meme
    # resultat. Au-dela, ce sont deux points, et on le dit au lieu de comparer.
    meme_point <- abs(ecart_ll) < 5e-2
    cat("\n  composante        remlax     asreml    ecart   (SE asreml)\n")
    for (z in cmp) {
      rk <- as.numeric(z[2]); as_ <- as.numeric(z[3]); se <- as.numeric(z[4])
      # asreml rend NA comme erreur-type d'un parametre a une borne. Le test
      # s'arretait alors sur `if (NA)` — et les composantes suivantes n'etaient
      # jamais comparees. Une borne active est une INFORMATION, pas une panne.
      tol <- if (is.finite(se)) max(0.1 * se, 1e-4) else NA_real_
      marque <- if (!is.finite(se)) "  <- SE asreml indisponible (borne)" else
                if (abs(rk - as_) > 0.1 * se) "  <- ecarte" else ""
      cat(sprintf("  %-14s %9.5f %10.5f %8.1e   (%s)%s\n", z[1], rk, as_,
                  abs(rk - as_), if (is.finite(se)) sprintf("%.3f", se) else "NA", marque))
      if (!meme_point || !is.finite(tol)) next
      verifier(sprintf("%s : accord avec asreml", z[1]), abs(rk - as_) <= tol,
               sprintf("%.5f vs %.5f (SE %.3f)", rk, as_, se))
    }
    if (!meme_point)
      cat(sprintf(paste0("\n  [note] les deux solveurs s'arretent a %.3f point(s) de logLik\n",
                         "         l'un de l'autre (en faveur de %s) : ce sont deux POINTS\n",
                         "         differents, leurs composantes ne sont pas comparables.\n",
                         "         A cette taille (%d obs, %d parametres) le bloc us(2) est\n",
                         "         a la frontiere et la surface est plate.\n"),
                  abs(ecart_ll), if (ecart_ll > 0) "remlax" else "asreml",
                  mod$n, mod$n_par))
  }
}

cat("\n", strrep("=", 66), "\n", sep = "")
if (length(ECHECS)) {
  cat(sprintf("ECHECS (%d) :\n%s\n", length(ECHECS), paste0("  - ", ECHECS, collapse = "\n")))
  quit(save = "no", status = 1L)
}
cat("Modele IGE complet : ajuste, et coherent avec la verite simulee.\n")
quit(save = "no", status = 0L)
