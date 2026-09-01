# =============================================================================
# BLUP, erreurs de prediction, et TEMPS : remlax contre asreml
# =============================================================================
# POURQUOI CE FICHIER EXISTE. Toute la validation existante porte sur la
# vraisemblance et les composantes de variance. Or un selectionneur utilise les
# BLUP et leur variance d'erreur de prediction. Un defaut dans ce chemin ne se
# verrait dans AUCUN test actuel — exactement comme le log|K| manquant ne se
# voyait pas dans les estimations, puisqu'une constante ne touche pas le
# gradient. C'est le trou de correction le plus grave du depot.
#
# LE TEMPS EST MESURE ICI ET NULLE PART AILLEURS. asreml n'existe que dans le
# conteneur, et c'est la seule comparaison de vitesse a la fois disponible et
# attendue par un lecteur. Les deux moteurs ajustent le MEME dispositif, dans le
# meme processus, a la suite.
#
# ECRIT DEFENSIVEMENT. Les noms exacts des sorties d'asreml (coefficients,
# erreurs types des effets aleatoires) ne sont pas verifiables hors du
# conteneur : ce script IMPRIME ce qu'il trouve avant de comparer, et une
# structure inattendue produit un diagnostic, non un echec muet.
suppressPackageStartupMessages({library(Matrix); library(asreml)})
source("R/remlax.R")
ok <- 0L; ko <- 0L
verifier <- function(nom, cond, detail = "") {
  if (isTRUE(cond)) { ok <<- ok + 1L; cat(sprintf("  OK    %-44s %s\n", nom, detail)) }
  else { ko <<- ko + 1L; cat(sprintf("  ECHEC %-44s %s\n", nom, detail)) }
}

set.seed(20260901)
nr <- 30L; nc <- 24L; n <- nr * nc; q <- 60L
geno <- factor(rep_len(seq_len(q), n), levels = seq_len(q))
u <- rnorm(q, 0, 1.1)
row <- factor(rep(seq_len(nr), each = nc)); col <- factor(rep(seq_len(nc), nr))
y <- 5 + u[as.integer(geno)] + rnorm(n, 0, 0.8)
d <- data.frame(y = y, geno = geno, row = row, col = col)

cat("=== 1. ajustement des deux cotes, temps mesure ===\n")
t_a <- system.time(a <- asreml(y ~ 1, random = ~ geno, data = d, trace = FALSE))[["elapsed"]]
t_r <- system.time(f <- rx_reml(y ~ 1, random = ~ geno, data = d,
                                hessian = TRUE, blups = TRUE, verbose = FALSE))[["elapsed"]]
cat(sprintf("  asreml %7.2f s | remlax %7.2f s | rapport %.2f\n", t_a, t_r, t_r / t_a))
verifier("logLik dans la convention asreml", abs(f$logLik_asreml - a$loglik) < 1e-5,
         sprintf("%.9f vs %.9f", f$logLik_asreml, a$loglik))

cat("\n=== 2. ce que chaque cote expose (diagnostic avant comparaison) ===\n")
cat("  remlax$blups :", paste(names(f$blups), collapse = ", "), "\n")
sa <- try(summary(a, coef = TRUE), silent = TRUE)
cat("  asreml summary(coef=TRUE) :",
    if (inherits(sa, "try-error")) "indisponible" else paste(names(sa), collapse = ", "), "\n")
cr <- if (!inherits(sa, "try-error") && !is.null(sa$coef.random)) sa$coef.random else NULL
if (is.null(cr)) cr <- try(a$coefficients$random, silent = TRUE)
cat("  colonnes des effets aleatoires asreml :",
    if (inherits(cr, "try-error") || is.null(cr)) "introuvables"
    else paste(colnames(as.matrix(cr)), collapse = " | "), "\n")

cat("\n=== 3. BLUP : les deux cotes predisent-ils les memes effets ? ===\n")
bl_r <- as.numeric(f$blups[["geno"]])
cr_m <- if (!is.null(cr) && !inherits(cr, "try-error")) as.matrix(cr) else NULL
if (!is.null(cr_m)) {
  lignes <- grep("^geno", rownames(cr_m))
  bl_a <- as.numeric(cr_m[lignes, 1])
  # asreml ordonne ses niveaux comme le facteur ; on aligne par NOM, jamais par position
  noms_a <- sub("^geno_?", "", rownames(cr_m)[lignes])
  ord <- match(levels(geno), noms_a)
  bl_a <- bl_a[ord]
  n_comm <- sum(!is.na(bl_a))
  cat(sprintf("  %d niveaux apparies sur %d\n", n_comm, q))
  if (n_comm >= q %/% 2) {
    ecart <- max(abs(bl_r - bl_a), na.rm = TRUE)
    corr <- cor(bl_r, bl_a, use = "complete.obs")
    verifier("BLUP : ecart maximal", ecart < 1e-4, sprintf("max|diff| = %.3e", ecart))
    verifier("BLUP : correlation", corr > 1 - 1e-8, sprintf("r = %.12f", corr))
  } else {
    verifier("BLUP : appariement des niveaux", FALSE,
             sprintf("seulement %d niveaux apparies", n_comm))
  }
  if (ncol(cr_m) >= 2) {
    se_a <- as.numeric(cr_m[lignes, 2])[ord]
    se_r <- if (!is.null(f$blup_se) && !is.null(f$blup_se[["geno"]]))
              as.numeric(f$blup_se[["geno"]]) else NULL
    if (!is.null(se_r)) {
      e <- max(abs(se_r - se_a), na.rm = TRUE)
      verifier("erreur de prediction des BLUP", e < 1e-4, sprintf("max|diff| = %.3e", e))
    } else {
      cat("  NOTE : remlax n'expose pas d'erreur-type de BLUP ($blup_se absent).\n")
      cat("         C'est un MANQUE a combler, pas un desaccord. Champs presents :",
          paste(grep("blup|se", names(f), value = TRUE), collapse = ", "), "\n")
    }
  }
} else {
  verifier("recuperation des BLUP asreml", FALSE, "structure inattendue, voir diagnostic")
}

cat("\n=== 4. effets fixes et leur variance ===\n")
b_a <- as.numeric(a$coefficients$fixed)
b_r <- as.numeric(f$beta)
verifier("effet fixe (intercept)", abs(b_r[1] - b_a[length(b_a)]) < 1e-5,
         sprintf("%.9f vs %.9f", b_r[1], b_a[length(b_a)]))

cat(sprintf("\n=== %d verifications, %d echec(s) | temps asreml %.2f s, remlax %.2f s ===\n",
            ok + ko, ko, t_a, t_r))
if (ko > 0) quit(status = 1)
