# ==============================================================================
# Le scan de remlax reproduit-il rrBLUP au SNP pres ?
# ==============================================================================
# POURQUOI UNE REFERENCE EXTERNE, ET POURQUOI SUR CE CAS-LA. `remlax.scan`
# accepte des V que les paquets de GWAS ne savent pas traiter : deux noyaux, une
# residuelle AR1, des sections heterogenes. Il n'existe donc aucune reference
# externe sur le cas qui nous interesse. Mais il en existe une sur le cas
# DEGENERE ou les deux sont censes coincider :
#
#     un seul noyau, residuelle iid, une observation par genotype
#
# C'est le modele d'EMMAX. Si les deux implementations s'accordent la, l'algebre
# du scan est validee contre du code tiers eprouve ; le reste n'est que la meme
# algebre avec une autre V, deja verifiee contre le test de Wald dans
# tests/python/test_scan.py.
#
# POURQUOI rrBLUP ET PAS statgenGWAS. Les deux conviendraient. rrBLUP::GWAS avec
# P3D = TRUE est l'EMMAX le plus cite en amelioration des plantes, et c'est du R
# PUR : il se charge partout, sans bibliotheque compilee. Le script
# `etude_scan_vs_statgen.R`, a cote, fait la meme chose avec statgenGWAS pour
# qui peut le charger.
#
# CE QUI DIFFERE ENTRE LES DEUX, ET QUI DOIT DONC ETRE RECONSTRUIT. rrBLUP
# reestime l'echelle residuelle A CHAQUE SNP, sur les residus du modele qui
# CONTIENT le SNP, et teste en F(1, n-2) :
#
#     s2_j   = resid_j' Hinv resid_j / v2       v2 = n - 2
#     Fstat  = beta_j^2 / (s2_j Winv_pp)
#
# Le scan garde l'echelle du modele NUL et teste en chi2 a 1 ddl. C'est le point
# documente en §3.5 de la note de conception : a V figee, un SNP de gros effet
# est teste sous une variance residuelle surestimee, donc conservativement. La
# relation entre les deux est exacte et se reconstruit sans rien recalculer en
# dimension n :
#
#     Fstat_j = chi2_j * v2 / (y'Py - num_j^2/den_j)
#
# ou num_j et den_j sont les sorties brutes du scan et y'Py un scalaire calcule
# une fois. La comparaison porte donc sur TROIS niveaux :
#   (A) beta_j          -> doit coincider a la precision machine (algebre du GLS)
#   (B) le score rrBLUP -> reconstruit depuis le scan, doit coincider aussi
#                          (l'ecart de convention est alors entierement explique)
#   (C) l'ampleur de la conservativite, mesuree la ou elle mord
#
# Lancement :  Rscript validation/etude_scan_vs_rrblup.R <dossier_sortie>
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)
sortie <- if (length(args)) args[1] else file.path(tempdir(), "scan_vs_rrblup")
dir.create(sortie, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(rrBLUP))
cat("rrBLUP", as.character(packageVersion("rrBLUP")), "\n")

set.seed(20260907)
q <- 250L      # genotypes = observations : une par genotype, comme EMMAX
p <- 800L

# --- marqueurs en codage -1/0/1, celui de rrBLUP -------------------------
frq <- runif(p, 0.10, 0.90)
G <- matrix(rbinom(q * p, 2, rep(frq, each = q)) - 1L, nrow = q)
G <- G[, apply(G, 2, stats::sd) > 0, drop = FALSE]
p <- ncol(G)
rownames(G) <- sprintf("g%03d", seq_len(q))
colnames(G) <- sprintf("snp%04d", seq_len(p))

K <- rrBLUP::A.mat(G)          # la parente de rrBLUP, pas une autre

# --- phenotype : fond polygenique + UN SNP causal de gros effet ----------
# Le gros effet est voulu : c'est la ou les deux conventions divergent le plus,
# donc la ou la comparaison est informative.
Lk <- t(chol(K + diag(1e-6, q)))
j0 <- sample.int(p, 1L)
y <- as.numeric(2.0 + 0.80 * scale(G[, j0])[, 1] +
                1.0 * (Lk %*% rnorm(q))[, 1] + rnorm(q, sd = 1.0))

geno <- data.frame(marker = colnames(G), chrom = 1L, pos = seq_len(p),
                   t(G), check.names = FALSE, stringsAsFactors = FALSE)
pheno <- data.frame(line = rownames(G), rdt = y, stringsAsFactors = FALSE)

# --- rrBLUP : composantes de variance une fois, puis un test par SNP -----
res <- rrBLUP::GWAS(pheno = pheno, geno = geno, K = K, n.PC = 0,
                    min.MAF = 0, P3D = TRUE, plot = FALSE)
# Les composantes que P3D a figees : mixed.solve sur le modele NUL, exactement
# ce que GWAS() fait en interne avant la boucle.
ms <- rrBLUP::mixed.solve(y = y, K = K, X = matrix(1, q, 1), SE = FALSE)
cat(sprintf("composantes (mixed.solve) : Vu = %.8g  Ve = %.8g\n", ms$Vu, ms$Ve))
cat(sprintf("SNP causal : %s  score rrBLUP = %.4f\n",
            colnames(G)[j0], res$rdt[res$marker == colnames(G)[j0]]))

utils::write.csv(data.frame(snp = res$marker, score = res$rdt),
                 file.path(sortie, "rrblup_scores.csv"), row.names = FALSE)
utils::write.table(y, file.path(sortie, "y.txt"), row.names = FALSE, col.names = FALSE)
utils::write.table(K, file.path(sortie, "K.txt"), row.names = FALSE, col.names = FALSE)
utils::write.table(G, file.path(sortie, "M.txt"), row.names = FALSE, col.names = FALSE)
writeLines(sprintf("%.17g", c(ms$Vu, ms$Ve, q, p, j0)),
           file.path(sortie, "meta.txt"))   # Vu, Ve, q, p, indice du SNP causal
cat("ecrit dans", sortie, "\n")
