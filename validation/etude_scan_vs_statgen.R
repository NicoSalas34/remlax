# ==============================================================================
# Le scan de remlax reproduit-il statgenGWAS au SNP pres ?
# ==============================================================================
# POURQUOI CE TEST EXISTE, ET POURQUOI IL EST LE SEUL POSSIBLE. `remlax.scan`
# accepte des V que statgenGWAS ne sait pas traiter : deux noyaux, une
# residuelle AR1, des sections heterogenes. Il n'existe donc PAS de reference
# externe sur le cas qui nous interesse. Mais il en existe une sur le cas
# DEGENERE ou les deux sont censes coincider exactement :
#
#     un seul noyau, residuelle iid, une observation par genotype
#
# C'est le modele d'EMMAX. Si les deux implementations s'accordent la, l'algebre
# du scan est validee contre du code tiers eprouve ; le reste (deux noyaux, AR1)
# n'est que la meme algebre avec une autre V, et c'est deja verifie contre le
# test de Wald dans tests/python/test_scan.py.
#
# CE QUI EST COMPARE, ET POURQUOI PAS LES p-VALEURS. On compare `effect` et
# `effectSe` — les quantites que les deux calculent. Les p-valeurs, elles,
# dependent de la LOI DE REFERENCE : statgenGWAS rapporte un test de Student,
# remlax un chi2 a 1 ddl. Les deux sont defendables a V figee et ne diffe`rent
# que par le denominateur des degres de liberte ; comparer les p-valeurs
# melangerait un desaccord d'algebre (grave) avec une convention de loi (pas
# grave, et documentee).
#
# ON FIGE LES COMPOSANTES DE VARIANCE. statgenGWAS les estime par EMMA, remlax
# par REML : rien ne garantit le meme optimum a la 12e decimale, et un ecart la
# se propagerait a tous les SNP. Comme c'est l'algebre du SCAN qu'on teste et
# non l'estimation (deja testee ailleurs dans remlax), on prend les composantes
# de statgenGWAS et on les impose au scan. Le desaccord residuel ne peut alors
# venir que du scan.
#
# Lancement :  Rscript validation/etude_scan_vs_statgen.R <dossier_sortie>
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)
sortie <- if (length(args)) args[1] else file.path(tempdir(), "scan_vs_statgen")
dir.create(sortie, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(statgenGWAS))
cat("statgenGWAS", as.character(packageVersion("statgenGWAS")), "\n")

set.seed(20260907)
q <- 250L      # genotypes = observations (une par genotype, comme EMMAX)
p <- 800L      # SNP

# --- genotypes en dosage 0/1/2, frequences variees -----------------------
frq <- runif(p, 0.10, 0.90)
G <- matrix(rbinom(q * p, 2, rep(frq, each = q)), nrow = q)
G <- G[, apply(G, 2, stats::sd) > 0, drop = FALSE]
p <- ncol(G)
rownames(G) <- sprintf("g%03d", seq_len(q))
colnames(G) <- sprintf("snp%04d", seq_len(p))

carte <- data.frame(chr = rep(1L, p), pos = seq_len(p))
rownames(carte) <- colnames(G)

# --- parente vanRaden, celle de statgenGWAS ------------------------------
K <- kinship(G, method = "vanRaden")

# --- phenotype : fond polygenique + un SNP causal ------------------------
Lk <- t(chol(K + diag(1e-6, q)))
j0 <- sample.int(p, 1L)
y <- 2.0 + 0.45 * scale(G[, j0])[, 1] +
     1.0 * (Lk %*% rnorm(q))[, 1] + rnorm(q, sd = 1.0)
pheno <- data.frame(genotype = rownames(G), rdt = y, stringsAsFactors = FALSE)

# --- statgenGWAS ---------------------------------------------------------
gd <- createGData(geno = G, map = carte, pheno = pheno, kin = K)
res <- runSingleTraitGwas(gd, traits = "rdt", kin = K,
                          GLSMethod = "single",     # pas de LOCO : un seul V
                          remlAlgo = "EMMA",
                          MAF = 0,                  # aucun SNP ecarte
                          thrType = "fixed", LODThr = 0)
gw <- as.data.frame(res$GWAResult$pheno)
vc <- res$GWASInfo$varComp
cat("composantes de variance (EMMA) :\n"); print(unlist(vc))

# --- ce que le scan doit relire -----------------------------------------
# Les doses sont recuperees TELLES QUE statgenGWAS les a vues (imputation et
# filtres compris) : comparer sur d'autres colonnes ne comparerait rien.
Gs <- gd$markers[rownames(G), gw$snp, drop = FALSE]
utils::write.csv(gw[, c("snp", "allFreq", "effect", "effectSe", "pValue", "LOD")],
                 file.path(sortie, "statgen_resultats.csv"), row.names = FALSE)
utils::write.csv(data.frame(genotype = rownames(G), y = y),
                 file.path(sortie, "pheno.csv"), row.names = FALSE)
utils::write.table(K, file.path(sortie, "K.txt"), row.names = FALSE, col.names = FALSE)
utils::write.table(Gs, file.path(sortie, "M.txt"), row.names = FALSE, col.names = FALSE)
writeLines(as.character(gw$snp), file.path(sortie, "snp.txt"))
writeLines(jsonlite::toJSON(list(varComp = vc, q = q, p = length(gw$snp),
                                 snp_causal = colnames(G)[j0]),
                            auto_unbox = TRUE),
           file.path(sortie, "meta.json"))
cat("ecrit dans", sortie, ":", length(gw$snp), "SNP,", q, "genotypes\n")
