#!/usr/bin/env Rscript
# ==============================================================================
# Validation (a) du chapitre 3 : rx_neighbourhood contre build_voisinage_matrices
# du depot IGE_analysis_2024-2025, sur les DONNEES REELLES.
# ------------------------------------------------------------------------------
# Skip si les donnees ne sont pas la (REMLAX_IGE_REPO absent). Ne modifie rien
# dans le depot IGE : il est lu pour ses fonctions et ses donnees.
#
#   REMLAX_IGE_REPO=/chemin/IGE_analysis_2024-2025 Rscript validation/ch3_neighbourhood_vs_ige.R
#
# Pour chaque geometrie : les quatre incidences par genotype (level) et
# l'incidence par plante (unit) sont comparees apres alignement des lignes par
# ID_GENERAL et des colonnes par nom de genotype sans tiret. Ecart attendu : 0
# ou zero machine (les deux calculs passent par dist() et un produit matriciel
# en double).
# ==============================================================================
suppressPackageStartupMessages({ library(Matrix) })
ige <- Sys.getenv("REMLAX_IGE_REPO", "")
if (!nzchar(ige) || !dir.exists(file.path(ige, "data", "processed"))) {
  cat("SKIP : REMLAX_IGE_REPO absent ou sans data/processed\n"); quit(status = 0)
}
racine <- normalizePath(file.path(dirname(sub("^--file=", "",
            grep("^--file=", commandArgs(), value = TRUE)[1])), ".."))
source(file.path(racine, "R", "remlax.R"))
source(file.path(racine, "R", "remlax_design.R"))
# les fonctions du depot IGE, sans son .Rprofile (renv) : on source le fichier
suppressPackageStartupMessages(source(file.path(ige, "R", "functions.R")))

design <- readRDS(file.path(ige, "data", "processed", "01_imported_data.rds"))$design
design <- design[design$Bac %in% 1:12, ]
design$Genotype <- as.character(design$Genotype)
cat(sprintf("design : %d plantes, %d bacs\n", nrow(design), length(unique(design$Bac))))

# geometries : les sept de tab:geom (06_model_spec.txt des univaries retenus) +
# trois cellules quelconques du cube, dont une a rayons decouples et dilution 0,5
geoms <- list()
for (d in list.dirs(file.path(ige, "output", "results_uni_C"), recursive = FALSE)) {
  sp <- readLines(file.path(d, "06_model_spec.txt"))
  v <- function(k) as.numeric(sub("^.*=", "", grep(paste0("^", k, "="), sp, value = TRUE)))
  geoms[[basename(d)]] <- c(ri = v("ORDRE_INTRA"), re = v("ORDRE_INTER"),
                            li = v("LAMBDA_INTRA"), le = v("LAMBDA_INTER"),
                            di = v("DILUTION_INTRA"), de = v("DILUTION_INTER"))
}
geoms[["cube_ri3_re8_l1_0.5_d0.5_1"]] <- c(ri = 3, re = 8, li = 1, le = 0.5, di = 0.5, de = 1)
geoms[["cube_ri8_re2_l2_2_d0_0.5"]]   <- c(ri = 8, re = 2, li = 2, le = 2, di = 0, de = 0.5)
geoms[["cube_ri1_re1_l0_0_d0_0"]]     <- c(ri = 1, re = 1, li = 0, le = 0, di = 0, de = 0)

paires <- c(Voisins_ble_ble = "Ble<-Ble", Voisins_luz_ble = "Ble<-Luzerne",
            Voisins_luz_luz = "Luzerne<-Luzerne", Voisins_ble_luz = "Luzerne<-Ble")
res <- list()
for (nm in names(geoms)) {
  g <- geoms[[nm]]
  params <- list(correction_distance = TRUE, lambda_intra = g[["li"]], lambda_inter = g[["le"]],
                 dilution_intra = g[["di"]], dilution_inter = g[["de"]],
                 ordre_intra = g[["ri"]], ordre_inter = g[["re"]], p = 1,
                 normalize = FALSE, colname_ID = "ID_GENERAL")
  t0 <- Sys.time()
  ref_g <- build_voisinage_matrices(design, max(g[["ri"]], g[["re"]]),
                                    modifyList(params, list(type_voisinage = "Genotype")))
  ref_u <- build_voisinage_matrices(design, max(g[["ri"]], g[["re"]]),
                                    modifyList(params, list(type_voisinage = "ID")))
  t_ref <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  t0 <- Sys.time()
  nb <- rx_neighbourhood(coord = design[, c("Ligne", "Colonne")], group = design$Espece,
                         block = design$Bac, level = gsub("-", "", design$Genotype),
                         id = design$ID_GENERAL,
                         rank = list("Ble<-Ble" = g[["ri"]], "Luzerne<-Luzerne" = g[["ri"]],
                                     "Ble<-Luzerne" = g[["re"]], "Luzerne<-Ble" = g[["re"]]),
                         reach = list("Ble<-Ble" = g[["li"]], "Luzerne<-Luzerne" = g[["li"]],
                                      "Ble<-Luzerne" = g[["le"]], "Luzerne<-Ble" = g[["le"]]),
                         dilution = list("Ble<-Ble" = g[["di"]], "Luzerne<-Luzerne" = g[["di"]],
                                         "Ble<-Luzerne" = g[["de"]], "Luzerne<-Ble" = g[["de"]]),
                         spacing = c(5, 5), output = "both", sparse = TRUE)
  t_new <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ecarts <- c()
  for (k in names(paires)) {
    R <- ref_g[[k]]; ids <- as.character(R$ID_GENERAL); R$ID_GENERAL <- NULL
    R <- as.matrix(R); rownames(R) <- ids
    M <- as.matrix(nb$level[[paires[[k]]]])
    stopifnot(setequal(rownames(R), rownames(M)), setequal(colnames(R), colnames(M)))
    M <- M[rownames(R), colnames(R), drop = FALSE]
    ecarts[k] <- max(abs(M - R))
  }
  # incidence par plante : la reference est une matrice 1920+1920 x 1920+1920 sur
  # tout le design ; nos quatre blocs unit en sont les sous-matrices par paire.
  ids <- as.character(ref_u$ID_GENERAL); ref_u$ID_GENERAL <- NULL
  U <- as.matrix(ref_u); rownames(U) <- ids; colnames(U) <- sub("^ID", "", colnames(U))
  for (pr in names(nb$unit)) {
    M <- as.matrix(nb$unit[[pr]])
    ecarts[paste0("unit ", pr)] <- max(abs(M - U[rownames(M), colnames(M), drop = FALSE]))
  }
  res[[nm]] <- data.frame(geometrie = nm, ri = g[["ri"]], re = g[["re"]], li = g[["li"]],
                          le = g[["le"]], di = g[["di"]], de = g[["de"]],
                          ecart_max_level = max(ecarts[names(paires)]),
                          ecart_max_unit = max(ecarts[grepl("^unit", names(ecarts))]),
                          s_ige = t_ref, s_remlax = t_new)
  cat(sprintf("%-28s ri %2g re %2g li %3g le %3g di %3g de %3g | level %.2e | unit %.2e | %.1fs vs %.1fs\n",
              nm, g[["ri"]], g[["re"]], g[["li"]], g[["le"]], g[["di"]], g[["de"]],
              res[[nm]]$ecart_max_level, res[[nm]]$ecart_max_unit, t_ref, t_new))
}
tab <- do.call(rbind, res)
out <- file.path(racine, "validation", "results", "ch3_neighbourhood_vs_ige.csv")
utils::write.csv(tab, out, row.names = FALSE)
cat(sprintf("\necart max global : level %.3e, unit %.3e -> %s\n",
            max(tab$ecart_max_level), max(tab$ecart_max_unit), out))
if (max(tab$ecart_max_level, tab$ecart_max_unit) > 1e-12) quit(status = 1)
