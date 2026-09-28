#!/usr/bin/env Rscript
# ==============================================================================
# 00_data.R — lecture des donnees du chapitre 3 (dispositif, phenotypes, GRM)
# ------------------------------------------------------------------------------
# Entree  : REMLAX_IGE_DATA = data/processed du depot IGE (01_imported_data.rds,
#           02_donnees_ble_clean.rds, 02_donnees_luz_clean.rds, 02_GK_matrix.rds,
#           02_GK_luz_matrix.rds). Aucun code du depot IGE n'est charge.
# Sortie  : <REMLAX_CH3_OUT>/data.rds (design des bacs 1-12, tables par espece,
#           GRM blendees 0,98 K + 0,02 I, univers de genotypes) et
#           data_summary.csv.
#   Rscript 00_data.R
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()

data <- ch3_read_data()
saveRDS(data, ch3_out("data.rds"))
res <- data.frame(
  quoi = c("plantes du design (bacs 1-12)", "bacs", "plantes ble", "plantes luzerne",
           "genotypes ble presents", "genotypes luzerne presents", "GRM ble", "GRM luzerne",
           "diag moyenne GRM ble blendee (univers)", "diag moyenne GRM luzerne blendee (univers)"),
  valeur = c(nrow(data$design), length(unique(data$design$Bac)), nrow(data$tables$Ble),
             nrow(data$tables$Luzerne), length(data$glev$Ble), length(data$glev$Luzerne),
             paste(dim(data$K$Ble), collapse = "x"), paste(dim(data$K$Luzerne), collapse = "x"),
             sprintf("%.5f", mean(diag(data$K$Ble[data$glev$Ble, data$glev$Ble]))),
             sprintf("%.5f", mean(diag(data$K$Luzerne[data$glev$Luzerne, data$glev$Luzerne])))))
utils::write.csv(res, ch3_out("data_summary.csv"), row.names = FALSE)
print(res, row.names = FALSE)
cat("ecrit :", ch3_out("data.rds"), "\n")
