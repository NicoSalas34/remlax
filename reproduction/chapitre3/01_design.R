#!/usr/bin/env Rscript
# ==============================================================================
# 01_design.R — incidences de voisinage pour une geometrie donnee
# ------------------------------------------------------------------------------
# Construit, pour l'espece recevante demandee, les incidences par genotype
# (termes IGE) et par plante (termes IEE) avec rx_neighbourhood, sur le design
# complet des bacs 1 a 12. Les incidences sont restreintes aux observations dans
# 02_fit_univariate.R / 05_fit_multivariate.R, pas ici.
#
#   Rscript 01_design.R --species Ble --rank_within 5 --rank_between 7 \
#       --reach_within 0 --reach_between 0 --dilution_within 0 --dilution_between 0
# Sortie : <REMLAX_CH3_OUT>/design_<species>_ri<>_re<>_li<>_le<>_di<>_de<>.rds et
#          un resume (nombre de voisins, sommes de poids, k a K = I par plante).
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(species = "Ble", rank_within = 5, rank_between = 7, reach_within = 0,
                   reach_between = 0, dilution_within = 0, dilution_between = 0))
data <- readRDS(ch3_out("data.rds"))
nb <- ch3_neighbourhood(data, a, a$species)
print(nb)
tag <- sprintf("design_%s_ri%g_re%g_li%g_le%g_di%g_de%g", a$species, a$rank_within, a$rank_between,
               a$reach_within, a$reach_between, a$dilution_within, a$dilution_between)
saveRDS(list(geom = a, nb = nb), ch3_out(paste0(tag, ".rds")))
other <- setdiff(c("Ble", "Luzerne"), a$species)
res <- data.frame(pair = nb$params$pair, rank = nb$params$rank, reach = nb$params$reach,
                  dilution = nb$params$dilution,
                  mean_neighbours = vapply(nb$n_neighbours, mean, 1),
                  mean_row_sum = vapply(nb$level, function(M) mean(Matrix::rowSums(M)), 1),
                  k_identity_all_units = vapply(nb$level, function(M) mean(Matrix::rowSums(M^2)), 1))
utils::write.csv(res, ch3_out(paste0(tag, "_summary.csv")), row.names = FALSE)
print(res, row.names = FALSE)
cat("ecrit :", ch3_out(paste0(tag, ".rds")), "\n")
