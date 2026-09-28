#!/usr/bin/env Rscript
# ==============================================================================
# 02_fit_univariate.R — une cellule : un caractere, une espece, une geometrie
# ------------------------------------------------------------------------------
# Modele : y = X b + DGE + IGE_intra (us 2x2 avec K de l'espece) + IGE_inter
# (K de l'autre espece) + IEE intra + IEE inter (iid par plante) + cinq termes
# spatiaux iid + residuelle iid. Effets fixes : moyenne, poids de graine, date de
# semis (ble). Options d'ajustement figees du cube : maxiter 3000, polish 25,
# floor -12, ceil 12, CPU.
#
#   Rscript 02_fit_univariate.R --species Ble --trait Hauteur.4 \
#       --rank_within 5 --rank_between 7 --reach_within 0 --reach_between 0 \
#       --dilution_within 0 --dilution_between 0 [--hessian FALSE] [--out <dir>]
#       [--tag <nom>] [--maxiter 3000] [--keep_model]
# Sorties dans <out>/<tag>/ : model_spec.txt, theta.csv, fit_summary.csv,
# sigmas.csv, exposure.csv, diagnostics.csv, fit.rds (et model.rds avec
# --keep_model). Le tag par defaut est celui du cube :
# ord_<species>_<trait>_li<>_le<>_di<>_de<>_ri<>_re<>.
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(species = "Ble", trait = "Hauteur.4", rank_within = 5, rank_between = 7,
                   reach_within = 0, reach_between = 0, dilution_within = 0, dilution_between = 0,
                   hessian = "TRUE", out = "", tag = "", maxiter = 3000, keep_model = FALSE,
                   data = ""))
tag <- if (nzchar(a$tag)) a$tag else ch3_tag(a)
out <- if (nzchar(a$out)) file.path(a$out, tag) else ch3_out("cells", tag)
data <- readRDS(if (nzchar(a$data)) a$data else ch3_out("data.rds"))
invisible(ch3_fit_cell(a, data, out, tag))

