#!/usr/bin/env Rscript
# ==============================================================================
# 08_slope_2A.R — hauteur propre d'une lignee contre son effet sur ses voisines,
# par classe d'allele a un QTL (fig3b_slope_2A)
# ------------------------------------------------------------------------------
# Entrees : les BLUP du multivarie (REMLAX_CH3_OUT/<name>/blups_gen_ble.csv,
# colonnes = etiquettes de Sigma, lignes = genotypes ; ou --blups <csv> au meme
# format que blups_mv_ble.csv du chapitre : Genotype, <trait>_DGE, <trait>_IGE)
# et une table de classes d'alleles (--classes <csv> : Genotype, <colonne de
# classe>). Pour chaque caractere voisin et chaque classe : pente de la
# regression de l'IGE sur le DGE de la hauteur, correlation de Pearson et IC de
# Fisher par rx_cor_z(n = ). Aucun test : la figure compare des BLUP.
#
#   Rscript 08_slope_2A.R --name mvC7 --classes classes_alleles_2A_4B2.csv \
#       [--class_col classe_2A] [--x Hauteur.4] \
#       [--y Hauteur.4,Biomasse_seche_totale.4,nb_grains] [--blups <csv>]
# Sorties dans REMLAX_CH3_OUT/<name>/slope/ : slope_2A.csv, fig3b_slope_2A.png
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(name = "mvC7", classes = "", class_col = "classe_2A", x = "Hauteur.4",
                   y = "Hauteur.4,Biomasse_seche_totale.4,nb_grains", blups = ""))
if (!nzchar(a$classes)) stop("--classes <csv> est requis (Genotype, classe)")
out <- ch3_out(a$name); od <- file.path(out, "slope"); dir.create(od, showWarnings = FALSE, recursive = TRUE)
if (nzchar(a$blups)) {
  B <- utils::read.csv(a$blups, stringsAsFactors = FALSE)
} else {
  b <- utils::read.csv(file.path(out, "blups_gen_ble.csv"), stringsAsFactors = FALSE, check.names = FALSE)
  names(b)[1] <- "Genotype"
  B <- data.frame(Genotype = b$Genotype, stringsAsFactors = FALSE)
  for (cn in names(b)[-1]) {
    if (startsWith(cn, "DGE_")) B[[paste0(sub("^DGE_", "", cn), "_DGE")]] <- b[[cn]]
    if (startsWith(cn, "IGEintra_")) B[[paste0(sub("^IGEintra_", "", cn), "_IGE")]] <- b[[cn]]
  }
}
C <- utils::read.csv(a$classes, stringsAsFactors = FALSE)
D <- merge(B, C[, c("Genotype", a$class_col)], by = "Genotype")
D$classe <- D[[a$class_col]]
ys <- strsplit(a$y, ",")[[1]]; xv <- paste0(a$x, "_DGE")
stat <- function(x, y) {
  b <- stats::coef(stats::lm(y ~ x)); r <- stats::cor(x, y); z <- rx_cor_z(r, n = length(x))
  data.frame(n = length(x), slope = b[[2]], intercept = b[[1]], r = r, ci_low = z$ci_low, ci_high = z$ci_high)
}
res <- list()
for (y in ys) {
  yv <- paste0(y, "_IGE")
  for (cl in c(sort(unique(D$classe)), "all")) {
    s <- if (cl == "all") D else D[D$classe == cl, ]
    res[[length(res) + 1L]] <- cbind(neighbours_trait = y, class = cl, stat(s[[xv]], s[[yv]]))
  }
}
res <- do.call(rbind, res)
utils::write.csv(res, file.path(od, "slope_2A.csv"), row.names = FALSE)
print(res, row.names = FALSE)
cls <- sort(unique(D$classe)); cols <- stats::setNames(c("#E69F00", "#0072B2", "#009E73", "#CC79A7")[seq_along(cls)], cls)
grDevices::png(file.path(od, "fig3b_slope_2A.png"), width = 900 * length(ys), height = 1000, res = 200)
op <- par(mfrow = c(1, length(ys)), mar = c(4, 4, 6, 1))
for (y in ys) {
  yv <- paste0(y, "_IGE")
  plot(D[[xv]], D[[yv]], col = cols[D$classe], pch = 16, cex = 0.7,
       xlab = sprintf("own %s (direct effect, BLUP)", a$x), ylab = sprintf("effect on neighbours' %s (BLUP)", y))
  abline(h = 0, col = "grey80")
  for (cl in cls) { s <- D[D$classe == cl, ]; abline(stats::lm(s[[yv]] ~ s[[xv]]), col = cols[[cl]], lwd = 1.5) }
  abline(stats::lm(D[[yv]] ~ D[[xv]]), col = "black", lty = 2)
  r <- res[res$neighbours_trait == y, ]
  mtext(sprintf("%s: slope %+.2g, r %+.2f [%+.2f; %+.2f] (n = %d)", r$class, r$slope, r$r, r$ci_low, r$ci_high, r$n),
        side = 3, line = rev(seq_len(nrow(r))) - 0.5, cex = 0.6, adj = 0,
        col = ifelse(r$class == "all", "black", cols[r$class]))
}
par(op); grDevices::dev.off()
cat("ecrit :", od, "\n")
