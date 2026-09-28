#!/usr/bin/env Rscript
# ==============================================================================
# 07_correlations.R — correlations genetiques, residuelles et de TBV avec leurs
# intervalles de Fisher (fig4, fig5, figS5, figS6, tab:supp:cor, tab:supp:rescor)
# ------------------------------------------------------------------------------
# Lit REMLAX_CH3_OUT/<name>/ratios/ratios.csv ecrit par 06_ratios.R.
#   Rscript 07_correlations.R --name mvC7 [--min_abs 0.2] [--width_max 1.5]
# Sorties dans REMLAX_CH3_OUT/<name>/correlations/ :
#   genetic_correlations.csv   r, SE, z, IC de Fisher, largeur, drapeau
#                              (NOT_ESTIMABLE quand la largeur depasse width_max)
#   residual_correlations.csv, tbv_correlations.csv
#   fig4_genetic_correlations_wheat.png, fig5_genetic_correlations_alfalfa.png
#   figS5_tbv.png (tau2 et correlations de TBV), figS6_residual_correlations.png
# Le seuil |r| >= min_abs est un filtre d'AFFICHAGE des figures, pas d'estimation.
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(name = "mvC7", min_abs = 0.2, width_max = 1.5))
out <- ch3_out(a$name); od <- file.path(out, "correlations"); dir.create(od, showWarnings = FALSE, recursive = TRUE)
rat <- utils::read.csv(file.path(out, "ratios", "ratios.csv"), stringsAsFactors = FALSE)
cz <- function(r) {
  z <- rx_cor_z(r$estimate, se = r$se, width_max = a$width_max)
  data.frame(term = r$target, pair = r$component, a = sub("~.*$", "", r$component), b = sub("^.*~", "", r$component),
             r = r$estimate, se = r$se, z = r$z, ci_low = z$ci_low, ci_high = z$ci_high, width = z$width,
             flag = ifelse(!is.na(z$informative) & !z$informative, "NOT_ESTIMABLE", r$flag), stringsAsFactors = FALSE)
}
gc <- cz(rat[rat$quantity == "cor", ]); rc <- cz(rat[rat$quantity == "rescor", ]); tc <- cz(rat[rat$quantity == "tbv_cor", ])
utils::write.csv(gc, file.path(od, "genetic_correlations.csv"), row.names = FALSE)
utils::write.csv(rc, file.path(od, "residual_correlations.csv"), row.names = FALSE)
utils::write.csv(tc, file.path(od, "tbv_correlations.csv"), row.names = FALSE)
cat(sprintf("%d correlations genetiques (%d non estimables), %d residuelles, %d de TBV\n",
            nrow(gc), sum(gc$flag == "NOT_ESTIMABLE"), nrow(rc), nrow(tc)))

matrice <- function(tab) {
  labs <- unique(c(tab$a, tab$b)); n <- length(labs)
  R <- matrix(NA_real_, n, n, dimnames = list(labs, labs)); F <- matrix("", n, n, dimnames = list(labs, labs))
  for (i in seq_len(nrow(tab))) { R[tab$a[i], tab$b[i]] <- R[tab$b[i], tab$a[i]] <- tab$r[i]
                                  F[tab$a[i], tab$b[i]] <- F[tab$b[i], tab$a[i]] <- tab$flag[i] }
  diag(R) <- 1; list(R = R, F = F)
}
dessiner <- function(tab, fn, titre, min_abs = a$min_abs) {
  if (!nrow(tab)) return(invisible())
  mm <- matrice(tab); R <- mm$R; F <- mm$F; n <- nrow(R)
  pal <- grDevices::colorRampPalette(c("#B2182B", "white", "#2166AC"))(101)
  grDevices::png(fn, width = 220 * n + 600, height = 220 * n + 400, res = 200)
  op <- par(mar = c(1, 12, 12, 1))
  Rp <- R; Rp[abs(Rp) < min_abs & row(Rp) != col(Rp)] <- NA
  image(seq_len(n), seq_len(n), t(Rp[n:1, ]), col = pal, zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "", main = titre)
  axis(3, at = seq_len(n), labels = colnames(R), las = 2, cex.axis = 0.7, tick = FALSE)
  axis(2, at = seq_len(n), labels = rev(rownames(R)), las = 1, cex.axis = 0.7, tick = FALSE)
  for (i in seq_len(n)) for (j in seq_len(n)) if (i != j) {
    v <- R[i, j]; f <- F[i, j]
    if (is.na(v) || abs(v) < min_abs) next
    text(j, n - i + 1, if (f == "NOT_ESTIMABLE") "n.e." else sprintf("%.2f", v), cex = 0.6,
         col = if (f == "NOT_ESTIMABLE") "grey40" else "black")
  }
  par(op); grDevices::dev.off()
}
dessiner(gc[gc$term == "gen_ble", ], file.path(od, "fig4_genetic_correlations_wheat.png"),
         sprintf("Genetic correlations, wheat (|r| >= %g shown)", a$min_abs))
dessiner(gc[gc$term == "gen_luz", ], file.path(od, "fig5_genetic_correlations_alfalfa.png"),
         sprintf("Genetic correlations, alfalfa (|r| >= %g shown)", a$min_abs))
dessiner(rc, file.path(od, "figS6_residual_correlations.png"), "Residual correlations, wheat", min_abs = 0)
# figS5 : tau2 (a) et correlations de TBV (b)
tau <- utils::read.csv(file.path(out, "ratios", "tau2.csv"), stringsAsFactors = FALSE)
tau <- tau[tau$quantity == "tau2", ]
grDevices::png(file.path(od, "figS5_tbv.png"), width = 2600, height = 1200, res = 200)
op <- par(mfrow = c(1, 2), mar = c(10, 4, 3, 1))
lab <- paste(tau$trait, tau$sense)
bp <- barplot(tau$estimate, names.arg = lab, las = 2, col = ifelse(tau$sense == "own", "#009E73", "#CC79A7"),
              ylab = "tau2 = Var(TBV) / Var(P)", main = "(a) tau2 with Wald intervals",
              ylim = c(min(0, tau$ci_low, na.rm = TRUE), max(tau$ci_high, na.rm = TRUE)))
arrows(bp, tau$ci_low, bp, tau$ci_high, angle = 90, code = 3, length = 0.02); abline(h = 1, lty = 2)
par(mar = c(1, 12, 12, 1))
if (nrow(tc)) {
  mm <- matrice(tc); R <- mm$R; n <- nrow(R)
  pal <- grDevices::colorRampPalette(c("#B2182B", "white", "#2166AC"))(101)
  image(seq_len(n), seq_len(n), t(R[n:1, ]), col = pal, zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
        main = "(b) correlations between total genetic values")
  axis(3, at = seq_len(n), labels = colnames(R), las = 2, cex.axis = 0.6, tick = FALSE)
  axis(2, at = seq_len(n), labels = rev(rownames(R)), las = 1, cex.axis = 0.6, tick = FALSE)
  for (i in seq_len(n)) for (j in seq_len(n)) if (i != j && !is.na(R[i, j]))
    text(j, n - i + 1, sprintf("%.2f", R[i, j]), cex = 0.5)
}
par(op); grDevices::dev.off()
cat("ecrit :", od, "\n")
