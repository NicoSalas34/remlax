#!/usr/bin/env Rscript
# ==============================================================================
# 04_cube_summary.R — bilan AIC du cube : tab:geom, comptes, figures S1-S3, S7-S13
# ------------------------------------------------------------------------------
# Entree : les tables cube_<species>_<trait>.csv ecrites par 03_cube.R
# --aggregate (dans REMLAX_CH3_OUT), ou, avec --reference <dossier>, les tables
# agregees du chapitre cube_C_<trait>.csv (colonnes du pipeline d'origine ; les
# deux rangs y sont lus dans le tag, la colonne `ordre` etant le rang nominal).
#
#   Rscript 04_cube_summary.R [--reference <dossier des cube_C_*.csv>] [--tol 2]
#
# Sorties (REMLAX_CH3_OUT/cube_summary/) :
#   tab_geom.csv      meilleure cellule par caractere, etendues marginales sur
#                     l'ensemble soutenu (dAIC <= tol), n_supported, n_product
#   counts.csv        cellules, pdHess FALSE, composantes au bord, manquantes ;
#                     meilleure cellule PD et son ecart d'AIC ; AIC_eff
#   figS7-S13_cube_<trait>.png  le cube deplie : 15 x 15 blocs (dilution x portee,
#                     intra en lignes, inter en colonnes) de 10 x 10 rangs, dAIC
#                     borne a 8
#   figS1_reach.png   profil de dAIC sur (portee intra, portee inter), minimise
#                     sur les quatre autres coordonnees, par caractere
#   figS2_dilution.png idem sur (dilution intra, dilution inter)
#   figS3_specification.png  les quatre etapes de la specification : rangs
#                     communs, portees decouplees, dilutions decouplees, les
#                     deux ; meilleur AIC sous chaque contrainte, par caractere
# Les figures S1 a S3 du chapitre viennent d'une campagne separee (16, 25 et 4
# ajustements par caractere) ; ici les memes comparaisons sont lues dans le cube,
# qui contient ces cellules.
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(reference = "", tol = 2, pattern = ""))
od <- ch3_out("cube_summary"); dir.create(od, showWarnings = FALSE)
coords <- c("rank_within", "rank_between", "reach_within", "reach_between", "dilution_within", "dilution_between")

lire <- function() {
  if (nzchar(a$reference)) {
    fs <- list.files(a$reference, pattern = "^cube_C_.*\\.csv$", full.names = TRUE)
    tabs <- lapply(fs, function(f) {
      x <- utils::read.csv(f, stringsAsFactors = FALSE)
      data.frame(species = x$espece, trait = x$trait,
                 rank_within = as.integer(sub(".*_ri([0-9]+)_re.*", "\\1", x$tag)),
                 rank_between = as.integer(sub(".*_re([0-9]+)$", "\\1", x$tag)),
                 reach_within = x$lambda_intra, reach_between = x$lambda_inter,
                 dilution_within = x$d_intra, dilution_between = x$d_inter,
                 n_obs = x$n_obs, n_par = x$n_var, logLik = x$logLik, AIC = x$AIC,
                 pd_hessian = as.logical(x$pdHess), n_at_bound = x$n_at_bound,
                 n_par_free = x$n_var - x$n_at_bound, converged = as.logical(x$converge),
                 stringsAsFactors = FALSE)
    })
  } else {
    fs <- list.files(ch3_out(), pattern = "^cube_.*\\.csv$", full.names = TRUE)
    tabs <- lapply(fs, function(f) utils::read.csv(f, stringsAsFactors = FALSE))
  }
  if (!length(tabs)) stop("aucune table de cube trouvee")
  do.call(rbind, tabs)
}
cube <- lire()
cube$n_par_free <- ifelse(is.na(cube$n_par_free), cube$n_par - cube$n_at_bound, cube$n_par_free)
cat(sprintf("%d cellules, %d caractere(s)\n", nrow(cube), length(unique(cube$trait))))

gs <- rx_grid_summary(cube, coords = coords, by = "trait", tol = a$tol, effective = TRUE)
print(gs)
# --- tab:geom -----------------------------------------------------------------
rg <- gs$ranges
tg <- gs$best
for (cc in coords) {
  r <- rg[rg$coord == cc, ]
  tg[[paste0(cc, "_range")]] <- sprintf("[%g; %g]", r$min[match(tg$.group, r$.group)], r$max[match(tg$.group, r$.group)])
}
tg$n_supported <- gs$n_supported$n_supported[match(tg$.group, gs$n_supported$.group)]
tg$n_product <- gs$n_supported$n_product[match(tg$.group, gs$n_supported$.group)]
names(tg)[names(tg) == ".group"] <- "trait"
utils::write.csv(tg, file.path(od, "tab_geom.csv"), row.names = FALSE)
cnt <- gs$counts; names(cnt)[names(cnt) == ".group"] <- "trait"
bp <- gs$best_pd
if (!is.null(bp)) {
  names(bp)[names(bp) == ".group"] <- "trait"
  cnt <- merge(cnt, bp[, c("trait", "delta_pd", "same_as_best")], by = "trait", all.x = TRUE)
} else cnt$delta_pd <- NA_real_   # aucune cellule avec Hessien (cube lance avec --hessian FALSE)
cnt <- merge(cnt, gs$effective$best_moved |> (\(x) { names(x)[1] <- "trait"; x })(), by = "trait", all.x = TRUE)
utils::write.csv(cnt, file.path(od, "counts.csv"), row.names = FALSE)
cat(sprintf("total : %d cellules, %d avec au moins une composante au bord, %d sans Hessien PD\n",
            sum(cnt$n_cells), sum(cnt$n_at_bound_pos, na.rm = TRUE), sum(cnt$n_pd_false, na.rm = TRUE)))
print(cnt, row.names = FALSE)

# --- figures ----------------------------------------------------------------------
pal <- grDevices::hcl.colors(64, "viridis")
noms_fig <- c(Hauteur.4 = "figS7_cube_height", Nb_talles.1 = "figS8_cube_tillers",
              nb_grains = "figS9_cube_grains", pred_azote.4 = "figS10_cube_leafN",
              pred_proteines_grain = "figS11_cube_protein",
              Biomasse_seche_totale.4 = "figS12_cube_biomass_wheat",
              Biomasse_seche.5 = "figS13_cube_biomass_alfalfa")
D <- gs$delta
for (t in unique(D$trait)) {
  d <- D[D$trait == t, ]
  lv <- sort(unique(d$reach_within)); dv <- sort(unique(d$dilution_within)); rk <- sort(unique(d$rank_within))
  nb <- length(lv) * length(dv); nr <- length(rk)
  M <- matrix(NA_real_, nb * nr, nb * nr)
  bl <- expand.grid(l = lv, dd = dv)   # portee varie le plus vite dans un bloc de dilution
  for (i in seq_len(nrow(bl))) for (j in seq_len(nrow(bl))) {
    s <- d[d$dilution_within == bl$dd[i] & d$reach_within == bl$l[i] &
           d$dilution_between == bl$dd[j] & d$reach_between == bl$l[j], ]
    if (!nrow(s)) next
    M[(i - 1) * nr + match(s$rank_within, rk), (j - 1) * nr + match(s$rank_between, rk)] <- pmin(s$delta_aic, 8)
  }
  fn <- file.path(od, paste0(if (t %in% names(noms_fig)) noms_fig[[t]] else paste0("fig_cube_", t), ".png"))
  grDevices::png(fn, width = 2000, height = 2000, res = 200)
  op <- par(mar = c(4, 4, 3, 1))
  image(seq_len(ncol(M)), seq_len(nrow(M)), t(M[nrow(M):1, ]), col = pal, zlim = c(0, 8), axes = FALSE,
        xlab = "between-group : dilution (blocks) x reach (sub-blocks) x rank (cells)",
        ylab = "within-group : dilution x reach x rank", main = sprintf("%s : dAIC (capped at 8)", t))
  abline(v = seq(0, ncol(M), by = nr) + 0.5, col = "grey70", lwd = 0.5)
  abline(h = seq(0, nrow(M), by = nr) + 0.5, col = "grey70", lwd = 0.5)
  abline(v = seq(0, ncol(M), by = nr * length(lv)) + 0.5, col = "white", lwd = 2)
  abline(h = seq(0, nrow(M), by = nr * length(lv)) + 0.5, col = "white", lwd = 2)
  axis(1, at = (seq_len(nb) - 0.5) * nr, labels = sprintf("d%g l%g", bl$dd, bl$l), cex.axis = 0.5, las = 2)
  axis(2, at = (seq_len(nb) - 0.5) * nr, labels = rev(sprintf("d%g l%g", bl$dd, bl$l)), cex.axis = 0.5, las = 1)
  par(op); grDevices::dev.off()
}
# S1 / S2 : profils minimises
profil <- function(d, cx, cy) {
  ag <- stats::aggregate(d$delta_aic, by = list(x = d[[cx]], y = d[[cy]]), FUN = min)
  xs <- sort(unique(ag$x)); ys <- sort(unique(ag$y))
  M <- matrix(NA_real_, length(xs), length(ys)); M[cbind(match(ag$x, xs), match(ag$y, ys))] <- ag$x
  M[] <- NA; M[cbind(match(ag$x, xs), match(ag$y, ys))] <- ag[[3]]
  list(x = xs, y = ys, M = M)
}
for (fig in list(list("figS1_reach.png", "reach_within", "reach_between", "reach within", "reach between"),
                 list("figS2_dilution.png", "dilution_within", "dilution_between", "dilution within", "dilution between"))) {
  traits <- unique(D$trait); k <- length(traits)
  grDevices::png(file.path(od, fig[[1]]), width = 600 * min(k, 4), height = 600 * ceiling(k / 4), res = 150)
  op <- par(mfrow = c(ceiling(k / 4), min(k, 4)), mar = c(4, 4, 3, 1))
  for (t in traits) {
    p <- profil(D[D$trait == t, ], fig[[2]], fig[[3]])
    image(seq_along(p$x), seq_along(p$y), pmin(p$M, 8), col = pal, zlim = c(0, 8), axes = FALSE,
          xlab = fig[[4]], ylab = fig[[5]], main = t)
    axis(1, at = seq_along(p$x), labels = p$x); axis(2, at = seq_along(p$y), labels = p$y)
    text(rep(seq_along(p$x), length(p$y)), rep(seq_along(p$y), each = length(p$x)),
         sprintf("%.1f", p$M), cex = 0.6, col = "white")
  }
  par(op); grDevices::dev.off()
}
# S3 : les quatre etapes de la specification
etapes <- do.call(rbind, lapply(unique(D$trait), function(t) {
  d <- D[D$trait == t, ]
  best <- function(m) if (any(m)) min(d$AIC[m]) else NA_real_
  com_r <- d$rank_within == d$rank_between; com_l <- d$reach_within == d$reach_between
  com_d <- d$dilution_within == d$dilution_between
  data.frame(trait = t,
             step1_common_all = best(com_r & com_l & com_d),
             step2_reach_decoupled = best(com_r & com_d),
             step3_dilution_decoupled = best(com_r & com_l),
             step4_reach_and_dilution = best(com_r),
             step5_ranks_decoupled = best(rep(TRUE, nrow(d))))
}))
etapes$gain_reach <- etapes$step1_common_all - etapes$step2_reach_decoupled
etapes$gain_dilution <- etapes$step1_common_all - etapes$step3_dilution_decoupled
etapes$gain_both <- etapes$step1_common_all - etapes$step4_reach_and_dilution
etapes$gain_ranks <- etapes$step4_reach_and_dilution - etapes$step5_ranks_decoupled
utils::write.csv(etapes, file.path(od, "specification_steps.csv"), row.names = FALSE)
grDevices::png(file.path(od, "figS3_specification.png"), width = 1600, height = 900, res = 150)
op <- par(mar = c(8, 4, 3, 1))
G <- t(as.matrix(etapes[, c("gain_reach", "gain_dilution", "gain_both", "gain_ranks")]))
colnames(G) <- etapes$trait
barplot(G, beside = TRUE, las = 2, ylab = "AIC gain over the previous step",
        legend.text = c("reach decoupled", "dilution decoupled", "both", "ranks decoupled"),
        args.legend = list(x = "topright", bty = "n"), main = "Specification steps read in the cube")
par(op); grDevices::dev.off()
cat("ecrit :", od, "\n")
