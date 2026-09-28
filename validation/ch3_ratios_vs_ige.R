#!/usr/bin/env Rscript
# ==============================================================================
# Validations (b), (c), (d) du chapitre 3 sur les sorties REELLES du depot IGE
# ------------------------------------------------------------------------------
#  (b) rx_exposure contre les k a K = I du cube (cube_C_Hauteur.4.csv), contre
#      06_nuisance_var.csv de mvC7 et contre les k ponderes de
#      08_variances_remlax.R (vx_expositions.csv de mvC7_geo) ;
#  (c) rx_ratios sur mvC7 (modele.rds, 06_theta_brut.csv, 06_hessian.csv) contre
#      08_variance_components_se.csv de mvC7_geo (338 quantites hors TBV) et
#      contre mvC7_tbvhomogene / tau2_par_caractere_mv.csv (TBV, tau2) ;
#  (d) rx_cor_z contre trois valeurs de tab:supp:cor.
# Skip si REMLAX_IGE_REPO est absent. Rien n'est ecrit dans le depot IGE.
# ==============================================================================
suppressPackageStartupMessages({ library(Matrix) })
ige <- Sys.getenv("REMLAX_IGE_REPO", "")
if (!nzchar(ige) || !dir.exists(file.path(ige, "output", "results", "06_models_multivariate_C"))) {
  cat("SKIP : REMLAX_IGE_REPO absent ou sans mvC7\n"); quit(status = 0)
}
racine <- normalizePath(file.path(dirname(sub("^--file=", "",
            grep("^--file=", commandArgs(), value = TRUE)[1])), ".."))
source(file.path(racine, "R", "remlax.R"))
source(file.path(racine, "R", "remlax_design.R"))
source(file.path(racine, "R", "remlax_ratios.R"))
bilan <- list()
note <- function(quoi, ecart, seuil, detail = "") {
  ok <- is.finite(ecart) && ecart <= seuil
  bilan[[length(bilan) + 1L]] <<- data.frame(controle = quoi, ecart = ecart, seuil = seuil,
                                             verdict = if (ok) "OK" else "ECHEC", detail = detail)
  cat(sprintf("  %-58s %.3e (seuil %.0e) %s %s\n", quoi, ecart, seuil, if (ok) "OK" else "ECHEC", detail))
}
dp <- file.path(ige, "data", "processed")
mv <- file.path(ige, "output", "results", "06_models_multivariate_C")
v08 <- file.path(ige, "output", "results", "08_variance")

# ------------------------------------------------------------------------------
# (b) expositions
# ------------------------------------------------------------------------------
cat("(b) rx_exposure\n")
design <- readRDS(file.path(dp, "01_imported_data.rds"))$design
design <- design[design$Bac %in% 1:12, ]
db <- readRDS(file.path(dp, "02_donnees_ble_clean.rds")); db <- db[db$Bac %in% 1:12, ]
keep <- !is.na(db$Hauteur.4) & !is.na(db$poids_graine) & !is.na(db$Date_semis)
ids_obs <- as.character(db$ID_GENERAL[keep])
cube <- utils::read.csv(file.path(ige, "figures_session", "data", "cube_C_Hauteur.4.csv"))
for (cell in list(c(ri = 1, re = 1), c(ri = 5, re = 5), c(ri = 5, re = 7))) {
  nb <- rx_neighbourhood(design[, c("Ligne", "Colonne")], design$Espece, block = design$Bac,
                         level = gsub("-", "", design$Genotype), id = design$ID_GENERAL,
                         rank = list("Ble<-Ble" = cell[["ri"]], "Ble<-Luzerne" = cell[["re"]]),
                         pairs = c("Ble<-Ble", "Ble<-Luzerne"), spacing = c(5, 5))
  Zi <- nb$level[["Ble<-Ble"]][ids_obs, ]; Ze <- nb$level[["Ble<-Luzerne"]][ids_obs, ]
  ki <- rx_exposure(NULL, Zi)$k_identity; ke <- rx_exposure(NULL, Ze)$k_identity
  tag <- sprintf("ord_Ble_Hauteur.4_li0_le0_di0_de0_ri%d_re%d", cell[["ri"]], cell[["re"]])
  r <- cube[cube$tag == tag, ]
  note(sprintf("k_identity intra, cube %s", tag), abs(ki - r$k_IGE_intra), 1e-10, sprintf("%.6f", ki))
  note(sprintf("k_identity inter, cube %s", tag), abs(ke - r$k_IGE_inter), 1e-10, sprintf("%.6f", ke))
}
# mvC7 : modele exporte, etiquettes de meta -> colnames des termes
m <- readRDS(file.path(mv, "modele.rds")); model <- m$model; meta <- m$meta
model$terms$gen_ble$colnames <- meta$lab_ble; model$terms$gen_luz$colnames <- meta$lab_luz
nuis <- utils::read.csv(file.path(mv, "06_nuisance_var.csv"))
vx <- utils::read.csv(file.path(v08, "mvC7_geo", "multivariate", "vx_expositions.csv"))
expo <- list()
for (i in seq_len(nrow(nuis))) {
  sp <- nuis$espece[i]; t <- nuis$trait[i]
  own <- if (sp == "Ble") "gen_ble" else "gen_luz"; oth <- if (sp == "Ble") "gen_luz" else "gen_ble"
  tag <- if (sp == "Ble") "ble" else "luz"
  e1 <- rx_exposure(model, direct = sprintf("%s:DGE_%s", own, t),
                    indirect = sprintf("%s:IGEintra_%s", own, t), rows = "auto")
  e2 <- rx_exposure(model, direct = sprintf("%s:DGE_%s", own, t),
                    indirect = sprintf("%s:IGEon_%s_%s", oth, tag, t), rows = "auto")
  b <- sprintf("b%02d", i)
  e3 <- rx_exposure(model, direct = sprintf("%s:DGE_%s", own, t), indirect = sprintf("iee_%s_01", b), rows = "auto")
  e4 <- rx_exposure(model, direct = sprintf("%s:DGE_%s", own, t), indirect = sprintf("iee2_%s_01", b), rows = "auto")
  expo[[i]] <- data.frame(target = t, group = sp, d = e1$d, k_within = e1$k, k_between = e2$k,
                          c = e1$c, S_within = e1$S, S_between = e2$S,
                          k_within_identity = e1$k_identity, k_between_identity = e2$k_identity,
                          k_iee_within = e3$k_identity, k_iee_between = e4$k_identity, n_rows = e1$n_rows,
                          k_other = sprintf("iee_%s_01=%.17g+iee2_%s_01=%.17g", b, e3$k_identity, b, e4$k_identity),
                          stringsAsFactors = FALSE)
  r <- vx[vx$trait == t, ]; n <- nuis[i, ]
  note(sprintf("%s d_DGE vs vx_expositions", t), abs(e1$d - r$d_DGE), 1e-10, sprintf("%.5f", e1$d))
  note(sprintf("%s k_intra pondere vs vx", t), abs(e1$k - r$k_IGE_intra_mod), 1e-10)
  note(sprintf("%s k_inter pondere vs vx", t), abs(e2$k - r$k_IGE_inter_mod), 1e-10)
  note(sprintf("%s c vs vx", t), abs(e1$c - r$c_IGE_intra_mod), 1e-10)
  note(sprintf("%s S_intra vs vx", t), abs(e1$S - r$S_IGE_intra), 1e-10)
  note(sprintf("%s k_identity intra vs 06_nuisance_var", t), abs(e1$k_identity - n$k_IGE_intra), 1e-10)
  note(sprintf("%s k_identity inter vs 06_nuisance_var", t), abs(e2$k_identity - n$k_IGE_inter), 1e-10)
  note(sprintf("%s k_IEE_intra vs vx", t), abs(e3$k_identity - r$k_IEE_intra), 1e-10)
  note(sprintf("%s k_IEE_inter vs vx", t), abs(e4$k_identity - r$k_IEE_inter), 1e-10)
}
E <- do.call(rbind, expo)
cat(sprintf("  d_DGE ble : %.4f a %.4f ; luzerne : %.4f\n", min(E$d[E$group == "Ble"]),
            max(E$d[E$group == "Ble"]), E$d[E$group == "Luzerne"]))

# ------------------------------------------------------------------------------
# (c) rx_ratios sur mvC7
# ------------------------------------------------------------------------------
cat("(c) rx_ratios sur mvC7\n")
th <- utils::read.csv(file.path(mv, "06_theta_brut.csv"))$theta
H <- as.matrix(utils::read.csv(file.path(mv, "06_hessian.csv"))); dimnames(H) <- NULL
tse <- utils::read.csv(file.path(mv, "06_theta_se.csv"))
fit <- structure(list(theta = th, hessian = H, par_floor = -12, par_ceil = 12,
                      se_theta = tse$se_theta, model = model), class = "rx_fit")
# controle de se_theta rx_read_result contre 06_theta_se.csv
se_r <- rx_se_theta(th, H, -12, 12)
ok <- is.finite(se_r) & is.finite(tse$se_theta)
note("se_theta rx_se_theta vs 06_theta_se.csv (max rel)", max(abs(se_r[ok] / tse$se_theta[ok] - 1)), 1e-6,
     sprintf("%d/%d finies", sum(ok), length(th)))
# residuelle : sections Ble (us 6x6, traits 1..6 dans l'ordre des blocs) et Luzerne (iid)
tr_ble <- nuis$trait[nuis$espece == "Ble"]
comps <- list()
for (i in seq_len(nrow(nuis))) {
  sp <- nuis$espece[i]; t <- nuis$trait[i]; b <- sprintf("b%02d", i)
  own <- if (sp == "Ble") "gen_ble" else "gen_luz"; oth <- if (sp == "Ble") "gen_luz" else "gen_ble"
  tag <- if (sp == "Ble") "ble" else "luz"
  resid <- if (sp == "Ble") sprintf("residual:Ble[%d]", match(t, tr_ble)) else "residual:Luzerne"
  comps[[i]] <- data.frame(target = t, group = sp, direct = sprintf("%s:DGE_%s", own, t),
                           indirect_within = sprintf("%s:IGEintra_%s", own, t),
                           indirect_between = sprintf("%s:IGEon_%s_%s", oth, tag, t),
                           other = paste(c(sprintf("spat_%s_%02d", b, 1:5), sprintf("iee_%s_01", b),
                                           sprintf("iee2_%s_01", b), resid), collapse = "+"),
                           stringsAsFactors = FALSE)
}
comps <- do.call(rbind, comps)
t0 <- Sys.time()
rat <- rx_ratios(fit, comps, exposure = E, model = model)
cat(sprintf("  rx_ratios : %d quantites en %.1f s ; controle se_theta %.8f\n", nrow(rat),
            as.numeric(difftime(Sys.time(), t0, units = "secs")), attr(rat, "check_se")))
note("mediane se_theta(rx_ratios) / 06_theta_se.csv", abs(attr(rat, "check_se") - 1), 1e-3)
ref <- utils::read.csv(file.path(v08, "mvC7_geo", "multivariate", "08_variance_components_se.csv"))
ref_t <- utils::read.csv(file.path(v08, "mvC7_tbvhomogene", "multivariate", "08_variance_components_se.csv"))
# correspondance des noms 08 -> rx_ratios
map_comp <- c(DGE = "direct", IGE_intra = "indirect_within", IGE_inter = "indirect_between",
              cov_DGE_IGE = "cov_direct_indirect", Residual = "RESID")
spat_lab <- c(Bac = 1, Ligne = 2, Colonne = 3, Bordure = 4, Orientation = 5)
mine <- function(q, target, comp) {
  r <- rat[rat$quantity == q & rat$target == target & rat$component == comp, ]
  if (nrow(r) != 1L) return(c(NA_real_, NA_real_)); c(r$estimate, r$se)
}
comp_de <- function(comp, i, sp, t) {
  b <- sprintf("b%02d", i)
  if (comp %in% names(map_comp)) {
    if (comp == "Residual") return(if (sp == "Ble") sprintf("residual:Ble[%d]", match(t, tr_ble)) else "residual:Luzerne")
    return(map_comp[[comp]])
  }
  if (comp %in% names(spat_lab)) return(sprintf("spat_%s_%02d", b, spat_lab[[comp]]))
  if (comp == "IEE_intra") return(sprintf("iee_%s_01", b))
  if (comp == "IEE_inter") return(sprintf("iee2_%s_01", b))
  NA_character_
}
d_est <- c(); d_se <- c(); n_cmp <- 0L; manq <- c()
for (r in seq_len(nrow(ref))) {
  x <- ref[r, ]; q <- x$quantite
  if (!q %in% c("var", "prop", "h2")) next
  if (x$composante %in% c("AR1", "Spatial")) next   # sommes ou composantes absentes du modele
  i <- which(nuis$trait == x$trait); if (!length(i)) { manq <- c(manq, x$trait); next }
  sp <- nuis$espece[i]
  if (q == "h2") {
    cm <- c(h2_DGE = "h2", h2_IGE_intra = "h2_indirect_within", h2_IGE_inter = "h2_indirect_between",
            h2_total = "h2_ext_within")[[x$composante]]
    v <- mine("h2", x$trait, cm)
  } else {
    cm <- comp_de(x$composante, i, sp, x$trait)
    v <- mine(if (q == "var") "var" else "share", x$trait, cm)
  }
  if (anyNA(v)) { manq <- c(manq, paste(q, x$trait, x$composante)); next }
  n_cmp <- n_cmp + 1L
  d_est <- c(d_est, abs(v[1] - x$estimate)); d_se <- c(d_se, abs(v[2] / x$se - 1))
}
note(sprintf("var/prop/h2 (%d quantites) : estimation, ecart abs max", n_cmp), max(d_est), 1e-8)
note(sprintf("var/prop/h2 (%d quantites) : SE, ecart relatif max", n_cmp), max(d_se, na.rm = TRUE), 1e-4)
if (length(manq)) cat("  non apparie :", paste(unique(manq), collapse = "; "), "\n")
# correlations genetiques : 08 ordonne les paires (col, row) du triangle superieur
cor_cmp <- function(q_ref, q_mine, sp_ref, term, lab) {
  rr <- ref[ref$quantite == q_ref & ref$espece == sp_ref, ]
  t <- length(lab); ij <- which(upper.tri(matrix(0, t, t)), arr.ind = TRUE)
  ij <- ij[order(ij[, "col"], ij[, "row"]), , drop = FALSE]
  if (nrow(rr) != nrow(ij)) stop("nombre de correlations inattendu pour ", sp_ref)
  de <- c(); ds <- c(); dl <- c(); dh <- c(); nne <- 0L
  for (k in seq_len(nrow(ij))) {
    a <- ij[k, "row"]; b <- ij[k, "col"]
    mi <- rat[rat$quantity == q_mine & rat$target == term & rat$component == paste0(lab[a], "~", lab[b]), ]
    stopifnot(nrow(mi) == 1L)
    de <- c(de, abs(mi$estimate - rr$estimate[k])); ds <- c(ds, abs(mi$se / rr$se[k] - 1))
    dl <- c(dl, abs(mi$ci_low - rr$ic_bas[k])); dh <- c(dh, abs(mi$ci_high - rr$ic_haut[k]))
    nne <- nne + ((mi$flag == "NOT_ESTIMABLE") != (rr$flag[k] == "NON_ESTIMABLE"))
  }
  list(est = max(de), se = max(ds, na.rm = TRUE), ic = max(c(dl, dh), na.rm = TRUE), flags = nne, n = nrow(ij))
}
lab_res <- as.character(seq_along(tr_ble))
for (z in list(list("cor", "cor", "Wheat", "gen_ble", meta$lab_ble),
               list("cor", "cor", "Alfalfa", "gen_luz", meta$lab_luz),
               list("rescor", "rescor", "Wheat", "residual:Ble", lab_res))) {
  cc <- cor_cmp(z[[1]], z[[2]], z[[3]], z[[4]], z[[5]])
  note(sprintf("%s %s (%d) : estimation", z[[1]], z[[3]], cc$n), cc$est, 1e-8)
  note(sprintf("%s %s (%d) : SE relatif", z[[1]], z[[3]], cc$n), cc$se, 1e-4)
  # 08 multiplie se_z par 1,96 ; rx_cor_z par qnorm(0.975) = 1,959964. L'ecart
  # relatif de 1,8e-5 sur le quantile se lit tel quel sur les bornes.
  note(sprintf("%s %s (%d) : IC de Fisher (1,96 contre qnorm)", z[[1]], z[[3]], cc$n), cc$ic, 1e-4)
  note(sprintf("%s %s (%d) : drapeaux NOT_ESTIMABLE discordants", z[[1]], z[[3]], cc$n), cc$flags, 0)
}
# TBV et tau2 contre mvC7_tbvhomogene et la table du 2026-09-23
tbv_ref <- ref_t[ref_t$quantite == "TBVvar", ]
de <- c(); ds <- c()
for (k in seq_len(nrow(tbv_ref))) {
  x <- tbv_ref[k, ]
  term <- if (x$espece == "Wheat") "gen_ble" else "gen_luz"
  cm <- if (grepl("^TBV intra ", x$composante)) paste0("own:", sub("^TBV intra ", "", x$composante))
        else paste0("cross:", sub("^TBV inter on ", "", x$composante))
  mi <- rat[rat$quantity == "tbv_var" & rat$target == term & rat$component == cm, ]
  stopifnot(nrow(mi) == 1L)
  de <- c(de, abs(mi$estimate - x$estimate)); ds <- c(ds, abs(mi$se / x$se - 1))
}
note(sprintf("TBVvar (%d) vs mvC7_tbvhomogene : estimation", nrow(tbv_ref)), max(de), 1e-8)
note(sprintf("TBVvar (%d) vs mvC7_tbvhomogene : SE relatif", nrow(tbv_ref)), max(ds), 1e-4)
tau <- utils::read.csv(file.path(ige, "figures_session", "data", "tau2_par_caractere_mv.csv"))
noms_fig <- c(hauteur = "Hauteur.4", talles = "Nb_talles.1", "nb de grains" = "nb_grains",
              "azote foliaire" = "pred_azote.4", proteines = "pred_proteines_grain",
              "biomasse ble" = "Biomasse_seche_totale.4", "biomasse luzerne" = "Biomasse_seche.5")
de <- c(); ds <- c()
for (k in seq_len(nrow(tau))) {
  t <- noms_fig[[tau$trait[k]]]; sp <- nuis$espece[nuis$trait == t]
  own <- if (sp == "Ble") "gen_ble" else "gen_luz"; oth <- if (sp == "Ble") "gen_luz" else "gen_ble"
  mi <- if (tau$sens[k] == "intra") rat[rat$quantity == "tau2" & rat$target == own & rat$component == paste0("own:", t), ]
        else rat[rat$quantity == "tau2" & rat$target == oth & rat$component == paste0("cross:", t), ]
  stopifnot(nrow(mi) == 1L)
  de <- c(de, abs(mi$estimate - tau$tau2[k])); ds <- c(ds, abs(mi$se / tau$se_tau2[k] - 1))
}
note(sprintf("tau2 (%d) vs tau2_par_caractere_mv.csv : estimation", nrow(tau)), max(de), 1e-8)
note(sprintf("tau2 (%d) vs tau2_par_caractere_mv.csv : SE relatif", nrow(tau)), max(ds), 1e-4)
her <- utils::read.csv(file.path(ige, "figures_session", "data", "heritabilites_etendues_mv.csv"))
de <- c()
for (k in seq_len(nrow(her))) {
  t <- noms_fig[[her[[1]][k]]]
  g <- function(cm) rat$estimate[rat$quantity == "h2" & rat$target == t & rat$component == cm]
  s <- function(cm) rat$se[rat$quantity == "h2" & rat$target == t & rat$component == cm]
  de <- c(de, abs(g("h2") - her$h2[k]), abs(s("h2") - her$se_h2[k]), abs(g("h2_ext_within") - her$h2_intra[k]),
          abs(g("h2_indirect_between") - her$part_part[k]))
}
note("heritabilites_etendues_mv.csv (h2, se, h2_ext_within, h2_between)", max(de), 1e-6)

# ------------------------------------------------------------------------------
# (d) rx_cor_z contre tab:supp:cor
# ------------------------------------------------------------------------------
cat("(d) rx_cor_z\n")
cz <- rx_cor_z(c(-0.937, 0.979, -0.153), se = c(0.076, 0.022, 1.093))
note("tab:supp:cor r=-0.937 se=0.076 -> [-0.99, -0.46] (arrondi 0.01)",
     max(abs(c(cz$ci_low[1], cz$ci_high[1]) - c(-0.99, -0.46))), 0.01,
     sprintf("[%.3f, %.3f]", cz$ci_low[1], cz$ci_high[1]))
note("tab:supp:cor r=0.979 se=0.022 -> [0.85, 1.00]",
     max(abs(c(cz$ci_low[2], cz$ci_high[2]) - c(0.85, 1.00))), 0.01,
     sprintf("[%.3f, %.3f]", cz$ci_low[2], cz$ci_high[2]))
note("tab:supp:cor r=-0.153 se=1.093 -> largeur > 1.5 (n.e.)",
     as.numeric(cz$informative[3]), 0, sprintf("largeur %.2f", cz$width[3]))

tab <- do.call(rbind, bilan)
out <- file.path(racine, "validation", "results", "ch3_ratios_vs_ige.csv")
utils::write.csv(tab, out, row.names = FALSE)
utils::write.csv(as.data.frame(rat), file.path(racine, "validation", "results", "ch3_mvC7_rx_ratios.csv"), row.names = FALSE)
cat(sprintf("\n%d controles, %d OK, %d ECHEC -> %s\n", nrow(tab), sum(tab$verdict == "OK"),
            sum(tab$verdict != "OK"), out))
if (any(tab$verdict != "OK")) quit(status = 1)
