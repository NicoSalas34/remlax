#!/usr/bin/env Rscript
# ==============================================================================
# 06_ratios.R — expositions, composantes mises a l'echelle, parts, heritabilites,
# tau2 et leurs erreurs-types (fig3, tab:partition)
# ------------------------------------------------------------------------------
# Lit REMLAX_CH3_OUT/<name>/model.rds et fit.rds (05 puis 05b), ou, avec
# --theta et --hessian, un theta et un Hessien externes poses sur le modele
# construit par 05 --build_only : c'est la voie « sans reajuster » qui relit les
# estimations du chapitre.
#
#   Rscript 06_ratios.R --name mvC7 [--theta theta.csv --hessian hessian.csv]
#       [--width_max 1.5] [--level 0.95]
# Sorties dans REMLAX_CH3_OUT/<name>/ratios/ :
#   exposure.csv      d, k (K et identite), c, S, n_eff, k des IEE, par caractere
#   components.csv    les references des composantes par caractere
#   ratios.csv        la table longue de rx_ratios (450 quantites sur mvC7)
#   tab_partition.csv parts en pourcent avec SE, une ligne par composante et
#                     caractere, asterisque a |z| > 1,96 (tab:partition)
#   heritabilities.csv h2, h2_ext_within (h2_intra,ext du chapitre),
#                     h2_indirect_between (h2_inter), h2_ext_total, r_DI
#   tau2.csv          Var(TBV) et tau2 propres et croises, IC de Wald
#   fig3_heritabilities_tau2.png
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(name = "mvC7", theta = "", hessian = "", width_max = 1.5, level = 0.95))
out <- ch3_out(a$name); od <- file.path(out, "ratios"); dir.create(od, showWarnings = FALSE, recursive = TRUE)
m <- readRDS(file.path(out, "model.rds")); model <- m$model; meta <- m$meta
if (nzchar(a$theta)) {
  th <- utils::read.csv(a$theta)
  H <- if (nzchar(a$hessian)) { H <- as.matrix(utils::read.csv(a$hessian)); dimnames(H) <- NULL; H } else NULL
  fit <- structure(list(theta = th$theta, hessian = H, par_floor = CH3_FIT_OPTIONS$floor,
                        par_ceil = CH3_FIT_OPTIONS$ceil, fixed_theta = integer(0),
                        se_theta = rx_se_theta(th$theta, H, CH3_FIT_OPTIONS$floor, CH3_FIT_OPTIONS$ceil)),
                   class = "rx_fit")
  cat(sprintf("theta externe (%d) et Hessien %s\n", length(fit$theta), if (is.null(H)) "absent" else "externe"))
} else fit <- readRDS(file.path(out, "fit.rds"))
if (length(fit$theta) != model$n_par) stop("theta a ", length(fit$theta), " valeurs pour ", model$n_par)
fit$sigmas <- NULL   # la carte theta -> Sigma est recontrolee par rx_ratios contre le solveur si present
ce <- ch3_components(model, meta)
utils::write.csv(ce$exposure, file.path(od, "exposure.csv"), row.names = FALSE)
utils::write.csv(ce$components, file.path(od, "components.csv"), row.names = FALSE)
rat <- rx_ratios(fit, ce$components, exposure = ce$exposure, model = model,
                 level = a$level, width_max = a$width_max)
print(rat)
utils::write.csv(as.data.frame(rat), file.path(od, "ratios.csv"), row.names = FALSE)
utils::write.csv(data.frame(index = seq_along(fit$theta), term = ch3_theta_labels(model), theta = fit$theta,
                            se_theta = attr(rat, "se_theta")), file.path(od, "se_theta.csv"), row.names = FALSE)

# --- tab:partition : parts en pourcent -----------------------------------------
sh <- rat[rat$quantity == "share", ]
sh$part_pct <- 100 * sh$estimate; sh$se_pct <- 100 * sh$se
sh$star <- ifelse(is.finite(sh$z) & abs(sh$z) > 1.96, "*", "")
sh$label <- sprintf("%.1f (%.1f)%s", sh$part_pct, sh$se_pct, sh$star)
tp <- sh[, c("target", "component", "part_pct", "se_pct", "z", "flag", "label")]
utils::write.csv(tp, file.path(od, "tab_partition.csv"), row.names = FALSE)
wide <- stats::reshape(tp[, c("target", "component", "label")], idvar = "component", timevar = "target", direction = "wide")
utils::write.csv(wide, file.path(od, "tab_partition_wide.csv"), row.names = FALSE)
# --- heritabilites --------------------------------------------------------------
h2 <- rat[rat$quantity == "h2", c("target", "component", "estimate", "se", "ci_low", "ci_high", "flag")]
utils::write.csv(h2, file.path(od, "heritabilities.csv"), row.names = FALSE)
# --- tau2 et Var(TBV) --------------------------------------------------------------
tb <- rat[rat$quantity %in% c("tbv_var", "tau2"), ]
tb$sense <- ifelse(startsWith(tb$component, "own:"), "own", "cross")
tb$trait <- sub("^(own|cross):", "", tb$component)
tb$emitter_term <- tb$target
tau <- tb[, c("trait", "sense", "emitter_term", "quantity", "estimate", "se", "ci_low", "ci_high", "flag")]
tau$entirely_above_1 <- tau$quantity == "tau2" & tau$ci_low > 1
utils::write.csv(tau, file.path(od, "tau2.csv"), row.names = FALSE)

# --- fig3 : heritabilites et tau2 ------------------------------------------------
traits <- meta$traits
pick <- function(q, cm) { r <- rat[rat$quantity == q & rat$component == cm, ]; r[match(traits, r$target), ] }
hh <- list(h2 = pick("h2", "h2"), ext = pick("h2", "h2_ext_within"), inter = pick("h2", "h2_indirect_between"))
t_own <- tau[tau$quantity == "tau2" & tau$sense == "own", ]; t_own <- t_own[match(traits, t_own$trait), ]
t_cr <- tau[tau$quantity == "tau2" & tau$sense == "cross", ]; t_cr <- t_cr[match(traits, t_cr$trait), ]
grDevices::png(file.path(od, "fig3_heritabilities_tau2.png"), width = 2200, height = 1000, res = 200)
op <- par(mfrow = c(1, 2), mar = c(9, 4, 3, 1))
M <- rbind(hh$h2$estimate, hh$ext$estimate, hh$inter$estimate); colnames(M) <- traits
bp <- barplot(M, beside = TRUE, las = 2, col = c("#0072B2", "#56B4E9", "#E69F00"), ylim = c(0, max(M + 2 * rbind(hh$h2$se, hh$ext$se, hh$inter$se), na.rm = TRUE)),
              ylab = "share of the phenotypic variance", main = "Heritabilities (a)")
S <- rbind(hh$h2$se, hh$ext$se, hh$inter$se)
arrows(bp, M - S, bp, M + S, angle = 90, code = 3, length = 0.02)
legend("topright", bty = "n", fill = c("#0072B2", "#56B4E9", "#E69F00"),
       legend = c("h2 (direct)", "h2 ext. within (direct + within IGE)", "h2 between (IGE received)"))
M2 <- rbind(t_own$estimate, t_cr$estimate); colnames(M2) <- traits; S2 <- rbind(t_own$se, t_cr$se)
bp2 <- barplot(M2, beside = TRUE, las = 2, col = c("#009E73", "#CC79A7"), ylim = c(0, max(M2 + S2, na.rm = TRUE)),
               ylab = "Var(TBV) / Var(P)", main = "tau2 (b)")
arrows(bp2, pmax(M2 - S2, 0), bp2, M2 + S2, angle = 90, code = 3, length = 0.02)
abline(h = 1, lty = 2, col = "grey40")
legend("topleft", bty = "n", fill = c("#009E73", "#CC79A7"), legend = c("own TBV", "TBV exerted by the other species"))
par(op); grDevices::dev.off()
cat("ecrit :", od, "\n")
