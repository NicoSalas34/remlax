#!/usr/bin/env Rscript
# ==============================================================================
# 05_fit_multivariate.R — le modele multivarie (mvC7 / mvD7) : sept caracteres
# ------------------------------------------------------------------------------
# Un bloc par caractere a sa geometrie (fichier --geometry : species, trait,
# rank_within, rank_between, reach_within, reach_between, dilution_within,
# dilution_between ; geometries/mvC7.csv est celle de l'ajustement du chapitre,
# geometries/tab_geom.csv celle des sept univaries retenus, dite mvD7).
# Termes : gen_ble us (2 T_ble + T_luz colonnes : DGE, IGE intra, IGE exerce sur
# la luzerne) avec K ble ; gen_luz us (2 T_luz + T_ble) avec K luzerne ; cinq
# termes spatiaux iid et deux IEE iid par bloc ; residuelle dsum : us entre les
# caracteres du ble (meme plante), iid pour la luzerne.
#
#   Rscript 05_fit_multivariate.R --geometry geometries/mvC7.csv --name mvC7 \
#       [--hessian FALSE] [--backend auto] [--maxiter 3000] [--build_only]
#       [--theta_init <theta.csv>]
# Sorties dans REMLAX_CH3_OUT/<name>/ : model.rds (model + meta), model_spec.txt,
# theta.csv, fit_summary.csv, sigmas_<terme>.csv, fit.rds. Avec --hessian FALSE
# (conseille : 198 parametres, le Hessien par differences finies coute 2 p
# evaluations), lancer ensuite 05b_hessian.R au theta enregistre.
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(geometry = file.path(ch3_here(), "geometries", "mvC7.csv"), name = "mvC7",
                   hessian = "FALSE", backend = "auto", maxiter = 3000, polish = 25,
                   build_only = FALSE, theta_init = "", data = ""))
out <- ch3_out(a$name); dir.create(out, showWarnings = FALSE, recursive = TRUE)
data <- readRDS(if (nzchar(a$data)) a$data else ch3_out("data.rds"))
g <- utils::read.csv(a$geometry, stringsAsFactors = FALSE)
t0 <- Sys.time()
blocks <- lapply(seq_len(nrow(g)), function(i) {
  nb <- ch3_neighbourhood(data, g[i, ], g$species[i])
  ch3_block(data, g$species[i], g$trait[i], nb)
})
st <- ch3_stack(blocks, data)
model <- st$model; meta <- st$meta
print(model)
saveRDS(list(model = model, meta = meta, geometry = g), file.path(out, "model.rds"))
writeLines(c(sprintf("name=%s", a$name), sprintf("geometry=%s", normalizePath(a$geometry)),
             sprintf("n=%d", model$n), sprintf("n_par=%d", model$n_par), sprintf("n_terms=%d", length(model$terms)),
             sprintf("blocks=%s", paste(meta$traits, collapse = ",")),
             sprintf("labels_ble=%s", paste(meta$labels$Ble, collapse = ",")),
             sprintf("labels_luz=%s", paste(meta$labels$Luzerne, collapse = ",")),
             sprintf("maxiter=%d", as.integer(a$maxiter)), sprintf("polish=%d", as.integer(a$polish)),
             sprintf("floor=%g", CH3_FIT_OPTIONS$floor), sprintf("ceil=%g", CH3_FIT_OPTIONS$ceil),
             sprintf("hessian=%s", a$hessian), sprintf("backend=%s", a$backend),
             sprintf("date=%s", format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))),
           file.path(out, "model_spec.txt"))
cat(sprintf("modele construit en %.0f s -> %s\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), out))
if (isTRUE(a$build_only)) quit(status = 0)

th0 <- NULL
if (nzchar(a$theta_init)) {
  th0 <- utils::read.csv(a$theta_init)$theta
  if (length(th0) != model$n_par) stop("theta_init : ", length(th0), " valeurs pour ", model$n_par, " parametres")
}
fit <- rx_fit(model, backend = a$backend, maxiter = as.integer(a$maxiter), polish = as.integer(a$polish),
              floor = CH3_FIT_OPTIONS$floor, ceil = CH3_FIT_OPTIONS$ceil,
              hessian = as.logical(a$hessian), blups = TRUE, theta_init = th0, verbose = TRUE)
utils::write.csv(data.frame(index = seq_along(fit$theta), term = ch3_theta_labels(model),
                            theta = fit$theta, se_theta = fit$se_theta),
                 file.path(out, "theta.csv"), row.names = FALSE)
utils::write.csv(data.frame(name = a$name, n_obs = fit$n_obs, n_par = fit$n_par, n_par_free = fit$n_par_free %||% NA,
                            logLik = fit$logLik, AIC = 2 * fit$n_par - 2 * fit$logLik,
                            converged = isTRUE(fit$conv_decrement) && isTRUE(fit$conv_grad_rel) && isTRUE(fit$conv_hessien_ok),
                            conv_decrement = fit$conv_decrement %||% NA, conv_grad_rel = fit$conv_grad_rel %||% NA,
                            conv_hessian_ok = fit$conv_hessien_ok %||% NA, newton_decrement = fit$newton_decrement %||% NA,
                            grad_rel = fit$grad_rel %||% NA, n_neg_eig = fit$n_neg_eig %||% NA,
                            n_null_dir = fit$n_null_dir %||% NA, cond = fit$cond %||% NA, n_at_bound = fit$n_at_bound,
                            n_iter = fit$n_iter, optim_msg = fit$scipy_message %||% NA, backend = fit$backend,
                            seconds = fit$secondes),
                 file.path(out, "fit_summary.csv"), row.names = FALSE)
for (nm in c("gen_ble", "gen_luz")) if (!is.null(fit$sigmas[[nm]]))
  utils::write.csv(as.data.frame(fit$sigmas[[nm]]), file.path(out, paste0("sigma_", nm, ".csv")))
for (nm in names(fit$sigmas_res))
  utils::write.csv(as.data.frame(fit$sigmas_res[[nm]]), file.path(out, paste0("sigma_residual_", nm, ".csv")))
for (nm in c("gen_ble", "gen_luz")) if (!is.null(fit$blups[[nm]]))
  utils::write.csv(as.data.frame(fit$blups[[nm]]), file.path(out, paste0("blups_", nm, ".csv")))
saveRDS(fit, file.path(out, "fit.rds"))
cat(sprintf("%s : logLik %.9f | %d parametres | %d obs | %.0f s\n", a$name, fit$logLik, fit$n_par, fit$n_obs, fit$secondes))
