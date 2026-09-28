#!/usr/bin/env Rscript
# ==============================================================================
# 05b_hessian.R — le Hessien au theta enregistre, sans deplacer l'estimation
# ------------------------------------------------------------------------------
# rx_fit(theta_init = theta, maxiter = 0, polish = 0, hessian = TRUE) evalue la
# vraisemblance et le Hessien de -2 logL au point relu. Le controle : la logLik
# relue egale celle de l'ajustement a 1e-9 relatif. Ecrit hessian.csv, met a
# jour theta.csv (se_theta) et fit.rds (hessian, se_theta, diagnostics).
#
#   Rscript 05b_hessian.R --name mvC7 [--backend auto]
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(name = "mvC7", backend = "auto"))
out <- ch3_out(a$name)
m <- readRDS(file.path(out, "model.rds")); model <- m$model
fit <- readRDS(file.path(out, "fit.rds"))
h <- rx_fit(model, backend = a$backend, theta_init = fit$theta, maxiter = 0L, polish = 0L,
            floor = fit$par_floor, ceil = fit$par_ceil, hessian = TRUE, blups = FALSE, verbose = TRUE)
ecart <- abs(h$logLik - fit$logLik) / abs(fit$logLik)
cat(sprintf("logLik relue %.9f, ajustement %.9f, ecart relatif %.2e\n", h$logLik, fit$logLik, ecart))
if (ecart > 1e-9) stop("la logLik au theta relu ne retrouve pas celle de l'ajustement")
utils::write.csv(as.data.frame(h$hessian), file.path(out, "hessian.csv"), row.names = FALSE)
for (k in c("hessian", "se_theta", "n_neg_eig", "n_null_dir", "cond", "newton_decrement",
            "conv_decrement", "conv_hessien_ok", "n_par_free", "n_at_bound"))
  fit[[k]] <- h[[k]]
saveRDS(fit, file.path(out, "fit.rds"))
th <- utils::read.csv(file.path(out, "theta.csv")); th$se_theta <- fit$se_theta
utils::write.csv(th, file.path(out, "theta.csv"), row.names = FALSE)
cat(sprintf("Hessien %dx%d : %d valeur(s) propre(s) negative(s), %d direction(s) quasi nulle(s), cond %.2e\n",
            nrow(h$hessian), ncol(h$hessian), h$n_neg_eig, h$n_null_dir, h$cond))
