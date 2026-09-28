#!/usr/bin/env Rscript
# ==============================================================================
# Le modele construit par reproduction/chapitre3 (00 + _common.R) contre le
# modele exporte par le pipeline IGE (modele.rds), SANS ajuster : y, X, chaque
# incidence, chaque facteur de parente, la residuelle. Deux cibles :
#   - l'univarie retenu de Hauteur.4 (output/results_uni_C/Hauteur.4/modele.rds) ;
#   - le multivarie mvC7 (output/results/06_models_multivariate_C/modele.rds).
# Skip sans REMLAX_IGE_REPO. Ecart attendu : 0 partout.
# ==============================================================================
suppressPackageStartupMessages({ library(Matrix) })
ige <- Sys.getenv("REMLAX_IGE_REPO", "")
if (!nzchar(ige)) { cat("SKIP : REMLAX_IGE_REPO absent\n"); quit(status = 0) }
racine <- normalizePath(file.path(dirname(sub("^--file=", "",
            grep("^--file=", commandArgs(), value = TRUE)[1])), ".."))
Sys.setenv(REMLAX_R = file.path(racine, "R"))
if (!nzchar(Sys.getenv("REMLAX_IGE_DATA"))) Sys.setenv(REMLAX_IGE_DATA = file.path(ige, "data", "processed"))
source(file.path(racine, "reproduction", "chapitre3", "_common.R"))
ch3_load_remlax()
data <- ch3_read_data()

comparer <- function(mine, ref, quoi) {
  e <- c()
  e["n"] <- abs(mine$n - ref$n)
  e["y"] <- max(abs(mine$y - ref$y))
  e["X"] <- if (all(dim(mine$X) == dim(ref$X))) max(abs(mine$X - ref$X)) else Inf
  for (nm in names(ref$terms)) {
    a <- mine$terms[[nm]]; b <- ref$terms[[nm]]
    if (is.null(a)) { e[paste0("term ", nm)] <- Inf; next }
    ok <- a$t == b$t && a$q == b$q && a$struct == b$struct
    if (!ok) { e[paste0("term ", nm)] <- Inf; next }
    d <- 0
    for (j in seq_len(a$t)) d <- max(d, max(abs(a$Zl[[j]] - b$Zl[[j]])))
    e[paste0("Z ", nm)] <- d
    if (!is.null(b$LK)) e[paste0("LK ", nm)] <- if (is.null(a$LK)) Inf else max(abs(a$LK - b$LK))
  }
  e["n_terms"] <- abs(length(mine$terms) - length(ref$terms))
  e["n_par"] <- abs(mine$n_par - ref$n_par)
  sa <- mine$residual$sections %||% list(mine$residual); sb <- ref$residual$sections %||% list(ref$residual)
  e["n_sections"] <- abs(length(sa) - length(sb))
  for (i in seq_along(sb)) {
    e[paste0("res rows ", sb[[i]]$name)] <- if (identical(as.integer(sa[[i]]$rows), as.integer(sb[[i]]$rows))) 0 else Inf
    e[paste0("res struct ", sb[[i]]$name)] <- if (identical(sa[[i]]$struct, sb[[i]]$struct)) 0 else Inf
    if (!is.null(sb[[i]]$trait))
      e[paste0("res trait ", sb[[i]]$name)] <- if (identical(as.integer(factor(sa[[i]]$trait)), as.integer(sb[[i]]$trait))) 0 else Inf
    e[paste0("res unit ", sb[[i]]$name)] <- if (identical(as.integer(sa[[i]]$unit), as.integer(sb[[i]]$unit))) 0 else Inf
  }
  cat(sprintf("\n%s : %d controles, ecart max %.3e\n", quoi, length(e), max(e)))
  for (k in names(e)) if (e[[k]] != 0) cat(sprintf("   %-28s %.3e\n", k, e[[k]]))
  data.frame(cible = quoi, controle = names(e), ecart = as.numeric(e))
}

# --- Hauteur.4 a la geometrie retenue -------------------------------------------
ref <- readRDS(file.path(ige, "output", "results_uni_C", "Hauteur.4", "modele.rds"))$model
g <- utils::read.csv(file.path(racine, "reproduction", "chapitre3", "geometries", "tab_geom.csv"))
gh <- g[g$trait == "Hauteur.4", ]
nb <- ch3_neighbourhood(data, gh, "Ble")
st <- ch3_stack(list(ch3_block(data, "Ble", "Hauteur.4", nb)), data)
r1 <- comparer(st$model, ref, "Hauteur.4 (results_uni_C)")

# --- mvC7 ---------------------------------------------------------------------------
ref <- readRDS(file.path(ige, "output", "results", "06_models_multivariate_C", "modele.rds"))$model
g <- utils::read.csv(file.path(racine, "reproduction", "chapitre3", "geometries", "mvC7.csv"))
blocks <- lapply(seq_len(nrow(g)), function(i) {
  nb <- ch3_neighbourhood(data, g[i, ], g$species[i])
  ch3_block(data, g$species[i], g$trait[i], nb)
})
st <- ch3_stack(blocks, data)
r2 <- comparer(st$model, ref, "mvC7 (06_models_multivariate_C)")
tab <- rbind(r1, r2)
out <- file.path(racine, "validation", "results", "ch3_model_vs_ige.csv")
utils::write.csv(tab, out, row.names = FALSE)
cat(sprintf("\n-> %s ; ecart max global %.3e\n", out, max(tab$ecart)))
if (max(tab$ecart) > 0) quit(status = 1)
