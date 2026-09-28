#!/usr/bin/env Rscript
# ==============================================================================
# 03_cube.R — la grille a six parametres : taches, tranches, reprise, agregation
# ------------------------------------------------------------------------------
# Trois modes.
#   --make_tasks : ecrit le fichier de taches d'un caractere, une ligne par
#     cellule du produit rangs (1..10 x 1..10) x portees ({0, 0.5, 1, 1.5, 2}^2)
#     x dilutions ({0, 0.5, 1}^2), soit 22 500 lignes, avec le tag du cube.
#       Rscript 03_cube.R --make_tasks --species Ble --trait Hauteur.4 \
#           [--ranks 1:10 --reaches 0,0.5,1,1.5,2 --dilutions 0,0.5,1] [--tasks f.csv]
#   --run : ajuste les cellules d'une TRANCHE ENTRELACEE du fichier de taches :
#     la tranche CUBE_I (1..CUBE_K) prend les lignes i telles que
#     (i - 1) %% CUBE_K == CUBE_I - 1, si bien que toutes les tranches avancent
#     dans le cube de front et qu'un arret partiel laisse un echantillon
#     representatif. Une cellule dont fit_summary.csv existe deja est sautee :
#     relancer la meme commande reprend la ou elle s'est arretee.
#       CUBE_I=3 CUBE_K=750 Rscript 03_cube.R --run --tasks f.csv [--out dir]
#       Rscript 03_cube.R --run --tasks f.csv --slice 3 --of 750   (equivalent)
#   --aggregate : rassemble les fit_summary.csv des cellules d'un fichier de
#     taches en une table cube_<species>_<trait>.csv (une ligne par cellule
#     faite) et compte les cellules manquantes.
# Les options d'ajustement sont celles de 02 (CH3_FIT_OPTIONS) ; --hessian FALSE
# rend la cellule deux fois moins chere et laisse pd_hessian a NA.
# ==============================================================================
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))), "_common.R"))
ch3_load_remlax()
a <- ch3_args(list(make_tasks = FALSE, run = FALSE, aggregate = FALSE, species = "Ble",
                   trait = "Hauteur.4", ranks = "1:10", reaches = "0,0.5,1,1.5,2",
                   dilutions = "0,0.5,1", tasks = "", out = "", slice = 0, of = 0,
                   hessian = "TRUE", maxiter = 3000, data = "", verbose = FALSE))
tasks_file <- if (nzchar(a$tasks)) a$tasks else ch3_out(sprintf("tasks_%s_%s.csv", a$species, a$trait))
out_dir <- if (nzchar(a$out)) a$out else ch3_out("cells")

if (isTRUE(a$make_tasks)) {
  rk <- eval(parse(text = a$ranks)); lv <- as.numeric(strsplit(a$reaches, ",")[[1]])
  dv <- as.numeric(strsplit(a$dilutions, ",")[[1]])
  g <- expand.grid(rank_within = rk, rank_between = rk, reach_within = lv, reach_between = lv,
                   dilution_within = dv, dilution_between = dv, KEEP.OUT.ATTRS = FALSE)
  g <- cbind(species = a$species, trait = a$trait, g, stringsAsFactors = FALSE)
  g$tag <- vapply(seq_len(nrow(g)), function(i) ch3_tag(as.list(g[i, ])), "")
  g$task <- seq_len(nrow(g))
  utils::write.csv(g, tasks_file, row.names = FALSE)
  cat(sprintf("%d taches -> %s\n", nrow(g), tasks_file))
  quit(status = 0)
}
if (!file.exists(tasks_file)) stop("fichier de taches introuvable : ", tasks_file, " (--make_tasks d'abord)")
g <- utils::read.csv(tasks_file, stringsAsFactors = FALSE)

if (isTRUE(a$run)) {
  i_sl <- if (a$slice > 0) as.integer(a$slice) else as.integer(Sys.getenv("CUBE_I", "1"))
  k_sl <- if (a$of > 0) as.integer(a$of) else as.integer(Sys.getenv("CUBE_K", "1"))
  if (is.na(i_sl) || is.na(k_sl) || i_sl < 1L || i_sl > k_sl) stop("tranche invalide : ", i_sl, "/", k_sl)
  mine <- which((seq_len(nrow(g)) - 1L) %% k_sl == i_sl - 1L)
  data <- readRDS(if (nzchar(a$data)) a$data else ch3_out("data.rds"))
  faites <- 0L; sautees <- 0L; echecs <- 0L
  cat(sprintf("tranche %d/%d : %d cellule(s) sur %d\n", i_sl, k_sl, length(mine), nrow(g)))
  for (i in mine) {
    cell <- as.list(g[i, ]); tag <- cell$tag; out <- file.path(out_dir, tag)
    if (file.exists(file.path(out, "fit_summary.csv"))) { sautees <- sautees + 1L; next }
    cell$hessian <- a$hessian; cell$maxiter <- a$maxiter; cell$keep_model <- FALSE
    t0 <- Sys.time()
    r <- try(ch3_fit_cell(cell, data, out, tag, verbose = isTRUE(a$verbose)), silent = TRUE)
    if (inherits(r, "try-error")) {
      echecs <- echecs + 1L
      writeLines(as.character(r), file.path(out, "ERREUR.txt"))
      cat(sprintf("  ECHEC %s : %s\n", tag, sub("\n.*", "", as.character(r))))
    } else faites <- faites + 1L
    cat(sprintf("  [%d/%d] %s %.0f s\n", match(i, mine), length(mine), tag,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
  cat(sprintf("tranche %d/%d terminee : %d faite(s), %d deja la, %d echec(s)\n",
              i_sl, k_sl, faites, sautees, echecs))
  quit(status = if (echecs) 1L else 0L)
}

if (isTRUE(a$aggregate)) {
  rows <- list(); manquantes <- character(0)
  for (i in seq_len(nrow(g))) {
    f <- file.path(out_dir, g$tag[i], "fit_summary.csv")
    if (!file.exists(f)) { manquantes <- c(manquantes, g$tag[i]); next }
    rows[[length(rows) + 1L]] <- utils::read.csv(f, stringsAsFactors = FALSE)
  }
  if (!length(rows)) stop("aucune cellule faite pour ", tasks_file)
  cube <- do.call(rbind, rows)
  cube <- cube[match(intersect(g$tag, cube$tag), cube$tag), ]
  fo <- ch3_out(sprintf("cube_%s_%s.csv", a$species, a$trait))
  utils::write.csv(cube, fo, row.names = FALSE)
  writeLines(manquantes, ch3_out(sprintf("cube_%s_%s_missing.txt", a$species, a$trait)))
  cat(sprintf("%d cellule(s) agregee(s), %d manquante(s) -> %s\n", nrow(cube), length(manquantes), fo))
  quit(status = 0)
}
stop("choisir --make_tasks, --run ou --aggregate")
