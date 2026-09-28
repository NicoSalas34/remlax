# ==============================================================================
# _common.R — fonctions partagees par les scripts reproduction/chapitre3/
# ------------------------------------------------------------------------------
# Rien ici ne vient du depot IGE_analysis_2024-2025 : les donnees sont lues dans
# REMLAX_IGE_DATA (son data/processed), le reste est remlax et du R de base.
#
# Variables d'environnement :
#   REMLAX_IGE_DATA  chemin de data/processed (obligatoire pour 00)
#   REMLAX_CH3_OUT   repertoire des sorties (defaut : <ce dossier>/output)
#   REMLAX_R         dossier R/ du depot remlax si le paquet n'est pas installe
#   RX_PY            interpreteur Python portant jax (voir docs/api-r.md, sect. 13)
#
# CONVENTIONS DU DISPOSITIF, reprises du chapitre et verifiees dans le guide :
#   - bacs 1 a 12 (le melange) ; grille 16 lignes x 20 colonnes par bac ;
#     pas de 5 cm ; luzerne en colonnes paires ;
#   - lignes du modele triees par (bac, colonne, ligne), observations completes
#     du caractere ET des covariables ; y centre-reduit ;
#   - effets fixes : moyenne ; pour le ble poids de graine et date de semis
#     (si plusieurs dates) ; rien de plus pour la luzerne ;
#   - termes, dans cet ordre : gen_ble (us), gen_luz (us), cinq termes spatiaux
#     iid par bloc (bac, bac:ligne, bac:colonne, bac:bordure, bac:orientation),
#     un IEE conspecifique et un IEE heterospecifique iid par bloc ;
#     residuelle en sections par espece (us entre caracteres, iid si un seul) ;
#   - parente : GRM de data/processed, K <- 0,98 K + 0,02 I, restreinte aux
#     genotypes presents, noms sans tiret.
# ==============================================================================

ch3_load_remlax <- function() {
  if (requireNamespace("remlax", quietly = TRUE)) {
    suppressPackageStartupMessages(library(remlax))
    return(invisible("package"))
  }
  rdir <- Sys.getenv("REMLAX_R", "")
  if (!nzchar(rdir)) {
    # le depot lui-meme, si ce script en fait partie
    cand <- normalizePath(file.path(ch3_here(), "..", "..", "R"), mustWork = FALSE)
    if (file.exists(file.path(cand, "remlax.R"))) rdir <- cand
  }
  if (!nzchar(rdir) || !file.exists(file.path(rdir, "remlax.R")))
    stop("remlax n'est ni installe ni trouve : installer rpkg/ ou definir REMLAX_R=<depot>/R.")
  suppressPackageStartupMessages(library(Matrix))
  for (f in c("remlax.R", "remlax_design.R", "remlax_ratios.R"))
    source(file.path(rdir, f), local = globalenv())
  invisible("source")
}

ch3_here <- function() {
  f <- grep("^--file=", commandArgs(), value = TRUE)
  if (length(f)) return(dirname(normalizePath(sub("^--file=", "", f[1]))))
  if (!is.null(sys.frames()) && length(sys.frames()))
    for (i in rev(seq_along(sys.frames()))) {
      of <- sys.frame(i)$ofile
      if (!is.null(of)) return(dirname(normalizePath(of)))
    }
  getwd()
}

ch3_out <- function(...) {
  o <- Sys.getenv("REMLAX_CH3_OUT", file.path(ch3_here(), "output"))
  dir.create(o, recursive = TRUE, showWarnings = FALSE)
  file.path(o, ...)
}

# Arguments --cle valeur d'une ligne de commande, avec defauts nommes.
ch3_args <- function(defauts) {
  a <- commandArgs(trailingOnly = TRUE)
  out <- defauts
  i <- 1L
  while (i <= length(a)) {
    if (!startsWith(a[i], "--")) stop("argument inattendu : ", a[i])
    k <- sub("^--", "", a[i])
    if (!k %in% names(defauts)) stop("option inconnue : --", k, " (connues : ",
                                     paste(names(defauts), collapse = ", "), ")")
    if (is.logical(defauts[[k]])) { out[[k]] <- TRUE; i <- i + 1L; next }
    if (i == length(a)) stop("--", k, " sans valeur")
    v <- a[i + 1L]
    out[[k]] <- if (is.numeric(defauts[[k]])) as.numeric(v) else v
    i <- i + 2L
  }
  out
}

# ------------------------------------------------------------------------------
# Donnees
# ------------------------------------------------------------------------------
ch3_read_data <- function(data_dir = Sys.getenv("REMLAX_IGE_DATA", ""), blocks = 1:12) {
  if (!nzchar(data_dir) || !dir.exists(data_dir))
    stop("REMLAX_IGE_DATA doit pointer sur data/processed du depot IGE.")
  f <- function(x) { p <- file.path(data_dir, x); if (!file.exists(p)) stop("introuvable : ", p); p }
  design <- readRDS(f("01_imported_data.rds"))$design
  design <- design[design$Bac %in% blocks, ]
  design$Genotype <- gsub("-", "", as.character(design$Genotype))
  design$id <- as.character(design$ID_GENERAL)
  tables <- list()
  for (sp in c("Ble", "Luzerne")) {
    d <- readRDS(f(if (sp == "Ble") "02_donnees_ble_clean.rds" else "02_donnees_luz_clean.rds"))
    d <- d[d$Bac %in% blocks, ]
    # les colonnes de voisinage d'une ancienne geometrie sont retirees : elles
    # portent les noms des genotypes et seraient prises pour des covariables
    d <- d[, !grepl("^[BL][0-9]+$", names(d)), drop = FALSE]
    d <- droplevels(d)
    d$Genotype <- gsub("-", "", as.character(d$Genotype))
    d$id <- as.character(d$ID_GENERAL)
    d$file_order <- seq_len(nrow(d))
    tables[[sp]] <- d
  }
  K <- list()
  for (sp in c("Ble", "Luzerne")) {
    G <- as.matrix(readRDS(f(if (sp == "Ble") "02_GK_matrix.rds" else "02_GK_luz_matrix.rds")))
    G <- G[order(rownames(G)), order(colnames(G)), drop = FALSE]
    dimnames(G) <- lapply(dimnames(G), function(x) gsub("-", "", x))
    # (1 - 0.02) et non 0.98 : la meme arithmetique que la reference, au bit pres
    Gb <- (1 - 0.02) * G + 0.02 * diag(nrow(G)); dimnames(Gb) <- dimnames(G)
    K[[sp]] <- Gb
  }
  glev <- lapply(tables, function(d) sort(unique(d$Genotype)))
  for (sp in names(glev)) {
    manque <- setdiff(glev[[sp]], rownames(K[[sp]]))
    if (length(manque)) stop(sp, " : ", length(manque), " genotype(s) absents de la GRM : ",
                             paste(head(manque, 3), collapse = ", "))
  }
  list(design = design, tables = tables, K = K, glev = glev, blocks = blocks)
}

# ------------------------------------------------------------------------------
# Geometrie et incidences
# ------------------------------------------------------------------------------
# geom : liste ou ligne de data.frame avec rank_within, rank_between,
# reach_within, reach_between, dilution_within, dilution_between ; species est
# le groupe recevant (les quatre paires sont construites pour cette espece).
ch3_neighbourhood <- function(data, geom, species) {
  other <- setdiff(c("Ble", "Luzerne"), species)
  pw <- paste0(species, "<-", species); pb <- paste0(species, "<-", other)
  d <- data$design
  rx_neighbourhood(coord = d[, c("Ligne", "Colonne")], group = d$Espece, block = d$Bac,
                   level = d$Genotype, id = d$id,
                   rank = stats::setNames(list(geom$rank_within, geom$rank_between), c(pw, pb)),
                   reach = stats::setNames(list(geom$reach_within, geom$reach_between), c(pw, pb)),
                   dilution = stats::setNames(list(geom$dilution_within, geom$dilution_between), c(pw, pb)),
                   spacing = c(5, 5), pairs = c(pw, pb), output = "both", sparse = TRUE)
}

CH3_SPATIAL <- c("Bac_f", "Bac_f:Ligne_f", "Bac_f:Colonne_f", "Bac_f:bordure", "Bac_f:orientation_bordure")

# Un bloc = un caractere d'une espece : lignes, y, X, incidences.
ch3_block <- function(data, species, trait, nb, min_obs = 50L, scale_y = TRUE) {
  other <- setdiff(c("Ble", "Luzerne"), species)
  d0 <- data$tables[[species]]
  if (!trait %in% names(d0)) stop(species, " : caractere '", trait, "' absent des donnees.")
  # ordre des lignes : bac, colonne, ligne (niveaux des facteurs)
  ord <- order(as.integer(d0$Bac_f), as.integer(d0$Colonne_f), as.integer(d0$Ligne_f))
  d0 <- d0[ord, ]
  # effets fixes
  keep <- !is.na(d0[[trait]])
  extra <- character(0)
  if (species == "Ble") {
    if (any(!is.na(d0$poids_graine[keep]))) extra <- c(extra, "poids_graine")
    if (length(unique(stats::na.omit(d0$Date_semis[keep]))) > 1L) extra <- c(extra, "Date_semis")
  }
  for (cv in extra) keep <- keep & !is.na(d0[[cv]])
  if (sum(keep) < min_obs) return(NULL)
  row_idx <- which(keep)
  d <- droplevels(d0[keep, , drop = FALSE])
  y <- as.numeric(d[[trait]]); if (scale_y) y <- as.numeric(scale(y))
  rhs <- if (length(extra)) stats::as.formula(paste("~", paste(extra, collapse = " + "))) else ~ 1
  X <- stats::model.matrix(rhs, d)
  X <- X[, apply(X, 2, function(cc) any(cc != 0)), drop = FALSE]
  qrX <- qr(X); if (qrX$rank < ncol(X)) X <- X[, qrX$pivot[seq_len(qrX$rank)], drop = FALSE]
  storage.mode(X) <- "double"
  # DGE : indicatrice du genotype focal sur l'univers de l'espece
  glev <- data$glev[[species]]
  gi <- match(d$Genotype, glev)
  if (anyNA(gi)) stop("genotype(s) hors univers : ", paste(unique(d$Genotype[is.na(gi)]), collapse = ", "))
  Zd <- Matrix::sparseMatrix(i = seq_len(nrow(d)), j = gi, x = 1, dims = c(nrow(d), length(glev)),
                             dimnames = list(NULL, glev))
  # IGE : incidences par genotype restreintes aux lignes, colonnes sur l'univers
  align <- function(M, lev) {
    M <- M[d$id, , drop = FALSE]
    out <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                dims = c(nrow(d), length(lev)), dimnames = list(NULL, lev))
    com <- intersect(colnames(M), lev)
    Mc <- methods::as(methods::as(M[, com, drop = FALSE], "generalMatrix"), "CsparseMatrix")
    out[, match(com, lev)] <- Mc
    methods::as(out, "CsparseMatrix")
  }
  Zw <- align(nb$level[[paste0(species, "<-", species)]], glev)
  Zb <- align(nb$level[[paste0(species, "<-", other)]], data$glev[[other]])
  # IEE : incidences par plante, colonnes = plantes emettrices ; l'ordre des
  # colonnes suit la table de l'espece (conspecifiques) ou le tri des identifiants
  # (heterospecifiques), les colonnes vides sont elaguees
  iee <- function(U, cols) {
    M <- U[d$id, cols, drop = FALSE]
    M <- methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    cs <- Matrix::colSums(M != 0); M <- M[, cs > 0, drop = FALSE]
    if (!ncol(M)) NULL else M
  }
  Uw <- nb$unit[[paste0(species, "<-", species)]]; Ub <- nb$unit[[paste0(species, "<-", other)]]
  # conspecifiques : identifiants en ordre NUMERIQUE ; heterospecifiques : ordre
  # alphabetique des identifiants. C'est l'ordre du pipeline de reference ; il ne
  # change pas la vraisemblance (variance iid) mais il fixe l'ordre des colonnes
  # dans le paquet ecrit pour le solveur.
  ord_num <- function(x) { v <- suppressWarnings(as.numeric(x)); if (anyNA(v)) order(x) else order(v) }
  ids_own <- colnames(Uw)
  Ziee  <- iee(Uw, ids_own[ord_num(ids_own)])
  Ziee2 <- iee(Ub, colnames(Ub)[order(colnames(Ub))])
  # spatial : une incidence creuse par effet, colonnes vides elaguees
  spat <- list()
  for (st in CH3_SPATIAL) {
    Zs <- tryCatch(Matrix::sparse.model.matrix(stats::as.formula(paste0("~ ", st, " - 1")), d),
                   error = function(e) NULL)
    if (is.null(Zs)) next
    cs <- Matrix::colSums(Zs != 0); Zs <- Zs[, cs > 0, drop = FALSE]
    if (ncol(Zs)) spat[[st]] <- methods::as(Zs, "CsparseMatrix")
  }
  list(species = species, trait = trait, other = other, n = nrow(d), y = y, X = X,
       ids = d$id, row_idx = row_idx, Zd = Zd, Zw = Zw, Zb = Zb, Ziee = Ziee, Ziee2 = Ziee2,
       spat = spat, extra = extra)
}

# Empile des blocs en un rx_model ; rend list(model, meta).
ch3_stack <- function(blocks, data, struct = "us", residual = "us") {
  blocks <- Filter(Negate(is.null), blocks)
  B <- length(blocks); if (!B) stop("aucun bloc")
  n_b <- vapply(blocks, function(b) b$n, integer(1)); off <- c(0L, cumsum(n_b)); n <- sum(n_b)
  y <- unlist(lapply(blocks, `[[`, "y"), use.names = FALSE)
  X <- as.matrix(Matrix::bdiag(lapply(blocks, function(b) as.matrix(b$X))))
  colnames(X) <- unlist(lapply(blocks, function(b) paste(b$trait, colnames(b$X), sep = ":")))
  esp <- vapply(blocks, `[[`, "", "species"); tr <- vapply(blocks, `[[`, "", "trait")
  place <- function(M, t) {
    T_ <- methods::as(methods::as(M, "generalMatrix"), "TsparseMatrix")
    Matrix::sparseMatrix(i = off[t] + T_@i + 1L, j = T_@j + 1L, x = T_@x, dims = c(n, ncol(M)),
                         dimnames = list(NULL, colnames(M)))
  }
  terms <- list(); labels <- list()
  for (sp in c("Ble", "Luzerne")) {
    own <- which(esp == sp); oth <- which(esp != sp)
    if (!length(own) && !length(oth)) next
    lab <- c(if (length(own)) paste0("DGE_", tr[own]), if (length(own)) paste0("IGEintra_", tr[own]),
             if (length(oth)) paste0("IGEon_", if (sp == "Ble") "luz" else "ble", "_", tr[oth]))
    glev <- data$glev[[sp]]
    Zl <- stats::setNames(vector("list", length(lab)), lab)
    for (t in own) {
      Zl[[paste0("DGE_", tr[t])]] <- place(blocks[[t]]$Zd, t)
      Zl[[paste0("IGEintra_", tr[t])]] <- place(blocks[[t]]$Zw, t)
    }
    for (t in oth) Zl[[paste0("IGEon_", if (sp == "Ble") "luz" else "ble", "_", tr[t])]] <- place(blocks[[t]]$Zb, t)
    nm <- if (sp == "Ble") "gen_ble" else "gen_luz"
    terms[[nm]] <- rx_term(nm, Zl, K = data$K[[sp]][glev, glev], struct = struct, colnames = lab)
    labels[[sp]] <- lab
  }
  for (t in seq_len(B)) for (s in seq_along(blocks[[t]]$spat)) {
    cle <- sprintf("spat_b%02d_%02d", t, s)
    terms[[cle]] <- rx_term(cle, list(place(blocks[[t]]$spat[[s]], t)), struct = "iid")
  }
  for (t in seq_len(B)) if (!is.null(blocks[[t]]$Ziee)) {
    cle <- sprintf("iee_b%02d_01", t); terms[[cle]] <- rx_term(cle, list(place(blocks[[t]]$Ziee, t)), struct = "iid")
  }
  for (t in seq_len(B)) if (!is.null(blocks[[t]]$Ziee2)) {
    cle <- sprintf("iee2_b%02d_01", t); terms[[cle]] <- rx_term(cle, list(place(blocks[[t]]$Ziee2, t)), struct = "iid")
  }
  # residuelle : une section par espece, us entre caracteres (unite = plante)
  sections <- list()
  for (sp in unique(esp)) {
    bt <- which(esp == sp)
    rows <- unlist(lapply(bt, function(t) off[t] + seq_len(n_b[t])))
    unit <- unlist(lapply(bt, function(t) blocks[[t]]$row_idx))
    if (length(bt) == 1L) {
      sections[[sp]] <- rx_residual("iid", unit = unit, rows = rows, name = sp)
    } else {
      trait_f <- factor(rep(tr[bt], n_b[bt]), levels = tr[bt])
      sections[[sp]] <- rx_residual(residual, trait = trait_f, unit = unit, rows = rows, name = sp)
    }
  }
  res <- if (length(sections) == 1L) sections[[1L]] else rx_residual(residual, sections = sections)
  model <- rx_model(y, X, terms = terms, residual = res)
  meta <- list(B = B, n = n, species = esp, traits = tr, offset = off, labels = labels,
               spatial = lapply(blocks, function(b) names(b$spat)),
               ids = lapply(blocks, `[[`, "ids"), extra = lapply(blocks, `[[`, "extra"))
  list(model = model, meta = meta)
}

# Les references de composantes et les expositions d'un modele empile, pour
# rx_ratios : une ligne par bloc.
ch3_components <- function(model, meta) {
  out <- list(); expo <- list()
  for (i in seq_len(meta$B)) {
    sp <- meta$species[i]; t <- meta$traits[i]; b <- sprintf("b%02d", i)
    own <- if (sp == "Ble") "gen_ble" else "gen_luz"; oth <- if (sp == "Ble") "gen_luz" else "gen_ble"
    tag <- if (sp == "Ble") "ble" else "luz"
    direct <- sprintf("%s:DGE_%s", own, t); iw <- sprintf("%s:IGEintra_%s", own, t)
    ib <- sprintf("%s:IGEon_%s_%s", oth, tag, t)
    if (is.null(model$terms[[oth]])) ib <- NA_character_
    spat <- sprintf("spat_%s_%02d", b, seq_along(meta$spatial[[i]]))
    iee <- c(if (!is.null(model$terms[[sprintf("iee_%s_01", b)]])) sprintf("iee_%s_01", b),
             if (!is.null(model$terms[[sprintf("iee2_%s_01", b)]])) sprintf("iee2_%s_01", b))
    n_sp <- sum(meta$species == sp)
    resid <- if (n_sp == 1L) paste0("residual:", sp) else sprintf("residual:%s:%s", sp, t)
    if (is.null(model$residual$sections)) resid <- "residual"
    out[[i]] <- data.frame(target = t, group = sp, direct = direct, indirect_within = iw,
                           indirect_between = ib, other = paste(c(spat, iee, resid), collapse = "+"),
                           stringsAsFactors = FALSE)
    e1 <- rx_exposure(model, direct = direct, indirect = iw, rows = "auto")
    e2 <- if (is.na(ib)) NULL else rx_exposure(model, direct = direct, indirect = ib, rows = "auto")
    ko <- vapply(iee, function(k) rx_exposure(model, direct = direct, indirect = k, rows = "auto")$k_identity, numeric(1))
    expo[[i]] <- data.frame(target = t, group = sp, d = e1$d, k_within = e1$k,
                            k_between = if (is.null(e2)) NA_real_ else e2$k, c = e1$c,
                            S_within = e1$S, S_between = if (is.null(e2)) NA_real_ else e2$S,
                            k_within_identity = e1$k_identity,
                            k_between_identity = if (is.null(e2)) NA_real_ else e2$k_identity,
                            n_eff_within = e1$n_eff, n_rows = e1$n_rows,
                            k_other = paste(sprintf("%s=%.17g", iee, ko), collapse = "+"),
                            stringsAsFactors = FALSE)
  }
  list(components = do.call(rbind, out), exposure = do.call(rbind, expo))
}

# Options d'ajustement figees du cube et des sept univaries retenus.
CH3_FIT_OPTIONS <- list(maxiter = 3000L, polish = 25L, floor = -12, ceil = 12, backend = "cpu")

# Etiquette de chaque theta : nom du terme ou de la section, dans l'ordre du solveur.
ch3_theta_labels <- function(model) {
  lab <- character(0)
  for (tm in model$terms) lab <- c(lab, rep(tm$name, tm$n_par))
  secs <- model$residual$sections
  if (is.null(secs)) secs <- list(model$residual)
  for (s in secs) {
    ts <- if (is.null(s$trait)) 1L else nlevels(factor(s$trait))
    np <- rx_n_params(s$struct, ts, if (is.null(s$rank)) 0L else s$rank) +
      rx_n_level(if (is.null(s$level)) "id" else s$level, if (is.null(s$order)) 0L else s$order, s$opts)
    lab <- c(lab, rep(if (is.null(s$name)) "residual" else paste0("residual:", s$name), np))
  }
  lab
}

# Ajuste UNE cellule (une espece, un caractere, une geometrie) et ecrit ses
# sorties dans `out`. `a` porte species, trait, rank_*, reach_*, dilution_*,
# hessian, maxiter, keep_model. Utilise par 02 (une cellule) et 03 (le cube).
ch3_fit_cell <- function(a, data, out, tag, verbose = TRUE) {
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  t0 <- Sys.time()
  nb <- ch3_neighbourhood(data, a, a$species)
  bl <- ch3_block(data, a$species, a$trait, nb)
  if (is.null(bl)) stop("moins de 50 observations pour ", a$trait)
  st <- ch3_stack(list(bl), data)
  model <- st$model; meta <- st$meta
  if (isTRUE(a$keep_model)) saveRDS(list(model = model, meta = meta, geom = a), file.path(out, "model.rds"))
  t_build <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (verbose) print(model)

  opts <- CH3_FIT_OPTIONS
  fit <- rx_fit(model, backend = opts$backend, maxiter = as.integer(a$maxiter), polish = opts$polish,
                floor = opts$floor, ceil = opts$ceil, hessian = as.logical(a$hessian), blups = TRUE,
                verbose = verbose)
  t_tot <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # --- le brut d'abord ------------------------------------------------------------
  utils::write.csv(data.frame(index = seq_along(fit$theta), term = ch3_theta_labels(model),
                              theta = fit$theta, se_theta = fit$se_theta),
                   file.path(out, "theta.csv"), row.names = FALSE)
  writeLines(c(sprintf("tag=%s", tag), sprintf("species=%s", a$species), sprintf("trait=%s", a$trait),
               sprintf("rank_within=%g", a$rank_within), sprintf("rank_between=%g", a$rank_between),
               sprintf("reach_within=%g", a$reach_within), sprintf("reach_between=%g", a$reach_between),
               sprintf("dilution_within=%g", a$dilution_within), sprintf("dilution_between=%g", a$dilution_between),
               sprintf("n=%d", model$n), sprintf("n_par=%d", model$n_par), sprintf("n_terms=%d", length(model$terms)),
               sprintf("fixed=%s", paste(colnames(model$X), collapse = ",")),
               sprintf("maxiter=%d", as.integer(a$maxiter)), sprintf("polish=%d", opts$polish),
               sprintf("floor=%g", opts$floor), sprintf("ceil=%g", opts$ceil), sprintf("backend=%s", fit$backend),
               sprintf("hessian=%s", a$hessian), sprintf("date=%s", format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))),
             file.path(out, "model_spec.txt"))
  # --- expositions et composantes -----------------------------------------------
  ce <- ch3_components(model, meta)
  E <- ce$exposure
  S <- fit$sigmas$gen_ble %||% fit$sigmas$gen_luz
  own <- if (a$species == "Ble") "gen_ble" else "gen_luz"; oth <- if (a$species == "Ble") "gen_luz" else "gen_ble"
  So <- fit$sigmas[[own]]; Sx <- fit$sigmas[[oth]]
  lab <- meta$labels[[a$species]]
  comp <- data.frame(component = c("var_direct", "var_indirect_within", "cov_direct_indirect",
                                   "var_indirect_between", "var_iee_within", "var_iee_between",
                                   paste0("var_spatial_", seq_along(meta$spatial[[1]])), "var_residual"),
                     estimate = c(So[1, 1], So[2, 2], So[1, 2], if (is.null(Sx)) NA else Sx[1, 1],
                                  if (is.null(fit$sigmas$iee_b01_01)) NA else fit$sigmas$iee_b01_01[1, 1],
                                  if (is.null(fit$sigmas$iee2_b01_01)) NA else fit$sigmas$iee2_b01_01[1, 1],
                                  vapply(seq_along(meta$spatial[[1]]), function(s) fit$sigmas[[sprintf("spat_b01_%02d", s)]][1, 1], 1),
                                  as.numeric(fit$sigma_res)[1]))
  utils::write.csv(comp, file.path(out, "sigmas.csv"), row.names = FALSE)
  utils::write.csv(E, file.path(out, "exposure.csv"), row.names = FALSE)
  pd <- if (is.null(fit$n_neg_eig)) NA else fit$n_neg_eig == 0
  summ <- data.frame(tag = tag, species = a$species, trait = a$trait,
                     rank_within = a$rank_within, rank_between = a$rank_between,
                     reach_within = a$reach_within, reach_between = a$reach_between,
                     dilution_within = a$dilution_within, dilution_between = a$dilution_between,
                     n_obs = fit$n_obs, n_par = fit$n_par, n_par_free = fit$n_par_free %||% NA,
                     logLik = fit$logLik, AIC = 2 * fit$n_par - 2 * fit$logLik,
                     AIC_eff = 2 * (fit$n_par_free %||% fit$n_par) - 2 * fit$logLik,
                     converged = isTRUE(fit$conv_decrement) && isTRUE(fit$conv_grad_rel) && isTRUE(fit$conv_hessien_ok),
                     conv_decrement = fit$conv_decrement %||% NA, conv_grad_rel = fit$conv_grad_rel %||% NA,
                     conv_hessian_ok = fit$conv_hessien_ok %||% NA,
                     newton_decrement = fit$newton_decrement %||% NA, grad_rel = fit$grad_rel %||% NA,
                     pd_hessian = pd, n_neg_eig = fit$n_neg_eig %||% NA, n_null_dir = fit$n_null_dir %||% NA,
                     cond = fit$cond %||% NA, n_at_bound = fit$n_at_bound, n_iter = fit$n_iter,
                     optim_msg = fit$scipy_message %||% NA,
                     var_direct = So[1, 1], var_indirect_within = So[2, 2], cov_direct_indirect = So[1, 2],
                     var_indirect_between = if (is.null(Sx)) NA else Sx[1, 1],
                     var_residual = as.numeric(fit$sigma_res)[1],
                     d = E$d, k_within = E$k_within, k_between = E$k_between, c = E$c,
                     S_within = E$S_within, S_between = E$S_between,
                     k_within_identity = E$k_within_identity, k_between_identity = E$k_between_identity,
                     seconds_build = t_build, seconds_total = t_tot, seconds_fit = fit$secondes %||% NA)
  utils::write.csv(summ, file.path(out, "fit_summary.csv"), row.names = FALSE)
  diag <- summ[, c("tag", "logLik", "AIC", "converged", "conv_decrement", "conv_grad_rel", "conv_hessian_ok",
                   "newton_decrement", "grad_rel", "pd_hessian", "n_neg_eig", "n_null_dir", "cond",
                   "n_at_bound", "n_iter", "optim_msg")]
  utils::write.csv(diag, file.path(out, "diagnostics.csv"), row.names = FALSE)
  fit$model <- NULL   # le modele est gros ; il se reconstruit depuis model_spec.txt
  saveRDS(fit, file.path(out, "fit.rds"))
  cat(sprintf("\n%s : logLik %.9f | AIC %.6f | %d par | %d obs | %.0f s | %s\n", tag, fit$logLik,
              summ$AIC, fit$n_par, fit$n_obs, t_tot, out))
  invisible(summ)
}

# Le tag d'une cellule, forme du cube : ord_<species>_<trait>_li<>_le<>_di<>_de<>_ri<>_re<>
ch3_tag <- function(a) {
  fmt <- function(x) sub("\\.?0+$", "", sprintf("%.4f", x))
  sprintf("ord_%s_%s_li%s_le%s_di%s_de%s_ri%d_re%d", a$species, a$trait, fmt(a$reach_within),
          fmt(a$reach_between), fmt(a$dilution_within), fmt(a$dilution_between),
          as.integer(a$rank_within), as.integer(a$rank_between))
}
