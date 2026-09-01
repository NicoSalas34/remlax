# ==============================================================================
# export_bundles.R — ecrit un dispositif serialise par modele, SANS RIEN AJUSTER.
#
# POURQUOI CE DECOUPAGE. Sur le cluster, R et JAX vivent dans deux conteneurs
# distincts et apptainer ne s'imbrique pas (chaine de bibliotheques manquantes :
# libsubid, libcrypt...). R ne peut donc pas appeler le solveur. Il ecrit les
# paquets, le conteneur JAX les ajuste, et la comparaison se fait sur des
# fichiers — ce qui est aussi ce qui rend la parite CPU/GPU verifiable, les deux
# backends lisant STRICTEMENT la meme entree.
#
#   Rscript scripts/tests/export_bundles.R --out=/chemin/bundles
#
# La batterie couvre TOUT le catalogue, y compris les structures ajoutees le
# 01/09/2026 (lvr, ilv, mtrn anisotrope, own, dsum).
# ==============================================================================
suppressPackageStartupMessages({ library(Matrix) })
getarg <- function(k, d) { a <- commandArgs(TRUE); i <- grep(paste0("^--", k, "="), a)
  if (length(i)) sub(paste0("^--", k, "="), "", a[i[1]]) else d }
RACINE <- getarg("racine", ".")
OUT    <- getarg("out", file.path(RACINE, "output", "tests", "bundles"))
source(file.path(RACINE, "R", "remlkit.R"))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

set.seed(20260901)
n_g <- 40; n_rep <- 4; n <- n_g * n_rep
gid <- factor(rep(sprintf("G%02d", 1:n_g), each = n_rep))
bloc <- factor(rep(1:n_rep, n_g))
# GRM plausible : produit de marqueurs centres
Mk <- matrix(rbinom(n_g * 300, 2, 0.3), n_g, 300)
Mc <- scale(Mk, scale = FALSE)
K <- tcrossprod(Mc) / mean(diag(tcrossprod(Mc)))
dimnames(K) <- list(levels(gid), levels(gid))
u  <- as.numeric(t(chol(K + diag(1e-6, n_g))) %*% rnorm(n_g))
d  <- data.frame(gid = gid, bloc = bloc, x = rnorm(n))
d$ligne <- factor(rep(1:10, length.out = n)); d$col <- factor(rep(1:16, each = 10))
d$lig_n <- as.numeric(as.character(d$ligne)); d$col_n <- as.numeric(as.character(d$col))
d$site  <- factor(ifelse(seq_len(n) <= n / 2, "A", "B"))
d$y <- 3 + 0.5 * d$x + u[as.integer(gid)] + rnorm(n, 0, 0.8)
Zx <- model.matrix(~ 0 + gid, d) * d$x       # pente par genotype, MEMES niveaux
colnames(Zx) <- levels(gid)
d$Zx <- Zx
Zx <- model.matrix(~ 0 + gid, d) * d$x       # pente par genotype, memes niveaux
colnames(Zx) <- levels(gid)
d$Zx <- Zx                                   # accessible depuis la formule

# --- jeu multi-caractere (format long) ----------------------------------------
tr <- c("t1", "t2", "t3")
dl <- do.call(rbind, lapply(tr, function(t_) transform(d, trait = t_, unite = seq_len(n))))
dl$trait <- factor(dl$trait)
Sg <- matrix(c(1, .6, .3, .6, 1.2, .4, .3, .4, .8), 3, 3)
Ug <- t(chol(Sg)) %*% matrix(rnorm(3 * n_g), 3, n_g)
dl$y <- 2 + Ug[cbind(as.integer(dl$trait), as.integer(dl$gid))] + rnorm(nrow(dl), 0, 0.7)

# --- champ spatial, une observation par cellule --------------------------------
ds <- d[!duplicated(paste(d$lig_n, d$col_n)), ]
# Une observation par (section, colonne) : ce que reclame une structure entre
# unites a l'interieur d'une section.
dsec <- d[!duplicated(paste(d$site, d$col)), ]

modeles <- list(
  iid        = list(f = y ~ 1 + x, r = ~ iid(gid), dat = d),
  vm_kin     = list(f = y ~ 1 + x, r = ~ vm(gid, K), dat = d),
  ar1        = list(f = y ~ 1, r = ~ ar1(col), dat = d),
  ar2        = list(f = y ~ 1, r = ~ ar2(col), dat = d),
  ar3        = list(f = y ~ 1, r = ~ ar3(col), dat = d),
  sar        = list(f = y ~ 1, r = ~ sar(col), dat = d),
  ma1        = list(f = y ~ 1, r = ~ ma1(col), dat = d),
  arma       = list(f = y ~ 1, r = ~ arma(col), dat = d),
  cor        = list(f = y ~ 1, r = ~ cor(col), dat = d),
  corb2      = list(f = y ~ 1, r = ~ corb(col, order = 2), dat = d),
  corg       = list(f = y ~ 1, r = ~ corg(bloc), dat = d),
  expo       = list(f = y ~ 1, r = ~ exp(col_n), dat = d),
  gau        = list(f = y ~ 1, r = ~ gau(col_n), dat = d),
  lvr        = list(f = y ~ 1, r = ~ lvr(col_n), dat = d),
  iexp       = list(f = y ~ 1, r = ~ iexp(lig_n, col_n), dat = ds),
  ieuc       = list(f = y ~ 1, r = ~ ieuc(lig_n, col_n), dat = ds),
  sph        = list(f = y ~ 1, r = ~ sph(lig_n, col_n), dat = ds),
  cir        = list(f = y ~ 1, r = ~ cir(lig_n, col_n), dat = ds),
  aexp       = list(f = y ~ 1, r = ~ aexp(lig_n, col_n), dat = ds),
  mtrn_iso   = list(f = y ~ 1, r = ~ mtrn(lig_n, col_n, phi = 3, nu = "0.8 F"), dat = ds),
  mtrn_aniso = list(f = y ~ 1, r = ~ mtrn(lig_n, col_n, phi = 3, nu = "1.5 F",
                                          delta = 1.5, alpha = 0.3), dat = ds),
  own_exp    = list(f = y ~ 1, r = ~ own(col, expr = "exp(-lag*exp(p1))", n_par = 1), dat = d),
  own_2par   = list(f = y ~ 1, r = ~ own(col, expr = "exp(-(lag/exp(p1))^(1+tanh(p2)))",
                                         n_par = 2), dat = d),
  ar1ar1_res = list(f = y ~ 1, r = ~ iid(gid), res = ~ ar1(ligne):ar1(col), dat = ds),
  us_mv      = list(f = y ~ 1, r = ~ us(gid), dat = dl, trait = "trait", unit = "unite"),
  fa2_mv     = list(f = y ~ 1, r = ~ fa(gid, rank = 2), dat = dl, trait = "trait", unit = "unite"),
  chol2_mv   = list(f = y ~ 1, r = ~ chol(gid, rank = 2), dat = dl, trait = "trait", unit = "unite"),
  ante1_mv   = list(f = y ~ 1, r = ~ ante(gid, rank = 1), dat = dl, trait = "trait", unit = "unite"),
  rr2_mv     = list(f = y ~ 1, r = ~ rr(gid, rank = 2), dat = dl, trait = "trait", unit = "unite"),
  corh_mv    = list(f = y ~ 1, r = ~ corh(gid), dat = dl, trait = "trait", unit = "unite"),
  us_res_mv  = list(f = y ~ 1, r = ~ iid(gid), res = ~ us(trait):units, dat = dl,
                    trait = "trait", unit = "unite"),
  dsum       = list(f = y ~ 1, r = ~ iid(gid), res = ~ dsum(~ units | site), dat = d),
  # dsum + ar1 : une seule observation par (section, colonne), sinon deux
  # lignes partagent une unite, leur correlation vaut 1 et R est singuliere.
  dsum_ar1   = list(f = y ~ 1, r = ~ iid(gid), res = ~ dsum(~ ar1(col) | site), dat = dsec),
  # str() : DEUX incidences sur les MEMES niveaux (effet direct et pente sur x),
  # avec une covariance libre entre elles. Les colonnes de la seconde doivent
  # porter les memes noms que les niveaux, sinon rk_term refuse — a raison.
  str_2      = list(f = y ~ 1, r = ~ str(~ gid + mm(Zx, name = "gid_x"), struct = "us"),
                    dat = d)
)

ok <- 0L; ko <- character(0)
for (nm in names(modeles)) {
  m <- modeles[[nm]]
  res <- tryCatch({
    dd <- m$dat
    tr_ <- if (!is.null(m$trait)) dd[[m$trait]] else NULL
    un_ <- if (!is.null(m$unit))  dd[[m$unit]]  else NULL
    X <- model.matrix(m$f, dd)
    asg <- attr(X, "assign")
    attr(X, "termes") <- c("(Intercept)", attr(terms(m$f), "term.labels"))[sort(unique(asg)) + 1L]
    terms_l <- .rk_parse_random(m$r, dd, trait = tr_)
    res_o <- .rk_parse_residual(m$res %||% "units", dd,
                                trait = if (!is.null(m$trait)) m$trait else NULL,
                                unit  = if (!is.null(m$unit))  m$unit  else NULL)
    mod <- rk_model(model.response(model.frame(m$f, dd)), X, terms_l, res_o, name = nm)
    rk_export(mod, file.path(OUT, nm))
    sprintf("%-12s n=%-5d p=%-2d %d parametre(s)", nm, mod$n, ncol(X), mod$n_par)
  }, error = function(e) structure(conditionMessage(e), class = "erreur"))
  if (inherits(res, "erreur")) { ko <- c(ko, sprintf("%-12s %s", nm, res)) }
  else { ok <- ok + 1L; cat("  ", res, "\n") }
}
cat(sprintf("\n%d/%d paquets ecrits dans %s\n", ok, length(modeles), OUT))
if (length(ko)) { cat("ECHECS :\n"); cat(paste0("  ", ko, collapse = "\n"), "\n"); quit(status = 1) }
