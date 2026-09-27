# ------------------------------------------------------------------------------
# FICHIER GENERE par rpkg/tools/sync_sources.R depuis R/remlax_scan.R.
# Ne pas editer ici : editer le script vivant, puis relancer la synchronisation.
# Les transformations appliquees sont enumerees en tete de sync_sources.R.
# ------------------------------------------------------------------------------

# ==============================================================================
# remlax_scan.R - GWAS a V figee, POSE A COTE de R/remlax.R
# ==============================================================================
# Ce fichier n'est pas source par R/remlax.R et n'en modifie rien. Il le
# SUPPOSE deja source : il appelle rx_export(), rx_pkg_dir(), rx_python_cmd()
# et rx_solver_args(), toutes existantes.
#
#   source("R/remlax.R")
#   source("R/remlax_scan.R")
#   fit  <- rx_fit(modele)
#   gwas <- rx_scan(fit, modele, marqueurs = M, carte = carte,
#                   incidences = c(dir = "gen", ind = "voisin"),
#                   tests = c("dir", "ind", "dir+ind+sim", "ind|dir+sim"))
#
# CE QUE FAIT rx_scan, ET CE QU'IL NE FAIT PAS. Il ne reajuste rien : il reecrit
# le theta DEJA estime par rx_fit dans le paquet et lance le scan avec
# --theta-in --maxiter 0, donc le solveur reassemble V et s'arrete. Les
# composantes de variance du scan sont exactement celles de l'ajustement que
# l'utilisateur a lance et regarde.
#
# POURQUOI LES INCIDENCES SONT NOMMEES PAR LEURS TERMES. Z_soi et Z_voisin
# existent deja : ce sont les incidences des termes DGE et IGE du modele
# ajuste. Les redemander a l'utilisateur ouvrirait la porte a un scan mene sur
# une incidence differente de celle du modele nul - un defaut silencieux. On
# nomme donc les TERMES, et le dispositif du scan est celui du modele par
# construction.
# ==============================================================================

# Comment atteindre `remlax.scan_cli`.
#
# ATTENTION, PIEGE : `rx_solver_args()` ne rend PAS un interpreteur, il rend
# deja les arguments du point d'entree de l'AJUSTEMENT - `c("-m",
# "remlax.cli")`, ou le chemin de `cli.py`, ou le contenu de $RX_CLI. Le
# reutiliser tel quel ici enchainerait deux modules sur la meme ligne de
# commande (`-m remlax.cli -m remlax.scan_cli`) et Python n'executerait que le
# premier : le scan ne tournerait jamais, et l'ajustement tournerait a sa place
# avec des arguments qu'il ne comprend pas. On refait donc la meme logique pour
# NOTRE point d'entree, avec sa propre variable d'environnement.
# SECOND PIEGE, mesure celui-la : lancer `scan_cli.py` PAR SON CHEMIN ne suffit
# pas. Python met alors `src/remlax/` sur sys.path, pas `src/`, donc
# `from remlax.bundle import Bundle` echoue par ModuleNotFoundError. Il faut
# soit le paquet installe, soit dire ou il est. On rend donc les arguments ET
# l'environnement, et l'appelant passe le second a system2(env=).
rx_scan_solver_args <- function() {
  if (nzchar(Sys.getenv("RX_SCAN_CLI")))
    return(list(args = strsplit(Sys.getenv("RX_SCAN_CLI"), " +")[[1]], env = character(0)))
  py <- rx_python_cmd()
  ok <- suppressWarnings(try(system2(py[1], shQuote(c(py[-1], "-c", "import remlax")),
                                     stdout = FALSE, stderr = FALSE), silent = TRUE))
  if (identical(as.integer(ok), 0L))
    return(list(args = c("-m", "remlax.scan_cli"), env = character(0)))
  # Paquet installe : scan_cli.py importe `remlax.bundle`, donc il faut le
  # repertoire qui CONTIENT remlax/ sur PYTHONPATH, soit inst/python/.
  inst <- system.file("python", package = "remlax")
  if (nzchar(inst) && file.exists(file.path(inst, "remlax", "scan_cli.py"))) {
    anc <- Sys.getenv("PYTHONPATH")
    return(list(args = c("-m", "remlax.scan_cli"),
                env = paste0("PYTHONPATH=",
                             if (nzchar(anc)) paste(inst, anc, sep = .Platform$path.sep)
                             else inst)))
  }
  base <- rx_pkg_dir()
  cands <- c(file.path(base, "..", "src"), file.path("src"))
  for (p in cands) {
    if (file.exists(file.path(p, "remlax", "scan_cli.py"))) {
      src <- normalizePath(p)
      anc <- Sys.getenv("PYTHONPATH")
      return(list(args = c("-m", "remlax.scan_cli"),
                  env = paste0("PYTHONPATH=",
                               if (nzchar(anc)) paste(src, anc, sep = .Platform$path.sep)
                               else src)))
    }
  }
  stop("remlax.scan_cli introuvable. Installer le paquet (`pip install -e .`), ",
       "ou definir $RX_SCAN_CLI, ou lancer depuis la racine du depot.",
       call. = FALSE)
}


# GWAS a V figee sur un modele deja ajuste
#
# @param fit sortie de `rx_fit` (ou `rx_reml`) : seul `fit$theta` est lu.
# @param model l'objet `rx_model` correspondant.
# @param marqueurs matrice des doses GENOTYPIQUES (q lignes x p SNP), lignes
#   nommees par les niveaux du terme, ou liste `list(dir = ., ind = .)` pour le
#   cas inter-especes (voisins d'une autre espece, donc autre jeu de SNP -
#   les tests conjoints sont alors refuses par le solveur).
#   Pour la famille `sim`, les doses doivent etre CENTREES (codage +/-1).
# @param carte data.frame optionnel a colonnes `snp`, `chr`, `pos`.
# @param incidences vecteur nomme : quels TERMES du modele portent l'incidence
#   du genotype focal (`dir`) et celle des voisins (`ind`).
# @param tests specifications, cf. docs/scan.md. La barre denote un test
#   conditionnel : `"ind|dir+sim"` teste l'effet ADDITIF du voisin sachant le
#   modele de Sato.
# @param maf filtre de frequence allelique mineure, calcule sur `marqueurs$dir`
#   selon `codage`. NULL pour ne rien filtrer.
# @param codage "pm1" (doses en -1/+1) ou "012". Sert au calcul de la MAF, et
#   c'est le seul endroit du dispositif ou le codage intervient.
# @param bloc largeur des blocs de SNP pour la famille `sim`.
# @return data.frame d'une ligne par SNP, plus l'attribut "meta" (lambda de
#   controle genomique par test, temps, -2logL du nul).
rx_scan <- function(fit, model, marqueurs, carte = NULL,
                    incidences = c(dir = "gen", ind = "voisin"),
                    tests = c("dir", "ind", "sim", "dir+ind+sim"),
                    maf = 0.05, codage = c("pm1", "012"),
                    backend = c("auto", "gpu", "cpu"), bloc = 1024L,
                    dir = NULL, verbose = TRUE, keep = FALSE) {
  backend <- match.arg(backend)
  codage  <- match.arg(codage)
  stopifnot(is.list(fit), !is.null(fit$theta), is.list(model))

  # ---- les incidences viennent des TERMES du modele ------------------------
  noms_termes <- vapply(model$terms, `[[`, "", "name")
  Z <- list()
  for (fam in names(incidences)) {
    if (!fam %in% c("dir", "ind"))
      stop("incidences : nom '", fam, "' inconnu ; attendu 'dir' et/ou 'ind'.",
           call. = FALSE)
    # Un terme peut porter PLUSIEURS incidences. C'est le cas normal du modele
    # DGE/IGE : une structure `us` 2x2 sur (direct, indirect) est UN terme a
    # t = 2, dont Zl[[1]] est l'incidence du genotype porte et Zl[[2]] celle des
    # voisins. On accepte donc `"terme:a"` pour designer la a-ieme, et on
    # l'EXIGE des que t > 1 - deviner l'indice attribuerait l'effet direct a
    # l'incidence de voisinage sans qu'aucune erreur ne le signale.
    spec <- as.character(incidences[[fam]])
    a <- 1L
    if (grepl(":", spec, fixed = TRUE)) {
      mor <- strsplit(spec, ":", fixed = TRUE)[[1]]
      spec <- mor[1]
      a <- suppressWarnings(as.integer(mor[2]))
      if (is.na(a)) stop("incidences['", fam, "'] : apres ':' il faut un entier, ",
                         "l'indice de la colonne du terme.", call. = FALSE)
    }
    k <- match(spec, noms_termes)
    if (is.na(k))
      stop("incidences['", fam, "'] = '", spec,
           "' : ce terme n'est pas dans le modele. Termes disponibles : ",
           paste(noms_termes, collapse = ", "), call. = FALSE)
    tm <- model$terms[[k]]
    if (tm$t > 1L && !grepl(":", as.character(incidences[[fam]]), fixed = TRUE))
      stop("terme '", tm$name, "' porte t = ", tm$t, " incidences. Preciser ",
           "laquelle : incidences['", fam, "'] = \"", spec, ":1\" pour la ",
           "premiere. Deviner l'indice attribuerait l'effet direct a ",
           "l'incidence de voisinage en silence.", call. = FALSE)
    if (a < 1L || a > tm$t)
      stop("incidences['", fam, "'] : indice ", a, " hors de 1..", tm$t,
           " pour le terme '", tm$name, "'.", call. = FALSE)
    Z[[fam]] <- tm$Zl[[a]]
  }
  if (!length(Z)) stop("incidences vide.", call. = FALSE)

  # ---- doses ---------------------------------------------------------------
  if (!is.list(marqueurs)) marqueurs <- list(dir = marqueurs, ind = marqueurs)
  for (fam in names(Z)) {
    if (is.null(marqueurs[[fam]]))
      stop("marqueurs : matrice absente pour la famille '", fam, "'.", call. = FALSE)
    M <- as.matrix(marqueurs[[fam]])
    if (nrow(M) != ncol(Z[[fam]]))
      stop("marqueurs[['", fam, "']] a ", nrow(M), " lignes mais l'incidence '",
           incidences[[fam]], "' a ", ncol(Z[[fam]]), " colonnes. Les doses sont ",
           "indexees par GENOTYPE, dans l'ordre des niveaux du terme.",
           call. = FALSE)
    marqueurs[[fam]] <- M
  }
  p <- ncol(marqueurs[[names(Z)[1]]])
  if (length(unique(vapply(marqueurs[names(Z)], ncol, 1L))) > 1L)
    stop("les matrices de doses n'ont pas le meme nombre de SNP.", call. = FALSE)

  # ---- filtre de frequence -------------------------------------------------
  garder <- rep(TRUE, p)
  if (!is.null(maf)) {
    Md <- marqueurs[[if ("dir" %in% names(Z)) "dir" else names(Z)[1]]]
    f <- if (codage == "pm1") (colMeans(Md) + 1) / 2 else colMeans(Md) / 2
    garder <- pmin(f, 1 - f) >= maf & apply(Md, 2, stats::sd) > 0
    if (verbose)
      message(sprintf("  rx_scan : %d SNP sur %d passent MAF >= %.3g",
                      sum(garder), p, maf))
  }

  # ---- paquet : rx_export intact, puis les tableaux du scan A COTE ---------
  garde_paquet <- FALSE
  if (is.null(dir)) {
    dir <- file.path(tempdir(), paste0("rx_scan_", as.integer(Sys.time())))
    on.exit(if (!keep && !garde_paquet) unlink(dir, recursive = TRUE), add = TRUE)
  }
  # QUATRIEME PIEGE : `tempdir()` peut etre RELATIF (c'est le cas quand $TMPDIR
  # l'est). Le chemin est alors resolu contre le repertoire courant du
  # processus PYTHON, pas celui de R, et le solveur cherche le paquet ailleurs.
  # On rend le chemin absolu avant de le passer.
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir <- normalizePath(dir, mustWork = TRUE)
  rx_export(model, dir)
  con <- file(file.path(dir, "in_theta.bin"), "wb")
  writeBin(as.double(fit$theta), con, size = 8); close(con)

  man <- list()
  put <- function(name, x, dtype = "f8") {
    if (dtype == "str") {
      writeLines(as.character(x), file.path(dir, paste0(name, ".txt")))
      man[[length(man) + 1L]] <<- list(name = name, dtype = "str",
                                       shape = I(length(x)))
      return(invisible())
    }
    shp <- if (is.matrix(x)) dim(x) else length(x)
    con <- file(file.path(dir, paste0(name, ".bin")), "wb")
    if (dtype == "f8") writeBin(as.double(as.vector(x)), con, size = 8)
    else               writeBin(as.integer(as.vector(x)), con, size = 4)
    close(con)
    man[[length(man) + 1L]] <<- list(name = name, dtype = dtype, shape = I(shp))
    invisible()
  }
  # Les doses sont-elles LES MEMES des deux cotes ? Si oui (cas intra-espece,
  # le cas courant), on ne les ecrit qu'UNE fois : 292 Mo au format du projet,
  # qu'il serait absurde de dupliquer, et l'alignement des colonnes pour les
  # tests conjoints devient certain au lieu d'etre a verifier.
  fams <- names(Z)
  partage <- length(fams) < 2 ||
    (identical(dim(marqueurs[[fams[1]]]), dim(marqueurs[[fams[2]]])) &&
       identical(marqueurs[[fams[1]]], marqueurs[[fams[2]]]))
  meta <- list(p = p)
  for (fam in fams) {
    S <- methods::as(methods::as(Z[[fam]], "CsparseMatrix"), "TsparseMatrix")
    pre <- paste0("scan_", fam)
    put(paste0(pre, "_zi"), S@i, "i4")     # 0-based, comme rx_export
    put(paste0(pre, "_zj"), S@j, "i4")
    put(paste0(pre, "_zx"), S@x)
    meta[[paste0("q_", fam)]] <- ncol(Z[[fam]])
    if (!partage) put(paste0("scan_M", fam), marqueurs[[fam]])
  }
  if (partage) put("scan_M", marqueurs[[fams[1]]])
  put("scan_garder", as.integer(garder), "i4")
  if (!is.null(carte)) {
    if (!is.null(carte$snp)) put("scan_snp", carte$snp, "str")
    if (!is.null(carte$chr)) put("scan_chr", carte$chr, "str")
    if (!is.null(carte$pos)) put("scan_pos", as.double(carte$pos))
  } else if (!is.null(colnames(marqueurs[[names(Z)[1]]]))) {
    put("scan_snp", colnames(marqueurs[[names(Z)[1]]]), "str")
  }
  writeLines(jsonlite::toJSON(list(arrays = man, meta = meta),
                              auto_unbox = TRUE),
             file.path(dir, "scan_manifest.json"))

  # ---- appel du solveur ----------------------------------------------------
  # TROISIEME PIEGE, et le plus sournois : `system2` construit une ligne de
  # commande et la fait passer par un SHELL. Or une specification de test
  # conditionnel contient une barre verticale - `"ind|dir+sim"` - que le shell
  # lit comme un TUBE : la commande est coupee en deux, la seconde moitie
  # (`dir+sim ...`) n'existe pas, et le tout rend 127 « commande introuvable ».
  # L'erreur ne dit rien du tube ; elle est indechiffrable si on ne sait pas.
  # Les valeurs partent donc protegees, y compris le chemin du paquet, qui peut
  # contenir des espaces.
  sol <- rx_scan_solver_args()
  args <- c(sol$args, shQuote(dir),
            "--tests", shQuote(paste(tests, collapse = ",")),
            "--backend", backend, "--theta-in", "--maxiter", "0",
            "--bloc", as.character(as.integer(bloc)))
  if (!verbose) args <- c(args, "--quiet")
  code <- system2(rx_python_cmd(), args, stdout = "", stderr = "", env = sol$env)
  if (!identical(as.integer(code), 0L)) {
    # On DESARME le nettoyage : le message dit ou est le paquet, il faut donc
    # qu'il y soit encore. Sans cela on annonce un chemin qu'on vient
    # d'effacer, ce qui rend le diagnostic impossible.
    garde_paquet <- TRUE
    stop("rx_scan : le solveur a rendu le code ", code,
         ". Paquet conserve pour diagnostic dans ", dir,
         "\n  relancer a la main : ", paste(c(rx_python_cmd(), args), collapse = " "),
         call. = FALSE)
  }

  # check.names = FALSE EST OBLIGATOIRE : sans lui, R remplace `+` et `|` par
  # des points, donc `p_dir+ind+sim` devient `p_dir.ind.sim` et
  # `p_sim|dir+ind` devient `p_sim.dir.ind`. Les colonnes deviennent
  # introuvables sous le nom du test qui les a produites, et un appelant qui
  # boucle sur `tests` saute alors les tests composes SANS RIEN DIRE - defaut
  # mesure : trois tests sur six absents du recapitulatif.
  out <- utils::read.csv(file.path(dir, "scan_resultats.csv"),
                         stringsAsFactors = FALSE, check.names = FALSE)
  attr(out, "meta") <- jsonlite::fromJSON(file.path(dir, "scan_meta.json"))
  out
}


# Seuil de Bonferroni sur les SNP effectivement testes
# @param res sortie de `rx_scan`
# @param test nom du test, tel que passe a `tests`
# @param alpha risque global
rx_scan_seuil <- function(res, test, alpha = 0.05) {
  col <- paste0("p_", test)
  if (is.null(res[[col]]))
    stop("colonne '", col, "' absente ; tests disponibles : ",
         paste(sub("^p_", "", grep("^p_", names(res), value = TRUE)),
               collapse = ", "), call. = FALSE)
  m <- sum(is.finite(res[[col]]))
  alpha / max(m, 1L)
}
