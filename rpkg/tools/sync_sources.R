# ==============================================================================
# sync_sources.R — produit les sources du paquet R installable (rpkg/) a partir
# des scripts vivants du depot (R/*.R et src/remlax/*.py).
# ------------------------------------------------------------------------------
# POURQUOI UNE COPIE TRANSFORMEE ET NON UN DEPLACEMENT. Les scripts R/remlax.R,
# R/remlax_tmb.R et R/remlax_scan.R sont charges par source() depuis les
# lanceurs du cluster (RX_REMLAX_R) et depuis IGE_analysis_2024-2025. Ils
# doivent rester utilisables tels quels. Le paquet est donc une SECONDE forme
# des memes sources, regeneree par ce script, jamais editee a la main.
#
# CE QUE LE SCRIPT CHANGE, ET RIEN D'AUTRE :
#   1. la ligne library(Matrix); library(jsonlite) disparait : un paquet
#      declare ses dependances dans DESCRIPTION et NAMESPACE ;
#   2. le bloc .RX_FILE_DIR (repertoire du fichier source(), lu dans la pile
#      d'appels) devient une constante NA : un paquet installe se localise par
#      system.file() ;
#   3. rx_here() et rx_pkg_dir() rendent le repertoire du paquet installe ;
#   4. rx_solver_args() et rx_scan_solver_args() essaient, avant l'arborescence
#      du depot, la copie du solveur Python livree dans inst/python/ ;
#   5. les commentaires roxygen (#') des scripts deviennent des commentaires
#      ordinaires : la documentation Rd du paquet vit dans R/remlax-docs.R et
#      ne doit pas etre generee deux fois pour un meme objet ;
#   6. la seconde definition de `%||%` (remlax_tmb.R) est retiree, un paquet
#      ne pouvant definir deux fois le meme objet ;
#   7. le tiret cadratin (U+2014) devient un tiret ASCII, R CMD check refusant
#      tout caractere non ASCII dans le code d'un paquet portable.
#
# CHAQUE TRANSFORMATION VERIFIE L'UNICITE DE SON MARQUEUR. Si le script vivant
# change de forme, la synchronisation s'arrete au lieu de produire un paquet
# qui differe des scripts sans le dire. C'est la meme regle que traduire.R.
#
#   Rscript rpkg/tools/sync_sources.R           # depuis la racine du depot
#   Rscript rpkg/tools/sync_sources.R --check   # verifie sans ecrire
# ==============================================================================

args  <- commandArgs(trailingOnly = TRUE)
check <- "--check" %in% args

racine <- normalizePath(file.path(dirname(sub("^--file=", "",
            grep("^--file=", commandArgs(), value = TRUE)[1])), "..", ".."))
if (!file.exists(file.path(racine, "R", "remlax.R")))
  stop("racine du depot introuvable depuis ", racine)
pkg <- file.path(racine, "rpkg")

lire <- function(f) readLines(f, warn = FALSE, encoding = "UTF-8")

# Remplace UN bloc delimite par une premiere ligne (regex) et une derniere
# ligne (regex, cherchee apres la premiere). Refuse 0 ou plusieurs occurrences.
remplacer_bloc <- function(x, debut, fin, par, nom) {
  i <- grep(debut, x)
  if (length(i) != 1L)
    stop(nom, " : marqueur de debut trouve ", length(i), " fois (attendu 1) : ", debut)
  j <- grep(fin, x)
  j <- j[j >= i]
  if (!length(j))
    stop(nom, " : marqueur de fin introuvable apres la ligne ", i, " : ", fin)
  j <- j[1]
  c(x[seq_len(i - 1L)], par, x[seq.int(j + 1L, length(x))])
}

# Insere des lignes AVANT la ligne unique qui correspond au motif.
inserer_avant <- function(x, motif, lignes, nom) {
  i <- grep(motif, x, fixed = TRUE)
  if (length(i) != 1L)
    stop(nom, " : motif trouve ", length(i), " fois (attendu 1) : ", motif)
  c(x[seq_len(i - 1L)], lignes, x[seq.int(i, length(x))])
}

# roxygen -> commentaire ordinaire
deroxygen <- function(x) sub("^(\\s*)#'", "\\1#", x)

# Le tiret cadratin (U+2014) des commentaires et d'un message d'erreur devient
# un tiret ASCII : R CMD check refuse tout caractere non ASCII dans le code
# d'un paquet portable. Le sens ne change pas.
ascii <- function(x) gsub("\u2014", "-", x, useBytes = FALSE)

entete <- function(src) c(
  "# ------------------------------------------------------------------------------",
  paste0("# FICHIER GENERE par rpkg/tools/sync_sources.R depuis ", src, "."),
  "# Ne pas editer ici : editer le script vivant, puis relancer la synchronisation.",
  "# Les transformations appliquees sont enumerees en tete de sync_sources.R.",
  "# ------------------------------------------------------------------------------",
  "")

# ---- R/remlax.R --------------------------------------------------------------
x <- lire(file.path(racine, "R", "remlax.R"))

x <- remplacer_bloc(x,
  "^suppressPackageStartupMessages\\(\\{ library\\(Matrix\\); library\\(jsonlite\\) \\}\\)$",
  "^suppressPackageStartupMessages\\(\\{ library\\(Matrix\\); library\\(jsonlite\\) \\}\\)$",
  "# (paquet : Matrix et jsonlite sont importes par NAMESPACE)",
  "remlax.R/library")

x <- remplacer_bloc(x, "^\\.RX_FILE_DIR <- local\\(\\{$", "^\\}\\)$",
  c("# (paquet : le repertoire du fichier source() n'a pas de sens ; voir rx_pkg_dir)",
    ".RX_FILE_DIR <- NA_character_"),
  "remlax.R/.RX_FILE_DIR")

x <- remplacer_bloc(x, "^rx_here <- function\\(\\)", "^rx_here <- function\\(\\)",
  "rx_here <- function() system.file(package = \"remlax\")",
  "remlax.R/rx_here")

x <- remplacer_bloc(x, "^rx_pkg_dir <- function\\(\\) \\{$", "^\\}$",
  c("rx_pkg_dir <- function() system.file(package = \"remlax\")"),
  "remlax.R/rx_pkg_dir")

x <- inserer_avant(x, "  base <- rx_pkg_dir()",
  c("  # Paquet installe : la copie du solveur livree dans inst/python/. cli.py",
    "  # place lui-meme son repertoire parent sur sys.path (cli.py l.19), donc le",
    "  # lancement par chemin suffit, sans PYTHONPATH.",
    "  inst <- system.file(\"python\", \"remlax\", \"cli.py\", package = \"remlax\")",
    "  if (nzchar(inst)) return(inst)"),
  "remlax.R/rx_solver_args")

x <- ascii(deroxygen(x))
remlax_R <- c(entete("R/remlax.R"), x)

# ---- R/remlax_tmb.R ----------------------------------------------------------
x <- lire(file.path(racine, "R", "remlax_tmb.R"))
x <- remplacer_bloc(x, "^`%\\|\\|%` <- function\\(a, b\\)", "^`%\\|\\|%` <- function\\(a, b\\)",
  "# (paquet : `%||%` est defini une fois, dans remlax.R)",
  "remlax_tmb.R/%||%")
x <- ascii(deroxygen(x))
remlax_tmb_R <- c(entete("R/remlax_tmb.R"), x)

# ---- R/remlax_scan.R ---------------------------------------------------------
x <- lire(file.path(racine, "R", "remlax_scan.R"))
x <- inserer_avant(x, "  base <- rx_pkg_dir()",
  c("  # Paquet installe : scan_cli.py importe `remlax.bundle`, donc il faut le",
    "  # repertoire qui CONTIENT remlax/ sur PYTHONPATH, soit inst/python/.",
    "  inst <- system.file(\"python\", package = \"remlax\")",
    "  if (nzchar(inst) && file.exists(file.path(inst, \"remlax\", \"scan_cli.py\"))) {",
    "    anc <- Sys.getenv(\"PYTHONPATH\")",
    "    return(list(args = c(\"-m\", \"remlax.scan_cli\"),",
    "                env = paste0(\"PYTHONPATH=\",",
    "                             if (nzchar(anc)) paste(inst, anc, sep = .Platform$path.sep)",
    "                             else inst)))",
    "  }"),
  "remlax_scan.R/rx_scan_solver_args")
x <- ascii(deroxygen(x))
remlax_scan_R <- c(entete("R/remlax_scan.R"), x)

# ---- Python : src/remlax -> inst/python/remlax --------------------------------
py_src <- file.path(racine, "src", "remlax")
py_files <- sort(list.files(py_src, pattern = "\\.py$", full.names = FALSE))
if (!length(py_files)) stop("aucun .py dans ", py_src)

# ---- ecriture ou controle ----------------------------------------------------
sorties <- list(
  "R/remlax.R"      = remlax_R,
  "R/remlax_tmb.R"  = remlax_tmb_R,
  "R/remlax_scan.R" = remlax_scan_R)

ecarts <- character(0)
for (nm in names(sorties)) {
  cible <- file.path(pkg, nm)
  if (check) {
    if (!file.exists(cible) || !identical(lire(cible), sorties[[nm]]))
      ecarts <- c(ecarts, nm)
  } else {
    dir.create(dirname(cible), recursive = TRUE, showWarnings = FALSE)
    writeLines(sorties[[nm]], cible, useBytes = TRUE)
  }
}
py_dst <- file.path(pkg, "inst", "python", "remlax")
for (f in py_files) {
  a <- file.path(py_src, f); b <- file.path(py_dst, f)
  if (check) {
    if (!file.exists(b) || !identical(lire(a), lire(b)))
      ecarts <- c(ecarts, file.path("inst/python/remlax", f))
  } else {
    dir.create(py_dst, recursive = TRUE, showWarnings = FALSE)
    file.copy(a, b, overwrite = TRUE)
  }
}
if (!check) {
  # fichiers de inst/python/remlax qui n'existent plus dans src/remlax
  orphelins <- setdiff(list.files(py_dst, pattern = "\\.py$"), py_files)
  if (length(orphelins)) file.remove(file.path(py_dst, orphelins))
  # trace de provenance
  sha <- tryCatch(system2("git", c("-C", shQuote(racine), "rev-parse", "HEAD"),
                          stdout = TRUE, stderr = FALSE), error = function(e) NA_character_)
  writeLines(c(paste0("source_git_head: ", if (length(sha)) sha[1] else NA),
               paste0("synced_on: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
               paste0("python_files: ", paste(py_files, collapse = " "))),
             file.path(pkg, "inst", "python", "PROVENANCE"))
}

if (check) {
  if (length(ecarts)) {
    cat("DESYNCHRONISE :", paste(ecarts, collapse = ", "), "\n")
    quit(status = 1L)
  }
  cat("synchronise : rpkg/ correspond aux scripts vivants\n")
} else {
  cat("ecrit :", paste(names(sorties), collapse = ", "),
      "+", length(py_files), "fichiers Python dans inst/python/remlax\n")
}
