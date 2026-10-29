# Arguments that select the solver entry point, placed before the bundle
# directory on the command line: $RX_CLI when set, otherwise the launcher of the
# copy shipped with this package followed by "cli". The shipped copy is used
# even when a Python package `remlax` is importable, so that the R interface
# and the solver always come from the same version.
rx_solver_args <- function() {
  if (nzchar(Sys.getenv("RX_CLI"))) return(strsplit(Sys.getenv("RX_CLI"), " +")[[1]])
  c(.rx_launcher(), "cli")
}

# Path of inst/python/remlax_run.py. The launcher runs one module of the shipped
# `remlax` Python package, imported as a package, without searching the current
# directory of the R session.
.rx_launcher <- function() {
  f <- system.file("python", "remlax_run.py", package = "remlax")
  if (!nzchar(f))
    stop("solveur Python introuvable dans le paquet installe (inst/python). ",
         "Reinstaller remlax, ou definir RX_CLI.", call. = FALSE)
  f
}

# Commande Python : $RX_PY, sinon $IGE_JAX_CMD, sinon $IGE_JAX_SIF, sinon python3
rx_python_cmd <- function() {
  if (nzchar(Sys.getenv("RX_PY"))) {
    py <- strsplit(Sys.getenv("RX_PY"), " +")[[1]]
    # RX_PY qui designe le DOSSIER d'un venv (celui que cree rx_install_python)
    # plutot que son interpreteur : erreur frequente, et le shell ne repondait
    # que "Permission denied". On prend l'interpreteur du venv.
    if (length(py) == 1L && dir.exists(py)) {
      cand <- file.path(py, c("bin/python", "bin/python3", "Scripts/python.exe"))
      cand <- cand[file.exists(cand)]
      if (length(cand)) return(cand[1])
    }
    return(py)
  }
  if (nzchar(Sys.getenv("IGE_JAX_CMD"))) return(strsplit(Sys.getenv("IGE_JAX_CMD"), " +")[[1]])
  sif <- Sys.getenv("IGE_JAX_SIF")
  if (nzchar(sif)) return(c("apptainer", "exec", "--nv", sif, "python3"))
  "python3"
}

# Cache des verifications d'interpreteur, par commande. Un `import jax` coute
# une a deux secondes : le payer a chaque ajustement d'une grille serait
# absurde, le payer une fois par session est invisible.
.rx_python_cache <- new.env(parent = emptyenv())

# L'interpreteur choisi porte-t-il jax, numpy et scipy ?
#
# Sans ce controle, un Python sans jax donnait un traceback Python de dix
# lignes finissant par « No module named 'jax' » puis « solveur REML : echec »,
# sans jamais dire QUEL interpreteur avait ete essaye ni COMMENT en designer
# un autre. C'est arrive au premier essai du paquet installe : python3 du
# systeme, pas de jax. Le diagnostic doit nommer l'interpreteur, la variable
# RX_PY et la commande d'installation.
#
# @param py commande Python decoupee (defaut : rx_python_cmd()).
# @param quiet ne rien imprimer.
# @return liste(ok, python, versions, message), invisible si ok.
rx_python_check <- function(py = rx_python_cmd(), quiet = FALSE) {
  cle <- paste(py, collapse = " ")
  res <- .rx_python_cache[[cle]]
  if (is.null(res)) res <- .rx_python_sonder(py, cle)
  if (!quiet) {
    if (res$ok) message("remlax : ", res$versions[["executable"]], " : jax ", res$versions[["jax"]],
                        ", numpy ", res$versions[["numpy"]], ", scipy ", res$versions[["scipy"]])
    else message(res$message)
  }
  invisible(res)
}

# Lance l'interpreteur une fois et met le resultat en cache.
.rx_python_sonder <- function(py, cle) {
  code <- paste0("import sys\n",
                 "try:\n    import jax, numpy, scipy\n",
                 "except Exception as e:\n    print('MANQUE', type(e).__name__, e); sys.exit(3)\n",
                 "print('OK', jax.__version__, numpy.__version__, scipy.__version__, sys.executable)\n")
  out <- suppressWarnings(tryCatch(
    .rx_system2(py[1], shQuote(c(py[-1], "-c", code)), stdout = TRUE, stderr = TRUE),
    error = function(e) structure(conditionMessage(e), status = 127L)))
  st <- attr(out, "status") %||% 0L
  ligne <- if (length(out)) out[length(out)] else ""
  res <- list(ok = identical(as.integer(st), 0L) && startsWith(ligne, "OK "),
              python = cle, versions = NULL, message = "")
  if (res$ok) {
    v <- strsplit(ligne, " ")[[1]]
    res$versions <- c(jax = v[2], numpy = v[3], scipy = v[4], executable = v[5])
  } else {
    res$message <- paste0(
      "remlax : l'interpreteur Python \"", cle, "\" ne porte pas jax, numpy et scipy",
      if (nzchar(ligne)) paste0(" (", ligne, ")") else "", ".\n",
      "  Designer un Python qui les a : Sys.setenv(RX_PY = \"/chemin/vers/python\")\n",
      "  ou dans ~/.Renviron : RX_PY=/chemin/vers/python\n",
      "  Pour en creer un : remlax::rx_install_python() (ou pip install jax numpy scipy).")
  }
  .rx_python_cache[[cle]] <- res
  res
}

# Arret net avant tout lancement du solveur si l'interpreteur ne convient pas.
.rx_python_assurer <- function(py) {
  r <- rx_python_check(py, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  invisible(r)
}

# Creer un environnement virtuel Python pour le solveur
#
# Cree `dir` par `python -m venv`, y installe jax, numpy et scipy par pip
# (`jax[cuda12]` si cuda = TRUE), verifie l'import, et imprime la ligne
# RX_PY a poser. Ne modifie ni ~/.Renviron ni la session : l'utilisateur
# choisit ou et comment la conserver. Sans reticulate, par choix.
#
# @param dir repertoire du venv (defaut : tools::R_user_dir("remlax", "data"),
#   repertoire de donnees utilisateur prevu par R pour un paquet).
# @param cuda installer la version CUDA 12 de jax.
# @param python interpreteur de base (>= 3.10) servant a creer le venv.
# @param upgrade reinstaller si le venv existe deja.
# @return chemin de l'interpreteur cree, invisible.
rx_install_python <- function(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"), cuda = FALSE,
                              python = "python3", upgrade = FALSE, ask = interactive(),
                              bayes = FALSE) {
  # Chemin ABSOLU : une RX_PY relative se resoudrait contre le repertoire
  # courant de chaque session, donc tantot un venv, tantot rien.
  dir <- normalizePath(dir, mustWork = FALSE)
  exe <- file.path(dir, if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python")
  if (!file.exists(exe) || upgrade) {
    # Politique du CRAN : ecrire hors du repertoire temporaire, et telecharger
    # plusieurs centaines de Mo, se fait sur confirmation en session interactive.
    if (isTRUE(ask)) {
      ok <- utils::askYesNo(paste0(
        "remlax : creer un environnement Python dans ", dir,
        " et y telecharger jax, numpy et scipy depuis PyPI (plusieurs centaines de Mo) ?"))
      if (!isTRUE(ok)) stop("remlax : installation annulee.", call. = FALSE)
    }
    if (!file.exists(exe)) {
      message("remlax : creation du venv ", dir)
      st <- system2(python, c("-m", "venv", shQuote(dir)))
      if (!identical(as.integer(st), 0L))
        stop("remlax : `", python, " -m venv` a echoue (code ", st, "). Python >= 3.10 requis.",
             call. = FALSE)
    }
    paquets <- c(if (cuda) "jax[cuda12]>=0.4.30" else "jax>=0.4.30", "numpy>=1.24", "scipy>=1.10",
                 # voie bayesienne (rx_bayes) : BlackJAX (et optax), optionnel
                 if (bayes) "blackjax>=1.7")
    message("remlax : pip install ", paste(paquets, collapse = " "))
    st <- system2(exe, c("-m", "pip", "install", "--upgrade", "--quiet", shQuote(paquets)))
    if (!identical(as.integer(st), 0L))
      stop("remlax : pip install a echoue (code ", st, ").", call. = FALSE)
  }
  rm(list = ls(.rx_python_cache), envir = .rx_python_cache)
  r <- rx_python_check(exe, quiet = TRUE)
  if (!r$ok) stop(r$message, call. = FALSE)
  message("remlax : interpreteur pret : ", exe, "\n",
          "  jax ", r$versions[["jax"]], ", numpy ", r$versions[["numpy"]],
          ", scipy ", r$versions[["scipy"]], "\n",
          "  Pour cette session : Sys.setenv(RX_PY = \"", exe, "\")\n",
          "  Pour toujours, dans ~/.Renviron : RX_PY=", exe, "\n",
          "  Pour le supprimer : remlax::rx_remove_python()")
  invisible(exe)
}

# Supprimer l'environnement cree par rx_install_python(). La politique du CRAN
# admet des donnees dans tools::R_user_dir() si leur contenu est gere : cette
# fonction en est le moyen. Le repertoire du paquet est retire s'il est vide.
rx_remove_python <- function(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"),
                             ask = interactive()) {
  dir <- normalizePath(dir, mustWork = FALSE)
  if (!dir.exists(dir)) {
    message("remlax : rien a supprimer, ", dir, " n'existe pas.")
    return(invisible(FALSE))
  }
  if (isTRUE(ask) && !isTRUE(utils::askYesNo(paste0("remlax : supprimer ", dir, " ?"))))
    return(invisible(FALSE))
  unlink(dir, recursive = TRUE)
  parent <- dirname(dir)
  if (identical(parent, normalizePath(tools::R_user_dir("remlax", "data"), mustWork = FALSE)) &&
      dir.exists(parent) && !length(list.files(parent, all.files = TRUE, no.. = TRUE)))
    unlink(parent, recursive = TRUE)
  rm(list = ls(.rx_python_cache), envir = .rx_python_cache)
  message("remlax : ", dir, " supprime.")
  invisible(!dir.exists(dir))
}
