# LANCEMENT DU SOUS-PROCESSUS PYTHON. Tout appel au moteur passe par ici,
# pour trois raisons tenues par la politique du CRAN.
#   1. Python ecrit par defaut ses .pyc dans __pycache__ a cote des sources,
#      soit dans la bibliotheque R ou le paquet est installe. Un paquet ne
#      doit ecrire que dans le repertoire temporaire de la session :
#      PYTHONDONTWRITEBYTECODE=1 l'empeche.
#   2. JAX sur CPU prend tous les coeurs. Pendant R CMD check
#      (_R_CHECK_LIMIT_CORES_ pose, ce que fait --as-cran), un paquet ne doit
#      pas en prendre plus de deux. Mesure sur 20 coeurs : OMP_NUM_THREADS=2
#      laisse 2,3 coeurs occupes, un seul fil de calcul et Eigen coupe en
#      laissent 1,2. On prend donc un fil. L'option remlax.threads fixe le
#      nombre hors check.
#   3. system2(env=) est ignore sous Windows. Les variables sont donc posees
#      par Sys.setenv() pour la duree de l'appel, puis restaurees.
.rx_threads <- function() {
  o <- getOption("remlax.threads")
  if (!is.null(o)) return(as.integer(o))
  lim <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_"))
  if (nzchar(lim) && lim != "false") return(1L)
  NULL
}

.rx_system2 <- function(command, args = character(), ..., env = character()) {
  v <- c(PYTHONDONTWRITEBYTECODE = "1")
  n <- .rx_threads()
  if (!is.null(n)) {
    v <- c(v, OMP_NUM_THREADS = n, OPENBLAS_NUM_THREADS = n, MKL_NUM_THREADS = n)
    if (n == 1L)
      v <- c(v, XLA_FLAGS = trimws(paste(Sys.getenv("XLA_FLAGS"),
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")))
  }
  v <- c(v, env)
  ancien <- Sys.getenv(names(v), unset = NA, names = TRUE)
  on.exit({
    for (k in names(ancien))
      if (is.na(ancien[[k]])) Sys.unsetenv(k)
      else do.call(Sys.setenv, stats::setNames(list(ancien[[k]]), k))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(v))
  system2(command, args, ...)
}

# Lance le solveur et arrete avec son message s'il echoue. En mode verbeux la
# progression (stdout) s'affiche en direct et la sortie d'erreur va dans un
# fichier temporaire : sous R, capturer stderr forcerait aussi la capture de
# stdout. Sans cela, un echec ne rendait que « echec » sans la cause.
.rx_lancer <- function(py, a, verbose, quoi) {
  # SOLVEUR PERSISTANT d'abord (remlax_serveur.R) : il evite le demarrage de
  # Python et l'import de JAX a chaque ajustement. NULL : pas de serveur
  # utilisable, on lance le sous-processus comme avant.
  ent <- rx_solver_args(); n0 <- length(ent)
  r <- if (identical(as.character(a[seq_len(n0)]), as.character(ent)))
    .rx_serveur_executer(py, a[-seq_len(n0)], verbose) else NULL
  if (!is.null(r)) {
    if (!identical(r$code, 0L))
      stop(paste(c(paste0(quoi, ". Sortie du solveur :"), utils::tail(r$sortie, 40)),
                 collapse = "\n"), call. = FALSE)
    return(invisible(r$sortie))
  }
  err <- tempfile("rx_err_"); on.exit(unlink(err), add = TRUE)
  # backend = "cpu" : JAX ne doit pas toucher au GPU. Sinon il l'initialise quand
  # meme et tente d'en reserver 75 % de la memoire, ce qui echoue bruyamment
  # (CUDA_ERROR_OUT_OF_MEMORY) des qu'un autre processus occupe la carte.
  ib <- match("--backend", a)
  env <- if (!is.na(ib) && identical(as.character(a[ib + 1L]), "cpu"))
    c(JAX_PLATFORMS = "cpu") else character()
  # instant du lancement, pour le chrono du solveur (chrono.py) : sa premiere
  # etape mesure alors le demarrage de Python
  env <- c(env, RX_T_LANCEMENT = format(as.numeric(Sys.time()), digits = 17))
  st <- suppressWarnings(.rx_system2(py[1], shQuote(c(py[-1], as.character(a))),
                                     stdout = if (verbose) "" else TRUE,
                                     stderr = if (verbose) err else TRUE, env = env))
  code <- if (verbose) as.integer(st) else as.integer(attr(st, "status") %||% 0L)
  sortie <- if (verbose) { if (file.exists(err)) readLines(err, warn = FALSE) else character() } else st
  if (!identical(code, 0L))
    stop(paste(c(paste0(quoi, ". Sortie du solveur :"), utils::tail(sortie, 40)),
               collapse = "\n"), call. = FALSE)
  if (verbose && length(sortie)) message(paste(sortie, collapse = "\n"))
  invisible(sortie)
}
