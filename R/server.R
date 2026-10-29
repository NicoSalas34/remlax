# =============================================================================
# remlax : SOLVEUR PERSISTANT (inst/python/remlax/serveur.py)
# =============================================================================
# Un sous-processus par ajustement paie a chaque fois le demarrage de Python et
# l'import de JAX : environ 1,5 s, plus que le calcul d'un petit modele (mesure
# 2026-10-22, n = 500 : 2,3 s hors solveur pour 0,15 s de calcul). Le serveur
# garde un processus ouvert pour la session R ; .rx_lancer() lui envoie la ligne
# de commande sur un socket local et relaie sa sortie ligne par ligne. En cas de
# probleme, .rx_lancer() revient au sous-processus.
#
# Desactive par options(remlax.server = FALSE), et d'office pendant R CMD check
# (pas de processus qui survive a un exemple).
# =============================================================================

.rx_serveur <- new.env(parent = emptyenv())

.rx_serveur_permis <- function() {
  o <- getOption("remlax.server")
  if (!is.null(o)) return(isTRUE(o))
  !nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")) && !nzchar(Sys.getenv("RX_CLI"))
}

.rx_serveur_vivant <- function() {
  !is.null(.rx_serveur$port) && .rx_serveur_ping()
}

.rx_serveur_ping <- function() {
  con <- tryCatch(suppressWarnings(socketConnection("127.0.0.1", .rx_serveur$port, blocking = TRUE,
                                                    open = "r+b", timeout = 5)),
                  error = function(e) NULL)
  if (is.null(con)) return(FALSE)
  close(con)
  TRUE
}

# Demarre le serveur pour la commande Python `py` ; rend TRUE s'il ecoute.
.rx_serveur_demarrer <- function(py) {
  srv <- tryCatch(c(.rx_launcher(), "serveur"), error = function(e) NULL)
  if (is.null(srv)) return(FALSE)
  f <- tempfile("rx_serveur_")
  jeton <- paste(sample(c(letters, LETTERS, 0:9), 40, TRUE), collapse = "")
  journal <- tempfile("rx_serveur_journal_")
  # XLA_PYTHON_CLIENT_PREALLOCATE=false : un serveur ouvert ne doit pas garder
  # 75 % de la memoire du GPU pendant toute la session.
  # XLA_PYTHON_CLIENT_MEM_FRACTION : sans preallocation, c'est seulement le
  # plafond de la reserve de JAX. Son defaut (75 %) faisait echouer le mode
  # econome sur une tranche MIG de 10 Go (cf. cli.py) ; le serveur sert les deux
  # modes et son backend s'initialise une seule fois, donc le plafond se pose ici.
  .rx_system2(py[1], shQuote(c(py[-1], srv, f, jeton, Sys.getpid())), wait = FALSE,
              stdout = journal, stderr = journal,
              env = c(XLA_PYTHON_CLIENT_PREALLOCATE = "false",
                      XLA_PYTHON_CLIENT_MEM_FRACTION =
                        Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")))
  t0 <- Sys.time()
  while (!file.exists(f) && difftime(Sys.time(), t0, units = "secs") < 120) Sys.sleep(0.05)
  if (!file.exists(f)) return(FALSE)
  v <- scan(f, quiet = TRUE)
  unlink(f)
  .rx_serveur$port <- as.integer(v[1]); .rx_serveur$pid <- as.integer(v[2])
  .rx_serveur$jeton <- jeton; .rx_serveur$py <- paste(py, collapse = " ")
  .rx_serveur$fils <- .rx_threads()
  .rx_serveur$proprietaire <- Sys.getpid()
  if (is.null(.rx_serveur$finaliseur)) {
    reg.finalizer(.rx_serveur, function(e) rx_server_stop(), onexit = TRUE)
    .rx_serveur$finaliseur <- TRUE
  }
  TRUE
}

#' Stop the persistent solver process
#'
#' remlax keeps one Python process open for the R session, so that each fit
#' does not pay again the start of Python and the import of JAX (about 1.5 s
#' per fit). The process starts with the first fit, stops when the R session
#' ends, and can be stopped earlier with this function; the next fit starts a
#' new one. `options(remlax.server = FALSE)` turns the mechanism off: every fit
#' then runs in a fresh subprocess, as before.
#' @return `TRUE`, invisibly, if a server was stopped.
#' @name rx_server_stop
rx_server_stop <- function() {
  if (is.null(.rx_serveur$port)) return(invisible(FALSE))
  con <- tryCatch(suppressWarnings(socketConnection("127.0.0.1", .rx_serveur$port, blocking = TRUE,
                                                    open = "r+b", timeout = 5)),
                  error = function(e) NULL)
  if (!is.null(con)) {
    writeLines(jsonlite::toJSON(list(jeton = .rx_serveur$jeton, ordre = "stop"), auto_unbox = TRUE), con)
    close(con)
  } else if (!is.null(.rx_serveur$pid)) {
    tools::pskill(.rx_serveur$pid)
  }
  .rx_serveur$port <- NULL; .rx_serveur$pid <- NULL
  invisible(TRUE)
}

# Execute la ligne de commande du solveur (sans l'entree cli.py) sur le serveur.
# Rend la liste(code, sortie), ou NULL si le serveur n'est pas utilisable : le
# solveur est alors lance en sous-processus.
.rx_serveur_executer <- function(py, argv, verbose) {
  if (!.rx_serveur_permis()) return(NULL)
  cle <- paste(py, collapse = " ")
  if (!is.null(.rx_serveur$port) && !identical(.rx_serveur$proprietaire, Sys.getpid())) {
    # processus fils (fork) : l'etat herite designe le serveur du PARENT, qui
    # traiterait les requetes une a une. On l'oublie sans l'arreter et on
    # lance celui de ce processus.
    .rx_serveur$port <- NULL; .rx_serveur$pid <- NULL; .rx_serveur$finaliseur <- NULL
  }
  if (!identical(.rx_serveur$py, cle) || !identical(.rx_serveur$fils, .rx_threads()))
    rx_server_stop()                         # autre interpreteur ou autre nombre de fils
  if (!.rx_serveur_vivant() && !.rx_serveur_demarrer(py)) return(NULL)
  con <- tryCatch(suppressWarnings(socketConnection("127.0.0.1", .rx_serveur$port, blocking = TRUE,
                                                    open = "r+b", timeout = 1e7)),
                  error = function(e) NULL)
  if (is.null(con)) return(NULL)
  on.exit(close(con))
  writeLines(jsonlite::toJSON(list(jeton = .rx_serveur$jeton, argv = as.character(argv),
                                       t_lancement = as.numeric(Sys.time())),
                              auto_unbox = TRUE), con)
  sortie <- character(); code <- NA_integer_
  repeat {
    l <- readLines(con, n = 1L, warn = FALSE, encoding = "UTF-8")
    if (!length(l)) break                    # serveur mort en cours de route
    if (startsWith(l, "\036RXFIN ")) {
      code <- as.integer(sub("^\036RXFIN (-?[0-9]+).*$", "\\1", l)); break
    }
    sortie <- c(sortie, l)
    if (verbose) { cat(l, "\n", sep = ""); utils::flush.console() }
  }
  if (is.na(code)) {
    rx_server_stop()
    code <- 1L
    sortie <- c(sortie, "[remlax] le processus solveur persistant s'est arrete en cours d'ajustement")
  }
  list(code = code, sortie = sortie)
}
