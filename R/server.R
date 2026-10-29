# ==============================================================================
# server.R - the persistent solver process (inst/python/remlax/serveur.py)
# ==============================================================================
# A subprocess per fit pays each time the start of Python and the import of
# JAX, about 1.5 s, more than the computation of a small model. The server
# keeps one process open for the R session; .rx_run_solver() sends it the
# command line on a local socket and relays its output line by line. When
# anything goes wrong, .rx_run_solver() falls back to a subprocess.
#
# It is turned off by options(remlax.server = FALSE), and always during
# R CMD check, so that no process outlives an example. The JSON keys of the
# protocol (jeton, argv, ordre, t_lancement) are those read by serveur.py.
# ==============================================================================

.rx_server <- new.env(parent = emptyenv())

.rx_server_allowed <- function() {
  o <- getOption("remlax.server")
  if (!is.null(o)) return(isTRUE(o))
  !nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")) && !nzchar(Sys.getenv("RX_CLI"))
}

.rx_server_alive <- function() {
  !is.null(.rx_server$port) && .rx_server_ping()
}

.rx_server_connect <- function(timeout) {
  tryCatch(suppressWarnings(socketConnection("127.0.0.1", .rx_server$port, blocking = TRUE,
                                             open = "r+b", timeout = timeout)),
           error = function(e) NULL)
}

.rx_server_ping <- function() {
  con <- .rx_server_connect(5)
  if (is.null(con)) return(FALSE)
  close(con)
  TRUE
}

# Starts the server for the Python command `py`; TRUE when it listens. It
# writes "<port> <pid>" to a file once it listens. XLA_PYTHON_CLIENT_PREALLOCATE
# = false keeps an open server from holding 75 % of the GPU memory for the
# whole session; without preallocation XLA_PYTHON_CLIENT_MEM_FRACTION is only
# the ceiling of JAX's pool, set here because the backend of the server is
# initialised once for every engine.
.rx_server_start <- function(py) {
  srv <- tryCatch(c(.rx_launcher(), "serveur"), error = function(e) NULL)
  if (is.null(srv)) return(FALSE)
  f <- tempfile("rx_serveur_")
  token <- paste(sample(c(letters, LETTERS, 0:9), 40, TRUE), collapse = "")
  log_file <- tempfile("rx_serveur_journal_")
  .rx_system2(py[1], shQuote(c(py[-1], srv, f, token, Sys.getpid())), wait = FALSE,
              stdout = log_file, stderr = log_file,
              env = c(XLA_PYTHON_CLIENT_PREALLOCATE = "false",
                      XLA_PYTHON_CLIENT_MEM_FRACTION =
                        Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")))
  t0 <- Sys.time()
  while (!file.exists(f) && difftime(Sys.time(), t0, units = "secs") < 120) Sys.sleep(0.05)
  if (!file.exists(f)) return(FALSE)
  v <- scan(f, quiet = TRUE)
  unlink(f)
  .rx_server$port <- as.integer(v[1]); .rx_server$pid <- as.integer(v[2])
  .rx_server$token <- token; .rx_server$py <- paste(py, collapse = " ")
  .rx_server$threads <- .rx_threads()
  .rx_server$owner <- Sys.getpid()
  if (is.null(.rx_server$finalizer)) {
    reg.finalizer(.rx_server, function(e) rx_server_stop(), onexit = TRUE)
    .rx_server$finalizer <- TRUE
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
  if (is.null(.rx_server$port)) return(invisible(FALSE))
  con <- .rx_server_connect(5)
  if (!is.null(con)) {
    writeLines(jsonlite::toJSON(list(jeton = .rx_server$token, ordre = "stop"), auto_unbox = TRUE), con)
    close(con)
  } else if (!is.null(.rx_server$pid)) {
    tools::pskill(.rx_server$pid)
  }
  .rx_server$port <- NULL; .rx_server$pid <- NULL
  invisible(TRUE)
}

# Runs the solver command line (without its entry point) on the server.
# Returns list(code, sortie), or NULL when the server is not usable. A forked
# child inherits the state of its parent's server, which would serve requests
# one at a time: it forgets it without stopping it and starts its own. Another
# interpreter or another thread count also needs a new server.
.rx_server_run <- function(py, argv, verbose) {
  if (!.rx_server_allowed()) return(NULL)
  key <- paste(py, collapse = " ")
  if (!is.null(.rx_server$port) && !identical(.rx_server$owner, Sys.getpid())) {
    .rx_server$port <- NULL; .rx_server$pid <- NULL; .rx_server$finalizer <- NULL
  }
  if (!identical(.rx_server$py, key) || !identical(.rx_server$threads, .rx_threads()))
    rx_server_stop()
  if (!.rx_server_alive() && !.rx_server_start(py)) return(NULL)
  con <- .rx_server_connect(1e7)
  if (is.null(con)) return(NULL)
  on.exit(close(con))
  writeLines(jsonlite::toJSON(list(jeton = .rx_server$token, argv = as.character(argv),
                                   t_lancement = as.numeric(Sys.time())),
                              auto_unbox = TRUE), con)
  # The server sends the solver's output line by line, then "\036RXFIN <code>"
  # (0x1E, an ASCII separator that R can read).
  output <- character(); code <- NA_integer_
  repeat {
    l <- readLines(con, n = 1L, warn = FALSE, encoding = "UTF-8")
    if (!length(l)) break
    if (startsWith(l, "\036RXFIN ")) {
      code <- as.integer(sub("^\036RXFIN (-?[0-9]+).*$", "\\1", l)); break
    }
    output <- c(output, l)
    if (verbose) { cat(l, "\n", sep = ""); utils::flush.console() }
  }
  if (is.na(code)) {
    rx_server_stop()
    code <- 1L
    output <- c(output, "[remlax] le processus solveur persistant s'est arrete en cours d'ajustement")
  }
  list(code = code, sortie = output)
}
