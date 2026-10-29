# ==============================================================================
# 5. AJUSTEMENT : delegation au solveur JAX
# ==============================================================================
# @param backend "auto" | "gpu" | "cpu" - choix de MACHINE, jamais de modele.
# @param vpredict expressions sur les composantes, facon asreml :
#   c(h2 = "V1/(V1+V2)", rg = "V2/sqrt(V1*V3)"). Les Vi sont numerotees dans
#   l'ordre rendu par `fit$composantes_noms`.
# @param wald TRUE pour les tests de Wald sur les effets fixes
# @param fixed_theta indices (1-based) des parametres a FIXER a leur depart
rx_fit <- function(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
                   maxiter = 3000L, polish = 25L, n_restarts = 0L,
                   hessian = TRUE, blups = TRUE, pev = FALSE,
                   vpredict = NULL, wald = FALSE,
                   kenward_roger = FALSE, predict = NULL, theta_init = NULL,
                   fixed_theta = NULL, floor = -12, ceil = 12,
                   verbose = TRUE, keep = FALSE,
                   optimizer = getOption("remlax.optimizer", Sys.getenv("RX_OPTIMIZER", "lbfgsb")),
                   engine = getOption("remlax.engine", Sys.getenv("RX_ENGINE", "dense")),
                   inversion = c("auto", "dense", "sparse", "hybrid"), bayes = NULL,
                   memory = getOption("remlax.memory", Sys.getenv("RX_MEMORY", "standard")),
                   finish = getOption("remlax.finish", Sys.getenv("RX_FINISH", "newton"))) {
  # optimizer : "lbfgsb" (defaut) ou "ai" (AI-REML, pas pris en echelle des
  # variances, a la maniere d'asreml). Polissage, diagnostics et sorties ne
  # changent pas. Le defaut se regle sans toucher aux appels, par
  # options(remlax.optimizer = "ai") ou la variable d'environnement RX_OPTIMIZER :
  # c'est ce qui permet de rejouer une suite de validation entiere sous AI-REML.
  optimizer <- match.arg(optimizer, c("lbfgsb", "ai", "auto"))
  # engine : "dense" (V formee) ou "mme" (equations du modele mixte). Un modele
  # hors du perimetre du moteur MME (residuelle none, rr, decalage) y bascule
  # vers le dense, avec un message du solveur.
  # "auto" : moteur MME et AI-REML quand le modele est dans le perimetre du
  # MME, sinon moteur dense et L-BFGS-B (econome au-dela de n = 5000)
  engine <- match.arg(engine, c("dense", "mme", "blocks", "auto"))
  engine_cli <- if (engine == "blocks") "blocs" else engine
  # inversion de C dans le moteur MME. "creuse" reste accepte (nom d'origine,
  # employe par les bancs et la suite de verification) ; le solveur attend
  # toujours "creuse".
  if (identical(inversion, "creuse")) inversion <- "sparse"
  inversion <- match.arg(inversion)
  inversion_cli <- switch(inversion, sparse = "creuse", hybrid = "hybride", inversion)
  # memory : "standard" ou "low" (le solveur dit "econome", econome.py). Meme
  # vraisemblance ; "low" garde moins de matrices n x n sur le GPU.
  memory <- match.arg(memory, c("standard", "low"))
  memory_cli <- if (memory == "low") "econome" else "standard"
  # finish : "newton" (polissage apres L-BFGS-B) ou "trust" (arret sur progres
  # lent et Newton a region de confiance, finition.py)
  finish <- match.arg(finish, c("newton", "trust"))
  finish_cli <- if (finish == "trust") "rc" else "newton"
  # LES BORNES DE theta SE DEMANDENT ICI, ET NULLE PART AILLEURS. Le CLI les
  # acceptait (--floor, --ceil) mais aucun argument R ne les transmettait : un
  # appelant qui croyait poser un plancher a -8 obtenait -12 sans un mot, et
  # un pipeline a enregistre PAR_FLOOR=-8 dans ses specs pendant que le solveur
  # travaillait a -12. theta est un LOG D'ECART-TYPE : var = exp(2 theta), donc
  # floor = -12 vaut var = 3,8e-11. Les valeurs effectives reviennent dans le
  # resultat (par_floor, par_ceil) : les lire la, ne jamais les recoder.
  floor <- as.numeric(floor); ceil <- as.numeric(ceil)
  if (length(floor) != 1L || length(ceil) != 1L || !is.finite(floor) || !is.finite(ceil) ||
      floor >= ceil)
    stop("rx_fit : floor et ceil doivent etre deux scalaires finis avec floor < ceil.",
         call. = FALSE)
  # UN MODELE DECLARE PAR SA PRECISION N'EST PAS AJUSTABLE ICI, ET LE TAIRE
  # COUTE CHER. rx_model() pose l'attribut `creux_seulement` des qu'un terme
  # porte Kinv sans LK ; personne ne le lisait, si bien que ce moteur ajustait
  # le modele en traitant le terme comme INDEPENDANT (level "id", ligne "K=I"
  # dans le journal) sans un mot. Cout mesure : la variance genetique de
  # Hauteur.4 sortait a 0,3019 -- exactement celle de l'ajustement univarie
  # SANS parente (0,3013) -- au lieu de 0,1849, celle de l'univarie AVEC
  # parente (0,1839). Toutes les heritabilites en decoulant etaient gonflees
  # d'un facteur 1,63, sans qu'aucun diagnostic ne le signale.
  if (isTRUE(attr(model$terms, "creux_seulement")) && !(as.character(engine)[1] %in% c("mme", "auto")))
    stop("rx_fit : ce modele contient un ou plusieurs termes declares par leur ",
         "PRECISION (Kinv) sans facteur de K. Le moteur dense a besoin de K et ",
         "ignorerait silencieusement la parente, en estimant les effets comme ",
         "INDEPENDANTS. Fournir K a rx_term(), ou ajuster avec engine = \"mme\".",
         call. = FALSE)

  backend <- match.arg(backend)
  # Kenward-Roger et predict supposent y lineaire en beta et u a V donnee ;
  # avec un decalage, seule y* = (I - M) y l'est, et M depend d'un parametre.
  if (!is.null(model$lag) && (isTRUE(kenward_roger) || !is.null(predict)))
    stop("rx_fit : Kenward-Roger et predict ne sont pas disponibles avec un decalage ",
         "spatial (rx_lag).", call. = FALSE)
  if (is.null(dir)) { dir <- tempfile("rx_"); on.exit(if (!keep) unlink(dir, recursive = TRUE)) }
  t_R0 <- Sys.time()
  rx_export(model, dir)
  # predict : L (l x p) et, par terme aleatoire, M (l x t*q). Ecrits en
  # column-major comme tout le reste du paquet.
  if (!is.null(predict)) {
    wbin <- function(nm, x) { con <- file(file.path(dir, nm), "wb")
      writeBin(as.double(as.vector(as.matrix(x))), con, size = 8); close(con) }
    wbin("pred_L.bin", predict$L)
    for (nm in names(predict$M %||% list())) wbin(paste0("pred_M_", nm, ".bin"), predict$M[[nm]])
  }
  # Depart a chaud. Avec maxiter = 0 et polish = 0, c'est une simple EVALUATION
  # de la vraisemblance au theta fourni : le seul moyen de savoir si deux
  # solveurs calculent la meme fonction ou s'arretent seulement a des endroits
  # differents.
  if (!is.null(theta_init)) {
    con <- file(file.path(dir, "in_theta.bin"), "wb")
    writeBin(as.double(theta_init), con, size = 8); close(con)
  }
  py <- rx_python_cmd()
  # PAS DE SONDE PREALABLE de l'interpreteur : elle lancait un Python juste pour
  # importer jax, numpy et scipy (1,5 s a chaque session R) avant que le solveur
  # ne refasse les memes imports. Elle ne sert plus qu'a expliquer un echec du
  # lancement (voir plus bas), avec le meme message qu'avant.
  a <- c(rx_solver_args(), dir, "--backend", backend, "--maxiter", maxiter, "--polish", polish,
         "--optimiseur", optimizer, "--moteur", engine_cli, "--inversion", inversion_cli,
         "--memoire", memory_cli, "--finition", finish_cli,
         .rx_bayes_args(bayes, dir),
         "--restarts", n_restarts,
         "--floor", format(floor, digits = 15), "--ceil", format(ceil, digits = 15),
         if (!is.null(vpredict)) c("--vpredict",
           paste(sprintf("%s=%s", names(vpredict), unlist(vpredict)), collapse = ";")) else NULL,
         if (isTRUE(wald)) "--wald" else NULL,
         if (isTRUE(kenward_roger)) "--kenward-roger" else NULL,
         if (!is.null(predict)) "--predict" else NULL,
         if (!is.null(theta_init)) "--theta-in" else NULL,
         if (!is.null(fixed_theta)) c("--fixed-theta",
           paste(as.integer(fixed_theta), collapse = ",")) else NULL,
         if (!hessian) "--no-hessian" else NULL, if (!blups) "--no-blups" else NULL,
         # DEFAUT A FALSE, deliberement. La PEV coute une matrice (t*q) x n par
         # terme : gratuite pour un modele univarie a quelques centaines de
         # genotypes, lourde des que t*q monte. Elle se demande donc, elle ne
         # s'impose pas. Sans elle H2_Cullis n'est pas calculable.
         # pev accepte TRUE (tous les termes) ou un VECTEUR DE NOMS. Sur le
         # modele IGE complet, 21 des 23 termes sont des nuisances dont la
         # variance d'erreur de prediction n'interesse personne, et les
         # demander toutes a epuise la memoire lors du premier essai reel.
         if (isTRUE(pev)) "--pev" else
           if (is.character(pev) && length(pev)) c("--pev-termes", paste(pev, collapse = ",")) else NULL,
         if (!verbose) "--quiet" else NULL)
  t_R1 <- Sys.time()
  tryCatch(.rx_lancer(py, a, verbose, "solveur REML : echec"), error = function(e) {
    # interpreteur sans jax, numpy ou scipy : le dire en clair, avec RX_PY
    .rx_python_assurer(py)
    stop(e)
  })
  t_R2 <- Sys.time()
  res <- rx_read_result(dir, model = model, fixed_theta = fixed_theta)
  t_R3 <- Sys.time()
  # CHRONO PAR ETAPES : celles du solveur (result.json, chrono.py) encadrees
  # par celles de R. "solveur_total" est le temps mur du lancement au retour ;
  # l'ecart avec la somme des etapes du solveur est la fin du processus.
  dt <- function(a, b) as.numeric(difftime(b, a, units = "secs"))
  ch_py <- unlist(res[["chrono"]])
  # demarrage_serveur : ce que le temps mur du solveur a de plus que ses etapes
  # (demarrage du serveur persistant au premier ajustement de la session, fin
  # du processus en sous-processus)
  res[["chrono"]] <- c(export_R = dt(t_R0, t_R1), ch_py,
                       demarrage_serveur = max(dt(t_R1, t_R2) - sum(ch_py), 0),
                       solveur_total = dt(t_R1, t_R2), lecture_R = dt(t_R2, t_R3))
  # le solveur nomme l'inversion creuse "creuse" ; l'interface dit "sparse"
  if (identical(res[["inversion"]], "creuse")) res[["inversion"]] <- "sparse"
  if (identical(res[["inversion"]], "hybride")) res[["inversion"]] <- "hybrid"
  res
}

# ERREURS-TYPES SUR theta. Le Hessien rendu est celui de -2 logL, donc
# cov(theta) = 2 H^-1 sur le sous-espace LIBRE : ni a une borne (lue dans
# par_floor / par_ceil, jamais recodee), ni fixe par fixed_theta. Ailleurs NA.
# Une valeur propre negative de H_f signale un point de selle : les erreurs-types
# y sont NA plutot qu'un nombre sans sens. Verifie contre le vpredict du solveur
# (test-ratios.R).
rx_se_theta <- function(theta, hessian, par_floor = -12, par_ceil = 12,
                        fixed_theta = NULL, tol_bound = 1e-7) {
  p <- length(theta)
  se <- rep(NA_real_, p)
  if (is.null(hessian) || !p) return(se)
  H <- as.matrix(hessian)
  if (nrow(H) != p || ncol(H) != p) return(se)
  libre <- !(theta <= par_floor + tol_bound | theta >= par_ceil - tol_bound)
  if (length(fixed_theta)) libre[as.integer(fixed_theta)] <- FALSE
  if (!any(libre)) return(se)
  Hf <- (H[libre, libre, drop = FALSE] + t(H[libre, libre, drop = FALSE])) / 2
  if (!all(is.finite(Hf))) return(se)
  ev <- eigen(Hf, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-12 * max(abs(ev), 1)) return(se)
  Vi <- tryCatch(solve(Hf), error = function(e) NULL)
  if (is.null(Vi)) return(se)
  d <- diag(Vi)
  se[libre] <- ifelse(d > 0, sqrt(2 * d), NA_real_)
  se
}

# =============================================================================
# remlax : moteur par les EQUATIONS DU MODELE MIXTE (facade R)
# =============================================================================
# Le calcul est entierement en Python (inst/python/remlax/mme.py, selinv.py) :
# C = W'R^-1W + diag(0, G^-1) formee comme combinaison de composantes fixes,
# AI-REML en echelle des variances (comme asreml), C dense par LAPACK/JAX (GPU
# possible) ou creuse par SuperLU et inversion selective supernodale.
#
# Cette fonction n'est qu'une facade de rx_fit(engine = "mme") : meme paquet
# binaire, memes parametres theta, memes sorties et diagnostics que le moteur
# dense. Un modele hors du perimetre (residuelle "none", structure rr,
# decalage rx_lag, noyaux gaussiens) est ajuste par le moteur dense, avec un
# message.
# =============================================================================

#' Fit an rx_model through the mixed model equations
#'
#' A shortcut for `rx_fit(model, engine = "mme", ...)`, with the average
#' information optimiser by default. The restricted likelihood is computed from
#' Henderson's mixed model equations, with `W = [X, Z]`,
#' \deqn{C = W' R^{-1} W + \mathrm{diag}(0, G^{-1}),}{C = W' R^-1 W + diag(0, G^-1),}
#' instead of the marginal covariance `V`. The cost then depends on `m`, the
#' number of fixed effects and random levels, and on the fill-in of the factor
#' of `C`, rather than on `n`.
#'
#' The model, the parameters `theta`, the result and its diagnostics are those
#' of [rx_fit()]: both engines maximise the same likelihood and agree to
#' rounding. A model outside the scope of the mixed model equations engine
#' (residual `"none"`, an `rr` structure, a spatial lag, Gaussian metric kernels)
#' is fitted by the dense engine, with a message.
#'
#' @param model an [rx_model] object.
#' @param inversion how the selected entries of `C^-1` are obtained.
#'   `"dense"` inverts `C` with LAPACK on the CPU or with JAX on the GPU.
#'   `"sparse"` factorises `C` with SuperLU and computes the selected inverse
#'   by the supernodal Takahashi recurrence, on the CPU, from a small C kernel
#'   compiled at first use (a NumPy fallback runs without a compiler).
#'   `"auto"` (default) takes the dense route when `m <= 1000`, when the
#'   pattern of `C` fills more than 30 % of its lower triangle, or when its
#'   factor fills more than half of it.
#'   `"hybrid"` splits the unknowns into a dense separator (fixed effects and
#'   the terms that link otherwise independent groups of observations, found
#'   automatically) and independent blocks. The blocks are factorised as dense
#'   matrices in batches and the dense Schur complement of the separator is
#'   factorised last, on the GPU when one is available. When a block is too
#'   large, the route falls back to `"auto"` with a message.
#' @param optimizer `"ai"` (default) or `"lbfgsb"`; see [rx_fit()].
#' @param ... passed to [rx_fit()]: `backend`, `maxiter`, `polish`,
#'   `hessian`, `blups`, `vpredict`, `verbose`, and the others.
#' @return An object of class `rx_fit`, with `moteur = "mme"` and the
#'   inversion actually used in `inversion`. Prediction error variances
#'   (`pev`) are the diagonal of the random-effect block of `C^-1`, read from
#'   the same inversion. A fit that ends with a large relative gradient is
#'   finished by the dense engine from the point reached, and `moteur` then
#'   reads `"mme+dense"`.
#' @seealso [rx_fit()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' \donttest{
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' f <- rx_fit_mme(m, backend = "cpu", verbose = FALSE)
#' f$logLik
#' }
#' @name rx_fit_mme
rx_fit_mme <- function(model, inversion = c("auto", "dense", "sparse", "hybrid"), optimizer = "ai", ...) {
  if (identical(inversion, "creuse")) inversion <- "sparse"
  inversion <- match.arg(inversion)
  rx_fit(model, engine = "mme", inversion = inversion, optimizer = optimizer, ...)
}
