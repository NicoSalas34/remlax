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
