# ==============================================================================
# remlax-docs.R — documentation Rd du paquet, SEPAREE du code.
# ------------------------------------------------------------------------------
# Les fichiers R/remlax.R, R/remlax_tmb.R et R/remlax_scan.R sont GENERES depuis
# les scripts vivants du depot par tools/sync_sources.R ; on n'y ecrit pas.
# Les pages d'aide vivent donc ici, sous forme de blocs roxygen sans objet
# (`NULL`), rattaches par @name. Le NAMESPACE est ecrit a la main : aucun
# @export n'apparait ici, et roxygen2 ne produit que man/*.Rd.
#
# La reference longue, avec la grammaire complete des formules et la
# correspondance avec asreml, est docs/api-r.md a la racine du depot. Les pages
# ci-dessous en sont la forme courte, accessible par ?rx_reml.
# ==============================================================================

#' remlax: a generic, differentiable REML solver for linear mixed models
#'
#' remlax maximises the restricted likelihood of
#' \deqn{y = X b + \sum_k Z_k u_k + e, \quad u_k \sim N(0, \Sigma_k \otimes K_k), \quad e \sim N(0, R).}
#' A *term* is a set of `t` incidence matrices (`n x q`) sharing the same `q`
#' levels, a covariance structure `Sigma` over those `t` columns, and a
#' relationship matrix `K` between levels. This single object covers a random
#' factor (`t = 1, K = I`), a genomic effect (`t = 1, K = GRM`), a multi-trait
#' `us`/`fa` model, and models where every column has its own weighted
#' incidence, such as indirect genetic effects.
#'
#' The R side never imports Python. [rx_export()] writes the design to a
#' directory (raw binary plus `manifest.json`), [rx_fit()] runs the solver as a
#' subprocess, and [rx_read_result()] reads the result back. The solver is a
#' JAX program shipped with the package under `inst/python/`; it also exists as
#' the Python package `remlax` (`pip install remlax`).
#'
#' @section Two engines:
#' [rx_fit()] is the dense engine: `V` is formed explicitly (`n x n`) and
#' differentiated, on CPU or GPU. [rx_fit_sparse()] is the sparse engine built
#' on the mixed model equations and RTMB, for the structures whose precision is
#' sparse and known in closed form ([rx_sparse_scope()] says which). Both take
#' the same [rx_model] object and maximise the same likelihood.
#'
#' @section Finding Python:
#' [rx_python_cmd()] returns the interpreter: `RX_PY`, then `IGE_JAX_CMD`, then
#' `IGE_JAX_SIF` (an Apptainer image), then `python3`. [rx_solver_args()]
#' locates the solver: `RX_CLI`, then `-m remlax.cli` if `import remlax`
#' succeeds, then the copy in `system.file("python", package = "remlax")`,
#' then the repository tree. An installed package therefore needs at most
#' `RX_PY` pointing to a Python that has jax, numpy and scipy.
#'
#' @section Conventions that define the model:
#' Column order of `Z`: `column = (a - 1) * q + level`, level fastest.
#' Order of `theta`: for each term the `Sigma` parameters then the between-level
#' parameters, terms in declaration order, then the residual, then each `dsum`
#' section. `theta` is a log standard deviation for a variance (`var = exp(2
#' theta)`) and the floor is `-12` by default. Row order in a separable product:
#' `level = (row - 1) * n_col + col`. Changing any of these on one side only
#' yields a different model with no error.
#'
#' @section Reading a fit:
#' Read `newton_decrement` (the log-likelihood still available under the local
#' quadratic model, comparable to 3.84 for a one-degree-of-freedom LRT) rather
#' than `max_grad`, which is several units at verified optima. `logLik`
#' includes `(n - p)/2 log(2 pi)` like lme4; asreml omits it, use
#' `logLik_asreml` to compare. `composantes_degenerees` names the terms with a
#' parameter at the floor: those are unidentified, not estimated at zero.
#'
#' @references Salas, N. (2026). remlax: a generic, differentiable REML solver
#'   for linear mixed models. <https://github.com/nsalas/remlax>
#' @name remlax-package
#' @aliases remlax
#' @keywords internal
NULL

# ------------------------------------------------------------------------------

#' Fit a linear mixed model by REML from formulas
#'
#' The formula front end. Builds `X`, the random terms and the residual from
#' an asreml-like grammar, assembles an [rx_model] and calls [rx_fit()].
#'
#' @param fixed two-sided formula for the fixed effects, e.g. `y ~ 1 + trt`.
#' @param random one-sided formula for the random terms; `NULL` is allowed when
#'   the residual is structured. Grammar: `gid` (indicator, `iid`),
#'   `iid(f)`, `vm(f, K = )` (relationship matrix), `diag(f)`, `us(f)`,
#'   `fa(f, k)`, `rr(f, k)`, `chol(f, k)`, `ante(f, k)`, `corh(f)`, between-level
#'   structures `ar1(f)`, `ar2`, `ar3`, `sar`, `ma1`, `ma2`, `arma`, `cor`,
#'   `corb(f, order = )`, `corg`, `exp`, `gau`, `lvr`, `iexp`, `igau`, `ieuc`,
#'   `sph`, `cir`, `aexp`, `agau`, `mtrn`, `own(expr = , n_par = )`, and the
#'   separable `ar1(row):ar1(col)`. `str()` puts one covariance over several
#'   terms. The complete table is in `docs/api-r.md` of the repository.
#' @param residual `"units"`, `"diag"`, `"us"`, or a one-sided formula such as
#'   `~ ar1(row):ar1(col)` or `~ dsum(~ units | section)`.
#' @param data a data frame.
#' @param trait name of the trait column for long-format multi-trait data, or a
#'   factor of length `n`. `NULL` means a single trait.
#' @param unit name of the unit column (two observations of the same unit on
#'   different traits are correlated under a `us` residual), or a vector.
#' @param backend `"auto"` takes the GPU if JAX sees one, `"gpu"` refuses
#'   rather than falling back, `"cpu"` forces the CPU. A machine choice, never a
#'   model choice: CPU and GPU agree to machine precision.
#' @param ... passed to [rx_fit()]: `vpredict`, `wald`, `kenward_roger`,
#'   `predict`, `fixed_theta`, `theta_init`, `n_restarts`, `polish`, `maxiter`,
#'   `hessian`, `blups`, `pev`, `verbose`, `keep`, `dir`.
#' @return An object of class `rx_fit` (see [rx_fit()]) carrying in addition
#'   `model`, `fixed`, `data`, `xlevels` and `call`, which [rx_predict()] needs.
#' @details Incomplete rows are dropped with a message. `attr(X, "assign")` is
#'   restored after that, otherwise a Wald test would examine each column in
#'   isolation instead of the whole term.
#' @examples
#' \dontrun{
#' set.seed(2026)
#' d <- expand.grid(gid = factor(1:60), bloc = factor(1:4))
#' g <- rnorm(60, 0, sqrt(1.5)); b <- rnorm(4, 0, sqrt(0.4))
#' d$y <- 12 + g[d$gid] + b[d$bloc] + rnorm(nrow(d))
#' fit <- rx_reml(y ~ 1, random = ~ gid + iid(bloc), data = d,
#'                vpredict = c(h2 = "V1/(V1+V2+V3)"), backend = "cpu")
#' print(fit)
#' fit$sigmas$gid
#' }
#' @seealso [rx_fit()], [rx_model()], [rx_predict()]
#' @usage
#' rx_reml(fixed, random = NULL, residual = "units", data,
#'         trait = NULL, unit = NULL,
#'         backend = c("auto", "gpu", "cpu"), ...)
#' @name rx_reml
NULL

#' Fit an rx_model with the dense engine
#'
#' Serialises the model with [rx_export()], runs the JAX solver as a
#' subprocess, and reads the result with [rx_read_result()]. Optimisation is
#' L-BFGS-B on an unconstrained parametrisation (a Cholesky factor of every
#' `Sigma`), followed by regularised Newton polishing.
#'
#' @param model an [rx_model] object.
#' @param backend `"auto"`, `"gpu"` or `"cpu"`; see [rx_reml()].
#' @param dir directory receiving the serialised design and the result. A
#'   temporary directory by default, removed unless `keep = TRUE`.
#' @param maxiter cap on L-BFGS-B iterations (default 3000).
#' @param polish cap on Newton polishing steps (default 25). Each step costs two
#'   gradients per parameter. Polishing stops early once the Newton decrement is
#'   negligible. Note that this phase prints nothing to the log: a silent log
#'   is not a stalled fit.
#' @param n_restarts number of perturbed restarts (default 0). The Newton
#'   decrement cannot detect a local optimum; restarts are the only way. Turn
#'   them on for any fit you intend to publish.
#' @param hessian,blups set to `FALSE` to skip the corresponding computation.
#'   Without the Hessian the curvature diagnostics are `NA`; the projected
#'   gradient is still reported.
#' @param pev `FALSE`, `TRUE`, or a character vector of term names: prediction
#'   error variance of the BLUPs, diagonal only, `var(u - u_hat) = G - G Z' P Z
#'   G`. Costs one `(t q) x n` matrix per term: on a large model request it for
#'   the terms you need, never for all.
#' @param vpredict named character vector of functions of the variance
#'   components, e.g. `c(h2 = "V1/(V1+V2)")`, with standard errors by the delta
#'   method. `fit$composantes_noms` gives the `Vi` numbering.
#' @param wald compute Wald tests of the fixed effects (conditional, type III).
#' @param kenward_roger compute Kenward-Roger denominator degrees of freedom and
#'   the adjusted covariance of the fixed effects. The second-order term is
#'   omitted; the field `second_ordre_omis` says whether the model contains
#'   parameters for which the omission is not exact.
#' @param predict `list(L = , M = )`, prediction matrices written for the
#'   solver. Users normally call [rx_predict()] instead.
#' @param theta_init starting value of `theta`, in the solver's order (see
#'   [rx_n_theta()]). With `maxiter = 0` and `polish = 0` the objective is only
#'   evaluated at that point.
#' @param fixed_theta 1-based indices of parameters held at their starting
#'   value. Indices, not combinations: a correlation of a `us` block depends on
#'   two `theta` and cannot be fixed this way.
#' @param verbose pass the solver's output through.
#' @param keep keep `dir` after the fit.
#' @return An object of class `rx_fit`, a list with `theta`, `logLik`,
#'   `logLik_asreml`, `n_par`, `n_obs`, `secondes`, `backend`, `sigmas` (named
#'   list of `Sigma` matrices), `sigmas_res` and `sigma_res`, `rho`
#'   (between-level parameters on the scale actually used), `blups` (named list
#'   of `q x t` matrices), `beta`, `vbeta`, `hessian`, `max_grad`,
#'   `newton_decrement`, `n_neg_eig`, `n_null_dir`, `cond`, `n_at_bound`,
#'   `n_fixed_out`, `n_par_free`, `composantes_degenerees`, `composantes_noms`,
#'   `optim_msg`, and, when requested, `vpredict`, `wald`, `kenward_roger`,
#'   `pev`, `predictions`. Read fields with `[[` rather than `$`: `$` does
#'   partial matching and `fit$sigmas` once matched `sigmas_res`.
#' @details A model containing a term declared by its precision (`Kinv`) without
#'   a factor of `K` is refused: the dense engine needs `K` and would otherwise
#'   treat the term as independent without a message. Use [rx_fit_sparse()] for
#'   such a term, or supply `K` to [rx_term()].
#' @seealso [rx_reml()], [rx_fit_sparse()], [rx_read_result()]
#' @usage
#' rx_fit(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
#'        maxiter = 3000L, polish = 25L, n_restarts = 0L,
#'        hessian = TRUE, blups = TRUE, pev = FALSE,
#'        vpredict = NULL, wald = FALSE,
#'        kenward_roger = FALSE, predict = NULL, theta_init = NULL,
#'        fixed_theta = NULL, verbose = TRUE, keep = FALSE)
#' @name rx_fit
NULL

#' Fit an rx_model with the sparse engine (RTMB)
#'
#' The second engine. `beta` and `u` are both declared random and TMB's Laplace
#' approximation, exact for a Gaussian linear model, integrates them out: this
#' gives the restricted likelihood. The cost depends on the fill-in of the mixed
#' model equations, not on `n`. Only structures whose precision is sparse and
#' known in closed form are in scope; see [rx_sparse_scope()].
#'
#' @param model an [rx_model] object, the same one [rx_fit()] takes.
#' @param theta_init warm start. With `maxiter = 0` the objective is evaluated
#'   at `theta_init`, which is how the two engines are compared on the same
#'   function rather than at their own stopping points.
#' @param maxiter iteration cap. Keep it equal to that of [rx_fit()] when
#'   comparing engines: a lower cap on one side turns a truncation into an
#'   apparent convergence.
#' @param verbose print progress.
#' @param sdreport compute TMB's standard-error report. It forms the covariance
#'   of all random effects, which is expensive as `q` grows. Set it to `FALSE`
#'   for a fair timing against `rx_fit(hessian = FALSE, blups = FALSE)`.
#' @return A list with the same main fields as an `rx_fit` object (`theta`,
#'   `logLik`, `sigmas`, `sigma_res`, `beta`, `blups`), plus `engine = "sparse"`.
#' @seealso [rx_fit()], [rx_sparse_scope()], [rx_tmb_available()]
#' @usage
#' rx_fit_sparse(model, theta_init = NULL, maxiter = 3000L, verbose = TRUE,
#'               sdreport = TRUE)
#' @name rx_fit_sparse
NULL

# ------------------------------------------------------------------------------

#' Assemble and check a model from explicit components
#'
#' The explicit route, for everything a formula cannot say: weighted
#' incidences, heterogeneous columns, one covariance shared by several
#' incidences. Checks and refuses: a term whose incidence has the wrong number
#' of rows; a `diag`/`us`/`fa` residual without several traits; sections that
#' do not partition the observations; an `X` not of full column rank
#' (`log|X'V^-1X|` would be `-Inf`); no random term together with an `iid`
#' residual (nothing to estimate). A model with no random term but a structured
#' residual is legal: it is the commonest spatial field model.
#'
#' @param y response vector of length `n`.
#' @param X fixed-effects design matrix (`n x p`). `attr(X, "assign")` is
#'   carried through so that Wald tests operate on whole terms.
#' @param terms list of [rx_term] objects.
#' @param residual an [rx_residual] object.
#' @param name a label used in the printed inventory.
#' @return An object of class `rx_model` with a `print` method.
#' @examples
#' \dontrun{
#' m <- rx_model(d$y, cbind(1, model.matrix(~ trt, d)[, -1]),
#'               terms = list(rx_term("gid", d$gid)),
#'               residual = rx_residual())
#' print(m)
#' }
#' @seealso [rx_term()], [rx_residual()], [rx_fit()]
#' @usage
#' rx_model(y, X, terms, residual = rx_residual(), name = "modele")
#' @name rx_model
NULL

#' Declare a random term
#'
#' A term is `t` incidence matrices sharing `q` levels, a structure `Sigma`
#' over the `t` columns, and a relationship `K` (or a correlation structure)
#' between the `q` levels.
#'
#' @param name key of the term in every output.
#' @param Z one of: a factor or character vector of length `n` (indicator
#'   incidence, `t = 1`, levels deduced); a list of `t` matrices `n x q`, one per
#'   column of `Sigma` (the form that covers weighted incidences); a matrix
#'   `n x (t q)` already stacked, with `t` supplied.
#' @param K `q x q` relationship matrix with `dimnames`, or `NULL` for the
#'   identity. `K` is reordered onto `levels`; a missing level is an error. A
#'   Cholesky with bending is applied if `K` is not positive definite, with a
#'   message giving the amount added.
#' @param struct structure of `Sigma`: `"iid"`, `"diag"`, `"us"`, `"fa"`,
#'   `"rr"`, `"chol"`, `"ante"`, `"corh"`. Note that `fa(k)` is loadings plus
#'   specific variances and stays full rank; strict reduced rank is `rr(k)`.
#' @param rank rank of `fa`/`rr`, band order of `chol`/`ante`.
#' @param t number of columns; required when `Z` is a stacked matrix.
#' @param levels level names when `Z` is a matrix.
#' @param level structure between levels: `"auto"` (`"fixed"` if `K` is given,
#'   `"id"` otherwise), or one of `id`, `fixed`, `cor`, `corb`, `corg`, `ar1`,
#'   `ar2`, `ar3`, `sar`, `ma1`, `ma2`, `arma`, `exp`, `gau`, `lvr`, `iexp`,
#'   `igau`, `ieuc`, `sph`, `cir`, `aexp`, `agau`, `mtrn`, `own`, `ar1ar1`,
#'   `sep`. A structure combined with a supplied `K` is refused: the structure
#'   is the covariance between levels.
#' @param dims `c(n_rows, n_cols)` for `level = "ar1ar1"`.
#' @param order order of `corb`/`corg`.
#' @param coord coordinates of the `q` levels: a vector (1D) or a two-column
#'   matrix (2D). A 2D metric structure with a one-column `coord` is refused.
#' @param opts named numeric vector of settings (`mtrn`, `own`).
#' @param expr expression for `level = "own"`.
#' @param parts for `level = "sep"`: a list of `(family, dimension)` pairs
#'   describing a separable Kronecker product with any number of factors.
#' @param Kinv sparse precision matrix of the levels, for the sparse engine
#'   only. A term declared by `Kinv` without `K` cannot be fitted by [rx_fit()].
#' @param Kinv_logdet log-determinant of `Kinv`, if already known; computed
#'   otherwise.
#' @return An object of class `rx_term`.
#' @details `struct = "iid"` with `t > 1` is accepted with a message: it means
#'   one variance shared by the `t` columns.
#' @seealso [rx_model()], [rx_residual()]
#' @usage
#' rx_term(name, Z, K = NULL, struct = "iid", rank = 0L,
#'         t = NULL, levels = NULL, level = "auto",
#'         dims = NULL, order = 0L, coord = NULL,
#'         opts = NULL, expr = NULL, parts = NULL,
#'         Kinv = NULL, Kinv_logdet = NULL)
#' @name rx_term
NULL

#' Declare the residual structure
#'
#' @param struct `"iid"`, `"diag"`, `"us"` or `"fa"` over traits.
#' @param trait trait factor of length `n`; `NULL` means a single trait.
#' @param unit unit identifier of length `n`: two observations of the same unit
#'   on different traits are correlated under `us`.
#' @param rank rank of `fa`.
#' @param level,order,coord,n_unit,dims,opts,expr structure between units,
#'   exactly as for a term (see [rx_term()]).
#' @param sections list of `rx_residual` objects for a direct sum (`dsum`), each
#'   carrying its `rows` and `name`.
#' @param rows,name row indices and label of a section.
#' @return An object of class `rx_residual`.
#' @seealso [rx_model()], [rx_term()]
#' @usage
#' rx_residual(struct = "iid", trait = NULL, unit = NULL, rank = 0L,
#'             level = "id", order = 0L, coord = NULL, n_unit = NULL,
#'             dims = NULL, opts = NULL, expr = NULL, sections = NULL,
#'             rows = NULL, name = NULL)
#' @name rx_residual
NULL

#' Two-dimensional P-spline basis (PS-ANOVA)
#'
#' Tensor-product B-spline basis with the PS-ANOVA decomposition of the
#' penalty. Nothing is added to the solver: a smooth surface is a random effect
#' with a known incidence.
#'
#' @param x,y coordinates of the `n` observations.
#' @param nseg number of segments in each direction.
#' @param deg degree of the B-splines.
#' @param pord order of the difference penalty.
#' @param prefix prefix of the term names.
#' @return `list(X = , terms = )`. `X` is the null-space part and must go into
#'   the fixed effects, otherwise the surface is penalised down to its linear
#'   component. `terms` is a list of three [rx_term] objects, `<prefix>_x`,
#'   `<prefix>_y`, `<prefix>_xy`, each with its own variance, which makes the
#'   smoothing anisotropic.
#' @usage
#' rx_spl2d(x, y, nseg = c(6L, 6L), deg = 3L, pord = 2L, prefix = "spl")
#' @name rx_spl2d
NULL

# ------------------------------------------------------------------------------

#' Serialise a model for the solver, and read a result back
#'
#' `rx_export()` writes `manifest.json` plus one `<name>.bin` (float64 or int32,
#' column-major) or `<name>.txt` per array into `dir`. `Z` is written as a COO
#' triplet (`zi`, `zj`, `zx`) with 0-based indices, stacked in the order
#' `column = (a - 1) q + level`. `rx_read_result()` reads `result.json` and the
#' binary arrays the solver wrote in the same directory.
#'
#' @param model an [rx_model] object.
#' @param dir directory.
#' @return `rx_export()` returns `dir` invisibly. `rx_read_result()` returns an
#'   object of class `rx_fit` (see [rx_fit()]).
#' @usage
#' rx_export(model, dir)
#' rx_read_result(dir)
#' @name rx_export
#' @aliases rx_read_result
NULL

# ------------------------------------------------------------------------------

#' Predictions from a fitted model
#'
#' A prediction is a linear combination of the fitted effects; all the work is
#' building the right combination. `classify` names the variables held at each
#' of their levels. Every other variable of the fixed model is averaged: a
#' factor over its levels, a covariate at its mean, as asreml does. If a
#' `classify` variable is the factor of a random term, its BLUP enters the
#' prediction and the standard error becomes a prediction error computed by the
#' solver. A combination outside the row space of `X` is flagged not estimable
#' and returned as `NA`.
#'
#' @param fit an `rx_fit` object produced by [rx_reml()] (it must carry the
#'   model frame).
#' @param classify variables to predict for, separated by `:` or `+`.
#' @param levels restrict the levels used.
#' @param at force specific values of other variables.
#' @param average `"equal"` or `"proportional"` to observed counts.
#' @param weights reweight the cells.
#' @param vcov `"simple"` or `"kenward-roger"` (needs a fit made with
#'   `kenward_roger = TRUE`; applies to the fixed part only).
#' @param sed attach the matrix of standard errors of differences
#'   (`attr(out, "sed")`) and their quadratic mean (`attr(out, "sed.moyen")`).
#' @param include_random `FALSE` gives the fixed-effect prediction only.
#' @param backend,verbose passed to the solver.
#' @return A data frame of class `rx_predict` with columns `predicted.value`,
#'   `std.error`, `estimable`.
#' @details Only single-trait random terms are covered. A `classify` on a
#'   multi-trait term is warned about and dropped from the random part.
#' @seealso [rx_reml()]
#' @usage
#' rx_predict(fit, classify, levels = NULL, at = NULL,
#'            average = c("equal", "proportional"), weights = NULL,
#'            vcov = c("simple", "kenward-roger"),
#'            sed = FALSE, include_random = TRUE,
#'            backend = c("auto", "gpu", "cpu"), verbose = FALSE)
#' @name rx_predict
NULL

# ------------------------------------------------------------------------------

#' Parameter counts
#'
#' `rx_n_params()` counts the parameters of a `Sigma` structure over `t`
#' columns, `rx_n_loadings()` the loadings of an `fa`/`rr` structure,
#' `rx_n_level()` the parameters of a between-level structure, and
#' `rx_n_theta()` the length of the whole `theta` vector of a model. These
#' reproduce `structures.py` on the Python side; any divergence would make the
#' two sides read `theta` differently with no error.
#'
#' `rx_n_theta()` is what you need to impose a common `theta` on both engines:
#' evaluated at different starting points, their log-likelihoods are not
#' comparable and the gap looks like a formulation defect.
#'
#' @param struct structure name.
#' @param t number of columns.
#' @param rank rank or band order.
#' @param r rank, for `rx_n_loadings()`.
#' @param level between-level structure name.
#' @param order order of `corb`/`corg`.
#' @param opts options (`mtrn`).
#' @param parts factor list of a `sep` structure.
#' @param model an [rx_model] object.
#' @return An integer.
#' @usage
#' rx_n_params(struct, t, rank = 0L)
#' rx_n_loadings(t, r)
#' rx_n_level(level, order = 0L, opts = NULL, parts = NULL)
#' rx_n_theta(model)
#' @name rx_n_params
#' @aliases rx_n_loadings rx_n_level rx_n_theta
NULL

#' Scope of the sparse engine
#'
#' `rx_sparse_scope()` says whether a set of terms and a residual are within
#' the sparse engine's perimeter (`Sigma` in `iid`/`diag`/`us`; levels in
#' `id`/`ar1`/`ar1ar1` or a supplied sparse precision `Kinv`; residual `iid` or
#' `diag`, no sections), and why not otherwise. `rx_tmb_available()` says
#' whether RTMB and Matrix can be loaded.
#'
#' @param terms list of [rx_term] objects.
#' @param residual an [rx_residual] object.
#' @return `rx_sparse_scope()`: `list(ok = , reason = )`.
#'   `rx_tmb_available()`: logical.
#' @seealso [rx_fit_sparse()]
#' @usage
#' rx_sparse_scope(terms, residual = NULL)
#' rx_tmb_available()
#' @name rx_sparse_scope
#' @aliases rx_tmb_available
NULL

#' Locating the Python solver
#'
#' `rx_python_cmd()` returns the command that starts Python, split on spaces:
#' the first of `RX_PY`, `IGE_JAX_CMD`, `IGE_JAX_SIF` (expanded to
#' `apptainer exec --nv <sif> python3`), `python3`. `rx_solver_args()` returns
#' the arguments that select the solver entry point: `RX_CLI` if set; then
#' `-m remlax.cli` if `import remlax` succeeds under that Python; then the copy
#' shipped in `system.file("python", package = "remlax")`; then
#' `src/remlax/cli.py` in a repository tree. It stops with a message if none is
#' found.
#'
#' @return A character vector.
#' @examples
#' \dontrun{
#' Sys.setenv(RX_PY = "/path/to/venv/bin/python")
#' rx_python_cmd()
#' rx_solver_args()
#' }
#' @usage
#' rx_python_cmd()
#' rx_solver_args()
#' @name rx_python_cmd
#' @aliases rx_solver_args
NULL

# ------------------------------------------------------------------------------

#' GWAS at fixed V on a fitted model
#'
#' `rx_scan()` does not refit. It writes the `theta` already estimated by
#' [rx_fit()] into the bundle and runs the scan with `--theta-in --maxiter 0`,
#' so the variance components of the scan are exactly those of the fit. The
#' incidences of the focal genotype and of the neighbours are named by the
#' *terms* of the model that carry them, never supplied as matrices, so the
#' scan uses the design of the null model by construction.
#'
#' @param fit an `rx_fit` object; only `fit$theta` is read.
#' @param model the corresponding [rx_model].
#' @param marqueurs matrix of genotype doses (`q` rows named by the term's
#'   levels, `p` SNP columns), or `list(dir = , ind = )` for the inter-species
#'   case (joint tests are then refused). The `sim` family requires centred
#'   doses (`+/-1` coding).
#' @param carte optional data frame with columns `snp`, `chr`, `pos`.
#' @param incidences named vector: which terms carry the focal incidence
#'   (`dir`) and the neighbour incidence (`ind`).
#' @param tests test specifications, e.g. `"dir"`, `"ind"`, `"sim"`,
#'   `"dir+ind+sim"`; a bar denotes a conditional test, `"ind|dir+sim"`.
#' @param maf minor allele frequency filter computed on `marqueurs$dir`; `NULL`
#'   to keep all.
#' @param codage `"pm1"` (doses in `-1/+1`) or `"012"`; used for the MAF only.
#' @param backend,dir,verbose,keep as in [rx_fit()].
#' @param bloc SNP block width for the `sim` family.
#' @return A data frame with one row per SNP and an attribute `"meta"` (genomic
#'   control lambda per test, timings, `-2 logL` of the null).
#'
#' `rx_scan_seuil()` returns the Bonferroni threshold `alpha / m` for a test
#' column, `m` being the number of finite p-values.
#' @param res result of `rx_scan()`.
#' @param test name of a test present in `res` (column `p_<test>`).
#' @param alpha family-wise level.
#' @usage
#' rx_scan(fit, model, marqueurs, carte = NULL,
#'         incidences = c(dir = "gen", ind = "voisin"),
#'         tests = c("dir", "ind", "sim", "dir+ind+sim"),
#'         maf = 0.05, codage = c("pm1", "012"),
#'         backend = c("auto", "gpu", "cpu"), bloc = 1024L,
#'         dir = NULL, verbose = TRUE, keep = FALSE)
#' rx_scan_seuil(res, test, alpha = 0.05)
#' @name rx_scan
#' @aliases rx_scan_seuil
NULL

# ------------------------------------------------------------------------------

#' Print methods
#'
#' `print.rx_model()` prints the inventory of terms with their structure, `t`,
#' `q`, relationship and parameter count. `print.rx_fit()` prints the
#' log-likelihood, the convergence diagnostics (`max|grad|`, Newton decrement,
#' number of negative eigenvalues), every `Sigma`, the residual, and the
#' `vpredict` table when present. `print.rx_predict()` prints the prediction
#' table and the mean standard error of differences when `sed = TRUE`.
#'
#' @param x the object.
#' @param ... ignored.
#' @return `x`, invisibly.
#' @usage
#' \method{print}{rx_fit}(x, ...)
#' \method{print}{rx_model}(x, ...)
#' \method{print}{rx_predict}(x, ...)
#' @name print.rx_fit
#' @aliases print.rx_model print.rx_predict
NULL
