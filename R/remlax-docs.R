# ==============================================================================
# remlax-docs.R — documentation Rd du paquet, SEPAREE du code.
# ------------------------------------------------------------------------------
# Les pages d'aide vivent ici, separees du code, sous forme de blocs roxygen sans objet
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
#' JAX program shipped with the package under `inst/python/`, so that the
#' interface and the engine always come from the same version.
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
#' locates the solver: `RX_CLI` if set, otherwise the copy shipped in
#' `system.file("python", package = "remlax")`. An installed package
#' therefore needs at most `RX_PY` pointing to a Python that has jax, numpy
#' and scipy.
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
#'   for linear mixed models. <https://github.com/NicoSalas34/remlax>
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
#' @seealso [rx_fit()], [rx_model()], [rx_predict()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(2026)
#' d <- expand.grid(gid = factor(1:30), bloc = factor(1:3))
#' g <- rnorm(30, 0, sqrt(1.5)); b <- rnorm(3, 0, sqrt(0.4))
#' d$y <- 12 + g[d$gid] + b[d$bloc] + rnorm(nrow(d))
#' fit <- rx_reml(y ~ 1, random = ~ gid + iid(bloc), data = d,
#'                backend = "cpu", verbose = FALSE)
#' fit
#' fit$sigmas$gid
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
#'   negligible. Each step prints one line when `verbose = TRUE`.
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
#' @param floor,ceil bounds on every `theta` (default `-12` and `12`). `theta`
#'   is a log standard deviation, `var = exp(2 theta)`, so the default floor is
#'   `var = 3.8e-11`. The values actually used come back as `par_floor` and
#'   `par_ceil` in the result; read them there rather than recoding them.
#' @param verbose pass the solver's output through: one line per L-BFGS-B
#'   iteration and one per Newton polishing step.
#' @param keep keep `dir` after the fit.
#' @return An object of class `rx_fit`, a list with `theta`, `logLik`,
#'   `logLik_asreml`, `n_par`, `n_obs`, `secondes`, `backend`, `sigmas` (named
#'   list of `Sigma` matrices), `sigmas_res` and `sigma_res`, `rho`
#'   (between-level parameters on the scale actually used), `blups` (named list
#'   of `q x t` matrices), `beta`, `vbeta`, `hessian`, `max_grad`,
#'   `newton_decrement`, `n_neg_eig`, `n_null_dir`, `cond`, `n_at_bound`,
#'   `n_fixed_out`, `n_par_free`, `par_floor`, `par_ceil`, `fixed_theta`,
#'   `se_theta` (standard errors of `theta`, `sqrt(2 diag(H^-1))` on the free
#'   subspace, `NA` elsewhere; see [rx_se_theta()]),
#'   `composantes_degenerees`, `composantes_noms`, `optim_msg` (read it first:
#'   `ABNORMAL` is a failed line search, not a maximum), and, when requested, `vpredict`, `wald`, `kenward_roger`,
#'   `pev`, `predictions`. Read fields with `[[` rather than `$`: `$` does
#'   partial matching and `fit$sigmas` once matched `sigmas_res`.
#' @details A model containing a term declared by its precision (`Kinv`) without
#'   a factor of `K` is refused: the dense engine needs `K` and would otherwise
#'   treat the term as independent without a message. Use [rx_fit_sparse()] for
#'   such a term, or supply `K` to [rx_term()].
#' @seealso [rx_reml()], [rx_fit_sparse()], [rx_read_result()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' f <- rx_fit(m, backend = "cpu", verbose = FALSE)
#' f$logLik
#' f$sigmas
#' @usage
#' rx_fit(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
#'        maxiter = 3000L, polish = 25L, n_restarts = 0L,
#'        hessian = TRUE, blups = TRUE, pev = FALSE,
#'        vpredict = NULL, wald = FALSE,
#'        kenward_roger = FALSE, predict = NULL, theta_init = NULL,
#'        fixed_theta = NULL, floor = -12, ceil = 12,
#'        verbose = TRUE, keep = FALSE)
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
#' @examplesIf rx_tmb_available()
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' f <- rx_fit_sparse(m, verbose = FALSE)
#' f$logLik
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
#' @seealso [rx_term()], [rx_residual()], [rx_fit()]
#' @examples
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' m
#' rx_n_theta(m)
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
#' @param colnames names of the `t` columns of `Sigma`; by default the names of
#'   the list `Z` when it has them. They are carried to `dimnames(fit$sigmas[[name]])`
#'   and to the columns of `fit$blups[[name]]`, and [rx_exposure()] and
#'   [rx_ratios()] refer to components by them (`"name:label"`).
#' @examples
#' d <- data.frame(gid = factor(rep(1:10, each = 3)))
#' tt <- rx_term("gid", d$gid)
#' tt$struct
#' @usage
#' rx_term(name, Z, K = NULL, struct = "iid", rank = 0L,
#'         t = NULL, levels = NULL, level = "auto",
#'         dims = NULL, order = 0L, coord = NULL,
#'         opts = NULL, expr = NULL, parts = NULL,
#'         Kinv = NULL, Kinv_logdet = NULL, colnames = NULL)
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
#' @examples
#' r <- rx_residual()
#' r$struct
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
#' @examples
#' lay <- expand.grid(row = 1:6, col = 1:8)
#' s <- rx_spl2d(lay$row, lay$col, nseg = c(3, 3))
#' dim(s$X)
#' names(s$terms)
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
#' @param model an [rx_model] object. In `rx_read_result()` it is optional: when
#'   given, the column names of the terms (`colnames` of [rx_term()]) are set as
#'   `dimnames` on `sigmas` and `blups`.
#' @param dir directory.
#' @param fixed_theta 1-based indices of parameters held fixed during the fit;
#'   they are excluded from the free subspace of `se_theta`.
#' @return `rx_export()` returns `dir` invisibly. `rx_read_result()` returns an
#'   object of class `rx_fit` (see [rx_fit()]), with `se_theta` computed by
#'   [rx_se_theta()].
#' @examples
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' dir <- file.path(tempdir(), "remlax_bundle")
#' rx_export(m, dir)
#' list.files(dir)
#' unlink(dir, recursive = TRUE)
#' @usage
#' rx_export(model, dir)
#' rx_read_result(dir, model = NULL, fixed_theta = NULL)
#' @name rx_export
#' @aliases rx_read_result
NULL

# ------------------------------------------------------------------------------

#' Distance-weighted neighbourhood incidences
#'
#' Builds, for every pair (receiving group, emitting group), the incidence of
#' the neighbours of each unit: by level (weights summed over the units of each
#' level, `n_r x q_e`) and by unit (`n_r x n_e`). Units of different blocks are
#' never neighbours. The window is taken in grid indices (`|d_row| <= rank`
#' and `|d_col| <= rank`, the unit itself excluded); the weight is taken on the
#' physical distance `delta = sqrt((s_1 d_row)^2 + (s_2 d_col)^2)`. Dilution
#' divides each row by `n_i^dilution`, `n_i` being the NUMBER of neighbours of
#' the unit in the emitting group; it is applied before the optional L2
#' normalisation. Level names are returned as given.
#'
#' @param coord `n x 2` matrix or data frame of INTEGER grid positions (row, column).
#' @param group factor of length `n`: the class of each unit.
#' @param block factor of length `n` or `NULL`: two units of different blocks are
#'   never neighbours.
#' @param level factor of length `n`: the level at which weights are summed in the
#'   by-level output. Required unless `output = "unit"`.
#' @param id unit identifiers, used as row names. Default `seq_len(n)`.
#' @param rank radius in grid steps: a scalar, a named `G x G` matrix (rows =
#'   receiving group, columns = emitting group), or a list named by pair
#'   `"receiver<-emitter"`.
#' @param reach kernel exponent `lambda`, same forms as `rank`. `0` gives unit weights.
#' @param dilution exponent `d`, same forms as `rank`.
#' @param kernel `"power"` (`delta^-lambda`), `"exponential"`
#'   (`exp(-lambda (delta - delta_0))`, `delta_0 = min(spacing)`) or `"none"`.
#' @param window `"chebyshev"` (square in grid indices) or `"euclidean"`
#'   (`delta <= rank * min(spacing)`).
#' @param spacing `c(row_step, col_step)` in physical units.
#' @param normalise divide each row by its L2 norm after dilution.
#' @param pairs `NULL` (all `G^2` pairs) or a character vector of `"receiver<-emitter"`.
#' @param output `"level"`, `"unit"` or `"both"`.
#' @param sparse return `dgCMatrix` objects.
#' @return An object of class `rx_neighbourhood`: `level` and `unit` (lists of
#'   matrices named by pair), `n_neighbours` (list of integer vectors), `params`
#'   (one row per pair with the parameters actually used).
#' @seealso [rx_exposure()], [rx_term()]
#' @examples
#' set.seed(1)
#' lay <- expand.grid(row = 1:6, col = 1:8)
#' lay$sp <- factor(ifelse(lay$row %% 2 == 1, "A", "B"))
#' lay$gen <- factor(paste0(lay$sp, sample(1:6, nrow(lay), replace = TRUE)))
#' nb <- rx_neighbourhood(lay[, c("row", "col")], group = lay$sp,
#'                        level = lay$gen, rank = 1)
#' nb
#' dim(nb$level[["A<-B"]])
#' @usage
#' rx_neighbourhood(coord, group, block = NULL, level = NULL, id = NULL,
#'                  rank, reach = 0, dilution = 0,
#'                  kernel = c("power", "exponential", "none"),
#'                  window = c("chebyshev", "euclidean"),
#'                  spacing = c(1, 1), normalise = FALSE,
#'                  pairs = NULL, output = c("level", "unit", "both"),
#'                  sparse = TRUE)
#' @name rx_neighbourhood
NULL

#' Exposure functionals of a (direct, indirect) pair of incidences
#'
#' Averages over `rows`: `d = mean((Z_d K_d Z_d')_ii)`, `k = mean((Z_n K Z_n')_ii)`,
#' `k_identity = mean(sum_g Z_n[i, g]^2)`, `c = mean((Z_d K Z_n')_ii)`,
#' `S = mean(sum_g Z_n[i, g])`, `n_eff = S^2 d / k`. The phenotypic variance
#' brought by an indirect component of variance `s2_I` is `k s2_I`, by the
#' direct component `d s2_D`, and the direct-indirect covariance enters with
#' coefficient `2 c`; the total genetic value of a level is `sqrt(d) (u_D + S u_I)`.
#'
#' Two forms. Matrix form: `rx_exposure(Z_direct, Z_indirect, K, rows, K_direct)`.
#' Model form: `rx_exposure(model, direct = "term:label", indirect = "term:label",
#' rows = "auto")`, where `K` is read from the term of the indirect incidence,
#' `K_direct` from the term of the direct one, and `rows = "auto"` selects the
#' rows where the direct incidence is non-zero (the observations of one target in
#' a stacked model).
#'
#' @param Z_direct `n x q` incidence of the direct effect, `NULL`, or an
#'   [rx_model] / `rx_fit` carrying its model.
#' @param Z_indirect `n x q_e` incidence of the indirect effect.
#' @param K `q_e x q_e` relationship matrix of the emitting group; `NULL` for identity.
#' @param rows row indices to average over; `NULL` for all; `"auto"` in the model form.
#' @param K_direct relationship matrix of the direct effect's group; defaults to `K`.
#' @param direct,indirect component references `"term:label"`, `"term[i]"` or
#'   `"term"` (model form).
#' @return A list: `d`, `k`, `k_identity`, `c`, `S`, `n_eff`, `n_rows`,
#'   `convention` (`"K"` or `"identity"`). `c` is `NA` without a direct incidence
#'   or when the two groups differ.
#' @seealso [rx_neighbourhood()], [rx_ratios()]
#' @examples
#' set.seed(1)
#' lay <- expand.grid(row = 1:6, col = 1:8)
#' lay$sp <- factor(ifelse(lay$row %% 2 == 1, "A", "B"))
#' lay$gen <- factor(paste0(lay$sp, sample(1:6, nrow(lay), replace = TRUE)))
#' nb <- rx_neighbourhood(lay[, c("row", "col")], group = lay$sp,
#'                        level = lay$gen, rank = 1)
#' rx_exposure(Z_indirect = nb$level[["A<-A"]])[c("k", "S")]
#' @usage
#' rx_exposure(Z_direct = NULL, Z_indirect = NULL, K = NULL, rows = NULL,
#'             K_direct = K, direct = NULL, indirect = NULL)
#' @name rx_exposure
NULL

#' Genomic relationship matrix (VanRaden method 1, any ploidy)
#'
#' `D = M k` (or `M` with `coding = "count"`), `p = colMeans(D) / k`,
#' `Z = D - k p`, `G = Z Z' / (k sum p (1 - p))`, then
#' `G_b = (1 - blend) G + blend I`.
#'
#' @param M individuals x markers matrix of allele doses, as a FRACTION of the
#'   ploidy (`coding = "fraction"`) or as counts `0..k` (`coding = "count"`).
#' @param ploidy ploidy `k`.
#' @param blend share of the identity added.
#' @param coding `"fraction"` or `"count"`.
#' @return A `q x q` matrix with the row names of `M` and an attribute
#'   `"denominator"`.
#' @examples
#' set.seed(1)
#' M <- matrix(rbinom(20 * 100, 2, 0.4) / 2, 20, 100,
#'             dimnames = list(paste0("g", 1:20), NULL))
#' G <- rx_grm(M)
#' round(G[1:3, 1:3], 3)
#' @usage
#' rx_grm(M, ploidy = 2, blend = 0, coding = c("fraction", "count"))
#' @name rx_grm
NULL

# ------------------------------------------------------------------------------

#' Confidence intervals of correlations on Fisher's z scale
#'
#' `se_z = se / max(1 - r^2, 1e-8)` (delta method) or `1 / sqrt(n - 3)`
#' (Pearson); `ci = tanh(atanh(r) -+ q se_z)`. An interval wider than
#' `width_max` is flagged non-informative. The reported `z` is the ratio of
#' the correlation to its standard error, on the correlation scale, not
#' `atanh(r)` divided by `se_z`.
#'
#' @param r correlations.
#' @param se standard errors of `r` on the correlation scale; exclusive with `n`.
#' @param n sample size of a Pearson correlation.
#' @param level confidence level.
#' @param width_max width above which the interval is declared non-informative;
#'   `NULL` not to judge.
#' @param clamp bound applied to `r` before `atanh`.
#' @return A data frame: `r`, `se`, `z`, `z_fisher`, `se_z`, `ci_low`, `ci_high`,
#'   `width`, `informative`, `method`, `level`, `width_max`.
#' @examples
#' rx_cor_z(c(-0.4, 0.1, 0.8), se = c(0.1, 0.2, 0.05))
#' rx_cor_z(0.3, n = 50)
#' @usage
#' rx_cor_z(r, se = NULL, n = NULL, level = 0.95, width_max = 1.5,
#'          clamp = 0.999999)
#' @name rx_cor_z
NULL

#' The map from theta to the covariance matrices
#'
#' `rx_sigma_of()` rebuilds one `Sigma` (`t x t`) from the theta of one
#' structure, with the solver's parametrisation: `theta` is a log standard
#' deviation, `us` is `L L'` with `L` lower triangular filled row by row,
#' diagonal `exp(theta)`. `rx_sigmas_from_theta()` splits a full `theta` in the
#' solver's order (terms, then residual sections) and returns every matrix with
#' `dimnames` (the `colnames` of the terms, the trait levels of the residual).
#' Residual sections are named `"residual"` or `"residual:<section>"`.
#' `rx_se_theta()` gives `sqrt(2 diag(H_f^-1))` on the free subspace (neither
#' at a bound read from `par_floor` / `par_ceil` nor fixed), `NA` elsewhere and
#' `NA` everywhere when `H_f` has a non-positive eigenvalue.
#'
#' @param th,theta theta vector (of one structure, or the whole model).
#' @param struct,t,rank structure, dimension, rank.
#' @param model an [rx_model].
#' @param hessian Hessian of `-2 logL`.
#' @param par_floor,par_ceil bounds actually used by the solver.
#' @param fixed_theta 1-based indices of fixed parameters.
#' @param tol_bound tolerance for declaring a parameter at a bound.
#' @return A matrix, a named list of matrices, or a numeric vector.
#' @examples
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' rx_sigma_of(c(0, 0.5, log(2)), "us", 2)
#' rx_sigmas_from_theta(rep(0, rx_n_theta(m)), m)
#' rx_se_theta(c(0.1, -0.2), hessian = diag(c(4, 9)))
#' @usage
#' rx_sigma_of(th, struct, t, rank = 0L)
#' rx_sigmas_from_theta(theta, model)
#' rx_se_theta(theta, hessian, par_floor = -12, par_ceil = 12,
#'             fixed_theta = NULL, tol_bound = 1e-7)
#' @name rx_sigmas_from_theta
#' @aliases rx_sigma_of rx_se_theta
NULL

#' Ratios, shares, heritabilities, tau2, correlations, and their standard errors
#'
#' One map `q(theta)` from the solver's theta to every requested quantity,
#' differentiated once by central finite differences; standard errors follow
#' from `SE = sqrt(J V J')` with `V = 2 H^-1` on the free subspace. Exposure
#' constants (`d`, `k`, `c`, `S`, see [rx_exposure()]) multiply a quantity and
#' its standard error alike. Before any derivative the map is checked against
#' `fit$sigmas` at `1e-8`.
#'
#' Formulas, per target: `V_D = d s2_D`, `V_IW = k_within s2_IW`,
#' `V_IB = k_between s2_IB`, `V_o = k_o s2_o`, `C = c cov_DI`;
#' `V_P = V_D + V_IW + V_IB + sum V_o + 2 C`; shares are `V_x / (V_P - 2 C)`;
#' `h2 = V_D / V_P`, `h2_ext_within = (V_D + 2 C + V_IW) / V_P`,
#' `h2_ext_total = h2_ext_within + V_IB / V_P`; `r_direct_indirect` on the raw
#' components. Total genetic values: `sqrt(d)` on the direct column and
#' `sqrt(d) S_within` on the indirect column of the emitting `Sigma`;
#' `sqrt(d_emitter) S_between` for the effect exerted on the other group;
#' `tau2 = Var(TBV) / V_P` of the RECEIVING target.
#'
#' @param fit an `rx_fit` with `theta`, `hessian`, `par_floor`, `par_ceil` (and
#'   `fixed_theta`, `se_theta` when present).
#' @param components data frame, one row per target: `target`, `direct`,
#'   `indirect_within` (`NA` if none), `indirect_between` (`NA` if none), `other`
#'   (references separated by `+`, additive components of the phenotypic
#'   variance). References are `"term"`, `"term[i]"`, `"term:label"`,
#'   `"residual"`, `"residual:section"`, `"residual:section[i]"`,
#'   `"residual:section:label"`.
#' @param exposure data frame, one row per target: `target`, `d`, `k_within`,
#'   `k_between`, `c`, `S_within`, `S_between`, `k_other`
#'   (`"ref=value+ref=value"`). `NULL` sets every constant to 1 and the column
#'   `scaled` to `FALSE`.
#' @param model the [rx_model] when `fit` does not carry it.
#' @param quantities subset of `"variances"`, `"shares"`, `"h2"`, `"h2_ext"`,
#'   `"tau2"`, `"correlations"`, `"residual_correlations"`, `"tbv"`.
#' @param scale apply `exposure`.
#' @param level,width_max passed to [rx_cor_z()] for correlations.
#' @param jacobian `"numeric"`; `"solver"` is not implemented.
#' @param curvature `"project"` (invert on the positive-curvature directions and
#'   flag `NOT_IDENTIFIED` what depends on the excluded ones) or `"refuse"`
#'   (every standard error `NA`) when `H` has a negative eigenvalue.
#' @param bound_tol tolerance for declaring a parameter at a bound.
#' @param dep_bound share of the Jacobian on bounded parameters above which a
#'   quantity is flagged `COND_BOUND`.
#' @param step relative step of the finite differences.
#' @return A data frame of class `rx_ratios`, one row per quantity: `target`,
#'   `quantity`, `component`, `estimate`, `se`, `z`, `ci_low`, `ci_high`, `flag`
#'   (`OK`, `NOT_ESTIMATED`, `FLOOR`, `COND_BOUND`, `NOT_IDENTIFIED`,
#'   `NOT_ESTIMABLE`, `NO_HESSIAN`), `dep_bound`, `dep_excluded`, `scaled`,
#'   `convention`. Attributes `V`, `free`, `se_theta`, `check_se` (median ratio
#'   to `fit$se_theta`), `sigmas`, `exposure`.
#' @seealso [rx_exposure()], [rx_cor_z()], [rx_sigmas_from_theta()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' fit <- rx_fit(m, backend = "cpu", verbose = FALSE)
#' comp <- data.frame(target = "y", direct = "gid", indirect_within = NA,
#'                    indirect_between = NA, other = "residual")
#' rx_ratios(fit, comp, model = m, quantities = c("variances", "h2"))
#' @usage
#' rx_ratios(fit, components, exposure = NULL, model = NULL,
#'           quantities = c("variances", "shares", "h2", "h2_ext", "tau2",
#'                          "correlations", "residual_correlations", "tbv"),
#'           scale = TRUE, level = 0.95, width_max = 1.5,
#'           jacobian = c("numeric", "solver"), curvature = c("project", "refuse"),
#'           bound_tol = 1e-7, dep_bound = 0.05, step = 1e-5)
#' @name rx_ratios
NULL

#' AIC summary of a grid of fits
#'
#' `AIC = 2 p - 2 logLik` (recomputed when absent, checked when present);
#' best cell = argmin AIC; supported set = cells with `AIC - min <= tol`;
#' marginal ranges of each coordinate over the supported set; counts of cells,
#' missing cells against the Cartesian product of the observed coordinate values,
#' non-PD Hessians and cells with a component at a bound; best PD cell and its
#' AIC distance to the best. With `effective = TRUE` the same summary under
#' `AIC_eff = 2 n_par_free - 2 logLik` and a `best_moved` flag per group.
#'
#' @param table data frame, one row per fit.
#' @param coords names of the columns that define a cell.
#' @param aic,loglik,n_par column names.
#' @param by column(s) separating non-comparable grids (the function stops when
#'   `n_obs` varies inside a group).
#' @param tol width of the supported set in AIC units.
#' @param pd,n_at_bound,n_par_free,n_obs optional column names.
#' @param effective also summarise under `AIC_eff`.
#' @return An object of class `rx_grid_summary`: `best`, `supported`, `ranges`,
#'   `n_supported` (with `n_product`), `counts`, `best_pd`, `delta` (the full
#'   table with `delta_aic` and `supported`), and `effective` when requested.
#' @examples
#' g <- expand.grid(order = 1:3, reach = c(0, 1, 2))
#' g$logLik <- -100 + c(1, 3, 3.5, 2, 5, 5.2, 1, 4, 4.1)
#' g$n_par <- 4
#' rx_grid_summary(g, coords = c("order", "reach"))
#' @usage
#' rx_grid_summary(table, coords, aic = "AIC", loglik = "logLik", n_par = "n_par",
#'                 by = NULL, tol = 2, pd = "pd_hessian", n_at_bound = "n_at_bound",
#'                 n_par_free = "n_par_free", n_obs = "n_obs", effective = FALSE)
#' @name rx_grid_summary
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
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(2026)
#' d <- expand.grid(gid = factor(1:60), bloc = factor(1:4))
#' d$trt <- factor(rep(c("a", "b"), length.out = nrow(d)))
#' d$y <- 12 + 0.5 * (d$trt == "b") + rnorm(60)[d$gid] + rnorm(nrow(d))
#' fit <- rx_reml(y ~ trt, random = ~ gid, data = d, backend = "cpu",
#'                verbose = FALSE)
#' rx_predict(fit, classify = "trt")
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
#' @examples
#' rx_n_params("us", 3)
#' rx_n_params("fa", 4, rank = 1)
#' rx_n_loadings(4, 1)
#' rx_n_level("ar1")
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
#' @examples
#' d <- data.frame(gid = factor(rep(1:10, each = 3)))
#' rx_sparse_scope(list(rx_term("gid", d$gid)))
#' rx_tmb_available()
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
#' the arguments that select the solver entry point: `RX_CLI` if set,
#' otherwise the copy shipped in `system.file("python", package = "remlax")`.
#' The shipped copy is used even when a Python package `remlax` is importable,
#' so that the interface and the engine are always those of the same version.
#' It stops with a message if the shipped copy is missing.
#'
#' @return A character vector.
#' @examples
#' rx_python_cmd()
#' basename(rx_solver_args())
#' @usage
#' rx_python_cmd()
#' rx_solver_args()
#' @name rx_python_cmd
#' @aliases rx_solver_args
NULL

#' Checking and installing the Python side
#'
#' `rx_python_check()` runs the chosen interpreter once and reports whether it
#' imports jax, numpy and scipy, with their versions. The result is cached per
#' command for the session. [rx_fit()], [rx_predict()] and [rx_scan()] call it
#' before launching the solver and stop with an actionable message otherwise:
#' the interpreter tried, the `RX_PY` variable to set, and the installation
#' command. `rx_install_python()` creates a virtual environment with
#' `python -m venv`, installs jax (or `jax[cuda12]`), numpy and scipy with pip,
#' checks the import and prints the `RX_PY` line to put in `~/.Renviron`. It
#' does not modify the session nor `~/.Renviron` itself. In an interactive
#' session it asks for confirmation before writing to `dir` and downloading
#' the Python packages. `rx_remove_python()` deletes that environment, and the
#' package's user data directory when it is left empty.
#'
#' @param py Python command, split on spaces (default `rx_python_cmd()`).
#' @param quiet do not print the diagnostic.
#' @param dir directory of the virtual environment (default: `venv` under
#'   `tools::R_user_dir("remlax", "data")`, the per-user data directory R
#'   provides for a package).
#' @param cuda install the CUDA 12 build of jax. It downloads the NVIDIA CUDA
#'   libraries, which are distributed under NVIDIA's proprietary licence; the
#'   default CPU build uses open-source packages only.
#' @param python base interpreter (>= 3.10) used to create the venv.
#' @param upgrade reinstall the packages if the venv already exists.
#' @param ask ask for confirmation before writing or deleting (default: in
#'   interactive sessions only).
#' @return `rx_python_check()`: a list with `ok`, `python`, `versions` and
#'   `message`, invisibly. `rx_install_python()`: the path of the created
#'   interpreter, invisibly. `rx_remove_python()`: `TRUE` if the directory was
#'   removed, invisibly.
#' @examples
#' chk <- rx_python_check(quiet = TRUE)
#' chk$ok
#' \dontrun{
#' # Downloads several hundred MB from PyPI.
#' exe <- rx_install_python()
#' Sys.setenv(RX_PY = exe)
#' rx_remove_python()
#' }
#' @usage
#' rx_python_check(py = rx_python_cmd(), quiet = FALSE)
#' rx_install_python(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"),
#'                   cuda = FALSE, python = "python3", upgrade = FALSE,
#'                   ask = interactive())
#' rx_remove_python(dir = file.path(tools::R_user_dir("remlax", "data"), "venv"),
#'                  ask = interactive())
#' @name rx_python_check
#' @aliases rx_install_python rx_remove_python
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
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(1)
#' d <- data.frame(gid = factor(rep(1:30, each = 4)), x = rnorm(120))
#' d$y <- 1 + 0.5 * d$x + rnorm(30)[d$gid] + rnorm(120)
#' m <- rx_model(d$y, cbind(1, d$x), terms = list(rx_term("gid", d$gid)))
#' \donttest{
#' fit <- rx_fit(m, backend = "cpu", verbose = FALSE)
#' M <- matrix(sample(c(-1, 1), 30 * 50, replace = TRUE), 30, 50,
#'             dimnames = list(levels(d$gid), paste0("snp", 1:50)))
#' gw <- rx_scan(fit, m, marqueurs = M, incidences = c(dir = "gid"),
#'               tests = "dir", backend = "cpu", verbose = FALSE)
#' head(gw)
#' rx_scan_seuil(gw, "dir")
#' }
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

#' Summary of a REML fit
#'
#' `summary()` on an object returned by [rx_reml()] or [rx_fit()] gathers what
#' `summary.asreml` reports, in the same layout: the call, the number of
#' observations and residual degrees of freedom, the log-likelihood, AIC and
#' BIC, the table of variance components, and the fixed effects. The
#' log-likelihood is given in two conventions: the complete REML
#' log-likelihood (`fit$logLik`, the one of 'lme4' and 'nlme') and the one of
#' 'asreml', which omits a constant. Convergence is judged on the slope
#' (projected gradient and Newton decrement), and a negative eigenvalue of the
#' Hessian is reported.
#'
#' The variance components are listed in the order `V1, V2, ...` used by
#' `vpredict`: the terms, then the residual (one section after the other with
#' `dsum`); within a matrix, the lower triangle row by row (`gid[2,1]` is the
#' covariance between traits 1 and 2); then the parameters of the structure
#' between levels, named `<term>!<structure><k>` in the order of its factors
#' (for `ar1(row):ar1(col)`, 1 is the row correlation and 2 the column one). Their standard errors come from the delta method with the exact
#' Jacobian of the components with respect to `theta`, and a covariance of
#' `theta` equal to `2 H^-1` on the directions of positive curvature of the
#' Hessian `H` of `-2 logL`. The `bound` column follows 'asreml': `P` a
#' variance, `U` an unconstrained parameter (covariance, correlation, range),
#' `B` a component at a bound or on a flat direction of the likelihood, whose
#' standard error is not reported, `F` a fixed component. AIC and BIC count the
#' variance parameters that are not fixed: `AIC = -2 logL + 2 k` and
#' `BIC = -2 logL + k log(n - p)`, with `p` the rank of `X`.
#'
#' @param object an `rx_fit` object.
#' @param coef also return the BLUPs of the random effects, with standard
#'   errors when the fit was made with `pev = TRUE`.
#' @param x a `summary.rx_fit` object.
#' @param digits significant digits.
#' @param ... ignored.
#' @return A list of class `summary.rx_fit`: `call`, `backend`, `n_obs`,
#'   `nedf`, `n_par`, `n_estimated`, `n_at_bound`, `loglik`, `loglik_asreml`,
#'   `aic`, `bic`, `criteria`, `converged`, `convergence`, `varcomp` (data frame
#'   with `component`, `std.error`, `z.ratio`, `bound`), `coef.fixed` (with
#'   `solution`, `std.error`, `z.ratio`), `coef.random` (when `coef = TRUE`),
#'   `vpredict` and `wald` when requested in the fit.
#' @seealso [rx_reml()], [rx_fit()]
#' @examplesIf rx_python_check(quiet = TRUE)$ok
#' set.seed(2026)
#' d <- expand.grid(gid = factor(1:60), bloc = factor(1:4))
#' d$y <- 12 + rnorm(60)[d$gid] + rnorm(4, 0, 0.6)[d$bloc] + rnorm(nrow(d))
#' fit <- rx_reml(y ~ 1, random = ~ gid + bloc, data = d, backend = "cpu",
#'                verbose = FALSE)
#' s <- summary(fit)
#' s
#' s$varcomp
#' @usage
#' \method{summary}{rx_fit}(object, coef = FALSE, ...)
#' \method{print}{summary.rx_fit}(x, digits = max(4L, getOption("digits") - 3L), ...)
#' @name summary.rx_fit
#' @aliases print.summary.rx_fit
NULL

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
#' \method{print}{rx_neighbourhood}(x, ...)
#' \method{print}{rx_ratios}(x, ...)
#' \method{print}{rx_grid_summary}(x, ...)
#' @name print.rx_fit
#' @aliases print.rx_model print.rx_predict print.rx_neighbourhood print.rx_ratios print.rx_grid_summary
NULL
