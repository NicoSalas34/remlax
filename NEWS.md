# remlax (development version)

* The RTMB engine is removed: `rx_fit_sparse()`, `rx_sparse_scope()` and
  `rx_tmb_available()` no longer exist, and RTMB and TMB leave Suggests. The
  mixed model equations engine (`engine = "mme"`) covers its scope, sparse
  `C` with a pedigree or a separable field, with more structures, the
  average information optimiser, and the GPU. `rx_n_theta()` stays, now
  defined with the other parameter counts. The 46 validation checks that
  compared the RTMB engine with the dense engine are removed from the
  shipped table, which now holds 353 checks.

* Average information REML on a rank-deficient `us` block. The face of the
  positive semidefinite cone now includes the rotations of the eigenvectors of
  `Sigma`, with an exact retraction and its curvature term in the AI matrix.
  Before, the face froze the eigenvectors at the moment an eigenvalue reached
  the floor. AI stopped there after a few iterations, the point was declared
  unconfirmed, and L-BFGS-B rotated the eigenvectors in small steps: 490
  iterations on `us(8)` with 1000 genotypes, up to 3000 on `us(12)`, often
  ending without convergence. Only the null directions whose gradient points
  out of the cone now stay blocked (partial face). The face threshold is
  `max(1e-8, 100 x floor)`. A point that satisfies the KKT conditions on the
  boundary is declared converged. On `us(12)` with 1000 genotypes, a fit on an
  A100 takes 135 to 151 s instead of 1500 to 6000 s, and the log-likelihoods
  of all engines agree to 1.5e-5. In the mixed model equations engine, a
  nearly singular `Sigma` can still stall AI on `us(8)` and larger. The dense
  engine then usually takes over and reaches the same optimum.
* `rx_scan_loco()`: genome-wide scan at fixed V with one null model per
  chromosome (leave one chromosome out). The relationship matrix of each null
  is computed without the chromosome being scanned, so the polygenic term no
  longer absorbs part of the effect of the tested SNP.
* `rx_scan_exact()`: refits the top SNPs of a scan with the SNP as a fixed
  effect and returns a Wald test with re-estimated variance components. At
  fixed V, the p-value of a SNP of large effect is about 6 times too large.
  The refit removes this loss. A REML likelihood ratio test is not available
  here, since two models with different fixed effects do not share the same
  restricted likelihood. Both functions take a `construire(K, X_sup)` function
  that builds the model, so the design of the null model is kept by
  construction.

* The solver limits its CPU threads to the cores it was given
  (`REMLAX_THREADS`, else `SLURM_CPUS_PER_TASK`, else `OMP_NUM_THREADS`)
  when the process may run on more cores than that. JAX sizes its thread
  pool on the processor affinity mask and ignores `OMP_NUM_THREADS`, so a
  user who set `OMP_NUM_THREADS = 4` on a 32-core workstation still got 32
  threads. The solver restricts the affinity to the least busy cores, on one
  NUMA node when possible, before NumPy and JAX start, and reports it in the
  log (`fils bornes a ...`) and in the result (`coeurs`). It does nothing
  when the scheduler already confines the job, as SLURM does on most
  clusters.
* `engine = "auto"` and `optimizer = "auto"` in `rx_fit()`: the mixed model
  equations engine with average information REML when the model is in its
  scope, otherwise the dense engine with L-BFGS-B (low memory mode above
  n = 5000). The defaults are unchanged.
* Starting values from the residuals of the fixed effects, per component and
  per residual section. With several environments or treatments the raw
  variance of `y` contains the differences between their means, which
  inflated every starting variance. On 5 treatments x 600 genotypes, the
  average information optimiser needs 7 iterations instead of 17.
* The average information optimiser no longer divides a variance by more than
  10 in one step, no longer moves a between-level parameter (correlation,
  range) by more than 0.5 on its internal scale, and frees variances stuck at
  the floor after it stops, as L-BFGS-B already did. On a likelihood with
  several optima (one-stage DROPS model, `ar1 x ar1` fields with a correlation
  near 0.997), full steps moved the correlations from one basin to another and
  the optimum reached depended on the number of BLAS threads. A variance held
  back on three accepted steps in a row may then reach the floor (Heywood case
  of a `fa`): 50 instead of 104 iterations on 5 treatments x 600 genotypes.
* The mixed model equations engine keeps its last evaluation: the finishing
  step and the outputs (`beta`, BLUPs, PEV) at the final point no longer
  factorise `C` again, and the full evaluation of an accepted average
  information step reuses the factorisation of its value-only trial and only
  computes the derivatives (about 10 s of Cholesky per accepted step on 5
  treatments x 3768 genotypes on CPU). A step that sent a residual variance to the floor
  left `-2 log L` without significant digits (huge `log|R|` and `log|C|`
  cancelling) and the optimiser stopped at a wrong point.
* Dense blocks in the dense inversion of the mixed model equations engine, on
  CPU and GPU: a term with a dense relationship matrix enters `C` block by
  block (`Sigma^-1[a, b] K^-1`), and the traces of the derivatives are read
  on the blocks of `C^-1`. These blocks no longer enter the sparse pattern of
  `C` unless a sparse or hybrid inversion is chosen. On 5 treatments x 3768
  genotypes the preparation drops from 18 s. On GPU, `C` is assembled on the
  device by a compiled function and `C^-1` never leaves it: only the traces of
  the blocks, the entries of `C^-1` outside them and its diagonal come back.
* CHOLMOD now always factorises in the order computed by remlax
  (`nmethods = 1`). By default CHOLMOD compared that order with its own and
  kept the cheaper, so on a dense `C` (40 %) the factor came back in another
  order and the selected inversion failed. A check of `L L' v = A v` guards
  the analysis, with SuperLU as fallback.
* Timing by stage. `rx_fit()` returns `chrono`, the seconds spent in each
  stage: export by R, Python start-up and imports, device selection, reading
  the model, preparation, compilation, optimisation, finishing, outputs,
  inference, writing, and reading back by R.
* Persistent compilation cache: programs compiled by XLA are kept on disk
  (`$REMLAX_JAX_CACHE`, else `<remlax cache>/jax`; `REMLAX_JAX_CACHE=0`
  disables it). A second fit of the same shape no longer recompiles.
* Starting values of terms in long format: the exposure was divided by the
  number of components although each row loads only its own, so every
  starting variance was `t` times too large (28 times on a fa1 across 28
  environments). The average information optimiser now takes full steps from
  its third iteration on that model, instead of thirteen damped ones.
* The average information optimiser tries its candidate steps on the value
  alone; the gradient and the AI matrix are computed at the accepted point
  only. On a fa1 across 28 environments, 42 s become 12 s.
* On CPU, the dense inversion of the mixed model equations engine uses LAPACK
  (Cholesky, solves and the inverse from the factor) instead of XLA, and
  `K^-1` is formed from the factor of `K` by triangular solve.
* The NumPy and SciPy wheels each ship their own OpenBLAS, whose threads
  competed for the cores during mixed model equations fits. NumPy's OpenBLAS
  now runs on one thread during those fits, while the LAPACK calls of SciPy
  keep their threads: one full evaluation drops from 1.46 s to 0.25 s on an
  8-core node (m = 1428). `REMLAX_BLAS_NUMPY=garder` keeps the default.
* On CPU only (`JAX_PLATFORMS=cpu`), the solver no longer loads the CUDA
  plugin of JAX, which took 1.65 s at every start-up, and no module creates a
  JAX array at import. The table of variance components is vectorised
  (1.9 s to 0.18 s on 407 components). The Python interpreter is probed only
  when the solver fails to start.
* One evaluation fewer per fit: the compilation is no longer timed by an extra
  evaluation, and the starting point is evaluated once.
* `theta_init` accepts `NA` entries, which take the automatic start. Fixing
  one parameter no longer needs a first call to learn the others.
* The sparse inversion of the mixed model equations engine orders `C` by
  nested dissection (METIS, through the optional `pymetis` package) and
  factorises it with the supernodal Cholesky of CHOLMOD (through the optional
  `cvxopt` package). Without them it keeps minimum degree and SuperLU. Every
  combination gives the dense values to 1e-14.
* `rx_fit(memory = "low")` (solver option `--memoire econome`) runs the dense
  engine with a smaller device footprint: no full-height incidence matrices on
  the device, a residual without structure between units scattered entry by
  entry, factors recomputed in the backward pass (`jax.checkpoint`), no copy
  of `dV` for groups covering most rows, and in the AI step the adjoint formed
  after `P Q`. Same likelihood, gradient and AI matrix as the standard mode,
  which stays the default and is unchanged. On a multi-trait IGE model
  (n = 12 760, 51 terms, 198 parameters), one evaluation now fits on a 10 GB
  MIG slice (peak 4.9 GB, 4.9 s), where the standard mode runs out of memory;
  the log-likelihood equals the CPU value to 2e-16 in relative terms and is
  reproducible bit for bit. The mode raises the JAX allocator ceiling to 95 %
  of the device (default 75 %), unless `XLA_PYTHON_CLIENT_MEM_FRACTION` is set.
* The mixed model equations engine returns prediction error variances
  (`pev`): the diagonal of the random-effect block of `C^-1`, read from the
  selected or dense inversion. They equal those of the dense engine to 3e-15.
* Separable `ar1` residuals on a grid (`ar1`, `ar1 x ar1`, `id`/`ar1`
  products, iid on one trait) keep `C` sparse in the mixed model equations
  engine. Observations are placed in their cells and each empty cell receives
  an indicator fixed effect, which leaves the restricted likelihood unchanged;
  `R^-1` is then sparse and known in closed form. At n = 8000, 5 s against
  80 s when `R` was formed and inverted.
* A persistent Python process serves the fits of an R session: the start of
  Python and the import of JAX are paid once. Compiled functions that depend
  only on the structure of a term are kept between fits. On a model with 500
  observations, a fit takes 0.3 to 1 s instead of 2.2 to 2.8 s.
  `rx_server_stop()` stops the process; `options(remlax.server = FALSE)`
  restores one subprocess per fit.
* Robustness of the average information optimiser: loadings of `fa` and `rr`
  structures are bounded at 100 starting standard deviations of their term
  (the common bound of `theta` let them run to 12 within a few AI steps); the
  mixed model equations engine refuses points where a `Sigma` has a
  condition number above about 1e10, instead of returning a wrong gradient;
  a fit of that engine that the optimiser declares unconverged and that ends
  with a relative gradient above 1e-5 is finished by the dense engine
  (`moteur = "mme+dense"`), except when a term is given by its precision
  `K^-1` alone, which the dense engine cannot represent, or when n exceeds
  20000; and when the AI step
  cannot be made admissible, bursts of L-BFGS-B iterations are interleaved
  before AI resumes. Three five-trait `fa(2)` fits of the benchmark that ended
  460 to 690 log-likelihood units below the optimum now converge to it.

* `rx_fit(bayes = rx_bayes())` (and `rx_reml(bayes = )`): posterior of the
  variance parameters by Hamiltonian Monte Carlo. By default no REML fit is
  run (`start = "map"`): the posterior mode and the Hessian of the negative log
  posterior at the mode give the starting point and the whitening, which
  matters when the REML fit alone takes hours. `start = "reml"` fits by REML
  first and whitens with `2 H^-1`; when that Hessian is not positive definite
  (an optimum at a boundary), the posterior mode takes over. Fixed
  and random effects are integrated out analytically, so the chains carry only
  the variance parameters, with no funnel between a variance and its effects.
  The sampler is BlackJAX's HMC with jittered trajectories, its step size and
  trajectory length adapted by the ChEES criterion across all chains: every
  chain shares the same trajectory length, so many short chains advance
  together in vectorised calls, on CPU or GPU. It works on whitened parameters,
  `theta = theta_0 + L z`, with `L L'` the Laplace covariance at the start. Chains are grouped into
  superchains with overdispersed starting points, and convergence is judged by
  the nested R-hat. Priors (`rx_prior()`): half-t, half-normal, half-Cauchy,
  inverse gamma or flat on standard deviations, LKJ on the correlations of
  `us` blocks, normal on `fa`/`rr` loadings, uniform on correlation parameters
  between levels, with per-term overrides; their Jacobians are checked by
  sampling the priors alone. The result carries the draws of `theta`, of the
  variance components and of the `vpredict` expressions, a summary, sampler
  diagnostics, and Rao-Blackwellised posterior means and standard deviations
  of the fixed and random effects. Requires the optional Python package
  blackjax (`rx_install_python(bayes = TRUE)`). On a model with 400
  observations, 8 chains of 100 draws take 50 s on a laptop CPU, with an
  effective sample size above 550 for every component. A spatial lag is not
  supported yet, and the posterior is always evaluated by the dense engine.
* `inversion = "sparse"` is the documented name of the sparse selected
  inversion (`"creuse"` is still accepted).

* `rx_fit(engine = "mme")` and `rx_fit_mme()`: a second engine inside the
  Python solver, on Henderson's mixed model equations
  `C = W'R^-1W + diag(0, G^-1)` instead of the marginal covariance `V`. `C`
  is assembled as a combination of fixed components (cross-products of the
  incidences, closed-form inverses of AR1 and separable AR1 x AR1
  correlations, `K^-1` of a pedigree) and of components recomputed at each
  evaluation (a dense `K(theta)`, a structured residual block). The
  likelihood, its gradient and the average information matrix agree with
  the dense engine to 1e-13. `inversion = "dense"` inverts `C` with LAPACK or
  JAX (GPU); `inversion = "sparse"` factorises `C` with SuperLU and computes
  the selected inverse by the supernodal Takahashi recurrence, in a small C
  kernel compiled at first use, with a NumPy fallback; `"auto"` chooses from
  `m` and the fill-in of `C` and of its factor. A model outside its scope
  (residual `"none"`, `rr`, a spatial lag, Gaussian metric kernels) is fitted
  by the dense engine with a message. Prediction error variances are not
  available from this engine. The default engine can be set with
  `options(remlax.engine = )` or `RX_ENGINE`.
* `rx_fit(optimizer = "ai")`: average information REML with steps on the
  variance scale (the entries of each `Sigma`), as in ASReml. A `us` block
  whose optimum is rank deficient is handled by an active set on the positive
  semidefinite cone (see the entry on the face of the cone above). When six halvings of the step do
  not decrease `-2 logL`, the step is damped in the manner of Levenberg and
  Marquardt (`AI + mu I`, `mu` increasing): on a five-trait `fa(2)` model the
  undamped run stalled after twelve iterations and fell back to L-BFGS-B
  (9.1 s), the damped run converges by AI alone (2.3 s) to the same
  log-likelihood. After an AI success, the curvature check of the Hessian
  (`hessian = TRUE`) detects saddle points. `optimizer = "ai_theta"` keeps the
  earlier variant with steps on the Cholesky scale. The default optimiser can
  be set with `options(remlax.optimizer = )` or `RX_OPTIMIZER`.
* Both engines replace `X` by the orthonormal factor of its QR decomposition
  and add `2 log|det R|` to the criterion: a fixed covariate on a large scale
  (a date, say) no longer costs precision in `log|X'V^-1X|`.
* Terms declared by their precision (`Kinv`, for instance the inverse
  relationship matrix of a pedigree) are exported to the solver and used
  directly by the mixed model equations engine.
* Separable `ar1(row):ar1(col)` residuals with missing cells: the grid
  coordinates of each observation are exported, so the residual covariance
  is that of the observed cells of the full grid.

* `rx_lag()` and the `lag` argument of `rx_model()` and `rx_reml()`: the
  structural (trait-based, spatial-lag) model `y = M y + X b + Z u + e`, with
  `M[(t,i),(u,j)] = Psi[t,u] W_t[i,j]`. `Psi` (free, diagonal, or a mask,
  with fixed entries if wanted) is estimated with the variance parameters, its
  standard errors come from the same observed Hessian, and the fit reports
  `log|det(I - M)|`, the spectral radius of `M` and the criterion used. Three
  criteria: `"reml_y"` (default, the exact restricted likelihood of `y`, with
  Harville's `|X_y'X_y|` term, which depends on `Psi` when the row sums of `W`
  vary), `"jacobian"` (restricted likelihood of `y*` plus the Jacobian only)
  and `"ml"`. `log|det(I - M)|` is exact on three paths: no feedback
  (`det = 1`), one symmetric `W` (its eigenvalues, computed once), dense LU
  otherwise. Outside the admissible region (a real eigenvalue of `M` at or
  above 1) the objective returns a finite penalty. Dense engine only; Kenward-Roger,
  `rx_predict()` and `rx_scan()` are refused with a lag.
* `rx_border(W, trait, unit, order = 1)`: border covariates (`W 1`, ...,
  `W^order 1`, scaled by the mean row sum) as one fixed-effect column per
  trait, for data in long format. With non-zero trait means the structural
  model implies a mean that depends on the number of neighbours; models
  without a lag need these columns to be unbiased, and structural models use
  them to keep border effects that are not interactions out of `Psi`
  (`vignette("structural")`, section "Trait means and border plants").
* `rx_residual("none")` (and `residual = "none"` in `rx_reml()`): no residual
  term, for models whose plant-level term carries the residual, the indirect
  environmental effects and their covariance (incidences `[I, W]`, structure
  `us`). A `V` that is not positive definite at the starting values is
  refused with an explicit message.
* Entries of a `us` structure that are not identified on their own (two
  off-diagonal entries with the same pattern in `V`, such as
  `Cov(res_t, IEE_u)` and `Cov(res_u, IEE_t)` of a plant term `[I, W]` with a
  symmetric `W`: `q(q-1)/2` of them in VP+) are detected from the incidences.
  The fit reports them (`alias`, `n_plat_attendu`), counts the identified
  parameters (`n_par_identifies`, used by the AIC and BIC of `summary()`),
  keeps their flat directions out of the convergence verdict and of the
  covariance of `theta`, codes them `N` in the component table and adds their
  identified sum with its standard error.
* `rx_model()` warns when a lag has a reciprocal pair `Psi[t,u]`, `Psi[u,t]`
  free together with a term carrying `W` on a trait (IEE*): the direction of
  transmission is then only weakly identified, and a mask is advised.
* The engine's `reml_from_V()` now also returns the gradient with respect to
  `y` (`2 P y`), needed to differentiate through `y* = (I - M) y`; a variant
  `ml_from_V()` gives the profile maximum likelihood.
* New vignette `structural`: the VP+, TB and TB+ models, the spatially
  correlated residual, and the choice of criterion. Cross-validated against
  the independent engine `igepath_reml` on three data sets (with and without
  feedback), two criteria and three models: log-likelihoods within 8e-7
  (1e-12 for TB) and estimates within 8e-7 relative.

# remlax 0.1.0

* First submission to CRAN.
* REML fitting of linear mixed models on the dense marginal covariance, with a
  'JAX' solver shipped in `inst/python` and called as a subprocess.
* Covariance structures of the 'ASReml-R' catalogue, neighbourhood incidences
  for direct and indirect genetic effects, ratios with standard errors, and a
  genome scan at fixed variance components.
* `summary()` of a fit, in the layout of `summary.asreml`: log-likelihood in
  both conventions, AIC, BIC, convergence, variance components with standard
  errors (delta method, exact Jacobian) and `P`/`U`/`B`/`F` codes, fixed
  effects, and BLUPs with `coef = TRUE`.
* `spl2d(x, y, nseg =, deg =, pord =, nest.div =, name =)` in the random
  formula of `rx_reml()`: a two-dimensional P-spline surface whose null part
  joins the fixed effects; `rx_predict()` averages it over the field.
* `rx_spl2d()` and `spl2d()` build the PS-ANOVA of 'SpATS' (`PSANOVA()`)
  matrix for matrix: five smooth components `f(x)`, `f(y)`, `f(x):y`,
  `x:f(y)`, `f(x):f(y)`, each with its own variance, nested bases through
  `nest.div`, and the defaults of 'SpATS' (`nseg = c(10, 10)`). Variance
  components, effective dimensions and fitted values agree with those of
  'SpATS'.
* `spl2d(x, y, at = bloc, at.levels =)`: one surface per level, in the manner
  of `spl2Dc()` in 'sommer', each with its own basis, five variances and null
  part. `rx_spatial_trend()` and `plot()` accept `surface = "all"`.
* Spatial output in the manner of 'SpATS': `rx_dimensions()` (effective,
  model and nominal dimensions), `rx_heritability()` (generalised
  heritability of Oakey et al. 2006), `rx_spatial_trend()` (surface on a grid,
  as a data frame), `plot()` (six panels, exported with `file =`), `fitted()`
  and `residuals()`.
* `rx_validate()` and the validation vignette compare the fits with 'lme4',
  'nlme' and 'sommer'; precomputed comparisons with 'asreml' are shipped in
  `inst/extdata`.
