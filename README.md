# remlax

remlax fits linear mixed models by restricted maximum likelihood (REML) from
R, with a formula interface close to ASReml-R. The computations run in a
Python solver shipped with the package. The same model can be fitted on the
marginal covariance `V` or on Henderson's mixed model equations, on CPU or on
GPU, without being rewritten.

## What remlax fits

remlax maximises the restricted likelihood of

```
y = X b + sum_k Z_k u_k + e,    u_k ~ N(0, Sigma_k (x) K_k),    e ~ N(0, R)
```

A random term groups `t` incidence matrices (`n x q`) that share the same `q`
levels, a covariance structure `Sigma` between those `t` columns, and a
relationship matrix `K` between the levels. This form covers the usual models.

| model | `t` | `Sigma` | `K` |
|---|---|---|---|
| random factor (blocks, genotypes) | 1 | one variance | identity |
| genomic BLUP | 1 | one variance | genomic relationship (`rx_grm()`) |
| multi-trait or multi-environment, G x E | number of traits or environments | `us`, `fa(k)`, `diag`, ... | identity or relationship |
| direct and indirect genetic effects | 2 or more, one weighted incidence per column | `us` | identity or relationship |
| spatial field | 1 | one variance | `ar1(row, col)` or a metric kernel; 2D P-splines through `spl2d()` |

The residual `R` takes its own structure: one variance, one variance per
group, a separable `ar1 x ar1` field, or a `us` between traits of the same
unit. The covariance catalogue follows ASReml-R (see "Structure catalogue"
below).

`rx_reml()` is the main function of the package. It reads the model from
formulas and fits it. For instance, a genomic G x E model in long format:

```r
rx_reml(y ~ env, random = ~ fa(gid, rank = 2, K = K) + iid(bloc),
        residual = ~ diag(env):units, data = d, trait = "env")
```

Designs that no factor describes also go through `rx_reml()`. `mm(Z)` adds a
supplied incidence matrix, and `str(~ gid + mm(Zw), struct = "us")` groups
several incidences in one term, as for direct and indirect genetic effects.
Every option of the solver (engine, optimiser, device, outputs) is an
argument of `rx_reml()`. At a lower level, `rx_model()` and `rx_term()` build
the same model object from matrices, and `rx_fit()` fits it with the same
options.

## How the solver works

R builds and checks the design, writes it to a temporary directory as raw
binary files with a JSON manifest, and calls the solver of the installed
package. The solver therefore always has the version of the R interface. R
never imports Python. A Python process starts at the first fit and stays open
for the session, which saves the start of Python and the import of JAX (about
1.5 s) on every later fit.

### Engines

An engine computes the restricted likelihood, its gradient and the average
information matrix. All engines compute the same likelihood. They differ in
the matrix they factorise, and so in the size that drives their cost. That
size is either `n`, the number of observations, or `m`, the size of the mixed
model equations. `m` counts the fixed effects plus the levels of every random
term, a term with `t` columns over `q` levels counting `t x q`. A genomic G x
E model on 2000 genotypes in 10 environments has `m` close to 20 000, whatever
the number of plots.

| engine | option of `rx_reml()` | factorises | cost grows with | devices |
|---|---|---|---|---|
| dense | `engine = "dense"` (default) | the marginal covariance `V` (`n x n`), differentiated by JAX | `n` | CPU, GPU |
| mixed model equations | `engine = "mme"` | `C = W'R^-1W + diag(0, G^-1)` (`m x m`), with `W = [X Z]` | `m` and the fill-in of `C` | CPU, GPU (dense inversion) |
| blocks (experimental) | `engine = "blocks"` | `V = A + U U'`, `A` block diagonal | the largest block and the width of `U` | CPU, GPU |

The dense engine covers every structure. It forms `V`, so it suits models with
at most a few tens of thousands of observations, many variance parameters, or
structures without a sparse inverse. Its low memory mode (`memory = "low"`)
keeps fewer `n x n` matrices alive and lets larger models fit on a GPU.

The mixed model equations engine suits models where `m` is small against `n`,
and sparse `C` (pedigree, separable fields). It inverts `C` densely, or computes
only the entries of `C^-1` it needs by a supernodal selected inversion of the
sparse Cholesky factor. It finds independent blocks of `C` and orders them by
nested dissection when `pymetis` is installed. A model outside its scope
(residual `"none"`, `rr`, a spatial lag, Gaussian metric kernels) is fitted by
the dense engine, with a message.

The blocks engine finds by itself the terms that link otherwise independent
groups of observations, for instance genetic terms across tanks, and writes
them as the low-rank part `U U'`. It processes the blocks in batches and runs
with L-BFGS-B only. It has been measured on one design so far: on a
two-species model with 12 760 observations, one evaluation took 0.90 s
against 4.85 s for the dense engine in low memory mode, on a slice of an
A100. It has not yet been compared with the mixed model equations engine.

### Optimisers

`optimizer = "lbfgsb"` (default) runs L-BFGS-B on an unconstrained
parametrisation. Every `Sigma` is parametrised by its Cholesky factor, so it
stays positive definite by construction.

`optimizer = "ai"` runs average information REML with steps on the variance
scale, as ASReml does. It usually needs far fewer evaluations than L-BFGS-B.
An active set keeps each `us` block on the positive semidefinite cone. When an
eigenvalue reaches zero, the step stays on the face of the cone and can still
rotate the eigenvectors, so an optimum of reduced rank is reached by AI
itself. A step that cannot be shortened usefully is damped in the manner of
Levenberg and Marquardt. When AI stalls or finds no admissible step, L-BFGS-B
resumes from the point reached.

Both optimisers end with the same regularised Newton polishing. Every fit
reports the largest gradient, the Newton decrement, the number of negative
eigenvalues of the Hessian, the null directions and the parameters at a
bound. `optim_msg` says why the optimiser stopped. `n_restarts` adds perturbed
restarts against secondary optima.

### CPU or GPU

`backend = "auto"` takes the GPU when JAX sees one. CPU and GPU compute the
same likelihood, to machine precision. A GPU pays a fixed cost of about 7 s
per fit (compilation and transfers), so it is slower than 8 CPU cores on small
models. On the benchmark of remlax-validation, an A100 was 5 to 30 times
faster than 8 CPU cores on multi-trait `us` and `fa` models and on sparse
multi-environment trials.

### Choosing automatically

`rx_reml(..., engine = "auto", optimizer = "auto")` takes the mixed model
equations engine with average information REML when the model is in its
scope, and otherwise the dense engine with L-BFGS-B, in low memory mode above
n = 5000. On genetic models with a relationship matrix (genomic BLUP, G x E in
`fa` or `us`), this combination was the fastest on the benchmarks against kmmr
and ASReml-R, on CPU and on GPU. When the number of levels is much larger than
`n` (many genotypes predicted from few observations), the dense engine is
faster. The automatic choice does not yet detect this case.

## What a fit returns

- Variance components with their standard errors, and functions of them with
  delta-method standard errors (`vpredict = c(h2 = "V1/(V1+V2+V3)")`).
- BLUPs of every term and, on request, their prediction error variances.
- Fixed effects with their covariance, conditional Wald tests (type III) and
  Kenward-Roger degrees of freedom.
- Predictions in the manner of `predict.asreml` (`rx_predict()`).
- For 2D P-splines, the spatial output of SpATS: maps, spatial trend,
  effective dimensions and generalised heritability (`plot()`,
  `rx_spatial_trend()`, `rx_dimensions()`, `rx_heritability()`).
- The time spent in each stage of the fit (`chrono`).
- The log-likelihood with the constant of lme4 (`logLik`) and with that of
  ASReml-R (`logLik_asreml`).

## Beyond REML

- Bayesian route. `rx_reml(..., bayes = rx_bayes(prior = rx_prior()))`
  samples the posterior of the variance parameters. Fixed and random effects
  are integrated out analytically. Many short Hamiltonian Monte Carlo chains,
  adapted by the ChEES criterion (BlackJAX), run together on CPU or GPU. It
  needs the optional Python package blackjax.
- Genome-wide scans. `rx_scan()` tests every SNP at the variance components of
  a fitted model, for direct effects, indirect effects (doses of the
  neighbours) and their joint or conditional tests. `rx_scan_loco()` uses one
  null model per chromosome, and `rx_scan_exact()` refits the top SNPs with
  re-estimated variance components.
- Neighbourhood designs. `rx_neighbourhood()` builds weighted incidences of
  the neighbours of each plant, by species, distance kernel and dilution.
  `rx_exposure()` computes the exposure that puts indirect variances on the
  phenotypic scale.
- Structural models. `rx_lag()` adds a spatial lag of the response,
  `y = M y + ...`, as in the VP+, TB and TB+ models.

## What it is not

- The dense engine is limited by `n`. It forms `V` explicitly: at n = 16 000,
  `V` takes 2.1 GB in double precision. Large `n` with a small `m`, or a sparse
  `C`, belongs to `engine = "mme"`.
- remlax does not fit generalised linear mixed models. Non-Gaussian responses
  would need an IRLS or Laplace layer above the solver.
- remlax is not a drop-in replacement for asreml. The objective is the same and
  the estimates agree. The algorithm, the iteration counts, the
  log-likelihood constant and the default Wald test type differ. These
  differences are documented in `vignette("structures", package = "remlax")`
  and in the other vignettes.

## Installation

```r
remotes::install_github("NicoSalas34/remlax", build_vignettes = TRUE)
```

From a clone, build the archive first. Installing the source folder directly
works, but R then skips the vignettes and `vignette(package = "remlax")` finds
none.

```sh
R CMD build remlax
R CMD INSTALL remlax_0.1.0.tar.gz
```

The package needs a Python interpreter (>= 3.10) with `jax >= 0.4.30`,
`numpy >= 1.24` and `scipy >= 1.10`. Either point `RX_PY` at one you have, or
let the package create one:

```r
library(remlax)
exe <- rx_install_python()             # CPU build; cuda = TRUE for the GPU build
Sys.setenv(RX_PY = exe)                # or put RX_PY=<exe> in ~/.Renviron
rx_python_check()                      # versions of jax, numpy, scipy seen by remlax
```

Two optional Python packages speed up the sparse inversion of the mixed model
equations: `pymetis` (nested dissection ordering by METIS) and `cvxopt` (the
supernodal Cholesky of CHOLMOD). Without them remlax keeps minimum degree and
SuperLU, which run out of memory on large, locally dense models.

`rx_install_python()` writes only when called, into
`tools::R_user_dir("remlax", "data")`, and asks for confirmation in an
interactive session. `rx_remove_python()` deletes that environment. A Python
without jax is refused before the solver starts, with the interpreter tried and
the line to fix.

During `R CMD check`, the solver runs on a single computation thread. The
option `remlax.threads` sets the number of threads otherwise.

The R side never imports Python. It writes the design to a directory as raw
binary plus a JSON manifest, calls the solver shipped in `inst/python/` as a
subprocess, and reads the results back. The solver is always the copy of the
installed package, so the interface and the solver are of the same version.
Environment variables:

| variable | meaning | default |
|---|---|---|
| `RX_PY` | how to start Python, space-separated | `python3` |
| `RX_CLI` | another solver entry point (a `cli.py` path, or `-m remlax.cli`) | the shipped copy |
| `RX_ENGINE` | default `engine` of `rx_reml()` and `rx_fit()` (`dense`, `mme`, `blocks`, `auto`) | `dense` |
| `RX_OPTIMIZER` | default `optimizer` of `rx_reml()` and `rx_fit()` (`lbfgsb`, `ai`, `ai_theta`, `auto`) | `lbfgsb` |
| `RX_MEMORY` | default `memory` of `rx_reml()` and `rx_fit()` (`standard`, `low`) | `standard` |
| `REMLAX_JAX_CACHE` | folder of the persistent XLA compilation cache; `0` disables it | `<user cache>/remlax/jax` |
| `REMLAX_BLAS_NUMPY` | `garder` keeps NumPy's OpenBLAS threads during mixed model equations fits | one thread |
| `REMLAX_THREADS` | number of CPU cores the solver may use; `tous` disables the limit. Without it, `SLURM_CPUS_PER_TASK`, then `OMP_NUM_THREADS` | no limit |
| `IGE_JAX_CMD` | alternative to `RX_PY` | none |
| `IGE_JAX_SIF` | Apptainer image; expands to `apptainer exec --nv <sif> python3` | none |

## A minimal R example

```r
library(remlax)

set.seed(2026)
n_gen <- 60; n_bloc <- 4
d <- expand.grid(gid = factor(seq_len(n_gen)), bloc = factor(seq_len(n_bloc)))
g <- rnorm(n_gen, 0, sqrt(1.5)); b <- rnorm(n_bloc, 0, sqrt(0.4))
d$y <- 12 + g[as.integer(d$gid)] + b[as.integer(d$bloc)] + rnorm(nrow(d))

fit <- rx_reml(fixed  = y ~ 1,
               random = ~ gid + iid(bloc),
               data   = d,
               vpredict = c(h2 = "V1/(V1+V2+V3)"),
               backend = "cpu")
print(fit)
```

```
Ajustement REML (cpu) : logLik -388.088838 | 3 parametres | 240 obs | 1.6 s
  max|grad| 3.42e-14 | decrement de Newton 1.43e-28 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 1.401
  Sigma[bloc] 1x1, diagonale : 0.00722
  residuelle : 0.8988
  vpredict :
    h2            0.60735   SE  0.05871   [V1/(V1+V2+V3)]
```

## Structure catalogue

Covariance structures for `Sigma` (over the `t` columns of a term). `w` is the
number of columns, `k` the rank or band order.

| remlax | asreml | Sigma | parameters |
|---|---|---|---|
| `iid` | `idv` | `s2 I` | 1 |
| `diag` | `idh` | `diag(s2_j)` | `w` |
| `us` | `corgh` | free symmetric PD | `w(w+1)/2` |
| `fa(k)` | `fa(k)` | `L L' + diag(psi)` | `n_loadings(w,k) + w` |
| `rr(k)` | `rr(k)` | `G G'`, rank `k` | `n_loadings(w,k)` |
| `chol(k)` | `chol(k)` | `L D L'`, band `k` | `(k+1)(w - k/2)` |
| `ante(k)` | `ante(k)` | `Sigma^-1 = U D U'`, band `k` | `(k+1)(w - k/2)` |
| `corh` | `corh` | heterogeneous variances, uniform correlation | `w + 1` |
| `fixed` | none | supplied matrix | 0 |

Correlation structures between the `q` levels of a term. The variance lives in
`Sigma`; these contribute a correlation matrix only, which is the `Sigma_h = D C D`
decomposition of the ASReml-R manual. An asreml `ar1v` is written here
`ar1(f)`, an `ar1h` is `ar1(f, struct = "diag")`.

| family | structures | parameters each |
|---|---|---|
| none / supplied | `id`, `fixed` | 0 |
| uniform | `cor` | 1 |
| stationary 1D | `ar1`, `sar`, `ma1` | 1 |
| | `ar2`, `ma2`, `arma` | 2 |
| | `ar3` | 3 |
| | `corb(order = b)` | `b` |
| general | `corg` | `q(q-1)/2` |
| metric 1D | `exp`, `gau`, `lvr` | 1 |
| metric 2D isotropic | `iexp`, `igau`, `ieuc`, `sph`, `cir` | 1 |
| metric 2D anisotropic | `aexp`, `agau` | 2 |
| Matern | `mtrn` | 0 to 4 (declared ones only) |
| user-defined | `own(expr =, n_par = k)` | `k` |
| separable | `ar1(row, col)` | 2 |

Full definitions, positivity constraints and internal parametrisations:
`vignette("structures", package = "remlax")`.

## Known limitations

- The dense engine forms `V`. See "What it is not" above.
- A Python process stays open during the R session. The first fit starts
  it; later fits reuse it, which avoids the start of Python and the import of
  JAX (about 1.5 s per fit). It stops with the R session or with
  `rx_server_stop()`; `options(remlax.server = FALSE)` runs every fit in a
  fresh subprocess instead.
- The log stays informative through polishing. L-BFGS-B prints one line
  per iteration (step adapted to the measured evaluation cost) and the Newton
  polishing prints one line per step. A log with no new line for a long time
  means the current evaluation is long, not that the fit has stalled.
- CPU fits are bitwise reproducible on one machine; GPU fits are not. On
  CPU, four independent runs of the same fit on the cluster returned exactly
  the same log-likelihood. That equality holds for the same binary on the same
  machine: across jax versions or CPU models the last bit can move (one ulp
  measured between jax 0.11.2 on two GitHub runners). On GPU, XLA reductions are not deterministic across runs:
  measured on an RTX A1000 with jax 0.11.1, two identical fits agree on
  `-2 logL` to 3e-13 but can stop at a `theta` 2e-8 apart, because the
  rounding noise changes which line-search step is accepted and the
  optimiser lands elsewhere on the same flat top. Evaluation at a fixed
  `theta` (`maxiter = 0, polish = 0`) is exact on both devices. Compare GPU fits with a
  tolerance, never with equality. The number of BLAS threads also moves the
  CPU log-likelihood in its sixth decimal, so bitwise comparison across
  machines requires the same thread count.
- Restarts are off by default (`n_restarts = 0`). The Newton decrement
  cannot detect a local optimum: it measures the ascent available locally,
  so it is zero at the top of a secondary hill. Turn restarts on for any fit
  you intend to publish.
- Range kernels (`sph`, `cir`, `lvr`) are multimodal in the range. The
  automatic start sweeps the deciles of the pairwise distances and descends
  briefly from the best three before the fit; a supplied `theta_init` is
  respected. On irregular 1D positions `lvr` can still need `n_restarts`
  (`vignette("structures")`).
- No `ilv`. asreml has one; its formula could not be recovered, and the
  natural candidate (a Euclidean tent) is not positive definite in two
  dimensions. An absent structure is preferred to a wrong one carrying an
  asreml name.
- Kenward-Roger omits the second-order term. It vanishes when `V` is
  linear in the variance parameters, which is true of every structure here
  except the between-level correlation parameters (an AR1 `phi`, a range).
  pbkrtest and SAS make the same choice. The `second_ordre_omis` field says
  whether the model contains parameters for which the omission is not exact.
- `rx_predict()` covers fixed effects and single-trait random terms. A
  `classify` on a multi-trait term is warned about and dropped from the random
  part.
- Wald tests are conditional (type III). asreml's default is sequential
  (type I). The two agree on the last term of the model.
- The log-likelihood constant differs from asreml. remlax includes
  `(n-p)/2 log(2 pi)` (like lme4); asreml omits it. Use `fit$logLik_asreml`
  when comparing.
- The sign of a metric correlation is not identified. `exp`, `gau`,
  `iexp`, `igau`, `ieuc`, `aexp` and `agau` raise `abs(tanh(theta))` to a
  distance, so `theta` and `-theta` give the same model. The reported `phi` is
  always positive and the report carries `signe_non_identifie`; there is no
  sign to interpret.
- Neighbourhood kernel ranges are not estimated by REML. Chaining a kernel
  parameter through `Z` into `V` would require rebuilding the incidence at
  every iteration. They can be profiled by refitting over a grid.
- No non-Gaussian responses.

## Validation

remlax is checked against asreml, lme4, sommer, nlme and pbkrtest on the same
models and the same data, against closed forms and against an independent
dense REML. The campaign of 2026-09-28 counts 353 checks, of which 341 passed.
The 12 others concern three models where asreml stops before the optimum, as
an independent computation shows. The results ship with the package:

```r
vignette("validation", package = "remlax")   # tables, figure, the failures explained, speed
rx_validation_results()                      # the table of checks
rx_validate()                                # rerun the comparisons against lme4, nlme, sommer here
```

The scripts, the asreml comparisons and the speed benchmark live in
[remlax-validation](https://github.com/NicoSalas34/remlax-validation).

## Documentation

`?rx_reml`, `?rx_fit` and the other help pages, the vignettes below, and the
reference manual, whose sources live in `manual/` (`Rscript manual/build.R`
builds the PDF; see `manual/README.md`).

| vignette | contents |
|---|---|
| `remlax` | one random factor, reading the output, heritability |
| `multi-trait` | several traits, `us` and `fa`, genomic relationship |
| `spatial` | separable AR1, metric kernels, 2D splines |
| `explicit-terms` | weighted incidences, direct and indirect genetic effects (the model of Salas et al. 2026) |
| `structural` | spatial lag of the response (`rx_lag`), residual `"none"`, the VP+, TB and TB+ models, REML criterion |
| `structures` | structure catalogue with formulas and parametrisations |
| `scan` | genome-wide scans at fixed covariance, one null per chromosome (LOCO), exact refit of the top SNPs |
| `r-interface` | every `rx_*` function, full formula grammar |
| `engine` | the Python solver behind the package: modules, engines, optimisers |
| `validation` | agreement with asreml, lme4, sommer, nlme; speed |

## Development

`tests/testthat/` holds the tests of the R interface; they skip when no
Python with jax is found. `tests-python/` holds the tests of the solver
(`python -m pytest -c tests-python/pytest.ini tests-python` from the root,
with `inst/python` on the path). Neither the solver tests nor the CI
configuration are part of the built package (`.Rbuildignore`).
`tests/README.md` maps every feature to the tests that cover it.

## Citation

`citation("remlax")`. The neighbourhood model used as the worked example of
the explicit-terms vignette is

Salas N, Montazeaud G, Bourke PM, Julier B, Baranger A, David J (2026). Multispecies
mixtures: an individual-centered quantitative genetic framework for complex
plant neighborhoods. bioRxiv. doi:10.64898/2026.05.27.728303.

## Licence

GNU General Public License, version 3 or later. See `LICENSE`.

The choice follows the company remlax keeps: lme4, sommer and breedR are all
GPL. It also matches the point of the project. A free alternative to a licensed solver stays free, and cannot be
taken into a closed product. The corresponding cost is real and worth stating:
a permissively licensed package cannot depend on remlax.
