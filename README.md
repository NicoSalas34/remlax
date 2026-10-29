# remlax

An R package for REML estimation of linear mixed models with an asreml-like
formula interface. The likelihood is computed by a Python solver shipped with
the package, either on the marginal covariance `V` with JAX (CPU or GPU) or on
Henderson's mixed model equations with a dense or sparse inversion, so the same
model runs on any of these routes without being rewritten.

## What it is

remlax maximises the **restricted likelihood** of

```
y = X b + sum_k Z_k u_k + e,    u_k ~ N(0, Sigma_k (x) K_k),    e ~ N(0, R)
```

A **term** is a set of `t` incidence matrices (n x q) sharing the same `q`
levels, a covariance structure `Sigma` over those `t` columns, and a
relationship matrix `K` between levels. That single object covers a simple
random factor (`t = 1, K = I`), a genomic effect (`t = 1, K = GRM`), a
multi-trait `us`/`fa` model (`t` = number of traits), and models where every
column has its **own** weighted incidence, such as indirect genetic effects.

Three engines compute the restricted likelihood of the same model:

| engine | call | works on | cost follows | devices |
|---|---|---|---|---|
| dense | `rx_fit()` (default) | `V`, `n x n`, differentiated by JAX | `n` | CPU, GPU |
| mixed model equations | `rx_fit(engine = "mme")`, `rx_fit_mme()` | `C = W'R^-1W + diag(0, G^-1)`, `m x m` | `m` and the fill-in of `C` | CPU, GPU (dense inversion) |
| sparse RTMB | `rx_fit_sparse()` | the mixed model equations through TMB | the fill-in of `C` | CPU |

The dense engine covers every structure. The mixed model equations engine
inverts `C` densely (LAPACK, or JAX on the GPU) or computes only the entries
of `C^-1` it needs, by a supernodal selected inversion of the sparse Cholesky
factor; a model outside its scope (residual `"none"`, `rr`, a spatial lag,
Gaussian metric kernels) falls back to the dense engine with a message.

Two optimisers maximise the likelihood. `optimizer = "lbfgsb"` (default) runs
L-BFGS-B on an unconstrained parametrisation: every `Sigma` is parametrised by
its Cholesky factor, so positive definiteness holds by construction.
`optimizer = "ai"` runs average information REML with steps on the variance
scale, as in ASReml, with an active set on the positive semidefinite cone and
Levenberg-Marquardt damping when the step cannot be shortened usefully. The
face of the cone follows the rotations of the eigenvectors of a rank-deficient
`us`, so an optimum on the boundary is reached by AI itself. It hands over to
L-BFGS-B when it stalls or finds no admissible step. Both end with the
same regularised Newton polishing and report the same diagnostics.

A Bayesian route samples the posterior of the variance parameters, from the
posterior mode and without a REML fit by default:
`rx_fit(model, bayes = rx_bayes(prior = rx_prior()))`. Fixed and
random effects are integrated out analytically, and many short chains of
Hamiltonian Monte Carlo, adapted by the ChEES criterion (BlackJAX), advance
together on CPU or GPU. It needs the optional Python package blackjax.

## What it is not

- **The dense engine is limited by `n`.** It forms `V` explicitly (n x n):
  at n = 16 000 that is 2.1 GB in float64. Large `n` with
  few levels, or a sparse `C` (pedigree, separable fields), belongs to
  `engine = "mme"`.
- **Not a generalised linear mixed model package.** Non-Gaussian responses
  would need an IRLS/Laplace layer above the solver.
- **Not a drop-in replacement for asreml.** The objective is the same and the
  estimates agree, but the algorithm, the reported iteration counts, the
  log-likelihood constant, and the default Wald test type all differ. Those
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
| `RX_ENGINE` | default `engine` of `rx_fit()` (`dense`, `mme`, `blocks`, `auto`) | `dense` |
| `RX_OPTIMIZER` | default `optimizer` of `rx_fit()` (`lbfgsb`, `ai`, `ai_theta`, `auto`) | `lbfgsb` |
| `RX_MEMORY` | default `memory` of `rx_fit()` (`standard`, `low`) | `standard` |
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

## Choosing an engine

`rx_fit(model, engine = "auto", optimizer = "auto")` takes the mixed model
equations engine with average information REML whenever the model is in its
scope, and the dense engine with L-BFGS-B otherwise. On genetic models with a
relationship matrix (genomic BLUP, G x E in `fa` or `us`), it is the fastest
combination on the benchmarks against kmmr and ASReml-R, on CPU and on GPU.
`rx_fit()$chrono` gives the seconds spent in each stage of a fit, from the
export by R to the reading back of the result.

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

- **The dense engine forms `V`.** See "What it is not" above.
- **A Python process stays open during the R session.** The first fit starts
  it; later fits reuse it, which avoids the start of Python and the import of
  JAX (about 1.5 s per fit). It stops with the R session or with
  `rx_server_stop()`; `options(remlax.server = FALSE)` runs every fit in a
  fresh subprocess instead.
- **The log stays informative through polishing.** L-BFGS-B prints one line
  per iteration (step adapted to the measured evaluation cost) and the Newton
  polishing prints one line per step. A log with no new line for a long time
  means the current evaluation is long, not that the fit has stalled.
- **CPU fits are bitwise reproducible on one machine; GPU fits are not.** On
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
- **Restarts are off by default** (`n_restarts = 0`). The Newton decrement
  cannot detect a local optimum: it measures the ascent available locally,
  so it is zero at the top of a secondary hill. Turn restarts on for any fit
  you intend to publish.
- **Range kernels (`sph`, `cir`, `lvr`) are multimodal in the range.** The
  automatic start sweeps the deciles of the pairwise distances and descends
  briefly from the best three before the fit; a supplied `theta_init` is
  respected. On irregular 1D positions `lvr` can still need `n_restarts`
  (`vignette("structures")`).
- **No `ilv`.** asreml has one; its formula could not be recovered, and the
  natural candidate (a Euclidean tent) is not positive definite in two
  dimensions. An absent structure is preferred to a wrong one carrying an
  asreml name.
- **Kenward-Roger omits the second-order term.** It vanishes when `V` is
  linear in the variance parameters, which is true of every structure here
  **except** the between-level correlation parameters (an AR1 `phi`, a range).
  pbkrtest and SAS make the same choice. The `second_ordre_omis` field says
  whether the model contains parameters for which the omission is not exact.
- **`rx_predict()` covers fixed effects and single-trait random terms.** A
  `classify` on a multi-trait term is warned about and dropped from the random
  part.
- **Wald tests are conditional (type III).** asreml's default is sequential
  (type I). The two agree on the last term of the model.
- **The log-likelihood constant differs from asreml.** remlax includes
  `(n-p)/2 log(2 pi)` (like lme4); asreml omits it. Use `fit$logLik_asreml`
  when comparing.
- **The sign of a metric correlation is not identified.** `exp`, `gau`,
  `iexp`, `igau`, `ieuc`, `aexp` and `agau` raise `abs(tanh(theta))` to a
  distance, so `theta` and `-theta` give the same model. The reported `phi` is
  always positive and the report carries `signe_non_identifie`; there is no
  sign to interpret.
- **Neighbourhood kernel ranges are not estimated.** Chaining a kernel
  parameter through `Z` into `V` would require rebuilding the incidence at
  every iteration.
- **No non-Gaussian responses.**

## Validation

remlax is checked against asreml, lme4, sommer, nlme and pbkrtest on the same
models and the same data, against closed forms and an independent dense REML,
and its RTMB engine against its dense engine: 399 checks in the campaign of
2026-09-28, 387 passed, the 12 others being three models where asreml stops
before the optimum, as shown by an independent computation. The results ship
with the package:

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
Python with jax is found. `tests-python/` holds the 207 tests of the solver
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

The choice follows the company remlax keeps: lme4, sommer, breedR and TMB are
all GPL, and the sparse engine calls RTMB, so a permissive licence would have
required keeping that engine at arm's length. It also matches the point of the
project - a free alternative to a licensed solver stays free, and cannot be
taken into a closed product. The corresponding cost is real and worth stating:
a permissively licensed package cannot depend on remlax.
