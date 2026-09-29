# remlax

An R package for REML estimation of linear mixed models with an asreml-like
formula interface. The likelihood is computed and differentiated by a JAX
engine shipped with the package, so the same model runs on CPU or on GPU
without being rewritten.

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

The likelihood is differentiated by automatic differentiation, with an
analytic vector-Jacobian product for `d(-2 logL)/dV`. Optimisation is L-BFGS-B
on an **unconstrained** parametrisation — positive definiteness of every
`Sigma` is guaranteed by construction, because a Cholesky factor is
parametrised, never `Sigma` itself — followed by regularised Newton polishing.

## What it is not

- **Not a sparse solver.** `V` is formed explicitly and densely (n x n). At
  n = 16 000 that is 2.1 GB in float64. Beyond that regime you need a sparse
  factorisation or Henderson's mixed model equations; `dense_Z()` and
  `assemble_V()` are the two functions that would have to change.
- **Not a generalised linear mixed model package.** Non-Gaussian responses
  would need an IRLS/Laplace layer above the solver.
- **Not a drop-in replacement for asreml.** The objective is the same and the
  estimates agree, but the algorithm, the reported iteration counts, the
  log-likelihood constant, and the default Wald test type all differ. Those
  differences are documented in `vignette("structures", package = "remlax")`
  and in the other vignettes.

## Installation

```r
remotes::install_github("NicoSalas34/remlax")
# or, from a clone
install.packages(".", repos = NULL, type = "source")
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

`rx_install_python()` writes only when called, into
`tools::R_user_dir("remlax", "data")`. A Python without jax is refused before
the solver starts, with the interpreter tried and the line to fix.

The R side never imports Python. It writes the design to a directory as raw
binary plus a JSON manifest, calls the engine shipped in `inst/python/` as a
subprocess, and reads the results back. The engine is always the copy of the
installed package, so the interface and the engine are of the same version.
Environment variables:

| variable | meaning | default |
|---|---|---|
| `RX_PY` | how to start Python, space-separated | `python3` |
| `RX_CLI` | another engine entry point (a `cli.py` path, or `-m remlax.cli`) | the shipped copy |
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
| `fixed` | — | supplied matrix | 0 |

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

- **`V` is dense.** See "What it is not" above.
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
  cannot detect a local optimum — it measures the ascent available *locally*,
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
and its sparse engine against its dense engine: 399 checks in the campaign of
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

`?rx_reml`, `?rx_fit` and the other help pages, and the vignettes:

| vignette | contents |
|---|---|
| `remlax` | one random factor, reading the output, heritability |
| `multi-trait` | several traits, `us` and `fa`, genomic relationship |
| `spatial` | separable AR1, metric kernels, 2D splines |
| `explicit-terms` | weighted incidences, direct and indirect genetic effects (the model of Salas et al. 2026) |
| `structures` | structure catalogue with formulas and parametrisations |
| `scan` | genome-wide scans at fixed covariance |
| `r-interface` | every `rx_*` function, full formula grammar |
| `engine` | the Python engine behind the package |
| `validation` | agreement with asreml, lme4, sommer, nlme; speed |

## Development

`tests/testthat/` holds the tests of the R interface; they skip when no
Python with jax is found. `tests-python/` holds the 179 tests of the engine
(`python -m pytest -c tests-python/pytest.ini tests-python` from the root,
with `inst/python` on the path). Neither the engine tests nor the CI
configuration are part of the built package (`.Rbuildignore`).
`tests/README.md` maps every feature to the tests that cover it.

## Citation

`citation("remlax")`. The neighbourhood model used as the worked example of
the explicit-terms vignette is

Salas N, Montazeaud G, Bourke PM, Baranger A, David J (2026). Multispecies
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
