# remlax

A generic REML solver for linear mixed models, written in JAX, driven from R by
an asreml-like formula interface. The same model runs on CPU or on GPU without
being rewritten.

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
  differences are documented in [docs/structures.md](docs/structures.md) and
  in the guides.
- **Two forms of the R interface, one source.** `R/remlax.R` is a single file
  you `source()`, and `rpkg/` is the same code as an installable R package
  (generated from it by `rpkg/tools/sync_sources.R`, with a copy of the Python
  solver under `inst/python/`). Use whichever suits: the scripts for a cluster
  job that sets `RX_REMLAX_R`, the package for `library(remlax)` and `?rx_fit`.

## Installation

### Python solver

```sh
git clone https://github.com/NicoSalas34/remlax
cd remlax
pip install -e .                 # CPU
pip install -e ".[cuda]"         # CUDA 12 build of JAX
```

Requirements: Python >= 3.10, `jax >= 0.4.30`, `numpy >= 1.24`,
`scipy >= 1.10`. Double precision is enabled at import time by
`remlax._x64`; do not disable it.

Check what JAX can see:

```sh
python -c "import remlax; print(remlax.device_report())"
```

### R interface

Either source the script:

```r
source("R/remlax.R")            # needs Matrix and jsonlite
```

or install the package, which ships its own copy of the Python solver and
documents every `rx_*` function:

```r
install.packages("rpkg", repos = NULL, type = "source")   # from a clone
# remotes::install_github("NicoSalas34/remlax", subdir = "rpkg")
library(remlax); ?rx_reml
```

The package still needs a Python interpreter with jax, numpy and scipy. Either
point `RX_PY` at one you have, or let the package create one:

```r
exe <- rx_install_python()             # ~/.remlax/venv, CPU build; cuda = TRUE for GPU
Sys.setenv(RX_PY = exe)                # or put RX_PY=<exe> in ~/.Renviron
rx_python_check()                      # versions of jax, numpy, scipy seen by remlax
```

A Python without jax is refused before the solver starts, with the
interpreter tried and the line to fix. See [rpkg/README.md](rpkg/README.md).

The R side never imports Python. It writes the design to a directory as raw
binary plus a JSON manifest, calls the solver as a subprocess, and reads the
results back from the same directory. Two environment variables control that
call:

| variable | meaning | default |
|---|---|---|
| `RX_PY` | how to start Python, space-separated | `python3` |
| `RX_CLI` | path to `cli.py`, or `-m remlax.cli` | auto-detected |
| `IGE_JAX_CMD` | alternative to `RX_PY` | — |
| `IGE_JAX_SIF` | Apptainer image; expands to `apptainer exec --nv <sif> python3` | — |

`RX_CLI` is resolved in three steps: the variable if set; then `-m
remlax.cli` if `import remlax` succeeds under `RX_PY`; then
`src/remlax/cli.py` relative to `R/remlax.R`. So an editable install needs
neither variable, and an uninstalled clone needs only `RX_PY`.

```sh
export RX_PY=/path/to/venv/bin/python
export RX_CLI=/path/to/remlax/src/remlax/cli.py    # only if not pip-installed
```

## A minimal R example

```r
source("R/remlax.R")

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

## A minimal Python example

The Python API takes plain dictionaries; it knows nothing about formulas,
factors or data frames.

```python
import numpy as np
from remlax.fit import fit_reml
from remlax.inference import component_names, vpredict

rng = np.random.default_rng(1)
n_gen, n_rep = 40, 5
n = n_gen * n_rep
gid = np.repeat(np.arange(n_gen), n_rep)
u = rng.normal(0, np.sqrt(2.0), n_gen)
y = 10.0 + u[gid] + rng.normal(0, 1.0, n)
X = np.ones((n, 1))

term = dict(name="genotype", struct="iid", t=1, rank=0, q=n_gen,
            zi=np.arange(n), zj=gid, zx=np.ones(n), LK=None)
res = dict(struct="iid", t=1, rank=0,
           trait=np.zeros(n, dtype=int), unit=np.arange(n))

fit = fit_reml([term], res, y, X, verbose=False)
print(component_names([term], res))
print(fit["sigmas"]["genotype"][0, 0], fit["sigma_res"][0, 0])

vp = vpredict(fit["theta"], fit["hessian"], [term], res, [("h2", "V1/(V1+V2)")])
print(vp["predictions"][0])
```

```
['genotype', 'residuelle']
1.5153589929029908 0.8959200739880405
{'nom': 'h2', 'expression': 'V1/(V1+V2)', 'valeur': 0.6284461...,
 'se': 0.0659438...}
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
[docs/structures.md](docs/structures.md).

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
  (`docs/structures.md`).
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

remlax is checked against asreml, lme4, sommer, pbkrtest, closed-form REML on
balanced designs, finite-difference gradients, and a random stress sweep; and
CPU against GPU on a bundle of designs covering every structure. Details and
measured agreements are in [docs/validation.md](docs/validation.md).

<!-- VALIDATION-SUMMARY -->

## Benchmarks

Timings and memory for CPU and GPU across model sizes are in
[docs/benchmarks.md](docs/benchmarks.md), produced by `benchmarks/bench.py`.

<!-- BENCHMARK-SUMMARY -->

## Documentation

| page | contents |
|---|---|
| [docs/index.md](docs/index.md) | entry point and reading order |
| [docs/guide/01-getting-started.md](docs/guide/01-getting-started.md) | one random factor, reading the output, heritability |
| [docs/guide/02-multi-trait.md](docs/guide/02-multi-trait.md) | several traits, `us` and `fa`, genomic relationship |
| [docs/guide/03-spatial.md](docs/guide/03-spatial.md) | separable AR1, metric kernels, 2D splines |
| [docs/guide/04-explicit-terms.md](docs/guide/04-explicit-terms.md) | weighted incidences, DGE/IGE with shared covariance |
| [docs/guide/05-chapter3-reproduction.md](docs/guide/05-chapter3-reproduction.md) | neighbourhood incidences, exposures, ratios with standard errors, grid summary; the chapter-3 reproduction scripts |
| [docs/api-r.md](docs/api-r.md) | R reference: every `rx_*` function, full formula grammar |
| [docs/api-python.md](docs/api-python.md) | Python reference: every public function |
| [docs/structures.md](docs/structures.md) | structure catalogue with formulas and parametrisations |
| [docs/note_remlax_fr.md](docs/note_remlax_fr.md) | design note (French): why each choice was made |

## Citation

```
Salas, N. (2026). remlax: a generic, differentiable REML solver for linear
mixed models. Version 0.1.0. https://github.com/NicoSalas34/remlax
```

`CITATION.cff` at the repository root carries the machine-readable form.

## Licence

GNU General Public License, version 3 or later. See `LICENSE`.

The choice follows the company remlax keeps: lme4, sommer, breedR and TMB are
all GPL, and the sparse engine calls RTMB, so a permissive licence would have
required keeping that engine at arm's length. It also matches the point of the
project - a free alternative to a licensed solver stays free, and cannot be
taken into a closed product. The corresponding cost is real and worth stating:
a permissively licensed package cannot depend on remlax.
