# Architecture of remlax

This guide is for whoever takes over the code. It says where each piece lives,
how a fit travels from R to Python and back, and where to change things. The
user documentation is in the vignettes; `vignettes/engine.Rmd` documents the
Python functions one by one. The measurements and bug histories behind the
design choices are in `dev/notes.md`.

## Two halves and one file format

remlax has an R front end (`R/`) and a Python/JAX engine
(`inst/python/remlax/`). They share no code and no runtime: R writes a model
to a directory (the bundle), starts Python on it, and reads the result back
from the same directory. R needs no Python package and Python needs no R
package.

```
rx_reml(formulas, data)            R/reml.R, formula_random.R, formula_residual.R, spl2d.R
  -> rx_model(y, X, terms, res)    R/model.R, term.R, residual.R, lag.R
  -> rx_fit(model)                 R/fit.R
       rx_export(model, dir)       R/bundle_write.R    manifest.json + raw arrays
       solver process              R/launch.R, python.R, server.R
         remlax_run.py cli dir     inst/python/remlax_run.py -> remlax.cli
         Bundle(dir)               bundle.py
         fit_reml(...)             fit.py and the engines
         result.json + out_*.bin   cli.py
       rx_read_result(dir)         R/bundle_read.R     -> rx_fit object
  -> print, summary, predict, ratios, scan, spatial outputs
```

The bundle is described in `R/bundle_write.R` (writer) and `bundle.py`
(reader); the result in `cli.py` (`_write_outputs`) and `R/bundle_read.R`.
A field added on one side must be added on the other.

## R side

| file | role |
|---|---|
| `reml.R` | `rx_reml()`: formulas to an `rx_model`, then `rx_fit()` |
| `formula_random.R`, `formula_residual.R`, `spl2d.R` | the asreml-style grammar of the random, residual and spline terms |
| `model.R`, `term.R`, `residual.R` | `rx_model()`, `rx_term()`, `rx_residual()`: the model object and its checks |
| `counts.R` | catalogues of structures and parameter counts (must agree with Python) |
| `lag.R` | spatial lag of the response |
| `fit.R` | `rx_fit()`: options, bundle, solver call, standard errors of theta |
| `bundle_write.R`, `bundle_read.R` | the bundle format, both directions |
| `python.R`, `launch.R`, `server.R` | locating Python, starting the solver, the persistent server |
| `print.R`, `summary.R` | printing and the asreml-style summary |
| `predict.R` | `rx_predict()`: the averaging matrix L in R, the algebra in Python |
| `ratios.R` | ratios with delta-method standard errors, AIC summary of a grid |
| `design.R` | neighbourhood incidences, exposure functionals, GRM |
| `scan.R`, `scan_loco.R` | genome-wide scans at fixed V |
| `spatial.R` | SpATS-style spatial outputs |
| `bayes.R` | the Bayesian route |
| `validate.R` | `rx_validate()` and the shipped validation tables |
| `remlax-docs.R` | all the Rd pages, as roxygen blocks on NULL |

Each user function reads as a sequence of named steps; the steps are internal
functions named `.rx_<verb>_<object>`. The NAMESPACE is written by hand and
roxygen2 only writes `man/*.Rd`.

## Python side

The engine knows incidence matrices and covariance structures, nothing about
formulas. A term is a dict (`name`, `struct`, `t`, `rank`, `q`, COO triplets
`zi`, `zj`, `zx`, `LK` or `Kinv`, level options `lvl`, `coord`, ...); the
residual is a dict with optional `sections`.

| layer | modules |
|---|---|
| structures | `structures.py` (Sigma from theta), `levels.py` (correlation between levels), `bessel.py` |
| model | `layout.py` (layout of theta), `incidence.py` (Z, supports, folded kinships, orthonormal X), `residual.py` (R), `model.py` (assembly of V, objective), `core.py` (REML from V with its gradient) |
| engines | dense (`model.py`), low-memory dense (`econome.py`), blocks (`blocs.py`, `decoupage.py`), mixed model equations (`mme.py`, `selinv.py`, `selinv_c.c`, `mme_hybride.py`) |
| fitting | `fit.py` (driver), `start.py` (starting values), `checks.py`, `ai.py` (AI-REML), `finition.py` (trust-region finish), `diagnostics.py`, `effects.py` (BLUPs and PEV at theta) |
| after the fit | `components.py`, `vpredict.py`, `fixed_tests.py`, `prediction.py`, `lag.py`, `bayes.py`, `scan.py` |
| process | `remlax_run.py` (launcher), `cli.py`, `scan_cli.py`, `serveur.py`, `bundle.py`, `device.py`, `fils.py`, `chrono.py`, `_x64.py` |

`fit.fit_reml` resolves the automatic choices and the prior weights, then
`_fit_model` runs: `_build_engine` (engine, fallbacks, objective),
`_start_theta` and `_check_start`, `_Objective` (compilation, counted and
memoised evaluations), `_bounds`, `_optimise` (L-BFGS-B or AI, with floor
escapes), the finish (`_check_ai_saddle`, `finition.finish` or
`_newton_polish`), and the result (`_result_core`, `_add_identification`,
`_add_lag_report`, `_add_degenerate`, `_add_sigmas`, `_add_effects`,
`_add_hessian_diagnostics`, `_dense_restart_if_needed`).

Every engine returns the same likelihood for the same theta. An engine that
cannot handle a model says why (`mme.out_of_scope`, `blocs.out_of_scope`)
and `fit.py` falls back to the dense engine.

## Invariants

- Column order of Z: column = trait index x q + level, level fastest, on both
  sides.
- Layout of theta: for each term the Sigma parameters then the level
  parameters, then each residual section, then the free entries of Psi.
  `layout.py` and `R/counts.R` must agree.
- The engines agree on -2logL to about 1e-9 relative; a larger gap is a bug.
- Result field names (`secondes`, `moteur`, `composantes_noms`, ...), command
  line flags and environment variables are part of the interface between R
  and Python and of user scripts.

## Where to change things

- A new Sigma structure: `structures.py` (`n_params`, `chol_sigma`,
  `theta0`), `R/counts.R`, the grammar in `formula_random.R`, and
  `ai.variance_scale_blocks` if it has a variance scale.
- A new level structure: `levels.py` (`LEVEL_STRUCTURES`, `level_corr`,
  `level_params_report`), `components._level_natural`, `R/counts.R`, and the
  start in `start._theta0_levels`.
- A new output of a fit: compute it in `fit.py` (or an engine), write it in
  `cli._write_outputs`, read it in `R/bundle_read.R`, print it in `R/print.R`.

## Checking a change

- `tests/testthat` (R, about 870 expectations; needs Python with JAX).
- `tests-python` (pytest, about 265 tests, 15 minutes on CPU).
- The safety nets of the cleanup, in remlax-validation/refacto: `golden.R`
  (36 fits through R, compared to the bit) and `golden_py.py` (engine values
  at fixed theta on every route, MME outputs and AI fits). Both pin the
  solver to one CPU core; XLA on CPU is not deterministic across runs
  otherwise.
- remlax-validation holds the campaigns against asreml, lme4, sommer, nlme and
  pbkrtest, and the simulation checks (`verif_simulations`).
