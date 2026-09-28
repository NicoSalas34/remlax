# remlax documentation

A generic REML solver for linear mixed models, written in JAX, driven from R
by an asreml-like formula interface. The same model runs on CPU or GPU without
being rewritten.

```
y = X b + sum_k Z_k u_k + e,    u_k ~ N(0, Sigma_k (x) K_k),    e ~ N(0, R)
```

See the [README](../README.md) for what the solver is, what it is not,
installation, and citation.

---

## Where to start

**If you have used lme4 or asreml**, read
[1. Getting started](guide/01-getting-started.md), then jump to
[api-r.md](api-r.md) for the formula grammar and the asreml name mapping.

**If you want to know whether a particular structure exists**, go straight to
[structures.md](structures.md) — it lists every one with its formula,
parameter count and constraints.

**If you are calling the solver from Python**, read
[api-python.md](api-python.md), starting with the two invariants at the top:
the column order of `Z` and the order of `theta`.

**If your model does not fit in a formula**, read
[4. Explicit terms](guide/04-explicit-terms.md).

---

## Guides

Five step-by-step guides. Every code block was executed and its real output
pasted underneath. The examples are small (a few hundred observations) and
each fit takes a couple of seconds, so they are quick to replay.

| | contents |
|---|---|
| [1. Getting started](guide/01-getting-started.md) | one random factor; reading the output and the diagnostic; heritability with `vpredict`; BLUPs; Wald and Kenward-Roger on an unbalanced design; predictions; the same model in Python |
| [2. Multi-trait models](guide/02-multi-trait.md) | long format; a genomic relationship matrix; `us` genetic and residual covariances; component numbering; heritabilities and genetic correlation; parameter counts of `diag`/`us`/`fa`/`rr`/`chol`/`ante`/`corh`; reduced rank, and an aliasing that looks like a model test |
| [3. Spatial models](guide/03-spatial.md) | separable AR1 as a residual and as a random term; the nugget that vanishes; metric kernels; Matern and its parameter-declaration convention; the duplicated-cell guard; user-defined correlations with `own()`; two-dimensional splines |
| [4. Explicit terms](guide/04-explicit-terms.md) | why a formula is not enough; `rx_term` with a list of weighted incidences; a DGE/IGE model with a shared covariance; the same model through `str()`; BLUPs of both effects; functions of the shared covariance |
| [5. Chapter-3 reproduction](guide/05-chapter3-reproduction.md) | a two-species neighbourhood design end to end with `rx_neighbourhood`, `rx_exposure`, `rx_ratios`, `rx_cor_z` and `rx_grid_summary`; the analysis -> script -> figure table; the numerical conventions; the measured differences to the original pipeline |

---

## Reference

| | contents |
|---|---|
| [api-r.md](api-r.md) | every `rx_*` function; the complete formula grammar for random terms and residuals, as tables with parameter counts; the mapping to asreml names (`idv`, `idh`, `corgh`, `ar1v`, `ar1h`, ...); how to read a fit |
| [api-python.md](api-python.md) | every public function of `structures`, `levels`, `model`, `fit`, `inference`, `bundle`, `device`, `core`, `bessel`, `cli`: exact signature, arguments, return value, and the invariants that would silently change the model |
| [structures.md](structures.md) | the catalogue: covariance structures for `Sigma` and correlation structures between levels, with formulas, parameter counts, positivity constraints and the parametrisation used; the deliberate divergences from the ASReml-R manual |

---

## Results

| | contents |
|---|---|
| [validation.md](validation.md) | agreement against asreml, lme4, sommer, pbkrtest, closed-form REML, finite-difference gradients, the random stress sweep, and CPU/GPU parity |
| [benchmarks.md](benchmarks.md) | timings and memory across model sizes on CPU and GPU, from `benchmarks/bench.py` |

Those two pages are written separately from the cluster runs; they are not
part of this documentation set.

---

## Design note

[note_remlax_fr.md](note_remlax_fr.md) (French) is the design note: why the
likelihood is parametrised the way it is, why the gradient is analytic in `V`,
how each structure was identified against asreml, what the stress sweep found,
and what is deliberately left out. It is the source of truth for everything
above.

---

## Conventions that matter

Three of them, repeated here because getting any of them wrong produces a
different model with no error message.

**Column order of `Z`.** `column = (a - 1) * q + level`, with the level
varying fastest; `Sigma (x) K` follows the same order, `Sigma` on the slow
index. R and Python share the convention.

**Order of `theta`.** For each term: the `Sigma` parameters, then the
between-level parameters. Terms in declaration order, then the residual, then
each `dsum` section in turn. `fit$composantes_noms` reflects it, and every
`vpredict` expression depends on it.

**Row order in a separable product.** `level = (row - 1) * n_col + col`: row
slowest, column fastest.

---

## Two numbers to read, and one to ignore

`newton_decrement` is the convergence criterion: the rise in log-likelihood
still available under the local quadratic model, comparable to the 3.84 of a
one-degree-of-freedom LRT. `max_grad` is **not** a convergence criterion — it
is several units at verified optima.

`composantes_degenerees` names components whose variance sits at the floor.
Those are not "estimated at zero", they are unidentified, and every standard
error or test touching them is void.

`logLik` includes the `(n-p)/2 log(2 pi)` constant, like lme4. Compare with
asreml using `logLik_asreml`.

---

## Reported scales

`fit$rho` reports, for every structure, the value **actually used** in the
correlation matrix — never an intermediate `theta`. Three keys tell you which
scale you are reading: `portee` for the range structures (`sph`, `cir`,
`lvr`), `borne_inf` alongside a `cor` correlation already rescaled onto its
positive-definite interval, and `signe_non_identifie` on the metric families,
whose reported `phi` is always positive because `theta` and `-theta` give the
same model. The per-family table is in
[structures.md](structures.md#which-scale-each-family-reports).

Three counters describe the parameter space, and they are not
interchangeable: `n_at_bound` (at `floor`/`ceil`), `n_fixed_out` (held by
`fixed_theta`, outside the free subspace without being at a bound), and
`n_par_free` (what remains). The Newton decrement is computed on the free
subspace only.
