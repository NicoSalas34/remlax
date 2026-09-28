# Python API reference

The Python package knows nothing about formulas, data frames or factors. It
knows **incidence matrices** and **covariance structures**. Everything
model-specific is built on the R side and serialised, or built by hand as
plain dictionaries.

```
remlax/
  _x64.py        enables float64 at import time
  core.py        V -> -2 logL, with the analytic VJP
  structures.py  theta -> Sigma
  levels.py      theta -> correlation between levels
  bessel.py      Matern kernel for arbitrary nu
  model.py       assembly of V and of the objective
  fit.py         optimisation, BLUPs, diagnostics
  inference.py   vpredict, Wald, Kenward-Roger, predict
  bundle.py      reader for the serialised design
  device.py      CPU/GPU selection
  cli.py         command-line entry point
```

`import remlax` re-exports `STRUCTURES`, `n_params`, `build_sigma`,
`chol_sigma`, `Bundle`, `assemble_V`, `neg2_reml`, `make_objective`,
`pick_device`, `device_report`, `fit_reml`.

---

## 0. Two invariants

Break either of these and you fit a different model with no error message.

### Column order of Z

```
column of Z = (col - 1) * q + level          -> level varies FASTEST
```

`Z` is `n x (t*q)`: the `t` incidence matrices of a term are stacked
horizontally, each `n x q`. `Sigma (x) K` follows the same order, with `Sigma`
on the slow index. `R/remlax.R` builds `Z` with this convention. If one side
changes and the other does not, the model silently becomes a different one.

### Order of theta

```
theta = [ term_1 : Sigma params, level params ]
        [ term_2 : Sigma params, level params ]
        ...
        [ residual section 1 : Sigma params, level params ]
        [ residual section 2 : ... ]
```

`split_theta` performs this cut, and `component_names` numbers `V1, V2, ...`
in the same order. Any `vpredict` expression depends on it.

### The term dictionary

Every function below that takes a `term` expects a dictionary with these keys.
Only the first nine are mandatory.

| key | type | meaning |
|---|---|---|
| `name` | str | key used in every output |
| `struct` | str | one of `STRUCTURES` |
| `t` | int | number of columns of `Sigma` |
| `rank` | int | rank or band order (0 when unused) |
| `q` | int | number of levels |
| `zi`, `zj`, `zx` | int64, int64, float64 arrays | COO triplet of `Z`, 0-based |
| `LK` | (q,q) array or `None` | Cholesky factor of `K`; `None` means `K = I` |
| `lvl` | str | between-level structure; absent means `"fixed"` if `LK` is set, else `"id"` |
| `lvl_order` | int | order of `corb`, `corg` |
| `dims` | tuple | `(n_rows, n_cols)` for `ar1ar1` |
| `coord` | (q,1) or (q,2) array | coordinates for metric structures |
| `lvl_opts` | dict of floats | options (`mtrn`, `own`) |
| `lvl_expr` | str | expression for `own` |

A supplied `K` without an explicit `lvl` key defaults to `"fixed"`, never to
`"id"`. Falling back to `"id"` would silently ignore the relationship matrix —
that is a failure that actually happened, producing two runs labelled "kinship"
that were exact duplicates of the runs without.

The residual dictionary uses the same keys plus `trait` (0-based trait index
per observation), `unit` (0-based unit index per observation), `n_unit`, and
optionally `sections`, a list of per-section dictionaries each carrying `rows`.

---

## 1. `remlax.structures`

### `STRUCTURES`

Tuple of accepted `Sigma` structures:
`("iid", "diag", "us", "fa", "rr", "chol", "ante", "corh", "fixed")`.

### `n_loadings(t, r)`

`sum_{i=1..t} min(i, r)` — the number of free entries of a `t x r` lower
trapezoidal matrix. Used by `fa` and `rr`.

### `n_params(struct, t, rank=0)`

Number of free parameters of a structure. Returns `int`. Raises `ValueError`
on an unknown name.

```python
>>> from remlax.structures import n_params
>>> [n_params(s, 4, 2) for s in ("iid","diag","us","fa","rr","chol","ante","corh")]
[1, 4, 10, 11, 7, 9, 9, 5]
```

(`chol`/`ante` with `rank=2` give 9; the table in `docs/structures.md` uses
`rank=1`.)

### `chol_sigma(theta, struct, t, rank=0, fixed=None)`

Returns the **factor** `L` such that `Sigma = L L'`, shape `(t, t)` or
`(t, m)` with `m > t` for `fa`. Returning the factor rather than `Sigma` is
what lets `assemble_V` form `V += B B'` without ever building `Sigma` and
refactorising it, and what guarantees positive definiteness even when a
variance runs to zero.

`fixed` is used only by `struct="fixed"` and must be the `Sigma` matrix
itself.

```python
>>> import numpy as np, jax.numpy as jnp
>>> from remlax.structures import chol_sigma
>>> np.asarray(chol_sigma(jnp.array([np.log(1.2), 0.5, np.log(0.9)]), "us", 2))
array([[1.2, 0. ],
       [0.5, 0.9]])
```

### `build_sigma(theta, struct, t, rank=0, fixed=None)`

`L L'` from the above. Shape `(t, t)`.

```python
>>> from remlax.structures import build_sigma
>>> np.asarray(build_sigma(jnp.array([np.log(1.2), 0.5, np.log(0.9)]), "us", 2))
array([[1.44, 0.6 ],
       [0.6 , 1.06]])
```

### `theta0(struct, t, rank=0, var=1.0)`

Starting point such that `Sigma ~ var * I`. Returns a numpy array of length
`n_params(struct, t, rank)`.

```python
>>> from remlax.structures import theta0
>>> theta0("us", 2, var=3.0)
array([0.549306, 0.      , 0.549306])
```

`rr` is a special case: equal loadings would give a rank-1 `Gamma`, so the
starting values are graded (`linspace(0.6, 0.3, nl)`) to start from full rank.

---

## 2. `remlax.levels`

### `LEVEL_STRUCTURES`

Dictionary mapping name to parameter count; `None` means "variable, see
`n_level_params`". Keys:

```
['aexp', 'agau', 'ar1', 'ar2', 'ar3', 'arma', 'cir', 'cor', 'corb', 'corg',
 'exp', 'fixed', 'gau', 'id', 'ieuc', 'iexp', 'igau', 'lvr', 'ma1', 'ma2',
 'mtrn', 'own', 'sar', 'sph']
```

`ar1ar1` and `sep` are handled by the functions but are not in this dictionary.

### `LEVEL_NEEDS_COORD`

Names that require a `coord` argument:
`("exp", "gau", "iexp", "igau", "ieuc", "sph", "cir", "aexp", "agau", "mtrn", "lvr")`.

### `MTRN_PARAMS`, `MTRN_DEFAUT`

`("phi", "nu", "delta", "alpha")` — the **canonical order** of the Matern
parameters, which fixes their place in `theta` and in every output. Changing
it would invalidate any fit already recorded. `MTRN_DEFAUT` is
`{"phi": 1.0, "nu": 0.5, "delta": 1.0, "alpha": 0.0, "lambda": 2.0}`.

### `n_level_params(kind, order=0, parts=None, opts=None)`

Number of parameters of a between-level structure. `parts` is a list of
`(kind, order)` for a separable product, whose counts are summed. `opts`
carries structure settings: for `mtrn`, the `est_<param>` flags decide which
of the four are estimated (only `phi` defaults to estimated); for `own`,
`n_par`.

```python
>>> from remlax.levels import n_level_params
>>> n_level_params("ar2"), n_level_params("corb", 3), n_level_params("corg", 5)
(2, 3, 10)
>>> n_level_params("mtrn", opts={"est_phi": 1.0, "est_nu": 1.0})
2
>>> n_level_params("own", opts={"n_par": 2})
2
```

### `level_corr(theta_lv, kind, q, order=0, coord=None, C_fixed=None, parts=None, dims=None, opts=None, expr=None)`

The `q x q` correlation matrix. Unit diagonal by contract — the variance lives
in `Sigma`.

```python
>>> from remlax.levels import level_corr
>>> np.round(np.asarray(level_corr(jnp.array([np.arctanh(0.7)]), "ar1", 4)), 6)
array([[1.   , 0.7  , 0.49 , 0.343],
       [0.7  , 1.   , 0.7  , 0.49 ],
       [0.49 , 0.7  , 1.   , 0.7  ],
       [0.343, 0.49 , 0.7  , 1.   ]])
```

### `level_chol(theta_lv, kind, q, dims=None, LK_fixed=None, order=0, coord=None, parts=None, jitter=1e-10, opts=None, expr=None)`

Factor `L` with `L L' = C`. Returns **`None` for `kind="id"`**, meaning "no
product to perform" — callers must handle that. Three cases are kept in closed
form: `id` (`None`), `fixed` (the factor already computed on the R side, no
refactorisation) and `ar1`. Everything else goes through a numerical Cholesky
with a small jitter, because a banded correlation (`ma1`, `corb`) can be
singular at the edge of its domain.

### `ar1_chol(rho, n)`

Closed-form Cholesky factor of an AR(1) correlation:
`L[i,0] = rho^i`, `L[i,j] = rho^(i-j) sqrt(1-rho^2)` for `1 <= j <= i`. Exact,
differentiable, no `n x n` factorisation.

```python
>>> from remlax.levels import ar1_chol
>>> L = np.asarray(ar1_chol(0.7, 4)); float(np.max(np.abs(L @ L.T - C)))
1.11e-16
```

### `level_params_report(theta_lv, kind, order=0, opts=None, q=None)`

Parameters on the asreml scale, for reporting. Returns a dict of lists of
floats. The contract is that **the value returned is the value actually used in
`C`**, never an intermediate `theta`.

`q` is **required for `kind="cor"`** and ignored otherwise. The uniform
correlation is positive definite only on `(-1/(q-1), 1)`, so `tanh(theta)` is
not the correlation of the model; without `q` the function refuses rather than
reporting the wrong scale:

```python
>>> level_params_report(np.array([1.0]), "cor")
ValueError: level_params_report(kind='cor') exige q : la correlation uniforme est
remise a l'echelle sur (-1/(q-1), 1), donc tanh(theta) n'est pas la valeur du modele.
```

Keys by family:

```
ar1     -> {'phi': [0.664037]}
sar     -> {'phi': [0.664037]}
ma1     -> {'phi': [0.664037]}
ma2     -> {'phi': [0.462117, -0.291313]}
arma    -> {'phi': [0.462117, 0.53705]}
ar2     -> {'phi': [0.444141, 0.379949], 'pacf': [0.716298, 0.379949]}
ar3     -> {'phi': [0.519134, 0.467611, -0.197375], 'pacf': [0.716298, 0.379949, -0.197375]}
corb    -> {'phi': [0.462117, -0.197375]}
corg    -> {}
cor     -> {'phi': [0.850996], 'borne_inf': [-0.25]}
exp     -> {'phi': [0.761594], 'signe_non_identifie': [1.0]}
gau     -> {'phi': [0.761594], 'signe_non_identifie': [1.0]}
iexp    -> {'phi': [0.761594], 'signe_non_identifie': [1.0]}
igau    -> {'phi': [0.761594], 'signe_non_identifie': [1.0]}
ieuc    -> {'phi': [0.761594], 'signe_non_identifie': [1.0]}
aexp    -> {'phi': [0.761594, 0.462117], 'signe_non_identifie': [1.0]}
agau    -> {'phi': [0.761594, 0.462117], 'signe_non_identifie': [1.0]}
sph     -> {'portee': [4.481689]}
cir     -> {'portee': [4.481689]}
lvr     -> {'portee': [4.481689]}
mtrn    -> {'phi': [1.349859], 'nu': [1.105171]}
own     -> {'own': [0.4]}
ar1ar1  -> {'phi': [0.664037, 0.197375]}
```

`borne_inf` is `-1/(q-1)`, the exact positivity bound of `cor`.
`signe_non_identifie` marks the metric families, which raise `abs(tanh(theta))`
to a distance: `theta` and `-theta` give the same `C`, so the reported `phi` is
always positive and floored at `1e-12`. `own` returns the raw unconstrained
parameters, since only the expression knows what they mean. Full table with the
reasoning: [structures.md](structures.md#which-scale-each-family-reports).

---

## 3. `remlax.model`

### `term_n_params(tm)`

`n_params(Sigma) + n_level_params(between levels)` for one term dictionary.

### `dense_Z(term, n)`

Builds the dense `n x (t*q)` incidence from the COO triplet. Dense on
purpose: at the target sizes (`n <= ~2e4`, `t*q <= ~3e3`) a dense product on
GPU beats a sparse one by enough to make the choice obvious. This function and
`assemble_V` are the two places to change for a sparse solver.

### `term_factor(theta_k, term, Z)`

`B_k = Z_k (L_Sigma (x) L_K)`, so that `Z_k (Sigma (x) K) Z_k' = B_k B_k'`.
Never forms the Kronecker product, which would be `(t q)^2` and unstorable
beyond a few thousand levels. `L_K` is rebuilt here at each evaluation when it
depends on parameters (an AR1 field), and is differentiated like the rest.

### `res_sections(res, n)`

The residual sections, **always as a list**. A plain residual becomes a single
section covering all rows. Returning the same shape in both cases avoids the
duplicated code path that would otherwise start to diverge.

### `sec_n_params(sec)`, `res_n_params(res)`, `n_theta(terms, res)`

Parameter counts for one section, for the whole residual, and for the whole
model.

### `residual_V(theta_r, res, n)`

The full `R`. With sections it is a direct sum (asreml's `dsum`), assembled by
scatter into an `n x n` matrix rather than by concatenation: the rows of a
section have no reason to be contiguous, and an implicit sort would match the
wrong rows of `y` without any warning.

### `split_theta(theta, terms, res)`

Returns `(list_of_term_thetas, residual_theta, offset_consumed)`.

### `assemble_V(theta, terms, Zs, res, n)`

`V = sum_k B_k B_k' + R`. `Zs` is the list of dense incidences from
`dense_Z`. Positive definiteness of `V` is guaranteed by construction, even
far from the optimum.

### `neg2_reml(theta, terms, Zs, res, y, X)`

`-2 logL_REML` including the `(n-p) log(2 pi)` constant.

### `make_objective(bundle_terms, res, y, X, scale=None)`

Returns `(fun_jac_scaled, fun_jac, sc, Zs)`.

- `fun_jac_scaled(theta) -> (float, ndarray)` — objective and gradient
  **divided by n**. This is what the optimiser sees.
- `fun_jac(theta) -> (float, ndarray)` — the same on the natural `-2 logL`
  scale, for diagnostics and outputs.
- `sc` — the scale factor used (`n` unless `scale` is given).
- `Zs` — the dense incidences, reusable.

The scaling is load-bearing, not cosmetic. `-2 logL` is of order `n` and so is
its gradient; L-BFGS-B starts with an approximate Hessian equal to the
identity, so its first step is `-gradient`, hundreds of units in parameter
space. Measured on a one-factor model without effective scaling: it stopped at
`-2 logL = 425.15` with `|grad| = 74.6`, against `414.3095` at the optimum.

---

## 4. `remlax.fit`

### `initial_theta(terms, res, y)`

Starting values: the phenotypic variance split equally between the terms and
the residual. **Exposure is taken into account**: a neighbourhood component
enters the phenotype as `s2 * k` with `k` the mean of `(Z K Z')_ii`, which can
reach 15 to 40 on some designs; dividing by `k` prevents `V` from starting
orders of magnitude too large. Between-level parameters start at zero
(`rho = 0`, unstructured), except range parameters, which start at a quarter
of the coordinate span, where the likelihood still has slope.

### `validate(terms, res, y, X)`

Raises `ValueError` on designs that would make `V` singular. Returns `None`
when everything is fine. The checks are:

- duplicated `(unit, trait)` pairs under any structured residual — the two
  rows would be perfectly correlated, `R` singular, `-2 logL` NaN;
- a trait index outside the declared residual `t`;
- an entirely null incidence for a term;
- non-finite values in `y` or `X`;
- `X` not of full column rank.

### `fit_reml(terms, res, y, X, theta_init=None, maxiter=3000, floor=-12.0, ceil=12.0, verbose=True, hessian=True, blups=True, tol=1e-14, polish=25, check=True, n_restarts=0, restart_sd=0.5, seed=0, fixed_idx=None)`

The fit. Arguments:

| argument | meaning |
|---|---|
| `terms`, `res`, `y`, `X` | the model |
| `theta_init` | starting vector; `initial_theta` if `None` |
| `maxiter` | L-BFGS-B iteration cap (`maxfun = 10 * maxiter`) |
| `floor`, `ceil` | wide bounds on every `theta`. Not needed for positivity — they only stop a variance escaping to `exp(-inf)`, where the gradient vanishes exponentially and the point becomes indistinguishable from an optimum |
| `verbose` | progress every 25 iterations, and warnings |
| `hessian` | compute the finite-difference Hessian and the diagnostic |
| `blups` | compute `beta`, `vbeta`, `Vi`, `Py` and the BLUPs |
| `tol` | `ftol` of L-BFGS-B |
| `polish` | maximum number of regularised Newton steps after L-BFGS-B |
| `check` | run `validate` first |
| `n_restarts` | perturbed restarts; the best optimum is kept |
| `restart_sd` | standard deviation of the perturbation |
| `seed` | RNG seed for the restarts |
| `fixed_idx` | 0-based indices held at their starting value |

Returns a `dict`. Keys, on a two-term model:

| key | contents |
|---|---|
| `theta` | the estimate, internal scale |
| `neg2_reml`, `logLik` | `-2 logL` and `logL`, constant **included** |
| `logLik_asreml`, `const_2pi` | asreml convention (constant removed) and the constant |
| `n_par`, `n_obs` | counts |
| `max_grad` | `max |grad|` on the natural scale |
| `newton_decrement` | `g' H^+ g`, the ascent still available — the stopping criterion that matters |
| `leak` | gradient component in the null space of `H` |
| `n_neg_eig`, `n_null_dir`, `lambda_min`, `lambda_max`, `cond` | spectrum of the free-subspace Hessian |
| `n_at_bound` | parameters at `floor`/`ceil`, **excluding** fixed ones |
| `n_fixed`, `n_fixed_out` | parameters held by `fixed_idx` (same count, two keys) |
| `n_par_free` | dimension of the actual free subspace |
| `composantes_degenerees` | names of terms with a parameter at the floor |
| `n_polish`, `n_iter`, `scipy_success`, `scipy_message`, `secondes` | optimiser trace |
| `sigmas` | dict name -> `Sigma` matrix |
| `sigmas_res`, `sigma_res` | per-section residual matrices, and the first one |
| `rho` | dict of between-level parameters on the asreml scale |
| `pacf` | partial autocorrelations for `ar2`/`ar3` |
| `beta`, `vbeta`, `Vi`, `Py` | fixed effects and the pieces needed downstream |
| `V_singuliere` | `True` if `V` had to be pseudo-inverted; `beta` and the BLUPs are then conditional |
| `blups` | dict name -> `(q, t)` array |
| `hessian` | the finite-difference Hessian of `-2 logL` |
| `n_restarts`, `restart_gain`, `restart_better` | present only when `n_restarts > 0` |

```
neg2_reml          822.0106277293249
logLik             -411.00531386466247
logLik_asreml      -191.3790044287457
const_2pi          219.62630943591677
n_par              3
n_obs              240
max_grad           5.888075907683343e-13
newton_decrement   5.470948439833614e-28
n_neg_eig          0
n_null_dir         0
n_at_bound         0
cond               4562.749084175374
n_polish           2
n_iter             18
scipy_success      True
V_singuliere       False
composantes_degenerees []
sigmas             {'gid': 1.5847079094518273, 'bloc': 0.002430569370277488}
sigma_res          1.1138165870456136
beta               [11.658782]
blups shapes       {'gid': (60, 1), 'bloc': (4, 1)}
```

**Do not read `max_grad` as a convergence criterion.** It is several units at
verified optima. `newton_decrement` is the quantity to read: it is the
remaining rise in log-likelihood under the local quadratic model, in the same
units as the 3.84 of a one-degree-of-freedom LRT, and it is invariant under
affine reparametrisation.

**`composantes_degenerees` is not a warning about a small variance.** A
variance that reaches the floor is not "estimated at zero", it is
**unidentified**, and its standard error, its variance share and any test
touching it are meaningless. The typical case is AR1/nugget aliasing, where
the field absorbs the whole residual; asreml refuses to converge there, and
remlax converges, so it has to say so.

**Restarts are the only way to detect a local optimum.** The Newton decrement
measures the ascent available *locally* and is zero at the top of a secondary
hill. `n_restarts > 0` perturbs the retained point and keeps the best; the
gain is reported in `restart_gain`.

**Range kernels (`sph`, `cir`, `lvr`) get a swept start.** When `theta_init`
is not supplied, `-2 logL` is evaluated over the deciles of the pairwise
distances of the design for each range parameter, the three best candidates
receive a short L-BFGS-B descent, and the fit starts from the best point
(`[portee]` line in the log). The likelihood in a range has a kink at every
distinct distance and often several hills; see `docs/structures.md`. A start
whose value or gradient is not finite raises `ValueError`.

**`fixed_idx` removes a parameter from the free subspace without putting it at
a bound.** The diagnostic receives `fixed_idx` and excludes those coordinates,
so the Newton decrement is computed on the directions the step can actually
take. Measured on the model above, with and without the first parameter fixed:

```
libre          n_par 3 | n_at_bound 0 | n_fixed 0 | n_fixed_out 0 | n_par_free 3
               max_grad 5.888076e-13 | newton_decrement 5.471e-28 | logLik -411.005314
fixed_idx=[0]  n_par 3 | n_at_bound 0 | n_fixed 1 | n_fixed_out 1 | n_par_free 2
               max_grad 5.117183e+01 | newton_decrement 4.510e-14 | logLik -414.682675
```

`max_grad` is 51.2 in the fixed case: that is the gradient along the pinned
direction, large and irrelevant. The decrement stays at `4.5e-14` and correctly
reports the two free parameters as converged. A pinned coordinate left inside
the free subspace would have made the decrement announce an optimum not
reached where it was.

The full key list:

```
['Py', 'V_singuliere', 'Vi', 'beta', 'blups', 'composantes_degenerees', 'cond',
 'const_2pi', 'hessian', 'k_eff', 'lambda_max', 'lambda_min', 'leak', 'logLik',
 'logLik_asreml', 'max_grad', 'n_at_bound', 'n_fixed', 'n_fixed_out', 'n_iter',
 'n_neg_eig', 'n_null_dir', 'n_obs', 'n_par', 'n_par_free', 'n_polish',
 'neg2_reml', 'newton_decrement', 'rho', 'scipy_message', 'scipy_success',
 'secondes', 'sigma_res', 'sigmas', 'sigmas_res', 'theta', 'vbeta']
```

---

## 5. `remlax.inference`

### `component_names(terms, res)`

The names of the components, in the order `vpredict` numbers them `V1, V2,
...`. A single-column term gives one name; a `t`-column term gives the lower
triangle of `Sigma`, row by row, as `name[i,j]`; between-level parameters
follow as `name!kind` (or `name!kind1`, `name!kind2`, ... when there are
several). The residual comes last.

```python
>>> component_names([t_gid, t_bloc], res)
['gid', 'bloc', 'residuelle']
```

**The number of components is not the number of parameters.** The lower
triangle of `Sigma` is enumerated whatever the structure, because those are
the quantities a `vpredict` expression refers to. A `diag` residual on three
traits has 3 free parameters but contributes 6 component names, three of which
are structurally zero.

### `components_from_theta(theta, terms, res)`

The component vector as a JAX array, in the order above. This is the function
the delta method differentiates, so it is written in JAX end to end.

### `vpredict(theta, H, terms, res, expressions, free=None)`

Functions of the variance components and their standard errors.
`expressions` is a list of `(name, string)` where the string uses `V1, V2,
...` and the usual Python operators, plus `sqrt`, `log`, `exp`, `abs`.

`H` is the Hessian of `-2 logL`, so the asymptotic covariance of `theta` is
`2 H^-1` and `var(g) = J' (2 H^-1) J`. `free` is a boolean mask excluding
fixed parameters.

Returns `{"composantes": [{"i", "nom", "valeur"}, ...],
"predictions": [{"nom", "expression", "valeur", "se"}, ...]}`.

```python
>>> vp = vpredict(fit["theta"], fit["hessian"], terms, res,
...               [("h2", "V1/(V1+V2+V3)"), ("vg", "V1")])
h2   V1/(V1+V2+V3)    0.586721  se 0.060482
vg   V1               1.584708  se 0.344311
```

Standard errors are conditional on the Hessian being invertible over the free
subspace. A parameter at an active bound has none, and a direction of zero
curvature makes the delta method silent rather than wrong: `se` is `NaN`,
never an invented number.

### `wald(y, X, V, termes_fixes=None)`

Wald tests on the fixed effects at **fixed `V`**: `var(beta) = (X'V^-1X)^-1`,
and for a group of columns `L`,
`F = (L b)' [L var(b) L']^-1 (L b) / rank`.

`termes_fixes` is a list of `(name, column_indices)`; without it, every column
is tested on its own, which is meaningless as soon as a factor has more than
two levels.

Returns `{"beta", "se_beta", "tests"}` where each test is
`{"terme", "ddl", "F", "chi2", "p"}`.

This is the **conditional** test (each term given all the others), that is,
type III. asreml's default is sequential (type I). The two coincide on the
last term of the model. Denominator degrees of freedom are not adjusted, so
the p-values are those of a `chi2/rank` and are asymptotic.

### `kenward_roger(theta, terms, res, y, X, H=None, termes_fixes=None)`

The Kenward & Roger (1997) adjustment: an inflated covariance of `beta` and
denominator degrees of freedom.

Returns, when available,
`{"disponible": True, "tests", "se_beta", "se_beta_brut", "vbeta_kr", "beta",
"second_ordre_omis"}`; each test carries `{"terme", "ddl", "denDF", "F",
"F_brut", "echelle", "p"}` and possibly a `note`. When the expected REML
information is not positive definite it returns
`{"disponible": False, "raison": ...}` rather than a number.

```
[{'terme': '(Intercept)', 'ddl': 1, 'denDF': 39.08261264539447,
  'F': 4293.294813883976, 'F_brut': 4293.2948138839765,
  'echelle': 0.9999999999999998, 'p': 1.406367256974862e-41}]
```

Three implementation points that are easy to get wrong:

- `W` is the inverse of the **expected** REML information
  `1/2 tr(P dV_i P dV_j)`, not of the observed Hessian. They agree
  asymptotically but not at the sample sizes where the adjustment matters.
- `A1` and `A2` contract `Theta` against `dPhi/dtheta = Phi P Phi`, not
  against `P`. With `P` they come out of order `n^3` and the degrees of
  freedom go **negative**.
- On an orthogonal design `A1` and `A2` vanish exactly; the code falls back to
  the unadjusted test with residual degrees of freedom rather than computing
  `0/0`, and says so in `note`.

The second-order term is omitted; `second_ordre_omis` is `True` when the model
contains between-level correlation parameters, for which the omission is not
exact. Cost is `O(p_theta^2 n^2)`: the first derivatives of `V` are formed in
full, one `n x n` matrix per parameter.

### `predict(theta, terms, res, y, X, L, M=None, vbeta=None)`

Predictions `p = L' beta (+ M' u)` and their **prediction** error covariance.

`L` is `l x p`, already averaged on the R side over the factors absent from
the `classify`. `M` is a dict `term_name -> (l, t*q)`; without it the
prediction carries fixed effects only, which is right when the random term is
averaged out and wrong when a genotype mean is asked for, since that contains
its BLUP.

The variance is not `var(p_hat)` but `var(p_hat - p)`:

```
L' Phi L  -  2 sym( L' Phi X'V^-1 Z G M )  +  M'G M - M'G Z' P Z G M
```

All three terms matter, and the middle one is **negative**: the correlation
between `beta_hat` and `u_hat` reduces the error of the total. No `G^-1` and
no Kronecker product is formed; `G = Sigma (x) K` enters only through its
action on `M`.

```python
>>> predict(fit["theta"], terms, res, y, X, L=np.ones((1, 1)), vbeta=fit["vbeta"])
{'valeur': [11.658782], 'se': [0.177934], ...}
>>> # with the BLUPs of the first two genotypes
{'valeur': [10.413671, 12.565078], 'se': [0.487996, 0.487996], ...}
```

---

## 6. `remlax.bundle`

### `Bundle(path)`

Reader for a design serialised by `rx_export()`: a directory containing
`manifest.json` plus one `<name>.bin` (raw binary, column-major) or
`<name>.txt` (one string per line) per array. Raw binary rather than `.npz` so
that the R side needs no Python dependency; column-major because that is R's
native order.

| member | returns |
|---|---|
| `.has(name)` | whether an array is present |
| `.get(name)` | the array, reshaped from the manifest (`order="F"`) |
| `.y` | response, `(n,)` float64 |
| `.X` | design matrix, `(n, p)` float64 |
| `.n` | number of observations |
| `.fixed_groups()` | `[(term name, column indices)]` from `X_assign`, or `None` |
| `.terms()` | list of term dictionaries, in R declaration order |
| `.residual()` | residual dictionary, with `sections` when `res_nsec > 1` |

`.residual()` checks that the sections partition the observations: a forgotten
row would leave `V` with a zero variance, a duplicated one would count it
twice.

---

## 7. `remlax.device`

### `pick_device(prefer="auto")`

Returns `(device, platform_name)`. `"auto"` takes a GPU if JAX sees one, the
CPU otherwise. `"gpu"` **raises** rather than falling back silently.
`"cpu"` raises if JAX exposes no CPU.

`jax.devices()` **without an argument** returns only the default platform's
devices: as soon as a GPU is visible it lists no CPU at all. Filtering its
output on `platform == "cpu"` therefore returned an empty list, and
`--backend=cpu` fell back silently to the GPU. `jax.devices("cpu")` must be
asked for explicitly.

### `device_report()`

Every device, CPU included, as `[{"platform", "kind"}, ...]`.

```python
>>> device_report()
[{'platform': 'cpu', 'kind': 'cpu'},
 {'platform': 'gpu', 'kind': 'NVIDIA RTX A1000 6GB Laptop GPU'}]
```

---

## 8. `remlax.core`

### `reml_from_V(V, y, X)`

`-2 logL_REML = (n-p) log(2 pi) + log|V| + log|X'V^-1X| + y'Py`, as a
`jax.custom_vjp`. The backward pass uses the analytic gradient

```
d(-2 logL)/dV = P - (Py)(Py)',   P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1
```

so it needs neither the Cholesky band nor the triangular solves that automatic
differentiation would memorise: three `n x n` matrices live at once instead of
about ten. That is what makes `n` of order 16 000 fit in a 10 GB GPU slice.

Because the rest of the package only ever sees a differentiable function of
`V`, **any** smooth parametrisation of `V` gets its gradient for free.

`BWD_CHUNK` (environment variable `REML_BWD_CHUNK`, default 0) enables a
column-blocked backward pass. It is kept only to document the dead end:
measured at `N = 16211` on a MIG slice, chunking used 8.11 GB against 7.34 GB
unchunked, because `fori_loop` carries the full matrix in its state on top of
the blocks. Leave it at 0.

---

## 9. `remlax.bessel`

### `matern(z, nu)`

`M(z, nu) = 2^(1-nu)/Gamma(nu) z^nu K_nu(z)`, with `M(0, nu) = 1`.
Differentiable in both arguments. This is the Matern **correlation**, not
`K_nu`: `z^nu K_nu(z)` is regular at `z = 0` whereas `K_nu(z)` diverges, and
computing `K_nu` then multiplying would put `0 * inf` on the diagonal of every
correlation matrix.

### `z_nu_kv(z, nu)`

`z^nu K_nu(z)`, computed in logs so `K_nu` is never formed (at `nu = 5`,
`z = 1e-8` it is about `1e41` while the product stays of order 1). `z` is
floored at `Z_MIN = 1e-12`.

Constants: `U_MAX = 30.0`, `H_PAS = 0.15`, `Z_MIN = 1e-12`. The trapezoidal
error behaves like `exp(-pi^2/h)`, which at that step is `exp(-65.8)` =
2.7e-29 - thirteen orders of magnitude below `eps` = 2.2e-16, not three as
the source docstring said before this was checked. The grid is therefore
fixed and needs no error control.

---

## 10. `remlax.cli`

```
python -m remlax.cli <bundle_dir> [options]
remlax <bundle_dir> [options]          # console script from pyproject.toml
```

Reads a serialised design, fits, and writes results back into the **same**
directory: `result.json` for scalars and dimensions, `out_*.bin` for arrays
(column-major float64). The output format mirrors the input format exactly, so
R needs no Python dependency and Python needs no R dependency.

| option | default | meaning |
|---|---|---|
| `--backend auto\|gpu\|cpu` | `auto` | device; `gpu` refuses if none is visible |
| `--maxiter N` | 3000 | L-BFGS-B cap |
| `--polish N` | 25 | Newton polishing steps |
| `--restarts N` | 0 | perturbed restarts |
| `--restart-sd X` | 0.5 | perturbation size |
| `--floor X`, `--ceil X` | -12, 12 | bounds on `theta` |
| `--no-hessian` | off | skip the Hessian and the diagnostic |
| `--no-blups` | off | skip `beta`, BLUPs and `Vi` |
| `--quiet` | off | no progress output |
| `--vpredict "n1=e1;n2=e2"` | — | expressions on the components |
| `--wald` | off | Wald tests on the fixed effects |
| `--kenward-roger` | off | KR degrees of freedom and adjusted covariance |
| `--predict` | off | read `pred_L.bin` (and `pred_M_<term>.bin`) and predict |
| `--only-predict` | off | **no fit**: read `in_theta.bin` and predict only |
| `--fixed-theta i,j` | — | 1-based indices held at their starting value |

`--only-predict` exists so that `rx_predict()` does not relaunch the whole
optimisation to recover a `theta` that is already known — and so that nothing
depends on the optimiser landing on the same optimum twice.

```
$ python -m remlax.cli bundle_demo --backend cpu --vpredict 'h2=V1/(V1+V2)'
[remlax] peripheriques : [{'platform': 'cpu', 'kind': 'cpu'}, {'platform': 'gpu', 'kind': 'NVIDIA RTX A1000 6GB Laptop GPU'}]
[remlax] backend retenu : cpu (cpu:0)
[remlax] n=120, 1 effet(s) fixe(s), 1 terme(s) aleatoire(s)
          gid            iid   t=1   q=30    K=I
[remlax] logLik = -199.430646551 | 2 parametres | 1.4 s
```

Files written:

```
['out_beta.bin', 'out_blup_gid.bin', 'out_hessian.bin', 'out_sigma_gid.bin',
 'out_sigma_res.bin', 'out_sigmares_residuelle.bin', 'out_theta.bin',
 'out_vbeta.bin', 'result.json']
```

Non-finite floats are written as JSON `null`, which R reads as `NA`. JSON has
no `NaN`, and a conforming reader such as jsonlite rejects the **whole** file
if one appears — an unavailable diagnostic would otherwise lose the entire fit.
