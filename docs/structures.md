# Structure catalogue

Every random term in remlax carries two structures:

```
u_k ~ N(0, Sigma_k (x) K_k)
       \_______/   \____/
       covariance   correlation
       over the t   between the
       columns      q levels
```

`Sigma` holds the **variances**; the between-level structure contributes a
**correlation matrix only**. This is the `Sigma_h = D C D` decomposition of
appendix C of the ASReml-R 4.2 manual, and it is why an asreml `ar1v` is
written here `ar1(f)` (`Sigma = s2`) and an `ar1h` is `ar1(f, struct = "diag")`.

Every parametrisation is **unconstrained**: the optimiser works on `R^p`
without bounds, and positive definiteness is guaranteed by construction. Two
devices do that work. A Cholesky factor is parametrised rather than the matrix
itself, so no eigenvalue can go negative; and a scalar that must live in an
interval is passed through `tanh` (a correlation) or `exp` (a variance, a
range, a shape).

The two reference implementations are `src/remlax/structures.py` (Sigma) and
`src/remlax/levels.py` (between levels). `R/remlax.R` reproduces the
parameter counts and must agree exactly: two different counts would split
`theta` at the wrong place, silently.

---

## 1. Covariance structures for Sigma

`w` is the number of columns of `Sigma`, `k` a rank or band order, and

```
n_loadings(w, k) = sum_{i=1..w} min(i, k)
```

| name | asreml | Sigma | free parameters | parametrisation |
|---|---|---|---|---|
| `iid` | `idv` | `exp(2 th) I_w` | 1 | `th` = log standard deviation |
| `diag` | `idh` | `diag(exp(2 th_j))` | `w` | `th_j` = log standard deviation |
| `us` | `corgh` | `L L'`, `L` lower triangular | `w(w+1)/2` | diagonal of `L` = `exp(th)`, off-diagonal free |
| `fa(k)` | `fa(k)` | `Lam Lam' + diag(psi^2)` | `n_loadings(w,k) + w` | `Lam` lower trapezoidal, free; `psi = exp(th)` |
| `rr(k)` | `rr(k)` | `Gam Gam'`, rank `k` | `n_loadings(w,k)` | `Gam` lower trapezoidal, free |
| `chol(k)` | `chol(k)` | `L D L'`, `L` unit lower triangular of band `k` | `(k+1)(w - k/2)` | band entries free, `D = diag(exp(2 th))` |
| `ante(k)` | `ante(k)` | `Sigma^-1 = U D U'`, `U` unit upper triangular of band `k` | `(k+1)(w - k/2)` | band entries free, `D = diag(exp(2 th))` |
| `corh` | `corh` | `D C D`, `C` uniform correlation | `w + 1` | `D = diag(exp(th))`, correlation bounded (below) |
| `fixed` | — | supplied matrix | 0 | Cholesky of the given `Sigma` |

Counts, computed by `remlax.structures.n_params`:

```
t        iid     diag    us      fa(2)   rr(2)   chol(1)  ante(1)  corh
2        1       2       3       5       3       3       3       3
4        1       4       10      11      7       7       7       5
6        1       6       21      17      11      11      11      7
8        1       8       36      23      15      15      15      9
```

Two consequences of that table are worth stating, because they cost time when
discovered by accident.

- **`fa(k)` is not always cheaper than `us`.** `fa` pays `w` extra specific
  variances on top of its loadings, so at `w = 4` a rank-2 factor-analytic
  model has 11 parameters against 10 for the unstructured one. The saving
  starts at `w = 5` for `k = 2`, and grows from there.
- **`rr(k)` is the reduced-rank structure that actually saves parameters**, at
  the price of a singular `Sigma`. That is legitimate here because `V` is
  assembled as `B B'`: a rank-deficient component does not make `V` singular
  as long as another term or the residual completes it.

### rr(k): a trapezoidal Gamma, and why the count differs from the manual

The manual gives `k*w` parameters for `rr(k)`. remlax counts
`n_loadings(w, k)`, which is smaller by `k(k-1)/2`, because `Gamma` is
constrained to be **lower trapezoidal**. Without that constraint `Gamma` is
defined only up to a `k x k` rotation: the likelihood has a flat ridge of
dimension `k(k-1)/2` and the Hessian is singular by exactly that amount. The
fitted model is identical; only the parameter count, the Hessian and every
standard error derived from it change. The same constraint is applied to `fa`.

### Saturated parametrisations

`chol(w-1)`, `ante(w-1)` and `rr(w)` all have `w(w+1)/2` parameters and
represent the same set of matrices as `us`:

```
us       10 parametres
chol(3)  10 parametres
ante(3)  10 parametres
rr(4)    10 parametres
fa(4)    14 parametres
```

`fa(w)` also spans every PD matrix but carries `w` redundant parameters
(`n_loadings(w,w) + w = w(w+1)/2 + w`); it reaches the same optimum with a
Hessian that is singular in `w` directions.

### An aliasing that looks like a model comparison

`us` and `rr(k)` differ by a diagonal. If the residual already carries a free
diagonal — `residual = ~ diag(trait):units` — then that difference is
absorbed, and the two models reach the **same** likelihood with different
parameter counts. Measured on a four-trait design with 70 genotypes:

```
us     : logLik  -457.8032, 14 parametres
rr(2)  : logLik  -457.8032, 11 parametres
LRT    : chi2 = -0.0000, 3 ddl, p = 1.0000
```

A likelihood-ratio test between them is meaningless: the three extra
parameters of `us` are aliased with the residual diagonal, not merely
unhelpful. `Sigma_us - Sigma_rr` is exactly diagonal here.

### corh: the bound on the uniform correlation

`corh` combines heterogeneous variances with a single correlation `r`. A
uniform correlation matrix `(1-r) I + r J` is positive definite only for
`r > -1/(w-1)`, so `tanh` is rescaled onto `(-1/(w-1), 1)` rather than
`(-1, 1)`:

```
theta  -5.0  r = -0.333273  min eig 1.8159e-04  (borne -1/3 = -0.333333)
theta   0.0  r =  0.333333  min eig 6.6667e-01
theta   5.0  r =  0.999939  min eig 6.0530e-05
```

Note the value at `theta = 0`: it is `1/3`, not `0`, because the mapping sends
the midpoint of `(-1, 1)` to the midpoint of `(-1/3, 1)`.

---

## 2. Correlation structures between levels

All of them return a correlation matrix `C` with unit diagonal. Every
stationary structure is Toeplitz, so remlax computes the autocorrelation
function `rho_0 .. rho_{q-1}` and expands `C[i,j] = rho_{|i-j|}`.

| name | `C` or `rho_k` | parameters | parametrisation |
|---|---|---|---|
| `id` | `I` | 0 | — |
| `fixed` | supplied (GRM, pedigree, computed kernel) | 0 | Cholesky supplied by R |
| `cor` | `C_ij = r` off-diagonal | 1 | `tanh` rescaled to `(-1/(q-1), 1)` |
| `ar1` | `rho_k = phi^k` | 1 | `phi = tanh(th)` |
| `ar2` | Yule-Walker from `phi_1, phi_2` | 2 | partial autocorrelations `tanh(th_i)`, then Levinson-Durbin |
| `ar3` | same at order 3 | 3 | idem |
| `sar` | AR(2) with `phi_1 = phi`, `phi_2 = -phi^2/4` | 1 | `phi = tanh(th)` |
| `ma1` | `rho_1 = -th/(1+th^2)`, 0 beyond | 1 | `th = tanh(.)` |
| `ma2` | `rho_1 = -t1(1-t2)/D`, `rho_2 = -t2/D`, `D = 1+t1^2+t2^2` | 2 | `t1, t2 = tanh(.)` |
| `arma` | `rho_1 = (1+ph*th)(ph+th)/(1+2 ph th+th^2)`, `rho_k = ph rho_{k-1}` | 2 | both `tanh` |
| `corb(b)` | `rho_k = tanh(th_k)` for `k <= b`, 0 beyond | `b` | `tanh` per band |
| `corg` | free correlation matrix | `q(q-1)/2` | Cholesky factor with **row-normalised** rows |
| `exp` | `C_ij = phi^{d_ij}`, 1D coordinates | 1 | `phi = |tanh(th)|` |
| `gau` | `C_ij = phi^{d_ij^2}` | 1 | `phi = |tanh(th)|` |
| `lvr` | `C_ij = max(0, 1 - dx_ij/f)` | 1 | `f = exp(th)`, a **range** |
| `iexp` | `phi^{|dx| + |dy|}` (city-block) | 1 | `phi = |tanh(th)|` |
| `igau` | `phi^{dx^2 + dy^2}` | 1 | `phi = |tanh(th)|` |
| `ieuc` | `phi^{d}`, `d` Euclidean | 1 | `phi = |tanh(th)|` |
| `sph` | `1 - 1.5 t + 0.5 t^3`, `t = min(d/f, 1)` | 1 | `f = exp(th)`, a range |
| `cir` | `1 - (2/pi)(t sqrt(1-t^2) + asin t)` | 1 | `f = exp(th)`, a range |
| `aexp` | `phi_1^{|dx|} phi_2^{|dy|}` | 2 | both `|tanh|` |
| `agau` | `phi_1^{dx^2} phi_2^{dy^2}` | 2 | both `|tanh|` |
| `mtrn` | anisotropic Matern (below) | 0 to 4 | `exp` except `alpha` |
| `own` | user-supplied expression | `n_par` | raw, unconstrained |
| `ar1ar1` | `C_row (x) C_col` | 2 | two `tanh` |

`exp`, `gau`, `lvr`, `iexp`, `igau`, `ieuc`, `sph`, `cir`, `aexp`, `agau` and
`mtrn` require coordinates. The 1D and 2D families are kept separate on the R
side, because passing a single-column coordinate to a 2D structure would be
accepted silently and would fit a different model.

### Where the parameters land in `theta`

For one term, `theta` is `[Sigma parameters, between-level parameters]`, in
that order; terms come first, in declaration order, then the residual (and,
for `dsum`, each section in turn with its own parameters). This split is
performed identically by `remlax.model.split_theta` and by the R side. It is
fixed, and changing it on one side only would produce a different model with
no visible error.

### Which scale each family reports

`level_params_report` reports on the asreml scale, and the rule it follows is
that **the value reported is the value actually used in `C`**, never an
intermediate `theta`. Two families need more than a plain `tanh` for that to
hold, and both are handled.

`fit$rho` on the R side, and the dictionary returned by
`level_params_report`, carry these keys:

| family | key | reported value |
|---|---|---|
| `cor` | `phi`, `borne_inf` | the correlation in `C`, already rescaled onto `(-1/(q-1), 1)`, plus that lower bound |
| `ar1`, `sar`, `ma1` | `phi` | `tanh(theta)`; the sign is meaningful |
| `ma2`, `arma` | `phi` (2) | `tanh(theta_i)`; the MA sign may differ from asreml, see below |
| `ar2`, `ar3` | `phi`, `pacf` | asreml's autoregressive coefficients, and the internal partial autocorrelations |
| `corb(b)` | `phi` (b) | `tanh(theta_k)` per band |
| `corg` | (none) | the free correlation matrix is not summarised by scalars |
| `exp`, `gau`, `iexp`, `igau`, `ieuc` | `phi`, `signe_non_identifie` | `abs(tanh(theta))`, floored at `1e-12` |
| `aexp`, `agau` | `phi` (2), `signe_non_identifie` | `abs(tanh(theta_i))` per axis |
| `sph`, `cir`, `lvr` | `portee` | the range `exp(theta)`, not a correlation |
| `mtrn` | `phi`, `nu`, `delta`, `alpha` | only the declared parameters; `exp(theta)` except `alpha`, which is an angle |
| `own` | `own` | the **raw** unconstrained parameters; the expression decides what they mean |
| `ar1ar1` | `phi` (2) | `tanh(theta_i)` for the row and column factors |

Measured, at `q = 4`:

```
theta -2.000000   phi rapporte -0.309352   C[0,1] -0.309352   borne_inf -0.333333
theta -0.500000   phi rapporte  0.025255   C[0,1]  0.025255   borne_inf -0.333333
theta  0.000000   phi rapporte  0.333333   C[0,1]  0.333333   borne_inf -0.333333
theta  1.040774   phi rapporte  0.852129   C[0,1]  0.852129   borne_inf -0.333333
```

The reported `phi` of `cor` is now the entry of `C`, to the digit. `q` is a
**required** argument of `level_params_report` for this structure, because
without it the rescaling cannot be applied:

```
ValueError: level_params_report(kind='cor') exige q : la correlation uniforme est
remise a l'echelle sur (-1/(q-1), 1), donc tanh(theta) n'est pas la valeur du modele.
```

Note again that `theta = 0` gives `1/3` and not `0`: the mapping sends the
midpoint of `(-1, 1)` to the midpoint of `(-1/(q-1), 1)`.

### The metric families: the sign of theta is not identified

The metric families raise `abs(tanh(theta))` to a distance, so `theta` and
`-theta` give **the same** correlation matrix. The reported `phi` is therefore
always positive, and the report carries `signe_non_identifie` to say why:

```
exp   phi(+theta)  0.761594  phi(-theta)  0.761594  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
gau   phi(+theta)  0.761594  phi(-theta)  0.761594  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
iexp  phi(+theta)  0.761594  phi(-theta)  0.761594  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
igau  phi(+theta)  0.761594  phi(-theta)  0.761594  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
ieuc  phi(+theta)  0.761594  phi(-theta)  0.761594  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
aexp  phi [0.761594 0.604368]  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
agau  phi [0.761594 0.604368]  max|C(+)-C(-)| 0.00e+00  signe_non_identifie 1
```

`max|C(+) - C(-)| = 0` exactly: this is a genuine non-identification of the
sign, not a reporting artefact. Two runs starting from opposite `theta` reach
the same likelihood and now report the same `phi`.

A consequence to recognise in output: a metric structure whose correlation has
collapsed reports `phi = 1e-12`, the floor of the clip, rather than a plain
zero. See [guide 3](guide/03-spatial.md#metric-kernels) for a fit where three
of them do exactly that.

`ar1` keeps its sign, because `phi^k` with negative `phi` is a legitimate
alternating correlation and the two are not the same model:

```
theta  -1.0  phi rapporte -0.761594   C[0,1] -0.761594
theta   1.0  phi rapporte  0.761594   C[0,1]  0.761594
```

### ar2 and ar3: partial autocorrelations, not phi

Bounding `|phi_i| < 1` is not enough. The stationarity region of an AR(2) is a
triangle, not a square, and a `phi` outside it produces a "correlation" that
is not positive definite — the fit then fails without saying why. remlax
parametrises the **partial autocorrelations**, each mapped into `(-1,1)` by
`tanh`, and derives the `phi` by the Levinson-Durbin recursion. That bijection
maps `(-1,1)^p` exactly onto the admissible region, so every point of the
unconstrained space is a valid AR:

```
theta [0.9, 0.9]     pacf [0.7163 0.7163]  phi [0.2032 0.7163]  min eig 9.84e-02
theta [-1.5, 1.2]    pacf [-0.9051 0.8337] phi [-0.1506 0.8337] min eig 2.08e-02
```

The reported `phi` remain those of asreml; `fit$pacf` carries the internal
parametrisation when you need it.

### cor: the exact positivity bound

`(1-r) I + r J` is positive definite for `r > -1/(q-1)` and nothing else. At
`q = 10`, `r = -0.3` gives an eigenvalue of `-1.7`. A plain `tanh` would allow
the whole of `(-1,1)` and the fit would fail silently, so the map sends
`(-1,1)` onto exactly the admissible interval.

### arma: a documented divergence from the manual

The formula printed in appendix C,
`rho_1 = (th - ph)(1 - th ph)/(1 + th^2 - 2 th ph)`, produces sequences that
are **not** positive definite over a large part of the square
`|th| < 1, |ph| < 1` — 204 out of 300 random draws, including `rho_1 = -0.976`
with `rho_2 = -0.944`, which is impossible. remlax uses the standard ARMA(1,1)
autocorrelation, valid over the whole square. The fitted model is the same; the
**sign** of the reported MA parameter may differ from asreml.

### lvr: a formula identified against asreml, not read in a manual

The help page of `lvr` gives the signature, not the formula. It was recovered
by comparing log-likelihood curves at fixed parameter against asreml, and is
the truncated tent

```
C_ij = max(0, 1 - d_ij / f)
```

the linear-with-sill model of geostatistics. It is positive definite **in one
dimension only**, being the autocorrelation of a rectangular pulse; there is
no two-dimensional analogue. `f` is a range, hence `exp(theta)` — positive and
unbounded — not a correlation in `tanh`.

The likelihood in `f` is **not unimodal**: the support of the tent changes
every time `f` crosses an integer, so an `optimize()`-style search is easily
trapped. The non-regression test compares against a fine grid for that reason.
See the next section for how the starting point is chosen.

### sph, cir, lvr: range parameters, kinks and the starting point

These three kernels are exactly zero beyond the range `f`. Two consequences.

**The gradient must be written with `where`, not `clip`.** `cir` contains
`sqrt(1 - t^2)` and `asin(t)`, whose derivative is infinite at `t = 1`. Writing
`t = min(d/f, 1)` and then the formula multiplies that infinite derivative by
the zero derivative of the clip: `0 x inf = NaN`. As soon as one pair of
positions lies beyond the range (always, in practice) the whole gradient was
NaN, L-BFGS-B stopped at iteration 0 and the starting point was returned as
the optimum, silently (validation of 2026-09-28 against asreml, test B7:
`n_iter = 0`, `-2 logL` 6.4 above the optimum). Both kernels are now evaluated
on a safe `t` (0 beyond the range) and selected with `jnp.where` on each side,
the only form whose gradient is finite everywhere. A start whose value or
gradient is not finite now raises an error instead of being returned.

**The likelihood in `f` has a kink at every distinct pairwise distance**, and
often several local maxima. On the 90-position irregular field of test B7 the
circular kernel has three (ranges 5.0, 7.4 and 10.9; `-2 logL` 6.1 apart
between the first and the best). asreml, started near 5, stops in the first
hill; so did remlax before this change. The automatic start therefore sweeps
`-2 logL` over the deciles of the pairwise distances of the design (one range
parameter at a time, other parameters at their start), gives the three best
candidates a ten-iteration L-BFGS-B descent, and starts the fit from the best
point reached. Cost: a few dozen evaluations, once. A `theta_init` supplied
by the caller is respected and not swept. On that field remlax now reaches
`logLik = 13.876` where asreml reports `10.848`, at the same kernel; the
difference is the hill, not the function.

The sweep is a heuristic, not a guarantee. On irregular one-dimensional
positions the tent kernel `lvr` can have dozens of local minima (68 between
ranges 2 and 25 on 60 random positions); there `n_restarts`, or a
`theta_init` from a grid, remains necessary. The pytest file
`tests/python/test_portee.py` checks the finite gradient, the sweep, and that
the fit reaches the global minimum of an independent dense profile on
two-dimensional fields (`sph`, `cir`) and on a regular transect (`lvr`).

### mtrn: the anisotropic Matern of Haskard et al. (2007)

```
u =  dx cos(a) + dy sin(a)        h = ( |sqrt(D) u|^l + |v / sqrt(D)|^l )^(1/l)
v = -dx sin(a) + dy cos(a)        rho = 2^(1-nu)/Gamma(nu) (h/f)^nu K_nu(h/f)
```

Parameters, in this canonical order: `phi` (range `f`), `nu` (shape), `delta`
(anisotropy ratio `D`), `alpha` (angle `a`). `lambda` (`l`) is fixed at 1
(city-block) or 2 (Euclidean) and is never estimated. `phi`, `nu` and `delta`
are taken in `exp(theta)`; `alpha` is an angle and stays free.

As in asreml, a parameter **absent from the call is held at its default** and
only declared parameters are estimated — which prevents `nu`, `delta` and
`alpha` from being fitted by accident on a design that does not identify them.
Defaults: `phi = 1` (estimated by default), `nu = 0.5`, `delta = 1`,
`alpha = 0`, `lambda = 2`.

`delta` acts **as a square root and in opposite directions on the two axes**.
The transformation preserves areas, so `delta` does not become confounded with
the range `phi`. Three other plausible conventions (`delta` on one axis only,
`delta` not square-rooted) shift the log-likelihood by 0.5 to 4 points.

`lambda = 1` with `nu != 0.5` is **not positive definite** in two dimensions.
asreml returns a number anyway; remlax refuses.

`K_nu` for arbitrary `nu` is computed in `src/remlax/bessel.py` by trapezoidal
quadrature of the integral representation, not by a series. The usual series
goes through `K_nu = pi/2 (I_-nu - I_nu)/sin(nu pi)`, which blows up at integer
`nu`. The integral is analytic in `nu`, has no order singularity, and its
trapezoidal error decays like `exp(-pi^2/h)`. What is computed is
`M(z,nu) = 2^(1-nu)/Gamma(nu) z^nu K_nu(z)` directly, never `K_nu` alone:
`z^nu K_nu(z)` is regular at `z = 0` whereas `K_nu(z)` diverges there.

### own: a user-defined structure

asreml's `own(obj, fun)` calls an R function at each evaluation. That is not
possible here: R and JAX do not share memory, and the **derivative** would be
missing — an opaque R function has none. remlax takes an **expression**
instead, which crosses the boundary as a string and is differentiated
automatically.

```r
random = ~ own(col, expr = "exp(-lag*exp(p1))", n_par = 1)
```

Variables available in the expression: `d` (Euclidean distance), `dx`, `dy`,
`lag` (`|i-j|`), `I`, `J`, `q`, and `p1..pk`, the **raw** unconstrained
parameters — it is the expression's job to apply `exp` or `tanh` if it wants
positivity or a bound. Functions available: `exp`, `log`, `sqrt`, `abs`,
`tanh`, `sinh`, `cosh`, `sin`, `cos`, `minimum`, `maximum`, `where`, `pi`,
`matern`.

`^` is translated to `**`. Without that translation, an expression written in
R produced `xor does not accept dtype float64`, because `^` is exclusive-or in
Python and does not even raise a readable error on floats.

The result is normalised to a correlation unless `normalise = FALSE`; the
scale is already carried by `Sigma`, and without normalisation the variance
would be counted twice.

`own` reproduces a built-in structure exactly when given its formula.
Measured on a 16 x 12 field, `own(col, "exp(-lag*exp(p1))")` against
`ar1(col)`:

```
own(col, exp(-lag*exp(p1))) : logLik -338.56966, p1 = -2.27076 -> rho(1) = 0.90192
ar1(col)                    : logLik -338.56966, phi = 0.90192
```

### ilv is not provided

asreml has an `ilv`; its formula could not be recovered. None of nine
candidate families reproduces its log-likelihood curve, and the natural
candidate — a Euclidean tent — is not positive definite in two dimensions and
returns NaN. An absent structure is preferable to a wrong one carrying an
asreml name. `ilv(x, y)` raises "terme aleatoire non reconnu".

### Separable products

`ar1(row, col)` builds `C_row (x) C_col` with the **row index slowest**:
level `= (row - 1) * n_col + col`. R constructs the incidence with the same
convention. Getting this backwards is a silent error, and a specific one: on a
design where one factor alternates, lag 1 of asreml becomes lag 2 of a
full-grid field, and the estimated correlation comes out as the **square** of
the right one, sign erased.

---

## 3. Residual structures

The residual is a term like any other, with the identity as incidence. One
formula covers everything:

```
R[i,j] = Sigma_trait[trait(i), trait(j)] * C_unit[unit(i), unit(j)]
```

| written | Sigma | C |
|---|---|---|
| `~ units` | `s2` | `I` |
| `~ us(trait):units` | `us` | `I` |
| `~ ar1(row):ar1(col)` | `s2` | `AR1 (x) AR1` |
| `~ exp(pos)` | `s2` | metric |
| `~ dsum(~ ... | site)` | per section | per section |

The `Sigma` side of a residual accepts `iid`, `diag`, `us` and `fa` only.

`dsum` makes `R` a direct sum: sections **partition** the observations, and
each carries its own parameters, so two sites or two trials can have residual
structures of different shapes inside one fit.

Two observations of the same unit **under the same trait** would have
correlation 1 under any residual structure: `R` is then singular, `V` with it,
and `-2 logL` is NaN. remlax refuses explicitly, for any structure between
units (the check used to cover `us`/`diag`/`fa` only, and a
`dsum(~ ar1(col) | site)` with several rows per column slipped through and
returned NaN in silence):

```
ValueError: structure 'own' entre unites : 180 couple(s) (unite, caractere) en
double. Chaque unite ne peut etre observee qu'une fois par caractere ; sinon la
matrice residuelle est singuliere et la vraisemblance vaut NaN.
```

The unit of the constraint is the **(unit, trait) pair**, not the unit. On
long-format data with `t` traits every cell of a residual field appears `t`
times by construction, and `R = Sigma_trait (x) C_cell` is perfectly regular
there: `Sigma_trait` carries the correlation within a cell. The R-side parser
enforces the same pair, so `us(trait):ar1(row):ar1(col)` is a legal formula:

```
residual = ~ ar1(row):ar1(col) : 1 couple(s) (cellule, caractere) en double. Un
champ residuel structure exige au plus une observation par cellule et par
caractere.
```

---

## 4. Two-dimensional splines

`rx_spl2d(x, y, nseg)` builds a tensor-product P-spline basis and returns the
PS-ANOVA decomposition: three random blocks (`_x`, `_y`, `_xy`), each with its
own variance so the smoothing can be anisotropic, plus a null-space part that
**must** go into the fixed effects. Nothing is added to the solver — a smooth
surface is a random effect with a known incidence, exactly as in sommer and
asreml.

---

## 5. Fixed parameters

`fixed_theta = c(3, 5)` (1-based indices into `theta`) bounds those parameters
to their starting value, which is asreml's `F` code. Bounding rather than
removing them keeps all the algebra unchanged. The `P` code (positive) is
automatic here, since log standard deviations are what is parametrised, and
`U` is the default.

A fixed parameter is **outside the free subspace without being at a bound**:
it is pinned to its starting value, which has nothing to do with `floor` or
`ceil`. Three separate counters say so, and they are deliberately not merged
into one — they do not mean the same thing:

| counter | meaning |
|---|---|
| `n_at_bound` | parameters sitting at `floor` or `ceil` (default `-12`/`12`), **excluding** fixed ones |
| `n_fixed_out` | parameters held by `fixed_theta` |
| `n_par_free` | dimension of the actual free subspace |

`n_fixed` is the same count as `n_fixed_out`, reported by `fit_reml` alongside
the diagnostic block. Fixed parameters are excluded from the covariance used by
`vpredict`.

This matters beyond bookkeeping. The Newton decrement is computed on the free
subspace only; a pinned parameter left inside it would contribute the slope
available along a direction the step cannot take, and the fit would report an
optimum not reached where it was. Measured on a two-component model with the
first parameter fixed:

```
libre          n_par 3 | n_at_bound 0 | n_fixed 0 | n_fixed_out 0 | n_par_free 3
               max_grad 5.888076e-13 | newton_decrement 5.471e-28
fixed_idx=[0]  n_par 3 | n_at_bound 0 | n_fixed 1 | n_fixed_out 1 | n_par_free 2
               max_grad 5.117183e+01 | newton_decrement 4.510e-14
```

`max_grad` is 51.2 in the fixed case — the gradient along the pinned
direction, which is large and irrelevant — while the Newton decrement stays at
`4.5e-14`, correctly reporting that the two free parameters are at their
optimum. This is one more reason to read the decrement and not the gradient.
