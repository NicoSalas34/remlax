# 3. Spatial models

A separable AR1 field, metric kernels, user-defined correlations, and
two-dimensional splines.

All blocks were executed with `backend = "cpu"` and their real output pasted
underneath.

---

## A field trial

A 16 x 12 grid, 192 plots, 48 genotypes replicated four times, with a
simulated field of AR1(0.6) across rows and AR1(0.3) across columns.

```r
source("R/remlkit.R")
set.seed(21)
nr <- 16; nc <- 12
d <- expand.grid(col = seq_len(nc), row = seq_len(nr))
d$row <- factor(d$row); d$col <- factor(d$col)
n <- nrow(d)
ar1 <- function(rho, q) rho^abs(outer(seq_len(q), seq_len(q), "-"))
Lf <- t(chol(kronecker(ar1(0.6, nr), ar1(0.3, nc)) + diag(1e-8, n)))
champ <- as.numeric(Lf %*% rnorm(n)) * sqrt(1.2)
ngen <- 48
d$gid <- factor(sample(rep(seq_len(ngen), length.out = n)))
g <- rnorm(ngen, 0, sqrt(0.9))
d$y <- 8 + g[as.integer(d$gid)] + champ + rnorm(n, 0, sqrt(0.35))
d$rown <- as.numeric(as.character(d$row))
d$coln <- as.numeric(as.character(d$col))
```

```
n = 192 parcelles, 48 genotypes, grille 16 x 12
```

The `rown`/`coln` numeric copies are for the metric structures, which need
coordinates rather than factor levels.

---

## The separable residual field

```r
f0 <- rk_reml(y ~ 1, random = ~ gid, residual = ~ units, data = d,
              backend = "cpu", verbose = FALSE)
f1 <- rk_reml(y ~ 1, random = ~ gid, residual = ~ ar1(row):ar1(col), data = d,
              backend = "cpu", verbose = FALSE)
```

```
Ajustement REML (cpu) : logLik -345.664616 | 2 parametres | 192 obs | 0.9 s
  Sigma[gid] 1x1, diagonale : 0.8006
  residuelle : 1.626

Ajustement REML (cpu) : logLik -320.758159 | 4 parametres | 192 obs | 1.7 s
  max|grad| 2.17e-08 | decrement de Newton 5.77e-18 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 0.7505
  residuelle : 1.591

rho ligne = 0.46281   rho colonne = 0.31226
LRT champ : chi2 = 49.813 pour 2 ddl, p = 1.52e-11
```

`~ ar1(row):ar1(col)` builds `R = s2 * (C_row (x) C_col)`. The correlations
are read from `f1$rho$residuelle`, a vector of two.

The row correlation comes out at 0.46 against a simulated 0.6, the column one
at 0.31 against 0.3. The row parameter is the harder of the two: 16 rows of 12
plots gives more replication of the column lag than of the row lag.

### Column order in a separable product

```
level = (row - 1) * n_col + col     -> row slowest, column fastest
```

R builds the incidence with that convention and `levels.py` builds
`L_row (x) L_col` in the same one. Getting it backwards is a silent error with
a recognisable signature: on a design where one factor alternates, lag 1 of
asreml becomes lag 2 of the full-grid field, and the estimated correlation
comes out as the **square** of the right one with its sign erased.

A structured residual field requires **at most one observation per cell**.
remlkit refuses otherwise, because two observations of the same cell would be
perfectly correlated, `R` singular, and `-2 logL` NaN.

---

## The nugget that vanishes

The same field can be written as a random term instead, leaving the residual
as a separate nugget.

```r
f2 <- rk_reml(y ~ 1, random = ~ gid + ar1(row, col), residual = ~ units,
              data = d, backend = "cpu", verbose = FALSE)
print(f2)
f2$rho
f2$composantes_noms
```

```
Ajustement REML (cpu) : logLik -320.758159 | 5 parametres | 192 obs | 3.0 s
  max|grad| 8.44e-11 | decrement de Newton 1.69e-28 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 0.7505
  Sigma[row_col] 1x1, diagonale : 1.591
  residuelle : 3.775e-11

$row_col
[1] 0.4628062 0.3122591

[1] "gid"             "row_col"         "row_col!ar1ar11" "row_col!ar1ar12"
[5] "residuelle"
```

Note `ar1(row, col)` with **two positional arguments**: that is the separable
product as a random term. Compare with `ar1(row):ar1(col)`, which is the
residual spelling.

The likelihood is identical to `f1` — `-320.758159` in both — and the nugget
has collapsed to `3.8e-11`. On a complete grid with one observation per cell,
an AR1 x AR1 random field and an iid residual are **not separately
identifiable**: the field can reproduce any nugget by lowering its
correlations. The solver says so:

```r
f2$composantes_degenerees
f2$n_at_bound
```

```
composantes_degenerees : residuelle
n_at_bound             : 1
```

A variance at the floor is not "estimated at zero", it is **unidentified**.
Its standard error, its share of variance, and any test touching it are void.
asreml refuses to converge in this situation; remlkit converges, so it has to
say so.

The practical rule: put the field in the **residual** when there is one
observation per cell. Put it in `random` only when the design gives the nugget
its own information — replicated cells, or a second uncorrelated source.

---

## Metric kernels

Metric structures work on coordinates rather than factor levels, so they
handle irregular positions and gaps that an AR1 cannot.

```r
essai <- function(fx)
  rk_reml(y ~ 1, random = ~ gid, residual = fx, data = d,
          backend = "cpu", verbose = FALSE)
```

```
~ units                            logLik  -345.6646  2 par.
~ ar1(row):ar1(col)                logLik  -320.7582  4 par.  residuelle=0.4628/0.3123
~ iexp(coln, rown)                 logLik  -345.6646  3 par.  residuelle=0
~ ieuc(coln, rown)                 logLik  -345.6646  3 par.  residuelle=0
~ sph(coln, rown)                  logLik  -327.8789  3 par.  residuelle!portee=4.64
~ aexp(coln, rown)                 logLik  -345.6646  4 par.  residuelle=0/0
```

Two things are visible here.

**`iexp`, `ieuc` and `aexp` collapsed to `phi = 0`** and reproduced the
independent-residual likelihood exactly. Their single correlation applies the
same decay in both directions of a field whose two directions have very
different correlations (0.6 and 0.3); the compromise is worse than no
structure at all, so the optimiser turns them off. `aexp` is anisotropic and
should do better, but it parametrises `phi_1^{|dx|} phi_2^{|dy|}`, which
decays as fast as an AR1 only along the axes, and here it also went to zero.

**`sph` found a range of 4.64** and recovered a substantial part of the field
(`-327.88` against `-345.66` independent, `-320.76` for the separable AR1). A
spherical kernel with a sill has a shape a separable AR1 does not, and it
identifies here because the field is genuinely isotropic over short distances.

The `portee` key rather than `phi` in the reported parameters signals that
`sph`, `cir` and `lvr` are parametrised by a **range** in `exp(theta)` —
positive and unbounded — not by a correlation in `tanh`.

**Read the reported `phi` of the metric families in absolute value.** They use
`|tanh(theta)|` internally, so the sign of the reported value is not
identified. `ar1` keeps its sign: a negative `phi` there is a legitimate
alternating correlation.

---

## Matern

`mtrn` is the anisotropic Matern of Haskard et al. (2007), with parameters
`phi` (range), `nu` (shape), `delta` (anisotropy ratio), `alpha` (angle), and
`lambda` fixed at 1 or 2.

```r
m1 <- rk_reml(y ~ 1, random = ~ gid, residual = ~ mtrn(coln, rown), data = d, ...)
m2 <- rk_reml(y ~ 1, random = ~ gid, residual = ~ mtrn(coln, rown, nu = 1.0), ...)
m3 <- rk_reml(y ~ 1, random = ~ gid, residual = ~ mtrn(coln, rown, nu = "1.0 F"), ...)
```

```
m1 : logLik  -321.8478  3 par.  residuelle!phi=1.1893
m2 : logLik  -321.7610  4 par.  residuelle!phi=1.8281 residuelle!nu=0.34253
m3 : logLik  -339.2455  3 par.  residuelle!phi=0.26242
```

The three differ only in what is said about `nu`, and that changes the model.

- `m1` does not mention `nu`: it stays at its default 0.5 and is **not
  estimated**. Three parameters.
- `m2` gives `nu = 1.0`: that is a **starting value**, and `nu` is estimated.
  Four parameters, and it settles at 0.343.
- `m3` gives `nu = "1.0 F"`: the asreml `F` code, fixed at 1.0. Three
  parameters, and the fit is 17 log-likelihood units worse than `m1` — the
  data want a rougher field than `nu = 1` allows.

This is asreml's convention: a parameter absent from the call keeps its
default and is not estimated. It prevents `nu`, `delta` and `alpha` from being
fitted by accident on a design that does not identify them.

The Matern reaches `-321.85` on 3 parameters where `sph` reached `-327.88` on
the same 3, and the separable AR1 `-320.76` on 4. All these counts are the
totals printed by the solver, genotype variance and residual variance
included; the structures themselves contribute one correlation parameter
(`sph`, `mtrn` at default `nu`) or two (`ar1(row):ar1(col)`). On this field the
separable structure is still the best description, which is unsurprising given
that it is the one that generated the data.

`K_nu` for arbitrary `nu` is computed by trapezoidal quadrature of the
integral representation, not by a series expansion — the usual series has a
singularity at integer `nu`. See
[api-python.md](../api-python.md#9-remlkitbessel).

---

## The duplicated-cell guard

Any structure between units requires one observation per (unit, trait) pair.
Passing a column factor as the unit on a 16-row field violates it:

```r
tryCatch(
  rk_reml(y ~ 1, random = ~ gid,
          residual = ~ own(coln, expr = "exp(-lag*exp(p1))", n_par = 1),
          data = d, backend = "cpu", verbose = FALSE),
  error = function(e) conditionMessage(e))
```

```
ValueError: structure 'own' entre unites : 180 couple(s) (unite, caractere) en
double. Chaque unite ne peut etre observee qu'une fois par caractere ; sinon la
matrice residuelle est singuliere et la vraisemblance vaut NaN.
```

The check covers **every** structure between units. It used to cover
`us`/`diag`/`fa` only, and a `dsum(~ ar1(col) | site)` with several rows per
column slipped through and returned NaN in silence.

---

## A user-defined correlation

`own()` takes an **expression**, not an R function. R and JAX do not share
memory, and an opaque R function would have no derivative. An expression
crosses the boundary as a string and is differentiated automatically.

Here it is used as a random term — where the levels are the 12 columns, so
there is no duplicate-cell problem — and it reproduces `ar1` exactly.

```r
o1 <- rk_reml(y ~ 1, random = ~ gid + own(col, expr = "exp(-lag*exp(p1))", n_par = 1),
              data = d, backend = "cpu", verbose = FALSE)
a1 <- rk_reml(y ~ 1, random = ~ gid + ar1(col), data = d,
              backend = "cpu", verbose = FALSE)
```

```
own(col, exp(-lag*exp(p1))) : logLik -338.56966, p1 = -2.27076 -> rho(1) = 0.90192
ar1(col)                    : logLik -338.56966, phi = 0.90192
```

`exp(-lag * exp(p1))` is `rho^lag` with `rho = exp(-exp(p1))`, which is an
AR(1) restricted to positive correlations, so the agreement is exact and not
approximate.

Variables available in the expression: `d` (Euclidean distance), `dx`, `dy`,
`lag` (`|i-j|`), `I`, `J`, `q`, and `p1..pk`, the **raw** unconstrained
parameters. It is the expression's job to apply `exp` or `tanh` if it wants
positivity or a bound. Functions: `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`,
`cosh`, `sin`, `cos`, `minimum`, `maximum`, `where`, `pi`, `matern`.

Write `^` if you prefer; it is translated to `**`. Without that translation an
expression copied from R produced `xor does not accept dtype float64`, because
`^` is exclusive-or in Python and does not even raise a readable error on
floats.

The result is normalised to a correlation unless `normalise = FALSE`: the
scale is already in `Sigma`, and without normalisation the variance would be
counted twice.

---

## Two-dimensional splines

`rk_spl2d()` builds a tensor-product P-spline basis with the PS-ANOVA
decomposition. It returns a null-space part for the fixed effects and three
random blocks.

```r
sp <- rk_spl2d(d$coln, d$rown, nseg = c(6, 6))
```

```
partie nulle (fixe) : 3 colonnes : spl_lin1, spl_lin2, spl_lin3
  terme spl_x    q = 14
  terme spl_y    q = 14
  terme spl_xy   q = 49
```

The null-space columns **must** go into the fixed effects. Left out, the
surface is penalised down to its linear component and the smoothing is biased.
They are centred and reduced to an independent basis by a rank-revealing QR,
so they sit next to an intercept without making `X` rank-deficient.

The three random blocks each carry **their own** variance, which is what makes
the smoothing anisotropic: a field is allowed to be rough across rows and
smooth across columns.

```r
X <- cbind("(Intercept)" = 1, sp$X)
mod <- rk_model(d$y, X,
                terms = c(list(rk_term("gid", d$gid)), sp$terms),
                residual = rk_residual("iid"))
print(mod)
fs <- rk_fit(mod, backend = "cpu", verbose = FALSE)
print(fs)
```

```
Modele REML 'modele' : 192 observations, 4 effets fixes
  terme gid            iid   t=1   q=48    K=I        (1 parametre)
  terme spl_x          iid   t=1   q=14    K=I        (1 parametre)
  terme spl_y          iid   t=1   q=14    K=I        (1 parametre)
  terme spl_xy         iid   t=1   q=49    K=I        (1 parametre)
  residuelle     iid   t=1  (1 parametre)
  total : 5 parametres de variance

Ajustement REML (cpu) : logLik -318.856345 | 5 parametres | 192 obs | 1.8 s
  max|grad| 7.67e-13 | decrement de Newton 1.39e-25 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 0.7719
  Sigma[spl_x] 1x1, diagonale : 2.082
  Sigma[spl_y] 1x1, diagonale : 0.702
  Sigma[spl_xy] 1x1, diagonale : 2.738
  residuelle : 1.003
```

`-318.86` with 5 parameters, against `-320.76` with 4 for the separable AR1.
The spline is competitive here, and the three variances are genuinely
different (2.08, 0.70, 2.74), which is the anisotropy the decomposition is
for.

Nothing was added to the solver: a smooth surface is a random effect with a
known incidence, exactly as in sommer and asreml. That is why the splines go
through `rk_model()` rather than through the formula interface.

---

## Structure summary for spatial work

| you have | write |
|---|---|
| complete grid, field in the residual | `residual = ~ ar1(row):ar1(col)` |
| complete grid, field as a random term | `random = ~ ar1(row, col)` — but watch the nugget |
| irregular positions | `iexp`, `ieuc`, `igau` on coordinates |
| a range and a sill | `sph`, `cir` |
| different behaviour along each axis | `aexp`, `agau`, or `mtrn` with `delta` |
| unknown roughness | `mtrn(x, y, nu = <start>)` |
| a smooth trend rather than a correlation | `rk_spl2d()` |
| a formula of your own | `own(f, expr =, n_par =)` |
| several trials with different fields | `residual = ~ dsum(~ ar1(row):ar1(col) | site)` |

`lvr` (the truncated tent) exists in one dimension only. `ilv` is **not
provided**: asreml's formula could not be recovered, and the natural candidate
is not positive definite in two dimensions. `ilv(x, y)` raises "terme
aleatoire non reconnu".

---

## Next

- [4. Explicit terms](04-explicit-terms.md) — weighted incidences and shared
  covariances.
- [structures.md](../structures.md) — every formula, constraint and
  parametrisation.
