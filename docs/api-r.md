# R interface reference

`R/remlkit.R` describes models; it fits none. It builds the incidences, checks
their consistency, serialises everything, and delegates the computation to the
JAX solver, which uses a GPU if the machine has one and a CPU otherwise
**without the model changing**.

```r
source("R/remlkit.R")     # requires Matrix and jsonlite
```

Two ways in. The formula interface, in the spirit of asreml:

```r
fit <- rk_reml(fixed    = y ~ 1 + traitement,
               random   = ~ vm(genotype, K = Kmat) + iid(bloc),
               residual = ~ units,
               data     = df,
               backend  = "auto")
```

and, for everything a formula cannot say — weighted incidences, heterogeneous
columns — the explicit path, the same machinery without the sugar:

```r
mod <- rk_model(y, X,
                terms    = list(rk_term("voisinage", list(W1, W2), K = Kb, struct = "us")),
                residual = rk_residual("us", trait = d$trait, unit = d$unit))
fit <- rk_fit(mod, backend = "auto")
```

---

## 1. Formula grammar: random terms

Each term of `random` is a call whose **first argument is the grouping
column**. `K =` and `rank =` are optional. Below, `w` is the number of traits
and `q` the number of levels of the grouping factor.

### Covariance over the columns of Sigma

| written | Sigma | parameters | needs `trait =` |
|---|---|---|---|
| `f` (bare name) | `s2` | 1 | no |
| `iid(f)` | `s2` | 1 | no |
| `vm(f, K)` or `vm(f, K = K)` | `s2`, relationship `K` | 1 | no |
| `mm(Z)` | `s2`, supplied incidence | 1 | no |
| `diag(f)` | `diag(s2_j)` | `w` | yes |
| `us(f)` | free symmetric PD | `w(w+1)/2` | yes |
| `fa(f, rank = k)` | `Lam Lam' + diag(psi)` | `n_loadings(w,k) + w` | yes |
| `rr(f, rank = k)` | `Gam Gam'`, rank `k` | `n_loadings(w,k)` | yes |
| `chol(f, rank = k)` | `L D L'`, band `k` | `(k+1)(w - k/2)` | yes |
| `ante(f, rank = k)` | `Sigma^-1 = U D U'`, band `k` | `(k+1)(w - k/2)` | yes |
| `corh(f)` | heterogeneous variances, uniform correlation | `w + 1` | yes |
| `str(~ a + b, struct = )` | one covariance for several terms | depends on `struct` | no |

`K =` is accepted by all of them: `us(gid, K = Kmat)` is a multi-trait model
with a relationship matrix. `rank =` only means something for `fa`, `rr`,
`chol` and `ante`. `mm()` accepts `name =` to control the output key.

`diag`, `us`, `fa`, `rr`, `chol`, `ante` and `corh` require `trait =` on
`rk_reml()`: the data are then in **long format**, one row per unit x trait,
and each column's incidence is built automatically. Long format is chosen over
a `cbind()` of responses because it handles traits measured on different
subsets of units with nothing special — which is the rule as soon as
phenotyping is incomplete.

### Correlation between levels

The variance stays in `Sigma`. `struct =` remains available to make it
heterogeneous across traits — the equivalent of asreml's `v`/`h` suffixes.

| written | correlation | parameters | notes |
|---|---|---|---|
| `cor(f)` | uniform | 1 | bounded to the PD interval |
| `ar1(f)` | `rho_k = phi^k` | 1 | |
| `ar2(f)`, `ar3(f)` | AR(2), AR(3) | 2, 3 | partial autocorrelations |
| `sar(f)` | constrained AR(2) | 1 | |
| `ma1(f)`, `ma2(f)` | MA(1), MA(2) | 1, 2 | |
| `arma(f)` | ARMA(1,1) | 2 | standard form, see below |
| `corb(f, order = b)` | banded | `b` | |
| `corg(f)` | free correlation matrix | `q(q-1)/2` | |
| `exp(x)`, `gau(x)` | metric 1D | 1 | numeric levels or `coord =` |
| `lvr(x)` | truncated tent, 1D only | 1 | parameter is a **range** |
| `iexp(x,y)`, `igau(x,y)`, `ieuc(x,y)` | metric 2D isotropic | 1 | |
| `sph(x,y)`, `cir(x,y)` | 2D with a range | 1 | parameter is a **range** |
| `aexp(x,y)`, `agau(x,y)` | 2D anisotropic | 2 | |
| `mtrn(x,y, phi=, nu=, delta=, alpha=, lambda=)` | anisotropic Matern | 0 to 4 | only declared parameters are estimated |
| `own(f, expr =, n_par = k, normalise =)` | user expression | `k` | |
| `ar1(r, c)` | `AR1 (x) AR1`, separable | 2 | two positional arguments |

Measured parameter counts, on a design with three traits and a `~ trait` fixed
part (the total includes an `iid` residual, hence one more than the term
itself):

```
~ gid                                          2 parametre(s) de variance
~ iid(gid)                                     2
~ vm(gid, K = K)                               2
~ vm(gid, K)                                   2
~ diag(gid)                                    4
~ us(gid)                                      7
~ us(gid, K = K)                               7
~ fa(gid, rank = 1)                            7
~ rr(gid, rank = 2)                            6
~ chol(gid, rank = 1)                          6
~ ante(gid, rank = 1)                          6
~ corh(gid)                                    5
~ mm(Zw)                                       2
~ str(~ gid + mm(Zw, name='w'), struct='us')   4
~ cor(col)                                     3
~ ar1(col)                                     3
~ ar2(col)                                     4
~ ar3(col)                                     5
~ sar(col)                                     3
~ ma1(col)                                     3
~ ma2(col)                                     4
~ arma(col)                                    4
~ corb(col, order = 2)                         4
~ corg(col)                                   12
~ ar1(row, col)                                4
~ exp(col)                                     3
~ gau(col)                                     3
~ lvr(col)                                     3
~ iexp(coln, rown)                             3
~ igau(coln, rown)                             3
~ ieuc(coln, rown)                             3
~ sph(coln, rown)                              3
~ cir(coln, rown)                              3
~ aexp(coln, rown)                             4
~ agau(coln, rown)                             4
~ mtrn(coln, rown)                             3
~ mtrn(coln, rown, nu = 1, delta = 2)          5
~ mtrn(coln, rown, nu = "1 F")                 3
~ own(col, expr = "...", n_par = 1)            3
~ ar1(col, struct = "diag")                    3
```

`corg(col)` on `q = 5` columns gives `5*4/2 = 10` correlation parameters, plus
one variance and one residual: 12.

### Correspondence with asreml names

asreml writes variance and correlation in one symbol. remlkit separates them,
so an asreml name maps to a pair.

| asreml | remlkit | comment |
|---|---|---|
| `idv(f)` | `iid(f)` | |
| `idh(f)` | `diag(f)` | |
| `corgh(f)` | `us(f)` | |
| `corh(f)` | `corh(f)` | |
| `fa(f, k)` | `fa(f, rank = k)` | |
| `rr(f, k)` | `rr(f, rank = k)` | fewer parameters, see below |
| `chol(f, k)` | `chol(f, rank = k)` | |
| `ante(f, k)` | `ante(f, rank = k)` | |
| `vm(f, K)` | `vm(f, K = K)` | |
| `ar1v(f)` | `ar1(f)` | `Sigma = s2` |
| `ar1h(f)` | `ar1(f, struct = "diag")` | `Sigma` diagonal |
| `ar1(f)` (correlation only) | `ar1(f)` | remlkit always carries a scale in `Sigma` |
| `ar2v`, `ar3v`, `sarv`, `ma1v`, ... | `ar2(f)`, `ar3(f)`, `sar(f)`, `ma1(f)`, ... | same rule |
| `corbv(f, b)` | `corb(f, order = b)` | |
| `corgv(f)` | `corg(f)` | |
| `expv(x)`, `gauv(x)` | `exp(x)`, `gau(x)` | |
| `iexpv(x,y)`, `ieucv(x,y)`, ... | `iexp(x,y)`, `ieuc(x,y)`, ... | |
| `sphv`, `cirv`, `aexpv`, `agauv` | `sph`, `cir`, `aexp`, `agau` | |
| `mtrnv(x,y,...)` | `mtrn(x,y,...)` | |
| `lvrv(x)` | `lvr(x)` | formula identified against asreml |
| `ilv(x,y)` | **absent** | formula could not be recovered |
| `own(obj, fun)` | `own(f, expr = "...", n_par = k)` | an expression, not an R function |
| `str(~ a + b, ~us(2):id(n))` | `str(~ a + b, struct = "us")` | |
| `dsum(~ ... | site)` | `dsum(~ ... | site)` | same |
| `units` | `units` | |
| `grp(x)` | `mm(Zmatrix)` | supply the incidence directly |

Three counts or conventions differ from the manual, deliberately.

- **`rr(k)` counts `n_loadings(w,k)`, not `k*w`**, because `Gamma` is
  constrained to be trapezoidal. The fitted model is identical; the Hessian is
  no longer singular in `k(k-1)/2` directions.
- **`arma` uses the standard ARMA(1,1) autocorrelation.** The formula printed
  in appendix C is not positive definite over a large part of its own
  parameter square. Same fitted model; the reported **sign** of the MA
  parameter may differ.
- **`ar2`/`ar3` are parametrised by partial autocorrelations.** The reported
  `phi` are still asreml's; `fit$pacf` carries the internal ones.

### `str()`: one covariance for several terms

```r
random = ~ str(~ gid + mm(W, name = "ige"), struct = "us", name = "dge_ige")
```

The grouped terms must share **exactly** the same levels — that shared
indexing is what gives a covariance between them any meaning — and each keeps
**its own** incidence, `mm()` included. This is what lets a direct effect and a
neighbourhood effect carried by the same genotypes be correlated.

Arguments: `struct` (default `"us"`), `K`, `rank`, `name`. Without `name`, the
key is built from the names of the incidences.

---

## 2. Formula grammar: the residual

Same grammar as the random terms. The formula reads as a product of factors
separated by `:`. **At most one factor bears on the traits** — the one whose
argument is the column passed as `trait =` — and the others bear on the units.

| written | R |
|---|---|
| `~ units` or `~ id(units)` | `s2 I` |
| `~ us(trait):units` | coupling between traits of the same unit |
| `~ diag(trait):units` | one variance per trait |
| `~ ar1(row):ar1(col)` | separable residual field |
| `~ exp(pos)` | metric decay |
| `~ dsum(~ <structure> | <section factor>)` | direct sum |

Verified:

```
res ~ units                                    2 parametre(s) de variance
res ~ us(trait):units                          7
res ~ diag(trait):units                        4
res ~ ar1(row):ar1(col)                        4
res ~ exp(col)                                 3
res ~ dsum(~ units | site)                     3
res ~ dsum(~ ar1(col) | site)                  5
```

Underneath there is a single formula:
`R[i,j] = Sigma_trait[t_i, t_j] * C_unit[u_i, u_j]`. The residual is a term
like any other whose incidence is the identity, so nothing had to be added to
the solver — only exposed.

The `Sigma` side of a residual accepts `iid`, `diag`, `us` and `fa` only.

### `dsum`: one residual structure per section

```r
residual = ~ dsum(~ ar1(row):ar1(col) | site)
residual = ~ dsum(~ ar1(row):ar1(col) + units | site, levels = list(1:3, 4))
```

The first form gives the **same shape** on every site with **parameters of its
own** for each. The second gives **different shapes**: `+` at the top level
separates structures, and `levels =` must then be a list of the same length,
assigning section levels to structures.

Sections partition the observations, so `R` is block-diagonal up to a
permutation. Both the R and the Python side check the partition: a forgotten
row would leave `V` with a zero variance, a duplicated one would count it
twice.

### A limitation of the residual parser

`us(trait):ar1(row):ar1(col)` **cannot be written as a formula** on
long-format data. The parser recognises the `ar1(r):ar1(c)` pair before
anything else and requires at most one observation per `(row, col)` cell,
which long format violates by construction:

```
residual = ~ ar1(row):ar1(col) : 30 cellule(s) en double. Un champ residuel
structure exige au plus une observation par cellule.
```

The solver supports the structure. Build it through `rk_residual()`, giving
the cell index as `unit`:

```r
cell <- (as.integer(d$row) - 1L) * nc + as.integer(d$col)
r <- rk_residual("diag", trait = d$trait, unit = cell,
                 level = "ar1ar1", dims = c(nr, nc), n_unit = nr * nc)
mod <- rk_model(d$y, X, list(rk_term("gid", d$gid)), r)
```

```
Ajustement REML (cpu) : logLik -79.255014 | 5 parametres | 60 obs
  Sigma[gid] 1x1, diagonale : 0.02091
  residuelle : 0.6537 0.9782
  rho : 0.118476 0.043092
  noms : gid, residuelle[1,1], residuelle[2,1], residuelle[2,2],
         residuelle!ar1ar11, residuelle!ar1ar12
```

---

## 3. `rk_reml()`

```r
rk_reml(fixed, random = NULL, residual = "units", data,
        trait = NULL, unit = NULL,
        backend = c("auto", "gpu", "cpu"), ...)
```

| argument | meaning |
|---|---|
| `fixed` | two-sided formula for the fixed effects |
| `random` | one-sided formula, grammar above; `NULL` allowed if the residual is structured |
| `residual` | `"units"`, `"diag"`, `"us"`, or a formula |
| `data` | data frame |
| `trait` | name of the trait column (long format), or a vector |
| `unit` | name of the unit column, or a vector |
| `backend` | machine, never model |
| `...` | passed to `rk_fit()`: `vpredict`, `wald`, `kenward_roger`, `predict`, `fixed_theta`, `n_restarts`, `polish`, `maxiter`, `hessian`, `blups`, `verbose`, `keep`, `dir` |

Incomplete rows are dropped with a message; `attr(X, "assign")` is restored
afterwards, without which a Wald test would examine each column in isolation
instead of the whole term.

The returned object is of class `rk_fit`, and additionally carries `model`,
`fixed`, `data`, `xlevels` and `call`, which is what `rk_predict()` needs.

---

## 4. `rk_fit()`

```r
rk_fit(model, backend = c("auto", "gpu", "cpu"), dir = NULL,
       maxiter = 3000L, polish = 25L, n_restarts = 0L,
       hessian = TRUE, blups = TRUE, vpredict = NULL, wald = FALSE,
       kenward_roger = FALSE, predict = NULL,
       fixed_theta = NULL, verbose = TRUE, keep = FALSE)
```

| argument | meaning |
|---|---|
| `model` | an `rk_model` |
| `backend` | `"auto"` takes the GPU if JAX sees one; `"gpu"` **refuses** rather than falling back silently |
| `dir` | where to write the serialised design; a temporary directory by default, removed unless `keep = TRUE` |
| `maxiter`, `polish` | optimiser caps |
| `n_restarts` | perturbed restarts; the only way to detect a local optimum |
| `hessian`, `blups` | switch off to save the corresponding computation |
| `vpredict` | named character vector, `c(h2 = "V1/(V1+V2)")` |
| `wald` | Wald tests on the fixed effects |
| `kenward_roger` | denominator degrees of freedom and adjusted covariance |
| `predict` | `list(L = , M = )`, matrices written for the solver |
| `fixed_theta` | **1-based** indices of parameters held at their starting value |
| `verbose` | passes the solver's output through |
| `keep` | keep `dir` after the fit |

### `vpredict`

The `Vi` are numbered in the order of `fit$composantes_noms` — a fixed,
documented order, because a silent renumbering would make every expression
wrong on the next run. Expressions use `V1, V2, ...` and the usual operators,
plus `sqrt`, `log`, `exp`, `abs`.

```r
fit <- rk_reml(y ~ 1, random = ~ gid + iid(bloc), data = d,
               vpredict = c(h2 = "V1/(V1+V2+V3)", vg = "V1",
                            pct = "100*V1/(V1+V2+V3)"))
fit$vpredict$composantes
fit$vpredict$predictions
```

```
  i        nom      valeur
1 1        gid 1.401362464
2 2       bloc 0.007220142
3 3 residuelle 0.898773260

  nom        expression     valeur         se
1  h2     V1/(V1+V2+V3)  0.6073456 0.05871483
2  vg                V1  1.4013625 0.30033238
3 pct 100*V1/(V1+V2+V3) 60.7345614 5.87148261
```

For a matrix structure the components are the **lower triangle of Sigma**, row
by row, then the between-level parameters:

```
  i             nom    valeur
1 1        gid[1,1] 2.1360812
2 2        gid[2,1] 0.4163982
3 3        gid[2,2] 0.2001556
4 4 residuelle[1,1] 0.5630491
5 5 residuelle[2,1] 0.3264990
6 6 residuelle[2,2] 1.9670976
```

Standard errors come from the delta method on the Hessian of `-2 logL`:
`var(g) = J' (2 H^-1) J`. They are `NA` when the Hessian is not invertible
over the free subspace — never an invented number.

### `wald`

Wald tests at fixed `V`, **conditional** (type III: each term given all the
others). asreml's default is sequential (type I); the two coincide on the last
term of the model. Without `kenward_roger = TRUE` the denominator degrees of
freedom are not adjusted, so the p-values are those of a `chi2/df` and are
asymptotically valid.

```
        terme ddl          F      chi2            p
1 (Intercept)   1 139.794105 139.79411 2.952830e-32
2         trt   2   5.091395  10.18279 6.149437e-03
```

### `kenward_roger`

Adds `fit$kenward_roger` with `tests`, `se_beta`, `se_beta_brut`,
`second_ordre_omis`, and `fit$vbeta_kr`. On the same unbalanced block design:

```
        terme ddl     denDF          F     F_brut   echelle            p
1 (Intercept)   1  9.508858 139.012320 139.012320 1.0000000 5.471722e-07
2         trt   2 20.188279   5.076542   5.076565 0.9999954 1.636762e-02

    se_brut     se_KR
1 0.4336283 0.4348459
2 0.4635461 0.4635461
3 0.4666903 0.4684138
second_ordre_omis : FALSE
```

`second_ordre_omis` is `TRUE` when the model carries between-level correlation
parameters, for which the omitted second-order term is not exactly zero.

### `fixed_theta` and `n_restarts`

```
   libre : logLik -155.75273, theta = -0.175965 -0.038052, n_at_bound = 0
   fixe  : logLik -155.80423, theta = -0.113874 -0.041241, n_at_bound = 0
```

A fixed parameter is bounded to its starting value; the algebra is unchanged.
It is counted in `n_fixed` and excluded from the covariance used by
`vpredict`. It is **not** counted in `n_at_bound`, which reports parameters at
the global `floor`/`ceil`.

```
   n_restarts 3, restart_better 0, restart_gain 0.000e+00, logLik -155.752735
```

`restart_better` counts the restarts that found a strictly better optimum, and
`restart_gain` is the improvement in `-2 logL`.

### `backend`

```
   backend='gpu' -> gpu, logLik -155.752734812
   backend='cpu' -> cpu, logLik -155.752734812
   ecart relatif sur -2logL : 1.82e-16
```

The device is fixed once, through `jax.default_device`. It is the only place
in the whole package where the choice of machine intervenes.

---

## 5. `rk_term()`

```r
rk_term(name, Z, K = NULL, struct = "iid", rank = 0L,
        t = NULL, levels = NULL, level = "auto",
        dims = NULL, order = 0L, coord = NULL,
        opts = NULL, expr = NULL)
```

| argument | meaning |
|---|---|
| `name` | key in every output |
| `Z` | three accepted forms, below |
| `K` | `q x q` relationship matrix with `dimnames`, or `NULL` for `I` |
| `struct` | `iid`, `diag`, `us`, `fa`, `rr`, `chol`, `ante`, `corh` |
| `rank` | rank of `fa`/`rr`, band order of `chol`/`ante` |
| `t` | number of columns; required when `Z` is a stacked matrix |
| `levels` | level names when `Z` is a matrix |
| `level` | between-level structure; `"auto"` is `"fixed"` if `K` is given, `"id"` otherwise |
| `dims` | `c(n_rows, n_cols)` for `level = "ar1ar1"` |
| `order` | order of `corb`/`corg` |
| `coord` | coordinates of the `q` levels; a vector (1D) or a two-column matrix (2D) |
| `opts` | named numeric vector of settings (`mtrn`, `own`) |
| `expr` | expression for `level = "own"` |

`Z` accepts:

1. a **factor or character vector** of length `n` — indicator incidence,
   `t = 1`, levels deduced;
2. a **list of `t` matrices** `n x q` — one per column of `Sigma`. This is the
   form that covers weighted incidences;
3. a **matrix `n x (t*q)`** already stacked, with `t` supplied.

`K` is reordered onto `levels` when both have names, and a missing level is an
error, not a silent drop. A Cholesky with **bending** is applied if `K` is not
positive definite — a GRM frequently is not (identical genotypes, more markers
than individuals) — with a message giving the amount added.

Checks that raise: a list of incidences with differing column counts; `rank`
out of `1..t` for `fa`/`rr` or `1..t-1` for `chol`/`ante`; a between-level
structure combined with a supplied `K` (the structure **is** the covariance
between levels); a metric structure without `coord`; a 2D metric structure
with a single-column `coord` — which would be accepted silently and would give
a different model; `own` without `expr`.

`struct = "iid"` with `t > 1` produces a message, not an error: it means one
variance shared by the `t` columns.

---

## 6. `rk_residual()`

```r
rk_residual(struct = "iid", trait = NULL, unit = NULL, rank = 0L,
            level = "id", order = 0L, coord = NULL, n_unit = NULL,
            dims = NULL, opts = NULL, expr = NULL, sections = NULL,
            rows = NULL, name = NULL)
```

`struct` is `iid`, `diag`, `us` or `fa`. `trait` is the trait factor (length
`n`); `NULL` means a single trait. `unit` is the unit identifier: two
observations of the **same** unit on different traits are correlated under
`us`. `level`, `order`, `coord`, `dims`, `opts` and `expr` describe the
structure between units, exactly as for a term. `sections` holds a list of
`rk_residual` objects for a direct sum, each carrying its `rows` and `name`.

---

## 7. `rk_model()`

```r
rk_model(y, X, terms, residual = rk_residual(), name = "modele")
```

Assembles and checks. Refuses: a term whose incidence has the wrong number of
rows; a `diag`/`us`/`fa` residual without multiple traits; sections that do
not partition the observations; an `X` that is not of full column rank —
`log|X'V^-1X|` would be `-Inf` and the restricted likelihood undefined.

A model with **no random term** is legal as long as the residual is
structured (`residual = ~ ar1(row):ar1(col)`), which is the commonest spatial
field trial model. A model with no random term and an `iid` residual is
refused: there is nothing to estimate.

`attr(X, "assign")` and `attr(X, "termes")` are carried through so that Wald
tests operate on whole terms.

`print.rk_model()` gives the inventory:

```
Modele REML 'modele' : 240 observations, 1 effets fixes
  terme dge_ige        us    t=2   q=60    K=I        (3 parametres)
  terme champ          iid   t=1   q=240   AR1xAR1    (3 parametres)
  residuelle     iid   t=1  (1 parametre)
  total : 7 parametres de variance
```

---

## 8. `rk_export()`

```r
rk_export(model, dir)
```

Serialises the design into `dir`: `manifest.json` plus one `<name>.bin` (raw
binary: float64, or int32 for indices; column-major) or `<name>.txt` (one
string per line) per array. No Python dependency on the R side, no R
dependency on the Python side.

```
manifest.json   res_lvl.txt      res_nsec.bin    res_rank.bin
res_struct.txt  res_t.bin        res_trait.bin   res_unit.bin
term_gid_levels.txt              term_gid_lvl.txt
term_gid_q.bin  term_gid_rank.bin term_gid_struct.txt
term_gid_t.bin  term_gid_zi.bin  term_gid_zj.bin term_gid_zx.bin
term_names.txt  X.bin            y.bin
```

`Z` is written as a COO triplet (`zi`, `zj`, `zx`) with **0-based** indices,
stacked in the order `column = (a-1)*q + level`.

---

## 9. `rk_predict()`

```r
rk_predict(fit, classify, levels = NULL, at = NULL,
           average = c("equal", "proportional"), weights = NULL,
           vcov = c("simple", "kenward-roger"),
           sed = FALSE, include_random = TRUE,
           backend = c("auto", "gpu", "cpu"), verbose = FALSE)
```

A prediction is a linear combination of the fitted effects; all the work is
building the right combination.

- `classify` names the variables held at each of their levels, separated by
  `:` or `+`.
- Every other variable of the fixed model is **averaged**: a factor over its
  levels (equal weights, or proportional to the observed counts with
  `average = "proportional"`), a covariate at its **mean** — which is what
  asreml does, and what explains a "by variety" prediction being given at
  `Column = 5.5`.
- `at` forces specific values, `levels` restricts the levels used, `weights`
  reweights the cells.
- If a `classify` variable is the factor of a random term, its **BLUP enters
  the prediction** and the standard error becomes a **prediction** error,
  computed by the solver where `V^-1` and the projector live.
  `include_random = FALSE` gives the fixed-effect prediction only.
- A combination outside the row space of `X` is **not estimable**; it is
  flagged and its value is `NA`, rather than returning a number that depends
  on the parametrisation chosen.
- `sed = TRUE` attaches the matrix of standard errors of differences
  (`attr(out, "sed")`) and their quadratic mean (`attr(out, "sed.moyen")`).

```r
pv <- rk_predict(f, classify = "trt", sed = TRUE)
```

```
Predictions
 trt predicted.value std.error estimable
   A         5.12698  0.433628      TRUE
   B         6.36507  0.433628      TRUE
   C         6.45335  0.436859      TRUE

  erreur-type moyenne des differences : 0.465645
```

`vcov = "kenward-roger"` requires a fit made with `kenward_roger = TRUE`. It
applies to the **fixed** part only; when a random term contributes, a warning
says that the random part keeps its usual prediction error.

Only single-trait random terms are covered. A `classify` on a multi-trait term
produces a warning and is dropped from the random part.

---

## 10. `rk_spl2d()`

```r
rk_spl2d(x, y, nseg = c(6L, 6L), deg = 3L, pord = 2L, prefix = "spl")
```

Tensor-product P-spline basis with the PS-ANOVA decomposition. Returns
`list(X = , terms = )`.

- `X` is the **null-space** part: it **must** go into the fixed effects,
  otherwise the surface is penalised down to its linear component and the
  smoothing is biased. Its columns are centred and reduced to an independent
  basis by a rank-revealing QR, so it can sit next to an intercept.
- `terms` is a list of three `rk_term` objects, `<prefix>_x`, `<prefix>_y` and
  `<prefix>_xy`, each with its **own** variance — which is what makes the
  smoothing anisotropic, a surface being allowed to be rough in one direction
  and smooth in the other.

```r
sp  <- rk_spl2d(d$coln, d$rown, nseg = c(6, 6))
X   <- cbind("(Intercept)" = 1, sp$X)
mod <- rk_model(d$y, X, terms = c(list(rk_term("gid", d$gid)), sp$terms))
```

```
partie nulle (fixe) : 3 colonnes : spl_lin1, spl_lin2, spl_lin3
  terme spl_x    q = 14
  terme spl_y    q = 14
  terme spl_xy   q = 49
```

Nothing is added to the solver: a smooth surface is a random effect with a
known incidence, exactly as in sommer and asreml.

---

## 11. Reading the result

`rk_fit` objects have a `print` method:

```
Ajustement REML (cpu) : logLik -388.088838 | 3 parametres | 240 obs | 1.6 s
  max|grad| 3.42e-14 | decrement de Newton 1.43e-28 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 1.401
  Sigma[bloc] 1x1, diagonale : 0.00722
  residuelle : 0.8988
```

| field | contents |
|---|---|
| `theta` | estimate on the internal scale |
| `logLik`, `logLik_asreml`, `const_2pi` | remlkit includes `(n-p)/2 log(2 pi)`, asreml omits it |
| `n_par`, `n_obs`, `secondes`, `backend` | |
| `sigmas` | named list of `Sigma` matrices |
| `sigmas_res`, `sigma_res` | per-section residual matrices, and the first one |
| `rho` | between-level parameters; a single-parameter structure keeps the bare key (`rho$residuelle`), several are prefixed (`rho[["residuelle!phi"]]`) |
| `pacf` | partial autocorrelations of `ar2`/`ar3` |
| `blups` | named list of `q x t` matrices |
| `beta`, `vbeta`, `vbeta_kr` | fixed effects and their covariance |
| `hessian` | Hessian of `-2 logL` |
| `max_grad`, `newton_decrement`, `n_neg_eig`, `n_null_dir`, `n_at_bound`, `cond` | diagnostics |
| `composantes_degenerees` | terms with a parameter at the floor |
| `composantes_noms` | the `Vi` numbering |
| `vpredict`, `wald`, `kenward_roger`, `predictions` | present when requested |

**Read `newton_decrement`, not `max_grad`.** The latter is several units at
verified optima. The former is the rise in log-likelihood still available
under the local quadratic model, so it is directly comparable to the 3.84 of a
one-degree-of-freedom LRT.

**`logLik` is not asreml's.** The difference is `(n-p)/2 log(2 pi)`, which is
219.6 on a design with `n = 240` — large enough to look like a disagreement
between models. Use `logLik_asreml` when comparing.

---

## 12. Environment variables

| variable | meaning |
|---|---|
| `RK_PY` | command that starts Python, space-separated |
| `RK_CLI` | path to `cli.py`, or `-m remlkit.cli` |
| `IGE_JAX_CMD` | fallback for `RK_PY` |
| `IGE_JAX_SIF` | Apptainer image; becomes `apptainer exec --nv <sif> python3` |

`rk_python_cmd()` returns the first of `RK_PY`, `IGE_JAX_CMD`, `IGE_JAX_SIF`,
`python3`. `rk_solver_args()` returns `RK_CLI` if set, then `-m remlkit.cli`
if `import remlkit` succeeds, then `src/remlkit/cli.py` relative to
`R/remlkit.R`.

The coupling goes through **files**, not reticulate. Two reasons: R and JAX
may live in different containers, and the file makes CPU/GPU parity verifiable
because both backends then read strictly the same input.

---

## 13. A trap worth knowing

`$` performs **partial matching** on R lists. When a `sigmas_res` field was
added to the result, the expression `r$sigmas` — not yet created at that point
of `rk_read_result()` — started matching `sigmas_res`. The terms' `Sigma`
matrices were appended to the residual list, and `fit$sigmas[[1]]` returned the
**residual** instead of the first term. Seven failures across three test
suites, all shifted by one, and the log-likelihood stayed correct throughout,
which made the diagnosis counter-intuitive.

`rk_read_result()` now uses `[[ ]]` only, which matches exactly. Use `[[ ]]`
when reading a fit programmatically.
