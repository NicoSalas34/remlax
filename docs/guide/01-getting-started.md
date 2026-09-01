# 1. Getting started

One random factor, reading the output, and heritability with `vpredict`.

Every block below was executed and its real output pasted underneath. The
examples use `backend = "cpu"` so that they reproduce identically on a machine
without a GPU; on small models the CPU is also the faster choice, because XLA
compilation dominates the computation.

```r
source("R/remlax.R")
```

If the Python package is not installed, point the interface at it first:

```sh
export RX_PY=/path/to/venv/bin/python
export RX_CLI=/path/to/remlax/src/remlax/cli.py
```

---

## Data

A randomised complete block design: 60 genotypes in 4 blocks.

```r
set.seed(2026)
n_gen <- 60; n_bloc <- 4
d <- expand.grid(gid = factor(seq_len(n_gen)), bloc = factor(seq_len(n_bloc)))
g <- rnorm(n_gen, 0, sqrt(1.5))
b <- rnorm(n_bloc, 0, sqrt(0.4))
d$y <- 12 + g[as.integer(d$gid)] + b[as.integer(d$bloc)] + rnorm(nrow(d), 0, sqrt(1.0))
str(d)
```

```
'data.frame':	240 obs. of  3 variables:
 $ gid : Factor w/ 60 levels "1","2","3","4",..: 1 2 3 4 5 6 7 8 9 10 ...
 $ bloc: Factor w/ 4 levels "1","2","3","4": 1 1 1 1 1 1 1 1 1 1 ...
 $ y   : num  10.8 10.3 10.3 13 10.1 ...
```

The true variances are 1.5 for the genotypes, 0.4 for the blocks and 1.0
residual.

---

## The fit

```r
fit <- rx_reml(fixed  = y ~ 1,
               random = ~ gid + iid(bloc),
               data   = d,
               backend = "cpu", verbose = FALSE)
print(fit)
```

```
Ajustement REML (cpu) : logLik -388.088838 | 3 parametres | 240 obs | 1.6 s
  max|grad| 3.42e-14 | decrement de Newton 1.43e-28 | 0 valeur(s) propre(s) negative(s)
  Sigma[gid] 1x1, diagonale : 1.401
  Sigma[bloc] 1x1, diagonale : 0.00722
  residuelle : 0.8988
```

A bare name in `random` means `iid()`; `~ gid` and `~ iid(gid)` are the same
term. The two spellings are mixed here only to show that they are.

A genotypic variance of 1.401 against a true 1.5, and a residual of 0.899
against 1.0: normal sampling variation at this size. The block variance came
out at 0.0072 against a true 0.4, which is what four blocks buy you — the
whole information about that component is four numbers.

---

## Reading the output

### Component names, and their order

```r
fit$composantes_noms
```

```
[1] "gid"        "bloc"       "residuelle"
```

This order is fixed and documented. It is the order in which terms were
declared, then the residual, and it is what `V1, V2, V3` refer to in a
`vpredict` expression. A silent renumbering would make every expression wrong
on the next run, so there is none.

### The matrices

```r
fit$sigmas$gid
fit$sigma_res
```

```
         [,1]
[1,] 1.401362

          [,1]
[1,] 0.8987733
```

`sigmas` is a named list of `Sigma` matrices, one per term, always as
matrices — `1 x 1` for a single-column term. `sigma_res` is the residual, and
`sigmas_res` is the per-section list when `dsum` is used.

Read a fit with `[[ ]]`, not `$`: partial matching on lists is a trap, and
`fit$sigmas` once matched `sigmas_res`. See
[api-r.md](../api-r.md#13-a-trap-worth-knowing).

### The log-likelihood, in two conventions

```r
c(logLik = fit$logLik, logLik_asreml = fit$logLik_asreml,
  n_par = fit$n_par, n_obs = fit$n_obs)
```

```
       logLik logLik_asreml         n_par         n_obs
    -388.0888     -168.4625        3.0000      240.0000
```

remlax includes the constant `(n-p)/2 log(2 pi)`, like lme4. asreml and
sommer omit it. Here the difference is 219.6 — big enough to look like a
disagreement between models rather than a convention. Compare
`logLik_asreml` with asreml, `logLik` with lme4.

---

## The diagnostic

```r
str(fit[c("max_grad", "newton_decrement", "n_neg_eig", "n_null_dir",
          "n_at_bound", "n_fixed", "n_fixed_out", "n_par_free",
          "cond", "n_polish", "scipy_message", "composantes_degenerees")])
```

```
List of 12
 $ max_grad              : num 3.42e-14
 $ newton_decrement      : num 1.43e-28
 $ n_neg_eig             : num 0
 $ n_null_dir            : num 0
 $ n_at_bound            : num 0
 $ n_fixed               : num 0
 $ n_fixed_out           : num 0
 $ n_par_free            : num 3
 $ cond                  : num 571
 $ n_polish              : num 2
 $ scipy_message         : chr "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
 $ composantes_degenerees: list()
```

The diagnostic answers three separate questions and never merges them into a
single boolean.

**Am I at the top?** `newton_decrement` is `g' H^+ g`, the rise in
log-likelihood still available under the local quadratic model. It is in the
same units as the 3.84 of a one-degree-of-freedom likelihood-ratio test, and
it is invariant under affine reparametrisation. **This is the criterion to
read.** `max_grad` is not: on these likelihoods it is several units at
verified optima.

**Is it a peak?** `n_neg_eig` counts negative eigenvalues of the Hessian and
`n_null_dir` counts flat directions. Both zero here. A non-zero `n_neg_eig`
means the optimiser stopped at a saddle; a non-zero `n_null_dir` means a
direction the data do not identify, and every standard error involving it is
meaningless.

**Which parameters?** Three counters, kept separate because they mean three
different things. `n_at_bound` counts parameters sitting at the global bounds
(`floor = -12`, `ceil = 12` on the log scale). `n_fixed_out` (and `n_fixed`,
the same count) counts parameters held by `fixed_theta`: those are outside the
free subspace **without** being at a bound, since they are pinned to their
starting value. `n_par_free` is the dimension of the free subspace that
remains. All three categories are excluded from the diagnostic, by the KKT
condition for the first and by construction for the second — the Newton
decrement is computed only along directions the step can actually take.

`composantes_degenerees` names terms with a parameter at the floor. A variance
at the floor is not "estimated at zero", it is **unidentified**: its standard
error, its share of variance and any test touching it are void. See
[guide 3](03-spatial.md#the-nugget-that-vanishes) for a fit where this fires.

`n_polish` counts the regularised Newton steps taken after L-BFGS-B. L-BFGS-B
stops on a relative tolerance and lands about `1e-7` from the optimum on the
variance components; the polishing brings that to machine precision. Each step
is accepted only if it improves the objective.

---

## Heritability with vpredict

```r
fit2 <- rx_reml(y ~ 1, random = ~ gid + iid(bloc), data = d,
                vpredict = c(h2  = "V1/(V1+V2+V3)",
                             vg  = "V1",
                             pct = "100*V1/(V1+V2+V3)"),
                backend = "cpu", verbose = FALSE)
fit2$vpredict$composantes
fit2$vpredict$predictions
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

`composantes` gives you the numbering to write against; `predictions` gives
the values and their standard errors.

The standard errors come from the delta method on the Hessian of `-2 logL`:
`var(g) = J' (2 H^-1) J`, the factor 2 accounting for `H` being the Hessian of
`-2 logL` rather than of `-logL`. The Jacobian is obtained by automatic
differentiation, so any expression you can write is differentiated exactly.

When the Hessian is not invertible over the free subspace — a parameter at a
bound, a flat direction — the standard error is `NA`. It is never replaced by
a number.

---

## BLUPs

```r
u <- fit$blups$gid
dim(u)
head(round(u[, 1], 4), 8)
var(u[, 1])
```

```
dim(BLUP gid) = 60 x 1
[1] -0.1065 -0.9229  0.0301  0.0813 -0.4069 -2.6160 -0.5659 -0.4012

var(BLUP) = 1.20772   (retrecissement attendu vs sigma2_g = 1.40136)
```

`blups` is a named list of `q x t` matrices: levels in rows, columns of
`Sigma` in columns. The variance of the BLUPs is below the estimated genetic
variance, which is the shrinkage a BLUP is for, not an error.

---

## Fixed effects: Wald and Kenward-Roger

Wald tests are worth their asymptotic status on a large design. On a small
one they are not, and that is exactly when a test is wanted. Take an
**unbalanced** block design: five blocks of 3 to 7 plots, three treatments,
27 plots in total.

```r
set.seed(7)
tailles <- c(3, 5, 7, 6, 6)
d <- data.frame(bloc = factor(rep(seq_along(tailles), tailles)))
d$trt <- factor(rep_len(c("A", "B", "C"), nrow(d)))
b <- rnorm(length(tailles), 0, 0.7)
d$y <- 5 + c(A = 0, B = 0.8, C = 1.4)[as.character(d$trt)] +
       b[as.integer(d$bloc)] + rnorm(nrow(d), 0, 1)
table(d$bloc, d$trt)
```

```
    A B C
  1 1 1 1
  2 2 2 1
  3 2 2 3
  4 2 2 2
  5 2 2 2
```

```r
f <- rx_reml(y ~ trt, random = ~ iid(bloc), data = d,
             wald = TRUE, kenward_roger = TRUE,
             backend = "cpu", verbose = FALSE)
print(f)
```

```
Ajustement REML (cpu) : logLik -39.216376 | 2 parametres | 27 obs | 0.7 s
  max|grad| 8.95e-09 | decrement de Newton 1.37e-18 | 0 valeur(s) propre(s) negative(s)
  Sigma[bloc] 1x1, diagonale : 0.3947
  residuelle : 0.9669
  Wald (conditionnel, chi2/ddl) :
    (Intercept)    ddl  1   F   139.794   p 2.953e-32
    trt            ddl  2   F     5.091   p 0.006149
```

```r
f$wald$tests
data.frame(coef = f$wald$beta, se = f$wald$se_beta)
```

```
        terme ddl          F      chi2            p
1 (Intercept)   1 139.794105 139.79411 2.952830e-32
2         trt   2   5.091395  10.18279 6.149437e-03

      coef        se
1 5.126985 0.4336283
2 1.238082 0.4635461
3 1.326362 0.4666903
```

Two things to know about this table.

**The test is conditional (type III):** each term given all the others.
asreml's default is sequential (type I). The two agree on the last term of the
model, which is the only case where the notions coincide.

**The degrees of freedom of the denominator are not adjusted.** The p-values
are those of a `chi2/df` and are asymptotically valid. On 27 observations that
is optimistic, which is what the next table fixes.

```r
f$kenward_roger$tests
data.frame(se_brut = f$kenward_roger$se_beta_brut,
           se_KR   = f$kenward_roger$se_beta)
f$kenward_roger$second_ordre_omis
```

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

The Kenward-Roger adjustment does two separate things: it inflates the
covariance of `beta` to account for `V` being **estimated** rather than known,
and it matches moments to obtain a denominator degrees of freedom and a scale
factor. Here `trt` gets 20.19 denominator degrees of freedom instead of the
asymptotic infinity, and `p` moves from 0.0061 to 0.0164 — a factor of nearly
three on the p-value, from 27 observations.

`second_ordre_omis` is `FALSE`: the model carries no between-level correlation
parameter, so the omitted second-order term is exactly zero here. It would be
`TRUE` with an AR1 or a range in the model.

---

## Predictions

```r
pv <- rx_predict(f, classify = "trt", sed = TRUE)
print(pv)
attr(pv, "sed")
```

```
Predictions
 trt predicted.value std.error estimable
   A         5.12698  0.433628      TRUE
   B         6.36507  0.433628      TRUE
   C         6.45335  0.436859      TRUE

  erreur-type moyenne des differences : 0.465645

          A         B         C
A        NA 0.4635461 0.4666903
B 0.4635461        NA 0.4666903
C 0.4666903 0.4666903        NA
```

Every variable of the fixed model that is not in the `classify` is averaged: a
factor over its levels, a covariate at its mean. `estimable` marks
combinations inside the row space of `X`; one outside it gets `NA` rather than
a number that would depend on the parametrisation chosen.

`sed = TRUE` attaches the matrix of standard errors of differences and their
quadratic mean.

The standard errors here are the unadjusted ones, because `vcov` defaults to
`"simple"`. Pass `vcov = "kenward-roger"` to use the adjusted covariance from
the fit above.

---

## The same model in Python

The Python API takes dictionaries and numpy arrays. Nothing about factors,
formulas or data frames reaches it.

```python
import numpy as np
from remlax.fit import fit_reml
from remlax.inference import component_names, vpredict

rng = np.random.default_rng(2026)
n_gen, n_bloc = 60, 4
gid  = np.tile(np.arange(n_gen), n_bloc)
bloc = np.repeat(np.arange(n_bloc), n_gen)
n = n_gen * n_bloc
g = rng.normal(0, np.sqrt(1.5), n_gen); b = rng.normal(0, np.sqrt(0.4), n_bloc)
y = 12 + g[gid] + b[bloc] + rng.normal(0, 1.0, n)
X = np.ones((n, 1))

t_gid  = dict(name="gid",  struct="iid", t=1, rank=0, q=n_gen,
              zi=np.arange(n), zj=gid,  zx=np.ones(n), LK=None)
t_bloc = dict(name="bloc", struct="iid", t=1, rank=0, q=n_bloc,
              zi=np.arange(n), zj=bloc, zx=np.ones(n), LK=None)
res = dict(struct="iid", t=1, rank=0,
           trait=np.zeros(n, dtype=int), unit=np.arange(n))
terms = [t_gid, t_bloc]

fit = fit_reml(terms, res, y, X, verbose=False)
print(component_names(terms, res))
vp = vpredict(fit["theta"], fit["hessian"], terms, res,
              [("h2", "V1/(V1+V2+V3)"), ("vg", "V1")])
```

```
['gid', 'bloc', 'residuelle']
logLik             -411.00531386466247
logLik_asreml      -191.3790044287457
n_par              3
n_obs              240
max_grad           5.888075907683343e-13
newton_decrement   5.470948439833614e-28
n_neg_eig          0
n_at_bound         0
n_polish           2
n_iter             18
scipy_success      True
sigmas             {'gid': 1.5847079094518273, 'bloc': 0.002430569370277488}
sigma_res          1.1138165870456136
beta               [11.658782]
blups shapes       {'gid': (60, 1), 'bloc': (4, 1)}

h2   V1/(V1+V2+V3)    0.586721  se 0.060482
vg   V1               1.584708  se 0.344311
```

The numbers differ from the R fit because R's `rnorm` and numpy's generator
produce different data, not because the models differ.

The `zi`, `zj`, `zx` triplet is `Z` in COO form with **0-based** indices, and
the column of `Z` for column `a` of `Sigma` and level `l` is `(a-1)*q + l` —
level varying fastest. That convention is shared with the R side; see
[api-python.md](../api-python.md#column-order-of-z).

---

## Next

- [2. Multi-trait models](02-multi-trait.md) — several traits, `us` and `fa`,
  genomic relationship.
- [3. Spatial models](03-spatial.md) — separable AR1, metric kernels, 2D
  splines.
- [4. Explicit terms](04-explicit-terms.md) — weighted incidences and shared
  covariances.
