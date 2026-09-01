# 4. Explicit terms

The `rk_term` / `rk_model` path, for models a formula cannot express:
incidences that are not indicator matrices, and covariances shared between
terms that carry different incidences.

The worked example is a direct/indirect genetic effects (DGE/IGE) model, the
case that forced the solver to be general in the first place. It is described
in section 5 of [the design note](../note_remlkit_fr.md).

All blocks were executed with `backend = "cpu"` and their real output pasted
underneath.

---

## Why a formula is not enough

A formula term is `structure(grouping_column)`: the incidence is the indicator
matrix of a factor, one 1 per row. That covers a genotype effect, a block
effect, a multi-trait model where every column is the same indicator restricted
to one trait.

It does not cover an incidence whose entries are **weights**. A neighbourhood
effect is exactly that: plot `i` is exposed to genotype `g` through

```
W[i, g] = sum over neighbours j of plot i carrying genotype g, of w_ij
```

with `w_ij` a distance kernel. `W` is not an indicator matrix, its rows do not
sum to 1, and no factor produces it.

The `rk_term` path takes the incidence directly. A **list of `t` matrices**
gives one incidence per column of `Sigma`, which is the general form: each
column may have its own incidence, and the covariance `Sigma` links them.

---

## The design

A 20 x 12 field, 240 plots, 60 genotypes, each replicated four times.

```r
source("R/remlkit.R")
set.seed(31)
nr <- 20; nc <- 12; ngen <- 60
d <- expand.grid(coln = seq_len(nc), rown = seq_len(nr))
n <- nrow(d)
d$gid <- factor(sample(rep(seq_len(ngen), length.out = n)))
d$row <- factor(d$rown); d$col <- factor(d$coln)

## Z_DGE : indicator of the plot's own genotype
Zd <- Matrix::sparseMatrix(i = seq_len(n), j = as.integer(d$gid), x = 1,
                           dims = c(n, ngen))

## W : weighted neighbourhood incidence, radius 2, kernel 1/distance
dist <- as.matrix(dist(cbind(d$rown, d$coln)))
pds  <- ifelse(dist > 0 & dist <= 2, 1 / dist, 0)
W    <- Matrix::Matrix(pds %*% Zd, sparse = TRUE)
```

```
Z_DGE : 240 x 60, 240 non nuls
W     : 240 x 60, 2388 non nuls, poids max 2.7071
exposition moyenne sum_g W[i,g] = 7.9298
```

Mean exposure is 7.93: the neighbourhood component enters the phenotype as
`sigma2_IGE * k` with `k` of that order, not as `sigma2_IGE`. That is why
`initial_theta` divides the starting variance by the mean of `(Z K Z')_ii` —
without it `V` starts orders of magnitude too large and the optimiser spends
its budget getting back.

```r
Sig <- matrix(c(1.00, -0.45, -0.45, 0.35), 2, 2)   # true cov(DGE, IGE)
U   <- matrix(rnorm(ngen * 2), ngen, 2) %*% chol(Sig)
ar1 <- function(rho, q) rho^abs(outer(seq_len(q), seq_len(q), "-"))
Lf  <- t(chol(kronecker(ar1(0.5, nr), ar1(0.2, nc)) + diag(1e-8, n)))
d$y <- 5 + as.numeric(Zd %*% U[, 1]) + as.numeric(W %*% U[, 2]) +
       as.numeric(Lf %*% rnorm(n)) * sqrt(0.6) + rnorm(n, 0, sqrt(0.4))
```

The direct and indirect effects are **negatively correlated** (-0.76), which
is the biologically interesting case: a genotype that performs well itself
depresses its neighbours.

---

## The explicit path

```r
X <- matrix(1, n, 1, dimnames = list(NULL, "(Intercept)"))

## one term, two columns, two DIFFERENT incidences, one free 2x2 covariance
tm_g <- rk_term("dge_ige", list(Zd, W), struct = "us",
                levels = as.character(seq_len(ngen)))

## the field, as a separable AR1 x AR1 random term
idx  <- (as.integer(d$row) - 1L) * nc + as.integer(d$col)
Zf   <- Matrix::sparseMatrix(i = seq_len(n), j = idx, x = 1, dims = c(n, nr * nc))
tm_f <- rk_term("champ", Zf, t = 1L, struct = "iid",
                level = "ar1ar1", dims = c(nr, nc))

mod <- rk_model(d$y, X, terms = list(tm_g, tm_f), residual = rk_residual("iid"))
print(mod)
```

```
Modele REML 'modele' : 240 observations, 1 effets fixes
  terme dge_ige        us    t=2   q=60    K=I        (3 parametres)
  terme champ          iid   t=1   q=240   AR1xAR1    (3 parametres)
  residuelle     iid   t=1  (1 parametre)
  total : 7 parametres de variance
```

The whole idea is in the first line: **one** term with `t = 2` columns sharing
the same 60 levels, and a free `2 x 2` covariance between those columns. The
two columns have different incidences — `Zd` and `W` — which no formula can
say.

`levels =` supplies the level names; they are what `K` would be matched
against if a relationship matrix were given, and what appears in the BLUP row
names.

For the field, `dims = c(nr, nc)` tells `level = "ar1ar1"` the shape of the
grid, and `idx = (row-1)*nc + col` must follow the same convention — row
slowest, column fastest.

```r
fit <- rk_fit(mod, backend = "cpu", verbose = FALSE)
print(fit)
```

```
Ajustement REML (cpu) : logLik -436.250563 | 7 parametres | 240 obs | 2.3 s
  max|grad| 5.53e-07 | decrement de Newton 5.21e-15 | 0 valeur(s) propre(s) negative(s)
  Sigma[dge_ige] 2x2, diagonale : 1.007 0.420
  Sigma[champ] 1x1, diagonale : 0.2011
  residuelle : 0.813
```

---

## Reading the shared covariance

```r
S <- fit$sigmas$dge_ige
round(S, 4)
```

```
Sigma(DGE, IGE) estimee :
        [,1]    [,2]
[1,]  1.0067 -0.3335
[2,] -0.3335  0.4200

valeurs simulees        :
      [,1]  [,2]
[1,]  1.00 -0.45
[2,] -0.45  0.35

correlation DGE-IGE : estimee -0.5129, simulee -0.7606
rho ligne 0.9081  rho colonne 0.1515  pepite 0.8130
```

The direct variance is recovered almost exactly (1.007 against 1.00), the
indirect one is 20% high (0.420 against 0.35), and the correlation is
attenuated (-0.51 against -0.76). Sixty genotypes is not many for a
covariance between a direct and an indirect effect; the standard errors below
quantify that.

The field parameters are further off (row 0.91 against a simulated 0.5) and
the nugget has taken 0.81 of the 0.6 + 0.4 simulated. A field and a
neighbourhood effect both act on nearby plots, so they compete for the same
signal; that is the identifiability problem this class of model lives with,
not a solver failure.

```r
fit$composantes_noms
```

```
[1] "dge_ige[1,1]"  "dge_ige[2,1]"  "dge_ige[2,2]"  "champ"
[5] "champ!ar1ar11" "champ!ar1ar12" "residuelle"
```

The `!` suffix marks a between-level parameter; the trailing digit
distinguishes the two AR1 correlations of the separable field.

---

## The same model as a formula, with str()

The formula grammar does reach this model, through `str()`. `mm()` supplies an
incidence directly, and `str()` puts one covariance over several terms.

```r
fit_f <- rk_reml(y ~ 1,
                 random = ~ str(~ gid + mm(W, name = "ige"), struct = "us",
                                name = "dge_ige") + ar1(row, col),
                 residual = ~ units, data = d,
                 backend = "cpu", verbose = FALSE)
```

```
Ajustement REML (cpu) : logLik -436.250563 | 7 parametres | 240 obs | 2.4 s
  Sigma[dge_ige] 2x2, diagonale : 1.007 0.420
  Sigma[row_col] 1x1, diagonale : 0.2011
  residuelle : 0.813

ecart de logLik formule vs explicite : 0.000e+00
ecart max sur Sigma                  : 0.000e+00
```

Exactly zero difference. The two paths build the same object; `rk_reml()` is
a parser in front of `rk_model()`.

The grouped terms must share **exactly** the same levels. That shared indexing
is what gives a covariance between them any meaning, and remlkit refuses
otherwise:

```
str() : le terme '<x>' n'a pas les memes niveaux que '<y>'. Une covariance
entre termes n'a de sens qu'a niveaux partages.
```

Use the explicit path when the incidences are computed rather than looked up,
when a term needs more than two columns with different incidences, or when the
model is generated programmatically. Use `str()` when the incidences are
already in the data frame's environment and the formula stays readable.

---

## BLUPs of both effects

```r
u <- fit$blups$dge_ige
dim(u)
round(head(u, 5), 4)
```

```
dim(blups) = 60 x 2  (q niveaux x t colonnes)
        [,1]    [,2]
[1,]  1.9972 -0.5476
[2,] -0.2518 -0.5141
[3,] -1.0051  1.3099
[4,] -1.0092  0.1448
[5,]  0.2021 -0.6163

correlation des BLUPs DGE/IGE  : -0.6676
correlation des vraies valeurs : -0.6984
```

Column 1 is the direct effect of each genotype, column 2 its effect on
neighbours. The correlation of the BLUPs (-0.668) is closer to the realised
correlation of the simulated effects (-0.698) than the estimated covariance
parameter was (-0.513) — BLUPs are shrunk towards the fitted covariance but
also carry the data's own information.

---

## Functions of the shared covariance

The interesting quantities in a DGE/IGE model are functions of `Sigma`: the
direct-indirect correlation, and the **total heritable variance** available to
selection, `V1 + 2*V2 + V3` for one neighbour.

```r
fv <- rk_fit(mod, backend = "cpu", verbose = FALSE,
             vpredict = c(var_dge = "V1", cov_di = "V2", var_ige = "V3",
                          r_di = "V2/sqrt(V1*V3)",
                          T2   = "V1 + 2*V2 + V3"))
fv$vpredict$composantes
fv$vpredict$predictions
```

```
  i           nom     valeur
1 1  dge_ige[1,1]  1.0066785
2 2  dge_ige[2,1] -0.3334784
3 3  dge_ige[2,2]  0.4200020
4 4         champ  0.2010646
5 5 champ!ar1ar11  0.9081110
6 6 champ!ar1ar12  0.1514925
7 7    residuelle  0.8130207

      nom     expression     valeur        se
1 var_dge             V1  1.0066785 0.2513742
2  cov_di             V2 -0.3334784 0.1130389
3 var_ige             V3  0.4200020 0.1009573
4    r_di V2/sqrt(V1*V3) -0.5128574 0.1486077
5      T2 V1 + 2*V2 + V3  0.7597236 0.2528442
```

Now the estimates can be read. The direct-indirect correlation is
`-0.51 +/- 0.15`: comfortably negative, and its distance from the simulated
-0.76 is about 1.7 standard errors. The total heritable variance is
`0.76 +/- 0.25`, well below the direct variance alone (1.01) — the negative
covariance removes a quarter of the selectable variance, which is the whole
point of fitting the model.

Note that the between-level parameters of the field are components too
(`V5`, `V6`), so a `vpredict` expression can involve an AR1 correlation.

---

## Does the indirect effect earn its parameters?

```r
tm_d <- rk_term("dge", d$gid)
mod0 <- rk_model(d$y, X, terms = list(tm_d, tm_f), residual = rk_residual("iid"))
fit0 <- rk_fit(mod0, backend = "cpu", verbose = FALSE)
```

```
DGE seul       : logLik  -480.8700, 5 parametres
DGE + IGE (us) : logLik  -436.2506, 7 parametres
LRT : chi2 = 89.2389, 2 ddl, p = 4.188e-20
```

Two extra parameters (the indirect variance and its covariance with the direct
effect) buy 44.6 log-likelihood units. Unlike the `us`/`rr` comparison in
[guide 2](02-multi-trait.md#reduced-rank-and-an-aliasing-that-looks-like-a-test),
this one is a genuine nested comparison: the null model is `Sigma` with its
second row and column set to zero, and nothing else in the model can absorb
`W`.

The usual caveat still applies: the null puts a variance on the boundary of
its space, so the reference distribution is a mixture rather than a plain
`chi2_2`, and the p-value above is conservative. At `4e-20` that hardly
matters.

---

## Checklist for the explicit path

| you need | write |
|---|---|
| a term whose incidence you computed | `rk_term(name, Zmatrix, t = 1)` |
| `t` columns with **different** incidences | `rk_term(name, list(Z1, ..., Zt), struct = "us")` |
| a stacked `n x (t*q)` incidence | `rk_term(name, Z, t = t)` — order `(a-1)*q + level` |
| a relationship matrix | `K =`, with `dimnames` matching `levels =` |
| a separable field as a term | `level = "ar1ar1", dims = c(nr, nc)` |
| a metric structure | `level = "iexp"`, `coord = <q x 2 matrix>` |
| a residual whose formula the parser refuses | `rk_residual(struct, trait =, unit =, level =, dims =)` |
| the design on disk for the solver | `rk_export(mod, dir)` |

Two invariants apply to everything on this page.

**Column order of `Z`:** `column = (a - 1) * q + level`, level varying
fastest. `Sigma (x) K` follows the same order. Changing it on one side only
gives a different model with no error message.

**Order of `theta`:** for each term, `Sigma` parameters then between-level
parameters; terms in declaration order, then the residual. That is what
`fit$composantes_noms` reflects and what every `vpredict` expression depends
on.

---

## Next

- [api-r.md](../api-r.md) — every argument of `rk_term`, `rk_residual`,
  `rk_model`, `rk_fit`.
- [api-python.md](../api-python.md) — the same model as Python dictionaries.
- [structures.md](../structures.md) — the catalogue.
