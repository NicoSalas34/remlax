# Multivariate genome scan and multi-trait Neighbor GWAS: design

Status: agreed on 2026-10-31 (choices below), implementation started. References are
collected in `multivariate_gwas_literature.md` (same folder).

## Where the univariate scan stands

`rx_scan()` (R/scan.R, inst/python/remlax/scan.py) is a scan at fixed V. The
variance components come from the null fit; each SNP enters as a fixed effect
and

    b = (S'PS)^-1 S'Py,   Var(b) = (S'PS)^-1,   chi2 = b_A' [(S'PS)^-1]_AA^-1 b_A

with S the covariates of the SNP and A the tested columns. Score and Wald
statistics coincide at fixed V. S has one column per family:

    dir  s = Z_dir m          direct dose of the focal genotype
    ind  s = Z_ind m          neighbour dose (weighted incidence of the neighbours)
    sim  s = (Z_dir m) * (Z_ind m)    focal x neighbour similarity (Sato et al. 2021)

All the work for `dir` and `ind` happens in genotype dimension through
c_f = Z_f'Py and C_fg = Z_f'PZ_g; `sim` needs P s for each SNP, the only step
in dimension n.

## Multivariate scan: the model

Long format, one row per unit x trait, T traits, n rows in all. The null model
is any multi-trait remlax model, for example

    y ~ trait,  random = ~ us(gen, K = G) + us(voisin, K = G),  residual = ~ us(trait):units

so V carries the genetic and residual covariances between traits, and traits
measured on different subsets of units need nothing special.

For SNP j, family f and trait k, the covariate is the family covariate kept on
the rows of trait k:

    s_{f,k} = D_k s_f,     D_k = diag(1[trait = k])

S_j = [s_{f,k}] has F x T columns (F families). The incidences D_k Z_f are
exactly the per-trait incidences `Zl` that a multi-trait term already holds,
so the projection is the univariate one applied to F x T columns:

    c_{f,k} = Z_f' D_k P y,    C_{(f,k),(g,l)} = Z_f' D_k P D_l Z_g

Cost of the projection: F x T products (n x n)(n x q). Cost per SNP for `dir`
and `ind`: O((FT)^2 q). For `sim`: T products in dimension n per SNP block.

## Tests

Each test is a linear hypothesis on b, chi2 with its df:

| written | hypothesis | df |
|---|---|---|
| `dir` | beta_dir = 0 on every trait (any effect) | T |
| `dir[y1]` | beta_dir,y1 = 0, other traits free | 1 |
| `dir@common` | one shared beta_dir = 0, in the model with a single column sum_k s_dir,k | 1 |
| `dir@het` | beta_dir,1 = ... = beta_dir,T (heterogeneity between traits) | T - 1 |
| `ind|dir`, `sim|dir+ind` | conditional tests, every trait on both sides | T x (tested families) |
| `dir+ind` | joint, every trait | 2T |

`@common` and `@het` follow the any / common / interaction effect tests of the
multi-trait LMM literature (LIMIX, Casale et al. 2015). At fixed V,
chi2(any) = chi2(common) + chi2(het) holds exactly for nested linear models;
this identity is a unit test. With T = 1 every test reduces to the current
univariate scan, which is a second unit test.

## Multi-trait Neighbor GWAS

Sato et al. (2021) test the focal effect and the neighbour similarity effect,
with a null model holding a self kinship K1 and a neighbour kinship K2, and
choose the neighbourhood scale s by the proportion of variance explained
(PVE). The multi-trait generalisation:

    y = X b + Z1 u1 + Z2 u2 + e
    u1 ~ N(0, Sigma_1 (x) K1),  u2 ~ N(0, Sigma_2 (x) K2),  e ~ N(0, Sigma_e (x) I)

with Sigma_1, Sigma_2, Sigma_e covariances between traits (us, or fa when T is
large). Per SNP: `dir` and `sim` per trait, the tests above, and the IGE model
of Muir/Bijma with `ind` as well. New outputs compared with the univariate
paper: the genetic correlation between traits of the neighbour effects
(Sigma_2), a PVE per trait for the self and neighbour terms, and the choice of
the scale s by the multi-trait REML likelihood (AIC over a grid of s).

Validation, since no published multi-trait Neighbor GWAS exists: the
univariate case against rNeighborGWAS on its example data; reduction T = 1;
exact algebra against a dense per-SNP GLS in R; asreml with the variance
parameters fixed and the SNP as a trait-specific fixed effect (Wald test) on a
few SNPs; type I error and power by simulation under a multi-trait IGE model.

## Implementation steps

1. scan.py: families split by trait, test grammar with `[traits]`, `@common`,
   `@het`; reduction to the univariate path when T = 1.
2. R/scan.R: multi-trait models (incidences taken from the `Zl` of a
   multi-trait term), outputs per trait, `rx_scan_threshold`.
3. LOCO (R/scan_loco.R) in the multi-trait case.
4. Tests: the two identities above, dense GLS in R, asreml at fixed variances.
5. Validation on published data: a multi-environment GWAS (DROPS maize, the
   environments as traits), rNeighborGWAS example data for the univariate
   Sato model, simulation for the multi-trait Sato model.
6. Documentation: vignette section, NEWS.

## Decisions (2026-10-31)

- Fixed-V scan on the whole genome, then an exact refit for the SNPs above a
  threshold: REML refit with the SNP columns in X (V re-estimated), Wald test
  of the same linear hypothesis, as the Wald test of GEMMA mvLMM. An LRT would
  need ML fits, which remlax has only for the spatial lag.
- All four multivariate tests: any effect (T df), per trait (1 df), common
  effect (1 df), heterogeneity (T - 1 df).
- Sato's K2 built by remlax, with the definitions of rNeighborGWAS 1.2.5
  (Sato, Yamamoto, Shimizu & Nagano 2021, Heredity 126:597,
  doi:10.1038/s41437-020-00401-w): with q markers coded +/-1, X1 the focal
  doses and X2 the similarity covariates of all markers,
      K1 = ((q - 1)/2 + X1 X1'/2) / (q - 1),   K2 = X2 X2' / (q - 1),
  both projected to the nearest positive definite matrix. A user-supplied K2
  stays possible. Sato fits no covariance between the self and neighbour
  polygenic terms; the multi-trait model keeps them independent by default
  and allows str() for a covariance (Sato & Hamazaki 2026 add one, one trait
  at a time).

## Notes from the literature (multivariate_gwas_literature.md)

- At fixed V, score, Wald and ML LRT coincide; GEMMA's score test and Sato's
  per-SNP LRT are this statistic. Conditional test of block b given a:
  U_b.a = U_b - I_ba I_aa^-1 U_a, I_b.a = I_bb - I_ba I_aa^-1 I_ab,
  T = U_b.a' I_b.a^-1 U_b.a.
- Fixed V is conservative in small related samples (Zhou & Stephens 2014:
  top p-values about 10 times too large in 656 related mice), hence the exact
  refit of the top SNPs.
- Animal IGE GWAS (Baud et al. 2021, Brinker et al. 2018) test direct and
  indirect effects jointly in the model and report conditional tests;
  conditioning on the local direct genotype is needed for calibration.
- Long format handles missing traits (MTMM); GEMMA and LIMIX need complete
  phenotypes.
