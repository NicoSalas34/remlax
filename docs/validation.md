# Validation

Every number on this page was produced on 2026-09-28 by the test suites of this repository at git revision `f576638 (+ working tree)` (working tree of that day, see the commit that follows it). The suites that need asreml or pbkrtest ran on the user's SLURM cluster inside the apptainer image `ige_reml.sif`; the others ran there too and, where noted, on a local CPU. The measurements were extracted from the logs by `validation/parse_logs.py` and this page is generated from the resulting table (`validation/results/checks_2026-09-28.csv`) by `validation/make_validation_md.py`. No value here was typed by hand.

**399 checks, 387 passed, 12 failed.**

Provenance of the rows:

- re-measured 2026-09-28, ssh:cluster cpu-dedicated, ige_reml.sif: 343 checks
- re-measured 2026-09-28, local CPU (conda remlax-r + remlax-jax): 56 checks

Software versions read from the log headers:

- R: 4.6.0 (ssh:cluster cpu-dedicated, ige_reml.sif)
- nlme: 3.1.169 (ssh:cluster cpu-dedicated, ige_reml.sif)
- lme4: 2.0.1 (ssh:cluster cpu-dedicated, ige_reml.sif)
- sommer: 4.4.5 (ssh:cluster cpu-dedicated, ige_reml.sif)
- asreml: 4.2.0.480 (ssh:cluster cpu-dedicated, ige_reml.sif)
- pbkrtest: 0.5.5 (ssh:cluster cpu-dedicated, ige_reml.sif)
- jax: 0.11.1 (ssh:cluster cpu-dedicated, ige_reml.sif)

The check labels in the tables below are in French. They are the strings the test suites print, quoted verbatim, and the suites are written in French like the rest of the code. Translating them would give a more readable table and a trace that no longer matches the logs, so they are left alone; the section headings and the commentary are in English.

Suites whose log was not available for this run, and therefore absent from the table: `02_stress_s50.log`, `02_stress_s100.log`, `21_parite.log`. See the last section.

## Summary

| Suite | Checks | Compared against | Passed |
|---|---|---|---|
| Internal algebra | 25 | the solver against itself and against closed-form results | 25/25 |
| Unit tests (pytest) | 1 | closed forms and independent numpy references | 1/1 |
| Random sweep | 1 | properties over randomly drawn designs | 1/1 |
| Against lme4 and sommer | 31 | two open-source reference implementations | 31/31 |
| Against nlme | 61 | the reference implementation for correlation structures | 61/61 |
| Against lme4, random regressions, nesting and PEV | 30 | the reference implementation for crossed and nested random effects | 30/30 |
| Against sommer, genomic and multi-trait models | 19 | the reference implementation for genomic relationship matrices | 19/19 |
| Against asreml, structures and inference | 28 | the field reference | 28/28 |
| Against asreml, the rest of the catalogue | 22 | the field reference | 22/22 |
| Against asreml, the remaining structures | 111 | the field reference | 99/111 |
| Against asreml, BLUPs | 4 | the field reference | 4/4 |
| A full model of known truth | 11 | asreml, and the simulated parameters | 11/11 |
| Sparse engine: parameterisation parity | 30 | the dense engine, at a common theta | 30/30 |
| Sparse engine: same likelihood as the dense engine | 16 | the dense engine, warm-started in both directions | 16/16 |
| Sparse engine: precision path | 9 | an independent dense-algebra REML | 9/9 |

## Internal algebra

Gradients against central finite differences, the Hessian against a second differentiation, log-determinant and quadratic form against a direct computation, parameter counts, and the refusal of designs that would make V singular. These need no other software: they are the properties the implementation must have to be an implementation of REML at all.

**1. structures : nombre de parametres, positivite, aller-retour**

| Check | Measured | Verdict |
|---|---|---|
| iid(t=1,r=0) : p=1, symetrique, PSD | min eig 1.70e+00 | pass |
| iid(t=4,r=0) : p=1, symetrique, PSD | min eig 1.70e+00 | pass |
| diag(t=4,r=0) : p=4, symetrique, PSD | min eig 1.70e+00 | pass |
| us(t=2,r=0) : p=3, symetrique, PSD | min eig 1.70e+00 | pass |
| us(t=6,r=0) : p=21, symetrique, PSD | min eig 1.70e+00 | pass |
| fa(t=5,r=1) : p=10, symetrique, PSD | min eig 1.70e+00 | pass |
| fa(t=5,r=3) : p=17, symetrique, PSD | min eig 1.70e+00 | pass |
| fa(t=8,r=4) : p=34, symetrique, PSD | min eig 1.70e+00 | pass |
| us(t=2) represente n'importe quelle matrice PSD | ecart 1.78e-15 | pass |
| us(t=5) represente n'importe quelle matrice PSD | ecart 8.88e-16 | pass |
| us(t=9) represente n'importe quelle matrice PSD | ecart 1.78e-15 | pass |

**2. gradient autodiff contre differences finies centrees**

| Check | Measured | Verdict |
|---|---|---|
| gradient exact (t=1, K=False) | ecart relatif 2.61e-09 | pass |
| gradient exact (t=1, K=True) | ecart relatif 1.09e-09 | pass |
| gradient exact (t=3, K=False) | ecart relatif 8.28e-09 | pass |
| gradient exact (t=3, K=True) | ecart relatif 2.23e-08 | pass |

**3. assemblage de V : symetrie, positivite, coherence Kronecker**

| Check | Measured | Verdict |
|---|---|---|
| V symetrique | - | pass |
| V definie positive | min eig 4.000e-01 | pass |
| V == Z (Sigma (x) K) Z' + R  (voie independante) | ecart max 5.33e-15 | pass |

**4. REML analytique : plan a un facteur, equilibre**

| Check | Measured | Verdict |
|---|---|---|
| a=20 m=6 : sigma2 analytiques retrouvees | ecarts 9.6e-10 / 8.3e-10, decrement 5.4e-17 | pass |
| a=12 m=10 : sigma2 analytiques retrouvees | ecarts 1.5e-10 / 2.1e-10, decrement 3.5e-18 | pass |
| a=40 m=3 : sigma2 analytiques retrouvees | ecarts 7.4e-10 / 4.1e-09, decrement 6.9e-16 | pass |

**5. reproductibilite et invariance au peripherique**

| Check | Measured | Verdict |
|---|---|---|
| deux appels identiques donnent le meme optimum | ecart 0.00e+00 | pass |
| depart perturbe -> meme optimum | ecart 0.00e+00 | pass |

**6. cas degeneres : le solveur refuse plutot que de mentir**

| Check | Measured | Verdict |
|---|---|---|
| structure inconnue rejetee | - | pass |
| X singuliere : erreur explicite | ValueError | pass |

## Unit tests (pytest)

The pytest suite under tests/python: one test per feature, each against an independent reference (numpy Cholesky REML written out longhand, closed-form correlation matrices, finite differences). Collected here as a single line; docs/tests-matrix.md maps every feature to its test.

**pytest, tests/python**

| Check | Measured | Verdict |
|---|---|---|
| 169 tests collected by pytest | 168 passed, 1 skipped, 0 failed, 0 errors in 121.20 s | pass |

## Random sweep

Configurations are drawn at random (structure, dimensions, number of traits, seed) and fitted. Nothing is compared to a reference: the sweep asks only whether the solver ever produces something impossible - an exception, a non-finite log-likelihood, a covariance matrix that is not positive semi-definite, a run that does not converge.

**random configuration sweep, seed 0**

| Check | Measured | Verdict |
|---|---|---|
| 200 configurations: no exception, finite logLik, PSD Sigma, converged | clean 198/200; unstable optima 2 (best-of-restart gains 0.0085, 0.0902 in -2logL) | pass |

## Against lme4 and sommer

The models both packages can fit: a single random factor on three layouts, two crossed factors, a genomic relationship matrix, and a bivariate unstructured covariance. Agreement is required on the restricted log-likelihood first, then on the variance components, the fixed effects and the BLUPs.

**1. un facteur aleatoire, contre lme4**

| Check | Measured | Verdict |
|---|---|---|
| a=25 m=5 : -2logL identique | 451.397867339 vs 451.397867339 | pass |
| a=25 m=5 : composantes de variance | s2g 1.361456/1.361456 | pass |
| a=25 m=5 : effets fixes | beta 1.7588 0.80093 | pass |
| a=25 m=5 : BLUPs | ecart max 7.46e-09 | pass |
| a=10 m=12 : -2logL identique | 391.341158079 vs 391.341158079 | pass |
| a=10 m=12 : composantes de variance | s2g 2.231905/2.231905 | pass |
| a=10 m=12 : effets fixes | beta 1.70765 0.74394 | pass |
| a=10 m=12 : BLUPs | ecart max 2.02e-10 | pass |
| a=60 m=3 : -2logL identique | 653.887073882 vs 653.887073882 | pass |
| a=60 m=3 : composantes de variance | s2g 2.502726/2.502726 | pass |
| a=60 m=3 : effets fixes | beta 2.30711 0.79389 | pass |
| a=60 m=3 : BLUPs | ecart max 7.71e-09 | pass |

**2. deux facteurs croises, contre lme4**

| Check | Measured | Verdict |
|---|---|---|
| croise : -2logL identique | 380.944211187 vs 380.944211187 | pass |
| croise : les trois variances | 1.94207 0.20123 0.90019 | pass |

**3. matrice de parente, contre sommer**

| Check | Measured | Verdict |
|---|---|---|
| parente : composantes de variance vs sommer | s2g 0.99215/0.99211 \| s2e 0.98662/0.98663 | pass |
| parente : la GRM ameliore l'ajustement (K vs I) | 515.7869 vs 516.7093 | pass |

**4. deux caracteres, structure us, contre sommer**

| Check | Measured | Verdict |
|---|---|---|
| us : covariance genetique 2x2 vs sommer | remlax 1.3169 0.8723 0.8723 1.0382 \| sommer 1.3169 0.8723 0.8723 1.0381 | pass |
| us : covariance residuelle 2x2 vs sommer | remlax 0.849 0.1019 0.1019 1.2783 \| sommer 0.849 0.1019 0.1019 1.2783 | pass |
| us : correlation genetique retrouvee | r 0.7460 vs 0.7460 | pass |

**5. structures emboitees : us doit dominer diag, qui domine iid**

| Check | Measured | Verdict |
|---|---|---|
| emboitement : -2logL decroissante iid >= diag >= us | iid=1324.4543 diag=1324.0042 us=1299.9284 | pass |

**6. incidence PONDEREE (le cas qu'une formule ne sait pas dire)**

| Check | Measured | Verdict |
|---|---|---|
| incidence ponderee : ajustement fini et defini | s2 1.5621 (vrai 1.21) \| s2e 0.6332 (vrai 0.64) \| decrement 1.0e-22 | pass |

**7. interface par formule : meme resultat que la voie explicite**

| Check | Measured | Verdict |
|---|---|---|
| formule ~ g : -2logL identique a lme4 | ecart 2.27e-13 | pass |
| formule ~ iid(g)+iid(b) : -2logL identique a lme4 | ecart 1.71e-13 | pass |
| formule us(gid) == voie explicite rx_term/rx_model | ecart 0.00e+00 | pass |

**8. bascule de backend : le modele ne change pas avec la machine**

| Check | Measured | Verdict |
|---|---|---|
| backend auto et cpu donnent le meme ajustement | cpu vs cpu, ecart 0.00e+00 | pass |
| le backend retenu est annonce dans le resultat | cpu | pass |

**9. erreurs : messages exploitables plutot que resultats douteux**

| Check | Measured | Verdict |
|---|---|---|
| terme inconnu rejete | - | pass |
| colonne absente rejetee | - | pass |
| structure multi-caractere sans trait rejetee | - | pass |
| X de rang deficient rejetee | - | pass |
| niveau absent de K rejete | - | pass |

## Against nlme

nlme::gls and nlme::lme on the same data: corAR1, corCAR1, corARMA (AR2 and ARMA(1,1)), corCompSymm, corExp, corGaus, corLin, corSpher, corSymm with varIdent (an unstructured covariance) and varIdent alone (one residual variance per group), then a random intercept with an AR1 residual. A grouped corStruct is written in remlax by declaring the group as the trait of the residual, residual = ~ id(g):ar1(t) with trait = g. Agreement on -2logL, the residual variance, the correlation parameters, the fixed effects and their standard errors.

**1. corAR1 : AR1 dans chaque groupe, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| constante de logLik : nlme = remlax$logLik | nlme -182.486858 \| remlax -182.486858 \| logLik_asreml -74.052112 (+(n-p)/2 log 2pi = 108.4347) | pass |
| corAR1 : -2logL identique | 364.973716966 vs 364.973716966 | pass |
| corAR1 : variance residuelle | 1.627385 vs 1.627385 | pass |
| corAR1 : effets fixes | ecart rel max 1.57e-09 | pass |
| corAR1 : erreurs-types des effets fixes | ecart rel max 1.84e-08 | pass |
| corAR1 : rho | 0.5338496 vs 0.5338496 | pass |

**2. corCAR1 : AR1 continu a temps irreguliers, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| corCAR1 : -2logL identique | 232.839518984 vs 232.839518984 | pass |
| corCAR1 : variance residuelle | 1.152159 vs 1.152159 | pass |
| corCAR1 : effets fixes | ecart rel max 2.04e-09 | pass |
| corCAR1 : erreurs-types des effets fixes | ecart rel max 1.74e-08 | pass |
| corCAR1 : phi (correlation par unite de temps) | 0.7287346 vs 0.7287346 (vrai 0.7) | pass |

**3. corARMA(p=2) : AR2, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| ar2 : -2logL identique | 364.591914889 vs 364.591914889 | pass |
| ar2 : variance residuelle | 1.899831 vs 1.899831 | pass |
| ar2 : effets fixes | ecart rel max 1.60e-09 | pass |
| ar2 : erreurs-types des effets fixes | ecart rel max 1.17e-08 | pass |
| ar2 : phi1 et phi2 | remlax 0.526414 0.145188 \| nlme 0.526414 0.145188 | pass |

**4. corARMA(p=1, q=1) : ARMA(1,1), contre gls**

| Check | Measured | Verdict |
|---|---|---|
| arma : -2logL identique | 348.507672823 vs 348.507672823 | pass |
| arma : variance residuelle | 1.461911 vs 1.461911 | pass |
| arma : effets fixes | ecart rel max 1.43e-08 | pass |
| arma : erreurs-types des effets fixes | ecart rel max 2.44e-08 | pass |
| arma : phi et theta (ordre remlax : theta, phi) | remlax theta 0.389765 phi 0.259672 \| nlme phi 0.259672 theta 0.389765 | pass |

**5. corCompSymm : correlation uniforme, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| cor : -2logL identique | 365.567018442 vs 365.567018442 | pass |
| cor : variance residuelle | 1.481197 vs 1.481197 | pass |
| cor : effets fixes | ecart rel max 1.93e-10 | pass |
| cor : erreurs-types des effets fixes | ecart rel max 5.00e-09 | pass |
| cor : rho uniforme | 0.2951065 vs 0.2951065 | pass |

**6. corExp : decroissance exponentielle en 1D, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| exp : -2logL identique | 229.403629489 vs 229.403629489 | pass |
| exp : variance residuelle | 0.633634 vs 0.633633 | pass |
| exp : effets fixes | ecart rel max 6.23e-08 | pass |
| exp : erreurs-types des effets fixes | ecart rel max 9.81e-07 | pass |
| exp : rho = exp(-1/portee nlme) | 0.6429754 vs 0.6429748 (portee 2.2642, vraie 4) | pass |
| exp : l'ajustement a bouge de son point de depart | 9 iterations, rho 0.6430 | pass |

**7. corGaus : decroissance gaussienne en 1D, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| gau : -2logL identique | 235.903503011 vs 235.903503011 | pass |
| gau : variance residuelle | 1.106024 vs 1.106024 | pass |
| gau : effets fixes | ecart rel max 1.41e-10 | pass |
| gau : erreurs-types des effets fixes | ecart rel max 1.96e-09 | pass |
| gau : rho = exp(-1/portee^2) | 0.8895193 vs 0.8895193 (portee 2.9226, vraie 3) | pass |

**8. corLin : tente lineaire tronquee (lvr), contre gls**

| Check | Measured | Verdict |
|---|---|---|
| lvr (portee gls) : -2logL identique | 188.616692460 vs 188.616692460 | pass |
| lvr (portee gls) : variance residuelle | 0.780188 vs 0.780188 | pass |
| lvr (portee gls) : effets fixes | ecart rel max 1.35e-14 | pass |
| lvr (portee gls) : erreurs-types des effets fixes | ecart rel max 1.92e-14 | pass |
| lvr : portee lue = portee imposee | 6.02241 vs 6.02241 (vraie 6) | pass |
| lvr : optimum libre de remlax pas moins bon que gls | remlax 188.616692 (portee 6.0224, 4 redemarrages) \| meilleur gls sur 2 departs 188.616692 (portee 6.0224) | pass |

**9. corSpher : spherique en 2D, contre gls**

| Check | Measured | Verdict |
|---|---|---|
| sph : -2logL identique | 297.607416797 vs 297.607416797 | pass |
| sph : variance residuelle | 0.982703 vs 0.982703 | pass |
| sph : effets fixes | ecart rel max 4.22e-10 | pass |
| sph : erreurs-types des effets fixes | ecart rel max 4.91e-09 | pass |
| sph : portee | 7.32780 vs 7.32780 (vraie 8) | pass |

**10. corSymm + varIdent : covariance non structuree (us), contre gls**

| Check | Measured | Verdict |
|---|---|---|
| us : -2logL identique | 458.854629436 vs 458.854629445 | pass |
| us : les 10 composantes de Sigma | ecart abs max / max var 5.78e-06 \| diag remlax 0.8898 1.5638 2.2088 0.953 \| nlme 0.8898 1.5638 2.2088 0.953 | pass |
| us : effets fixes | ecart rel max 6.50e-07 | pass |

**11. varIdent seul : une variance residuelle par groupe (diag), contre gls**

| Check | Measured | Verdict |
|---|---|---|
| diag : -2logL identique | 508.562150916 vs 508.562150916 | pass |
| diag : les cinq variances | remlax 0.206 0.9976 1.8157 4.1048 8.9001 \| nlme 0.206 0.9976 1.8157 4.1048 8.9001 | pass |
| diag : effets fixes | ecart rel max 1.81e-07 | pass |

**12. lme : intercept aleatoire + corAR1 intra-groupe, contre lme**

| Check | Measured | Verdict |
|---|---|---|
| lme AR1 : -2logL identique | 348.881810570 vs 348.881810570 | pass |
| lme AR1 : variance du groupe | 1.545201 vs 1.545200 | pass |
| lme AR1 : variance residuelle | 0.888761 vs 0.888762 | pass |
| lme AR1 : rho | 0.3242477 vs 0.3242487 | pass |
| lme AR1 : effets fixes | ecart rel max 3.66e-07 | pass |
| lme AR1 : erreurs-types des effets fixes | ecart rel max 2.92e-07 | pass |
| lme AR1 : BLUPs | ecart max 8.76e-07 | pass |

## Against lme4, random regressions, nesting and PEV

Correlated and independent random slopes (a two-column us and diag term), a nested design, three crossed factors on an unbalanced layout, very unequal group sizes. Besides -2logL, components, fixed effects and their standard errors: the BLUPs, and the prediction error variance checked against the mixed-model-equations formula computed by hand (lme4's condVar follows a different convention, without the uncertainty on beta, and is checked against its own formula).

**1. pentes aleatoires correlees (1 + x | g) : us a deux colonnes**

| Check | Measured | Verdict |
|---|---|---|
| pentes us : -2logL identique | 708.105175507 vs 708.105175507 | pass |
| pentes us : effets fixes | ecart rel max 8.52e-10 | pass |
| pentes us : erreurs-types des effets fixes | ecart rel max 4.55e-08 | pass |
| pentes us : les trois composantes de Sigma | remlax 1.04933 0.31753 0.51161 \| lme4 1.04933 0.31753 0.51161 | pass |
| pentes us : variance residuelle | 0.649963 vs 0.649963 | pass |
| pentes us : BLUPs (intercepts et pentes) | ecart max 2.25e-08 sur 60 valeurs | pass |
| pentes us : PEV remlax = formule des MME (avec beta) | ecart rel max 9.22e-15 \| PEV moyenne 0.10677 | pass |
| pentes us : condVar lme4 = formule sans beta | ecart rel max 2.67e-08 \| condVar moyenne 0.08346 (les deux conventions different) | pass |

**2. pentes aleatoires independantes (1 + x || g) : diag a deux colonnes**

| Check | Measured | Verdict |
|---|---|---|
| pentes diag : -2logL identique | 768.020691545 vs 768.020691545 | pass |
| pentes diag : effets fixes | ecart rel max 9.72e-10 | pass |
| pentes diag : erreurs-types des effets fixes | ecart rel max 9.51e-09 | pass |
| pentes diag : les deux variances | remlax 1.18772 0.49438 \| lme4 1.18772 0.49438 | pass |
| pentes diag : covariance nulle par construction | 0.0e+00 | pass |

**3. emboitement (1 | g/h) : deux termes iid**

| Check | Measured | Verdict |
|---|---|---|
| emboite : -2logL identique | 764.300248907 vs 764.300248907 | pass |
| emboite : effets fixes | ecart rel max 2.60e-14 | pass |
| emboite : erreurs-types des effets fixes | ecart rel max 5.07e-08 | pass |
| emboite : variances g et h dans g | g 1.53208/1.53208 \| h:g 0.26695/0.26695 \| res 1.05729/1.05729 | pass |
| emboite : BLUPs des deux termes | ecart max 1.61e-08 / 2.46e-08 | pass |

**4. trois facteurs croises, dispositif desequilibre**

| Check | Measured | Verdict |
|---|---|---|
| 3 croises : -2logL identique | 1232.919208193 vs 1232.919208193 | pass |
| 3 croises : effets fixes | ecart rel max 8.93e-10 | pass |
| 3 croises : erreurs-types des effets fixes | ecart rel max 7.16e-08 | pass |
| 3 croises : les quatre variances | remlax 0.81044 0.69056 0.18664 1.02067 \| lme4 0.81044 0.69056 0.18664 1.02067 | pass |
| 3 croises : BLUPs des trois termes | ecart max 1.57e-08 | pass |
| 3 croises : PEV des trois termes = formule des MME | ecart rel max 1.23e-14 | pass |
| 3 croises : PEV de g decroit avec la replication | cor(PEV, n_rep) = -0.982 | pass |

**5. un facteur, tailles de groupe tres inegales, deux covariables**

| Check | Measured | Verdict |
|---|---|---|
| desequilibre : -2logL identique | 888.893080406 vs 888.893080406 | pass |
| desequilibre : effets fixes | ecart rel max 3.37e-11 | pass |
| desequilibre : erreurs-types des effets fixes | ecart rel max 8.05e-10 | pass |
| desequilibre : composantes de variance | s2g 1.603755/1.603755 \| s2e 1.214474/1.214474 | pass |
| desequilibre : BLUPs | ecart max 8.85e-10 (tailles de 1 a 12) | pass |

## Against sommer, genomic and multi-trait models

A GRM on one term with BLUPs and PEV; three traits with us genetic and us residual covariance and a GRM; the same with diag and the likelihood ratio between the two; additive plus dominance (two relationship matrices on the same factor); a multi-environment trial with one genetic variance per environment. sommer is tightened to tolParConvLL = 1e-12; its log-likelihood carries a data-dependent constant, so only differences of -2logL are compared.

**1. une GRM, univarie : composantes, effets fixes, BLUP, PEV**

| Check | Measured | Verdict |
|---|---|---|
| GRM : composantes de variance | s2g 1.350769/1.350769 \| s2e 0.844907/0.844907 | pass |
| GRM : effets fixes | ecart rel max 2.15e-12 | pass |
| GRM : erreurs-types des effets fixes | ecart rel max 3.21e-10 | pass |
| GRM : BLUPs | ecart max 2.67e-10 | pass |
| GRM : PEV des BLUPs | ecart rel max 7.24e-10 \| PEV moyenne 0.17607 | pass |
| GRM : constante de logLik de sommer (mesuree) | sommer -42.2549 \| remlax logLik_asreml -135.2526 \| ecart 92.9977 ((n-p)/2 log var(y) = 87.6572) | pass |

**2. trois caracteres, us genetique avec GRM et us residuelle**

| Check | Measured | Verdict |
|---|---|---|
| us3 + GRM : les 6 composantes genetiques | ecart rel max 2.33e-09 \| diag remlax 1.3404 1.0058 1.0875 \| sommer 1.3404 1.0058 1.0875 | pass |
| us3 + GRM : les 6 composantes residuelles | ecart rel max 5.14e-10 | pass |
| us3 + GRM : les 3 correlations genetiques | remlax 0.5164 0.1735 0.6232 \| sommer 0.5164 0.1735 0.6232 | pass |
| us3 + GRM : effets fixes (3 moyennes) | ecart rel max 2.15e-15 | pass |
| us3 + GRM : BLUPs des 3 caracteres | ecart max 1.66e-09 sur 180 valeurs | pass |

**3. trois caracteres, diag genetique avec GRM et diag residuelle ; LRT us vs diag**

| Check | Measured | Verdict |
|---|---|---|
| diag3 + GRM : les 3 variances genetiques | remlax 1.3399 0.9738 1.0889 \| sommer 1.3399 0.9738 1.0889 | pass |
| diag3 + GRM : les 3 variances residuelles | remlax 0.7849 1.3983 0.6668 \| sommer 0.7849 1.3983 0.6668 | pass |
| LRT us vs diag : meme difference de -2logL | remlax 81.06687 \| sommer 81.06687 (6 ddl) | pass |

**4. additif + dominance : deux matrices de parente sur le meme facteur**

| Check | Measured | Verdict |
|---|---|---|
| A + D : les trois composantes | remlax 0.93906 0.76455 0.78469 \| sommer 0.93906 0.76455 0.78469 | pass |
| A + D : BLUPs additifs et de dominance | ecart max 6.78e-08 | pass |

**5. multi-environnement : une variance genetique par milieu avec GRM**

| Check | Measured | Verdict |
|---|---|---|
| MET : les 4 variances genetiques | remlax 0.5255 0.8048 1.1331 0.9474 \| sommer 0.5255 0.8048 1.1331 0.9474 | pass |
| MET : les 4 variances residuelles | remlax 0.7188 1.1889 1.5606 0.3662 \| sommer 0.7188 1.1889 1.5606 0.3662 | pass |
| MET : effets fixes | ecart rel max 2.12e-10 | pass |

## Against asreml, structures and inference

The structures a plant breeder actually writes: a one-dimensional AR1, a separable AR1xAR1 field, two-dimensional splines, and a covariance shared between two terms through str(). Then the inference layer: heritability and its delta-method standard error, Wald tests, and the equivalences between saturated parameterisations.

**1. AR1 unidimensionnel**

| Check | Measured | Verdict |
|---|---|---|
| AR1 : rho | 0.6276741 vs 0.6276741 | pass |
| AR1 : variance | 1.1690526 vs 1.1690526 | pass |
| AR1 : residuelle | - | pass |
| AR1 : logLik (convention asreml) | -111.075133372 vs -111.075133372 | pass |

**2. AR1 x AR1 separable (champ spatial)**

| Check | Measured | Verdict |
|---|---|---|
| AR1xAR1 : variance du champ | 1.2088279 vs 1.2088279 | pass |
| AR1xAR1 : rho ligne | 0.5940549 vs 0.5940549 | pass |
| AR1xAR1 : rho colonne | 0.3397100 vs 0.3397100 | pass |
| AR1xAR1 : residuelle | - | pass |
| AR1xAR1 : logLik | -154.818019348 vs -154.818019348 | pass |

**3. spline 2D (MEMES matrices de base des deux cotes)**

| Check | Measured | Verdict |
|---|---|---|
| spline 2D : variance spl_x | 6.072477e+00 vs 6.072478e+00 | pass |
| spline 2D : variance spl_y | 6.211014e+00 vs 6.210686e+00 | pass |
| spline 2D : variance spl_xy | 3.775135e-11 vs 2.473758e-10 | pass |
| spline 2D : residuelle | - | pass |
| spline 2D : logLik | -13.508534360 vs -13.508534389 | pass |
| spline 2D : composante nulle signalee degeneree | spl_xy | pass |

**4. str() : covariance PARTAGEE entre deux termes**

| Check | Measured | Verdict |
|---|---|---|
| str : var terme 1 | 1.1103369 vs 1.1103369 | pass |
| str : covariance | 0.5137383 vs 0.5137383 | pass |
| str : var terme 2 | 0.6966672 vs 0.6966671 | pass |
| str : residuelle | - | pass |
| str : logLik | -133.657051701 vs -133.657051701 | pass |

**5. vpredict et Wald**

| Check | Measured | Verdict |
|---|---|---|
| vpredict : estimation de h2 | 0.6685333 vs 0.6685333 | pass |
| vpredict : ERREUR-TYPE de h2 (delta method) | 0.0500972 vs 0.0500998 | pass |
| Wald : ddl du terme trt | 2 | pass |
| Wald : p-valeur du dernier terme (seq == cond) | 8.776e-09 vs 8.776e-09 | pass |

**6. parametrisations SATUREES : chol(t-1), ante(t-1), fa(t), rr(t)**

| Check | Measured | Verdict |
|---|---|---|
| chol(gid, rank=3)  == us | ecart 5.68e-13 | pass |
| ante(gid, rank=3)  == us | ecart 2.27e-13 | pass |
| fa(gid, rank=4)    == us | ecart 0.00e+00 | pass |
| rr(gid, rank=4)    == us | ecart 1.14e-13 | pass |

## Against asreml, the rest of the catalogue

The structures that were identified by matching asreml's output rather than from documentation - lvr, the anisotropic Matern, user-written structures, sectioned residuals - plus prediction and the Kenward-Roger denominator degrees of freedom against pbkrtest.

**1. lvr : tente tronquee max(0, 1 - d/phi)**

| Check | Measured | Verdict |
|---|---|---|
| lvr : logLik = profil R independant | remlax -1.9143976 \| R -1.9143976 \| portee 5.8896 vs 5.8896 | pass |
| lvr : asreml n'atteint pas un meilleur optimum | asreml -2.819575 <= remlax -1.914398 | pass |

**2. mtrn : Matern anisotrope, parametres FIXES des deux cotes**

| Check | Measured | Verdict |
|---|---|---|
| mtrn isotrope nu=0.8 : logLik = reference besselK | remlax 1.0419875 \| R 1.0419875 | pass |
| mtrn isotrope nu=0.8 : logLik = asreml | asreml 1.0419875 \| remlax 1.0419875 | pass |
| mtrn nu=1.5 delta=2 : logLik = reference besselK | remlax -16.5344994 \| R -16.5344994 | pass |
| mtrn nu=1.5 delta=2 : logLik = asreml | asreml -16.5344994 \| remlax -16.5344994 | pass |
| mtrn delta=2 alpha=.6 : logLik = reference besselK | remlax 2.5900227 \| R 2.5900227 | pass |
| mtrn nu=0.5 lambda=1 : logLik = reference besselK | remlax -6.9246740 \| R -6.9246740 | pass |
| mtrn nu=0.5 lambda=1 : logLik = asreml | asreml -6.9246740 \| remlax -6.9246740 | pass |

**3. mtrn : la portee est bien ESTIMEE quand on la libere**

| Check | Measured | Verdict |
|---|---|---|
| mtrn : phi estime = argmax du profil R | phi 0.39696 vs 0.39696 (vrai 2.5) \| logLik 5.3370863 vs 5.3370863 | pass |

**5. dsum : une residuelle par section**

| Check | Measured | Verdict |
|---|---|---|
| dsum : deux sections ameliorent la vraisemblance | 1 section -142.6365 -> 2 sections -130.4130 (+12.22) | pass |
| dsum : les deux variances residuelles sont separees | 0.2203 / 1.2541 (vraies 0.25 et 1.96) | pass |
| dsum : logLik = asreml | asreml -43.113853 \| remlax -43.113852 | pass |

**5b. dsum : des structures DIFFERENTES selon la section**

| Check | Measured | Verdict |
|---|---|---|
| dsum : trois sections, deux AR1 et une iid | 3 sections \| 6 parametres \| rho sur A+B | pass |

**6. predict : moyennes ajustees et leurs erreurs-types**

| Check | Measured | Verdict |
|---|---|---|
| predict : valeurs = asreml | ecart max 1.95e-12 | pass |
| predict : erreurs-types = asreml | ecart max 4.42e-10 | pass |
| predict : erreur-type des differences disponible | sed moyen 0.18603 | pass |

**7. Kenward-Roger vs pbkrtest**

| Check | Measured | Verdict |
|---|---|---|
| K-R : ddl du denominateur = pbkrtest | remlax 21.76897 \| pbkrtest 21.76896 | pass |
| K-R : statistique F = pbkrtest | remlax 2.790148 \| pbkrtest 2.790148 | pass |
| K-R : erreurs-types ajustees = pbkrtest | ecart max 1.30e-07 | pass |
| K-R : l'ajustement GONFLE les erreurs-types | rapport moyen 1.03753 | pass |

**8. predict avec la part aleatoire (BLUP + erreur de prediction)**

| Check | Measured | Verdict |
|---|---|---|
| predict : le BLUP entre dans la moyenne par bloc | ecart-type avec BLUP 0.45961, sans 0.00e+00 | pass |

## Against asreml, the remaining structures

Grouped residual series ar2, ar3, ma1, ma2, arma, sar, cor, corb and corg; the two-dimensional metric kernels iexp, igau, ieuc, aexp, agau, sph and cir on an irregular field; the non-saturated multi-trait structures fa(1), rr(1), corh, ante(1), chol(1) and diag on four traits with a GRM; the separable product id x ar1 x ar1 with shared correlations; and the multi-trait spatial residual us(trait):ar1(row):ar1(col).

**A1. ar2 : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ar2 : logLik (convention asreml) | -72.335196521 vs -72.335196521 | pass |
| ar2 : variance residuelle | 1.675628 vs 1.675628 | pass |
| ar2 : parametres de correlation (ensemble) | remlax 0.5325 0.19233 \| asreml 0.5325 0.19233 | pass |
| ar2 : remlax pas moins bon qu'asreml | -72.335197 vs -72.335197 (ecart -5.68e-14) | pass |
| ar2 : effets fixes (a l'ordre pres) | remlax 0.6225 0.4419 \| asreml 0.6225 0.4419 | pass |

**A2. ar3 : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ar3 : logLik (convention asreml) | -75.186557795 vs -75.186557795 | pass |
| ar3 : variance residuelle | 1.658159 vs 1.658159 | pass |
| ar3 : parametres de correlation (ensemble) | remlax 0.4335 0.10803 0.21949 \| asreml 0.4335 0.10803 0.21949 | pass |
| ar3 : remlax pas moins bon qu'asreml | -75.186558 vs -75.186558 (ecart +4.26e-14) | pass |
| ar3 : effets fixes (a l'ordre pres) | remlax 1.0652 0.4518 \| asreml 1.0652 0.4518 | pass |

**A3. ma1 : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ma1 : logLik (convention asreml) | -71.441165627 vs -71.441165627 | pass |
| ma1 : variance residuelle | 1.173080 vs 1.173080 | pass |
| ma1 : parametres de correlation (ensemble) | remlax -0.52845 \| asreml -0.52845 | pass |
| ma1 : remlax pas moins bon qu'asreml | -71.441166 vs -71.441166 (ecart -2.84e-14) | pass |
| ma1 : effets fixes (a l'ordre pres) | remlax 1.0329 0.6044 \| asreml 1.0329 0.6044 | pass |

**A4. ma2 : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ma2 : logLik (convention asreml) | -72.524782986 vs -72.558053561 | FAIL |
| ma2 : variance residuelle | 1.400230 vs 1.369831 | FAIL |
| ma2 : parametres de correlation (ensemble) | remlax -0.58011 -0.44931 \| asreml -0.57924 -0.44714 | FAIL |
| ma2 : remlax pas moins bon qu'asreml | -72.524783 vs -72.558054 (ecart +3.33e-02) | pass |
| ma2 : effets fixes (a l'ordre pres) | remlax 0.8983 0.4977 \| asreml 0.8994 0.4962 | FAIL |

**A5. arma : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| arma : logLik (convention asreml) | -73.124116717 vs -73.124116717 | pass |
| arma : variance residuelle | 1.376202 vs 1.376202 | pass |
| arma : parametres de correlation (ensemble) | remlax 0.28594 0.35587 \| asreml 0.35587 0.28594 | pass |
| arma : remlax pas moins bon qu'asreml | -73.124117 vs -73.124117 (ecart -5.68e-14) | pass |
| arma : effets fixes (a l'ordre pres) | remlax 0.8109 0.3658 \| asreml 0.8109 0.3658 | pass |

**A6. sar : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| sar : logLik (convention asreml) | -70.706817237 vs -70.706817237 | pass |
| sar : variance residuelle | 1.369386 vs 1.369386 | pass |
| sar : parametres de correlation (ensemble) | remlax 0.63555 \| asreml 0.63555 | pass |
| sar : remlax pas moins bon qu'asreml | -70.706817 vs -70.706817 (ecart -2.84e-14) | pass |
| sar : effets fixes (a l'ordre pres) | remlax 1.0847 0.3773 \| asreml 1.0847 0.3773 | pass |

**A7. cor : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| cor : logLik (convention asreml) | -96.140426816 vs -96.140426816 | pass |
| cor : variance residuelle | 1.515622 vs 1.515622 | pass |
| cor : parametres de correlation (ensemble) | remlax 0.21775 \| asreml 0.21775 | pass |
| cor : remlax pas moins bon qu'asreml | -96.140427 vs -96.140427 (ecart +4.26e-14) | pass |
| cor : effets fixes (a l'ordre pres) | remlax 0.9673 0.6452 \| asreml 0.9673 0.6452 | pass |

**A8. corb : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| corb : logLik (convention asreml) | -69.307901067 vs -69.307901067 | pass |
| corb : variance residuelle | 1.092528 vs 1.092528 | pass |
| corb : parametres de correlation (ensemble) | remlax 0.41316 0.14577 \| asreml 0.41316 0.14577 | pass |
| corb : remlax pas moins bon qu'asreml | -69.307901 vs -69.307901 (ecart -1.42e-14) | pass |
| corb : effets fixes (a l'ordre pres) | remlax 0.9803 0.4266 \| asreml 0.9803 0.4266 | pass |

**A9. corg : residuelle groupee, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| corg : logLik (convention asreml) | -97.912513106 vs -97.912513106 | pass |
| corg : variance residuelle | 0.905272 vs 0.905272 | pass |
| corg : parametres de correlation (ensemble) | remlax 0.38785 0.16896 0.22701 -0.07416 0.19222 0.43027 \| asreml 0.38785 0.16896 0.22701 -0.07416 0.19222 0.43027 | pass |
| corg : remlax pas moins bon qu'asreml | -97.912513 vs -97.912513 (ecart +1.85e-13) | pass |
| corg : effets fixes (a l'ordre pres) | remlax 0.9558 0.4911 \| asreml 0.9558 0.4911 | pass |

**B1. iexp : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| iexp : logLik (convention asreml) | -4.778255069 vs -4.778255069 | pass |
| iexp : variance | 1.476882 vs 1.476882 | pass |
| iexp : parametres du noyau (ensemble) | remlax 0.87186 \| asreml 0.87186 | pass |
| iexp : remlax pas moins bon qu'asreml | -4.778255 vs -4.778255 (ecart +8.88e-14) | pass |
| iexp : meme logLik aux parametres d'asreml | -4.778255069 vs -4.778255069 (remlax evalue aux parametres d'asreml) | pass |

**B2. igau : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| igau : logLik (convention asreml) | -3.293847858 vs -3.293847858 | pass |
| igau : variance | 1.287447 vs 1.287447 | pass |
| igau : parametres du noyau (ensemble) | remlax 0.80114 \| asreml 0.80114 | pass |
| igau : remlax pas moins bon qu'asreml | -3.293848 vs -3.293848 (ecart +4.25e-12) | pass |
| igau : meme logLik aux parametres d'asreml | -3.293847858 vs -3.293847858 (remlax evalue aux parametres d'asreml) | pass |

**B3. ieuc : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ieuc : logLik (convention asreml) | -4.016578104 vs -4.016578104 | pass |
| ieuc : variance | 0.769838 vs 0.769838 | pass |
| ieuc : parametres du noyau (ensemble) | remlax 0.70585 \| asreml 0.70585 | pass |
| ieuc : remlax pas moins bon qu'asreml | -4.016578 vs -4.016578 (ecart +2.13e-14) | pass |
| ieuc : meme logLik aux parametres d'asreml | -4.016578104 vs -4.016578104 (remlax evalue aux parametres d'asreml) | pass |

**B4. aexp : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| aexp : logLik (convention asreml) | -37.151545485 vs -37.151545489 | pass |
| aexp : variance | 1.424754 vs 1.424748 | pass |
| aexp : parametres du noyau (ensemble) | remlax 0.75991 0.65095 \| asreml 0.75991 0.65096 | pass |
| aexp : remlax pas moins bon qu'asreml | -37.151545 vs -37.151545 (ecart +4.57e-09) | pass |
| aexp : meme logLik aux parametres d'asreml | -37.151545486 vs -37.151545489 (remlax evalue aux parametres d'asreml) | pass |

**B5. agau : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| agau : logLik (convention asreml) | 17.316795826 vs 17.316795825 | pass |
| agau : variance | 0.605183 vs 0.605183 | pass |
| agau : parametres du noyau (ensemble) | remlax 0.94404 0.24205 \| asreml 0.94404 0.24205 | pass |
| agau : remlax pas moins bon qu'asreml | 17.316796 vs 17.316796 (ecart +1.44e-10) | pass |
| agau : meme logLik aux parametres d'asreml | 17.316795826 vs 17.316795825 (remlax evalue aux parametres d'asreml) | pass |

**B6. sph : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| sph : logLik (convention asreml) | 3.823231014 vs 3.753094938 | FAIL |
| sph : variance | 1.024529 vs 1.006357 | FAIL |
| sph : parametres du noyau (ensemble) | remlax 8.6514 \| asreml 8.75447 | FAIL |
| sph : remlax pas moins bon qu'asreml | 3.823231 vs 3.753095 (ecart +7.01e-02) | pass |
| sph : meme logLik aux parametres d'asreml | 3.783476190 vs 3.753094938 (remlax evalue aux parametres d'asreml) | FAIL |

**B7. cir : noyau metrique 2D, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| cir : logLik (convention asreml) | 13.875897073 vs 10.847517451 | FAIL |
| cir : variance | 0.846601 vs 0.623517 | FAIL |
| cir : parametres du noyau (ensemble) | remlax 7.40768 \| asreml 4.98345 | FAIL |
| cir : remlax pas moins bon qu'asreml | 13.875897 vs 10.847517 (ecart +3.03e+00) | pass |
| cir : meme logLik aux parametres d'asreml | 10.844976334 vs 10.847517451 (remlax evalue aux parametres d'asreml) | FAIL |

**C1. us(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| us : logLik (convention asreml) | -475.359451222 vs -475.359451222 | pass |
| us : remlax pas moins bon qu'asreml | -475.359451 vs -475.359451 (ecart +6.93e-12) | pass |
| us : Sigma genetique | ecart rel max 6.41e-08 \| diag remlax 1.209 2.6058 0.711 1.3245 \| asreml 1.209 2.6058 0.711 1.3245 | pass |
| us : les 4 variances residuelles | remlax 0.798 0.8956 0.5819 0.9622 \| asreml 0.798 0.8956 0.5819 0.9622 | pass |

**C2. diag(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| diag : logLik (convention asreml) | -493.193233887 vs -493.193233887 | pass |
| diag : remlax pas moins bon qu'asreml | -493.193234 vs -493.193234 (ecart -3.18e-12) | pass |
| diag : Sigma genetique | ecart rel max 4.00e-08 \| diag remlax 1.2052 2.6244 0.7152 1.3477 \| asreml 1.2052 2.6244 0.7152 1.3477 | pass |
| diag : les 4 variances residuelles | remlax 0.7998 0.8921 0.5804 0.9565 \| asreml 0.7998 0.8921 0.5804 0.9565 | pass |

**C3. fa1(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| fa1 : logLik (convention asreml) | -480.160993199 vs -480.160993199 | pass |
| fa1 : remlax pas moins bon qu'asreml | -480.160993 vs -480.160993 (ecart -5.88e-11) | pass |
| fa1 : les 4 variances residuelles | remlax 0.7967 0.8955 0.5807 0.9543 \| asreml 0.7967 0.8955 0.5807 0.9543 | pass |

**C4. rr1(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| rr1 : logLik (convention asreml) | -564.439985070 vs -564.440028141 | pass |
| rr1 : remlax pas moins bon qu'asreml | -564.439985 vs -564.440028 (ecart +4.31e-05) | pass |
| rr1 : les 4 variances residuelles | remlax 1.9695 1.0314 1.3332 1.5949 \| asreml 1.9693 1.0321 1.3332 1.5944 | pass |

**C5. corh(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| corh : logLik (convention asreml) | -492.756261237 vs -492.756261237 | pass |
| corh : remlax pas moins bon qu'asreml | -492.756261 vs -492.756261 (ecart -9.09e-13) | pass |
| corh : les 4 variances residuelles | remlax 0.7972 0.894 0.5795 0.9562 \| asreml 0.7972 0.894 0.5795 0.9562 | pass |

**C6. ante1(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| ante1 : logLik (convention asreml) | -491.875918525 vs -491.875918525 | pass |
| ante1 : remlax pas moins bon qu'asreml | -491.875919 vs -491.875919 (ecart +3.44e-11) | pass |
| ante1 : les 4 variances residuelles | remlax 0.7999 0.8922 0.5821 0.9573 \| asreml 0.7999 0.8922 0.5821 0.9573 | pass |

**C7. chol1(trait) x GRM, 4 caracteres, residuelle diag, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| chol1 : logLik (convention asreml) | -492.112977719 vs -492.112977719 | pass |
| chol1 : remlax pas moins bon qu'asreml | -492.112978 vs -492.112978 (ecart -4.87e-11) | pass |
| chol1 : les 4 variances residuelles | remlax 0.7999 0.8922 0.581 0.9575 \| asreml 0.7999 0.8922 0.581 0.9575 | pass |

**D. sep : id(bloc) x ar1(col) x ar1(row), correlations partagees, contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| sep : logLik (convention asreml) | -349.658684512 vs -349.658684512 | pass |
| sep : variance du champ | 1.742834 vs 1.742834 | pass |
| sep : rho colonnes et rho lignes | remlax 0.30873 0.59191 \| asreml 0.30873 0.59191 | pass |
| sep : residuelle | 0.772377 vs 0.772377 | pass |

**E. residuelle multi-caractere spatiale us(trait):ar1(row):ar1(col), contre asreml**

| Check | Measured | Verdict |
|---|---|---|
| us:ar1:ar1 : logLik (convention asreml) | -97.115201240 vs -97.115201240 | pass |
| us:ar1:ar1 : rho ligne et colonne | remlax 0.50089 0.36206 \| asreml 0.50089 0.36206 | pass |
| us:ar1:ar1 : Sigma residuelle 2x2 | remlax 1.1591 0.6273 0.6273 0.8517 \| asreml 1.1591 0.6273 0.6273 0.8517 | pass |
| us:ar1:ar1 : Sigma genetique 2x2 | remlax 0.5743 0.1596 0.1596 0.4802 \| asreml 0.5743 0.1596 0.1596 0.4802 | pass |

## Against asreml, BLUPs

BLUPs and their ordering on a one-factor model, level by level.

**1. ajustement des deux cotes, temps mesure**

| Check | Measured | Verdict |
|---|---|---|
| logLik dans la convention asreml | -260.296848172 vs -260.296848182 | pass |

**3. BLUP : les deux cotes predisent-ils les memes effets ?**

| Check | Measured | Verdict |
|---|---|---|
| BLUP : ecart maximal | max\|diff\| = 3.155e-06 | pass |
| BLUP : correlation | r = 1.000000000000 | pass |

**4. effets fixes et leur variance**

| Check | Measured | Verdict |
|---|---|---|
| effet fixe (intercept) | 5.110533510 vs 5.110533510 | pass |

## A full model of known truth

The complete direct and indirect genetic effects model: a weighted neighbourhood incidence, a covariance shared between the direct and indirect terms, a separable field and a nugget - eight variance parameters - fitted on simulated data whose true values are known, by remlax and by asreml.

| Check | Measured | Verdict |
|---|---|---|
| le signe de cov(DGE, IGE intra) est retrouve | -0.4221 vs -0.35 | pass |
| l'ajustement est a un optimum | decrement 8.7e-23 | pass |
| remlax n'est pas moins bon qu'asreml | ecart +2.05e-02 en faveur de remlax | pass |
| var_DGE : accord avec asreml | 1.09524 vs 1.09592 (SE 0.189) | pass |
| cov_DGE_IGE : accord avec asreml | -0.42211 vs -0.41939 (SE 0.180) | pass |
| var_IGEintra : accord avec asreml | 0.16268 vs 0.17627 (SE 0.354) | pass |
| var_IGEinter : accord avec asreml | 0.54369 vs 0.53748 (SE 0.341) | pass |
| var_champ : accord avec asreml | 0.61009 vs 0.61022 (SE 0.164) | pass |
| rho_ligne : accord avec asreml | 0.53375 vs 0.53215 (SE 0.131) | pass |
| rho_colonne : accord avec asreml | 0.11931 vs 0.11850 (SE 0.111) | pass |
| pepite : accord avec asreml | 0.39824 vs 0.39694 (SE 0.156) | pass |

## Sparse engine: parameterisation parity

Sigma, level correlations and their closed-form sparse precisions read identically by the RTMB sparse engine and the JAX dense engine, including K^-1 K = I checked by product.

**1. Sigma(theta), entre caracteres**

| Check | Measured | Verdict |
|---|---|---|
| Sigma : iid, t = 1, 1 parametres | ecart 0.000e+00 | pass |
| Sigma : iid, t = 2, 1 parametres | ecart 0.000e+00 | pass |
| Sigma : iid, t = 3, 1 parametres | ecart 0.000e+00 | pass |
| Sigma : iid, t = 5, 1 parametres | ecart 0.000e+00 | pass |
| Sigma : diag, t = 1, 1 parametres | ecart 0.000e+00 | pass |
| Sigma : diag, t = 2, 2 parametres | ecart 0.000e+00 | pass |
| Sigma : diag, t = 3, 3 parametres | ecart 4.441e-16 | pass |
| Sigma : diag, t = 5, 5 parametres | ecart 5.551e-17 | pass |
| Sigma : us, t = 1, 1 parametres | ecart 2.220e-16 | pass |
| Sigma : us, t = 2, 3 parametres | ecart 0.000e+00 | pass |
| Sigma : us, t = 3, 6 parametres | ecart 0.000e+00 | pass |
| Sigma : us, t = 5, 15 parametres | ecart 0.000e+00 | pass |

**2. K^-1(theta) : forme close R contre inverse numerique Python**

| Check | Measured | Verdict |
|---|---|---|
| ar1 : q = 5, phi = -0.0261 : K^-1 K = I | ecart 4.571e-18 | pass |
| ar1 : q = 5 : log\|K\| analytique contre numerique | ecart 0.000e+00 | pass |
| ar1 : q = 12, phi = 0.5222 : K^-1 K = I | ecart 2.220e-16 | pass |
| ar1 : q = 12 : log\|K\| analytique contre numerique | ecart 8.882e-16 | pass |
| ar1 : q = 40, phi = -0.4347 : K^-1 K = I | ecart 5.551e-17 | pass |
| ar1 : q = 40 : log\|K\| analytique contre numerique | ecart 5.329e-15 | pass |
| ar1ar1 : 3x4, phi = (0.512, -0.049) : K^-1 K = I | ecart 7.772e-16 | pass |
| ar1ar1 : 3x4 : log\|K\| analytique contre numerique | ecart 3.997e-15 | pass |
| ar1ar1 : 5x6, phi = (0.303, 0.748) : K^-1 K = I | ecart 1.554e-15 | pass |
| ar1ar1 : 5x6 : log\|K\| analytique contre numerique | ecart 2.487e-14 | pass |

**3. forme quadratique ecrite en vecteurs contre u' K^-1 u**

| Check | Measured | Verdict |
|---|---|---|
| ar1 : q = 6 | ecart 2.665e-15 | pass |
| ar1 : q = 25 | ecart 3.553e-15 | pass |
| ar1ar1 : 4x5 | ecart 0.000e+00 | pass |
| ar1ar1 : 7x3 | ecart 0.000e+00 | pass |

**4. perimetre**

| Check | Measured | Verdict |
|---|---|---|
| refuse noyau metrique iexp | : terme 's' : structure entre niveaux 'iexp' a un inverse plein, le creux n'y ga | pass |
| refuse parente dense LK | : terme 'g' : structure entre niveaux 'fixed' a un inverse plein, le creux n'y g | pass |
| refuse fa entre caracteres | : terme 'g' : structure entre caracteres 'fa' hors perimetre creux (retenues : i | pass |
| accepte us(2) + ar1 + residuelle diag | - | pass |

## Sparse engine: same likelihood as the dense engine

Each engine is evaluated at the other's optimum; comparing two optima alone would only show where each stopped.

| Check | Measured | Verdict |
|---|---|---|
| 0 : dense reevalue a son propre theta | ecart 0.000e+00 | pass |
| A : creux evalue au theta du dense | ecart 2.274e-13 | pass |
| B : dense evalue au theta du creux | ecart -2.274e-13 | pass |
| C : les deux optima a moins de 1e-4 | ecart -1.137e-13 | pass |
| 0 : dense reevalue a son propre theta | ecart 0.000e+00 | pass |
| A : creux evalue au theta du dense | ecart 2.046e-12 | pass |
| B : dense evalue au theta du creux | ecart 9.095e-13 | pass |
| C : les deux optima a moins de 1e-4 | ecart -2.559e-09 | pass |
| 0 : dense reevalue a son propre theta | ecart 0.000e+00 | pass |
| A : creux evalue au theta du dense | ecart 1.819e-12 | pass |
| B : dense evalue au theta du creux | ecart 5.684e-13 | pass |
| C : les deux optima a moins de 1e-4 | ecart -3.623e-10 | pass |
| 0 : dense reevalue a son propre theta | ecart 0.000e+00 | pass |
| A : creux evalue au theta du dense | ecart 1.023e-12 | pass |
| B : dense evalue au theta du creux | ecart 1.137e-13 | pass |
| C : les deux optima a moins de 1e-4 | ecart -6.118e-09 | pass |

## Sparse engine: precision path

A term declared by its precision matrix gives the same likelihood as the same term declared by its relationship matrix, arbitrated by a REML written out longhand in the test file.

**1. la fente accepte une precision creuse, et refuse les melanges**

| Check | Measured | Verdict |
|---|---|---|
| Kinv portee par le terme | ecart +0.000e+00 | pass |
| niveau resolu en 'prec' | ecart +0.000e+00 | pass |
| dans le perimetre creux | ecart +0.000e+00 | pass |
| refuse K et Kinv ensemble | ecart +0.000e+00 | pass |

**2. la MEME parente par deux voies donne la MEME vraisemblance**

| Check | Measured | Verdict |
|---|---|---|
| arbitre contre moteur dense, au theta du dense | ecart -1.137e-13 | pass |
| arbitre contre moteur creux, au theta du dense | ecart +0.000e+00 | pass |
| les deux moteurs, au theta du dense | ecart +1.137e-13 | pass |
| optima des deux moteurs | ecart +5.684e-13 | pass |

**3. log|K| fourni par l'utilisateur : meme resultat**

| Check | Measured | Verdict |
|---|---|---|
| log\|K\| fourni contre log\|K\| calcule | ecart +0.000e+00 | pass |

## What could not be measured here

This section exists so that nothing on the page above is mistaken for something it is not.

**CPU against GPU.** The 34-design parity sweep (`tests/python/parite_gpu.py`, evaluation at a common theta on both devices) was not replayed on this date: the whole A100 became free late in the day and the time went to the speed benchmark, which fitted the same ten models on 4 CPU cores and on the whole card and reached the same optimum on each (|logLik gap| at most 1e-6 at six printed decimals, `benchmarks/results/bench_2026-09-28_*.csv`). The parity recorded on 2026-09-01 (34/34, relative gap at most 6.4e-15) stands for the code of that date. `tests/slurm_valid_gpu.slurm` replays it.

**Structures with no external reference.** `mtrn` with all four parameters free, `own` (user-written kernels) and `sep` beyond three factors have no counterpart in asreml, lme4, sommer or nlme; they are checked against closed forms (besselK profile, the same kernel written as `exp`) and against the dense reference. The sparse RTMB engine is compared to the dense engine, never to a third program.

**What agreement means here.** Two REML implementations agree on the restricted log-likelihood at the optimum to 1e-6 absolute, on the variance components to 1e-4 relative (looser on purpose: on a flat surface two optimisers stop at slightly different theta for the same likelihood) and on fixed effects to 1e-6 relative. asreml stops on its own criterion and is compared at 1e-5 on the log-likelihood and 1e-3 on components; sommer is tightened to tolParConvLL = 1e-12 and compared at 1e-4. Iteration counts are never compared: average information, quasi-Newton and derivative-free optimisers do not count the same thing.

**The comparison against the project's own RTMB engine at full scale.** The figure of roughly 36 hours for RTMB against roughly 15 minutes on a full A100, at n = 16211, was measured in the originating project (IGE_analysis_2024-2025) before this repository existed. It is quoted in the paper as a prior measurement with its date, and it was not re-run here: it needs both the full card and that project's data.

## Re-running all of it

```sh
# inside a container or an environment carrying JAX, R, lme4, sommer, nlme
export PYTHONPATH=$PWD/src RX_PY=python3

python3 -m pytest tests/python -q                 # unit tests
python3 tests/python/test_remlax_core.py          # internal algebra
python3 tests/python/stress_remlax.py --n 50 --seed 0    # random sweep
Rscript  tests/R/test_remlax.R                    # lme4, sommer
Rscript  tests/R/test_remlax_nlme.R               # nlme, correlation structures
Rscript  tests/R/test_remlax_lme4.R               # lme4, slopes, nesting, PEV
Rscript  tests/R/test_remlax_sommer.R             # sommer, genomic multi-trait
Rscript  tests/R/test_remlax_asreml.R             # asreml, structures
Rscript  tests/R/test_remlax_asreml2.R            # asreml, catalogue, K&R
Rscript  tests/R/test_remlax_asreml3.R            # asreml, remaining structures
Rscript  tests/R/test_remlax_asreml_blup.R        # asreml, BLUPs
Rscript  tests/R/test_remlax_ige.R                # full model, known truth
Rscript  tests/R/test_remlax_tmb_parity.R         # sparse engine (RTMB)
Rscript  tests/R/test_remlax_tmb_vs_dense.R
Rscript  tests/R/test_remlax_tmb_prec.R

Rscript  tests/R/export_bundles.R --out=bundles   # serialise the catalogue
python3  tests/python/parite_gpu.py bundles --tol 1e-8   # CPU against GPU

python3 validation/parse_logs.py <logs>=<provenance> --out validation/results/checks_<date>.csv
python3 validation/make_validation_md.py --csv ... --versions ... --rev <sha> --date <date>
```

asreml requires a licence, and the `test_remlax_asreml*.R` suites check one out at run time. The other suites need no licence. The CPU/GPU parity check needs a machine with a GPU; everything else runs on a CPU.

