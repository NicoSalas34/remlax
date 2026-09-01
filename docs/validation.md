# Validation

Every number on this page was produced by a job run on 2026-09-01 on the user's SLURM cluster, inside the apptainer image `ige_reml.sif`, which carries JAX 0.11.1, R 4.6.0 and licensed copies of asreml and pbkrtest. The measurements were extracted from the job logs by `validation/parse_logs.py` and this page is generated from the resulting table by `validation/make_validation_md.py`. No value here was typed by hand.

**154 checks, 154 passed, 0 failed.**

The check labels in the tables below are in French. They are the strings the test suites print, quoted verbatim, and the suites are written in French like the rest of the code. Translating them would give a more readable table and a trace that no longer matches the logs, so they are left alone; the section headings and the commentary are in English.

A note on naming. The solver was called `remlkit` when these measurements were taken and was renamed `remlax` afterwards. The recorded table still prints the former name in the columns that quote a log line verbatim. Rewriting a recorded measurement to match a name chosen later would falsify the trace, so it is left as measured.

## Summary

| Suite | Checks | Compared against | Passed |
|---|---|---|---|
| Internal algebra | 25 | the solver against itself and against closed-form results | 25/25 |
| Against lme4 and sommer | 31 | two open-source reference implementations | 31/31 |
| Against asreml, structures and inference | 28 | the field reference | 28/28 |
| Against asreml, the rest of the catalogue | 22 | the field reference | 22/22 |
| A full model of known truth | 11 | asreml, and the simulated parameters | 11/11 |
| Random sweep | 3 | properties over randomly drawn designs | 3/3 |
| CPU against GPU | 34 | the other backend, on the same input | 34/34 |

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
| us : covariance genetique 2x2 vs sommer | remlkit 1.3169 0.8723 0.8723 1.0382 \| sommer 1.3169 0.8723 0.8723 1.0381 | pass |
| us : covariance residuelle 2x2 vs sommer | remlkit 0.849 0.1019 0.1019 1.2783 \| sommer 0.849 0.1019 0.1019 1.2783 | pass |
| us : correlation genetique retrouvee | r 0.7460 vs 0.7460 | pass |

**5. structures emboitees : us doit dominer diag, qui domine iid**

| Check | Measured | Verdict |
|---|---|---|
| emboitement : -2logL decroissante iid >= diag >= us | iid=1324.4543 diag=1324.0042 us=1299.9284 | pass |

**6. incidence PONDEREE (le cas qu'une formule ne sait pas dire)**

| Check | Measured | Verdict |
|---|---|---|
| incidence ponderee : ajustement fini et defini | s2 1.5621 (vrai 1.21) \| s2e 0.6332 (vrai 0.64) \| decrement 4.0e-22 | pass |

**7. interface par formule : meme resultat que la voie explicite**

| Check | Measured | Verdict |
|---|---|---|
| formule ~ g : -2logL identique a lme4 | ecart 1.14e-13 | pass |
| formule ~ iid(g)+iid(b) : -2logL identique a lme4 | ecart 1.71e-13 | pass |
| formule us(gid) == voie explicite rk_term/rk_model | ecart 0.00e+00 | pass |

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
| spline 2D : variance spl_y | 6.211016e+00 vs 6.210686e+00 | pass |
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
| chol(gid, rank=3)  == us | ecart 2.27e-13 | pass |
| ante(gid, rank=3)  == us | ecart 0.00e+00 | pass |
| fa(gid, rank=4)    == us | ecart 3.41e-13 | pass |
| rr(gid, rank=4)    == us | ecart 2.27e-13 | pass |

## Against asreml, the rest of the catalogue

The structures that were identified by matching asreml's output rather than from documentation - lvr, the anisotropic Matern, user-written structures, sectioned residuals - plus prediction and the Kenward-Roger denominator degrees of freedom against pbkrtest.

**1. lvr : tente tronquee max(0, 1 - d/phi)**

| Check | Measured | Verdict |
|---|---|---|
| lvr : logLik = profil R independant | remlkit -1.9143976 \| R -1.9143976 \| portee 5.8896 vs 5.8896 | pass |
| lvr : asreml n'atteint pas un meilleur optimum | asreml -2.819575 <= remlkit -1.914398 | pass |

**2. mtrn : Matern anisotrope, parametres FIXES des deux cotes**

| Check | Measured | Verdict |
|---|---|---|
| mtrn isotrope nu=0.8 : logLik = reference besselK | remlkit 1.0419875 \| R 1.0419875 | pass |
| mtrn isotrope nu=0.8 : logLik = asreml | asreml 1.0419875 \| remlkit 1.0419875 | pass |
| mtrn nu=1.5 delta=2 : logLik = reference besselK | remlkit -16.5344994 \| R -16.5344994 | pass |
| mtrn nu=1.5 delta=2 : logLik = asreml | asreml -16.5344994 \| remlkit -16.5344994 | pass |
| mtrn delta=2 alpha=.6 : logLik = reference besselK | remlkit 2.5900227 \| R 2.5900227 | pass |
| mtrn nu=0.5 lambda=1 : logLik = reference besselK | remlkit -6.9246740 \| R -6.9246740 | pass |
| mtrn nu=0.5 lambda=1 : logLik = asreml | asreml -6.9246740 \| remlkit -6.9246740 | pass |

**3. mtrn : la portee est bien ESTIMEE quand on la libere**

| Check | Measured | Verdict |
|---|---|---|
| mtrn : phi estime = argmax du profil R | phi 0.39696 vs 0.39696 (vrai 2.5) \| logLik 5.3370863 vs 5.3370863 | pass |

**5. dsum : une residuelle par section**

| Check | Measured | Verdict |
|---|---|---|
| dsum : deux sections ameliorent la vraisemblance | 1 section -142.6365 -> 2 sections -130.4130 (+12.22) | pass |
| dsum : les deux variances residuelles sont separees | 0.2203 / 1.2541 (vraies 0.25 et 1.96) | pass |
| dsum : logLik = asreml | asreml -43.113853 \| remlkit -43.113852 | pass |

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
| K-R : ddl du denominateur = pbkrtest | remlkit 21.76897 \| pbkrtest 21.76896 | pass |
| K-R : statistique F = pbkrtest | remlkit 2.790148 \| pbkrtest 2.790148 | pass |
| K-R : erreurs-types ajustees = pbkrtest | ecart max 1.30e-07 | pass |
| K-R : l'ajustement GONFLE les erreurs-types | rapport moyen 1.03753 | pass |

**8. predict avec la part aleatoire (BLUP + erreur de prediction)**

| Check | Measured | Verdict |
|---|---|---|
| predict : le BLUP entre dans la moyenne par bloc | ecart-type avec BLUP 0.45961, sans 0.00e+00 | pass |

## A full model of known truth

The complete direct and indirect genetic effects model: a weighted neighbourhood incidence, a covariance shared between the direct and indirect terms, a separable field and a nugget - eight variance parameters - fitted on simulated data whose true values are known, by remlax and by asreml.

| Check | Measured | Verdict |
|---|---|---|
| le signe de cov(DGE, IGE intra) est retrouve | -0.4221 vs -0.35 | pass |
| l'ajustement est a un optimum | decrement 8.5e-23 | pass |
| remlkit n'est pas moins bon qu'asreml | ecart +2.05e-02 en faveur de remlkit | pass |
| var_DGE : accord avec asreml | 1.09524 vs 1.09592 (SE 0.189) | pass |
| cov_DGE_IGE : accord avec asreml | -0.42211 vs -0.41939 (SE 0.180) | pass |
| var_IGEintra : accord avec asreml | 0.16268 vs 0.17627 (SE 0.354) | pass |
| var_IGEinter : accord avec asreml | 0.54369 vs 0.53748 (SE 0.341) | pass |
| var_champ : accord avec asreml | 0.61009 vs 0.61022 (SE 0.164) | pass |
| rho_ligne : accord avec asreml | 0.53375 vs 0.53215 (SE 0.131) | pass |
| rho_colonne : accord avec asreml | 0.11931 vs 0.11850 (SE 0.111) | pass |
| pepite : accord avec asreml | 0.39824 vs 0.39694 (SE 0.156) | pass |

## Random sweep

Configurations are drawn at random (structure, dimensions, number of traits, seed) and fitted. Nothing is compared to a reference: the sweep asks only whether the solver ever produces something impossible - an exception, a non-finite log-likelihood, a covariance matrix that is not positive semi-definite, a run that does not converge.

**random configuration sweep, seed 0**

| Check | Measured | Verdict |
|---|---|---|
| 50 configurations: no exception, finite logLik, PSD Sigma, converged | clean 49/50; unstable optima 1 (best-of-restart gains 0.0085 in -2logL) | pass |

**random configuration sweep, seed 50**

| Check | Measured | Verdict |
|---|---|---|
| 50 configurations: no exception, finite logLik, PSD Sigma, converged | clean 50/50; unstable optima 0 | pass |

**random configuration sweep, seed 100**

| Check | Measured | Verdict |
|---|---|---|
| 50 configurations: no exception, finite logLik, PSD Sigma, converged | clean 48/50; unstable optima 2 (best-of-restart gains 0.0306, 0.0033 in -2logL) | pass |

## CPU against GPU

Every design of the structure catalogue is serialised once, then fitted twice in the same process: on the CPU and on a slice of an A100. The model must not depend on the machine that fits it.

**structure catalogue, 34 serialised designs**

| Check | Measured | Verdict |
|---|---|---|
| design 'aexp': same -2logL on CPU and on an A100 MIG slice | -2logL 601.62640591 vs 601.62640591 \| rel gap 5.67e-16 \| max\|dtheta\| 9.14e-10 \| 1.8 s CPU vs 7.0 s MIG | pass |
| design 'ante1_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1284.22057669 vs 1284.22057669 \| rel gap 1.77e-16 \| max\|dtheta\| 3.11e-15 \| 2.0 s CPU vs 4.6 s MIG | pass |
| design 'ar1': same -2logL on CPU and on an A100 MIG slice | -2logL 585.15663362 vs 585.15663362 \| rel gap 1.94e-16 \| max\|dtheta\| 9.99e-16 \| 1.0 s CPU vs 3.0 s MIG | pass |
| design 'ar1ar1_res': same -2logL on CPU and on an A100 MIG slice | -2logL 513.75395410 vs 513.75395410 \| rel gap 0.00e+00 \| max\|dtheta\| 3.12e-16 \| 1.0 s CPU vs 2.9 s MIG | pass |
| design 'ar2': same -2logL on CPU and on an A100 MIG slice | -2logL 583.38706024 vs 583.38706024 \| rel gap 0.00e+00 \| max\|dtheta\| 2.66e-15 \| 1.5 s CPU vs 2.4 s MIG | pass |
| design 'ar3': same -2logL on CPU and on an A100 MIG slice | -2logL 581.02165146 vs 581.02165146 \| rel gap 3.91e-16 \| max\|dtheta\| 5.16e-08 \| 1.6 s CPU vs 1.9 s MIG | pass |
| design 'arma': same -2logL on CPU and on an A100 MIG slice | -2logL 583.22954846 vs 583.22954846 \| rel gap 1.95e-16 \| max\|dtheta\| 4.12e-04 \| 0.8 s CPU vs 1.1 s MIG | pass |
| design 'chol2_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1281.59614340 vs 1281.59614340 \| rel gap 0.00e+00 \| max\|dtheta\| 1.05e-15 \| 1.0 s CPU vs 0.9 s MIG | pass |
| design 'cir': same -2logL on CPU and on an A100 MIG slice | -2logL 597.67667518 vs 597.67667518 \| rel gap 0.00e+00 \| max\|dtheta\| 0.00e+00 \| 0.8 s CPU vs 1.0 s MIG | pass |
| design 'cor': same -2logL on CPU and on an A100 MIG slice | -2logL 585.55424610 vs 585.55424610 \| rel gap 0.00e+00 \| max\|dtheta\| 4.77e-15 \| 0.4 s CPU vs 0.6 s MIG | pass |
| design 'corb2': same -2logL on CPU and on an A100 MIG slice | -2logL 583.23582112 vs 583.23582112 \| rel gap 0.00e+00 \| max\|dtheta\| 2.02e-14 \| 0.5 s CPU vs 0.6 s MIG | pass |
| design 'corg': same -2logL on CPU and on an A100 MIG slice | -2logL 601.62640596 vs 601.62640596 \| rel gap 9.45e-16 \| max\|dtheta\| 6.26e-11 \| 1.3 s CPU vs 3.1 s MIG | pass |
| design 'corh_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1281.60150032 vs 1281.60150032 \| rel gap 0.00e+00 \| max\|dtheta\| 1.33e-08 \| 1.1 s CPU vs 0.7 s MIG | pass |
| design 'dsum': same -2logL on CPU and on an A100 MIG slice | -2logL 515.18686079 vs 515.18686079 \| rel gap 2.21e-16 \| max\|dtheta\| 1.25e-16 \| 0.7 s CPU vs 0.8 s MIG | pass |
| design 'dsum_ar1': same -2logL on CPU and on an A100 MIG slice | -2logL 52.15394381 vs 52.15394381 \| rel gap 2.72e-16 \| max\|dtheta\| 8.88e-16 \| 1.0 s CPU vs 1.5 s MIG | pass |
| design 'expo': same -2logL on CPU and on an A100 MIG slice | -2logL 585.55424610 vs 585.55424610 \| rel gap 0.00e+00 \| max\|dtheta\| 1.33e-09 \| 0.5 s CPU vs 0.9 s MIG | pass |
| design 'fa2_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1281.59614340 vs 1281.59614340 \| rel gap 1.77e-16 \| max\|dtheta\| 1.15e-07 \| 1.7 s CPU vs 3.2 s MIG | pass |
| design 'gau': same -2logL on CPU and on an A100 MIG slice | -2logL 585.55424610 vs 585.55424610 \| rel gap 1.94e-16 \| max\|dtheta\| 1.33e-09 \| 0.4 s CPU vs 0.4 s MIG | pass |
| design 'ieuc': same -2logL on CPU and on an A100 MIG slice | -2logL 601.62640591 vs 601.62640591 \| rel gap 3.78e-16 \| max\|dtheta\| 5.02e-10 \| 0.5 s CPU vs 0.6 s MIG | pass |
| design 'iexp': same -2logL on CPU and on an A100 MIG slice | -2logL 601.62640591 vs 601.62640591 \| rel gap 3.78e-16 \| max\|dtheta\| 5.00e-10 \| 0.4 s CPU vs 0.6 s MIG | pass |
| design 'iid': same -2logL on CPU and on an A100 MIG slice | -2logL 498.81465569 vs 498.81465569 \| rel gap 2.28e-16 \| max\|dtheta\| 9.37e-10 \| 0.4 s CPU vs 0.8 s MIG | pass |
| design 'lvr': same -2logL on CPU and on an A100 MIG slice | -2logL 590.86761450 vs 590.86761450 \| rel gap 0.00e+00 \| max\|dtheta\| 4.44e-16 \| 0.6 s CPU vs 0.8 s MIG | pass |
| design 'ma1': same -2logL on CPU and on an A100 MIG slice | -2logL 583.30849573 vs 583.30849573 \| rel gap 0.00e+00 \| max\|dtheta\| 4.69e-13 \| 0.6 s CPU vs 0.8 s MIG | pass |
| design 'mtrn_aniso': same -2logL on CPU and on an A100 MIG slice | -2logL 630.20843955 vs 630.20843955 \| rel gap 1.80e-16 \| max\|dtheta\| 0.00e+00 \| 2.8 s CPU vs 255.0 s MIG | pass |
| design 'mtrn_iso': same -2logL on CPU and on an A100 MIG slice | -2logL 589.40743274 vs 589.40743274 \| rel gap 0.00e+00 \| max\|dtheta\| 3.22e-13 \| 3.4 s CPU vs 1.0 s MIG | pass |
| design 'own_2par': same -2logL on CPU and on an A100 MIG slice | -2logL 607.77082821 vs 607.77082821 \| rel gap 0.00e+00 \| max\|dtheta\| 0.00e+00 \| 0.7 s CPU vs 0.9 s MIG | pass |
| design 'own_exp': same -2logL on CPU and on an A100 MIG slice | -2logL 585.55424610 vs 585.55424610 \| rel gap 6.41e-15 \| max\|dtheta\| 2.82e-03 \| 0.7 s CPU vs 0.7 s MIG | pass |
| design 'rr2_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1343.60530125 vs 1343.60530125 \| rel gap 0.00e+00 \| max\|dtheta\| 6.66e-16 \| 1.0 s CPU vs 2.9 s MIG | pass |
| design 'sar': same -2logL on CPU and on an A100 MIG slice | -2logL 585.04086512 vs 585.04086512 \| rel gap 0.00e+00 \| max\|dtheta\| 1.74e-13 \| 0.6 s CPU vs 0.8 s MIG | pass |
| design 'sph': same -2logL on CPU and on an A100 MIG slice | -2logL 587.42826915 vs 587.42826915 \| rel gap 1.94e-16 \| max\|dtheta\| 2.21e-04 \| 0.6 s CPU vs 0.8 s MIG | pass |
| design 'str_2': same -2logL on CPU and on an A100 MIG slice | -2logL 512.47455420 vs 512.47455420 \| rel gap 0.00e+00 \| max\|dtheta\| 4.53e-09 \| 0.8 s CPU vs 3.4 s MIG | pass |
| design 'us_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1281.59614340 vs 1281.59614340 \| rel gap 1.77e-16 \| max\|dtheta\| 9.71e-09 \| 1.1 s CPU vs 0.7 s MIG | pass |
| design 'us_res_mv': same -2logL on CPU and on an A100 MIG slice | -2logL 1395.95209251 vs 1395.95209251 \| rel gap 0.00e+00 \| max\|dtheta\| 4.61e-16 \| 1.0 s CPU vs 2.3 s MIG | pass |
| design 'vm_kin': same -2logL on CPU and on an A100 MIG slice | -2logL 494.64877180 vs 494.64877180 \| rel gap 5.75e-16 \| max\|dtheta\| 7.49e-16 \| 0.5 s CPU vs 1.0 s MIG | pass |

## What could not be measured here

This section exists so that nothing on the page above is mistaken for something it is not.

**The full A100 card.** Every GPU number above and in [benchmarks.md](benchmarks.md) comes from one MIG slice, which exposes 7 of the card's 108 streaming multiprocessors. The card itself was occupied by other work. A slice is the right instrument for checking that CPU and GPU agree - arithmetic does not depend on how many multiprocessors run it - but any timing measured on it is a pessimistic bound on the hardware, not a property of the solver.

**Two structures on the slice.** The Matern kernels, isotropic and anisotropic, could not be fitted on the slice at 1200 levels: the Bessel quadrature asks for a single 6.5 GB allocation, above what a 10 GB slice can give. They fit without difficulty on the CPU, and the CPU/GPU parity check above did pass for both at 160 levels. This is a limit of the slice, and it is measured, not inferred.

**The comparison against the project's own RTMB engine.** The figure of roughly 36 hours for RTMB against roughly 15 minutes on a full A100, at n = 16211, was measured in the originating project (IGE_analysis_2024-2025) before this repository existed. It is quoted in the paper as a prior measurement with its date, and it was not re-run here: it needs both the full card and that project's data.

## Re-running all of it

```sh
# inside a container or an environment carrying JAX, R, lme4, sommer
export PYTHONPATH=$PWD/src RX_PY=python3

python3 tests/python/test_remlax_core.py          # internal algebra
python3 tests/python/stress_remlax.py --n 50 --seed 0    # random sweep
Rscript  tests/R/test_remlax.R                    # lme4, sommer
Rscript  tests/R/test_remlax_asreml.R             # asreml, structures
Rscript  tests/R/test_remlax_asreml2.R            # asreml, catalogue, K&R
Rscript  tests/R/test_remlax_ige.R                # full model, known truth

Rscript  tests/R/export_bundles.R --out=bundles   # serialise the catalogue
python3  tests/python/parite_gpu.py bundles --tol 1e-8   # CPU against GPU
```

asreml requires a licence, and the two `test_remlax_asreml*.R` suites check one out at run time. The other suites need no licence. The CPU/GPU parity check needs a machine with a GPU; everything else runs on a CPU.

