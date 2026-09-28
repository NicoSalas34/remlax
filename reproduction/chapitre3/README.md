# Reproduction du chapitre 3 avec remlax

Ces scripts refont chaque analyse du chapitre 3 (effets genetiques indirects dans une association ble dur x luzerne) avec le paquet remlax seul. Aucun code du depot d'analyse d'origine n'est charge. Seules ses donnees traitees sont lues.

Le guide complet, en anglais, avec la table analyse -> script -> sortie -> figure ou table du chapitre, les conventions numeriques et la section de verification, est dans `docs/guide/05-chapter3-reproduction.md`.

## Prerequis

- R >= 4.1 avec Matrix et jsonlite. Le paquet remlax installe (`R CMD INSTALL rpkg/`), ou bien la variable `REMLAX_R` pointant sur le dossier `R/` du depot : les scripts sourcent alors `remlax.R`, `remlax_design.R` et `remlax_ratios.R`.
- Un interpreteur Python avec jax, numpy et scipy, designe par `RX_PY` (par exemple `RX_PY=/chemin/envs/remlax-jax/bin/python`). Le CPU suffit pour une cellule univariee. Le multivarie a 198 parametres demande un GPU ou de la patience.
- Les donnees traitees du chapitre : `01_imported_data.rds`, `02_donnees_ble_clean.rds`, `02_donnees_luz_clean.rds`, `02_GK_matrix.rds`, `02_GK_luz_matrix.rds`. Elles ne sont pas publiques a ce jour.

## Variables d'environnement

| variable | role |
|---|---|
| `REMLAX_IGE_DATA` | chemin du dossier `data/processed` qui contient les cinq fichiers ci-dessus |
| `REMLAX_CH3_OUT` | dossier des sorties ; defaut `reproduction/chapitre3/output` |
| `REMLAX_R` | dossier `R/` du depot remlax si le paquet n'est pas installe |
| `RX_PY` | commande Python portant jax |
| `CUBE_I`, `CUBE_K` | tranche courante et nombre de tranches du cube (03) |

## La chaine

```
00_data.R            lecture des donnees, GRM blendees, univers des genotypes    -> data.rds
01_design.R          incidences de voisinage d'une geometrie (inspection)          -> design_*.rds
02_fit_univariate.R  une cellule : un caractere, une espece, une geometrie          -> cells/<tag>/
03_cube.R            taches, tranches entrelacees, reprise, agregation du cube      -> cube_<esp>_<trait>.csv
03_cube.sbatch       le cube en tableau de taches SLURM
04_cube_summary.R    tab:geom, comptes, figures S1-S3 et S7-S13                    -> cube_summary/
05_fit_multivariate.R  mvC7 ou mvD7 : sept caracteres, us 13x13 et 8x8             -> <name>/
05b_hessian.R        le Hessien au theta enregistre                                 -> <name>/hessian.csv
06_ratios.R          expositions, parts, heritabilites, tau2 (fig3, tab:partition)  -> <name>/ratios/
07_correlations.R    correlations genetiques, residuelles, de TBV (fig4, 5, S5, S6) -> <name>/correlations/
08_slope_2A.R        pente et correlation des BLUP par classe d'allele (fig3b)      -> <name>/slope/
_common.R            fonctions partagees (lecture, blocs, empilement, cellule)
geometries/          mvC7.csv (geometrie du multivarie du chapitre), tab_geom.csv (les sept retenues)
```

Exemple complet sur un caractere, en local :

```sh
export REMLAX_IGE_DATA=/chemin/IGE_analysis_2024-2025/data/processed
export REMLAX_R=/chemin/remlax/R
export RX_PY=/chemin/envs/remlax-jax/bin/python
cd reproduction/chapitre3
Rscript 00_data.R
Rscript 02_fit_univariate.R --species Ble --trait Hauteur.4 \
    --rank_within 5 --rank_between 7 --reach_within 0 --reach_between 0 \
    --dilution_within 0 --dilution_between 0
```

Le cube d'un caractere :

```sh
Rscript 03_cube.R --make_tasks --species Ble --trait Hauteur.4 --tasks tasks_Hauteur.4.csv
sbatch --array=1-750 --export=ALL,TASKS=tasks_Hauteur.4.csv,CUBE_K=750 03_cube.sbatch
Rscript 03_cube.R --aggregate --species Ble --trait Hauteur.4 --tasks tasks_Hauteur.4.csv
Rscript 04_cube_summary.R
```

Le multivarie et ses sorties :

```sh
Rscript 05_fit_multivariate.R --geometry geometries/mvC7.csv --name mvC7 --hessian FALSE --backend auto
Rscript 05b_hessian.R --name mvC7
Rscript 06_ratios.R --name mvC7
Rscript 07_correlations.R --name mvC7
Rscript 08_slope_2A.R --name mvC7 --classes <classes_alleles.csv>
```

## Ce qui est fige

Les options d'ajustement de toute la campagne sont dans `CH3_FIT_OPTIONS` (`_common.R`) : `maxiter 3000`, `polish 25`, `floor -12`, `ceil 12`, CPU pour les cellules. Le tag d'une cellule est celui du cube d'origine, `ord_<espece>_<trait>_li<>_le<>_di<>_de<>_ri<>_re<>`, avec `li`/`le` les portees, `di`/`de` les dilutions et `ri`/`re` les rangs conspecifique et heterospecifique.

## Verification

Le modele construit par ces scripts pour l'univarie retenu de la hauteur et pour le multivarie mvC7 est identique au modele exporte par le pipeline d'origine : y, X, chaque incidence, la residuelle, a l'ecart 0 ; les facteurs de Cholesky des parentes a 1e-15 (`validation/ch3_model_vs_ige.R`). Les autres ecarts mesures sont dans le guide, section Verification.
