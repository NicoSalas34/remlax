# Guide 05: reproducing the chapter-3 analyses with remlax alone

Chapter 3 of the thesis estimates direct and indirect genetic effects in a durum wheat x alfalfa intercrop: 3 840 plants on 12 trays, a 16 x 20 grid with a 5 cm step, alfalfa in the even columns, 181 wheat lines and 106 alfalfa families with genomic relationship matrices. The scripts under `reproduction/chapitre3/` redo every analysis of the chapter with the remlax package and its two companion files `R/remlax_design.R` and `R/remlax_ratios.R`. No code of the original analysis repository is loaded; only its processed data are read (`REMLAX_IGE_DATA`). The French README in that folder gives the prerequisites, the environment variables and the command lines.

## 1. Analysis -> script -> output -> chapter

| analysis of the chapter | script | main outputs (under `REMLAX_CH3_OUT`) | figure or table |
|---|---|---|---|
| Design, phenotypes, blended GRM, genotype universe | `00_data.R` | `data.rds`, `data_summary.csv` | sec. 2.5, 2.6 |
| Neighbourhood incidences of a geometry (by genotype and by plant) | `01_design.R` (`rx_neighbourhood`) | `design_<species>_ri<>_re<>_li<>_le<>_di<>_de<>.rds` and summary | sec. 2.5, 2.6 |
| One univariate cell: 12 variance components, exposures | `02_fit_univariate.R` (`rx_term`, `rx_model`, `rx_fit`, `rx_exposure`) | `cells/<tag>/{theta,fit_summary,sigmas,exposure,diagnostics}.csv`, `fit.rds` | the seven retained fits, sec. 3.1 |
| The six-parameter grid, 22 500 cells per trait, interleaved slices, resume, aggregation | `03_cube.R`, `03_cube.sbatch` | `tasks_<species>_<trait>.csv`, `cells/`, `cube_<species>_<trait>.csv` | sec. 2.6, 3.1 |
| Best cell, supported set, marginal ranges, counts at bounds and non-PD Hessians, AIC_eff | `04_cube_summary.R` (`rx_grid_summary`) | `cube_summary/tab_geom.csv`, `counts.csv`, `specification_steps.csv` | tab:geom, sec. 3.1 counts |
| Unfolded cube per trait | `04_cube_summary.R` | `cube_summary/figS7_cube_height.png` ... `figS13_cube_biomass_alfalfa.png` | fig S7 to S13 |
| Reach and dilution decoupling, specification steps | `04_cube_summary.R` | `cube_summary/figS1_reach.png`, `figS2_dilution.png`, `figS3_specification.png` | fig S1 to S3 (see the note below) |
| Multi-trait model (mvC7: 13 x 13 and 8 x 8 `us`, `dsum` residual) | `05_fit_multivariate.R`, `05b_hessian.R` | `<name>/model.rds`, `theta.csv`, `fit_summary.csv`, `sigma_*.csv`, `blups_*.csv`, `hessian.csv` | sec. 2.6, 3.2 |
| Exposures d, k, c, S; scaled components; shares; heritabilities; tau2 with standard errors | `06_ratios.R` (`rx_exposure`, `rx_ratios`) | `<name>/ratios/{exposure,ratios,tab_partition,heritabilities,tau2}.csv`, `fig3_heritabilities_tau2.png` | fig 3, tab:partition, sec. 3.3, 3.4 |
| Genetic, residual and TBV correlations with Fisher intervals | `07_correlations.R` (`rx_cor_z`) | `<name>/correlations/*.csv`, `fig4_genetic_correlations_wheat.png`, `fig5_genetic_correlations_alfalfa.png`, `figS5_tbv.png`, `figS6_residual_correlations.png` | fig 4, fig 5, fig S5, fig S6, tab:supp:cor, tab:supp:rescor |
| Slope and correlation of BLUPs by allele class | `08_slope_2A.R` (`rx_cor_z(n = )`) | `<name>/slope/slope_2A.csv`, `fig3b_slope_2A.png` | fig 3b |

Note on figures S1 to S3. In the chapter they come from a separate campaign (16, 25 and 4 fits per trait). The reproduction reads the same comparisons inside the cube, which contains those cells: `figS1_reach.png` is the AIC profile over (reach within, reach between) minimised over the four other coordinates, `figS2_dilution.png` the same over the dilutions, and `specification_steps.csv` / `figS3_specification.png` the best AIC under the successive constraints (common radius, reach and dilution; then reach decoupled; then dilution decoupled; then both; then radii decoupled).

## 2. Numerical conventions

These are the conventions of the code that produced the chapter, now carried by the functions, and checked by `validation/ch3_*.R`.

- Grid and distance. One tray at a time (`block`). Integer positions (row, column), physical distance `delta = 5 sqrt(d_row^2 + d_col^2)` cm (`spacing = c(5, 5)`).
- Window. A neighbour satisfies `|d_row| <= r` and `|d_col| <= r` and `delta > 0`: a Chebyshev square of side `2r + 1`, the plant itself excluded. The radius is that of the PAIR (receiving species, emitting species): `rank_within` for conspecifics, `rank_between` for heterospecifics. The nominal rank of the original tables (`ordre`) is the maximum of the two and is not a coordinate.
- Kernel. `w = delta^-lambda` with `delta` in centimetres, so `w = (5 delta_grid)^-lambda`. At `lambda = 0` every weight is 1. The factor `5^-lambda` is absorbed by the indirect variance and by `k`; it changes neither the likelihood nor any scaled quantity.
- Dilution. Each row is divided by `n_i^d`, `n_i` being the NUMBER of neighbours of the focal plant in the emitting class, not the sum of weights; `d = 0` leaves the matrix intact. Dilution comes before the (unused) L2 normalisation.
- Genotype incidence. `Z[i, g] = sum of w_ij over the neighbours j of genotype g`, then dilution; an entry can exceed 1. Hyphens are removed from genotype names before the call, so that they match the GRM.
- Plant incidence (IEE). `Z_E[i, j] = w_ij` diluted, built on the whole design, then restricted to the observed rows; the conspecific and heterospecific IEE are its two column sub-blocks, empty columns pruned. The IEE kernel follows the IGE kernel of the trait.
- Rows of the model. Sorted by (tray, column, row); complete observations of the trait AND of the covariates; the response is centred and reduced.
- Fixed effects. Intercept; for wheat seed weight and sowing date (when more than one date); nothing else for alfalfa.
- Terms, in the solver's order. `gen_ble` (`us`, columns `DGE_<t>`, `IGEintra_<t>`, `IGEon_luz_<t'>`), `gen_luz` (`us`, `DGE_<t'>`, `IGEintra_<t'>`, `IGEon_ble_<t>`), five spatial `iid` terms per block (`Bac_f`, `Bac_f:Ligne_f`, `Bac_f:Colonne_f`, `Bac_f:bordure`, `Bac_f:orientation_bordure`), one conspecific and one heterospecific IEE `iid` term per block; residual `dsum` by species, `us` between the traits of a species (same plant = same unit), `iid` when a species has one trait.
- Relationship. `K <- (1 - 0.02) K_raw + 0.02 I`, restricted to the genotypes present (181 wheat, 106 alfalfa), passed as covariance (`rx_term(K = )`). The effect exerted on the other species belongs to the vector of the species that EXERTS it and is weighted by its relationship matrix.
- Two conventions of `k`. `k_identity = mean_i sum_g Z[i, g]^2` (`K = I`, what the grid tables carry) and `k = mean_i (Z K Z')_ii` (weighted by the emitter's relationship, what tab:partition, fig 3 and tau2 use). `d = mean_i K[g(i), g(i)]` over the observed plants of the trait (1.915 to 1.920 for wheat, 0.948 for alfalfa); `c = mean_i (Z_d K Z_n')_ii`, zero across species; `S = mean_i sum_g Z[i, g]`; `n_eff = S^2 d / k`. All means are taken over the rows of the trait (`rows = "auto"`).
- Scaling. `V_D = d s2_D`, `V_IW = k_within s2_IW`, `V_IB = k_between s2_IB`, `V_IEE = k_identity(Z_E) s2_IEE`, `C = c cov_DI`; `V_P = sum + 2 C`; shares have the denominator without `2 C`; `h2 = V_D / V_P`; the chapter's `h2_intra,ext` is `h2_ext_within = (V_D + 2 C + V_IW) / V_P`, its `h2_inter` is `h2_indirect_between`, and `h2_total,ext = h2_ext_within + h2_indirect_between` (`h2_ext_total`). The direct-indirect correlation is computed on the raw components.
- Total genetic values. Own: `sqrt(d) (u_D + S_within u_IW)`; exerted on the other species: `sqrt(d_emitter) S_between u_IB`, `d_emitter` being the mean `d` of the emitter's traits. `tau2 = Var(TBV) / V_P` of the RECEIVING trait, placed in the map so that the delta method propagates the covariance between numerator and denominator.
- Theta and Sigma. `theta` is a log standard deviation; `us` is `L L'`, `L` lower triangular filled row by row, diagonal `exp(theta)`; theta order = terms, then residual sections.
- Standard errors. `H` is the Hessian of `-2 logL`; `cov(theta) = 2 H^-1` on the free subspace (neither at the bounds read in `par_floor` / `par_ceil`, nor fixed). Negative curvature: spectral projection and the `NOT_IDENTIFIED` flag on what depends on the removed directions; more than 5 % of a Jacobian on bounded parameters: `COND_BOUND`.
- Correlations. `se_z = se / max(1 - r^2, 1e-8)`, `ci = tanh(atanh(r) -+ q se_z)`, `NOT_ESTIMABLE` when the width exceeds 1.5 (the chapter's threshold since 2026-09-17). `|r| >= 0.2` in fig 4 and fig 5 is a display filter. Pearson correlations of BLUPs use `se_z = 1 / sqrt(n - 3)`.
- Grid. `AIC = 2 p - 2 logLik` with `p = 12`; `AIC_eff = 2 n_par_free - 2 logLik`; best cell = argmin; supported set = `AIC - min <= 2`; marginal ranges over the supported set; `pd_hessian = (n_neg_eig == 0)`; no cell is filtered.
- Fit options, frozen (`CH3_FIT_OPTIONS`). `maxiter 3000`, `polish 25`, `floor -12`, `ceil 12`, CPU. The original tables record `PAR_FLOOR = -8` while the solver bounded at -12; the reproduction bounds at -12, which is what the recorded `theta` show (values at -12 in the retained fits).

## 3. Verification

Every number below was measured on 2026-09-27/28 with R 4.5 (conda env `remlax-r`, OpenBLAS), jax 0.10.2 on CPU, against the outputs of the original pipeline. The scripts that measure them are `validation/ch3_neighbourhood_vs_ige.R`, `validation/ch3_model_vs_ige.R`, `validation/ch3_ratios_vs_ige.R`; their tables are under `validation/results/ch3_*.csv`. They skip when `REMLAX_IGE_REPO` is not set.

### 3.1 Incidences: `rx_neighbourhood` against the reference constructor

Ten geometries on the real design (the seven retained ones of tab:geom, plus ranks 3/8 with reaches 1/0.5 and dilutions 0.5/1, ranks 8/2 with reaches 2/2 and dilutions 0/0.5, and rank 1): the four genotype incidences and the plant incidence, aligned by plant identifier and genotype name.

```
REMLAX_IGE_REPO=<repo> Rscript validation/ch3_neighbourhood_vs_ige.R
```

Maximal absolute difference: `0` for every geometry, on both the level and the unit incidences (`validation/results/ch3_neighbourhood_vs_ige.csv`). The two computations share `dist()` and a dense product in double precision, and the order of the operations is the same.

### 3.2 Exposures: `rx_exposure` against the grid tables and the reference exposures

```
REMLAX_IGE_REPO=<repo> Rscript validation/ch3_ratios_vs_ige.R      # sections (b), (c), (d)
```

- `k_identity` of the height incidences against `k_IGE_intra` / `k_IGE_inter` of `cube_C_Hauteur.4.csv` at three cells (ranks 1/1, 5/5, 5/7): differences `3e-15` to `1e-13` (values 1.87653, 5.57797, 39.92496, 56.91485, 100.19319).
- On the exported mvC7 model, for the seven traits: `d` against `d_DGE`, weighted `k` within and between against `k_IGE_intra_mod` / `k_IGE_inter_mod`, `c`, `S`, and the IEE `k` against `vx_expositions.csv`, `k_identity` against `06_nuisance_var.csv`: 63 comparisons, maximal difference `5.4e-13`. `d` = 1.9148 to 1.9196 for wheat, 0.9481 for alfalfa, as in the legend of tab:partition.

### 3.3 Ratios: `rx_ratios` on mvC7 against the reference tables

Inputs: `modele.rds`, `06_theta_brut.csv`, `06_hessian.csv` of mvC7 (198 parameters, 12 760 observations), exposures from `rx_exposure`. Reference: `08_variance_components_se.csv` of `mvC7_geo` (338 quantities), `mvC7_tbvhomogene` and `tau2_par_caractere_mv.csv` of 2026-09-23 for TBV and tau2, `heritabilites_etendues_mv.csv`.

| quantities | estimate, max abs. diff. | SE, max rel. diff. | note |
|---|---|---|---|
| `se_theta` (198) against `06_theta_se.csv` | `1.1e-14` relative | | factor 2 confirmed |
| 189 variances, shares, heritabilities | `5.6e-16` | `2.4e-8` | the reference differentiates by Richardson extrapolation, `rx_ratios` by central differences with a relative step `1e-5` |
| 78 wheat and 28 alfalfa genetic correlations | `5.6e-16` | `1.7e-7` | Fisher intervals at `2.3e-5`: the reference multiplies by 1.96, `rx_cor_z` by `qnorm(0.975)`; `NOT_ESTIMABLE` flags identical (16 of 106) |
| 15 residual correlations | `5.6e-16` | `1.7e-10` | intervals at `9.7e-7` |
| 14 TBV variances | `4.7e-15` | `5.4e-10` | against `mvC7_tbvhomogene` |
| 14 tau2 | `4.9e-15` | `2.4e-8` | against `tau2_par_caractere_mv.csv` |
| `h2`, `se_h2`, `h2_intra` (= `h2_ext_within`), `part_part` (= `h2_indirect_between`) | `4.0e-11` | | against `heritabilites_etendues_mv.csv` (6 digits written) |

`rx_cor_z` on the three values of tab:supp:cor: `-0.937 (0.076)` gives `[-0.994; -0.456]`, `0.979 (0.022)` gives `[0.844; 0.997]`, `-0.153 (1.093)` gives a width of 1.95, flagged non-estimable; the table rounds to two decimals.

### 3.4 The model built by the reproduction scripts against the exported models

```
REMLAX_IGE_REPO=<repo> Rscript validation/ch3_model_vs_ige.R
```

For the retained univariate fit of `Hauteur.4` (ranks 5/7, reaches 0/0, dilutions 0/0) and for mvC7 (seven blocks at the geometry of `geometries/mvC7.csv`): `y`, `X`, every incidence of every term (9 and 51 terms), the residual sections (rows, structure, trait, unit) and the parameter count are identical, difference `0`. The Cholesky factors of the two relationship matrices differ by `1.0e-15` (wheat) and `2.2e-16` (alfalfa): the same blended `K` factorised by another LAPACK build; `chol()` of the reference's own `K` in this environment differs from the stored factor by the same amount. This is the only difference between the two model files, and it is at machine precision.

### 3.5 End-to-end on `Hauteur.4`

`00_data.R` then `02_fit_univariate.R --species Ble --trait Hauteur.4 --rank_within 5 --rank_between 7 --reach_within 0 --reach_between 0 --dilution_within 0 --dilution_between 0`, CPU, options frozen, against `output/results_uni_C/Hauteur.4/` (fit of 2026-09-22 on the same machine):

| quantity | reproduction | reference (2026-09-22) | difference |
|---|---|---|---|
| logLik | -2326.9409736131 | -2326.9409731085 | `5.0e-7` absolute, `2.2e-10` relative (the reproduction is the lower of the two) |
| L-BFGS-B | 327 iterations, `ABNORMAL` | 339 iterations, `ABNORMAL` | same stopping reason, different path |
| gradient (relative, projected) | `3.2e-9` | `2.4e-9` | |
| Newton decrement | `6.2e-15` | `2.7e-12` | |
| PD Hessian | yes | yes | |
| `k_identity` within / between, `d` | 39.92496 / 100.19319 / 1.91957 | 39.92496 / 100.19319 / 1.91957 | `0` |
| `theta` of the 9 identified components (`gen_ble` x3, `gen_luz`, `spat_b01_01`, `_03`, `_04`, `iee2_b01_01`, residual) | | | max `|dtheta| = 2.7e-6` (variances at `5e-6` relative) |
| `se_theta` of those 9 | | | max `3.4e-3` relative (`gen_luz`), `2e-3` or less for the eight others |
| `theta` of the 3 degenerate components (`spat_b01_02`, `spat_b01_05`, `iee_b01_01`) | -10.23, -10.11, -9.64 | -12.00, -11.74, -10.73 | variances `1e-9` against `1e-11`, both zero at the scale of the phenotype; `se_theta` of `1e3` on both sides |

The likelihood is reproduced to `2e-10` relative, better than the `1e-6` criterion, and every identified component to `5e-6`. The three components whose `theta` differ sit on the plateau where the likelihood is flat (their standard errors on the theta scale are `10^3` to `10^4`, and one of them is at the floor in the reference): the two runs stop, both with an `ABNORMAL` line search, at two points of the same plateau. Two causes, both measured: the Cholesky factor of the blended GRM differs at `1e-15` between the reference and this environment (3.4), and the number of BLAS threads differs (`user 119 min for 8 min wall` here), so `V`, the objective and the L-BFGS-B path differ at machine precision from the first iteration; the reference takes 339 iterations, the reproduction 327. With 11 of 12 `theta` unchanged to `3e-6` and one component at its floor in the reference, the comparison is at the limit of what an `ABNORMAL` stop allows. The recorded output is `reproduction/chapitre3/output/cells/ord_Ble_Hauteur.4_li0_le0_di0_de0_ri5_re7/` (`fit_summary.csv`, `theta.csv`, `sigmas.csv`, `exposure.csv`, `diagnostics.csv`).

`06_ratios.R` and `07_correlations.R` were then run on mvC7 without refitting (`05_fit_multivariate.R --build_only`, then `06_ratios.R --theta 06_theta_brut.csv --hessian 06_hessian.csv`): the 14 tau2 equal `tau2_par_caractere_mv.csv` at `1.0e-15` (SE at `2.4e-8` relative), the 21 heritabilities equal `heritabilites_etendues_mv.csv` at `7.7e-17`, the 77 shares of tab:partition equal `08_variance_components_se.csv` at `0` (SE at `2.4e-8` relative), 106 genetic correlations of which 16 non-estimable as in tab:supp:cor. `08_slope_2A.R` was run on the chapter's BLUP table and allele classes (`figures_session/data`): rare allele, neighbours' height, `r = 0.47 [0.20; 0.67]`, n = 44; frequent allele `r = 0.17 [0.00; 0.33]`, n = 137.

### 3.6 The grid summary on the seven reference cube tables

`04_cube_summary.R --reference <figures_session/data>` on the seven `cube_C_<trait>.csv` (157 500 cells): the best cells and the marginal ranges of `tab_geom.csv` are those of tab:geom for the seven traits; supported-set sizes 16 (height), 27 (tillers), 130 (grain count), 38 (leaf N), 6 (protein), 134 (wheat biomass), 2 (alfalfa biomass); 78 810 cells with at least one component at a bound; 26 338 cells without a PD Hessian; best PD cell of wheat biomass 0.1423 AIC unit above the best; the best cell moves under `AIC_eff` for five of the seven traits. All match the chapter (sec. 3.1).

### 3.7 The cube launcher on a mini-grid

Local run, `REMLAX_CH3_OUT=.../output_minicube`, with `--maxiter 20 --hessian FALSE` to keep each cell under a minute (the frozen options are used in production; here the launcher, not the estimates, is under test):

```
Rscript 03_cube.R --make_tasks --species Ble --trait Hauteur.4 --ranks "c(1,2)" --reaches 0 --dilutions 0 --tasks tasks_mini4.csv
CUBE_I=1 CUBE_K=2 Rscript 03_cube.R --run --tasks tasks_mini4.csv --maxiter 20 --hessian FALSE
Rscript 03_cube.R --run --tasks tasks_mini4.csv --slice 2 --of 2 --maxiter 20 --hessian FALSE
Rscript 03_cube.R --aggregate --species Ble --trait Hauteur.4 --tasks tasks_mini4.csv
Rscript 04_cube_summary.R
```

Four tasks (ranks 1 and 2 on both sides, reach 0, dilution 0). Slice 1 (tasks 1 and 3) had been fitted by an earlier, interrupted run of a wider grid: the second run reported `0 faite(s), 2 deja la` and fitted nothing, which is the resume. Slice 2 fitted tasks 2 and 4 (about 60 s each at `maxiter 20`). The aggregation found 4 cells and 0 missing and wrote `cube_Ble_Hauteur.4.csv`; `04_cube_summary.R` on it gives the best cell (ranks 1/2), 2 supported cells, no bounded component, and `delta_pd = NA` since no Hessian was computed. At `maxiter 20` the cell of ranks 1/2 reaches `logLik = -2341.94714270` against `-2341.94714261` in the chapter's cube table (converged run); the cell of ranks 1/1 is still 0.52 above its converged value, as expected from a truncated run.

### 3.8 What was not redone

- The 157 500 cells of the cube were not refitted (about 66 000 core-hours). The launcher is proven on a mini-grid; the summary is proven on the seven aggregated tables of the chapter.
- mvC7 was not refitted (198 parameters, several GPU hours). `05_fit_multivariate.R --build_only` builds the model, which equals the exported one (3.4); `06_ratios.R --theta --hessian` and `07_correlations.R` were run on the chapter's estimates placed on that model and reproduce the tables (3.3).
- The retained univariate fits of the six other traits were not refitted; only `Hauteur.4` was run end to end.
- `08_slope_2A.R` was run on the chapter's BLUP table and allele classes, not on BLUPs refitted here.
