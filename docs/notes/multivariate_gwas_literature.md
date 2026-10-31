# Multivariate and neighbour GWAS: literature for a fixed-V scan

Compiled on 2026-10-31 for two planned extensions of the remlax genome-wide scan. One is a multi-trait scan with an unstructured or factor-analytic null model. The other is a multi-trait version of Neighbor GWAS with direct, indirect and similarity covariates. Sources are journal articles and preprints read in full text through PubMed Central or the publisher, plus package source code. The code was read from GEMMA `src/mvlmm.cpp` (GitHub master), limix and glimix-core (GitHub master) and rNeighborGWAS 1.2.5 (CRAN tarball). Statements marked [unverified] were not checked against the primary text.

## Multivariate LMM GWAS

### The shared model

Every single-variant method below fits the same Kronecker model. For n individuals and t traits, $\mathrm{vec}(Y) \sim N\big((I_t \otimes W)\,\mathrm{vec}(A) + (I_t \otimes x)\,\beta,\; V_g \otimes K + V_e \otimes I_n\big)$, with a t-vector of SNP effects $\beta$ ([GEMMA manual](https://git.genenetwork.org/pangemma/plain/doc/manual.tex); [Furlotte and Eskin 2015](https://doi.org/10.1534/genetics.114.171447); [Lippert et al. 2014](https://doi.org/10.1101/003905)). Methods differ in whether $V_g, V_e$ are refitted for each SNP or kept at the null estimates. Their statistic may be a Wald, likelihood ratio or score test. Some allow trait design matrices that encode constrained hypotheses, while others test only $\beta = 0$.

### GEMMA mvLMM

GEMMA computes exact tests of $H_0: \beta = 0$ against $\beta \neq 0$ with t degrees of freedom ([Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848)). For each SNP it refits $V_g$ and $V_e$ by EM followed by Newton-Raphson. In the source code, the LRT uses ML refits, the Wald test uses REML refits, and the score test evaluates the GLS quadratic form at the null estimates `V_g_null`, `V_e_null` with $\chi^2_t$ ([mvlmm.cpp](https://github.com/genetics-statistics/GEMMA/blob/master/src/mvlmm.cpp), function `MphCalcP` and the `a_mode` branches). An optional small-sample correction (`PCRT`) rescales the Wald and LRT statistics from Hessian terms; its derivation is not documented in the manual [unverified].

Speed comes from a double rotation. The eigendecomposition $K = U \Lambda U^\top$ rotates individuals. A second eigendecomposition of $V_e^{-1/2} V_g V_e^{-1/2} = Q D Q^\top$ rotates traits. After both rotations every element of $\tilde Y = Q^\top V_e^{-1/2} Y^\top U$ is independent with variance $d_l \lambda_k + 1$ (visible in `MphCalcP`; same transform in [Furlotte and Eskin 2015](https://doi.org/10.1534/genetics.114.171447)). After the single $O(n^3)$ eigendecomposition, no per-SNP step repeats a cubic operation in n ([Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848); exact per-SNP order garbled in the PMC text [unverified]). According to its authors, the algorithm handles one variance component besides the residual, requires complete phenotypes, and becomes statistically fragile around ten traits because $V_g, V_e$ carry $t(t+1)$ parameters ([Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848)). GEMMA's manual recommends fewer than ten phenotypes and imputing partially missing ones with `-predict` before testing ([GEMMA manual](https://git.genenetwork.org/pangemma/plain/doc/manual.tex)).

### MTMM

MTMM stacks traits in long format and tests with generalised least squares F-tests ([Korte et al. 2012](https://doi.org/10.1038/ng.2376)). The covariance comes from ASReml once under the null. Only a scalar in front of it is re-estimated for each marker, in the spirit of EMMAX and P3D. With two traits this fixes five of six variance parameters. The tests are defined on the model $y = \sum_i t_i \alpha_i + x\beta + (x \circ t_2)\gamma + \varepsilon$. A full test compares the model with $\beta$ and $\gamma$ to one with neither, with two degrees of freedom for two traits. An interaction test drops $\gamma$ only and detects trait-specific or opposite effects. A common-effect test compares the model with $x$ to one without, with one degree of freedom. Korte et al. advise normalising traits before the interaction and common tests because both depend on trait scaling. Their Henderson covariance model generalises to traits measured on different sets of individuals, so incomplete phenotypes fit the long format.

### Matrix-variate mvLMM

Furlotte and Eskin reformulate the same model as a matrix-variate normal and apply the double rotation above, with a matrix L that satisfies $L \Psi L^\top = I$ and $L \Phi L^\top = D$ ([Furlotte and Eskin 2015](https://doi.org/10.1534/genetics.114.171447)). The association test keeps $\Psi$ and $\Phi$ at their null estimates. It is a Wald-type F statistic with p numerator degrees of freedom, cited from McCulloch and Neuhaus; the denominator formula is garbled in the PMC text [unverified]. The model assumes a complete matrix. In the cohort analysis, missing phenotypes were replaced by the phenotypic mean.

### LIMIX and mtSet

The LIMIX preprint introduced trait design matrices for single-variant tests ([Lippert et al. 2014](https://doi.org/10.1101/003905)). An any-effect test uses $A = I_t$ and has t degrees of freedom. A common-effect test uses $A = 1_t^\top$ and has one. For trait p, a specific-effect test adds one degree of freedom over the common design. A GxE test compares $I_t \otimes 1_2^\top$ with $I_t \otimes I_2$. Trait covariances are estimated on the null model only, and the scaling factors are refitted for each test. LIMIX's fast path assumes a fully observed phenotype matrix.

The current limix code keeps this logic ([limix `qtl/_scan.py`](https://github.com/limix/limix/blob/master/limix/qtl/_scan.py)). It fits $\mathrm{vec}(Y) \sim N((A \otimes M)\,\mathrm{vec}(\mathsf{A}),\; C_0 \otimes K + C_1 \otimes I)$ by ML (`restricted=False`) under $H_0$. Blocks $(A_0 \otimes G_i)$ and $(A_1 \otimes G_i)$ are then added, and LRTs are returned for $H_0$ vs $H_1$, $H_0$ vs $H_2$ and $H_1$ vs $H_2$. Per candidate, only the fixed effects and one scale $s$ are optimised, with $K$ fixed ([glimix-core `_kron2sum_scan.py`](https://github.com/limix/glimix-core/blob/master/glimix_core/lmm/_kron2sum_scan.py)). In `Kron2Sum` the default rank of $C_0$ is 1, so the multi-trait null uses a rank-one genetic covariance unless changed. Its constructor raises an error on non-finite values in Y, so missing traits are not supported. Degrees of freedom are counted as the number of fitted effect sizes, with no rank check (`_mt_result.py`). The design $A_0 = 1_t$, $A_1 = I_t$ would then count t degrees of freedom for a contrast of rank $t-1$ (inferred from code; untested).

mtSet tests a genomic region as a random effect with its own trait covariance, against a null with genetic and noise trait covariances ([Casale et al. 2015](https://doi.org/10.1038/nmeth.3439)). This set test bears on variance components and gives no per-SNP fixed effects [unverified beyond title and abstract].

### GridLMM and MegaLMM

GridLMM is univariate and handles several random effects ([Runcie and Crawford 2019](https://doi.org/10.1371/journal.pgen.1007978)). It fits on a grid of variance proportions and reuses one Cholesky factor per grid point across markers. Its outputs are Wald F-tests at REML estimates, approximate LRTs and Bayes factors. The authors report that the null-LMM approach underestimates significance for large-effect markers. They also report that a fine grid is barely distinguishable from the exact LMM. The grid grows quickly with the number of variance components, with a practical limit near six to eight.

MegaLMM is a Bayesian factor model for thousands of traits with several random effects and large proportions of missing data ([Runcie et al. 2021](https://doi.org/10.1186/s13059-021-02416-w)). It targets genomic prediction and marginalises over missing records. The paper states that double diagonalisation applies only to models with a single random effect and no missing data. That limitation also applies to a multi-trait neighbour model with two kinships.

### Other multivariate approaches

MultiPhen reverses the regression ([O'Reilly et al. 2012](https://doi.org/10.1371/journal.pone.0034861)). Genotype is the response of a proportional-odds model with all phenotypes as predictors, and the test is an LRT for model fit. It does not model relatedness. MultiABEL applies a GRAMMAR+ mixed-model transformation to each phenotype, then a MANOVA with Pillai's trace for each variant ([Shen et al. 2017](https://doi.org/10.1038/s41467-017-00453-3)). The paper shows that the MANOVA statistics are equivalent for a single-marker test. Stephens partitions traits into unassociated, directly associated and indirectly associated groups, with Bayes factors over partitions ([Stephens 2013](https://doi.org/10.1371/journal.pone.0065245)). That framework gives a principled version of the conditional "trait k given the others" question [unverified beyond abstract]. PHENIX imputes missing phenotypes with a multi-trait mixed model before any scan ([Dahl et al. 2016](https://doi.org/10.1038/ng.3513)) [unverified beyond title and its use as a comparator in MegaLMM].

### Calibration with fixed or refitted V

Evidence points the same way in all sources. Fixing V at its null estimate yields conservative p-values in small, strongly related samples and near-exact ones in large samples with small effects. Zhou and Stephens state that the approximate LRT of MTMM is guaranteed to underestimate the exact LRT ([Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848)). In mouse data with 656 individuals and strong relatedness, MTMM p-values were larger than expected under the null, by almost an order of magnitude at the top. For 5,255 weakly related humans the two methods agreed. EM alone left the LRT unconverged and produced the same deflation. Onogi reports multivariate F-tests at null variance components on simulated plant data ([Onogi 2019](https://doi.org/10.3389/fgene.2019.00030)). These tests did not inflate p-values, and p-values became larger as the number of traits grew. Their denominator degrees of freedom ignore variance estimation, so the author suggests permutation, parametric bootstrap or Kenward-Roger. Univariate precedents are EMMAX and P3D ([Kang et al. 2010](https://doi.org/10.1038/ng.548); [Zhang et al. 2010](https://doi.org/10.1038/ng.546)).

### Incomplete phenotypes

| Method | Missing traits | Source |
|---|---|---|
| GEMMA mvLMM | complete matrix required; drop or impute with `-predict` | [Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848), manual |
| MTMM | long format; traits on different individuals allowed | [Korte et al. 2012](https://doi.org/10.1038/ng.2376) |
| Matrix-variate mvLMM | complete matrix; mean imputation in the application | [Furlotte and Eskin 2015](https://doi.org/10.1534/genetics.114.171447) |
| LIMIX 2014, limix 3 | fast path needs full Y; limix 3 raises an error on NaN | [Lippert et al. 2014](https://doi.org/10.1101/003905), glimix-core source |
| Multivariate F (remlf90) | missing records dropped from the stacked model | [Onogi 2019](https://doi.org/10.3389/fgene.2019.00030) |
| MegaLMM | handled by marginalisation; no GWAS | [Runcie et al. 2021](https://doi.org/10.1186/s13059-021-02416-w) |
| PHENIX | imputation before testing | [Dahl et al. 2016](https://doi.org/10.1038/ng.3513) |
| MultiPhen, MultiABEL | not stated [unverified] | |

## Neighbor GWAS

### Authors and scope

The Heredity paper is by Yasuhiro Sato, Eiji Yamamoto, Kentaro K. Shimizu and Atsushi J. Nagano ([Sato et al. 2021](https://doi.org/10.1038/s41437-020-00401-w)). The brief named "Sato, Yamamichi, Broman, Kudoh", which does not match CrossRef or PubMed. Kudoh co-authored the earlier field study of frequency-dependent herbivory in *Arabidopsis halleri* ([Sato and Kudoh 2017](https://doi.org/10.1086/692603)). The method is univariate. It was applied to herbivory on 199 *Arabidopsis thaliana* accessions.

### Model and neighbour covariate

The model comes from the two-dimensional Ising model ([Sato et al. 2021](https://doi.org/10.1038/s41437-020-00401-w)). Each locus is coded $x_i \in \{-1, +1\}$. Phenotypes follow $y_i = \beta_0 + \beta_1 x_i + \beta_2 \frac{1}{L_s} \sum_{j \in N_s(i)} x_i x_j + u_i + e_i$, with $L_s$ the number of neighbours within scale s. Each product $x_i x_j$ equals +1 for identical alleles and −1 for different ones. The neighbour term therefore measures allelic similarity between the focal plant and its neighbours. It corresponds to the Sato similarity covariate of remlax. No separate neighbour-dose term is fitted by the package.

The code fixes the details ([rNeighborGWAS 1.2.5](https://cran.r-project.org/package=rNeighborGWAS), `nei_coval.R`). Neighbours are all individuals of the same group whose Euclidean distance from the focal plant is at most `scale`. Its mask uses squared distance different from zero, so any plant at the same coordinates as the focal one is also excluded. The covariate is $x_i \cdot \overline{x}_{N_s(i)}$, with missing genotypes skipped in the mean and the final NA set to 0. With a finite `alpha`, each neighbour genotype is weighted by an exponential or Gaussian kernel. That weighted sum is divided by the neighbour count, so the weights are not normalised. Heterozygotes are coded 0 in the 2026 preprint, so the product is 0 when either plant is heterozygous ([Sato and Hamazaki 2026](https://doi.org/10.64898/2026.03.31.715746)). Sato et al. warn that the self and neighbour covariates become collinear when the minor allele frequency is low or the scale is broad. As $s \to \infty$ the covariate tends to $x_i$ times the population mean genotype.

### Kinship matrices

Background effects enter as a pair of random effects ([Sato et al. 2021](https://doi.org/10.1038/s41437-020-00401-w); `nei_lmm.R`). With q markers and the $\pm 1$ matrix $X_1$, the self kinship is $K_1 = \big((q-1)/2 + X_1 X_1^\top/2\big)/(q-1)$. It approximates the proportion of shared alleles. The neighbour matrix is $K_2 = X_2 X_2^\top/(q-1)$, where $X_2$ holds the similarity covariates of all markers. $K_2$ is therefore a similarity kinship. It differs from the classic IGE neighbour kinship $Z_N K Z_N^\top$. Both matrices are projected with `Matrix::nearPD`. The null is $V = \sigma_1^2 K_1 + \sigma_2^2 K_2 + \sigma_e^2 I$, fitted by AI-REML in `gaston::lmm.aireml`, with no covariance between the two effects. An asymmetric option adds a third matrix built from $x_i \cdot (x_i \overline{x}_{N})$.

### Tests

All SNP tests keep the variance components fixed. The package fits the null once, forms $K = \hat\tau_1 K_1 + \hat\tau_2 K_2$, refits the ratio of K to the residual by `lmm.diago`, and uses one eigendecomposition. For each SNP it compares ML profile log-likelihoods (`lmm.diago.profile.likelihood` returns the full likelihood profiled over $\beta$, per the [gaston documentation](https://rdrr.io/cran/gaston/man/lmm.diago.likelihood.html)). It reports $\chi^2_1$ p-values. Self effects are tested against the intercept model under $K_1$ alone. For the neighbour effect, the model with self and neighbour is compared with the model with self only, under the two-kinship V. This is a conditional test of similarity given direct effect. The paper recommends testing self effects without the neighbour component, because the extra variance component weakens the correction for structure. For binary traits the package uses a per-SNP logistic mixed model with Wald $z^2$. At fixed V, the profile LRT for $\beta$ equals the GLS Wald statistic exactly, so Neighbor GWAS is a fixed-V Wald scan in practice.

### Proportion of variance explained

Sato et al. define PVE in single, partial and net forms ([Sato et al. 2021](https://doi.org/10.1038/s41437-020-00401-w)). The single PVE of self effects is $\sigma_1^2/(\sigma_1^2+\sigma_e^2)$ in the one-kinship model. Partial PVEs come from the two-kinship model, with $\sigma_k^2/(\sigma_1^2+\sigma_2^2+\sigma_e^2)$ for $k = 1, 2$. The net neighbour PVE is (partial self + partial neighbour) − single self. `calc_PVEnei` scans a sequence of scales and tests $\sigma_2^2$ by a naive $\chi^2_1$ LRT. `delta_PVE` picks the effective scale where PVE_nei stops increasing. Simulations showed that the first nearest neighbours gave the most power unless the true range was broad.

### Follow-ups

Neighbor QTL carries the covariate into interval mapping with conditional genotype probabilities and adds dominance deviations for neighbour effects ([Sato et al. 2021b](https://doi.org/10.1093/g3journal/jkab017)). Combined with Ising-based genomic prediction, Neighbor GWAS selected *A. thaliana* genotype pairs. Three predicted pairs reduced herbivore damage by 18–30% in mixtures ([Sato et al. 2024](https://doi.org/10.1038/s41467-024-52374-7)). A GWAS of neighbour similarity on growth of 98 genotypes found negative frequency-dependent selection at the top SNP ([Sato 2026](https://doi.org/10.1098/rstb.2025.0179)). Sato and Wuest review genetic mapping of plant–plant interactions ([Sato and Wuest 2024](https://doi.org/10.1093/pcp/pcae092)). Wuest and Niklaus mapped a biodiversity effect in *A. thaliana* mixtures to one chromosomal region ([Wuest and Niklaus 2018](https://doi.org/10.1038/s41559-018-0708-y)).

The most recent extension is a bioRxiv preprint, also an eLife reviewed preprint ([Sato and Hamazaki 2026](https://doi.org/10.64898/2026.03.31.715746); code at github.com/yassato/ngwas2). It adds a covariance $\rho_{12}\sigma_1\sigma_2$ between polygenic self and neighbour effects through asymmetric cross matrices $K_{12} \neq K_{21}$. The authors test $\rho_{12}$ by LRT, reuse REML estimates for the SNP scan, and apply the method to aspen, apple and grapevine. Each trait and year is analysed separately. The equations were images in the HTML, so the cross-kernel formulas remain [unverified].

## Indirect genetic effect GWAS in animals

All animal IGE GWAS found are univariate and keep variance components at their null values. They differ in how the direct and indirect SNP effects enter. Baud et al. sum the cage mates' dosages into a social genotype ([Baud et al. 2021](https://doi.org/10.1186/s13059-021-02415-x)). The null has correlated polygenic DGE and IGE, direct and indirect environmental effects, and a cage effect, after the model of [Baud et al. 2017](https://doi.org/10.1371/journal.pgen.1006498). The igeGWAS compares models with and without the local social genotype, both including the local direct genotype. It is a 1-df LLR test with the covariance estimated once per phenotype. The dgeGWAS conditions on the local social genotype in the same way. Conditioning was needed for calibration, because each mouse serves as focal and partner, and this correlation arises even among unrelated animals. FDR came from 100 genome-wide permutations of the tested genotype matrix. With a constant number of cage mates, sum and mean give identical p-values. Variable group sizes make them differ, and power depends on which aggregation is true.

Brinker et al. fit the direct SNP effect and the summed indirect SNP effect of cage mates together in ASReml ([Brinker et al. 2018](https://doi.org/10.1186/s12711-018-0409-7)). Variance components were fixed from the model without SNP, and F values were corrected by genomic control when λ exceeded 1.10. Duijvesteijn, Knol and Bijma fit the same two-term single-SNP regression for androstenone in pigs ([Duijvesteijn et al. 2014](https://doi.org/10.2527/jas.2014-7863)). Biscarini et al. reported an across-line SNP study of direct and associative effects on feather damage ([Biscarini et al. 2010](https://doi.org/10.1007/s10519-010-9370-0)) [unverified beyond title]. The modelling issues of the variance-component side are reviewed by [Bijma 2014](https://doi.org/10.1038/hdy.2013.15). None of these studies reports a joint 2-df test of DGE and IGE as the main scan. Each reports conditional 1-df tests or per-term F-tests.

## Multi-trait IGE GWAS

No published multi-trait neighbour or IGE GWAS was found. Searches in PubMed and on the web on 2026-10-31 used "multivariate neighbor GWAS", "multi-trait indirect genetic effects GWAS" and "social genetic effects GWAS". The closest work is univariate with a DGE–IGE covariance ([Sato and Hamazaki 2026](https://doi.org/10.64898/2026.03.31.715746)). Multi-trait IGE variance models exist without the GWAS step, for example in multispecies mixtures ([Salas et al. 2026](https://doi.org/10.64898/2026.05.27.728303), preprint).

## Implications for a fixed-V multivariate scan

### Statistics at fixed V

Let y stack the N observed records over traits, after dropping missing ones, and $X_0$ hold the null fixed effects. The null covariance $\hat V$ is restricted to observed rows and fixed, for example $\hat\Sigma_D \otimes K + \hat\Sigma_I \otimes K_N + \hat\Sigma_{DI} \otimes K Z_N^\top + \text{transpose} + \hat R \otimes I$. Define $P = \hat V^{-1} - \hat V^{-1} X_0 (X_0^\top \hat V^{-1} X_0)^{-1} X_0^\top \hat V^{-1}$. For one SNP, $X_s = S (I_t \otimes C_s)$, where $C_s$ holds the chosen covariate families (direct, indirect, similarity) and S selects observed rows. Then the score and information are $U = X_s^\top P y$ and $\mathcal I = X_s^\top P X_s$, with $\hat\beta = \mathcal I^{-1} U$ and $\mathrm{Cov}(\hat\beta) = \mathcal I^{-1}$.

For $H_0: L\beta = 0$ with rank r, $T = (L\hat\beta)^\top (L \mathcal I^{-1} L^\top)^{-1} (L\hat\beta) \sim \chi^2_r$. With V fixed, the Wald, score and ML likelihood ratio statistics coincide exactly. GEMMA's score test and Sato's profile LRT are both this quantity. For a conditional test of SNP block b given SNP block a, $U_{b\cdot a} = U_b - \mathcal I_{ba}\mathcal I_{aa}^{-1}U_a$ and $\mathcal I_{b\cdot a} = \mathcal I_{bb} - \mathcal I_{ba}\mathcal I_{aa}^{-1}\mathcal I_{ab}$. The statistic is $T = U_{b\cdot a}^\top \mathcal I_{b\cdot a}^{-1} U_{b\cdot a}$.

| Hypothesis | L or design | df |
|---|---|---|
| Any effect of one family on t traits | identity on that family's t coefficients | t |
| Joint direct + indirect + similarity, all traits | identity on all 3t | 3t |
| Conditional family given another (ind given dir, sim given dir) | block test with the other family in $X_s$ | t |
| Common effect across traits | $X_c = X_s 1_t$, $T = (1^\top U)^2/(1^\top \mathcal I 1)$ | 1 |
| Heterogeneity given a common effect | contrasts $\beta_k - \beta_1$, $k = 2..t$ | t − 1 |
| Single trait k inside the joint model | $L = e_k^\top$ | 1 |
| Univariate Sato test (t = 1) | sim given dir, two-kinship V | 1 |

Nested statistics are additive at fixed V: the any-effect statistic equals the common-effect statistic plus the heterogeneity statistic. Common and heterogeneity tests depend on trait scaling, as Korte et al. point out, so traits should be standardised or the common direction weighted. A factor-analytic null fits the same formulas, since only $\hat V$ changes.

### Optional scale refit and F form

MTMM, LIMIX and limix 3 refit one scalar s per SNP. With GLS residual sums $\mathrm{RSS}_j = r_j^\top \hat V^{-1} r_j$, the ML LRT becomes $N \log(\mathrm{RSS}_0/\mathrm{RSS}_1)$. Onogi uses $F = T/(r\,\hat\sigma^2)$ with $\hat\sigma^2 = \mathrm{RSS}_1/(N - p)$ and $F(r, N - p)$, p being the number of fixed effects in the alternative ([Onogi 2019](https://doi.org/10.3389/fgene.2019.00030)). Both forms remain conservative-leaning approximations because V is not re-estimated.

### Computation

The double rotation of GEMMA and Furlotte–Eskin requires one kinship plus residual and a complete Y. A multi-trait neighbour null has at least two kinships with different trait covariances, plus missing records, so it falls outside both conditions ([Runcie et al. 2021](https://doi.org/10.1186/s13059-021-02416-w)). At fixed V, a practical route factorises $\hat V$ once and applies $P$ to all SNP columns in blocks by matrix products. Per-SNP work then reduces to $t f \times t f$ systems. This route is a design inference without a published source.

### Calibration and checks

Fixed-V tests are conservative in small related samples and near exact at large n ([Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848); [Onogi 2019](https://doi.org/10.3389/fgene.2019.00030); [Runcie and Crawford 2019](https://doi.org/10.1371/journal.pgen.1007978)). A refit of the full V at the top hits, as GEMMA does for every SNP, gives an exact LRT where it matters. Permutation of the tested genotype matrix, with the conditioning covariates left unpermuted, controls FDR in IGE scans ([Baud et al. 2021](https://doi.org/10.1186/s13059-021-02415-x)). Degrees of freedom must equal the rank of L, whereas the limix source counts columns. Sato's similarity covariate is defined under $\pm 1$ (or −1/0/+1) coding and changes meaning if doses are centred first.

## Key references

| Reference | Role here |
|---|---|
| [Zhou and Stephens 2014](https://doi.org/10.1038/nmeth.2848) | exact mvLMM LRT, double rotation, MTMM deflation |
| [Korte et al. 2012](https://doi.org/10.1038/ng.2376) | full, common and interaction tests; fixed V with scale refit |
| [Lippert et al. 2014](https://doi.org/10.1101/003905) | trait design matrices: any, common, specific, GxE |
| [Furlotte and Eskin 2015](https://doi.org/10.1534/genetics.114.171447) | matrix-variate model, fixed-V F test |
| [Onogi 2019](https://doi.org/10.3389/fgene.2019.00030) | multivariate F at null V with missing records |
| [Sato et al. 2021](https://doi.org/10.1038/s41437-020-00401-w) | Neighbor GWAS model, K1 and K2, PVE |
| [Sato and Hamazaki 2026](https://doi.org/10.64898/2026.03.31.715746) | DGE–IGE covariance in the neighbour model (preprint) |
| [Baud et al. 2021](https://doi.org/10.1186/s13059-021-02415-x) | conditional igeGWAS, permutation FDR |
| [Runcie et al. 2021](https://doi.org/10.1186/s13059-021-02416-w) | missing data and multiple random effects beyond rotation |
| [Runcie and Crawford 2019](https://doi.org/10.1371/journal.pgen.1007978) | null-LMM calibration, multiple random effects by grid |
