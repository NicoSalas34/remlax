# remlax — solveur REML générique, CPU et GPU

**Projet** : IGE Analysis 2024–2025 · **Écrit le** : 31 août 2026 · **Étendu et validé contre asreml** : 1er septembre 2026 · **Catalogue de structures + modèle IGE complet** : 1er septembre 2026

Un solveur de vraisemblance restreinte **indépendant du modèle IGE**, pilotable
depuis R avec une syntaxe proche d'asreml, calculé en JAX, et qui utilise le GPU
s'il y en a un sans que le modèle change.

---

## 1. Ce que c'est

```
y = X β + Σ_k Z_k u_k + e,   u_k ~ N(0, Σ_k ⊗ K_k),   e ~ N(0, R)
```

Un **terme** = un ensemble de `t` incidences (n × q) partageant les mêmes `q`
niveaux, une structure de covariance Σ sur ces `t` colonnes, et une parenté K
entre niveaux. Ce seul objet couvre :

| cas | t | K |
|---|---|---|
| facteur aléatoire simple | 1 | I |
| effet génétique avec GRM | 1 | GRM |
| multi-caractère `us` / `fa` | nb de caractères | I ou GRM |
| modèle IGE de ce projet | DGE + IGE intra + IGE inter | GRM par espèce |

Le dernier cas est celui qui impose la généralité : chaque colonne y a **sa
propre** incidence (voisinage pondéré), pas une indicatrice de caractère.

## 2. Interface

Par formule, dans l'esprit d'asreml :

```r
source("R/remlax.R")

fit <- rx_reml(fixed    = y ~ 1 + traitement,
               random   = ~ vm(genotype, K = Kmat) + iid(bloc),
               residual = "units",
               data     = df,
               backend  = "auto")
```

Grammaire des termes aléatoires — le premier argument est la colonne de
groupement, `K =` et `rank =` sont optionnels :

| écriture | Σ | paramètres |
|---|---|---|
| `iid(f)` ou `f` | σ² | 1 |
| `vm(f, K)` | σ², parenté K | 1 |
| `diag(f)` | diagonale entre caractères | t |
| `us(f)` | libre entre caractères | t(t+1)/2 |
| `fa(f, rank = k)` | ΛΛ' + diag(ψ) | nload + t |
| `mm(Z)` | incidence **fournie** | 1 |
| `ar1(f)` | σ² × AR1(ρ) entre niveaux | 2 |
| `ar1(r, c)` | σ² × AR1(ρ_r) ⊗ AR1(ρ_c) | 3 |
| `str(~ a + b, struct = "us")` | **une** covariance pour **plusieurs** termes | t(t+1)/2 |

### Catalogue de corrélations entre niveaux (annexe C du manuel ASReml-R 4.2)

**Stationnaires 1D** : `cor` · `ar1` · `ar2` · `ar3` · `sar` · `ma1` · `ma2` ·
`arma` · `corb(order=)` · `corg`.
**Métriques 1D** : `exp(coord=)` · `gau(coord=)` · `lvr`.
**Métriques 2D** : `iexp` (city-block) · `igau` · `ieuc` (euclidienne) ·
`sph` · `cir` (à portée) · `aexp` · `agau` (anisotropes) · `mtrn` (Matérn).
**Définie par l'utilisateur** : `own(f, expr=, n_par=)`.
**Séparable** : `ar1(r, c)`.

Les modèles à portée (`sph`, `cir`, `lvr`, `mtrn`) prennent un paramètre de
**distance**, donc en `exp(θ)` — positif, non borné — et non une corrélation en
`tanh`.

#### Deux formules qui ne sont pas dans le manuel

Les pages d'aide de `lvr` et `mtrn` donnent la signature, pas la formule. Les
deux ont été **identifiées contre asreml**, en comparant des courbes de logLik à
paramètre fixé.

**`lvr`** est la tente tronquée `C_ij = max(0, 1 − d_ij/φ)` — le modèle linéaire
à palier de la géostatistique, défini positif en 1D parce que c'est
l'autocorrélation d'un créneau. φ est une **portée**, contrainte U chez asreml.
Identification (q = 12, même jeu, φ imposé des deux côtés) :

| φ | 1 | 1,5 | 2 | 2,528 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|
| asreml | −4,518854 | −3,301014 | −3,831719 | −3,419241 | −5,217616 | −4,338156 | −2,652383 |
| `max(0,1−d/φ)` | −4,518854 | −3,301014 | −3,831719 | −3,419241 | −5,217616 | −4,338156 | −2,652383 |

Six familles plausibles ont été écartées avant celle-là (`I + φ(max−d)`,
`min(i,j)`, sphérique, `exp(−d/φ)`...), toutes à plus de 0,6 point.

La courbe n'est **pas** unimodale : le support change chaque fois que φ franchit
un entier. Sur ce jeu asreml s'arrête à φ = 2,53 (−3,419) alors que l'optimum
global est vers φ = 47 (−2,723) — un optimum local, pas une erreur de formule.
Le test de non-régression compare donc à une **grille fine**, pas à un
`optimize()`, qui s'y ferait piéger de la même façon.

**`mtrn`** est la Matérn anisotrope de Haskard et al. (2007) :

```
u =  dx·cos α + dy·sin α          h = ( |√δ·u|^λ + |v/√δ|^λ )^(1/λ)
v = −dx·sin α + dy·cos α          ρ = 2^(1−ν)/Γ(ν) · (h/φ)^ν · K_ν(h/φ)
```

δ agit **en racine et en sens inverse sur les deux axes** : la transformation
préserve les aires, donc δ ne se confond pas avec la portée φ. Trois autres
conventions (δ sur un seul axe, δ non racine) décalent la logLik de 0,5 à 4
points. Vérifié contre asreml à 3×10⁻¹⁰ sur (δ=2, α=0) et (δ=2, α=0,6), et à
1,5×10⁻¹⁰ sur ν ∈ {0,5 ; 0,8 ; 1,5} en isotrope. λ = 1 (city-block) est vérifié
exactement à ν = 0,5 ; **λ = 1 avec ν ≠ 0,5 n'est pas défini positif** en deux
dimensions (valeur propre −0,049 sur une grille 6×5), asreml rend pourtant un
nombre — le nôtre refuse.

#### ν quelconque : K_ν en JAX

`jax.scipy.special` n'expose pas la fonction de Bessel modifiée K_ν, et un
rappel vers scipy n'aurait pas de dérivée — or ν et φ sont justement des
paramètres à estimer. `scripts/gpu/remlax/bessel.py` calcule donc directement

```
M(z,ν) = 2^(1−ν)/Γ(ν) · z^ν · K_ν(z) = ∫₀^∞ exp(ν log z + ν u − z cosh u) du  (symétrisé)
```

par **quadrature trapézoïdale** sur une grille fixe. Le choix n'est pas
esthétique : la première version, par différence de séries hypergéométriques
(K_ν = π/2·(I₋ν − I_ν)/sin νπ), donne une erreur absolue de **3,5×10⁶ à ν = 3**
— la formule explose à ν entier et perd déjà 8 chiffres à ν = 3,7. L'intégrale
est analytique en ν, n'a aucune singularité d'ordre, et son erreur de trapèze
décroît en exp(−π²/h) parce que les coefficients de Poisson valent K_{iξ}(z).

Vérifié contre `scipy.special.kv` sur ν ∈ [0,05 ; 8] (entiers compris) et
z ∈ [0 ; 80] : **erreur absolue maximale 1,5×10⁻¹⁴**, dérivées en z et en ν
correctes à 8×10⁻¹⁰.

#### `own` : structure écrite par l'utilisateur

`own(obj, fun)` d'asreml appelle une fonction R à chaque évaluation. Impossible
ici : R et JAX ne partagent pas d'espace mémoire, et surtout il faudrait la
**dérivée**, qu'une fonction R opaque n'a pas. On prend donc une **expression**,
qui traverse la frontière comme une chaîne et se différentie automatiquement :

```r
random = ~ own(col, expr = "exp(-lag*exp(p1))", n_par = 1)
```

Variables : `d` (distance euclidienne), `dx`, `dy`, `lag` (|i−j|), `I`, `J`,
`p1..pk` (paramètres **bruts**, non contraints — à l'expression d'appliquer
`exp`/`tanh` si elle veut de la positivité). `^` est traduit en `**` : sans
cela, une expression écrite en R produisait
`xor does not accept dtype float64`. Le résultat est normalisé en corrélation
sauf `normalise = FALSE`, l'échelle étant déjà portée par Σ.

**`ilv` n'est pas fourni.** asreml en a un ; sa formule n'a pas pu être
retrouvée — aucune des neuf familles essayées ne reproduit sa courbe de logLik,
et la tente euclidienne, candidate naturelle, n'est pas définie positive en 2D
(elle rend des NaN). Mieux vaut pas de structure qu'une fausse sous le nom
d'asreml.

### Structures de Σ

`iid`(=idv) · `diag`(=idh) · `us`(=corgh) · `fa(k)` · `rr(k)` · `chol(k)` ·
`ante(k)` · `corh`.

Décomptes conformes au manuel — `chol(k)` et `ante(k)` valent (k+1)(ω−k/2) —
à une exception documentée : `rr(k)` compte `n_loadings(ω,k)` et non kω, parce
que Γ est contraint trapézoïdal. Sans cette contrainte Γ n'est défini qu'à une
rotation près, la vraisemblance a une crête plate de dimension k(k−1)/2 et le
Hessien est singulier d'autant. Le modèle ajusté est le même.

**Contrôle de cohérence** : `chol(ω−1)`, `ante(ω−1)`, `fa(ω)` et `rr(ω)` sont
des paramétrisations **saturées** du même Σ. Vérifié : toutes atteignent la
logLik de `us` à ≤ 3,4×10⁻¹³, avec le même Σ à 7×10⁻¹⁴.

La **variance** vit dans Σ, la structure entre niveaux ne porte qu'une
**corrélation** : c'est la décomposition Σ_h = D C D du manuel. Un `ar1v`
d'asreml s'écrit ici `ar1(f)` (Σ = σ²), un `ar1h` s'écrit
`ar1(f, struct = "diag")`.

Deux choix qui évitent des échecs muets :

- **ar2 / ar3 paramétrés par les corrélations partielles.** Poser |φᵢ| < 1 ne
  suffit pas — la région de stationnarité d'un AR(2) est un triangle, pas un
  carré, et un φ hors région donne une « corrélation » non définie positive.
  Levinson-Durbin envoie (−1,1)^p exactement sur la région admissible ; les φ
  rapportés restent ceux d'asreml.
- **`cor` borné à θ > −1/(q−1)**, la condition exacte de positivité (à q = 10,
  θ = −0,3 donne une valeur propre de −1,7).

Une divergence assumée avec le manuel : la formule imprimée pour `arma`,
ρ₁ = (θ−φ)(1−θφ)/(1+θ²−2θφ), produit des suites **non** définies positives sur
204 tirages de 300 dans le carré |θ|<1, |φ|<1 (dont ρ₁ = −0,976 avec
ρ₂ = −0,944, impossible). J'utilise la forme standard de l'ARMA(1,1). Le modèle
ajusté est le même ; seul le signe du paramètre MA rapporté peut différer.

### Résiduelle : même grammaire que les termes aléatoires

```r
residual = ~ units                            # σ² I
residual = ~ us(trait):units                  # couplage entre caractères
residual = ~ ar1(row):ar1(col)                # champ résiduel séparable
residual = ~ diag(trait):ar1(row):ar1(col)    # les deux
```

Sous le capot une seule formule : `R[i,j] = Σ_trait[t_i,t_j] · C_unité[u_i,u_j]`.
La résiduelle est un terme comme un autre, dont l'incidence est l'identité — il
n'y avait donc rien à ajouter au solveur, seulement à l'exposer.

`str()` est l'équivalent du `str(~ a + b, ~us(2):id(n))` d'asreml : les termes
groupés partagent leurs niveaux et reçoivent une covariance commune. C'est ce
qui permet de corréler un effet direct et un effet de voisinage portés par les
mêmes génotypes — le cas qui a motivé tout ceci. Chaque terme garde **son**
incidence, `mm()` compris.

Splines 2D par `rx_spl2d(x, y, nseg)` : produit tensoriel de P-splines,
décomposition PS-ANOVA en trois blocs aléatoires (`_x`, `_y`, `_xy`) plus une
partie nulle qui **doit** aller dans les effets fixes. Aucune machinerie
nouvelle côté solveur — une surface lisse est un effet aléatoire à incidence
connue, exactement comme dans sommer et asreml.

Les structures multi-caractères exigent `trait =` (données en **format long**,
une ligne par unité × caractère). Ce format est choisi plutôt qu'un `cbind()` de
réponses parce qu'il gère sans rien de spécial les caractères mesurés sur des
sous-ensembles d'unités différents — ce qui est la règle dès que le phénotypage
est incomplet.

Pour ce qu'une formule ne sait pas dire — incidence **pondérée**, colonnes
hétérogènes — la voie explicite, même machinerie sans le sucre :

```r
mod <- rx_model(y, X,
                terms = list(rx_term("voisinage", list(W1, W2), K = Kb, struct = "us")),
                residual = rx_residual("us", trait = d$trait, unit = d$unit))
fit <- rx_fit(mod, backend = "auto")
```

## 3. CPU ou GPU

`backend = "auto"` prend le GPU si `jax.devices()` en voit un, sinon le CPU, et
le dit. `"gpu"` **refuse explicitement** s'il n'y en a pas plutôt que de basculer
en silence. Le périphérique est fixé une fois, par `jax.default_device` : c'est
le seul endroit du code où le choix de machine intervient.

Le couplage R ↔ Python passe par des **fichiers** (binaire brut + manifeste
JSON), pas par reticulate. Deux raisons : sur ce cluster R et JAX vivent dans
deux conteneurs distincts (`ige_pipeline.sif`, `jax_gpu.sif`), et le fichier rend
la parité vérifiable puisque les deux backends lisent alors strictement la même
entrée. `RX_PY`, `IGE_JAX_CMD` ou `IGE_JAX_SIF` disent comment appeler Python.

## 4. Méthode, et ce qu'elle partage avec asreml

**Même objectif** : la vraisemblance restreinte. Donc même optimum, et des
estimations qui coïncident — c'est vérifié ci-dessous.

**Chemin différent** : asreml utilise l'information moyenne (AI-REML) dès la
première itération. Ici, L-BFGS-B sur une paramétrisation **non contrainte** fait
l'approche, puis un **polissage de Newton régularisé** finit le travail. La
positivité de Σ est garantie par construction (on paramètre un facteur de
Cholesky, jamais Σ), donc aucune contrainte à gérer.

Deux points où cette mécanique s'est révélée décisive :

- **Mise à l'échelle de l'objectif.** `-2logL` est d'ordre *n*, son gradient
  aussi. L-BFGS-B démarre avec un hessien approché égal à l'identité : son
  premier pas vaut `-gradient`, soit des centaines d'unités dans l'espace des
  paramètres. Sans division par *n*, mesure sur un modèle à un facteur : arrêt à
  `-2logL = 425,15` avec |grad| = 74,6, contre **414,3095** à l'optimum. Onze
  points de log-vraisemblance perdus en silence.
- **Polissage.** Depuis un L-BFGS-B volontairement arrêté tôt, l'erreur sur σ²
  passe de 2×10⁻² à 9×10⁻¹⁰ et le décrément de Newton de 4×10⁻³ à 9×10⁻¹⁸.

Le critère d'arrêt rapporté n'est pas `max|grad|` — il vaut plusieurs unités à
des optima vérifiés — mais le **décrément de Newton** `g' H⁻¹ g`, qui est la
montée de logLik encore disponible, donc comparable au 3,84 d'un LRT à 1 ddl.

## 5. Validation

### Inférence : vpredict et Wald

```r
f <- rx_reml(y ~ trt, random = ~ gid, data = d,
             vpredict = c(h2 = "V1/(V1+V2)"), wald = TRUE)
```

Les `Vi` sont numérotées dans l'ordre de `f$composantes_noms` — ordre **fixe et
documenté**, parce qu'une renumérotation silencieuse rendrait toute expression
fausse au run suivant.

| | remlax | asreml |
|---|---|---|
| h² | 0,6685333 | 0,6685333 |
| SE(h²), delta method | 0,0500972 | 0,0500998 |
| Wald `trt`, 2 ddl | p = 8,776×10⁻⁹ | p = 8,776×10⁻⁹ |

**Le Wald de remlax est CONDITIONNEL** (type III : chaque terme sachant tous
les autres) ; celui d'asreml est **séquentiel** par défaut (type I). Les deux
coïncident sur le dernier terme du modèle, seul cas où les notions se
rejoignent — c'est donc lui que le test compare. Sans `kenward_roger = TRUE`
les degrés de liberté du dénominateur ne sont pas ajustés : les p-valeurs sont
celles d'un χ²/ddl, valides asymptotiquement.

### Kenward-Roger, et predict

```r
f  <- rx_reml(y ~ trt + x, random = ~ iid(bloc), data = d, kenward_roger = TRUE)
pv <- rx_predict(f, classify = "trt", sed = TRUE)
```

Comparé à `pbkrtest::KRmodcomp`, l'implémentation de référence, sur un
dispositif en blocs **déséquilibré** (n = 27, 5 blocs de 3 à 7 parcelles) :

| | remlax | pbkrtest |
|---|---|---|
| ddl du dénominateur | 21,76897 | 21,76896 |
| F (`trt`, 2 ddl) | 2,790148 | 2,790148 |
| p | 0,0834186 | 0,0834186 |
| erreurs-types ajustées | à 1,3×10⁻⁷ | — |

Deux points sans lesquels le résultat est faux, tous deux trouvés en comparant :

- **W est l'inverse de l'information REML ESPÉRÉE** ½tr(P ∂V_i P ∂V_j), pas du
  Hessien observé. Les deux coïncident asymptotiquement, pas à n = 27 : le
  Hessien donnait 21,83 ddl au lieu de 21,769.
- **A₁ et A₂ contractent Θ contre ∂Φ/∂θ = Φ P Φ**, pas contre P. Avec P, ils
  valent ~n³ au lieu de O(1) et les degrés de liberté sortent **négatifs**
  (−0,56 sur ce jeu).

Le **terme du second ordre est omis**, et ce n'est pas un oubli : mesuré, il
n'est pas invariant par reparamétrisation. Sur un dispositif en blocs, le même
modèle donne Φ_A = 0,4375 en variances contre 0,3465 en log-écarts-types (21 %
d'écart) alors que la partie du premier ordre coïncide à 2×10⁻¹⁶. Ce terme
corrige le biais du second ordre de l'estimateur *dans la paramétrisation
choisie* : il n'a de sens qu'exprimé dans les composantes naturelles, où il
s'annule dès que V y est linéaire. pbkrtest et SAS font le même choix.

`predict` reproduit asreml : valeurs à **1,7×10⁻¹⁰**, erreurs-types à
**1,2×10⁻⁸** sur `classify = "trt"` avec une covariable (fixée à sa moyenne) et
un bloc aléatoire moyenné. Avec `include_random = TRUE`, le BLUP entre dans la
prédiction et l'erreur devient une erreur de **prédiction** :

    L'ΦL − 2 sym(L'Φ X'V⁻¹ Z G M) + M'G M − M'G Z' P Z G M

Les trois termes comptent ; le deuxième est négatif, la corrélation entre β̂ et
û réduisant l'erreur du total.

### Parité CPU/GPU sur tout le catalogue

`scripts/tests/export_bundles.R` sérialise **34 dispositifs** couvrant chaque
structure, `scripts/tests/parite_gpu.py` les ajuste sur les deux backends. Les
deux lisent le *même* paquet — c'est l'intérêt du format sur fichier.

Résultat : −2logL identique, écart relatif ≤ 1,1×10⁻¹⁴, le plus souvent
**exactement 0**. Rejoué sur une **tranche MIG A100** du cluster (job 10326144,
image `ige_reml.sif`) : mêmes 34 modèles, écart ≤ 4,1×10⁻¹⁵.

Une observation pratique : sur une tranche MIG (7 SM sur 108), la **compilation
XLA domine largement le calcul**. `mtrn` anisotrope y prend 252 s contre 2,8 s
sur CPU pour un modèle de 160 observations — la quadrature de Bessel produit un
noyau coûteux à compiler, pas à exécuter. Pour de petits modèles, le CPU est le
bon choix ; le GPU paie sur la taille de V, pas sur le nombre de modèles.

Le vrai apport du balayage n'est pas la parité mais les **trois bugs qu'aucune
suite n'avait vus** — auxquels s'ajoute un quatrième, trouvé en relançant les
suites existantes après coup et qui mérite d'être connu :

> **`$` fait de l'appariement PARTIEL sur les listes R.** En ajoutant un champ
> `sigmas_res` au résultat, l'expression `r$sigmas` — pas encore créée à ce
> point de `rx_read_result()` — s'est mise à s'apparier sur `sigmas_res`. Les Σ
> des termes étaient donc ajoutés à la liste des résiduelles, et
> `fit$sigmas[[1]]` rendait la **résiduelle** au lieu du premier terme. Sept
> échecs sur trois suites, tous dus à ce décalage d'un cran — et la logLik, elle,
> restait juste, ce qui rendait le diagnostic contre-intuitif.
> `rx_read_result()` n'utilise plus que `[[ ]]`, qui apparie exactement.


1. `own` : `^`, écrit en R, était le XOR de Python
   (`xor does not accept dtype float64`). Traduit en `**`.
2. `dsum(~ ar1(col) | site)` avec plusieurs lignes par colonne : R singulière,
   −2logL = NaN **en silence**. `validate()` ne contrôlait les unités dupliquées
   que sous us/diag/fa ; le contrôle porte désormais sur toute structure entre
   unités.
3. `ilv` rendait NaN — ce qui a conduit à le retirer.

Paramètres **fixés** (le code `F` d'asreml) : `fixed_theta = c(3, 5)` borne ces
paramètres à leur valeur de départ. Le diagnostic les voit alors « à une borne »
et les exclut du sous-espace libre, ce qui est le bon comportement pour les
degrés de liberté d'un LRT ultérieur. Le code `P` (positif) est automatique —
on paramètre des log-écarts-types — et `U` est le défaut.

### Le modèle IGE complet, sur données simulées de vérité connue

`scripts/tests/test_remlax_ige.R` reproduit en petit le modèle des scripts
05/06 : DGE + IGE intra sur **voisinage pondéré** avec covariance `us(2)` libre,
IGE inter, champ AR1×AR1, pépite. Huit paramètres de variance.

Dispositif 40×30 avec 90/60 génotypes, espèces alternées par colonne, noyau
puissance d⁻¹ à l'ordre 6 ou exponentiel exp(−κ(d−δ₀)). asreml ajuste le **même**
modèle via `str(~ geno + grp(nb), ~us(2):id(q)) + grp(xb) + ar1(row):ar1(col)`.

| composante | remlax | asreml | écart | (SE asreml) |
|---|---|---|---|---|
| var_DGE | 1,09524 | 1,09592 | 6,7×10⁻⁴ | 0,189 |
| cov(DGE, IGE intra) | −0,42211 | −0,41939 | 2,7×10⁻³ | 0,180 |
| var_IGE intra | 0,16268 | 0,17627 | 1,4×10⁻² | 0,354 |
| var_IGE inter | 0,54369 | 0,53748 | 6,2×10⁻³ | 0,341 |
| var champ | 0,61009 | 0,61022 | 1,3×10⁻⁴ | 0,164 |
| ρ ligne | 0,53375 | 0,53215 | 1,6×10⁻³ | 0,131 |
| ρ colonne | 0,11931 | 0,11850 | 8,1×10⁻⁴ | 0,111 |
| pépite | 0,39824 | 0,39694 | 1,3×10⁻³ | 0,156 |

Les huit composantes coïncident à **moins de 4 % d'une erreur-type**, et remlax
s'arrête 0,021 point de logLik au-dessus d'asreml.

#### La taille du dispositif décide de ce que le test prouve

Le test tournait sur 16×30 avec **18/12 génotypes** : 96 observations pour 8
paramètres de variance. À cette taille le bloc `us(2)` est à la frontière
(r = −1,0000) **des deux côtés** — asreml rend `NA` comme erreur-type de trois
paramètres du champ, sa façon de dire qu'il est sur une arête — et les deux
solveurs s'arrêtent à **0,305 point de logLik** l'un de l'autre. Comparer leurs
composantes ne prouvait alors rien : ce sont deux points différents d'une même
surface plate, et exiger qu'ils coïncident revient à demander à un maximum
d'égaler un non-maximum.

Ce qui sépare DGE et IGE intra, c'est le **nombre de génotypes**, pas le nombre
de parcelles : ils sont portés par les mêmes individus. Mesuré — l'écart de
logLik entre les deux solveurs :

| dispositif | génotypes | n | écart logLik | composantes |
|---|---|---|---|---|
| 16×12 | 18 / 12 | 96 | 0,305 | non comparables |
| 40×30 | 18 / 12 | 600 | 0,015 | encore à la frontière |
| 40×30 | 90 / 60 | 600 | **0,021** | **les 8 à < 4 % d'un SE** |

Le test compare donc les composantes seulement quand les deux logLik se
rejoignent à moins de 0,05 — très en dessous du 1,92 d'un LRT à 1 ddl — et le
dit explicitement sinon, au lieu de conclure à un désaccord.

**Un piège que ce test a exhumé.** Le champ AR1×AR1 doit être indexé sur les
colonnes **réellement occupées**, pas sur la grille complète : les espèces
alternent par colonne, donc le décalage 1 d'asreml est le décalage 2 du champ
sur grille pleine. Avant correction : remlax ρ_col = −0,4856, asreml +0,2358,
et (−0,4856)² = 0,2358 **exactement** — le carré effaçant au passage le signe.
C'est le même piège que sur le dispositif réel.

### Contre asreml — la référence du domaine

`scripts/tests/test_remlax_asreml.R` — 20 vérifications, licence enterprise
locale.

| structure | accord |
|---|---|
| `ar1(f)` | ρ, variance, résiduelle et **logLik identiques à 10⁻⁸** |
| `ar1(r, c)` séparable | variance, ρ_ligne, ρ_colonne, résiduelle, logLik à 10⁻⁸ |
| spline 2D, **mêmes matrices de base** | variances et logLik à 3×10⁻⁸ |
| `str(~ a + b, us)` vs `str(~ gid + gid:w, ~us(2):id(40))` | les trois éléments de Σ et la **logLik à 1,9×10⁻¹²** |

**Convention de log-vraisemblance** : asreml omet la constante
`(n−p)/2·log(2π)`, remlax l'inclut (comme lme4). Le champ `logLik_asreml` fait
la conversion — vérifiée à 1,3×10⁻⁸. Sans elle l'écart vaut 219,6 sur un jeu à
n = 240 et ressemble à un désaccord de modèle.

Un cas où **asreml refuse et remlax converge** : champ AR1×AR1 sans réplication,
où le champ absorbe toute la résiduelle. asreml s'arrête sur « 1 singularities in
the Average Information matrix » ; remlax rend un optimum (décrément 1,8×10⁻²⁶)
avec la résiduelle à 4×10⁻¹¹ — et le **signale** via
`composantes_degenerees`. Même diagnostic, exprimé autrement.

### Contre des implémentations écrites par d'autres

`scripts/tests/test_remlax.R` — 31 vérifications.

| cas | référence | résultat |
|---|---|---|
| un facteur aléatoire (3 plans) | `lme4::lmer` | −2logL identique à **10⁻¹³**, BLUPs à 7×10⁻⁹ |
| deux facteurs croisés | `lme4::lmer` | −2logL identique à 10⁻¹³ |
| parenté génomique | `sommer::mmer` | variances à 4×10⁻⁵ |
| 2 caractères, `us` génétique + résiduelle | `sommer::mmer` | G et R à la 4ᵉ décimale, corrélation génétique **0,7460 vs 0,7460** |
| formule vs voie explicite | lui-même | écart exactement 0 |

### Contre l'algèbre

`scripts/tests/test_remlax_core.py` — 22 vérifications : REML analytique d'un
plan équilibré retrouvée à 10⁻⁹ ; gradient autodiff contre différences finies à
10⁻⁸ ; `V == Z (Σ ⊗ K) Z' + R` calculé par une voie indépendante à **5×10⁻¹⁵** ;
`us` représente n'importe quelle matrice PSD à 10⁻¹⁵.

### Sous balayage aléatoire

`scripts/tests/stress_remlax.py` tire des centaines de configurations, y compris
pénibles (variance nulle, incidences dégénérées, forte parenté, plans très
déséquilibrés), et vérifie des **propriétés** plutôt que des valeurs : pas
d'exception, logLik finie, décrément négligeable, Σ symétriques et PSD, optimum
stable au redémarrage.

Le balayage a trouvé trois défauts réels, tous corrigés :

1. **15 plantages sur 120** dans la décomposition spectrale : le Hessien par
   différences finies contient des non-finis quand un pas sort du domaine où V
   est définie positive. Corrigé par bascule en différence avant/arrière, et
   diagnostic qui se déclare indisponible au lieu de planter.
2. **11 logLik NaN**, toutes avec résiduelle `us` : le générateur produisait des
   couples (unité, caractère) en double, ce qui rend R singulière. Le solveur
   **refuse** maintenant explicitement — c'est une erreur qu'un utilisateur ferait.
3. **`Singular matrix`** sur `inv(V)` : remplacé par une résolution avec repli
   sur la pseudo-inverse, en signalant que β et les BLUPs sont alors conditionnels.

Après correction : **150 configurations, 0 exception, 0 logLik non finie, 0 Σ non
PSD, 0 non convergé**. Restaient 2 cas d'**optimum local**, où un redémarrage
depuis un autre point trouve mieux. Le décrément de Newton ne peut pas les
détecter : il mesure la montée disponible *localement*, donc il vaut zéro au
sommet d'une colline secondaire. D'où l'option `n_restarts`, qui perturbe le
point retenu et garde le meilleur — elle récupère 0,0085 et 0,017 de `-2logL` sur
les deux cas concernés.

## 6. Limites connues

- **V est formée explicitement** (n × n dense). À n = 16 000 cela fait 2,1 Go en
  float64 : c'est le régime visé, pas au-delà. Passer à plus grand demanderait
  une factorisation creuse ou les équations de Henderson — `dense_Z()` et
  `assemble_V()` sont les deux seuls endroits à changer.
- **Les redémarrages ne sont pas actifs par défaut** (`n_restarts = 0`), parce
  qu'ils multiplient le coût. Les activer sur tout ajustement destiné à être
  publié.
- ~~Vérifié sur CPU seulement~~ — **levé le 01/09/2026**. JAX CUDA installé sur
  le poste (RTX A1000 6 Go) ; les suites tournent sur GPU. Parité mesurée sur le
  même modèle : **−2logL identique au bit près**, Σ à 2,8×10⁻¹⁵, BLUPs à
  3,5×10⁻¹⁴, β à 4,2×10⁻¹⁶.

  Piège corrigé au passage : `jax.devices()` **sans argument** ne rend que la
  plateforme par défaut. Dès qu'un GPU est visible elle ne liste plus aucun CPU,
  donc filtrer sa sortie sur `platform == "cpu"` rendait une liste vide et
  `--backend=cpu` retombait silencieusement sur le GPU. Il faut demander
  `jax.devices("cpu")`.
- **Reste dehors** : les modèles non gaussiens (GLM), qui demanderaient une
  couche IRLS/Laplace au-dessus du solveur — un chantier, pas un ajout. Et
  `ilv`, dont la formule d'asreml n'a pas pu être retrouvée (voir §2).
- **Kenward-Roger sans le terme du second ordre.** La formule complète porte
  −R_ij/4 avec R_ij = U′(∂²V/∂σᵢ∂σⱼ)U, nul dès que V est linéaire dans les
  paramètres de variance — vrai de toutes les structures d'ici **sauf** les
  paramètres de corrélation entre niveaux (φ d'un AR1, une portée). C'est ce que
  font pbkrtest et SAS. Le champ `second_ordre_omis` dit si le modèle contient
  des paramètres pour lesquels l'omission n'est pas exacte.
- **`predict()` couvre les effets fixes et un terme aléatoire à un caractère.**
  Un classify portant sur un terme multi-caractère est signalé et ignoré dans la
  part aléatoire.
- **La portée du noyau de voisinage n'est pas estimée** : le chaînage
  κ → Z → V demanderait de reconstruire l'incidence à chaque itération, ce que
  `scripts/gpu/reml_uni_kernel.py` fait déjà et qu'il faudrait porter ici.
