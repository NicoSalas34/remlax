"""Scan GWAS a V figee, POSE A COTE du solveur.

Ce fichier n'est importe par aucun autre module de remlax et n'en modifie
aucun. Il consomme la SORTIE de `fit_reml` — rien de plus — et n'a besoin ni de
JAX ni d'un GPU : `numpy` suffit, parce que tout ce qui etait differentiable a
deja ete fait par le solveur.

CE QUE C'EST. Un GWAS de type EMMAX (Kang et al. 2010) : les composantes de
variance sont estimees UNE FOIS sous l'hypothese nulle, puis chaque SNP est
teste comme effet fixe a V figee. Pour une colonne s ajoutee a X,

    beta_s = (s'Ps)^-1 s'Py      var(beta_s) = (s'Ps)^-1      chi2 = (s'Py)^2/(s'Ps)
    P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

soit exactement la formule de `inference.wald` reecrite pour une colonne. V ne
bougeant pas, le score et le Wald coincident : une seule formule.

POURQUOI CE N'EST PAS UNE BOUCLE SUR LES SNP. Les covariables marqueurs vivent
au niveau GENOTYPE (q de l'ordre de quelques centaines), pas au niveau
observation (n de l'ordre de la dizaine de milliers). En projetant P une seule
fois sur les incidences du dispositif,

    c_f = Z_f' P y        C_fg = Z_f' P Z_g

les p statistiques s'obtiennent par des produits matriciels en dimension q :

    num_f(j)   = m_j' c_f
    den_fg(j)  = m_j' C_fg m_j

Le cout passe de O(n^2 p) a O(q^2 p). Et la projection elle-meme ne coute
AUCUNE factorisation nouvelle, parce que `fit_reml` renvoie deja `Vi`, `vbeta`
et `Py` :

    C_fg = Z_f' Vi Z_g - (Z_f' Vi X) vbeta (X' Vi Z_g)

TROIS FAMILLES DE COVARIABLES, ET POURQUOI ELLES SONT TROIS ET PAS UNE.
Avec m_j les doses du SNP j, Z_dir l'incidence du genotype porte par la plante
et Z_ind celle des genotypes de ses voisins (ponderee) :

    dir : s = Z_dir m_j                      effet genetique DIRECT du SNP
    ind : s = Z_ind m_j                      effet genetique INDIRECT (dose du voisin)
    sim : s = (Z_dir m_j) * (Z_ind m_j)      SIMILARITE allelique focal x voisin

La troisieme est la covariable de Sato et al. (2021, 2024) : en codage +/-1 le
produit vaut +1 quand le voisin porte le meme allele et -1 sinon. C'est aussi,
exactement, le terme d'INTERACTION entre l'effet direct et l'effet du voisin,
donc un DGE x IGE. Le modele de Sato porte `dir` et `sim` mais PAS `ind` ; le
modele d'IGE additif au sens de Muir (2005) et Bijma (2011) porte `dir` et
`ind` mais pas `sim`. Aucun des deux ne teste ce que l'autre teste, et les
tests emboites sur les trois familles a la fois valent mieux que chacun (voir
docs/scan.md).

REGIMES DE COUT. `dir` et `ind` sont LINEAIRES en m : tout se passe en
dimension q, en O(q^2 p). `sim` est BILINEAIRE en m — elle depend du produit
des doses du focal et du voisin — donc elle ne se reduit pas a l'espace
genotype. Ses termes croises avec `dir` et `ind` restent bon marche, parce que
    s_sim' P (Z_f m) = (s_sim' P Z_f) m
et P Z_f est deja calcule par la projection ; seul le terme s_sim' P s_sim
exige un produit en dimension n, soit O(n^2) par SNP. C'est le seul poste cher
du fichier, et il est ecrit comme un produit matriciel par blocs pour que
BLAS — ou un tableau JAX passe en `xp` — en fasse ce qu'il peut.

CODAGE DES MARQUEURS. `sim` n'a le sens d'une similarite que si les doses sont
CENTREES (codage +/-1, ou dose centree). En codage 0/1/2 le produit n'est pas
une similarite mais une quantite sans interpretation simple. `scan()` refuse
donc la famille `sim` sur une matrice de doses manifestement non centree,
plutot que de rendre un resultat que rien ne signale comme faux.
"""
import numpy as np

FAMILLES = ("dir", "ind", "sim")

# Familles lineaires en m : celles qui vivent entierement en dimension genotype.
LINEAIRES = ("dir", "ind")


# ==============================================================================
# Projection : P vue depuis les incidences du dispositif
# ==============================================================================
class Projection:
    """c_f = Z_f' P y, C_fg = Z_f' P Z_g, et P Z_f garde pour la famille `sim`.

    Attributs
    ---------
    c    : dict famille -> vecteur (q_f,)
    C    : dict (famille, famille) -> matrice (q_f, q_g), symetrique en (f,g)
    PZ   : dict famille -> matrice (n, q_f), utile aux termes croises de `sim`
    Py   : vecteur (n,), recopie de fit["Py"]
    q    : dict famille -> q_f
    """

    def __init__(self, c, C, PZ, Py, Z, Vi, X, vbeta, ViX):
        self.c, self.C, self.PZ, self.Py, self.Z = c, C, PZ, Py, Z
        # REFERENCES, pas des copies : Vi pese 2,1 Go a n = 16 000 et il existe
        # deja dans `fit`. `np.asarray` sur un tableau float64 rend le meme
        # objet, donc rien n'est duplique. Ces attributs ne servent qu'a la
        # famille `sim`, seule a avoir besoin de P s pour un s qui n'est pas
        # une colonne d'incidence.
        self._Vi, self._X, self._vbeta, self._ViX = Vi, X, vbeta, ViX
        self.q = {f: Z[f].shape[1] for f in Z}
        self.n = len(Py)

    def __repr__(self):
        return ("Projection(n=%d, %s)"
                % (self.n, ", ".join("%s: q=%d" % (f, q)
                                     for f, q in sorted(self.q.items()))))


def _dense(Z):
    """Accepte numpy, matrice numpy ou creuse scipy ; rend un tableau 2-D."""
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float64)


def projeter(fit, X, incidences):
    """Projette P sur les incidences, une fois pour toutes les familles.

    Parametres
    ----------
    fit         : dict rendu par `fit_reml` ; doit porter `Vi`, `vbeta`, `Py`
                  (c'est le cas des que `blups=True` ou `hessian=True`, donc par
                  defaut). Aucune autre cle n'est lue.
    X           : matrice des effets fixes du modele nul (n x p_fixe), la meme
                  qui a servi a l'ajustement.
    incidences  : dict {"dir": Z_dir, "ind": Z_ind}. Au moins une cle. Les deux
                  peuvent avoir un nombre de colonnes DIFFERENT : c'est le cas
                  inter-especes, ou les voisins d'un ble sont des luzernes.

    Cout : un produit (n x n) par (n x q) par famille. Aucune factorisation.
    """
    for cle in ("Vi", "vbeta", "Py"):
        if cle not in fit:
            raise KeyError(
                "fit['%s'] absent : `scan` lit la sortie de fit_reml et a besoin "
                "de Vi, vbeta et Py. Ils sont produits des que blups=True ou "
                "hessian=True (le defaut) ; un ajustement lance avec "
                "--no-blups --no-hessian ne les contient pas." % cle)
    inconnues = set(incidences) - set(LINEAIRES)
    if inconnues:
        raise ValueError("incidences : cle(s) inconnue(s) %s ; attendu parmi %s"
                         % (sorted(inconnues), list(LINEAIRES)))
    if not incidences:
        raise ValueError("incidences vide : fournir au moins 'dir' ou 'ind'.")

    Vi = np.asarray(fit["Vi"], dtype=np.float64)
    vbeta = np.asarray(fit["vbeta"], dtype=np.float64)
    Py = np.asarray(fit["Py"], dtype=np.float64).ravel()
    X = _dense(X)
    n = len(Py)
    if Vi.shape != (n, n):
        raise ValueError("Vi est %s mais Py a %d elements." % (Vi.shape, n))

    Z = {f: _dense(Zf) for f, Zf in incidences.items()}
    for f, Zf in Z.items():
        if Zf.shape[0] != n:
            raise ValueError("incidence '%s' a %d lignes, attendu %d."
                             % (f, Zf.shape[0], n))

    ViX = Vi @ X
    # P Z = Vi Z - ViX vbeta (X' Vi Z). L'ordre evite de former P (n x n).
    PZ, c, C = {}, {}, {}
    for f, Zf in Z.items():
        ViZ = Vi @ Zf
        PZ[f] = ViZ - ViX @ (vbeta @ (X.T @ ViZ))
        c[f] = Zf.T @ Py
    for f in Z:
        for g in Z:
            if (g, f) in C:
                C[(f, g)] = C[(g, f)].T
            else:
                C[(f, g)] = Z[f].T @ PZ[g]
    return Projection(c, C, PZ, Py, Z, Vi, X, vbeta, ViX)


# ==============================================================================
# Analyse des specifications de test
# ==============================================================================
def _parse_test(spec):
    """'sim|dir+ind' -> (modele, testees). 'dir+ind' -> (les deux, les deux).

    A gauche de la barre, ce qui est TESTE ; a droite, ce sur quoi on
    conditionne. Le modele est l'union des deux. Sans barre, on teste tout le
    modele — c'est le test conjoint.
    """
    if "|" in spec:
        gauche, droite = spec.split("|", 1)
        testees = [t.strip() for t in gauche.split("+") if t.strip()]
        cond = [t.strip() for t in droite.split("+") if t.strip()]
    else:
        testees = [t.strip() for t in spec.split("+") if t.strip()]
        cond = []
    modele = list(dict.fromkeys(testees + cond))
    mauvaises = [f for f in modele if f not in FAMILLES]
    if mauvaises:
        raise ValueError("test '%s' : famille(s) inconnue(s) %s ; attendu parmi %s"
                         % (spec, mauvaises, list(FAMILLES)))
    if not testees:
        raise ValueError("test '%s' : rien de teste a gauche de la barre." % spec)
    return modele, testees


def familles_requises(tests):
    """Les familles de covariables qu'il faut calculer pour cette liste de tests."""
    besoin = []
    for spec in tests:
        modele, _ = _parse_test(spec)
        for f in modele:
            if f not in besoin:
                besoin.append(f)
    return [f for f in FAMILLES if f in besoin]


# ==============================================================================
# Scan
# ==============================================================================
def _marqueurs(M, familles):
    """Normalise M en dict famille -> matrice (q_f x p).

    Un seul tableau => les MEMES SNP des deux cotes (cas intra-espece). Un dict
    => une matrice par famille, ce qui autorise le cas inter-especes MAIS
    interdit alors les tests conjoints, puisque la colonne j des deux matrices
    ne designe pas le meme SNP. `scan` verifie ce point.
    """
    if isinstance(M, dict):
        manquantes = [f for f in familles if f not in M and f in LINEAIRES]
        if manquantes:
            raise ValueError("M : matrice de doses absente pour %s" % manquantes)
        out = {f: np.asarray(M[f], dtype=np.float64) for f in M}
    else:
        Ma = np.asarray(M, dtype=np.float64)
        out = {f: Ma for f in LINEAIRES}
    ps = {f: Mat.shape[1] for f, Mat in out.items()}
    if len(set(ps.values())) > 1:
        raise ValueError("M : nombres de SNP incompatibles entre familles : %s" % ps)
    return out, next(iter(ps.values()))


def _memes_snp(Mf, familles):
    """Les familles portent-elles le MEME jeu de SNP ?

    Le critere ne peut pas etre l'identite d'OBJET : `scan_cli` relit les doses
    depuis deux fichiers, donc deux tableaux distincts, meme dans le cas
    intra-espece ou ce sont les memes SNP. C'est le CONTENU qui tranche. Les
    dimensions d'abord — un nombre de genotypes different signe le cas
    inter-especes et suffit a conclure — puis l'egalite element par element, une
    seule passe.
    """
    lin = [f for f in familles if f in LINEAIRES and f in Mf]
    if len(lin) < 2:
        return True
    ref = Mf[lin[0]]
    for f in lin[1:]:
        A = Mf[f]
        if A is ref:
            continue
        if A.shape != ref.shape or not np.array_equal(A, ref):
            return False
    return True


def scan(proj, M, tests=("dir", "ind", "sim"), garder=None, bloc=1024,
         xp=None, verbose=False):
    """Statistiques par SNP pour les tests demandes, a V figee.

    Parametres
    ----------
    proj    : `Projection` rendue par `projeter`.
    M       : matrice (q x p) des doses genotypiques, ou dict famille -> matrice.
              Pour la famille `sim`, les doses doivent etre CENTREES (codage
              +/-1 typiquement) : sans quoi le produit n'est pas une similarite.
    tests   : liste de specifications, par exemple
                "dir"            effet direct seul, 1 ddl (GWAS standard)
                "ind"            effet indirect additif seul, 1 ddl
                "sim"            similarite seule, 1 ddl (modele de Sato)
                "dir+ind"        conjoint, 2 ddl
                "dir+ind+sim"    conjoint, 3 ddl
                "ind|dir"        indirect SACHANT le direct, 1 ddl
                "sim|dir+ind"    similarite sachant les deux effets additifs
              Les tests conditionnels sont ceux d'asreml et de `inference.wald` :
              chaque terme sachant les autres.
    garder  : masque booleen (p,) des SNP a traiter ; les autres sortent en NaN.
              Le filtre de frequence allelique appartient a l'appelant.
    bloc    : largeur des blocs de SNP pour la famille `sim` (le seul poste en
              dimension n). Regle la memoire de pointe, pas le resultat.
    xp      : module tableau pour le produit lourd de `sim` (defaut numpy).
              Passer `jax.numpy` y substitue le GPU sans changer le reste.

    Retour
    ------
    dict de vecteurs (p,) :
      num_<f>, den_<f>_<g>          quantites brutes du score
      chi2_<test>, ddl_<test>, p_<test>
      beta_<test>, se_<test>        pour les tests a 1 ddl seulement
      r_dir_ind                     correlation entre covariables directe et
                                    indirecte, CONDITIONNELLEMENT a V et X
      ddl et p sont des vecteurs pour rester ecrivables tel quel en CSV.
    """
    from scipy.stats import chi2 as _chi2
    xp = np if xp is None else xp
    besoin = familles_requises(tests)
    Mf, p = _marqueurs(M, besoin)

    conjoints = [t for t in tests if len(_parse_test(t)[0]) > 1]
    if conjoints and not _memes_snp(Mf, besoin):
        raise ValueError(
            "tests conjoints %s demandes avec des jeux de marqueurs DIFFERENTS "
            "d'une famille a l'autre : la colonne j ne designe alors pas le meme "
            "SNP des deux cotes (cas inter-especes). Lancer les familles "
            "separement." % conjoints)

    for f in besoin:
        if f in LINEAIRES and f not in proj.q:
            raise ValueError("famille '%s' demandee mais l'incidence correspondante "
                             "n'a pas ete projetee (incidences fournies : %s)"
                             % (f, sorted(proj.q)))
    if "sim" in besoin:
        for f in LINEAIRES:
            if f not in proj.q:
                raise ValueError("la famille 'sim' est le produit des covariables "
                                 "directe et indirecte : les DEUX incidences sont "
                                 "necessaires (fournies : %s)" % sorted(proj.q))
        for f in LINEAIRES:
            mu = float(np.abs(Mf[f].mean()))
            sd = float(Mf[f].std())
            if sd > 0 and mu > 0.25 * sd:
                raise ValueError(
                    "famille 'sim' : les doses de '%s' ne sont pas centrees "
                    "(moyenne %.3g pour un ecart-type %.3g). Le produit des doses "
                    "n'est une similarite allelique qu'en codage centre (+/-1). "
                    "Centrer les doses, ou ne pas demander 'sim'." % (f, mu, sd))

    if garder is None:
        garder = np.ones(p, dtype=bool)
    garder = np.asarray(garder, dtype=bool)
    if garder.shape != (p,):
        raise ValueError("garder a %s elements, attendu (%d,)" % (garder.shape, p))
    idx = np.flatnonzero(garder)

    nan = lambda: np.full(p, np.nan)
    num = {f: nan() for f in besoin}
    den = {}
    for i, f in enumerate(besoin):
        for g in besoin[i:]:
            den[(f, g)] = nan()

    # ---- familles lineaires : tout en dimension genotype -------------------
    lin = [f for f in besoin if f in LINEAIRES]
    for f in lin:
        num[f][idx] = Mf[f][:, idx].T @ proj.c[f]
    for i, f in enumerate(lin):
        for g in lin[i:]:
            A = proj.C[(f, g)] @ Mf[g][:, idx]
            den[(f, g)][idx] = np.einsum("ij,ij->j", Mf[f][:, idx], A)

    # ---- famille sim : par blocs, seul poste en dimension n ----------------
    if "sim" in besoin:
        Mdir, Mind = Mf["dir"], Mf["ind"]
        Zdir, Zind = proj.Z["dir"], proj.Z["ind"]
        Py = proj.Py
        PZdir, PZind = proj.PZ["dir"], proj.PZ["ind"]
        # P n'est JAMAIS forme. Pour les colonnes d'incidence, P Z est deja
        # calcule ; s3 n'en est pas une, donc on refait
        #   P s3 = Vi s3 - ViX vbeta (X' Vi s3)
        # par blocs de SNP. C'est le seul poste en O(n^2) du fichier.
        Vi = proj._Vi
        for d0 in range(0, len(idx), int(bloc)):
            j = idx[d0:d0 + int(bloc)]
            u = Zdir @ Mdir[:, j]                       # (n, b)
            v = Zind @ Mind[:, j]                       # (n, b)
            S3 = u * v
            num["sim"][j] = S3.T @ Py
            # s_sim' P (Z_f m) = (s_sim' P Z_f) m : somme sur les q GENOTYPES,
            # une valeur par SNP du bloc -> "ij,ij->i", pas "->j".
            if ("dir", "sim") in den:
                den[("dir", "sim")][j] = np.einsum(
                    "ij,ij->i", S3.T @ PZdir, Mdir[:, j].T)
            if ("ind", "sim") in den:
                den[("ind", "sim")][j] = np.einsum(
                    "ij,ij->i", S3.T @ PZind, Mind[:, j].T)
            S3x = xp.asarray(S3)
            PS3 = xp.asarray(Vi) @ S3x
            PS3 = PS3 - xp.asarray(proj._ViX) @ (
                xp.asarray(proj._vbeta) @ (xp.asarray(proj._X).T @ PS3))
            den[("sim", "sim")][j] = np.asarray(
                xp.einsum("ij,ij->j", S3x, PS3))
            if verbose:
                print("  [scan sim] %d / %d SNP" % (min(d0 + int(bloc), len(idx)),
                                                    len(idx)), flush=True)

    # ---- tests ------------------------------------------------------------
    out = {}
    for f in besoin:
        out["num_" + f] = num[f]
    for (f, g), v in den.items():
        out["den_%s_%s" % (f, g)] = v

    def bloc_den(fams, j):
        k = len(fams)
        Dm = np.empty((len(j), k, k))
        for a, fa in enumerate(fams):
            for b, fb in enumerate(fams):
                key = (fa, fb) if (fa, fb) in den else (fb, fa)
                Dm[:, a, b] = den[key][j]
        return Dm

    for spec in tests:
        modele, testees = _parse_test(spec)
        k, kt = len(modele), len(testees)
        chi2v, betav, sev = nan(), nan(), nan()
        Dm = bloc_den(modele, idx)
        Nv = np.stack([num[f][idx] for f in modele], axis=1)        # (m, k)
        pos = [modele.index(f) for f in testees]
        ok = np.isfinite(Dm).all(axis=(1, 2)) & np.isfinite(Nv).all(axis=1)
        if k == 1:
            d = Dm[:, 0, 0]
            bon = ok & (d > 0)
            b = Nv[bon, 0] / d[bon]
            chi2v[idx[bon]] = Nv[bon, 0] ** 2 / d[bon]
            betav[idx[bon]] = b
            sev[idx[bon]] = 1.0 / np.sqrt(d[bon])
        else:
            # beta = D^-1 num ; var(beta) = D^-1 ; test du sous-ensemble A :
            #   chi2 = beta_A' [(D^-1)_AA]^-1 beta_A
            Di = np.full_like(Dm, np.nan)
            bon = ok.copy()
            if bon.any():
                sel = np.flatnonzero(bon)
                try:
                    Di[sel] = np.linalg.inv(Dm[sel])
                except np.linalg.LinAlgError:
                    for s in sel:
                        try:
                            Di[s] = np.linalg.inv(Dm[s])
                        except np.linalg.LinAlgError:
                            bon[s] = False
            beta = np.einsum("mab,mb->ma", Di, Nv)
            sub = Di[:, pos][:, :, pos]
            bA = beta[:, pos]
            for m in np.flatnonzero(bon):
                try:
                    W = np.linalg.inv(sub[m])
                except np.linalg.LinAlgError:
                    continue
                chi2v[idx[m]] = float(bA[m] @ W @ bA[m])
            if kt == 1:
                betav[idx[bon]] = bA[bon, 0]
                d = sub[bon, 0, 0]
                sev[idx[bon]] = np.sqrt(np.maximum(d, 0.0))
        out["chi2_" + spec] = chi2v
        out["ddl_" + spec] = np.full(p, float(kt))
        with np.errstate(invalid="ignore"):
            out["p_" + spec] = _chi2.sf(chi2v, kt)
        if kt == 1:
            out["beta_" + spec] = betav
            out["se_" + spec] = sev

    if ("dir", "ind") in den:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r_dir_ind"] = den[("dir", "ind")] / np.sqrt(
                den[("dir", "dir")] * den[("ind", "ind")])
    return out


def lambda_gc(chi2, ddl=1):
    """Facteur de controle genomique : mediane(chi2) / mediane theorique."""
    from scipy.stats import chi2 as _chi2
    v = np.asarray(chi2, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v) / _chi2.ppf(0.5, ddl))
