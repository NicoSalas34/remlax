"""Structures de covariance ENTRE NIVEAUX — catalogue façon asreml.

Toutes rendent une matrice de CORRELATION C (diagonale unite). La variance vit
dans Sigma (structures.py) : c'est la decomposition Sigma_h = D C D du manuel
asreml (annexe C), ou D = diag(sigma_i). Un `ar1v` d'asreml est donc ici
`struct="iid"` cote Sigma + `level="ar1"` cote niveaux ; un `ar1h`, `struct="diag"`
+ `level="ar1"`.

TOUTES LES STRUCTURES STATIONNAIRES SONT TOEPLITZ. Plutot que d'implementer
chaque recursion sur la matrice, on calcule la FONCTION D'AUTOCORRELATION
rho_0..rho_{q-1} puis on deploie C[i,j] = rho_{|i-j|}. Un seul chemin de code,
une seule chose a verifier par structure, et la definition tient en trois lignes.

CATALOGUE (definitions de l'annexe C du manuel ASReml-R 4.2)

  id      C = I                                              0 parametre
  fixed   C fournie (GRM, pedigree, noyau calcule)           0
  cor     C_ij = theta  (i != j)          correlation uniforme    1
  ar1     rho_k = phi^k                                      1
  ar2     rho_1 = phi1/(1-phi2), puis Yule-Walker            2
  ar3     idem a l'ordre 3                                   3
  sar     AR(2) contraint : phi1 = phi, phi2 = -phi^2/4      1
  ma1     rho_1 = -theta/(1+theta^2), 0 au-dela              1
  ma2     rho_1, rho_2 donnes, 0 au-dela                     2
  arma    rho_1 = (th-ph)(1-th*ph)/(1+th^2-2*th*ph),
          rho_k = ph * rho_{k-1}                             2
  corb    rho_k = phi_k pour k < b, 0 au-dela  (bande)       b
  exp     C_ij = phi^|x_i-x_j|   (coordonnees, pas forcement
          equidistantes ; en 2D, distance euclidienne)       1
  gau     C_ij = phi^{(x_i-x_j)^2}                           1
  lvr     C_ij = max(0, 1 - d_ij/phi)   "variance lineaire"  1
          Tente tronquee. Definie positive en 1D SEULEMENT :
          c'est l'autocorrelation d'un creneau, ce qui n'a pas
          d'equivalent en 2D. phi est une PORTEE, pas une
          correlation, d'ou exp(theta).
          PAS D'EQUIVALENT 2D ICI. asreml a un `ilv`, mais sa
          formule n'a pas pu etre retrouvee : aucune des neuf
          familles essayees ne reproduit sa courbe de logLik
          (cf. docs/note_remlax.md), et la tente euclidienne
          n'est pas definie positive — elle rend des NaN. On
          prefere ne pas la fournir plutot que d'en fournir une
          fausse sous le nom d'asreml.
  mtrn    Matern anisotrope (phi, nu, delta, alpha ; lambda
          fixe a 1 ou 2)                             0 a 4 estimes
  own     structure DEFINIE PAR L'UTILISATEUR : expression
          arithmetique de d/dx/dy/lag et de p1..pk       k

lvr ET mtrn ONT ETE IDENTIFIES CONTRE ASREML, pas lus dans le manuel (leurs
pages d'aide ne donnent pas les formules). Voir docs/note_remlax.md pour les
courbes de logLik qui les fixent.

STATIONNARITE GARANTIE PAR CONSTRUCTION pour ar2/ar3. Mettre |phi_i| < 1 ne
suffit PAS : la region de stationnarite d'un AR(2) est un triangle, pas un carre,
et un phi hors region donne une "correlation" qui n'est pas definie positive —
l'ajustement echoue alors sans qu'on sache pourquoi. On parametre donc les
CORRELATIONS PARTIELLES (chacune dans (-1,1) par tanh, sans contrainte), et on
en deduit les phi par Levinson-Durbin. La bijection PACF <-> region stationnaire
est exacte, et les phi rapportes restent ceux d'asreml.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
import jax.numpy as jnp
import numpy as np

try:
    from .bessel import matern as _matern
except ImportError:
    from bessel import matern as _matern

# nom -> nombre de parametres (None = variable, cf. n_level_params)
LEVEL_STRUCTURES = {
    "id": 0, "fixed": 0, "cor": 1, "ar1": 1, "ar2": 2, "ar3": 3,
    "sar": 1, "ma1": 1, "ma2": 2, "arma": 2, "corb": None,
    # 1D irregulier
    "exp": 1, "gau": 1,
    # 2D irregulier isotrope / anisotrope (annexe C, "power models")
    "iexp": 1, "igau": 1, "ieuc": 1, "sph": 1, "cir": 1,
    "aexp": 2, "agau": 2,
    # metriques a portee : le parametre est une DISTANCE, pas une correlation
    "lvr": 1,
    # nombre variable : cf. n_level_params
    "mtrn": None, "own": None,
    # correlation generale (corg) : omega(omega-1)/2 parametres
    "corg": None,
}

# Ordre CANONIQUE des parametres de mtrn. Il fixe la place de chacun dans theta
# et dans les sorties ; le changer invaliderait tout ajustement deja enregistre.
MTRN_PARAMS = ("phi", "nu", "delta", "alpha")
MTRN_DEFAUT = {"phi": 1.0, "nu": 0.5, "delta": 1.0, "alpha": 0.0, "lambda": 2.0}
# "sep" : produit separable de plusieurs facteurs (ar1 x ar1, exp x ar1, ...).
# "ar1ar1" est conserve comme raccourci du cas le plus courant.
# structures qui exigent des coordonnees
LEVEL_NEEDS_COORD = ("exp", "gau", "iexp", "igau", "ieuc", "sph", "cir",
                     "aexp", "agau", "mtrn", "lvr")


def n_level_params(kind, order=0, parts=None, opts=None):
    """Nombre de parametres.

    `parts` : liste de (kind, order) pour un produit separable, qui additionne
    ses facteurs. `opts` : reglages de la structure (mtrn : quels parametres
    sont estimes ; own : combien l'utilisateur en declare).
    """
    if kind == "ar1ar1":
        return 2
    if kind == "sep":
        # INCOHERENCE A NE PAS LAISSER PASSER EN SILENCE. `level_chol` lit le
        # second element de `parts` comme la DIMENSION du facteur, alors que
        # cette ligne le lisait comme son ORDRE. Pour `id` et `ar1` les deux
        # coincident par accident — leur compte ne depend pas de l'ordre — mais
        # pour `ar2`, `arma` ou une famille metrique le compte serait faux, et
        # rien ne le signalerait : theta serait decoupe autrement des deux cotes.
        # On lit donc la famille SEULE, ce qui est correct pour toute famille
        # dont le compte ne depend pas de l'ordre, et on refuse les autres.
        # DEUX FAMILLES DE REFUS, et le second manquait. `corb`, `corg`, `mtrn`
        # et `own` ont un nombre de parametres qui depend de `order` ou de
        # `opts`, que `parts` ne transporte pas : le compte tombait a zero et le
        # facteur devenait une identite SANS AUCUN MESSAGE (mesure : un sep
        # (corb, 4) x (ar1, 5) comptait 1 parametre au lieu de 5). Les familles
        # metriques exigent des coordonnees, que `parts` ne porte pas non plus.
        deps = ("ar2", "ar3", "ma2", "arma", "corb", "corg", "mtrn", "own") + LEVEL_NEEDS_COORD
        mauvais = [k for k, _ in (parts or []) if k in deps]
        if mauvais:
            raise ValueError(
                "produit separable : les familles %s ne sont pas admises dans "
                "`parts`, qui ne porte que (famille, dimension) : leur nombre de "
                "parametres depend d'un ordre ou d'options, ou elles exigent des "
                "coordonnees. Les declarer ici decouperait theta differemment de "
                "level_chol, ou donnerait une identite en silence. Utiliser "
                "`expr` pour ces cas." % sorted(set(mauvais)))
        return sum(n_level_params(k) for k, _ in (parts or []))
    if kind == "corb":
        return int(order)
    if kind == "corg":
        q = int(order)                      # ici `order` porte omega
        return q * (q - 1) // 2
    if kind == "mtrn":
        # Comme asreml : un parametre ABSENT de l'appel est tenu fixe a sa
        # valeur par defaut ; seuls les parametres declares sont estimes. C'est
        # ce qui evite d'ajuster nu, delta et alpha par accident sur un
        # dispositif qui ne les identifie pas.
        o = opts or {}
        return int(sum(float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5
                       for p in MTRN_PARAMS))
    if kind == "own":
        return int((opts or {}).get("n_par", 0))
    n = LEVEL_STRUCTURES.get(kind)
    if n is None:
        raise ValueError("structure de niveaux inconnue : %s" % kind)
    return n


# ==============================================================================
# Fonctions d'autocorrelation
# ==============================================================================
def _levinson_phi(pacf):
    """Coefficients AR phi_1..phi_p depuis les correlations partielles.

    Recursion de Levinson-Durbin. Elle envoie (-1,1)^p exactement sur la region
    de stationnarite : tout jeu de pacf admissible donne un AR stationnaire, donc
    une matrice de correlation definie positive.
    """
    p = len(pacf)
    phi = jnp.zeros(p)
    for k in range(p):
        pk = pacf[k]
        if k == 0:
            phi = phi.at[0].set(pk)
        else:
            prev = phi[:k]
            phi = phi.at[:k].set(prev - pk * prev[::-1])
            phi = phi.at[k].set(pk)
    return phi


def _acf_ar(phi, q):
    """ACF d'un AR(p) : Yule-Walker pour rho_1..rho_p, puis recursion.

    rho_k = sum_i phi_i rho_{|k-i|}. Pour k = 1..p c'est un systeme lineaire en
    rho ; au-dela c'est une simple recursion.
    """
    p = len(phi)
    if p == 0:
        return jnp.concatenate([jnp.ones(1), jnp.zeros(q - 1)])
    # systeme A rho = b avec rho = (rho_1..rho_p)
    A = jnp.eye(p)
    b = jnp.zeros(p)
    for k in range(1, p + 1):
        b = b.at[k - 1].add(phi[k - 1])          # terme i = k : phi_k * rho_0
        for i in range(1, p + 1):
            if i == k:
                continue
            j = abs(k - i)                        # rho_{|k-i|}
            if j == 0:
                b = b.at[k - 1].add(phi[i - 1])
            else:
                A = A.at[k - 1, j - 1].add(-phi[i - 1])
    rho = jnp.linalg.solve(A, b)
    acf = jnp.concatenate([jnp.ones(1), rho])
    for k in range(p + 1, q):
        acf = jnp.concatenate([acf, jnp.sum(phi * acf[k - 1:k - p - 1:-1])[None]])
    return acf[:q]


def _acf(theta, kind, q, order=0):
    """rho_0..rho_{q-1} d'une structure stationnaire."""
    if kind == "id":
        return jnp.concatenate([jnp.ones(1), jnp.zeros(q - 1)])
    if kind == "cor":
        # Correlation uniforme : C = (1-t) I + t J n'est definie positive que
        # pour t > -1/(q-1) (verifie numeriquement : a q=10, t=-0.3 donne une
        # valeur propre de -1.7). tanh seul autoriserait tout (-1,1) et
        # l'ajustement echouerait sans dire pourquoi. On envoie donc (-1,1) sur
        # (-1/(q-1), 1), ce qui couvre EXACTEMENT le domaine admissible.
        lo = -1.0 / max(q - 1, 1)
        t = lo + (jnp.tanh(theta[0]) + 1.0) * 0.5 * (1.0 - lo)
        return jnp.concatenate([jnp.ones(1), jnp.full(q - 1, t)])
    if kind == "ar1":
        phi = jnp.tanh(theta[0])
        return phi ** jnp.arange(q)
    if kind in ("ar2", "ar3"):
        p = 2 if kind == "ar2" else 3
        return _acf_ar(_levinson_phi(jnp.tanh(theta[:p])), q)
    if kind == "sar":
        # AR(2) contraint : phi1 = phi, phi2 = -phi^2/4. Toujours stationnaire
        # (racines reelles doubles), donc pas de reparametrisation necessaire.
        ph = jnp.tanh(theta[0])
        return _acf_ar(jnp.array([ph, -ph ** 2 / 4.0]), q)
    if kind == "ma1":
        th = jnp.tanh(theta[0])
        r1 = -th / (1.0 + th ** 2)
        return jnp.concatenate([jnp.ones(1), r1[None], jnp.zeros(max(q - 2, 0))])[:q]
    if kind == "ma2":
        t1, t2 = jnp.tanh(theta[0]), jnp.tanh(theta[1])
        den = 1.0 + t1 ** 2 + t2 ** 2
        r1 = -t1 * (1.0 - t2) / den
        r2 = -t2 / den
        return jnp.concatenate([jnp.ones(1), r1[None], r2[None],
                                jnp.zeros(max(q - 3, 0))])[:q]
    if kind == "arma":
        # ACF d'un ARMA(1,1) : rho_1 = (1+phi*th)(phi+th)/(1+2*phi*th+th^2),
        # puis rho_k = phi * rho_{k-1}. C'est la forme STANDARD, valide sur tout
        # le carre |phi|<1, |th|<1 (stationnaire et inversible).
        # La formule imprimee a l'annexe C du manuel, (th-ph)(1-th*ph)/
        # (1+th^2-2*th*ph), donne des suites NON definies positives sur une
        # bonne part de ce carre (mesure : 204 tirages sur 300, dont
        # rho_1 = -0.976 avec rho_2 = -0.944, impossible). Convention de signe
        # de theta possiblement opposee ; le modele ajuste est le meme, seul le
        # signe du parametre MA rapporte peut differer d'asreml.
        th, ph = jnp.tanh(theta[0]), jnp.tanh(theta[1])
        r1 = (1.0 + ph * th) * (ph + th) / (1.0 + 2.0 * ph * th + th ** 2)
        acf = jnp.concatenate([jnp.ones(1), r1[None]])
        for k in range(2, q):
            acf = jnp.concatenate([acf, (ph * acf[k - 1])[None]])
        return acf[:q]
    if kind == "corb":
        b = int(order)
        vals = jnp.tanh(theta[:b])
        return jnp.concatenate([jnp.ones(1), vals, jnp.zeros(max(q - 1 - b, 0))])[:q]
    raise ValueError("pas d'ACF pour la structure %s" % kind)


def _tril_off(t):
    """Indices du triangle inferieur STRICT, ligne par ligne."""
    ii, jj = [], []
    for i in range(t):
        for j in range(i):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)


def _toeplitz(acf, q):
    i = jnp.arange(q)
    return acf[jnp.abs(i[:, None] - i[None, :])]


# ==============================================================================
# Matrice de correlation
# ==============================================================================
def _mtrn_valeurs(theta_lv, opts):
    """(phi, nu, delta, alpha, lambda) de mtrn : les estimes viennent de theta,
    les autres de opts. phi, nu et delta sont POSITIFS (exp) ; alpha est un
    angle, donc libre."""
    o = opts or {}
    val, k = {}, 0
    for p in MTRN_PARAMS:
        estime = float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5
        if estime:
            t = theta_lv[k]; k += 1
            val[p] = t if p == "alpha" else jnp.exp(t)
        else:
            val[p] = jnp.asarray(float(o.get(p, MTRN_DEFAUT[p])))
    val["lambda"] = float(o.get("lambda", MTRN_DEFAUT["lambda"]))
    return val


def _distance_anisotrope(dx, dy, delta, alpha, lam):
    """Metrique de Haskard et al. (2007), telle qu'ASReml l'emploie.

        u =  dx cos(alpha) + dy sin(alpha)
        v = -dx sin(alpha) + dy cos(alpha)
        h = ( |sqrt(delta) u|^lambda + |v/sqrt(delta)|^lambda )^(1/lambda)

    delta AGIT EN RACINE, et en sens INVERSE sur les deux axes : la
    transformation preserve les aires, donc delta ne se confond pas avec la
    portee phi. Trois autres conventions plausibles (delta sur un seul axe, ou
    non racine) ont ete essayees et ECARTEES : elles decalent la logLik de 0.5 a
    4 points sur le meme ajustement (cf. docs/note_remlax.md). Verifie contre
    asreml a 3e-10 sur (delta=2, alpha=0) et (delta=2, alpha=0.6).
    """
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    sd = jnp.sqrt(delta)
    a, b = jnp.abs(sd * u), jnp.abs(v / sd)
    if abs(lam - 2.0) < 1e-12:
        return jnp.sqrt(a ** 2 + b ** 2)
    if abs(lam - 1.0) < 1e-12:
        return a + b
    return (a ** lam + b ** lam) ** (1.0 / lam)


def _own_corr(theta_lv, q, expr, coord=None, opts=None):
    """Structure DEFINIE PAR L'UTILISATEUR : expression arithmetique.

    POURQUOI PAS LA FONCTION R D'ASREML. own(obj, fun) d'asreml appelle une
    fonction R a chaque evaluation. Ici R et JAX vivent dans deux conteneurs
    distincts, et surtout il faudrait la DERIVEE : une fonction R opaque n'en a
    pas. L'expression, elle, traverse la frontiere comme une chaine et se
    differentie automatiquement.

    Variables disponibles : d (distance euclidienne), dx, dy (distances par
    axe), lag (|i-j|), I, J (identite, matrice de uns), p1..pk (parametres
    BRUTS, non contraints : c'est a l'expression d'appliquer exp/tanh si elle
    veut de la positivite ou une borne), et les fonctions usuelles + matern.

    NORMALISATION. Sauf opts["normalise"] = 0, on rend D^-1/2 C D^-1/2 : le
    contrat de ce module est une matrice de CORRELATION, et l'echelle est deja
    portee par Sigma. Sans cela la variance serait comptee deux fois.
    """
    if not expr:
        raise ValueError("own : aucune expression fournie.")
    # L'expression est ECRITE EN R, ou `^` est la puissance. En Python c'est le
    # OU EXCLUSIF, qui sur des flottants ne leve meme pas une erreur lisible :
    # "xor does not accept dtype float64". On traduit, plutot que d'exiger de
    # l'utilisateur qu'il ecrive du Python dans une formule R.
    expr = expr.replace("^", "**")
    i = jnp.arange(q)
    lag = jnp.abs(i[:, None] - i[None, :]).astype(jnp.float64)
    if coord is not None:
        x = jnp.asarray(coord)
        if x.ndim == 1:
            x = x[:, None]
        dx = x[:, None, 0] - x[None, :, 0]
        dy = (x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1 else jnp.zeros_like(dx)
    else:
        dx = i[:, None] - i[None, :]
        dx = dx.astype(jnp.float64)
        dy = jnp.zeros_like(dx)
    env = {"d": jnp.sqrt(dx ** 2 + dy ** 2), "dx": jnp.abs(dx), "dy": jnp.abs(dy),
           "lag": lag, "I": jnp.eye(q), "J": jnp.ones((q, q)), "q": float(q),
           "exp": jnp.exp, "log": jnp.log, "sqrt": jnp.sqrt, "abs": jnp.abs,
           "tanh": jnp.tanh, "sinh": jnp.sinh, "cosh": jnp.cosh, "sin": jnp.sin,
           "cos": jnp.cos, "minimum": jnp.minimum, "maximum": jnp.maximum,
           "where": jnp.where, "pi": float(np.pi), "matern": _matern}
    for k in range(len(theta_lv)):
        env["p%d" % (k + 1)] = theta_lv[k]
    C = eval(expr, {"__builtins__": {}}, env)          # noqa: S307
    C = 0.5 * (C + C.T)
    if float((opts or {}).get("normalise", 1.0)) > 0.5:
        s = jnp.sqrt(jnp.clip(jnp.diagonal(C), 1e-300, None))
        C = C / s[:, None] / s[None, :]
    return C


def level_corr(theta_lv, kind, q, order=0, coord=None, C_fixed=None, parts=None,
               dims=None, opts=None, expr=None):
    """Matrice de correlation C (q x q) de la structure."""
    if kind == "own":
        return _own_corr(theta_lv, q, expr, coord=coord, opts=opts)
    if kind == "id":
        return jnp.eye(q)
    if kind == "ar1ar1":
        if dims is None or len(dims) != 2:
            raise ValueError("ar1ar1 exige dims = (n_lignes, n_colonnes)")
        nr, nc = int(dims[0]), int(dims[1])
        Cr = _toeplitz(_acf(theta_lv[:1], "ar1", nr), nr)
        Cc = _toeplitz(_acf(theta_lv[1:2], "ar1", nc), nc)
        return jnp.kron(Cr, Cc)
    if kind == "fixed":
        return jnp.asarray(C_fixed)
    if kind == "corg":
        # Correlation GENERALE : toutes les correlations libres, diagonale unite.
        # Parametrisation par un facteur de Cholesky a LIGNES NORMEES : c'est la
        # seule facon d'obtenir a la fois une diagonale exactement unitaire et la
        # positivite, sans contrainte a gerer. Parametrer directement les phi_ij
        # produirait des matrices non definies positives des que omega depasse 3.
        w = int(order) if order else q
        ii, jj = _tril_off(w)
        L = jnp.eye(w).at[ii, jj].set(theta_lv[:len(ii)])
        L = L / jnp.sqrt(jnp.sum(L ** 2, axis=1, keepdims=True))
        return L @ L.T
    if kind in LEVEL_NEEDS_COORD:
        x = jnp.asarray(coord)
        if x.ndim == 1:
            x = x[:, None]
        dx = jnp.abs(x[:, None, 0] - x[None, :, 0])
        dy = (jnp.abs(x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1
              else jnp.zeros_like(dx))
        d_eu = jnp.sqrt(dx ** 2 + dy ** 2)
        # `ph` est calcule A LA DEMANDE. En tete de bloc, il lisait theta_lv[0]
        # avant meme de savoir de quelle structure il s'agit : un mtrn dont TOUS
        # les parametres sont fixes n'a aucun theta, et l'ajustement s'arretait
        # sur "index is out of bounds for axis 0 with size 0" — alors que c'est
        # precisement le cas le plus simple, celui d'une correlation imposee.
        def _ph():
            return jnp.clip(jnp.abs(jnp.tanh(theta_lv[0])), 1e-12, 1 - 1e-12)
        if kind == "exp":                      # 1D : |x_i - x_j|
            return _ph() ** d_eu
        if kind == "gau":                      # 1D : (x_i - x_j)^2
            return _ph() ** (d_eu ** 2)
        if kind == "iexp":                     # 2D isotrope, metrique CITY-BLOCK
            return _ph() ** (dx + dy)             # (et non euclidienne : cf. manuel)
        if kind == "igau":
            return _ph() ** (dx ** 2 + dy ** 2)
        if kind == "ieuc":
            return _ph() ** d_eu
        if kind == "aexp":                     # anisotrope : un phi par axe
            p2 = jnp.clip(jnp.abs(jnp.tanh(theta_lv[1])), 1e-12, 1 - 1e-12)
            return (_ph() ** dx) * (p2 ** dy)
        if kind == "agau":
            p2 = jnp.clip(jnp.abs(jnp.tanh(theta_lv[1])), 1e-12, 1 - 1e-12)
            return (_ph() ** (dx ** 2)) * (p2 ** (dy ** 2))
        if kind == "lvr":
            # "Variance lineaire" : tente tronquee, correlation qui decroit
            # LINEAIREMENT jusqu'a s'annuler a la portee phi. Definie positive
            # en une dimension seulement.
            return jnp.clip(1.0 - dx / jnp.exp(theta_lv[0]), 0.0, None)
        if kind in ("sph", "cir"):
            # Modeles a PORTEE : la correlation s'annule exactement au-dela de
            # phi. Le parametre est une distance, pas une correlation : il est
            # donc pris en exp(theta) (positif, non borne).
            #
            # GRADIENT AU-DELA DE LA PORTEE. Une premiere ecriture ecretait
            # t = min(d/phi, 1) puis appliquait la formule. Pour cir, la formule
            # contient sqrt(1 - t^2) et arcsin(t), dont la derivee est INFINIE
            # en t = 1 ; la regle de derivation la multiplie par la derivee de
            # l'ecretage, qui vaut 0 : 0 x inf = NaN. Des qu'une paire depasse
            # la portee (toujours, en pratique), le gradient entier etait NaN,
            # L-BFGS-B s'arretait a l'iteration 0 et le solveur rendait le point
            # de depart comme optimum, sans un mot (test asreml3 B7 du
            # 2026-09-28 : n_iter = 0, -2logL a 6,4 de l'optimum). On ecrit donc
            # la formule sur un t SUR (0 hors portee), puis on selectionne :
            # jnp.where dans les deux sens, la seule ecriture dont le gradient
            # est fini de chaque cote. sph (polynome) n'avait pas ce defaut ;
            # il suit la meme ecriture pour n'avoir qu'une convention.
            rg = jnp.exp(theta_lv[0])
            dedans = d_eu < rg
            t = jnp.where(dedans, d_eu / rg, 0.0)
            if kind == "sph":
                f = 1.0 - 1.5 * t + 0.5 * t ** 3
            else:
                f = 1.0 - (2.0 / jnp.pi) * (t * jnp.sqrt(1.0 - t ** 2) + jnp.arcsin(t))
            return jnp.where(dedans, f, 0.0)
        if kind == "mtrn":
            # Matern ANISOTROPE complet (Haskard et al. 2007), tel qu'asreml
            # l'implemente : portee phi, forme nu, rapport d'anisotropie delta,
            # angle alpha, metrique lambda (1 = city-block, 2 = euclidienne).
            # nu QUELCONQUE : K_nu vient de bessel.py (quadrature), pas d'une
            # interpolation entre demi-entiers.
            v = _mtrn_valeurs(theta_lv, opts)
            sx = x[:, None, 0] - x[None, :, 0]
            sy = (x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1 else jnp.zeros_like(sx)
            h = _distance_anisotrope(sx, sy, v["delta"], v["alpha"], v["lambda"])
            return _matern(h / v["phi"], v["nu"])
    if kind == "sep":
        # Produit separable : C = C_1 (x) C_2 (x) ... ; l'ordre des niveaux est
        # celui d'un kron, le PREMIER facteur variant le plus LENTEMENT.
        # `parts` porte (famille, DIMENSION du facteur), exactement comme dans
        # level_chol. Une version anterieure lisait le second element comme un
        # ORDRE et passait q = None a chaque facteur : jnp.eye(None) levait une
        # TypeError des le premier appel, donc une residuelle a structure `sep`
        # (_section_V passe par level_corr) etait inatteignable. Trouve par le
        # test test_levels_catalogue.py::test_sep_level_corr_egale_kron.
        C, o = None, 0
        for (k, qi) in (parts or []):
            p = n_level_params(k)
            C_i = level_corr(theta_lv[o:o + p], k, int(qi))
            C = C_i if C is None else jnp.kron(C, C_i)
            o += p
        return C
    return _toeplitz(_acf(theta_lv, kind, q, order), q)


def level_chol(theta_lv, kind, q, dims=None, LK_fixed=None, order=0,
               coord=None, parts=None, jitter=1e-10, opts=None, expr=None):
    """Facteur L tel que L L' = C. None signifie identite (aucun produit a faire).

    Cas particuliers gardes en forme close :
      - `id` -> None, pour eviter un produit par l'identite ;
      - `fixed` -> le facteur deja calcule cote R (pas de refactorisation) ;
      - `ar1` -> forme close, exacte jusqu'a |phi| proche de 1.
    Les autres passent par une Cholesky numerique, avec un jitter proportionnel
    a la taille : une correlation de bande (ma1, corb) peut etre singuliere aux
    bords de son domaine.
    """
    if kind == "id":
        return None
    if kind == "fixed":
        return jnp.asarray(LK_fixed)
    if kind == "ar1":
        return ar1_chol(jnp.tanh(theta_lv[0]), q)
    if kind == "ar1ar1":
        nr, nc = dims
        return jnp.kron(ar1_chol(jnp.tanh(theta_lv[0]), nr),
                        ar1_chol(jnp.tanh(theta_lv[1]), nc))
    if kind == "sep":
        # Produit separable de facteurs : L = L_1 (x) L_2, exact et moins cher
        # qu'une Cholesky sur le produit.
        L, o = None, 0
        for (k, qi) in (parts or []):
            p = n_level_params(k)
            Li = level_chol(theta_lv[o:o + p], k, qi)
            Li = jnp.eye(qi) if Li is None else Li
            L = Li if L is None else jnp.kron(L, Li)
            o += p
        return L
    C = level_corr(theta_lv, kind, q, order=order, coord=coord, opts=opts,
                   expr=expr, dims=dims)
    return jnp.linalg.cholesky(C + jitter * jnp.eye(q))


def ar1_chol(rho, n):
    """L tel que L L' = correlation AR(1), en forme close.

        L[i,0] = rho^i ;  L[i,j] = rho^(i-j) sqrt(1-rho^2) pour 1 <= j <= i.

    Exact, derivable, et sans matrice n x n a factoriser.
    """
    i = jnp.arange(n)
    d = i[:, None] - i[None, :]
    s = jnp.sqrt(jnp.clip(1.0 - rho ** 2, 1e-300, None))
    L = jnp.where(d >= 0, rho ** jnp.abs(d) * s, 0.0)
    return L.at[:, 0].set(rho ** i)


def level_params_report(theta_lv, kind, order=0, opts=None, q=None, parts=None):
    """Parametres sur l'echelle d'asreml (phi, theta), pour les sorties.

    REGLE : cette fonction rend la valeur EFFECTIVEMENT UTILISEE dans C, jamais
    le theta intermediaire. Deux familles ne se contentent donc pas de tanh, et
    l'oubli faisait rapporter un nombre absent du modele ajuste :

      cor            la correlation uniforme n'est definie positive que sur
                     (-1/(q-1), 1) ; _acf envoie tanh(theta) sur cet intervalle.
                     Rapporter tanh(theta) donnait une valeur hors du modele
                     des que q etait petit (a q = 3, tanh = -0,9 devient -0,45).
                     `q` est donc REQUIS ici ; sans lui on refuse plutot que de
                     rapporter la mauvaise echelle.
      familles       exp, gau, iexp, igau, ieuc, aexp, agau elevent |tanh(theta)|
      metriques      a une distance : le SIGNE de theta n'est pas identifie
                     (theta et -theta donnent le meme C). On rapporte donc la
                     valeur absolue, et le champ `signe_non_identifie` le dit.
    """
    # PRODUIT SEPARABLE : rapporter CHAQUE facteur, sous un nom qui dit sa
    # position. Sans cela un champ id (x) ar1 (x) ar1 ne rapportait rien du tout
    # — `rho` revenait vide — et les correlations n'etaient lisibles qu'en
    # decodant theta a la main. Le nom porte l'indice du facteur parce que deux
    # facteurs de la meme famille sont la norme ici, pas l'exception.
    if kind == "sep":
        out, o = {}, 0
        for idx, (k, qi) in enumerate(parts or [], start=1):
            pk = n_level_params(k)
            if pk:
                sub = level_params_report(theta_lv[o:o + pk], k, order, q=qi)
                for nm, v in (sub or {}).items():
                    out["%s_%d_%s" % (k, idx, nm)] = v
            o += pk
        return out

    th = np.asarray(theta_lv, dtype=float)
    absta = lambda j: float(np.clip(np.abs(np.tanh(th[j])), 1e-12, 1 - 1e-12))  # noqa: E731
    if kind == "ar1ar1":
        return {"phi": [float(np.tanh(th[0])), float(np.tanh(th[1]))]}
    if kind == "cor":
        if q is None:
            raise ValueError(
                "level_params_report(kind='cor') exige q : la correlation uniforme "
                "est remise a l'echelle sur (-1/(q-1), 1), donc tanh(theta) n'est "
                "pas la valeur du modele.")
        lo = -1.0 / max(int(q) - 1, 1)
        return {"phi": [float(lo + (np.tanh(th[0]) + 1.0) * 0.5 * (1.0 - lo))],
                "borne_inf": [float(lo)]}
    if kind in ("ar1", "sar", "ma1"):
        return {"phi": [float(np.tanh(th[0]))]}
    if kind in ("exp", "gau"):
        return {"phi": [absta(0)], "signe_non_identifie": [1.0]}
    if kind in ("ar2", "ar3"):
        p = 2 if kind == "ar2" else 3
        return {"phi": [float(v) for v in np.asarray(_levinson_phi(jnp.tanh(jnp.asarray(th[:p]))))],
                "pacf": [float(np.tanh(v)) for v in th[:p]]}
    if kind in ("ma2", "arma"):
        return {"phi": [float(np.tanh(v)) for v in th[:2]]}
    if kind in ("aexp", "agau"):
        return {"phi": [absta(0), absta(1)], "signe_non_identifie": [1.0]}
    if kind in ("iexp", "igau", "ieuc"):
        return {"phi": [absta(0)], "signe_non_identifie": [1.0]}
    if kind in ("sph", "cir", "lvr"):
        return {"portee": [float(np.exp(th[0]))]}
    if kind == "mtrn":
        o = opts or {}
        out, k = {}, 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5:
                out[p] = [float(th[k]) if p == "alpha" else float(np.exp(th[k]))]
                k += 1
        return out
    if kind == "own":
        return {"own": [float(v) for v in th]}
    if kind == "corg":
        return {}
    if kind == "corb":
        return {"phi": [float(np.tanh(v)) for v in th[:int(order)]]}
    return {}
