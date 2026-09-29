"""Inference sur un ajustement REML : vpredict, Wald, contraintes.

Trois choses distinctes, souvent confondues :

  vpredict   fonctions des COMPOSANTES DE VARIANCE (heritabilite, correlation
             genetique, part de variance) et leurs erreurs-types. Delta method
             sur le Hessien de -2logL : var(g) = J' (2 H^-1) J, ou le facteur 2
             vient de ce que H est le Hessien de -2logL et non de -logL.

  wald       tests sur les EFFETS FIXES. var(beta) = (X' V^-1 X)^-1 a V fixe ;
             les tests conditionnels d'asreml testent chaque terme sachant tous
             les autres, ce qui est le F de type III.

  contraintes  chaque parametre peut etre libre (U), positif (P) ou fixe (F),
             comme les codes d'asreml. "Positif" est ici automatique : on
             parametre des log-ecarts-types. "Fixe" retire le parametre du
             vecteur optimise, ce qui change les degres de liberte et donc
             tout LRT qui suit — d'ou la trace explicite.

LIMITE ASSUMEE : les erreurs-types de vpredict sont conditionnelles au fait que
le Hessien soit inversible sur le sous-espace libre. Un parametre a une borne
active n'en a pas, et une direction de courbure nulle rend la delta method
muette plutot que fausse : on rend NaN, jamais un chiffre invente.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


def component_names(terms, res):
    """Noms des composantes, dans l'ordre ou vpredict les numerote (V1, V2, ...).

    L'ordre est celui des termes puis de la residuelle, et pour une structure
    matricielle celui du triangle inferieur ligne par ligne. Il est FIXE et
    documente : une sortie qui renumerote silencieusement rendrait toute
    expression vpredict fausse a la prochaine execution.
    """
    try:
        from .structures import n_params
    except ImportError:
        from structures import n_params
    noms = []
    for tm in terms:
        t = tm["t"]
        if t == 1:
            noms.append("%s" % tm["name"])
        else:
            for i in range(t):
                for j in range(i + 1):
                    noms.append("%s[%d,%d]" % (tm["name"], i + 1, j + 1))
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        nl = _n_niv(tm)
        for k in range(nl):
            noms.append("%s!%s%s" % (tm["name"], kind, "" if nl == 1 else str(k + 1)))
    try:
        from .model import res_sections
    except ImportError:
        from model import res_sections
    secs = res_sections(res, 0)
    for sec in secs:
        pre = sec.get("name", "residuelle") if len(secs) > 1 else "residuelle"
        t = sec["t"]
        if t == 1:
            noms.append(pre)
        else:
            for i in range(t):
                for j in range(i + 1):
                    noms.append("%s[%d,%d]" % (pre, i + 1, j + 1))
        nl = _n_niv(sec)
        kind = sec.get("lvl", "id")
        for k in range(nl):
            noms.append("%s!%s%s" % (pre, kind, "" if nl == 1 else str(k + 1)))
    return noms


def _n_niv(tm):
    """Parametres de la structure ENTRE NIVEAUX d'un terme ou d'une section."""
    try:
        from .levels import n_level_params
    except ImportError:
        from levels import n_level_params
    kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    return n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                          opts=tm.get("lvl_opts"))


# Structures dont TOUS les parametres sont des correlations (tanh), et celles
# dont le premier est une PORTEE (exp). vpredict doit rendre l'echelle
# naturelle : une expression comme "V3" designe la correlation phi, pas le
# theta interne qui la produit.
# Correlations rendues DIRECTEMENT par tanh(theta).
_NIV_TANH = ("ar1", "ar2", "ar3", "sar", "ma1", "ma2", "arma", "corb", "ar1ar1")
# Familles metriques : C utilise |tanh(theta)| eleve a une distance, donc le
# SIGNE de theta n'est pas identifie. Rendre tanh(theta) ici affichait un signe
# arbitraire, et une expression vpredict portant sur ce parametre differentiait
# la mauvaise fonction du cote ou tanh est negatif.
_NIV_ABS_TANH = ("exp", "gau", "iexp", "igau", "ieuc", "aexp", "agau")
# Portees : le parametre est une DISTANCE, prise en exp(theta).
_NIV_PORTEE = ("sph", "cir", "lvr", "ilv")


def _niv_naturel(th_niv, tm):
    """Parametres de niveaux sur leur echelle NATURELLE, en JAX (differentiable)."""
    try:
        from .levels import MTRN_PARAMS
    except ImportError:
        from levels import MTRN_PARAMS
    kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    nl = _n_niv(tm)
    if nl == 0:
        return []
    if kind in _NIV_TANH:
        return [jnp.tanh(th_niv[j]) for j in range(nl)]
    if kind in _NIV_ABS_TANH:
        return [jnp.abs(jnp.tanh(th_niv[j])) for j in range(nl)]
    if kind == "cor":
        # Meme remise a l'echelle que _acf : (-1,1) -> (-1/(q-1), 1).
        q = int(tm.get("q") or tm.get("n_unit") or 1)
        lo = -1.0 / max(q - 1, 1)
        return [lo + (jnp.tanh(th_niv[0]) + 1.0) * 0.5 * (1.0 - lo)]
    if kind in _NIV_PORTEE:
        return [jnp.exp(th_niv[j]) for j in range(nl)]
    if kind == "mtrn":
        o, out, k = tm.get("lvl_opts") or {}, [], 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5:
                out.append(th_niv[k] if p == "alpha" else jnp.exp(th_niv[k]))
                k += 1
        return out
    return [th_niv[j] for j in range(nl)]      # corg, own : parametres bruts


def components_from_theta(theta, terms, res):
    """Vecteur des composantes (variances et covariances) a partir de theta.

    C'est la fonction que la delta method differentie. Elle doit donc etre
    ecrite en JAX de bout en bout, et rendre les composantes DANS L'ORDRE de
    component_names().
    """
    try:
        from .structures import build_sigma, n_params
        from .model import split_theta
        from .levels import level_params_report
    except ImportError:
        from structures import build_sigma, n_params
        from model import split_theta
        from levels import level_params_report
    th_terms, th_res, _ = split_theta(theta, terms, res)
    out = []
    for k, tm in enumerate(terms):
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        S = build_sigma(th_terms[k][:ns], tm["struct"], tm["t"], tm["rank"])
        t = tm["t"]
        if t == 1:
            out.append(S[0, 0])
        else:
            for i in range(t):
                for j in range(i + 1):
                    out.append(S[i, j])
        out.extend(_niv_naturel(th_terms[k][ns:], tm))
    try:
        from .model import res_sections, sec_n_params
    except ImportError:
        from model import res_sections, sec_n_params
    o = 0
    for sec in res_sections(res, 0):
        p_s = sec_n_params(sec)
        th_s = th_res[o:o + p_s]
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        R = build_sigma(th_s[:ns], sec["struct"], sec["t"], sec["rank"]) \
            if sec["struct"] != "iid" else jnp.exp(2 * th_s[0]) * jnp.eye(sec["t"])
        t = sec["t"]
        if t == 1:
            out.append(R[0, 0])
        else:
            for i in range(t):
                for j in range(i + 1):
                    out.append(R[i, j])
        out.extend(_niv_naturel(th_s[ns:], sec))
        o += p_s
    return jnp.stack(out)


VPREDICT_FONCTIONS = ("sqrt", "log", "exp", "abs")


def check_vpredict(expressions, terms, res):
    """Verifie les expressions vpredict AVANT l'ajustement.

    Sans ce controle, une expression citant une composante absente (V3 dans un
    modele qui n'en a que deux) n'echouait qu'apres l'optimisation, par un
    NameError Python qui ne disait pas quelles composantes existaient.
    Leve ValueError avec la liste V1 = nom, V2 = nom, ...
    """
    import ast
    noms = component_names(terms, res)
    permis = {"V%d" % (i + 1) for i in range(len(noms))} | set(VPREDICT_FONCTIONS)
    liste = ", ".join("V%d = %s" % (i + 1, n) for i, n in enumerate(noms))
    for nom, expr in expressions:
        try:
            arbre = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError("vpredict '%s' : expression illisible \"%s\" (%s). "
                             "Composantes du modele : %s." % (nom, expr, e.msg, liste))
        inconnus = sorted({n.id for n in ast.walk(arbre) if isinstance(n, ast.Name)} - permis)
        if inconnus:
            raise ValueError("vpredict '%s' : %s inconnu(s) dans \"%s\". Ce modele a %d "
                             "composante(s) : %s. Fonctions admises : %s."
                             % (nom, ", ".join(inconnus), expr, len(noms), liste,
                                ", ".join(VPREDICT_FONCTIONS)))


def vpredict(theta, H, terms, res, expressions, free=None):
    """Fonctions des composantes et leurs erreurs-types.

    expressions : liste de (nom, chaine) ou la chaine utilise V1, V2, ... et les
    operateurs Python usuels. Exemple : ("h2", "V1/(V1+V2)").

    H est le Hessien de -2logL. La covariance asymptotique de theta est donc
    2 H^-1, et var(g) = J' (2 H^-1) J avec J = dg/dtheta.
    """
    theta = np.asarray(theta, dtype=float)
    comp_fn = lambda th: components_from_theta(th, terms, res)  # noqa: E731
    comps = np.asarray(comp_fn(jnp.asarray(theta)))
    noms = component_names(terms, res)

    if free is None:
        free = np.ones(len(theta), dtype=bool)
    free = np.asarray(free, dtype=bool)
    Vth = np.full((len(theta), len(theta)), np.nan)
    if H is not None and free.sum() > 0:
        Hf = np.asarray(H)[np.ix_(free, free)]
        try:
            w, U = np.linalg.eigh(Hf)
            tol = 1e-8 * max(abs(w).max(), 1.0)
            if np.all(w > tol):
                Vth_f = 2.0 * (U @ np.diag(1.0 / w) @ U.T)
                Vth = np.zeros((len(theta), len(theta)))
                Vth[np.ix_(free, free)] = Vth_f
        except np.linalg.LinAlgError:
            pass

    res_out = []
    for nom, expr in expressions:
        def g(th, expr=expr):
            c = comp_fn(th)
            env = {"V%d" % (i + 1): c[i] for i in range(c.shape[0])}
            env.update({"sqrt": jnp.sqrt, "log": jnp.log, "exp": jnp.exp, "abs": jnp.abs})
            return eval(expr, {"__builtins__": {}}, env)          # noqa: S307
        val = float(g(jnp.asarray(theta)))
        J = np.asarray(jax.grad(g)(jnp.asarray(theta)))
        se = float(np.sqrt(max(J @ Vth @ J, 0.0))) if np.all(np.isfinite(Vth)) else np.nan
        res_out.append(dict(nom=nom, expression=expr, valeur=val, se=se))
    return dict(composantes=[{"i": i + 1, "nom": noms[i], "valeur": float(comps[i])}
                             for i in range(len(noms))],
                predictions=res_out)


def wald(y, X, V, termes_fixes=None):
    """Tests de Wald sur les effets fixes, a V FIXE.

      var(beta) = (X' V^-1 X)^-1
      Pour un groupe de colonnes L : F = (L beta)' [L var(beta) L']^-1 (L beta) / rang

    C'est le test CONDITIONNEL d'asreml (chaque terme sachant les autres), pas
    l'incrementiel. Le denominateur de degres de liberte n'est pas ajuste
    (Kenward-Roger n'est pas implemente) : les p-valeurs sont donc celles d'un
    chi2/rang, valides asymptotiquement. C'est ecrit ici pour qu'on ne les prenne
    pas pour des F exacts a petit echantillon.
    """
    n, p = X.shape
    Vi = np.linalg.pinv(V)
    A = X.T @ Vi @ X
    Ai = np.linalg.pinv(A)
    beta = Ai @ (X.T @ Vi @ y)
    out = []
    groupes = termes_fixes or [("colonne %d" % (j + 1), [j]) for j in range(p)]
    for nom, cols in groupes:
        cols = list(cols)
        L = np.zeros((len(cols), p))
        for r, c in enumerate(cols):
            L[r, c] = 1.0
        Lb = L @ beta
        M = L @ Ai @ L.T
        rg = np.linalg.matrix_rank(M)
        if rg == 0:
            out.append(dict(terme=nom, ddl=0, F=np.nan, p=np.nan)); continue
        stat = float(Lb @ np.linalg.pinv(M) @ Lb)
        from scipy.stats import chi2
        out.append(dict(terme=nom, ddl=int(rg), F=stat / rg,
                        chi2=stat, p=float(chi2.sf(stat, rg))))
    return dict(beta=[float(b) for b in beta],
                se_beta=[float(np.sqrt(max(v, 0))) for v in np.diag(Ai)],
                tests=out)


# ==============================================================================
# Kenward-Roger
# ==============================================================================
def _info_reml(D, P_mat):
    """Information REML ESPEREE : I_ij = 1/2 tr(P dV_i P dV_j), P = V^-1 - V^-1X Phi X'V^-1.

    C'est CETTE matrice, et non le Hessien observe, que Kenward et Roger
    utilisent. Les deux coincident asymptotiquement, mais pas a l'echantillon
    fini ou l'ajustement sert : sur un essai a 27 observations, le Hessien
    donnait 21.83 ddl de denominateur contre 21.769 pour l'esperee, qui est la
    valeur de pbkrtest. Un ecart petit, mais il n'y a aucune raison de
    l'accepter quand la bonne quantite se calcule aussi bien.
    """
    k = len(D)
    M = [P_mat @ Di for Di in D]
    I = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            I[i, j] = I[j, i] = 0.5 * float(np.sum(M[i] * M[j].T))
    return I


def kenward_roger(theta, terms, res, y, X, H=None, termes_fixes=None):
    """Ajustement de Kenward & Roger (1997) : variance de beta et ddl du denominateur.

    CE QUE CORRIGE LA METHODE. Le test de Wald ordinaire traite V comme CONNUE.
    Elle est estimee : la variance de beta_chapeau est donc sous-estimee et la
    statistique trop grande. Sur un dispositif de taille reelle l'effet est
    negligeable ; sur un essai a peu de blocs ou de sites — le cas ou l'on tient
    justement a un test — il ne l'est pas.

    DEUX CORRECTIONS, distinctes :

      1. Phi_A, covariance AJUSTEE de beta
             Phi_A = Phi + 2 Phi [ sum_ij W_ij (Q_ij - P_i Phi P_j) ] Phi
         avec Phi = (X'V^-1X)^-1, U = V^-1 X, P_i = U' dV_i U,
         Q_ij = (dV_i U)' V^-1 (dV_j U), et W l'inverse de l'information REML
         ESPEREE.

      2. un facteur d'echelle lambda et des ddl de denominateur m, obtenus en
         appariant les deux premiers moments de lambda F* a ceux d'un F(l, m).
         Les moments font intervenir dPhi/dtheta_i = Phi P_i Phi, PAS P_i :
         contracter Theta contre P_i au lieu de dPhi donne des A1, A2 d'ordre
         n^3 au lieu de O(1), et des ddl NEGATIFS.

    VALIDE CONTRE pbkrtest::KRmodcomp (implementation de reference) sur un
    dispositif en blocs desequilibre : ddl 21.76897 contre 21.76896, facteur
    d'echelle 0.9998304 contre 0.9998304, erreurs-types ajustees identiques a
    1e-7.

    LE TERME DU SECOND ORDRE EST OMIS, ET CE N'EST PAS UN OUBLI. La formule
    complete porte aussi -R_ij/4 avec R_ij = U' d2V/dsigma_i dsigma_j U. Il
    s'annule des que V est LINEAIRE dans les parametres de variance, ce qui est
    le cas de toutes les structures d'ici sauf les parametres de CORRELATION
    entre niveaux (le phi d'un AR1, une portee). C'est aussi ce que font
    pbkrtest et SAS. On l'a essaye : calcule dans la parametrisation INTERNE
    (log-ecarts-types, facteurs de Cholesky), il n'est pas invariant — sur un
    dispositif en blocs, le meme modele donnait Phi_A = 0.4375 en variances
    contre 0.3465 en log-ecarts-types, 21 % d'ecart, alors que la partie du
    premier ordre coincide a 2e-16. Ce terme corrige le BIAIS DU SECOND ORDRE
    de l'estimateur DANS LA PARAMETRISATION CHOISIE et n'a de sens qu'exprime
    dans les composantes naturelles. `second_ordre_omis` dit si le modele
    contient des parametres pour lesquels l'omission n'est pas exacte.

    COUT : les derivees premieres de V sont formees en ENTIER (n x n), une par
    parametre de variance, puis appariees deux a deux pour l'information. C'est
    O(p_theta^2 n^2) et c'est le prix de la methode, pas un choix.
    """
    try:
        from .model import assemble_V, dense_Z
    except ImportError:
        from model import assemble_V, dense_Z
    theta = np.asarray(theta, dtype=float)
    n, p = X.shape
    k = len(theta)
    Zs = [dense_Z(t, n) for t in terms]

    def V_de(th):
        return assemble_V(th, terms, Zs, res, n)

    V = np.asarray(V_de(jnp.asarray(theta)))
    try:
        Vi = np.linalg.solve(V, np.eye(n))
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    U = Vi @ X                                    # n x p
    Phi = np.linalg.pinv(X.T @ U)                 # (X'V^-1X)^-1
    P_mat = Vi - U @ Phi @ U.T                    # projecteur REML

    D = []
    for i in range(k):
        e = np.zeros(k); e[i] = 1.0
        _, dV = jax.jvp(V_de, (jnp.asarray(theta),), (jnp.asarray(e),))
        D.append(np.asarray(dV))
    I = _info_reml(D, P_mat)
    w, Qv = np.linalg.eigh(0.5 * (I + I.T))
    tol = 1e-10 * max(abs(w).max(), 1.0)
    if not np.all(w > tol):
        return dict(disponible=False,
                    raison="information REML non definie positive (%d direction(s) "
                           "plate(s)) : la covariance des parametres de variance "
                           "n'existe pas, donc l'ajustement de Kenward-Roger non plus."
                           % int((w <= tol).sum()))
    W = Qv @ np.diag(1.0 / w) @ Qv.T

    GU = [Di @ U for Di in D]                     # dV_i U
    Pt = [U.T @ g for g in GU]                    # X'V^-1 dV_i V^-1 X
    dPhi = [Phi @ Pi @ Phi for Pi in Pt]          # dPhi/dtheta_i
    S = np.zeros((p, p))
    for i in range(k):
        Vig = Vi @ GU[i]
        for j in range(k):
            S += W[i, j] * (GU[j].T @ Vig - Pt[i] @ Phi @ Pt[j])
    Phi_A = Phi + 2.0 * Phi @ S @ Phi

    beta = Phi @ (X.T @ Vi @ y)
    ddl_resid = float(n - np.linalg.matrix_rank(X))
    from scipy.stats import f as f_dist
    groupes = termes_fixes or [("colonne %d" % (j + 1), [j]) for j in range(p)]
    tests = []
    for nom, cols in groupes:
        cols = list(cols)
        L = np.zeros((p, len(cols)))
        for r, c in enumerate(cols):
            L[c, r] = 1.0
        l = int(np.linalg.matrix_rank(L))
        if l == 0:
            continue
        Lb = L.T @ beta
        Fstar = float(Lb @ np.linalg.pinv(L.T @ Phi_A @ L) @ Lb) / l
        Th = L @ np.linalg.pinv(L.T @ Phi @ L) @ L.T
        tD = np.array([np.trace(Th @ dPhi[i]) for i in range(k)])
        A1 = float(tD @ W @ tD)
        TD = [Th @ dPhi[i] for i in range(k)]
        A2 = float(sum(W[i, j] * np.trace(TD[i] @ TD[j])
                       for i in range(k) for j in range(k)))
        # DISPOSITIF ORTHOGONAL : A1 et A2 s'annulent exactement (dV_i V^-1 X
        # reste dans l'espace de X, il n'y a rien a corriger). g et rho seraient
        # alors des 0/0 et m sortirait NEGATIF. On retombe explicitement sur le
        # test non ajuste et les ddl residuels, qui est la bonne reponse.
        if abs(A1) < 1e-12 and abs(A2) < 1e-12:
            tests.append(dict(terme=nom, ddl=l, denDF=ddl_resid, F=float(Fstar),
                              F_brut=float(Fstar), echelle=1.0,
                              p=float(f_dist.sf(Fstar, l, ddl_resid)),
                              note="dispositif orthogonal : aucun ajustement"))
            continue
        B = (A1 + 6.0 * A2) / (2.0 * l)
        g = ((l + 1) * A1 - (l + 4) * A2) / ((l + 2) * A2)
        den = 3.0 * l + 2.0 * (1.0 - g)
        c1, c2, c3 = g / den, (l - g) / den, (l + 2.0 - g) / den
        Etoile = 1.0 / (1.0 - A2 / l)
        Vetoile = (2.0 / l) * (1.0 + c1 * B) / ((1.0 - c2 * B) ** 2 * (1.0 - c3 * B))
        rho = Vetoile / (2.0 * Etoile ** 2)
        m_ddl = 4.0 + (l + 2.0) / (l * rho - 1.0)
        lam = m_ddl / (Etoile * (m_ddl - 2.0)) if m_ddl > 2.0 else 1.0
        note = None
        if not np.isfinite(m_ddl) or m_ddl <= 2.0:
            # L'appariement des moments est sorti de son domaine : le dire, et
            # rendre le test non ajuste plutot qu'un p issu d'un F(l, m<=2).
            m_ddl, lam = ddl_resid, 1.0
            note = "appariement des moments hors domaine : test NON ajuste"
        FKR = lam * Fstar
        d = dict(terme=nom, ddl=l, denDF=float(m_ddl), F=float(FKR),
                 F_brut=float(Fstar), echelle=float(lam),
                 p=float(f_dist.sf(FKR, l, m_ddl)))
        if note:
            d["note"] = note
        tests.append(d)
    niv = sum(_n_niv(t) for t in terms)
    try:
        from .model import res_sections
        niv += sum(_n_niv(s) for s in res_sections(res, n))
    except ImportError:
        pass
    return dict(disponible=True, tests=tests,
                se_beta=[float(np.sqrt(max(v, 0))) for v in np.diag(Phi_A)],
                se_beta_brut=[float(np.sqrt(max(v, 0))) for v in np.diag(Phi)],
                vbeta_kr=Phi_A, beta=[float(b) for b in beta],
                second_ordre_omis=bool(niv > 0))


# ==============================================================================
# predict
# ==============================================================================
def predict(theta, terms, res, y, X, L, M=None, vbeta=None):
    """Predictions p = L' beta (+ M' u) et leur covariance d'ERREUR DE PREDICTION.

    L : l x p, une ligne par prediction, deja MOYENNEE cote R sur les facteurs
    absents du `classify`. Toute la semantique de predict vit dans la
    construction de L ; il ne reste ici que de l'algebre.

    M : dict nom_de_terme -> l x (t*q), la part ALEATOIRE de chaque prediction.
    Absent, la prediction ne porte que les effets fixes — ce qui est juste quand
    le terme aleatoire est moyenne, et FAUX quand on demande la moyenne d'un
    genotype : celle-la contient son BLUP.

    VARIANCE. Pour p = L'beta + M'u ce n'est pas var(p_chapeau) mais
    var(p_chapeau - p), l'erreur de PREDICTION :

        L' Phi L  -  2 sym( L' Phi X'V^-1 Z G M )  +  M'G M - M'G Z' P Z G M

    avec Phi = (X'V^-1X)^-1 et P = V^-1 - V^-1 X Phi X'V^-1. Les trois termes
    comptent : ne garder que le premier sous-estime l'erreur d'autant plus que
    le terme aleatoire porte de signal, et le deuxieme est NEGATIF — la
    correlation entre beta_chapeau et u_chapeau reduit l'erreur du total.

    Aucun G^-1 n'est forme, et aucun Kronecker : G = Sigma (x) K n'intervient
    que par son ACTION sur M, soit Sigma M_mat K'.
    """
    try:
        from .model import assemble_V, dense_Z, split_theta
        from .structures import build_sigma, n_params
        from .levels import level_chol
    except ImportError:
        from model import assemble_V, dense_Z, split_theta
        from structures import build_sigma, n_params
        from levels import level_chol
    theta = np.asarray(theta, dtype=float)
    n, p = X.shape
    L = np.atleast_2d(np.asarray(L, dtype=float))
    if L.shape[1] != p:
        raise ValueError("predict : L a %d colonnes pour %d effets fixes."
                         % (L.shape[1], p))
    Zs = [dense_Z(t, n) for t in terms]
    V = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    try:
        Vi = np.linalg.solve(V, np.eye(n))
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    U = Vi @ X
    Phi = np.linalg.pinv(X.T @ U) if vbeta is None else np.asarray(vbeta)
    beta = Phi @ (X.T @ Vi @ y)
    Pm = Vi - U @ Phi @ U.T
    Py = Pm @ y
    val = L @ beta
    cov = L @ Phi @ L.T
    th_terms, _, _ = split_theta(jnp.asarray(theta), terms, res)
    for k, tm in enumerate(terms):
        nom = tm["name"]
        if not M or M.get(nom) is None:
            continue
        Mk = np.atleast_2d(np.asarray(M[nom], dtype=float))       # l x (t*q)
        t_, q_ = tm["t"], tm["q"]
        if Mk.shape != (L.shape[0], t_ * q_):
            raise ValueError("predict : M['%s'] est %s, attendu (%d, %d)."
                             % (nom, Mk.shape, L.shape[0], t_ * q_))
        ns = n_params(tm["struct"], t_, tm["rank"])
        Sig = np.asarray(build_sigma(th_terms[k][:ns], tm["struct"], t_, tm["rank"]))
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        LK = (np.asarray(tm["LK"]) if kind == "fixed" and tm.get("LK") is not None
              else level_chol(th_terms[k][ns:], kind, q_, dims=tm.get("dims"),
                              LK_fixed=tm.get("LK"), order=tm.get("lvl_order", 0),
                              coord=tm.get("coord"), parts=tm.get("lvl_parts"),
                              opts=tm.get("lvl_opts"), expr=tm.get("lvl_expr")))
        Kq = np.eye(q_) if LK is None else np.asarray(LK) @ np.asarray(LK).T
        Mb = Mk.reshape(Mk.shape[0], t_, q_)
        GM = np.einsum("ab,lbq,qr->lar", Sig, Mb, Kq).reshape(Mk.shape[0], t_ * q_)
        Zd = np.asarray(Zs[k])
        ZGM = GM @ Zd.T                                            # l x n
        val = val + GM @ (Zd.T @ Py)                               # + M' u_chapeau
        A = L @ Phi @ (X.T @ Vi) @ ZGM.T                           # l x l
        cov = cov - A - A.T + Mk @ GM.T - ZGM @ Pm @ ZGM.T
    return dict(valeur=[float(v) for v in np.asarray(val).ravel()],
                se=[float(np.sqrt(max(v, 0))) for v in np.diag(cov)],
                cov=np.asarray(cov), beta=[float(b) for b in beta])
