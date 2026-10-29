"""AI-REML : optimiseur a information moyenne (Gilmour, Thompson et Cullis 1995).

CE QUE FAIT CE MODULE. Il remplace L-BFGS-B par un schema de type Newton dont la
courbure est la matrice d'information moyenne de -2 log L_R :

    AI_ij = y' P V_i P V_j P y = q_i' P q_j,     q_i = V_i P y,  V_i = dV/dtheta_i

et le pas, amorti a la Levenberg-Marquardt, vaut
    delta = -(AI + mu diag(AI))^-1 g,            g = gradient de -2 log L_R.

POURQUOI LES q_i SE CALCULENT SANS MATRICE n x n. Un terme contribue
Z (Sigma (x) K) Z' a V. Pour un vecteur v fixe, posons U = Z'v range en (t, q) :

    V_k v = Z vec(Sigma U K)     et    d(V_k v)/dtheta = Z vec(dSigma U K + Sigma U dK).

La derivee porte sur une matrice t x q, et JAX la calcule par jacfwd pour tous les
parametres du terme d'un coup. Le cout est un produit Z x (t q, p_k) par terme,
de l'ordre d'une evaluation de V, et la memoire supplementaire est n x p.
La residuelle `us` par unite se traite de meme en regroupant les observations par
unite : R v = (S applique a la matrice unite x caractere). Les autres residuelles
passent par la section dense, derivee parametre par parametre (lax.map) pour ne
jamais tenir plus d'une matrice n x n a la fois.

LE GRADIENT reste celui de make_objective (regle de derivation fermee sur V) :
l'AI ne sert qu'a la courbure. Le modele lineaire en V n'est pas suppose : avec la
parametrisation de Cholesky, l'AI est la partie de Gauss-Newton du Hessien.

LIMITES DECLAREES. Pas de modele structurel (lag) : son critere n'est pas une
vraisemblance REML en V seul.
"""
import time

# Decrement (unites de -2logL) sous lequel une boucle AI peut se declarer
# convergee sans polissage : le seuil du diagnostic final (fit.SEUIL_DECREMENT).
SEUIL_SUCCES = 1e-4
# demi-pas tentes sur la direction AI avant de passer a l'amortissement de
# Levenberg-Marquardt (ai_reml_var)
N_DEMI_PAS = 6
# SALVES L-BFGS-B INTERCALEES. Quand l'AI ne trouve plus de pas admissible, ou
# stagne avec un decrement encore grand, quelques iterations de L-BFGS-B
# l'eloignent de la region ou la matrice AI est presque singuliere, puis l'AI
# reprend (a la maniere des iterations EM qu'asreml intercale). MESURE : sur
# fa(2) a 5 caracteres, n = 1500 et 2000, trois ajustements du banc finissaient
# a 460-690 unites de logL sous l'optimum (AI bloquee, repli L-BFGS-B arrete
# tot), selon le BLAS et le nombre de fils.
SALVE_ITER = 20
SALVES_MAX = 5

import jax
import jax.numpy as jnp
import numpy as np

from .model import (split_theta, term_n_params, res_sections, sec_n_params,
                    support_groups, restrict_Z, assemble_V_groupes, prefold_fixed_K,
                    orthonormal_X, section_matvec)
from .structures import chol_sigma, n_params
from .levels import level_chol


def _sigma_et_K(theta_k, term):
    """Sigma (t x t) et K (q x q, ou None pour l'identite) d'un terme.

    Meme logique que model.term_factor, dont on reprend les cles : un terme a K
    fournie sans "lvl" est "fixed" ; un niveau parametre reconstruit L_K.
    """
    t, q = term["t"], term["q"]
    n_sig = n_params(term["struct"], t, term["rank"])
    Ls = chol_sigma(theta_k[:n_sig], term["struct"], t, term["rank"])
    S = Ls @ Ls.T
    kind = term.get("lvl") or ("fixed" if term.get("LK") is not None else "id")
    if kind == "fixed":
        LK = jnp.asarray(term["LK"])
    else:
        LK = level_chol(theta_k[n_sig:], kind, q, dims=term.get("dims"),
                        LK_fixed=term.get("LK"), order=term.get("lvl_order", 0),
                        coord=term.get("coord"), parts=term.get("lvl_parts"),
                        opts=term.get("lvl_opts"), expr=term.get("lvl_expr"))
    K = None if LK is None else LK @ LK.T
    return S, K


def make_ai(terms, res, y, X):
    """Rend ai(theta) -> matrice AI (p x p) de -2 log L_R, compilee.

    Les supports et incidences restreintes sont recalcules ici, comme dans
    make_objective : ils ne dependent pas de theta.
    """
    n = y.shape[0]
    # P ne depend que de l'espace engendre par X : Q (model.orthonormal_X) donne
    # la meme matrice AI, mieux conditionnee.
    Xq, ldRx = orthonormal_X(X)
    yj, Xj = jnp.asarray(y), jnp.asarray(Xq)
    groupes = support_groups(terms, n)
    Zr = [None] * len(terms)
    rows_of = [None] * len(terms)
    for rows, ks in groupes:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)
            rows_of[k] = jnp.asarray(np.asarray(rows, dtype=np.int64))
    # Parente fixe pliee dans l'incidence (model.prefold_fixed_K) : sans cela
    # K = L_K L_K' et U K seraient calcules a la compilation.
    terms, Zr = prefold_fixed_K(terms, Zr)
    secs = res_sections(res, n)

    def colonnes_terme(th_k, term, Z, rows, v):
        """d(V_k v)/dtheta_k : matrice n x p_k (lignes hors support nulles)."""
        t, q = term["t"], term["q"]
        U = (Z.T @ v[rows]).reshape(t, q)

        def g(th):
            S, K = _sigma_et_K(th, term)
            M = U if K is None else U @ K
            return (S @ M).reshape(t * q)

        J = jax.jacfwd(g)(th_k)                       # (t q, p_k)
        Qr = Z @ J                                    # (|rows|, p_k)
        return jnp.zeros((n, J.shape[1])).at[rows].set(Qr)

    def colonnes_section(th_s, sec, v):
        """d(R_s v)/dtheta_s : matrice n x p_s, sans former R_s quand la
        structure le permet (model.section_matvec)."""
        rows = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        vs = v[rows]
        J = jax.jacfwd(lambda th: section_matvec(th, sec, vs))(th_s)
        return jnp.zeros((n, J.shape[1])).at[rows].set(J)

    def ai(theta, Zr):
        V = assemble_V_groupes(theta, terms, groupes, Zr, res, n)
        L = jnp.linalg.cholesky(V)

        def sol(b):
            return jax.scipy.linalg.cho_solve((L, True), b)

        ViX = sol(Xj)
        A = Xj.T @ ViX
        Py = sol(yj) - ViX @ jnp.linalg.solve(A, Xj.T @ sol(yj))
        th_terms, th_res, _ = split_theta(theta, terms, res)
        blocs = [colonnes_terme(th_terms[k], tm, Zr[k], rows_of[k], Py)
                 for k, tm in enumerate(terms)]
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            blocs.append(colonnes_section(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(blocs, axis=1)            # (n, p)
        ViQ = sol(Q)
        PQ = ViQ - ViX @ jnp.linalg.solve(A, Xj.T @ ViQ)
        AI = Q.T @ PQ
        return 0.5 * (AI + AI.T)

    # incidences en ARGUMENT (voir model.make_objective) : jamais repliees
    _ai = jax.jit(ai)
    Zr_dev = Zr

    def out(theta):
        return _ai(theta, Zr_dev)
    # Outils exposes pour make_fga, qui les reutilise : une seule copie des
    # incidences restreintes et des colonnes de la matrice AI.
    out.outils = dict(groupes=groupes, Zr=Zr, rows_of=rows_of, secs=secs,
                      colonnes_terme=colonnes_terme, colonnes_section=colonnes_section,
                      yj=yj, Xj=Xj, n=n, terms=terms, ldRx=ldRx)
    return out


def _gram_rentable(term, rows):
    """Terme dont la contribution sigma2 Z Z' vaut d'etre PRECALCULEE : une
    composante, sans parente ni structure entre niveaux, variance iid, et au moins
    autant de niveaux que la moitie des lignes de son support. C'est le cas des
    IEE ; un terme spatial a peu de niveaux garde son facteur, moins cher."""
    lvl = term.get("lvl")
    return (term["t"] == 1 and term.get("LK") is None and lvl in (None, "", "id")
            and term["struct"] == "iid" and term["q"] >= 0.5 * len(rows))


def make_fga(terms, res, y, X):
    """Rend fga(theta) -> (-2logL, gradient, matrice AI), d'UNE factorisation de V.

    LE CALCUL. V est assemblee comme dans le moteur (supports, sections), sauf les
    termes ou sigma2 Z Z' est rentable a precalculer (_gram_rentable) : leur bloc
    vaut exp(2 theta) G, G = Z Z' fixe. Puis, a partir de la seule Cholesky de V :
        -2logL = (n-p) log 2pi + log|V| + log|X'V^-1X| + y'Py
        gradient : adjoint de V, Vbar = P - (Py)(Py)', propage a travers
                   l'assemblage par jax.vjp (pas de seconde factorisation) ;
        AI = Q' P Q, colonnes de Q comme dans make_ai.
    Ce n'est PAS une approximation : c'est le meme calcul que make_objective et
    make_ai, sans refaire la factorisation entre les deux.
    """
    base = make_ai(terms, res, y, X).outils
    terms = base["terms"]          # parente fixe deja pliee dans Zr
    groupes, Zr, rows_of, secs = base["groupes"], base["Zr"], base["rows_of"], base["secs"]
    colonnes_terme, colonnes_section = base["colonnes_terme"], base["colonnes_section"]
    yj, Xj, n = base["yj"], base["Xj"], base["n"]
    p_x = Xj.shape[1]
    G = [None] * len(terms)
    for rows, ks in groupes:
        for k in ks:
            if _gram_rentable(terms[k], rows):
                G[k] = Zr[k] @ Zr[k].T
    from .model import _section_V
    from .model import term_factor, sec_n_params as _snp

    # Un bloc qui couvre TOUTES les lignes dans l'ordre s'ajoute sans
    # dispersion : la dispersion dans un zeros(n, n) et sa transposee (un
    # rassemblement) coutaient chacune une matrice n x n de plus.
    tout = np.arange(n, dtype=np.int64)

    def _plein(r):
        r = np.asarray(r, dtype=np.int64)
        return r.shape[0] == n and np.array_equal(r, tout)

    sec_plein = [_plein(sec["rows"]) for sec in secs]
    grp_plein = [_plein(rows) for rows, _ in groupes]

    def _ajouter(V, M, rows, plein):
        if plein:
            return M if V is None else V + M
        idx = jnp.asarray(np.asarray(rows, dtype=np.int64))
        V = jnp.zeros((n, n)) if V is None else V
        return V.at[idx[:, None], idx[None, :]].add(M)

    def V_of(theta, Zr, G):
        th_terms, th_res, _ = split_theta(theta, terms, res)
        V = None
        o = 0
        for s_i, sec in enumerate(secs):
            pnb = _snp(sec)
            V = _ajouter(V, _section_V(th_res[o:o + pnb], sec), sec["rows"], sec_plein[s_i])
            o += pnb
        for g_i, (rows, ks) in enumerate(groupes):
            bloc = None
            fac = [term_factor(th_terms[k], terms[k], Zr[k]) for k in ks if G[k] is None]
            if fac:
                Bm = jnp.concatenate(fac, axis=1)
                bloc = Bm @ Bm.T
            for k in ks:
                if G[k] is not None:
                    c = jnp.exp(2.0 * th_terms[k][0]) * G[k]
                    bloc = c if bloc is None else bloc + c
            if bloc is not None:
                V = _ajouter(V, bloc, rows, grp_plein[g_i])
        return jnp.zeros((n, n)) if V is None else V

    def fga(theta, Zr, G):
        # MEMOIRE. P n'est jamais formee : l'adjoint de V et P Q s'ecrivent avec
        # V^-1 et un correctif de rang p. Mesure (ar1 x ar1, n = 2000) : 7,25
        # matrices n x n de temporaires avant, ce qui faisait 3,7 Go a n = 8000
        # et l'echec de l'AI sur une tranche MIG de 10 Go.
        V, vjp_V = jax.vjp(lambda th: V_of(th, Zr, G), theta)
        L = jnp.linalg.cholesky(V)
        logdetV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        Vi = jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
        ViX = Vi @ Xj
        A = Xj.T @ ViX
        La = jnp.linalg.cholesky(A)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(La)))
        ViXA = ViX @ jax.scipy.linalg.cho_solve((La, True), jnp.eye(p_x))
        Py = Vi @ yj - ViXA @ (ViX.T @ yj)
        f = (n - p_x) * np.log(2.0 * np.pi) + logdetV + logdetA + base["ldRx"] + yj @ Py
        # adjoint : P - Py Py' = V^-1 - [ViXA, Py] [ViX, Py]'
        g = vjp_V(Vi - jnp.concatenate([ViXA, Py[:, None]], axis=1)
                  @ jnp.concatenate([ViX, Py[:, None]], axis=1).T)[0]
        th_terms, th_res, _ = split_theta(theta, terms, res)
        blocs = []
        for k, tm in enumerate(terms):
            if G[k] is not None:
                c = jnp.exp(2.0 * th_terms[k][0])
                col = 2.0 * c * (G[k] @ Py[rows_of[k]])
                blocs.append(jnp.zeros((n, 1)).at[rows_of[k], 0].set(col))
            else:
                blocs.append(colonnes_terme(th_terms[k], tm, Zr[k], rows_of[k], Py))
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            blocs.append(colonnes_section(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(blocs, axis=1)
        PQ = Vi @ Q - ViXA @ (ViX.T @ Q)
        AI = Q.T @ PQ
        return f, g, 0.5 * (AI + AI.T)

    # incidences et Gram en ARGUMENTS (voir model.make_objective)
    _fga = jax.jit(fga)
    Zr_dev, G_dev = Zr, G

    def fga_appel(theta):
        return _fga(theta, Zr_dev, G_dev)
    return fga_appel


# ==============================================================================
# AI-REML EN ECHELLE DES VARIANCES (optimizer = "ai", par defaut des AI)
# ==============================================================================
# POURQUOI. Le solveur parametre chaque Sigma par son facteur de Cholesky, a
# diagonale en logarithme : V n'est pas lineaire en theta, et un pas de scoring
# pris en theta n'est pas celui que prend asreml. Asreml travaille sur les
# ENTREES des Sigma (phi), ou V est lineaire et ou la matrice AI est
# l'information moyenne exacte ; c'est ce qui lui donne quelques dizaines
# d'iterations la ou l'AI en theta rampe (us9, us12 a n ~ 500 ; mv3_ref).
#
# COMMENT, SANS NOUVELLE FONCTION COMPILEE. theta = theta(phi) est un
# changement de variables lisse. Avec J = d theta / d phi :
#     g_phi = J' g_theta,      AI_phi = J' AI_theta J
# (AI se transforme comme une information de Fisher : c'est exact.) On garde
# donc fga, et seul le PAS est pris en phi. Les structures iid, diag et us sont
# mises a l'echelle des variances ; les autres parametres (poids de fa, rho,
# portees, rr...) restent en theta.
#
# REGION ADMISSIBLE. Un candidat est PROJETE : valeurs propres d'un bloc us
# tronquees a 1e-10 fois la plus grande, puis theta borne a [floor, ceil] et
# phi recalcule depuis theta. Le pas plein peut ainsi atteindre une frontiere
# (variance nulle, correlation de 1), la ou la division du pas ramperait.
def _blocs_echelle(terms, res):
    """Decoupage de theta : (debut, longueur, nature, t), nature dans
    {"iid", "diag", "us", "id"} ("id" : laisse en theta)."""
    blocs, o = [], 0

    def sigma(o, st, t, rank):
        ns = n_params(st, t, rank)
        if not ns:
            return []
        if st == "fa":
            # poids de Lambda en theta (lineaires), variances specifiques psi^2
            # en echelle des variances : en log psi, une variance poussee au
            # plancher n'en remonte plus (gradient nul), comme pour diag.
            nl = ns - int(t)
            return [(o, nl, "id", 0), (o + nl, int(t), "diag", int(t))]
        nat = st if st in ("iid", "diag", "us") else "id"
        return [(o, ns, nat, int(t))]

    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        p = term_n_params(tm)
        blocs += sigma(o, tm["struct"], tm["t"], tm["rank"])
        if p > ns:
            blocs.append((o + ns, p - ns, "id", -1))          # niveaux (rho...) : t = -1
        o += p
    for sec in res_sections(res, 0):
        p = sec_n_params(sec)
        if p == 0:
            continue
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        blocs += sigma(o, sec["struct"], sec["t"], sec["rank"])
        if p > ns:
            blocs.append((o + ns, p - ns, "id", -1))
        o += p
    return blocs, o


def _tril(t):
    return np.tril_indices(t)            # ligne par ligne, comme structures._tril_indices


def _theta_vers_phi(theta, blocs, xp=jnp):
    out = []
    for (o, ln, nat, t) in blocs:
        th = theta[o:o + ln]
        if nat in ("iid", "diag"):
            out.append(xp.exp(2.0 * th))
        elif nat == "us":
            ii, jj = _tril(t)
            L = xp.zeros((t, t))
            if xp is jnp:
                L = L.at[ii, jj].set(th)
                L = L.at[np.arange(t), np.arange(t)].set(xp.exp(xp.diagonal(L)))
            else:
                L = L.copy(); L[ii, jj] = th
                L[np.arange(t), np.arange(t)] = np.exp(np.diagonal(L))
            S = L @ L.T
            out.append(S[ii, jj])
        else:
            out.append(th)
    return xp.concatenate(out) if out else xp.zeros(0)


def _phi_vers_theta(phi, blocs, xp=jnp):
    out = []
    for (o, ln, nat, t) in blocs:
        ph = phi[o:o + ln]
        if nat in ("iid", "diag"):
            out.append(0.5 * xp.log(ph))
        elif nat == "us":
            ii, jj = _tril(t)
            if xp is jnp:
                S = jnp.zeros((t, t)).at[ii, jj].set(ph).at[jj, ii].set(ph)
                L = jnp.linalg.cholesky(S)
                th = L[ii, jj]
                th = jnp.where(ii == jj, jnp.log(jnp.where(ii == jj, th, 1.0)), th)
            else:
                S = np.zeros((t, t)); S[ii, jj] = ph; S[jj, ii] = ph
                L = np.linalg.cholesky(S)
                th = L[ii, jj].copy()
                d = ii == jj
                th[d] = np.log(th[d])
            out.append(th)
        else:
            out.append(ph)
    return xp.concatenate(out) if out else xp.zeros(0)


import os as _os_ai
REDUCTION_MAX = float(_os_ai.environ.get("REMLAX_AI_REDUCTION", "0.1"))
# BRIDES DU PAS AI (voir essayer() dans ai_reml_var) : une variance au plus
# divisee par 10 par pas, un parametre de niveau (rho en echelle atanh, portee
# en log) au plus deplace de 0,5. Le test de rapport des regions de confiance
# (gain reel >= RAPPORT_MIN x gain predit) reste une option : sur les cas
# testes il ne changeait aucun optimum et coutait jusqu'a 60 % de temps.
# N_BRIDE_PLANCHER : une variance bridee sur 3 pas acceptes de suite peut
# rejoindre le plancher (cas de Heywood d'un fa : 104 -> 50 iterations sur
# Krause fa2, 600 genotypes, meme optimum ; aucun autre cas du banc ne change).
PAS_NIV = float(_os_ai.environ.get("REMLAX_AI_PAS_NIV", "0.5"))
RAPPORT_MIN = float(_os_ai.environ.get("REMLAX_AI_RAPPORT", "0"))
N_BRIDE_PLANCHER = int(_os_ai.environ.get("REMLAX_AI_N_BRIDE", "3"))


def ai_reml_var(fga, th0, bornes, terms, res, maxiter=200, tol_dec=1e-6, verbose=True,
                compteur=None, plancher_us=1e-10, fun_sc=None, f_seul=None):
    """AI-REML dont le pas est pris en echelle des variances (voir plus haut).

    Arret : decrement g' AI^-1 g / 2 sous tol_dec (unites de -2logL) ; ou trois
    gains de suite sous 1e-6 (optimum a la frontiere : le pas pointe dehors, le
    decrement reste grand) ; ou aucun pas admissible (echec declare, fit_reml
    garde alors le polissage de Newton en repli).
    """
    blocs, p = _blocs_echelle(terms, res)
    lo = np.array([b[0] for b in bornes], dtype=np.float64)
    hi = np.array([b[1] for b in bornes], dtype=np.float64)
    if p != len(lo):
        raise ValueError("ai : %d parametres decoupes pour %d bornes" % (p, len(lo)))
    est_var = np.zeros(p, dtype=bool)
    est_niv = np.zeros(p, dtype=bool)
    for (o, ln, nat, t) in blocs:
        if nat == "id" and t == -1:
            est_niv[o:o + ln] = True
    for (o, ln, nat, t) in blocs:
        if nat in ("iid", "diag"):
            est_var[o:o + ln] = True
        elif nat == "us":
            ii, jj = _tril(t)
            est_var[o:o + ln] = ii == jj
    jac = jax.jit(jax.jacfwd(lambda ph: _phi_vers_theta(ph, blocs)))

    def theta_de(phi):
        return np.clip(_phi_vers_theta(phi, blocs, xp=np), lo, hi)

    def projeter(phi):
        """phi admissible le plus proche, puis aller-retour par theta borne."""
        phi = np.array(phi, dtype=np.float64)
        for (o, ln, nat, t) in blocs:
            if nat in ("iid", "diag"):
                phi[o:o + ln] = np.maximum(phi[o:o + ln], np.exp(2.0 * lo[o:o + ln]))
            elif nat == "us":
                ii, jj = _tril(t)
                S = np.zeros((t, t)); S[ii, jj] = phi[o:o + ln]; S[jj, ii] = phi[o:o + ln]
                w, U = np.linalg.eigh(S)
                if w[-1] <= 0:
                    return None
                plancher = plancher_us * w[-1]
                if w[0] < plancher:
                    S = (U * np.maximum(w, plancher)) @ U.T
                    phi[o:o + ln] = S[ii, jj]
        th = theta_de(phi)
        return _theta_vers_phi(th, blocs, xp=np), th

    def evaluer(th):
        if compteur is not None:
            compteur["n_eval"] += 1
        f_, g_, A_ = fga(jnp.asarray(th))
        return float(f_), np.asarray(g_, dtype=np.float64), np.asarray(A_, dtype=np.float64)

    theta = np.clip(np.asarray(th0, dtype=np.float64), lo, hi)
    phi = _theta_vers_phi(theta, blocs, xp=np)
    f, g_t, A_t = evaluer(theta)
    msg, ok, n_it, petits = "maxiter atteint", False, 0, 0
    n_salves = 0

    def salve(pourquoi):
        """SALVE_ITER iterations de L-BFGS-B depuis theta ; rend True si -2logL baisse."""
        nonlocal theta, phi, f, g_t, A_t, n_salves, n_it
        if fun_sc is None or n_salves >= SALVES_MAX:
            return False
        from scipy.optimize import minimize
        n_salves += 1
        r = minimize(fun_sc, theta, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=SALVE_ITER, maxfun=3 * SALVE_ITER,
                                  ftol=1e-15, gtol=1e-12))
        th_n = np.clip(np.asarray(r.x, dtype=np.float64), lo, hi)
        f_n, g_n, A_n = evaluer(th_n)
        n_it += int(r.nfev)
        gain_ = f - f_n
        if verbose:
            print("  [ai] %s : salve L-BFGS-B %d/%d (%d iterations), gain %.3e"
                  % (pourquoi, n_salves, SALVES_MAX, int(r.nit), gain_), flush=True)
        if not (np.isfinite(f_n) and gain_ > 1e-10 * max(1.0, abs(f))):
            return False
        theta, f, g_t, A_t = th_n, f_n, g_n, A_n
        phi = _theta_vers_phi(theta, blocs, xp=np)
        return True
    n_bride = np.zeros(p, dtype=np.int64)        # pas acceptes de suite ou la bride a joue
    mu_mem = None                                # amortissement LM retenu de l'iteration precedente
    dernier_bride = np.zeros(p, dtype=bool)
    t0 = time.time()
    for it in range(int(maxiter)):
        if not (np.isfinite(f) and np.all(np.isfinite(A_t))):
            msg = "valeur ou matrice AI non finie"
            break
        J = np.asarray(jac(jnp.asarray(phi)), dtype=np.float64)
        g = J.T @ g_t
        A = J.T @ A_t @ J
        # Ensemble actif, comme L-BFGS-B : un parametre a sa borne et pousse vers
        # l'exterieur sort du pas. En phi, le sens du gradient d'une variance est
        # celui de theta (changement de variable monotone).
        bas = (theta <= lo + 1e-12) & (g > 0)
        haut = (theta >= hi - 1e-12) & (g < 0)
        fixe = (hi - lo) <= 0
        libre = ~(bas | haut | fixe)
        # un bloc us dont une variance est au plancher reste libre : la
        # projection sur le cone s'en charge.
        for (o, ln, nat, t) in blocs:
            if nat == "us":
                libre[o:o + ln] = ~fixe[o:o + ln]
        if not libre.any():
            msg, ok = "tous les parametres sont a leur borne", True
            break
        # ENSEMBLE ACTIF SUR LE CONE. Un bloc us dont une valeur propre est au
        # plancher, et dont le gradient pousse vers l'exterieur le long de ses
        # vecteurs propres nuls (v' dF/dSigma v > 0), est restreint a sa FACE :
        # Sigma = U S U', U orthogonal aux directions nulles. V reste lineaire
        # en S, donc l'AI de la face, T' AI T, est encore exacte. Sans cela le
        # pas de Newton pointe hors du cone, la projection le ramene au meme
        # point, et la boucle cale (us9 a n = 500 : 1,4 unite sous l'optimum).
        #
        # LA FACE TOURNE. Les matrices de rang r forment une variete dont l'espace
        # tangent compte, outre U S U' (S symetrique r x r), les directions
        # U B V0' + V0 B' U' (B r x (t - r)) qui font tourner l'image de Sigma.
        # Sans elles, la face figeait U au moment ou une valeur propre touchait le
        # plancher : l'AI s'arretait sur un point qui n'est pas stationnaire au
        # bord, et L-BFGS-B le faisait tourner a petits pas. MESURE (banc absolu,
        # us(8), q = 1000, A100) : AI arretee en 9 iterations sur la face, puis
        # 490 iterations de L-BFGS-B et 4 pas de polissage pour 1,98 unite de
        # -2logL ; us(12) : jusqu'a 3000 iterations.
        # Le long d'une rotation, Sigma suit la retraction exacte
        #     (U S + V0 B')(S^-1)(S U' + B V0')
        # dont le terme d'ordre 2, V0 B' S^-1 B V0', ajoute a la courbure le terme
        # 2 kron(S^-1, V0' G V0) (G = dF/dSigma). L'AI seule l'ignore et
        # proposerait des pas trop longs des qu'une valeur propre gardee est petite.
        cols, n_faces, faces, n_col = [], 0, [], 0
        for (o, ln, nat, t) in blocs:
            idx = np.arange(o, o + ln)
            M = None
            if nat == "us" and t >= 2:
                ii, jj = _tril(t)
                S = np.zeros((t, t)); S[ii, jj] = phi[o:o + ln]; S[jj, ii] = phi[o:o + ln]
                w_s, U_s = np.linalg.eigh(S)
                # seuil de la face a 100 fois le plancher de projection (au moins
                # 1e-8) : au seuil egal au plancher (MME, 1e-8), une valeur propre
                # projetee tombait d'un cote ou de l'autre par arrondi, et la face
                # changeait de rang a chaque iteration (us(12) : 11, 9, 10).
                nuls = w_s <= max(1e-8, 100.0 * plancher_us) * w_s[-1]
                if nuls.any() and not nuls.all():
                    gb = g[o:o + ln]
                    Gm = np.zeros((t, t)); Gm[ii, jj] = np.where(ii == jj, gb, 0.5 * gb)
                    Gm = Gm + np.tril(Gm, -1).T
                    V0 = U_s[:, nuls]
                    # FACE PARTIELLE. Seules les directions nulles ou le gradient
                    # pousse vers l'exterieur (valeurs propres > 0 de V0' G V0)
                    # restent bloquees ; les autres rejoignent l'image, ou leur
                    # variance peut remonter. Exiger toutes les directions du meme
                    # cote liberait le bloc entier des qu'une seule voulait
                    # remonter : le pas de Newton sortait alors du cone par les
                    # autres, et aucun pas n'etait admissible (us(12), MME).
                    gam, Qg = np.linalg.eigh(V0.T @ Gm @ V0)
                    act = gam > 0
                    if act.any():
                        V0a = V0 @ Qg[:, act]
                        Ur = np.concatenate([U_s[:, ~nuls], V0 @ Qg[:, ~act]], axis=1)
                        # r1 vecteurs de l'image proprement dite, suivis des
                        # directions liberees (variance au plancher). Les rotations
                        # ne portent que sur les r1 premiers : vers une direction
                        # liberee, la courbure 2/s (s ~ plancher) est sans objet.
                        r1 = int((~nuls).sum())
                        r = Ur.shape[1]; m0 = t - r
                        Sr = Ur.T @ S @ Ur
                        Sr = 0.5 * (Sr + Sr.T)
                        Srinv = np.linalg.inv(Sr[:r1, :r1])
                        ik, jk = np.tril_indices(r)
                        M = np.zeros((p, len(ik) + r1 * m0))
                        for c, (k, l) in enumerate(zip(ik, jk)):
                            E = np.zeros((r, r)); E[k, l] = 1.0; E[l, k] = 1.0
                            M[o:o + ln, c] = (Ur @ E @ Ur.T)[ii, jj]
                        c0 = len(ik)
                        for k in range(r1):
                            for m in range(m0):
                                D = np.outer(Ur[:, k], V0a[:, m])
                                M[o:o + ln, c0 + k * m0 + m] = (D + D.T)[ii, jj]
                        faces.append(dict(o=o, ln=ln, ii=ii, jj=jj, r=r, r1=r1, m0=m0, V0=V0a,
                                          Ur=Ur, Sr=Sr, Srinv=0.5 * (Srinv + Srinv.T),
                                          Gnn=np.diag(gam[act]), c0=n_col + c0))
                        n_faces += 1
            if M is not None:
                cols.append(M)
                n_col += M.shape[1]
            else:
                lib = idx[libre[idx]]
                if len(lib):
                    E = np.zeros((p, len(lib))); E[lib, np.arange(len(lib))] = 1.0
                    cols.append(E)
                    n_col += len(lib)
        T = np.concatenate(cols, axis=1)
        Af, gf = T.T @ A @ T, T.T @ g
        for fc in faces:
            sl = slice(fc["c0"], fc["c0"] + fc["r1"] * fc["m0"])
            Af[sl, sl] += 2.0 * np.kron(fc["Srinv"], fc["Gnn"])

        def dphi_de(xi):
            """pas reduit -> pas en phi. Sur une face, Sigma suit la retraction
            exacte (U S' + V0 B') S'^-1 (S' U' + B V0'), S' = S + A : le rang
            reste r. Avec le seul terme d'ordre 2, le pas dans l'image (A)
            relevait les valeurs propres nulles au-dessus du seuil de la face,
            qui se perdait une iteration sur deux."""
            d = T @ xi
            for fc in faces:
                r, r1, m0 = fc["r"], fc["r1"], fc["m0"]
                c_a = fc["c0"] - r * (r + 1) // 2
                ik, jk = np.tril_indices(r)
                Am = np.zeros((r, r)); Am[ik, jk] = xi[c_a:fc["c0"]]; Am[jk, ik] = xi[c_a:fc["c0"]]
                Sp = fc["Sr"] + Am
                Bm = np.zeros((r, m0))
                Bm[:r1] = xi[fc["c0"]:fc["c0"] + r1 * m0].reshape(r1, m0)
                try:
                    Lp = np.linalg.cholesky(Sp)
                except np.linalg.LinAlgError:
                    continue                       # pas trop long : projection lineaire
                F = fc["Ur"] @ Lp + fc["V0"] @ np.linalg.solve(Lp, Bm).T
                S_new = F @ F.T
                S_old = fc["Ur"] @ fc["Sr"] @ fc["Ur"].T
                o, ln = fc["o"], fc["ln"]
                d[o:o + ln] = (S_new - S_old)[fc["ii"], fc["jj"]]
            return d
        w, Vv = np.linalg.eigh(0.5 * (Af + Af.T))
        lam = max(1e-12, 1e-10 * abs(w).max())
        wr = np.where(w > lam, w, lam)
        pas_l = -(Vv @ ((Vv.T @ gf) / wr))
        dec = -0.5 * float(gf @ pas_l)
        if dec < tol_dec:
            if n_faces:
                # Point de KKT sur le bord du cone : stationnaire sur toute la
                # variete de rang r (rotations comprises) et gradient qui pousse
                # vers l'exterieur le long des directions nulles (V0' G V0 > 0).
                # Avant les rotations, ce point etait seulement stationnaire a U
                # fige, d'ou le repli L-BFGS-B (us9 a n = 500 : -798,76 sur la
                # face, -796,94 apres repli).
                msg, ok = ("decrement AI sous le seuil, optimum au bord du cone "
                           "(%d bloc(s) us de rang incomplet)" % n_faces), True
            else:
                msg, ok = "decrement AI sous le seuil", True
            break
        pas = dphi_de(pas_l)

        def essayer(dphi):
            # ESSAI PAR LA SEULE VALEUR (f_seul), quand elle est fournie : un pas
            # rejete n'a besoin ni du gradient ni de la matrice AI, qui coutent
            # l'inversion selective et les traces. MESURE sur drops2 (MME dense,
            # m = 1428) : 0,10 s la valeur seule contre 0,49 s l'evaluation
            # complete, et une iteration amortie essaie jusqu'a 20 pas.
            nonlocal n_it
            # UNE VARIANCE NE TOMBE PAS AU PLANCHER EN UN PAS : au plus divisee
            # par 10 (REDUCTION_MAX, regle d'asreml). Sans cela un pas AI envoyait la variance
            # residuelle d'une section a 1e-8 fois sa valeur ; log|R| et log|C|
            # y deviennent enormes et se compensent, -2logL perd ses chiffres et
            # l'AI se declarait convergee sur un point faux (Krause, 5
            # traitements, 600 genotypes : 3364,93 contre 3338,25 a l'optimum).
            # LE FACTEUR DECIDE AUSSI DE L'OPTIMUM ATTEINT quand la vraisemblance
            # en a plusieurs. MESURE sur DROPS une etape us(5) (champs ar1 a rho
            # proche de 1) : /10 atteint -1146,19, /100 -1384,0 et /1000 -1189,6,
            # trois points stationnaires confirmes par le moteur dense. /10 est
            # le seul facteur juste sur les trois cas testes ; il ralentit un fa2
            # (104 iterations contre 48 a /1000).
            # UN PARAMETRE DE NIVEAU NE SAUTE PAS (|pas| <= PAS_NIV). Sur une
            # vraisemblance a plusieurs optimums (DROPS une etape, champs ar1 a
            # rho ~ 0,997), un pas AI plein deplacait rho d'un bassin a l'autre,
            # et l'optimum atteint dependait du nombre de fils BLAS (-1146,19,
            # -1305,6 ou -1462,5 pour le meme modele). Bride a 0,5 : optimum juste
            # sur DROPS fa1 et us, avec 6 et 8 fils, sans iteration de plus
            # ailleurs (Krause, DROPS deux etapes).
            dphi = np.array(dphi, dtype=np.float64)
            dphi[est_niv] = np.clip(dphi[est_niv], -PAS_NIV, PAS_NIV)
            cible = phi + dphi
            # bride, sauf pour une variance deja bridee sur N_BRIDE_PLANCHER pas
            # acceptes de suite : elle va vraiment vers zero (cas de Heywood
            # d'un fa) et rejoint le plancher sans dix paliers de plus
            libre_bas = est_var & (n_bride >= N_BRIDE_PLANCHER)
            borne_bas = np.where(libre_bas, -np.inf, REDUCTION_MAX * phi)
            bride_ = est_var & (cible < borne_bas)
            cible[est_var] = np.maximum(cible[est_var], borne_bas[est_var])
            dernier_bride[:] = bride_
            pr = projeter(cible)
            if pr is None:
                return None
            phi_c, th_c = pr
            n_it += 1
            # TEST DE RAPPORT (option, REMLAX_AI_RAPPORT > 0) : le gain reel doit
            # valoir au moins RAPPORT_MIN du gain predit par le modele
            # quadratique de l'AI. Pres de l'optimum (gain predit sous 1e-6),
            # toute baisse reste acceptee.
            d_ = phi_c - phi
            pred = -float(g @ d_ + 0.5 * d_ @ A @ d_)

            def admis(f_c):
                if not (np.isfinite(f_c) and f_c <= f + 1e-10 * max(1.0, abs(f))):
                    return False
                if RAPPORT_MIN > 0 and pred > 1e-6:
                    return (f - f_c) >= RAPPORT_MIN * pred
                return True
            if f_seul is not None:
                if compteur is not None:
                    compteur["n_eval"] += 1
                if not admis(float(f_seul(th_c))):
                    return None
            f_c, g_c, A_c = evaluer(th_c)
            if admis(f_c):
                return (phi_c, th_c, f_c, g_c, A_c)
            return None

        alpha, acc, mu = 1.0, None, None
        echelle = max(abs(w).max(), 1e-12)

        def pas_lm(mu_):
            return dphi_de(-(Vv @ ((Vv.T @ gf) / (np.maximum(w, 0.0) + mu_))))

        if mu_mem is None:
            for h in range(N_DEMI_PAS):
                acc = essayer(pas if alpha == 1.0 else dphi_de(alpha * pas_l))
                if acc is not None:
                    break
                alpha *= 0.5
        else:
            # un amortissement a servi a l'iteration precedente : le pas de
            # Newton plein d'abord (un seul essai, sans les demi-pas). Sans lui,
            # remlax restait amorti plusieurs iterations ou le pas plein passait
            # (drops2 fa1 : 7,2 s en v10, 14,5 s avec le seul mu garde).
            acc = essayer(pas)
        if acc is None:
            # AMORTISSEMENT DE LEVENBERG-MARQUARDT. Quand la direction de Newton
            # est portee par une valeur propre quasi nulle de l'AI, la reduire ne
            # sert a rien : chaque pas accepte est minuscule et le decrement ne
            # baisse pas (fa de rang 2, n = 500 : decrement fige a 1,26e3, pas de
            # 3e-5, douze iterations pour rien). (AI + mu I) tourne la direction
            # vers le gradient a mesure que mu croit.
            # MU GARDE D'UNE ITERATION A L'AUTRE. Repartir de 1e-6 x l'echelle a
            # chaque iteration amortie refaisait les demi-pas puis toute la serie
            # des mu : une douzaine d'essais par iteration pour retrouver le meme
            # mu (fa2 a t = 8, q = 1000 : trois iterations amorties de 20 a 35 s
            # contre 4 s pour une iteration pleine). Tant qu'un amortissement a
            # servi, on repart du dernier mu accepte divise par 10, sans demi-pas.
            k0 = 0 if mu_mem is None else max(0, int(np.floor(np.log10(mu_mem / echelle))) + 6)
            for k in range(k0, 14):
                mu = echelle * 10.0 ** (k - 6)
                acc = essayer(pas_lm(mu))
                if acc is not None:
                    alpha = 1.0
                    break
            if acc is None and mu_mem is not None:
                # le mu retenu ne convient plus : retour au pas de Newton
                mu, mu_mem = None, None
                alpha = 1.0
                for h in range(N_DEMI_PAS):
                    acc = essayer(pas if alpha == 1.0 else dphi_de(alpha * pas_l))
                    if acc is not None:
                        break
                    alpha *= 0.5
        # mu retenu pour la suite : divise par 10 a chaque succes, oublie quand
        # il devient negligeable devant l'AI (retour au pas de Newton)
        if acc is not None and mu is not None:
            mu_mem = mu / 10.0 if mu / 10.0 > 1e-6 * echelle else None
        elif acc is not None:
            mu_mem = None
        if acc is None:
            if dec >= SEUIL_SUCCES and salve("aucun pas admissible (decrement %.1e)" % dec):
                petits = 0
                continue
            msg, ok = ("aucun pas admissible (decrement %.1e)" % dec,
                       dec < SEUIL_SUCCES)
            break
        gain = f - acc[2]
        phi, theta, f, g_t, A_t = acc
        n_bride = np.where(dernier_bride, n_bride + 1, 0)
        if verbose:
            print("  aiv  %4d | -2logL %14.6f | gain %.3e | decrement %.3e | %s | %6.1f s%s"
                  % (it + 1, f, gain, dec, ("pas %.3g" % alpha) if mu is None else ("lm mu %.1e" % mu),
                     time.time() - t0,
                     "".join(" | face rang %d/%d" % (fc["r"], fc["r"] + fc["m0"]) for fc in faces)),
                  flush=True)
        petits = petits + 1 if abs(gain) < 1e-6 else 0
        if petits >= 3 and dec >= SEUIL_SUCCES and salve("stagnation (decrement %.1e)" % dec):
            petits = 0
            continue
        if petits >= 3:
            # Optimum a la frontiere si le decrement est deja petit ; sinon la
            # boucle rampe et l'echec est declare, pour que fit_reml enchaine le
            # polissage de Newton sur le vrai Hessien.
            ok = dec < SEUIL_SUCCES
            msg = ("trois gains de suite sous 1e-6 (optimum a la frontiere)" if ok else
                   "stagnation : trois gains sous 1e-6 pour un decrement de %.1e" % dec)
            break
    return theta, dict(nit=n_it, success=ok, message="AI-REML (variances) : %s" % msg)
