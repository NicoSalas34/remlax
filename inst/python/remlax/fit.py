"""Ajustement REML : optimisation, BLUPs, erreurs-types, diagnostic.

L'optimiseur est L-BFGS-B sur une parametrisation NON CONTRAINTE : aucune borne
n'est necessaire pour garantir la positivite (cf. structures.py), les bornes
larges posees ici ne servent qu'a empecher une variance de fuir vers exp(-inf),
ou le gradient s'annule exponentiellement et rend le point indistinguable d'un
optimum.

CRITERE D'ARRET. On ne se fie pas a max|grad| : sur ces vraisemblances il vaut
plusieurs unites a des optima verifies. Le diagnostic rendu porte le DECREMENT
DE NEWTON g' H^-1 g, qui est la montee de logLik encore disponible sous le
modele quadratique local — donc une quantite comparable au 3.84 d'un LRT a
1 ddl, et invariante par reparametrisation affine.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

try:
    from .model import (make_objective, assemble_V, split_theta, n_theta, dense_Z,
                        term_factor, term_n_params, res_sections, sec_n_params,
                        alias_entrees)
    from .structures import build_sigma, n_params, theta0 as struct_theta0
    from .levels import n_level_params, level_params_report, level_chol
    from .structures import sigma_loadings, n_loadings
    from .lag import (prepare_lag, lag_n_params, ystar as lag_ystar, lag_terms,
                      lag_report, theta_from_psi)
except ImportError:
    from model import (make_objective, assemble_V, split_theta, n_theta, dense_Z,
                       term_factor, term_n_params, res_sections, sec_n_params,
                       alias_entrees)
    from structures import build_sigma, n_params, theta0 as struct_theta0
    from levels import n_level_params, level_params_report, level_chol
    from structures import sigma_loadings, n_loadings
    from lag import (prepare_lag, lag_n_params, ystar as lag_ystar, lag_terms,
                     lag_report, theta_from_psi)


def _chrono(nom):
    """Marque d'etape (chrono.py) ; sans effet si le module est absent."""
    try:
        from . import chrono
    except ImportError:
        try:
            import chrono
        except ImportError:
            return
    chrono.marque(nom)


def _somme_diag_ZKZ(Z, LK):
    """Somme des (Z K Z')_ii, K = LK LK', Z creuse (CSR).

    Deux voies de meme resultat. Par Z LK : nnz(Z) x q operations. Par Z'Z :
    trace(Z K Z') = somme des G_jl K_jl, G = Z'Z, avec K_jl = LK_j . LK_l
    calcule aux seuls non-zeros de G, soit nnz(G) x q operations. Avec des
    repetitions, nnz(G) est bien plus petit que nnz(Z) : MESURE sur une GBLUP
    a 128 000 observations et 2000 genotypes, nnz(Z) = 128 000 contre
    nnz(G) = 2000, et 4 s par appel par la premiere voie."""
    q = LK.shape[0]
    G = (Z.T @ Z).tocoo()
    pas = max(1, int(2 ** 25 // max(LK.shape[1], 1)))
    if G.nnz < Z.nnz:
        tot = 0.0
        for a in range(0, G.nnz, pas):
            r, c = G.row[a:a + pas], G.col[a:a + pas]
            tot += float(np.dot(G.data[a:a + pas], np.einsum("ij,ij->i", LK[r], LK[c])))
        return tot
    tot = 0.0
    for a in range(0, Z.shape[0], pas):
        ZL = Z[a:a + pas] @ LK
        tot += float(np.sum(ZL * ZL))
    return tot


def _exposition(tm, n):
    """k = moyenne des (Z K Z')_ii par colonne : facteur par lequel la variance
    d'un terme entre dans le phenotype (voir initial_theta)."""
    # Z en CREUX, lignes (observation, composante) : la version dense formait
    # Z (n x tq) puis Z LK par einsum, sans BLAS. MESURE sur une GBLUP a 3768
    # genotypes : 45 s sur les 100 de l'ajustement, pour un simple point de
    # depart. Le produit creux x dense coute nnz(Z) x q et se fait par paquets
    # de lignes pour borner la memoire.
    # garde dans le terme : initial_theta et bornes_loadings l'appellent
    # plusieurs fois par ajustement, sur le meme terme
    cle = ("_expo", n)
    if cle in tm:
        return tm[cle]
    import scipy.sparse as sp
    t, q = int(tm["t"]), int(tm["q"])
    zi = np.asarray(tm["zi"], dtype=np.int64); zj = np.asarray(tm["zj"], dtype=np.int64)
    Zs = sp.csr_matrix((np.asarray(tm["zx"], dtype=np.float64), (zi * t + zj // q, zj % q)),
                       shape=(n * t, q))
    if tm["LK"] is None:
        tot = float(Zs.multiply(Zs).sum())
    else:
        tot = _somme_diag_ZKZ(Zs, np.asarray(tm["LK"], dtype=np.float64))
    # MOYENNE SUR LES COUPLES (observation, composante) CHARGES, et non sur
    # n x t. En format long chaque ligne ne charge que SA composante (son
    # environnement, son caractere) : diviser par n x t sous-estimait
    # l'exposition d'un facteur t, et chaque variance de depart etait t fois
    # trop grande. MESURE sur un fa1 a 28 environnements (drops2) : variances
    # specifiques de depart 39 contre 0,03 a 0,7 a l'optimum, -2logL 10136
    # contre 5595, et 13 iterations AI amorties avant le premier pas plein.
    # Quand chaque ligne charge toutes les composantes, le compte vaut n x t et
    # rien ne change.
    n_charges = int(np.count_nonzero(np.diff(Zs.indptr)))
    k = tot / max(n_charges, 1)
    tm[cle] = k if k > 1e-12 else 1.0
    return tm[cle]


# BORNE DES LOADINGS fa ET rr, en multiples de l'ecart-type de depart du terme.
# Les loadings sont des parametres LINEAIRES, sur l'echelle des ecarts-types :
# la borne commune de theta (12, faite pour des log-ecarts-types) les laissait
# partir a 12 en quelques pas d'AI, ou Sigma = Lambda Lambda' + psi^2 devient de
# rang deficient et ou le moteur MME n'evalue plus rien (banc 2026-10-22, fa a 5
# caracteres : valeurs propres 0, 0, 0, 261, 759 pour des variances vraies
# d'ordre 1). Cent ecarts-types ne genent aucun optimum reel.
BORNE_LOADINGS_SD = 100.0


def bornes_loadings(terms, res, y):
    """[(debut, longueur, borne)] des loadings fa/rr dans theta."""
    vy = float(np.var(np.asarray(y))) or 1.0
    part = vy / (len(terms) + 1.0)
    out, o = [], 0
    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        if tm["struct"] in ("fa", "rr"):
            nl = n_loadings(tm["t"], tm["rank"]) if tm["struct"] == "fa" else ns
            B = BORNE_LOADINGS_SD * float(np.sqrt(part / _exposition(tm, len(y))))
            out.append((o, nl, B))
        o += term_n_params(tm)
    for sec in res_sections(res, len(y)):
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        if sec["struct"] in ("fa", "rr"):
            nl = n_loadings(sec["t"], sec["rank"]) if sec["struct"] == "fa" else ns
            out.append((o, nl, BORNE_LOADINGS_SD * float(np.sqrt(part))))
        o += sec_n_params(sec)
    return out


def _expo_composantes(tm, n, e):
    """Par composante c d'un terme : exposition k_c (moyenne des (Z_c K Z_c')_ii
    sur les lignes qui chargent c) et variance moyenne v_c des residus e sur
    ces lignes. Rend (k, v), deux vecteurs de longueur t ; NaN ou aucune ligne :
    la composante prend la moyenne des autres."""
    import scipy.sparse as sp
    t, q = int(tm["t"]), int(tm["q"])
    zi = np.asarray(tm["zi"], dtype=np.int64); zj = np.asarray(tm["zj"], dtype=np.int64)
    zx = np.asarray(tm["zx"], dtype=np.float64)
    comp = zj // q
    k = np.full(t, np.nan); v = np.full(t, np.nan)
    LK = None if tm.get("LK") is None else np.asarray(tm["LK"], dtype=np.float64)
    for c in range(t):
        m = comp == c
        if not m.any():
            continue
        lignes, inv = np.unique(zi[m], return_inverse=True)
        Zc = sp.csr_matrix((zx[m], (inv, zj[m] % q)), shape=(len(lignes), q))
        tot = float(Zc.multiply(Zc).sum()) if LK is None else _somme_diag_ZKZ(Zc, LK)
        k[c] = tot / len(lignes)
        v[c] = float(np.mean(e[lignes] ** 2))
    for x in (k, v):
        ok = np.isfinite(x) & (x > 1e-12)
        x[~ok] = np.mean(x[ok]) if ok.any() else 1.0
    return k, v


def _theta0_composantes(struct, t, rank, var):
    """Depart d'une Sigma diagonale de variances `var` (une par composante) ;
    les structures sans variance par composante prennent leur moyenne."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-8)
    sd = np.sqrt(var)
    if struct == "diag":
        return np.log(sd)
    if struct == "us":
        ii, jj = np.tril_indices(t)
        th = np.zeros(len(ii))
        th[ii == jj] = np.log(sd)
        return th
    if struct == "fa":
        lam = [0.1 * sd[i] for i in range(t) for j in range(min(i + 1, rank))]
        return np.concatenate([np.asarray(lam), np.log(sd)])
    return struct_theta0(struct, t, rank, var=float(np.mean(var)))


def initial_theta(terms, res, y, X=None):
    """Depart : variance repartie a parts egales entre termes.

    VARIANCE DES RESIDUS DES EFFETS FIXES quand X est fourni, et non celle de y
    brut : en plusieurs environnements ou traitements, y porte les ecarts de
    moyenne entre eux, et les variances de depart en etaient gonflees. MESURE
    sur 5 traitements x 3768 genotypes (Krause) : -2logL de depart 49 000 contre
    20 311 a l'optimum, et 7 iterations AI amorties (920 s sur 1216) avant le
    premier pas plein.

    PAR COMPOSANTE ET PAR SECTION. Un terme a plusieurs composantes (G x E en
    format long) part d'une Sigma diagonale dont chaque variance vient des
    residus des lignes de sa composante ; une residuelle a plusieurs sections
    part de la variance des residus de chaque section.

    L'EXPOSITION EST PRISE EN COMPTE. Une composante de voisinage entre dans le
    phenotype par sigma^2 * k avec k = moyenne des (Z K Z')_ii, qui vaut 15 a 40
    sur certains dispositifs. Partir de sigma^2 = part * var(y) y mettrait donc
    plusieurs fois la variance phenotypique dans CHAQUE terme, et V demarrerait
    des ordres de grandeur trop grande. On divise par k.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if X is not None and np.size(X):
        X = np.asarray(X, dtype=np.float64)
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        ddl = max(n - int(np.linalg.matrix_rank(X)), 1)
        e = e * np.sqrt(n / ddl)                   # variance sans biais
    else:
        e = y - float(np.mean(y))
    vy = float(np.mean(e ** 2)) or 1.0
    nt = len(terms) + 1.0
    th = []
    for tm in terms:
        t = int(tm["t"])
        if X is not None and t > 1 and tm["struct"] in ("diag", "us", "fa"):
            k_c, v_c = _expo_composantes(tm, n, e)
            th.append(_theta0_composantes(tm["struct"], t, tm["rank"], v_c / nt / k_c))
        else:
            k = _exposition(tm, n)
            th.append(struct_theta0(tm["struct"], tm["t"], tm["rank"], var=vy / nt / k))
        # rho de depart a 0 (tanh(0)) : champ non structure. Partir d'un rho
        # eleve fait demarrer dans un regime ou la vraisemblance est plate.
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        nl = n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                            opts=tm.get("lvl_opts"))
        if nl:
            th.append(_theta0_niveaux(kind, nl, tm.get("lvl_opts"), tm.get("coord")))
    for sec in res_sections(res, n):
        rows = np.asarray(sec.get("rows", np.arange(n)), dtype=np.int64)
        ts = int(sec["t"])
        if X is not None and len(rows):
            if ts > 1 and sec["struct"] in ("diag", "us", "fa") and sec.get("trait") is not None:
                tr = np.asarray(sec["trait"], dtype=np.int64)
                v_c = np.array([float(np.mean(e[rows[tr == c]] ** 2)) if np.any(tr == c) else vy
                                for c in range(ts)])
                th.append(_theta0_composantes(sec["struct"], ts, sec["rank"], v_c / nt))
            else:
                th.append(struct_theta0(sec["struct"], ts, sec["rank"],
                                        var=float(np.mean(e[rows] ** 2)) / nt))
        else:
            th.append(struct_theta0(sec["struct"], ts, sec["rank"], var=vy / nt))
        nlr = n_level_params(sec.get("lvl", "id"), sec.get("lvl_order", 0),
                             sec.get("lvl_parts"), opts=sec.get("lvl_opts"))
        if nlr:
            th.append(_theta0_niveaux(sec.get("lvl", "id"), nlr,
                                      sec.get("lvl_opts"), sec.get("coord")))
    return np.concatenate([np.asarray(x, dtype=np.float64) for x in th])


RHO_METRIQUES = ("exp", "gau", "iexp", "igau", "ieuc", "aexp", "agau")


def _pas_typique(v):
    """Ecart MEDIAN entre coordonnees consecutives distinctes d'un axe.

    La mediane plutot que le minimum : deux positions presque confondues (pas
    de 0.01 sur une etendue de 30) ramenaient rho0 = 0.5^(1/0.01) a l'ecretage,
    c'est-a-dire au point a gradient nul que ce depart doit eviter.
    """
    d = np.diff(np.unique(np.asarray(v, dtype=float)))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _theta0_rho_metrique(kind, nl, coord=None):
    """Depart des structures rho^d : correlation 0.5 au plus proche voisin.

    rho0 = 0.5^(1/d0) pour les noyaux exponentiels, 0.5^(1/d0^2) pour les
    gaussiens, d0 etant le pas median de l'axe (un par axe pour aexp et
    agau, le plus petit des deux pas medians pour les isotropes). rho0 est borne a
    [1e-3, 0.99] pour que theta = atanh(rho0) reste a une echelle ou le
    gradient est exploitable ; sans coordonnees, rho0 = 0.5.
    """
    th = np.full(nl, np.arctanh(0.5))
    if coord is None:
        return th
    c = np.atleast_2d(np.asarray(coord, dtype=float))
    if c.shape[0] == 1 and c.shape[1] > 1:
        c = c.T
    pas = [_pas_typique(c[:, j]) for j in range(c.shape[1])]
    gaussien = kind in ("gau", "igau", "agau")

    def rho0(d0):
        r = 0.5 ** (1.0 / (d0 ** 2 if gaussien else d0))
        return float(np.clip(r, 1e-3, 0.99))

    if kind in ("aexp", "agau"):
        for j in range(min(nl, len(pas))):
            th[j] = np.arctanh(rho0(pas[j]))
        return th
    th[0] = np.arctanh(rho0(min(pas)))
    return th


def _theta0_niveaux(kind, nl, opts=None, coord=None):
    """Depart des parametres de niveaux.

    Zero (donc rho = 0, champ non structure) pour les structures parametrees en
    tanh. Pour les structures a PORTEE le parametre est un log-distance : partir
    de zero voudrait dire "portee = 1", soit une correlation nulle des le
    premier voisin sur un dispositif dont les coordonnees vont de 1 a 40. On
    part donc du quart de l'etendue des coordonnees, la ou la vraisemblance a
    encore de la pente.
    """
    if kind == "mtrn":
        # mtrn a plusieurs parametres, chacun avec sa propre echelle. Les
        # valeurs de depart declarees par l'utilisateur (mtrn(..., phi = 3))
        # sont dans opts sous init_* ; sans elles, phi partirait de 1 et nu, qui
        # est un exposant, aussi.
        try:
            from .levels import MTRN_PARAMS, MTRN_DEFAUT
        except ImportError:
            from levels import MTRN_PARAMS, MTRN_DEFAUT
        o = opts or {}
        th, k = np.zeros(nl), 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) <= 0.5:
                continue
            v = float(o.get("init_" + p, np.nan))
            if not np.isfinite(v):
                v = MTRN_DEFAUT[p]
                if p == "phi" and coord is not None:
                    c = np.atleast_2d(np.asarray(coord, dtype=float))
                    v = max(0.25 * float(np.ptp(c, axis=0).max()), 1e-3)
            th[k] = v if p == "alpha" else np.log(max(v, 1e-8))
            k += 1
        return th
    if kind in RHO_METRIQUES:
        # Les structures en rho^d (exp, gau, iexp, igau, ieuc, aexp, agau)
        # prennent rho = |tanh(theta)| ecrete a 1e-12. Partir de theta = 0
        # mettait rho EXACTEMENT sur l'ecretage, ou le gradient est nul par
        # construction (clip actif, et sign(0) = 0 pour |.|) : L-BFGS-B
        # concluait a la convergence sans bouger, et un champ exponentiel sur
        # des positions irregulieres restait a rho = 0 quelle que soit la
        # correlation dans les donnees. Trouve le 2026-09-28 en comparant a
        # nlme::gls(corExp) : -2logL a 65 unites de l'optimum, retrouve a
        # 1e-9 pres avec un depart a chaud. On part d'une correlation de 0.5
        # au pas median de chaque axe.
        return _theta0_rho_metrique(kind, nl, coord)
    if kind not in ("sph", "cir", "lvr", "ilv"):
        return np.zeros(nl)
    etendue = 1.0
    if coord is not None:
        c = np.atleast_2d(np.asarray(coord, dtype=float))
        if c.shape[0] == 1 and c.shape[1] > 1:
            c = c.T
        etendue = float(max(np.ptp(c, axis=0).max(), 1.0))
    th = np.zeros(nl)
    th[0] = np.log(max(0.25 * etendue, 1e-3))
    return th


A_PORTEE = ("sph", "cir", "lvr")


def _positions_portee(terms, res, n):
    """Indices dans theta des parametres de PORTEE, avec leurs coordonnees.

    Suit exactement la disposition d'initial_theta : pour chaque terme, les
    parametres de structure puis ceux de niveaux ; puis chaque section de la
    residuelle de meme. Rend une liste de (indice, coord).
    """
    out, k = [], 0
    for tm in terms:
        k += n_params(tm["struct"], tm["t"], tm["rank"])
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        nl = n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                            opts=tm.get("lvl_opts"))
        if nl and kind in A_PORTEE:
            out.append((k, tm.get("coord")))
        k += nl
    for sec in res_sections(res, n):
        k += n_params(sec["struct"], sec["t"], sec["rank"])
        kind = sec.get("lvl", "id")
        nl = n_level_params(kind, sec.get("lvl_order", 0), sec.get("lvl_parts"),
                            opts=sec.get("lvl_opts"))
        if nl and kind in A_PORTEE:
            out.append((k, sec.get("coord")))
        k += nl
    return out


def _candidats_portee(coord, n_cand=9):
    """Portees candidates : quantiles des distances entre paires de positions.

    Entre deux distances consecutives du dispositif, la vraisemblance d'un
    noyau a portee est lisse ; elle a un pli a chaque distance. Balayer les
    quantiles des distances (10 % a 90 %) place un candidat dans chaque
    region ou la forme du noyau change vraiment.
    """
    if coord is None:
        return np.array([])
    c = np.atleast_2d(np.asarray(coord, dtype=float))
    if c.shape[0] == 1 and c.shape[1] > 1:
        c = c.T
    if c.shape[0] > 2000:
        rng = np.random.default_rng(0)
        c = c[rng.choice(c.shape[0], 2000, replace=False)]
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    d = d[np.triu_indices(c.shape[0], 1)]
    d = d[d > 0]
    if d.size == 0:
        return np.array([])
    q = np.quantile(d, np.linspace(0.1, 0.9, n_cand))
    return np.log(np.unique(q))


def _balayage_portee(fun_jac, th0, terms, res, n, verbose=False, fixed_idx=None,
                     fun_sc=None, bornes=None, n_garde=3, maxiter_local=10):
    """Depart des parametres de PORTEE choisi sur un balayage de -2logL.

    POURQUOI. La vraisemblance d'un noyau a portee (sph, cir, lvr) a souvent
    PLUSIEURS maxima locaux en la portee : sur le champ irregulier du test
    asreml3 B7 (90 positions), le noyau circulaire en a trois (portees 5.0,
    7.4 et 10.9 ; -2logL a 6,1 d'ecart entre le premier et le meilleur), et
    asreml comme remlax partaient dans la premiere colline. Un depart au quart
    de l'etendue ne vaut donc rien de general.

    COMMENT. (1) -2logL est evaluee sur les quantiles des distances du
    dispositif, un parametre de portee a la fois, les autres parametres a leur
    depart. (2) Comparer des candidats a variance NON ajustee ne suffit pas :
    sur un champ de 60 positions a quatre minima locaux, le candidat le mieux
    classe a variance de depart menait a la colline a 104.4, celle a 101.8
    etant un cran plus loin. Les n_garde meilleurs candidats recoivent donc
    chacun une COURTE descente L-BFGS-B (maxiter_local iterations, sur
    l'objectif mis a l'echelle et sous les memes bornes que l'ajustement), et
    l'on part du meilleur point atteint. Cout : une dizaine d'evaluations par
    candidat retenu, une fois. Ne s'applique qu'au depart AUTOMATIQUE : un
    theta_init fourni par l'appelant est respecte. Rend (theta, n_eval).
    """
    pos = _positions_portee(terms, res, n)
    if not pos:
        return th0, 0
    fixes = set(int(i) for i in (fixed_idx or []))
    th = np.array(th0, dtype=np.float64)
    n_eval = 0
    for k, coord in pos:
        if k in fixes:
            continue
        cands = _candidats_portee(coord)
        if cands.size == 0:
            continue
        cands = np.append(cands, th[k])
        vals = []
        for c in cands:
            t = th.copy()
            t[k] = c
            try:
                v = float(fun_jac(t)[0])
            except Exception:
                v = np.nan
            n_eval += 1
            vals.append(v if np.isfinite(v) else np.inf)
        vals = np.asarray(vals)
        ordre = [int(j) for j in np.argsort(vals) if np.isfinite(vals[j])][:max(int(n_garde), 1)]
        if not ordre:
            continue
        meilleur_th, meilleur_v = th.copy(), vals[-1]
        if fun_sc is None or maxiter_local <= 0:
            j = ordre[0]
            if vals[j] < meilleur_v - 1e-10:
                meilleur_th[k], meilleur_v = cands[j], vals[j]
        else:
            for j in ordre:
                t = th.copy()
                t[k] = cands[j]
                try:
                    r = minimize(fun_sc, t, jac=True, method="L-BFGS-B", bounds=bornes,
                                 options=dict(maxiter=int(maxiter_local),
                                              maxfun=20 * int(maxiter_local)))
                    v = float(fun_jac(r.x)[0])
                    n_eval += int(r.nfev) + 1
                except Exception:
                    continue
                if np.isfinite(v) and v < meilleur_v - 1e-10:
                    meilleur_th, meilleur_v = np.asarray(r.x, dtype=np.float64), v
        if verbose and meilleur_v < vals[-1] - 1e-10:
            print("  [portee] depart deplace de %.4g a %.4g (-2logL %.6f -> %.6f)"
                  % (np.exp(th[k]), np.exp(meilleur_th[k]), vals[-1], meilleur_v), flush=True)
        th = meilleur_th
    return th, n_eval


def validate(terms, res, y, X):
    """Refus explicite des dispositifs qui rendraient V singuliere.

    Chaque controle correspond a une erreur qu'on a effectivement vue, et qui
    se manifestait par un NaN silencieux plutot que par un message.
    """
    n = len(y)
    for sec in res_sections(res, n):
        _valider_section(sec, len(np.asarray(sec["trait"])))
    _valide_reste(terms, y, X, n)


def _valider_section(res, n):
    tr, un = np.asarray(res["trait"]), np.asarray(res["unit"])
    kind = res.get("lvl", "id")
    # Deux lignes de la MEME unite et du MEME caractere ont, sous n'importe
    # quelle structure residuelle, une correlation de 1 : elles sont identiques
    # dans R, qui devient singuliere, et V avec elle. Le controle ne portait que
    # sur us/diag/fa ; une residuelle iid AVEC structure entre unites
    # (residual = ~ dsum(~ ar1(col) | site), plusieurs lignes par colonne)
    # passait donc au travers et rendait -2logL = NaN, sur CPU comme sur GPU,
    # sans un mot. Trouve par le balayage de parite.
    if res["struct"] in ("us", "diag", "fa") or kind != "id":
        key = un.astype(np.int64) * (int(tr.max()) + 1) + tr.astype(np.int64)
        if len(np.unique(key)) != n:
            d = n - len(np.unique(key))
            quoi = ("structure residuelle '%s'" % res["struct"] if kind == "id"
                    else "structure '%s' entre unites" % kind)
            raise ValueError(
                "%s : %d couple(s) (unite, caractere) en double. Chaque unite ne "
                "peut etre observee qu'une fois par caractere ; sinon la matrice "
                "residuelle est singuliere et la vraisemblance vaut NaN."
                % (quoi, d))
    if res["struct"] in ("us", "diag", "fa"):
        if int(tr.max()) + 1 > res["t"]:
            raise ValueError("indice de caractere %d hors de la structure residuelle (t=%d)"
                             % (tr.max(), res["t"]))


def _valide_reste(terms, y, X, n):
    for tm in terms:
        # somme des |z| par colonne sur les triplets : la version precedente
        # formait Z dense (n x tq), 2 Go et 4 s a n = 128 000, q = 2000. Les
        # doublons de triplets s'additionnent, comme dans Zc[zi, zj] += zx.
        ncol = tm["t"] * tm["q"]
        zj = np.asarray(tm["zj"], dtype=np.int64); zx = np.asarray(tm["zx"], dtype=np.float64)
        if zj.size == 0:
            raise ValueError("terme '%s' : incidence entierement nulle." % tm["name"])
        zi = np.asarray(tm["zi"], dtype=np.int64)
        cle = zi * ncol + zj
        u, inv = np.unique(cle, return_inverse=True)
        par_cellule = np.abs(np.bincount(inv, weights=zx, minlength=len(u)))
        if not np.any(par_cellule > 0):
            raise ValueError("terme '%s' : incidence entierement nulle." % tm["name"])
    if not np.all(np.isfinite(y)):
        raise ValueError("y contient des valeurs non finies.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des valeurs non finies.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("X est de rang %d pour %d colonnes."
                         % (np.linalg.matrix_rank(X), X.shape[1]))


# gradient relatif au-dela duquel un ajustement MME est repris par le moteur
# dense (les ajustements convergés sont sous 1e-6, SEUIL_GRAD_REL). 1e-3 etait
# trop laxiste : sur mtigee8 (208 parametres, bloc us(16) de rang incomplet), les
# trois voies MME s'arretaient a 2-5e-4, 0,13 a 0,20 unite de logL sous
# l'optimum, sans declencher la reprise (banc IEE, 2026-10-23).
GRAD_REL_REPLI_DENSE = 1e-5
N_MAX_REPLI_DENSE = 20000


def effets_denses(theta, terms, Zs, res, y_fit, X, blups=True, pev=False,
                  V=None, z_de=None):
    """beta, vbeta, Vi, P y, BLUP et PEV a theta donne, sur le moteur dense.

    Extrait de fit_reml pour servir aussi a chaque tirage de l'echantillonneur
    bayesien (bayes.py), qui en tire les moments a posteriori de beta et des
    effets aleatoires. Rend un dict avec les memes cles que fit_reml.

    V et z_de servent au mode econome (econome.py) : V deja assemblee (sur
    l'hote), et z_de(k) qui rend l'incidence pleine hauteur du terme k a la
    demande, au lieu de la liste Zs gardee sur le peripherique.
    """
    n = len(y_fit)
    th_terms, _, _ = split_theta(jnp.asarray(theta), terms, res)
    out = {}
    if V is None:
        V = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    else:
        V = np.asarray(V)
    # inv() leve "Singular matrix" des que V est numeriquement singuliere,
    # ce qui detruirait un ajustement par ailleurs valide (une variance a
    # zero suffit). On resout, et on retombe sur la pseudo-inverse en
    # signalant que beta et les BLUPs sont alors conditionnels.
    try:
        Vi = np.linalg.solve(V, np.eye(n))
        out["V_singuliere"] = False
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V, rcond=1e-12)
        out["V_singuliere"] = True
    XtVi = X.T @ Vi
    A = XtVi @ X
    try:
        beta = np.linalg.solve(A, XtVi @ y_fit)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ (XtVi @ y_fit)
    out["beta"] = beta
    # (X'V^-1X)^-1 : matrice de covariance des effets fixes A V CONNUE.
    # Necessaire a predict() et point de depart de l'ajustement de
    # Kenward-Roger ; la recalculer plus tard exigerait de refaire V.
    out["vbeta"] = np.linalg.pinv(A)
    out["Vi"] = Vi
    resid = y_fit - X @ beta
    Py = Vi @ resid - Vi @ X @ (np.linalg.pinv(A) @ (XtVi @ resid))
    out["Py"] = Py
    # QUELS TERMES. `pev=True` la calcule pour TOUS les termes, ce qui est
    # une mauvaise valeur par defaut sur un modele reel : le modele IGE
    # complet a 23 termes dont 21 sont des nuisances — effets spatiaux de
    # bloc et effets environnementaux indirects — et personne ne veut la
    # variance d'erreur de prediction d'un effet de bloc. Chacun coute
    # pourtant une matrice (t*q) x n, et le total a tue le processus par
    # manque de memoire lors du premier essai reel. `pev` accepte donc une
    # LISTE DE NOMS, et le pipeline ne demande que les termes genetiques.
    pev_noms = ([tm["name"] for tm in terms] if pev is True
                else ([] if not pev else list(pev)))
    inconnus = set(pev_noms) - {tm["name"] for tm in terms}
    if inconnus:
        raise ValueError("pev : terme(s) inconnu(s) %s ; termes du modele : %s"
                         % (sorted(inconnus), [tm["name"] for tm in terms]))
    if blups and pev_noms:
        # P, formee UNE fois : elle ne depend pas du terme. C'est la matrice
        # dont depend toute la PEV, et la former dans la boucle la
        # recalculerait pour chaque terme.
        Pmat = Vi - (Vi @ X) @ out["vbeta"] @ (X.T @ Vi)
    if blups:
        # u_k = (Sigma_k (x) K_k) Z_k' P y = B_k B_k' P y au facteur pres :
        # on passe par B pour ne jamais former Sigma (x) K.
        out["blups"] = {}
        for k, tm in enumerate(terms):
            Zk = Zs[k] if z_de is None else z_de(k)
            B = np.asarray(term_factor(jnp.asarray(th_terms[k]), tm, Zk))  # noqa: F841
            # u = (Sigma (x) K) Z' Py ; avec B = Z (L (x) L_K) on a
            # (Sigma (x) K) Z' = (L (x) L_K) B'
            bt = B.T @ Py
            ns = n_params(tm["struct"], tm["t"], tm["rank"])
            Ls = np.asarray(build_sigma(jnp.asarray(th_terms[k][:ns]), tm["struct"],
                                        tm["t"], tm["rank"]))
            # K effective : peut dependre de theta (AR1), d'ou la
            # reconstruction plutot qu'un simple tm["LK"].
            kind_ = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
            from_lvl = (jnp.asarray(tm["LK"]) if kind_ == "fixed" and tm["LK"] is not None
                        else level_chol(jnp.asarray(th_terms[k][ns:]), kind_,
                                        tm["q"], dims=tm.get("dims"),
                                        LK_fixed=tm.get("LK"),
                                        order=tm.get("lvl_order", 0),
                                        coord=tm.get("coord"),
                                        parts=tm.get("lvl_parts"),
                                        opts=tm.get("lvl_opts"),
                                        expr=tm.get("lvl_expr")))
            Kq = (np.eye(tm["q"]) if from_lvl is None
                  else np.asarray(from_lvl) @ np.asarray(from_lvl).T)
            Zd = np.asarray(Zk)
            G = np.kron(Ls, Kq)
            u = G @ (Zd.T @ Py)
            out["blups"][tm["name"]] = u.reshape(tm["t"], tm["q"]).T
            if tm["name"] in pev_noms:
                # VARIANCE D'ERREUR DE PREDICTION, diagonale seulement.
                #
                #     var(u - u_chapeau) = G - G Z' P Z G
                #
                # avec P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1, deja formable
                # ici puisque Vi et vbeta sont sous la main. On ne rend que
                # la DIAGONALE : c'est tout ce que demande une fiabilite ou
                # une heritabilite de Cullis, et la matrice pleine pese
                # (t*q)^2.
                #
                # POURQUOI CETTE SORTIE EXISTE. Sans elle H2_Cullis n'est
                # pas calculable, et la suite de validation contre asreml
                # l'avait releve comme la seule sortie que le logiciel de
                # reference donne et que remlax ne donnait pas.
                M = G @ Zd.T                      # (t*q) x n
                d_ = np.diag(G) - np.einsum("ij,ij->i", M @ Pmat, M)
                out.setdefault("pev", {})[tm["name"]] = \
                    np.maximum(d_, 0.0).reshape(tm["t"], tm["q"]).T
    return out


_BLAS_REGLE = [False]        # contexte BlasNumpyUnFil deja ouvert


def fit_reml(terms, res, y, X, *args, poids=None, **kwargs):
    """Ajustement REML (voir _fit_reml_coeur pour les options).

    poids : poids a priori w_i > 0 des observations (None : aucun). La residuelle
      devient R = W^-1/2 R0 W^-1/2, W = diag(w), R0 la residuelle declaree. Avec
      une residuelle iid dont la variance est fixee a 1 (fixed_idx), R = diag(1/w) :
      variances residuelles connues, comme dans une analyse en deux etapes.

    COMMENT. Multiplier les lignes de y, X et Z par sqrt(w_i) ramene exactement a
    la residuelle R0, et aucun moteur n'a a connaitre les poids. Seule la
    vraisemblance change, d'une constante : -2logL(y) = -2logL(y*) - somme log w_i.
    beta, BLUP, PEV et Hessien sont inchanges ; P y et V^-1 sont ramenes a
    l'echelle d'origine (P y = W^1/2 P* y*, V^-1 = W^1/2 V*^-1 W^1/2).
    """
    # CHOIX AUTOMATIQUE (moteur = "auto", optimizer = "auto") : moteur MME et
    # AI-REML quand le modele est dans le perimetre du MME, sinon moteur dense
    # et L-BFGS-B (econome au-dela de n = 5000). C'est la combinaison la plus
    # rapide au banc (DROPS, Krause) et celle d'asreml.
    if kwargs.get("moteur") == "auto" or kwargs.get("optimizer") == "auto":
        kwargs = dict(kwargs)
        try:
            from .mme import hors_perimetre as _hp
        except ImportError:
            from mme import hors_perimetre as _hp
        dans_mme = _hp(terms, res, kwargs.get("lag")) is None
        if kwargs.get("moteur") == "auto":
            kwargs["moteur"] = "mme" if dans_mme else "dense"
            if kwargs["moteur"] == "dense" and len(y) > 5000 and kwargs.get("memoire", "standard") == "standard":
                kwargs["memoire"] = "econome"
        if kwargs.get("optimizer") == "auto":
            kwargs["optimizer"] = "ai" if kwargs["moteur"] == "mme" else "lbfgsb"
    if kwargs.get("moteur") == "mme" and not _BLAS_REGLE[0]:
        # moteur MME : produits NumPy minces, gros calcul en LAPACK (SciPy) ;
        # les deux OpenBLAS ne doivent pas se disputer les coeurs (device.py)
        try:
            from .device import BlasNumpyUnFil
        except ImportError:
            from device import BlasNumpyUnFil
        _BLAS_REGLE[0] = True
        try:
            with BlasNumpyUnFil():
                return fit_reml(terms, res, y, X, *args, poids=poids, **kwargs)
        finally:
            _BLAS_REGLE[0] = False
    if poids is None:
        return _fit_reml_coeur(terms, res, y, X, *args, **kwargs)
    w = np.asarray(poids, dtype=np.float64).ravel()
    if w.shape[0] != len(y) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("poids : %d valeurs finies et > 0 attendues" % len(y))
    sw = np.sqrt(w)
    y2 = np.asarray(y, dtype=np.float64) * sw
    X2 = np.asarray(X, dtype=np.float64) * sw[:, None]
    terms2 = [dict(tm, zx=np.asarray(tm["zx"], dtype=np.float64)
                   * sw[np.asarray(tm["zi"], dtype=np.int64)]) for tm in terms]
    out = _fit_reml_coeur(terms2, res, y2, X2, *args, **kwargs)
    c = -float(np.sum(np.log(w)))                  # constante de -2logL
    for k in ("neg2_reml",):
        if k in out:
            out[k] = out[k] + c
    for k in ("logLik", "logLik_asreml"):
        if k in out:
            out[k] = out[k] - 0.5 * c
    if "Py" in out:
        out["Py"] = np.asarray(out["Py"]) * sw
    if "Vi" in out:
        out["Vi"] = np.asarray(out["Vi"]) * sw[:, None] * sw[None, :]
    out["poids"] = True
    return out


def _fit_reml_coeur(terms, res, y, X, theta_init=None, maxiter=3000,
             floor=-12.0, ceil=12.0, verbose=True, hessian=True,
             blups=True, pev=False, tol=1e-14, polish=25, check=True,
             n_restarts=0, restart_sd=0.5, seed=0, fixed_idx=None, lag=None,
             optimizer="lbfgsb", moteur="dense", inversion="auto",
             memoire="standard", finition="newton", arret_lent=(100, 0.05)):
    """Ajustement REML.

    finition : "newton" (defaut : L-BFGS-B jusqu'a sa tolerance, puis polissage
      de Newton) ou "rc" (module finition.py) : L-BFGS-B s'arrete des que son
      gain sur les arret_lent[0] dernieres iterations tombe sous arret_lent[1]
      (unites de -2logL), puis un Newton a region de confiance finit, en
      traitant directions plates et variances qui s'eteignent.

    memoire : "standard" (defaut) ou "econome" (moteur dense seulement, module
      econome.py). Meme vraisemblance, meme gradient, meme matrice AI ; le mode
      econome garde moins de matrices n x n et aucune incidence pleine hauteur
      sur le peripherique. Sans effet sur le moteur MME.

    optimizer : "lbfgsb" (defaut) ou "ai" (AI-REML amorti, module ai.py). Le
      polissage de Newton, les diagnostics et les sorties sont les memes.

    moteur : "dense" (V formee, defaut) ou "mme" (equations du modele mixte,
      module mme.py ; inversion "auto", "dense" ou "creuse"). Les deux rendent
      la meme vraisemblance en theta ; un modele hors du perimetre du moteur MME
      (residuelle none, rr, decalage) bascule vers le dense avec un message.
      "blocs" (module blocs.py) : decoupage automatique en blocs relies par des
      termes globaux, V = A + U U' traitee bloc par bloc (vmap) ; meme
      vraisemblance. Sans decoupage, ou hors perimetre, bascule vers le moteur
      dense econome avec un message. Optimiseur L-BFGS-B seulement.

    lag : decalage spatial (lag.py). theta porte alors en queue les entrees
      libres de Psi, l'objectif est le critere structurel, et beta, les BLUP et
      les tests portent sur y* = (I - M) y au point final.

    n_restarts : redemarrages depuis un point perturbe, le meilleur optimum est
      conserve. C'est le SEUL moyen de detecter un optimum local : le decrement
      de Newton ne le voit pas — il mesure la montee disponible LOCALEMENT, et
      vaut donc zero au sommet d'une colline secondaire. Mesure sur 150
      configurations tirees au hasard : 2 avaient un meilleur optimum ailleurs,
      a 0,009 et 0,09 de -2logL.
    """
    n = len(y)
    if check:
        validate(terms, res, y, X)
    lag = prepare_lag(lag, y, X)
    if n_restarts > 0:
        base = fit_reml(terms, res, y, X, theta_init=theta_init, maxiter=maxiter,
                        floor=floor, ceil=ceil, verbose=verbose, hessian=hessian,
                        blups=blups, pev=pev, tol=tol, polish=polish, check=False,
                        moteur=moteur, inversion=inversion, memoire=memoire, finition=finition, arret_lent=arret_lent,
                        fixed_idx=fixed_idx, lag=lag, optimizer=optimizer)
        rng = np.random.default_rng(seed)
        best, n_better = base, 0
        for i in range(int(n_restarts)):
            th = np.clip(base["theta"] + rng.normal(0, restart_sd, len(base["theta"])),
                         floor, ceil)
            try:
                if fixed_idx:
                    th[np.array(fixed_idx)] = base["theta"][np.array(fixed_idx)]
                cand = fit_reml(terms, res, y, X, theta_init=th, maxiter=maxiter,
                                floor=floor, ceil=ceil, verbose=False, hessian=hessian,
                                blups=blups, pev=pev, tol=tol, polish=polish, check=False,
                                moteur=moteur, inversion=inversion, memoire=memoire, finition=finition, arret_lent=arret_lent,
                                fixed_idx=fixed_idx, lag=lag, optimizer=optimizer)
            except Exception:
                continue
            if np.isfinite(cand["neg2_reml"]) and cand["neg2_reml"] < best["neg2_reml"] - 1e-8:
                best, n_better = cand, n_better + 1
        best["n_restarts"] = int(n_restarts)
        best["restart_gain"] = float(base["neg2_reml"] - best["neg2_reml"])
        best["restart_better"] = n_better
        if verbose and n_better:
            print("  [restarts] %d redemarrage(s) ont trouve mieux : gain %.4g en -2logL"
                  % (n_better, best["restart_gain"]), flush=True)
        return best
    M_mme = None
    if moteur == "mme" and lag is None and res.get("struct") != "none":
        # PLANCHER DES VARIANCES. Le moteur MME utilise Sigma^-1 : une variance a
        # exp(2 x -12) = 4e-11 rendrait C trop mal conditionnee pour que log|C|
        # garde ses chiffres. Plancher a 1e-8 fois la variance residuelle des
        # moindres carres.
        _res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        _vy = float(_res @ _res) / max(n - X.shape[1], 1)
        floor = max(floor, 0.5 * float(np.log(1e-8 * _vy)))
    if moteur == "mme":
        try:
            from .mme import make_fga_mme, hors_perimetre
        except ImportError:
            from mme import make_fga_mme, hors_perimetre
        raison = hors_perimetre(terms, res, lag)
        if raison:
            if any(tm.get("Kinv") is not None and tm.get("LK") is None for tm in terms):
                raise ValueError("moteur MME hors perimetre (%s), et le moteur dense ne sait pas "
                                 "utiliser une precision Kinv sans facteur de K : il ignorerait la "
                                 "parente. Fournir K." % raison)
            if verbose:
                print("  [moteur] MME hors perimetre (%s) -> moteur dense" % raison, flush=True)
            moteur = "dense"
    if moteur == "blocs":
        try:
            from .blocs import hors_perimetre as hp_blocs, PlanBlocs, make_objective_blocs
        except ImportError:
            from blocs import hors_perimetre as hp_blocs, PlanBlocs, make_objective_blocs
        raison = hp_blocs(terms, res, lag)
        plan_blocs = None
        if raison is None:
            plan_blocs = PlanBlocs(terms, res, y, X, verbose=verbose)
            if plan_blocs.n_blocs < 2:
                raison = "aucun decoupage en blocs (un seul bloc)"
        if raison is None and optimizer in ("ai", "ai_theta"):
            raison = "optimiseur AI non pris en charge par le moteur par blocs"
        if raison:
            if verbose:
                print("  [moteur] blocs : %s -> moteur dense econome" % raison, flush=True)
            moteur, memoire = "dense", "econome"
    if moteur == "mme":
        fga_mme, fun_jac, M_mme = make_fga_mme(terms, res, y, X, inversion=inversion,
                                               verbose=verbose)
        sc = float(n)
        Zs = None

        def fun_sc(theta, _fj=fun_jac, _sc=sc):
            v_, g_ = _fj(theta)
            return v_ / _sc, g_ / _sc
    elif moteur == "blocs":
        fun_sc, fun_jac, sc = make_objective_blocs(plan_blocs)
        Zs = None
    else:
        if memoire == "econome" and lag is not None:
            if verbose:
                print("  [memoire] mode econome sans decalage spatial -> mode standard",
                      flush=True)
            memoire = "standard"
        if memoire == "econome":
            try:
                from .econome import PlanEconome, make_objective_econome
            except ImportError:
                from econome import PlanEconome, make_objective_econome
            plan_eco = PlanEconome(terms, res, y, X)
            fun_sc, fun_jac, sc = make_objective_econome(plan_eco)
            Zs = None
            if verbose:
                print("  [memoire] econome : %s" % plan_eco.resume(), flush=True)
        elif memoire == "standard":
            fun_sc, fun_jac, sc, Zs = make_objective(terms, res, y, X, lag=lag)
        else:
            raise ValueError("memoire : 'standard' ou 'econome', pas %r" % (memoire,))
    p_cov = n_theta(terms, res)
    if theta_init is not None:
        th0 = np.array(theta_init, dtype=np.float64)
        # ENTREES MANQUANTES (NaN) : depart automatique pour celles-la. Fixer un
        # seul parametre (la residuelle a 1 d'un modele pondere) n'oblige plus
        # a un premier appel du solveur pour connaitre les autres valeurs.
        if np.isnan(th0).any():
            _ti = initial_theta(terms, res, y, X)
            if th0.shape[0] == _ti.shape[0]:
                th0 = np.where(np.isnan(th0), _ti, th0)
    else:
        th0 = initial_theta(terms, res, y, X)
        if lag is not None:
            # Psi part de `init` (unites de W) quand il est fourni, de 0 sinon :
            # 0 est toujours dans la region admissible (A = I).
            ini = np.asarray(lag.get("init", np.zeros((lag["q"], lag["q"]))),
                             dtype=np.float64).reshape(lag["q"], lag["q"])
            ini = np.where(np.isfinite(ini), ini, 0.0)
            th0 = np.concatenate([th0, theta_from_psi(ini, lag)])
    p = p_cov + lag_n_params(lag)
    if th0.shape[0] != p:
        raise ValueError("theta initial de longueur %d, attendu %d" % (th0.shape[0], p))
    if lag is not None:
        _ok = bool(lag_terms(jnp.asarray(th0[p_cov:]), lag, X)[2])
        if not _ok:
            raise ValueError(
                "Psi initial hors de la region admissible : I - M est singuliere ou de "
                "determinant negatif (pas d'equilibre). Partir de Psi = 0 ou d'une valeur "
                "plus petite.")
    if res.get("struct") == "none" and moteur == "dense":
        # SANS RESIDUELLE, V n'est definie positive que si les termes couvrent
        # chaque observation d'une variance propre. Le dire ici, en clair,
        # plutot que de laisser la Cholesky rendre NaN et l'optimiseur refuser
        # un point de depart "non fini".
        if memoire == "econome":
            try:
                from .econome import V_hote
            except ImportError:
                from econome import V_hote
            V0 = V_hote(plan_eco, th0)
        else:
            V0 = np.asarray(assemble_V(jnp.asarray(th0), terms, Zs, res, n))
        try:
            np.linalg.cholesky(V0)
        except np.linalg.LinAlgError:
            w = np.linalg.eigvalsh(0.5 * (V0 + V0.T))
            raise ValueError(
                "residuelle 'none' : V n'est pas definie positive au point de depart "
                "(plus petite valeur propre %.3g, %d valeur(s) propre(s) <= 0 sur %d). "
                "Sans residuelle, les termes doivent donner a CHAQUE observation une "
                "variance propre : par exemple un terme de plante dont les incidences "
                "sont [I place sur le caractere t, W_t place sur le caractere t], "
                "structure us sur 2q colonnes." % (w.min(), int((w <= 0).sum()), n))

    # ------------------------------------------------------------------
    # COMBIEN COUTE LA COMPILATION, ET COMBIEN UNE EVALUATION
    # ------------------------------------------------------------------
    # XLA compile le programme au PREMIER appel. Le premier chrono paie donc
    # compilation + evaluation, le second l'evaluation seule, et leur difference
    # est la compilation. C'est la seule facon de l'obtenir : le compilateur ne
    # la rapporte pas, et le solveur etant invoque hors processus depuis R, on
    # ne peut pas la mesurer de l'exterieur en comparant deux appels.
    #
    # POURQUOI CELA MERITE UNE EVALUATION SUPPLEMENTAIRE. La compilation est
    # payee UNE FOIS et ne se divise pas par le nombre d'iterations : sans la
    # separer, un ajustement court se voit attribuer un cout unitaire qui est
    # surtout du temps de compilation. Mesure sur parente dense, elle valait
    # jusqu'a 66 % du temps total. Le surcout est d'une evaluation sur plusieurs
    # dizaines a plusieurs centaines.
    #
    # UNE SEULE EVALUATION DE MESURE, ET ELLE SERT. La version precedente en
    # faisait une seconde en un point perturbe, pour la seule mesure : une
    # evaluation perdue par ajustement (60 s sur mvD7 en MME creux). Le temps
    # d'une evaluation est maintenant lu sur la deuxieme evaluation de
    # l'optimiseur, et la premiere, au point de depart, est gardee en memoire :
    # L-BFGS-B commence par ce meme point et la retrouve sans la recalculer.
    _chrono("preparation")
    _t = time.time()
    _v0 = fun_jac(th0)
    try:
        import jax as _jax
        _jax.block_until_ready(_v0[0])
    except Exception:
        pass
    _t_premier = time.time() - _t
    _memo = {"fj": (np.array(th0, dtype=np.float64).tobytes(), _v0)}
    _chrono("compilation")

    # ANNONCER LA COMPILATION DES QU'ELLE EST FINIE. Sur le modele multivarie
    # complet de Salas et al. (2026), XLA compile un programme portant 5,47 Go de
    # constantes : cette phase peut durer longtemps, et pendant ce temps un
    # journal muet ne permet pas de distinguer "il compile" de "il est bloque".
    # Cette ligne est le premier signe de vie du solveur.
    if verbose:
        print("  [compilation] premier appel (compilation + une evaluation) %.1f s | %d parametres"
              % (_t_premier, p), flush=True)

    hist = {"n": 0, "t0": time.time(), "n_eval": 0,
            # DERNIERE VALEUR VUE, pour que le rapport soit GRATUIT. La version
            # precedente rappelait fun_jac() juste pour imprimer : une
            # evaluation entiere par ligne, pour une information deja calculee.
            "v": float("nan"), "gmax": float("nan")}

    # COMPTER LES EVALUATIONS PLUTOT QUE LES PREDIRE. La reconstruction
    # compilation + n_eval x cout_unitaire = total echouait d'un facteur deux
    # (rapport median 0,48 sur 32 cellules) parce que je predisais le nombre
    # d'evaluations par n_iter + 2p. Or L-BFGS-B en fait PLUSIEURS par iteration
    # dans sa recherche lineaire, et chaque pas de polissage ajoute un Hessien
    # complet, soit 2p de plus. Le compte reel est la seule facon honnete de
    # fermer le bilan : un facteur d'ajustement masquerait le poste manquant.
    _fun_jac_brut = fun_jac
    hist["t_eval"] = []                 # durees des evaluations (hors la premiere)

    def _evaluer_memo(cle_memo, th, brut, facteur):
        cle = np.asarray(th, dtype=np.float64).tobytes()
        m_ = _memo.get(cle_memo)
        if m_ is not None and m_[0] == cle:
            return m_[1]
        if cle_memo == "sc":
            m_ = _memo.get("fj")
            if m_ is not None and m_[0] == cle:
                v_, g_ = m_[1]
                return float(v_) / facteur, np.asarray(g_, dtype=np.float64) / facteur
        t_ = time.time()
        hist["n_eval"] += 1
        r_ = brut(th)
        if len(hist["t_eval"]) < 50:
            hist["t_eval"].append(time.time() - t_)
        _memo[cle_memo] = (cle, r_)
        return r_

    def fun_jac(th):
        v, g = _evaluer_memo("fj", th, _fun_jac_brut, 1.0)
        hist["v"], hist["gmax"] = float(v), float(np.max(np.abs(g)))
        return v, g

    _fun_sc_brut = fun_sc

    def fun_sc(th):
        v, g = _evaluer_memo("sc", th, _fun_sc_brut, sc)
        # REMISE A L'ECHELLE. L-BFGS-B minimise -2logL / sc (cf. make_objective) ;
        # imprimer sa valeur brute donnerait un nombre sans signification pour le
        # lecteur. C'est fun_sc que l'optimiseur appelle, donc c'est ici qu'il
        # faut retenir la valeur — la retenir dans fun_jac laisserait NaN.
        hist["v"], hist["gmax"] = float(v) * sc, float(np.max(np.abs(g))) * sc
        return v, g

    lent = dict(actif=(finition == "rc" and optimizer == "lbfgsb"), f=[])

    def cb(_):
        hist["n"] += 1
        # ARRET SUR PROGRES LENT (finition "rc") : la longue traine de L-BFGS-B
        # est laissee au Newton a region de confiance (finition.py).
        if lent["actif"]:
            lent["f"].append(hist["v"])
            k_, seuil_ = arret_lent
            if len(lent["f"]) > k_ and lent["f"][-k_ - 1] - lent["f"][-1] < seuil_:
                hist["arret_lent"] = True
                raise StopIteration
        # LE PAS D'IMPRESSION S'ADAPTE AU COUT MESURE D'UNE ITERATION. Un pas
        # fixe de 25 rendait le journal MUET sur le modele reel, ou une
        # evaluation coute plusieurs secondes : 52 minutes sans une ligne, et
        # aucun moyen de distinguer la progression d'un blocage. Le seuil porte
        # sur le temps MESURE des evaluations deja faites, et non suppose.
        if not verbose:
            return
        te_ = float(np.median(hist["t_eval"])) if hist["t_eval"] else _t_premier
        pas = 1 if te_ > 0.25 else (10 if te_ > 0.02 else 50)
        if hist["n"] == 1 or hist["n"] % pas == 0:
            print("  iter %4d | -2logL %14.6f | max|grad| %.3e | %6.1f s | %d eval"
                  % (hist["n"], hist["v"], hist["gmax"],
                     time.time() - hist["t0"], hist["n_eval"]), flush=True)

    # L-BFGS-B travaille sur l'objectif MIS A L'ECHELLE (cf. make_objective).
    # Parametres FIXES (le code F d'asreml) : on les borne a leur valeur de
    # depart. Les borner plutot que de les retirer du vecteur garde toute
    # l'algebre inchangee ; le diagnostic les verra "a une borne" et les
    # exclura du sous-espace libre, ce qui est exactement le comportement
    # voulu pour les degres de liberte.
    bornes = [(floor, ceil)] * p
    th0 = np.array(th0, dtype=np.float64)          # copie : th0 peut etre theta_init de l'appelant
    for (o_, ln_, B_) in bornes_loadings(terms, res, y):
        for j in range(o_, o_ + ln_):
            bornes[j] = (-B_, B_)
            th0[j] = float(np.clip(th0[j], -B_, B_))
    if fixed_idx:
        for j in fixed_idx:
            bornes[j] = (float(th0[j]), float(th0[j]))
    # maxiter = 0 SIGNIFIE « n'optimise pas », et il faut le court-circuiter
    # explicitement. Passer maxiter=0 a L-BFGS-B ne rend PAS le point de depart :
    # mesure sur un champ ar1, theta ressortait deplace de 1,6e-2 vers l'optimum
    # (maxfun = 10 * 0 = 0 n'est pas interprete comme « aucune evaluation »).
    #
    # POURQUOI CELA COMPTE. Avec --theta-in, maxiter = 0 et polish = 0, le
    # solveur est censé EVALUER la vraisemblance au theta impose : c'est l'usage
    # meme du depart a chaud, et le seul moyen de distinguer deux solveurs qui
    # calculent une fonction differente de deux solveurs qui s'arretent
    # ailleurs. Un deplacement silencieux invalide exactement cette comparaison,
    # en rendant une valeur plus haute que celle demandee — d'autant plus haute
    # qu'on est loin de l'optimum. C'est ce qui faisait apparaitre un ecart de
    # 8e-2 entre le moteur dense et le moteur creux la ou un calcul REML
    # independant, en algebre dense, donnait EXACTEMENT la valeur du creux.
    if int(maxiter) <= 0:
        theta = np.asarray(th0, dtype=np.float64)
    else:
        # DEPART DES PORTEES SUR BALAYAGE (cf. _balayage_portee) : seulement
        # pour le depart automatique, et hors parametres fixes.
        if theta_init is None:
            th0, _ = _balayage_portee(fun_jac, th0, terms, res, n, verbose=verbose,
                                      fixed_idx=fixed_idx, fun_sc=fun_sc, bornes=bornes)
        # LE POINT DE DEPART DOIT AVOIR UNE VALEUR ET UN GRADIENT FINIS. Avec
        # un gradient NaN, L-BFGS-B rend le point de depart a l'iteration 0 en
        # se declarant converge, et rien ne le distinguait d'un ajustement
        # reussi (cir, 2026-09-28). On refuse explicitement.
        _v0, _g0 = fun_jac(th0)
        if not np.isfinite(float(_v0)) or not np.all(np.isfinite(np.asarray(_g0))):
            raise ValueError(
                "-2logL ou son gradient n'est pas fini au point de depart "
                "(valeur %r, %d composante(s) du gradient non finie(s)) : "
                "l'optimiseur ne peut pas demarrer. Verifier le noyau ou fournir "
                "theta_init." % (float(_v0), int(np.sum(~np.isfinite(np.asarray(_g0))))))
        # callback=cb : SANS LUI LE RAPPEL EST DU CODE MORT. Il existait, avait
        # l'air de rapporter la progression, et n'a jamais ete passe ici — d'ou
        # 52 minutes d'ajustement sans une ligne sur le modele reel, et aucun
        # moyen de distinguer la progression d'un blocage.
        if optimizer in ("ai", "ai_theta"):
            # AI-REML : meme objectif et meme gradient, courbure par l'information
            # moyenne (ai.py). Le modele structurel n'est pas couvert.
            if lag is not None:
                raise ValueError("optimizer='ai' : modele structurel (lag) non pris en charge.")
            try:
                from .ai import make_fga, ai_reml_fga
            except ImportError:
                from ai import make_fga, ai_reml_fga
            from types import SimpleNamespace
            # UNE factorisation par point : valeur, gradient et matrice AI ensemble.
            if moteur == "mme":
                fga = fga_mme
            elif memoire == "econome":
                try:
                    from .econome import make_fga_econome
                except ImportError:
                    from econome import make_fga_econome
                fga = make_fga_econome(plan_eco)
            else:
                fga = make_fga(terms, res, y, X)
            if optimizer == "ai":
                try:
                    from .ai import ai_reml_var
                except ImportError:
                    from ai import ai_reml_var
                # valeur seule pour les essais de pas : MME sans derivees ; ailleurs
                # valeur et gradient (sans la matrice AI)
                if moteur == "mme":
                    f_seul = lambda th_: M_mme.evaluer(np.asarray(th_), complet=False)["f"]
                else:
                    f_seul = lambda th_: float(fun_jac(th_)[0])
                th_ai, info = ai_reml_var(fga, th0, bornes, terms, res, maxiter=maxiter,
                                          verbose=verbose, compteur=hist,
                                          plancher_us=1e-8 if moteur == "mme" else 1e-10,
                                          fun_sc=fun_sc, f_seul=f_seul)
                if not info["success"]:
                    # REPLI : L-BFGS-B repris du point atteint. AI en echelle des
                    # variances cale quand l'optimum est de rang incomplet (bord
                    # du cone d'une Sigma us) ; L-BFGS-B, en theta de Cholesky,
                    # y glisse a un gradient par iteration. MESURE sur us9 a
                    # n = 500 : le polissage de Newton y arrivait aussi, mais en
                    # 4665 evaluations (Hessien par differences finies).
                    if verbose:
                        print("  [ai] %s -> L-BFGS-B depuis ce point" % info["message"],
                              flush=True)
                    r2 = minimize(fun_sc, th_ai, jac=True, method="L-BFGS-B",
                                  bounds=bornes, callback=cb,
                                  options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                               ftol=tol, gtol=1e-10))
                    th_ai = np.asarray(r2.x, dtype=np.float64)
                    info = dict(nit=info["nit"] + int(r2.nit), success=False,
                                message="%s ; repli L-BFGS-B : %s" % (info["message"], r2.message))
            else:
                # "ai_theta" : le pas en theta de Cholesky, variante anterieure,
                # gardee pour comparaison.
                th_ai, info = ai_reml_fga(fga, th0, bornes, maxiter=maxiter,
                                          verbose=verbose, compteur=hist)
            r = SimpleNamespace(x=th_ai, **info)
            # SORTIE DU PLANCHER apres l'AI aussi (cf. _evasion_plancher)
            if optimizer == "ai":
                th_ref = np.array(th0, dtype=np.float64)
                _ti = initial_theta(terms, res, y, X)
                th_ref[:len(_ti)] = np.maximum(th_ref[:len(_ti)], _ti)
                for _tour in range(2):
                    th_ev = _evasion_plancher(fun_jac, np.asarray(r.x, dtype=np.float64), th_ref,
                                              bornes, terms, res, floor, verbose=verbose)
                    if th_ev is None:
                        break
                    th_ai, info2 = ai_reml_var(fga, th_ev, bornes, terms, res, maxiter=maxiter,
                                               verbose=verbose, compteur=hist,
                                               plancher_us=1e-8 if moteur == "mme" else 1e-10,
                                               fun_sc=fun_sc, f_seul=f_seul)
                    r = SimpleNamespace(x=th_ai, nit=int(getattr(r, "nit", 0)) + int(info2["nit"]),
                                        success=info2["success"],
                                        message="%s ; apres sortie du plancher : %s"
                                        % (getattr(r, "message", ""), info2["message"]))
        elif optimizer == "lbfgsb":
            r = minimize(fun_sc, th0, jac=True, method="L-BFGS-B",
                         bounds=bornes, callback=cb,
                         options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                      ftol=tol, gtol=1e-10))
            if hist.get("arret_lent") and verbose:
                print("  [finition] L-BFGS-B arrete a l'iteration %d : gain < %g sur %d iterations"
                      % (hist["n"], arret_lent[1], arret_lent[0]), flush=True)
            # SORTIE DU PLANCHER (cf. _evasion_plancher) : une variance poussee
            # pres du plancher en log n'en remonte pas, son gradient y est nul.
            # Reference de remontee : le depart fourni, ou le depart automatique
            # s'il est plus haut (un theta_init lui-meme au plancher ne dit rien).
            th_ref = np.array(th0, dtype=np.float64)
            _ti = initial_theta(terms, res, y, X)
            th_ref[:len(_ti)] = np.maximum(th_ref[:len(_ti)], _ti)
            for _tour in range(3):
                th_ev = _evasion_plancher(fun_jac, np.asarray(r.x, dtype=np.float64), th_ref,
                                          bornes, terms, res, floor, verbose=verbose)
                if th_ev is None:
                    break
                nit_av = int(r.nit)
                r = minimize(fun_sc, th_ev, jac=True, method="L-BFGS-B",
                             bounds=bornes, callback=cb,
                             options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                          ftol=tol, gtol=1e-10))
                r.nit = nit_av + int(r.nit)
        else:
            raise ValueError("optimizer inconnu : %r (lbfgsb, ai, ai_theta)" % (optimizer,))
        theta = np.asarray(r.x, dtype=np.float64)
    t_optim = time.time() - hist["t0"]
    _chrono("optimisation")
    # APRES UN AI-REML CONVERGE, PAS DE POLISSAGE. AI s'arrete sur un decrement
    # de 1e-6 en -2logL : le polissage de Newton n'a plus rien a gagner, et chaque
    # pas coute un Hessien par differences finies (2p gradients). Mesure sur la
    # luzerne : 25 pas pour des gains de 1e-8, le temps total multiplie par 4 a 5.
    # Le Hessien des diagnostics (hessian=True) reste calcule, une seule fois.
    # Si AI echoue, le polissage est garde : il sert alors de repli.
    polish_eff = int(polish)
    H_cache = None
    if int(maxiter) > 0 and optimizer.startswith("ai") and bool(getattr(r, "success", False)):
        polish_eff = 0
        # POINT SELLE. La matrice AI est semi-definie positive par construction :
        # elle ne voit pas une courbure negative, et la boucle AI peut s'arreter
        # sur un point stationnaire qui n'est pas un minimum. Verifie par
        # simulation (fa de rang 2, 5 caracteres) : decrement AI sous 1e-6,
        # Hessien a une valeur propre negative, logLik 1,6e-2 sous l'optimum.
        # Un Hessien par differences finies (2p gradients) tranche ; sur le
        # sous-espace libre, une valeur propre negative renvoie a L-BFGS-B puis
        # au polissage.
        # Le controle n'est fait qu'avec hessian=True : le Hessien est alors
        # calcule de toute facon pour les diagnostics, et il est reutilise. Avec
        # hessian=False, 2p gradients de plus seraient un cout que l'appelant a
        # refuse (400 gradients a 200 parametres).
        lo_ = np.array([b[0] for b in bornes]); hi_ = np.array([b[1] for b in bornes])
        libre_ = (theta > lo_ + 1e-9) & (theta < hi_ - 1e-9)
        if hessian and libre_.any():
            H_ai = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
            H_cache = (theta.copy(), H_ai)
            w_ai = np.linalg.eigvalsh(H_ai[np.ix_(libre_, libre_)])
            if w_ai.min() < -1e-6 * max(1.0, np.abs(w_ai).max()):
                if verbose:
                    print("  [ai] point selle (valeur propre du Hessien %.2e) -> L-BFGS-B depuis ce point"
                          % w_ai.min(), flush=True)
                r2 = minimize(fun_sc, theta, jac=True, method="L-BFGS-B", bounds=bornes,
                              callback=cb, options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                                        ftol=tol, gtol=1e-10))
                theta = np.asarray(r2.x, dtype=np.float64)
                r = SimpleNamespace(x=theta, nit=int(getattr(r, "nit", 0)) + int(r2.nit), success=False,
                                    message="%s ; point selle, repli L-BFGS-B : %s"
                                    % (getattr(r, "message", ""), r2.message))
                polish_eff = int(polish)

    # --- polissage de Newton -------------------------------------------------
    # L-BFGS-B s'arrete sur une tolerance RELATIVE de l'objectif : il rend un
    # point a l'optimum a ~1e-7 pres sur les composantes de variance, ce qui se
    # voit quand on compare a une solution analytique. Quelques pas de Newton
    # regularises (le Hessien est calcule de toute facon pour le diagnostic)
    # ramenent l'ecart au niveau de la precision machine. C'est aussi ce qui
    # rapproche le comportement de l'AI-REML d'asreml, qui est un schema de type
    # Newton sur les parametres de variance : meme objectif, meme optimum, et
    # desormais la meme facon d'y finir.
    #
    # Chaque pas est ACCEPTE seulement s'il ameliore l'objectif (recherche
    # lineaire par bissection). Sans ce garde-fou, une courbure negative ou un
    # Hessien mal conditionne eloignerait du sommet en pretendant s'en approcher.
    # Le polissage tourne jusqu'a ce que le decrement de Newton soit negligeable
    # devant le seuil d'un LRT (3.84), pas un nombre fixe de fois : c'est le
    # critere qui a un sens, et il coute deux gradients par parametre et par pas.
    if finition == "rc" and int(maxiter) > 0 and polish_eff > 0:
        try:
            from .finition import finir
        except ImportError:
            from finition import finir
        lo_ = np.array([b_[0] for b_ in bornes], dtype=np.float64)
        hi_ = np.array([b_[1] for b_ in bornes], dtype=np.float64)
        theta, info_fin = finir(fun_jac, theta, lo_, hi_,
                                hessien=lambda th: _hessian_fd(fun_jac, th, floor=lo_, ceil=hi_),
                                verbose=verbose, compteur=hist)
        hist["finition"] = info_fin
        polish_eff = 0
    n_polish = 0
    # LE POLISSAGE PARLE, COMME L-BFGS-B. Il etait muet : sur un modele a 198
    # parametres il a dure 61,5 h et 9 979 evaluations APRES la derniere ligne
    # d'iteration, et un journal fige pendant quatre jours a fait diagnostiquer
    # un blocage la ou l'ajustement travaillait. Chaque pas coute 2p gradients
    # (Hessien par differences finies) : une ligne par pas est bon marche et
    # suffit a distinguer la progression d'un arret.
    if polish_eff > 0:
        if verbose:
            print("  polissage de Newton : jusqu'a %d pas, %d gradients par pas"
                  % (polish_eff, 2 * p), flush=True)
        for _ in range(polish_eff):
            t_pas = time.time()
            f_cur, g_cur = fun_jac(theta)
            H = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
            if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g_cur)):
                if verbose:
                    print("  polissage arrete : Hessien ou gradient non fini", flush=True)
                break
            try:
                w, Vv = np.linalg.eigh(0.5 * (H + H.T))
            except np.linalg.LinAlgError:
                break
            lam = max(1e-10, 1e-8 * abs(w).max())
            w_reg = np.where(w > lam, w, lam)          # regularisation : pas de pas
            step = -(Vv @ ((Vv.T @ g_cur) / w_reg))    # dans une direction plate
            done = False
            if fixed_idx:
                step[np.array(fixed_idx)] = 0.0
            for alpha in (1.0, 0.5, 0.25, 0.1):
                cand = np.clip(theta + alpha * step, floor, ceil)
                f_try, _ = fun_jac(cand)
                if np.isfinite(f_try) and f_try < f_cur - 1e-13:   # gain ABSOLU :
                    # un seuil relatif a |f| (~1e4) rejetterait des gains de 1e-11
                    # qui deplacent encore les variances au 7e chiffre.
                    theta, done = cand, True
                    break
            if not done:
                if verbose:
                    print("  polissage arrete : aucun pas n'ameliore -2logL (deja au sommet)",
                          flush=True)
                break                                   # deja au sommet
            n_polish += 1
            if verbose:
                # decrement de Newton g' H^+ g / 2 en unites de logLik, sur le
                # Hessien regularise : c'est la montee encore disponible sous
                # le modele quadratique, comparable au 3,84 d'un LRT a 1 ddl.
                dec = 0.5 * float(g_cur @ (Vv @ ((Vv.T @ g_cur) / w_reg)))
                print("  polish %3d | -2logL %14.6f | gain %.3e | decrement %.3e | %6.1f s | %d eval"
                      % (n_polish, f_try, f_cur - f_try, dec,
                         time.time() - t_pas, hist["n_eval"]), flush=True)
            if float(g_cur @ g_cur) < 1e-24:
                break

    f_end, g_end = fun_jac(theta)
    _chrono("finition")
    # temps d'une evaluation : mediane des evaluations de l'optimiseur ; la
    # compilation est l'exces du premier appel sur cette mediane
    _t_eval = float(np.median(hist["t_eval"])) if hist["t_eval"] else float(_t_premier)
    if moteur == "mme" and getattr(M_mme, "_durees", None):
        _t_eval = float(np.median(M_mme._durees))         # evaluations completes calculees
    _t_compil = max(_t_premier - _t_eval, 0.0)
    # DEUX CONVENTIONS DE LOG-VRAISEMBLANCE, et il faut les deux.
    # remlax et lme4 incluent la constante (n-p)/2 * log(2*pi) ; asreml et
    # sommer l'omettent. Comparer les deux nombres sans le savoir donne un ecart
    # de plusieurs centaines (219.6 sur un jeu a n=240, p=1) qu'on prendrait
    # pour un desaccord de modele. VERIFIE a 1.3e-08 sur ce meme jeu.
    n_p = int(np.linalg.matrix_rank(X))
    cst = 0.5 * (len(y) - n_p) * np.log(2.0 * np.pi)
    out = dict(theta=theta, neg2_reml=f_end, logLik=-0.5 * f_end, n_polish=n_polish,
               logLik_asreml=-0.5 * f_end + cst, const_2pi=float(cst),
               n_par=int(p), n_obs=int(n),
               max_grad=float(np.max(np.abs(g_end))),
               # Avec maxiter = 0 l'optimiseur n'a pas tourne : on le DIT, au
               # lieu de rendre un statut de convergence qui n'a pas de sens.
               scipy_success=(True if int(maxiter) <= 0 else bool(r.success)),
               scipy_message=("evaluation seule (maxiter = 0), aucune optimisation"
                              if int(maxiter) <= 0 else str(r.message)),
               n_iter=(0 if int(maxiter) <= 0 else int(r.nit)),
               secondes=time.time() - hist["t0"],
               # LES BORNES VOYAGENT AVEC LE RESULTAT. Tout consommateur qui a
               # besoin du sous-espace libre — erreurs-types, diagnostics — doit
               # les LIRE et non les recoder : une borne en dur cote R valait -8
               # la ou le solveur borne a -12, et onze composantes de variance
               # etaient exclues du calcul des erreurs-types sans raison.
               par_floor=float(floor), par_ceil=float(ceil),
               # Mesures separees : la compilation est payee UNE fois, une
               # evaluation autant de fois qu'il y a d'iterations. Les melanger
               # attribue a l'algebre du temps de compilateur.
               compile_s=_t_compil, eval_s=_t_eval, first_call_s=_t_premier,
               n_eval=int(hist["n_eval"]),
               # Optimiseur et temps de sa seule phase (avant polissage) : c'est
               # ce que compare un banc d'essai entre L-BFGS-B et AI-REML.
               optimiseur=str(optimizer), secondes_optim=float(t_optim))

    th_terms, th_res, _ = split_theta(theta, terms, res)
    # ENTREES NON IDENTIFIEES SEPAREMENT (alias_entrees) : motifs de V egaux.
    # Elles laissent autant de directions plates, attendues, dans le Hessien ;
    # le nombre de parametres identifies (ddl d'un LRT, AIC) en tient compte.
    alias = alias_entrees(terms, n)
    n_plat = int(sum(len(g["entrees"]) - 1 for g in alias))
    _chrono("alias")
    out["alias"] = [dict(terme=g["terme"], entrees=[[int(i), int(j)] for i, j in g["entrees"]])
                    for g in alias]
    out["n_plat_attendu"] = n_plat
    out["n_par_identifies"] = int(p - n_plat)
    if n_plat and verbose:
        print("  [identification] %d direction(s) plate(s) par construction : %s"
              % (n_plat, "; ".join("%s %s" % (g["terme"], " = ".join(
                  "[%d,%d]" % (i + 1, j + 1) for i, j in g["entrees"])) for g in alias)),
              flush=True)
    # REPONSE ANALYSEE. Sans decalage, y ; avec, y* = (I - M) y au point final :
    # beta, les BLUP et P y sont ceux du modele ordinaire de y*.
    y_fit = (np.asarray(y, dtype=np.float64) if lag is None
             else np.asarray(lag_ystar(jnp.asarray(theta[p_cov:]), lag, y)))
    if lag is not None:
        out["lag"] = lag_report(theta[p_cov:], lag, X, f_end)
        out["ystar"] = y_fit
        if verbose:
            Lr = out["lag"]
            print("  [decalage] critere %s | chemin %s | log|det A| %.6f | rayon spectral de M %.4f"
                  % (Lr["criterion"], Lr["chemin"], Lr["logdet_A"], Lr["rayon_spectral"]),
                  flush=True)

    # --- composantes degenerees -----------------------------------------------
    # Une variance qui part au plancher n'est pas "estimee a zero" : elle est
    # NON IDENTIFIEE, et tout ce qui la concerne (son ecart-type, sa part de
    # variance, les tests qui la touchent) est sans objet. Le cas typique est
    # l'aliasing AR1 <-> pepite : le champ absorbe toute la residuelle. asreml
    # refuse alors de converger ("singularities in the Average Information
    # matrix") ; ici l'ajustement aboutit, donc il FAUT le signaler, sans quoi
    # on publierait une residuelle de 1e-11 comme un resultat.
    seuil = floor + 0.5
    degen = []
    o = 0
    for k, tm in enumerate(terms):
        p_k = term_n_params(tm)
        if np.any(theta[o:o + p_k] <= seuil):
            degen.append(tm["name"])
        o += p_k
    if np.any(theta[o:p_cov] <= seuil):
        degen.append("residuelle")
    out_degen = degen
    out["composantes_degenerees"] = out_degen
    out["n_fixed"] = 0 if not fixed_idx else len(fixed_idx)
    if out_degen and verbose:
        print("  [attention] composante(s) au plancher (non identifiee(s)) : %s"
              % ", ".join(out_degen), flush=True)
    out["sigmas"], out["rho"] = {}, {}
    for k, tm in enumerate(terms):
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        out["sigmas"][tm["name"]] = np.asarray(build_sigma(
            jnp.asarray(th_terms[k][:ns]), tm["struct"], tm["t"], tm["rank"]))
        kind_ = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        r_ = level_params_report(np.asarray(th_terms[k][ns:]), kind_,
                                 tm.get("lvl_order", 0), opts=tm.get("lvl_opts"),
                                 q=tm.get("q"), parts=tm.get("lvl_parts"))
        # Lambda et les variances specifiques, quand la structure en a. Sigma
        # assemblee ne suffit pas : sa decomposition n'est pas unique.
        lo = sigma_loadings(np.asarray(th_terms[k][:ns]), tm["struct"],
                            tm["t"], tm["rank"])
        if lo is not None:
            out.setdefault("loadings", {})[tm["name"]] = lo
        if r_.get("pacf"):
            out.setdefault("pacf", {})[tm["name"]] = r_["pacf"]
        _ranger_niveaux(out, tm["name"], kind_, r_)
    secs = res_sections(res, n)
    out["sigmas_res"], o_s = {}, 0
    for sec in secs:
        p_s = sec_n_params(sec)
        th_s = th_res[o_s:o_s + p_s]
        ns_r = n_params(sec["struct"], sec["t"], sec["rank"])
        nom = sec.get("name", "residuelle") if len(secs) > 1 else "residuelle"
        S = (np.asarray(build_sigma(jnp.asarray(th_s[:ns_r]), sec["struct"],
                                    sec["t"], sec["rank"]))
             if sec["struct"] in ("us", "fa", "diag")
             else np.array([[float(np.exp(2 * th_s[0]))]]))
        out["sigmas_res"][nom] = S
        lo_r = sigma_loadings(np.asarray(th_s[:ns_r]), sec["struct"],
                              sec["t"], sec["rank"])
        if lo_r is not None:
            out.setdefault("loadings_res", {})[nom] = lo_r
        r_res = level_params_report(np.asarray(th_s[ns_r:]), sec.get("lvl", "id"),
                                    sec.get("lvl_order", 0), opts=sec.get("lvl_opts"),
                                    q=sec.get("n_unit", len(np.unique(np.asarray(sec["unit"])))),
                                    parts=sec.get("lvl_parts"))
        _ranger_niveaux(out, nom, sec.get("lvl", "id"), r_res)
        o_s += p_s
    # Residuelle "none" : aucune section, donc pas de sigma_res (et non une
    # matrice nulle, qui se lirait comme une variance estimee a zero).
    if out["sigmas_res"]:
        out["sigma_res"] = out["sigmas_res"][list(out["sigmas_res"])[0]]

    out["moteur"] = moteur
    out["memoire"] = memoire if moteur == "dense" else "standard"
    if moteur == "blocs":
        out["blocs"] = plan_blocs.resume()
    if moteur == "mme":
        # beta, vbeta et BLUP lus dans les equations du modele mixte : V n'est
        # jamais formee (elle serait n x n, ce que ce moteur evite).
        so = M_mme.sorties(theta, pev=(pev if blups else ()))
        out["beta"], out["vbeta"], out["V_singuliere"] = so["beta"], so["vbeta"], False
        if blups:
            out["blups"] = so["blups"]
            if "pev" in so:
                out["pev"] = so["pev"]
        out["inversion"] = M_mme.inversion
    if moteur in ("dense", "blocs") and (blups or hessian) and (memoire == "econome"
                                                                 or moteur == "blocs"):
        # Moteur par blocs : les sorties (beta, BLUP, PEV) passent par la voie
        # econome, qui forme V une fois sur le peripherique puis la ramene sur
        # l'hote. Version 1 : une sortie par blocs reste a ecrire.
        try:
            from .econome import V_hote, z_hote, PlanEconome
        except ImportError:
            from econome import V_hote, z_hote, PlanEconome
        if moteur == "blocs":
            plan_eco = PlanEconome(terms, res, y, X)
        out.update(effets_denses(theta, terms, None, res, y_fit, X, blups=blups, pev=pev,
                                 V=V_hote(plan_eco, theta),
                                 z_de=lambda k: z_hote(terms[k], n)))
    elif moteur == "dense" and (blups or hessian):
        out.update(effets_denses(theta, terms, Zs, res, y_fit, X, blups=blups, pev=pev))
    _chrono("effets")
    if hessian:
        if H_cache is not None and np.array_equal(H_cache[0], theta):
            H = H_cache[1]                      # celui du controle de point selle
        else:
            H = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
        out["hessian"] = H
        out.update(_diagnostic(H, g_end, theta, floor, ceil, fixed_idx=fixed_idx,
                               f_obj=out.get("neg2_reml"), n_plat_attendu=n_plat))
    else:
        # SANS HESSIEN, CE QUI RESTE DISPONIBLE L'EST GRATUITEMENT. g_end est
        # deja en memoire : supprimer le gradient projete avec le reste laissait
        # un ajustement de 39 minutes sans AUCUN critere, alors que celui-ci
        # suffit a dire qu'un point n'est pas un optimum. Le decrement et le
        # signe de la courbure restent absents, et le verdict le DIT plutot que
        # de valoir vrai par defaut.
        out.update(_diagnostic_gradient(g_end, theta, floor, ceil,
                                        fixed_idx=fixed_idx,
                                        f_obj=out.get("neg2_reml")))
        out.update(dict(newton_decrement=np.nan, conv_decrement=None,
                        conv_hessien_ok=None, n_neg_eig=None, cond=np.nan))
    _chrono("hessien")
    # FIN PAR LE MOTEUR DENSE. Un ajustement MME qui s'arrete avec un gradient
    # relatif encore grand est repris par le moteur dense depuis le point
    # atteint : le moteur MME doit inverser Sigma et K, ce que le moteur dense
    # ne fait jamais, et il peut s'enfermer pres d'une Sigma de rang deficient
    # (fa a 5 caracteres, banc 2026-10-22). Le champ `moteur` le dit.
    gr_ = out.get("grad_rel")
    # Le moteur dense ne sait pas representer un terme donne par sa seule
    # precision K^-1 (pedigree) : il le traiterait comme independant, et la
    # reprise rendrait un AUTRE modele (verif_simulations, parente_pedigree_kinv :
    # logLik -684,65 au lieu de -657,59). Pas de reprise dans ce cas.
    kinv_seul = any(tm.get("Kinv") is not None and tm.get("LK") is None for tm in terms)
    echec_opt = not bool(out.get("scipy_success", False))
    if (moteur == "mme" and int(maxiter) > 0 and kinv_seul and verbose and echec_opt
            and not (gr_ is not None and gr_ <= GRAD_REL_REPLI_DENSE)):
        print("  [moteur] MME non converge (gradient relatif %s), sans reprise dense : un terme "
              "n'a que sa precision K^-1" % ("%.2e" % gr_ if gr_ is not None else "n.d."), flush=True)
    # La reprise ne vise que les ajustements que l'optimiseur n'a PAS declares
    # convergés (AI bloquee, repli L-BFGS-B) : un ajustement AI convergé par son
    # decrement peut garder un gradient relatif de quelques 1e-5 (pedigree a
    # 500 individus : decrement < 1e-6, a 5e-7 de l'optimum). Et pas au-dela de
    # n = 20000, ou le moteur dense ne tient plus en memoire.
    if (moteur == "mme" and int(maxiter) > 0 and not kinv_seul and echec_opt
            and len(y) <= N_MAX_REPLI_DENSE
            and not (gr_ is not None and gr_ <= GRAD_REL_REPLI_DENSE)):
        if verbose:
            print("  [moteur] MME non converge (gradient relatif %s) : fin par le moteur dense "
                  "depuis ce point" % ("%.2e" % gr_ if gr_ is not None else "n.d."), flush=True)
        out_d = fit_reml(terms, res, y, X, theta_init=out["theta"], maxiter=maxiter,
                         floor=floor, ceil=ceil, verbose=verbose, hessian=hessian, blups=blups,
                         pev=pev, tol=tol, polish=polish, check=False, n_restarts=0,
                         fixed_idx=fixed_idx, lag=lag, optimizer=optimizer, moteur="dense",
                         inversion=inversion, memoire=memoire, finition=finition,
                         arret_lent=arret_lent)
        out_d["moteur"] = "mme+dense"
        out_d["inversion"] = out.get("inversion")
        out_d["secondes"] = float(out_d.get("secondes", 0.0)) + float(out.get("secondes", 0.0))
        return out_d
    return out


def _ranger_niveaux(out, nom, kind, rapport):
    """Range les parametres de niveaux dans `rho`, sous des cles NON AMBIGUES.

    Une structure a UN parametre garde la cle nue (`rho["residuelle"]`) : c'est
    la convention historique, et un ar1 n'a qu'un rho. Des qu'il y en a
    plusieurs — mtrn porte phi, nu, delta, alpha — la cle nue ne dirait pas
    LEQUEL. On prefixe alors : `residuelle!phi`, `residuelle!nu`, ...
    """
    multi = kind == "mtrn"
    for cle, v in rapport.items():
        if cle == "pacf":
            continue
        if cle == "phi" and not multi:
            out["rho"][nom] = v
        else:
            out["rho"]["%s!%s" % (nom, cle)] = v


def _evasion_plancher(fun_jac, theta, th0, bornes, terms, res, floor, marge=3.0,
                      verbose=True):
    """Fait sortir du plancher les variances qui n'y ont rien a faire.

    Une variance parametree en log (iid, diag, variances specifiques d'un fa)
    a un gradient en theta egal a 2 psi dF/dpsi : pres du plancher il est nul,
    quel que soit dF/dpsi, et L-BFGS-B s'y declare converge. MESURE sur un fa1
    a 28 environnements pondere (drops2) : les 28 variances specifiques a
    exp(2 x -11,4), -2logL 5469 contre 3507 a l'optimum, ou elles valent 0,03 a
    0,8. Le SIGNE du gradient reste lisible : negatif, monter la variance fait
    baisser -2logL. Ces variances sont remontees ensemble vers leur valeur de
    depart (psi0 e^-2, puis e^-6, puis e^-10) ; le point est retenu des que
    -2logL baisse, et None est rendu sinon (rien a faire ou aucun gain).
    """
    try:
        from .ai import _blocs_echelle
    except ImportError:
        from ai import _blocs_echelle
    blocs, p = _blocs_echelle(terms, res)
    if p != len(theta):
        return None
    log_var = np.zeros(p, dtype=bool)
    for (o, ln, nat, t) in blocs:
        if nat in ("iid", "diag"):
            log_var[o:o + ln] = True
    lo = np.array([b[0] for b in bornes]); hi = np.array([b[1] for b in bornes])
    libre = hi > lo
    bas = log_var & libre & (theta <= floor + marge)
    if not bas.any():
        return None
    f0, g0 = fun_jac(theta)
    g0 = np.asarray(g0, dtype=np.float64)
    cand = bas & (g0 < 0)
    if not cand.any() or not np.isfinite(float(f0)):
        return None
    for recul in (1.0, 3.0, 5.0):
        th = theta.copy()
        th[cand] = np.clip(np.maximum(th0[cand] - recul, theta[cand]), lo[cand], hi[cand])
        f1, _ = fun_jac(th)
        if np.isfinite(float(f1)) and float(f1) < float(f0) - 1e-8 * max(1.0, abs(float(f0))):
            if verbose:
                print("  [plancher] %d variance(s) remontee(s) (log psi0 - %g) : -2logL %.6f -> %.6f"
                      % (int(cand.sum()), recul, float(f0), float(f1)), flush=True)
            return th
    return None


def _hessian_fd(fun_jac, theta, eps=1e-5, floor=None, ceil=None):
    """Hessien par differences finies CENTREES sur le gradient analytique.

    2p evaluations de gradient. C'est ce que fait TMB::sdreport en interne, et
    c'est plus sur qu'un hessien autodiff sur une factorisation de Cholesky, ou
    la derivee seconde amplifie le mauvais conditionnement.

    ROBUSTESSE. Un pas de +-eps peut sortir du domaine ou V est definie positive
    (variance collee au plancher, incidence degeneree) : le gradient rendu est
    alors non fini, et une colonne entiere de H l'est aussi. Mesure sur un
    balayage de 120 configurations aleatoires : 15 plantaient dans eigh() avec
    "Eigenvalues did not converge". On bascule donc en differences AVANT ou
    ARRIERE selon le cote qui reste evaluable, et si les deux echouent la
    colonne est mise a NaN — le diagnostic la signalera au lieu de planter.
    """
    p = len(theta)
    H = np.full((p, p), np.nan)
    f0, g0 = fun_jac(theta)
    for j in range(p):
        e = np.zeros(p); e[j] = eps
        tp = theta + e if ceil is None else np.minimum(theta + e, ceil)
        tm = theta - e if floor is None else np.maximum(theta - e, floor)
        _, gp = fun_jac(tp)
        _, gm = fun_jac(tm)
        ok_p, ok_m = np.all(np.isfinite(gp)), np.all(np.isfinite(gm))
        if ok_p and ok_m:
            H[:, j] = (gp - gm) / (tp[j] - tm[j])
        elif ok_p and np.all(np.isfinite(g0)):
            H[:, j] = (gp - g0) / (tp[j] - theta[j])      # difference avant
        elif ok_m and np.all(np.isfinite(g0)):
            H[:, j] = (g0 - gm) / (theta[j] - tm[j])      # difference arriere
    bad = ~np.isfinite(H)
    if bad.any():
        H[bad] = 0.0                       # neutre : ces directions seront
        np.fill_diagonal(H, np.where(~np.isfinite(np.diagonal(H)) | (np.diagonal(H) == 0)
                                     & bad.any(axis=0), 0.0, np.diagonal(H)))
    return 0.5 * (H + H.T)


# Seuils des criteres de convergence. Ils sont RAPPORTES, pas imposes : le
# critere d'arret reste celui de L-BFGS-B suivi du polissage de Newton. Les
# changer en criteres d'arret modifierait tous les ajustements, donc cela se
# fera apres validation contre une reference a tolerance tres serree.
#
# DECREMENT : g' H^+ g est le double du gain de vraisemblance restant dans le
# modele quadratique local. Il est invariant par reparametrisation et s'exprime
# dans l'unite du chi-deux, donc 1e-4 signifie qu'il reste un dix-milliemme
# d'unite a gagner — tres au-dessous de toute pertinence inferentielle, et assez
# serre pour des tests de rapport de vraisemblance ou les modeles compares
# different de plusieurs unites.
#
# GRADIENT RELATIF PROJETE : second test, car le decrement suppose un Hessien
# utilisable. Quand H est mal conditionne ou a des directions nulles, le
# decrement devient ininterpretable et c'est le gradient qui tranche. C'est la
# meme division du travail que dans les logiciels etablis, ou un critere de
# gradient sert de repli quand le Hessien n'est pas defini positif.
SEUIL_DECREMENT = 1e-4
SEUIL_GRAD_REL = 1e-6


def _verdict(d):
    """Trois reponses distinctes, jamais fusionnees : chacune peut manquer."""
    dec, gr = d.get("newton_decrement"), d.get("grad_rel")
    v = {}
    v["conv_decrement"] = (None if dec is None or not np.isfinite(dec)
                           else bool(dec < SEUIL_DECREMENT))
    v["conv_grad_rel"] = (None if gr is None or not np.isfinite(gr)
                          else bool(gr < SEUIL_GRAD_REL))
    # Un Hessien a valeur propre negative dit que le point n'est PAS un maximum,
    # quelle que soit la petitesse du gradient : c'est un test de nature, pas de
    # proximite, et il ne se remplace pas par un seuil.
    v["conv_hessien_ok"] = (None if d.get("n_neg_eig", -1) < 0
                            else bool(d["n_neg_eig"] == 0 and d.get("n_null_dir", 0) == 0))
    v["seuil_decrement"] = SEUIL_DECREMENT
    v["seuil_grad_rel"] = SEUIL_GRAD_REL
    return v


def _diagnostic_gradient(g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None,
                         f_obj=None, ech=None):
    """Les criteres qui ne demandent QU'UN GRADIENT, deja calcule.

    POURQUOI CETTE FONCTION EXISTE. Le bloc de diagnostic etait conditionne EN
    BLOC sur `hessian` : avec hessian=False les TROIS criteres sortaient NA, y
    compris le gradient projete, qui ne demande pourtant aucun Hessien. Mesure
    sur l'ordre 0 du balayage IGE : 39 minutes d'ajustement et pas un seul
    critere rapporte, alors que celui-la etait disponible gratuitement — g_end
    est deja en memoire. Le decrement de Newton et le signe de la courbure, eux,
    exigent bien H et restent absents.

    LA PROJECTION EST INDISPENSABLE. Une composante de variance tenue a sa borne
    laisse un gradient non nul dans cette direction : sans projeter sur le cone
    admissible, aucun critere ne peut jamais etre satisfait, et c'est exactement
    le defaut du critere max_grad de la grille de reference.
    """
    p_ = len(theta)
    at_bound = (theta <= floor + tol_bound) | (theta >= ceil - tol_bound)
    fixed = np.zeros(p_, bool)
    if fixed_idx:
        fixed[np.asarray(list(fixed_idx), dtype=int)] = True
    free = ~at_bound & ~fixed
    out = dict(n_at_bound=int((at_bound & ~fixed).sum()),
               n_fixed_out=int(fixed.sum()),
               n_par_free=int(free.sum()))
    if free.sum() == 0:
        return dict(out, grad_proj_max=np.nan, grad_rel=np.nan, conv_grad_rel=None)
    gf = g[free]
    if not np.all(np.isfinite(gf)):
        return dict(out, grad_proj_max=np.nan, grad_rel=np.nan, conv_grad_rel=None)
    gpm = float(np.max(np.abs(gf)))
    # MISE A L'ECHELLE de chaque parametre et de l'objectif, comme dans le
    # diagnostic complet : un gradient brut n'est pas invariant d'echelle.
    e = np.maximum(np.abs(theta[free]), 1.0) if ech is None else np.asarray(ech)[free]
    if f_obj is None or not np.isfinite(f_obj):
        grad_rel = np.nan
    else:
        grad_rel = float(np.max(np.abs(gf) * e) / max(abs(float(f_obj)), 1.0))
    return dict(out, grad_proj_max=gpm, grad_rel=grad_rel,
                conv_grad_rel=(None if not np.isfinite(grad_rel) else bool(grad_rel < 1e-6)))


def _diagnostic(H, g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None, f_obj=None,
                n_plat_attendu=0):
    """Trois questions distinctes, jamais fusionnees en un booleen.

      1. suis-je au sommet ?   decrement de Newton g' H^+ g, en unites de logLik
      2. est-ce un pic ?       spectre de H (valeurs propres nulles / negatives)
      3. quels parametres ?    ceux a une borne active sont hors du sous-espace
                               libre (condition KKT) et n'y participent pas

    UN PARAMETRE TENU PAR fixed_theta EST AUSSI HORS DU SOUS-ESPACE LIBRE, et il
    n'apparait PAS dans n_at_bound : il est pince a sa valeur de depart, qui n'a
    aucune raison d'etre floor ou ceil. Sans le retirer ici, le decrement de
    Newton comptait la pente disponible le long d'une direction interdite au pas,
    et annoncait donc un optimum moins bon qu'il ne l'est ; le spectre melangeait
    de meme des directions gelees aux directions reelles. Les deux comptes sont
    rendus separement (n_at_bound, n_fixed_out) plutot que fusionnes : ils ne
    veulent pas dire la meme chose, seul leur total definit le sous-espace libre.
    """
    p = len(theta)
    at_bound = (theta <= floor + tol_bound) | (theta >= ceil - tol_bound)
    fixed = np.zeros(p, bool)
    if fixed_idx:
        fixed[np.asarray(list(fixed_idx), dtype=int)] = True
    free = ~at_bound & ~fixed
    out = dict(n_at_bound=int((at_bound & ~fixed).sum()),
               n_fixed_out=int(fixed.sum()),
               n_par_free=int(free.sum()))
    if free.sum() == 0:
        return dict(out, newton_decrement=np.nan, grad_proj_max=np.nan,
                    grad_rel=np.nan, n_neg_eig=0, n_null_dir=0,
                    cond=np.nan, conv_decrement=None, conv_grad_rel=None,
                    conv_hessien_ok=None)
    Hf, gf = H[np.ix_(free, free)], g[free]
    if not np.all(np.isfinite(Hf)) or not np.all(np.isfinite(gf)):
        return dict(out, newton_decrement=np.nan, leak=np.nan, n_neg_eig=-1,
                    n_null_dir=-1, cond=np.nan,
                    diag_note="Hessien ou gradient non fini : diagnostic indisponible")
    try:
        w, Vv = np.linalg.eigh(Hf)
    except np.linalg.LinAlgError as e:
        # eigh peut echouer sur une matrice extremement mal conditionnee ; on le
        # DIT plutot que de laisser l'exception detruire un ajustement valide.
        return dict(out, newton_decrement=np.nan, leak=np.nan, n_neg_eig=-1,
                    n_null_dir=-1, cond=np.nan,
                    diag_note="decomposition spectrale echouee (%s)" % e)
    tol = 1e-8 * max(abs(w).max(), 1.0)
    # DIRECTIONS PLATES ATTENDUES (alias_entrees) : les n plus petites valeurs
    # propres en module sont celles de la vraisemblance constante le long d'une
    # difference d'entrees non identifiee. Elles ne disent rien de la
    # convergence : on les met de cote pour compter valeurs propres negatives et
    # nulles, sans quoi un VP+ a W symetrique serait declare non converge a
    # chaque fois. Le decrement de Newton les exclut deja (pseudo-inverse).
    attendu = np.zeros(len(w), dtype=bool)
    if n_plat_attendu:
        attendu[np.argsort(np.abs(w))[:min(int(n_plat_attendu), len(w))]] = True
    keep = (w > tol) & ~attendu
    dec = float(gf @ (Vv[:, keep] @ ((Vv[:, keep].T @ gf) / w[keep]))) if keep.any() else np.nan
    leak = float(np.max(np.abs(Vv[:, ~keep].T @ gf))) if (~keep).any() else 0.0

    # ------------------------------------------------------------------
    # GRADIENT PROJETE, ET SA VERSION RELATIVE
    # ------------------------------------------------------------------
    # PROJETE. Sous contraintes de bornes, la condition d'optimalite n'est pas
    # g = 0 mais g = 0 dans les directions LIBRES et g pointant vers l'exterieur
    # aux bornes actives. On projette donc le gradient sur le cone admissible :
    # a une borne inferieure seule une composante NEGATIVE (qui pousserait vers
    # l'interieur) temoigne d'une non-optimalite ; une composante positive est
    # retenue par la contrainte et vaut zero apres projection. Sans cela, un
    # sigma^2 a zero laisse un gradient non nul indefiniment et aucun critere de
    # gradient ne peut jamais etre satisfait.
    g_proj = np.array(g, dtype=np.float64, copy=True)
    lo = theta <= floor + tol_bound
    hi = theta >= ceil - tol_bound
    g_proj[lo] = np.minimum(g_proj[lo], 0.0)
    g_proj[hi] = np.maximum(g_proj[hi], 0.0)
    if fixed_idx:
        g_proj[fixed] = 0.0          # direction interdite au pas : ne compte pas
    gpm = float(np.max(np.abs(g_proj))) if p else np.nan

    # RELATIF. Une norme de gradient brute depend de l'echelle des parametres et
    # de n : sur un modele a 16 000 observations, -2logL vaut des dizaines de
    # milliers et un gradient de 37 est petit en relatif. On rapporte donc chaque
    # composante a l'echelle de SON parametre et a celle de l'objectif, dans
    # l'esprit du critere de Dennis et Schnabel. Le facteur max(|theta_i|, 1)
    # evite de diviser par un parametre proche de zero.
    ech = np.maximum(np.abs(np.asarray(theta, dtype=np.float64)), 1.0)
    # f_obj est la valeur de -2logL AU POINT COURANT, passee par l'appelant. Une
    # version anterieure la cherchait dans `out`, qui ne la contient jamais : le
    # denominateur valait donc 1 et la normalisation annoncee ne se produisait
    # pas. Le symptome etait visible — grad_rel sortait bit a bit egal a
    # grad_proj_max. Si l'appelant ne la fournit pas, on le DIT en rendant NaN
    # plutot qu'en normalisant silencieusement par 1.
    if f_obj is None or not np.isfinite(f_obj):
        grad_rel = np.nan
    else:
        f_ech = max(abs(float(f_obj)), 1.0)
        grad_rel = float(np.max(np.abs(g_proj) * ech) / f_ech) if p else np.nan

    res = dict(out, newton_decrement=dec, leak=leak,
                grad_proj_max=gpm, grad_rel=grad_rel,
                n_neg_eig=int(((w < -tol) & ~attendu).sum()),
                n_null_dir=int(((np.abs(w) <= tol) & ~attendu).sum()),
                n_plat_exclues=int(attendu.sum()),
                lambda_min=float(w[~attendu].min()) if (~attendu).any() else np.nan,
                lambda_max=float(w.max()),
                cond=(float(abs(w[~attendu]).max() / max(abs(w[~attendu]).min(), 1e-300))
                      if (~attendu).any() else np.nan),
                k_eff=int(keep.sum()) + int(at_bound.sum() == 0) * 0)
    return dict(res, **_verdict(res))
