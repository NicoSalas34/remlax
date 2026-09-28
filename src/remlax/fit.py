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
                        term_factor, term_n_params, res_sections, sec_n_params)
    from .structures import build_sigma, n_params, theta0 as struct_theta0
    from .levels import n_level_params, level_params_report, level_chol
    from .structures import sigma_loadings
except ImportError:
    from model import (make_objective, assemble_V, split_theta, n_theta, dense_Z,
                       term_factor, term_n_params, res_sections, sec_n_params)
    from structures import build_sigma, n_params, theta0 as struct_theta0
    from levels import n_level_params, level_params_report, level_chol
    from structures import sigma_loadings


def initial_theta(terms, res, y):
    """Depart : variance phenotypique repartie a parts egales entre termes.

    L'EXPOSITION EST PRISE EN COMPTE. Une composante de voisinage entre dans le
    phenotype par sigma^2 * k avec k = moyenne des (Z K Z')_ii, qui vaut 15 a 40
    sur certains dispositifs. Partir de sigma^2 = part * var(y) y mettrait donc
    plusieurs fois la variance phenotypique dans CHAQUE terme, et V demarrerait
    des ordres de grandeur trop grande. On divise par k.
    """
    vy = float(np.var(np.asarray(y))) or 1.0
    part = vy / (len(terms) + 1.0)
    th = []
    for tm in terms:
        Z = np.zeros((len(y), tm["t"] * tm["q"]))
        Z[tm["zi"], tm["zj"]] += tm["zx"]
        Zb = Z.reshape(len(y), tm["t"], tm["q"])
        if tm["LK"] is None:
            k = float(np.mean(np.sum(Zb ** 2, axis=(1, 2)))) / max(tm["t"], 1)
        else:
            ZL = np.einsum("ntq,qp->ntp", Zb, tm["LK"])
            k = float(np.mean(np.sum(ZL ** 2, axis=(1, 2)))) / max(tm["t"], 1)
        k = k if k > 1e-12 else 1.0
        th.append(struct_theta0(tm["struct"], tm["t"], tm["rank"], var=part / k))
        # rho de depart a 0 (tanh(0)) : champ non structure. Partir d'un rho
        # eleve fait demarrer dans un regime ou la vraisemblance est plate.
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        nl = n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                            opts=tm.get("lvl_opts"))
        if nl:
            th.append(_theta0_niveaux(kind, nl, tm.get("lvl_opts"), tm.get("coord")))
    for sec in res_sections(res, len(y)):
        th.append(struct_theta0(sec["struct"], sec["t"], sec["rank"], var=part))
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
        Zc = np.zeros((n, tm["t"] * tm["q"]))
        Zc[tm["zi"], tm["zj"]] += tm["zx"]
        vide = int((np.abs(Zc).sum(axis=0) == 0).sum())
        if vide == Zc.shape[1]:
            raise ValueError("terme '%s' : incidence entierement nulle." % tm["name"])
    if not np.all(np.isfinite(y)):
        raise ValueError("y contient des valeurs non finies.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des valeurs non finies.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("X est de rang %d pour %d colonnes."
                         % (np.linalg.matrix_rank(X), X.shape[1]))


def fit_reml(terms, res, y, X, theta_init=None, maxiter=3000,
             floor=-12.0, ceil=12.0, verbose=True, hessian=True,
             blups=True, pev=False, tol=1e-14, polish=25, check=True,
             n_restarts=0, restart_sd=0.5, seed=0, fixed_idx=None):
    """Ajustement REML.

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
    if n_restarts > 0:
        base = fit_reml(terms, res, y, X, theta_init=theta_init, maxiter=maxiter,
                        floor=floor, ceil=ceil, verbose=verbose, hessian=hessian,
                        blups=blups, pev=pev, tol=tol, polish=polish, check=False,
                        fixed_idx=fixed_idx)
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
                                fixed_idx=fixed_idx)
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
    fun_sc, fun_jac, sc, Zs = make_objective(terms, res, y, X)
    th0 = np.asarray(theta_init if theta_init is not None
                     else initial_theta(terms, res, y), dtype=np.float64)
    p = n_theta(terms, res)
    if th0.shape[0] != p:
        raise ValueError("theta initial de longueur %d, attendu %d" % (th0.shape[0], p))

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
    _t = time.time()
    _v0 = fun_jac(th0)
    try:
        import jax as _jax
        _jax.block_until_ready(_v0[0])
    except Exception:
        pass
    _t_premier = time.time() - _t
    _t = time.time()
    _v1 = fun_jac(th0 * 1.0001 + 1e-6)
    try:
        _jax.block_until_ready(_v1[0])
    except Exception:
        pass
    _t_eval = time.time() - _t
    _t_compil = max(_t_premier - _t_eval, 0.0)

    # ANNONCER LA COMPILATION DES QU'ELLE EST FINIE. Sur le modele multivarie
    # complet d'IGE_analysis, XLA compile un programme portant 5,47 Go de
    # constantes : cette phase peut durer longtemps, et pendant ce temps un
    # journal muet ne permet pas de distinguer "il compile" de "il est bloque".
    # Cette ligne est le premier signe de vie du solveur.
    if verbose:
        print("  [compilation] %.1f s | une evaluation : %.3f s | %d parametres"
              % (_t_compil, _t_eval, p), flush=True)

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

    def fun_jac(th):
        hist["n_eval"] += 1
        v, g = _fun_jac_brut(th)
        hist["v"], hist["gmax"] = float(v), float(np.max(np.abs(g)))
        return v, g

    _fun_sc_brut = fun_sc

    def fun_sc(th):
        hist["n_eval"] += 1
        v, g = _fun_sc_brut(th)
        # REMISE A L'ECHELLE. L-BFGS-B minimise -2logL / sc (cf. make_objective) ;
        # imprimer sa valeur brute donnerait un nombre sans signification pour le
        # lecteur. C'est fun_sc que l'optimiseur appelle, donc c'est ici qu'il
        # faut retenir la valeur — la retenir dans fun_jac laisserait NaN.
        hist["v"], hist["gmax"] = float(v) * sc, float(np.max(np.abs(g))) * sc
        return v, g

    def cb(_):
        hist["n"] += 1
        # LE PAS D'IMPRESSION S'ADAPTE AU COUT MESURE D'UNE ITERATION. Un pas
        # fixe de 25 rendait le journal MUET sur le modele reel, ou une
        # evaluation coute plusieurs secondes : 52 minutes sans une ligne, et
        # aucun moyen de distinguer la progression d'un blocage. Le seuil porte
        # sur _t_eval, qui est MESURE juste au-dessus, et non suppose.
        if not verbose:
            return
        pas = 1 if _t_eval > 0.25 else (10 if _t_eval > 0.02 else 50)
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
        r = minimize(fun_sc, th0, jac=True, method="L-BFGS-B",
                     bounds=bornes, callback=cb,
                     options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                  ftol=tol, gtol=1e-10))
        theta = np.asarray(r.x, dtype=np.float64)

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
    n_polish = 0
    # LE POLISSAGE PARLE, COMME L-BFGS-B. Il etait muet : sur un modele a 198
    # parametres il a dure 61,5 h et 9 979 evaluations APRES la derniere ligne
    # d'iteration, et un journal fige pendant quatre jours a fait diagnostiquer
    # un blocage la ou l'ajustement travaillait. Chaque pas coute 2p gradients
    # (Hessien par differences finies) : une ligne par pas est bon marche et
    # suffit a distinguer la progression d'un arret.
    if polish > 0:
        if verbose:
            print("  polissage de Newton : jusqu'a %d pas, %d gradients par pas"
                  % (int(polish), 2 * p), flush=True)
        for _ in range(int(polish)):
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
               n_eval=int(hist["n_eval"]))

    th_terms, th_res, _ = split_theta(theta, terms, res)

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
    if np.any(theta[o:] <= seuil):
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
    out["sigma_res"] = out["sigmas_res"][list(out["sigmas_res"])[0]]

    V = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    if blups or hessian:
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
            beta = np.linalg.solve(A, XtVi @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ (XtVi @ y)
        out["beta"] = beta
        # (X'V^-1X)^-1 : matrice de covariance des effets fixes A V CONNUE.
        # Necessaire a predict() et point de depart de l'ajustement de
        # Kenward-Roger ; la recalculer plus tard exigerait de refaire V.
        out["vbeta"] = np.linalg.pinv(A)
        out["Vi"] = Vi
        resid = y - X @ beta
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
                B = np.asarray(term_factor(jnp.asarray(th_terms[k]), tm, Zs[k]))  # noqa: F841
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
                Zd = np.asarray(Zs[k])
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
    if hessian:
        H = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
        out["hessian"] = H
        out.update(_diagnostic(H, g_end, theta, floor, ceil, fixed_idx=fixed_idx,
                               f_obj=out.get("neg2_reml")))
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


def _diagnostic(H, g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None, f_obj=None):
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
    keep = w > tol
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
                n_neg_eig=int((w < -tol).sum()), n_null_dir=int((np.abs(w) <= tol).sum()),
                lambda_min=float(w.min()), lambda_max=float(w.max()),
                cond=float(abs(w).max() / max(abs(w).min(), 1e-300)),
                k_eff=int(keep.sum()) + int(at_bound.sum() == 0) * 0)
    return dict(res, **_verdict(res))
