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
except ImportError:
    from model import (make_objective, assemble_V, split_theta, n_theta, dense_Z,
                       term_factor, term_n_params, res_sections, sec_n_params)
    from structures import build_sigma, n_params, theta0 as struct_theta0
    from levels import n_level_params, level_params_report, level_chol


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
             blups=True, tol=1e-14, polish=25, check=True,
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
                        blups=blups, tol=tol, polish=polish, check=False,
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
                                blups=blups, tol=tol, polish=polish, check=False,
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

    hist = {"n": 0, "t0": time.time()}

    def cb(_):
        hist["n"] += 1
        if verbose and hist["n"] % 25 == 0:
            v, g = fun_jac(_)
            print("  iter %4d | -2logL %14.6f | max|grad| %.3e | %5.1f s"
                  % (hist["n"], v, np.max(np.abs(g)), time.time() - hist["t0"]), flush=True)

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
    r = minimize(fun_sc, th0, jac=True, method="L-BFGS-B",
                 bounds=bornes,
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
    if polish > 0:
        for _ in range(int(polish)):
            f_cur, g_cur = fun_jac(theta)
            H = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
            if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g_cur)):
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
                break                                   # deja au sommet
            n_polish += 1
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
               scipy_success=bool(r.success), scipy_message=str(r.message),
               n_iter=int(r.nit), secondes=time.time() - hist["t0"])

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
                                 q=tm.get("q"))
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
        r_res = level_params_report(np.asarray(th_s[ns_r:]), sec.get("lvl", "id"),
                                    sec.get("lvl_order", 0), opts=sec.get("lvl_opts"),
                                    q=sec.get("n_unit", len(np.unique(np.asarray(sec["unit"])))))
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
                u = np.kron(Ls, Kq) @ (Zd.T @ Py)
                out["blups"][tm["name"]] = u.reshape(tm["t"], tm["q"]).T
    if hessian:
        H = _hessian_fd(fun_jac, theta, floor=floor, ceil=ceil)
        out["hessian"] = H
        out.update(_diagnostic(H, g_end, theta, floor, ceil, fixed_idx=fixed_idx))
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


def _diagnostic(H, g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None):
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
        return dict(out, newton_decrement=np.nan, n_neg_eig=0, n_null_dir=0, cond=np.nan)
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
    return dict(out, newton_decrement=dec, leak=leak,
                n_neg_eig=int((w < -tol).sum()), n_null_dir=int((np.abs(w) <= tol).sum()),
                lambda_min=float(w.min()), lambda_max=float(w.max()),
                cond=float(abs(w).max() / max(abs(w).min(), 1e-300)),
                k_eff=int(keep.sum()) + int(at_bound.sum() == 0) * 0)
