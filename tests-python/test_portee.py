# ==============================================================================
# test_portee.py — noyaux a PORTEE (sph, cir, lvr) : gradient fini au-dela de
#                  la portee, depart par balayage, optimum global
# ------------------------------------------------------------------------------
# Origine : validation asreml3 du 2026-09-28, section B7. Le noyau circulaire
# rendait un gradient NaN des qu'une paire depassait la portee (0 x inf dans la
# derivee de l'ecretage), L-BFGS-B s'arretait a l'iteration 0 et le point de
# depart etait rendu comme optimum. Sur le meme champ, la vraisemblance a trois
# maxima locaux en la portee ; asreml et remlax partaient dans le mauvais.
# ==============================================================================
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import outils_dispositifs as od
from remlax import levels as Lv
from remlax import fit as F

RNG = np.random.default_rng(2026)
N = 60
COORD = np.column_stack([RNG.uniform(0, 20, N), RNG.uniform(0, 20, N)])
D = np.sqrt(((COORD[:, None, :] - COORD[None, :, :]) ** 2).sum(-1))


def _noyau_np(kind, D, rg):
    h = np.minimum(D / rg, 1.0)
    if kind == "sph":
        return 1.0 - 1.5 * h + 0.5 * h ** 3
    if kind == "cir":
        return 1.0 - (2.0 / np.pi) * (h * np.sqrt(1.0 - h ** 2) + np.arcsin(h))
    return np.clip(1.0 - D / rg, 0.0, None)          # lvr, 1D


def _champ(kind, rg_vrai, seed):
    rng = np.random.default_rng(seed)
    C = _noyau_np(kind, D, rg_vrai) + 1e-6 * np.eye(N)
    e = np.linalg.cholesky(C) @ rng.normal(size=N)
    return 2.0 + e + rng.normal(0, 0.1, N)


def _profil_reference(kind, y, portees):
    """-2logL REML minimale sur une grille de portees, variance en forme close.

    Pour V = s2 C, le minimum en s2 de -2logL est atteint en s2 = y'P0y/(n-p)
    avec P0 calcule sous C. Aucun emprunt a remlax.
    """
    X = np.ones((N, 1))
    out = []
    for rg in portees:
        C = _noyau_np(kind, D, rg) + 1e-10 * np.eye(N)
        Ci = np.linalg.inv(C)
        A = X.T @ Ci @ X
        beta = np.linalg.solve(A, X.T @ Ci @ y)
        r = y - X @ beta
        yPy = float(r @ Ci @ r)
        s2 = yPy / (N - 1)
        out.append(od.neg2logL_reference(s2 * C, y, X))
    return np.array(out)


# ------------------------------------------------------------------------------
# 1. gradient FINI quand des paires depassent la portee
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["sph", "cir"])
def test_gradient_fini_au_dela_de_la_portee(kind):
    # portee 5 sur un champ de 20 x 20 : la plupart des paires sont au-dela
    def somme(th):
        return jnp.sum(Lv.level_corr(th, kind, N, coord=COORD))
    th = jnp.array([np.log(5.0)])
    g = np.asarray(jax.grad(somme)(th))
    assert np.all(np.isfinite(g)), g
    assert abs(g[0]) > 0.0
    # et la valeur suit toujours la formule de reference
    C = np.asarray(Lv.level_corr(th, kind, N, coord=COORD))
    assert np.allclose(C, _noyau_np(kind, D, 5.0), atol=1e-12)


# ------------------------------------------------------------------------------
# 2. balayage des portees : candidats et choix du meilleur
# ------------------------------------------------------------------------------
def test_candidats_portee_sont_des_quantiles_de_distances():
    c = F._candidats_portee(COORD)
    assert c.ndim == 1 and 5 <= c.size <= 9
    d = D[np.triu_indices(N, 1)]
    assert np.exp(c).min() >= d.min() - 1e-12 and np.exp(c).max() <= d.max() + 1e-12
    assert F._candidats_portee(None).size == 0
    assert F._candidats_portee(np.zeros((3, 2))).size == 0          # positions confondues


def test_balayage_choisit_le_meilleur_candidat_et_respecte_les_fixes():
    y = _champ("cir", 8.0, 11)
    X = np.ones((N, 1))
    res = od.residuelle("iid", np.zeros(N, np.int64), np.arange(N), 1, lvl="cir", coord=COORD)
    from remlax.model import make_objective
    _, fun_jac, _, _ = make_objective([], res, y, X)
    th0 = F.initial_theta([], res, y)
    assert F._positions_portee([], res, N) == [(1, COORD)] or \
        F._positions_portee([], res, N)[0][0] == 1
    th, n_eval = F._balayage_portee(fun_jac, th0, [], res, N)
    assert n_eval >= 6
    v0 = float(fun_jac(th0)[0]); v1 = float(fun_jac(th)[0])
    assert v1 <= v0 + 1e-10
    # le parametre de portee fixe n'est pas touche
    th_f, _ = F._balayage_portee(fun_jac, th0, [], res, N, fixed_idx=[1])
    assert th_f[1] == th0[1]


# ------------------------------------------------------------------------------
# 3. l'ajustement atteint le MINIMUM GLOBAL du profil de reference
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,seed", [("cir", 11), ("cir", 12), ("sph", 21), ("sph", 22)])
def test_fit_atteint_le_minimum_global_du_profil(kind, seed):
    y = _champ(kind, 8.0, seed)
    X = np.ones((N, 1))
    res = od.residuelle("iid", np.zeros(N, np.int64), np.arange(N), 1, lvl=kind, coord=COORD)
    fit = F.fit_reml([], res, y, X, verbose=False, hessian=False, blups=False)
    assert fit["n_iter"] > 0
    portees = np.linspace(2.0, 16.0, 281)
    prof = _profil_reference(kind, y, portees)
    # remlax n'est jamais au-dessus du meilleur point de la grille (pas 0.05 :
    # la grille peut manquer le minimum exact de tres peu), et le trouve dans la
    # bonne colline : sa portee est a moins d'un pas de grille de l'argmin.
    assert fit["neg2_reml"] <= prof.min() + 1e-6, (fit["neg2_reml"], prof.min())
    rg_fit = float(np.exp(fit["theta"][1]))
    assert abs(rg_fit - portees[np.argmin(prof)]) <= 0.06, (rg_fit, portees[np.argmin(prof)])


def test_lvr_1d_atteint_le_minimum_global_du_profil():
    # Transect REGULIER. Sur des positions irregulieres, la vraisemblance de
    # lvr (noyau triangulaire) a un pli a chaque distance distincte : 68 minima
    # locaux entre 2 et 25 sur 60 positions tirees au hasard, et aucun balayage
    # raisonnable ne garantit le minimum global (cf. docs/structures.md). Sur
    # un pas regulier les plis sont aux entiers et le balayage suffit.
    x = np.arange(N, dtype=float) * 0.5
    Dl = np.abs(x[:, None] - x[None, :])
    rng = np.random.default_rng(31)
    C = np.clip(1.0 - Dl / 9.0, 0.0, None) + 1e-6 * np.eye(N)
    y = 1.0 + np.linalg.cholesky(C) @ rng.normal(size=N) + rng.normal(0, 0.1, N)
    X = np.ones((N, 1))
    res = od.residuelle("iid", np.zeros(N, np.int64), np.arange(N), 1, lvl="lvr", coord=x.reshape(N, 1))
    fit = F.fit_reml([], res, y, X, verbose=False, hessian=False, blups=False)
    portees = np.linspace(2.0, 25.0, 461)
    prof = []
    for rg in portees:
        Cr = np.clip(1.0 - Dl / rg, 0.0, None) + 1e-10 * np.eye(N)
        Ci = np.linalg.inv(Cr); A = X.T @ Ci @ X
        r = y - X @ np.linalg.solve(A, X.T @ Ci @ y)
        s2 = float(r @ Ci @ r) / (N - 1)
        prof.append(od.neg2logL_reference(s2 * Cr, y, X))
    assert fit["neg2_reml"] <= min(prof) + 1e-6


# ------------------------------------------------------------------------------
# 4. un depart a gradient non fini est REFUSE, pas rendu comme optimum
# ------------------------------------------------------------------------------
def test_depart_non_fini_est_refuse(monkeypatch):
    y = _champ("sph", 8.0, 41)
    X = np.ones((N, 1))
    res = od.residuelle("iid", np.zeros(N, np.int64), np.arange(N), 1, lvl="sph", coord=COORD)
    import remlax.model as M
    brut = M.make_objective

    def casse(*a, **k):
        fun_sc, fun_jac, sc, Zs = brut(*a, **k)

        def fun_jac_nan(th):
            v, g = fun_jac(th)
            return v, jnp.asarray(g) * jnp.nan
        return fun_sc, fun_jac_nan, sc, Zs
    monkeypatch.setattr(F, "make_objective", casse)
    with pytest.raises(ValueError, match="pas fini au point de depart"):
        F.fit_reml([], res, y, X, verbose=False, hessian=False, blups=False)