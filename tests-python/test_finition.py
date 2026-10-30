"""Finition par Newton a region de confiance (finition.py)."""
import numpy as np
import pytest

import outils_dispositifs  # noqa: F401  (chemin vers inst/python)

from remlax import _x64  # noqa: F401
from remlax.finition import _trust_region_step, decrement, finish
from remlax.fit import fit_reml


def test_pas_rc_newton_et_bord():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5)); H = A @ A.T + np.eye(5); g = rng.normal(size=5)
    d, pred = _trust_region_step(g, H, 1e6)                     # region large : pas de Newton
    assert np.allclose(d, -np.linalg.solve(H, g))
    d, pred = _trust_region_step(g, H, 0.1)                     # region petite : sur le bord
    assert abs(np.linalg.norm(d) - 0.1) < 1e-8 and pred > 0


def test_pas_rc_courbure_negative():
    H = np.diag([2.0, -1.0, 1e-12]); g = np.array([1.0, 0.5, 1e-3])
    d, pred = _trust_region_step(g, H, 0.5)
    assert np.linalg.norm(d) <= 0.5 + 1e-10 and pred > 0


def test_finir_direction_plate_et_plancher():
    """f = (x0 - 1)^2 + 1e-6 (x1 - 3)^2 + exp(2 x2) : direction plate (x1) et une
    'variance' x2 qui s'eteint vers le plancher -12."""
    def fj(x):
        f = (x[0] - 1) ** 2 + 1e-6 * (x[1] - 3) ** 2 + np.exp(2 * x[2])
        g = np.array([2 * (x[0] - 1), 2e-6 * (x[1] - 3), 2 * np.exp(2 * x[2])])
        return f, g

    def hess(x):
        return np.diag([2.0, 2e-6, 4 * np.exp(2 * x[2])])
    lo, hi = np.full(3, -12.0), np.full(3, 12.0)
    x, info = finish(fj, np.array([0.0, 0.0, 0.0]), lo, hi, hess, verbose=False)
    assert info["success"]
    assert abs(x[0] - 1) < 1e-6
    assert x[2] == -12.0                              # plaque au plancher
    assert fj(x)[0] - (1e-6 * 0 + np.exp(-24)) < 1e-5  # gain restant negligeable


def test_decrement():
    H = np.diag([2.0, 1e-20])
    # direction plate sans pente : negligee
    assert abs(decrement(np.array([2.0, 1e-9]), H) - 1.0) < 1e-6
    # direction plate AVEC pente : un vrai gain, la convergence est refusee
    assert decrement(np.array([2.0, 1.0]), H) > 1e3
    # courbure negative significative : pas un minimum
    assert decrement(np.array([1.0, 1.0]), np.diag([2.0, -1.0])) == float("inf")


@pytest.mark.parametrize("fab", ["essais", "ige"])
def test_meme_optimum(fab):
    import test_econome as te
    import test_blocs as tb
    terms, res, y, X = (tb.cas_essais if fab == "essais" else te.cas_ige)()
    kw = dict(verbose=False, hessian=False, blups=False)
    r1 = fit_reml(terms, res, y, X, finish="newton", **kw)
    r2 = fit_reml(terms, res, y, X, finish="trust", **kw)
    if fab == "essais":
        # y de bruit pur, vraisemblance plate et multimodale : les optimums
        # locaux connus vont de 1494,3194 a 1494,3599, et le chemin suivi
        # depend de l'ordre des sommes (CPU ou GPU). Les deux finitions doivent
        # tomber dans cette plage.
        assert abs(r2["neg2_reml"] - r1["neg2_reml"]) < 0.05
    else:
        assert r2["neg2_reml"] - r1["neg2_reml"] < 1e-6
    # Le nombre d'evaluations n'est plus compare. Depuis les departs calcules
    # sur les residus des effets fixes (2026-10-26), L-BFGS-B finit pres de
    # l'optimum et le polissage de Newton ne coute presque rien : la region de
    # confiance fait 430 evaluations contre 412 sur "ige", 7022 contre 2290 sur
    # "essais" (vraisemblance plate). Elle reste une option, pas le defaut.
