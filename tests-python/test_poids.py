"""Poids a priori (fit.fit_reml(poids=)) contre une reference numpy independante.

Reference : V = Z G Z' + sigma2 W^-1, -2logL REML calcule directement. Les poids
ne sont connus que du solveur par mise a l'echelle des lignes : la reference
verifie la vraisemblance d'origine, constante comprise.
"""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml


def dispositif(graine=0, n=150, q=20):
    rng = np.random.default_rng(graine)
    g = rng.integers(0, q, n)
    tm = od.terme_facteur("g", g, q)
    res = od.residuelle_iid(n)
    X = np.c_[np.ones(n), rng.normal(size=n)]
    w = rng.uniform(0.3, 3.0, n)
    y = X @ np.array([1.0, 0.5]) + rng.normal(size=q)[g] + rng.normal(size=n) / np.sqrt(w)
    return [tm], res, y, X, w, g, q


@pytest.mark.parametrize("engine", ["dense", "mme"])
def test_vraisemblance_ponderee(engine):
    terms, res, y, X, w, g, q = dispositif()
    th = np.array([0.2, -0.1])                       # log sd du terme, log sd residuel
    r = fit_reml(terms, res, y, X, theta_init=th, maxiter=0, polish=0, verbose=False,
                 hessian=False, blups=True, weights=w, engine=engine)
    Z = od.dense_Z_np(terms[0], len(y))
    V = np.exp(2 * th[0]) * Z @ Z.T + np.exp(2 * th[1]) * np.diag(1.0 / w)
    ref = od.neg2logL_reference(V, y, X)
    assert abs(r["neg2_reml"] - ref) < 1e-8 * abs(ref)


def test_variance_residuelle_fixee():
    """Residuelle fixee a 1 (fixed_idx) : R = diag(1/w), variances connues."""
    terms, res, y, X, w, g, q = dispositif(graine=1)
    th0 = np.array([0.0, 0.0])
    r = fit_reml(terms, res, y, X, theta_init=th0, fixed_idx=[1], verbose=False,
                 hessian=False, blups=False, weights=w)
    assert r["theta"][1] == 0.0
    Z = od.dense_Z_np(terms[0], len(y))
    V = np.exp(2 * r["theta"][0]) * Z @ Z.T + np.diag(1.0 / w)
    assert abs(r["neg2_reml"] - od.neg2logL_reference(V, y, X)) < 1e-8 * abs(r["neg2_reml"])
