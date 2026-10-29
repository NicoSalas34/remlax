"""Sortie du plancher apres L-BFGS-B (fit._evasion_plancher).

Une variance en log pres du plancher a un gradient en theta nul : L-BFGS-B
s'y arrete. Le test pousse la variance d'un terme au plancher, verifie que
_evasion_plancher la remonte et que -2logL baisse, puis qu'un ajustement parti
de ce point retrouve l'optimum.
"""
import numpy as np

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml, _evasion_plancher
from remlax.model import make_objective


def dispositif(graine=0, n=200, q=25):
    rng = np.random.default_rng(graine)
    g = rng.integers(0, q, n)
    tm = od.terme_facteur("g", g, q)
    res = od.residuelle_iid(n)
    X = np.ones((n, 1))
    y = 1.0 + 1.2 * rng.normal(size=q)[g] + rng.normal(size=n)
    return [tm], res, y, X


def test_variance_remontee_du_plancher():
    terms, res, y, X = dispositif()
    _, fun_jac, _, _ = make_objective(terms, res, y, X)
    floor, ceil = -12.0, 12.0
    th0 = np.array([0.0, 0.0])
    coince = np.array([-11.5, 0.0])                  # variance genetique au plancher
    f0, g0 = fun_jac(coince)
    assert abs(g0[0]) < 1e-6                         # gradient en theta quasi nul
    th = _evasion_plancher(fun_jac, coince, th0, [(floor, ceil)] * 2, terms, res, floor,
                           verbose=False)
    assert th is not None and th[0] > -2.0
    assert fun_jac(th)[0] < f0
    # sans rien au plancher : rien a faire
    assert _evasion_plancher(fun_jac, th0, th0, [(floor, ceil)] * 2, terms, res, floor,
                             verbose=False) is None


def test_ajustement_parti_du_plancher():
    terms, res, y, X = dispositif(graine=1)
    ref = fit_reml(terms, res, y, X, verbose=False, hessian=False, blups=False)
    r = fit_reml(terms, res, y, X, theta_init=np.array([-11.5, 0.0]), verbose=False,
                 hessian=False, blups=False, optimizer="lbfgsb")
    assert abs(r["logLik"] - ref["logLik"]) < 1e-5
