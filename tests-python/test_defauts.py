"""Defauts corriges apres le nettoyage de 2026-10 : un test par defaut."""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml


def _terme_kinv(graine=2, q=15, rep=4):
    rng = np.random.default_rng(graine)
    A = np.eye(q) + 0.3 * (np.ones((q, q)) - np.eye(q)) / q
    codes = np.repeat(np.arange(q), rep)
    n = len(codes)
    tm = od.terme_facteur("anim", codes, q)
    tm.update(Kinv=np.linalg.inv(A), Kinv_logdet=float(-np.linalg.slogdet(A)[1]), LK=None)
    y = rng.normal(size=n)
    return [tm], od.residuelle_iid(n), y, np.ones((n, 1))


@pytest.mark.parametrize("moteur", ["dense", "blocs"])
def test_terme_kinv_seul_refuse_hors_mme(moteur):
    terms, res, y, X = _terme_kinv()
    with pytest.raises(ValueError, match="precision Kinv"):
        fit_reml(terms, res, y, X, moteur=moteur, verbose=False, hessian=False, blups=False)


def test_terme_kinv_seul_accepte_par_le_mme():
    terms, res, y, X = _terme_kinv()
    r = fit_reml(terms, res, y, X, moteur="mme", verbose=False, hessian=False, blups=False)
    assert np.isfinite(r["logLik"])


def test_terme_kinv_seul_refuse_par_la_voie_bayesienne():
    from remlax.bayes import prepare
    terms, res, y, X = _terme_kinv()
    with pytest.raises(ValueError, match="precision Kinv"):
        prepare(terms, res, y, X)
