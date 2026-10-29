"""Voie bayesienne (bayes.py).

Les jacobiens se verifient en echantillonnant la LOI A PRIORI SEULE : si le
changement de variable theta -> echelle naturelle est faux, les marges tirees
s'ecartent des lois declarees. Une erreur de jacobien ne se voit pas autrement,
car elle deforme la loi a posteriori sans rien casser.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inst", "python"))

from remlax import _x64  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from remlax import bayes as B  # noqa: E402

pytest.importorskip("blackjax")

# Ces tests n'echantillonnent que des lois a priori : le CPU suffit, et le GPU
# peut etre occupe par un autre processus (cuSolver echoue alors a la creation).
CPU = jax.devices("cpu")[0]


@pytest.fixture(autouse=True)
def _sur_cpu():
    with jax.default_device(CPU):
        yield


def _prior_seul(terms, res, prior, n_par, chaines=8, tirages=500, seed=0):
    """Tire la loi a priori seule avec l'echantillonneur de production (ChEES)."""
    y = np.random.default_rng(1).standard_normal(50)
    X = np.ones((50, 1))
    p = B._resoudre_prior(prior, terms, res, y, X)
    lp, _, _ = B.make_log_prior(terms, res, p)
    z0 = 0.5 * np.random.default_rng(seed).standard_normal((chaines, n_par))
    z, diag = B.run_sampler(lp, z0, warmup=300, draws=tirages, seed=seed, verbose=False)
    assert diag["divergences"] == 0
    return z.reshape(-1, n_par)


def test_jacobien_ecart_type_demi_normal():
    terms = [dict(name="g", struct="iid", t=1, rank=0, q=5, LK=None)]
    res = dict(struct="iid", t=1, rank=0)
    th = _prior_seul(terms, res, {"sd": {"famille": "half_normal", "echelle": 1.0}}, 2)
    sd = np.exp(th[:, 0])
    # mediane d'une demi-normale d'echelle 1 : 0,6745
    assert abs(np.median(sd) - 0.6745) < 0.05
    assert abs(np.quantile(sd, 0.9) - 1.645) < 0.15


def test_jacobien_lkj_uniforme_en_dimension_2():
    terms = [dict(name="g", struct="us", t=2, rank=0, q=5, LK=None)]
    res = dict(struct="iid", t=1, rank=0)
    th = _prior_seul(terms, res, {"sd": {"famille": "half_normal", "echelle": 1.0}, "lkj": 1.0}, 4)
    S = np.array([np.asarray(B.build_sigma(jnp.asarray(t[:3]), "us", 2, 0)) for t in th])
    r = S[:, 1, 0] / np.sqrt(S[:, 0, 0] * S[:, 1, 1])
    sd1 = np.sqrt(S[:, 0, 0])
    # LKJ(1) en dimension 2 : r uniforme sur (-1, 1)
    assert abs(np.mean(np.abs(r)) - 0.5) < 0.05
    assert abs(np.quantile(r, 0.25) + 0.5) < 0.08
    assert abs(np.median(sd1) - 0.6745) < 0.06


def test_jacobien_tanh_uniforme():
    terms = [dict(name="g", struct="iid", t=1, rank=0, q=5, LK=None, lvl="ar1")]
    res = dict(struct="iid", t=1, rank=0)
    th = _prior_seul(terms, res, {"sd": {"famille": "half_normal", "echelle": 1.0}}, 3)
    rho = np.tanh(th[:, 1])
    assert abs(np.mean(np.abs(rho)) - 0.5) < 0.05
    assert abs(np.quantile(rho, 0.9) - 0.8) < 0.06


def test_prior_terme_inconnu_refuse():
    terms = [dict(name="g", struct="iid", t=1, rank=0, q=5, LK=None)]
    res = dict(struct="iid", t=1, rank=0)
    with pytest.raises(ValueError, match="inconnu"):
        B._resoudre_prior({"termes": {"autre": {}}}, terms, res, np.zeros(5), np.ones((5, 1)))


def test_rhat_emboite():
    rng = np.random.default_rng(0)
    melange = rng.standard_normal((8, 200))
    assert abs(B.nested_rhat(melange, 4) - 1.0) < 0.02
    decale = melange + np.repeat(np.arange(4), 2)[:, None]       # super-chaines en desaccord
    assert B.nested_rhat(decale, 4) > 1.5
