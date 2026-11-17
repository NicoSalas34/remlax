"""Automatic choice of engine and optimiser (engine = "auto", optimizer = "auto").

The rule follows the absolute benchmark of remlax-validation (banc_kmmr/absolu,
fig_rule): one evaluation of the dense MME costs less than one of the V form
while dim(C) / n stays at or below 4, on 8 CPU cores as on an A100. Above, the
V form is cheaper (16 000 genotypes predicted from 1 000 plants: 128 s in MME
against 15 s in V on CPU). AI-REML is chosen on both engines: on the same
benchmark the V form with AI beat L-BFGS-B on every case but one, and
L-BFGS-B declared convergence 2.3e-3 below the optimum on a us(8) of nearly
rank 2 (verif_simulations, boundary_us8_near_rank2).
"""
import numpy as np

import outils_dispositifs as od
from remlax.fit import _resolve_auto, MME_MAX_DIMC_OVER_N


def _plan(q, n, seed=0):
    """n observations of a factor with q levels (levels beyond n unobserved)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, min(q, n), n)
    terms = [od.terme_facteur("g", g, q)]
    return terms, od.residuelle_iid(n), rng.normal(size=n)


def _auto(terms, res, y, **kw):
    kw = dict(dict(engine="auto", optimizer="auto"), **kw)
    return _resolve_auto(terms, res, y, kw)


def test_few_levels_against_n_take_the_mme_with_ai():
    terms, res, y = _plan(q=50, n=400)
    k = _auto(terms, res, y)
    assert (k["engine"], k["optimizer"]) == ("mme", "ai")


def test_many_more_levels_than_observations_take_the_v_form_with_ai():
    terms, res, y = _plan(q=2000, n=300)         # dim(C) / n = 6.7
    k = _auto(terms, res, y)
    assert (k["engine"], k["optimizer"]) == ("dense", "ai")


def test_the_threshold_is_inclusive_on_the_mme_side():
    n = 100
    q = int(MME_MAX_DIMC_OVER_N * n)             # random levels = 4 n
    terms, res, y = _plan(q=q, n=n)
    assert _auto(terms, res, y)["engine"] == "mme"
    terms, res, y = _plan(q=q + 1, n=n)
    assert _auto(terms, res, y)["engine"] == "dense"


def test_a_spatial_lag_keeps_the_dense_engine_and_l_bfgs_b():
    terms, res, y = _plan(q=50, n=400)
    k = _auto(terms, res, y, lag={"W": None})
    assert (k["engine"], k["optimizer"]) == ("dense", "lbfgsb")


def test_explicit_choices_are_left_alone():
    terms, res, y = _plan(q=2000, n=300)
    k = _auto(terms, res, y, engine="mme")
    assert (k["engine"], k["optimizer"]) == ("mme", "ai")
    k = _auto(terms, res, y, optimizer="lbfgsb")
    assert (k["engine"], k["optimizer"]) == ("dense", "lbfgsb")
    k = _resolve_auto(terms, res, y, dict(engine="dense", optimizer="lbfgsb"))
    assert (k["engine"], k["optimizer"]) == ("dense", "lbfgsb")


def test_a_term_given_only_by_its_precision_keeps_the_mme():
    terms, res, y = _plan(q=2000, n=300)
    terms[0]["Kinv"] = np.eye(2000)              # no factor LK: dense would ignore it
    assert _auto(terms, res, y)["engine"] == "mme"
