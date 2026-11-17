"""AI-REML ends where the convergence criteria of remlax hold.

AI stopped on its own decrement (1e-6 in -2logL units) and skipped the Newton
polishing, so a converged AI fit could fail the relative gradient criterion
(GRAD_REL_THRESHOLD = 1e-6) that summary() and rx_scan_loco() read: on a
random intercept with 30 levels x 4, decrement 2.7e-7 but grad_rel 2.5e-5
after 2 iterations, while L-BFGS-B with polishing reached 1.6e-11.
"""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax.fit import fit_reml


def _plan(seed):
    rng = np.random.default_rng(seed)
    q, rep = 30, 4
    g = np.repeat(np.arange(q), rep)
    x = rng.normal(size=q * rep)
    y = 1 + 0.5 * x + rng.normal(size=q)[g] + rng.normal(size=q * rep)
    X = np.column_stack([np.ones(q * rep), x])
    return [od.terme_facteur("gid", g, q)], od.residuelle_iid(q * rep), y, X


@pytest.mark.parametrize("engine", ["dense", "mme"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_converged_ai_fit_meets_the_gradient_criterion(engine, seed):
    terms, res, y, X = _plan(seed)
    f = fit_reml(terms, res, y, X, optimizer="ai", engine=engine, verbose=False)
    assert f["conv_decrement"] is True
    assert f["conv_grad_rel"] is True, f["grad_rel"]
    ref = fit_reml(terms, res, y, X, optimizer="lbfgsb", engine="dense", verbose=False)
    assert abs(f["neg2_reml"] - ref["neg2_reml"]) < 1e-6


def _plan_weak(seed):
    """A well identified factor and a weak one (variance 0.01, standard error
    about twice as large), as sigma2 D in the A+D+AA pig model of
    remlax-validation/correctness/published_data, where AI stopped 1.2e-3 away from asreml
    and gaston with a decrement threshold of 1e-6."""
    rng = np.random.default_rng(seed)
    q, rep, qw = 40, 3, 60
    g = np.repeat(np.arange(q), rep)
    w = rng.integers(0, qw, q * rep)
    y = rng.normal(size=q)[g] + rng.normal(0, 0.1, qw)[w] + rng.normal(size=q * rep)
    X = np.ones((q * rep, 1))
    return ([od.terme_facteur("g", g, q), od.terme_facteur("w", w, qw)],
            od.residuelle_iid(q * rep), y, X)


@pytest.mark.parametrize("engine", ["dense", "mme"])
@pytest.mark.parametrize("seed", [4, 5, 6])
def test_ai_reaches_a_weak_component_as_closely_as_newton_polishing(engine, seed):
    terms, res, y, X = _plan_weak(seed)
    f = fit_reml(terms, res, y, X, optimizer="ai", engine=engine, verbose=False)
    ref = fit_reml(terms, res, y, X, optimizer="lbfgsb", engine="dense", verbose=False)
    v, v_ref = np.exp(2 * f["theta"]), np.exp(2 * ref["theta"])
    keep = v_ref > 1e-6                            # components off the floor
    assert np.max(np.abs(v[keep] / v_ref[keep] - 1)) < 1e-4, (v, v_ref)
