"""Batched exact refits (refit_batch.py) against fit_reml run SNP by SNP.

Model: a focal genotype term and a neighbour genotype term, both with a
kinship, iid residual. For each SNP the columns dir = Z_g m and
ind = W m are appended to X; the Wald test of the SNP coefficients
(1 or 2 df) must equal the one computed from fit_reml's beta and vbeta, and
the batched optimum must be at least as good as fit_reml's.
"""
import numpy as np
import pytest

import outils_dispositifs  # noqa: F401  (path to inst/python)

from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml, _bounds
from remlax.refit_batch import refit_batch


def _chol(q, rng):
    A = rng.normal(size=(q, q))
    return np.linalg.cholesky(A @ A.T / q + 0.5 * np.eye(q))


def _model(seed=3, q=30, n=240, p_snp=6):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, q, n)
    nb = rng.integers(0, q, (n, 3))
    W = np.zeros((n, q))
    for j in range(3):
        W[np.arange(n), nb[:, j]] += 1.0 / 3.0
    Wi, Wj = np.nonzero(W)
    terms = [dict(name="gen", struct="iid", t=1, rank=0, q=q, zi=np.arange(n), zj=g,
                  zx=np.ones(n), LK=_chol(q, rng)),
             dict(name="vois", struct="iid", t=1, rank=0, q=q, zi=Wi, zj=Wj,
                  zx=W[Wi, Wj], LK=_chol(q, rng))]
    res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64),
               unit=np.arange(n), n_unit=n, lvl="id")
    M = rng.choice([-1.0, 1.0], size=(q, p_snp))
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = (X @ np.array([2.0, 0.5]) + 0.8 * rng.normal(size=q)[g]
         + W @ rng.normal(size=q) * 0.6 + rng.normal(size=n)
         + 0.7 * M[g, 0])                                 # SNP 0 has a direct effect
    S = np.stack([np.column_stack([M[g, j], W @ M[:, j]]) for j in range(p_snp)])
    return terms, res, y, X, S


@pytest.mark.parametrize("cols,L", [([0], np.eye(1)), ([0, 1], np.eye(2)),
                                    ([0, 1], np.array([[0.0, 1.0]]))])
def test_refit_batch_equals_fit_reml_snp_by_snp(cols, L):
    terms, res, y, X, S = _model()
    S = S[:, :, cols]
    f0 = fit_reml(terms, res, y, X, verbose=False, hessian=False)
    bornes, th0 = _bounds(f0["theta"], len(f0["theta"]), -12.0, 12.0, terms, res, y, None)
    lo, hi = np.array([b[0] for b in bornes]), np.array([b[1] for b in bornes])
    r = refit_batch(terms, res, y, X, S, L, th0, lo, hi, batch=4)
    assert r["converged"].all()
    k = len(cols)
    for s in range(S.shape[0]):
        f = fit_reml(terms, res, y, np.column_stack([X, S[s]]), verbose=False, hessian=False,
                     theta_init=f0["theta"])
        b = np.asarray(f["beta"])[-k:]
        Vb = np.asarray(f["vbeta"])[-k:, -k:]
        Lb = L @ b
        chi2 = Lb @ np.linalg.solve(L @ Vb @ L.T, Lb)
        assert r["neg2_reml"][s] <= f["neg2_reml"] + 1e-7
        assert abs(r["neg2_reml"][s] - f["neg2_reml"]) < 1e-5
        assert r["chi2"][s] == pytest.approx(chi2, rel=1e-5, abs=1e-5)
        # beta in units of its standard error: fit_reml stops within its own tolerance
        assert np.all(np.abs(r["beta"][s] - b) <= 1e-4 * np.sqrt(np.diag(Vb)))
    if np.any(L[:, 0] != 0):
        assert np.argmax(r["chi2"]) == 0                   # the causal SNP (direct effect)
