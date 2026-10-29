"""Somme des (Z K Z')_ii : la voie par Z'Z et la voie par Z LK concordent."""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
from remlax.start import _sum_diag_ZKZ  # noqa: E402


def directe(Z, LK):
    K = LK @ LK.T
    return float(np.trace(Z.toarray() @ K @ Z.toarray().T))


def test_repetitions_une_entree_par_ligne():
    rng = np.random.default_rng(1)
    q, n = 30, 400                         # nnz(Z'Z) = q < nnz(Z) = n : voie Z'Z
    Z = sp.csr_matrix((rng.normal(size=n), (np.arange(n), rng.integers(0, q, n))), shape=(n, q))
    LK = np.linalg.cholesky(np.cov(rng.normal(size=(q, 3 * q))) + np.eye(q))
    assert np.isclose(_sum_diag_ZKZ(Z, LK), directe(Z, LK), rtol=1e-12)


def test_voisinage_plusieurs_entrees_par_ligne():
    rng = np.random.default_rng(2)
    q, n = 40, 25                          # lignes chargees de 6 voisins : voie Z LK
    rows = np.repeat(np.arange(n), 6); cols = rng.integers(0, q, rows.size)
    Z = sp.csr_matrix((rng.normal(size=rows.size), (rows, cols)), shape=(n, q))
    LK = np.linalg.cholesky(np.cov(rng.normal(size=(q, 3 * q))) + np.eye(q))
    assert np.isclose(_sum_diag_ZKZ(Z, LK), directe(Z, LK), rtol=1e-12)
