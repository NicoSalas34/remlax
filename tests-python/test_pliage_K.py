"""Pliage de la parente dans l'incidence (model.prefold_fixed_K) : la V obtenue
avec le terme plie, compresse ou non, egale Z (Sigma (x) K) Z'."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
from remlax import _x64  # noqa: F401,E402
from remlax.model import prefold_fixed_K, restrict_Z  # noqa: E402


def terme(n, t, q, q_obs, rng):
    """t composantes, chaque ligne charge une composante et un genotype parmi
    les q_obs premiers : les q - q_obs autres sont dans K sans observation."""
    comp = rng.integers(0, t, n); gen = rng.integers(0, q_obs, n)
    A = rng.normal(size=(q, 2 * q))
    K = A @ A.T / (2 * q) + 0.05 * np.eye(q)
    return dict(name="g", t=t, q=q, zi=np.arange(n), zj=comp * q + gen, zx=np.ones(n),
                LK=np.linalg.cholesky(K), struct="us", rank=0), K


def V_pliee(tm2, ZL, S):
    t, q2 = tm2["t"], tm2["q"]
    B = [ZL[:, a * q2:(a + 1) * q2] for a in range(t)]
    return sum(S[a, b] * B[a] @ B[b].T for a in range(t) for b in range(t))


def verifier(n, t, q, q_obs, compresse):
    rng = np.random.default_rng(n + q)
    tm, K = terme(n, t, q, q_obs, rng)
    Z = np.zeros((n, t * q)); Z[tm["zi"], tm["zj"]] = 1.0
    L = rng.normal(size=(t, t)); S = L @ L.T + np.eye(t)
    V_ref = Z @ np.kron(S, K) @ Z.T
    Zr = restrict_Z(tm, np.arange(n), n)
    (tm2,), (ZL,) = prefold_fixed_K([tm], [Zr])
    assert (tm2["q"] < q) == compresse
    assert np.allclose(V_pliee(tm2, np.asarray(ZL), S), V_ref, rtol=1e-10, atol=1e-10)
    # second appel : resultat garde dans le terme d'origine
    (tm3,), (ZL3,) = prefold_fixed_K([tm], [Zr])
    assert tm3["q"] == tm2["q"] and np.array_equal(np.asarray(ZL3), np.asarray(ZL))


def test_compresse_genotypes_a_predire():
    verifier(n=60, t=2, q=200, q_obs=40, compresse=True)


def test_sans_compression_plan_complet():
    verifier(n=400, t=2, q=50, q_obs=50, compresse=False)
