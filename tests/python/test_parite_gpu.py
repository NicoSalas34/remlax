"""Parite CPU / GPU : meme -2logL et meme gradient a 1e-12 relatif.

Le test S'IGNORE (pytest.skip) si JAX ne voit aucun peripherique cuda : sur
une machine sans GPU il n'y a rien a comparer, et un echec dirait le contraire
de ce qui se passe. Avec un GPU, le meme dispositif est evalue deux fois via
jax.default_device, une fois par peripherique, sur STRICTEMENT les memes
tableaux d'entree.

Le script tests/python/parite_gpu.py fait la meme chose sur les 34 paquets
serialises du catalogue (validation/results, cpu_gpu_parity) ; ce test est la
version courte, collectable par pytest.
"""
import numpy as np
import pytest

import outils_dispositifs as od
import jax
import jax.numpy as jnp
from remlax.model import make_objective, n_theta

TOL_REL = 1e-12


def _gpus():
    out = []
    for plat in ("cuda", "gpu", "rocm"):
        try:
            out += list(jax.devices(plat))
        except RuntimeError:
            pass
    return out


def _dispositif():
    rng = np.random.default_rng(5)
    q, nr, nc = 15, 6, 8
    n = nr * nc
    cell = np.arange(n)
    g = rng.integers(0, q, n)
    A = rng.normal(size=(q, q)); LK = np.linalg.cholesky(A @ A.T / q + np.eye(q))
    terms = [od.terme_facteur("g", g, q, LK=LK),
             od.terme_facteur("champ", cell, n, lvl="ar1ar1", dims=(nr, nc))]
    res = od.residuelle_iid(n)
    y = rng.normal(size=n); X = np.column_stack([np.ones(n), rng.normal(size=n)])
    th = rng.normal(0, 0.3, n_theta(terms, res))
    return terms, res, y, X, th


def _evalue(dev, terms, res, y, X, th):
    with jax.default_device(dev):
        _, fj, _, _ = make_objective(terms, res, y, X)
        v, g = fj(th)
        jax.block_until_ready(v)
    return float(v), np.asarray(g)


def test_parite_cpu_gpu_valeur_et_gradient():
    gpus = _gpus()
    if not gpus:
        pytest.skip("aucun peripherique cuda visible par JAX : parite CPU/GPU non testable ici")
    terms, res, y, X, th = _dispositif()
    v_cpu, g_cpu = _evalue(jax.devices("cpu")[0], terms, res, y, X, th)
    v_gpu, g_gpu = _evalue(gpus[0], terms, res, y, X, th)
    assert abs(v_cpu - v_gpu) <= TOL_REL * abs(v_cpu), (v_cpu, v_gpu)
    ech = np.maximum(np.abs(g_cpu), 1.0)
    assert np.all(np.abs(g_cpu - g_gpu) <= TOL_REL * ech), np.abs(g_cpu - g_gpu) / ech


def test_deux_evaluations_cpu_bit_a_bit_identiques():
    # la partie de la parite qui se verifie PARTOUT : le calcul est deterministe
    terms, res, y, X, th = _dispositif()
    cpu = jax.devices("cpu")[0]
    v1, g1 = _evalue(cpu, terms, res, y, X, th)
    v2, g2 = _evalue(cpu, terms, res, y, X, th)
    assert v1 == v2 and np.array_equal(g1, g2)
