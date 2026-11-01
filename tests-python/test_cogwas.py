"""Co-GWAS (cogwas.py) against a dense GLS in dimension n, and its diagonal
against the `sim` family of scan.py.

For every pair the reference builds s_dir = Z_g m_a, s_ind = Z_n m'_b and
u = s_dir * s_ind, then b = (S'PS)^-1 S'Py and chi2 = b_u^2 / [(S'PS)^-1]_uu
for S = [conditioning columns..., u], with P written out (n x n).
"""
import numpy as np
import pytest

import outils_dispositifs  # noqa: F401  (path to inst/python)

from remlax import _x64  # noqa: F401
from remlax.cogwas import PairProjection, cogwas, TESTS
from remlax.scan import project, scan


def _data(seed=5, n=150, qg=12, qf=9, pa=7, pb=6, same=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, qg, n)
    Zg = np.zeros((n, qg)); Zg[np.arange(n), g] = 1.0
    qn = qg if same else qf
    nb = rng.integers(0, qn, (n, 3))
    Zn = np.zeros((n, qn))
    for j in range(3):
        Zn[np.arange(n), nb[:, j]] += 1.0 / 3.0
    Kg = np.cov(rng.standard_normal((qg, 3 * qg))) + 0.3 * np.eye(qg)
    Kn = np.cov(rng.standard_normal((qn, 3 * qn))) + 0.3 * np.eye(qn)
    V = 0.8 * Zg @ Kg @ Zg.T + 0.5 * Zn @ Kn @ Zn.T + np.eye(n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    Ma = rng.choice([-1.0, 1.0], size=(qg, pa))
    Mb = Ma.copy() if same else rng.choice([-1.0, 1.0], size=(qn, pb))
    u0 = (Zg @ Ma[:, 1]) * (Zn @ Mb[:, 2])
    y = X @ np.array([1.0, 0.3]) + np.linalg.cholesky(V) @ rng.normal(size=n) + 0.6 * u0
    Vi = np.linalg.inv(V)
    vbeta = np.linalg.inv(X.T @ Vi @ X)
    P = Vi - Vi @ X @ vbeta @ X.T @ Vi
    fit = dict(Vi=Vi, vbeta=vbeta, Py=P @ y)
    return dict(fit=fit, X=X, Zg=Zg, Zn=Zn, Ma=Ma, Mb=Mb, P=P, y=y)


def _brute(d, a, b, cols):
    sd = d["Zg"] @ d["Ma"][:, a]; sn = d["Zn"] @ d["Mb"][:, b]
    c = {"dir": sd, "ind": sn, "pair": sd * sn}
    S = np.column_stack([c[k] for k in cols])
    D = S.T @ d["P"] @ S
    Di = np.linalg.inv(D)
    bt = Di @ (S.T @ d["P"] @ d["y"])
    return bt[-1] ** 2 / Di[-1, -1], bt[-1], np.sqrt(Di[-1, -1])


@pytest.mark.parametrize("test", TESTS)
def test_every_pair_matches_dense_gls(test):
    d = _data()
    proj = PairProjection(d["fit"], d["X"], d["Zg"], d["Zn"])
    pa, pb = d["Ma"].shape[1], d["Mb"].shape[1]
    r = cogwas(proj, d["Ma"], d["Mb"], tests=[test], top=pa * pb, block_a=3, block_b=4)[test]
    cols = test.split("|")[1].split("+") + ["pair"] if "|" in test else ["pair"]
    ref = np.array([[_brute(d, a, b, cols)[0] for b in range(pb)] for a in range(pa)])
    got = np.full((pa, pb), np.nan)
    got[r["a"], r["b"]] = r["chi2"]
    assert np.allclose(got, ref, rtol=1e-9)
    assert np.all(np.diff(r["chi2"]) <= 0)                       # sorted, largest first
    assert (r["a"][0], r["b"][0]) == (1, 2)                      # the simulated pair
    a, b = r["a"][0], r["b"][0]
    _, beta, se = _brute(d, a, b, cols)
    assert r["beta"][0] == pytest.approx(beta, rel=1e-9) and r["se"][0] == pytest.approx(se, rel=1e-9)
    from scipy.stats import chi2 as c2
    assert np.allclose(r["min_p_a"], c2.sf(ref.max(axis=1), 1), rtol=1e-8)
    assert np.allclose(r["min_p_b"], c2.sf(ref.max(axis=0), 1), rtol=1e-8)


def test_top_keeps_the_largest_pairs():
    d = _data(seed=6, pa=9, pb=11)
    proj = PairProjection(d["fit"], d["X"], d["Zg"], d["Zn"])
    full = cogwas(proj, d["Ma"], d["Mb"], tests=["pair|dir+ind"], top=99, block_a=2, block_b=3)
    few = cogwas(proj, d["Ma"], d["Mb"], tests=["pair|dir+ind"], top=5, block_a=4, block_b=5)
    assert np.allclose(few["pair|dir+ind"]["chi2"], full["pair|dir+ind"]["chi2"][:5], rtol=1e-12)


def test_diagonal_of_the_same_species_is_the_sim_test():
    """Same species, same loci: the pair (a, a) is Sato's similarity, so
    pair|dir+ind on the diagonal equals the scan's sim|dir+ind."""
    d = _data(seed=7, same=True, pa=8)
    proj = PairProjection(d["fit"], d["X"], d["Zg"], d["Zn"])
    p = d["Ma"].shape[1]
    r = cogwas(proj, d["Ma"], d["Mb"], tests=["pair|dir+ind", "pair"], top=p * p)
    sp = project(d["fit"], d["X"], {"dir": d["Zg"], "ind": d["Zn"]})
    sc = scan(sp, d["Ma"], tests=["sim|dir+ind", "sim"])
    for t_pair, t_sim in (("pair|dir+ind", "sim|dir+ind"), ("pair", "sim")):
        got = np.full((p, p), np.nan)
        got[r[t_pair]["a"], r[t_pair]["b"]] = r[t_pair]["chi2"]
        assert np.allclose(np.diag(got), sc["chi2_" + t_sim], rtol=1e-9), t_pair


def test_jax_device_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    d = _data(seed=8)
    a = cogwas(PairProjection(d["fit"], d["X"], d["Zg"], d["Zn"]), d["Ma"], d["Mb"],
               tests=["pair|dir+ind"], top=42)["pair|dir+ind"]
    b = cogwas(PairProjection(d["fit"], d["X"], d["Zg"], d["Zn"], xp=jnp), d["Ma"], d["Mb"],
               tests=["pair|dir+ind"], top=42)["pair|dir+ind"]
    assert np.array_equal(a["a"], b["a"]) and np.array_equal(a["b"], b["b"])
    assert np.allclose(a["chi2"], b["chi2"], rtol=1e-10)
