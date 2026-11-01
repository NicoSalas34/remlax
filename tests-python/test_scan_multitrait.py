"""Multi-trait scan at fixed V: every test against a brute-force GLS in
dimension n, for a known V (no fit needed: the scan reads Vi, vbeta, Py).

Long format, T traits, a direct and a neighbour incidence; the covariate of
family f for trait k is D_k s_f. For each SNP the reference builds S, then
    b = (S'PS)^-1 S'Py,  chi2 = (L b)' [L (S'PS)^-1 L']^-1 (L b).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inst", "python"))
from remlax.scan import project, scan  # noqa: E402


def _data(seed=4, T=2, q=25, n_unit=120, p=12):
    rng = np.random.default_rng(seed)
    gen = rng.integers(0, q, n_unit)
    nb = rng.integers(0, q, (n_unit, 3))
    Zu = np.zeros((n_unit, q)); Zu[np.arange(n_unit), gen] = 1.0
    Wu = np.zeros((n_unit, q))
    for j in range(3):
        Wu[np.arange(n_unit), nb[:, j]] += 1.0 / 3.0
    # long format: unit-major, trait-minor, with one record of trait 2 missing
    rows = [(i, k) for i in range(n_unit) for k in range(T) if not (i == 5 and k == 1)]
    unit = np.array([r[0] for r in rows]); trait = np.array([r[1] for r in rows])
    n = len(rows)
    Zdir = [np.where((trait == k)[:, None], Zu[unit], 0.0) for k in range(T)]
    Zind = [np.where((trait == k)[:, None], Wu[unit], 0.0) for k in range(T)]
    K = np.cov(rng.standard_normal((q, 3 * q))) + 0.2 * np.eye(q)
    Sg = np.array([[1.0, 0.4], [0.4, 0.5]])[:T, :T]
    Se = np.array([[1.0, -0.3], [-0.3, 2.0]])[:T, :T]
    Zfull = sum(Zdir)
    Vg = np.zeros((n, n))
    for a in range(T):
        for b in range(T):
            Vg += Sg[a, b] * Zdir[a] @ K @ Zdir[b].T
    Ve = Se[trait[:, None], trait[None, :]] * (unit[:, None] == unit[None, :])
    V = Vg + Ve
    X = np.stack([(trait == k).astype(float) for k in range(T)], axis=1)
    y = np.linalg.cholesky(V) @ rng.standard_normal(n) + X @ np.arange(1, T + 1) * 10.0
    M = rng.choice([-1.0, 1.0], size=(q, p))
    Vi = np.linalg.inv(V)
    vbeta = np.linalg.inv(X.T @ Vi @ X)
    P = Vi - Vi @ X @ vbeta @ X.T @ Vi
    fit = dict(Vi=Vi, vbeta=vbeta, Py=P @ y)
    return dict(fit=fit, X=X, y=y, P=P, Zdir=Zdir, Zind=Zind, Zfull=Zfull,
                Wfull=sum(Zind), M=M, trait=trait, T=T, n=n, p=p)


def _brute(d, j, fams, L):
    m = d["M"][:, j]
    cols = []
    for f in fams:
        for k in range(d["T"]):
            mask = (d["trait"] == k).astype(float)
            if f == "dir":
                s = d["Zfull"] @ m
            elif f == "ind":
                s = d["Wfull"] @ m
            else:
                s = (d["Zfull"] @ m) * (d["Wfull"] @ m)
            cols.append(mask * s)
    S = np.stack(cols, axis=1)
    I = S.T @ d["P"] @ S
    b = np.linalg.solve(I, S.T @ d["P"] @ d["y"])
    Lb = L @ b
    return float(Lb @ np.linalg.solve(L @ np.linalg.solve(I, L.T), Lb)), b


@pytest.mark.parametrize("spec,fams,L", [
    ("dir", ["dir"], np.eye(2)),
    ("dir[2]", ["dir"], np.array([[0.0, 1.0]])),
    ("dir@het", ["dir"], np.array([[-1.0, 1.0]])),
    ("ind|dir", ["ind", "dir"], np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])),
    ("dir+ind", ["dir", "ind"], np.eye(4)),
    ("sim|dir", ["sim", "dir"], np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])),
    ("sim[1]|dir+ind", ["sim", "dir", "ind"], np.eye(6)[[0]]),
])
def test_tests_match_brute_force_gls(spec, fams, L):
    d = _data()
    proj = project(d["fit"], d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    r = scan(proj, d["M"], tests=[spec])
    for j in range(d["p"]):
        ref, _ = _brute(d, j, fams, L)
        assert r["chi2_" + spec][j] == pytest.approx(ref, rel=1e-8)
    assert r["df_" + spec][0] == L.shape[0]


def test_common_effect_and_identity_any_common_het():
    d = _data()
    proj = project(d["fit"], d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    r = scan(proj, d["M"], tests=["dir", "dir@common", "dir@het"])
    for j in range(d["p"]):
        m = d["M"][:, j]
        s = d["Zfull"] @ m                       # sum_k D_k s = s
        I = s @ d["P"] @ s
        assert r["chi2_dir@common"][j] == pytest.approx((s @ d["P"] @ d["y"]) ** 2 / I, rel=1e-8)
    np.testing.assert_allclose(r["chi2_dir"], r["chi2_dir@common"] + r["chi2_dir@het"],
                               rtol=1e-9)
    assert r["df_dir@common"][0] == 1 and r["df_dir@het"][0] == 1


def test_per_trait_effects_and_single_matrix_split():
    d = _data()
    proj = project(d["fit"], d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    r = scan(proj, d["M"], tests=["dir"])
    for j in range(3):
        _, b = _brute(d, j, ["dir"], np.eye(2))
        assert r["beta_dir[1]"][j] == pytest.approx(b[0], rel=1e-8)
        assert r["beta_dir[2]"][j] == pytest.approx(b[1], rel=1e-8)
    # the same scan from full-height incidences and the trait of each row
    proj2 = project(d["fit"], d["X"], {"dir": d["Zfull"], "ind": d["Wfull"]},
                    trait=d["trait"], traits=["1", "2"])
    r2 = scan(proj2, d["M"], tests=["dir", "sim|dir+ind", "ind@common|dir"])
    r1 = scan(proj, d["M"], tests=["dir", "sim|dir+ind", "ind@common|dir"])
    for k in ("chi2_dir", "chi2_sim|dir+ind", "chi2_ind@common|dir"):
        np.testing.assert_allclose(r2[k], r1[k], rtol=1e-10)


def test_one_trait_is_the_univariate_scan():
    d = _data(T=1)
    Z1, W1 = d["Zdir"][0], d["Zind"][0]
    a = scan(project(d["fit"], d["X"], {"dir": Z1, "ind": W1}), d["M"],
             tests=["dir", "ind|dir", "sim|dir+ind"])
    b = scan(project(d["fit"], d["X"], {"dir": [Z1], "ind": [W1]}), d["M"],
             tests=["dir", "ind|dir", "sim|dir+ind"])
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    with pytest.raises(ValueError, match="@het needs at least two traits"):
        scan(project(d["fit"], d["X"], {"dir": Z1}), d["M"], tests=["dir@het"])


def test_grammar_errors():
    d = _data()
    proj = project(d["fit"], d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    with pytest.raises(ValueError, match="unknown trait"):
        scan(proj, d["M"], tests=["dir[zz]"])
    with pytest.raises(ValueError, match="cannot be combined"):
        scan(proj, d["M"], tests=["dir[1]@common"])
    with pytest.raises(ValueError, match="both with and"):
        scan(proj, d["M"], tests=["dir@common|dir"])


@pytest.mark.parametrize("T", [1, 2])
def test_sim_on_a_jax_device_matches_numpy(T):
    """The `sim` step computed with jax.numpy (on the default JAX device, a
    GPU when there is one) gives the statistics of the numpy path."""
    jax = pytest.importorskip("jax")
    from remlax import _x64  # noqa: F401
    import jax.numpy as jnp
    d = _data(T=T, seed=9, p=40)
    proj = project(d["fit"], d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    tests = ["sim", "sim|dir+ind", "dir+ind+sim"] + (["sim@het|dir", "sim[2]|dir"] if T == 2 else [])
    ref = scan(proj, d["M"], tests=tests, block=7)
    gpu = scan(proj, d["M"], tests=tests, block=7, xp=jnp)
    for key, v in ref.items():
        v = np.asarray(v, dtype=float)
        assert np.allclose(np.asarray(gpu[key], dtype=float), v, rtol=1e-10, atol=1e-12,
                           equal_nan=True), key
