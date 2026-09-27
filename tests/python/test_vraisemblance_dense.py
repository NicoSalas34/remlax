"""-2 logL de remlax contre un calcul dense NAIF en numpy, sur plusieurs modeles.

La reference assemble V a la main — Sigma (x) K explicitement, R terme a
terme — puis applique la definition de la vraisemblance restreinte par deux
Cholesky numpy (outils_dispositifs.neg2logL_reference). Aucune fonction de
model.py n'est utilisee du cote reference. Les modeles couvrent : K = I, K
fournie, multi-caractere long avec residuelle us, dsum a deux sections de
formes differentes, produit separable id (x) ar1 (x) ar1, residuelle ar1ar1,
structure fa avec residuelle diag, et une famille metrique avec coordonnees.
Puis les refus de validate() et les comptages de theta.
"""
import numpy as np
import pytest

import outils_dispositifs as od
import jax
import jax.numpy as jnp
from remlax import structures as S
from remlax.model import (dense_Z, neg2_reml, n_theta, split_theta, make_objective,
                          assemble_V, term_n_params)
from remlax.fit import validate, initial_theta, fit_reml
from remlax.inference import component_names

RNG = np.random.default_rng(11)
TOL = 1e-8


def _kron_G(theta_sig, tm, K):
    Sig = np.asarray(S.build_sigma(jnp.asarray(theta_sig), tm["struct"], tm["t"], tm["rank"]))
    return np.kron(Sig, K)


def _V_ref(theta, terms, Ks, res_fn, n):
    """V par la definition : sum Z (Sigma (x) K) Z' + R, K et R fournis a la main."""
    V = np.zeros((n, n))
    o = 0
    for tm, K in zip(terms, Ks):
        p = term_n_params(tm)
        th = theta[o:o + p]
        ns = S.n_params(tm["struct"], tm["t"], tm["rank"])
        Z = od.dense_Z_np(tm, n)
        Kq = K(th[ns:]) if callable(K) else K
        V += Z @ _kron_G(th[:ns], tm, Kq) @ Z.T
        o += p
    return V + res_fn(theta[o:])


def _compare(terms, res, y, X, theta, Ks, res_fn):
    n = len(y)
    Zs = [dense_Z(t, n) for t in terms]
    v = float(neg2_reml(jnp.asarray(theta), terms, Zs, res, jnp.asarray(y), jnp.asarray(X)))
    V_ref = _V_ref(theta, terms, Ks, res_fn, n)
    V_rx = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    assert np.allclose(V_rx, V_ref, atol=1e-9, rtol=1e-9), np.abs(V_rx - V_ref).max()
    ref = od.neg2logL_reference(V_ref, y, X)
    assert abs(v - ref) <= TOL * max(1.0, abs(ref)), (v, ref)
    return v


def _iid_R(n, theta_r):
    return lambda th: np.exp(2 * th[0]) * np.eye(n)


def test_iid_K_identite():
    n, q = 90, 15
    g = RNG.integers(0, q, n)
    tm = od.terme_facteur("g", g, q)
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.column_stack([np.ones(n), RNG.normal(size=n)])
    theta = np.array([0.2, -0.1])
    _compare([tm], res, y, X, theta, [np.eye(q)], _iid_R(n, None))


def test_parente_fournie_LK():
    n, q = 80, 12
    A = RNG.normal(size=(q, q)); K = A @ A.T / q + np.eye(q)
    LK = np.linalg.cholesky(K)
    g = RNG.integers(0, q, n)
    tm = od.terme_facteur("g", g, q, LK=LK)           # lvl absent -> "fixed"
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    theta = np.array([0.3, 0.05])
    _compare([tm], res, y, X, theta, [K], _iid_R(n, None))


def test_multitrait_us_long_avec_residuelle_us():
    q, t, nu = 20, 3, 40
    unit = np.repeat(np.arange(nu), t); trait = np.tile(np.arange(t), nu)
    g = RNG.integers(0, q, nu)[unit]
    n = nu * t
    tm = od.terme_long("g", g, trait, q, t, struct="us")
    res = od.residuelle("us", trait, unit, t)
    y = RNG.normal(size=n)
    X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    theta = RNG.normal(0, 0.4, n_theta([tm], res))
    ns = S.n_params("us", t)

    def R_fn(th):
        Sr = np.asarray(S.build_sigma(jnp.asarray(th[:ns]), "us", t))
        return Sr[trait[:, None], trait[None, :]] * (unit[:, None] == unit[None, :])
    _compare([tm], res, y, X, theta, [np.eye(q)], R_fn)


def test_dsum_deux_sections_de_formes_differentes():
    # section A : 50 lignes, residuelle iid ; section B : 40 lignes en 4 colonnes
    # de 10, ar1 entre unites (une observation par unite). Les lignes des deux
    # sections sont ENTREMELEES pour verifier la dispersion par `rows`.
    nA, nB = 50, 40
    n = nA + nB
    perm = RNG.permutation(n)
    rowsA, rowsB = np.sort(perm[:nA]), np.sort(perm[nA:])
    secA = dict(struct="iid", t=1, rank=0, trait=np.zeros(nA, np.int64),
                unit=np.arange(nA), n_unit=nA, lvl="id", rows=rowsA, name="A")
    unitB = np.arange(nB)
    secB = dict(struct="iid", t=1, rank=0, trait=np.zeros(nB, np.int64),
                unit=unitB, n_unit=nB, lvl="ar1", rows=rowsB, name="B")
    res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64), unit=np.arange(n),
               n_unit=n, sections=[secA, secB])
    q = 10
    g = RNG.integers(0, q, n)
    tm = od.terme_facteur("g", g, q)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    theta = np.array([0.1, -0.2, 0.3, 0.5])         # g ; A ; B (var, rho)
    assert n_theta([tm], res) == 4

    def R_fn(th):
        R = np.zeros((n, n))
        R[np.ix_(rowsA, rowsA)] = np.exp(2 * th[0]) * np.eye(nA)
        R[np.ix_(rowsB, rowsB)] = np.exp(2 * th[1]) * od.ar1_corr_np(np.tanh(th[2]), nB)
        return R
    _compare([tm], res, y, X, theta, [np.eye(q)], R_fn)
    assert component_names([tm], res) == ["g", "A", "B", "B!ar1"]


def test_produit_separable_id_ar1_ar1():
    nb, nr, nc = 2, 4, 5
    q = nb * nr * nc
    n = 2 * q
    cell = np.tile(np.arange(q), 2)
    tm = od.terme_facteur("champ", cell, q, lvl="sep",
                          lvl_parts=[("id", nb), ("ar1", nc), ("ar1", nr)])
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    theta = np.array([0.2, 0.4, -0.3, -0.1])
    assert term_n_params(tm) == 3

    def K_fn(th):
        return np.kron(np.eye(nb), np.kron(od.ar1_corr_np(np.tanh(th[0]), nc),
                                            od.ar1_corr_np(np.tanh(th[1]), nr)))
    _compare([tm], res, y, X, theta, [K_fn], _iid_R(n, None))


def test_residuelle_ar1ar1_avec_us_caractere():
    nr, nc, t = 5, 6, 2
    cell = np.arange(nr * nc)
    unit = np.tile(cell, t); trait = np.repeat(np.arange(t), nr * nc)
    n = nr * nc * t
    res = od.residuelle("us", trait, unit, t, lvl="ar1ar1", dims=(nr, nc))
    q = 8
    g = RNG.integers(0, q, nr * nc)[unit]
    tm = od.terme_long("g", g, trait, q, t, struct="diag")
    y = RNG.normal(size=n)
    X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    theta = RNG.normal(0, 0.3, n_theta([tm], res))
    ns = S.n_params("us", t)

    def R_fn(th):
        Sr = np.asarray(S.build_sigma(jnp.asarray(th[:ns]), "us", t))
        C = np.kron(od.ar1_corr_np(np.tanh(th[ns]), nr), od.ar1_corr_np(np.tanh(th[ns + 1]), nc))
        return Sr[trait[:, None], trait[None, :]] * C[unit[:, None], unit[None, :]]
    _compare([tm], res, y, X, theta, [np.eye(q)], R_fn)


def test_fa_avec_residuelle_diag():
    q, t, nu = 15, 3, 30
    unit = np.repeat(np.arange(nu), t); trait = np.tile(np.arange(t), nu)
    g = RNG.integers(0, q, nu)[unit]
    n = nu * t
    tm = od.terme_long("g", g, trait, q, t, struct="fa", rank=1)
    res = od.residuelle("diag", trait, unit, t)
    y = RNG.normal(size=n)
    X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    theta = RNG.normal(0, 0.3, n_theta([tm], res))

    def R_fn(th):
        return np.diag(np.exp(2 * th[trait]))
    _compare([tm], res, y, X, theta, [np.eye(q)], R_fn)


def test_metrique_exp_avec_coordonnees():
    q, n = 12, 70
    coord = np.sort(RNG.uniform(0, 20, q))
    pos = RNG.integers(0, q, n)
    tm = od.terme_facteur("pos", pos, q, lvl="exp", coord=coord.reshape(q, 1))
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    theta = np.array([0.1, 0.9, 0.0])

    def K_fn(th):
        ph = abs(np.tanh(th[0]))
        return ph ** np.abs(coord[:, None] - coord[None, :])
    _compare([tm], res, y, X, theta, [K_fn], _iid_R(n, None))


def test_terme_pondere_et_exposition_dans_initial_theta():
    # incidence PONDEREE (voisinage) : les poids passent par zx ; initial_theta
    # divise par l'exposition moyenne k = mean(sum_j w_ij^2).
    n, q = 60, 10
    g = RNG.integers(0, q, n)
    w = RNG.uniform(2, 4, n)
    tm = od.terme_facteur("vois", g, q, poids=w)
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    theta = np.array([-0.2, 0.1])
    Z = od.dense_Z_np(tm, n)
    assert np.allclose(Z.sum(axis=1), w)
    _compare([tm], res, y, X, theta, [np.eye(q)], _iid_R(n, None))
    th0 = initial_theta([tm], res, y)
    k = float(np.mean(w ** 2))
    part = np.var(y) / 2.0
    assert th0[0] == pytest.approx(0.5 * np.log(part / k))
    assert th0[1] == pytest.approx(0.5 * np.log(part))


def test_split_theta_et_make_objective_echelle():
    n, q = 50, 8
    g = RNG.integers(0, q, n)
    tm = od.terme_facteur("g", g, q, lvl="ar1")
    res = od.residuelle_iid(n)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    assert n_theta([tm], res) == 3
    th = np.array([0.1, 0.2, 0.3])
    parts, th_r, o = split_theta(th, [tm], res)
    assert np.allclose(parts[0], [0.1, 0.2]) and np.allclose(th_r, [0.3]) and o == 3
    fsc, fj, sc, Zs = make_objective([tm], res, y, X)
    assert sc == n
    v1, g1 = fsc(th); v2, g2 = fj(th)
    assert v2 == pytest.approx(v1 * n) and np.allclose(g2, g1 * n)
    # gradient analytique contre differences finies centrees
    eps = 1e-6
    for j in range(3):
        e = np.zeros(3); e[j] = eps
        fd = (fj(th + e)[0] - fj(th - e)[0]) / (2 * eps)
        assert fd == pytest.approx(g2[j], rel=1e-5, abs=1e-6)


def test_validate_refuse_les_dispositifs_singuliers():
    n, q = 40, 5
    g = RNG.integers(0, q, n)
    tm = od.terme_facteur("g", g, q)
    y = RNG.normal(size=n); X = np.ones((n, 1))
    # (unite, caractere) en double sous une residuelle us
    unit = np.repeat(np.arange(n // 2), 2); trait = np.zeros(n, np.int64)
    with pytest.raises(ValueError, match="en double"):
        validate([tm], od.residuelle("us", trait, unit, 2), y, X)
    # meme unite deux fois sous une structure ar1 entre unites, residuelle iid
    with pytest.raises(ValueError, match="en double"):
        validate([tm], od.residuelle("iid", trait, unit, 1, lvl="ar1"), y, X)
    # indice de caractere hors de t
    tr2 = np.tile([0, 1, 2], n // 3 + 1)[:n]
    with pytest.raises(ValueError, match="hors de la structure"):
        validate([tm], od.residuelle("diag", tr2, np.arange(n), 2), y, X)
    # incidence entierement nulle
    tm0 = dict(tm); tm0["zx"] = np.zeros(n)
    with pytest.raises(ValueError, match="entierement nulle"):
        validate([tm0], od.residuelle_iid(n), y, X)
    # y non fini ; X non finie ; X de rang deficient
    y2 = y.copy(); y2[3] = np.nan
    with pytest.raises(ValueError, match="non finies"):
        validate([tm], od.residuelle_iid(n), y2, X)
    X2 = X.copy(); X2[0, 0] = np.inf
    with pytest.raises(ValueError, match="non finies"):
        validate([tm], od.residuelle_iid(n), y, X2)
    with pytest.raises(ValueError, match="rang"):
        validate([tm], od.residuelle_iid(n), y, np.column_stack([X, X]))
    # structure residuelle inconnue et theta de mauvaise longueur
    with pytest.raises(ValueError, match="structure residuelle inconnue"):
        Zs = [dense_Z(tm, n)]
        neg2_reml(jnp.zeros(2), [tm], Zs, od.residuelle("corh", trait, np.arange(n), 1), y, X)
    with pytest.raises(ValueError, match="longueur"):
        fit_reml([tm], od.residuelle_iid(n), y, X, theta_init=np.zeros(5), verbose=False,
                 maxiter=0, polish=0, hessian=False, blups=False)


def test_initial_theta_a_la_bonne_longueur_pour_les_portees():
    n, q = 40, 8
    coord = np.column_stack([RNG.uniform(0, 10, q), RNG.uniform(0, 10, q)])
    for kind, nl in (("sph", 1), ("mtrn", 1), ("aexp", 2)):
        tm = od.terme_facteur("s", RNG.integers(0, q, n), q, lvl=kind, coord=coord)
        res = od.residuelle_iid(n)
        th0 = initial_theta([tm], res, RNG.normal(size=n))
        assert len(th0) == 2 + nl
        if kind == "sph":
            # portee de depart : le quart de l'etendue des coordonnees
            assert th0[1] == pytest.approx(np.log(0.25 * np.ptp(coord, axis=0).max()))
