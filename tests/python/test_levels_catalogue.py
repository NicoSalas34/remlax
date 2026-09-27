"""Catalogue des structures ENTRE NIVEAUX : chaque famille a un theta fixe.

Pour chaque nom de levels.py, la matrice C rendue par level_corr est comparee
a une formule ecrite ICI, independamment : ACF d'un AR par l'equation de
Lyapunov, ACF d'un MA par ses coefficients psi, metriques par leur definition,
Matern par scipy.special.kv. Puis : diagonale unite, semi-definie positive,
level_chol reconstruit C, n_level_params et level_params_report coherents.
"""
import numpy as np
import pytest
from scipy.special import kv, gamma as gamma_fn

import outils_dispositifs as od
import jax.numpy as jnp
from remlax import levels as Lv
from remlax.bessel import matern

RNG = np.random.default_rng(7)
Q = 6
X1 = np.array([0.0, 1.0, 2.5, 4.0, 4.5, 7.0])            # positions 1D irregulieres
X2 = np.column_stack([X1, np.array([0.0, 2.0, 1.0, 3.0, 0.5, 2.5])])


def _d(coord):
    c = np.atleast_2d(coord.T).T if coord.ndim == 1 else coord
    dx = np.abs(c[:, None, 0] - c[None, :, 0])
    dy = np.abs(c[:, None, 1] - c[None, :, 1]) if c.shape[1] > 1 else np.zeros_like(dx)
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def _matern_ref(z, nu):
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    m = z > 0
    out[m] = 2 ** (1 - nu) / gamma_fn(nu) * z[m] ** nu * kv(nu, z[m])
    return out


# theta fixes, differents de zero et de signe mixte
TH = {
    "id": [], "cor": [0.4], "ar1": [0.6], "ar2": [0.5, -0.3], "ar3": [0.4, 0.2, -0.3],
    "sar": [0.7], "ma1": [0.5], "ma2": [0.4, -0.3], "arma": [0.5, 0.3],
    "corb": [0.3, 0.1],   # une bande n'est PD que sur une partie du carre "corg": list(RNG.normal(0, 0.7, Q * (Q - 1) // 2)),
    "exp": [0.8], "gau": [-0.6], "lvr": [np.log(3.0)],
    "iexp": [0.7], "igau": [0.5], "ieuc": [-0.7], "sph": [np.log(4.0)], "cir": [np.log(5.0)],
    "aexp": [0.7, -0.4], "agau": [0.6, 0.3],
    "mtrn": [np.log(2.0), np.log(1.5)], "own": [-0.3], "ar1ar1": [0.5, -0.3],
}


def _reference(kind, th, q=Q):
    th = np.asarray(th, dtype=float)
    tanh = np.tanh
    if kind == "id":
        return np.eye(q)
    if kind == "cor":
        lo = -1.0 / (q - 1)
        t = lo + (tanh(th[0]) + 1) / 2 * (1 - lo)
        return (1 - t) * np.eye(q) + t * np.ones((q, q))
    if kind == "ar1":
        return od.ar1_corr_np(tanh(th[0]), q)
    if kind in ("ar2", "ar3"):
        phi = od.levinson_np(tanh(th))
        return od.toeplitz_np(od.acf_ar_lyapunov(phi, q))
    if kind == "sar":
        ph = tanh(th[0])
        return od.toeplitz_np(od.acf_ar_lyapunov([ph, -ph ** 2 / 4], q))
    if kind == "ma1":
        return od.toeplitz_np(od.acf_ma_psi([1.0, -tanh(th[0])], q))
    if kind == "ma2":
        return od.toeplitz_np(od.acf_ma_psi([1.0, -tanh(th[0]), -tanh(th[1])], q))
    if kind == "arma":
        # x_t = ph x_{t-1} + e_t + th e_{t-1} : psi_0 = 1, psi_j = (ph + th) ph^(j-1)
        t_, ph = tanh(th[0]), tanh(th[1])
        psi = np.concatenate([[1.0], (ph + t_) * ph ** np.arange(0, 400)])
        return od.toeplitz_np(od.acf_ma_psi(psi, q))
    if kind == "corb":
        acf = np.zeros(q); acf[0] = 1; acf[1:1 + len(th)] = tanh(th)
        return od.toeplitz_np(acf)
    if kind == "corg":
        L = np.eye(q); k = 0
        for i in range(q):
            for j in range(i):
                L[i, j] = th[k]; k += 1
        L = L / np.sqrt((L ** 2).sum(axis=1))[:, None]
        return L @ L.T
    dx, dy, de = _d(X1 if kind in ("exp", "gau", "lvr") else X2)
    ph = abs(tanh(th[0]))
    if kind == "exp":
        return ph ** de
    if kind == "gau":
        return ph ** (de ** 2)
    if kind == "lvr":
        return np.maximum(0.0, 1.0 - dx / np.exp(th[0]))
    if kind == "iexp":
        return ph ** (dx + dy)
    if kind == "igau":
        return ph ** (dx ** 2 + dy ** 2)
    if kind == "ieuc":
        return ph ** de
    if kind == "aexp":
        return ph ** dx * abs(tanh(th[1])) ** dy
    if kind == "agau":
        return ph ** (dx ** 2) * abs(tanh(th[1])) ** (dy ** 2)
    if kind in ("sph", "cir"):
        t = np.minimum(de / np.exp(th[0]), 1.0)
        if kind == "sph":
            return 1 - 1.5 * t + 0.5 * t ** 3
        return 1 - 2 / np.pi * (t * np.sqrt(1 - t ** 2) + np.arcsin(t))
    if kind == "mtrn":
        # phi et nu estimes, delta = 2, alpha = 0.6, lambda = 2 (Haskard)
        phi, nu = np.exp(th[0]), np.exp(th[1])
        sx = X2[:, None, 0] - X2[None, :, 0]; sy = X2[:, None, 1] - X2[None, :, 1]
        ca, sa = np.cos(0.6), np.sin(0.6)
        u = sx * ca + sy * sa; v = -sx * sa + sy * ca
        h = np.sqrt((np.sqrt(2.0) * u) ** 2 + (v / np.sqrt(2.0)) ** 2)
        return _matern_ref(h / phi, nu)
    if kind == "own":
        # expr "exp(-lag*exp(p1))" : un AR1 de correlation exp(-exp(p1))
        return od.ar1_corr_np(np.exp(-np.exp(th[0])), q)
    if kind == "ar1ar1":
        return np.kron(od.ar1_corr_np(tanh(th[0]), 2), od.ar1_corr_np(tanh(th[1]), 3))
    raise KeyError(kind)


def _appel(kind, th):
    """Arguments de level_corr / level_chol pour chaque famille."""
    kw = {}
    if kind in ("exp", "gau", "lvr"):
        kw["coord"] = X1
    elif kind in Lv.LEVEL_NEEDS_COORD:
        kw["coord"] = X2
    if kind == "corb":
        kw["order"] = len(th)
    if kind == "corg":
        kw["order"] = Q
    if kind == "mtrn":
        kw["opts"] = {"est_phi": 1.0, "est_nu": 1.0, "delta": 2.0, "alpha": 0.6, "lambda": 2.0}
    if kind == "own":
        kw["expr"] = "exp(-lag*exp(p1))"
        kw["opts"] = {"n_par": 1.0}
    if kind == "ar1ar1":
        kw["dims"] = (2, 3)
    return kw


@pytest.mark.parametrize("kind", sorted(TH))
def test_level_corr_egale_formule_independante(kind):
    th = jnp.asarray(np.asarray(TH[kind], dtype=float))
    C = np.asarray(Lv.level_corr(th, kind, Q, **_appel(kind, TH[kind])))
    ref = _reference(kind, TH[kind])
    assert C.shape == (Q, Q)
    assert np.allclose(C, ref, atol=1e-9), np.abs(C - ref).max()
    assert np.allclose(np.diag(C), 1.0, atol=1e-9)
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > -1e-9


@pytest.mark.parametrize("kind", sorted(TH))
def test_level_chol_reconstruit_C(kind):
    th = jnp.asarray(np.asarray(TH[kind], dtype=float))
    kw = _appel(kind, TH[kind])
    L = Lv.level_chol(th, kind, Q, **kw)
    if kind == "id":
        assert L is None
        return
    C = np.asarray(Lv.level_corr(th, kind, Q, **kw))
    L = np.asarray(L)
    assert np.allclose(L @ L.T, C, atol=1e-7)


def test_n_level_params_par_famille():
    attendu = {k: len(v) for k, v in TH.items()}
    for kind, n in attendu.items():
        kw = _appel(kind, TH[kind])
        got = Lv.n_level_params(kind, order=kw.get("order", 0), opts=kw.get("opts"))
        assert got == n, (kind, got, n)
    assert Lv.n_level_params("fixed") == 0
    assert Lv.n_level_params("mtrn") == 1                      # seule phi par defaut
    assert Lv.n_level_params("mtrn", opts={"est_phi": 0.0}) == 0
    assert Lv.n_level_params("mtrn", opts={"est_nu": 1.0, "est_alpha": 1.0}) == 3
    assert Lv.n_level_params("own", opts={"n_par": 3}) == 3
    assert Lv.n_level_params("corg", order=5) == 10
    assert Lv.n_level_params("sep", parts=[("id", 3), ("ar1", 4), ("ar1", 5)]) == 2
    with pytest.raises(ValueError):
        Lv.n_level_params("inconnue")


def test_level_params_report_rend_la_valeur_utilisee():
    r = Lv.level_params_report(np.array([0.4]), "cor", q=Q)
    lo = -1.0 / (Q - 1)
    assert np.isclose(r["phi"][0], lo + (np.tanh(0.4) + 1) / 2 * (1 - lo))
    assert np.isclose(r["borne_inf"][0], lo)
    with pytest.raises(ValueError):
        Lv.level_params_report(np.array([0.4]), "cor")            # q obligatoire
    for kind in ("exp", "gau", "iexp", "igau", "ieuc"):
        r = Lv.level_params_report(np.array([-0.7]), kind)
        assert r["phi"][0] == pytest.approx(abs(np.tanh(-0.7)))
        assert r["signe_non_identifie"] == [1.0]
    r = Lv.level_params_report(np.array([np.log(4.0)]), "sph")
    assert r["portee"][0] == pytest.approx(4.0)
    r = Lv.level_params_report(np.array([0.5, -0.3]), "ar2")
    assert np.allclose(r["phi"], od.levinson_np(np.tanh([0.5, -0.3])))
    assert np.allclose(r["pacf"], np.tanh([0.5, -0.3]))
    r = Lv.level_params_report(np.array([np.log(2.0), np.log(1.5)]), "mtrn",
                               opts={"est_phi": 1.0, "est_nu": 1.0})
    assert r["phi"][0] == pytest.approx(2.0) and r["nu"][0] == pytest.approx(1.5)
    r = Lv.level_params_report(np.array([0.5, -0.3]), "ar1ar1")
    assert np.allclose(r["phi"], np.tanh([0.5, -0.3]))
    r = Lv.level_params_report(np.array([0.5, -0.3]), "sep",
                               parts=[("id", 3), ("ar1", 4), ("ar1", 5)])
    assert set(r) == {"ar1_2_phi", "ar1_3_phi"}
    assert r["ar1_2_phi"][0] == pytest.approx(np.tanh(0.5))
    assert Lv.level_params_report(np.zeros(3), "corg", order=3) == {}
    assert Lv.level_params_report(np.array([0.2, 0.3]), "own")["own"] == [0.2, 0.3]


def test_ar1_chol_forme_close():
    L = np.asarray(Lv.ar1_chol(0.7, 5))
    assert np.allclose(L @ L.T, od.ar1_corr_np(0.7, 5), atol=1e-14)
    assert np.allclose(L[:, 0], 0.7 ** np.arange(5))


def test_ar2_ar3_stationnaires_pour_toute_pacf():
    # la region de stationnarite est atteinte EXACTEMENT par tanh(pacf)
    for _ in range(20):
        th = jnp.asarray(RNG.normal(0, 2.0, 3))
        for kind, p in (("ar2", 2), ("ar3", 3)):
            C = np.asarray(Lv.level_corr(th[:p], kind, 12))
            assert np.linalg.eigvalsh(C).min() > 0


def test_sep_level_corr_egale_kron():
    th = jnp.array([0.5, -0.3])
    parts = [("id", 2), ("ar1", 3), ("ar1", 4)]
    C = np.asarray(Lv.level_corr(th, "sep", 24, parts=parts))
    ref = np.kron(np.eye(2), np.kron(od.ar1_corr_np(np.tanh(0.5), 3),
                                     od.ar1_corr_np(np.tanh(-0.3), 4)))
    assert np.allclose(C, ref, atol=1e-12)
    L = np.asarray(Lv.level_chol(th, "sep", 24, parts=parts))
    assert np.allclose(L @ L.T, ref, atol=1e-12)


def test_sep_refuse_les_familles_a_compte_variable():
    for fam in ("corb", "corg", "mtrn", "own", "exp", "sph"):
        with pytest.raises(ValueError):
            Lv.n_level_params("sep", parts=[(fam, 4), ("ar1", 5)])


def test_own_normalise_et_sans_normalisation():
    th = jnp.array([0.2])
    C1 = np.asarray(Lv.level_corr(th, "own", 4, expr="exp(-lag*p1) + 0.5*I"))
    assert np.allclose(np.diag(C1), 1.0)
    C0 = np.asarray(Lv.level_corr(th, "own", 4, expr="exp(-lag*p1) + 0.5*I",
                                  opts={"normalise": 0.0}))
    assert np.allclose(np.diag(C0), 1.5)
    # `^` est traduit en puissance (l'expression est ecrite en R)
    C2 = np.asarray(Lv.level_corr(th, "own", 4, expr="tanh(p1)^lag"))
    assert np.allclose(C2, od.ar1_corr_np(np.tanh(0.2), 4))
    with pytest.raises(ValueError):
        Lv.level_corr(th, "own", 4, expr=None)


def test_ar1ar1_exige_dims_et_fixed_rend_la_matrice():
    with pytest.raises(ValueError):
        Lv.level_corr(jnp.zeros(2), "ar1ar1", 6)
    Cf = np.asarray(od.ar1_corr_np(0.3, 4))
    assert np.allclose(np.asarray(Lv.level_corr(jnp.zeros(0), "fixed", 4, C_fixed=Cf)), Cf)
    Lf = np.linalg.cholesky(Cf)
    assert np.allclose(np.asarray(Lv.level_chol(jnp.zeros(0), "fixed", 4, LK_fixed=Lf)), Lf)


def test_matern_contre_scipy_et_cas_fermes():
    z = np.array([0.0, 1e-6, 0.3, 1.0, 2.5, 6.0])
    for nu in (0.5, 0.8, 1.5, 2.5, 4.0):
        m = np.asarray(matern(jnp.asarray(z), nu))
        assert np.allclose(m, _matern_ref(z, nu), rtol=1e-9, atol=1e-12), nu
        assert m[0] == pytest.approx(1.0)
    assert np.allclose(np.asarray(matern(jnp.asarray(z), 0.5)), np.exp(-z), rtol=1e-9)
    assert np.allclose(np.asarray(matern(jnp.asarray(z), 1.5)), (1 + z) * np.exp(-z), rtol=1e-9)


def test_mtrn_lambda_1_city_block_et_tout_fixe():
    # lambda = 1 : metrique city-block ; aucun parametre estime -> theta vide
    opts = {"est_phi": 0.0, "phi": 2.0, "nu": 0.5, "lambda": 1.0}
    C = np.asarray(Lv.level_corr(jnp.zeros(0), "mtrn", Q, coord=X2, opts=opts))
    dx, dy, _ = _d(X2)
    assert np.allclose(C, np.exp(-(dx + dy) / 2.0), atol=1e-9)
    assert Lv.n_level_params("mtrn", opts=opts) == 0
