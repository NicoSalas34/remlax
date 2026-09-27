"""Chaque option de fit_reml, une par une, sur de petits dispositifs simules.

theta_init avec maxiter = 0 et polish = 0 est une EVALUATION exacte ; fixed_idx
tient la valeur ; floor et ceil sont respectes et rendus ; hessian et blups
peuvent etre coupes ; pev accepte True ou une liste de noms, refuse un nom
inconnu, decroit avec la replication et suit la formule G - G Z' P Z G ;
n_restarts retrouve le meme optimum et rend ses compteurs ; les cles de `rho`
suivent la convention nue / prefixee ; la constante asreml est celle annoncee.
"""
import numpy as np
import pytest

import outils_dispositifs as od
import jax.numpy as jnp
from remlax import structures as S
from remlax.model import dense_Z, neg2_reml, n_theta
from remlax.fit import fit_reml

RNG = np.random.default_rng(2026)



def _sur_gpu():
    """Un peripherique cuda est-il le peripherique par defaut ?"""
    import jax
    return any(d.platform == "gpu" for d in jax.devices())


def _egal_selon_peripherique(a, b):
    """CPU : egalite bit a bit, mesuree sur le cluster (quatre executions de la
    meme cellule a logLik EXACTEMENT egale). GPU : les reductions XLA ne sont
    pas deterministes d'une execution a l'autre ; mesure sur RTX A1000 avec
    jax 0.11.1 : 1,2e-14 sur theta et 3,4e-13 sur -2logL entre deux appels
    identiques. On exige donc 1e-12 relatif sur GPU, l'egalite sur CPU."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if _sur_gpu():
        return bool(np.allclose(a, b, rtol=1e-12, atol=1e-12))
    return bool(np.array_equal(a, b))

def plan(q=20, rep=4, sg=1.0, sb=0.4, se=1.0, seed=0):
    """Plan en blocs complets : q genotypes x rep blocs."""
    rng = np.random.default_rng(seed)
    g = np.tile(np.arange(q), rep); b = np.repeat(np.arange(rep), q)
    n = q * rep
    y = 10 + rng.normal(0, sg, q)[g] + rng.normal(0, sb, rep)[b] + rng.normal(0, se, n)
    X = np.ones((n, 1))
    terms = [od.terme_facteur("g", g, q), od.terme_facteur("bloc", b, rep)]
    return terms, od.residuelle_iid(n), y, X


def _n2(terms, res, y, X, theta):
    Zs = [dense_Z(t, len(y)) for t in terms]
    return float(neg2_reml(jnp.asarray(theta), terms, Zs, res, jnp.asarray(y), jnp.asarray(X)))


def test_theta_init_maxiter_0_polish_0_est_une_evaluation_exacte():
    terms, res, y, X = plan()
    th = np.array([0.3, -0.5, 0.1])
    f = fit_reml(terms, res, y, X, theta_init=th, maxiter=0, polish=0, verbose=False,
                 hessian=False, blups=False)
    assert np.array_equal(f["theta"], th)
    assert _egal_selon_peripherique(f["neg2_reml"], _n2(terms, res, y, X, th))
    assert f["n_iter"] == 0 and f["n_polish"] == 0
    assert "evaluation seule" in f["scipy_message"]
    assert f["logLik"] == pytest.approx(-0.5 * f["neg2_reml"])


def test_fixed_idx_tient_la_valeur_et_les_compteurs():
    terms, res, y, X = plan()
    libre = fit_reml(terms, res, y, X, verbose=False)
    th0 = libre["theta"].copy(); th0[0] += 0.4
    fixe = fit_reml(terms, res, y, X, theta_init=th0, fixed_idx=[0], verbose=False)
    assert fixe["theta"][0] == th0[0]
    assert fixe["n_fixed"] == 1 and fixe["n_fixed_out"] == 1
    assert fixe["n_par_free"] == 2 and fixe["n_at_bound"] == 0
    assert libre["n_fixed"] == 0 and libre["n_par_free"] == 3
    assert fixe["neg2_reml"] >= libre["neg2_reml"] - 1e-9
    # le decrement est calcule sur le sous-espace LIBRE : converge malgre le
    # gradient non nul le long de la direction pincee
    assert fixe["newton_decrement"] < 1e-4
    assert fixe["max_grad"] > fixe["newton_decrement"]


def test_floor_ceil_respectes_et_rendus():
    terms, res, y, X = plan()
    f = fit_reml(terms, res, y, X, verbose=False)
    assert f["par_floor"] == -12.0 and f["par_ceil"] == 12.0
    # y multiplie par 1000 : log(sd) des composantes ~ 7, un plafond a 1 mord
    fc = fit_reml(terms, res, y * 1000.0, X, floor=-3.0, ceil=1.0, verbose=False)
    assert fc["par_floor"] == -3.0 and fc["par_ceil"] == 1.0
    assert np.all(fc["theta"] <= 1.0 + 1e-12) and np.all(fc["theta"] >= -3.0 - 1e-12)
    assert fc["n_at_bound"] >= 1
    assert fc["n_par_free"] == 3 - fc["n_at_bound"]
    with pytest.raises(ValueError):
        fit_reml(terms, res, y, X, theta_init=np.zeros(3), floor=2.0, ceil=1.0,
                 verbose=False, hessian=False, blups=False)


def test_composante_au_plancher_signalee_degeneree():
    # aucun effet de bloc simule : sa variance part au plancher
    terms, res, y, X = plan(q=30, rep=3, sb=0.0, seed=3)
    f = fit_reml(terms, res, y, X, verbose=False)
    assert "bloc" in f["composantes_degenerees"]
    # le seuil de degenerescence est floor + 0.5 : un parametre peut etre
    # declare degenere sans etre EXACTEMENT a la borne (n_at_bound, tol 1e-7)
    assert f["theta"][1] <= -12.0 + 0.5
    assert f["n_at_bound"] in (0, 1)


def test_hessian_false_garde_le_gradient_projete():
    terms, res, y, X = plan()
    f = fit_reml(terms, res, y, X, hessian=False, verbose=False)
    assert "hessian" not in f
    assert np.isnan(f["newton_decrement"]) and f["conv_decrement"] is None
    assert np.isfinite(f["grad_proj_max"]) and np.isfinite(f["grad_rel"])
    assert f["conv_grad_rel"] in (True, False)
    assert "blups" in f and "beta" in f            # blups=True par defaut


def test_blups_false_et_hessian_false_ne_calculent_pas_beta():
    terms, res, y, X = plan()
    f = fit_reml(terms, res, y, X, hessian=False, blups=False, verbose=False)
    for k in ("blups", "beta", "vbeta", "Vi", "Py", "hessian"):
        assert k not in f
    f2 = fit_reml(terms, res, y, X, hessian=True, blups=False, verbose=False)
    assert "blups" not in f2 and "beta" in f2 and "hessian" in f2


def test_pev_liste_true_inconnu_et_formule():
    terms, res, y, X = plan()
    f_all = fit_reml(terms, res, y, X, pev=True, verbose=False)
    assert set(f_all["pev"]) == {"g", "bloc"}
    f_g = fit_reml(terms, res, y, X, pev=["g"], verbose=False)
    assert set(f_g["pev"]) == {"g"}
    assert f_g["pev"]["g"].shape == (20, 1)
    with pytest.raises(ValueError, match="inconnu"):
        fit_reml(terms, res, y, X, pev=["zzz"], verbose=False)
    # formule : diag(G - G Z' P Z G), P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1
    n = len(y)
    th = f_g["theta"]
    Z = od.dense_Z_np(terms[0], n)
    G = np.exp(2 * th[0]) * np.eye(20)
    Vi = f_g["Vi"]
    P = Vi - Vi @ X @ np.linalg.inv(X.T @ Vi @ X) @ X.T @ Vi
    pev_ref = np.diag(G - G @ Z.T @ P @ Z @ G)
    assert np.allclose(f_g["pev"]["g"].ravel(), pev_ref, atol=1e-8)
    # le BLUP suit u = G Z' P y
    assert np.allclose(f_g["blups"]["g"].ravel(), G @ Z.T @ P @ y, atol=1e-8)


def test_pev_decroit_avec_la_replication():
    th = np.array([0.0, -0.5, 0.0])
    pevs = []
    for rep in (2, 6):
        terms, res, y, X = plan(q=20, rep=rep, seed=1)
        f = fit_reml(terms, res, y, X, theta_init=th, maxiter=0, polish=0, pev=["g"],
                     hessian=False, verbose=False)
        pevs.append(float(np.mean(f["pev"]["g"])))
    assert pevs[1] < pevs[0]
    assert 0 < pevs[1] < 1.0                     # bornee par var(u) = exp(0) = 1


def test_n_restarts_retrouve_le_meme_optimum():
    terms, res, y, X = plan()
    base = fit_reml(terms, res, y, X, verbose=False)
    f = fit_reml(terms, res, y, X, n_restarts=2, restart_sd=0.7, seed=5, verbose=False)
    assert f["n_restarts"] == 2
    assert f["restart_gain"] >= 0.0
    assert f["restart_better"] in (0, 1, 2)
    assert f["neg2_reml"] <= base["neg2_reml"] + 1e-8
    assert abs(f["neg2_reml"] - base["neg2_reml"]) < 1e-6
    assert np.allclose(f["theta"], base["theta"], atol=1e-3)


def test_deux_appels_identiques_meme_resultat_et_polish_0():
    terms, res, y, X = plan()
    a = fit_reml(terms, res, y, X, verbose=False)
    b = fit_reml(terms, res, y, X, verbose=False)
    if _sur_gpu():
        # Sur GPU la TRAJECTOIRE change d'une execution a l'autre (le bruit
        # d'arrondi fait accepter un autre pas), donc l'arret tombe a un autre
        # point de la meme surface plate : mesure 2e-8 sur theta pour 3e-13 sur
        # -2logL. C'est la vraisemblance qui est reproductible, pas le chemin.
        assert np.allclose(a["neg2_reml"], b["neg2_reml"], rtol=1e-10)
        assert np.allclose(a["theta"], b["theta"], atol=1e-6)
    else:
        assert np.array_equal(a["theta"], b["theta"])
    c = fit_reml(terms, res, y, X, polish=0, verbose=False, hessian=False, blups=False)
    assert c["n_polish"] == 0
    assert a["n_polish"] >= 0 and a["neg2_reml"] <= c["neg2_reml"] + 1e-9


def test_constante_asreml_et_sigmas():
    terms, res, y, X = plan()
    f = fit_reml(terms, res, y, X, verbose=False)
    n, p = X.shape
    assert f["const_2pi"] == pytest.approx(0.5 * (n - p) * np.log(2 * np.pi))
    assert f["logLik_asreml"] - f["logLik"] == pytest.approx(f["const_2pi"])
    assert set(f["sigmas"]) == {"g", "bloc"}
    assert f["sigmas"]["g"].shape == (1, 1)
    assert f["sigmas"]["g"][0, 0] == pytest.approx(np.exp(2 * f["theta"][0]))
    assert list(f["sigmas_res"]) == ["residuelle"]
    assert f["sigma_res"][0, 0] == pytest.approx(np.exp(2 * f["theta"][2]))
    assert f["n_par"] == 3 and f["n_obs"] == n
    assert f["V_singuliere"] is False
    assert f["blups"]["g"].shape == (20, 1)
    assert f["hessian"].shape == (3, 3)
    assert f["scipy_success"] and f["newton_decrement"] < 1e-4


def test_verbose_imprime_la_compilation(capsys):
    terms, res, y, X = plan()
    fit_reml(terms, res, y, X, verbose=True, hessian=False, blups=False, maxiter=5, polish=1)
    out = capsys.readouterr().out
    assert "[compilation]" in out and "iter" in out


def test_rho_cles_nues_prefixees_et_pacf():
    n, q = 80, 10
    g = RNG.integers(0, q, n); y = RNG.normal(size=n); X = np.ones((n, 1))
    coord = np.column_stack([RNG.uniform(0, 5, q), RNG.uniform(0, 5, q)])
    res = od.residuelle_iid(n)
    th_c = np.array([0.0, 0.3, 0.0])
    kw = dict(maxiter=0, polish=0, hessian=False, blups=False, verbose=False)
    f = fit_reml([od.terme_facteur("g", g, q, lvl="ar1")], res, y, X, theta_init=th_c, **kw)
    assert set(f["rho"]) == {"g"} and f["rho"]["g"][0] == pytest.approx(np.tanh(0.3))
    f = fit_reml([od.terme_facteur("g", g, q, lvl="cor")], res, y, X, theta_init=th_c, **kw)
    assert set(f["rho"]) == {"g", "g!borne_inf"}
    f = fit_reml([od.terme_facteur("g", g, q, lvl="ar2")], res, y, X,
                 theta_init=np.array([0.0, 0.3, -0.2, 0.0]), **kw)
    assert set(f["rho"]) == {"g"} and len(f["rho"]["g"]) == 2
    assert np.allclose(f["pacf"]["g"], np.tanh([0.3, -0.2]))
    f = fit_reml([od.terme_facteur("g", g, q, lvl="ieuc", coord=coord)], res, y, X,
                 theta_init=np.array([0.0, -0.4, 0.0]), **kw)
    assert set(f["rho"]) == {"g", "g!signe_non_identifie"}
    assert f["rho"]["g"][0] == pytest.approx(abs(np.tanh(-0.4)))
    opts = {"est_phi": 1.0, "est_nu": 1.0}
    f = fit_reml([od.terme_facteur("g", g, q, lvl="mtrn", coord=coord, lvl_opts=opts)],
                 res, y, X, theta_init=np.array([0.0, np.log(2.0), np.log(1.5), 0.0]), **kw)
    assert set(f["rho"]) == {"g!phi", "g!nu"}
    assert f["rho"]["g!phi"][0] == pytest.approx(2.0)


def test_loadings_et_sections_nommees():
    q, t, nu = 12, 3, 30
    unit = np.repeat(np.arange(nu), t); trait = np.tile(np.arange(t), nu)
    g = RNG.integers(0, q, nu)[unit]; n = nu * t
    tm = od.terme_long("g", g, trait, q, t, struct="fa", rank=1)
    res = od.residuelle("diag", trait, unit, t)
    y = RNG.normal(size=n); X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    th = RNG.normal(0, 0.2, n_theta([tm], res))
    f = fit_reml([tm], res, y, X, theta_init=th, maxiter=0, polish=0, hessian=False,
                 blups=False, verbose=False)
    assert f["loadings"]["g"]["Lambda"].shape == (t, 1)
    assert f["loadings"]["g"]["psi"].shape == (t,)
    assert f["sigmas_res"]["residuelle"].shape == (t, t)
    # dsum : les sections portent leur nom
    n2 = 60
    rows = np.arange(n2)
    secs = [dict(struct="iid", t=1, rank=0, trait=np.zeros(30, np.int64), unit=np.arange(30),
                 n_unit=30, lvl="id", rows=rows[:30], name="siteA"),
            dict(struct="iid", t=1, rank=0, trait=np.zeros(30, np.int64), unit=np.arange(30),
                 n_unit=30, lvl="id", rows=rows[30:], name="siteB")]
    res2 = dict(struct="iid", t=1, rank=0, trait=np.zeros(n2, np.int64), unit=np.arange(n2),
                n_unit=n2, sections=secs)
    tm2 = od.terme_facteur("g", RNG.integers(0, 6, n2), 6)
    f2 = fit_reml([tm2], res2, RNG.normal(size=n2), np.ones((n2, 1)), verbose=False,
                  hessian=False, blups=False)
    assert list(f2["sigmas_res"]) == ["siteA", "siteB"]
    assert f2["n_par"] == 3
