"""vpredict, wald, kenward_roger et predict contre des formules exactes.

Plan en blocs complets equilibre (t traitements x b blocs, bloc aleatoire) :
la GLS coincide avec les moindres carres ordinaires, le REML rend exactement
les estimateurs de l'analyse de variance, et le test F du traitement a
(t-1)(b-1) degres de liberte au denominateur. C'est le cas ou Wald et
Kenward-Roger ont une reponse en forme close, ecrite ici sans le solveur.

vpredict est verifie par differences finies sur la fonction des composantes ;
predict par la combinaison lineaire calculee a la main, y compris la part
aleatoire et sa covariance d'erreur de prediction.
"""
import numpy as np
import pytest
from scipy.stats import chi2, f as f_dist

import outils_dispositifs as od
import jax.numpy as jnp
from remlax.model import dense_Z, assemble_V
from remlax.fit import fit_reml
from remlax.inference import (component_names, components_from_theta, vpredict, wald,
                              kenward_roger, predict)

T, B = 4, 6


def rcbd(seed=1, sb=0.7):
    rng = np.random.default_rng(seed)
    n = T * B
    trt = np.repeat(np.arange(T), B); blk = np.tile(np.arange(B), T)
    y = 5 + 0.5 * trt + rng.normal(0, sb, B)[blk] + rng.normal(0, 1, n)
    X = np.zeros((n, T)); X[:, 0] = 1
    for i in range(1, T):
        X[trt == i, i] = 1
    term = od.terme_facteur("bloc", blk, B)
    res = od.residuelle_iid(n)
    return [term], res, y, X, trt, blk


def anova(y, trt, blk):
    ybar = y.mean()
    mt = np.array([y[trt == i].mean() for i in range(T)])
    mb = np.array([y[blk == j].mean() for j in range(B)])
    sst = B * ((mt - ybar) ** 2).sum(); ssb = T * ((mb - ybar) ** 2).sum()
    sse = ((y - ybar) ** 2).sum() - sst - ssb
    mse = sse / ((T - 1) * (B - 1))
    return dict(F=(sst / (T - 1)) / mse, mse=mse, sb2=(ssb / (B - 1) - mse) / T)


@pytest.fixture(scope="module")
def ajuste():
    terms, res, y, X, trt, blk = rcbd()
    f = fit_reml(terms, res, y, X, verbose=False)
    a = anova(y, trt, blk)
    assert a["sb2"] > 0                                   # sinon REML != ANOVA
    return terms, res, y, X, trt, blk, f, a


def test_reml_equilibre_retrouve_l_anova(ajuste):
    terms, res, y, X, trt, blk, f, a = ajuste
    assert f["sigma_res"][0, 0] == pytest.approx(a["mse"], rel=1e-7)
    assert f["sigmas"]["bloc"][0, 0] == pytest.approx(a["sb2"], rel=1e-6)


def test_wald_egale_le_F_de_l_anova(ajuste):
    terms, res, y, X, trt, blk, f, a = ajuste
    n = len(y)
    V = np.asarray(assemble_V(jnp.asarray(f["theta"]), terms, [dense_Z(terms[0], n)], res, n))
    w = wald(y, X, V, termes_fixes=[("int", [0]), ("trt", [1, 2, 3])])
    t_trt = w["tests"][1]
    assert t_trt["ddl"] == 3
    assert t_trt["F"] == pytest.approx(a["F"], rel=1e-8)
    assert t_trt["chi2"] == pytest.approx(3 * a["F"], rel=1e-8)
    assert t_trt["p"] == pytest.approx(chi2.sf(3 * a["F"], 3), rel=1e-8)
    assert np.allclose(w["beta"], f["beta"], atol=1e-8)
    # erreur-type d'un contraste de traitement : sqrt(2 MSE / b)
    assert w["se_beta"][1] == pytest.approx(np.sqrt(2 * a["mse"] / B), rel=1e-7)
    # sans groupes : une ligne par colonne
    w1 = wald(y, X, V)
    assert [t["terme"] for t in w1["tests"]] == ["colonne %d" % j for j in range(1, 5)]
    assert all(t["ddl"] == 1 for t in w1["tests"])


def test_kenward_roger_ddl_exacts_du_plan_equilibre(ajuste):
    terms, res, y, X, trt, blk, f, a = ajuste
    kr = kenward_roger(f["theta"], terms, res, y, X, termes_fixes=[("int", [0]), ("trt", [1, 2, 3])])
    assert kr["disponible"] is True
    t_trt = kr["tests"][1]
    assert t_trt["denDF"] == pytest.approx((T - 1) * (B - 1), abs=1e-9)
    assert t_trt["F"] == pytest.approx(a["F"], rel=1e-8)
    assert t_trt["echelle"] == pytest.approx(1.0, abs=1e-12)
    assert t_trt["p"] == pytest.approx(f_dist.sf(a["F"], T - 1, (T - 1) * (B - 1)), rel=1e-8)
    assert kr["se_beta"][1] == pytest.approx(np.sqrt(2 * a["mse"] / B), rel=1e-7)
    assert kr["se_beta"][1] == pytest.approx(kr["se_beta_brut"][1], rel=1e-9)
    assert kr["second_ordre_omis"] is False
    assert kr["vbeta_kr"].shape == (T, T)


def test_kenward_roger_sans_terme_aleatoire_rend_les_ddl_residuels():
    terms, res, y, X, trt, blk = rcbd(seed=2)
    f = fit_reml([], res, y, X, verbose=False)
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = ((y - X @ beta) ** 2).sum() / (n - T)
    assert f["sigma_res"][0, 0] == pytest.approx(s2, rel=1e-8)
    kr = kenward_roger(f["theta"], [], res, y, X, termes_fixes=[("trt", [1, 2, 3])])
    t_trt = kr["tests"][0]
    assert t_trt["denDF"] == pytest.approx(n - T, abs=1e-9)
    # F classique du traitement : (SS_trt / 3) / s2
    ybar = y.mean(); mt = np.array([y[trt == i].mean() for i in range(T)])
    F_cl = (B * ((mt - ybar) ** 2).sum() / 3) / s2
    assert t_trt["F"] == pytest.approx(F_cl, rel=1e-8)


def test_kenward_roger_second_ordre_signale_avec_une_correlation():
    n, q = 60, 10
    rng = np.random.default_rng(4)
    g = rng.integers(0, q, n)
    terms = [od.terme_facteur("g", g, q, lvl="ar1")]
    res = od.residuelle_iid(n)
    y = rng.normal(size=n); X = np.ones((n, 1))
    th = np.array([0.0, 0.3, 0.0])
    kr = kenward_roger(th, terms, res, y, X)
    assert kr["second_ordre_omis"] is True
    assert kr["disponible"] in (True, False)


def test_component_names_et_components_from_theta():
    q, t, nu = 5, 2, 10
    unit = np.repeat(np.arange(nu), t); trait = np.tile(np.arange(t), nu)
    g = np.arange(nu)[unit] % q
    tm = od.terme_long("g", g, trait, q, t, struct="us")
    tm2 = od.terme_facteur("b", unit % 3, 3, lvl="ar2")
    res = od.residuelle("diag", trait, unit, t)
    noms = component_names([tm, tm2], res)
    assert noms == ["g[1,1]", "g[2,1]", "g[2,2]", "b", "b!ar21", "b!ar22",
                    "residuelle[1,1]", "residuelle[2,1]", "residuelle[2,2]"]
    th = np.array([np.log(1.2), 0.5, np.log(0.9), 0.1, 0.4, -0.2, np.log(0.7), np.log(1.1)])
    c = np.asarray(components_from_theta(jnp.asarray(th), [tm, tm2], res))
    assert np.allclose(c[:3], [1.44, 0.6, 1.06])
    assert c[3] == pytest.approx(np.exp(0.2))
    assert np.allclose(c[4:6], np.tanh([0.4, -0.2]))          # echelle naturelle
    assert np.allclose(c[6:], [0.49, 0.0, 1.21])                # diag : covariance nulle


def test_vpredict_delta_contre_differences_finies(ajuste):
    terms, res, y, X, trt, blk, f, a = ajuste
    H = f["hessian"]; th = f["theta"]
    vp = vpredict(th, H, terms, res, [("h2", "V1/(V1+V2)"), ("sd", "sqrt(V2)"),
                                       ("lv", "log(V1)")])
    comps = {c["nom"]: c["valeur"] for c in vp["composantes"]}
    assert comps["bloc"] == pytest.approx(np.exp(2 * th[0]))
    assert comps["residuelle"] == pytest.approx(np.exp(2 * th[1]))
    pred = {p["nom"]: p for p in vp["predictions"]}
    v1, v2 = comps["bloc"], comps["residuelle"]
    assert pred["h2"]["valeur"] == pytest.approx(v1 / (v1 + v2))
    assert pred["sd"]["valeur"] == pytest.approx(np.sqrt(v2))
    assert pred["lv"]["valeur"] == pytest.approx(np.log(v1))
    # delta method a la main : J par differences finies, Vth = 2 H^-1
    Vth = 2.0 * np.linalg.inv(H)
    fns = {"h2": lambda t_: np.exp(2 * t_[0]) / (np.exp(2 * t_[0]) + np.exp(2 * t_[1])),
           "sd": lambda t_: np.exp(t_[1]), "lv": lambda t_: 2 * t_[0]}
    eps = 1e-6
    for nom, fn in fns.items():
        J = np.array([(fn(th + eps * e) - fn(th - eps * e)) / (2 * eps) for e in np.eye(2)])
        assert pred[nom]["se"] == pytest.approx(np.sqrt(J @ Vth @ J), rel=1e-5)
    # sans Hessien : NaN, jamais un chiffre invente ; parametre fixe exclu
    vp0 = vpredict(th, None, terms, res, [("h2", "V1/(V1+V2)")])
    assert np.isnan(vp0["predictions"][0]["se"])
    vpf = vpredict(th, H, terms, res, [("lv", "log(V1)")], free=np.array([False, True]))
    assert vpf["predictions"][0]["se"] == 0.0                # V1 fixe : variance nulle
    vpf2 = vpredict(th, H, terms, res, [("sd", "sqrt(V2)")], free=np.array([False, True]))
    assert vpf2["predictions"][0]["se"] == pytest.approx(np.sqrt(2.0 / H[1, 1]) * np.exp(th[1]), rel=1e-6)


def test_predict_contre_combinaison_lineaire_a_la_main(ajuste):
    terms, res, y, X, trt, blk, f, a = ajuste
    n = len(y); th = f["theta"]
    Z = od.dense_Z_np(terms[0], n)
    V = np.exp(2 * th[0]) * Z @ Z.T + np.exp(2 * th[1]) * np.eye(n)
    Vi = np.linalg.inv(V)
    Phi = np.linalg.inv(X.T @ Vi @ X)
    beta = Phi @ X.T @ Vi @ y
    P = Vi - Vi @ X @ Phi @ X.T @ Vi
    # moyennes par traitement : L = [1, e_i]
    L = np.zeros((T, T)); L[:, 0] = 1
    for i in range(1, T):
        L[i, i] = 1
    pr = predict(th, terms, res, y, X, L, vbeta=f["vbeta"])
    assert np.allclose(pr["valeur"], L @ beta, atol=1e-8)
    assert np.allclose(pr["se"], np.sqrt(np.diag(L @ Phi @ L.T)), atol=1e-8)
    assert np.allclose(pr["beta"], beta, atol=1e-8)
    # moyenne du traitement 1 DANS chaque bloc : part aleatoire M = indicatrice du bloc
    Lb = np.tile(L[0], (B, 1))
    M = {"bloc": np.eye(B)}
    G = np.exp(2 * th[0]) * np.eye(B)
    u = G @ Z.T @ P @ y
    prb = predict(th, terms, res, y, X, Lb, M=M, vbeta=f["vbeta"])
    assert np.allclose(prb["valeur"], Lb @ beta + u, atol=1e-8)
    C_fix = Lb @ Phi @ Lb.T
    A = Lb @ Phi @ X.T @ Vi @ Z @ G
    cov_ref = C_fix - A - A.T + G - G @ Z.T @ P @ Z @ G
    assert np.allclose(prb["cov"], cov_ref, atol=1e-8)
    assert np.allclose(prb["se"], np.sqrt(np.diag(cov_ref)), atol=1e-8)
    # le BLUP relu est celui de fit_reml
    assert np.allclose(u, f["blups"]["bloc"].ravel(), atol=1e-8)
    with pytest.raises(ValueError, match="colonnes"):
        predict(th, terms, res, y, X, np.ones((1, T + 1)))
    with pytest.raises(ValueError, match="attendu"):
        predict(th, terms, res, y, X, Lb, M={"bloc": np.eye(B + 1)})
