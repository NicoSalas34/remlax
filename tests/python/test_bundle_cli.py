"""Aller-retour du paquet serialise, puis le CLI drapeau par drapeau.

Le paquet est ECRIT ICI, a la main, dans le format de rx_export (manifest.json
+ binaires column-major + .txt), sans passer par R : c'est ce qui rend le
contrat de format testable des deux cotes. Bundle doit le relire a l'identique,
y compris les sections dsum, les options de niveaux et les coordonnees.

Le CLI est lance en sous-processus sur ce paquet avec chaque drapeau au moins
une fois ; result.json et les out_*.bin sont relus et compares a un appel
direct de fit_reml / neg2_reml.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import outils_dispositifs as od
import jax
import jax.numpy as jnp
from remlax.bundle import Bundle
from remlax.model import dense_Z, neg2_reml, n_theta
from remlax.fit import fit_reml
from remlax.device import pick_device, device_report

RNG = np.random.default_rng(99)


# ==============================================================================
# Ecriture d'un paquet, format rx_export
# ==============================================================================
class Ecrivain:
    def __init__(self, path, forme="liste"):
        self.path, self.man, self.forme = path, [], forme
        os.makedirs(path, exist_ok=True)

    def put(self, name, x, dtype="f8"):
        if dtype == "str":
            with open(os.path.join(self.path, name + ".txt"), "w") as f:
                f.write("\n".join(str(v) for v in x) + "\n")
            self.man.append(dict(name=name, dtype="str", shape=[len(x)]))
            return
        a = np.asarray(x)
        dt = np.float64 if dtype == "f8" else np.int32
        a.astype(dt).ravel(order="F").tofile(os.path.join(self.path, name + ".bin"))
        self.man.append(dict(name=name, dtype=dtype, shape=list(a.shape)))

    def close(self):
        if self.forme == "liste":
            man = self.man
        else:
            man = {"arrays": {d["name"]: d for d in self.man}, "meta": {"origine": "test"}}
        with open(os.path.join(self.path, "manifest.json"), "w") as f:
            json.dump(man, f)


def ecrire_paquet(path, forme="liste", dsum=False):
    """Deux termes (g iid K=I ; bloc avec K fournie), X a deux colonnes groupees,
    residuelle iid ou dsum a deux sections."""
    q, rep = 12, 4
    n = q * rep
    g = np.tile(np.arange(q), rep); b = np.repeat(np.arange(rep), q)
    A = RNG.normal(size=(rep, rep)); K = A @ A.T / rep + np.eye(rep)
    LK = np.linalg.cholesky(K)
    x = RNG.normal(size=n)
    y = 3 + 0.5 * x + RNG.normal(0, 1, q)[g] + RNG.normal(0, 0.5, rep)[b] + RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    w = Ecrivain(path, forme)
    w.put("y", y); w.put("X", X)
    w.put("X_assign", [0, 1], "i4"); w.put("X_termes", ["(Intercept)", "x"], "str")
    w.put("term_names", ["g", "bloc"], "str")
    for nm, code, qq in (("g", g, q), ("bloc", b, rep)):
        w.put("term_%s_zi" % nm, np.arange(n), "i4")
        w.put("term_%s_zj" % nm, code, "i4")
        w.put("term_%s_zx" % nm, np.ones(n))
        w.put("term_%s_t" % nm, [1], "i4"); w.put("term_%s_q" % nm, [qq], "i4")
        w.put("term_%s_rank" % nm, [0], "i4"); w.put("term_%s_struct" % nm, ["iid"], "str")
        w.put("term_%s_levels" % nm, ["L%d" % i for i in range(qq)], "str")
    w.put("term_bloc_LK", LK)                      # pas de cle lvl : "fixed" implicite
    w.put("term_g_lvl", ["id"], "str")
    w.put("res_struct", ["iid"], "str"); w.put("res_lvl", ["id"], "str")
    w.put("res_t", [1], "i4"); w.put("res_rank", [0], "i4")
    w.put("res_trait", np.zeros(n), "i4"); w.put("res_unit", np.arange(n), "i4")
    if dsum:
        rows = [np.arange(0, n, 2), np.arange(1, n, 2)]
        w.put("res_nsec", [2], "i4"); w.put("res_secnames", ["pair", "impair"], "str")
        for k, r in enumerate(rows):
            pre = "res_s%d" % k
            w.put(pre + "_struct", ["iid"], "str"); w.put(pre + "_lvl", ["id"], "str")
            w.put(pre + "_t", [1], "i4"); w.put(pre + "_rank", [0], "i4")
            w.put(pre + "_trait", np.zeros(len(r)), "i4")
            w.put(pre + "_unit", np.arange(len(r)), "i4")
            w.put(pre + "_rows", r, "i4")
    else:
        w.put("res_nsec", [0], "i4")
    w.close()
    return dict(y=y, X=X, g=g, b=b, q=q, rep=rep, K=K, LK=LK)


def test_bundle_relit_le_paquet_a_l_identique(tmp_path):
    for forme in ("liste", "dict"):
        d = ecrire_paquet(str(tmp_path / forme), forme)
        bd = Bundle(str(tmp_path / forme))
        assert bd.n == len(d["y"])
        assert np.array_equal(bd.y, d["y"]) and np.array_equal(bd.X, d["X"])
        assert bd.has("term_g_zi") and not bd.has("absent")
        assert bd.fixed_groups() == [("(Intercept)", [0]), ("x", [1])]
        tm = bd.terms()
        assert [t["name"] for t in tm] == ["g", "bloc"]
        assert tm[0]["lvl"] == "id" and tm[0]["LK"] is None
        assert tm[1]["lvl"] == "fixed" and np.allclose(tm[1]["LK"], d["LK"])
        assert np.array_equal(tm[0]["zj"], d["g"]) and tm[0]["q"] == d["q"]
        assert tm[0]["zi"].dtype == np.int64 and tm[0]["zx"].dtype == np.float64
        r = bd.residual()
        assert r["struct"] == "iid" and r["n_unit"] == bd.n and "sections" not in r
        assert bd.meta == ({} if forme == "liste" else {"origine": "test"})
    with pytest.raises(FileNotFoundError):
        Bundle(str(tmp_path / "vide"))


def test_bundle_options_de_niveaux_et_sections(tmp_path):
    p = str(tmp_path / "opts")
    d = ecrire_paquet(p, dsum=True)
    w = Ecrivain(p); w.man = json.load(open(os.path.join(p, "manifest.json")))
    coord = RNG.uniform(0, 5, (d["q"], 2))
    w.put("term_g_lvl", ["mtrn"], "str"); w.put("term_g_coord", coord)
    w.put("term_g_lvloptk", ["est_phi", "est_nu", "delta"], "str")
    w.put("term_g_lvloptv", [1.0, 1.0, 2.0])
    w.put("term_bloc_lvlorder", [2], "i4"); w.put("term_bloc_dims", [2, 2], "i4")
    w.put("term_bloc_lvlpartk", ["id", "ar1"], "str"); w.put("term_bloc_lvlpartq", [2, 2], "i4")
    w.put("term_bloc_lvlexpr", ["exp(-lag*p1)"], "str")
    w.close()
    bd = Bundle(p)
    tm = {t["name"]: t for t in bd.terms()}
    assert tm["g"]["lvl"] == "mtrn" and np.allclose(tm["g"]["coord"], coord)
    assert tm["g"]["lvl_opts"] == {"est_phi": 1.0, "est_nu": 1.0, "delta": 2.0}
    assert tm["bloc"]["lvl_order"] == 2 and tm["bloc"]["dims"] == (2, 2)
    assert tm["bloc"]["lvl_parts"] == [("id", 2), ("ar1", 2)]
    assert tm["bloc"]["lvl_expr"] == "exp(-lag*p1)"
    r = bd.residual()
    assert len(r["sections"]) == 2
    assert [s["name"] for s in r["sections"]] == ["pair", "impair"]
    assert np.array_equal(r["sections"][0]["rows"], np.arange(0, bd.n, 2))
    # partition rompue : une ligne oubliee
    w = Ecrivain(p); w.man = json.load(open(os.path.join(p, "manifest.json")))
    w.put("res_s1_rows", np.arange(1, bd.n - 2, 2), "i4")
    w.put("res_s1_trait", np.zeros(len(np.arange(1, bd.n - 2, 2))), "i4")
    w.put("res_s1_unit", np.arange(len(np.arange(1, bd.n - 2, 2))), "i4")
    w.close()
    with pytest.raises(ValueError, match="partitionnent"):
        Bundle(p).residual()


# ==============================================================================
# CLI
# ==============================================================================
def cli(paquet, *args, module=True):
    env = dict(os.environ, PYTHONPATH=od._SRC)
    cmd = ([od.PY, "-m", "remlax.cli"] if module
           else [od.PY, os.path.join(od._SRC, "remlax", "cli.py")])
    return subprocess.run(cmd + [paquet] + list(args), capture_output=True, text=True,
                          env=env, cwd=od.RACINE)


def _lire(paquet, nom, shape=None):
    a = np.fromfile(os.path.join(paquet, nom + ".bin"), dtype=np.float64)
    return a if shape is None else a.reshape(shape, order="F")


def _json(paquet):
    with open(os.path.join(paquet, "result.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def paquet(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("cli") / "paquet")
    d = ecrire_paquet(p)
    bd = Bundle(p)
    d.update(path=p, terms=bd.terms(), res=bd.residual(), y=bd.y, X=bd.X)
    return d


def test_cli_ajustement_complet_et_inference(paquet):
    p = paquet["path"]
    r = cli(p, "--backend", "cpu", "--maxiter", "500", "--polish", "5",
            "--floor", "-10", "--ceil", "10", "--pev",
            "--vpredict", "h2=V1/(V1+V2+V3);vg=V1", "--wald", "--kenward-roger")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[remlax] backend retenu : cpu" in r.stdout and "logLik" in r.stdout
    out = _json(p)
    assert out["backend"] == "cpu"
    assert out["par_floor"] == -10.0 and out["par_ceil"] == 10.0
    assert out["n_par"] == 3 and out["n_obs"] == len(paquet["y"])
    assert out["composantes_noms"] == ["g", "bloc", "residuelle"]
    assert [p_["nom"] for p_ in out["vpredict"]["predictions"]] == ["h2", "vg"]
    assert out["vpredict"]["predictions"][0]["se"] > 0
    assert [t["terme"] for t in out["wald"]["tests"]] == ["(Intercept)", "x"]
    assert out["kenward_roger"]["disponible"] is True
    assert set(out["pev_dims"]) == {"g", "bloc"}
    assert set(out["blup_dims"]) == {"g", "bloc"} and out["sigma_dims"]["g"] == [1, 1]
    for f_ in ("out_theta", "out_beta", "out_vbeta", "out_vbeta_kr", "out_hessian",
               "out_sigma_g", "out_sigma_bloc", "out_sigma_res", "out_sigmares_residuelle",
               "out_blup_g", "out_pev_g"):
        assert os.path.exists(os.path.join(p, f_ + ".bin")), f_
    # accord avec un appel direct
    f = fit_reml(paquet["terms"], paquet["res"], paquet["y"], paquet["X"], verbose=False,
                 floor=-10, ceil=10, maxiter=500, polish=5)
    th = _lire(p, "out_theta")
    assert np.allclose(th, f["theta"], atol=1e-6)
    assert out["logLik"] == pytest.approx(f["logLik"], abs=1e-6)
    assert np.allclose(_lire(p, "out_beta"), f["beta"], atol=1e-6)
    assert np.allclose(_lire(p, "out_blup_g", (paquet["q"], 1)), f["blups"]["g"], atol=1e-5)
    assert _lire(p, "out_vbeta_kr", (2, 2)).shape == (2, 2)


def test_cli_theta_in_evaluation_seule_et_sans_hessien(paquet):
    p = paquet["path"]
    th = np.array([0.2, -0.4, 0.1])
    th.tofile(os.path.join(p, "in_theta.bin"))
    r = cli(p, "--backend", "cpu", "--theta-in", "--maxiter", "0", "--polish", "0",
            "--no-hessian", "--no-blups", "--quiet", module=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""                               # --quiet
    out = _json(p)
    assert np.array_equal(_lire(p, "out_theta"), th)
    n = len(paquet["y"])
    Zs = [dense_Z(t, n) for t in paquet["terms"]]
    v = float(neg2_reml(jnp.asarray(th), paquet["terms"], Zs, paquet["res"],
                        jnp.asarray(paquet["y"]), jnp.asarray(paquet["X"])))
    assert out["neg2_reml"] == pytest.approx(v, rel=1e-12)
    assert out["n_iter"] == 0 and "evaluation seule" in out["scipy_message"]
    assert out["newton_decrement"] is None                     # NaN -> null
    assert out["blup_dims"] == {} and "hessian" not in out
    assert not os.path.exists(os.path.join(p, "out_beta.bin")) or \
        os.path.getmtime(os.path.join(p, "out_beta.bin")) < os.path.getmtime(os.path.join(p, "result.json"))


def test_cli_restarts_fixed_theta_et_pev_termes(paquet):
    p = paquet["path"]
    r = cli(p, "--backend", "cpu", "--restarts", "1", "--restart-sd", "0.3",
            "--fixed-theta", "1", "--theta-in", "--pev-termes", "g", "--quiet")
    assert r.returncode == 0, r.stderr
    out = _json(p)
    assert out["n_restarts"] == 1 and out["restart_gain"] >= 0
    assert out["n_fixed"] == 1 and out["n_par_free"] == 2
    assert _lire(p, "out_theta")[0] == 0.2                     # tenu a in_theta[0]
    assert set(out["pev_dims"]) == {"g"}


def test_cli_predict_et_only_predict(paquet):
    p = paquet["path"]
    L = np.array([[1.0, 0.0], [1.0, 1.0]])
    L.ravel(order="F").tofile(os.path.join(p, "pred_L.bin"))
    M = np.zeros((2, paquet["q"])); M[0, 0] = 1; M[1, 3] = 1
    M.ravel(order="F").tofile(os.path.join(p, "pred_M_g.bin"))
    r = cli(p, "--backend", "cpu", "--predict", "--quiet", "--maxiter", "300")
    assert r.returncode == 0, r.stderr
    out = _json(p)
    th = _lire(p, "out_theta")
    assert len(out["predictions"]["valeur"]) == 2
    assert os.path.exists(os.path.join(p, "out_pred_cov.bin"))
    # only-predict : aucun ajustement, theta relu, memes predictions
    th.tofile(os.path.join(p, "in_theta.bin"))
    r2 = cli(p, "--backend", "cpu", "--only-predict", "--quiet")
    assert r2.returncode == 0, r2.stderr
    out2 = _json(p)
    # les booleens sortent en 1.0 dans result.json (float() sur un bool)
    assert out2["only_predict"] in (True, 1.0) and out2["logLik"] is None
    assert np.allclose(out2["predictions"]["valeur"], out["predictions"]["valeur"], atol=1e-8)
    assert np.allclose(out2["predictions"]["se"], out["predictions"]["se"], atol=1e-8)
    cov = _lire(p, "out_pred_cov", (2, 2))
    assert np.allclose(np.sqrt(np.diag(cov)), out2["predictions"]["se"])


def test_cli_dsum_et_backend_gpu(tmp_path):
    p = str(tmp_path / "dsum")
    ecrire_paquet(p, dsum=True)
    r = cli(p, "--backend", "cpu", "--quiet", "--no-hessian", "--maxiter", "200")
    assert r.returncode == 0, r.stderr
    out = _json(p)
    assert out["n_par"] == 4
    assert out["composantes_noms"] == ["g", "bloc", "pair", "impair"]
    assert set(out["sigma_res_dims"]) == {"pair", "impair"}
    a_gpu = any(d["platform"] != "cpu" for d in device_report())
    r = cli(p, "--backend", "gpu", "--quiet", "--no-hessian", "--maxiter", "5")
    if a_gpu:
        assert r.returncode == 0
    else:
        assert r.returncode != 0 and "GPU demande" in (r.stderr + r.stdout)


def test_device_pick_et_report():
    d, plat = pick_device("cpu")
    assert plat == "cpu"
    d2, plat2 = pick_device("auto")
    assert plat2 in ("cpu", "gpu", "cuda", "rocm", "tpu")
    rep = device_report()
    assert any(r["platform"] == "cpu" for r in rep)
    a_gpu = any(r["platform"] != "cpu" for r in rep)
    if a_gpu:
        assert pick_device("gpu")[1] != "cpu"
    else:
        with pytest.raises(RuntimeError):
            pick_device("gpu")
    assert jax.config.jax_enable_x64 is True
