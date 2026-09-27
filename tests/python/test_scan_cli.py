"""scan_cli en sous-processus : le scan a V figee depuis un paquet serialise.

Le paquet (rx_export) et les tableaux du scan (scan_manifest.json) sont ecrits
ici a la main. Le CLI est lance avec --theta-in --maxiter 0, comme le fait
rx_scan cote R, et sa table CSV est comparee a un appel direct de
scan.projeter / scan.scan sur le meme ajustement.
"""
import json
import os
import subprocess

import numpy as np
import pytest

import outils_dispositifs as od
from test_bundle_cli import Ecrivain
from remlax.bundle import Bundle
from remlax.fit import fit_reml
from remlax.scan import projeter, scan

RNG = np.random.default_rng(123)


def _paquet_scan(path, p_snp=30):
    q, rep = 20, 3
    n = q * rep
    g = np.tile(np.arange(q), rep)
    # voisin : le genotype suivant dans l'ordre du bloc, poids 1
    voisin = np.roll(g, 1)
    M = RNG.choice([-1.0, 1.0], size=(q, p_snp))
    y = RNG.normal(size=n) + 0.8 * M[g, 0] + RNG.normal(0, 0.7, q)[g]
    X = np.ones((n, 1))
    w = Ecrivain(path)
    w.put("y", y); w.put("X", X)
    w.put("term_names", ["gen", "vois"], "str")
    for nm, code in (("gen", g), ("vois", voisin)):
        w.put("term_%s_zi" % nm, np.arange(n), "i4"); w.put("term_%s_zj" % nm, code, "i4")
        w.put("term_%s_zx" % nm, np.ones(n)); w.put("term_%s_t" % nm, [1], "i4")
        w.put("term_%s_q" % nm, [q], "i4"); w.put("term_%s_rank" % nm, [0], "i4")
        w.put("term_%s_struct" % nm, ["iid"], "str"); w.put("term_%s_lvl" % nm, ["id"], "str")
    w.put("res_struct", ["iid"], "str"); w.put("res_lvl", ["id"], "str")
    w.put("res_t", [1], "i4"); w.put("res_rank", [0], "i4")
    w.put("res_trait", np.zeros(n), "i4"); w.put("res_unit", np.arange(n), "i4")
    w.put("res_nsec", [0], "i4")
    w.close()
    # tableaux du scan, a cote
    sm = Ecrivain(path); sm.man = []
    for fam, code in (("dir", g), ("ind", voisin)):
        sm.put("scan_%s_zi" % fam, np.arange(n), "i4"); sm.put("scan_%s_zj" % fam, code, "i4")
        sm.put("scan_%s_zx" % fam, np.ones(n))
    sm.put("scan_M", M)
    sm.put("scan_snp", ["s%d" % i for i in range(p_snp)], "str")
    garder = np.ones(p_snp, dtype=int); garder[-1] = 0
    sm.put("scan_garder", garder, "i4")
    with open(os.path.join(path, "scan_manifest.json"), "w") as f:
        json.dump({"arrays": sm.man, "meta": {"q_dir": q, "q_ind": q, "p": p_snp}}, f)
    return dict(g=g, voisin=voisin, M=M, y=y, X=X, n=n, q=q, p=p_snp)


def test_scan_cli_egale_scan_direct(tmp_path):
    p = str(tmp_path / "scan")
    d = _paquet_scan(p)
    bd = Bundle(p)
    terms, res = bd.terms(), bd.residual()
    f = fit_reml(terms, res, bd.y, bd.X, verbose=False, hessian=False)
    f["theta"].tofile(os.path.join(p, "in_theta.bin"))
    env = dict(os.environ, PYTHONPATH=od._SRC)
    tests = "dir,ind,sim,dir+ind+sim,ind|dir"
    r = subprocess.run([od.PY, "-m", "remlax.scan_cli", p, "--tests", tests, "--backend", "cpu",
                        "--theta-in", "--maxiter", "0", "--bloc", "7", "--quiet"],
                       capture_output=True, text=True, env=env, cwd=od.RACINE)
    assert r.returncode == 0, r.stderr + r.stdout
    csv = os.path.join(p, "scan_resultats.csv")
    assert os.path.exists(csv)
    with open(csv) as fh:
        entete = fh.readline().strip().split(",")
        lignes = [l.strip().split(",") for l in fh]
    assert len(lignes) == d["p"] and entete[0] == "snp" and lignes[0][0] == "s0"
    for t in tests.split(","):
        assert "chi2_" + t in entete and "p_" + t in entete
    meta = json.load(open(os.path.join(p, "scan_meta.json")))
    assert meta["p"] == d["p"] and meta["tests"] == tests.split(",")
    assert meta["neg2_reml"] == pytest.approx(f["neg2_reml"], rel=1e-10)
    # accord avec le scan direct
    Zd = od.dense_Z_np(terms[0], d["n"]); Zi = od.dense_Z_np(terms[1], d["n"])
    proj = projeter(f, bd.X, {"dir": Zd, "ind": Zi})
    garder = np.ones(d["p"], bool); garder[-1] = False
    ref = scan(proj, d["M"], tests=tests.split(","), garder=garder, bloc=7)
    col = entete.index("chi2_dir")
    chi_cli = np.array([float(l[col]) if l[col] else np.nan for l in lignes])
    assert np.allclose(chi_cli[:-1], ref["chi2_dir"][:-1], rtol=1e-8)
    assert np.isnan(chi_cli[-1])                                # SNP ecarte par garder
    col = entete.index("chi2_ind|dir")
    chi_cli = np.array([float(l[col]) if l[col] else np.nan for l in lignes])
    assert np.allclose(chi_cli[:-1], ref["chi2_ind|dir"][:-1], rtol=1e-8)
    # le SNP causal est le plus significatif en direct
    assert int(np.nanargmax(ref["chi2_dir"])) == 0
    # sans --theta-in et maxiter 0 : refus explicite
    r2 = subprocess.run([od.PY, "-m", "remlax.scan_cli", p, "--tests", "dir", "--backend", "cpu",
                         "--maxiter", "0", "--quiet"],
                        capture_output=True, text=True, env=env, cwd=od.RACINE)
    assert r2.returncode != 0 and "theta" in (r2.stderr + r2.stdout)
