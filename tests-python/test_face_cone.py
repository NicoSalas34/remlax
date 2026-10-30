"""AI-REML sur un bloc us de rang incomplet : la face du cone doit tourner.

Sigma genetique simulee de rang 2 sur 4 caracteres. A l'optimum, deux valeurs
propres sont au plancher. Avant le correctif (v14), la face figeait les
vecteurs propres au moment ou une valeur propre touchait le plancher : l'AI
s'arretait sur un point non stationnaire au bord, se declarait en echec, et le
repli L-BFGS-B faisait tourner la face a petits pas. Le test exige que l'AI
conclue SEULE (pas de repli), en peu d'iterations, a la meme vraisemblance que
L-BFGS-B, et que la Sigma estimee soit bien de rang incomplet (au moins une
valeur propre au plancher, ce qui engage la face).
"""
import numpy as np

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml
from remlax.structures import build_sigma


def cas_rang_2(graine=4, q=80, t=4, rep=2):
    rng = np.random.default_rng(graine)
    L = rng.normal(size=(t, 2))
    G = L @ L.T                                   # rang 2
    u = rng.normal(size=(q, 2)) @ L.T             # q x t, cov G
    codes, trait = [], []
    for g in range(q):
        for a in range(t):
            for _ in range(rep):
                codes.append(g); trait.append(a)
    codes, trait = np.asarray(codes), np.asarray(trait)
    n = len(codes)
    X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    y = X @ rng.normal(size=t) + u[codes, trait] + rng.normal(size=n)
    terms = [od.terme_long("gen", codes, trait, q, t, struct="us")]
    res = od.residuelle("diag", trait, np.arange(n), t)
    return terms, res, y, X, G


def test_ai_conclut_seule_au_bord_du_cone():
    terms, res, y, X, _ = cas_rang_2()
    kw = dict(verbose=False, hessian=False, blups=False, pev=False)
    r_ai = fit_reml(terms, res, y, X, optimizer="ai", **kw)
    r_lb = fit_reml(terms, res, y, X, optimizer="lbfgsb", **kw)
    msg = r_ai["scipy_message"]
    assert "L-BFGS-B fallback" not in msg, msg
    assert r_ai["scipy_success"], msg
    assert "edge of the cone" in msg, msg
    assert r_ai["n_iter"] <= 60, r_ai["n_iter"]
    # meme optimum que L-BFGS-B, et l'AI au moins aussi haut
    assert r_ai["neg2_reml"] <= r_lb["neg2_reml"] + 1e-6
    assert abs(r_ai["neg2_reml"] - r_lb["neg2_reml"]) <= 1e-5
    assert r_ai.get("conv_decrement") is not False


def test_sigma_estimee_de_rang_incomplet():
    terms, res, y, X, _ = cas_rang_2()
    r = fit_reml(terms, res, y, X, optimizer="ai", verbose=False, hessian=False,
                 blups=False, pev=False)
    S = np.asarray(build_sigma(np.asarray(r["theta"])[:10], "us", 4))
    w = np.linalg.eigvalsh(S)
    # une valeur propre au plancher, les autres franchement positives
    assert w[0] / w[-1] < 1e-6, w
    assert w[1] / w[-1] > 1e-3, w
