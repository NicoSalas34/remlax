"""Defauts corriges apres le nettoyage de 2026-10 : un test par defaut."""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.fit import fit_reml


def _terme_kinv(graine=2, q=15, rep=4):
    rng = np.random.default_rng(graine)
    A = np.eye(q) + 0.3 * (np.ones((q, q)) - np.eye(q)) / q
    codes = np.repeat(np.arange(q), rep)
    n = len(codes)
    tm = od.terme_facteur("anim", codes, q)
    tm.update(Kinv=np.linalg.inv(A), Kinv_logdet=float(-np.linalg.slogdet(A)[1]), LK=None)
    y = rng.normal(size=n)
    return [tm], od.residuelle_iid(n), y, np.ones((n, 1))


@pytest.mark.parametrize("moteur", ["dense", "blocs"])
def test_terme_kinv_seul_refuse_hors_mme(moteur):
    terms, res, y, X = _terme_kinv()
    with pytest.raises(ValueError, match="precision Kinv"):
        fit_reml(terms, res, y, X, moteur=moteur, verbose=False, hessian=False, blups=False)


def test_terme_kinv_seul_accepte_par_le_mme():
    terms, res, y, X = _terme_kinv()
    r = fit_reml(terms, res, y, X, moteur="mme", verbose=False, hessian=False, blups=False)
    assert np.isfinite(r["logLik"])


def test_terme_kinv_seul_refuse_par_la_voie_bayesienne():
    from remlax.bayes import prepare
    terms, res, y, X = _terme_kinv()
    with pytest.raises(ValueError, match="precision Kinv"):
        prepare(terms, res, y, X)


def _dispositif_simple(graine=4, q=12, rep=5, struct="iid", t=1):
    rng = np.random.default_rng(graine)
    codes = np.repeat(np.arange(q), rep)
    n = len(codes)
    tm = od.terme_facteur("g", codes, q, t=t, struct=struct, rank=1 if struct == "rr" else 0)
    return [tm], od.residuelle_iid(n), rng.normal(size=n), np.ones((n, 1))


def test_plancher_mme_seulement_si_le_moteur_reste_mme():
    kw = dict(verbose=False, hessian=False, blups=False)
    terms, res, y, X = _dispositif_simple()
    r = fit_reml(terms, res, y, X, moteur="mme", **kw)
    assert r["moteur"] == "mme" and r["par_floor"] > -12.0
    # Hors perimetre du MME (structure rr) : bascule vers le dense, plancher d'origine.
    terms, res, y, X = _dispositif_simple(struct="rr", t=2)
    r = fit_reml(terms, res, y, X, moteur="mme", **kw)
    assert r["moteur"] == "dense" and r["par_floor"] == -12.0


def test_grad_rel_identique_avec_et_sans_hessien():
    """Les deux voies du diagnostic (avec et sans Hessien) rendent le meme
    grad_rel, y compris quand un parametre a sa borne est pousse vers
    l'interieur par le gradient : il compte dans les deux."""
    from remlax.diagnostics import diagnostic, gradient_diagnostic
    theta = np.array([-12.0, 0.5, 2.0])
    g = np.array([-30.0, 1e-3, -2e-3])               # le premier pousse vers l'interieur
    H = np.eye(3)
    a = diagnostic(H, g, theta, -12.0, 12.0, f_obj=500.0)
    b = gradient_diagnostic(g, theta, -12.0, 12.0, f_obj=500.0)
    assert a["grad_rel"] == b["grad_rel"]
    assert a["grad_proj_max"] == b["grad_proj_max"] == 30.0


def test_iterations_ai_comptent_les_pas_acceptes(capsys):
    """n_iter de l'AI compte les iterations acceptees (une ligne aiv chacune),
    pas les points essayes. Le cas ar1 de test_econome essaie trois fois plus
    de points qu'il ne fait d'iterations."""
    from test_econome import cas_ar1
    terms, res, y, X = cas_ar1()
    r = fit_reml(terms, res, y, X, optimizer="ai", verbose=True, hessian=False, blups=False)
    lignes = [l for l in capsys.readouterr().out.splitlines() if l.lstrip().startswith("aiv")]
    assert r["scipy_success"]
    assert r["n_iter"] == len(lignes)


def test_repli_du_noyau_c_annonce(monkeypatch, capsys):
    """Sans noyau C chargeable, la bascule vers la boucle NumPy est dite."""
    from remlax import selinv
    monkeypatch.setattr(selinv, "_C_KERNEL", None)
    monkeypatch.setenv("REMLAX_SELINV_LIB", "/chemin/absent/libselinv.so")
    assert selinv._c_kernel() is None
    assert "indisponible" in capsys.readouterr().err
    monkeypatch.setattr(selinv, "_C_KERNEL", None)


def test_residuelle_sep_refusee():
    from remlax.checks import validate
    n = 12
    res = od.residuelle("iid", np.zeros(n), np.arange(n), 1, lvl="sep")
    terms = [od.terme_facteur("g", np.arange(n) % 3, 3)]
    with pytest.raises(ValueError, match="ar1ar1"):
        validate(terms, res, np.zeros(n), np.ones((n, 1)))
