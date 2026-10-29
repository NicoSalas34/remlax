"""Le moteur par blocs (blocs.py) rend-il la MEME vraisemblance que le moteur dense ?

Dispositifs :
  - cas_ige (test_econome) : deux especes, bacs, parente fixe sur les termes
    genetiques, termes locaux scalaires, residuelle us par plante ;
  - cas_essais : essais relies par un terme genotypique us a parente fixe et un
    terme AR1 global sur les genotypes, termes locaux us(2) par essai,
    residuelle us(2) par unite, un terme local AR1 confine a un essai (rendu
    global par le plan).
Sont compares valeur et gradient, puis un ajustement complet. Les modeles hors
perimetre doivent basculer vers le moteur dense econome.
"""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.model import make_objective
from remlax.blocs import PlanBlocs, make_objective_blocs, hors_perimetre, _paquets
from remlax.fit import fit_reml, initial_theta
from test_econome import cas_ige, cas_ar1, cas_sections_bloc, _chol_spd, _rel

TOL_F = 1e-11
TOL_G = 1e-8


def cas_essais(graine=4, n_essais=4, n_geno=25, rep=3):
    rng = np.random.default_rng(graine)
    lignes = []
    u = 0
    for e in range(n_essais):
        for g in range(n_geno):
            for _ in range(rep):
                for a in range(2):
                    if rng.random() < 0.9:
                        lignes.append((e, g, u, a))
                u += 1
    E, Gn, U, A = map(np.asarray, zip(*lignes))
    n = len(E)
    perm = rng.permutation(n)
    E, Gn, U, A = E[perm], Gn[perm], U[perm], A[perm]
    terms = [
        # global : genotype us(2), parente fixe
        dict(name="geno", struct="us", t=2, rank=0, q=n_geno, zi=np.arange(n),
             zj=A * n_geno + Gn, zx=np.ones(n), LK=_chol_spd(n_geno, rng), lvl="fixed"),
        # global : AR1 sur les genotypes (structure entre niveaux parametree)
        od.terme_facteur("geno_ar1", Gn, n_geno, lvl="ar1"),
    ]
    for e in range(n_essais):
        rr = np.flatnonzero(E == e)
        # local : bloc us(2) dans l'essai
        blocs_e = rng.integers(0, 5, len(rr))
        terms.append(dict(name="bloc_%d" % e, struct="us", t=2, rank=0, q=5, zi=rr,
                          zj=A[rr] * 5 + blocs_e, zx=np.ones(len(rr)), LK=None))
    # local AR1 confine a l'essai 0 : le plan doit le rendre global
    r0 = np.flatnonzero(E == 0)
    terms.append(dict(name="rang_e0", struct="iid", t=1, rank=0, q=8, zi=r0,
                      zj=rng.integers(0, 8, len(r0)), zx=np.ones(len(r0)), LK=None, lvl="ar1"))
    res = od.residuelle("us", A, U, 2)
    X = np.zeros((n, 2 * n_essais))
    X[np.arange(n), E * 2 + A] = 1.0
    y = rng.normal(size=n) + X @ rng.normal(size=X.shape[1])
    return terms, res, y, X


CAS = {"ige": cas_ige, "essais": cas_essais}


def _thetas(terms, res, y, k=3, graine=11):
    rng = np.random.default_rng(graine)
    th0 = initial_theta(terms, res, y)
    return [th0] + [th0 + rng.normal(0, 0.3, len(th0)) for _ in range(k - 1)]


@pytest.mark.parametrize("nom", list(CAS))
def test_objectif_identique(nom):
    terms, res, y, X = CAS[nom]()
    plan = PlanBlocs(terms, res, y, X)
    assert plan.n_blocs >= 2
    _, fj_std, _, _ = make_objective(terms, res, y, X)
    _, fj_blo, _ = make_objective_blocs(plan)
    for th in _thetas(terms, res, y):
        v1, g1 = fj_std(th)
        v2, g2 = fj_blo(th)
        assert abs(v1 - v2) <= TOL_F * abs(v1), (nom, v1, v2)
        assert _rel(g2, g1) <= TOL_G, (nom, _rel(g2, g1))


def test_plan_essais():
    terms, res, y, X = cas_essais()
    plan = PlanBlocs(terms, res, y, X)
    noms = {terms[k]["name"] for k in plan.globaux}
    assert {"geno", "geno_ar1", "rang_e0"} <= noms        # rang_e0 : local AR1 rendu global
    assert plan.n_blocs >= 4


def test_lots_identiques():
    """Traiter un paquet par lots de 1 bloc (scan) ou d'un coup (vmap) : meme valeur."""
    terms, res, y, X = cas_essais()
    th = _thetas(terms, res, y)[1]
    v_un = make_objective_blocs(PlanBlocs(terms, res, y, X, octets_lot=1))[1](th)
    v_tout = make_objective_blocs(PlanBlocs(terms, res, y, X))[1](th)
    assert abs(v_un[0] - v_tout[0]) <= TOL_F * abs(v_un[0])
    assert _rel(v_un[1], v_tout[1]) <= TOL_G


def test_paquets():
    g = _paquets(np.array([900, 910, 150, 160, 905, 155]), lambda m: m ** 3)
    tailles = sorted(sorted(int(x) for x in np.array([900, 910, 150, 160, 905, 155])[i]) for i in g)
    assert tailles == [[150, 155, 160], [900, 905, 910]]


@pytest.mark.parametrize("fab", [cas_ar1, cas_sections_bloc])
def test_hors_perimetre(fab):
    terms, res, y, X = fab()
    assert hors_perimetre(terms, res) is not None


def test_ajustement_identique():
    """Meme optimum (valeur), puis memes sorties au MEME theta.

    theta lui-meme ne se compare pas : sur ce dispositif (y bruit pur) plusieurs
    composantes sont mal identifiees, la vraisemblance est plate dans leur
    direction et deux optimiseurs s'arretent a des theta differents de meme
    valeur."""
    # Le moteur par blocs part du point du moteur dense et ne doit pas faire
    # PIRE. Sur ce y de bruit pur la vraisemblance est plate et multimodale
    # (1494,3194, 1494,3593, 1494,3599 selon le depart et le moteur) :
    # L-BFGS-B s'y arrete sur une reduction relative, et repartir du point
    # atteint peut encore descendre. L'egalite des valeurs se verifie plus bas,
    # au meme theta.
    terms, res, y, X = cas_essais()
    kw = dict(verbose=False, hessian=False, blups=False)
    r1 = fit_reml(terms, res, y, X, **kw)
    r2 = fit_reml(terms, res, y, X, moteur="blocs", theta_init=r1["theta"], **kw)
    assert r2["moteur"] == "blocs"
    assert r2["neg2_reml"] <= r1["neg2_reml"] + 1e-8 * abs(r1["neg2_reml"])
    kw0 = dict(theta_init=r1["theta"], maxiter=0, polish=0, verbose=False, hessian=False,
               blups=True, pev=True)
    s1 = fit_reml(terms, res, y, X, **kw0)
    s2 = fit_reml(terms, res, y, X, moteur="blocs", **kw0)
    assert _rel(s2["beta"], s1["beta"]) <= 1e-8
    for k in s1["blups"]:
        assert _rel(s2["blups"][k], s1["blups"][k]) <= 1e-8, k
        assert _rel(s2["pev"][k], s1["pev"][k]) <= 1e-6, k


def test_bascule_sans_decoupage():
    terms, res, y, X = cas_ar1()                # residuelle ar1 x ar1 : hors perimetre
    r = fit_reml(terms, res, y, X, moteur="blocs", verbose=False, hessian=False, blups=False)
    assert r["moteur"] == "dense" and r["memoire"] == "econome"


def test_reproductible():
    terms, res, y, X = cas_ige()
    _, fj, _ = make_objective_blocs(PlanBlocs(terms, res, y, X))
    th = _thetas(terms, res, y)[1]
    a, b = fj(th), fj(th)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
