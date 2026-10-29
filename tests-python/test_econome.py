"""Le mode econome (econome.py) rend-il la MEME vraisemblance que le mode standard ?

Chaque cas couvre un mode d'assemblage du plan econome :
  - residuelle "paires" (us par unite avec caracteres manquants, iid) ;
  - residuelle "plein" (ar1 x ar1 dense) et "bloc" (section partielle ar1) ;
  - groupes "plein" (support = toutes les lignes), "haut" (support >= la moitie)
    et "bloc" (termes locaux a un bac) ;
  - parente fixe pliee, structure entre niveaux parametree, fa.
Sont compares : valeur et gradient de l'objectif, la sortie (f, g, AI) de l'AI,
puis un ajustement complet (theta, logLik, beta, BLUP, PEV).

Tolerances RELATIVES : l'ordre des sommes flottantes differe entre les modes.
"""
import numpy as np
import pytest

import outils_dispositifs as od
from remlax import _x64  # noqa: F401
from remlax.model import make_objective, n_theta
from remlax.ai import make_fga
from remlax.econome import PlanEconome, make_objective_econome, make_fga_econome
from remlax.fit import fit_reml, initial_theta

TOL_F = 1e-11
TOL_G = 1e-8


def _chol_spd(q, rng):
    A = rng.normal(size=(q, q))
    K = A @ A.T / q + 0.5 * np.eye(q)
    return np.linalg.cholesky(K)


def cas_ige(graine=1, n_bacs=3, n_ble=40, n_luz=12, t_b=3, q_b=15, q_l=6):
    """Miniature de mvD7 : deux especes, bacs, format long avec manquants.
    Les tailles sont des parametres pour que le banc de temps (hors tests)
    puisse fabriquer un cas plus grand de la meme forme."""
    rng = np.random.default_rng(graine)
    lignes, trait, unit, esp, bac, geno = [], [], [], [], [], []
    u = 0
    for b in range(n_bacs):
        for _ in range(n_ble):
            g = rng.integers(q_b)
            for a in range(t_b):
                if rng.random() < 0.85:          # caracteres manquants
                    trait.append(a); unit.append(u); esp.append(0); bac.append(b); geno.append(g)
            u += 1
        for _ in range(n_luz):
            trait.append(0); unit.append(u); esp.append(1); bac.append(b)
            geno.append(rng.integers(q_l)); u += 1
    trait, unit, esp, bac, geno = map(np.asarray, (trait, unit, esp, bac, geno))
    n = len(trait)
    # melanger les lignes : les supports ne sont pas contigus
    perm = rng.permutation(n)
    trait, unit, esp, bac, geno = trait[perm], unit[perm], esp[perm], bac[perm], geno[perm]
    rb, rl = np.flatnonzero(esp == 0), np.flatnonzero(esp == 1)
    terms = []
    # genetique ble (support = lignes de ble, > n/2 : mode "haut"), parente fixe
    terms.append(dict(name="gen_ble", struct="us", t=t_b, rank=0, q=q_b,
                      zi=rb, zj=trait[rb] * q_b + geno[rb], zx=np.ones(len(rb)),
                      LK=_chol_spd(q_b, rng), lvl="fixed"))
    # genetique luzerne (support < n/2 : "bloc"), K fournie
    terms.append(dict(name="gen_luz", struct="iid", t=1, rank=0, q=q_l,
                      zi=rl, zj=geno[rl], zx=np.ones(len(rl)), LK=_chol_spd(q_l, rng),
                      lvl="fixed"))
    # effet commun a toutes les lignes ("plein") : unite, ponderee
    terms.append(dict(name="plante", struct="iid", t=1, rank=0, q=int(unit.max()) + 1,
                      zi=np.arange(n), zj=unit, zx=rng.random(n) + 0.5, LK=None))
    # termes locaux a chaque bac : spatial (indicatrice) et IEE (voisins ponderes)
    for b in range(n_bacs):
        rr = np.flatnonzero(bac == b)
        terms.append(dict(name="spat_%d" % b, struct="iid", t=1, rank=0, q=6, zi=rr,
                          zj=rng.integers(0, 6, len(rr)), zx=np.ones(len(rr)), LK=None))
        k = 4
        zi = np.repeat(rr, k)
        terms.append(dict(name="iee_%d" % b, struct="iid", t=1, rank=0, q=n_ble + n_luz,
                          zi=zi, zj=rng.integers(0, n_ble + n_luz, len(zi)),
                          zx=rng.random(len(zi)), LK=None))
    secs = [dict(name="ble", struct="us", t=t_b, rank=0, trait=trait[rb], unit=unit[rb],
                 n_unit=int(unit.max()) + 1, lvl="id", rows=rb),
            dict(name="luz", struct="iid", t=1, rank=0, trait=np.zeros(len(rl), np.int64),
                 unit=unit[rl], n_unit=int(unit.max()) + 1, lvl="id", rows=rl)]
    res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64), unit=unit,
               n_unit=int(unit.max()) + 1, sections=secs)
    X = np.zeros((n, t_b + 1))
    X[rb, trait[rb]] = 1.0
    X[rl, t_b] = 1.0
    y = rng.normal(size=n) + X @ rng.normal(size=t_b + 1)
    return terms, res, y, X


def cas_ar1(graine=2):
    """Residuelle us x ar1 x ar1 dense ("plein"), niveau ar1 parametre, fa."""
    rng = np.random.default_rng(graine)
    nr, nc, t = 6, 7, 2
    cell = np.arange(nr * nc)
    unit = np.tile(cell, t)
    trait = np.repeat(np.arange(t), nr * nc)
    n = len(unit)
    res = od.residuelle("us", trait, unit, t, lvl="ar1ar1", dims=(nr, nc))
    q = 10
    g = rng.integers(0, q, nr * nc)[unit]
    terms = [od.terme_long("g", g, trait, q, t, struct="fa", rank=1),
             od.terme_facteur("col", cell[unit] % nc, nc, lvl="ar1")]
    X = np.zeros((n, t)); X[np.arange(n), trait] = 1.0
    y = rng.normal(size=n)
    return terms, res, y, X


def cas_sections_bloc(graine=3):
    """Deux sections entremelees : iid ("paires") et ar1 partielle ("bloc")."""
    rng = np.random.default_rng(graine)
    nA, nB = 60, 40
    n = nA + nB
    perm = rng.permutation(n)
    rowsA, rowsB = np.sort(perm[:nA]), np.sort(perm[nA:])
    secA = dict(struct="iid", t=1, rank=0, trait=np.zeros(nA, np.int64),
                unit=np.arange(nA), n_unit=nA, lvl="id", rows=rowsA, name="A")
    secB = dict(struct="iid", t=1, rank=0, trait=np.zeros(nB, np.int64),
                unit=np.arange(nB), n_unit=nB, lvl="ar1", rows=rowsB, name="B")
    res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64), unit=np.arange(n),
               n_unit=n, sections=[secA, secB])
    terms = [od.terme_facteur("g", rng.integers(0, 12, n), 12)]
    X = np.ones((n, 1))
    y = rng.normal(size=n)
    return terms, res, y, X


CAS = {"ige": cas_ige, "ar1": cas_ar1, "sections": cas_sections_bloc}


def _thetas(terms, res, y, k=3, graine=7):
    rng = np.random.default_rng(graine)
    th0 = initial_theta(terms, res, y)
    return [th0] + [th0 + rng.normal(0, 0.3, len(th0)) for _ in range(k - 1)]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("adjoint", ["inv", "cho"])
@pytest.mark.parametrize("nom", list(CAS))
def test_objectif_identique(nom, adjoint):
    terms, res, y, X = CAS[nom]()
    _, fj_std, _, _ = make_objective(terms, res, y, X)
    _, fj_eco, _ = make_objective_econome(PlanEconome(terms, res, y, X), adjoint=adjoint)
    for th in _thetas(terms, res, y):
        v1, g1 = fj_std(th)
        v2, g2 = fj_eco(th)
        assert abs(v1 - v2) <= TOL_F * abs(v1), (nom, v1, v2)
        assert _rel(g2, g1) <= TOL_G, (nom, _rel(g2, g1))


@pytest.mark.parametrize("adjoint", ["inv", "cho"])
@pytest.mark.parametrize("nom", list(CAS))
def test_ai_identique(nom, adjoint):
    terms, res, y, X = CAS[nom]()
    fga_std = make_fga(terms, res, y, X)
    fga_eco = make_fga_econome(PlanEconome(terms, res, y, X), adjoint=adjoint)
    for th in _thetas(terms, res, y):
        f1, g1, A1 = fga_std(th)
        f2, g2, A2 = fga_eco(th)
        assert abs(float(f1) - float(f2)) <= TOL_F * abs(float(f1))
        assert _rel(g2, g1) <= TOL_G
        assert _rel(A2, A1) <= TOL_G


def test_plan_couvre_tous_les_modes():
    p = PlanEconome(*cas_ige())
    assert {d["mode"] for d in p.plan_grp} == {"plein", "haut", "bloc"}
    assert {d["mode"] for d in p.plan_secs} == {"paires"}
    p2 = PlanEconome(*cas_ar1())
    assert [d["mode"] for d in p2.plan_secs] == ["plein"]
    p3 = PlanEconome(*cas_sections_bloc())
    assert [d["mode"] for d in p3.plan_secs] == ["paires", "bloc"]


@pytest.mark.parametrize("optimiseur", ["lbfgsb", "ai"])
def test_ajustement_identique(optimiseur):
    terms, res, y, X = cas_ige()
    kw = dict(verbose=False, hessian=True, blups=True, pev=True, optimizer=optimiseur)
    r1 = fit_reml(terms, res, y, X, memoire="standard", **kw)
    r2 = fit_reml(terms, res, y, X, memoire="econome", **kw)
    assert r2.get("memoire") == "econome"
    assert abs(r1["neg2_reml"] - r2["neg2_reml"]) <= 1e-8 * abs(r1["neg2_reml"])
    # Une variance collee a la borne (theta pres de -12) est dans une direction
    # plate : sa position exacte, et les BLUP ~1e-10 du terme, ne sont pas
    # identifies. theta se compare hors de la borne, les BLUP en absolu.
    loin = (np.abs(r1["theta"]) < 10) & (np.abs(r2["theta"]) < 10)
    assert _rel(r2["theta"][loin], r1["theta"][loin]) <= 1e-4
    assert _rel(r2["beta"], r1["beta"]) <= 1e-5
    echelle = float(np.std(y))
    for k in r1["blups"]:
        assert np.max(np.abs(r2["blups"][k] - r1["blups"][k])) <= 1e-5 * echelle, k
        assert np.max(np.abs(r2["pev"][k] - r1["pev"][k])) <= 1e-5 * echelle ** 2, k


@pytest.mark.parametrize("nom", list(CAS))
def test_sorties_identiques_au_meme_theta(nom):
    """beta, vbeta, BLUP et PEV au MEME theta : seule la voie de calcul differe."""
    terms, res, y, X = CAS[nom]()
    th = _thetas(terms, res, y)[1]
    kw = dict(theta_init=th, maxiter=0, polish=0, verbose=False, hessian=False,
              blups=True, pev=True)
    r1 = fit_reml(terms, res, y, X, memoire="standard", **kw)
    r2 = fit_reml(terms, res, y, X, memoire="econome", **kw)
    assert np.array_equal(r1["theta"], r2["theta"])
    assert _rel(r2["beta"], r1["beta"]) <= 1e-9
    assert _rel(r2["vbeta"], r1["vbeta"]) <= 1e-9
    for k in r1["blups"]:
        assert _rel(r2["blups"][k], r1["blups"][k]) <= 1e-8, k
        assert _rel(r2["pev"][k], r1["pev"][k]) <= 1e-8, k


def test_reproductible_a_l_identique():
    """Deux evaluations du mode econome au meme point rendent les memes bits."""
    terms, res, y, X = cas_ige()
    _, fj, _ = make_objective_econome(PlanEconome(terms, res, y, X))
    th = _thetas(terms, res, y)[1]
    v1, g1 = fj(th)
    v2, g2 = fj(th)
    assert v1 == v2 and np.array_equal(g1, g2)
