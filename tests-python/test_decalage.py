"""Modele structurel a decalage spatial (lag.py) et residuelle "none".

Ce qui est verifie, et contre quoi :
  - la valeur de l'objectif, pour les trois criteres (reml_y, jacobian, ml) et
    les trois chemins de log|det A| (nilpotent, rapide, general), contre un
    calcul dense ECRIT ICI en numpy (V^-1, P, slogdet(A), A^-1 X), sans rien
    reprendre du moteur ;
  - le gradient automatique (theta de covariance ET Psi), contre des
    differences finies centrees, avec et sans retroaction ;
  - la cotangente de y ajoutee a reml_from_V : 2 P y ;
  - Psi fixe a 0 (aucune entree libre) : l'ajustement est celui du modele
    sans decalage, bit pour bit ;
  - la region admissible : penalite finie de gradient nul hors region, y
    compris quand deux facteurs det(I - lambda Psi) changent de signe ensemble ;
  - residuelle "none" : V non definie positive refusee avec un message clair,
    et un modele VP+ (terme de plante [I, W] us) ajuste.
"""
import os
import sys

import jax
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from remlax.core import reml_from_V  # noqa: E402
from remlax.model import make_objective, assemble_V, dense_Z  # noqa: E402
from remlax.fit import fit_reml  # noqa: E402
from remlax.lag import prepare_lag, PENALITE  # noqa: E402


def reine(nr, nc):
    """Voisinage reine, poids 1, sur une grille nr x nc."""
    N = nr * nc
    W = np.zeros((N, N))
    for a in range(N):
        ra, ca = divmod(a, nc)
        for b in range(N):
            rb, cb = divmod(b, nc)
            if a != b and abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                W[a, b] = 1.0
    return W


def jeu(nr=6, nc=5, q=2, ng=10, seed=1):
    rng = np.random.default_rng(seed)
    N = nr * nc
    W = reine(nr, nc)
    tr = np.repeat(np.arange(q), N)
    un = np.tile(np.arange(N), q)
    y = rng.normal(size=N * q)
    X = np.zeros((N * q, q))
    X[np.arange(N * q), tr] = 1.0
    g = rng.integers(0, ng, N)
    term = dict(name="gen", struct="us", t=q, rank=0, q=ng, zi=np.arange(N * q),
                zj=tr * ng + g[un], zx=np.ones(N * q), LK=None, lvl="id")
    res = dict(struct="us", t=q, rank=0, trait=tr, unit=un, lvl="id")
    return W, tr, un, y, X, term, res


def lag_de(W_list, tr, un, q, N, mask, crit, force=None):
    return dict(q=q, N=N, W=[sp.csr_matrix(w) for w in W_list], trait=tr, unit=un,
                mask=np.asarray(mask, bool), value=np.zeros((q, q)), init=np.zeros((q, q)),
                traits=["T%d" % (k + 1) for k in range(q)], criterion=crit, force_chemin=force)


def brute(th_cov, Psi, W_list, term, res, y, X, crit):
    """-2logL dense, ecrit sans le moteur (sauf l'assemblage de V)."""
    n = len(y)
    q = Psi.shape[0]
    N = W_list[0].shape[0]
    M = np.zeros((n, n))
    for t in range(q):
        Wt = W_list[0] if len(W_list) == 1 else W_list[t]
        for u in range(q):
            M[t * N:(t + 1) * N, u * N:(u + 1) * N] = Psi[t, u] * Wt
    A = np.eye(n) - M
    ys = A @ y
    V = np.asarray(assemble_V(jnp.asarray(th_cov), [term], [dense_Z(term, n)], res, n))
    Vi = np.linalg.inv(V)
    XVX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XVX, X.T @ Vi)
    p = X.shape[1]
    if crit == "ml":
        v = n * np.log(2 * np.pi) + np.linalg.slogdet(V)[1] + ys @ P @ ys
    else:
        v = (n - p) * np.log(2 * np.pi) + np.linalg.slogdet(V)[1] + np.linalg.slogdet(XVX)[1] + ys @ P @ ys
    v -= 2 * np.linalg.slogdet(A)[1]
    if crit == "reml_y":
        Xy = np.linalg.solve(A, X)
        v += -np.linalg.slogdet(Xy.T @ Xy)[1] + np.linalg.slogdet(X.T @ X)[1]
    return v


TH_COV = np.array([0.1, 0.2, -0.3, 0.0, 0.3, -0.1])
PLEIN = np.ones((2, 2), bool)
TRI = np.array([[0, 0], [1, 0]], bool)     # T2 <- T1 seulement : sans boucle


@pytest.mark.parametrize("crit", ["reml_y", "jacobian", "ml"])
@pytest.mark.parametrize("cas", ["rapide", "general", "nilpotent", "general_2W", "nilpotent_2W"])
def test_valeur_contre_calcul_dense_et_gradient_contre_differences_finies(crit, cas):
    W, tr, un, y, X, term, res = jeu()
    N, q = W.shape[0], 2
    W2 = 0.5 * W
    W2[0, 1], W2[1, 0] = 2.0, 0.0          # W propre au caractere 2, non symetrique
    Wl = [W, W2] if cas.endswith("_2W") else [W]
    mask = TRI if cas.startswith("nilpotent") else PLEIN
    force = "general" if cas == "general" else None
    L = prepare_lag(lag_de(Wl, tr, un, q, N, mask, crit, force), y, X)
    attendu = {"rapide": "rapide", "general": "general", "nilpotent": "nilpotent",
               "general_2W": "general", "nilpotent_2W": "nilpotent"}[cas]
    assert L["chemin"] == attendu
    th_lag = np.array([0.4, -0.5, 0.6, 0.2])[:int(mask.sum())] if mask.sum() == 4 else np.array([0.7])
    th = np.concatenate([TH_COV, th_lag])
    _, fj, _, _ = make_objective([term], res, y, X, lag=L)
    v, g = fj(th)
    Psi = np.zeros((q, q))
    Psi[np.nonzero(mask)] = th_lag / L["scale"]
    assert abs(v - brute(TH_COV, Psi, Wl, term, res, y, X, crit)) < 1e-9 * max(1.0, abs(v))
    eps = 1e-6
    gfd = np.array([(fj(th + eps * e)[0] - fj(th - eps * e)[0]) / (2 * eps) for e in np.eye(len(th))])
    assert np.max(np.abs(g - gfd)) < 1e-6 * max(1.0, np.max(np.abs(g)))


def test_rapide_egal_general_sur_le_meme_modele():
    W, tr, un, y, X, term, res = jeu(seed=4)
    N = W.shape[0]
    th = np.concatenate([TH_COV, [0.3, -0.6, 0.5, 0.25]])
    for crit in ("reml_y", "jacobian", "ml"):
        vr, gr = make_objective([term], res, y, X, lag=prepare_lag(
            lag_de([W], tr, un, 2, N, PLEIN, crit), y, X))[1](th)
        vg, gg = make_objective([term], res, y, X, lag=prepare_lag(
            lag_de([W], tr, un, 2, N, PLEIN, crit, "general"), y, X))[1](th)
        assert abs(vr - vg) < 1e-9 * abs(vr)
        assert np.max(np.abs(gr - gg)) < 1e-7 * np.max(np.abs(gr))


def test_cotangente_de_y_vaut_2_P_y():
    rng = np.random.default_rng(0)
    n = 25
    B = rng.normal(size=(n, n))
    V = B @ B.T / n + np.eye(n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)
    g = np.asarray(jax.grad(reml_from_V, argnums=1)(jnp.asarray(V), jnp.asarray(y), jnp.asarray(X)))
    Vi = np.linalg.inv(V)
    P = Vi - Vi @ X @ np.linalg.solve(X.T @ Vi @ X, X.T @ Vi)
    assert np.max(np.abs(g - 2 * P @ y)) < 1e-10


def test_psi_fixe_a_zero_reproduit_le_modele_sans_decalage():
    W, tr, un, y, X, term, res = jeu(seed=2)
    N = W.shape[0]
    sans = fit_reml([term], res, y, X, verbose=False, hessian=False, blups=True)
    for crit in ("reml_y", "jacobian"):
        L = lag_de([W], tr, un, 2, N, np.zeros((2, 2), bool), crit)
        avec = fit_reml([term], res, y, X, verbose=False, hessian=False, blups=True, lag=L)
        # Egalite BIT A BIT sur CPU. Sur GPU, les reductions XLA ne sont pas
        # deterministes d'un appel a l'autre (README, "Known limitations") :
        # -2logL coincide, mais l'optimiseur peut s'arreter a un theta voisin
        # sur le meme plateau. On y tolere donc l'ecart d'arrondi.
        if jax.default_backend() == "cpu":
            assert avec["neg2_reml"] == sans["neg2_reml"]
            assert np.array_equal(avec["theta"], sans["theta"])
            assert np.array_equal(avec["beta"], sans["beta"])
        else:
            assert abs(avec["neg2_reml"] - sans["neg2_reml"]) <= 1e-9 * abs(sans["neg2_reml"])
            # atol 1e-4 sur theta : une variance de ce jeu est au plancher
            # (theta ~ -11,93, var ~ 4e-11), direction plate ou l'arret varie
            np.testing.assert_allclose(avec["theta"], sans["theta"], rtol=1e-6, atol=1e-4)
            np.testing.assert_allclose(avec["beta"], sans["beta"], rtol=1e-6, atol=1e-7)
        assert avec["lag"]["logdet_A"] == 0.0 and avec["lag"]["chemin"] == "nilpotent"


def test_hors_region_penalite_finie_et_gradient_nul():
    W, tr, un, y, X, term, res = jeu()
    N = W.shape[0]
    L = prepare_lag(lag_de([W], tr, un, 2, N, PLEIN, "reml_y"), y, X)
    lmax = float(np.max(L["lam_np"]))
    fj = make_objective([term], res, y, X, lag=L)[1]
    # Psi = diag(c, c) avec c lambda_max > 1 : chaque det(I - lambda Psi) vaut
    # (1 - lambda c)^2 > 0, mais deux facteurs ont change de signe ensemble.
    c = 1.3 / lmax
    th = np.concatenate([TH_COV, [c * L["scale"], 0.0, 0.0, c * L["scale"]]])
    v, g = fj(th)
    assert v == PENALITE and np.all(g == 0.0)
    # Et une valeur propre reelle seule au-dela de 1/lambda_max.
    th = np.concatenate([TH_COV, [c * L["scale"], 0.0, 0.0, 0.0]])
    v, g = fj(th)
    assert v == PENALITE and np.all(np.isfinite(g))
    # Juste en dessous : pas de penalite.
    c = 0.9 / lmax
    th = np.concatenate([TH_COV, [c * L["scale"], 0.0, 0.0, c * L["scale"]]])
    assert fj(th)[0] < 1e6


def test_psi_initial_hors_region_refuse():
    W, tr, un, y, X, term, res = jeu()
    N = W.shape[0]
    L = lag_de([W], tr, un, 2, N, PLEIN, "reml_y")
    L["init"] = np.array([[2.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="region admissible"):
        fit_reml([term], res, y, X, verbose=False, hessian=False, blups=False, lag=L)


def _vp_plus(W, tr, un, g, ng, q, N):
    """Terme genetique [Z, WZ] us 2q et terme de plante [I, W] us 2q."""
    WZcol = sp.csr_matrix(W) @ sp.csr_matrix((np.ones(N), (np.arange(N), g)), shape=(N, ng))
    zi, zj, zx = [], [], []
    for k in range(q):                                  # DGE_k
        r = np.arange(N) + k * N
        zi.append(r); zj.append(k * ng + g); zx.append(np.ones(N))
    for k in range(q):                                  # IGE_k
        C = WZcol.tocoo()
        zi.append(C.row + k * N); zj.append((q + k) * ng + C.col); zx.append(C.data)
    gen = dict(name="gen", struct="us", t=2 * q, rank=0, q=ng, zi=np.concatenate(zi),
               zj=np.concatenate(zj), zx=np.concatenate(zx), LK=None, lvl="id")
    Wc = sp.csr_matrix(W).tocoo()
    zi, zj, zx = [], [], []
    for k in range(q):                                  # res_k : I
        zi.append(np.arange(N) + k * N); zj.append(k * N + np.arange(N)); zx.append(np.ones(N))
    for k in range(q):                                  # IEE_k : W
        zi.append(Wc.row + k * N); zj.append((q + k) * N + Wc.col); zx.append(Wc.data)
    plante = dict(name="plante", struct="us", t=2 * q, rank=0, q=N, zi=np.concatenate(zi),
                  zj=np.concatenate(zj), zx=np.concatenate(zx), LK=None, lvl="id")
    return gen, plante


def test_residuelle_none_refusee_si_V_singuliere_et_vp_plus_ajuste():
    W, tr, un, y, X, term, res = jeu(nr=5, nc=4, ng=6, seed=3)
    N, q = W.shape[0], 2
    aucune = dict(struct="none", t=q, rank=0, trait=tr, unit=un, lvl="id")
    # Seul le terme genetique : rang <= 2 x 6 < n, V singuliere.
    with pytest.raises(ValueError, match="residuelle 'none'"):
        fit_reml([term], aucune, y, X, verbose=False, hessian=False, blups=False)
    g = np.asarray(term["zj"][:N]) % 6
    gen, plante = _vp_plus(W, tr, un, g, 6, q, N)
    r = fit_reml([gen, plante], aucune, y, X, verbose=False, hessian=False, blups=False, maxiter=300)
    assert np.isfinite(r["neg2_reml"]) and "sigma_res" not in r and r["sigmas_res"] == {}


def test_alias_vp_plus_q_moins_1_sur_2_directions_plates():
    """Terme de plante [I, W] : Cov(res_t, IEE_u) et Cov(res_u, IEE_t) ont le
    meme motif quand W est symetrique (theorie_v2.md T1b) ; pas quand W ne
    l'est pas ; et le terme genetique [Z, WZ] n'en a pas (replication)."""
    from remlax.model import aliased_entries
    for q, attendu in ((2, 1), (3, 3)):
        W, tr, un, y, X, term, res = jeu(nr=5, nc=4, q=q, ng=6, seed=3)
        N = W.shape[0]
        g = np.asarray(term["zj"][:N]) % 6
        gen, plante = _vp_plus(W, tr, un, g, 6, q, N)
        al = aliased_entries([gen, plante], N * q)
        assert [a["terme"] for a in al] == ["plante"] * attendu
        assert all(len(a["entrees"]) == 2 for a in al)
        if q == 2:
            assert sorted(al[0]["entrees"]) == [(2, 1), (3, 0)]   # IEE1-res2 = IEE2-res1
        Wa = W.copy()
        Wa[0, 1] = 2.0                                          # W non symetrique
        gen, plante = _vp_plus(Wa, tr, un, g, 6, q, N)
        assert aliased_entries([gen, plante], N * q) == []


def test_vp_plus_directions_plates_exclues_du_verdict_et_sommes_identifiees():
    from remlax.components import varcomp_table
    W, tr, un, y, X, term, res = jeu(nr=5, nc=4, ng=6, seed=3)
    N, q = W.shape[0], 2
    aucune = dict(struct="none", t=q, rank=0, trait=tr, unit=un, lvl="id")
    g = np.asarray(term["zj"][:N]) % 6
    gen, plante = _vp_plus(W, tr, un, g, 6, q, N)
    r = fit_reml([gen, plante], aucune, y, X, verbose=False, blups=False, maxiter=500)
    assert r["n_plat_attendu"] == 1 and r["n_par_identifies"] == r["n_par"] - 1 == 19
    assert r["n_plat_exclues"] == 1 and r["n_neg_eig"] == 0
    vc = varcomp_table(r["theta"], r["hessian"], [gen, plante], aucune, alias=r["alias"])
    code = {d["nom"]: d for d in vc}
    assert code["plante[3,2]"]["contrainte"] == "N" and code["plante[4,1]"]["contrainte"] == "N"
    s = code["plante[3,2]+plante[4,1]"]
    assert abs(s["valeur"] - (code["plante[3,2]"]["valeur"] + code["plante[4,1]"]["valeur"])) < 1e-12
    assert np.isfinite(s["se"]) and s["se"] > 0
