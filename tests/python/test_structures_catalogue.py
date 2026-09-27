"""Catalogue des structures de Sigma : chaque nom de STRUCTURES, un par un.

Pour chaque structure : le nombre de parametres suit la formule de l'annexe C
(recalculee ici a la main), le facteur rendu par chol_sigma reconstruit une
Sigma symetrique et semi-definie positive pour un theta quelconque, theta0
donne bien var * I, et sigma_loadings relit Lambda depuis theta.

Les formules de comptage sont ecrites INDEPENDAMMENT dans ce fichier : le test
ne fait pas confiance a n_params pour verifier n_params.
"""
import numpy as np
import pytest

import outils_dispositifs  # noqa: F401  (ajoute src/ au chemin)
import jax.numpy as jnp
from remlax import structures as S

RNG = np.random.default_rng(20260927)


def _n_load(t, r):
    return sum(min(i, r) for i in range(1, t + 1))


def _attendu(struct, t, r):
    return {
        "iid": 1, "diag": t, "us": t * (t + 1) // 2,
        "fa": _n_load(t, r) + t, "rr": _n_load(t, r),
        "chol": int((r + 1) * (t - r / 2.0)), "ante": int((r + 1) * (t - r / 2.0)),
        "corh": t + 1, "fixed": 0,
    }[struct]


CAS = [("iid", 1, 0), ("iid", 3, 0), ("diag", 3, 0), ("us", 2, 0), ("us", 4, 0),
       ("fa", 4, 1), ("fa", 4, 2), ("rr", 4, 1), ("rr", 4, 2), ("rr", 5, 3),
       ("chol", 4, 1), ("chol", 4, 3), ("ante", 4, 1), ("ante", 4, 3),
       ("corh", 3, 0), ("corh", 5, 0), ("fixed", 3, 0)]


@pytest.mark.parametrize("struct,t,r", CAS)
def test_n_params_suit_annexe_c(struct, t, r):
    assert S.n_params(struct, t, r) == _attendu(struct, t, r)


@pytest.mark.parametrize("struct,t,r", CAS)
def test_sigma_symetrique_psd_a_theta_quelconque(struct, t, r):
    p = S.n_params(struct, t, r)
    th = jnp.asarray(RNG.normal(0, 0.8, p))
    fixed = None
    if struct == "fixed":
        A = RNG.normal(size=(t, t)); fixed = A @ A.T + np.eye(t)
    L = S.chol_sigma(th, struct, t, r, fixed=fixed)
    assert L.shape[0] == t
    Sig = np.asarray(S.build_sigma(th, struct, t, r, fixed=fixed))
    assert Sig.shape == (t, t)
    assert np.allclose(Sig, Sig.T, atol=1e-12)
    w = np.linalg.eigvalsh(Sig)
    assert w.min() >= -1e-10
    if struct == "rr":
        # rang reduit PUR : exactement `r` valeurs propres non nulles
        assert int(np.sum(w > 1e-10 * w.max())) == min(r, t)
    elif struct == "fixed":
        assert np.allclose(Sig, fixed, atol=1e-8)
    else:
        assert w.min() > 0


@pytest.mark.parametrize("struct,t,r", [c for c in CAS if c[0] != "fixed"])
def test_theta0_donne_var_fois_identite_ou_presque(struct, t, r):
    th = S.theta0(struct, t, r, var=2.5)
    assert th.shape == (S.n_params(struct, t, r),)
    Sig = np.asarray(S.build_sigma(jnp.asarray(th), struct, t, r))
    if struct in ("fa", "rr"):
        # fa demarre avec de petits loadings, rr avec des loadings echelonnes :
        # la diagonale est de l'ordre de var, pas exactement var.
        assert np.all(np.diag(Sig) > 0)
    elif struct == "corh":
        # theta0 met le parametre de correlation a 0, ce qui vaut le MILIEU de
        # l'intervalle (-1/(t-1), 1), pas une correlation nulle : diagonale
        # exacte, correlation strictement dans la borne.
        assert np.allclose(np.diag(Sig), 2.5, atol=1e-10)
        r = Sig[0, 1] / 2.5
        assert -1.0 / (t - 1) < r < 1.0
        assert r == pytest.approx((1.0 - 1.0 / (t - 1)) / 2.0)
    else:
        assert np.allclose(Sig, 2.5 * np.eye(t), atol=1e-10)


def test_sigma_loadings_relit_lambda_et_psi():
    t, r = 4, 2
    th = RNG.normal(size=S.n_params("fa", t, r))
    lo = S.sigma_loadings(th, "fa", t, r)
    Lam = lo["Lambda"]
    assert Lam.shape == (t, r)
    assert np.allclose(np.triu(Lam, 1), 0.0)         # trapezoidale inferieure
    nl = _n_load(t, r)
    assert np.allclose(lo["psi"], np.exp(th[nl:nl + t]) ** 2)
    Sig = np.asarray(S.build_sigma(jnp.asarray(th), "fa", t, r))
    assert np.allclose(Sig, Lam @ Lam.T + np.diag(lo["psi"]), atol=1e-12)
    lo_rr = S.sigma_loadings(th[:nl], "rr", t, r)
    assert lo_rr["psi"] is None
    assert S.sigma_loadings(th, "us", t, r) is None


def test_us_diagonale_de_L_positive_et_ordre_ligne_par_ligne():
    th = jnp.array([np.log(1.2), 0.5, np.log(0.9)])
    L = np.asarray(S.chol_sigma(th, "us", 2))
    assert np.allclose(L, [[1.2, 0.0], [0.5, 0.9]])


def test_corh_correlation_uniforme_dans_la_borne_de_positivite():
    t = 4
    for x in (-8.0, 0.0, 8.0):
        th = jnp.array([0.0] * t + [x])
        Sig = np.asarray(S.build_sigma(th, "corh", t))
        r = Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])
        assert -1.0 / (t - 1) - 1e-9 <= r <= 1.0 + 1e-9
        assert np.allclose(Sig - np.diag(np.diag(Sig)),
                           r * (1 - np.eye(t)) * np.sqrt(np.outer(np.diag(Sig), np.diag(Sig))),
                           atol=1e-8)


def test_ante_est_l_inverse_d_une_bande():
    # Sigma^-1 = U D U' avec U unitriangulaire superieure a bande `rank`
    t, r = 4, 1
    th = RNG.normal(size=S.n_params("ante", t, r))
    Sig = np.asarray(S.build_sigma(jnp.asarray(th), "ante", t, r))
    P = np.linalg.inv(Sig)
    # une precision de bande 1 est TRIDIAGONALE
    assert np.allclose(P[np.abs(np.subtract.outer(range(t), range(t))) > 1], 0.0, atol=1e-8)


def test_chol_est_une_bande_sur_L():
    t, r = 5, 1
    th = RNG.normal(size=S.n_params("chol", t, r))
    L = np.asarray(S.chol_sigma(jnp.asarray(th), "chol", t, r))
    assert np.allclose(L[np.subtract.outer(range(t), range(t)) > 1], 0.0)


def test_structure_inconnue_refusee_partout():
    with pytest.raises(ValueError):
        S.n_params("toto", 3)
    with pytest.raises(ValueError):
        S.chol_sigma(jnp.zeros(3), "toto", 3)
    with pytest.raises(ValueError):
        S.theta0("toto", 3)
