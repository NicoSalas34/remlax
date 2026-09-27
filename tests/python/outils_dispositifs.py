"""Petits dispositifs simules et calculs de reference INDEPENDANTS du solveur.

Ce module n'est pas un fichier de test (pas de prefixe test_) : il est importe
par les suites test_*.py de ce repertoire. Tout ce qu'il calcule est ecrit en
numpy pur, sans passer par remlax, pour que la comparaison ait un sens.

Conventions rappelees ici parce qu'elles sont load-bearing :
  - Z : colonne = a * q + niveau (le NIVEAU varie le plus vite), triplet COO
    0-based ;
  - theta d'un terme : parametres de Sigma PUIS parametres entre niveaux ;
  - -2 logL REML = (n - p) log(2 pi) + log|V| + log|X'V^-1 X| + y'Py.
"""
import os
import sys

import numpy as np

_ICI = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_ICI, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

PY = sys.executable
RACINE = os.path.normpath(os.path.join(_ICI, "..", ".."))


# ==============================================================================
# Construction de termes
# ==============================================================================
def terme_facteur(name, codes, q, t=1, struct="iid", rank=0, LK=None, poids=None,
                  **extra):
    """Terme a incidence INDICATRICE (ou ponderee par `poids`) sur `codes`.

    codes : entiers 0..q-1, un par observation. Avec t > 1, la meme incidence
    est repetee pour chaque colonne de Sigma (les poids peuvent alors etre un
    tableau n x t).
    """
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    zi = np.tile(np.arange(n, dtype=np.int64), t)
    zj = np.concatenate([a * q + codes for a in range(t)]).astype(np.int64)
    if poids is None:
        zx = np.ones(n * t)
    else:
        zx = np.asarray(poids, dtype=np.float64).reshape(n, t, order="F").ravel(order="F")
    d = dict(name=name, struct=struct, t=int(t), rank=int(rank), q=int(q),
             zi=zi, zj=zj, zx=zx, LK=LK)
    d.update(extra)
    return d


def terme_long(name, codes, trait, q, t, struct="us", rank=0, LK=None, **extra):
    """Terme multi-caractere en FORMAT LONG : la ligne i (caractere trait[i])
    n'entre que dans la colonne de Sigma de son caractere."""
    codes = np.asarray(codes, dtype=np.int64)
    trait = np.asarray(trait, dtype=np.int64)
    n = len(codes)
    d = dict(name=name, struct=struct, t=int(t), rank=int(rank), q=int(q),
             zi=np.arange(n, dtype=np.int64), zj=(trait * q + codes).astype(np.int64),
             zx=np.ones(n), LK=LK)
    d.update(extra)
    return d


def residuelle_iid(n):
    return dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64),
                unit=np.arange(n, dtype=np.int64), n_unit=int(n))


def residuelle(struct, trait, unit, t, rank=0, lvl="id", **extra):
    d = dict(struct=struct, t=int(t), rank=int(rank),
             trait=np.asarray(trait, np.int64), unit=np.asarray(unit, np.int64),
             n_unit=int(np.max(unit)) + 1, lvl=lvl)
    d.update(extra)
    return d


# ==============================================================================
# References numpy
# ==============================================================================
def dense_Z_np(term, n):
    Z = np.zeros((n, term["t"] * term["q"]))
    np.add.at(Z, (term["zi"], term["zj"]), term["zx"])
    return Z


def neg2logL_reference(V, y, X):
    """-2 logL REML par Cholesky numpy, sans rien emprunter a remlax."""
    n, p = X.shape
    Lv = np.linalg.cholesky(V)
    logdetV = 2.0 * np.sum(np.log(np.diag(Lv)))
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    La = np.linalg.cholesky(A)
    logdetA = 2.0 * np.sum(np.log(np.diag(La)))
    beta = np.linalg.solve(A, X.T @ Vi @ y)
    r = y - X @ beta
    yPy = float(r @ Vi @ r)
    return (n - p) * np.log(2.0 * np.pi) + logdetV + logdetA + yPy


def ar1_corr_np(rho, q):
    i = np.arange(q)
    return rho ** np.abs(i[:, None] - i[None, :])


def toeplitz_np(acf):
    q = len(acf)
    i = np.arange(q)
    return np.asarray(acf)[np.abs(i[:, None] - i[None, :])]


def acf_ar_lyapunov(phi, q):
    """ACF d'un AR(p) par l'equation de Lyapunov discrete (forme d'etat).

    Independant de Yule-Walker : Gamma = F Gamma F' + e1 e1', puis
    rho_k = (F^k Gamma)[0, 0] / Gamma[0, 0].
    """
    from scipy.linalg import solve_discrete_lyapunov
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    F = np.zeros((p, p))
    F[0, :] = phi
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    Q = np.zeros((p, p)); Q[0, 0] = 1.0
    G = solve_discrete_lyapunov(F, Q)
    acf = np.empty(q)
    Fk = np.eye(p)
    for k in range(q):
        acf[k] = (Fk @ G)[0, 0] / G[0, 0]
        Fk = F @ Fk
    return acf


def acf_ma_psi(psi, q):
    """ACF d'un processus MA(infini) tronque de coefficients psi (psi_0 = 1)."""
    psi = np.asarray(psi, dtype=float)
    m = len(psi)
    g = np.array([np.sum(psi[:m - k] * psi[k:]) if k < m else 0.0 for k in range(q)])
    return g / g[0]


def levinson_np(pacf):
    """Coefficients AR depuis les correlations partielles (Durbin-Levinson)."""
    pacf = list(pacf)
    phi = []
    for k, pk in enumerate(pacf):
        if k == 0:
            phi = [pk]
        else:
            prev = phi[:]
            phi = [prev[j] - pk * prev[k - 1 - j] for j in range(k)] + [pk]
    return np.array(phi)
