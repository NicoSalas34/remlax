"""Structures de covariance : theta (libre, non contraint) -> Sigma (t x t).

CONVENTION UNIQUE, PARTAGEE AVEC LE COTE R (R/remlax.R). Toute divergence
ici casse silencieusement la correspondance des parametres : les fonctions de
ce fichier sont donc la reference, et R doit les reproduire a l'identique.

Toutes les parametrisations sont NON CONTRAINTES : l'optimiseur travaille sur
R^p sans bornes, et la positivite de Sigma est garantie par construction (on
parametre un facteur de Cholesky, jamais Sigma directement). C'est ce qui
permet d'utiliser L-BFGS-B sans contrainte et de ne jamais rendre un Sigma
non defini positif, meme loin de l'optimum.

  iid     Sigma = exp(2*theta) * I_t                        1 parametre
  diag    Sigma = diag(exp(2*theta_j))                      t parametres
  us      Sigma = L L',  L triangulaire inferieure          t(t+1)/2
          diagonale de L = exp(theta) (positive), hors-diagonale libre
  fa(r)   Sigma = Lambda Lambda' + diag(exp(2*psi))         nload(t,r) + t
          Lambda trapezoidale inferieure (identifiabilite : sans la
          contrainte triangulaire, Lambda n'est definie qu'a une rotation pres)
  fixed   Sigma = valeur fournie, 0 parametre (pour tester a Sigma impose)
"""
try:                       # active float64, y compris en import direct
    from . import _x64     # noqa: F401
except ImportError:
    import _x64            # noqa: F401
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

STRUCTURES = ("iid", "diag", "us", "fa", "rr", "chol", "ante", "corh", "fixed")


def n_loadings(t, r):
    """Nombre de loadings d'une matrice t x r triangulaire inferieure."""
    return int(np.sum(np.minimum(np.arange(1, t + 1), r)))


def n_params(struct, t, rank=0):
    """Nombre de parametres libres d'une structure.

    Les decomptes suivent l'annexe C du manuel ASReml-R :
      diag = idh    omega
      us = corgh    omega(omega+1)/2
      fa(k)         k*omega + omega
      rr(k)         k*omega
      chol(k)       (k+1)(omega - k/2)
      ante(k)       (k+1)(omega - k/2)
      corh          omega + 1
    """
    t = int(t)
    if struct == "iid":
        return 1
    if struct == "diag":
        return t
    if struct == "us":
        return t * (t + 1) // 2
    if struct == "fa":
        return n_loadings(t, rank) + t
    if struct == "rr":
        # Le manuel annonce k*omega. On contraint Gamma a etre TRAPEZOIDALE
        # INFERIEURE, comme pour fa : sans cette contrainte Gamma n'est definie
        # qu'a une rotation pres, la vraisemblance a une crete plate de
        # dimension k(k-1)/2, et le Hessien est singulier d'autant. Le modele
        # ajuste est le meme ; le decompte de parametres, lui, est le bon.
        return n_loadings(t, rank)
    if struct in ("chol", "ante"):
        k = int(rank)
        return int((k + 1) * (t - k / 2.0))
    if struct == "corh":
        return t + 1
    if struct == "fixed":
        return 0
    raise ValueError("structure inconnue : %s" % struct)


def _band_idx(t, k):
    """Indices (i, j) de la bande inferieure stricte 1 <= i-j <= k, ligne par ligne."""
    ii, jj = [], []
    for i in range(t):
        for j in range(max(0, i - k), i):
            ii.append(i); jj.append(j)
    return np.array(ii, dtype=int), np.array(jj, dtype=int)


def _tril_indices(t):
    """Indices (i, j) du triangle inferieur, EN ORDRE LIGNE PAR LIGNE.

    L'ordre importe : c'est lui qui associe chaque theta a une case de L. Il est
    fixe ici, ligne par ligne (i croissant, puis j <= i), et R fait pareil.
    """
    ii, jj = [], []
    for i in range(t):
        for j in range(i + 1):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)


def chol_sigma(theta, struct, t, rank=0, fixed=None):
    """Facteur L tel que Sigma = L L' (t x t ou t x k).

    On rend le FACTEUR et non Sigma : l'assemblage de V s'en sert directement
    (V += (Z (L (x) L_K)) (.)'), ce qui evite de former Sigma puis de le
    refactoriser, et garantit la positivite meme quand une variance part a zero.
    """
    if struct == "iid":
        return jnp.exp(theta[0]) * jnp.eye(t)
    if struct == "diag":
        return jnp.diag(jnp.exp(theta[:t]))
    if struct == "us":
        ii, jj = _tril_indices(t)
        L = jnp.zeros((t, t)).at[ii, jj].set(theta[: len(ii)])
        d = jnp.exp(jnp.diagonal(L))          # diagonale positive
        return L.at[jnp.arange(t), jnp.arange(t)].set(d)
    if struct == "fa":
        nl = n_loadings(t, rank)
        lam = jnp.zeros((t, rank))
        idx = 0
        rows, cols = [], []
        for i in range(t):
            for j in range(min(i + 1, rank)):
                rows.append(i); cols.append(j)
        lam = lam.at[jnp.array(rows), jnp.array(cols)].set(theta[:nl])
        psi = jnp.exp(theta[nl:nl + t])
        # [Lambda | diag(psi)] est un facteur de Lambda Lambda' + diag(psi^2)
        return jnp.concatenate([lam, jnp.diag(psi)], axis=1)
    if struct == "rr":
        # Rang reduit : Sigma = Gamma Gamma', SINGULIERE par construction (rang
        # `rank`). Le facteur est Gamma lui-meme. C'est licite ici parce que V
        # est assemblee en B B' : une composante de rang deficient ne rend pas V
        # singuliere tant qu'un autre terme ou la residuelle la complete.
        lam = jnp.zeros((t, rank))
        rows, cols = [], []
        for i in range(t):
            for j in range(min(i + 1, rank)):
                rows.append(i); cols.append(j)
        return lam.at[jnp.array(rows), jnp.array(cols)].set(theta[:len(rows)])
    if struct == "chol":
        # Sigma = L D L', L unitriangulaire inferieure a bande `rank`.
        # Facteur : L D^(1/2).
        k = int(rank)
        ii, jj = _band_idx(t, k)
        L = jnp.eye(t)
        if len(ii):
            L = L.at[ii, jj].set(theta[:len(ii)])
        d = jnp.exp(theta[len(ii):len(ii) + t])
        return L * d[None, :]
    if struct == "ante":
        # Antedependance : Sigma^-1 = U D U', U unitriangulaire SUPERIEURE a
        # bande `rank`. Donc Sigma = (U')^-1 D^-1 U^-1, et un facteur est
        # (U')^-1 D^(-1/2). (U')^-1 se calcule par substitution triangulaire,
        # jamais par une inversion generale.
        k = int(rank)
        ii, jj = _band_idx(t, k)          # bande de U' (inferieure)
        Ut = jnp.eye(t)
        if len(ii):
            Ut = Ut.at[ii, jj].set(theta[:len(ii)])
        d = jnp.exp(theta[len(ii):len(ii) + t])
        A = jax.scipy.linalg.solve_triangular(Ut, jnp.eye(t), lower=True)
        return A / d[None, :]
    if struct == "corh":
        # Variances heterogenes + correlation UNIFORME. La borne de positivite
        # est theta > -1/(t-1), la meme que pour `cor` entre niveaux.
        sd = jnp.exp(theta[:t])
        lo = -1.0 / max(t - 1, 1)
        r = lo + (jnp.tanh(theta[t]) + 1.0) * 0.5 * (1.0 - lo)
        C = (1.0 - r) * jnp.eye(t) + r * jnp.ones((t, t))
        Lc = jnp.linalg.cholesky(C + 1e-12 * jnp.eye(t))
        return sd[:, None] * Lc
    if struct == "fixed":
        S = jnp.asarray(fixed)
        return jnp.linalg.cholesky(S + 1e-12 * jnp.eye(t))
    raise ValueError("structure inconnue : %s" % struct)


def build_sigma(theta, struct, t, rank=0, fixed=None):
    """Sigma = L L' (t x t)."""
    L = chol_sigma(theta, struct, t, rank, fixed)
    return L @ L.T


def theta0(struct, t, rank=0, var=1.0):
    """Point de depart raisonnable : Sigma ~ var * I."""
    s = 0.5 * np.log(max(var, 1e-8))
    if struct == "iid":
        return np.array([s])
    if struct == "diag":
        return np.full(t, s)
    if struct == "us":
        ii, jj = _tril_indices(t)
        th = np.zeros(len(ii))
        th[ii == jj] = s                     # diagonale en log, hors-diag a 0
        return th
    if struct == "fa":
        nl = n_loadings(t, rank)
        return np.concatenate([np.full(nl, 0.1 * np.exp(s)), np.full(t, s)])
    if struct == "rr":
        # Loadings tous egaux donneraient un Gamma de rang 1 : on echelonne pour
        # partir d'un point de rang plein.
        nl = n_loadings(t, rank)
        return np.linspace(0.6, 0.3, nl) * np.sqrt(max(var, 1e-8))
    if struct in ("chol", "ante"):
        k = int(rank); nb = len(_band_idx(t, k)[0])
        return np.concatenate([np.zeros(nb), np.full(t, s if struct == "chol" else -s)])
    if struct == "corh":
        return np.concatenate([np.full(t, s), np.zeros(1)])
    if struct == "fixed":
        return np.zeros(0)
    raise ValueError("structure inconnue : %s" % struct)
