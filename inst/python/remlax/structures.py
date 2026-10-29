"""Covariance structures: theta (free, unconstrained) -> Sigma (t x t).

The convention is shared with the R side (R/remlax.R). Any divergence here
silently breaks the mapping of parameters. The functions of this file are the
reference, and R must reproduce them exactly.

Every parametrisation is unconstrained. The optimiser works on R^p with no
bounds. Sigma is positive by construction, because the code parametrises a
Cholesky factor and never Sigma itself. L-BFGS-B can therefore run without
constraints, and Sigma stays positive definite even far from the optimum.

  iid     Sigma = exp(2*theta) * I_t                        1 parameter
  diag    Sigma = diag(exp(2*theta_j))                      t parameters
  us      Sigma = L L',  L lower triangular                 t(t+1)/2
          diagonal of L = exp(theta) (positive), off-diagonal free
  fa(r)   Sigma = Lambda Lambda' + diag(exp(2*psi))         nload(t,r) + t
          Lambda lower trapezoidal (identifiability: without the triangular
          constraint, Lambda is defined only up to a rotation)

The other structures (rr, chol, ante, corh) are described where chol_sigma
builds them.
"""
# turns on float64, including on a direct import
from . import _x64     # noqa: F401
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

STRUCTURES = ("iid", "diag", "us", "fa", "rr", "chol", "ante", "corh")


def sigma_loadings(theta, struct, t, rank):
    """Lambda and the specific variances of a factor structure.

    `chol_sigma` returns a factor, and the interface returns the assembled
    Sigma. Neither gives Lambda separately. Sigma = Lambda Lambda' + diag(psi)
    has no unique decomposition: without the trapezoidal constraint, Lambda is
    defined only up to a rotation. Lambda therefore cannot be recovered from
    Sigma. It is read from theta.

    The internal parametrisation holds psi = exp(theta), and
    Sigma = Lambda Lambda' + diag(psi^2). The output returns psi^2, the
    specific variance. This is the convention of the asreml loadings tables,
    where the diagonal of Sigma equals sum(V_k^2) + psi.

    Returns None for any structure without factors instead of raising. A
    caller looping over heterogeneous terms then need not know in advance
    which ones have factors.
    """
    if struct not in ("fa", "rr"):
        return None
    th = np.asarray(theta, dtype=float)
    r = int(rank)
    nl = n_loadings(t, r)
    lam = np.zeros((t, r))
    k = 0
    for i in range(t):
        for j in range(min(i + 1, r)):
            lam[i, j] = th[k]; k += 1
    # `rr` is a pure reduced rank: Sigma = Gamma Gamma', with no specific
    # variance. Zeros would wrongly read as a measurement.
    psi = np.exp(th[nl:nl + t]) ** 2 if struct == "fa" else None
    return {"Lambda": lam, "psi": psi, "rank": r, "struct": struct}


def n_loadings(t, r):
    """Number of loadings of a lower triangular t x r matrix."""
    return int(np.sum(np.minimum(np.arange(1, t + 1), r)))


def n_params(struct, t, rank=0):
    """Number of free parameters of a structure.

    The counts follow Appendix C of the ASReml-R manual:
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
        # The manual states k*omega. Gamma is constrained to be lower
        # trapezoidal, as for fa. Without this constraint Gamma is defined only
        # up to a rotation, the likelihood has a flat ridge of dimension
        # k(k-1)/2, and the Hessian is singular by as much. The fitted model is
        # the same, and this parameter count is the correct one.
        return n_loadings(t, rank)
    if struct in ("chol", "ante"):
        k = int(rank)
        return int((k + 1) * (t - k / 2.0))
    if struct == "corh":
        return t + 1
    raise ValueError("structure inconnue : %s" % struct)


def _band_idx(t, k):
    """Indices (i, j) of the strict lower band 1 <= i-j <= k, row by row."""
    ii, jj = [], []
    for i in range(t):
        for j in range(max(0, i - k), i):
            ii.append(i); jj.append(j)
    return np.array(ii, dtype=int), np.array(jj, dtype=int)


def _tril_indices(t):
    """Indices (i, j) of the lower triangle, in row-by-row order.

    The order maps each theta to a cell of L. It is fixed here row by row
    (increasing i, then j <= i), and R uses the same order.
    """
    ii, jj = [], []
    for i in range(t):
        for j in range(i + 1):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)


def chol_sigma(theta, struct, t, rank=0):
    """Factor L such that Sigma = L L' (t x t or t x k).

    The assembly of V uses the factor directly (V += (Z (L (x) L_K)) (.)').
    This avoids forming Sigma and factorising it again. It also keeps V
    positive even when a variance goes to zero.
    """
    if struct == "iid":
        return jnp.exp(theta[0]) * jnp.eye(t)
    if struct == "diag":
        return jnp.diag(jnp.exp(theta[:t]))
    if struct == "us":
        ii, jj = _tril_indices(t)
        L = jnp.zeros((t, t)).at[ii, jj].set(theta[: len(ii)])
        d = jnp.exp(jnp.diagonal(L))          # positive diagonal
        return L.at[jnp.arange(t), jnp.arange(t)].set(d)
    if struct == "fa":
        nl = n_loadings(t, rank)
        lam = jnp.zeros((t, rank))
        rows, cols = [], []
        for i in range(t):
            for j in range(min(i + 1, rank)):
                rows.append(i); cols.append(j)
        lam = lam.at[jnp.array(rows), jnp.array(cols)].set(theta[:nl])
        psi = jnp.exp(theta[nl:nl + t])
        # [Lambda | diag(psi)] is a factor of Lambda Lambda' + diag(psi^2)
        return jnp.concatenate([lam, jnp.diag(psi)], axis=1)
    if struct == "rr":
        # Reduced rank: Sigma = Gamma Gamma', singular by construction (rank
        # `rank`). The factor is Gamma itself. This is allowed because V is
        # assembled as B B'. A rank-deficient component leaves V nonsingular as
        # long as another term or the residual completes it.
        lam = jnp.zeros((t, rank))
        rows, cols = [], []
        for i in range(t):
            for j in range(min(i + 1, rank)):
                rows.append(i); cols.append(j)
        return lam.at[jnp.array(rows), jnp.array(cols)].set(theta[:len(rows)])
    if struct == "chol":
        # Sigma = L D L', with L unit lower triangular of band `rank`.
        # Factor: L D^(1/2).
        k = int(rank)
        ii, jj = _band_idx(t, k)
        L = jnp.eye(t)
        if len(ii):
            L = L.at[ii, jj].set(theta[:len(ii)])
        d = jnp.exp(theta[len(ii):len(ii) + t])
        return L * d[None, :]
    if struct == "ante":
        # Antedependence: Sigma^-1 = U D U', with U unit upper triangular of
        # band `rank`. Hence Sigma = (U')^-1 D^-1 U^-1, and a factor is
        # (U')^-1 D^(-1/2). (U')^-1 is computed by triangular substitution,
        # never by a general inversion.
        k = int(rank)
        ii, jj = _band_idx(t, k)          # band of U' (lower)
        Ut = jnp.eye(t)
        if len(ii):
            Ut = Ut.at[ii, jj].set(theta[:len(ii)])
        d = jnp.exp(theta[len(ii):len(ii) + t])
        A = jax.scipy.linalg.solve_triangular(Ut, jnp.eye(t), lower=True)
        return A / d[None, :]
    if struct == "corh":
        # Heterogeneous variances and a uniform correlation. The positivity
        # bound is r > -1/(t-1), the same as for `cor` between levels.
        sd = jnp.exp(theta[:t])
        lo = -1.0 / max(t - 1, 1)
        r = lo + (jnp.tanh(theta[t]) + 1.0) * 0.5 * (1.0 - lo)
        C = (1.0 - r) * jnp.eye(t) + r * jnp.ones((t, t))
        Lc = jnp.linalg.cholesky(C + 1e-12 * jnp.eye(t))
        return sd[:, None] * Lc
    raise ValueError("structure inconnue : %s" % struct)


def build_sigma(theta, struct, t, rank=0):
    """Sigma = L L' (t x t)."""
    L = chol_sigma(theta, struct, t, rank)
    return L @ L.T


def theta0(struct, t, rank=0, var=1.0):
    """Reasonable starting point: Sigma ~ var * I."""
    s = 0.5 * np.log(max(var, 1e-8))
    if struct == "iid":
        return np.array([s])
    if struct == "diag":
        return np.full(t, s)
    if struct == "us":
        ii, jj = _tril_indices(t)
        th = np.zeros(len(ii))
        th[ii == jj] = s                     # log diagonal, off-diagonal at 0
        return th
    if struct == "fa":
        nl = n_loadings(t, rank)
        return np.concatenate([np.full(nl, 0.1 * np.exp(s)), np.full(t, s)])
    if struct == "rr":
        # Equal loadings would give a Gamma of rank 1. They are staggered so
        # that the start has full rank.
        nl = n_loadings(t, rank)
        return np.linspace(0.6, 0.3, nl) * np.sqrt(max(var, 1e-8))
    if struct in ("chol", "ante"):
        k = int(rank); nb = len(_band_idx(t, k)[0])
        return np.concatenate([np.zeros(nb), np.full(t, s if struct == "chol" else -s)])
    if struct == "corh":
        return np.concatenate([np.full(t, s), np.zeros(1)])
    raise ValueError("structure inconnue : %s" % struct)
