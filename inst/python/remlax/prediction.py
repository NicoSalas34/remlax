"""Predictions L' beta (+ M' u) and their prediction error covariance.

R builds L (rx_predict(): one row per prediction, already averaged over the
factors absent from the classify) and M (the random part); only the algebra
lives here.
"""
from . import _x64  # noqa: F401

import jax.numpy as jnp
import numpy as np

from .layout import split_theta
from .incidence import dense_Z
from .model import assemble_V
from .effects import term_sigma_K


def predict(theta, terms, res, y, X, L, M=None, vbeta=None):
    """Predictions p = L' beta (+ M' u) and the covariance of p_hat - p.

    L: l x p. M: dict term name -> l x (t*q), the random part of each
    prediction. Without M the prediction carries only the fixed effects,
    right when the random term is averaged and wrong for a genotype mean,
    which contains its BLUP.

    The covariance is that of the prediction error:

        L' Phi L  -  2 sym( L' Phi X'V^-1 Z G M )  +  M'G M - M'G Z' P Z G M

    with Phi = (X'V^-1X)^-1 and P = V^-1 - V^-1 X Phi X'V^-1. The second term
    is negative: the correlation between beta-hat and u-hat reduces the error
    of the total. G = Sigma (x) K enters only through its action on M,
    Sigma M_mat K', without forming G or G^-1."""
    theta = np.asarray(theta, dtype=float)
    n, p = X.shape
    L = np.atleast_2d(np.asarray(L, dtype=float))
    if L.shape[1] != p:
        raise ValueError("predict : L a %d colonnes pour %d effets fixes."
                         % (L.shape[1], p))
    Zs = [dense_Z(t, n) for t in terms]
    V = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    try:
        Vi = np.linalg.solve(V, np.eye(n))
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    U = Vi @ X
    Phi = np.linalg.pinv(X.T @ U) if vbeta is None else np.asarray(vbeta)
    beta = Phi @ (X.T @ Vi @ y)
    Pm = Vi - U @ Phi @ U.T
    Py = Pm @ y
    val = L @ beta
    cov = L @ Phi @ L.T
    th_terms, _, _ = split_theta(jnp.asarray(theta), terms, res)
    for k, tm in enumerate(terms):
        name = tm["name"]
        if not M or M.get(name) is None:
            continue
        Mk = np.atleast_2d(np.asarray(M[name], dtype=float))       # l x (t*q)
        t_, q_ = tm["t"], tm["q"]
        if Mk.shape != (L.shape[0], t_ * q_):
            raise ValueError("predict : M['%s'] est %s, attendu (%d, %d)."
                             % (name, Mk.shape, L.shape[0], t_ * q_))
        Sig, Kq = term_sigma_K(tm, th_terms[k])
        Mb = Mk.reshape(Mk.shape[0], t_, q_)
        GM = np.einsum("ab,lbq,qr->lar", Sig, Mb, Kq).reshape(Mk.shape[0], t_ * q_)
        Zd = np.asarray(Zs[k])
        ZGM = GM @ Zd.T                                            # l x n
        val = val + GM @ (Zd.T @ Py)                               # + M' u_hat
        A = L @ Phi @ (X.T @ Vi) @ ZGM.T                           # l x l
        cov = cov - A - A.T + Mk @ GM.T - ZGM @ Pm @ ZGM.T
    return dict(value=[float(v) for v in np.asarray(val).ravel()],
                se=[float(np.sqrt(max(v, 0))) for v in np.diag(cov)],
                cov=np.asarray(cov), beta=[float(b) for b in beta])
