"""The residual covariance R.

R is the direct sum of its sections (asreml's dsum). Each section is
Sigma_trait (t x t) times a correlation between units C_unit, entry by entry:

    R[i,j] = Sigma_t[trait(i), trait(j)] * C_u[unit(i), unit(j)]

One formula covers every residual = ~ ... of asreml:
  ~ units                          Sigma = sigma^2,  C = I
  ~ us(trait):units                Sigma = us,       C = I (traits of a unit coupled)
  ~ ar1(row):ar1(col)              Sigma = sigma^2,  C = AR1 x AR1
  ~ us(trait):ar1(row):ar1(col)    both
"""
from . import _x64  # noqa: F401

import jax.numpy as jnp
import numpy as np

from .layout import res_sections, sec_n_params
from .structures import chol_sigma, n_params
from .levels import level_corr, _acf, _toeplitz


def _section_S(theta_s, sec):
    """Sigma_trait (t x t) of a residual section; shared with the sparse
    residual of the low-memory mode (econome.py) and the blocks engine."""
    st, t = sec["struct"], sec["t"]
    n_sig = n_params(st, t, sec["rank"])
    if st == "iid":
        return jnp.exp(2.0 * theta_s[0]) * jnp.eye(t)
    if st == "diag":
        return jnp.diag(jnp.exp(2.0 * theta_s[:t]))
    if st in ("us", "fa"):
        Ls = chol_sigma(theta_s[:n_sig], st, t, sec["rank"])
        return Ls @ Ls.T
    raise ValueError("unknown residual structure: %s" % st)


def _section_V(theta_s, sec):
    """R of one section, Sigma_trait x C_unit entry by entry.

    With a single trait, Sigma is a scalar and the n x n matrix S[tr, tr] is
    not formed. When the units are already in the order 0..n_u-1, C is used as
    is, without the gather C[un, un]. Without structure between units, two
    observations are correlated only on the same unit (different traits)."""
    st, t = sec["struct"], sec["t"]
    kind = sec.get("lvl", "id")
    tr = jnp.asarray(sec["trait"])
    un = jnp.asarray(sec["unit"])
    n_sig = n_params(st, t, sec["rank"])
    S = _section_S(theta_s, sec)
    un_np = np.asarray(sec["unit"], dtype=np.int64)
    if kind == "id":
        Cu = (un[:, None] == un[None, :]).astype(S.dtype)
    else:
        n_u = int(sec.get("n_unit", 0)) or int(np.max(un_np) + 1)
        C = level_corr(theta_s[n_sig:], kind, n_u, order=sec.get("lvl_order", 0),
                       coord=sec.get("coord"), dims=sec.get("dims"),
                       opts=sec.get("lvl_opts"), expr=sec.get("lvl_expr"))
        if un_np.shape[0] == n_u and np.array_equal(un_np, np.arange(n_u)):
            Cu = C
        else:
            Cu = C[un[:, None], un[None, :]]
    if t == 1:
        return S[0, 0] * Cu
    return S[tr[:, None], tr[None, :]] * Cu


def _toeplitz_ar1(th, q):
    """AR1 correlation (q x q), same parametrisation as level_corr("ar1ar1")."""
    return _toeplitz(_acf(th, "ar1", q), q)


def section_matvec(theta_s, sec, vs):
    """R_s vs without forming R_s when the structure allows it:

        (R v)_i = sum_b Sigma[tr_i, b] (C V_b)[un_i],  V_b[u] = v of unit u, trait b

    C = I and C = AR1 (x) AR1 need no n x n matrix:
    (A_r (x) A_c) v = vec(A_r M A_c'), M the rows x columns grid. Other
    structures fall back to the dense section. Used for the columns of the AI
    matrix (ai.py)."""
    st, t = sec["struct"], sec["t"]
    kind = sec.get("lvl", "id")
    if kind not in ("id", "ar1ar1") or st not in ("iid", "diag", "us", "fa"):
        return _section_V(theta_s, sec) @ vs
    n_sig = n_params(st, t, sec["rank"])
    if st == "iid":
        S = jnp.exp(2.0 * theta_s[0]) * jnp.eye(t)
    elif st == "diag":
        S = jnp.diag(jnp.exp(2.0 * theta_s[:t]))
    else:
        Ls = chol_sigma(theta_s[:n_sig], st, t, sec["rank"])
        S = Ls @ Ls.T
    tr = np.asarray(sec["trait"], dtype=np.int64)
    un = np.asarray(sec["unit"], dtype=np.int64)
    if kind == "id":
        _, un_c = np.unique(un, return_inverse=True)
        n_u = int(un_c.max()) + 1
    else:
        un_c = un
        nr, nc = int(sec["dims"][0]), int(sec["dims"][1])
        n_u = nr * nc
    Vm = jnp.zeros((n_u, t)).at[un_c, tr].add(vs)
    if kind == "ar1ar1":
        th_l = theta_s[n_sig:]
        Cr = _toeplitz_ar1(th_l[:1], nr)
        Cc = _toeplitz_ar1(th_l[1:2], nc)
        Mb = Vm.T.reshape(t, nr, nc)
        Vm = jnp.einsum("rs,bsc,dc->brd", Cr, Mb, Cc).reshape(t, n_u).T
    return (Vm @ S.T)[un_c, tr]


def residual_V(theta_r, res, n):
    """Full R, the direct sum of the sections. The sections partition the
    observations, each with its own parameters. R is assembled by scattering
    into an n x n matrix: the rows of a section need not be contiguous in the
    data."""
    secs = res_sections(res, n)
    if not secs:
        return jnp.zeros((n, n))
    if len(secs) == 1 and "rows" not in res and res.get("sections") is None:
        return _section_V(theta_r, secs[0])
    R = jnp.zeros((n, n))
    o = 0
    for sec in secs:
        p = sec_n_params(sec)
        idx = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        R = R.at[idx[:, None], idx[None, :]].add(_section_V(theta_r[o:o + p], sec))
        o += p
    return R
