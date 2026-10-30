"""Assembly of V and the REML likelihood.

MODEL
    y = X beta + sum_k Z_k u_k + e
    u_k ~ N(0, Sigma_k (x) K_k)          Sigma_k: t_k x t_k, K_k: q_k x q_k
    e   ~ N(0, R)

    V = sum_k Z_k (Sigma_k (x) K_k) Z_k' + R

ASSEMBLY BY FACTORS
    Sigma (x) K is never assembled. B_k = Z_k (L_Sigma (x) L_K) is formed and
    V += B_k B_k'. V is then positive semi-definite by construction, even when
    a variance goes to zero; one GEMM replaces an explicit Kronecker product of
    size (t q)^2; and the gradient follows the same path, with no formula
    derived by hand.

    All terms enter in one product, Sum_k B_k B_k' = [B_1 ... B_K][B_1 ... B_K]',
    so there is a single n x n intermediate (a chain V = V + B_k B_k' keeps one
    n x n buffer per term; dev/notes.md). Automatic differentiation of the
    single product is already optimal.

LIKELIHOOD
    -2 logL_REML = log|V| + log|X'V^-1 X| + y' P y  (+ constant)
    reml_from_V (core.py) carries the analytic gradient with respect to V,
    d(-2logL)/dV = P - (Py)(Py)', whatever the way V was built: any smooth
    parametrisation of V is differentiated for free.

Related modules: layout.py (layout of theta), incidence.py (Z and X),
residual.py (R).
"""
from . import _x64  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from .core import reml_from_V, ml_from_V
from .lag import (prepare_lag, ystar as lag_ystar, lag_terms,
                  PENALITE as LAG_PENALITE)
from .structures import chol_sigma, n_params
from .levels import level_chol
from .layout import (level_kind, term_n_params, res_sections, sec_n_params,  # noqa: F401
                     res_n_params, n_theta, split_theta)
from .incidence import (dense_Z, term_support, restrict_Z, support_groups,  # noqa: F401
                        orthonormal_X, prefold_fixed_K)
from .residual import _section_S, _section_V, section_matvec, residual_V  # noqa: F401


def term_factor(theta_k, term, Z):
    """B_k = Z_k (L_Sigma (x) L_K), such that Z_k (Sigma (x) K) Z_k' = B_k B_k'.
    L_K may depend on parameters (AR1 structure); it is then rebuilt at each
    evaluation and differentiated like the rest."""
    t, q = term["t"], term["q"]
    n_sig = n_params(term["struct"], t, term["rank"])
    Ls = chol_sigma(theta_k[:n_sig], term["struct"], t, term["rank"])   # t x m
    m = Ls.shape[1]
    kind = level_kind(term)
    if kind == "fixed":
        LK = term["LK"]
    else:
        LK = level_chol(theta_k[n_sig:], kind, q, dims=term.get("dims"),
                        LK_fixed=term.get("LK"), order=term.get("lvl_order", 0),
                        coord=term.get("coord"), parts=term.get("lvl_parts"),
                        opts=term.get("lvl_opts"), expr=term.get("lvl_expr"))
    if LK is None:
        # K = I: (L_Sigma (x) I) acts block by block, no product to form.
        Zb = Z.reshape(Z.shape[0], t, q)                 # (n, trait, level)
        B = jnp.einsum("ntq,tm->nmq", Zb, Ls).reshape(Z.shape[0], m * q)
    else:
        LKj = jnp.asarray(LK)
        Zb = Z.reshape(Z.shape[0], t, q)
        ZL = jnp.einsum("ntq,qp->ntp", Zb, LKj)          # Z (I (x) L_K)
        B = jnp.einsum("ntp,tm->nmp", ZL, Ls).reshape(Z.shape[0], m * q)
    return B


def assemble_V(theta, terms, Zs, res, n):
    """V = R + [B_1 ... B_K][B_1 ... B_K]' with full-height incidences Zs."""
    th_terms, th_res, _ = split_theta(theta, terms, res)
    R = residual_V(th_res, res, n)
    if not terms:
        return R
    B = jnp.concatenate([term_factor(th_terms[k], tm, Zs[k])
                         for k, tm in enumerate(terms)], axis=1)
    return R + B @ B.T


def assemble_V_by_support(theta, terms, groups, Zr, res, n):
    """V assembled by sub-blocks, each term on its own support (groups from
    incidence.support_groups, Zr the restricted incidences).

    Restricted factors store no zeros, and the residual sections are scattered
    directly into V, without a dense n x n residual. V is the same matrix, but
    the order of the sums changes and floating-point addition is not
    associative: compare with relative tolerances, never exact equality."""
    th_terms, th_res, _ = split_theta(theta, terms, res)
    V = jnp.zeros((n, n))
    o = 0
    for sec in res_sections(res, n):
        pnb = sec_n_params(sec)
        idx = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        V = V.at[idx[:, None], idx[None, :]].add(_section_V(th_res[o:o + pnb], sec))
        o += pnb
    for rows, ks in groups:
        B = jnp.concatenate([term_factor(th_terms[k], terms[k], Zr[k]) for k in ks],
                            axis=1)
        idx = jnp.asarray(rows)
        V = V.at[idx[:, None], idx[None, :]].add(B @ B.T)
    return V


def neg2_reml(theta, terms, Zs, res, y, X):
    V = assemble_V(theta, terms, Zs, res, y.shape[0])
    return reml_from_V(V, y, X)


def neg2_reml_by_support(theta, terms, groups, Zr, res, y, X):
    V = assemble_V_by_support(theta, terms, groups, Zr, res, y.shape[0])
    return reml_from_V(V, y, X)


def aliased_entries(terms, n, n_vec=3, seed=12345, tol=1e-9):
    """Entries of a `us` Sigma that are not identified separately.

    The contribution of a term to V is sum_{a>=b} Sigma[a,b] P_ab with
    P_ab = Z_a K Z_b' + Z_b K Z_a' (a != b). Two entries with equal patterns P
    enter V only through their sum: only the sum is identified, and the
    likelihood is flat along their difference. This is the case of the plant
    term [I, W] of VP+ as soon as W is symmetric (dev/notes.md).

    Patterns are compared by their action on a few random vectors (P_ab R, cost
    O(nnz)): two different patterns agree on three Gaussian vectors with
    probability zero. Computed once, outside the differentiated function. Only
    `us` structures with level kind id or fixed are examined: their entries
    are free, so a common pattern is exactly a flat direction.

    Returns a list of groups dict(term, entries = [(i, j), ...] 0-based,
    i >= j), each group losing len(entries) - 1 dimensions."""
    import scipy.sparse as sp_
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n, n_vec))
    out = []
    for tm in terms:
        if tm["struct"] != "us" or tm["t"] < 2:
            continue
        if level_kind(tm) not in ("id", "fixed"):
            continue
        t, q = tm["t"], tm["q"]
        Z = sp_.csr_matrix((tm["zx"], (tm["zi"], tm["zj"])), shape=(n, t * q))
        blocks = [Z[:, a * q:(a + 1) * q] for a in range(t)]
        LK = tm.get("LK")
        # Z_b' R, then K, once per column block.
        ZtR = [np.asarray(B.T @ R) for B in blocks]
        if LK is not None:
            LK = np.asarray(LK)
            ZtR = [LK @ (LK.T @ v) for v in ZtR]
        sig = {}
        for a in range(t):
            for b in range(a):
                v = np.asarray(blocks[a] @ ZtR[b]) + np.asarray(blocks[b] @ ZtR[a])
                nrm = float(np.linalg.norm(v))
                if nrm == 0.0:
                    continue
                sig[(a, b)] = (v, nrm)
        keys = list(sig)
        seen = set()
        for k, c in enumerate(keys):
            if c in seen:
                continue
            grp = [c]
            for d in keys[k + 1:]:
                if d in seen:
                    continue
                (v1, n1), (v2, n2) = sig[c], sig[d]
                if np.linalg.norm(v1 - v2) <= tol * max(n1, n2):
                    grp.append(d)
                    seen.add(d)
            if len(grp) > 1:
                out.append(dict(term=tm["name"], entries=grp))
    return out


def neg2_structural(theta, terms, groups, Zr, res, y, X, lag, X_lag=None):
    """-2logL of the structural model: the criterion of lag.py on
    y* = (I - M) y. Returns (value, admissible); without a lag,
    neg2_reml_by_support and True."""
    n = y.shape[0]
    if lag is None:
        return neg2_reml_by_support(theta, terms, groups, Zr, res, y, X), jnp.asarray(True)
    p_cov = n_theta(terms, res)
    th_lag = theta[p_cov:]
    ys = lag_ystar(th_lag, lag, y)
    # The mean term of the lag uses the original X; the REML part accepts the
    # orthonormal X, the constant being added by the caller.
    ldA, ldm, ok = lag_terms(th_lag, lag, X if X_lag is None else X_lag)
    V = assemble_V_by_support(theta, terms, groups, Zr, res, n)
    base = (ml_from_V if lag["criterion"] == "ml" else reml_from_V)(V, ys, X)
    # -2 log|det A| (change of variable) and, for reml_y, -2 x the mean term
    # (lag.py); zero for jacobian and ml.
    val = base - 2.0 * ldA - 2.0 * ldm
    return jnp.where(ok, val, LAG_PENALITE), ok


def make_neg2(bundle_terms, res, y, X, lag=None):
    """-2 logL (restricted, constant included) as a pure JAX function of theta.

    Returns (f, Zr, Zs): f(theta, Zr) is traceable, differentiable and
    vectorisable (vmap), unscaled. It is the objective of make_objective before
    its division by n, and the energy of the Bayesian sampler (bayes.py).

    The restricted incidences Zr are arguments of f, not captured: captured,
    they would be program constants that XLA tries to fold at compilation,
    which made the compilation of a large ar1 x ar1 field fail. Zs (full
    height) is kept for the identity tests against the full-height form."""
    n = y.shape[0]
    # Orthonormal X for the likelihood only (orthonormal_X). With a lag, its
    # mean term keeps the original X (X_lag), and the ML criterion has no
    # constant: it depends on X only through its span.
    Xq, ldRx = orthonormal_X(X)
    if lag is not None and lag.get("criterion") == "ml":
        ldRx = 0.0
    yj, Xj = jnp.asarray(y), jnp.asarray(Xq)
    Xl = jnp.asarray(np.asarray(X, dtype=np.float64))
    groups = support_groups(bundle_terms, n)
    Zr = [None] * len(bundle_terms)
    for rows, ks in groups:
        for k in ks:
            Zr[k] = restrict_Z(bundle_terms[k], rows, n)
    Zs = [dense_Z(t, n) for t in bundle_terms]
    terms_v, Zr = prefold_fixed_K(bundle_terms, Zr)
    lag = prepare_lag(lag, y, X)

    def f(theta, Zr_a):
        val, ok = neg2_structural(jnp.asarray(theta), terms_v, groups, Zr_a,
                                  res, yj, Xj, lag, X_lag=Xl)
        # The constant of X is added to a likelihood only: outside the
        # admissible region of the lag, the penalty stays exactly PENALITE.
        return jnp.where(ok, val + ldRx, val)

    return f, Zr, Zs


def make_objective(bundle_terms, res, y, X, scale=None, lag=None):
    """(fun_jac_scaled, fun_jac, sc, Zs) on theta.

    `lag` (dictionary of lag.py, prepared or not) adds the spatial lag: the
    objective becomes the structural criterion and theta ends with the free
    entries of Psi.

    Scaling by n: -2logL and its gradient are of order n, and L-BFGS-B starts
    with an identity Hessian approximation, so its first step is -gradient.
    Unscaled, every parameter would land on the bounds at the second
    evaluation and V would become singular (dev/notes.md). The optimiser sees
    fun_jac_scaled; fun_jac returns -2logL on its natural scale for the
    diagnostics and outputs."""
    n = y.shape[0]
    sc = float(n) if scale is None else float(scale)
    f_raw, Zr, Zs = make_neg2(bundle_terms, res, y, X, lag=lag)

    def f(theta, Zr_a):
        return f_raw(theta, Zr_a) / sc

    fg = jax.jit(jax.value_and_grad(f))

    def fun_jac_scaled(theta):
        v, g = fg(jnp.asarray(theta), Zr)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc

    return fun_jac_scaled, fun_jac, sc, Zs
