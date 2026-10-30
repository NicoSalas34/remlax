"""Tests on the fixed effects: conditional Wald tests and the Kenward-Roger
adjustment.

wald() treats V as known: var(beta) = (X' V^-1 X)^-1, and each term is tested
given all the others (asreml's conditional test, type III F), with chi2/rank
p-values. kenward_roger() corrects the variance of beta for the estimation of
V and gives denominator degrees of freedom.
"""
from . import _x64  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from .components import n_level
from .layout import res_sections
from .incidence import dense_Z
from .model import assemble_V


def _column_groups(fixed_terms, p):
    """The tested groups of columns: the fixed terms, or one per column."""
    return fixed_terms or [("column %d" % (j + 1), [j]) for j in range(p)]


def wald(y, X, V, fixed_terms=None):
    """Conditional Wald tests at fixed V.

      var(beta) = (X' V^-1 X)^-1
      for a group of columns L: F = (L beta)' [L var(beta) L']^-1 (L beta) / rank

    The denominator degrees of freedom are not adjusted: the p-values are
    those of a chi2/rank, valid asymptotically, not exact small-sample F."""
    from scipy.stats import chi2
    n, p = X.shape
    Vi = np.linalg.pinv(V)
    A = X.T @ Vi @ X
    Ai = np.linalg.pinv(A)
    beta = Ai @ (X.T @ Vi @ y)
    out = []
    for name, cols in _column_groups(fixed_terms, p):
        cols = list(cols)
        L = np.zeros((len(cols), p))
        for r, c in enumerate(cols):
            L[r, c] = 1.0
        Lb = L @ beta
        M = L @ Ai @ L.T
        rg = np.linalg.matrix_rank(M)
        if rg == 0:
            out.append(dict(term=name, df=0, F=np.nan, p=np.nan)); continue
        stat = float(Lb @ np.linalg.pinv(M) @ Lb)
        out.append(dict(term=name, df=int(rg), F=stat / rg,
                        chi2=stat, p=float(chi2.sf(stat, rg))))
    return dict(beta=[float(b) for b in beta],
                se_beta=[float(np.sqrt(max(v, 0))) for v in np.diag(Ai)],
                tests=out)


# ------------------------------------------------------------------------------
# Kenward-Roger
# ------------------------------------------------------------------------------

def _expected_reml_info(D, P_mat):
    """Expected REML information I_ij = 1/2 tr(P dV_i P dV_j),
    P = V^-1 - V^-1 X Phi X' V^-1. Kenward and Roger use this matrix, not the
    observed Hessian; the two differ at finite sample size (dev/notes.md)."""
    k = len(D)
    M = [P_mat @ Di for Di in D]
    I = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            I[i, j] = I[j, i] = 0.5 * float(np.sum(M[i] * M[j].T))
    return I


def kenward_roger(theta, terms, res, y, X, fixed_terms=None):
    """Kenward & Roger (1997) adjustment: variance of beta and denominator
    degrees of freedom.

    The ordinary Wald test treats V as known; it is estimated, so the variance
    of beta-hat is underestimated and the statistic too large, which matters
    on trials with few blocks or sites. Two separate corrections:

      Phi_A, the adjusted covariance of beta,
          Phi_A = Phi + 2 Phi [ sum_ij W_ij (Q_ij - P_i Phi P_j) ] Phi
      with Phi = (X'V^-1X)^-1, U = V^-1 X, P_i = U' dV_i U,
      Q_ij = (dV_i U)' V^-1 (dV_j U), W the inverse of the expected REML
      information;

      a scale factor lambda and denominator df m matching the first two
      moments of lambda F* to those of an F(l, m). The moments involve
      dPhi/dtheta_i = Phi P_i Phi, not P_i.

    The second-order term -R_ij/4 (R_ij = U' d2V/dsigma_i dsigma_j U) is
    omitted, as in pbkrtest and SAS: it vanishes when V is linear in the
    variance parameters, true of every structure here except correlation
    parameters between levels, and computed in the internal parametrisation it
    is not invariant (dev/notes.md). second_order_omitted says whether the model
    has parameters for which the omission is not exact.

    Cost: the first derivatives of V are formed in full (n x n), one per
    variance parameter, then paired for the information: O(p_theta^2 n^2).
    Validated against pbkrtest::KRmodcomp (dev/notes.md)."""
    theta = np.asarray(theta, dtype=float)
    n, p = X.shape
    k = len(theta)
    Zs = [dense_Z(t, n) for t in terms]

    def V_of(th):
        return assemble_V(th, terms, Zs, res, n)

    V = np.asarray(V_of(jnp.asarray(theta)))
    try:
        Vi = np.linalg.solve(V, np.eye(n))
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    U = Vi @ X                                    # n x p
    Phi = np.linalg.pinv(X.T @ U)                 # (X'V^-1X)^-1
    P_mat = Vi - U @ Phi @ U.T                    # REML projector
    D = []
    for i in range(k):
        e = np.zeros(k); e[i] = 1.0
        _, dV = jax.jvp(V_of, (jnp.asarray(theta),), (jnp.asarray(e),))
        D.append(np.asarray(dV))
    I = _expected_reml_info(D, P_mat)
    w, Qv = np.linalg.eigh(0.5 * (I + I.T))
    tol = 1e-10 * max(abs(w).max(), 1.0)
    if not np.all(w > tol):
        return dict(available=False,
                    reason="REML information not positive definite (%d flat "
                           "direction(s)). The covariance of the variance parameters "
                           "does not exist, so neither does the Kenward-Roger adjustment."
                           % int((w <= tol).sum()))
    W = Qv @ np.diag(1.0 / w) @ Qv.T
    Phi_A, dPhi = _kr_adjusted_cov(D, U, Vi, Phi, W, k, p)
    beta = Phi @ (X.T @ Vi @ y)
    df_resid = float(n - np.linalg.matrix_rank(X))
    tests = []
    for name, cols in _column_groups(fixed_terms, p):
        d = _kr_test(name, cols, beta, Phi, Phi_A, dPhi, W, k, p, df_resid)
        if d is not None:
            tests.append(d)
    n_lvl = sum(n_level(t) for t in terms)
    n_lvl += sum(n_level(s) for s in res_sections(res, n))
    return dict(available=True, tests=tests,
                se_beta=[float(np.sqrt(max(v, 0))) for v in np.diag(Phi_A)],
                se_beta_raw=[float(np.sqrt(max(v, 0))) for v in np.diag(Phi)],
                vbeta_kr=Phi_A, beta=[float(b) for b in beta],
                second_order_omitted=bool(n_lvl > 0))


def _kr_adjusted_cov(D, U, Vi, Phi, W, k, p):
    """Phi_A, and the derivatives dPhi/dtheta_i = Phi P_i Phi."""
    GU = [Di @ U for Di in D]                     # dV_i U
    Pt = [U.T @ g for g in GU]                    # X'V^-1 dV_i V^-1 X
    dPhi = [Phi @ Pi @ Phi for Pi in Pt]
    S = np.zeros((p, p))
    for i in range(k):
        Vig = Vi @ GU[i]
        for j in range(k):
            S += W[i, j] * (GU[j].T @ Vig - Pt[i] @ Phi @ Pt[j])
    return Phi + 2.0 * Phi @ S @ Phi, dPhi


def _kr_test(name, cols, beta, Phi, Phi_A, dPhi, W, k, p, df_resid):
    """Kenward-Roger F test of one group of columns, or None when the group is
    empty. On an orthogonal design A1 and A2 vanish exactly (dV_i V^-1 X stays
    in the span of X) and the unadjusted test with the residual df is
    returned; when the moment matching leaves its domain (m <= 2), the
    unadjusted test too, with a note."""
    from scipy.stats import f as f_dist
    cols = list(cols)
    L = np.zeros((p, len(cols)))
    for r, c in enumerate(cols):
        L[c, r] = 1.0
    l = int(np.linalg.matrix_rank(L))
    if l == 0:
        return None
    Lb = L.T @ beta
    Fstar = float(Lb @ np.linalg.pinv(L.T @ Phi_A @ L) @ Lb) / l
    Th = L @ np.linalg.pinv(L.T @ Phi @ L) @ L.T
    tD = np.array([np.trace(Th @ dPhi[i]) for i in range(k)])
    A1 = float(tD @ W @ tD)
    TD = [Th @ dPhi[i] for i in range(k)]
    A2 = float(sum(W[i, j] * np.trace(TD[i] @ TD[j])
                   for i in range(k) for j in range(k)))
    if abs(A1) < 1e-12 and abs(A2) < 1e-12:
        return dict(term=name, df=l, denDF=df_resid, F=float(Fstar),
                    F_raw=float(Fstar), scale=1.0,
                    p=float(f_dist.sf(Fstar, l, df_resid)),
                    note="orthogonal design: no adjustment")
    B = (A1 + 6.0 * A2) / (2.0 * l)
    g = ((l + 1) * A1 - (l + 4) * A2) / ((l + 2) * A2)
    den = 3.0 * l + 2.0 * (1.0 - g)
    c1, c2, c3 = g / den, (l - g) / den, (l + 2.0 - g) / den
    E_star = 1.0 / (1.0 - A2 / l)
    V_star = (2.0 / l) * (1.0 + c1 * B) / ((1.0 - c2 * B) ** 2 * (1.0 - c3 * B))
    rho = V_star / (2.0 * E_star ** 2)
    m_df = 4.0 + (l + 2.0) / (l * rho - 1.0)
    lam = m_df / (E_star * (m_df - 2.0)) if m_df > 2.0 else 1.0
    note = None
    if not np.isfinite(m_df) or m_df <= 2.0:
        m_df, lam = df_resid, 1.0
        note = "moment matching out of range: test NOT adjusted"
    FKR = lam * Fstar
    d = dict(term=name, df=l, denDF=float(m_df), F=float(FKR),
             F_raw=float(Fstar), scale=float(lam),
             p=float(f_dist.sf(FKR, l, m_df)))
    if note:
        d["note"] = note
    return d
