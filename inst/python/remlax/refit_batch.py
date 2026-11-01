"""Exact refits of many SNPs at once, vectorised over SNPs (jax.vmap).

rx_scan_exact refits the top SNPs of a scan one by one: REML with the SNP
columns appended to X, then the Wald test of the tested hypothesis L b = 0.
Every refit shares the random part of the model (terms, residual, y) and
the null start theta0; only the fixed effects differ, X_s = [X, S_s]. REML
depends on X_s through its orthonormal basis Q_s and the constant
2 log|det R_s| (incidence.orthonormal_X). One traced function of
(theta, Q_s) therefore serves every SNP, and a batch of SNPs is one call on
the device.

Optimiser, per SNP. Newton with the exact Hessian (theta is short, so
jax.hessian is cheap next to the n^3 of a Cholesky). The Hessian restricted
to the free parameters is made positive definite by its absolute
eigenvalues; the step norm is capped at max_step; the best of the step
lengths 1, 1/2, ..., 1/32 and 0 is kept, within the box bounds of fit_reml.
A parameter at a bound with the gradient pushing outwards is held
(finition._free). A SNP stops when its Newton decrement falls below tol and
is no longer moved; the batch stops when all have stopped or after maxit.

Outputs per SNP: theta, -2logL, decrement, iterations, converged, and on
the raw scale of the SNP columns: beta, its covariance block, the Wald
chi2 of L b = 0 with L b and its standard errors.
"""
import numpy as np

from . import _x64  # noqa: F401
import jax
import jax.numpy as jnp

from .incidence import support_groups, restrict_Z, prefold_fixed_K
from .model import neg2_reml_by_support, assemble_V_by_support, _center_j

STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def make_batch_neg2(terms, res, y):
    """f(theta, Q, ldR, Zr): -2logL REML of the model with fixed effects of
    orthonormal basis Q (constant ldR = 2 log|det R|). Returns (f, Zr, V_of)."""
    n = y.shape[0]
    groups = support_groups(terms, n)
    Zr = [None] * len(terms)
    for rows, ks in groups:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)
    terms_v, Zr = prefold_fixed_K(terms, Zr)
    yj = jnp.asarray(y)

    def f(theta, Q, ldR, Zr_a):
        return neg2_reml_by_support(theta, terms_v, groups, Zr_a, res, _center_j(yj, Q), Q) + ldR

    def V_of(theta, Zr_a):
        return assemble_V_by_support(theta, terms_v, groups, Zr_a, res, n)
    return f, Zr, V_of


def _qr_batch(X, S):
    """Orthonormal bases Q (B, n, p+k), R factors and 2 log|det R| of [X, S_s]."""
    B = S.shape[0]
    Xs = np.concatenate([np.broadcast_to(X, (B,) + X.shape), S], axis=2)
    Q, R = np.linalg.qr(Xs)
    ldR = 2.0 * np.sum(np.log(np.abs(np.diagonal(R, axis1=1, axis2=2))), axis=1)
    return Q, R, ldR


def _newton_step(f, lo, hi, max_step, tol):
    """One Newton step for one SNP, to be vmapped. Returns (theta, f, decrement)."""
    vg = jax.value_and_grad(f)
    hess = jax.hessian(f)
    steps = jnp.asarray(STEPS)

    def step(theta, Q, ldR, Zr, done):
        f0, g = vg(theta, Q, ldR, Zr)
        H = hess(theta, Q, ldR, Zr)
        free = ~(((theta <= lo + 1e-10) & (g > 0)) | ((theta >= hi - 1e-10) & (g < 0)))
        fr = free.astype(theta.dtype)
        gf = g * fr
        Hf = 0.5 * (H + H.T) * jnp.outer(fr, fr) + jnp.diag(1.0 - fr)
        w, U = jnp.linalg.eigh(Hf)
        wa = jnp.maximum(jnp.abs(w), jnp.maximum(1e-8 * jnp.max(jnp.abs(w)), 1e-10))
        a = U.T @ gf
        dec = 0.5 * jnp.sum(a ** 2 / wa)
        d = -(U @ (a / wa)) * fr
        nd = jnp.linalg.norm(d)
        d = d * jnp.minimum(1.0, max_step / jnp.maximum(nd, 1e-300))
        cand = jnp.clip(theta[None, :] + steps[:, None] * d[None, :], lo, hi)
        fc = jax.vmap(lambda t: f(t, Q, ldR, Zr))(cand)
        fc = jnp.where(jnp.isfinite(fc), fc, jnp.inf)
        j = jnp.argmin(fc)
        better = fc[j] < f0
        th_new = jnp.where(better, cand[j], theta)
        f_new = jnp.where(better, fc[j], f0)
        stop = done | (dec < tol)
        return (jnp.where(stop, theta, th_new), jnp.where(stop, f0, f_new), dec)
    return step


def _wald(V_of, Zr, theta, Qn, R, y, k, L):
    """beta, covariance block and Wald chi2 of the SNP columns at theta, for one SNP."""
    V = V_of(theta, Zr)
    C = jnp.linalg.cholesky(V)
    sol = lambda b: jax.scipy.linalg.cho_solve((C, True), b)  # noqa: E731
    ViQ = sol(Qn)
    A = Qn.T @ ViQ
    Ai = jnp.linalg.inv(A)
    bQ = Ai @ (Qn.T @ sol(y))
    Ri = jax.scipy.linalg.solve_triangular(R, jnp.eye(R.shape[0]), lower=False)
    beta = Ri @ bQ
    cov = Ri @ Ai @ Ri.T
    b = beta[-k:]
    Vb = cov[-k:, -k:]
    Lb = L @ b
    LVL = L @ Vb @ L.T
    chi2 = Lb @ jnp.linalg.solve(LVL, Lb)
    return b, Vb, Lb, jnp.sqrt(jnp.diag(LVL)), chi2


def refit_batch(terms, res, y, X, S, L, theta0, lo, hi, maxit=50, tol=1e-9,
                max_step=1.0, batch=64, verbose=False):
    """REML refits with the SNP columns S[s] (n x k) appended to X, for every s.

    S     : array (n_snp, n, k), the columns of each SNP (raw scale).
    L     : (r, k) tested hypothesis L b = 0 on the SNP coefficients.
    theta0: start (the null fit); lo, hi: box bounds of theta (fit._bounds).
    batch : SNPs per device call; memory grows as batch x n^2 x len(theta).
    Returns a dict of arrays with one row per SNP."""
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 2:
        S = S[:, :, None]
    L = np.atleast_2d(np.asarray(L, dtype=np.float64))
    n_snp, n, k = S.shape
    f, Zr, V_of = make_batch_neg2(terms, res, y)
    lo_j, hi_j = jnp.asarray(lo, dtype=jnp.float64), jnp.asarray(hi, dtype=jnp.float64)
    step = jax.jit(jax.vmap(_newton_step(f, lo_j, hi_j, max_step, tol),
                            in_axes=(0, 0, 0, None, 0)))
    wald = jax.jit(jax.vmap(lambda th, Qn, R: _wald(V_of, Zr, th, Qn, R, jnp.asarray(y), k,
                                                    jnp.asarray(L)),
                            in_axes=(0, 0, 0)))
    fv = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, None)))
    out = {key: [] for key in ("theta", "neg2_reml", "decrement", "n_iter", "converged",
                               "beta", "vbeta", "Lb", "se_Lb", "chi2")}
    for s0 in range(0, n_snp, int(batch)):
        Sb = S[s0:s0 + int(batch)]
        Q, R, ldR = _qr_batch(X, Sb)
        Bn = Sb.shape[0]
        th = jnp.tile(jnp.clip(jnp.asarray(theta0, dtype=jnp.float64), lo_j, hi_j), (Bn, 1))
        Qj, ldRj = jnp.asarray(Q), jnp.asarray(ldR)
        done = jnp.zeros(Bn, dtype=bool)
        fval = fv(th, Qj, ldRj, Zr)
        dec = jnp.full(Bn, jnp.inf)
        n_it = np.zeros(Bn, dtype=int)
        for it in range(int(maxit)):
            th, fval, dec = step(th, Qj, ldRj, Zr, done)
            n_it += ~np.asarray(done)
            done = done | (dec < tol)
            if bool(jnp.all(done)):
                break
        b, Vb, Lb, se, chi2 = wald(th, Qj, jnp.asarray(R))
        for key, v in (("theta", th), ("neg2_reml", fval), ("decrement", dec), ("n_iter", n_it),
                       ("converged", done), ("beta", b), ("vbeta", Vb), ("Lb", Lb),
                       ("se_Lb", se), ("chi2", chi2)):
            out[key].append(np.asarray(v))
        if verbose:
            print("  [refit_batch] %d / %d SNP, %d converged"
                  % (min(s0 + Bn, n_snp), n_snp, int(np.sum(np.concatenate(out["converged"])))),
                  flush=True)
    return {key: np.concatenate(v) for key, v in out.items()}
