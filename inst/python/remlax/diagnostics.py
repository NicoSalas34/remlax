"""Hessian and convergence diagnostics of a REML fit.

The stopping rule stays that of L-BFGS-B followed by Newton polishing. The
diagnostics below are reported, not imposed. max|grad| is not used: on these
likelihoods it is several units at verified optima. The reported criteria are
the Newton decrement g' H^+ g (the logLik ascent still available under the
local quadratic model, in chi-square units and invariant under affine
reparametrisation), the relative projected gradient, and the sign of the
curvature.
"""
import numpy as np


def hessian_fd(fun_jac, theta, eps=1e-5, floor=None, ceil=None):
    """Hessian by central finite differences of the analytic gradient: 2p
    gradient evaluations, as TMB::sdreport does, safer than an autodiff
    Hessian through a Cholesky factorisation.

    A step of +-eps can leave the domain where V is positive definite; the
    gradient is then not finite. The column falls back to a forward or
    backward difference on the side that can be evaluated, and is set to zero
    (a flat direction for the diagnostics) when neither can."""
    p = len(theta)
    H = np.full((p, p), np.nan)
    f0, g0 = fun_jac(theta)
    for j in range(p):
        e = np.zeros(p); e[j] = eps
        tp = theta + e if ceil is None else np.minimum(theta + e, ceil)
        tm = theta - e if floor is None else np.maximum(theta - e, floor)
        _, gp = fun_jac(tp)
        _, gm = fun_jac(tm)
        ok_p, ok_m = np.all(np.isfinite(gp)), np.all(np.isfinite(gm))
        if ok_p and ok_m:
            H[:, j] = (gp - gm) / (tp[j] - tm[j])
        elif ok_p and np.all(np.isfinite(g0)):
            H[:, j] = (gp - g0) / (tp[j] - theta[j])      # forward difference
        elif ok_m and np.all(np.isfinite(g0)):
            H[:, j] = (g0 - gm) / (theta[j] - tm[j])      # backward difference
    bad = ~np.isfinite(H)
    if bad.any():
        H[bad] = 0.0
    return 0.5 * (H + H.T)


# Thresholds of the convergence criteria, reported only. A decrement of 1e-4
# leaves a ten-thousandth of a chi-square unit to gain. The relative gradient
# decides when the Hessian is ill-conditioned or has null directions, where
# the decrement cannot be read.
DECREMENT_THRESHOLD = 1e-4
GRAD_REL_THRESHOLD = 1e-6


def _verdict(d):
    """Three separate answers, never merged; each may be missing. A negative
    eigenvalue says the point is not a maximum whatever the gradient."""
    dec, gr = d.get("newton_decrement"), d.get("grad_rel")
    v = {}
    v["conv_decrement"] = (None if dec is None or not np.isfinite(dec)
                           else bool(dec < DECREMENT_THRESHOLD))
    v["conv_grad_rel"] = (None if gr is None or not np.isfinite(gr)
                          else bool(gr < GRAD_REL_THRESHOLD))
    v["conv_hessien_ok"] = (None if d.get("n_neg_eig", -1) < 0
                            else bool(d["n_neg_eig"] == 0 and d.get("n_null_dir", 0) == 0))
    v["seuil_decrement"] = DECREMENT_THRESHOLD
    v["seuil_grad_rel"] = GRAD_REL_THRESHOLD
    return v


def _free_subspace(theta, floor, ceil, tol_bound, fixed_idx):
    """Parameters at a bound, fixed parameters, and the free ones; with the
    counts n_at_bound (fixed excluded), n_fixed_out and n_par_free."""
    p = len(theta)
    at_bound = (theta <= floor + tol_bound) | (theta >= ceil - tol_bound)
    fixed = np.zeros(p, bool)
    if fixed_idx:
        fixed[np.asarray(list(fixed_idx), dtype=int)] = True
    free = ~at_bound & ~fixed
    counts = dict(n_at_bound=int((at_bound & ~fixed).sum()),
                  n_fixed_out=int(fixed.sum()),
                  n_par_free=int(free.sum()))
    return fixed, free, counts


def gradient_diagnostic(g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None,
                        f_obj=None, ech=None):
    """The criteria that need only a gradient, already computed; used when
    the fit is run without Hessian.

    The gradient is taken on the free subspace: a variance held at its bound
    keeps a non-zero gradient in that direction, and no criterion could ever be
    met otherwise. Each component is scaled by max(|theta_i|, 1) and the
    objective by max(|f|, 1)."""
    fixed, free, out = _free_subspace(theta, floor, ceil, tol_bound, fixed_idx)
    if free.sum() == 0:
        return dict(out, grad_proj_max=np.nan, grad_rel=np.nan, conv_grad_rel=None)
    gf = g[free]
    if not np.all(np.isfinite(gf)):
        return dict(out, grad_proj_max=np.nan, grad_rel=np.nan, conv_grad_rel=None)
    gpm = float(np.max(np.abs(gf)))
    e = np.maximum(np.abs(theta[free]), 1.0) if ech is None else np.asarray(ech)[free]
    if f_obj is None or not np.isfinite(f_obj):
        grad_rel = np.nan
    else:
        grad_rel = float(np.max(np.abs(gf) * e) / max(abs(float(f_obj)), 1.0))
    return dict(out, grad_proj_max=gpm, grad_rel=grad_rel,
                conv_grad_rel=(None if not np.isfinite(grad_rel) else bool(grad_rel < 1e-6)))


def diagnostic(H, g, theta, floor, ceil, tol_bound=1e-7, fixed_idx=None, f_obj=None,
               n_plat_attendu=0):
    """Three separate questions, never merged into one boolean:

      at the top?        Newton decrement g' H^+ g, in logLik units
      a peak?            spectrum of H (null and negative eigenvalues)
      which parameters?  those at an active bound or fixed are outside the free
                         subspace (KKT condition) and take no part

    A parameter held by fixed_theta is outside the free subspace but not
    counted in n_at_bound: it is pinned at its start, which has no reason to
    be floor or ceil. n_at_bound and n_fixed_out are reported separately; only
    their total defines the free subspace.

    The n_plat_attendu smallest eigenvalues in modulus are the expected flat
    directions of entries not identified separately (aliased_entries): they are
    set aside when counting negative and null eigenvalues."""
    fixed, free, out = _free_subspace(theta, floor, ceil, tol_bound, fixed_idx)
    if free.sum() == 0:
        return dict(out, newton_decrement=np.nan, grad_proj_max=np.nan,
                    grad_rel=np.nan, n_neg_eig=0, n_null_dir=0,
                    cond=np.nan, conv_decrement=None, conv_grad_rel=None,
                    conv_hessien_ok=None)
    Hf, gf = H[np.ix_(free, free)], g[free]
    if not np.all(np.isfinite(Hf)) or not np.all(np.isfinite(gf)):
        return dict(out, newton_decrement=np.nan, leak=np.nan, n_neg_eig=-1,
                    n_null_dir=-1, cond=np.nan,
                    diag_note="Hessien ou gradient non fini : diagnostic indisponible")
    try:
        w, Vv = np.linalg.eigh(Hf)
    except np.linalg.LinAlgError as e:
        return dict(out, newton_decrement=np.nan, leak=np.nan, n_neg_eig=-1,
                    n_null_dir=-1, cond=np.nan,
                    diag_note="decomposition spectrale echouee (%s)" % e)
    tol = 1e-8 * max(abs(w).max(), 1.0)
    expected = np.zeros(len(w), dtype=bool)
    if n_plat_attendu:
        expected[np.argsort(np.abs(w))[:min(int(n_plat_attendu), len(w))]] = True
    keep = (w > tol) & ~expected
    dec = float(gf @ (Vv[:, keep] @ ((Vv[:, keep].T @ gf) / w[keep]))) if keep.any() else np.nan
    leak = float(np.max(np.abs(Vv[:, ~keep].T @ gf))) if (~keep).any() else 0.0
    gpm, grad_rel = _projected_gradient(g, theta, floor, ceil, tol_bound, fixed_idx, fixed,
                                        f_obj)
    res = dict(out, newton_decrement=dec, leak=leak,
               grad_proj_max=gpm, grad_rel=grad_rel,
               n_neg_eig=int(((w < -tol) & ~expected).sum()),
               n_null_dir=int(((np.abs(w) <= tol) & ~expected).sum()),
               n_plat_exclues=int(expected.sum()),
               lambda_min=float(w[~expected].min()) if (~expected).any() else np.nan,
               lambda_max=float(w.max()),
               cond=(float(abs(w[~expected]).max() / max(abs(w[~expected]).min(), 1e-300))
                     if (~expected).any() else np.nan),
               k_eff=int(keep.sum()))
    return dict(res, **_verdict(res))


def _projected_gradient(g, theta, floor, ceil, tol_bound, fixed_idx, fixed, f_obj):
    """max |g| projected on the admissible cone, and its relative version.

    Under box bounds, optimality is g = 0 in the free directions and g pointing
    outwards at active bounds: at a lower bound only a negative component
    shows non-optimality, a positive one is held by the constraint and
    projects to zero. The relative version scales each component by
    max(|theta_i|, 1) and the objective by max(|f|, 1) (Dennis and Schnabel);
    it is NaN when f_obj, the value of -2logL at theta, is not given."""
    p = len(theta)
    g_proj = np.array(g, dtype=np.float64, copy=True)
    lo = theta <= floor + tol_bound
    hi = theta >= ceil - tol_bound
    g_proj[lo] = np.minimum(g_proj[lo], 0.0)
    g_proj[hi] = np.maximum(g_proj[hi], 0.0)
    if fixed_idx:
        g_proj[fixed] = 0.0
    gpm = float(np.max(np.abs(g_proj))) if p else np.nan
    ech = np.maximum(np.abs(np.asarray(theta, dtype=np.float64)), 1.0)
    if f_obj is None or not np.isfinite(f_obj):
        grad_rel = np.nan
    else:
        f_ech = max(abs(float(f_obj)), 1.0)
        grad_rel = float(np.max(np.abs(g_proj) * ech) / f_ech) if p else np.nan
    return gpm, grad_rel
