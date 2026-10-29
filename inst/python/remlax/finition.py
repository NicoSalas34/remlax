"""Finish of a REML fit: trust-region Newton.

On large multivariate models (mvD7: 198 parameters) L-BFGS-B spends hours in a
long tail, gaining a few thousandths of -2logL per hundred iterations. Two
properties of the problem explain it: vanishing variances (in theta = log
sigma the gradient is proportional to sigma^2 and vanishes near the floor) and
flat directions (poorly identified covariances). A second-order method
converges in a few steps, provided both cases are handled.

At each step the trust-region subproblem

    min_d  g'd + d'Hd / 2   subject to   ||d|| <= Delta

is solved exactly by the eigendecomposition of H restricted to the free
parameters (More and Sorensen): d(mu) = -V (V'g) / (w + mu), mu chosen so that
||d|| = Delta when the Newton step leaves the region or H is not positive
definite. A flat or negative-curvature direction is thus bounded by Delta. The
ratio of actual to predicted gain sets Delta.

The Hessian costs 2p gradients (central differences). It is computed at the
start, then updated by SR1 at each accepted step (one evaluation per step); it
is recomputed only when the quadratic model stops predicting the gains (two
poor ratios in a row), and once more to confirm convergence.

Vanishing variances: a free parameter near the floor that the gradient pushes
down is set to the floor when -2logL rises by no more than floor_tol, which
avoids the asymptotic descent towards -12. If the gradient later pushes it
back inside, it becomes free again.

The stop rests on the Newton decrement g'H^+g / 2 (gain of -2logL still
available in the local model) and on the gain of the last steps, not on a
relative tolerance of the objective.
"""
import time

import numpy as np


def _free(theta, g, lo, hi, tol=1e-10):
    """Parameters not held at a bound by a gradient pushing outwards."""
    low = (theta <= lo + tol) & (g > 0)
    high = (theta >= hi - tol) & (g < 0)
    return ~(low | high)


def _trust_region_step(g, H, delta):
    """Exact trust-region step (eigendecomposition). Returns (d, predicted gain)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    a = V.T @ g
    w_min = w.min()

    def d_of(mu):
        return -V @ (a / (w + mu))

    if w_min > 0:
        d = d_of(0.0)
        if np.linalg.norm(d) <= delta:
            return d, -(g @ d + 0.5 * d @ H @ d)
    # Hard case (More and Sorensen): negative curvature with a nearly zero
    # gradient on its eigenvector. d(mu) cannot reach the boundary through mu
    # alone; a move along that eigenvector completes it.
    i_min = int(np.argmin(w))
    if w_min < 0 and abs(a[i_min]) <= 1e-8 * max(np.linalg.norm(g), 1e-300):
        mu = -w_min
        a2 = a.copy(); a2[i_min] = 0.0
        ok = np.abs(w + mu) > 1e-14 * max(1.0, abs(w).max())
        d0 = -V[:, ok] @ (a2[ok] / (w[ok] + mu))
        rest = delta ** 2 - d0 @ d0
        if rest > 0:
            d = d0 + np.sqrt(rest) * V[:, i_min]
            return d, -(g @ d + 0.5 * d @ H @ d)
    # mu with ||d(mu)|| = delta, mu > max(0, -w_min): bisection on a log scale.
    m_lo = max(0.0, -w_min) + 1e-12 * max(1.0, abs(w).max())
    m_hi = m_lo + np.linalg.norm(g) / delta + abs(w).max()
    while np.linalg.norm(d_of(m_hi)) > delta:
        m_hi *= 2.0
    for _ in range(100):
        m = np.sqrt(m_lo * m_hi) if m_lo > 0 else 0.5 * (m_lo + m_hi)
        if np.linalg.norm(d_of(m)) > delta:
            m_lo = m
        else:
            m_hi = m
        if m_hi - m_lo <= 1e-10 * m_hi:
            break
    d = d_of(m_hi)
    return d, -(g @ d + 0.5 * d @ H @ d)


def decrement(g, H, rel=1e-8, plancher=1e-6):
    """Gain of -2logL still available in the local model (Newton decrement).

    Regularised, not truncated: each direction counts with the curvature
    max(w, plancher), so a flat direction is neglected only when the gradient
    on it is negligible too (dev/notes.md). A significant negative curvature
    returns +inf: not a minimum."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    a = V.T @ g
    scale = max(abs(w).max(), 1e-300)
    if np.any((w < -rel * scale) & (np.abs(a) > 1e-8 * max(np.linalg.norm(g), 1e-300))):
        return float("inf")
    return 0.5 * float(np.sum(a ** 2 / np.maximum(w, max(plancher, rel * scale))))


def finish(fun_jac, theta, lo, hi, hessian, maxit=300, tol_dec=1e-6, tol_gain=1e-7,
           delta0=1.0, delta_max=10.0, floor_window=6.0, floor_tol=1e-6,
           verbose=True, counter=None):
    """Trust-region Newton from theta.

    fun_jac(theta) -> (-2logL, gradient) on the natural scale; hessian(theta)
    -> H (p x p). lo, hi: bounds (vectors). counter: the dict whose "n_eval"
    counts the evaluations. Returns (theta, info)."""
    t0 = time.time()
    theta = np.clip(np.asarray(theta, dtype=np.float64), lo, hi)
    n_eval = [0]

    def evaluate(th):
        n_eval[0] += 1
        if counter is not None:
            counter["n_eval"] = counter.get("n_eval", 0) + 1
        v, g = fun_jac(th)
        return float(v), np.asarray(g, dtype=np.float64)

    f, g = evaluate(theta)
    H = hessian(theta); n_hess = 1; fresh = True
    delta = delta0
    poor, small = 0, 0
    msg, ok = "maxit atteint", False

    def to_floor(theta, f, g):
        """Sets to the floor the free variances that the gradient pushes down."""
        cand = (theta > lo + 1e-10) & (theta < lo + floor_window) & (g > 0)
        if not cand.any():
            return theta, f, g, 0
        th2 = theta.copy(); th2[cand] = lo[cand]
        f2, g2 = evaluate(th2)
        if np.isfinite(f2) and f2 <= f + floor_tol:
            return th2, f2, g2, int(cand.sum())
        return theta, f, g, 0

    theta, f, g, n_fl = to_floor(theta, f, g)
    if verbose:
        print("  [finition] depart -2logL %.6f | %d parametre(s) plaque(s) au plancher | Hessien %d gradients"
              % (f, n_fl, 2 * len(theta)), flush=True)
    for it in range(maxit):
        L = _free(theta, g, lo, hi)
        gL, HL = g[L], H[np.ix_(L, L)]
        dec = decrement(gL, HL)
        if dec < tol_dec and small >= 1:
            # Before concluding: set vanishing variances to the floor (one
            # evaluation); if some are, the problem has changed face.
            theta, f, g, n_fl = to_floor(theta, f, g)
            if n_fl:
                if verbose:
                    print("  [finition] %d parametre(s) plaque(s) au plancher" % n_fl, flush=True)
                small = 0
                continue
            if fresh:
                msg, ok = "decrement %.2e < %.0e" % (dec, tol_dec), True
                break
            H = hessian(theta); n_hess += 1; fresh = True        # confirm with a fresh Hessian
            continue
        dL, pred = _trust_region_step(gL, HL, delta)
        d = np.zeros_like(theta); d[L] = dL
        th_new = np.clip(theta + d, lo, hi)
        s = th_new - theta
        f_new, g_new = evaluate(th_new)
        gain = f - f_new if np.isfinite(f_new) else -np.inf
        rho = gain / pred if pred > 0 else -1.0
        if rho < 0.25:
            delta = max(0.25 * np.linalg.norm(s), 1e-8)
        elif rho > 0.75 and np.linalg.norm(dL) >= 0.99 * delta:
            delta = min(2.0 * delta, delta_max)
        if gain > 0:
            y = g_new - g
            # SR1 update of H, kept when the denominator is not degenerate.
            r = y - H @ s
            den = r @ s
            if abs(den) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
                H = H + np.outer(r, r) / den
                fresh = False
            theta, f, g = th_new, f_new, g_new
            small = small + 1 if gain < tol_gain else 0
            poor = 0 if rho >= 0.25 else poor + 1
        else:
            poor += 1
        if verbose:
            print("  [finition] %3d | -2logL %.8f | gain %.2e | rho %6.2f | Delta %.2e | decrement %.2e | %d libres | %.0f s"
                  % (it + 1, f, gain, rho, delta, dec, int(L.sum()), time.time() - t0), flush=True)
        if poor >= 2:
            # The quadratic model no longer predicts: fresh Hessian, and floor.
            theta, f, g, n_fl = to_floor(theta, f, g)
            H = hessian(theta); n_hess += 1; fresh = True; poor = 0
            delta = max(delta, 1e-3)
            if verbose and n_fl:
                print("  [finition] %d parametre(s) plaque(s) au plancher" % n_fl, flush=True)
        if delta <= 1e-8 and fresh:
            msg, ok = "region de confiance effondree (Delta %.1e)" % delta, dec < 1e-3
            break
    info = dict(success=ok, message="finition : %s" % msg, n_eval=n_eval[0], n_hessiens=n_hess,
                decrement=decrement(g[_free(theta, g, lo, hi)], H[np.ix_(*(2 * [_free(theta, g, lo, hi)]))]),
                secondes=time.time() - t0)
    if verbose:
        print("  [finition] fin : %s | -2logL %.8f | %d evaluations, %d Hessiens | %.0f s"
              % (msg, f, n_eval[0], n_hess, info["secondes"]), flush=True)
    return theta, info
