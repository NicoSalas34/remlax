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
    run = _Finish(fun_jac, hessian, lo, hi, counter, verbose, tol_dec=tol_dec,
                  tol_gain=tol_gain, delta_max=delta_max, floor_window=floor_window,
                  floor_tol=floor_tol)
    return run.run(theta, maxit, delta0)


class _Finish:
    """State of one trust-region Newton finish: theta, -2logL, gradient, H, Delta."""

    def __init__(self, fun_jac, hessian, lo, hi, counter, verbose, tol_dec, tol_gain,
                 delta_max, floor_window, floor_tol):
        self.t0 = time.time()
        self.fun_jac, self.hessian, self.lo, self.hi = fun_jac, hessian, lo, hi
        self.counter, self.verbose = counter, verbose
        self.tol_dec, self.tol_gain, self.delta_max = tol_dec, tol_gain, delta_max
        self.floor_window, self.floor_tol = floor_window, floor_tol
        self.n_eval, self.n_hess = 0, 0

    def say(self, fmt, *args):
        if self.verbose:
            print(fmt % args, flush=True)

    def evaluate(self, th):
        self.n_eval += 1
        if self.counter is not None:
            self.counter["n_eval"] = self.counter.get("n_eval", 0) + 1
        v, g = self.fun_jac(th)
        return float(v), np.asarray(g, dtype=np.float64)

    def refresh_hessian(self):
        self.H = self.hessian(self.theta)
        self.n_hess += 1
        self.fresh = True

    def to_floor(self):
        """Sets to the floor the free variances that the gradient pushes down.

        Kept when -2logL rises by no more than floor_tol. Returns how many."""
        lo = self.lo
        cand = (self.theta > lo + 1e-10) & (self.theta < lo + self.floor_window) & (self.g > 0)
        if not cand.any():
            return 0
        th2 = self.theta.copy(); th2[cand] = lo[cand]
        f2, g2 = self.evaluate(th2)
        if np.isfinite(f2) and f2 <= self.f + self.floor_tol:
            self.theta, self.f, self.g = th2, f2, g2
            return int(cand.sum())
        return 0

    def run(self, theta, maxit, delta0):
        self.theta = np.clip(np.asarray(theta, dtype=np.float64), self.lo, self.hi)
        self.f, self.g = self.evaluate(self.theta)
        self.refresh_hessian()
        self.delta = delta0
        self.poor, self.small = 0, 0
        msg, ok = "maxit reached", False
        n_fl = self.to_floor()
        self.say("  [finish] start -2logL %.6f | %d parameter(s) set to the floor | Hessian %d gradients",
                 self.f, n_fl, 2 * len(self.theta))
        for it in range(maxit):
            stop = self.iterate(it)
            if stop:
                msg, ok = stop
                break
        info = self.report(msg, ok)
        return self.theta, info

    def iterate(self, it):
        """One iteration. Returns (message, success) when the finish stops."""
        L = _free(self.theta, self.g, self.lo, self.hi)
        gL, HL = self.g[L], self.H[np.ix_(L, L)]
        dec = decrement(gL, HL)
        if dec < self.tol_dec and self.small >= 1:
            return self.near_stop(dec)
        gain, rho = self.step(L, gL, HL)
        self.say("  [finish] %3d | -2logL %.8f | gain %.2e | rho %6.2f | Delta %.2e | decrement %.2e | %d free | %.0f s",
                 it + 1, self.f, gain, rho, self.delta, dec, int(L.sum()), time.time() - self.t0)
        if self.poor >= 2:
            self.rebuild()
        if self.delta <= 1e-8 and self.fresh:
            return "trust region collapsed (Delta %.1e)" % self.delta, dec < 1e-3
        return None

    def near_stop(self, dec):
        """Small decrement after small gains: floor, then stop on a fresh Hessian."""
        # Before concluding: set vanishing variances to the floor (one
        # evaluation); if some are, the problem has changed face.
        n_fl = self.to_floor()
        if n_fl:
            self.say("  [finish] %d parameter(s) set to the floor", n_fl)
            self.small = 0
            return None
        if self.fresh:
            return "decrement %.2e < %.0e" % (dec, self.tol_dec), True
        self.refresh_hessian()        # confirm with a fresh Hessian
        return None

    def step(self, L, gL, HL):
        """Trust-region step on the free parameters, accepted when it gains."""
        dL, pred = _trust_region_step(gL, HL, self.delta)
        d = np.zeros_like(self.theta); d[L] = dL
        th_new = np.clip(self.theta + d, self.lo, self.hi)
        s = th_new - self.theta
        f_new, g_new = self.evaluate(th_new)
        gain = self.f - f_new if np.isfinite(f_new) else -np.inf
        rho = gain / pred if pred > 0 else -1.0
        self.update_radius(rho, s, dL)
        if gain > 0:
            self.accept(th_new, f_new, g_new, s, gain, rho)
        else:
            self.poor += 1
            # A step refused while the model itself predicts less than
            # tol_gain has nothing left to find. It counts as a small gain,
            # otherwise a step that lands on the optimum with a large gain
            # leaves small at 0 and the finish cycles until maxit.
            if pred < self.tol_gain:
                self.small += 1
        return gain, rho

    def update_radius(self, rho, s, dL):
        if rho < 0.25:
            self.delta = max(0.25 * np.linalg.norm(s), 1e-8)
        elif rho > 0.75 and np.linalg.norm(dL) >= 0.99 * self.delta:
            self.delta = min(2.0 * self.delta, self.delta_max)

    def accept(self, th_new, f_new, g_new, s, gain, rho):
        y = g_new - self.g
        # SR1 update of H, kept when the denominator is not degenerate.
        r = y - self.H @ s
        den = r @ s
        if abs(den) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
            self.H = self.H + np.outer(r, r) / den
            self.fresh = False
        self.theta, self.f, self.g = th_new, f_new, g_new
        self.small = self.small + 1 if gain < self.tol_gain else 0
        self.poor = 0 if rho >= 0.25 else self.poor + 1

    def rebuild(self):
        """The quadratic model no longer predicts: fresh Hessian, and floor."""
        n_fl = self.to_floor()
        self.refresh_hessian()
        self.poor = 0
        self.delta = max(self.delta, 1e-3)
        if n_fl:
            self.say("  [finish] %d parameter(s) set to the floor", n_fl)

    def report(self, msg, ok):
        L = _free(self.theta, self.g, self.lo, self.hi)
        info = dict(success=ok, message="finish: %s" % msg, n_eval=self.n_eval,
                    n_hessians=self.n_hess, decrement=decrement(self.g[L], self.H[np.ix_(L, L)]),
                    seconds=time.time() - self.t0)
        self.say("  [finish] end: %s | -2logL %.8f | %d evaluations, %d Hessians | %.0f s",
                 msg, self.f, self.n_eval, self.n_hess, info["seconds"])
        return info
