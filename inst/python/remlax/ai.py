"""AI-REML: average information optimiser (Gilmour, Thompson and Cullis 1995).

The curvature is the average information matrix of -2 log L_R,

    AI_ij = y' P V_i P V_j P y = q_i' P q_j,     q_i = V_i P y,  V_i = dV/dtheta_i,

and the step, damped in the Levenberg-Marquardt manner, is
    delta = -(AI + mu diag(AI))^-1 g,            g = gradient of -2 log L_R.

The q_i need no n x n matrix. A term contributes Z (Sigma (x) K) Z' to V; for a
fixed vector v, with U = Z'v arranged as (t, q),

    V_k v = Z vec(Sigma U K)   and   d(V_k v)/dtheta = Z vec(dSigma U K + Sigma U dK).

JAX computes this derivative by jacfwd for all the parameters of the term at
once; the cost is a product Z x (t q, p_k) per term, and the extra memory
n x p. A residual `us` by unit is handled the same way by grouping the
observations by unit; other residuals go through the dense section.

The gradient stays that of make_objective; the AI only gives the curvature.
With the Cholesky parametrisation, the AI is the Gauss-Newton part of the
Hessian. The structural model (lag) is not covered.

ai_reml_var() takes the step in the variance scale (see _AIRemlVar).
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from .layout import level_kind, split_theta, term_n_params, res_sections, sec_n_params
from .incidence import support_groups, restrict_Z, prefold_fixed_K, orthonormal_X
from .residual import section_matvec
from .model import assemble_V_by_support
from .structures import chol_sigma, n_params
from .levels import level_chol

# Decrement (units of -2logL) under which an AI loop may declare convergence
# without polishing: the threshold of the final diagnostic
# (diagnostics.DECREMENT_THRESHOLD).
SUCCESS_DECREMENT = 1e-4
# Half steps tried on the AI direction before Levenberg-Marquardt damping.
N_HALF_STEPS = 6
# Interleaved L-BFGS-B salvos: when AI finds no admissible step, or stalls with
# a decrement still large, a few L-BFGS-B iterations move it away from the
# region where the AI matrix is nearly singular, then AI resumes (as asreml
# interleaves EM iterations; dev/notes.md).
SALVO_ITER = 20
SALVOS_MAX = 5
# Step limits (see _AIRemlVar.try_step): a variance divided by at most 10 per
# step, a level parameter (rho on the atanh scale, range in log) moved by at
# most 0.5. The trust-region ratio test (actual gain >= RATIO_MIN x predicted
# gain) stays optional. A variance clamped on N_CLAMP_TO_FLOOR accepted steps
# in a row may reach the floor (Heywood case of an fa).
REDUCTION_MAX = float(os.environ.get("REMLAX_AI_REDUCTION", "0.1"))
LEVEL_STEP_MAX = float(os.environ.get("REMLAX_AI_LEVEL_STEP", "0.5"))
RATIO_MIN = float(os.environ.get("REMLAX_AI_RATIO", "0"))
N_CLAMP_TO_FLOOR = int(os.environ.get("REMLAX_AI_N_CLAMP", "3"))


def _sigma_and_K(theta_k, term):
    """Sigma (t x t) and K (q x q, or None for the identity) of a term, with the
    logic of model.term_factor."""
    t, q = term["t"], term["q"]
    n_sig = n_params(term["struct"], t, term["rank"])
    Ls = chol_sigma(theta_k[:n_sig], term["struct"], t, term["rank"])
    S = Ls @ Ls.T
    kind = level_kind(term)
    if kind == "fixed":
        LK = jnp.asarray(term["LK"])
    else:
        LK = level_chol(theta_k[n_sig:], kind, q, dims=term.get("dims"),
                        LK_fixed=term.get("LK"), order=term.get("lvl_order", 0),
                        coord=term.get("coord"), parts=term.get("lvl_parts"),
                        opts=term.get("lvl_opts"), expr=term.get("lvl_expr"))
    K = None if LK is None else LK @ LK.T
    return S, K


def make_ai(terms, res, y, X):
    """ai(theta) -> AI matrix (p x p) of -2 log L_R, compiled. The supports and
    restricted incidences are computed here as in make_objective; P depends on
    X only through its span, so its orthonormal Q gives the same AI, better
    conditioned. The returned function carries .tools, reused by make_fga."""
    n = y.shape[0]
    Xq, ldRx = orthonormal_X(X)
    yj, Xj = jnp.asarray(y), jnp.asarray(Xq)
    groups = support_groups(terms, n)
    Zr = [None] * len(terms)
    rows_of = [None] * len(terms)
    for rows, ks in groups:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)
            rows_of[k] = jnp.asarray(np.asarray(rows, dtype=np.int64))
    # Fixed kinship folded into the incidence (incidence.prefold_fixed_K).
    terms, Zr = prefold_fixed_K(terms, Zr)
    secs = res_sections(res, n)

    def term_columns(th_k, term, Z, rows, v):
        """d(V_k v)/dtheta_k: n x p_k matrix (zero outside the support)."""
        t, q = term["t"], term["q"]
        U = (Z.T @ v[rows]).reshape(t, q)

        def g(th):
            S, K = _sigma_and_K(th, term)
            M = U if K is None else U @ K
            return (S @ M).reshape(t * q)

        J = jax.jacfwd(g)(th_k)                       # (t q, p_k)
        Qr = Z @ J                                    # (|rows|, p_k)
        return jnp.zeros((n, J.shape[1])).at[rows].set(Qr)

    def section_columns(th_s, sec, v):
        """d(R_s v)/dtheta_s: n x p_s matrix, without forming R_s when the
        structure allows it (residual.section_matvec)."""
        rows = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        vs = v[rows]
        J = jax.jacfwd(lambda th: section_matvec(th, sec, vs))(th_s)
        return jnp.zeros((n, J.shape[1])).at[rows].set(J)

    def ai(theta, Zr):
        V = assemble_V_by_support(theta, terms, groups, Zr, res, n)
        L = jnp.linalg.cholesky(V)

        def sol(b):
            return jax.scipy.linalg.cho_solve((L, True), b)

        ViX = sol(Xj)
        A = Xj.T @ ViX
        Py = sol(yj) - ViX @ jnp.linalg.solve(A, Xj.T @ sol(yj))
        th_terms, th_res, _ = split_theta(theta, terms, res)
        blocks = [term_columns(th_terms[k], tm, Zr[k], rows_of[k], Py)
                  for k, tm in enumerate(terms)]
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            blocks.append(section_columns(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(blocks, axis=1)           # (n, p)
        ViQ = sol(Q)
        PQ = ViQ - ViX @ jnp.linalg.solve(A, Xj.T @ ViQ)
        AI = Q.T @ PQ
        return 0.5 * (AI + AI.T)

    # Incidences as arguments (see model.make_neg2): never folded.
    _ai = jax.jit(ai)
    Zr_dev = Zr

    def out(theta):
        return _ai(theta, Zr_dev)
    out.tools = dict(groups=groups, Zr=Zr, rows_of=rows_of, secs=secs,
                     term_columns=term_columns, section_columns=section_columns,
                     yj=yj, Xj=Xj, n=n, terms=terms, ldRx=ldRx)
    return out


def _gram_pays(term, rows):
    """Term whose contribution sigma2 Z Z' is worth precomputing: one
    component, no kinship or level structure, iid variance, and at least half
    as many levels as the rows of its support (the IEE terms); a spatial term
    with few levels keeps its factor, cheaper."""
    lvl = term.get("lvl")
    return (term["t"] == 1 and term.get("LK") is None and lvl in (None, "", "id")
            and term["struct"] == "iid" and term["q"] >= 0.5 * len(rows))


def make_fga(terms, res, y, X):
    """fga(theta) -> (-2logL, gradient, AI matrix) from one factorisation of V.

    V is assembled as in the engine (supports, sections), except the terms
    whose sigma2 Z Z' is worth precomputing (_gram_pays): their block is
    exp(2 theta) G, G = Z Z' fixed. Then, from the single Cholesky of V:
        -2logL = (n-p) log 2pi + log|V| + log|X'V^-1X| + y'Py
        gradient: adjoint of V, Vbar = P - (Py)(Py)', propagated through the
                  assembly by jax.vjp (no second factorisation);
        AI = Q' P Q, columns of Q as in make_ai.
    The same computation as make_objective and make_ai, without refactorising
    between the two. P is never formed: the adjoint of V and P Q are written
    with V^-1 and a rank-p correction (dev/notes.md)."""
    base = make_ai(terms, res, y, X).tools
    terms = base["terms"]          # fixed kinship already folded into Zr
    groups, Zr, rows_of, secs = base["groups"], base["Zr"], base["rows_of"], base["secs"]
    term_columns, section_columns = base["term_columns"], base["section_columns"]
    yj, Xj, n = base["yj"], base["Xj"], base["n"]
    p_x = Xj.shape[1]
    G = [None] * len(terms)
    for rows, ks in groups:
        for k in ks:
            if _gram_pays(terms[k], rows):
                G[k] = Zr[k] @ Zr[k].T
    from .residual import _section_V
    from .model import term_factor

    # A block covering every row in order is added without scattering: the
    # scatter into zeros(n, n) and its transpose cost one n x n matrix each.
    every = np.arange(n, dtype=np.int64)

    def _covers_all(r):
        r = np.asarray(r, dtype=np.int64)
        return r.shape[0] == n and np.array_equal(r, every)

    sec_full = [_covers_all(sec["rows"]) for sec in secs]
    grp_full = [_covers_all(rows) for rows, _ in groups]

    def _add(V, M, rows, full):
        if full:
            return M if V is None else V + M
        idx = jnp.asarray(np.asarray(rows, dtype=np.int64))
        V = jnp.zeros((n, n)) if V is None else V
        return V.at[idx[:, None], idx[None, :]].add(M)

    def V_of(theta, Zr, G):
        th_terms, th_res, _ = split_theta(theta, terms, res)
        V = None
        o = 0
        for s_i, sec in enumerate(secs):
            pnb = sec_n_params(sec)
            V = _add(V, _section_V(th_res[o:o + pnb], sec), sec["rows"], sec_full[s_i])
            o += pnb
        for g_i, (rows, ks) in enumerate(groups):
            block = None
            fac = [term_factor(th_terms[k], terms[k], Zr[k]) for k in ks if G[k] is None]
            if fac:
                Bm = jnp.concatenate(fac, axis=1)
                block = Bm @ Bm.T
            for k in ks:
                if G[k] is not None:
                    c = jnp.exp(2.0 * th_terms[k][0]) * G[k]
                    block = c if block is None else block + c
            if block is not None:
                V = _add(V, block, rows, grp_full[g_i])
        return jnp.zeros((n, n)) if V is None else V

    def fga(theta, Zr, G):
        V, vjp_V = jax.vjp(lambda th: V_of(th, Zr, G), theta)
        L = jnp.linalg.cholesky(V)
        logdetV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        Vi = jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
        ViX = Vi @ Xj
        A = Xj.T @ ViX
        La = jnp.linalg.cholesky(A)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(La)))
        ViXA = ViX @ jax.scipy.linalg.cho_solve((La, True), jnp.eye(p_x))
        Py = Vi @ yj - ViXA @ (ViX.T @ yj)
        f = (n - p_x) * np.log(2.0 * np.pi) + logdetV + logdetA + base["ldRx"] + yj @ Py
        # adjoint: P - Py Py' = V^-1 - [ViXA, Py] [ViX, Py]'
        g = vjp_V(Vi - jnp.concatenate([ViXA, Py[:, None]], axis=1)
                  @ jnp.concatenate([ViX, Py[:, None]], axis=1).T)[0]
        th_terms, th_res, _ = split_theta(theta, terms, res)
        blocks = []
        for k, tm in enumerate(terms):
            if G[k] is not None:
                c = jnp.exp(2.0 * th_terms[k][0])
                col = 2.0 * c * (G[k] @ Py[rows_of[k]])
                blocks.append(jnp.zeros((n, 1)).at[rows_of[k], 0].set(col))
            else:
                blocks.append(term_columns(th_terms[k], tm, Zr[k], rows_of[k], Py))
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            blocks.append(section_columns(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(blocks, axis=1)
        PQ = Vi @ Q - ViXA @ (ViX.T @ Q)
        AI = Q.T @ PQ
        return f, g, 0.5 * (AI + AI.T)

    # Incidences and Gram matrices as arguments (see model.make_neg2).
    _fga = jax.jit(fga)
    Zr_dev, G_dev = Zr, G

    def fga_call(theta):
        return _fga(theta, Zr_dev, G_dev)
    return fga_call


# ==============================================================================
# AI-REML in the variance scale (optimizer = "ai")
# ==============================================================================
# Each Sigma is parametrised by its Cholesky factor with a log diagonal: V is
# not linear in theta, and a scoring step taken in theta is not asreml's.
# asreml works on the entries of Sigma (phi), where V is linear and the AI
# matrix is the exact average information; hence a few tens of iterations
# where AI in theta crawls. theta = theta(phi) is a smooth change of
# variables; with J = d theta / d phi,
#     g_phi = J' g_theta,      AI_phi = J' AI_theta J
# (AI transforms as a Fisher information: exact). fga is kept and only the
# step is taken in phi. iid, diag and us structures go to the variance scale;
# the other parameters (fa loadings, rho, ranges, rr...) stay in theta.
#
# Admissible region: a candidate is projected (eigenvalues of a us block
# truncated at us_floor times the largest, then theta bounded to
# [floor, ceil] and phi recomputed from theta), so a full step can reach a
# boundary (zero variance, correlation of 1).
def variance_scale_blocks(terms, res):
    """Layout of theta for the variance scale: [(start, length, kind, t)],
    kind in {"iid", "diag", "us", "id"} ("id": left in theta; t = -1 marks
    level parameters). fa: loadings in theta, specific variances psi^2 in the
    variance scale (in log psi, a variance pushed to the floor never comes
    back, as for diag). Returns (blocks, p)."""
    blocks, o = [], 0

    def sigma(o, st, t, rank):
        ns = n_params(st, t, rank)
        if not ns:
            return []
        if st == "fa":
            nl = ns - int(t)
            return [(o, nl, "id", 0), (o + nl, int(t), "diag", int(t))]
        kind = st if st in ("iid", "diag", "us") else "id"
        return [(o, ns, kind, int(t))]

    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        p = term_n_params(tm)
        blocks += sigma(o, tm["struct"], tm["t"], tm["rank"])
        if p > ns:
            blocks.append((o + ns, p - ns, "id", -1))
        o += p
    for sec in res_sections(res, 0):
        p = sec_n_params(sec)
        if p == 0:
            continue
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        blocks += sigma(o, sec["struct"], sec["t"], sec["rank"])
        if p > ns:
            blocks.append((o + ns, p - ns, "id", -1))
        o += p
    return blocks, o


def _tril(t):
    return np.tril_indices(t)            # row by row, as structures._tril_indices


def _theta_to_phi(theta, blocks, xp=jnp):
    out = []
    for (o, ln, kind, t) in blocks:
        th = theta[o:o + ln]
        if kind in ("iid", "diag"):
            out.append(xp.exp(2.0 * th))
        elif kind == "us":
            ii, jj = _tril(t)
            L = xp.zeros((t, t))
            if xp is jnp:
                L = L.at[ii, jj].set(th)
                L = L.at[np.arange(t), np.arange(t)].set(xp.exp(xp.diagonal(L)))
            else:
                L = L.copy(); L[ii, jj] = th
                L[np.arange(t), np.arange(t)] = np.exp(np.diagonal(L))
            S = L @ L.T
            out.append(S[ii, jj])
        else:
            out.append(th)
    return xp.concatenate(out) if out else xp.zeros(0)


def _phi_to_theta(phi, blocks, xp=jnp):
    out = []
    for (o, ln, kind, t) in blocks:
        ph = phi[o:o + ln]
        if kind in ("iid", "diag"):
            out.append(0.5 * xp.log(ph))
        elif kind == "us":
            ii, jj = _tril(t)
            if xp is jnp:
                S = jnp.zeros((t, t)).at[ii, jj].set(ph).at[jj, ii].set(ph)
                L = jnp.linalg.cholesky(S)
                th = L[ii, jj]
                th = jnp.where(ii == jj, jnp.log(jnp.where(ii == jj, th, 1.0)), th)
            else:
                S = np.zeros((t, t)); S[ii, jj] = ph; S[jj, ii] = ph
                L = np.linalg.cholesky(S)
                th = L[ii, jj].copy()
                d = ii == jj
                th[d] = np.log(th[d])
            out.append(th)
        else:
            out.append(ph)
    return xp.concatenate(out) if out else xp.zeros(0)


def ai_reml_var(fga, th0, bornes, terms, res, maxiter=200, tol_dec=1e-6, verbose=True,
                counter=None, us_floor=1e-10, fun_sc=None, f_only=None):
    """AI-REML with the step taken in the variance scale.

    Stops when the decrement g' AI^-1 g / 2 falls under tol_dec (units of
    -2logL); after three gains in a row under 1e-6 (optimum on the boundary:
    the step points outwards and the decrement stays large); or when no step
    is admissible (declared failure; fit_reml then falls back to L-BFGS-B and
    Newton polishing). counter: the dict whose "n_eval" counts the
    evaluations; fun_sc: the scaled objective for the L-BFGS-B salvos; f_only:
    the value alone, for the step trials. Returns (theta, info)."""
    return _AIRemlVar(fga, th0, bornes, terms, res, tol_dec, verbose, counter,
                      us_floor, fun_sc, f_only).run(maxiter)


class _AIRemlVar:
    """State of one AI-REML run in the variance scale.

    Each iteration: the free set (_free_set), the reduced system on the free
    parameters and the faces of the us blocks of incomplete rank
    (_reduced_system), the Newton step and its decrement, then the search for
    an admissible step (_search: half steps, then Levenberg-Marquardt damping)
    and, when it fails or the loop stalls, an L-BFGS-B salvo (_salvo)."""

    def __init__(self, fga, th0, bornes, terms, res, tol_dec, verbose, counter, us_floor,
                 fun_sc, f_only):
        self.fga, self.tol_dec, self.verbose, self.counter = fga, tol_dec, verbose, counter
        self.us_floor, self.fun_sc, self.f_only = us_floor, fun_sc, f_only
        blocks, p = variance_scale_blocks(terms, res)
        self.blocks, self.p = blocks, p
        self.lo = np.array([b[0] for b in bornes], dtype=np.float64)
        self.hi = np.array([b[1] for b in bornes], dtype=np.float64)
        if p != len(self.lo):
            raise ValueError("ai: %d parameters split for %d bounds" % (p, len(self.lo)))
        self.is_var = np.zeros(p, dtype=bool)
        self.is_level = np.zeros(p, dtype=bool)
        for (o, ln, kind, t) in blocks:
            if kind == "id" and t == -1:
                self.is_level[o:o + ln] = True
        for (o, ln, kind, t) in blocks:
            if kind in ("iid", "diag"):
                self.is_var[o:o + ln] = True
            elif kind == "us":
                ii, jj = _tril(t)
                self.is_var[o:o + ln] = ii == jj
        self.jac = jax.jit(jax.jacfwd(lambda ph: _phi_to_theta(ph, blocks)))
        self.theta = np.clip(np.asarray(th0, dtype=np.float64), self.lo, self.hi)
        self.phi = _theta_to_phi(self.theta, blocks, xp=np)
        self.f, self.g_t, self.A_t = self.evaluate(self.theta)
        self.n_it = 0
        self.n_salvos = 0
        self.n_clamped = np.zeros(p, dtype=np.int64)   # accepted steps in a row where the clamp acted
        self.mu_mem = None                             # LM damping kept from the previous iteration
        self.last_clamped = np.zeros(p, dtype=bool)

    # ------------------------------------------------------------- basic steps
    def theta_of(self, phi):
        return np.clip(_phi_to_theta(phi, self.blocks, xp=np), self.lo, self.hi)

    def project(self, phi):
        """Nearest admissible phi, then the round trip through bounded theta;
        returns (phi, theta) or None."""
        phi = np.array(phi, dtype=np.float64)
        for (o, ln, kind, t) in self.blocks:
            if kind in ("iid", "diag"):
                phi[o:o + ln] = np.maximum(phi[o:o + ln], np.exp(2.0 * self.lo[o:o + ln]))
            elif kind == "us":
                ii, jj = _tril(t)
                S = np.zeros((t, t)); S[ii, jj] = phi[o:o + ln]; S[jj, ii] = phi[o:o + ln]
                w, U = np.linalg.eigh(S)
                if w[-1] <= 0:
                    return None
                floor_ = self.us_floor * w[-1]
                if w[0] < floor_:
                    S = (U * np.maximum(w, floor_)) @ U.T
                    phi[o:o + ln] = S[ii, jj]
        th = self.theta_of(phi)
        return _theta_to_phi(th, self.blocks, xp=np), th

    def evaluate(self, th):
        if self.counter is not None:
            self.counter["n_eval"] += 1
        f_, g_, A_ = self.fga(jnp.asarray(th))
        return float(f_), np.asarray(g_, dtype=np.float64), np.asarray(A_, dtype=np.float64)

    def _salvo(self, why):
        """SALVO_ITER L-BFGS-B iterations from theta; True when -2logL drops."""
        if self.fun_sc is None or self.n_salvos >= SALVOS_MAX:
            return False
        from scipy.optimize import minimize
        self.n_salvos += 1
        r = minimize(self.fun_sc, self.theta, jac=True, method="L-BFGS-B",
                     bounds=list(zip(self.lo, self.hi)),
                     options=dict(maxiter=SALVO_ITER, maxfun=3 * SALVO_ITER,
                                  ftol=1e-15, gtol=1e-12))
        th_n = np.clip(np.asarray(r.x, dtype=np.float64), self.lo, self.hi)
        f_n, g_n, A_n = self.evaluate(th_n)
        self.n_it += int(r.nit)
        gain_ = self.f - f_n
        if self.verbose:
            print("  [ai] %s: L-BFGS-B salvo %d/%d (%d iterations), gain %.3e"
                  % (why, self.n_salvos, SALVOS_MAX, int(r.nit), gain_), flush=True)
        if not (np.isfinite(f_n) and gain_ > 1e-10 * max(1.0, abs(self.f))):
            return False
        self.theta, self.f, self.g_t, self.A_t = th_n, f_n, g_n, A_n
        self.phi = _theta_to_phi(self.theta, self.blocks, xp=np)
        return True

    # --------------------------------------------------------------- iteration
    def _free_set(self, g):
        """Active set, as in L-BFGS-B: a parameter at its bound pushing outwards
        leaves the step (in phi the gradient of a variance has the sign of
        theta's). A us block with a variance at the floor stays free; the
        projection on the cone handles it."""
        theta, lo, hi = self.theta, self.lo, self.hi
        low = (theta <= lo + 1e-12) & (g > 0)
        high = (theta >= hi - 1e-12) & (g < 0)
        fixed = (hi - lo) <= 0
        free = ~(low | high | fixed)
        for (o, ln, kind, t) in self.blocks:
            if kind == "us":
                free[o:o + ln] = ~fixed[o:o + ln]
        return free

    def _reduced_system(self, g, A, free):
        """Basis T of the step, reduced AI and gradient, and the faces.

        Active set on the cone: a us block with an eigenvalue at the floor
        whose gradient pushes outwards along its null eigenvectors
        (v' dF/dSigma v > 0) is restricted to its face, Sigma = U S U', U
        orthogonal to the null directions; V stays linear in S, so the face AI
        T' AI T is still exact. The face turns: its tangent space also has the
        directions U B V0' + V0 B' U' that rotate the image of Sigma. Along a
        rotation Sigma follows the exact retraction
            (U S + V0 B')(S^-1)(S U' + B V0'),
        whose second-order term adds 2 kron(S^-1, V0' G V0) to the curvature
        (G = dF/dSigma). The face is partial: only the null directions where the
        gradient pushes outwards (eigenvalues > 0 of V0' G V0) stay blocked;
        the others join the image, where their variance can rise. The face
        threshold is 100 times the projection floor (at least 1e-8), so that a
        projected eigenvalue does not fall on either side by rounding
        (dev/notes.md)."""
        p, phi = self.p, self.phi
        cols, n_faces, faces, n_col = [], 0, [], 0
        for (o, ln, kind, t) in self.blocks:
            idx = np.arange(o, o + ln)
            M = None
            if kind == "us" and t >= 2:
                ii, jj = _tril(t)
                S = np.zeros((t, t)); S[ii, jj] = phi[o:o + ln]; S[jj, ii] = phi[o:o + ln]
                w_s, U_s = np.linalg.eigh(S)
                null = w_s <= max(1e-8, 100.0 * self.us_floor) * w_s[-1]
                if null.any() and not null.all():
                    gb = g[o:o + ln]
                    Gm = np.zeros((t, t)); Gm[ii, jj] = np.where(ii == jj, gb, 0.5 * gb)
                    Gm = Gm + np.tril(Gm, -1).T
                    V0 = U_s[:, null]
                    gam, Qg = np.linalg.eigh(V0.T @ Gm @ V0)
                    act = gam > 0
                    if act.any():
                        V0a = V0 @ Qg[:, act]
                        Ur = np.concatenate([U_s[:, ~null], V0 @ Qg[:, ~act]], axis=1)
                        # r1 vectors of the image proper, then the released
                        # directions (variance at the floor). Rotations act on the
                        # r1 first only: towards a released direction the
                        # curvature 2/s (s ~ floor) is meaningless.
                        r1 = int((~null).sum())
                        r = Ur.shape[1]; m0 = t - r
                        Sr = Ur.T @ S @ Ur
                        Sr = 0.5 * (Sr + Sr.T)
                        Srinv = np.linalg.inv(Sr[:r1, :r1])
                        ik, jk = np.tril_indices(r)
                        M = np.zeros((p, len(ik) + r1 * m0))
                        for c, (k, l) in enumerate(zip(ik, jk)):
                            E = np.zeros((r, r)); E[k, l] = 1.0; E[l, k] = 1.0
                            M[o:o + ln, c] = (Ur @ E @ Ur.T)[ii, jj]
                        c0 = len(ik)
                        for k in range(r1):
                            for m in range(m0):
                                D = np.outer(Ur[:, k], V0a[:, m])
                                M[o:o + ln, c0 + k * m0 + m] = (D + D.T)[ii, jj]
                        faces.append(dict(o=o, ln=ln, ii=ii, jj=jj, r=r, r1=r1, m0=m0, V0=V0a,
                                          Ur=Ur, Sr=Sr, Srinv=0.5 * (Srinv + Srinv.T),
                                          Gnn=np.diag(gam[act]), c0=n_col + c0))
                        n_faces += 1
            if M is not None:
                cols.append(M)
                n_col += M.shape[1]
            else:
                lib = idx[free[idx]]
                if len(lib):
                    E = np.zeros((p, len(lib))); E[lib, np.arange(len(lib))] = 1.0
                    cols.append(E)
                    n_col += len(lib)
        T = np.concatenate(cols, axis=1)
        Af, gf = T.T @ A @ T, T.T @ g
        for fc in faces:
            sl = slice(fc["c0"], fc["c0"] + fc["r1"] * fc["m0"])
            Af[sl, sl] += 2.0 * np.kron(fc["Srinv"], fc["Gnn"])
        return T, Af, gf, faces, n_faces

    @staticmethod
    def _dphi_of(xi, T, faces):
        """Reduced step -> step in phi. On a face, Sigma follows the exact
        retraction (U S' + V0 B') S'^-1 (S' U' + B V0'), S' = S + A: the rank
        stays r."""
        d = T @ xi
        for fc in faces:
            r, r1, m0 = fc["r"], fc["r1"], fc["m0"]
            c_a = fc["c0"] - r * (r + 1) // 2
            ik, jk = np.tril_indices(r)
            Am = np.zeros((r, r)); Am[ik, jk] = xi[c_a:fc["c0"]]; Am[jk, ik] = xi[c_a:fc["c0"]]
            Sp = fc["Sr"] + Am
            Bm = np.zeros((r, m0))
            Bm[:r1] = xi[fc["c0"]:fc["c0"] + r1 * m0].reshape(r1, m0)
            try:
                Lp = np.linalg.cholesky(Sp)
            except np.linalg.LinAlgError:
                continue                       # step too long: linear projection
            F = fc["Ur"] @ Lp + fc["V0"] @ np.linalg.solve(Lp, Bm).T
            S_new = F @ F.T
            S_old = fc["Ur"] @ fc["Sr"] @ fc["Ur"].T
            o, ln = fc["o"], fc["ln"]
            d[o:o + ln] = (S_new - S_old)[fc["ii"], fc["jj"]]
        return d

    def try_step(self, dphi, g, A):
        """Candidate phi + dphi, limited and projected; (phi, theta, f, g, AI)
        when admissible, None otherwise.

        A variance is divided by at most 1/REDUCTION_MAX per step (asreml's
        rule), except one clamped on N_CLAMP_TO_FLOOR accepted steps in a row,
        which really goes to zero (Heywood case). A level parameter moves by at
        most LEVEL_STEP_MAX. Both limits also decide the optimum reached when
        the likelihood has several (dev/notes.md). The value alone (f_only),
        when given, screens the candidate before the full evaluation."""
        phi, f = self.phi, self.f
        dphi = np.array(dphi, dtype=np.float64)
        dphi[self.is_level] = np.clip(dphi[self.is_level], -LEVEL_STEP_MAX, LEVEL_STEP_MAX)
        target = phi + dphi
        free_low = self.is_var & (self.n_clamped >= N_CLAMP_TO_FLOOR)
        bound_low = np.where(free_low, -np.inf, REDUCTION_MAX * phi)
        clamped = self.is_var & (target < bound_low)
        target[self.is_var] = np.maximum(target[self.is_var], bound_low[self.is_var])
        self.last_clamped[:] = clamped
        pr = self.project(target)
        if pr is None:
            return None
        phi_c, th_c = pr
        # Ratio test (option, REMLAX_AI_RATIO > 0): the actual gain must be at
        # least RATIO_MIN of the gain predicted by the quadratic model of the
        # AI; near the optimum (predicted gain under 1e-6) any decrease passes.
        d_ = phi_c - phi
        pred = -float(g @ d_ + 0.5 * d_ @ A @ d_)

        def admissible(f_c):
            if not (np.isfinite(f_c) and f_c <= f + 1e-10 * max(1.0, abs(f))):
                return False
            if RATIO_MIN > 0 and pred > 1e-6:
                return (f - f_c) >= RATIO_MIN * pred
            return True
        if self.f_only is not None:
            if self.counter is not None:
                self.counter["n_eval"] += 1
            if not admissible(float(self.f_only(th_c))):
                return None
        f_c, g_c, A_c = self.evaluate(th_c)
        if admissible(f_c):
            return (phi_c, th_c, f_c, g_c, A_c)
        return None

    def _search(self, step_l, step, w, Vv, gf, T, faces, g, A):
        """An admissible step: the Newton step with up to N_HALF_STEPS halvings
        (or, after a damped iteration, the full Newton step once), then
        Levenberg-Marquardt damping (AI + mu I), mu kept from one iteration to
        the next while it serves. Returns (accepted, alpha, mu)."""
        alpha, acc, mu = 1.0, None, None
        scale = max(abs(w).max(), 1e-12)

        def step_lm(mu_):
            return self._dphi_of(-(Vv @ ((Vv.T @ gf) / (np.maximum(w, 0.0) + mu_))), T, faces)

        if self.mu_mem is None:
            for h in range(N_HALF_STEPS):
                acc = self.try_step(step if alpha == 1.0 else self._dphi_of(alpha * step_l, T, faces),
                                    g, A)
                if acc is not None:
                    break
                alpha *= 0.5
        else:
            acc = self.try_step(step, g, A)
        if acc is None:
            k0 = 0 if self.mu_mem is None else max(0, int(np.floor(np.log10(self.mu_mem / scale))) + 6)
            for k in range(k0, 14):
                mu = scale * 10.0 ** (k - 6)
                acc = self.try_step(step_lm(mu), g, A)
                if acc is not None:
                    alpha = 1.0
                    break
            if acc is None and self.mu_mem is not None:
                # The kept mu no longer serves: back to the Newton step.
                mu, self.mu_mem = None, None
                alpha = 1.0
                for h in range(N_HALF_STEPS):
                    acc = self.try_step(step if alpha == 1.0 else self._dphi_of(alpha * step_l, T, faces),
                                        g, A)
                    if acc is not None:
                        break
                    alpha *= 0.5
        # mu kept for the next iteration, divided by 10 at each success,
        # forgotten when negligible against the AI.
        if acc is not None and mu is not None:
            self.mu_mem = mu / 10.0 if mu / 10.0 > 1e-6 * scale else None
        elif acc is not None:
            self.mu_mem = None
        return acc, alpha, mu

    def run(self, maxiter):
        msg, ok, small = "maxiter reached", False, 0
        t0 = time.time()
        for it in range(int(maxiter)):
            if not (np.isfinite(self.f) and np.all(np.isfinite(self.A_t))):
                msg = "value or AI matrix not finite"
                break
            J = np.asarray(self.jac(jnp.asarray(self.phi)), dtype=np.float64)
            g = J.T @ self.g_t
            A = J.T @ self.A_t @ J
            free = self._free_set(g)
            if not free.any():
                msg, ok = "all parameters are at their bound", True
                break
            T, Af, gf, faces, n_faces = self._reduced_system(g, A, free)
            w, Vv = np.linalg.eigh(0.5 * (Af + Af.T))
            lam = max(1e-12, 1e-10 * abs(w).max())
            wr = np.where(w > lam, w, lam)
            step_l = -(Vv @ ((Vv.T @ gf) / wr))
            dec = -0.5 * float(gf @ step_l)
            if dec < self.tol_dec:
                if n_faces:
                    # KKT point on the edge of the cone: stationary on the whole
                    # rank-r manifold (rotations included), gradient pushing
                    # outwards along the null directions.
                    msg, ok = ("AI decrement under the threshold, optimum on the edge of the cone "
                               "(%d us block(s) of incomplete rank)" % n_faces), True
                else:
                    msg, ok = "AI decrement under the threshold", True
                break
            step = self._dphi_of(step_l, T, faces)
            acc, alpha, mu = self._search(step_l, step, w, Vv, gf, T, faces, g, A)
            if acc is None:
                if dec >= SUCCESS_DECREMENT and self._salvo("no admissible step (decrement %.1e)" % dec):
                    small = 0
                    continue
                msg, ok = ("no admissible step (decrement %.1e)" % dec,
                           dec < SUCCESS_DECREMENT)
                break
            gain = self.f - acc[2]
            self.phi, self.theta, self.f, self.g_t, self.A_t = acc
            self.n_it += 1
            self.n_clamped = np.where(self.last_clamped, self.n_clamped + 1, 0)
            if self.verbose:
                print("  aiv  %4d | -2logL %14.6f | gain %.3e | decrement %.3e | %s | %6.1f s%s"
                      % (it + 1, self.f, gain, dec, ("step %.3g" % alpha) if mu is None else ("lm mu %.1e" % mu),
                         time.time() - t0,
                         "".join(" | face rank %d/%d" % (fc["r"], fc["r"] + fc["m0"]) for fc in faces)),
                      flush=True)
            small = small + 1 if abs(gain) < 1e-6 else 0
            if small >= 3 and dec >= SUCCESS_DECREMENT and self._salvo("stagnation (decrement %.1e)" % dec):
                small = 0
                continue
            if small >= 3:
                # Optimum on the boundary when the decrement is already small;
                # otherwise the loop crawls and failure is declared, so that
                # fit_reml goes on to Newton polishing on the true Hessian.
                ok = dec < SUCCESS_DECREMENT
                msg = ("three gains in a row under 1e-6 (optimum on the boundary)" if ok else
                       "stagnation: three gains under 1e-6 for a decrement of %.1e" % dec)
                break
        return self.theta, dict(nit=self.n_it, success=ok, message="AI-REML (variances): %s" % msg)
