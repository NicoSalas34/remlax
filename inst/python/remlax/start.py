"""Starting values of theta.

initial_theta() splits the residual variance of the fixed effects equally
between the terms and the residual, divided by the exposure of each term, and
starts level parameters where the likelihood has a slope. loading_bounds()
gives the box bounds of the fa/rr loadings. _range_sweep() chooses the start of
range parameters (sph, cir, lvr) on a sweep of -2logL.
"""
from . import _x64  # noqa: F401

import numpy as np
from scipy.optimize import minimize

from .layout import level_kind, term_n_params, res_sections, sec_n_params
from .structures import n_params, theta0 as struct_theta0, n_loadings
from .levels import n_level_params


# ------------------------------------------------------------------------------
# Exposure of a term
# ------------------------------------------------------------------------------

def _sum_diag_ZKZ(Z, LK):
    """Sum of the (Z K Z')_ii, K = LK LK', Z sparse (CSR).

    Two routes with the same result: through Z LK, nnz(Z) x q operations; or
    through trace(Z K Z') = sum of G_jl K_jl, G = Z'Z, with K_jl = LK_j . LK_l
    computed at the non-zeros of G only, nnz(G) x q operations. With
    replicates, nnz(G) is much smaller than nnz(Z)."""
    G = (Z.T @ Z).tocoo()
    pas = max(1, int(2 ** 25 // max(LK.shape[1], 1)))
    if G.nnz < Z.nnz:
        tot = 0.0
        for a in range(0, G.nnz, pas):
            r, c = G.row[a:a + pas], G.col[a:a + pas]
            tot += float(np.dot(G.data[a:a + pas], np.einsum("ij,ij->i", LK[r], LK[c])))
        return tot
    tot = 0.0
    for a in range(0, Z.shape[0], pas):
        ZL = Z[a:a + pas] @ LK
        tot += float(np.sum(ZL * ZL))
    return tot


def _exposure(tm, n):
    """k = mean of the (Z K Z')_ii over the loaded (observation, component)
    pairs: the factor by which the variance of a term enters the phenotype.

    Z is built sparse, one row per (observation, component). The mean is taken
    over the loaded pairs, not over n x t: in long format each row loads only
    its own component, and dividing by n x t would underestimate the exposure
    by a factor t. The value is cached in the term (key ("_expo", n)), since
    initial_theta and loading_bounds call it several times per fit."""
    cle = ("_expo", n)
    if cle in tm:
        return tm[cle]
    import scipy.sparse as sp
    t, q = int(tm["t"]), int(tm["q"])
    zi = np.asarray(tm["zi"], dtype=np.int64); zj = np.asarray(tm["zj"], dtype=np.int64)
    Zs = sp.csr_matrix((np.asarray(tm["zx"], dtype=np.float64), (zi * t + zj // q, zj % q)),
                       shape=(n * t, q))
    if tm["LK"] is None:
        tot = float(Zs.multiply(Zs).sum())
    else:
        tot = _sum_diag_ZKZ(Zs, np.asarray(tm["LK"], dtype=np.float64))
    n_loaded = int(np.count_nonzero(np.diff(Zs.indptr)))
    k = tot / max(n_loaded, 1)
    tm[cle] = k if k > 1e-12 else 1.0
    return tm[cle]


def _component_exposure(tm, n, e):
    """For each component c of a term: exposure k_c (mean of (Z_c K Z_c')_ii
    over the rows that load c) and mean variance v_c of the residuals e on
    those rows. Returns (k, v), two vectors of length t; a component with NaN
    or no row takes the mean of the others."""
    import scipy.sparse as sp
    t, q = int(tm["t"]), int(tm["q"])
    zi = np.asarray(tm["zi"], dtype=np.int64); zj = np.asarray(tm["zj"], dtype=np.int64)
    zx = np.asarray(tm["zx"], dtype=np.float64)
    comp = zj // q
    k = np.full(t, np.nan); v = np.full(t, np.nan)
    LK = None if tm.get("LK") is None else np.asarray(tm["LK"], dtype=np.float64)
    for c in range(t):
        m = comp == c
        if not m.any():
            continue
        rows, inv = np.unique(zi[m], return_inverse=True)
        Zc = sp.csr_matrix((zx[m], (inv, zj[m] % q)), shape=(len(rows), q))
        tot = float(Zc.multiply(Zc).sum()) if LK is None else _sum_diag_ZKZ(Zc, LK)
        k[c] = tot / len(rows)
        v[c] = float(np.mean(e[rows] ** 2))
    for x in (k, v):
        ok = np.isfinite(x) & (x > 1e-12)
        x[~ok] = np.mean(x[ok]) if ok.any() else 1.0
    return k, v


# ------------------------------------------------------------------------------
# Bounds of the loadings
# ------------------------------------------------------------------------------

# Bound of the fa and rr loadings, in multiples of the starting standard
# deviation of the term. Loadings are linear parameters on the standard
# deviation scale: the common bound of theta (12, made for log standard
# deviations) let them reach 12 in a few AI steps, where Lambda Lambda' + psi
# becomes rank deficient (dev/notes.md). A hundred standard deviations hinder
# no real optimum.
LOADING_BOUND_SD = 100.0


def loading_bounds(terms, res, y):
    """[(start, length, bound)] of the fa/rr loadings in theta."""
    vy = float(np.var(np.asarray(y))) or 1.0
    part = vy / (len(terms) + 1.0)
    out, o = [], 0
    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        if tm["struct"] in ("fa", "rr"):
            nl = n_loadings(tm["t"], tm["rank"]) if tm["struct"] == "fa" else ns
            B = LOADING_BOUND_SD * float(np.sqrt(part / _exposure(tm, len(y))))
            out.append((o, nl, B))
        o += term_n_params(tm)
    for sec in res_sections(res, len(y)):
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        if sec["struct"] in ("fa", "rr"):
            nl = n_loadings(sec["t"], sec["rank"]) if sec["struct"] == "fa" else ns
            out.append((o, nl, LOADING_BOUND_SD * float(np.sqrt(part))))
        o += sec_n_params(sec)
    return out


# ------------------------------------------------------------------------------
# Covariance parameters
# ------------------------------------------------------------------------------

def _theta0_components(struct, t, rank, var):
    """Start of a diagonal Sigma with variances `var` (one per component);
    structures without a variance per component take their mean."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-8)
    sd = np.sqrt(var)
    if struct == "diag":
        return np.log(sd)
    if struct == "us":
        ii, jj = np.tril_indices(t)
        th = np.zeros(len(ii))
        th[ii == jj] = np.log(sd)
        return th
    if struct == "fa":
        lam = [0.1 * sd[i] for i in range(t) for j in range(min(i + 1, rank))]
        return np.concatenate([np.asarray(lam), np.log(sd)])
    return struct_theta0(struct, t, rank, var=float(np.mean(var)))



def initial_theta(terms, res, y, X=None):
    """Starting theta: the variance split equally between the terms and the
    residual.

    With X, the variance is that of the residuals of the fixed effects, not of
    raw y, which carries the differences between environments or treatments.
    A term with several components (G x E in long format) starts from a
    diagonal Sigma whose variances come from the residuals of the rows of each
    component; a residual with several sections starts from the residual
    variance of each section. Each variance is divided by the exposure k of
    its term: a neighbour component enters the phenotype as sigma^2 k, and k
    reaches 15 to 40 on some designs."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if X is not None and np.size(X):
        X = np.asarray(X, dtype=np.float64)
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        df = max(n - int(np.linalg.matrix_rank(X)), 1)
        e = e * np.sqrt(n / df)                    # unbiased variance
    else:
        e = y - float(np.mean(y))
    vy = float(np.mean(e ** 2)) or 1.0
    nt = len(terms) + 1.0
    th = []
    for tm in terms:
        t = int(tm["t"])
        if X is not None and t > 1 and tm["struct"] in ("diag", "us", "fa"):
            k_c, v_c = _component_exposure(tm, n, e)
            th.append(_theta0_components(tm["struct"], t, tm["rank"], v_c / nt / k_c))
        else:
            k = _exposure(tm, n)
            th.append(struct_theta0(tm["struct"], tm["t"], tm["rank"], var=vy / nt / k))
        kind = level_kind(tm)
        nl = n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                            opts=tm.get("lvl_opts"))
        if nl:
            th.append(_theta0_levels(kind, nl, tm.get("lvl_opts"), tm.get("coord")))
    for sec in res_sections(res, n):
        th.append(_theta0_section(sec, e, vy, nt, n, X is not None))
        nlr = n_level_params(sec.get("lvl", "id"), sec.get("lvl_order", 0),
                             sec.get("lvl_parts"), opts=sec.get("lvl_opts"))
        if nlr:
            th.append(_theta0_levels(sec.get("lvl", "id"), nlr,
                                     sec.get("lvl_opts"), sec.get("coord")))
    return np.concatenate([np.asarray(x, dtype=np.float64) for x in th])


def _theta0_section(sec, e, vy, nt, n, has_X):
    """Start of the covariance parameters of a residual section."""
    rows = np.asarray(sec.get("rows", np.arange(n)), dtype=np.int64)
    ts = int(sec["t"])
    if has_X and len(rows):
        if ts > 1 and sec["struct"] in ("diag", "us", "fa") and sec.get("trait") is not None:
            tr = np.asarray(sec["trait"], dtype=np.int64)
            v_c = np.array([float(np.mean(e[rows[tr == c]] ** 2)) if np.any(tr == c) else vy
                            for c in range(ts)])
            return _theta0_components(sec["struct"], ts, sec["rank"], v_c / nt)
        return struct_theta0(sec["struct"], ts, sec["rank"],
                             var=float(np.mean(e[rows] ** 2)) / nt)
    return struct_theta0(sec["struct"], ts, sec["rank"], var=vy / nt)


# ------------------------------------------------------------------------------
# Level parameters
# ------------------------------------------------------------------------------

RHO_METRIC_KINDS = ("exp", "gau", "iexp", "igau", "ieuc", "aexp", "agau")
RANGE_KINDS = ("sph", "cir", "lvr")


def _typical_step(v):
    """Median gap between consecutive distinct coordinates of an axis. The
    median, not the minimum: two nearly coincident positions would push rho0
    to the clipping point, where the gradient vanishes."""
    d = np.diff(np.unique(np.asarray(v, dtype=float)))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _as_columns(coord):
    c = np.atleast_2d(np.asarray(coord, dtype=float))
    if c.shape[0] == 1 and c.shape[1] > 1:
        c = c.T
    return c


def _theta0_rho_metric(kind, nl, coord=None):
    """Start of the rho^d structures: correlation 0.5 at the nearest neighbour.

    rho0 = 0.5^(1/d0) for exponential kernels, 0.5^(1/d0^2) for Gaussian ones,
    d0 being the median step of the axis (one per axis for aexp and agau, the
    smaller of the two for the isotropic ones). rho0 is bounded to
    [1e-3, 0.99] so that theta = atanh(rho0) keeps a usable gradient; without
    coordinates, rho0 = 0.5."""
    th = np.full(nl, np.arctanh(0.5))
    if coord is None:
        return th
    c = _as_columns(coord)
    pas = [_typical_step(c[:, j]) for j in range(c.shape[1])]
    gaussian = kind in ("gau", "igau", "agau")

    def rho0(d0):
        r = 0.5 ** (1.0 / (d0 ** 2 if gaussian else d0))
        return float(np.clip(r, 1e-3, 0.99))

    if kind in ("aexp", "agau"):
        for j in range(min(nl, len(pas))):
            th[j] = np.arctanh(rho0(pas[j]))
        return th
    th[0] = np.arctanh(rho0(min(pas)))
    return th


def _theta0_levels(kind, nl, opts=None, coord=None):
    """Start of the level parameters.

    - mtrn: the starting values declared by the user (opts init_*), otherwise
      the defaults, phi at a quarter of the coordinate range;
    - rho^d structures: correlation 0.5 at the median step (theta = 0 would put
      rho exactly on the clipping point, where the gradient is zero);
    - range structures: the parameter is a log distance, started at a quarter
      of the coordinate range, where the likelihood still has a slope;
    - other tanh-parametrised structures: zero, an unstructured field."""
    if kind == "mtrn":
        from .levels import MTRN_PARAMS, MTRN_DEFAUT
        o = opts or {}
        th, k = np.zeros(nl), 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) <= 0.5:
                continue
            v = float(o.get("init_" + p, np.nan))
            if not np.isfinite(v):
                v = MTRN_DEFAUT[p]
                if p == "phi" and coord is not None:
                    c = np.atleast_2d(np.asarray(coord, dtype=float))
                    v = max(0.25 * float(np.ptp(c, axis=0).max()), 1e-3)
            th[k] = v if p == "alpha" else np.log(max(v, 1e-8))
            k += 1
        return th
    if kind in RHO_METRIC_KINDS:
        return _theta0_rho_metric(kind, nl, coord)
    if kind not in ("sph", "cir", "lvr", "ilv"):
        return np.zeros(nl)
    extent = 1.0
    if coord is not None:
        c = _as_columns(coord)
        extent = float(max(np.ptp(c, axis=0).max(), 1.0))
    th = np.zeros(nl)
    th[0] = np.log(max(0.25 * extent, 1e-3))
    return th


# ------------------------------------------------------------------------------
# Sweep of the range parameters
# ------------------------------------------------------------------------------

def _range_positions(terms, res, n):
    """Indices in theta of the range parameters, with their coordinates, as
    [(index, coord)]. Follows the layout of initial_theta: for each term the
    structure parameters then the level parameters, then each residual
    section likewise."""
    out, k = [], 0
    for tm in terms:
        k += n_params(tm["struct"], tm["t"], tm["rank"])
        kind = level_kind(tm)
        nl = n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                            opts=tm.get("lvl_opts"))
        if nl and kind in RANGE_KINDS:
            out.append((k, tm.get("coord")))
        k += nl
    for sec in res_sections(res, n):
        k += n_params(sec["struct"], sec["t"], sec["rank"])
        kind = sec.get("lvl", "id")
        nl = n_level_params(kind, sec.get("lvl_order", 0), sec.get("lvl_parts"),
                            opts=sec.get("lvl_opts"))
        if nl and kind in RANGE_KINDS:
            out.append((k, sec.get("coord")))
        k += nl
    return out


def _range_candidates(coord, n_cand=9):
    """Candidate log ranges: the 10 % to 90 % quantiles of the distances
    between pairs of positions. The likelihood of a range kernel is smooth
    between two consecutive distances of the design and has a kink at each, so
    one candidate falls in each region where the kernel changes shape."""
    if coord is None:
        return np.array([])
    c = _as_columns(coord)
    if c.shape[0] > 2000:
        rng = np.random.default_rng(0)
        c = c[rng.choice(c.shape[0], 2000, replace=False)]
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    d = d[np.triu_indices(c.shape[0], 1)]
    d = d[d > 0]
    if d.size == 0:
        return np.array([])
    q = np.quantile(d, np.linspace(0.1, 0.9, n_cand))
    return np.log(np.unique(q))


def _range_sweep(fun_jac, th0, terms, res, n, verbose=False, fixed_idx=None,
                 fun_sc=None, bornes=None, n_garde=3, maxiter_local=10):
    """Start of the range parameters (sph, cir, lvr) chosen on a sweep of
    -2logL, which often has several local maxima in the range.

    For one range parameter at a time, the others at their start, -2logL is
    evaluated at the candidates of _range_candidates and at the current value.
    Ranking candidates at unadjusted variances is not enough, so the n_garde
    best each get a short L-BFGS-B descent (maxiter_local iterations, on the
    scaled objective and under the bounds of the fit), and the best point
    reached is kept. Only used for the automatic start: a theta_init given by
    the caller is respected. Returns (theta, n_eval)."""
    pos = _range_positions(terms, res, n)
    if not pos:
        return th0, 0
    fixes = set(int(i) for i in (fixed_idx or []))
    th = np.array(th0, dtype=np.float64)
    n_eval = 0
    for k, coord in pos:
        if k in fixes:
            continue
        cands = _range_candidates(coord)
        if cands.size == 0:
            continue
        cands = np.append(cands, th[k])
        vals = []
        for c in cands:
            t = th.copy()
            t[k] = c
            try:
                v = float(fun_jac(t)[0])
            except Exception:
                v = np.nan
            n_eval += 1
            vals.append(v if np.isfinite(v) else np.inf)
        vals = np.asarray(vals)
        order = [int(j) for j in np.argsort(vals) if np.isfinite(vals[j])][:max(int(n_garde), 1)]
        if not order:
            continue
        best_th, best_v = th.copy(), vals[-1]
        if fun_sc is None or maxiter_local <= 0:
            j = order[0]
            if vals[j] < best_v - 1e-10:
                best_th[k], best_v = cands[j], vals[j]
        else:
            for j in order:
                t = th.copy()
                t[k] = cands[j]
                try:
                    r = minimize(fun_sc, t, jac=True, method="L-BFGS-B", bounds=bornes,
                                 options=dict(maxiter=int(maxiter_local),
                                              maxfun=20 * int(maxiter_local)))
                    v = float(fun_jac(r.x)[0])
                    n_eval += int(r.nfev) + 1
                except Exception:
                    continue
                if np.isfinite(v) and v < best_v - 1e-10:
                    best_th, best_v = np.asarray(r.x, dtype=np.float64), v
        if verbose and best_v < vals[-1] - 1e-10:
            print("  [range] start moved from %.4g to %.4g (-2logL %.6f -> %.6f)"
                  % (np.exp(th[k]), np.exp(best_th[k]), vals[-1], best_v), flush=True)
        th = best_th
    return th, n_eval
