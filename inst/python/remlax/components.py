"""Variance components of a fit: names, values and the component table.

component_names() fixes the numbering V1, V2, ... used by vpredict;
components_from_theta() maps theta to the components in that order, in JAX so
that the delta method can differentiate it; varcomp_table() gives each
component with its delta-method standard error and a constraint code, in the
manner of summary.asreml.

The standard errors assume the Hessian invertible on the free subspace. A
parameter at an active bound has none, and a direction of zero curvature makes
the delta method silent: NaN is returned, never an invented number.
"""
from . import _x64  # noqa: F401
import re

import jax
import jax.numpy as jnp
import numpy as np

from .layout import level_kind, res_sections, sec_n_params, split_theta
from .structures import build_sigma, n_params
from .levels import n_level_params, MTRN_PARAMS



def n_level(tm):
    """Number of parameters of the structure between levels of a term or
    section."""
    return n_level_params(level_kind(tm), tm.get("lvl_order", 0), tm.get("lvl_parts"),
                          opts=tm.get("lvl_opts"))


def _matrix_names(prefix, t):
    """prefix for t = 1, otherwise prefix[i,j] over the lower triangle row by
    row."""
    if t == 1:
        return [prefix]
    return ["%s[%d,%d]" % (prefix, i + 1, j + 1) for i in range(t) for j in range(i + 1)]


def _level_names(prefix, kind, nl):
    return ["%s!%s%s" % (prefix, kind, "" if nl == 1 else str(k + 1)) for k in range(nl)]


def component_names(terms, res, lag=None):
    """Names of the components, in the order vpredict numbers them (V1, V2,
    ...): the terms then the residual, each matrix by its lower triangle row
    by row, then its level parameters. The order is fixed and documented:
    renumbering silently would break every vpredict expression. With a
    spatial lag, the free entries of Psi (Psi[receiver<-emitter], receiver by
    receiver) come last, so the numbers of the variance components do not
    move."""
    names = []
    for tm in terms:
        names += _matrix_names("%s" % tm["name"], tm["t"])
        names += _level_names(tm["name"], level_kind(tm), n_level(tm))
    secs = res_sections(res, 0)
    for sec in secs:
        pre = sec.get("name", "residual") if len(secs) > 1 else "residual"
        names += _matrix_names(pre, sec["t"])
        names += _level_names(pre, sec.get("lvl", "id"), n_level(sec))
    if lag is not None:
        from .lag import lag_component_names
        names.extend(lag_component_names(lag))
    return names


# Level parameters returned as correlations tanh(theta).
_LEVEL_TANH = ("ar1", "ar2", "ar3", "sar", "ma1", "ma2", "arma", "corb", "ar1ar1")
# Metric families: C uses |tanh(theta)| raised to a distance, so the sign of
# theta is not identified and |tanh(theta)| is returned.
_LEVEL_ABS_TANH = ("exp", "gau", "iexp", "igau", "ieuc", "aexp", "agau")
# Ranges: the parameter is a distance, exp(theta).
_LEVEL_RANGE = ("sph", "cir", "lvr", "ilv")


def _level_natural(th_lvl, tm):
    """Level parameters on their natural scale, in JAX (differentiable): a
    vpredict expression such as "V3" names the correlation, not the internal
    theta that produces it."""
    kind = level_kind(tm)
    nl = n_level(tm)
    if nl == 0:
        return []
    if kind in _LEVEL_TANH:
        return [jnp.tanh(th_lvl[j]) for j in range(nl)]
    if kind in _LEVEL_ABS_TANH:
        return [jnp.abs(jnp.tanh(th_lvl[j])) for j in range(nl)]
    if kind == "cor":
        # Same rescaling as _acf: (-1, 1) -> (-1/(q-1), 1).
        q = int(tm.get("q") or tm.get("n_unit") or 1)
        lo = -1.0 / max(q - 1, 1)
        return [lo + (jnp.tanh(th_lvl[0]) + 1.0) * 0.5 * (1.0 - lo)]
    if kind in _LEVEL_RANGE:
        return [jnp.exp(th_lvl[j]) for j in range(nl)]
    if kind == "mtrn":
        o, out, k = tm.get("lvl_opts") or {}, [], 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5:
                out.append(th_lvl[k] if p == "alpha" else jnp.exp(th_lvl[k]))
                k += 1
        return out
    return [th_lvl[j] for j in range(nl)]      # corg, own: raw parameters


def _append_block(out, S, t, th_lvl, tm):
    """Entries of S (lower triangle row by row, one extraction) and the level
    parameters on their natural scale."""
    if t == 1:
        out.append(jnp.atleast_1d(S[0, 0]))
    else:
        ii, jj = np.tril_indices(t)
        out.append(S[ii, jj])
    nv = _level_natural(th_lvl, tm)
    if len(nv):
        out.append(jnp.stack([jnp.asarray(v_) for v_ in nv]))


def components_from_theta(theta, terms, res, lag=None):
    """Vector of the components (variances, covariances, level parameters,
    entries of Psi) in the order of component_names(). Written in JAX end to
    end: the delta method differentiates it."""
    th_terms, th_res, _ = split_theta(theta, terms, res)
    out = []
    for k, tm in enumerate(terms):
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        S = build_sigma(th_terms[k][:ns], tm["struct"], tm["t"], tm["rank"])
        _append_block(out, S, tm["t"], th_terms[k][ns:], tm)
    o = 0
    for sec in res_sections(res, 0):
        p_s = sec_n_params(sec)
        th_s = th_res[o:o + p_s]
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        R = build_sigma(th_s[:ns], sec["struct"], sec["t"], sec["rank"]) \
            if sec["struct"] != "iid" else jnp.exp(2 * th_s[0]) * jnp.eye(sec["t"])
        _append_block(out, R, sec["t"], th_s[ns:], sec)
        o += p_s
    if lag is not None:
        # Psi in units of W: theta_lag / s (lag.py). Linear, so the delta
        # method is exact for these components.
        nl = int(np.asarray(lag["mask"], dtype=bool).sum())
        if nl:
            th_lag = theta[len(theta) - nl:]
            out.append(jnp.asarray(th_lag[:nl]) / float(lag["scale"]))
    return jnp.concatenate(out) if out else jnp.zeros(0)


def varcomp_table(theta, H, terms, res, floor=-12.0, ceil=12.0, fixed_idx=None, tol=1e-7,
                  lag=None, alias=None):
    """Table of the variance components, in the manner of summary.asreml.

    For each component (in the order V1, V2, ... of vpredict): its value, its
    delta-method standard error, and a constraint code:
      F  depends only on fixed parameters;
      B  depends on a parameter at a bound, or on a nearly flat direction;
      P  variance (diagonal of a Sigma, or a single-trait term);
      U  covariance or level parameter (correlation, range...);
      N  entry not identified separately (alias, model.aliased_entries): only
         its sum with the entries of the same pattern is. Those sums are
         appended at the end of the table, named "a+b", with their standard
         error.
    The Jacobian is exact (JAX, compiled); the covariance of theta is 2 H^-1
    on the positive-curvature directions of the free Hessian. A standard
    error is NaN for F and B."""
    theta = np.asarray(theta, dtype=float)
    p = len(theta)
    comp_fn = lambda th: components_from_theta(th, terms, res, lag=lag)  # noqa: E731
    vals = np.asarray(jax.jit(comp_fn)(jnp.asarray(theta)))
    J = np.asarray(jax.jit(jax.jacfwd(comp_fn))(jnp.asarray(theta))).reshape(len(vals), p)
    names = component_names(terms, res, lag=lag)
    fixed = np.zeros(p, dtype=bool)
    if fixed_idx:
        fixed[np.asarray(fixed_idx, dtype=int)] = True
    bound = (theta <= floor + tol) | (theta >= ceil - tol)
    free = ~fixed & ~bound
    Vth, Uz = _theta_cov_positive(H, free, p, alias)
    idx_free = np.where(free)[0]
    out = []
    for k in range(len(vals)):
        code = _constraint_code(J[k], names[k], fixed, bound, Uz, idx_free)
        se = float("nan")
        if Vth is not None and code not in ("F", "B"):
            se = float(np.sqrt(max(J[k] @ Vth @ J[k], 0.0)))
        out.append(dict(i=k + 1, name=names[k], value=float(vals[k]), se=se, constraint=code))
    _add_alias_sums(out, J, vals, names, fixed, bound, Vth, alias)
    return out


def _theta_cov_positive(H, free, p, alias):
    """Covariance 2 H^-1 of theta on the positive-curvature directions of the
    free Hessian (p x p, zero outside the free block), and the flat directions
    Uz set aside (or None). A variance tending to zero without reaching the
    floor leaves a nearly flat direction; inverting it would make every
    standard error absurd, so it is set aside, as asreml does. The known flat
    directions of aliased entries are set aside even when finite-difference
    noise puts them just above the threshold."""
    Vth, Uz = None, None
    if H is not None and free.sum() > 0:
        Hf = np.asarray(H)[np.ix_(free, free)]
        try:
            w, U = np.linalg.eigh(0.5 * (Hf + Hf.T))
            keep = w > 1e-8 * max(abs(w).max(), 1.0)
            n_flat = int(sum(len(g["entries"]) - 1 for g in (alias or [])))
            if n_flat:
                keep[np.argsort(np.abs(w))[:min(n_flat, len(w))]] = False
            if keep.any():
                Vth = np.zeros((p, p))
                Up = U[:, keep]
                Vth[np.ix_(free, free)] = 2.0 * (Up @ np.diag(1.0 / w[keep]) @ Up.T)
            Uz = U[:, ~keep] if (~keep).any() else None
        except np.linalg.LinAlgError:
            Vth = None
    return Vth, Uz


def _constraint_code(Jk, name, fixed, bound, Uz, idx_free):
    """F, B, P or U for one component (see varcomp_table). A component whose
    gradient lies more than 5 % in the flat directions counts as at a bound.
    An entry of Psi is not a variance: U."""
    dep = np.abs(Jk) > 1e-12 * max(1.0, np.abs(Jk).max())
    flat = False
    if Uz is not None:
        jl = Jk[idx_free]
        n2 = float(jl @ jl)
        flat = n2 > 0 and float(np.sum((jl @ Uz) ** 2)) / n2 > 0.05
    if dep.any() and np.all(fixed[dep]):
        return "F"
    if np.any(bound[dep]) or flat:
        return "B"
    m = re.search(r"\[(\d+),(\d+)\]$", name)
    return "U" if ("!" in name or name.startswith("Psi[")
                   or (m and m.group(1) != m.group(2))) else "P"


def _add_alias_sums(out, J, vals, names, fixed, bound, Vth, alias):
    """Aliased entries: code N and no standard error, then the identified sum
    of each group appended to the table."""
    pos = {d["name"]: k for k, d in enumerate(out)}
    for g in (alias or []):
        idx = [pos.get("%s[%d,%d]" % (g["term"], i + 1, j + 1)) for i, j in g["entries"]]
        if any(k is None for k in idx):
            continue
        for k in idx:
            out[k]["constraint"] = "N"
            out[k]["se"] = float("nan")
        Js = np.sum(J[idx], axis=0)
        dep = np.abs(Js) > 1e-12 * max(1.0, np.abs(Js).max())
        code = ("F" if dep.any() and np.all(fixed[dep]) else
                "B" if np.any(bound[dep]) else "U")
        se = (float(np.sqrt(max(Js @ Vth @ Js, 0.0)))
              if Vth is not None and code == "U" else float("nan"))
        out.append(dict(i=len(out) + 1, name="+".join(names[k] for k in idx),
                        value=float(np.sum(vals[idx])), se=se, constraint=code))
