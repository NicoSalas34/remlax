"""vpredict: functions of the variance components and their standard errors.

An expression uses V1, V2, ... (numbered by components.component_names) and
the usual Python operators, for example ("h2", "V1/(V1+V2)"). Its standard
error comes from the delta method on the Hessian H of -2logL:
var(g) = J' (2 H^-1) J, the factor 2 because H is the Hessian of -2logL and
not of -logL.
"""
from . import _x64  # noqa: F401
import ast

import jax
import jax.numpy as jnp
import numpy as np

from .components import component_names, components_from_theta

VPREDICT_FONCTIONS = ("sqrt", "log", "exp", "abs")


def check_vpredict(expressions, terms, res, lag=None):
    """Checks the vpredict expressions before the fit, so that a component
    absent from the model fails at once with the list of the components.
    Raises ValueError with V1 = name, V2 = name, ..."""
    names = component_names(terms, res, lag=lag)
    allowed = {"V%d" % (i + 1) for i in range(len(names))} | set(VPREDICT_FONCTIONS)
    listing = ", ".join("V%d = %s" % (i + 1, n) for i, n in enumerate(names))
    for name, expr in expressions:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError("vpredict '%s': unreadable expression \"%s\" (%s). "
                             "Components of the model: %s." % (name, expr, e.msg, listing))
        unknown = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - allowed)
        if unknown:
            raise ValueError("vpredict '%s': unknown %s in \"%s\". This model has %d "
                             "component(s): %s. Allowed functions: %s."
                             % (name, ", ".join(unknown), expr, len(names), listing,
                                ", ".join(VPREDICT_FONCTIONS)))


def vpredict(theta, H, terms, res, expressions, free=None, lag=None, alias=None):
    """Values and standard errors of the expressions, with the table of the
    components. free marks the free parameters; alias, the groups of entries
    not identified separately."""
    theta = np.asarray(theta, dtype=float)
    comp_fn = lambda th: components_from_theta(th, terms, res, lag=lag)  # noqa: E731
    comps = np.asarray(comp_fn(jnp.asarray(theta)))
    names = component_names(terms, res, lag=lag)
    if free is None:
        free = np.ones(len(theta), dtype=bool)
    free = np.asarray(free, dtype=bool)
    Vth = _theta_cov_all_positive(H, free, len(theta), alias)
    out = []
    for name, expr in expressions:
        def g(th, expr=expr):
            c = comp_fn(th)
            env = {"V%d" % (i + 1): c[i] for i in range(c.shape[0])}
            env.update({"sqrt": jnp.sqrt, "log": jnp.log, "exp": jnp.exp, "abs": jnp.abs})
            return eval(expr, {"__builtins__": {}}, env)          # noqa: S307
        val = float(g(jnp.asarray(theta)))
        J = np.asarray(jax.grad(g)(jnp.asarray(theta)))
        se = float(np.sqrt(max(J @ Vth @ J, 0.0))) if np.all(np.isfinite(Vth)) else np.nan
        out.append(dict(name=name, expression=expr, value=val, se=se))
    return dict(components=[{"i": i + 1, "name": names[i], "value": float(comps[i])}
                             for i in range(len(names))],
                predictions=out)


def _theta_cov_all_positive(H, free, p, alias):
    """2 H^-1 on the free subspace, the known flat directions of aliased
    entries set aside; all NaN unless every other eigenvalue of the free
    Hessian is positive. An expression that depends on an aliased entry alone
    then gets a meaningless standard error; one of identified components keeps
    its own."""
    Vth = np.full((p, p), np.nan)
    if H is not None and free.sum() > 0:
        Hf = np.asarray(H)[np.ix_(free, free)]
        try:
            w, U = np.linalg.eigh(Hf)
            tol = 1e-8 * max(abs(w).max(), 1.0)
            keep = np.ones(len(w), dtype=bool)
            n_flat = int(sum(len(g["entries"]) - 1 for g in (alias or [])))
            if n_flat:
                keep[np.argsort(np.abs(w))[:min(n_flat, len(w))]] = False
            if np.all(w[keep] > tol):
                Ug, wg = U[:, keep], w[keep]
                Vth = np.zeros((p, p))
                Vth[np.ix_(free, free)] = 2.0 * (Ug @ np.diag(1.0 / wg) @ Ug.T)
        except np.linalg.LinAlgError:
            pass
    return Vth
