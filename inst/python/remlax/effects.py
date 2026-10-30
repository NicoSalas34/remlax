"""Fixed and random effects at a given theta, on the dense engine.

dense_effects() serves the end of a dense fit (fit.py) and every draw of the
Bayesian sampler (bayes.py), which takes the posterior moments of beta and of
the random effects from it.
"""
from . import _x64  # noqa: F401

import jax.numpy as jnp
import numpy as np

from .layout import level_kind, split_theta
from .model import assemble_V
from .structures import build_sigma, n_params
from .levels import level_chol


def dense_effects(theta, terms, Zs, res, y_fit, X, blups=True, pev=False,
                  V=None, z_de=None):
    """beta, vbeta, Vi, P y, BLUPs and PEV at theta; a dict with the keys of
    fit_reml.

    V and z_de serve the low-memory mode (econome.py): V already assembled on
    the host, and z_de(k) returning the full-height incidence of term k on
    demand, instead of the list Zs kept on the device.

    pev: True for every term, a list of term names, or False. A PEV costs a
    (t*q) x n matrix per term, so a real model asks only for its genetic
    terms."""
    n = len(y_fit)
    th_terms, _, _ = split_theta(jnp.asarray(theta), terms, res)
    out = {}
    if V is None:
        V = np.asarray(assemble_V(jnp.asarray(theta), terms, Zs, res, n))
    else:
        V = np.asarray(V)
    Vi = _inverse_V(V, n, out)
    XtVi = X.T @ Vi
    A = XtVi @ X
    try:
        beta = np.linalg.solve(A, XtVi @ y_fit)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ (XtVi @ y_fit)
    out["beta"] = beta
    # (X'V^-1X)^-1, covariance of the fixed effects at known V: needed by
    # predict() and the start of the Kenward-Roger adjustment.
    out["vbeta"] = np.linalg.pinv(A)
    out["Vi"] = Vi
    resid = y_fit - X @ beta
    Py = Vi @ resid - Vi @ X @ (np.linalg.pinv(A) @ (XtVi @ resid))
    out["Py"] = Py
    pev_names = _pev_names(pev, terms)
    if blups and pev_names:
        # P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1, formed once for all terms.
        Pmat = Vi - (Vi @ X) @ out["vbeta"] @ (X.T @ Vi)
    if blups:
        out["blups"] = {}
        for k, tm in enumerate(terms):
            Zd = np.asarray(Zs[k] if z_de is None else z_de(k))
            G = _term_G(tm, th_terms[k])
            # u_k = G_k Z_k' P y, G_k = Sigma_k (x) K_k.
            u = G @ (Zd.T @ Py)
            out["blups"][tm["name"]] = u.reshape(tm["t"], tm["q"]).T
            if tm["name"] in pev_names:
                # Prediction error variance, diagonal only (what a reliability
                # or a Cullis heritability needs):
                #     var(u - u_hat) = G - G Z' P Z G
                M = G @ Zd.T                      # (t*q) x n
                d_ = np.diag(G) - np.einsum("ij,ij->i", M @ Pmat, M)
                out.setdefault("pev", {})[tm["name"]] = \
                    np.maximum(d_, 0.0).reshape(tm["t"], tm["q"]).T
    return out


def _inverse_V(V, n, out):
    """V^-1, or its pseudo-inverse when V is numerically singular (a variance
    at zero is enough); out["V_singular"] says which, since beta and the
    BLUPs are then conditional."""
    try:
        Vi = np.linalg.solve(V, np.eye(n))
        out["V_singular"] = False
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V, rcond=1e-12)
        out["V_singular"] = True
    return Vi


def _pev_names(pev, terms):
    names = ([tm["name"] for tm in terms] if pev is True
             else ([] if not pev else list(pev)))
    unknown = set(names) - {tm["name"] for tm in terms}
    if unknown:
        raise ValueError("pev : terme(s) inconnu(s) %s ; termes du modele : %s"
                         % (sorted(unknown), [tm["name"] for tm in terms]))
    return names


def term_sigma_K(tm, th_k):
    """Sigma (t x t) and K (q x q) of a term at its parameters th_k. K is
    rebuilt from theta, since it may depend on it (AR1 levels)."""
    ns = n_params(tm["struct"], tm["t"], tm["rank"])
    Ls = np.asarray(build_sigma(jnp.asarray(th_k[:ns]), tm["struct"], tm["t"], tm["rank"]))
    kind_ = level_kind(tm)
    from_lvl = (jnp.asarray(tm["LK"]) if kind_ == "fixed" and tm["LK"] is not None
                else level_chol(jnp.asarray(th_k[ns:]), kind_,
                                tm["q"], dims=tm.get("dims"),
                                LK_fixed=tm.get("LK"),
                                order=tm.get("lvl_order", 0),
                                coord=tm.get("coord"),
                                parts=tm.get("lvl_parts"),
                                opts=tm.get("lvl_opts"),
                                expr=tm.get("lvl_expr")))
    Kq = (np.eye(tm["q"]) if from_lvl is None
          else np.asarray(from_lvl) @ np.asarray(from_lvl).T)
    return Ls, Kq


def _term_G(tm, th_k):
    """G = Sigma (x) K of a term."""
    Ls, Kq = term_sigma_K(tm, th_k)
    return np.kron(Ls, Kq)
