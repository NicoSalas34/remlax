"""Layout of theta: which parameters belong to which term or residual section.

theta = [term 1 | term 2 | ... | residual section 1 | ... | free entries of Psi]

Each term carries the parameters of its Sigma, then those of its structure
between levels. R and Python must cut theta at the same places; every module
reads the layout from here.
"""
import numpy as np

from .structures import n_params
from .levels import n_level_params
from .lag import lag_n_params


def level_kind(tm):
    """Kind of the structure between levels of a term. A K given without an
    explicit "lvl" key means "fixed": falling back to "id" would ignore the
    relationship matrix silently."""
    return tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")


def term_n_params(tm):
    """Parameters of a term: those of Sigma, then those of the level structure."""
    return (n_params(tm["struct"], tm["t"], tm["rank"])
            + n_level_params(level_kind(tm), tm.get("lvl_order", 0), tm.get("lvl_parts"),
                             opts=tm.get("lvl_opts")))


def res_sections(res, n):
    """Sections of the residual, always as a list: an ordinary residual is a
    single section covering every row, so there is one code path. A residual
    "none" has no section, hence no parameter and R = 0; V must then be
    positive definite through the terms alone (fit.py checks it at the start)."""
    if res.get("struct") == "none":
        return []
    secs = res.get("sections")
    if secs:
        return secs
    s = dict(res)
    s["rows"] = np.arange(n, dtype=np.int64)
    return [s]


def sec_n_params(sec):
    """Parameters of a residual section: Sigma, then the level structure."""
    return (n_params(sec["struct"], sec["t"], sec["rank"])
            + n_level_params(sec.get("lvl", "id"), sec.get("lvl_order", 0),
                             sec.get("lvl_parts"), opts=sec.get("lvl_opts")))


def res_n_params(res):
    return sum(sec_n_params(s) for s in res_sections(res, 0))


def n_theta(terms, res, lag=None):
    """Length of theta: terms, residual, then the free entries of Psi. The lag
    parameters come last, so code that ignores the lag (assemble_V,
    kenward_roger, predict) still cuts the head of theta at the right place."""
    return sum(term_n_params(t) for t in terms) + res_n_params(res) + lag_n_params(lag)


def split_theta(theta, terms, res):
    """theta cut by term, then the residual block; returns (per-term list,
    residual block, end of the residual block)."""
    out, o = [], 0
    for tm in terms:
        p = term_n_params(tm)
        out.append(theta[o:o + p]); o += p
    pr = res_n_params(res)
    return out, theta[o:o + pr], o + pr
