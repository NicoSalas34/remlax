"""Checks of a model before fitting.

validate() refuses designs that would make V singular. Each check stands for
an error seen in practice, which used to show as a silent NaN rather than a
message.
"""
import numpy as np

from .layout import res_sections


def validate(terms, res, y, X):
    n = len(y)
    for sec in res_sections(res, n):
        _check_section(sec, len(np.asarray(sec["trait"])))
    _check_terms_y_X(terms, y, X, n)


def _check_section(res, n):
    """Two rows of the same unit and trait have correlation 1 under any
    residual structure: they are identical in R, which becomes singular. The
    check covers us/diag/fa and every structure between units."""
    tr, un = np.asarray(res["trait"]), np.asarray(res["unit"])
    kind = res.get("lvl", "id")
    if res["struct"] in ("us", "diag", "fa") or kind != "id":
        key = un.astype(np.int64) * (int(tr.max()) + 1) + tr.astype(np.int64)
        if len(np.unique(key)) != n:
            d = n - len(np.unique(key))
            what = ("structure residuelle '%s'" % res["struct"] if kind == "id"
                    else "structure '%s' entre unites" % kind)
            raise ValueError(
                "%s : %d couple(s) (unite, caractere) en double. Chaque unite ne "
                "peut etre observee qu'une fois par caractere ; sinon la matrice "
                "residuelle est singuliere et la vraisemblance vaut NaN."
                % (what, d))
    if res["struct"] in ("us", "diag", "fa"):
        if int(tr.max()) + 1 > res["t"]:
            raise ValueError("indice de caractere %d hors de la structure residuelle (t=%d)"
                             % (tr.max(), res["t"]))


def _check_terms_y_X(terms, y, X, n):
    """Incidences not entirely zero, y and X finite, X of full column rank. The
    sum of |z| per cell is taken on the triplets (duplicates add up, as in
    Z[zi, zj] += zx), without forming Z."""
    for tm in terms:
        ncol = tm["t"] * tm["q"]
        zj = np.asarray(tm["zj"], dtype=np.int64); zx = np.asarray(tm["zx"], dtype=np.float64)
        if zj.size == 0:
            raise ValueError("terme '%s' : incidence entierement nulle." % tm["name"])
        zi = np.asarray(tm["zi"], dtype=np.int64)
        cle = zi * ncol + zj
        u, inv = np.unique(cle, return_inverse=True)
        per_cell = np.abs(np.bincount(inv, weights=zx, minlength=len(u)))
        if not np.any(per_cell > 0):
            raise ValueError("terme '%s' : incidence entierement nulle." % tm["name"])
    if not np.all(np.isfinite(y)):
        raise ValueError("y contient des valeurs non finies.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des valeurs non finies.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("X est de rang %d pour %d colonnes."
                         % (np.linalg.matrix_rank(X), X.shape[1]))
