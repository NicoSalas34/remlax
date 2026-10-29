"""Incidence matrices Z and the fixed design X as the likelihood sees them.

Column order of Z: column = (trait - 1) * q + level, level fastest; Sigma (x) K
follows the same order (Sigma on the slow index). R/ builds Z with this
convention; changing it on one side only would give another model without any
visible error.

The supports, restricted incidences and folded kinships below are constant in
theta: they are computed once, outside the differentiated function.
"""
from . import _x64  # noqa: F401

import jax.numpy as jnp
import numpy as np

from .layout import level_kind


def dense_Z(term, n):
    """Dense Z (n x t*q) from the COO triplets. Dense, not sparse: at the
    target sizes a dense product on GPU beats a sparse one."""
    Z = jnp.zeros((n, term["t"] * term["q"]))
    return Z.at[term["zi"], term["zj"]].add(term["zx"])


def term_support(term, n):
    """The rows where the incidence of the term is non-zero, sorted. A term of
    the IGE model concerns one block and one trait; restricting its factor to
    those rows avoids storing zeros over the full height."""
    return np.unique(np.asarray(term["zi"], dtype=np.int64))


def restrict_Z(term, rows, n):
    """Z of the term restricted to `rows`, zi renumbered in the support. An
    index outside the support is an inconsistency, asserted rather than left
    to produce a silently wrong matrix."""
    pos = np.full(n, -1, dtype=np.int64)
    pos[rows] = np.arange(len(rows), dtype=np.int64)
    zi = pos[np.asarray(term["zi"], dtype=np.int64)]
    assert zi.min() >= 0, "une entree de l'incidence tombe hors du support"
    Z = jnp.zeros((len(rows), term["t"] * term["q"]))
    return Z.at[zi, np.asarray(term["zj"], dtype=np.int64)].add(
        jnp.asarray(term["zx"]))


def support_groups(terms, n):
    """Terms grouped by identical support, as [(rows, [k, ...])]. Only terms
    with exactly the same support can share one product: for disjoint
    supports, (B_1 + B_2)(B_1 + B_2)' has cross terms B_1 B_2', so merging
    them would give a wrong V."""
    by_key, order = {}, []
    for k, tm in enumerate(terms):
        rows = term_support(tm, n)
        key = rows.tobytes()
        if key not in by_key:
            by_key[key] = (rows, [])
            order.append(key)
        by_key[key][1].append(k)
    return [by_key[c] for c in order]


def orthonormal_X(X):
    """X = Q R (reduced QR): returns Q and 2 log|det R|.

    REML depends on X only through its span, plus a constant:
        log|X'V^-1X| = log|Q'V^-1Q| + 2 log|det R|.
    A raw column nearly collinear with the intercept (a sowing date, ~20 000)
    brings X'V^-1X to 1e12 and log|X'V^-1X| loses digits. Only the likelihood
    functions use Q; beta, Wald and predictions keep X."""
    Xn = np.asarray(X, dtype=np.float64)
    if Xn.shape[1] == 0:
        return Xn, 0.0
    Q, R = np.linalg.qr(Xn)
    return Q, float(2.0 * np.sum(np.log(np.abs(np.diag(R)))))


def prefold_fixed_K(terms, Zr):
    """Folds a fixed relationship matrix into the incidence: Z <- Z (I (x) L_K).

    A given K depends on no parameter, and neither does Z (I (x) L_K). Left in
    the compiled function, XLA computes this product of two captured constants
    during compilation, in its interpreter. Formed once on the host, it leaves
    the compiled program. V = Z (Sigma (x) K) Z' = ZL (Sigma (x) I) ZL' and its
    derivatives are the same; only the likelihood functions see the folded
    term, the BLUPs and outputs keep the original one.

    The product is sparse (Z has a few non-zeros per row) and cached in the
    original term for the second call (likelihood, then AI).

    Exact compression: V depends on Z_a L_K only through the products
    (Z_a L_K)(Z_b L_K)'. When the loaded rows, all components stacked, are
    fewer than q (genotypes in K without phenotype), L_K is replaced by a basis
    of the span of those rows: M' = Q R (reduced QR of M', M the stacked rows,
    R x q) gives (Z_a L_K)(Z_b L_K)' = C_a C_b' with C = R'. The folded term
    then has R columns per component instead of q, for exactly the same V."""
    import scipy.sparse as sp_
    terms2, Zr2 = [], []
    for tm, Z in zip(terms, Zr):
        if level_kind(tm) == "fixed" and tm.get("LK") is not None:
            ZL, q2 = _folded(tm, np.asarray(Z), sp_)
            tm2 = dict(tm)
            tm2["LK"] = None
            tm2["lvl"] = "id"
            tm2["q"] = q2
            terms2.append(tm2)
            Zr2.append(jnp.asarray(ZL))
        else:
            terms2.append(tm)
            Zr2.append(Z)
    return terms2, Zr2


def _folded(tm, Zn, sp_):
    """(Z (I (x) L_K), number of columns per component), compressed when the
    loaded rows are fewer than 0.8 q; cached in tm under ("_plie", shape)."""
    key = ("_plie", Zn.shape)
    if key not in tm:
        t, q = int(tm["t"]), int(tm["q"])
        LK = np.asarray(tm["LK"], dtype=np.float64)
        Zs = sp_.csr_matrix(Zn)
        blocks, rows = [], []
        for a in range(t):
            Za = Zs[:, a * q:(a + 1) * q].tocsr()
            blocks.append(np.asarray(Za @ LK))
            rows.append(np.flatnonzero(np.diff(Za.indptr)))
        R = int(sum(len(l) for l in rows))
        if 0 < R < 0.8 * q:
            M = np.vstack([blocks[a][rows[a]] for a in range(t)])      # R x q
            C = np.linalg.qr(M.T, mode="r").T                          # R x R
            ZL = np.zeros((Zn.shape[0], t * R))
            o = 0
            for a in range(t):
                ZL[rows[a], a * R:(a + 1) * R] = C[o:o + len(rows[a])]
                o += len(rows[a])
            q2 = R
        else:
            ZL, q2 = np.hstack(blocks), q
        tm[key] = (ZL, q2)
    return tm[key]
