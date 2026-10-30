"""Blocks engine: V = A + U U', with the blocks of A processed in batches (vmap).

Computation. decoupage.py splits the random terms into local and global
terms, and the observations into blocks. Local terms and the residual only
couple observations of the same block. Then

    V = A + U U',   A = diag(A_1, ..., A_B),   U = [B_k, k global]  (n x w)

where A_b = R_b + the sum of the local B_k B_k' restricted to block b. For
each block, A_b = L_b L_b' is factorised and W_b = L_b^-1 [U_b, X_b, y_b] is
solved. The cross-product matrix T = sum_b W_b' W_b equals
[U X y]' A^-1 [U X y]. The bordered matrix

    H = T + diag(1_w, 0_p, 0)

has M = I + U'A^-1U, the Woodbury capacitance, as its leading diagonal block.
The Cholesky factor L_H of H gives the rest by successive Schur complements:

    log|V|        = sum_b log|A_b| + log|M|      (first w diagonal entries of L_H)
    log|X'V^-1X|  = next p diagonal entries of L_H
    y'Py          = square of the last diagonal entry of L_H

This yields -2 logL REML without ever forming V or Sigma^-1. A singular Sigma
does no harm, since M >= I. The gradient comes from automatic differentiation.

Vmap. Blocks of similar sizes are grouped into buckets. Each block is padded
to the size of the largest block of its bucket, with the identity on the
diagonal of A and zeros in U, X and y. The padding changes neither log|A_b|
nor W_b. Within a bucket, vmap turns all the factorisations into one batch
(batched Cholesky and triangular solves). Buckets are chosen by dynamic
programming to limit the work wasted on padding. A bucket is traversed in
batches of blocks (lax.scan under jax.checkpoint). Memory is then bounded by
one batch, and the arithmetic is unchanged. The batch size follows a memory
target (OCTETS_LOT). A bucket that fits within it is one batch.

Scope (version 1). Local terms have K = I (any structure between traits) or
a given K. A local term with a parametrised structure between levels (ar1,
...) is made global. Global terms may be of any kind. The residual may have
sections without structure between units (lvl "id"), with any structure
between traits. Outside this scope, out_of_scope() returns the reason, and
fit_reml falls back to the low-memory dense engine (econome.py).

Organisation. out_of_scope() checks the scope. _paquets() groups the blocks
into buckets. BlockPlan builds the padded host arrays once.
make_neg2_blocks() returns -2 logL as a pure JAX function, and
make_objective_blocks() wraps it with its gradient.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp

from .layout import level_kind, split_theta, res_sections, sec_n_params
from .incidence import orthonormal_X
from .residual import _section_S
from .structures import chol_sigma, n_params
from .levels import level_chol
from .decoupage import BlockSplit, factor_width

LOG2PI = float(np.log(2.0 * np.pi))
# Maximum number of buckets.
PAQUETS_MAX = 4
# Each extra bucket costs one more compilation and one more batch. It must save
# at least this fraction of the cost of a single bucket.
SURCOUT_PAQUET = 0.10
# Memory target for one batch of blocks (scan).
OCTETS_LOT = 2 * 2 ** 30


def _term_kind(tm):
    """Kind of a term between levels.

    "id" (K = I), "fixed" (K given), "kinv" (only the precision K^-1 given), or
    the name of a parametrised structure.
    """
    lvl = level_kind(tm)
    if tm.get("Kinv") is not None and tm.get("LK") is None:
        return "kinv"
    return "id" if lvl in ("", None) else lvl


def out_of_scope(terms, res, lag=None):
    """Reason why the model is outside the scope of this engine, or None."""
    if lag is not None:
        return "spatial lag"
    if res.get("struct") == "none":
        return "residual 'none'"
    for sec in res_sections(res, 1):
        if sec.get("lvl", "id") != "id":
            return "residual with a structure between units (%s)" % sec.get("lvl")
    for tm in terms:
        if _term_kind(tm) == "kinv":
            return "term '%s' given only by its precision K^-1" % tm["name"]
    return None


def _paquets(sizes, cost, kmax=PAQUETS_MAX):
    """Split the blocks, sorted by size, into at most kmax contiguous buckets.

    The partition minimises the total cost, where each block of a bucket pays
    the cost of the largest block of that bucket. Dynamic programming in
    O(B^2 kmax). Returns a list of arrays of block indices.
    """
    order = np.argsort(sizes, kind="stable")
    t = np.asarray(sizes)[order]
    B = len(t)
    inf = float("inf")
    best = [[inf] * (B + 1) for _ in range(kmax + 1)]
    arg = [[0] * (B + 1) for _ in range(kmax + 1)]
    best[0][0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(1, B + 1):
            for i in range(j):
                if best[k - 1][i] == inf:
                    continue
                v = best[k - 1][i] + (j - i) * cost(int(t[j - 1]))
                if v < best[k][j]:
                    best[k][j], arg[k][j] = v, i
    # Each extra bucket must save at least SURCOUT_PAQUET of the cost of a
    # single bucket.
    k = min(range(1, kmax + 1), key=lambda kk: best[kk][B] * (1.0 + SURCOUT_PAQUET * (kk - 1)))
    groups, j = [], B
    while j > 0:
        i = arg[k][j]
        groups.append(order[i:j])
        j, k = i, k - 1
    return groups[::-1]


def _z_dense(tm, n):
    Z = np.zeros((n, int(tm["t"]) * int(tm["q"])))
    np.add.at(Z, (np.asarray(tm["zi"], dtype=np.int64), np.asarray(tm["zj"], dtype=np.int64)),
              np.asarray(tm["zx"], dtype=np.float64))
    return Z


class BlockPlan:
    """Block split, buckets and padded arrays, computed once on the host.

    Attributes read by other modules: n_blocs, globaux (indices of the global
    terms) and resume().
    """

    def __init__(self, terms, res, y, X, decoupage=None, octets_lot=OCTETS_LOT, verbose=False):
        n = int(len(y))
        self.n, self.terms, self.res = n, terms, res
        Xq, self.ldRx = orthonormal_X(X)
        p = Xq.shape[1]
        self.p = p
        D = decoupage or BlockSplit(terms, res, n).search()
        G = set(D["globaux"])
        # A local term with a parametrised structure between levels becomes global.
        for k, tm in enumerate(terms):
            if k not in G and _term_kind(tm) not in ("id", "fixed"):
                G.add(k)
        if G != set(D["globaux"]):
            D = dict(D, blocs=BlockSplit(terms, res, n).block_labels(sorted(G)))
        self.globaux = sorted(G)
        self.local_terms = [k for k in range(len(terms)) if k not in G and len(terms[k]["zi"])]
        labels = np.asarray(D["blocs"])
        self.n_blocks = int(labels.max()) + 1
        sizes = np.bincount(labels)
        self.w = int(sum(factor_width(terms[k]) for k in self.globaux))
        r = self.w + p + 1
        self.r = r

        # Scalar local terms (t = 1, iid or diag, K = I) share one matrix.
        def is_scalar(tm):
            return int(tm["t"]) == 1 and tm["struct"] in ("iid", "diag") and _term_kind(tm) == "id"
        self.local_scalar = [k for k in self.local_terms if is_scalar(terms[k])]
        self.local_general = [k for k in self.local_terms if not is_scalar(terms[k])]

        # Residual: each section owns t^2 codes, one per pair of traits.
        secs = res_sections(res, n)
        self.secs = secs
        offset, code_offset = 0, []
        for sec in secs:
            code_offset.append(offset)
            offset += int(sec["t"]) ** 2
        self.n_code = offset

        # Dense incidences of the global terms (fixed kinship folded in) and of
        # the general local terms.
        Zfull = {}
        for k in self.globaux + self.local_general:
            tm = terms[k]
            Z = _z_dense(tm, n)
            if _term_kind(tm) == "fixed" and tm.get("LK") is not None:
                t, q = int(tm["t"]), int(tm["q"])
                Z = np.einsum("ntq,qp->ntp", Z.reshape(n, t, q),
                              np.asarray(tm["LK"])).reshape(n, t * q)
            Zfull[k] = Z

        # Cost of a block of size m: Cholesky, triangular solve, cross products.
        def cost(m):
            return m ** 3 / 3.0 + m * m * r + m * r * r

        self.buckets = []
        for block_ids in _paquets(sizes, cost):
            self.buckets.append(self._build_bucket(block_ids, labels, sizes, Xq,
                                                   np.asarray(y, float), Zfull, secs,
                                                   code_offset, octets_lot))
        if verbose:
            print("  [blocks] %s" % self.summary(), flush=True)

    # Buckets
    def _build_bucket(self, block_ids, labels, sizes, Xq, y, Zfull, secs, code_offset,
                      batch_bytes):
        """Padded host arrays of one bucket, as a dict of shape (nb, m, ...) arrays."""
        terms, r = self.terms, self.r
        nb = len(block_ids)
        m = int(sizes[block_ids].max())
        rows = -np.ones((nb, m), dtype=np.int64)
        for a, b in enumerate(block_ids):
            rr = np.flatnonzero(labels == b)
            rows[a, :len(rr)] = rr
        mask = rows >= 0
        sel = np.where(mask, rows, 0)

        def gather(M):                             # (n, ...) -> (nb, m, ...), zeros outside the block
            out = M[sel]
            out[~mask] = 0.0
            return out

        P = dict(nb=nb, m=m)
        P["mask"] = mask.astype(np.float64)
        P["y"] = gather(y)
        P["X"] = gather(Xq)
        P["Zg"] = [gather(Zfull[k]).reshape(nb, m, int(terms[k]["t"]), int(terms[k]["q"]))
                   for k in self.globaux]
        P["Zs"], P["Es"] = self._scalar_local_arrays(rows, mask)
        P["Zl"], P["local_general"] = self._general_local_arrays(gather, Zfull, nb, m)
        P["Ir"], P["Jr"], P["Er"] = self._residual_pairs(rows, mask, secs, code_offset)

        # Batches: number of blocks processed together, from the memory target.
        c_max = P["Zs"].shape[2]
        n_cols = c_max + sum(z.shape[2] * z.shape[3] for z in P["Zl"])
        bytes_per_block = 8.0 * (3 * m * m + m * (n_cols + r) * 2 + r * r)
        batch = int(max(1, min(nb, batch_bytes // max(bytes_per_block, 1.0))))
        n_batches = int(math.ceil(nb / batch))
        P["batch"], P["n_batches"] = batch, n_batches
        return P

    def _scalar_local_arrays(self, rows, mask):
        """Incidence Zs (nb, m, c) and selector Es (nb, c, n_sc + 1) of the scalar local terms.

        A column of a block is one (term, level) pair present in that block.
        Es maps each column to its term. An unused column points to an extra
        dummy term whose standard deviation is zero.
        """
        terms, n = self.terms, self.n
        nb = rows.shape[0]
        m = rows.shape[1]
        n_sc = len(self.local_scalar)
        block_of = np.full(n, -1, dtype=np.int64)
        pos_in_block = np.full(n, -1, dtype=np.int64)
        for a in range(nb):
            rr = rows[a][mask[a]]
            block_of[rr] = a
            pos_in_block[rr] = np.arange(len(rr))
        e_block, e_row, e_key, e_x = [], [], [], []
        for j, k in enumerate(self.local_scalar):
            tm = terms[k]
            zi = np.asarray(tm["zi"], dtype=np.int64)
            a = block_of[zi]
            ok = a >= 0
            if not ok.any():
                continue
            e_block.append(a[ok])
            e_row.append(pos_in_block[zi[ok]])
            e_key.append(np.stack([np.full(ok.sum(), j),
                                   np.asarray(tm["zj"], dtype=np.int64)[ok]], axis=1))
            e_x.append(np.asarray(tm["zx"], dtype=np.float64)[ok])
        if e_block:
            eb = np.concatenate(e_block)
            er = np.concatenate(e_row)
            ek = np.concatenate(e_key)
            ex = np.concatenate(e_x)
            # column = rank of (term, level) among those of the block
            block_key = np.stack([eb, ek[:, 0], ek[:, 1]], axis=1)
            uniq, inv = np.unique(block_key, axis=0, return_inverse=True)
            inv = inv.ravel()
            start = np.searchsorted(uniq[:, 0], np.arange(nb))
            col_of_key = np.arange(len(uniq)) - start[uniq[:, 0]]
            c_max = int(col_of_key.max()) + 1
            Zs = np.zeros((nb, m, c_max))
            np.add.at(Zs, (eb, er, col_of_key[inv]), ex)
            Es = np.zeros((nb, c_max, n_sc + 1))
            Es[:, :, n_sc] = 1.0                       # empty column -> zero dummy term
            Es[uniq[:, 0], col_of_key, n_sc] = 0.0
            Es[uniq[:, 0], col_of_key, uniq[:, 1]] = 1.0
        else:
            Zs = np.zeros((nb, m, 1))
            Es = np.zeros((nb, 1, n_sc + 1))
            Es[:, :, n_sc] = 1.0
        return Zs, Es

    def _general_local_arrays(self, gather, Zfull, nb, m):
        """Incidences (nb, m, t, q) of the general local terms present in the bucket.

        With K = I, only the levels used in each block are kept, padded to the
        widest block. With a folded K, every level is kept.
        """
        Zl, kept = [], []
        for k in self.local_general:
            tm = self.terms[k]
            t, q = int(tm["t"]), int(tm["q"])
            Zk = gather(Zfull[k]).reshape(nb, m, t, q)
            if not np.any(Zk):
                continue
            if _term_kind(tm) == "id":
                levels = [np.flatnonzero(np.any(Zk[a] != 0, axis=(0, 1))) for a in range(nb)]
                q_max = max(max(len(v) for v in levels), 1)
                Zc = np.zeros((nb, m, t, q_max))
                for a, v in enumerate(levels):
                    Zc[a, :, :, :len(v)] = Zk[a][:, :, v]
                Zk = Zc
            Zl.append(Zk)
            kept.append(k)
        return Zl, kept

    def _residual_pairs(self, rows, mask, secs, code_offset):
        """Pairs (i, j) of rows of the same unit in each block, and their Sigma codes.

        Returns Ir, Jr (nb, n_pairs) with local row positions, and the one-hot
        matrix Er (nb, n_pairs, n_code) that maps vec(Sigma) to the pair values.
        Padding pairs have a zero row in Er.
        """
        n = self.n
        nb = rows.shape[0]
        I_, J_, C_ = [], [], []
        for a in range(nb):
            rr = rows[a][mask[a]]
            pos = np.full(n, -1, dtype=np.int64)
            pos[rr] = np.arange(len(rr))
            ii, jj, cc = [], [], []
            for s, sec in enumerate(secs):
                sec_rows = np.asarray(sec["rows"], dtype=np.int64)
                ok = pos[sec_rows] >= 0
                if not ok.any():
                    continue
                lr = pos[sec_rows[ok]]
                unit = np.asarray(sec["unit"], dtype=np.int64)[ok]
                trait = np.asarray(sec["trait"], dtype=np.int64)[ok]
                t = int(sec["t"])
                o = np.argsort(unit, kind="stable")
                lr, unit, trait = lr[o], unit[o], trait[o]
                starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
                ends = np.r_[starts[1:], len(unit)]
                for d0, d1 in zip(starts, ends):
                    for x in range(d0, d1):
                        for z in range(d0, d1):
                            ii.append(lr[x]); jj.append(lr[z])
                            cc.append(code_offset[s] + trait[x] * t + trait[z])
            I_.append(ii); J_.append(jj); C_.append(cc)
        n_pairs = max(max(len(v) for v in I_), 1)
        Ip = np.zeros((nb, n_pairs), dtype=np.int64)
        Jp = np.zeros((nb, n_pairs), dtype=np.int64)
        Er = np.zeros((nb, n_pairs, self.n_code))
        for a in range(nb):
            L = len(I_[a])
            Ip[a, :L], Jp[a, :L] = I_[a], J_[a]
            Er[a, np.arange(L), C_[a]] = 1.0          # padding pairs: zero row
        return Ip, Jp, Er

    def summary(self):
        txt = ", ".join("%d x %d" % (P["nb"], P["m"]) for P in self.buckets)
        return ("%d blocks in %d buckets (%s), global terms %s, w = %d"
                % (self.n_blocks, len(self.buckets), txt,
                   [self.terms[k]["name"] for k in self.globaux], self.w))

    # Device arrays
    def device_arrays(self):
        """Device arrays of each bucket, grouped by batch: shape (n_batches, batch, ...)."""
        out = []
        for P in self.buckets:
            batch, nbt, nb = P["batch"], P["n_batches"], P["nb"]
            pad = nbt * batch - nb

            def to_batches(a):
                a = np.asarray(a)
                if pad:          # dummy blocks: zero mask, hence A = I and W = 0
                    a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], dtype=a.dtype)])
                return jnp.asarray(a.reshape((nbt, batch) + a.shape[1:]))
            out.append(dict(
                mask=to_batches(P["mask"]), y=to_batches(P["y"]), X=to_batches(P["X"]),
                Zg=[to_batches(z) for z in P["Zg"]], Zs=to_batches(P["Zs"]),
                Es=to_batches(P["Es"]), Zl=[to_batches(z) for z in P["Zl"]],
                Ir=to_batches(P["Ir"]), Jr=to_batches(P["Jr"]), Er=to_batches(P["Er"])))
        return out


# ==============================================================================
# -2 logL
# ==============================================================================
def make_neg2_blocks(plan):
    """f(theta, arrays) -> -2 logL REML (constant included), a pure JAX function."""
    terms, res, p, w, r = plan.terms, plan.res, plan.p, plan.w, plan.r
    n = plan.n
    glob, local_scalar = plan.globaux, plan.local_scalar
    local_general_per_bucket = [P["local_general"] for P in plan.buckets]

    def sigma_factor(theta_k, tm):
        t, rk = int(tm["t"]), int(tm["rank"])
        return chol_sigma(theta_k[:n_params(tm["struct"], t, rk)], tm["struct"], t, rk)

    def level_factor(theta_k, tm):
        t, rk = int(tm["t"]), int(tm["rank"])
        ns = n_params(tm["struct"], t, rk)
        return level_chol(theta_k[ns:], _term_kind(tm), int(tm["q"]), dims=tm.get("dims"),
                          LK_fixed=tm.get("LK"), order=tm.get("lvl_order", 0),
                          coord=tm.get("coord"), parts=tm.get("lvl_parts"),
                          opts=tm.get("lvl_opts"), expr=tm.get("lvl_expr"))

    def f(theta, tabs):
        th_t, th_r, _ = split_theta(theta, terms, res)
        Ls = {k: sigma_factor(th_t[k], terms[k]) for k in glob + plan.local_terms}
        LKp = {k: level_factor(th_t[k], terms[k]) for k in glob
               if _term_kind(terms[k]) not in ("id", "fixed")}
        o = 0
        Svec = []
        for sec in plan.secs:
            pn = sec_n_params(sec)
            Svec.append(_section_S(th_r[o:o + pn], sec).reshape(-1))
            o += pn
        Svec = jnp.concatenate(Svec)
        sig_sc = jnp.concatenate([jnp.stack([Ls[k][0, 0] for k in local_scalar])
                                  if local_scalar else jnp.zeros(0),
                                  jnp.zeros(1)])

        def one_block(d, local_general):
            m = d["y"].shape[0]
            # A = R + B_loc B_loc' + identity on the padding
            vals = d["Er"] @ Svec
            A = jnp.zeros((m, m)).at[d["Ir"], d["Jr"]].add(vals)
            A = A + jnp.diag(1.0 - d["mask"])
            factors = [d["Zs"] * (d["Es"] @ sig_sc)[None, :]]
            for Zk, k in zip(d["Zl"], local_general):
                factors.append(jnp.einsum("ntq,tm->nmq", Zk, Ls[k]).reshape(m, -1))
            Bl = jnp.concatenate(factors, axis=1)
            A = A + Bl @ Bl.T
            # global columns U, then X and y
            us = []
            for Zk, k in zip(d["Zg"], glob):
                if k in LKp:
                    Zk = jnp.einsum("ntq,qp->ntp", Zk, LKp[k])
                us.append(jnp.einsum("ntq,tm->nmq", Zk, Ls[k]).reshape(m, -1))
            RHS = jnp.concatenate(us + [d["X"], d["y"][:, None]], axis=1)
            L = jnp.linalg.cholesky(A)
            Wb = jax.scipy.linalg.solve_triangular(L, RHS, lower=True)
            return Wb.T @ Wb, 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        T = jnp.zeros((r, r))
        ld = jnp.zeros(())
        for tab, local_general in zip(tabs, local_general_per_bucket):
            def scan_body(c, d, local_general=local_general):
                Tb, lb = jax.vmap(lambda dd: one_block(dd, local_general))(d)
                return (c[0] + Tb.sum(0), c[1] + lb.sum()), None
            (T, ld), _ = jax.lax.scan(jax.checkpoint(scan_body), (T, ld), tab)
        H = T + jnp.diag(jnp.concatenate([jnp.ones(w), jnp.zeros(p + 1)]))
        LH = jnp.linalg.cholesky(H)
        dH = jnp.diag(LH)
        return ((n - p) * LOG2PI + ld + 2.0 * jnp.sum(jnp.log(dH[:w + p]))
                + dH[-1] ** 2 + plan.ldRx)
    return f


def make_objective_blocks(plan, scale=None):
    """(fun_jac_scaled, fun_jac, sc), with the same contract as model.make_objective."""
    sc = float(plan.n) if scale is None else float(scale)
    f = make_neg2_blocks(plan)
    tabs = plan.device_arrays()
    fg = jax.jit(jax.value_and_grad(lambda th, tb: f(th, tb) / sc))

    def fun_jac_scaled(theta):
        v, g = fg(jnp.asarray(theta), tabs)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc
    return fun_jac_scaled, fun_jac, sc
