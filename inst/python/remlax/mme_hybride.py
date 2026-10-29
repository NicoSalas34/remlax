"""Hybrid inversion of C for the MME engine: dense separator, dense blocks in batches.

Split. The unknowns of C fall into two sets. D holds the fixed effects and the
levels of the GLOBAL terms, those that link the blocks of observations (found
by decoupage.py). S holds everything else. Once D is removed, Css splits into
independent connected components (the tanks of a trial, the environments).
Fill-in makes each component nearly full, so each is factorised as a DENSE
matrix. Components of similar sizes are processed together (vmap), as in
blocs.py.

Elimination (same algebra as kmmr, Melsen et al. 2026, sections 3.3 to 3.6):
  Css = blocks L_b L_b'          log|C| = sum log|Css_b| + log|Sch|
  Q   = Css^-1 Csd               Sch = Cdd - Cds Q  (Schur complement, dense)
  C x = r:  g = Css^-1 r_s;  x_d = Sch^-1 (r_d - Q' r_s);  x_s = g - Q x_d
  C^-1:  dd = Sch^-1;  sd = -Q Sch^-1;  ss = Css^-1 + Q Sch^-1 Q'
Since Cds Css^-1 r_s = Q' r_s, Csd need not be kept after the factorisation.
Only the entries of C^-1 on the pattern of C are returned
(selected_inverse), as for the sparse inversion. The ss blocks are computed
densely, block by block.

Difference with kmmr. kmmr chooses the dense part by a fixed rule (a dense
Kronecker structure) and factorises the sparse part with CHOLMOD on CPU. Here
the separator comes from the automatic split. The sparse part, made of
independent blocks, stays on the GPU in dense batches.

Scope. The components of Css must stay of moderate size (TAILLE_MAX_BLOC).
Otherwise OutOfScope is raised and the MME engine keeps its dense or
sparse inversion.

Organisation. dense_set lists the unknowns of the separator.
HybridFactor builds the batches and the index tables once (__init__),
compiles the factorisation, solve and inversion with JAX (_compile), then
offers the interface of selinv.SparseFactor (factorise_values, resoudre,
selected_inverse).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import jax
import jax.numpy as jnp

from .blocs import _paquets

TAILLE_MAX_BLOC = 8000


class OutOfScope(Exception):
    """The hybrid inversion does not apply to this model (component too large)."""
    pass


def dense_set(MME_T, p_dense, globaux):
    """Indices of C in the separator: fixed effects [0, p_dense) and the
    levels of the global terms (indices in the order of the terms of
    mme.MME)."""
    idx = [np.arange(p_dense, dtype=np.int64)]
    for k in globaux:
        d = MME_T[k]
        idx.append(np.arange(d["off"], d["off"] + d["t"] * d["q"], dtype=np.int64))
    return np.unique(np.concatenate(idx))


class HybridFactor:
    """Same interface as selinv.SparseFactor: factorise_values, resoudre,
    selected_inverse (entries of C^-1 at the positions (ci, cj) of the
    pattern).

    ci, cj: positions of the stored entries of C (upper triangle); m: order
    of C; dense_idx: unknowns of the separator (dense_set); taille_max:
    largest component of Css accepted.

    paquets: one dict per batch of blocks, with nb (number of blocks), m
    (common padded size), rows (unknowns of S in each block, -1 for padding)
    and mask (1 for a real row, 0 for padding).
    """

    def __init__(self, ci, cj, m, dense_idx, taille_max=TAILLE_MAX_BLOC, verbose=False):
        ci = np.asarray(ci, dtype=np.int64); cj = np.asarray(cj, dtype=np.int64)
        self.m = m
        is_dense = np.zeros(m, dtype=bool); is_dense[np.asarray(dense_idx, dtype=np.int64)] = True
        D = np.flatnonzero(is_dense); S = np.flatnonzero(~is_dense)
        self.D, self.S = D, S
        nD, nS = len(D), len(S)
        self.nD = nD
        dpos = np.full(m, -1, dtype=np.int64); dpos[D] = np.arange(nD)
        spos = np.full(m, -1, dtype=np.int64); spos[S] = np.arange(nS)
        cls_dd = is_dense[ci] & is_dense[cj]
        cls_ss = ~is_dense[ci] & ~is_dense[cj]
        cls_sd = ~(cls_dd | cls_ss)
        # Connected components of Css.
        e = np.flatnonzero(cls_ss)
        G = sp.coo_matrix((np.ones(len(e), dtype=np.int8), (spos[ci[e]], spos[cj[e]])), shape=(nS, nS))
        nc, lab = connected_components(G, directed=False)
        sizes = np.bincount(lab, minlength=nc)
        if nS and sizes.max() > taille_max:
            raise OutOfScope("composante de %d inconnues apres retrait du separateur (max %d)"
                                % (sizes.max(), taille_max))

        def cost(t):
            return t ** 3 / 3.0 + 2.0 * t * t * nD + t * nD * nD
        groups = _paquets(sizes, cost) if nS else []
        # Position of each unknown of S: (batch, slot, local rank).
        batch_of = np.full(nS, -1, dtype=np.int64); slot = np.zeros(nS, dtype=np.int64)
        loc = np.zeros(nS, dtype=np.int64)
        self.batches = []
        for g_i, comps in enumerate(groups):
            nmax = int(sizes[comps].max())
            rows = -np.ones((len(comps), nmax), dtype=np.int64)
            for a, c in enumerate(comps):
                members = np.flatnonzero(lab == c)
                batch_of[members] = g_i; slot[members] = a; loc[members] = np.arange(len(members))
                rows[a, :len(members)] = members
            self.batches.append(dict(nb=len(comps), m=nmax, rows=rows,
                                     mask=(rows >= 0).astype(np.float64)))
        # Entries of the pattern, by class.
        self.e_dd = np.flatnonzero(cls_dd)
        self.dd_r, self.dd_c = dpos[ci[self.e_dd]], dpos[cj[self.e_dd]]
        es = np.flatnonzero(cls_sd)
        s_of = np.where(is_dense[ci[es]], cj[es], ci[es])
        d_of = np.where(is_dense[ci[es]], ci[es], cj[es])
        self.sd = []
        self.ss = []
        for g_i in range(len(self.batches)):
            sel = batch_of[spos[s_of]] == g_i
            self.sd.append(dict(e=es[sel], b=slot[spos[s_of[sel]]], i=loc[spos[s_of[sel]]], d=dpos[d_of[sel]]))
            sel2 = batch_of[spos[ci[e]]] == g_i
            ee = e[sel2]
            self.ss.append(dict(e=ee, b=slot[spos[ci[ee]]], i=loc[spos[ci[ee]]], j=loc[spos[cj[ee]]]))
        self.n_entrees = len(ci)
        # Rank in S and (batch, slot, local rank) of each unknown of S.
        self._spos = spos
        self._dpos = dpos
        self._batch_of, self._slot, self._loc = batch_of, slot, loc
        self._compile()
        if verbose:
            print("  [mme] hybride : separateur %d inconnues | %d blocs en %d paquets (%s)"
                  % (nD, nc, len(self.batches),
                     ", ".join("%d x %d" % (P["nb"], P["m"]) for P in self.batches)), flush=True)

    # ------------------------------------------------------------------ computation
    def _compile(self):
        """Index tables on the device and the compiled factorisation, solve
        and inversion."""
        # The indices are ARGUMENTS of the compiled functions. Captured as
        # constants, they would make XLA try to fold arrays of tens of
        # millions of entries at compile time (see model.make_neg2).
        nD = self.nD
        # 32-bit integers: on large models these arrays hold tens of millions
        # of entries (dev/notes.md).
        i32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.int32))
        self._idx = dict(
            dd=tuple(i32(a) for a in (self.e_dd, self.dd_r, self.dd_c)),
            sd=[{k: i32(v) for k, v in d.items()} for d in self.sd],
            ss=[{k: i32(v) for k, v in d.items()} for d in self.ss],
            mask=[jnp.asarray(p["mask"]) for p in self.batches])
        n_ent = self.n_entrees

        def batched_cho_solve(L, B):
            return jax.vmap(lambda Lb, Bb: jax.scipy.linalg.cho_solve((Lb, True), Bb))(L, B)

        def factorise(x, idx):
            dd_e, dd_r, dd_c = idx["dd"]
            Cdd = jnp.zeros((nD, nD)).at[dd_r, dd_c].set(x[dd_e]).at[dd_c, dd_r].set(x[dd_e])
            ld = jnp.zeros(())
            Ls, Qs = [], []
            for msk, a, b in zip(idx["mask"], idx["sd"], idx["ss"]):
                nb, m = msk.shape
                Css = jnp.zeros((nb, m, m))
                Css = Css.at[b["b"], b["i"], b["j"]].set(x[b["e"]]).at[b["b"], b["j"], b["i"]].set(x[b["e"]])
                Css = Css + jax.vmap(jnp.diag)(1.0 - msk)
                Csd = jnp.zeros((nb, m, nD)).at[a["b"], a["i"], a["d"]].set(x[a["e"]])
                L = jnp.linalg.cholesky(Css)
                ld = ld + 2.0 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(L)))
                Q = batched_cho_solve(L, Csd)
                Cdd = Cdd - jnp.einsum("bnd,bne->de", Csd, Q)
                Ls.append(L); Qs.append(Q)
            Lsch = jnp.linalg.cholesky(Cdd)
            ld = ld + 2.0 * jnp.sum(jnp.log(jnp.diag(Lsch)))
            return ld, Ls, Qs, Lsch

        def solve(Ls, Qs, Lsch, Bs, Bd):
            Xi = Bd
            Gs = []
            for L, Q, B in zip(Ls, Qs, Bs):
                Gs.append(batched_cho_solve(L, B))
                Xi = Xi - jnp.einsum("bnd,bnk->dk", Q, B)
            Td = jax.scipy.linalg.cho_solve((Lsch, True), Xi)
            Ts = [Gb - jnp.einsum("bnd,dk->bnk", Q, Td) for Gb, Q in zip(Gs, Qs)]
            return Ts, Td

        def inverse(Ls, Qs, Lsch, idx):
            dd_e, dd_r, dd_c = idx["dd"]
            Si = jax.scipy.linalg.cho_solve((Lsch, True), jnp.eye(nD))
            out = jnp.zeros(n_ent).at[dd_e].set(Si[dd_r, dd_c])
            for L, Q, a, b in zip(Ls, Qs, idx["sd"], idx["ss"]):
                nb, m = L.shape[0], L.shape[1]
                M = jnp.einsum("bnd,de->bne", Q, Si)                 # Q Sch^-1
                out = out.at[a["e"]].set(-M[a["b"], a["i"], a["d"]])
                Ci = batched_cho_solve(L, jnp.broadcast_to(jnp.eye(m), (nb, m, m)))
                Ci = Ci + jnp.einsum("bnd,bkd->bnk", M, Q)
                out = out.at[b["e"]].set(Ci[b["b"], b["i"], b["j"]])
            return out

        self._f_fact = jax.jit(factorise)
        self._f_res = jax.jit(solve)
        self._f_inv = jax.jit(inverse)

    def factorise_values(self, x):
        """x: values of C on the pattern (ci, cj). Returns log|C|."""
        ld, Ls, Qs, Lsch = self._f_fact(jnp.asarray(x), self._idx)
        ld = float(ld)
        if not np.isfinite(ld):
            raise np.linalg.LinAlgError("C n'est pas definie positive (hybride)")
        self._state = (Ls, Qs, Lsch)
        return ld

    def solve(self, B):
        """C^-1 B from the current factorisation (B a vector or a matrix)."""
        B = np.asarray(B, dtype=np.float64)
        is_vector = B.ndim == 1
        if is_vector:
            B = B[:, None]
        Ls, Qs, Lsch = self._state
        Bd = jnp.asarray(B[self.D])
        Bs = []
        for P in self.batches:
            rows = P["rows"]
            sel = np.where(rows >= 0, self.S[np.maximum(rows, 0)], 0)
            Bb = B[sel]
            Bb[rows < 0] = 0.0
            Bs.append(jnp.asarray(Bb))
        Ts, Td = self._f_res(Ls, Qs, Lsch, Bs, Bd)
        X = np.zeros_like(B)
        X[self.D] = np.asarray(Td)
        for P, T in zip(self.batches, Ts):
            rows = P["rows"]; ok = rows >= 0
            X[self.S[rows[ok]]] = np.asarray(T)[ok]
        return X[:, 0] if is_vector else X

    def selected_inverse(self):
        """C^-1 at the positions (ci, cj) of the pattern."""
        Ls, Qs, Lsch = self._state
        return np.asarray(self._f_inv(Ls, Qs, Lsch, self._idx))
