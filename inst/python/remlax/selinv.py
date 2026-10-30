"""Sparse Cholesky factorisation and supernodal selected inversion.

The MME engine (mme.py) needs log|C|, solves with C and the entries of C^-1
on the pattern of C at every iteration. For a sparse C (pedigree), the full
inverse is dense and out of reach. Takahashi's selected inversion computes
C^-1 only on the pattern of the factor, at a cost close to one factorisation.

Factorisation. The elimination order P is computed once, by METIS nested
dissection when pymetis is installed and by the minimum degree of SuperLU on
C + C' otherwise. Every later factorisation receives C already permuted and
keeps the natural order, so the pattern of L does not change from one
iteration to the next. CHOLMOD (shipped inside cvxopt) factorises when it is
installed. SuperLU (SciPy) is the fallback. It factorises P C P' without
pivoting and in symmetric mode: U = D L', so C_p = L D L' and the Cholesky
factor is L D^(1/2).

Supernodes. Consecutive columns with the same structure form a supernode,
stored as a dense block (rows x columns, column major). Small supernodes are
merged with their parent when the added fill stays moderate (CHOLMOD's
rules). The blocks then become large enough for BLAS, at the cost of a few
explicit zeros.

Selected inversion. Z = (L L')^-1 is computed supernode by supernode, from
the last to the first:
    Lt   = L_RJ L_JJ^-1
    Z_RJ = - Z_RR Lt
    Z_JJ = (L_JJ L_JJ')^-1 - Lt' Z_RJ
R holds the rows of J below its diagonal block. Z_RR lies in the pattern of
the ancestor supernodes, which are already computed. A small C kernel
(selinv_c.c) runs this loop when a C compiler is available. A NumPy loop
gives the same result otherwise.

Organisation. _c_kernel compiles and loads the C kernel. metis_order computes
the METIS order. SparseFactor holds the symbolic analysis (order, supernodes,
extraction map) and the numeric steps (factoriser or factorise_values,
resoudre, selected_inverse).
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_triangular


_C_KERNEL = None


def _c_kernel():
    """C kernel of the selected inversion (src/selinv_c.c of the R package).

    R compiles it at installation into the package library and passes the
    path of that library in REMLAX_SELINV_LIB; nothing is compiled at run
    time. In a development checkout, without that variable, the source found
    of the repository (src/selinv_c.c) is compiled once into
    $REMLAX_CACHE (or the temporary directory). When neither is available, or the library cannot be
    loaded, the NumPy loop takes over with the same result, several times
    slower (dev/notes.md). BLAS and LAPACK are those of SciPy, passed as
    pointers.
    """
    global _C_KERNEL
    if _C_KERNEL is not None:
        return _C_KERNEL or None
    try:
        so = os.environ.get("REMLAX_SELINV_LIB") or _dev_kernel()
        if not so:
            raise RuntimeError("C kernel unavailable")
        lib = ctypes.CDLL(so)
        fn = lib.rx_selinv_brut
        fn.restype = ctypes.c_int
        P = ctypes.POINTER
        fn.argtypes = [ctypes.c_int, P(ctypes.c_longlong), P(ctypes.c_int), P(ctypes.c_int),
                       P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double), P(ctypes.c_double),
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        from scipy.linalg import cython_blas, cython_lapack
        api = ctypes.pythonapi
        api.PyCapsule_GetName.restype = ctypes.c_char_p
        api.PyCapsule_GetName.argtypes = [ctypes.py_object]
        api.PyCapsule_GetPointer.restype = ctypes.c_void_p
        api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        def ptr(mod, name):
            cap = mod.__pyx_capi__[name]
            return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
        _C_KERNEL = (fn, ptr(cython_blas, "dgemm"), ptr(cython_blas, "dtrsm"),
                     ptr(cython_lapack, "dpotri"))
    except Exception as e:
        _C_KERNEL = False
        # Said once per process: the NumPy loop gives the same result, but
        # several times more slowly.
        print("[remlax] C kernel of the selected inversion unavailable (%s). NumPy loop, "
              "same result, slower." % e, file=sys.stderr, flush=True)
    return _C_KERNEL or None


def _dev_kernel():
    """Path of the kernel compiled from a development checkout, or None.

    Only used when the solver runs outside the installed R package: the
    source sits in src/ of the repository, two levels above this package. The
    library goes to $REMLAX_CACHE, or to a folder of the temporary directory."""
    src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "src", "selinv_c.c"))
    if not os.path.exists(src):
        return None
    import tempfile
    cache = os.environ.get("REMLAX_CACHE") or os.path.join(tempfile.gettempdir(), "remlax_cache")
    with open(src, "rb") as f:
        h = hashlib.sha1(f.read()).hexdigest()[:12]
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, "libselinv_%s.so" % h)
    if not os.path.exists(so):
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            return None
        tmp = "%s.%d.tmp" % (so, os.getpid())
        subprocess.run([cc, "-O2", "-shared", "-fPIC", src, "-o", tmp],
                       check=True, capture_output=True)
        os.replace(tmp, so)
    return so


def _set_cholmod_options():
    """Set the CHOLMOD options, which are global in cvxopt.

    The factor is supernodal, without postordering, and only the supplied
    ordering is tried (nmethods = 1). By default CHOLMOD compares the
    supplied ordering with AMD and keeps the better one, which silently
    changed the order of the factor.
    """
    from cvxopt import cholmod
    cholmod.options["supernodal"] = 2
    cholmod.options["postorder"] = False
    cholmod.options["nmethods"] = 1


def _splu(A, order):
    """SuperLU in symmetric mode, without pivoting, with column order `order`."""
    return spla.splu(A, permc_spec=order, diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def metis_order(A):
    """Nested dissection order (METIS, through pymetis) of the symmetric
    pattern A, or None when pymetis is missing.

    The minimum degree order of SuperLU (MMD) degrades as soon as C has very
    full rows (genetic terms with a dense relationship matrix, IGE
    neighbourhoods). METIS cuts the graph of C recursively by separators,
    which keeps the factor much smaller (dev/notes.md). Returns perm, the
    order to apply: the old unknown in position k is perm[k].
    """
    try:
        import pymetis
    except ImportError:
        return None
    G = sp.csr_matrix(A, dtype=np.float64)
    G = (G - sp.diags(G.diagonal())).tocsr()
    G.eliminate_zeros()
    if G.nnz == 0:
        return np.arange(A.shape[0], dtype=np.int64)
    try:
        adj = pymetis.CSRAdjacency(G.indptr.astype(np.int32), G.indices.astype(np.int32))
        perm, _ = pymetis.nested_dissection(adjacency=adj)
    except (AttributeError, TypeError):
        perm, _ = pymetis.nested_dissection(xadj=G.indptr, adjncy=G.indices)
    return np.asarray(perm, dtype=np.int64)


class SparseFactor:
    """Successive factorisations of matrices with the SAME pattern.

    motif: symmetric sparse matrix (full pattern, both triangles).
    ii, jj: wanted entries of C^-1 (original indices, 0-based).
    order: "auto" or "metis" (METIS when available), anything else for the
        minimum degree of SuperLU.
    backend: "auto" or "cholmod" (CHOLMOD when available), anything else for
        SuperLU.
    """

    def __init__(self, motif, ii=None, jj=None, order="auto", backend="auto"):
        A = sp.csc_matrix(motif, dtype=np.float64).copy()
        A.data[:] = 1.0
        m = A.shape[0]
        self.m = m
        # Diagonally dominant matrix with the same pattern, used for the analysis.
        dg = np.asarray(abs(A).sum(axis=1)).ravel() + 1.0
        A0 = (A - sp.diags(A.diagonal()) + sp.diags(dg)).tocsc()
        # Elimination order: METIS nested dissection when pymetis is installed
        # (order="auto" or "metis"), minimum degree of SuperLU otherwise.
        metis_perm = metis_order(A0) if order in ("auto", "metis") else None
        if order == "metis" and metis_perm is None:
            raise ImportError("METIS order requested but pymetis is not installed")
        self.order = "metis" if metis_perm is not None else "mmd"
        if metis_perm is not None:
            self.perm = metis_perm
        else:
            lu0 = _splu(A0, "MMD_AT_PLUS_A")
            # SciPy convention: perm_c[i] is the NEW position of the old column
            # i (Pr A Pc = L U, Pc[i, perm_c[i]] = 1). The order to apply, old
            # column in position k, is therefore argsort(perm_c). Read the
            # other way round, the permutation destroys the minimum degree
            # order and the factor becomes nearly dense (dev/notes.md).
            self.perm = np.argsort(np.asarray(lu0.perm_c, dtype=np.int64))
        self.pinv = np.empty(m, dtype=np.int64); self.pinv[self.perm] = np.arange(m)
        # Factorisation engine: CHOLMOD (through cvxopt, which ships it) when
        # it is installed, SuperLU otherwise. CHOLMOD uses the symmetry (one
        # factor, half the memory of an LU). It also lacks the internal limit
        # of SuperLU, which refuses factors larger than about 2 GB.
        self.backend = "superlu"
        if backend in ("auto", "cholmod"):
            try:
                import cvxopt  # noqa: F401
                from cvxopt import cholmod  # noqa: F401
                self.backend = "cholmod"
            except ImportError:
                if backend == "cholmod":
                    raise
        Ap0 = A0[self.perm][:, self.perm].tocsc()
        if self.backend == "cholmod":
            try:
                self._init_cholmod(Ap0)
                if ii is not None:
                    self.prepare_extraction(ii, jj)
                    self._cholmod_entries(ii, jj)
                return
            except RuntimeError:
                if backend == "cholmod":
                    raise
                self.backend = "superlu"           # fallback: SuperLU, same order
        lu1 = _splu(Ap0, "NATURAL")
        if not (np.array_equal(lu1.perm_c, np.arange(m)) and np.array_equal(lu1.perm_r, np.arange(m))):
            # SuperLU permuted again (postorder of the tree): compose the two.
            q = np.asarray(lu1.perm_c, dtype=np.int64)
            self.perm = self.perm[np.argsort(q)]
            self.pinv[self.perm] = np.arange(m)
            Ap0 = A0[self.perm][:, self.perm].tocsc()
            lu1 = _splu(Ap0, "NATURAL")
        L = sp.csc_matrix(lu1.L)
        L.sort_indices()
        self._L_indices = L.indices.copy(); self._L_indptr = L.indptr.copy()
        self._build_supernodes(L)
        self.lu = None
        if ii is not None:
            self.prepare_extraction(ii, jj)

    # ------------------------------------------------------------------ CHOLMOD
    @staticmethod
    def _cvx(vals, I, J, m):
        from cvxopt import spmatrix, matrix
        return spmatrix(matrix(np.ascontiguousarray(vals, dtype=np.float64)), I, J, (m, m))

    def _init_cholmod(self, Ap0):
        """Symbolic analysis of CHOLMOD in OUR order (identity on the matrix
        already permuted, no postordering), then the pattern of the factor L
        for the supernodes of the selected inversion."""
        from cvxopt import cholmod, matrix
        m = self.m
        _set_cholmod_options()
        # Random off-diagonal values: with 1 everywhere, exact cancellations
        # would remove entries from the pattern returned by getfactor.
        S = sp.triu(Ap0, 1).tocoo()
        v = np.random.default_rng(0).uniform(0.5, 1.5, S.nnz)
        S = sp.coo_matrix((v, (S.row, S.col)), shape=(m, m))
        S = S + S.T
        dg = np.asarray(S.sum(axis=1)).ravel() + 1.0
        T = sp.tril(S + sp.diags(dg)).tocoo()
        I = matrix(T.row.astype(np.int64)); J = matrix(T.col.astype(np.int64))
        A = self._cvx(T.data, I, J, m)
        self._identity_perm = matrix(np.arange(m, dtype=np.int64))
        self.F = cholmod.symbolic(A, p=self._identity_perm)
        cholmod.numeric(A, self.F)
        colptr, rowind, val = (np.array(v).ravel() for v in cholmod.getfactor(self.F).CCS)
        colptr = colptr.astype(np.int64); rowind = rowind.astype(np.int64)
        # Are the rows sorted within each column? If not, keep the sorting
        # permutation once.
        col = np.repeat(np.arange(m), np.diff(colptr))
        order = np.lexsort((rowind, col))
        self._L_row_order = None if np.array_equal(order, np.arange(len(order))) else order
        if self._L_row_order is not None:
            rowind = rowind[order]
        L = sp.csc_matrix((np.ones(len(rowind)), rowind, colptr), shape=(m, m))
        self._L_indices = L.indices.copy(); self._L_indptr = L.indptr.copy()
        self._diag_pos = self._L_indptr[:-1]                 # diagonal at the top of each column
        self._L_keys = np.repeat(np.arange(m, dtype=np.int64), np.diff(self._L_indptr)) * m + self._L_indices
        if not np.array_equal(self._L_indices[self._diag_pos], np.arange(m)):
            raise RuntimeError("CHOLMOD: diagonal missing at the head of a column")
        # Safeguard: the factor must be the one of OUR order (L L' v = A v).
        # Without nmethods = 1, CHOLMOD also tried its own orders and kept
        # the cheapest, and the selected inversion then failed.
        Lv = sp.csc_matrix((val[self._L_row_order] if self._L_row_order is not None else val,
                            self._L_indices, self._L_indptr), shape=(m, m))
        v = np.random.default_rng(1).normal(size=m)
        Af = (sp.tril(S + sp.diags(dg)) + sp.triu(S, 1)).tocsr()
        if np.linalg.norm(Lv @ (Lv.T @ v) - Af @ v) > 1e-8 * np.linalg.norm(Af @ v):
            raise RuntimeError("CHOLMOD: the factor is not in the imposed order")
        self._build_supernodes(L)
        self.lu = None

    def _cholmod_entries(self, ii, jj):
        """Permuted lower-triangle indices of the entries (ii, jj) of C. The
        values of C then go straight to CHOLMOD, without rebuilding the full
        sparse matrix at each evaluation."""
        # Fixed pattern, copied values. Building a cvxopt spmatrix from
        # triplets (duplicates summed) at every evaluation was the dominant
        # cost on large models (dev/notes.md). The pattern is therefore set
        # once, in the CCS order of cvxopt (column, then row). Each evaluation
        # only copies the values, summed by bincount on the CCS position of
        # each entry.
        from cvxopt import matrix, spmatrix
        a = self.pinv[np.asarray(ii, dtype=np.int64)]
        b = self.pinv[np.asarray(jj, dtype=np.int64)]
        I = np.maximum(a, b); J = np.minimum(a, b)
        key = J * self.m + I                                 # CCS: column major
        uniq, pos = np.unique(key, return_inverse=True)
        self._entry_pos = pos.astype(np.int64)
        self._entry_nnz = len(uniq)
        Iu = (uniq % self.m).astype(np.int64); Ju = (uniq // self.m).astype(np.int64)
        self._A = spmatrix(matrix(np.ones(len(uniq))), matrix(Iu), matrix(Ju), (self.m, self.m))
        # cvxopt stores its values in sorted CCS order: checked once.
        Ic = np.array(self._A.I).ravel(); Jc = np.array(self._A.J).ravel()
        if not (np.array_equal(Ic, Iu) and np.array_equal(Jc, Ju)):
            raise RuntimeError("CHOLMOD: unexpected CCS order from cvxopt")

    def factorise_values(self, x):
        """Factorise C given by its values at the entries (ii, jj). Returns log|C|."""
        from cvxopt import matrix
        v = np.bincount(self._entry_pos, weights=np.asarray(x, dtype=np.float64),
                        minlength=self._entry_nnz)
        self._A.V = matrix(v)
        return self._cholmod_numeric(self._A)

    def _cholmod_numeric(self, A):
        from cvxopt import cholmod
        # getfactor converts the factor in place and leaves it symbolic
        # (cholmod_factor_to_sparse), so the analysis is redone every time.
        # In our fixed order it is linear in nnz(A), negligible against the
        # numeric factorisation.
        try:
            _set_cholmod_options()
            self.F = cholmod.symbolic(A, p=self._identity_perm)
            cholmod.numeric(A, self.F)
        except ArithmeticError:
            raise np.linalg.LinAlgError("C is not positive definite (CHOLMOD)")
        colptr, rowind, val = (np.array(v).ravel() for v in cholmod.getfactor(self.F).CCS)
        if len(val) == len(self._L_indices) and self._L_row_order is None and \
                np.array_equal(rowind, self._L_indices):
            pass                                          # same pattern as the analysis
        else:
            # Converting the supernodal factor drops exact numerical zeros.
            # The returned pattern is INCLUDED in the pattern of the analysis
            # (done on a diagonally dominant matrix). Each value is put back
            # at its column x row key, as for SuperLU.
            col = np.repeat(np.arange(self.m, dtype=np.int64), np.diff(colptr.astype(np.int64)))
            key = col * self.m + rowind.astype(np.int64)
            pos = np.searchsorted(self._L_keys, key)
            if np.any(pos >= len(self._L_keys)) or np.any(self._L_keys[np.minimum(pos, len(self._L_keys) - 1)] != key):
                raise RuntimeError("CHOLMOD: the pattern of the factor leaves that of the analysis")
            v2 = np.zeros(len(self._L_indices)); v2[pos] = val
            val = v2
        d = val[self._diag_pos]
        if np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise np.linalg.LinAlgError("C is not positive definite")
        self._L_values = val
        self._flat = np.zeros(self.off[-1])
        self._flat[self._flat_map] = val
        return float(2.0 * np.sum(np.log(d)))

    def _solve_supernodes(self, Bp):
        """L L' X = Bp by supernodes: one BLAS trsm and one gemm per block.
        The SciPy triangular solve (spsolve_triangular) proceeds row by row
        without BLAS, and it dominated the evaluation on large models
        (dev/notes.md)."""
        from scipy.linalg import solve_triangular
        X = np.array(Bp, dtype=np.float64, order="C")
        is_vector = X.ndim == 1
        if is_vector:
            X = X[:, None]
        blocks = []
        for s_ in range(self.ns):
            nc, nr = int(self.ncol[s_]), int(self.nrow[s_])
            blk = self._flat[self.off[s_]:self.off[s_ + 1]].reshape(nc, nr).T
            blocks.append((int(self.sup[s_]), int(self.sup[s_ + 1]), blk[:nc], blk[nc:], self.lignes[s_][nc:]))
        for c0, c1, Ljj, Lr, R in blocks:
            Xc = solve_triangular(Ljj, X[c0:c1], lower=True, check_finite=False)
            X[c0:c1] = Xc
            if len(R):
                X[R] -= Lr @ Xc
        for c0, c1, Ljj, Lr, R in reversed(blocks):
            Xc = X[c0:c1]
            if len(R):
                Xc = Xc - Lr.T @ X[R]
            X[c0:c1] = solve_triangular(Ljj, Xc, lower=True, trans="T", check_finite=False)
        return X[:, 0] if is_vector else X

    # ------------------------------------------------------------------ structure
    def _build_supernodes(self, L):
        """Supernodes of the factor L, relaxed merging, and the index tables
        of the numeric steps.

        Sets sup (first column of each supernode), lignes (rows of each
        supernode), ncol, nrow, off (offset of each block in the flat
        storage) and sn_of (supernode of each column). Also sets the map from
        the entries of L (CSC order) to the flat storage, the arrays of the C
        kernel, and the gather plan of Z_RR for the selected inversion.
        """
        m = self.m
        ip, ix = L.indptr, L.indices
        cnt = np.diff(ip)
        # Fundamental supernodes.
        starts = [0]
        for j in range(1, m):
            jm = j - 1
            r0 = ix[ip[jm]:ip[jm + 1]]
            # Column j extends the supernode if L[j, j-1] is structurally
            # nonzero and both columns have the same structure below the
            # diagonal.
            if cnt[jm] == cnt[j] + 1 and len(r0) > 1 and r0[1] == j:
                continue
            starts.append(j)
        starts.append(m)
        sup = np.array(starts, dtype=np.int64)
        rows_of = []
        for s in range(len(sup) - 1):
            j0 = sup[s]
            rows_of.append(ix[ip[j0]:ip[j0 + 1]].astype(np.int64))      # rows of the first column
        # Relaxed merging with the contiguous parent (CHOLMOD's rules).
        def trapezoid(nr, nc):          # entries of an nr x nc trapezoidal block
            return nc * nr - nc * (nc - 1) // 2
        merged_sup, merged_rows = [sup[0]], []
        cur_c0, cur_rows = sup[0], rows_of[0]
        cur_struct = trapezoid(len(rows_of[0]), sup[1] - sup[0])
        for s in range(1, len(sup) - 1):
            c0, rows = sup[s], rows_of[s]
            ncol_cur = c0 - cur_c0
            parent = len(cur_rows) > ncol_cur and cur_rows[ncol_cur] == c0
            merge = False
            s_struct = trapezoid(len(rows), sup[s + 1] - c0)
            if parent:
                # The rows of J below its block lie in {columns of J+1} U
                # {rows of J+1} (a property of the factor). The union is the
                # block of J+1 extended by the columns of J.
                ncol = int(sup[s + 1] - cur_c0)
                nr_new = ncol_cur + len(rows)
                after = trapezoid(nr_new, ncol)
                zeros = (after - cur_struct - s_struct) / max(after, 1)
                merge = (ncol <= 4 or (ncol <= 16 and zeros <= 0.8) or
                         (ncol <= 48 and zeros <= 0.1) or zeros <= 0.05)
            if merge:
                cur_rows = np.concatenate([np.arange(cur_c0, c0, dtype=np.int64), rows])
                cur_struct = cur_struct + s_struct
            else:
                merged_rows.append(cur_rows); merged_sup.append(c0)
                cur_c0, cur_rows, cur_struct = c0, rows, s_struct
        merged_rows.append(cur_rows); merged_sup.append(m)
        self.sup = np.array(merged_sup, dtype=np.int64)
        self.lignes = merged_rows
        ns = len(self.sup) - 1
        self.ns = ns
        self.ncol = np.diff(self.sup)
        self.nrow = np.array([len(r) for r in merged_rows], dtype=np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.nrow * self.ncol)]).astype(np.int64)
        self.sn_of = np.repeat(np.arange(ns), self.ncol)
        # Map from the entries of L (CSC order) to the flat block storage.
        col_of = np.repeat(np.arange(m), cnt)
        J = self.sn_of[col_of]
        lrow = np.empty(len(ix), dtype=np.int64)
        for s in range(ns):
            a, b = ip[self.sup[s]], ip[self.sup[s + 1]]
            lrow[a:b] = np.searchsorted(self.lignes[s], ix[a:b])
        lcol = col_of - self.sup[J]
        self._flat_map = self.off[J] + lrow + self.nrow[J] * lcol
        self._col_of = col_of
        self._keys = col_of * m + ix.astype(np.int64)        # increasing (sorted CSC)
        # Arrays of the C kernel (CHOLMOD supernodal layout).
        self._c_sup = self.sup.astype(np.int32)
        self._c_pi = np.concatenate([[0], np.cumsum(self.nrow)]).astype(np.int32)
        self._c_s = np.concatenate(self.lignes).astype(np.int32)
        self._c_off = self.off.astype(np.int64)
        self._c_sn = self.sn_of.astype(np.int32)
        # Gather plan of Z_RR: for each J, the supernodes K that hold the
        # columns of R, and the positions in their blocks.
        self._gather = []
        for s in range(ns):
            nJ = self.ncol[s]; R = self.lignes[s][nJ:]
            if not len(R):
                self._gather.append(None); continue
            Ks = self.sn_of[R]
            groups = []
            for K in np.unique(Ks):
                pc = np.where(Ks == K)[0]
                pr = np.where(R >= R[pc[0]])[0]
                rK = np.searchsorted(self.lignes[K], R[pr])
                if np.any(rK >= self.nrow[K]) or np.any(self.lignes[K][np.minimum(rK, self.nrow[K] - 1)] != R[pr]):
                    raise RuntimeError("selinv: non-chordal pattern (supernodes %d and %d)" % (s, K))
                cK = R[pc] - self.sup[K]
                groups.append((int(K), pr, pc, rK, cK))
            self._gather.append(groups)

    def prepare_extraction(self, ii, jj):
        """Positions, in the flat block storage, of the wanted entries of C^-1."""
        pi_, pj_ = self.pinv[np.asarray(ii, dtype=np.int64)], self.pinv[np.asarray(jj, dtype=np.int64)]
        lo, co = np.maximum(pi_, pj_), np.minimum(pi_, pj_)
        K = self.sn_of[co]
        r = np.empty(len(lo), dtype=np.int64)
        for k in np.unique(K):
            sel = np.where(K == k)[0]
            rr = np.searchsorted(self.lignes[k], lo[sel])
            if np.any(rr >= self.nrow[k]) or np.any(self.lignes[k][np.minimum(rr, self.nrow[k] - 1)] != lo[sel]):
                raise RuntimeError("selinv: requested entry outside the pattern of the factor")
            r[sel] = rr
        self._extract = self.off[K] + r + self.nrow[K] * (co - self.sup[K])

    # ----------------------------------------------------------------- numeric
    def factorise(self, C):
        """Factorise C (same pattern as at construction). Returns log|C|."""
        Cp = sp.csc_matrix(C)[self.perm][:, self.perm].tocsc()
        if self.backend == "cholmod":
            from cvxopt import matrix
            T = sp.tril(Cp).tocoo()
            return self._cholmod_numeric(self._cvx(T.data, matrix(T.row.astype(np.int64)),
                                                   matrix(T.col.astype(np.int64)), self.m))
        lu = _splu(Cp, "NATURAL")
        if not (np.array_equal(lu.perm_c, np.arange(self.m)) and np.array_equal(lu.perm_r, np.arange(self.m))):
            raise np.linalg.LinAlgError("selinv: SuperLU permuted, non-symmetric factorisation")
        d = lu.U.diagonal()
        if np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise np.linalg.LinAlgError("C is not positive definite")
        self.lu, self.d = lu, d
        return float(np.sum(np.log(d)))

    def solve(self, B):
        """C^-1 B from the current factorisation (B a vector or a matrix)."""
        B = np.asarray(B, dtype=np.float64)
        if self.backend == "cholmod":
            Xp = self._solve_supernodes(B[self.perm])
            X = np.empty_like(Xp); X[self.perm] = Xp
            return X
        Xp = self.lu.solve(B[self.perm])
        X = np.empty_like(Xp); X[self.perm] = Xp
        return X

    def selected_inverse(self):
        """C^-1 at the prepared entries (prepare_extraction)."""
        if self.backend == "cholmod":
            return self._selinv_flat(self._flat.copy())
        L = sp.csc_matrix(self.lu.L)
        L.sort_indices()
        if np.array_equal(L.indptr, self._L_indptr) and np.array_equal(L.indices, self._L_indices):
            flat_map, col_of = self._flat_map, self._col_of
        else:
            # SuperLU sometimes drops exact zeros (dev/notes.md). The actual
            # pattern is INCLUDED in the pattern of the analysis, where each
            # entry is found again by its column x row key.
            col_of = np.repeat(np.arange(self.m), np.diff(L.indptr))
            key = col_of * self.m + L.indices
            pos = np.searchsorted(self._keys, key)
            if np.any(pos >= len(self._keys)) or np.any(self._keys[np.minimum(pos, len(self._keys) - 1)] != key):
                raise RuntimeError("selinv: the pattern of the factor leaves that of the analysis")
            flat_map = self._flat_map[pos]
        flat = np.zeros(self.off[-1])
        flat[flat_map] = L.data * np.sqrt(self.d)[col_of]
        return self._selinv_flat(flat)

    def _selinv_flat(self, flat):
        """Selected inversion from the Cholesky factor stored by supernodes
        (flat). Returns C^-1 at the prepared entries."""
        kernel = _c_kernel()
        if kernel is not None:
            fn, pg, pt, pp = kernel
            Z = np.zeros(self.off[-1])
            P = ctypes.POINTER
            status = fn(int(self.ns), self._c_off.ctypes.data_as(P(ctypes.c_longlong)),
                        self._c_sup.ctypes.data_as(P(ctypes.c_int)), self._c_pi.ctypes.data_as(P(ctypes.c_int)),
                        self._c_s.ctypes.data_as(P(ctypes.c_int)), self._c_sn.ctypes.data_as(P(ctypes.c_int)),
                        flat.ctypes.data_as(P(ctypes.c_double)), Z.ctypes.data_as(P(ctypes.c_double)),
                        pg, pt, pp)
            if status != 0:
                raise np.linalg.LinAlgError("selinv (C kernel): status %d" % status)
            return Z[self._extract]
        Zb = [None] * self.ns
        for s in range(self.ns - 1, -1, -1):
            nJ, nr = int(self.ncol[s]), int(self.nrow[s])
            Lb = flat[self.off[s]:self.off[s + 1]].reshape((nr, nJ), order="F")
            Ljj = np.tril(Lb[:nJ])
            Linv = solve_triangular(Ljj, np.eye(nJ), lower=True, check_finite=False)
            Zjj = Linv.T @ Linv
            if nr > nJ:
                nR = nr - nJ
                Lt = Lb[nJ:] @ Linv                                   # L_RJ L_JJ^-1
                Zrr = np.zeros((nR, nR))
                for (K, pr, pc, rK, cK) in self._gather[s]:
                    Zrr[np.ix_(pr, pc)] = Zb[K][np.ix_(rK, cK)]
                il = np.triu_indices(nR, 1)
                Zrr[il] = Zrr.T[il]
                ZRJ = -Zrr @ Lt
                Zjj = Zjj - Lt.T @ ZRJ
                Zb[s] = np.vstack([Zjj, ZRJ])
            else:
                Zb[s] = Zjj
        Z = np.empty(self.off[-1])
        for s in range(self.ns):
            Z[self.off[s]:self.off[s + 1]] = Zb[s].ravel(order="F")
        return Z[self._extract]
