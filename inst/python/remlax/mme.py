"""Engine on the mixed model equations (MME), AI-REML in the manner of asreml.

The engine never forms V. With W = [X Z]:
    C = W' R^-1 W + diag(0, G^-1),      G = (+)_k Sigma_k (x) K_k
    -2 logL = (n-p) log 2pi + log|R| + log|G| + log|C| + log|det R_x|^2 + y'Py
    y'Py = y'R^-1 y - r' C^-1 r,         r = W' R^-1 y
(X is replaced by Q of its QR decomposition: incidence.orthonormal_X.)

Interface: like model.make_objective and ai.make_fga, the engine returns the
value, the gradient and the AI matrix in theta, with the conventions of the
dense engine (same parameters, structures and bounds), so the AI optimiser,
its fallbacks, the polishing and the diagnostics of fit.py serve both engines.

Derivatives are taken first with respect to the internal coordinates (the
entries of each Sigma and the level parameters), where the formulas are
simple, then brought to theta by the exact JAX Jacobian:
    g_theta = J' s,      AI_theta = Q_theta' P Q_theta,  Q_theta = Q J.
  Term k, entry (a, b) of Sigma_k:
    M = q Sigma^-1 - Sigma^-1 (T + U'K^-1U) Sigma^-1,  T_ab = <C^-1_{kk,ab}, K^-1>
    score M_aa, or 2 M_ab; working column Z vec(U Sigma^-1 E_ab).
  Level parameter i of term k:
    score t dlog|K| + <Sigma^-1, T'> + <Sigma^-1, U' dK^-1 U>;
    working column Z vec(dK K^-1 U).
  Residual by units, pattern g: H_g = S^-1 (n_g S - E'E - D) S^-1.
  Structured residual (dense R per section):
    tr(R^-1 dR) - <C^-1, W'R^-1 dR R^-1 W> - Py' dR Py; column dR Py.
  AI = Q'R^-1Q - (W'R^-1Q)' C^-1 (W'R^-1Q).

C is a combination of components on a common pattern: fixed components
(residual patterns W_a'W_b, sparse parts of K^-1) whose coefficient alone
changes, and variable components (dense K^-1 of a parametrised structure,
block W'R^-1W of a structured residual) whose values change. The traces of
the score are scalar products between C^-1 read on that pattern and the
components.

Inversion: dense C by LAPACK (SciPy) on CPU or by JAX on GPU; sparse C by
the supernodal Cholesky and selected inversion of selinv.py, on CPU; hybrid
(mme_hybride.py): dense separator and dense blocks in batches.

Out of scope (fallback to the dense engine, see out_of_scope): residual
"none" (the equations need R^-1), rr structure (singular Sigma), spatial lag
(y transformed by a parameter), Gaussian kernels (K nearly singular).
"""
from . import _x64  # noqa: F401
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .layout import level_kind, split_theta, term_n_params, res_sections, sec_n_params
from .incidence import orthonormal_X
from .residual import _section_V
from .structures import build_sigma, n_params
from .levels import level_chol
from .selinv import SparseFactor

LOG2PI = float(np.log(2.0 * np.pi))


# Conditioning guard of the Sigma inverted by the MME engine: minimal ratio of
# the diagonals of the Cholesky factor (1e-5, a condition number of about
# 1e10). Optima on the edge of the cone stay admissible: the relative floor of
# the us blocks there is 1e-8 on the eigenvalues (ratio about 1e-4).
DIAG_CHOL_MIN = 1e-5


# Compiled Sigma functions, cached by structure (type, t, rank): in the
# persistent server a second fit of the same structure reuses them.
_SIGMA_CACHE = {}


def _sigma_functions(st, t, rk):
    """(Sigma(theta), Jacobian of its vech in theta), compiled."""
    key = (st, int(t), int(rk))
    if key not in _SIGMA_CACHE:
        jpa = jnp.asarray(_pairs(st, t))

        def vech(th):
            S = build_sigma(th, st, t, rk)
            return S[jpa[:, 0], jpa[:, 1]]
        _SIGMA_CACHE[key] = (jax.jit(lambda th: build_sigma(th, st, t, rk)),
                             jax.jit(jax.jacfwd(vech)))
    return _SIGMA_CACHE[key]


def out_of_scope(terms, res, lag=None):
    """Why the model belongs to the dense engine, or None."""
    if lag is not None:
        return "spatial lag (rx_lag): y transformed by a parameter"
    if res.get("struct") == "none":
        return "residual 'none': the mixed model equations require R^-1"
    for tm in terms:
        if tm["struct"] == "rr":
            return "rr structure of the term '%s': singular Sigma, no inverse" % tm["name"]
    for sec in res_sections(res, 0):
        if sec["struct"] == "rr":
            return "rr structure of the residual: singular"
    # Gaussian kernels: the dense engine needs only a factor (with a jitter),
    # the MME inverts K, whose inverse then loses its digits (dev/notes.md).
    gauss = ("gau", "igau", "agau")
    for tm in terms:
        lvl = tm.get("lvl")
        if lvl in gauss or (lvl == "sep" and any(k in gauss for k, _ in (tm.get("lvl_parts") or []))):
            return "Gaussian kernel of the term '%s': K nearly singular, its inverse is unusable" % tm["name"]
    for sec in res_sections(res, 0):
        if sec.get("lvl") in gauss:
            return "Gaussian kernel of the residual: R nearly singular"
    return None


# ------------------------------------------------------------------------------
# Level structures in closed sparse form (products of id and ar1)
# ------------------------------------------------------------------------------
def _ar1_parts(n):
    """A^-1 = c0 I + c1 D + c2 T for A_ij = rho^|i-j| (D: interior diagonal)."""
    I = sp.identity(n, format="csr")
    dd = np.ones(n)
    dd[0] = 0.0; dd[-1] = 0.0
    D = sp.diags(dd, format="csr")
    T = (sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr")
         if n > 1 else sp.csr_matrix((1, 1)))
    return [I, D, T]


def _ar1_c(r):
    u = 1.0 - r * r
    return np.array([1.0 / u, r * r / u, -r / u])


def _ar1_dc(r):                                       # d c / d rho
    u = 1.0 - r * r
    return np.array([2 * r / u ** 2, 2 * r / u ** 2, -(1 + r * r) / u ** 2])


def _ar1_corr(r, n):
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return r ** h


def _ar1_dcorr(r, n):
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.where(h == 0, 0.0, h * r ** np.maximum(h - 1, 0))


class _SparseLevels:
    """K = (x)_f A_f, each factor id or ar1, rho = tanh(theta) as in the dense
    engine (levels.ar1_chol). K^-1 is a sum of Kronecker products of fixed
    parts; only their coefficients depend on rho."""

    def __init__(self, factors):
        self.fac = [(k, int(d)) for k, d in factors]
        self.ar1 = [i for i, (k, _) in enumerate(self.fac) if k == "ar1"]
        self.np = len(self.ar1)
        self.q = int(np.prod([d for _, d in self.fac]))
        lists = [(_ar1_parts(d) if k == "ar1" else [sp.identity(d, format="csr")])
                 for (k, d) in self.fac]
        self.combi = [()]
        for L in lists:
            self.combi = [c + (j,) for c in self.combi for j in range(len(L))]
        self.parts = []
        for c in self.combi:
            M = None
            for f, j in enumerate(c):
                M = lists[f][j] if M is None else sp.kron(M, lists[f][j], format="csr")
            self.parts.append(sp.csr_matrix(M))

    def coef(self, th, i_der=None):
        """Coefficients of the parts in K^-1, or their derivative in theta_i."""
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        out = np.ones(len(self.combi))
        for h, c in enumerate(self.combi):
            for j, f in enumerate(self.ar1):
                if j == i_der:
                    out[h] *= _ar1_dc(rho[j])[c[f]] * (1.0 - rho[j] ** 2)
                else:
                    out[h] *= _ar1_c(rho[j])[c[f]]
        return out

    def logdet(self, th):
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        return sum((self.q // self.fac[f][1]) * (self.fac[f][1] - 1) * np.log(1 - rho[i] ** 2)
                   for i, f in enumerate(self.ar1))

    def dlogdet(self, th, i):
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        d = self.fac[self.ar1[i]][1]
        return -(self.q // d) * (d - 1) * 2.0 * rho[i]       # d/dtheta, rho' = 1 - rho^2

    def apply_dK(self, th, i, B):
        """(dK / dtheta_i) B without a q x q matrix, factor by factor; K B when
        i is None."""
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        mats = []
        for j, (k, d) in enumerate(self.fac):
            if k == "id":
                mats.append(None)
                continue
            ia = self.ar1.index(j)
            mats.append(_ar1_dcorr(rho[ia], d) * (1.0 - rho[ia] ** 2) if ia == i
                        else _ar1_corr(rho[ia], d))
        dims = [d for _, d in self.fac]
        B = np.asarray(B, dtype=np.float64)
        out = np.empty_like(B)
        for c in range(B.shape[1]):
            T = B[:, c].reshape(dims)
            for ax, Mx in enumerate(mats):
                if Mx is not None:
                    T = np.moveaxis(np.tensordot(Mx, T, axes=([1], [ax])), 0, ax)
            out[:, c] = T.ravel()
        return out


def _sparse_factors(tm):
    """[(kind, dimension)] of a level structure that is a product of id and
    ar1 factors, or None."""
    lvl = level_kind(tm)
    if lvl == "ar1":
        return [("ar1", int(tm["q"]))]
    if lvl == "ar1ar1":
        return [("ar1", int(tm["dims"][0])), ("ar1", int(tm["dims"][1]))]
    if lvl == "sep" and tm.get("lvl_parts") and all(k in ("id", "ar1") for k, _ in tm["lvl_parts"]):
        return [(k, int(d)) for k, d in tm["lvl_parts"]]
    return None


def _pairs(st, t):
    """(a, b) entries of Sigma that carry a parameter: the diagonal for iid and
    diag, the lower triangle row by row otherwise."""
    if st in ("iid", "diag"):
        return np.column_stack([np.arange(t), np.arange(t)])
    return np.array([(a, b) for a in range(t) for b in range(a + 1)], dtype=np.int64).reshape(-1, 2)


class _Inadmissible(Exception):
    """A point where a Cholesky fails or a matrix is too ill-conditioned: the
    evaluation returns f = inf, and the optimiser rejects the step."""


def _inverse_from_factor(LK):
    """K^-1 from a factor K = LK LK'.

    Triangular factor (the case of rx_term, Cholesky of K): LK^-1 by
    substitution (dtrsm, parallel BLAS 3), then K^-1 = LK^-T LK^-1 by dsyrk.
    Triangularity is not tested on the whole matrix: the triangular solve is
    tried and checked by K^-1 (LK LK' v) = v on a random vector. Any other
    factor, or a failed check: general LU."""
    import scipy.linalg as _sl
    q = LK.shape[0]

    def full(Li):
        Ki = _sl.blas.dsyrk(1.0, Li, trans=1)        # upper triangle of Li' Li
        Ki += Ki.T
        Ki[np.diag_indices(q)] *= 0.5
        return Ki

    if LK.shape[0] == LK.shape[1] and q > 1:
        v = np.random.default_rng(0).normal(size=q)
        w = LK @ (LK.T @ v)
        # First row (resp. column) zero off the diagonal: candidate lower (resp.
        # upper) triangular. A non-triangular factor leaves a residual of order
        # 1, a triangular one of order cond(K) x 1e-16.
        for lower, ok in ((True, (LK[0, 1:] == 0).all()), (False, (LK[1:, 0] == 0).all())):
            if not ok:
                continue
            Ki = full(_sl.solve_triangular(LK, np.eye(q), lower=lower, check_finite=False))
            if np.linalg.norm(Ki @ w - v) <= 1e-4 * np.linalg.norm(v):
                return Ki
    Li = np.linalg.solve(LK, np.eye(q))
    return full(Li)


def _block_values(M, diag):
    """Values of a q x q block in the order of the triplets of its component:
    upper triangle row by row for a diagonal block, the whole block otherwise."""
    if diag:
        return M[np.triu_indices(M.shape[0])]
    return M.ravel()


# ------------------------------------------------------------------------------
# The engine
# ------------------------------------------------------------------------------
class MME:
    """The MME engine of one model. evaluate(theta) returns -2logL and, when
    full, its gradient and AI matrix in theta; outputs(theta) returns beta,
    vbeta, the BLUPs and the PEV.

    Construction: the residual on a grid (_setup_grid), the terms
    (_setup_terms), the residual sections (_setup_sections), the pattern of C
    and its components (_components), then the inversion route
    (_choose_inversion)."""

    def __init__(self, terms, res, y, X, inversion="auto", verbose=True, separateur=None):
        t0 = time.time()
        reason = out_of_scope(terms, res)
        if reason:
            raise ValueError("MME engine: " + reason)
        self.verbose = verbose
        self.y = y = np.asarray(y, dtype=np.float64)
        self.X = X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        self.n, self.p = n, p
        self.p_sortie = p
        Xq, self.ldRx = orthonormal_X(X)
        self.Rx = np.linalg.qr(X)[1]
        self.terms, self.res = terms, res
        self._n_obs0 = n                      # original design (hybrid split)
        self._separateur = separateur         # names of the imposed separator terms (hybrid)
        Xq = self._setup_grid(res, Xq)
        self._setup_terms(terms)
        self._setup_sections(res)
        self.W = sp.hstack([sp.csc_matrix(Xq)] + [d["Z"] for d in self.T], format="csc")
        self._components()
        self._choose_inversion(inversion)
        if verbose:
            m = self.m
            print("  [mme] m = %d | pattern of C %.1f %% | inversion %s | %d components | preparation %.1f s"
                  % (m, 100.0 * min(1.0, (self.nU + self._n_block_entries) / (m * (m + 1) / 2)),
                     self.inversion, self.n_comp, time.time() - t0), flush=True)

    def _setup_grid(self, res, Xq):
        """Residual on a grid (ar1, ar1 x ar1, product of id/ar1) of a single
        iid section with one trait: R^-1 = sigma^-2 (x) A_f^-1 is sparse and
        known in closed form. Observations are placed in their cell; each
        missing cell gets an indicator fixed effect that absorbs it. The REML
        of the augmented model is exactly that of the observed data
        (log|V_F| + log|M'V_F^-1 M| = log|V_O|, and n_F - p_F = n - p). C then
        stays sparse and R is never inverted. Returns the design of the fixed
        effects, augmented when the grid applies (y, n and p are updated)."""
        self.grille = None
        n, p, y = self.n, self.p, self.y
        secs0 = res_sections(res, n)
        if len(secs0) == 1 and secs0[0]["struct"] == "iid" and int(secs0[0]["t"]) == 1:
            s0 = secs0[0]
            fac0 = _sparse_factors(dict(s0, q=int(s0.get("n_unit", 0) or 0)))
            un0 = np.asarray(s0.get("unit", []), dtype=np.int64)
            if fac0 is not None and len(un0) == n:
                nF = int(np.prod([dd for _, dd in fac0]))
                if (nF > 0 and un0.min() >= 0 and un0.max() < nF
                        and len(np.unique(un0)) == n):
                    mis = np.setdiff1d(np.arange(nF), un0)
                    yF = np.zeros(nF); yF[un0] = y
                    XqF = np.zeros((nF, p + len(mis)))
                    XqF[un0, :p] = Xq
                    XqF[mis, p + np.arange(len(mis))] = 1.0
                    self.grille = dict(un=un0, mis=mis, nF=nF, fac=fac0)
                    self.y = yF
                    Xq = XqF
                    self.n, self.p = nF, p + len(mis)
        return Xq

    def _setup_terms(self, terms):
        """One dict per term: its columns in C (off), its incidence Z, its
        Sigma functions, and the form of K^-1: "fixe" (identity, given K or
        given precision Kinv), "creux" (product of id and ar1 in closed form)
        or "dense" (parametrised kernel, K built exactly as the dense engine
        does, jitter included)."""
        n, p = self.n, self.p
        self.T = []
        off = p
        for tm in terms:
            t, q = int(tm["t"]), int(tm["q"])
            st, rk = tm["struct"], int(tm["rank"])
            nsig = n_params(st, t, rk)
            nlv = term_n_params(tm) - nsig
            pa = _pairs(st, t)
            sig_f, sig_j = _sigma_functions(st, t, rk)
            d = dict(name=tm["name"], t=t, q=q, struct=st, rank=rk, nsig=nsig, nlv=nlv, off=off,
                     paires=pa, Z=sp.csc_matrix(
                         (tm["zx"], (self._row(tm["zi"]), tm["zj"])), shape=(n, t * q)),
                     sig=sig_f, sig_jac=sig_j)
            lvl = level_kind(tm)
            fac = _sparse_factors(tm)
            if tm.get("Kinv") is not None:
                # Given precision (pedigree), whatever the declared level kind.
                d.update(forme="fixe", parts=[sp.csr_matrix(tm["Kinv"])],
                         ldK=float(tm["Kinv_logdet"]))
            elif lvl == "id":
                d.update(forme="fixe", parts=[sp.identity(q, format="csr")], ldK=0.0)
            elif lvl == "fixed":
                LK = np.asarray(tm["LK"], dtype=np.float64)
                d.update(forme="fixe", parts=[sp.csr_matrix(_inverse_from_factor(LK))],
                         ldK=float(2.0 * np.sum(np.log(np.abs(np.diag(LK))))))
            elif fac is not None:
                nc = _SparseLevels(fac)
                if nc.q != q or nc.np != nlv:
                    raise ValueError("MME engine: levels '%s' of the term '%s' inconsistent"
                                     % (lvl, tm["name"]))
                d.update(forme="creux", niv=nc, parts=nc.parts)
            else:
                # K exactly as in the dense engine, through the factor of
                # level_chol, jitter included: level_corr alone would give
                # another matrix, hence another likelihood.
                def K_of(th, lvl=lvl, q=q, tm=tm):
                    L = level_chol(th, lvl, q, dims=tm.get("dims"), order=tm.get("lvl_order", 0),
                                   coord=tm.get("coord"), parts=tm.get("lvl_parts"),
                                   opts=tm.get("lvl_opts"), expr=tm.get("lvl_expr"))
                    return L @ L.T
                d.update(forme="dense", K=jax.jit(K_of), dK=jax.jit(jax.jacfwd(K_of)),
                         parts=[sp.csr_matrix(np.ones((q, q)))])
            self.T.append(d)
            off += t * q
        self.m = off

    def _setup_sections(self, res):
        """One dict per residual section: on a grid, its sparse levels; without
        structure between units ("id"), its patterns of observed traits per
        unit (motifs, each with the rows of its units); otherwise the compiled
        dense R of the section and its Jacobian."""
        n = self.n
        self.S = []
        for sec in res_sections(res, n):
            rows = np.asarray(sec["rows"], dtype=np.int64)
            st, t, rk = sec["struct"], int(sec["t"]), int(sec["rank"])
            nsig = n_params(st, t, rk)
            lvl = sec.get("lvl", "id")
            d = dict(rows=rows, struct=st, t=t, nsig=nsig, np_=sec_n_params(sec), lvl=lvl)
            if self.grille is not None:
                niv = _SparseLevels(self.grille["fac"])
                d.update(forme="grille", niv=niv, parts=niv.parts)
                self.S.append(d)
                continue
            if lvl == "id":
                pa = _pairs(st, t)
                sig_f, sig_j = _sigma_functions(st, t, rk)
                d.update(paires=pa, sig=sig_f, sig_jac=sig_j)
                tr = np.asarray(sec["trait"], dtype=np.int64)
                un = np.asarray(sec["unit"], dtype=np.int64)
                by_unit = {}
                for i_, (u_, a_) in enumerate(zip(un, tr)):
                    by_unit.setdefault(int(u_), {})[int(a_)] = int(rows[i_])
                grp = {}
                for dct in by_unit.values():
                    tau = tuple(sorted(dct))
                    grp.setdefault(tau, []).append([dct[a_] for a_ in tau])
                d["motifs"] = [dict(tau=np.array(tau, dtype=np.int64), I=np.array(Ls, dtype=np.int64),
                                    n=len(Ls)) for tau, Ls in grp.items()]
            else:
                d.update(R=jax.jit(lambda th, sec=sec: _section_V(th, sec)),
                         dR=jax.jit(jax.jacfwd(lambda th, sec=sec: _section_V(th, sec))))
            self.S.append(d)

    def _row(self, i):
        """Row of W of an observation: its cell on a grid."""
        i = np.asarray(i, dtype=np.int64)
        return i if self.grille is None else self.grille["un"][i]

    # ------------------------------------------------------------ pattern of C
    def _components(self, defer=True):
        """Pattern of C and its components. defer: full blocks (dense kinship)
        are not entered yet; they serve only the sparse and hybrid inversions,
        and _choose_inversion materialises them then."""
        self._blocks_deferred = False
        self._n_block_entries = 0
        W, m, y = self.W, self.m, self.y
        I_, J_, X_, var = [], [], [], []

        def add(i, j, x, variable=False):
            I_.append(np.asarray(i, dtype=np.int64)); J_.append(np.asarray(j, dtype=np.int64))
            X_.append(np.asarray(x, dtype=np.float64)); var.append(variable)
            return len(I_) - 1

        def triu_trip(M):
            T = sp.triu(sp.csr_matrix(M)).tocoo()
            keep = T.data != 0
            return T.row[keep], T.col[keep], T.data[keep]
        for d in self.S:
            if d.get("forme") == "grille":
                d["ic"], d["Wy"], d["yPy"] = [], [], []
                for P in d["parts"]:
                    PW = P @ W
                    d["ic"].append(add(*triu_trip(W.T @ PW)))
                    d["Wy"].append(np.asarray(PW.T @ y).ravel())
                    d["yPy"].append(float(y @ (P @ y)))
                d["yPy"] = np.asarray(d["yPy"])
                continue
            if d["lvl"] == "id":
                for mo in d["motifs"]:
                    tg = len(mo["tau"])
                    Wa = [W[mo["I"][:, a], :] for a in range(tg)]
                    Ya = y[mo["I"]]
                    mo["Wy"] = [np.asarray(Wa[a].T @ Ya) for a in range(tg)]
                    mo["yy"] = Ya.T @ Ya
                    mo["ic"] = -np.ones((tg, tg), dtype=np.int64)
                    for a in range(tg):
                        for b in range(a, tg):
                            M = Wa[a].T @ Wa[b]
                            if a != b:
                                M = M + M.T
                            mo["ic"][a, b] = mo["ic"][b, a] = add(*triu_trip(M))
            else:
                Ws = sp.csr_matrix(W[d["rows"], :])          # by rows: indices are columns
                cols = np.unique(Ws.indices)
                d["cols"], d["Ws"] = cols, Ws[:, cols].toarray()
                ii, jj = np.triu_indices(len(cols))
                d["triu"] = (ii, jj)
                d["ic"] = add(cols[ii], cols[jj], np.zeros(len(ii)), True)
                d["ys"] = y[d["rows"]]
        self._block_of = {}                  # component -> (term, part, a, b) of a full block
        for k_, d in enumerate(self.T):
            t, q = d["t"], d["q"]
            full = d["struct"] not in ("iid", "diag")
            d["ic"] = -np.ones((t, t, len(d["parts"])), dtype=np.int64)
            for h, P in enumerate(d["parts"]):
                Pt = sp.csr_matrix(P).tocoo()
                # Full block (dense kinship, kernel): in the dense inversion it
                # is added to C in one piece (see _prepare_blocks).
                full_block = Pt.nnz >= 0.5 * q * q and q >= 16
                for a in range(t):
                    for b in range(a, t):
                        if not full and a != b:
                            continue
                        ia, ib = d["off"] + a * q, d["off"] + b * q
                        if full_block and defer:
                            empty = np.zeros(0)
                            j = add(empty, empty, empty, d["forme"] == "dense")
                            self._blocks_deferred = True
                            self._n_block_entries += (q * (q + 1)) // 2 if a == b else q * q
                        elif a == b:
                            sel = Pt.row <= Pt.col
                            j = add(ia + Pt.row[sel], ia + Pt.col[sel], Pt.data[sel], d["forme"] == "dense")
                        else:
                            j = add(ia + Pt.row, ib + Pt.col, Pt.data, d["forme"] == "dense")
                        d["ic"][a, b, h] = d["ic"][b, a, h] = j
                        if full_block:
                            self._block_of[j] = (k_, h, a, b)
        ii = np.concatenate(I_ + [np.arange(m)])
        jj = np.concatenate(J_ + [np.arange(m)])
        U = sp.csc_matrix((np.ones(len(ii)), (ii, jj)), shape=(m, m))
        U.sum_duplicates(); U.sort_indices()
        self.ci = U.indices.astype(np.int64)
        self.cj = np.repeat(np.arange(m), np.diff(U.indptr)).astype(np.int64)
        self._U_indptr = U.indptr
        key = self.cj * m + self.ci
        lens = np.array([len(x) for x in X_], dtype=np.int64)
        self.n_comp = len(X_)
        self.c_debut = np.concatenate([[0], np.cumsum(lens)])
        self.c_pos = np.searchsorted(key, np.concatenate(J_) * m + np.concatenate(I_))
        self.c_id = np.repeat(np.arange(self.n_comp), lens)
        self.c_val0 = np.concatenate(X_)
        self.c_var = np.array(var, dtype=bool)
        self.nU = len(key)
        self.poids = np.where(self.ci == self.cj, 1.0, 2.0)

    def _choose_inversion(self, inversion):
        """Inversion route. auto: dense up to m = 1000 or when the pattern of C
        fills more than 30 % of a triangle, or when the sparse factor would
        fill more than half of it; sparse otherwise."""
        m = self.m
        self.factor = None
        self._block_mode = False
        n_U = self.nU + getattr(self, "_n_block_entries", 0)       # deferred blocks included
        if inversion == "auto" and m > 1000 and n_U <= 0.3 * m * (m + 1) / 2 \
                and getattr(self, "_blocks_deferred", False):
            self._components(defer=False)
        if inversion in ("sparse", "hybrid") and getattr(self, "_blocks_deferred", False):
            self._components(defer=False)
        if inversion == "auto":
            if m <= 1000 or n_U > 0.3 * m * (m + 1) / 2:
                inversion = "dense"
            else:
                try:
                    F = SparseFactor(self._pattern(), self.ci, self.cj)
                    fill = F.off[-1] / (m * (m + 1) / 2)
                except MemoryError:
                    # SuperLU refuses a factor that fills too much: C is nearly full.
                    F, fill = None, 1.0
                inversion = "dense" if fill > 0.5 else "sparse"
                if inversion == "sparse":
                    self.factor = F
        elif inversion == "sparse":
            self.factor = SparseFactor(self._pattern(), self.ci, self.cj)
        elif inversion == "hybrid":
            # Dense separator (fixed effects and the global terms found by
            # decoupage.py) and dense blocks in batches (mme_hybride.py); out of
            # scope, back to the automatic choice.
            from .mme_hybride import HybridFactor, dense_set, OutOfScope
            from .decoupage import BlockSplit
            if self._separateur is not None:
                names = [tm["name"] for tm in self.terms]
                glob = [names.index(nm) for nm in self._separateur]
            else:
                glob = BlockSplit(self.terms, self.res, self._n_obs0).search(
                    espace="niveaux", p=self.p_sortie)["globaux"]
            try:
                self.factor = HybridFactor(self.ci, self.cj, m,
                                              dense_set(self.T, self.p_sortie, glob),
                                              verbose=self.verbose)
            except OutOfScope as e:
                if self.verbose:
                    print("  [mme] hybrid out of scope (%s) -> automatic inversion" % e, flush=True)
                return self._choose_inversion("auto")
        self.inversion = inversion
        if inversion == "dense":
            self._jci, self._jcj = jnp.asarray(self.ci), jnp.asarray(self.cj)
            # On CPU, dense algebra goes through LAPACK (SciPy), not XLA: C
            # formed by indexing, factorised by dpotrf, inverted from the factor
            # by dpotri. On GPU, XLA (cuSOLVER) stays the right choice.
            try:
                dev = next(iter(jnp.zeros(()).devices()))
                self._dense_cpu = dev.platform == "cpu"
            except Exception:
                self._dense_cpu = False
            if self._block_of:
                self._prepare_blocks()

    # --------------------------------------------------- dense inversion by blocks
    def _prepare_blocks(self):
        """Dense inversion on CPU with full blocks (dense kinship, kernel).

        Each component (a, b) of a term with a dense kinship carries q^2
        entries. The blocks are added to C in one piece
        (C[A, B] += Sigma^-1[a, b] K^-1) and their traces are read on the blocks
        of C^-1 (sum of the entrywise product), instead of going through the
        sparse pattern at each evaluation. The other components (residual,
        fixed effects) keep the sparse pattern, restricted to their entries."""
        m = self.m
        self._block_mode = True
        is_block = np.zeros(self.n_comp, dtype=bool)
        is_block[list(self._block_of)] = True
        ent = ~is_block[self.c_id]                       # entries outside the blocks
        self._sp_ent = np.nonzero(ent)[0]
        u, inv = np.unique(self.c_pos[self._sp_ent], return_inverse=True)
        self._sp_u, self._sp_inv = u, inv.astype(np.int64)
        self._sp_i, self._sp_j = self.ci[u], self.cj[u]
        self._sp_hors_diag = self._sp_i != self._sp_j
        self._sp_poids = np.where(self._sp_hors_diag, 2.0, 1.0)
        # Constant blocks (given kinship): the full matrix, once per part.
        self._full_parts = {}
        for j, (k, h, a, b) in self._block_of.items():
            d = self.T[k]
            if d["forme"] != "dense" and (k, h) not in self._full_parts:
                self._full_parts[(k, h)] = np.asarray(d["parts"][h].toarray(), dtype=np.float64)
        self._diag_idx = np.arange(m)
        self._gpu = None
        if not self._dense_cpu:
            self._prepare_blocks_gpu()

    def _prepare_blocks_gpu(self):
        """Block mode on GPU: C assembled on the device, C^-1 never leaves it.

        The K^-1 blocks stay on the device, C is assembled by a compiled
        function (static structure, values as arguments), and only the traces
        of the blocks, the entries of C^-1 outside the blocks and its diagonal
        come back. One operation per term, C[term] += kron(S, K^-1), S (t x t)
        carrying the coefficients Sigma^-1[a, b]; the traces of all the blocks
        of a term come out of one einsum. A term with a parametrised dense
        kernel (derivatives of K) keeps the host route."""
        if any(self.T[k]["forme"] == "dense" and self.T[k]["nlv"] > 0 for (k, h, a, b) in self._block_of.values()):
            return
        m = self.m
        keys = sorted({(k, h) for (k, h, a, b) in self._block_of.values()})
        key_idx = {c: i for i, c in enumerate(keys)}
        order = sorted(self._block_of)
        groups = []
        for (k, h) in keys:
            d = self.T[k]
            t, q = d["t"], d["q"]
            members = [(n_, self._block_of[j][2], self._block_of[j][3]) for n_, j in enumerate(order)
                       if self._block_of[j][:2] == (k, h)]
            groups.append((d["off"], t, q, key_idx[(k, h)],
                           np.array([x[0] for x in members], dtype=np.int32),
                           np.array([x[1] for x in members], dtype=np.int32),
                           np.array([x[2] for x in members], dtype=np.int32)))
        hd = self._sp_hors_diag
        si, sj = jnp.asarray(self._sp_i), jnp.asarray(self._sp_j)
        sih, sjh = jnp.asarray(self._sp_i[hd]), jnp.asarray(self._sp_j[hd])
        hd_j = jnp.asarray(np.nonzero(hd)[0])
        groups = tuple(groups)

        def assemble(coefs, xs, Ps, r):
            C = jnp.zeros((m, m), dtype=xs.dtype)
            C = C.at[si, sj].add(xs)
            C = C.at[sjh, sih].add(xs[hd_j])
            for (off, t, q, c_, n_i, a_i, b_i) in groups:
                S = jnp.zeros((t, t), dtype=xs.dtype).at[a_i, b_i].set(coefs[n_i]).at[b_i, a_i].set(coefs[n_i])
                C = C.at[off:off + t * q, off:off + t * q].add(jnp.kron(S, Ps[c_]))
            L = jnp.linalg.cholesky(C)
            ld = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            sol = jax.scipy.linalg.cho_solve((L, True), r)
            return L, ld, sol

        def traces(L, Ps):
            Ci = jax.scipy.linalg.cho_solve((L, True), jnp.eye(m, dtype=L.dtype))
            tr = jnp.zeros(len(order), dtype=L.dtype)
            for (off, t, q, c_, n_i, a_i, b_i) in groups:
                B = Ci[off:off + t * q, off:off + t * q].reshape(t, q, t, q)
                Tm = jnp.einsum("aibj,ij->ab", B, Ps[c_])
                tr = tr.at[n_i].set(Tm[a_i, b_i])
            return tr, Ci[sj, si], jnp.diag(Ci)

        self._gpu = dict(ordre=order, cles=keys, assembler=jax.jit(assemble), traces=jax.jit(traces),
                         Ps=None)

    def _block_parts_gpu(self, termes):
        """The K^-1 blocks on the device, kept when none depends on theta."""
        g = self._gpu
        if g["Ps"] is not None and not any(self.T[k]["forme"] == "dense" for (k, h) in g["cles"]):
            return g["Ps"]
        Ps = tuple(jnp.asarray(termes[k]["Kinv"] if self.T[k]["forme"] == "dense" else self._full_parts[(k, h)])
                   for (k, h) in g["cles"])
        g["Ps"] = Ps
        return Ps

    def _block_part(self, j, termes):
        k, h, a, b = self._block_of[j]
        if self.T[k]["forme"] == "dense":
            return termes[k]["Kinv"]
        return self._full_parts[(k, h)]

    def _assemble_blocks(self, coef, val, termes):
        """C on the host in block mode: the sparse entries, then each block."""
        m = self.m
        C = np.zeros((m, m), order="F")
        w = coef[self.c_id[self._sp_ent]] * val[self._sp_ent]
        xs = np.bincount(self._sp_inv, weights=w, minlength=len(self._sp_u))
        C[self._sp_i, self._sp_j] += xs
        h_ = self._sp_hors_diag
        C[self._sp_j[h_], self._sp_i[h_]] += xs[h_]
        for j, (k, h, a, b) in self._block_of.items():
            c = coef[j]
            if c == 0.0:
                continue
            d = self.T[k]
            q = d["q"]
            ia, ib = d["off"] + a * q, d["off"] + b * q
            P = self._block_part(j, termes)
            C[ia:ia + q, ib:ib + q] += c * P
            if a != b:
                C[ib:ib + q, ia:ia + q] += c * P.T
        return C

    def _cinv_full(self, et):
        """Full C^-1 from the factor, cached in the state."""
        if "Cinv" not in et and not self._dense_cpu:
            # Factor on the GPU: C^-1 formed there, then brought to the host.
            L = et["L"]
            et["Cinv"] = np.asarray(jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.m, dtype=L.dtype)))
        if "Cinv" not in et:
            import scipy.linalg as _sl
            Ci, info = _sl.lapack.dpotri(et["L"], lower=1)
            if info != 0:
                raise np.linalg.LinAlgError("dpotri: C singular")
            # dpotri fills the lower triangle only.
            Ci = np.tril(Ci)
            Ci += np.tril(Ci, -1).T
            et["Cinv"] = Ci
        return et["Cinv"]

    def _pattern(self, x=None):
        """Symmetric sparse C on its pattern, with values x (ones by default)."""
        m = self.m
        v = np.ones(self.nU) if x is None else x
        U = sp.csc_matrix((v, self.ci, self._U_indptr), shape=(m, m))
        return (U + sp.triu(U, 1).T).tocsc()

    def _rinv(self, B, et):
        """R^-1 B section by section."""
        B = np.asarray(B, dtype=np.float64)
        one = B.ndim == 1
        if one:
            B = B[:, None]
        out = np.zeros_like(B)
        for s, d in enumerate(self.S):
            if d.get("forme") == "grille":
                out[d["rows"]] = sum(ch * (P @ B[d["rows"]]) for ch, P in zip(et["Rinv"][s], d["parts"]))
                continue
            if d["lvl"] == "id":
                for g, mo in enumerate(d["motifs"]):
                    I = mo["I"]; tg = I.shape[1]
                    Bg = B[I.ravel()].reshape(mo["n"], tg, B.shape[1])
                    out[I.ravel()] = np.einsum("nbc,ba->nac", Bg, et["Sg_inv"][s][g]).reshape(-1, B.shape[1])
            else:
                out[d["rows"]] = et["Rinv"][s] @ B[d["rows"]]
        return out[:, 0] if one else out

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, theta, full=True):
        """-2logL at theta ("f") and the solution of C ("sol"); with full, the
        gradient ("g") and the AI matrix ("AI") in theta. f = inf at an
        inadmissible point.

        The last evaluation is cached: the same theta and request (or a full
        evaluation already made) return it with its state. After a value-only
        evaluation, a full one at the same theta reuses the factorisation and
        computes only the derivatives (the AI step trial, then its full
        evaluation)."""
        key = np.asarray(theta, dtype=np.float64).tobytes()
        last = getattr(self, "_last", None)
        if last is not None and last[0] == key and (last[1] or not full):
            self._state = last[3]
            return dict(last[2])
        if last is not None and last[0] == key and full and last[4] is not None:
            t0_ = time.time()
            self._state = last[3]
            g, AI = self._derivatives(*last[4])
            out = dict(last[2]); out.update(g=g, AI=AI)
            self._durations = getattr(self, "_durations", [])
            if len(self._durations) < 200:
                self._durations.append(time.time() - t0_ + last[5])
            self._last = (key, True, dict(out), self._state, None, 0.0)
            return out
        t0_ = time.time()
        out = self._evaluate(theta, full)
        if full:
            # Duration of the full evaluations actually computed (not the
            # cache): the one reported as eval_s.
            self._durations = getattr(self, "_durations", [])
            if len(self._durations) < 200:
                self._durations.append(time.time() - t0_)
        if np.isfinite(out.get("f", np.inf)):
            self._last = (key, full, dict(out), self._state,
                          None if full else getattr(self, "_derivative_inputs", None),
                          time.time() - t0_)
        return out

    def _evaluate(self, theta, full=True):
        """The terms (coefficients of C, log|G|), the residual sections
        (coefficients, log|R|, y'R^-1y, r), the factorisation of C, then -2logL
        and, when full, the derivatives."""
        theta = np.asarray(theta, dtype=np.float64)
        th_t, th_r, _ = split_theta(theta, self.terms, self.res)
        th_t = [np.asarray(v, dtype=np.float64) for v in th_t]
        th_r = np.asarray(th_r, dtype=np.float64)
        m, n, p = self.m, self.n, self.p
        et = dict(Sg_inv={}, Rinv={})
        coef = np.zeros(self.n_comp)
        val = self.c_val0.copy()
        r = np.zeros(m)
        try:
            ldG, termes = self._eval_terms(th_t, coef, val)
            yRy, ldR, sections = self._eval_sections(th_r, coef, val, r, et)
            ldC, sol = self._factorise(coef, val, r, termes, et)
        except _Inadmissible:
            return dict(f=np.inf)
        f = (n - p) * LOG2PI + ldR + ldG + ldC + self.ldRx + yRy - float(r @ sol)
        out = dict(f=float(f), sol=sol)
        self._state = et
        if not full:
            # What the derivatives need later at the same point (evaluate).
            self._derivative_inputs = (et, termes, sections, sol, val)
            return out
        self._derivative_inputs = None
        g, AI = self._derivatives(et, termes, sections, sol, val)
        out.update(g=g, AI=AI)
        return out

    def _eval_terms(self, th_t, coef, val):
        """Coefficients Sigma^-1[a, b] c_K of the term components of C (and the
        values of a parametrised dense K^-1), and log|G|. Returns (log|G|, one
        dict per term with Si, ths, thl, cK and Kinv when dense). A Sigma whose
        Cholesky diagonal ratio falls under DIAG_CHOL_MIN, or a dense K
        conditioned beyond 1e12, makes the point inadmissible: their inverses
        no longer have significant digits (dev/notes.md)."""
        ldG = 0.0
        termes = []
        for k, d in enumerate(self.T):
            ths, thl = th_t[k][:d["nsig"]], th_t[k][d["nsig"]:]
            Sig = np.asarray(d["sig"](jnp.asarray(ths)), dtype=np.float64)
            try:
                cS = np.linalg.cholesky(Sig)
            except np.linalg.LinAlgError:
                raise _Inadmissible()
            dS = np.diag(cS)
            if dS.min() < DIAG_CHOL_MIN * dS.max():
                raise _Inadmissible()
            Si = np.linalg.inv(Sig)
            info = dict(Si=Si, ths=ths, thl=thl)
            if d["forme"] == "fixe":
                cK, ldK = np.ones(1), d["ldK"]
            elif d["forme"] == "creux":
                cK, ldK = d["niv"].coef(thl), d["niv"].logdet(thl)
            else:
                K = np.asarray(d["K"](jnp.asarray(thl)), dtype=np.float64)
                try:
                    cKK = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    raise _Inadmissible()
                dg = np.diag(cKK)
                if dg.min() < 1e-6 * dg.max():
                    raise _Inadmissible()
                Kinv = np.linalg.inv(K)
                info["Kinv"] = Kinv
                cK, ldK = np.ones(1), 2.0 * np.sum(np.log(np.diag(cKK)))
            info["cK"] = cK
            ldG += d["q"] * 2.0 * np.sum(np.log(np.diag(cS))) + d["t"] * ldK
            for h in range(len(d["parts"])):
                for a in range(d["t"]):
                    for b in range(a, d["t"]):
                        j = d["ic"][a, b, h]
                        if j < 0:
                            continue
                        coef[j] = Si[a, b] * cK[h]
                        if d["forme"] == "dense" and not (self._block_mode and j in self._block_of):
                            val[self.c_debut[j]:self.c_debut[j + 1]] = _block_values(info["Kinv"], a == b)
            termes.append(info)
        return ldG, termes

    def _eval_sections(self, th_r, coef, val, r, et):
        """Coefficients of the residual components of C, r = W'R^-1y (added in
        place), log|R| and y'R^-1y, section by section. Fills et["Rinv"] and
        et["Sg_inv"]. Returns (y'R^-1y, log|R|, one dict per section)."""
        n = self.n
        yRy = ldR = 0.0
        sections = []
        o = 0
        for s, d in enumerate(self.S):
            th_s = th_r[o:o + d["np_"]]
            o += d["np_"]
            info = dict(th=th_s)
            if d.get("forme") == "grille":
                s2 = float(np.exp(2.0 * th_s[0]))
                thl = th_s[1:]
                c = d["niv"].coef(thl) / s2
                ldR += n * np.log(s2) + d["niv"].logdet(thl)
                for h, j in enumerate(d["ic"]):
                    coef[j] = c[h]
                    r += c[h] * d["Wy"][h]
                yRy += float(c @ d["yPy"])
                et["Rinv"][s] = c
                info.update(s2=s2, thl=thl, c=c)
                sections.append(info)
                continue
            if d["lvl"] == "id":
                Sr = np.asarray(d["sig"](jnp.asarray(th_s[:d["nsig"]])), dtype=np.float64)
                info["Sr"] = Sr
                et["Sg_inv"][s] = []
                for mo in d["motifs"]:
                    Sg = Sr[np.ix_(mo["tau"], mo["tau"])]
                    try:
                        cg = np.linalg.cholesky(Sg)
                    except np.linalg.LinAlgError:
                        raise _Inadmissible()
                    if np.diag(cg).min() < DIAG_CHOL_MIN * np.diag(cg).max():
                        raise _Inadmissible()
                    ldR += mo["n"] * 2.0 * np.sum(np.log(np.diag(cg)))
                    Sgi = np.linalg.inv(Sg)
                    et["Sg_inv"][s].append(Sgi)
                    yRy += float(np.sum(Sgi * mo["yy"]))
                    tg = len(mo["tau"])
                    for a in range(tg):
                        r += mo["Wy"][a] @ Sgi[:, a]
                        for b in range(a, tg):
                            coef[mo["ic"][a, b]] = Sgi[a, b]
            else:
                R = np.asarray(d["R"](jnp.asarray(th_s)), dtype=np.float64)
                try:
                    cR = np.linalg.cholesky(R)
                except np.linalg.LinAlgError:
                    raise _Inadmissible()
                Ri = np.linalg.inv(R)
                ldR += 2.0 * np.sum(np.log(np.diag(cR)))
                A = Ri @ d["Ws"]
                j = d["ic"]
                coef[j] = 1.0
                val[self.c_debut[j]:self.c_debut[j + 1]] = (d["Ws"].T @ A)[d["triu"]]
                Riy = Ri @ d["ys"]
                yRy += float(d["ys"] @ Riy)
                r[d["cols"]] += d["Ws"].T @ Riy
                et["Rinv"][s] = Ri
                info.update(Ri=Ri, A=A)
            sections.append(info)
        return yRy, ldR, sections

    def _factorise(self, coef, val, r, termes, et):
        """Assembles and factorises C by the chosen route, solves C sol = r.
        Returns (log|C|, sol); the factor goes to et["L"] for the dense routes."""
        m = self.m
        x = None if self._block_mode else np.bincount(self.c_pos, weights=coef[self.c_id] * val,
                                                      minlength=self.nU)
        try:
            if self.inversion == "dense" and self._dense_cpu:
                import scipy.linalg as _sl
                if self._block_mode:
                    C = self._assemble_blocks(coef, val, termes)
                else:
                    C = np.zeros((m, m))
                    C[self.ci, self.cj] = x
                    C[self.cj, self.ci] = x
                L, info = _sl.lapack.dpotrf(C, lower=1, clean=1, overwrite_a=1)
                if info != 0:
                    raise _Inadmissible()
                ldC = float(2.0 * np.sum(np.log(np.diag(L))))
                if not np.isfinite(ldC):
                    raise _Inadmissible()
                sol = _sl.cho_solve((L, True), r, check_finite=False)
                et["L"] = L
            elif self.inversion == "dense":
                if self._block_mode and self._gpu is not None:
                    g_ = self._gpu
                    w_ = coef[self.c_id[self._sp_ent]] * val[self._sp_ent]
                    xs = np.bincount(self._sp_inv, weights=w_, minlength=len(self._sp_u))
                    L, ld_, sol = g_["assembler"](jnp.asarray(coef[g_["ordre"]]), jnp.asarray(xs),
                                                  self._block_parts_gpu(termes), jnp.asarray(r))
                    ldC = float(ld_)
                    if not np.isfinite(ldC):
                        raise _Inadmissible()
                    sol = np.asarray(sol)
                    et["L"] = L
                    et["termes_g"] = termes
                    return ldC, sol
                if self._block_mode:
                    # GPU without the compiled route: C assembled by blocks on the host.
                    C = jnp.asarray(self._assemble_blocks(coef, val, termes))
                else:
                    xj = jnp.asarray(x)
                    C = jnp.zeros((m, m)).at[self._jci, self._jcj].set(xj).at[self._jcj, self._jci].set(xj)
                L = jnp.linalg.cholesky(C)
                del C
                ldC = float(2.0 * jnp.sum(jnp.log(jnp.diag(L))))
                if not np.isfinite(ldC):
                    raise _Inadmissible()
                sol = np.asarray(jax.scipy.linalg.cho_solve((L, True), jnp.asarray(r)))
                et["L"] = L
            elif self.inversion == "hybrid":
                ldC = self.factor.factorise_values(x)
                sol = self.factor.solve(r)
            elif getattr(self.factor, "backend", "") == "cholmod":
                # CHOLMOD: the values go directly in the permuted order.
                ldC = self.factor.factorise_values(x)
                sol = self.factor.solve(r)
            else:
                ldC = self.factor.factorise(self._pattern(x))
                sol = self.factor.solve(r)
        except (np.linalg.LinAlgError, RuntimeError):
            raise _Inadmissible()
        return ldC, sol

    # ------------------------------------------------------------------ derivatives
    def _solve(self, et, B):
        """C^-1 B from the current factorisation."""
        if self.inversion == "dense" and self._dense_cpu:
            import scipy.linalg as _sl
            return _sl.cho_solve((et["L"], True), np.asarray(B, dtype=np.float64), check_finite=False)
        if self.inversion == "dense":
            return np.asarray(jax.scipy.linalg.cho_solve((et["L"], True), jnp.asarray(B)))
        return self.factor.solve(B)

    def _cinv_pattern(self, et):
        """Entries of C^-1 on the pattern of C (ci, cj)."""
        if self.inversion == "dense" and self._dense_cpu:
            import scipy.linalg as _sl
            Ci, info = _sl.lapack.dpotri(et["L"], lower=1)
            if info != 0:
                raise np.linalg.LinAlgError("dpotri: C singular")
            # dpotri fills only the lower triangle: (ci, cj) is the upper
            # triangle of C (ci <= cj), read at (cj, ci).
            return Ci[self.cj, self.ci]
        if self.inversion == "dense":
            Ci = jax.scipy.linalg.cho_solve((et["L"], True), jnp.eye(self.m))
            return np.asarray(Ci[self._jci, self._jcj])
        return self.factor.selected_inverse()

    def _derivatives(self, et, termes, sections, sol, val):
        """Gradient and AI matrix in theta: the scores and working columns of
        each term and section in internal coordinates, then the Jacobian to
        theta (see the module docstring)."""
        trace, cv_at = self._trace_functions(et, termes, val)
        e = self.y - self.W @ sol
        Py = self._rinv(e, et)
        blocs_J, s_int, Q = [], [], []
        for k, d in enumerate(self.T):
            self._term_scores(d, termes[k], sol, trace, blocs_J, s_int, Q)
        for s, d in enumerate(self.S):
            self._section_scores(s, d, sections[s], et, e, Py, trace, cv_at, blocs_J, s_int, Q)
        return self._gradient_and_ai(blocs_J, s_int, Q, et)

    def _trace_functions(self, et, termes, val):
        """trace(j, a, b, vals=None, full=None): <C^-1, component j> (with vals
        or a full block, against those values instead of the component's own),
        divided by 2 off the diagonal; and cv_at(pos), the entries of C^-1 at
        positions of the pattern. Three routes: block mode on GPU (traces and
        entries computed on the device), block mode on CPU (full C^-1), and the
        pattern of C otherwise."""
        if self._block_mode and self._gpu is not None:
            tr_g, cv_u_g, dg_g = self._gpu["traces"](et["L"], self._block_parts_gpu(termes))
            tr_g, cv_u, et["Cinv_diag"] = np.asarray(tr_g), np.asarray(cv_u_g), np.asarray(dg_g)
            e_ = self._sp_ent
            tr_comp = np.bincount(self.c_id[e_], weights=(cv_u * self._sp_poids)[self._sp_inv] * val[e_],
                                  minlength=self.n_comp)
            for n_, j in enumerate(self._gpu["ordre"]):
                k_, h_, a_, b_ = self._block_of[j]
                tr_comp[j] = float(tr_g[n_]) * (2.0 if a_ != b_ else 1.0)

            def cv_at(pos):
                return cv_u[np.searchsorted(self._sp_u, pos)]

            def trace(j, a, b, vals=None, plein=None):
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cv_at(pos) * np.where(self.ci[pos] == self.cj[pos], 1.0, 2.0)
                                    * vals)) / (1.0 if a == b else 2.0)
        elif self._block_mode:
            Ci = self._cinv_full(et)
            cv_u = Ci[self._sp_j, self._sp_i]                    # at the entries outside blocks
            e_ = self._sp_ent
            tr_comp = np.bincount(self.c_id[e_], weights=(cv_u * self._sp_poids)[self._sp_inv] * val[e_],
                                  minlength=self.n_comp)

            def block_ci(j):
                k, h, a, b = self._block_of[j]
                d = self.T[k]; q = d["q"]
                ia, ib = d["off"] + a * q, d["off"] + b * q
                return Ci[ia:ia + q, ib:ib + q]

            for j in self._block_of:
                tr_comp[j] = float(np.sum(block_ci(j) * self._block_part(j, termes)))
                k_, h_, a_, b_ = self._block_of[j]
                if a_ != b_:
                    tr_comp[j] *= 2.0                            # convention: divided below

            def cv_at(pos):
                return Ci[self.cj[pos], self.ci[pos]]

            def trace(j, a, b, vals=None, plein=None):
                if j in self._block_of:
                    if vals is None:
                        return tr_comp[j] / (1.0 if a == b else 2.0)
                    return float(np.sum(block_ci(j) * plein))
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cv_at(pos) * np.where(self.ci[pos] == self.cj[pos], 1.0, 2.0)
                                    * vals)) / (1.0 if a == b else 2.0)
        else:
            cv = self._cinv_pattern(et)
            cvw = cv * self.poids
            tr_comp = np.bincount(self.c_id, weights=cvw[self.c_pos] * val, minlength=self.n_comp)

            def cv_at(pos):
                return cv[pos]

            def trace(j, a, b, vals=None, plein=None):
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cvw[pos] * vals)) / (1.0 if a == b else 2.0)
        return trace, cv_at

    def _term_scores(self, d, info, sol, trace, blocs_J, s_int, Q):
        """Scores and working columns of one term (entries of Sigma, then level
        parameters), and its Jacobian block from internal coordinates to
        theta."""
        Si, q, t = info["Si"], d["q"], d["t"]
        U = sol[d["off"]:d["off"] + t * q].reshape(q, t, order="F")
        nparts = len(d["parts"])
        Tp = np.zeros((t, t, nparts))
        for h in range(nparts):
            for a in range(t):
                for b in range(a, t):
                    j = d["ic"][a, b, h]
                    if j >= 0:
                        Tp[a, b, h] = Tp[b, a, h] = trace(j, a, b)
        if d["forme"] == "dense":
            KiU = info["Kinv"] @ U
            Tm = Tp[:, :, 0]
        else:
            PU = [P @ U for P in d["parts"]]
            KiU = sum(info["cK"][h] * PU[h] for h in range(nparts))
            Tm = np.tensordot(Tp, info["cK"], axes=([2], [0]))
        M = q * Si - Si @ (Tm + U.T @ KiU) @ Si
        for (a, b) in d["paires"]:
            E = np.zeros((t, t)); E[a, b] = 1.0; E[b, a] = 1.0
            s_int.append(M[a, a] if a == b else 2.0 * M[a, b])
            Q.append(d["Z"] @ (U @ Si @ E).ravel(order="F"))
        if d["nlv"]:
            if d["forme"] == "creux":
                niv = d["niv"]
                for i in range(d["nlv"]):
                    dc = niv.coef(info["thl"], i_der=i)
                    Td = np.tensordot(Tp, dc, axes=([2], [0]))
                    dKiU = sum(dc[h] * PU[h] for h in range(nparts))
                    s_int.append(t * niv.dlogdet(info["thl"], i) + float(np.sum(Si * Td))
                                 + float(np.sum(Si * (U.T @ dKiU))))
                    Q.append(d["Z"] @ niv.apply_dK(info["thl"], i, KiU).ravel(order="F"))
            else:
                dK = np.asarray(d["dK"](jnp.asarray(info["thl"])), dtype=np.float64)
                Ki = info["Kinv"]
                for i in range(d["nlv"]):
                    dKi = dK[:, :, i]
                    dKinv = -Ki @ dKi @ Ki
                    Td = np.zeros((t, t))
                    for a in range(t):
                        for b in range(a, t):
                            j = d["ic"][a, b, 0]
                            if j >= 0:
                                Td[a, b] = Td[b, a] = (
                                    trace(j, a, b, True, plein=dKinv) if (self._block_mode and j in self._block_of)
                                    else trace(j, a, b, _block_values(dKinv, a == b)))
                    s_int.append(t * float(np.sum(Ki * dKi)) + float(np.sum(Si * Td))
                                 + float(np.sum(Si * (U.T @ (dKinv @ U)))))
                    Q.append(d["Z"] @ (dKi @ KiU).ravel(order="F"))
        Js = np.asarray(d["sig_jac"](jnp.asarray(info["ths"])), dtype=np.float64)
        Jk = np.zeros((len(d["paires"]) + d["nlv"], d["nsig"] + d["nlv"]))
        Jk[:len(d["paires"]), :d["nsig"]] = Js
        Jk[len(d["paires"]):, d["nsig"]:] = np.eye(d["nlv"])
        blocs_J.append(Jk)

    def _section_scores(self, s, d, info, et, e, Py, trace, cv_at, blocs_J, s_int, Q):
        """Scores and working columns of one residual section, and its Jacobian
        block."""
        n = self.n
        if d.get("forme") == "grille":
            # d(-2logL)/dtheta = dlog|R| + sum_h dc_h [tr(C^-1 W'P_hW) + e'P_h e],
            # R^-1 = sum_h c_h P_h; working vectors dR/dtheta P y, P y = R^-1 e.
            c, thl, s2, niv = info["c"], info["thl"], info["s2"], d["niv"]
            trh = np.array([trace(j, 0, 0) for j in d["ic"]])
            ePe = np.array([float(e @ (P @ e)) for P in d["parts"]])
            s_int.append(2.0 * n - 2.0 * float(c @ (trh + ePe)))   # theta = log sigma
            Q.append(2.0 * e)
            for i in range(niv.np):
                dc = niv.coef(thl, i_der=i) / s2
                s_int.append(niv.dlogdet(thl, i) + float(dc @ (trh + ePe)))
                Q.append(s2 * niv.apply_dK(thl, i, Py[:, None])[:, 0])
            blocs_J.append(np.eye(d["np_"]))
            return
        if d["lvl"] == "id":
            t = d["t"]
            H = np.zeros((t, t))
            for g, mo in enumerate(d["motifs"]):
                tg = len(mo["tau"])
                Sgi = et["Sg_inv"][s][g]
                Eg = e[mo["I"].ravel()].reshape(mo["n"], tg)
                D = np.zeros((tg, tg))
                for a in range(tg):
                    for b in range(a, tg):
                        D[a, b] = D[b, a] = trace(mo["ic"][a, b], a, b)
                Sg = info["Sr"][np.ix_(mo["tau"], mo["tau"])]
                H[np.ix_(mo["tau"], mo["tau"])] += Sgi @ (mo["n"] * Sg - Eg.T @ Eg - D) @ Sgi
            for (a, b) in d["paires"]:
                s_int.append(H[a, a] if a == b else 2.0 * H[a, b])
                qv = np.zeros(n)
                for mo in d["motifs"]:
                    tg = len(mo["tau"])
                    Eab = np.zeros((tg, tg))
                    ia = np.where(mo["tau"] == a)[0]; ib = np.where(mo["tau"] == b)[0]
                    if len(ia) and len(ib):
                        Eab[ia[0], ib[0]] = 1.0; Eab[ib[0], ia[0]] = 1.0
                    Pg = Py[mo["I"].ravel()].reshape(mo["n"], tg)
                    qv[mo["I"].ravel()] = (Pg @ Eab).ravel()
                Q.append(qv)
            Js = np.asarray(d["sig_jac"](jnp.asarray(info["th"][:d["nsig"]])), dtype=np.float64)
            Jk = np.zeros((len(d["paires"]), d["np_"]))
            Jk[:, :d["nsig"]] = Js
            blocs_J.append(Jk)
            return
        dR = np.asarray(d["dR"](jnp.asarray(info["th"])), dtype=np.float64)
        Ri, A = info["Ri"], info["A"]
        ii, jj = d["triu"]
        nc = len(d["cols"])
        Cb = np.zeros((nc, nc))
        vb = cv_at(self.c_pos[self.c_debut[d["ic"]]:self.c_debut[d["ic"] + 1]])
        Cb[ii, jj] = vb; Cb[jj, ii] = vb
        Pys = Py[d["rows"]]
        for i in range(d["np_"]):
            dRi = dR[:, :, i]
            s_int.append(float(np.sum(Ri * dRi)) - float(np.sum(Cb * (A.T @ dRi @ A)))
                         - float(Pys @ dRi @ Pys))
            qv = np.zeros(n); qv[d["rows"]] = dRi @ Pys
            Q.append(qv)
        blocs_J.append(np.eye(d["np_"]))

    def _gradient_and_ai(self, blocs_J, s_int, Q, et):
        """g = J' s and AI = Q_theta' R^-1 Q_theta - (W'R^-1Q_theta)' C^-1
        (W'R^-1Q_theta), Q_theta = Q J, J block-diagonal."""
        nI = sum(b.shape[0] for b in blocs_J)
        nT = sum(b.shape[1] for b in blocs_J)
        J = np.zeros((nI, nT))
        oi = oj = 0
        for b in blocs_J:
            J[oi:oi + b.shape[0], oj:oj + b.shape[1]] = b
            oi += b.shape[0]; oj += b.shape[1]
        g = J.T @ np.asarray(s_int)
        Qm = np.column_stack(Q) @ J
        RQ = self._rinv(Qm, et)
        WRQ = np.asarray(self.W.T @ RQ)
        AI = Qm.T @ RQ - WRQ.T @ self._solve(et, WRQ)
        return g, 0.5 * (AI + AI.T)

    # ---------------------------------------------------------------- final outputs
    def outputs(self, theta, pev=()):
        """beta (scale of X), vbeta, BLUPs per term (q x t) and, for the terms
        named in `pev`, the diagonal prediction error variance.

        C = W'R^-1W + diag(0, G^-1) is formed on the true R and G, so the u
        block of C^-1 is Var(u - u_hat) (Henderson 1975), as in asreml. Its
        diagonal is read in the selected inversion (the pattern of C always
        contains the diagonal) or in the dense inverse. Replacing X by Q does
        not change that block."""
        ev = self.evaluate(theta, full=False)
        sol, p = ev["sol"], self.p_sortie
        E = np.zeros((self.m, p))
        E[np.arange(p), np.arange(p)] = 1.0
        Cqq = self._solve(self._state, E)[:p, :]
        Ri = np.linalg.inv(self.Rx)
        vb = Ri @ Cqq @ Ri.T
        out = dict(beta=Ri @ sol[:p], vbeta=0.5 * (vb + vb.T),
                   blups={d["name"]: sol[d["off"]:d["off"] + d["t"] * d["q"]].reshape(d["q"], d["t"], order="F")
                          for d in self.T})
        names = [d["name"] for d in self.T] if pev is True else list(pev or [])
        if names:
            unknown = set(names) - {d["name"] for d in self.T}
            if unknown:
                raise ValueError("pev: unknown term(s) %s" % sorted(unknown))
            out["pev"] = self._pev(names)
        return out

    def _pev(self, names):
        """Diagonal of the u block of C^-1 for the named terms, as q x t."""
        if not hasattr(self, "_pos_diag"):
            diag = np.where(self.ci == self.cj)[0]
            self._pos_diag = np.full(self.m, -1, dtype=np.int64)
            self._pos_diag[self.ci[diag]] = diag
        pev = {}
        if self._block_mode:
            if self._gpu is not None:
                if "Cinv_diag" not in self._state:
                    self._state["Cinv_diag"] = np.asarray(
                        self._gpu["traces"](self._state["L"], self._block_parts_gpu(self._state["termes_g"]))[2])
                dg = self._state["Cinv_diag"]
            else:
                dg = np.diag(self._cinv_full(self._state))
            for d in self.T:
                if d["name"] in names:
                    pev[d["name"]] = np.maximum(dg[d["off"]:d["off"] + d["t"] * d["q"]], 0.0).reshape(
                        d["q"], d["t"], order="F")
            return pev
        cv = self._cinv_pattern(self._state)
        for d in self.T:
            if d["name"] in names:
                pos = self._pos_diag[d["off"]:d["off"] + d["t"] * d["q"]]
                pev[d["name"]] = np.maximum(np.asarray(cv)[pos], 0.0).reshape(
                    d["q"], d["t"], order="F")
        return pev


def make_fga_mme(terms, res, y, X, inversion="auto", verbose=True):
    """(fga, fun_jac, engine): fga(theta) -> (f, g, AI); fun_jac(theta) -> (f, g)."""
    M = MME(terms, res, y, X, inversion=inversion, verbose=verbose)

    def fga(theta):
        r = M.evaluate(np.asarray(theta), full=True)
        if not np.isfinite(r["f"]):
            k = len(theta)
            return np.inf, np.full(k, np.nan), np.full((k, k), np.nan)
        return r["f"], r["g"], r["AI"]

    def fun_jac(theta):
        f, g, _ = fga(theta)
        return float(f), np.asarray(g, dtype=np.float64)
    return fga, fun_jac, M
