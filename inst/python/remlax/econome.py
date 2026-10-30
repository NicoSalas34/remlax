"""Dense engine in low-memory mode.

This module computes the same likelihood, gradient and AI matrix as the
standard dense engine (model.make_objective, ai.make_fga,
effects.dense_effects). Only the way they are computed changes. The aim is to
reduce the number of n x n matrices and factors alive on the GPU at the same
time. The standard mode stays the default and is left untouched.
fit_reml(..., memory="low") selects this one.

The standard mode keeps several large objects that this mode avoids.

- Full-height incidences. The standard mode creates the incidences Zs
  (dense_Z, n x sum of t q) on the GPU in make_neg2 and keeps them until the
  BLUPs at the end of the fit. They serve no purpose during the optimisation.
  Here they are never formed on the GPU. The BLUPs build the incidence of one
  term at a time, on the host (z_on_host).
- Dense residual without structure between units (lvl "id"). _section_V
  builds the unit-equality matrix (n_s x n_s), Sigma[tr, tr] and their
  product, and its backward pass gathers dV[idx, idx]. A residual us(trait)
  by plant has only one entry per pair of traits of the same plant. Here R is
  scattered entry by entry, and the backward pass reads only those entries.
- Factors B_k kept between the forward and backward passes. Here each group
  contribution sits under jax.checkpoint. B is recomputed in the backward
  pass, at a cost of n x w operations, negligible next to the Cholesky.
- Gathering dV[idx, idx] for a group whose support covers most rows costs an
  n x n copy. Here such a factor is extended to full height (n x w, mode
  "haut"), and the backward pass forms dV B without a copy.
- In the AI evaluator (ai.make_fga), V^-1 is formed by cho_solve against the
  identity while L stays alive for the solves of P Q, and the adjoint is then
  formed next to it. Here P Q is computed first by triangular solves (n x p).
  The adjoint V^-1 - ... is formed afterwards, once L is no longer needed.

The algebra is unchanged, and V is the same matrix. The order of the
floating-point sums differs, so equality with the standard mode is tested to
a relative tolerance.

Organisation. LowMemoryPlan holds everything that does not depend on theta and
assembles V. reml_from_V_cho and reml_from_V_inv map V to -2 logL with a
custom backward pass. make_objective_low_memory and make_fga_low_memory build the
objective and the AI evaluator. V_on_host and z_on_host serve the end-of-fit
outputs on the host. Memory measurements on large models are in
dev/notes.md.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .layout import split_theta, res_sections, sec_n_params
from .incidence import support_groups, restrict_Z, prefold_fixed_K, orthonormal_X
from .residual import _section_V, _section_S, section_matvec
from .model import term_factor
from .core import _reml_fwd
from .ai import _sigma_and_K

# A group whose support covers at least this fraction of the rows is extended
# to full height (mode "haut", see the module docstring). Below it, scattering
# a block of size |idx|^2 costs less than n x w extra operations.
FRACTION_HAUTEUR = 0.5
# Above this size, the indicator matrix of the sparse residual (pairs x t^2)
# is not formed, and the values are obtained by a gather.
OCTETS_MAX_INDICATRICES = 256 * 2 ** 20


def _unit_pairs(sec):
    """Pairs (i, j) of rows of the same unit, as indices local to the section.

    All pairs are returned, diagonal included and in both directions. R is
    scattered as is, without symmetrisation.
    """
    units = np.asarray(sec["unit"], dtype=np.int64)
    order = np.argsort(units, kind="stable")
    sorted_units = units[order]
    starts = np.flatnonzero(np.r_[True, sorted_units[1:] != sorted_units[:-1]])
    sizes = np.diff(np.r_[starts, len(sorted_units)])
    I_, J_ = [], []
    for m in np.unique(sizes):
        d = starts[sizes == m]
        groups = order[d[:, None] + np.arange(m)[None, :]]          # (g, m)
        I_.append(np.repeat(groups, m, axis=1).ravel())
        J_.append(np.tile(groups, (1, m)).ravel())
    return np.concatenate(I_), np.concatenate(J_)


def z_on_host(term, n):
    """Full-height incidence of one term, in NumPy on the host (see incidence.dense_Z)."""
    Z = np.zeros((n, term["t"] * term["q"]))
    np.add.at(Z, (np.asarray(term["zi"], dtype=np.int64),
                  np.asarray(term["zj"], dtype=np.int64)), np.asarray(term["zx"]))
    return Z


class LowMemoryPlan:
    """Everything that does not depend on theta, computed once.

    terms: terms after folding the fixed kinship into the incidence
    (incidence.prefold_fixed_K). Zr: incidences restricted to their support,
    on the device. plan_secs and plan_grp: how each residual section and each
    group of terms enters V. Their "mode" values are read by the tests.
    """

    def __init__(self, terms, res, y, X):
        n = int(y.shape[0])
        self.n, self.res, self.terms0 = n, res, terms
        Xq, self.ldRx = orthonormal_X(X)
        self.yj, self.Xj = jnp.asarray(y), jnp.asarray(Xq)
        self.groups = support_groups(terms, n)
        Zr = [None] * len(terms)
        rows_of = [None] * len(terms)
        for rows, ks in self.groups:
            for k in ks:
                Zr[k] = restrict_Z(terms[k], rows, n)
                rows_of[k] = np.asarray(rows, dtype=np.int64)
        self.terms, self.Zr = prefold_fixed_K(terms, Zr)
        self.rows_of = rows_of
        all_rows = np.arange(n, dtype=np.int64)

        def is_full(r):
            return r.shape[0] == n and np.array_equal(r, all_rows)

        # Residual sections
        self.secs = res_sections(res, n)
        self.plan_secs = []
        for sec in self.secs:
            rows = np.asarray(sec["rows"], dtype=np.int64)
            if sec.get("lvl", "id") == "id":
                i, j = _unit_pairs(sec)
                tr = np.asarray(sec["trait"], dtype=np.int64)
                t = int(sec["t"])
                d = dict(mode="paires", I=jnp.asarray(rows[i]), J=jnp.asarray(rows[j]))
                code = tr[i] * t + tr[j]
                if len(code) * t * t * 8 <= OCTETS_MAX_INDICATRICES:
                    # values = E vec(Sigma). The backward pass is then the
                    # product E' dvals, which is deterministic. The gather
                    # Sigma[a, b] would have a backward pass by atomic additions.
                    E = np.zeros((len(code), t * t))
                    E[np.arange(len(code)), code] = 1.0
                    d["E"] = jnp.asarray(E)
                else:
                    d["code"] = jnp.asarray(code)
                self.plan_secs.append(d)
            elif is_full(rows):
                self.plan_secs.append(dict(mode="plein"))
            else:
                self.plan_secs.append(dict(mode="bloc", idx=jnp.asarray(rows)))

        # Groups of terms
        self.plan_grp = []
        for rows, ks in self.groups:
            rows = np.asarray(rows, dtype=np.int64)
            if is_full(rows):
                mode = "plein"
            elif rows.shape[0] >= FRACTION_HAUTEUR * n:
                mode = "haut"
            else:
                mode = "bloc"
            self.plan_grp.append(dict(mode=mode, ks=list(ks), idx=jnp.asarray(rows),
                                      contrib=self._contribution(ks, rows, mode)))

    def _contribution(self, ks, rows, mode):
        """B B' of one group, under jax.checkpoint (B is recomputed in the backward pass)."""
        terms, n = self.terms, self.n
        idx = jnp.asarray(rows)

        def f(ths, Zs_):
            B = jnp.concatenate([term_factor(th, terms[k], Z)
                                 for th, k, Z in zip(ths, ks, Zs_)], axis=1)
            if mode == "haut":
                B = jnp.zeros((n, B.shape[1]), B.dtype).at[idx].set(B)
            return B @ B.T
        return jax.checkpoint(f)

    def V(self, theta, Zr):
        """V = R + the sum of the group contributions, assembled in place."""
        n = self.n
        th_terms, th_res, _ = split_theta(theta, self.terms, self.res)
        V = jnp.zeros((n, n))
        o = 0
        for sec, d in zip(self.secs, self.plan_secs):
            pnb = sec_n_params(sec)
            th_s = th_res[o:o + pnb]
            o += pnb
            if d["mode"] == "paires":
                S = _section_S(th_s, sec)
                vals = d["E"] @ S.reshape(-1) if "E" in d else S.reshape(-1)[d["code"]]
                V = V.at[d["I"], d["J"]].add(vals)
            elif d["mode"] == "plein":
                V = V + _section_V(th_s, sec)
            else:
                idx = d["idx"]
                V = V.at[idx[:, None], idx[None, :]].add(_section_V(th_s, sec))
        for d in self.plan_grp:
            ks = d["ks"]
            M = d["contrib"]([th_terms[k] for k in ks], [Zr[k] for k in ks])
            if d["mode"] == "bloc":
                idx = d["idx"]
                V = V.at[idx[:, None], idx[None, :]].add(M)
            else:
                V = V + M
        return V

    def summary(self):
        """Short text with the modes chosen, for the log."""
        from collections import Counter
        s = Counter(d["mode"] for d in self.plan_secs)
        g = Counter(d["mode"] for d in self.plan_grp)
        return ("residual %s | groups %s | adjoint %s"
                % (", ".join("%s %d" % kv for kv in sorted(s.items())) or "none",
                   ", ".join("%s %d" % kv for kv in sorted(g.items())) or "none",
                   chosen_adjoint()))


# ==============================================================================
# Kernel V -> -2logL, with a backward pass by cho_solve (see core.reml_from_V)
# ==============================================================================
# The value and the gradient are those of core.reml_from_V: the forward pass is
# its own (_reml_fwd), and only the backward pass changes. The shared kernel
# forms V^-1 = L^-T L^-1 by a solve against the identity (n^3 operations),
# then a general product (2 n^3). cho_solve does two triangular solves (2 n^3
# in total) in the buffer of the identity. The number of live n x n matrices
# is the same (L and that buffer), with a third fewer operations.
#
# On GPU, however, cho_solve costs one more n x n matrix, because XLA does not
# run the two solves in place. On CPU, cho_solve is faster and RAM does not
# limit it. Hence the adjoint mode: "inv" (minimal memory) on GPU, "cho"
# (fewer operations) on CPU. The environment variable REMLAX_ADJOINT forces
# either one. Timings and memory: dev/notes.md.
def chosen_adjoint():
    import os
    v = os.environ.get("REMLAX_ADJOINT", "").strip().lower()
    if v in ("inv", "cho"):
        return v
    return "cho" if jax.default_backend() == "cpu" else "inv"


def _V_inverse(L, mode):
    """V^-1 from the Cholesky factor, following the chosen trade-off."""
    n = L.shape[0]
    if mode == "cho":
        return jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(n), lower=True)
    return Linv.T @ Linv


@jax.custom_vjp
def reml_from_V_cho(V, y, X):
    return _reml_fwd(V, y, X)[0]


def _fwd_cho(V, y, X):
    val, res = _reml_fwd(V, y, X)
    return val, res


def _bwd_shared(res, g, mode):
    L, ViX, La, r = res
    Z = jax.scipy.linalg.solve_triangular(La, ViX.T, lower=True)   # A^-1/2 ViX'
    W = _V_inverse(L, mode)                                          # V^-1
    W = W - Z.T @ Z                                                  # P
    W = W - jnp.outer(r, r)                                          # P - (Py)(Py)'
    return (g * W, 2.0 * g * r, None)


reml_from_V_cho.defvjp(_fwd_cho, lambda res, g: _bwd_shared(res, g, "cho"))


@jax.custom_vjp
def reml_from_V_inv(V, y, X):
    return _reml_fwd(V, y, X)[0]


reml_from_V_inv.defvjp(_fwd_cho, lambda res, g: _bwd_shared(res, g, "inv"))


# ==============================================================================
# Objective (value and gradient), same contract as model.make_objective
# ==============================================================================
def make_neg2_low_memory(plan, adjoint=None):
    """-2logL (constant included) as a pure JAX function of (theta, Zr)."""
    kernel = reml_from_V_cho if (adjoint or chosen_adjoint()) == "cho" else reml_from_V_inv

    def f(theta, Zr_a):
        V = plan.V(jnp.asarray(theta), Zr_a)
        return kernel(V, plan.yj, plan.Xj) + plan.ldRx
    return f


def make_objective_low_memory(plan, scale=None, adjoint=None):
    """Return (fun_jac_scaled, fun_jac, sc), scaled by n as in model.make_objective."""
    sc = float(plan.n) if scale is None else float(scale)
    f_raw = make_neg2_low_memory(plan, adjoint=adjoint)
    fg = jax.jit(jax.value_and_grad(lambda th, Z: f_raw(th, Z) / sc))
    Zr = plan.Zr

    def fun_jac_scaled(theta):
        v, g = fg(jnp.asarray(theta), Zr)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc
    return fun_jac_scaled, fun_jac, sc


# ==============================================================================
# AI: value, gradient and AI matrix from one factorisation (see ai.make_fga)
# ==============================================================================
def make_fga_low_memory(plan, adjoint=None):
    """fga(theta) -> (-2logL, gradient, AI), as ai.make_fga.

    The columns of Q follow the same formulas as ai.make_ai (term_columns,
    section_columns). They are copied here so that they share the incidences
    of the plan instead of building a second copy on the device.
    """
    n, terms, secs, res = plan.n, plan.terms, plan.secs, plan.res
    yj, Xj = plan.yj, plan.Xj
    p_x = Xj.shape[1]
    adjoint_mode = adjoint or chosen_adjoint()
    rows_of = [jnp.asarray(r) for r in plan.rows_of]

    def term_columns(th_k, term, Z, rows, v):
        t, q = term["t"], term["q"]
        U = (Z.T @ v[rows]).reshape(t, q)

        def g(th):
            S, K = _sigma_and_K(th, term)
            M = U if K is None else U @ K
            return (S @ M).reshape(t * q)
        J = jax.jacfwd(g)(th_k)
        return jnp.zeros((n, J.shape[1])).at[rows].set(Z @ J)

    def section_columns(th_s, sec, v):
        rows = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        J = jax.jacfwd(lambda th: section_matvec(th, sec, v[rows]))(th_s)
        return jnp.zeros((n, J.shape[1])).at[rows].set(J)

    def fga(theta, Zr):
        V, vjp_V = jax.vjp(lambda th: plan.V(th, Zr), theta)
        L = jnp.linalg.cholesky(V)
        del V
        logdetV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        def solve(b):
            return jax.scipy.linalg.cho_solve((L, True), b)
        ViX = solve(Xj)
        A = Xj.T @ ViX
        La = jnp.linalg.cholesky(A)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(La)))
        ViXA = ViX @ jax.scipy.linalg.cho_solve((La, True), jnp.eye(p_x))
        Py = solve(yj) - ViXA @ (ViX.T @ yj)
        f = (n - p_x) * np.log(2.0 * np.pi) + logdetV + logdetA + plan.ldRx + yj @ Py
        th_terms, th_res, _ = split_theta(theta, terms, res)
        columns = [term_columns(th_terms[k], tm, Zr[k], rows_of[k], Py)
                   for k, tm in enumerate(terms)]
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            columns.append(section_columns(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(columns, axis=1)
        PQ = solve(Q) - ViXA @ (ViX.T @ Q)
        AI = Q.T @ PQ
        # Adjoint of V, formed last: L is needed only for it from here on. The
        # memory/operations trade-off follows chosen_adjoint().
        W = _V_inverse(L, adjoint_mode)
        W = W - jnp.concatenate([ViXA, Py[:, None]], axis=1) \
            @ jnp.concatenate([ViX, Py[:, None]], axis=1).T
        g = vjp_V(W)[0]
        return f, g, 0.5 * (AI + AI.T)

    fga_jit = jax.jit(fga)
    Zr = plan.Zr

    def fga_call(theta):
        return fga_jit(jnp.asarray(theta), Zr)
    return fga_call


# ==============================================================================
# End-of-fit outputs (beta, BLUP, PEV), see effects.dense_effects
# ==============================================================================
def V_on_host(plan, theta):
    """V at theta, brought back to the host (NumPy). The GPU memory is freed afterwards."""
    Vf = jax.jit(plan.V)
    V = np.asarray(Vf(jnp.asarray(theta), plan.Zr))
    return V
