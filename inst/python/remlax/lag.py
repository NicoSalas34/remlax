"""Structural spatial lag model: y = M y + X b + Z u + e.

Model
    The response y is in long format (n rows). Each row carries a trait t and
    a unit i.

        y = M y + X b + sum_k Z_k u_k + e,     M[(t,i),(u,j)] = Psi[t,u] * W_t[i,j]

    Psi (q x q) holds the effect of the emitting trait u (column) of a
    neighbour j on the receiving trait t (row) of plant i. W_t (N x N, zero
    diagonal) holds the neighbourhood weights. They are either shared by all
    traits or specific to each receiving trait. With A = I - M invertible,
    y* = A y follows the ordinary model of the package (mean X b, covariance
    V(theta)). Everything remlax does on V therefore applies unchanged to y*.

Criteria (-2logL in the package convention; see also vignettes/structural.Rmd)
    jacobian   -2 REML(y*) - 2 log|det A|
               Change of variable y -> y*. The Harville term (the log|X'X| of
               the contrasts) is taken on X, the fixed-effect matrix of y*. It
               is therefore constant in Psi.
    reml_y     jacobian - log|X_y'X_y| + log|X'X|,    X_y = A^-1 X
               (default) The exact REML of y. The mean of y is X_y b. X_y
               depends on Psi as soon as the row sums of W vary (border
               plants). The contrasts of y are orthogonal to X_y, so the
               Harville term of y, log|X_y'X_y|, varies with Psi. This term
               does not vanish as N grows. Take psi_v = Psi s, the effect on
               the whole-neighbourhood scale (see Parametrisation). At
               psi_v = +0.6 on one trait, X_y ~ X / (1 - psi_v) in the
               interior, a shift of about log(1/0.4) = 0.9 in logLik
               (dev/notes.md). Without the term, the criterion depends on
               the scale of b in the parametrisation of y*.
    ml         -2 ML(y*) - 2 log|det A|, likelihood profiled over b. It is
               exact for y, since ML has no Harville term.
    The criteria coincide up to a constant when Psi = 0. jacobian and reml_y
    coincide exactly when W has constant row sums.

log|det A| paths, chosen once from the structure and never from theta
    nilpotent  The pattern of possibly nonzero entries of Psi (free entries,
               and fixed entries with a nonzero value) is an acyclic graph.
               Then M^q = 0, so det A = 1 exactly and A^-1 = sum_{m<q} M^m.
               There is no feedback loop and no equilibrium question.
    rapide     A single symmetric W, and each (trait, unit) pair observed
               exactly once. W = U diag(lambda) U' is computed once, and
               log|det A| = sum_k log det(I_q - lambda_k Psi). Each eigenvalue
               costs one q x q determinant.
    general    Everything else (one W per trait, nonsymmetric W, incomplete
               design). A dense LU factorisation of A (n x n) is used. Its
               cost is of the same order as the Cholesky factorisation of V
               already paid at each evaluation.
    The path names are returned to R as strings and keep their French
    spelling (rapide means fast).

Admissible region. The region kept is the set of points reachable from
Psi = 0 without crossing a singularity along the ray t Psi:

    det(I - t M) > 0 for all t in (0, 1]
    <=> every real eigenvalue of M is below 1.

Requiring every det(I_q - lambda_k Psi) > 0 is a weaker test. With
Psi = diag(c, c), each det_k equals (1 - lambda_k c)^2 >= 0. A point where two
factors have changed sign passes that test, yet it is reached only by crossing
det A = 0 (dev/notes.md). On the fast path the eigenvalues of M are lambda_k mu_i, with mu_i
the eigenvalues of Psi. The condition then involves only the eigenvalues of
Psi (q x q) and the two extreme eigenvalues of W. It is checked exactly at
each evaluation. On the general path the spectrum of M (n x n, nonsymmetric)
would cost about ten Cholesky factorisations. Only the necessary condition
det A > 0 is therefore checked at each evaluation. The exact condition is
checked at the final point and reported (`admissible`). A spectral radius
above 1 with all real eigenvalues below 1 is still admitted. A is then
invertible and the simultaneous model is defined, even though the equilibrium
is no longer the limit of the iteration y <- M y + .... The spectral radius
is reported.

Outside the region the objective equals a finite penalty (PENALITE) with zero
gradient. L-BFGS-B sees it as a steep rise and backtracks in its line search.
An infinite or NaN value would stop the optimiser. The gradient of the log of
a negative number would also spread NaN to the rest of the gradient. The code
therefore uses a double `where` guard: it computes on a safe matrix and
selects the result afterwards.

Parametrisation. For free entries, theta_lag = Psi * s, with s the mean row
sum of |W|. Inside the optimiser Psi is thus on the whole-neighbourhood scale:
one unit of theta is the effect of all neighbours together. This puts Psi on
the same scale as the log standard deviations. It also prevents a W with small
weights (1/d^2) from making the gradient in Psi orders of magnitude larger
than the others. Outputs are in units of W (Psi = theta_lag / s).

Organisation. prepare_lag computes the constants once per fit. psi_from_theta,
theta_from_psi and ystar map between theta, Psi and the response. lag_terms
returns the Psi-dependent terms of the criterion inside the differentiated
objective. lag_report and lag_component_names build the outputs at the final
point.
"""
from . import _x64  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp

CRITERES = ("reml_y", "jacobian", "ml")

# Value of -2logL outside the admissible region. It is finite (see the module
# docstring) and far above any value reachable on a data set of realistic size.
PENALITE = 1e12


def lag_n_params(lag):
    """Number of free entries of Psi (0 without a lag)."""
    if lag is None:
        return 0
    return int(np.asarray(lag["mask"], dtype=bool).sum())


def _nilpotent(B):
    """Whether the boolean pattern B (q x q) is nilpotent, that is an acyclic graph."""
    q = B.shape[0]
    P = B.astype(np.int64)
    Bi = B.astype(np.int64)
    for _ in range(q):
        P = (P @ Bi > 0).astype(np.int64)
    # P is the pattern of B^(q+1). B is nilpotent iff B^q = 0, iff B^(q+1) = 0.
    # A cycle leaves B^m nonzero for every m.
    return not P.any()


def prepare_lag(lag, y, X):
    """Constant precomputations of the lag, kept outside the differentiated function.

    The function is idempotent. A dictionary already prepared is returned as
    is, so the restarts of fit_reml do not redo the eigendecomposition.
    """
    if lag is None or lag.get("_prepare"):
        return lag
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, p = X.shape
    q, N = int(lag["q"]), int(lag["N"])
    tr = np.asarray(lag["trait"], dtype=np.int64)
    unit = np.asarray(lag["unit"], dtype=np.int64)
    if tr.shape != (n,) or unit.shape != (n,):
        raise ValueError("lag : `trait` et `unit` doivent avoir %d elements (un par ligne de y)." % n)
    if tr.min() < 0 or tr.max() >= q or unit.min() < 0 or unit.max() >= N:
        raise ValueError("lag : indice de caractere hors de 0..%d ou d'unite hors de 0..%d." % (q - 1, N - 1))
    key = tr * N + unit
    if len(np.unique(key)) != n:
        raise ValueError("lag : %d couple(s) (caractere, unite) en double. Chaque unite ne "
                         "peut porter qu'une observation par caractere : sinon M n'est pas "
                         "definie." % (n - len(np.unique(key))))
    Ws = [sp.csr_matrix(W, dtype=np.float64) for W in lag["W"]]
    if len(Ws) not in (1, q):
        raise ValueError("lag : %d matrice(s) W pour %d caracteres (1 ou %d attendues)."
                         % (len(Ws), q, q))
    for k, W in enumerate(Ws):
        if W.shape != (N, N):
            raise ValueError("lag : W[%d] est %s, attendu (%d, %d)." % (k, W.shape, N, N))
        if np.any(W.diagonal() != 0):
            raise ValueError("lag : W[%d] a une diagonale non nulle. Une plante n'est pas sa "
                             "propre voisine ; un effet sur soi-meme rendrait Psi non identifie "
                             "face aux effets fixes et a la residuelle." % k)
    mask = np.asarray(lag["mask"], dtype=bool).reshape(q, q)
    value = np.asarray(lag.get("value", np.zeros((q, q))), dtype=np.float64).reshape(q, q)
    value = np.where(mask, 0.0, value)
    crit = str(lag.get("criterion", "reml_y"))
    if crit not in CRITERES:
        raise ValueError("lag : critere '%s' inconnu (%s)." % (crit, ", ".join(CRITERES)))

    # Scale: mean row sum of |W|, averaged over the W matrices.
    s = float(np.mean([np.asarray(abs(W).sum(axis=1)).mean() for W in Ws]))
    s = s if s > 0 else 1.0

    widx = np.zeros(n, dtype=np.int64) if len(Ws) == 1 else tr
    # Q[r, u] = (W_{t(r)} y_u)[unit(r)], so that M y = sum_u Psi[t(r), u] Q[r, u].
    # y* is therefore linear in Psi with Q constant. Computing it costs n q per
    # evaluation instead of an n x n product.
    yfull = np.zeros((q, N))
    yfull[tr, unit] = y
    WY = [np.asarray(W @ yfull.T) for W in Ws]            # (N, q) each
    Q = np.empty((n, q))
    for k in range(len(Ws)):
        rows = np.flatnonzero(widx == k)
        Q[rows] = WY[k][unit[rows]]

    pattern = mask | (value != 0)
    complete = n == q * N
    single_W = len(Ws) == 1 or all((abs(W - Ws[0]).max() if (W - Ws[0]).nnz else 0.0) == 0
                                   for W in Ws[1:])
    W0 = Ws[0]
    asym = abs(W0 - W0.T).max() if W0.nnz else 0.0
    sym = asym <= 1e-12 * max(1.0, abs(W0).max() if W0.nnz else 1.0)
    if _nilpotent(pattern):
        path = "nilpotent"
    elif single_W and sym and complete:
        path = "rapide"
    else:
        path = "general"
    # Forcing the path is meant for tests only, to compare the fast and general
    # paths on the same model. The nilpotent path requires a nilpotent pattern.
    force = lag.get("force_chemin")
    if force:
        if force not in ("rapide", "general", "nilpotent"):
            raise ValueError("lag : force_chemin = %r inconnu." % force)
        if force == "nilpotent" and path != "nilpotent":
            raise ValueError("lag : chemin nilpotent force sur un motif de Psi avec boucle.")
        if force == "rapide" and not (single_W and sym and complete):
            raise ValueError("lag : chemin rapide force sans W unique symetrique et plan complet.")
        path = force

    mi, mj = np.nonzero(mask)
    # When Psi is identically zero (no free entry, no nonzero fixed value),
    # A = I and the objective must equal the one of the model without a lag,
    # to the bit. The code therefore avoids slogdet(X'X) in JAX, which differs
    # from the numpy log|X'X| in the last digit, and any other floating-point
    # detour.
    empty = not pattern.any()
    out = dict(lag, _prepare=True, vide=empty, q=q, N=N, n=n, p=p, trait=tr, unit=unit, mask=mask,
               value=value, criterion=crit, scale=s, chemin=path, Q=jnp.asarray(Q),
               tr_j=jnp.asarray(tr), mi=mi, mj=mj, value_j=jnp.asarray(value),
               traits=list(lag.get("traits") or ["T%d" % (k + 1) for k in range(q)]),
               logXX=(float(np.linalg.slogdet(X.T @ X)[1]) if p else 0.0))
    if path == "rapide":
        lam, U = np.linalg.eigh(W0.toarray())
        pos = np.empty((q, N), dtype=np.int64)
        pos[tr, unit] = np.arange(n)
        Xt = np.einsum("ik,tip->ktp", U, X[pos]) if p else np.zeros((N, q, 0))
        out.update(lam=jnp.asarray(lam), lam_np=lam, Xt=jnp.asarray(Xt),
                   lam_min=float(lam.min()), lam_max=float(lam.max()))
    else:
        # Long W: Wl[r, s] = W_{t(r)}[unit(r), unit(s)], dense n x n like V.
        Wl = np.empty((n, n))
        for k, W in enumerate(Ws):
            rows = np.flatnonzero(widx == k)
            if rows.size:
                Wd = W[unit[rows]].toarray()
                Wl[rows] = Wd[:, unit]
        out.update(Wl=jnp.asarray(Wl))
    return out


def psi_from_theta(th_lag, L):
    """Psi (q x q, in units of W) from the free parameters."""
    Psi = L["value_j"]
    if len(L["mi"]):
        Psi = Psi.at[L["mi"], L["mj"]].set(jnp.asarray(th_lag) / L["scale"])
    return Psi


def theta_from_psi(Psi, L):
    """Inverse of psi_from_theta on the free entries, used for starting values."""
    Psi = np.asarray(Psi, dtype=np.float64)
    return Psi[L["mi"], L["mj"]] * L["scale"]


def ystar(th_lag, L, y):
    """y* = (I - M) y = y - sum_u Psi[t(r), u] Q[r, u]."""
    if L.get("vide"):
        return jnp.asarray(y)
    Psi = psi_from_theta(th_lag, L)
    return jnp.asarray(y) - jnp.sum(Psi[L["tr_j"], :] * L["Q"], axis=1)


def _real_eigenvalues_admissible(Psi, lam_min, lam_max):
    """Whether every real eigenvalue mu of Psi gives mu lambda < 1 at both ends of the spectrum of W.

    Plain numpy, called outside differentiation.
    """
    Psi = np.asarray(Psi, dtype=np.float64)
    mu = np.linalg.eigvals(Psi)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(mu))) if mu.size else 1.0)
    re = mu.real[np.abs(mu.imag) <= tol]
    return np.bool_(np.all(re * lam_max < 1.0) and np.all(re * lam_min < 1.0))


def _ok_rapide(Psi, L):
    """Exact region condition of the fast path (module docstring), under jit.

    It is a numpy callback on the q x q matrix Psi. Psi goes through
    stop_gradient because the result has no derivative. Without it,
    differentiation would look for a derivative rule of the callback.
    """
    lmin, lmax = L["lam_min"], L["lam_max"]
    return jax.pure_callback(lambda P: _real_eigenvalues_admissible(P, lmin, lmax),
                             jax.ShapeDtypeStruct((), jnp.bool_),
                             jax.lax.stop_gradient(Psi))


def _M_dense(Psi, L):
    """Dense M (n x n) on the general and nilpotent paths."""
    tr = L["tr_j"]
    return Psi[tr[:, None], tr[None, :]] * L["Wl"]


def lag_terms(th_lag, L, X):
    """Terms of the criterion that depend on Psi beyond y*.

    Returns (log|det A|, mean term, ok). The mean term is
    1/2 log|X_y'X_y| - 1/2 log|X'X|, and 0 when the criterion does not use it.
    ok tells whether the point lies in the admissible region (JAX boolean).
    """
    if L.get("vide"):
        return jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(True)
    Psi = psi_from_theta(th_lag, L)
    q = L["q"]
    need_mean = L["criterion"] == "reml_y" and L["p"] > 0
    Xj = jnp.asarray(X)
    if L["chemin"] == "nilpotent":
        ldA, ok = jnp.asarray(0.0), jnp.asarray(True)
        if need_mean:
            M = _M_dense(Psi, L)
            Xy = Xj
            for _ in range(q - 1):            # A^-1 X = sum_{m<q} M^m X, exact
                Xy = Xj + M @ Xy
        else:
            Xy = None
    elif L["chemin"] == "rapide":
        lam = L["lam"]
        B = jnp.eye(q)[None] - lam[:, None, None] * Psi[None]
        d = jnp.linalg.det(B)
        ok = jnp.all(d > 0) & _ok_rapide(Psi, L)
        # Double guard: compute on a safe matrix, then select (module docstring).
        Bs = jnp.where(ok, B, jnp.eye(q)[None])
        ldA = jnp.sum(jnp.linalg.slogdet(Bs)[1])
        if need_mean:
            Yk = jnp.linalg.solve(Bs, L["Xt"])                 # (N, q, p)
            G = jnp.einsum("ktp,ktm->pm", Yk, Yk)
            return ldA, 0.5 * (jnp.linalg.slogdet(G)[1] - L["logXX"]), ok
        return ldA, jnp.asarray(0.0), ok
    else:
        n = L["n"]
        A = jnp.eye(n) - _M_dense(Psi, L)
        lu, piv = jax.scipy.linalg.lu_factor(A)
        dg = jnp.diagonal(lu)
        n_swaps = jnp.sum(piv != jnp.arange(n))
        sign = jnp.prod(jnp.sign(dg)) * (1 - 2 * (n_swaps % 2))
        ok = (sign > 0) & jnp.all(dg != 0)
        ldA = jnp.sum(jnp.log(jnp.abs(jnp.where(dg == 0, 1.0, dg))))
        Xy = jax.scipy.linalg.lu_solve((lu, piv), Xj) if need_mean else None
    if need_mean:
        G = Xy.T @ Xy
        return ldA, 0.5 * (jnp.linalg.slogdet(G)[1] - L["logXX"]), ok
    return ldA, jnp.asarray(0.0), ok


def lag_report(th_lag, L, X, neg2):
    """Lag outputs at the final point (numpy), including the spectral radius of M."""
    Psi = np.asarray(psi_from_theta(jnp.asarray(th_lag), L))
    crit = L["criterion"]
    # The mean term is reported even for the criteria that leave it out. The
    # user can then move from one criterion to another without refitting.
    L2 = dict(L, criterion="reml_y")
    ldA, ldm, ok = lag_terms(jnp.asarray(th_lag), L2, X)
    ldA, ldm, ok = float(ldA), float(ldm), bool(ok)
    if L["chemin"] == "nilpotent":
        rho = 0.0
    elif L["chemin"] == "rapide":
        rho = float(np.max(np.abs(L["lam_np"])) * np.max(np.abs(np.linalg.eigvals(Psi))))
    else:
        # General path: the spectrum of M, computed once, gives the spectral
        # radius and the exact region condition. Only det A > 0 was checked
        # during the optimisation.
        mu = np.linalg.eigvals(np.asarray(_M_dense(jnp.asarray(Psi), L)))
        rho = float(np.max(np.abs(mu))) if mu.size else 0.0
        tol = 1e-10 * max(1.0, rho)
        ok = ok and bool(np.all(mu.real[np.abs(mu.imag) <= tol] < 1.0))
    ll = -0.5 * float(neg2)
    if crit == "reml_y":
        ll_jac = ll - ldm
        ll_ys = ll_jac - ldA
    elif crit == "jacobian":
        ll_jac = ll
        ll_ys = ll - ldA
    else:
        ll_jac = float("nan")
        ll_ys = ll - ldA
    return dict(Psi=Psi, logdet_A=ldA, logdet_mean=ldm, admissible=ok, criterion=crit,
                chemin=L["chemin"], rayon_spectral=rho, logLik_jacobian=ll_jac,
                logLik_ystar=ll_ys, scale=float(L["scale"]), n_free=int(len(L["mi"])),
                traits=list(L["traits"]), mask=np.asarray(L["mask"]))


def lag_component_names(L):
    """Names of the free entries in theta order, as Psi[receiver<-emitter]."""
    names = L.get("traits") or ["T%d" % (k + 1) for k in range(int(L["q"]))]
    mi, mj = np.nonzero(np.asarray(L["mask"], dtype=bool).reshape(int(L["q"]), int(L["q"])))
    return ["Psi[%s<-%s]" % (names[i], names[j]) for i, j in zip(mi, mj)]
