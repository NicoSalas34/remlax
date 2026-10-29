"""Covariance structures between levels, an asreml-style catalogue.

Every structure returns a correlation matrix C (unit diagonal); the variance
lives in Sigma (structures.py). This is the decomposition Sigma_h = D C D of
the asreml manual (appendix C), D = diag(sigma_i): asreml's `ar1v` is here
struct = "iid" with level = "ar1", and `ar1h` is struct = "diag" with
level = "ar1".

Stationary structures are Toeplitz: the autocorrelation function
rho_0..rho_{q-1} is computed, then C[i,j] = rho_{|i-j|}. One code path, one
thing to check per structure.

CATALOGUE (definitions of appendix C of the ASReml-R 4.2 manual)

  id      C = I                                              0 parameters
  fixed   C given (GRM, pedigree, computed kernel)           0
  cor     C_ij = theta (i != j), uniform correlation         1
  ar1     rho_k = phi^k                                      1
  ar2     rho_1 = phi1/(1-phi2), then Yule-Walker            2
  ar3     the same at order 3                                3
  sar     constrained AR(2): phi1 = phi, phi2 = -phi^2/4     1
  ma1     rho_1 = -theta/(1+theta^2), 0 beyond               1
  ma2     rho_1, rho_2 given, 0 beyond                       2
  arma    ARMA(1,1), rho_k = ph * rho_{k-1}                  2
  corb    rho_k = phi_k for k < b, 0 beyond (band)           b
  exp     C_ij = phi^|x_i-x_j| (coordinates, not
          necessarily equidistant)                           1
  gau     C_ij = phi^{(x_i-x_j)^2}                           1
  lvr     C_ij = max(0, 1 - d_ij/phi), "linear variance":
          a truncated tent, positive definite in 1D only;
          phi is a range, hence exp(theta). No 2D version:
          asreml's `ilv` formula could not be recovered and
          the Euclidean tent is not positive definite.       1
  mtrn    anisotropic Matern (phi, nu, delta, alpha;
          lambda fixed at 1 or 2)                   0 to 4 estimated
  own     user-defined: arithmetic expression of d/dx/dy/lag
          and p1..pk                                         k

lvr and mtrn were identified against asreml, not read in the manual, whose
help pages do not give the formulas (dev/notes.md).

ar2/ar3 are stationary by construction. |phi_i| < 1 is not enough: the
stationarity region of an AR(2) is a triangle. The partial correlations are
parametrised instead (each in (-1, 1) through tanh), and the phi follow by
Levinson-Durbin; the map PACF <-> stationary region is exact, and the reported
phi are those of asreml.
"""
from . import _x64  # noqa: F401
import jax.numpy as jnp
import numpy as np

from .bessel import matern as _matern

# name -> number of parameters (None: variable, see n_level_params)
LEVEL_STRUCTURES = {
    "id": 0, "fixed": 0, "cor": 1, "ar1": 1, "ar2": 2, "ar3": 3,
    "sar": 1, "ma1": 1, "ma2": 2, "arma": 2, "corb": None,
    # irregular 1D
    "exp": 1, "gau": 1,
    # irregular 2D, isotropic or anisotropic (appendix C, "power models")
    "iexp": 1, "igau": 1, "ieuc": 1, "sph": 1, "cir": 1,
    "aexp": 2, "agau": 2,
    # range metrics: the parameter is a distance, not a correlation
    "lvr": 1,
    # variable number: see n_level_params
    "mtrn": None, "own": None,
    # general correlation (corg): omega(omega-1)/2 parameters
    "corg": None,
}

# Canonical order of the mtrn parameters: their place in theta and in the
# outputs. Changing it would invalidate every recorded fit.
MTRN_PARAMS = ("phi", "nu", "delta", "alpha")
MTRN_DEFAUT = {"phi": 1.0, "nu": 0.5, "delta": 1.0, "alpha": 0.0, "lambda": 2.0}
# "sep": separable product of several factors (ar1 x ar1, exp x ar1, ...);
# "ar1ar1" is kept as a shortcut for the most common case.
# Structures that need coordinates:
LEVEL_NEEDS_COORD = ("exp", "gau", "iexp", "igau", "ieuc", "sph", "cir",
                     "aexp", "agau", "mtrn", "lvr")


def n_level_params(kind, order=0, parts=None, opts=None):
    """Number of parameters.

    parts: list of (kind, dimension) of a separable product, which adds up its
    factors. opts: settings of the structure (mtrn: which parameters are
    estimated; own: how many the user declares). order: the band of corb, the
    omega of corg."""
    if kind == "ar1ar1":
        return 2
    if kind == "sep":
        # `parts` carries (family, dimension), not the order or options, so
        # families whose count depends on an order or options, and metric
        # families that need coordinates, are refused: theta would be cut
        # differently from level_chol, or a factor would become an identity
        # silently (dev/notes.md).
        deps = ("ar2", "ar3", "ma2", "arma", "corb", "corg", "mtrn", "own") + LEVEL_NEEDS_COORD
        bad = [k for k, _ in (parts or []) if k in deps]
        if bad:
            raise ValueError(
                "produit separable : les familles %s ne sont pas admises dans "
                "`parts`, qui ne porte que (famille, dimension) : leur nombre de "
                "parametres depend d'un ordre ou d'options, ou elles exigent des "
                "coordonnees. Les declarer ici decouperait theta differemment de "
                "level_chol, ou donnerait une identite en silence. Utiliser "
                "`expr` pour ces cas." % sorted(set(bad)))
        return sum(n_level_params(k) for k, _ in (parts or []))
    if kind == "corb":
        return int(order)
    if kind == "corg":
        q = int(order)                      # here `order` carries omega
        return q * (q - 1) // 2
    if kind == "mtrn":
        # As in asreml, a parameter absent from the call is held at its
        # default; only the declared parameters are estimated.
        o = opts or {}
        return int(sum(float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5
                       for p in MTRN_PARAMS))
    if kind == "own":
        return int((opts or {}).get("n_par", 0))
    n = LEVEL_STRUCTURES.get(kind)
    if n is None:
        raise ValueError("structure de niveaux inconnue : %s" % kind)
    return n


# ==============================================================================
# Autocorrelation functions
# ==============================================================================
def _levinson_phi(pacf):
    """AR coefficients phi_1..phi_p from the partial correlations, by the
    Levinson-Durbin recursion, which maps (-1, 1)^p exactly onto the
    stationarity region."""
    p = len(pacf)
    phi = jnp.zeros(p)
    for k in range(p):
        pk = pacf[k]
        if k == 0:
            phi = phi.at[0].set(pk)
        else:
            prev = phi[:k]
            phi = phi.at[:k].set(prev - pk * prev[::-1])
            phi = phi.at[k].set(pk)
    return phi


def _acf_ar(phi, q):
    """ACF of an AR(p): rho_k = sum_i phi_i rho_{|k-i|}, a linear system in
    rho for k = 1..p (Yule-Walker), a recursion beyond."""
    p = len(phi)
    if p == 0:
        return jnp.concatenate([jnp.ones(1), jnp.zeros(q - 1)])
    # system A rho = b with rho = (rho_1..rho_p)
    A = jnp.eye(p)
    b = jnp.zeros(p)
    for k in range(1, p + 1):
        b = b.at[k - 1].add(phi[k - 1])          # term i = k: phi_k * rho_0
        for i in range(1, p + 1):
            if i == k:
                continue
            j = abs(k - i)                        # rho_{|k-i|}
            if j == 0:
                b = b.at[k - 1].add(phi[i - 1])
            else:
                A = A.at[k - 1, j - 1].add(-phi[i - 1])
    rho = jnp.linalg.solve(A, b)
    acf = jnp.concatenate([jnp.ones(1), rho])
    for k in range(p + 1, q):
        acf = jnp.concatenate([acf, jnp.sum(phi * acf[k - 1:k - p - 1:-1])[None]])
    return acf[:q]


def _acf(theta, kind, q, order=0):
    """rho_0..rho_{q-1} of a stationary structure."""
    if kind == "id":
        return jnp.concatenate([jnp.ones(1), jnp.zeros(q - 1)])
    if kind == "cor":
        # Uniform correlation: C = (1-t) I + t J is positive definite only for
        # t > -1/(q-1). (-1, 1) is mapped onto (-1/(q-1), 1), exactly the
        # admissible domain.
        lo = -1.0 / max(q - 1, 1)
        t = lo + (jnp.tanh(theta[0]) + 1.0) * 0.5 * (1.0 - lo)
        return jnp.concatenate([jnp.ones(1), jnp.full(q - 1, t)])
    if kind == "ar1":
        phi = jnp.tanh(theta[0])
        return phi ** jnp.arange(q)
    if kind in ("ar2", "ar3"):
        p = 2 if kind == "ar2" else 3
        return _acf_ar(_levinson_phi(jnp.tanh(theta[:p])), q)
    if kind == "sar":
        # Constrained AR(2): phi1 = phi, phi2 = -phi^2/4, always stationary
        # (double real roots).
        ph = jnp.tanh(theta[0])
        return _acf_ar(jnp.array([ph, -ph ** 2 / 4.0]), q)
    if kind == "ma1":
        th = jnp.tanh(theta[0])
        r1 = -th / (1.0 + th ** 2)
        return jnp.concatenate([jnp.ones(1), r1[None], jnp.zeros(max(q - 2, 0))])[:q]
    if kind == "ma2":
        t1, t2 = jnp.tanh(theta[0]), jnp.tanh(theta[1])
        den = 1.0 + t1 ** 2 + t2 ** 2
        r1 = -t1 * (1.0 - t2) / den
        r2 = -t2 / den
        return jnp.concatenate([jnp.ones(1), r1[None], r2[None],
                                jnp.zeros(max(q - 3, 0))])[:q]
    if kind == "arma":
        # ACF of an ARMA(1,1): rho_1 = (1+phi*th)(phi+th)/(1+2*phi*th+th^2),
        # then rho_k = phi * rho_{k-1}; the standard form, valid on the whole
        # square |phi| < 1, |th| < 1. The formula printed in appendix C of the
        # manual gives sequences that are not positive definite on much of
        # that square (dev/notes.md); only the sign of the reported MA
        # parameter may differ from asreml.
        th, ph = jnp.tanh(theta[0]), jnp.tanh(theta[1])
        r1 = (1.0 + ph * th) * (ph + th) / (1.0 + 2.0 * ph * th + th ** 2)
        acf = jnp.concatenate([jnp.ones(1), r1[None]])
        for k in range(2, q):
            acf = jnp.concatenate([acf, (ph * acf[k - 1])[None]])
        return acf[:q]
    if kind == "corb":
        b = int(order)
        vals = jnp.tanh(theta[:b])
        return jnp.concatenate([jnp.ones(1), vals, jnp.zeros(max(q - 1 - b, 0))])[:q]
    raise ValueError("pas d'ACF pour la structure %s" % kind)


def _tril_off(t):
    """Indices of the strict lower triangle, row by row."""
    ii, jj = [], []
    for i in range(t):
        for j in range(i):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)


def _toeplitz(acf, q):
    i = jnp.arange(q)
    return acf[jnp.abs(i[:, None] - i[None, :])]


# ==============================================================================
# Correlation matrix
# ==============================================================================
def _mtrn_valeurs(theta_lv, opts):
    """(phi, nu, delta, alpha, lambda) of mtrn: the estimated ones from theta,
    the others from opts. phi, nu and delta are positive (exp); alpha is an
    angle, unconstrained."""
    o = opts or {}
    val, k = {}, 0
    for p in MTRN_PARAMS:
        estime = float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5
        if estime:
            t = theta_lv[k]; k += 1
            val[p] = t if p == "alpha" else jnp.exp(t)
        else:
            val[p] = jnp.asarray(float(o.get(p, MTRN_DEFAUT[p])))
    val["lambda"] = float(o.get("lambda", MTRN_DEFAUT["lambda"]))
    return val


def _distance_anisotrope(dx, dy, delta, alpha, lam):
    """Metric of Haskard et al. (2007), as ASReml uses it:

        u =  dx cos(alpha) + dy sin(alpha)
        v = -dx sin(alpha) + dy cos(alpha)
        h = ( |sqrt(delta) u|^lambda + |v/sqrt(delta)|^lambda )^(1/lambda)

    delta acts as a square root, in opposite directions on the two axes: the
    map preserves areas, so delta is not confounded with the range phi
    (dev/notes.md)."""
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    sd = jnp.sqrt(delta)
    a, b = jnp.abs(sd * u), jnp.abs(v / sd)
    if abs(lam - 2.0) < 1e-12:
        return jnp.sqrt(a ** 2 + b ** 2)
    if abs(lam - 1.0) < 1e-12:
        return a + b
    return (a ** lam + b ** lam) ** (1.0 / lam)


def _own_corr(theta_lv, q, expr, coord=None, opts=None):
    """User-defined structure: an arithmetic expression.

    asreml's own(obj, fun) calls an R function at each evaluation; here the
    derivative is needed, which an opaque R function does not give. The
    expression crosses the boundary as a string and is differentiated
    automatically. It is written in R, so `^` is translated to `**` (in Python
    `^` is the exclusive or).

    Variables: d (Euclidean distance), dx, dy (distances per axis), lag
    (|i-j|), I, J (identity, matrix of ones), p1..pk (raw unconstrained
    parameters: the expression applies exp or tanh itself), and the usual
    functions plus matern.

    Unless opts["normalise"] = 0, D^-1/2 C D^-1/2 is returned: this module
    returns correlation matrices, the scale being carried by Sigma."""
    if not expr:
        raise ValueError("own : aucune expression fournie.")
    expr = expr.replace("^", "**")
    i = jnp.arange(q)
    lag = jnp.abs(i[:, None] - i[None, :]).astype(jnp.float64)
    if coord is not None:
        x = jnp.asarray(coord)
        if x.ndim == 1:
            x = x[:, None]
        dx = x[:, None, 0] - x[None, :, 0]
        dy = (x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1 else jnp.zeros_like(dx)
    else:
        dx = i[:, None] - i[None, :]
        dx = dx.astype(jnp.float64)
        dy = jnp.zeros_like(dx)
    env = {"d": jnp.sqrt(dx ** 2 + dy ** 2), "dx": jnp.abs(dx), "dy": jnp.abs(dy),
           "lag": lag, "I": jnp.eye(q), "J": jnp.ones((q, q)), "q": float(q),
           "exp": jnp.exp, "log": jnp.log, "sqrt": jnp.sqrt, "abs": jnp.abs,
           "tanh": jnp.tanh, "sinh": jnp.sinh, "cosh": jnp.cosh, "sin": jnp.sin,
           "cos": jnp.cos, "minimum": jnp.minimum, "maximum": jnp.maximum,
           "where": jnp.where, "pi": float(np.pi), "matern": _matern}
    for k in range(len(theta_lv)):
        env["p%d" % (k + 1)] = theta_lv[k]
    C = eval(expr, {"__builtins__": {}}, env)          # noqa: S307
    C = 0.5 * (C + C.T)
    if float((opts or {}).get("normalise", 1.0)) > 0.5:
        s = jnp.sqrt(jnp.clip(jnp.diagonal(C), 1e-300, None))
        C = C / s[:, None] / s[None, :]
    return C


def level_corr(theta_lv, kind, q, order=0, coord=None, C_fixed=None, parts=None,
               dims=None, opts=None, expr=None):
    """Correlation matrix C (q x q) of the structure."""
    if kind == "own":
        return _own_corr(theta_lv, q, expr, coord=coord, opts=opts)
    if kind == "id":
        return jnp.eye(q)
    if kind == "ar1ar1":
        if dims is None or len(dims) != 2:
            raise ValueError("ar1ar1 exige dims = (n_lignes, n_colonnes)")
        nr, nc = int(dims[0]), int(dims[1])
        Cr = _toeplitz(_acf(theta_lv[:1], "ar1", nr), nr)
        Cc = _toeplitz(_acf(theta_lv[1:2], "ar1", nc), nc)
        return jnp.kron(Cr, Cc)
    if kind == "fixed":
        return jnp.asarray(C_fixed)
    if kind == "corg":
        return _corg_corr(theta_lv, q, order)
    if kind in LEVEL_NEEDS_COORD:
        return _metric_corr(theta_lv, kind, coord, opts)
    if kind == "sep":
        return _sep_corr(theta_lv, parts)
    return _toeplitz(_acf(theta_lv, kind, q, order), q)


def _corg_corr(theta_lv, q, order):
    """General correlation: every correlation free, unit diagonal, through a
    Cholesky factor with normalised rows, the parametrisation that gives both
    an exact unit diagonal and positive definiteness without constraint."""
    w = int(order) if order else q
    ii, jj = _tril_off(w)
    L = jnp.eye(w).at[ii, jj].set(theta_lv[:len(ii)])
    L = L / jnp.sqrt(jnp.sum(L ** 2, axis=1, keepdims=True))
    return L @ L.T


def _sep_corr(theta_lv, parts):
    """Separable product C = C_1 (x) C_2 (x) ...; the first factor varies the
    slowest, as in a kron. parts carries (family, dimension of the factor), as
    in level_chol."""
    C, o = None, 0
    for (k, qi) in (parts or []):
        p = n_level_params(k)
        C_i = level_corr(theta_lv[o:o + p], k, int(qi))
        C = C_i if C is None else jnp.kron(C, C_i)
        o += p
    return C


def _metric_corr(theta_lv, kind, coord, opts):
    """Correlation of the families defined on coordinates (LEVEL_NEEDS_COORD).

    phi = |tanh(theta)| clipped to (1e-12, 1 - 1e-12) is computed on demand: an
    mtrn with every parameter fixed has no theta. iexp uses the city-block
    metric, as in the manual.

    Range models (sph, cir) vanish beyond the range phi = exp(theta). The
    formula is written on a safe t (0 beyond the range) and then selected with
    jnp.where on both sides: clipping t at 1 gave 0 x inf = NaN in the
    gradient of cir (sqrt(1 - t^2) and arcsin(t) have infinite derivatives at
    t = 1), and L-BFGS-B returned the start (dev/notes.md)."""
    x = jnp.asarray(coord)
    if x.ndim == 1:
        x = x[:, None]
    dx = jnp.abs(x[:, None, 0] - x[None, :, 0])
    dy = (jnp.abs(x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1
          else jnp.zeros_like(dx))
    d_eu = jnp.sqrt(dx ** 2 + dy ** 2)

    def _ph():
        return jnp.clip(jnp.abs(jnp.tanh(theta_lv[0])), 1e-12, 1 - 1e-12)
    if kind == "exp":                      # 1D: |x_i - x_j|
        return _ph() ** d_eu
    if kind == "gau":                      # 1D: (x_i - x_j)^2
        return _ph() ** (d_eu ** 2)
    if kind == "iexp":                     # 2D isotropic, city-block metric
        return _ph() ** (dx + dy)
    if kind == "igau":
        return _ph() ** (dx ** 2 + dy ** 2)
    if kind == "ieuc":
        return _ph() ** d_eu
    if kind == "aexp":                     # anisotropic: one phi per axis
        p2 = jnp.clip(jnp.abs(jnp.tanh(theta_lv[1])), 1e-12, 1 - 1e-12)
        return (_ph() ** dx) * (p2 ** dy)
    if kind == "agau":
        p2 = jnp.clip(jnp.abs(jnp.tanh(theta_lv[1])), 1e-12, 1 - 1e-12)
        return (_ph() ** (dx ** 2)) * (p2 ** (dy ** 2))
    if kind == "lvr":
        # Linear variance: the correlation decreases linearly to zero at the
        # range; positive definite in one dimension only.
        return jnp.clip(1.0 - dx / jnp.exp(theta_lv[0]), 0.0, None)
    if kind in ("sph", "cir"):
        rg = jnp.exp(theta_lv[0])
        inside = d_eu < rg
        t = jnp.where(inside, d_eu / rg, 0.0)
        if kind == "sph":
            f = 1.0 - 1.5 * t + 0.5 * t ** 3
        else:
            f = 1.0 - (2.0 / jnp.pi) * (t * jnp.sqrt(1.0 - t ** 2) + jnp.arcsin(t))
        return jnp.where(inside, f, 0.0)
    if kind == "mtrn":
        # Full anisotropic Matern (Haskard et al. 2007) as in asreml: range phi,
        # shape nu (any value, K_nu from bessel.py), anisotropy ratio delta,
        # angle alpha, metric lambda (1 city-block, 2 Euclidean).
        v = _mtrn_valeurs(theta_lv, opts)
        sx = x[:, None, 0] - x[None, :, 0]
        sy = (x[:, None, 1] - x[None, :, 1]) if x.shape[1] > 1 else jnp.zeros_like(sx)
        h = _distance_anisotrope(sx, sy, v["delta"], v["alpha"], v["lambda"])
        return _matern(h / v["phi"], v["nu"])


def level_chol(theta_lv, kind, q, dims=None, LK_fixed=None, order=0,
               coord=None, parts=None, jitter=1e-10, opts=None, expr=None):
    """Factor L such that L L' = C; None means identity (no product to form).

    id gives None; fixed, the factor computed on the R side; ar1, ar1ar1 and
    sep, closed forms (a kron of factors for sep). The others go through a
    numerical Cholesky with a jitter: a band correlation (ma1, corb) can be
    singular at the edges of its domain."""
    if kind == "id":
        return None
    if kind == "fixed":
        return jnp.asarray(LK_fixed)
    if kind == "ar1":
        return ar1_chol(jnp.tanh(theta_lv[0]), q)
    if kind == "ar1ar1":
        nr, nc = dims
        return jnp.kron(ar1_chol(jnp.tanh(theta_lv[0]), nr),
                        ar1_chol(jnp.tanh(theta_lv[1]), nc))
    if kind == "sep":
        L, o = None, 0
        for (k, qi) in (parts or []):
            p = n_level_params(k)
            Li = level_chol(theta_lv[o:o + p], k, qi)
            Li = jnp.eye(qi) if Li is None else Li
            L = Li if L is None else jnp.kron(L, Li)
            o += p
        return L
    C = level_corr(theta_lv, kind, q, order=order, coord=coord, opts=opts,
                   expr=expr, dims=dims)
    return jnp.linalg.cholesky(C + jitter * jnp.eye(q))


def ar1_chol(rho, n):
    """L such that L L' = AR(1) correlation, in closed form:

        L[i,0] = rho^i;  L[i,j] = rho^(i-j) sqrt(1-rho^2) for 1 <= j <= i."""
    i = jnp.arange(n)
    d = i[:, None] - i[None, :]
    s = jnp.sqrt(jnp.clip(1.0 - rho ** 2, 1e-300, None))
    L = jnp.where(d >= 0, rho ** jnp.abs(d) * s, 0.0)
    return L.at[:, 0].set(rho ** i)


def level_params_report(theta_lv, kind, order=0, opts=None, q=None, parts=None):
    """Parameters on the asreml scale (phi, theta) for the outputs: the value
    actually used in C, never the intermediate theta.

      cor        reported on (-1/(q-1), 1), as in _acf; q is required.
      metric     exp, gau, iexp, igau, ieuc, aexp, agau raise |tanh(theta)| to
      families   a distance: the sign of theta is not identified, so the
                 absolute value is reported and signe_non_identifie says so.
      sep        each factor, named <family>_<position>_<parameter>, since two
                 factors of the same family are the norm.
    """
    if kind == "sep":
        out, o = {}, 0
        for idx, (k, qi) in enumerate(parts or [], start=1):
            pk = n_level_params(k)
            if pk:
                sub = level_params_report(theta_lv[o:o + pk], k, order, q=qi)
                for nm, v in (sub or {}).items():
                    out["%s_%d_%s" % (k, idx, nm)] = v
            o += pk
        return out

    th = np.asarray(theta_lv, dtype=float)
    absta = lambda j: float(np.clip(np.abs(np.tanh(th[j])), 1e-12, 1 - 1e-12))  # noqa: E731
    if kind == "ar1ar1":
        return {"phi": [float(np.tanh(th[0])), float(np.tanh(th[1]))]}
    if kind == "cor":
        if q is None:
            raise ValueError(
                "level_params_report(kind='cor') exige q : la correlation uniforme "
                "est remise a l'echelle sur (-1/(q-1), 1), donc tanh(theta) n'est "
                "pas la valeur du modele.")
        lo = -1.0 / max(int(q) - 1, 1)
        return {"phi": [float(lo + (np.tanh(th[0]) + 1.0) * 0.5 * (1.0 - lo))],
                "borne_inf": [float(lo)]}
    if kind in ("ar1", "sar", "ma1"):
        return {"phi": [float(np.tanh(th[0]))]}
    if kind in ("exp", "gau"):
        return {"phi": [absta(0)], "signe_non_identifie": [1.0]}
    if kind in ("ar2", "ar3"):
        p = 2 if kind == "ar2" else 3
        return {"phi": [float(v) for v in np.asarray(_levinson_phi(jnp.tanh(jnp.asarray(th[:p]))))],
                "pacf": [float(np.tanh(v)) for v in th[:p]]}
    if kind in ("ma2", "arma"):
        return {"phi": [float(np.tanh(v)) for v in th[:2]]}
    if kind in ("aexp", "agau"):
        return {"phi": [absta(0), absta(1)], "signe_non_identifie": [1.0]}
    if kind in ("iexp", "igau", "ieuc"):
        return {"phi": [absta(0)], "signe_non_identifie": [1.0]}
    if kind in ("sph", "cir", "lvr"):
        return {"portee": [float(np.exp(th[0]))]}
    if kind == "mtrn":
        o = opts or {}
        out, k = {}, 0
        for p in MTRN_PARAMS:
            if float(o.get("est_" + p, 1.0 if p == "phi" else 0.0)) > 0.5:
                out[p] = [float(th[k]) if p == "alpha" else float(np.exp(th[k]))]
                k += 1
        return out
    if kind == "own":
        return {"own": [float(v) for v in th]}
    if kind == "corg":
        return {}
    if kind == "corb":
        return {"phi": [float(np.tanh(v)) for v in th[:int(order)]]}
    return {}
