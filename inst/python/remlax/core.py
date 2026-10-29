"""Kernel of the computation: V -> -2logL, with a closed-form gradient in V.

The kernel comes unchanged from the analysis code of the multispecies model of
Salas et al. (2026), where it was written and validated. Functions specific to
that project (assembly of a V with genetic blocks, residual covariance per
plant) were left out, because a generic solver does not need them.

The module defines a `custom_vjp` because the gradient of -2logL with respect
to V is known in closed form,

    d(-2logL)/dV = P - (Py)(Py)',      P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

The backward pass therefore needs neither the Cholesky band nor the triangular
solves that automatic differentiation would store. About ten n x n matrices
kept alive drop to three, which lets the solver run at n near 16 000 on a
10 GB GPU slice.

The rest of the package sees only a differentiable function of V. Any
construction of V (covariance structure, parametrised neighbourhood kernel)
thus gets its gradient with no extra code.

The backward pass also returns the gradient with respect to y,

    d(-2logL)/dy = 2 P y = 2 r

which costs nothing because the forward pass already computes r. It is needed
when the analysed response is a parametrised transform of the data,
y* = (I - M(Psi)) y, as in the structural spatial-lag model (lag.py). The
derivative of -2logL with respect to Psi then goes through y*. Without this
component automatic differentiation would see it as zero and return a wrong
gradient with no error.

`ml_from_V` is the same function without the term log|X'V^-1 X| and with the
constant N log 2pi. It is -2logL_ML profiled over beta. Its gradient in V is
V^-1 - r r' instead of P - r r'. Only the `ml` criterion of the structural
model uses it.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

LOG2PI = float(np.log(2.0 * np.pi))

# ==============================================================================
# Kernel: V -> -2logL, with a closed-form gradient in V
# ==============================================================================
@partial(jax.custom_vjp, nondiff_argnums=())
def reml_from_V(V, y, X):
    return _reml_fwd(V, y, X)[0]


def _reml_fwd(V, y, X, ml=False):
    N, p = X.shape
    L = jnp.linalg.cholesky(V)
    logdetV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    sol = lambda b: jax.scipy.linalg.solve_triangular(
        L.T, jax.scipy.linalg.solve_triangular(L, b, lower=True), lower=False)
    ViX = sol(X)
    Viy = sol(y)
    A = X.T @ ViX
    La = jnp.linalg.cholesky(A)
    logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(La)))
    XtViy = X.T @ Viy
    beta = jax.scipy.linalg.solve_triangular(
        La.T, jax.scipy.linalg.solve_triangular(La, XtViy, lower=True), lower=False)
    r = Viy - ViX @ beta                     # = P y
    yPy = y @ r
    if ml:
        # Profiled -2logL_ML: no log|X'V^-1X| and no p removed from the constant.
        val = N * LOG2PI + logdetV + yPy
    else:
        val = (N - p) * LOG2PI + logdetV + logdetA + yPy
    return val, (L, ViX, La, r)


def _fwd(V, y, X):
    val, res = _reml_fwd(V, y, X)
    return val, (res, X)


def _bwd(saved, g):
    return _bwd_shared(saved, g, ml=False)


def _bwd_shared(saved, g, ml=False):
    """P - (Py)(Py)', written to minimise the number of N x N matrices alive
    at the same time. On a 10 GB GPU slice each N x N matrix weighs about
    2 GB, and the order of operations decides whether the computation fits.

    A naive version forms Vinv = solve(L', solve(L, I)) and then
    W = Vinv - Z'Z - r r'. It keeps I, the intermediate result and Vinv alive
    together, and materialises Z'Z and r r' as well, up to seven N x N
    matrices.

    This version does a single triangular solve (Linv) and then forms
    Linv' Linv. The identity and Linv die early, and L is no longer needed.
    The subtractions are sequential, so XLA can reuse the buffer of W instead
    of materialising Z'Z and r r' beside it.
    """
    (L, ViX, La, r), X = saved
    N = L.shape[0]
    Z = jax.scipy.linalg.solve_triangular(La, ViX.T, lower=True)     # A^-1/2 ViX' (p x N)

    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(N), lower=True)
    W = Linv.T @ Linv                   # = V^-1
    del Linv
    if not ml:
        W = W - Z.T @ Z                 # = P
    W = W - jnp.outer(r, r)             # = P - (Py)(Py)'   (ML: V^-1 - r r')
    # Cotangent of y: 2 P y = 2 r (see the module docstring). It is returned
    # even when y is not differentiated, since it is a vector of size N
    # already computed.
    return (g * W, 2.0 * g * r, None)


reml_from_V.defvjp(_fwd, _bwd)


# ==============================================================================
# ML variant (`ml` criterion of the structural model)
# ==============================================================================
@partial(jax.custom_vjp, nondiff_argnums=())
def ml_from_V(V, y, X):
    return _reml_fwd(V, y, X, ml=True)[0]


def _fwd_ml(V, y, X):
    val, res = _reml_fwd(V, y, X, ml=True)
    return val, (res, X)


def _bwd_ml(saved, g):
    return _bwd_shared(saved, g, ml=True)


ml_from_V.defvjp(_fwd_ml, _bwd_ml)
