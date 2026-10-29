"""Matern kernel for any nu, in JAX and differentiable.

The module computes the Matern correlation

    M(z, nu) = 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z),      M(0, nu) = 1

and never K_nu alone. z^nu K_nu(z) is regular at z = 0, whereas K_nu(z)
diverges there. Computing K_nu and then multiplying by z^nu would put 0 * inf
on the diagonal of every correlation matrix.

jax.scipy.special has no K_nu (it stops at i0/i1). A callback to scipy
(pure_callback) would force a host synchronisation at each evaluation and would
have no derivative. The range and nu are precisely the parameters to estimate.

Method: quadrature of the integral representation

    K_nu(z) = integral_0^inf exp(-z cosh u) cosh(nu u) du

evaluated by the trapezoidal rule on a fixed grid. The integral has no
singularity at integer nu. The usual series go through
K_nu = pi/2 (I_-nu - I_nu)/sin(nu pi), which blows up at integer nu and loses
digits near it (dev/notes.md). The integral is analytic in nu.

The trapezoidal rule converges spectrally on this integrand. With step h the
error behaves as exp(-pi^2/h), because the Poisson coefficients equal
K_{i xi}(z), which decays as exp(-pi xi/2). At h = 0.15 this gives
exp(-65.8) = 2.7e-29, far below eps = 2.2e-16. The grid can therefore stay
fixed, with no error control. A single code path also leaves one thing to
check, and the kernel reduces to a weighted sum, which a GPU does well.

The computation runs in logs. At small z, K_nu is close to
Gamma(nu) 2^(nu-1) z^-nu, which is 1e+41 for nu = 5 and z = 1e-8, while
z^nu K_nu stays of order 1. Both exponentials are therefore formed with
nu*log(z) already in the exponent.

z is bounded below at 1e-12. Below that, the peak of the integrand
(u* = log(2 nu/z)) would leave the grid and truncation would bite. At
z = 1e-12 the correlation already equals 1 to within 1e-12 for every nu >= 0.5.
"""
from . import _x64  # noqa: F401
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import gammaln

U_MAX = 30.0     # upper bound of the quadrature
H_PAS = 0.15     # trapezoidal step: error ~ exp(-pi^2/h) = exp(-66)
Z_MIN = 1e-12    # floor on z (see the module docstring)

# Nodes and weights are NumPy arrays. A JAX array created at import would
# initialise the XLA backend when the module loads, before the solver has
# chosen its device.
_U = np.arange(0.0, U_MAX + 0.5 * H_PAS, H_PAS)
_W = np.where(np.arange(_U.shape[0]) == 0, 0.5 * H_PAS, H_PAS)  # trapezoid


def z_nu_kv(z, nu):
    """z^nu K_nu(z), regular at z = 0, where it tends to Gamma(nu) 2^(nu-1).
    Computed in logs so that K_nu is never formed."""
    z = jnp.clip(jnp.asarray(z, dtype=jnp.float64), Z_MIN, None)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    lz = jnp.log(z)
    u = _U.reshape((-1,) + (1,) * jnp.ndim(z))          # broadcast over z
    ch = jnp.cosh(u)
    base = nu * lz - z[None, ...] * ch                  # nu log z - z cosh u
    # cosh(nu u) = (e^{nu u} + e^{-nu u})/2, each half kept in the exponent
    e_p = jnp.exp(base + nu * u)
    e_m = jnp.exp(base - nu * u)
    w = _W.reshape(u.shape[:1] + (1,) * jnp.ndim(z))
    return 0.5 * jnp.sum(w * (e_p + e_m), axis=0)


def matern(z, nu):
    """Matern correlation M(z, nu), with z >= 0, nu > 0 and M(0, nu) = 1."""
    z = jnp.asarray(z, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    v = jnp.exp((1.0 - nu) * jnp.log(2.0) - gammaln(nu)) * z_nu_kv(z, nu)
    return jnp.where(z <= 1e-13, jnp.ones_like(v), v)
