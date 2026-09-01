"""Noyau de Matern pour un nu QUELCONQUE, en JAX et differentiable.

CE QU'ON CALCULE

    M(z, nu) = 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z),      M(0, nu) = 1

C'est la correlation de Matern, pas K_nu. La distinction n'est pas cosmetique :
z^nu K_nu(z) est REGULIER en z = 0 alors que K_nu(z) y diverge. Calculer K_nu
puis multiplier par z^nu ferait passer 0 * inf par la diagonale de toute
matrice de correlation.

POURQUOI PAS jax.scipy : K_nu n'y est pas (jax.scipy.special s'arrete a i0/i1).
Un rappel vers scipy (pure_callback) forcerait une synchronisation hote a chaque
evaluation ET n'aurait pas de derivee — or la portee et nu sont justement les
parametres qu'on veut estimer.

METHODE : QUADRATURE DE LA REPRESENTATION INTEGRALE

    K_nu(z) = integrale_0^inf exp(-z cosh u) cosh(nu u) du

evaluee par la regle du TRAPEZE sur une grille fixe. Trois raisons de la
preferer aux series :

  1. AUCUNE SINGULARITE EN nu ENTIER. Les developpements usuels passent par
     K_nu = pi/2 (I_-nu - I_nu)/sin(nu pi), qui explose des que nu est entier et
     perd, meme a cote, autant de chiffres que le rapport des deux termes.
     MESURE sur la version en series : erreur absolue 3.5e+06 a nu = 3, et deja
     4e-08 a nu = 3.7. L'integrale, elle, est analytique en nu.
  2. CONVERGENCE SPECTRALE. Pour cet integrande, l'erreur du trapeze de pas h
     se comporte en exp(-pi^2/h) (les coefficients de Poisson valent K_{i xi}(z),
     qui decroit en exp(-pi xi/2)). A h = 0.15 cela fait exp(-66), soit trois
     ordres de grandeur sous la precision machine : la grille est fixe, sans
     controle d'erreur a faire.
  3. UN SEUL CHEMIN DE CODE, donc une seule chose a verifier, et un noyau qui se
     reduit a une somme ponderee — ce qu'un GPU fait bien.

CALCUL EN LOG. On n'evalue jamais K_nu seul : a z petit il vaut
Gamma(nu) 2^(nu-1) z^-nu, soit 1e+41 pour nu = 5 et z = 1e-8, alors que le
produit z^nu K_nu reste d'ordre 1. Les deux exponentielles sont donc formees
avec nu*log(z) DEJA DANS L'EXPOSANT.

DOMAINE. z est borne en bas a 1e-12 : au-dela le pic de l'integrande
(u* = log(2 nu/z)) sortirait de la grille et la troncature mordrait. A z = 1e-12
la correlation vaut deja 1 a 1e-12 pres pour tout nu >= 0.5.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
import jax.numpy as jnp
from jax.scipy.special import gammaln

U_MAX = 30.0     # borne superieure de la quadrature
H_PAS = 0.15     # pas du trapeze : erreur ~ exp(-pi^2/h) = exp(-66)
Z_MIN = 1e-12    # plancher sur z (cf. en-tete)

_U = jnp.arange(0.0, U_MAX + 0.5 * H_PAS, H_PAS)
_W = jnp.where(jnp.arange(_U.shape[0]) == 0, 0.5 * H_PAS, H_PAS)  # trapeze


def z_nu_kv(z, nu):
    """z^nu K_nu(z), regulier en z = 0 (y vaut Gamma(nu) 2^(nu-1) z^0... non :
    tend vers Gamma(nu) 2^(nu-1)). Calcule en log pour ne jamais former K_nu."""
    z = jnp.clip(jnp.asarray(z, dtype=jnp.float64), Z_MIN, None)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    lz = jnp.log(z)
    u = _U.reshape((-1,) + (1,) * jnp.ndim(z))          # diffusion sur z
    ch = jnp.cosh(u)
    base = nu * lz - z[None, ...] * ch                  # nu log z - z cosh u
    # cosh(nu u) = (e^{nu u} + e^{-nu u})/2, chaque moitie gardee dans l'exposant
    e_p = jnp.exp(base + nu * u)
    e_m = jnp.exp(base - nu * u)
    w = _W.reshape(u.shape[:1] + (1,) * jnp.ndim(z))
    return 0.5 * jnp.sum(w * (e_p + e_m), axis=0)


def matern(z, nu):
    """Correlation de Matern M(z, nu) ; z >= 0, nu > 0 ; M(0, nu) = 1."""
    z = jnp.asarray(z, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    v = jnp.exp((1.0 - nu) * jnp.log(2.0) - gammaln(nu)) * z_nu_kv(z, nu)
    return jnp.where(z <= 1e-13, jnp.ones_like(v), v)
