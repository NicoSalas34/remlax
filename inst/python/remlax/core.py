"""Coeur du calcul : V -> -2logL, avec gradient analytique par rapport a V.

Extrait tel quel du code d'analyse du modele multi-especes de Salas et al. (2026),
ou ce noyau a ete ecrit, mesure et valide. Seules les fonctions specifiques a ce
projet (assemblage d'un V a blocs genetiques, covariance residuelle par plante)
ont ete laissees de cote : elles ne concernent pas un solveur generique.

POURQUOI UN `custom_vjp`. Le gradient de -2logL par rapport a V est connu
analytiquement,

    d(-2logL)/dV = P - (Py)(Py)',      P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

donc la passe arriere n'a besoin ni de la bande de la Cholesky ni des
resolutions triangulaires memorisees par la differentiation automatique : on
passe d'une dizaine de matrices n x n conservees a trois. C'est ce qui rend le
solveur utilisable a n de l'ordre de 16 000 sur une tranche de GPU de 10 Go.

Le reste du paquet ne voit qu'une fonction differentiable de V : toute
construction de V — structure de covariance, noyau de voisinage parametre —
recoit donc son gradient sans code supplementaire.

GRADIENT PAR RAPPORT A y. La passe arriere rend aussi

    d(-2logL)/dy = 2 P y = 2 r

qui ne coute rien (r est deja calcule par la passe avant). Il etait inutile tant
que y etait une donnee ; il devient necessaire des que la reponse analysee est
une TRANSFORMEE parametree des donnees, y* = (I - M(Psi)) y, comme dans le
modele structurel a decalage spatial (lag.py) : la derivee de -2logL par rapport
a Psi passe par y*, et sans cette composante la differentiation automatique
l'aurait vue nulle — un gradient faux, sans erreur.

ML. `ml_from_V` est la meme fonction sans le terme log|X'V^-1 X| (et avec la
constante N log 2pi) : -2logL_ML profilee en beta. Son gradient par rapport a V
est V^-1 - r r' au lieu de P - r r'. Elle ne sert qu'au critere `ml` du modele
structurel.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

LOG2PI = float(np.log(2.0 * np.pi))

# ==============================================================================
# Coeur : V -> -2logL, avec gradient analytique par rapport a V
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
        # -2logL_ML profilee : ni log|X'V^-1X| ni retrait de p dans la constante.
        val = N * LOG2PI + logdetV + yPy
    else:
        val = (N - p) * LOG2PI + logdetV + logdetA + yPy
    return val, (L, ViX, La, r)


def _fwd(V, y, X):
    val, res = _reml_fwd(V, y, X)
    return val, (res, X)


def _bwd(saved, g):
    return _bwd_commun(saved, g, ml=False)


def _bwd_commun(saved, g, ml=False):
    """P - (Py)(Py)', ecrit pour minimiser le nombre de matrices N x N VIVANTES
    simultanement. Sur une A100 entiere (80 Go) la formulation naive passe sans
    qu'on y pense ; sur un huitieme de carte (10 Go) chaque matrice N x N pese
    2,1 Go et l'ordre des operations decide si le calcul tient ou non.

    Version precedente :
        Vinv = solve(L', solve(L, I))        <- I, le resultat intermediaire ET
                                                 Vinv vivants en meme temps
        W    = Vinv - Z'Z - r r'             <- Z'Z et r r' materialises en plus
    soit jusqu'a sept matrices N x N.

    Ici :
      - une seule resolution triangulaire (Linv), puis Linv' Linv : l'identite
        et Linv meurent tot, et L n'est plus necessaire ensuite ;
      - les retraits sont sequentiels, donc XLA peut reutiliser le tampon de W
        au lieu de materialiser Z'Z et r r' a cote.
    """
    (L, ViX, La, r), X = saved
    N = L.shape[0]
    Z = jax.scipy.linalg.solve_triangular(La, ViX.T, lower=True)     # A^-1/2 ViX' (p x N)

    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(N), lower=True)
    W = Linv.T @ Linv                   # = V^-1
    del Linv
    if not ml:
        W = W - Z.T @ Z                 # = P
    W = W - jnp.outer(r, r)             # = P - (Py)(Py)'   (ML : V^-1 - r r')
    # Cotangente de y : 2 P y = 2 r (cf. en-tete). Rendue meme quand y
    # n'est pas differentie : un vecteur de taille N, deja calcule.
    return (g * W, 2.0 * g * r, None)


reml_from_V.defvjp(_fwd, _bwd)


# ==============================================================================
# Variante ML (critere `ml` du modele structurel)
# ==============================================================================
@partial(jax.custom_vjp, nondiff_argnums=())
def ml_from_V(V, y, X):
    return _reml_fwd(V, y, X, ml=True)[0]


def _fwd_ml(V, y, X):
    val, res = _reml_fwd(V, y, X, ml=True)
    return val, (res, X)


def _bwd_ml(saved, g):
    return _bwd_commun(saved, g, ml=True)


ml_from_V.defvjp(_fwd_ml, _bwd_ml)

