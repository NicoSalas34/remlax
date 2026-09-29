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
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

LOG2PI = float(np.log(2.0 * np.pi))

# Largeur des blocs de colonnes de la passe arriere.
#
# MESURE, 2026-08-26, tranche MIG 1g.10gb, N = 16211 :
#     REML_BWD_CHUNK=0     -> 7,34 Go
#     REML_BWD_CHUNK=2048  -> 8,11 Go   (PIRE)
# Le decoupage est CONTRE-PRODUCTIF ici : fori_loop transporte W dans son etat,
# donc la matrice complete reste vivante pendant toute la boucle EN PLUS des
# blocs ; le remplissage a un multiple de k l agrandit, et la retaille finale
# en cree une copie. On ajoute deux matrices pour en economiser une.
# Laisse a 0 ; conserve uniquement pour documenter la piste et eviter qu elle
# soit retentee.
BWD_CHUNK = int(os.environ.get("REML_BWD_CHUNK", "0"))



# ==============================================================================
# Coeur : V -> -2logL, avec gradient analytique par rapport a V
# ==============================================================================
@partial(jax.custom_vjp, nondiff_argnums=())
def reml_from_V(V, y, X):
    return _reml_fwd(V, y, X)[0]


def _reml_fwd(V, y, X):
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
    val = (N - p) * LOG2PI + logdetV + logdetA + yPy
    return val, (L, ViX, La, r)


def _fwd(V, y, X):
    val, res = _reml_fwd(V, y, X)
    return val, (res, X)


def _bwd(saved, g):
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

    if BWD_CHUNK <= 0 or BWD_CHUNK >= N:
        Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(N), lower=True)
        W = Linv.T @ Linv                   # = V^-1
        del Linv
        W = W - Z.T @ Z                     # = P
        W = W - jnp.outer(r, r)             # = P - (Py)(Py)'
        return (g * W, None, None)

    # ---- variante a memoire bornee -------------------------------------------
    # On n a jamais besoin de V^-1 en entier d un coup : sa k-ieme tranche de
    # colonnes est simplement la resolution de V x = E_blk. On ne materialise
    # donc ni l identite N x N (2,1 Go) ni Linv (2,1 Go) : seules L et W
    # restent vivantes, plus des blocs N x k.
    #
    #   V^-1[:, blk] = L^-T L^-1 E_blk        (deux resolutions triangulaires)
    #   W[:, blk]    = V^-1[:, blk] - Z' Z[:, blk] - r r[blk]'
    #
    # Le nombre d operations est INCHANGE (N^3 au total) ; on perd seulement
    # un peu de parallelisme par appel. C est le compromis voulu : un calcul
    # legerement plus long contre une empreinte qui tient dans une tranche.
    k = int(BWD_CHUNK)
    nchunk = (N + k - 1) // k
    rows = jnp.arange(N)

    # N n est pas forcement multiple de k : on travaille sur une largeur
    # arrondie au bloc superieur, puis on retaille.
    Npad = nchunk * k
    Wp = jnp.zeros((N, Npad), dtype=L.dtype)
    Zp = jnp.pad(Z, ((0, 0), (0, Npad - N)))
    rp = jnp.pad(r, (0, Npad - N))

    def body(c, W):
        start = c * k
        idx = start + jnp.arange(k)
        E = (rows[:, None] == idx[None, :]).astype(L.dtype)
        blk = jax.scipy.linalg.solve_triangular(
            L.T, jax.scipy.linalg.solve_triangular(L, E, lower=True), lower=False)
        blk = blk - Z.T @ jax.lax.dynamic_slice(Zp, (0, start), (Zp.shape[0], k))
        blk = blk - r[:, None] * jax.lax.dynamic_slice(rp, (start,), (k,))[None, :]
        return jax.lax.dynamic_update_slice(W, blk, (0, start))

    Wp = jax.lax.fori_loop(0, nchunk, body, Wp)
    return (g * Wp[:, :N], None, None)


reml_from_V.defvjp(_fwd, _bwd)

