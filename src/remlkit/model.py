"""Assemblage de V et vraisemblance REML — generique.

MODELE
    y = X beta + sum_k Z_k u_k + e
    u_k ~ N(0, Sigma_k (x) K_k)          Sigma_k : t_k x t_k, K_k : q_k x q_k
    e   ~ N(0, R)

    V = sum_k Z_k (Sigma_k (x) K_k) Z_k' + R

ASSEMBLAGE PAR FACTEUR, ET POURQUOI
    On n'assemble jamais Sigma (x) K. On forme B_k = Z_k (L_Sigma (x) L_K) puis
    V += B_k B_k'. Trois raisons :
      - la positivite de V est garantie par CONSTRUCTION, meme quand une
        variance part a zero ou que l'optimiseur s'egare loin de l'optimum ;
      - un seul GEMM au lieu d'un produit de Kronecker explicite de taille
        (t q)^2, qu'on ne pourrait pas stocker des que t q depasse quelques
        milliers ;
      - le gradient passe par le meme chemin, donc aucune formule a deriver a
        la main.

    ORDRE DES COLONNES DE Z : colonne = (col_trait - 1) * q + niveau, c'est-a-dire
    NIVEAU le plus rapide. Sigma (x) K suit le meme ordre (Sigma sur l'index lent).
    R/remlkit.R construit Z avec cette convention ; en changer d'un cote sans
    l'autre donnerait un modele different sans aucune erreur visible.

VRAISEMBLANCE
    -2 logL_REML = log|V| + log|X'V^-1 X| + y' P y  (+ constante)
    Le noyau reml_from_V de remlkit/core.py porte le gradient
    analytique par rapport a V (d(-2logL)/dV = P - (Py)(Py)'), qui est agnostique
    a la facon dont V a ete batie : toute parametrisation lisse de V est donc
    differentiee gratuitement.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .core import reml_from_V
except ImportError:
    from core import reml_from_V  # noqa: F401  (import direct, hors paquet)

try:
    from .structures import chol_sigma, n_params
    from .levels import level_chol, n_level_params, level_corr
except ImportError:
    from structures import chol_sigma, n_params
    from levels import level_chol, n_level_params, level_corr


def term_n_params(tm):
    """Parametres d'un terme : ceux de Sigma, PUIS ceux de la structure de niveaux.

    L'ordre compte : R et Python doivent decouper theta au meme endroit.
    """
    kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    return (n_params(tm["struct"], tm["t"], tm["rank"])
            + n_level_params(kind, tm.get("lvl_order", 0), tm.get("lvl_parts"),
                             opts=tm.get("lvl_opts")))


# ==============================================================================
# Incidences creuses
# ==============================================================================
def dense_Z(term, n):
    """Z dense (n x t*q) depuis le triplet COO.

    Dense et pas creux : les tailles visees (n <= ~2e4, t*q <= ~3e3) tiennent
    largement en memoire, et un produit dense sur GPU bat un produit creux d'un
    facteur qui rend le choix evident. Pour des dispositifs beaucoup plus grands
    il faudrait revenir a un format creux ; la fonction est le seul endroit a
    changer.
    """
    Z = jnp.zeros((n, term["t"] * term["q"]))
    return Z.at[term["zi"], term["zj"]].add(term["zx"])


def term_factor(theta_k, term, Z):
    """B_k = Z_k (L_Sigma (x) L_K), tel que Z_k (Sigma (x) K) Z_k' = B_k B_k'.

    L_K peut dependre de parametres (structure AR1) : il est alors reconstruit
    ici a chaque evaluation, et differentie comme le reste.
    """
    t, q = term["t"], term["q"]
    n_sig = n_params(term["struct"], t, term["rank"])
    Ls = chol_sigma(theta_k[:n_sig], term["struct"], t, term["rank"])   # t x m
    m = Ls.shape[1]
    # DEFAUT SUR : une K fournie sans cle "lvl" explicite signifie "fixed".
    # Faire retomber sur "id" ferait IGNORER la parente en silence — c'est
    # exactement la panne qu'a connue le portage GPU du modele IGE, ou deux runs
    # etiquetes "kinship" etaient des doublons exacts des runs sans.
    kind = term.get("lvl") or ("fixed" if term.get("LK") is not None else "id")
    if kind == "fixed":
        LK = term["LK"]
    else:
        LK = level_chol(theta_k[n_sig:], kind, q, dims=term.get("dims"),
                        LK_fixed=term.get("LK"), order=term.get("lvl_order", 0),
                        coord=term.get("coord"), parts=term.get("lvl_parts"),
                        opts=term.get("lvl_opts"), expr=term.get("lvl_expr"))
    if LK is None:
        # K = I : (L_Sigma (x) I) agit bloc par bloc, pas de produit a former.
        Zb = Z.reshape(Z.shape[0], t, q)                 # (n, trait, niveau)
        B = jnp.einsum("ntq,tm->nmq", Zb, Ls).reshape(Z.shape[0], m * q)
    else:
        LKj = jnp.asarray(LK)
        Zb = Z.reshape(Z.shape[0], t, q)
        ZL = jnp.einsum("ntq,qp->ntp", Zb, LKj)          # Z (I (x) L_K)
        B = jnp.einsum("ntp,tm->nmp", ZL, Ls).reshape(Z.shape[0], m * q)
    return B


# ==============================================================================
# Residuelle
# ==============================================================================
def res_sections(res, n):
    """Sections de la residuelle, TOUJOURS sous forme de liste.

    Une residuelle ordinaire est une section unique couvrant toutes les lignes.
    Rendre la meme forme dans les deux cas evite le double chemin de code qui,
    sinon, se met a diverger des qu'on touche a l'un des deux.
    """
    secs = res.get("sections")
    if secs:
        return secs
    s = dict(res)
    s["rows"] = np.arange(n, dtype=np.int64)
    return [s]


def sec_n_params(sec):
    return (n_params(sec["struct"], sec["t"], sec["rank"])
            + n_level_params(sec.get("lvl", "id"), sec.get("lvl_order", 0),
                             sec.get("lvl_parts"), opts=sec.get("lvl_opts")))


def _section_V(theta_s, sec):
    """R d'UNE section : Sigma_caractere x C_unite, produit terme a terme.

        R[i,j] = Sigma_t[trait(i), trait(j)] * C_u[unit(i), unit(j)]

    UNE seule formule couvre tout ce qu'asreml ecrit avec `residual = ~ ...` :
      ~ units                  Sigma = sigma^2,   C = I    -> R = sigma^2 I
      ~ us(trait):units        Sigma = us,        C = I    -> couplage entre
                                                              caracteres d'une
                                                              meme unite
      ~ ar1(row):ar1(col)      Sigma = sigma^2,   C = AR1 x AR1
      ~ us(trait):ar1(row):ar1(col)   les deux a la fois
    C'est la meme algebre que pour un terme aleatoire ; la seule difference est
    que l'incidence est l'identite, donc il n'y a rien a multiplier.

    Le cas `us` couple les observations d'une MEME unite mesuree sur plusieurs
    traits. C'est la structure qui, sur le modele IGE, vaut 2228 points de LRT
    pour 22 parametres : la negliger n'est pas une approximation innocente.
    """
    st, t = sec["struct"], sec["t"]
    kind = sec.get("lvl", "id")
    tr = jnp.asarray(sec["trait"])
    un = jnp.asarray(sec["unit"])
    n_sig = n_params(st, t, sec["rank"])

    # --- partie caractere -----------------------------------------------------
    if st == "iid":
        S = jnp.exp(2.0 * theta_s[0]) * jnp.eye(t)
    elif st == "diag":
        S = jnp.diag(jnp.exp(2.0 * theta_s[:t]))
    elif st in ("us", "fa"):
        Ls = chol_sigma(theta_s[:n_sig], st, t, sec["rank"])
        S = Ls @ Ls.T
    else:
        raise ValueError("structure residuelle inconnue : %s" % st)
    St = S[tr[:, None], tr[None, :]]

    # --- partie unite ---------------------------------------------------------
    if kind == "id":
        # Sans structure entre unites, deux observations ne sont correlees que
        # si elles portent sur la MEME unite (caracteres differents).
        Cu = (un[:, None] == un[None, :]).astype(St.dtype)
    else:
        n_u = int(sec.get("n_unit", 0)) or int(np.max(np.asarray(sec["unit"])) + 1)
        C = level_corr(theta_s[n_sig:], kind, n_u, order=sec.get("lvl_order", 0),
                       coord=sec.get("coord"), dims=sec.get("dims"),
                       opts=sec.get("lvl_opts"), expr=sec.get("lvl_expr"))
        Cu = C[un[:, None], un[None, :]]
    return St * Cu


def residual_V(theta_r, res, n):
    """R complete : somme directe des sections (le dsum d'asreml).

    Les sections PARTITIONNENT les observations. R est donc bloc-diagonale a une
    permutation pres, et chaque section porte SES PROPRES parametres : c'est ce
    qui permet a deux sites, deux annees ou deux essais d'avoir des residuelles
    de formes differentes dans un seul ajustement.

    On assemble par diffusion dans une matrice n x n plutot que par
    concatenation : les lignes d'une section n'ont aucune raison d'etre
    contigues dans les donnees, et un tri implicite ferait correspondre les
    mauvaises lignes de y sans rien signaler.
    """
    secs = res_sections(res, n)
    if len(secs) == 1 and "rows" not in res and res.get("sections") is None:
        return _section_V(theta_r, secs[0])
    R = jnp.zeros((n, n))
    o = 0
    for sec in secs:
        p = sec_n_params(sec)
        idx = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        R = R.at[idx[:, None], idx[None, :]].add(_section_V(theta_r[o:o + p], sec))
        o += p
    return R


# ==============================================================================
# V complet et objectif
# ==============================================================================
def split_theta(theta, terms, res):
    """Decoupe le vecteur global selon l'ordre des termes, puis la residuelle."""
    out, o = [], 0
    for tm in terms:
        p = term_n_params(tm)
        out.append(theta[o:o + p]); o += p
    pr = res_n_params(res)
    return out, theta[o:o + pr], o + pr


def assemble_V(theta, terms, Zs, res, n):
    V = residual_V(split_theta(theta, terms, res)[1], res, n)
    th_terms, _, _ = split_theta(theta, terms, res)
    for k, tm in enumerate(terms):
        B = term_factor(th_terms[k], tm, Zs[k])
        V = V + B @ B.T
    return V


def neg2_reml(theta, terms, Zs, res, y, X):
    V = assemble_V(theta, terms, Zs, res, y.shape[0])
    return reml_from_V(V, y, X)


def res_n_params(res):
    return sum(sec_n_params(s) for s in res_sections(res, 0))


def n_theta(terms, res):
    return sum(term_n_params(t) for t in terms) + res_n_params(res)


def make_objective(bundle_terms, res, y, X, scale=None):
    """Rend (fn, grad_fn) sur theta, mis a l'echelle.

    MISE A L'ECHELLE PAR n : -2logL est d'ordre n et son gradient aussi. Or
    L-BFGS-B demarre avec un hessien approche egal a l'identite : son premier
    pas vaut -gradient, soit des centaines d'unites dans l'espace des parametres
    si l'objectif n'est pas normalise. Tous les parametres se retrouvent alors
    projetes sur les bornes des la deuxieme evaluation et V devient singuliere.
    Le facteur est retire des valeurs rendues a l'appelant.
    """
    n = y.shape[0]
    sc = float(n) if scale is None else float(scale)
    yj, Xj = jnp.asarray(y), jnp.asarray(X)
    Zs = [dense_Z(t, n) for t in bundle_terms]

    def f(theta):
        return neg2_reml(jnp.asarray(theta), bundle_terms, Zs, res, yj, Xj) / sc

    fg = jax.jit(jax.value_and_grad(f))

    def fun_jac_scaled(theta):
        """CE QUE VOIT L'OPTIMISEUR : objectif et gradient DIVISES par n.

        Rendre la valeur remultipliee annulerait la mise a l'echelle, qui est
        tout l'interet : L-BFGS-B demarre avec un hessien approche egal a
        l'identite, donc son premier pas vaut -gradient. Sur -2logL brut le
        gradient vaut des dizaines a des centaines, le premier pas envoie les
        parametres a l'autre bout de l'espace, la recherche lineaire echoue et
        l'optimiseur rend un point tres loin du sommet en annoncant un succes.
        MESURE sur un modele a un facteur : sans mise a l'echelle effective,
        arret a -2logL = 425.15 avec |grad| = 74.6, contre 414.3095 a l'optimum.
        """
        v, g = fg(jnp.asarray(theta))
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        """Echelle NATURELLE (-2logL), pour le diagnostic et les sorties."""
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc

    return fun_jac_scaled, fun_jac, sc, Zs
