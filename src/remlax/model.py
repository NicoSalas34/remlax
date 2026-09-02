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
    R/remlax.R construit Z avec cette convention ; en changer d'un cote sans
    l'autre donnerait un modele different sans aucune erreur visible.

VRAISEMBLANCE
    -2 logL_REML = log|V| + log|X'V^-1 X| + y' P y  (+ constante)
    Le noyau reml_from_V de remlax/core.py porte le gradient
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

from functools import partial

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


def term_support(term, n):
    """Les lignes ou l'incidence du terme est non nulle, triees.

    POURQUOI CELA VAUT LA PEINE. Un terme du modele IGE ne concerne qu'un bloc
    et un caractere : son incidence est nulle sur toutes les autres lignes. Le
    facteur B_k stocke pourtant ces zeros en flottants doubles sur toute la
    hauteur n. MESURE sur le modele a 5 caracteres (n = 8987) : un terme d'IEE
    porte q = 1920 colonnes pour ~1800 lignes utiles, soit 131,6 Mo dont 80 %
    de zeros ; les dix termes d'IEE font 1316 Mo, 82 % de la memoire des
    facteurs.

    Le support est CONSTANT — il ne depend pas de theta — donc il se calcule
    une fois, hors de la fonction derivee.
    """
    return np.unique(np.asarray(term["zi"], dtype=np.int64))


def restrict_Z(term, rows, n):
    """Z du terme, restreint aux lignes `rows`.

    On renumerote zi dans le repere du support. Un indice absent du support
    serait une incoherence entre le support et l'incidence, donc on l'assertit
    plutot que de le laisser produire une matrice silencieusement fausse.
    """
    pos = np.full(n, -1, dtype=np.int64)
    pos[rows] = np.arange(len(rows), dtype=np.int64)
    zi = pos[np.asarray(term["zi"], dtype=np.int64)]
    assert zi.min() >= 0, "une entree de l'incidence tombe hors du support"
    Z = jnp.zeros((len(rows), term["t"] * term["q"]))
    return Z.at[zi, np.asarray(term["zj"], dtype=np.int64)].add(
        jnp.asarray(term["zx"]))


def support_groups(terms, n):
    """Regroupe les termes par support IDENTIQUE.

    La concaternation exige des hauteurs egales, donc seuls des termes partageant
    exactement le meme support peuvent entrer dans un meme produit. Le
    regroupement est naturel dans ce modele : les cinq termes spatiaux d'un
    caractere, son terme d'IEE et son terme d'IEE croise portent tous les memes
    lignes.

    ET IL NE PEUT PAS ETRE PLUS AGRESSIF. Deux termes a supports DISJOINTS ne
    peuvent pas partager de colonnes : (B_1 + B_2)(B_1 + B_2)' fait apparaitre
    les termes croises B_1 B_2', non nuls sur S_1 x S_2. Fusionner des supports
    disjoints donnerait donc un V faux.
    """
    par_cle, ordre = {}, []
    for k, tm in enumerate(terms):
        rows = term_support(tm, n)
        cle = rows.tobytes()
        if cle not in par_cle:
            par_cle[cle] = (rows, [])
            ordre.append(cle)
        par_cle[cle][1].append(k)
    return [par_cle[c] for c in ordre]


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


# ==============================================================================
# L'ASSEMBLAGE DE V TIENT EN UN SEUL PRODUIT
# ==============================================================================
# POURQUOI, ET LE CHIFFRE QUI L'A EXIGE. La chaine naive
#
#     V = R ;  pour chaque terme :  V = V + B_k B_k'
#
# cree une matrice n x n NEUVE a chaque tour, et XLA n'en reutilise pas les
# tampons : le pic mesure vaut un tampon PAR TERME. Sur le modele multivarie
# complet d'IGE_analysis — 65 termes a n = 16211 — cela fait 65 x 1,96 = 127 Go,
# et le job a echoue en demandant 134 Gio sur une A100 de 80 Go.
#
# L'IDENTITE QUI RESOUT. La somme des contributions est un seul produit :
#
#     Sum_k B_k B_k' = [B_1 ... B_K] [B_1 ... B_K]'
#
# Exact, et il n'y a plus qu'UN intermediaire n x n. Le cout est la matrice
# concatenee, de largeur Sum_k m_k q_k : sur le modele reel, 16211 x 42186, soit
# 5,1 Go contre 2,0 pour V. Mesure sur un modele a 41 termes : le pic passe de
# 126,1 a 16,2 Mo, soit un facteur 7,8, a valeur et gradient identiques.
#
# ET IL N'Y A PAS DE REGLE DE DERIVATION A ECRIRE. Une premiere tentative avait
# donne a l'assemblage sa propre regle, sur l'idee que la passe ARRIERE gardait
# les intermediaires. Elle etait correcte — valeur et gradient identiques a la
# chaine, y compris sous jit — et parfaitement inutile : le pic est dans la passe
# AVANT, et la regle ne l'a pas bouge d'un pour cent. Avec un seul produit, la
# derivation automatique rend deja dB = 2 dV B, ce qui est optimal.
# ==============================================================================
def assemble_V(theta, terms, Zs, res, n):
    th_terms, th_res, _ = split_theta(theta, terms, res)
    R = residual_V(th_res, res, n)
    if not terms:
        return R
    B = jnp.concatenate([term_factor(th_terms[k], tm, Zs[k])
                         for k, tm in enumerate(terms)], axis=1)
    return R + B @ B.T


def assemble_V_groupes(theta, terms, groupes, Zr, res, n):
    """V assemblee par SOUS-BLOCS, chaque terme sur son seul support.

    CE QUE CETTE FORME EVITE, ET QUI A ETE MESURE.

    1. Les zeros stockes. Un facteur pleine hauteur porte n lignes la ou le
       terme n'en concerne qu'une fraction. Sur le modele a 5 caracteres :
       1598 Mo de facteurs contre ~400 Mo restreints, soit un facteur 4.

    2. La residuelle densifiee. residual_V() formait un n x n dense par une
       chaine de mises a jour fonctionnelles — une par section, 47 sur le jeu
       complet — puis assemble_V l'additionnait au produit, allouant un
       TROISIEME n x n. C'est le meme defaut que la chaine d'additions par
       terme corrigee auparavant, transpose dans une autre fonction. Ici les
       sections sont dispersees DIRECTEMENT dans V.

    CE QU'ELLE NE CHANGE PAS : l'algebre. V est la meme matrice. Mais l'ORDRE
    des sommations change, et l'addition flottante n'est pas associative : les
    tests comparent donc a des tolerances RELATIVES, jamais a l'egalite exacte.
    """
    th_terms, th_res, _ = split_theta(theta, terms, res)
    V = jnp.zeros((n, n))

    # --- residuelle : dispersion directe, sans n x n intermediaire ----------
    o = 0
    for sec in res_sections(res, n):
        pnb = sec_n_params(sec)
        idx = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        V = V.at[idx[:, None], idx[None, :]].add(_section_V(th_res[o:o + pnb], sec))
        o += pnb

    # --- termes, un produit par support ------------------------------------
    for rows, ks in groupes:
        B = jnp.concatenate([term_factor(th_terms[k], terms[k], Zr[k]) for k in ks],
                            axis=1)
        idx = jnp.asarray(rows)
        V = V.at[idx[:, None], idx[None, :]].add(B @ B.T)
    return V


def neg2_reml(theta, terms, Zs, res, y, X):
    V = assemble_V(theta, terms, Zs, res, y.shape[0])
    return reml_from_V(V, y, X)


def neg2_reml_groupes(theta, terms, groupes, Zr, res, y, X):
    V = assemble_V_groupes(theta, terms, groupes, Zr, res, y.shape[0])
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
    # SUPPORTS CALCULES UNE FOIS, hors de la fonction derivee : ils ne
    # dependent pas de theta. Zr[k] est le facteur du terme k restreint aux
    # lignes de son groupe.
    groupes = support_groups(bundle_terms, n)
    Zr = [None] * len(bundle_terms)
    for rows, ks in groupes:
        for k in ks:
            Zr[k] = restrict_Z(bundle_terms[k], rows, n)
    # Zs pleine hauteur : conserve pour le test d'identite contre l'ancienne
    # forme, et pour le chemin de repli si un jour un terme couvrait tout n.
    Zs = [dense_Z(t, n) for t in bundle_terms]

    # LES INCIDENCES RESTENT CAPTUREES, ET C'EST MESURE. Les passer en argument
    # de la fonction compilee ne gagne rien : tampon 4580,3 Mo contre 4548,7 et
    # 160,8 Mo d'arguments en plus, sur un dispositif de la forme du modele IGE.
    # L'avertissement "2,90 GB of constants were captured" que j'avais lu venait
    # des facteurs PLEINE HAUTEUR (2769 Mo) captures par la forme de reference
    # compilee A COTE pour le test d'identite — pas de cet objectif, dont les
    # incidences restreintes ne pesent que 161 Mo.
    def f(theta):
        return neg2_reml_groupes(jnp.asarray(theta), bundle_terms, groupes, Zr,
                                 res, yj, Xj) / sc

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
