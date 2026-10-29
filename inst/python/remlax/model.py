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
from . import _x64  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from .core import reml_from_V, ml_from_V
from .lag import (prepare_lag, lag_n_params, ystar as lag_ystar, lag_terms,
                  PENALITE as LAG_PENALITE)

from .structures import chol_sigma, n_params
from .levels import level_chol, n_level_params, level_corr, _acf, _toeplitz


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


def orthonormal_X(X):
    """X = Q R (QR reduite) : rend Q et 2 log|det R|.

    La REML ne depend de X que par son espace engendre, plus un terme constant :
        log|X'V^-1X| = log|Q'V^-1Q| + 2 log|det R|.
    POURQUOI. Une colonne brute comme une date de semis (~20 000, presque
    colineaire a la constante) porte X'V^-1X a 1e12 : log|X'V^-1X| y perd des
    chiffres. MESURE sur le modele IGE ble : -2logL faux de 4e-7 a l'interieur,
    de 1,4e-4 sur un sous-modele. Avec Q, le calcul est exact et la constante est
    rajoutee telle quelle : la valeur rendue reste -2logL du modele en X.
    Seules les fonctions de vraisemblance utilisent Q ; beta, Wald et
    predictions gardent X.
    """
    Xn = np.asarray(X, dtype=np.float64)
    if Xn.shape[1] == 0:
        return Xn, 0.0
    Q, R = np.linalg.qr(Xn)
    return Q, float(2.0 * np.sum(np.log(np.abs(np.diag(R)))))


def prefold_fixed_K(terms, Zr):
    """Plie une parente FIXE dans l'incidence : Z <- Z (I (x) L_K).

    POURQUOI. Une K fournie ne depend d'aucun parametre, donc Z (I (x) L_K) non
    plus. Laisse dans la fonction compilee, ce produit porte sur deux CONSTANTES
    capturees : XLA le calcule pendant la compilation ("constant folding"), dans
    son interpreteur, en n x q x q operations. MESURE : 28 s de compilation pour
    n = 3000, q = 1500 ; 633 s a n = 8000 sur MIG, et l'echec de compilation du
    modele ige a n = 8037. Forme une fois sur l'hote en NumPy, le produit sort
    du programme compile.

    CE QUI NE CHANGE PAS. V = Z (Sigma (x) K) Z' = ZL (Sigma (x) I) ZL', et ses
    derivees par rapport a theta sont les memes : le terme plie est le meme
    terme, decrit comme un terme sans parente. Seules les fonctions de
    vraisemblance le voient ; les BLUP et les sorties gardent le terme d'origine.

    COUT. Le produit se fait en creux (Z n'a qu'une poignee de non-zeros par
    ligne) : l'einsum dense precedent coutait n x t x q^2 operations hors
    BLAS, 66 s pour n = 1000, q = 8000 (genotypes a predire), et il etait fait
    deux fois, pour la vraisemblance et pour l'AI. Le resultat est garde dans
    le terme d'origine pour le second appel.

    COMPRESSION EXACTE. V ne depend de Z_a L_K (a = composante) que par les
    produits (Z_a L_K)(Z_b L_K)'. Quand les lignes chargees, toutes
    composantes empilees, sont moins nombreuses que q (genotypes non
    phenotypes dans K), on remplace L_K par une base de l'espace de ces
    lignes : M' = Q R (QR reduite de M', M = les lignes empilees, R x q), et
    M = R' Q' donne (Z_a L_K)(Z_b L_K)' = C_a C_b', C = R' (R x R). Le terme
    plie a alors q = R colonnes par composante au lieu de q, pour la meme V
    exactement. Les BLUP de tous les genotypes restent calcules sur le terme
    d'origine.
    """
    import scipy.sparse as sp_
    terms2, Zr2 = [], []
    for tm, Z in zip(terms, Zr):
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        if kind == "fixed" and tm.get("LK") is not None:
            t, q = int(tm["t"]), int(tm["q"])
            Zn = np.asarray(Z)
            cle = ("_plie", Zn.shape)
            if cle not in tm:
                LK = np.asarray(tm["LK"], dtype=np.float64)
                Zs = sp_.csr_matrix(Zn)
                blocs, lignes = [], []
                for a in range(t):
                    Za = Zs[:, a * q:(a + 1) * q].tocsr()
                    blocs.append(np.asarray(Za @ LK))
                    lignes.append(np.flatnonzero(np.diff(Za.indptr)))
                R = int(sum(len(l) for l in lignes))
                if 0 < R < 0.8 * q:
                    M = np.vstack([blocs[a][lignes[a]] for a in range(t)])      # R x q
                    C = np.linalg.qr(M.T, mode="r").T                           # R x R
                    ZL = np.zeros((Zn.shape[0], t * R))
                    o = 0
                    for a in range(t):
                        ZL[lignes[a], a * R:(a + 1) * R] = C[o:o + len(lignes[a])]
                        o += len(lignes[a])
                    q2 = R
                else:
                    ZL, q2 = np.hstack(blocs), q
                tm[cle] = (ZL, q2)
            ZL, q2 = tm[cle]
            tm2 = dict(tm)
            tm2["LK"] = None
            tm2["lvl"] = "id"
            tm2["q"] = q2
            terms2.append(tm2)
            Zr2.append(jnp.asarray(ZL))
        else:
            terms2.append(tm)
            Zr2.append(Z)
    return terms2, Zr2


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
    # RESIDUELLE ABSENTE ("none") : aucune section, donc aucun parametre et
    # R = 0. V doit alors etre definie positive par les seuls termes (par
    # exemple un terme de plante [I, W] us qui porte residuelle, IEE et leur
    # covariance) ; fit_reml le verifie au point de depart.
    if res.get("struct") == "none":
        return []
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


def _section_S(theta_s, sec):
    """Sigma_caractere (t x t) d'une section residuelle.

    Partie commune a _section_V et a la residuelle creuse du mode econome
    (econome.py) : une seule definition des structures iid, diag, us et fa.
    """
    st, t = sec["struct"], sec["t"]
    n_sig = n_params(st, t, sec["rank"])
    if st == "iid":
        return jnp.exp(2.0 * theta_s[0]) * jnp.eye(t)
    if st == "diag":
        return jnp.diag(jnp.exp(2.0 * theta_s[:t]))
    if st in ("us", "fa"):
        Ls = chol_sigma(theta_s[:n_sig], st, t, sec["rank"])
        return Ls @ Ls.T
    raise ValueError("structure residuelle inconnue : %s" % st)


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
    S = _section_S(theta_s, sec)
    # MEMOIRE. Un seul caractere : Sigma est un scalaire, on ne forme pas la
    # matrice n x n S[tr, tr]. Unites deja dans l'ordre 0..n_u-1 : C s'emploie
    # telle quelle, sans le rassemblement C[un, un] qui en faisait une copie.
    un_np = np.asarray(sec["unit"], dtype=np.int64)
    if kind == "id":
        # Sans structure entre unites, deux observations ne sont correlees que
        # si elles portent sur la MEME unite (caracteres differents).
        Cu = (un[:, None] == un[None, :]).astype(S.dtype)
    else:
        n_u = int(sec.get("n_unit", 0)) or int(np.max(un_np) + 1)
        C = level_corr(theta_s[n_sig:], kind, n_u, order=sec.get("lvl_order", 0),
                       coord=sec.get("coord"), dims=sec.get("dims"),
                       opts=sec.get("lvl_opts"), expr=sec.get("lvl_expr"))
        if un_np.shape[0] == n_u and np.array_equal(un_np, np.arange(n_u)):
            Cu = C
        else:
            Cu = C[un[:, None], un[None, :]]
    if t == 1:
        return S[0, 0] * Cu
    return S[tr[:, None], tr[None, :]] * Cu


def _toeplitz_ar1(th, q):
    """Correlation AR1 (q x q), meme parametrisation que level_corr("ar1ar1")."""
    return _toeplitz(_acf(th, "ar1", q), q)


def section_matvec(theta_s, sec, vs):
    """R_s vs SANS former R_s quand la structure le permet.

        (R v)_i = sum_b Sigma[tr_i, b] (C V_b)[un_i],  V_b[u] = v de l'unite u, caractere b

    C = I (pas de structure entre unites) et C = AR1 (x) AR1 se traitent sans
    matrice n x n : (A_r (x) A_c) v = vec(A_r M A_c'), M la grille rangs x
    colonnes. Les autres structures retombent sur la section dense.
    Sert aux colonnes de la matrice AI (ai.py), ou la forme dense coutait une
    matrice tangente n x n par parametre.
    """
    st, t = sec["struct"], sec["t"]
    kind = sec.get("lvl", "id")
    if kind not in ("id", "ar1ar1") or st not in ("iid", "diag", "us", "fa"):
        return _section_V(theta_s, sec) @ vs
    n_sig = n_params(st, t, sec["rank"])
    if st == "iid":
        S = jnp.exp(2.0 * theta_s[0]) * jnp.eye(t)
    elif st == "diag":
        S = jnp.diag(jnp.exp(2.0 * theta_s[:t]))
    else:
        Ls = chol_sigma(theta_s[:n_sig], st, t, sec["rank"])
        S = Ls @ Ls.T
    tr = np.asarray(sec["trait"], dtype=np.int64)
    un = np.asarray(sec["unit"], dtype=np.int64)
    if kind == "id":
        _, un_c = np.unique(un, return_inverse=True)
        n_u = int(un_c.max()) + 1
    else:
        un_c = un
        nr, nc = int(sec["dims"][0]), int(sec["dims"][1])
        n_u = nr * nc
    Vm = jnp.zeros((n_u, t)).at[un_c, tr].add(vs)
    if kind == "ar1ar1":
        th_l = theta_s[n_sig:]
        Cr = _toeplitz_ar1(th_l[:1], nr)
        Cc = _toeplitz_ar1(th_l[1:2], nc)
        Mb = Vm.T.reshape(t, nr, nc)
        Vm = jnp.einsum("rs,bsc,dc->brd", Cr, Mb, Cc).reshape(t, n_u).T
    return (Vm @ S.T)[un_c, tr]


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
    if not secs:
        return jnp.zeros((n, n))
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
# complet de Salas et al. (2026) — 65 termes a n = 16211 — cela fait 65 x 1,96 = 127 Go,
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


def n_theta(terms, res, lag=None):
    """Longueur de theta : termes, residuelle, puis entrees libres de Psi.

    Les parametres du decalage viennent EN DERNIER : split_theta ne lit que la
    tete du vecteur, si bien que tout le code qui ne connait pas le decalage
    (assemble_V, kenward_roger, predict) continue de decouper au bon endroit.
    """
    return sum(term_n_params(t) for t in terms) + res_n_params(res) + lag_n_params(lag)


def alias_entrees(terms, n, n_vec=3, seed=12345, tol=1e-9):
    """Entrees d'une Sigma `us` qui ne sont PAS identifiees separement.

    POURQUOI. La contribution d'un terme a V est sum_{a>=b} Sigma[a,b] P_ab avec
    P_ab = Z_a K Z_b' + Z_b K Z_a' (a != b). Deux entrees de MOTIFS P EGAUX
    n'entrent dans V que par leur somme : seule la somme est identifiee, et la
    vraisemblance est plate dans la direction de leur difference. C'est le cas
    du terme de plante [I, W] de VP+ des que W est symetrique :
    Cov(residuelle_t, IEE_u) et Cov(residuelle_u, IEE_t) ont le meme motif, soit
    q(q-1)/2 directions plates (theorie_v2.md, T1b). Le terme genetique [Z, WZ]
    n'en a pas des que la replication rend W Z Z' non symetrique ; le test le
    dit au cas par cas au lieu de le supposer.

    COMMENT. Les motifs sont compares par leur action sur quelques vecteurs
    aleatoires (P_ab R, cout O(nnz)) : deux motifs differents ne coincident sur
    trois vecteurs gaussiens qu'avec une probabilite nulle. Calcule une fois,
    hors de la fonction derivee (les motifs ne dependent pas de theta).

    Seules les structures `us` sont examinees : leurs entrees sont libres, donc
    un motif commun y est exactement une direction plate. Les termes a structure
    entre niveaux parametree (ar1, ...) sont ignores.

    Rend une liste de groupes : dict(terme, entrees = [(i, j), ...] 0-based,
    i >= j), chaque groupe faisant perdre len(entrees) - 1 dimensions.
    """
    import scipy.sparse as sp_
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n, n_vec))
    out = []
    for tm in terms:
        if tm["struct"] != "us" or tm["t"] < 2:
            continue
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        if kind not in ("id", "fixed"):
            continue
        t, q = tm["t"], tm["q"]
        Z = sp_.csr_matrix((tm["zx"], (tm["zi"], tm["zj"])), shape=(n, t * q))
        blocs = [Z[:, a * q:(a + 1) * q] for a in range(t)]
        LK = tm.get("LK")
        # Z_b' R, puis K, une fois par colonne
        ZtR = [np.asarray(B.T @ R) for B in blocs]
        if LK is not None:
            LK = np.asarray(LK)
            ZtR = [LK @ (LK.T @ v) for v in ZtR]
        sig = {}
        for a in range(t):
            for b in range(a):
                v = np.asarray(blocs[a] @ ZtR[b]) + np.asarray(blocs[b] @ ZtR[a])
                nrm = float(np.linalg.norm(v))
                if nrm == 0.0:
                    continue
                sig[(a, b)] = (v, nrm)
        cles = list(sig)
        vu = set()
        for k, c in enumerate(cles):
            if c in vu:
                continue
            grp = [c]
            for d in cles[k + 1:]:
                if d in vu:
                    continue
                (v1, n1), (v2, n2) = sig[c], sig[d]
                if np.linalg.norm(v1 - v2) <= tol * max(n1, n2):
                    grp.append(d)
                    vu.add(d)
            if len(grp) > 1:
                out.append(dict(terme=tm["name"], entrees=grp))
    return out


def neg2_structurel(theta, terms, groupes, Zr, res, y, X, lag, X_lag=None):
    """-2logL du modele structurel : critere de lag.py sur y* = (I - M) y.

    Rend (valeur, admissible). Sans decalage : neg2_reml_groupes et True.
    """
    n = y.shape[0]
    if lag is None:
        return neg2_reml_groupes(theta, terms, groupes, Zr, res, y, X), jnp.asarray(True)
    p_cov = n_theta(terms, res)
    th_lag = theta[p_cov:]
    ys = lag_ystar(th_lag, lag, y)
    # Le terme de moyenne du decalage porte sur X d'origine ; la partie REML
    # accepte X orthonormee (model.orthonormal_X), la constante etant rajoutee
    # par l'appelant.
    ldA, ldm, ok = lag_terms(th_lag, lag, X if X_lag is None else X_lag)
    V = assemble_V_groupes(theta, terms, groupes, Zr, res, n)
    base = (ml_from_V if lag["criterion"] == "ml" else reml_from_V)(V, ys, X)
    # -2 log|det A| (changement de variable) et, pour reml_y, -2 x le terme de
    # moyenne (lag.py) ; nul pour jacobian et ml.
    val = base - 2.0 * ldA - 2.0 * ldm
    return jnp.where(ok, val, LAG_PENALITE), ok


def make_neg2(bundle_terms, res, y, X, lag=None):
    """-2 logL (restreinte, constante comprise) comme fonction JAX PURE de theta.

    Rend (f, Zr, Zs) : f(theta, Zr) est tracable, derivable et vectorisable
    (vmap), sans mise a l'echelle. C'est l'objectif de make_objective avant sa
    division par n, et l'energie de l'echantillonneur bayesien (bayes.py), qui
    a besoin de la fonction elle-meme et non d'un appel Python. Zr (incidences
    restreintes) se passe en argument pour la meme raison que dans
    make_objective : capturees, elles seraient des constantes repliees par XLA.
    """
    n = y.shape[0]
    # X orthonormee pour la seule vraisemblance (orthonormal_X). Avec un
    # decalage, son terme de moyenne garde X d'origine (X_lag), et le critere ML
    # n'a pas de constante : il ne depend de X que par son espace engendre.
    Xq, ldRx = orthonormal_X(X)
    if lag is not None and lag.get("criterion") == "ml":
        ldRx = 0.0
    yj, Xj = jnp.asarray(y), jnp.asarray(Xq)
    Xl = jnp.asarray(np.asarray(X, dtype=np.float64))
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
    termes_v, Zr = prefold_fixed_K(bundle_terms, Zr)

    # LES INCIDENCES RESTENT CAPTUREES, ET C'EST MESURE. Les passer en argument
    # de la fonction compilee ne gagne rien : tampon 4580,3 Mo contre 4548,7 et
    # 160,8 Mo d'arguments en plus, sur un dispositif de la forme du modele IGE.
    # L'avertissement "2,90 GB of constants were captured" que j'avais lu venait
    # des facteurs PLEINE HAUTEUR (2769 Mo) captures par la forme de reference
    # compilee A COTE pour le test d'identite — pas de cet objectif, dont les
    # incidences restreintes ne pesent que 161 Mo.
    lag = prepare_lag(lag, y, X)

    # LES INCIDENCES SONT DES ARGUMENTS de la fonction compilee. Capturees,
    # elles sont des constantes du programme : XLA tente d'en replier les
    # operations a la compilation, et une incidence dense de 8037 x 8037 (champ
    # ar1 x ar1 du modele ige a n = 8037, 516 Mo) faisait echouer la
    # compilation. En argument, rien n'est repliable et la memoire est la meme.
    def f(theta, Zr_a):
        val, ok = neg2_structurel(jnp.asarray(theta), termes_v, groupes, Zr_a,
                                  res, yj, Xj, lag, X_lag=Xl)
        # la constante de X ne s'ajoute qu'a une vraisemblance : hors de la
        # region admissible du decalage, la penalite reste exactement PENALITE
        return jnp.where(ok, val + ldRx, val)

    return f, Zr, Zs


def make_objective(bundle_terms, res, y, X, scale=None, lag=None):
    """Rend (fn, grad_fn) sur theta, mis a l'echelle.

    `lag` (dictionnaire de lag.py, prepare ou non) ajoute le decalage spatial :
    l'objectif devient le critere structurel de lag.py, et theta porte en queue
    les entrees libres de Psi.

    MISE A L'ECHELLE PAR n : -2logL est d'ordre n et son gradient aussi. Or
    L-BFGS-B demarre avec un hessien approche egal a l'identite : son premier
    pas vaut -gradient, soit des centaines d'unites dans l'espace des parametres
    si l'objectif n'est pas normalise. Tous les parametres se retrouvent alors
    projetes sur les bornes des la deuxieme evaluation et V devient singuliere.
    Le facteur est retire des valeurs rendues a l'appelant.
    """
    n = y.shape[0]
    sc = float(n) if scale is None else float(scale)
    f_brut, Zr, Zs = make_neg2(bundle_terms, res, y, X, lag=lag)

    def f(theta, Zr_a):
        return f_brut(theta, Zr_a) / sc

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
        v, g = fg(jnp.asarray(theta), Zr)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        """Echelle NATURELLE (-2logL), pour le diagnostic et les sorties."""
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc

    return fun_jac_scaled, fun_jac, sc, Zs
