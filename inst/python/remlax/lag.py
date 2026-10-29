"""Modele structurel a decalage spatial (spatial lag) : y = M y + X b + Z u + e.

MODELE
    Reponse longue y (n lignes ; chaque ligne porte un caractere t et une unite i).

        y = M y + X b + sum_k Z_k u_k + e,     M[(t,i),(u,j)] = Psi[t,u] * W_t[i,j]

    Psi (q x q) : effet du caractere EMETTEUR u (colonne) d'une voisine j sur le
    caractere RECEVEUR t (ligne) de la plante i ; W_t (N x N, diagonale nulle) :
    poids de voisinage, communs a tous les caracteres ou propres a chaque
    caractere receveur. Avec A = I - M inversible, y* = A y suit le modele
    ordinaire du paquet (moyenne X b, covariance V(theta)) : tout ce que remlax
    sait faire sur V s'applique donc sans changement a y*.

CRITERES (-2logL, convention du paquet ; voir aussi vignettes/structural.Rmd)
    jacobian   -2 REML(y*) - 2 log|det A|
               Changement de variable y -> y*. Le terme de Harville (le
               log|X'X| des contrastes) est pris sur X, la matrice des effets
               fixes de y*, donc constant en Psi.
    reml_y     jacobian - log|X_y'X_y| + log|X'X|,    X_y = A^-1 X
               (DEFAUT) La REML EXACTE de y. La moyenne de y est X_y b et non
               X b ; or X_y depend de Psi des que les sommes de lignes de W
               varient (plantes de bordure). Les contrastes de y sont orthogonaux
               a X_y, et le terme de Harville de y, log|X_y'X_y|, n'est donc pas
               constant en Psi. Ce n'est pas un terme en O(1/N) : a psi_v = +0,6
               sur un caractere, X_y ~ X / (1 - psi_v) a l'interieur, soit un
               decalage d'environ log(1/0,4) = 0,9 en logLik
               (moteur igepath_reml, moteur_v2.md section 2). Sans lui, le
               critere depend de l'echelle de b dans la parametrisation de y*.
    ml         -2 ML(y*) - 2 log|det A|, vraisemblance profilee en b. Exacte
               pour y (le terme de Harville n'existe pas en ML).
    Les trois coincident a une constante pres quand Psi = 0, et jacobian et
    reml_y coincident exactement quand W a des sommes de lignes constantes.

log|det A|, TROIS CHEMINS, choisis une fois sur la structure (jamais sur theta)
    nilpotent  le motif des entrees non nulles possibles de Psi (libres ou
               fixees non nulles) est un graphe SANS CYCLE : M^q = 0, donc
               det A = 1 exactement et A^-1 = sum_{m<q} M^m. Pas de boucle de
               retroaction, pas de question d'equilibre.
    rapide     une seule W, symetrique, et chaque couple (caractere, unite)
               observe exactement une fois. W = U diag(lambda) U' (une fois),
               et log|det A| = sum_k log det(I_q - lambda_k Psi) : q x q par
               valeur propre. C'est la formule du moteur igepath_reml.
    general    tout le reste (W par caractere, W non symetrique, plan
               incomplet) : factorisation LU dense de A (n x n), du meme ordre
               que la Cholesky de V deja payee a chaque evaluation.

REGION ADMISSIBLE. La region retenue est celle que l'on atteint depuis Psi = 0
sans traverser de singularite le long du rayon t Psi :

    det(I - t M) > 0 pour tout t dans (0, 1]
    <=> toute valeur propre REELLE de M est < 1.

Ce n'est PAS "tous les det(I_q - lambda_k Psi) > 0" : avec Psi = diag(c, c),
chaque det_k vaut (1 - lambda_k c)^2 >= 0, et un point ou DEUX facteurs ont
change de signe passerait ce test alors qu'on n'y arrive qu'en traversant
det A = 0 (trouve par le test de penalite, Psi = diag(0,96 ; 0,96) sur une
grille reine). Les valeurs propres de M valent lambda_k mu_i (mu_i celles de
Psi) sur le chemin rapide, donc la condition porte sur les seules valeurs
propres de Psi, q x q, et sur les deux valeurs propres extremes de W : elle
est verifiee EXACTEMENT a chaque evaluation. Sur le chemin general, seule la
condition necessaire det A > 0 est verifiee a chaque evaluation (le spectre
de M, n x n non symetrique, couterait dix Cholesky) ; la condition exacte est
verifiee au point final et rapportee (`admissible`). Un rayon spectral > 1
avec toutes les valeurs propres reelles < 1 reste admis : A est inversible et
le modele simultane est defini, meme si l'equilibre n'est plus la limite de
l'iteration y <- M y + ... ; le rayon spectral est rapporte.

Hors de la region, l'objectif vaut une PENALITE finie, de gradient nul :
L-BFGS-B la voit comme une montee brutale et rebrousse chemin dans sa
recherche lineaire. Une valeur infinie ou NaN arreterait l'optimiseur, et le
gradient d'un log de nombre negatif empoisonnerait le reste par NaN (d'ou la
double garde `where` : on calcule sur une matrice SURE et on choisit ensuite).

PARAMETRISATION. theta_lag = Psi * s pour les entrees libres, avec s la somme
moyenne des |lignes| de W. Psi est ainsi a l'echelle "voisinage entier" (le
psi_v du moteur igepath_reml) dans l'optimiseur : une unite de theta y vaut
l'effet de toutes les voisines, ce qui met Psi a la meme echelle que les
log-ecarts-types et evite qu'un W a petits poids (1/d^2) ne rende le gradient en
Psi des ordres de grandeur plus grand que les autres. Les sorties sont en
unites de W (Psi = theta_lag / s).
"""
from . import _x64  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp

CRITERES = ("reml_y", "jacobian", "ml")

# Valeur de -2logL hors de la region admissible. Finie (cf. en-tete) et tres
# au-dessus de toute valeur atteignable sur un jeu de taille realiste.
PENALITE = 1e12


def lag_n_params(lag):
    """Nombre d'entrees LIBRES de Psi (0 sans decalage)."""
    if lag is None:
        return 0
    return int(np.asarray(lag["mask"], dtype=bool).sum())


def _nilpotent(B):
    """Le motif booleen B (q x q) est-il nilpotent (graphe sans cycle) ?"""
    q = B.shape[0]
    P = B.astype(np.int64)
    Bi = B.astype(np.int64)
    for _ in range(q):
        P = (P @ Bi > 0).astype(np.int64)
    # P vaut B^(q+1) en motif ; B nilpotent <=> B^q = 0 <=> B^(q+1) = 0 aussi,
    # et un cycle laisse B^m non nul pour tout m.
    return not P.any()


def prepare_lag(lag, y, X):
    """Precalculs CONSTANTS du decalage, hors de la fonction derivee.

    Idempotente : un dictionnaire deja prepare est rendu tel quel (les
    redemarrages de fit_reml ne refont pas la decomposition spectrale).
    """
    if lag is None or lag.get("_prepare"):
        return lag
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, p = X.shape
    q, N = int(lag["q"]), int(lag["N"])
    tr = np.asarray(lag["trait"], dtype=np.int64)
    un = np.asarray(lag["unit"], dtype=np.int64)
    if tr.shape != (n,) or un.shape != (n,):
        raise ValueError("lag : `trait` et `unit` doivent avoir %d elements (un par ligne de y)." % n)
    if tr.min() < 0 or tr.max() >= q or un.min() < 0 or un.max() >= N:
        raise ValueError("lag : indice de caractere hors de 0..%d ou d'unite hors de 0..%d." % (q - 1, N - 1))
    cle = tr * N + un
    if len(np.unique(cle)) != n:
        raise ValueError("lag : %d couple(s) (caractere, unite) en double. Chaque unite ne "
                         "peut porter qu'une observation par caractere : sinon M n'est pas "
                         "definie." % (n - len(np.unique(cle))))
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

    # Echelle : somme moyenne des |lignes| de W, moyennee sur les W.
    s = float(np.mean([np.asarray(abs(W).sum(axis=1)).mean() for W in Ws]))
    s = s if s > 0 else 1.0

    widx = np.zeros(n, dtype=np.int64) if len(Ws) == 1 else tr
    # Q[r, u] = (W_{t(r)} y_u)[unit(r)] : M y = sum_u Psi[t(r), u] Q[r, u]. y* est
    # donc LINEAIRE en Psi avec Q constant, et son calcul coute n q par
    # evaluation au lieu d'un produit n x n.
    yfull = np.zeros((q, N))
    yfull[tr, un] = y
    WY = [np.asarray(W @ yfull.T) for W in Ws]            # (N, q) chacun
    Q = np.empty((n, q))
    for k in range(len(Ws)):
        rows = np.flatnonzero(widx == k)
        Q[rows] = WY[k][un[rows]]

    motif = mask | (value != 0)
    complet = n == q * N
    unique_W = len(Ws) == 1 or all((abs(W - Ws[0]).max() if (W - Ws[0]).nnz else 0.0) == 0
                                   for W in Ws[1:])
    W0 = Ws[0]
    asym = abs(W0 - W0.T).max() if W0.nnz else 0.0
    sym = asym <= 1e-12 * max(1.0, abs(W0).max() if W0.nnz else 1.0)
    if _nilpotent(motif):
        chemin = "nilpotent"
    elif unique_W and sym and complet:
        chemin = "rapide"
    else:
        chemin = "general"
    # Forcage du chemin, pour les TESTS seulement (comparer rapide et general
    # sur le meme modele). Le chemin nilpotent exige le motif nilpotent.
    force = lag.get("force_chemin")
    if force:
        if force not in ("rapide", "general", "nilpotent"):
            raise ValueError("lag : force_chemin = %r inconnu." % force)
        if force == "nilpotent" and chemin != "nilpotent":
            raise ValueError("lag : chemin nilpotent force sur un motif de Psi avec boucle.")
        if force == "rapide" and not (unique_W and sym and complet):
            raise ValueError("lag : chemin rapide force sans W unique symetrique et plan complet.")
        chemin = force

    mi, mj = np.nonzero(mask)
    # Psi identiquement nulle (aucune entree libre, aucune valeur fixee non
    # nulle) : A = I, et l'objectif doit etre CELUI du modele sans decalage,
    # bit pour bit. On ne passe donc ni par slogdet(X'X) en JAX (qui differe du
    # log|X'X| numpy au dernier chiffre) ni par aucun autre detour flottant.
    vide = not motif.any()
    out = dict(lag, _prepare=True, vide=vide, q=q, N=N, n=n, p=p, trait=tr, unit=un, mask=mask,
               value=value, criterion=crit, scale=s, chemin=chemin, Q=jnp.asarray(Q),
               tr_j=jnp.asarray(tr), mi=mi, mj=mj, value_j=jnp.asarray(value),
               traits=list(lag.get("traits") or ["T%d" % (k + 1) for k in range(q)]),
               logXX=(float(np.linalg.slogdet(X.T @ X)[1]) if p else 0.0))
    if chemin == "rapide":
        lam, U = np.linalg.eigh(W0.toarray())
        pos = np.empty((q, N), dtype=np.int64)
        pos[tr, un] = np.arange(n)
        Xt = np.einsum("ik,tip->ktp", U, X[pos]) if p else np.zeros((N, q, 0))
        out.update(lam=jnp.asarray(lam), lam_np=lam, Xt=jnp.asarray(Xt),
                   lam_min=float(lam.min()), lam_max=float(lam.max()))
    else:
        # W "longue" : Wl[r, s] = W_{t(r)}[unit(r), unit(s)], n x n dense comme V.
        Wl = np.empty((n, n))
        for k, W in enumerate(Ws):
            rows = np.flatnonzero(widx == k)
            if rows.size:
                Wd = W[un[rows]].toarray()
                Wl[rows] = Wd[:, un]
        out.update(Wl=jnp.asarray(Wl))
    return out


def psi_from_theta(th_lag, L):
    """Psi (q x q, unites de W) depuis les parametres libres."""
    Psi = L["value_j"]
    if len(L["mi"]):
        Psi = Psi.at[L["mi"], L["mj"]].set(jnp.asarray(th_lag) / L["scale"])
    return Psi


def theta_from_psi(Psi, L):
    """Inverse de psi_from_theta sur les entrees libres (depart)."""
    Psi = np.asarray(Psi, dtype=np.float64)
    return Psi[L["mi"], L["mj"]] * L["scale"]


def ystar(th_lag, L, y):
    """y* = (I - M) y = y - sum_u Psi[t(r), u] Q[r, u]."""
    if L.get("vide"):
        return jnp.asarray(y)
    Psi = psi_from_theta(th_lag, L)
    return jnp.asarray(y) - jnp.sum(Psi[L["tr_j"], :] * L["Q"], axis=1)


def _vp_reelles_admissibles(Psi, lam_min, lam_max):
    """Toute valeur propre reelle mu de Psi verifie mu lambda < 1 aux deux
    extremes du spectre de W (numpy ; appele hors derivation)."""
    Psi = np.asarray(Psi, dtype=np.float64)
    mu = np.linalg.eigvals(Psi)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(mu))) if mu.size else 1.0)
    re = mu.real[np.abs(mu.imag) <= tol]
    return np.bool_(np.all(re * lam_max < 1.0) and np.all(re * lam_min < 1.0))


def _ok_rapide(Psi, L):
    """Condition exacte de la region (en-tete), sous jit : rappel numpy sur Psi
    q x q, a travers stop_gradient (ok n'a pas de derivee, et sans lui la
    differentiation chercherait une regle de derivation du rappel)."""
    lmin, lmax = L["lam_min"], L["lam_max"]
    return jax.pure_callback(lambda P: _vp_reelles_admissibles(P, lmin, lmax),
                             jax.ShapeDtypeStruct((), jnp.bool_),
                             jax.lax.stop_gradient(Psi))


def _M_dense(Psi, L):
    tr = L["tr_j"]
    return Psi[tr[:, None], tr[None, :]] * L["Wl"]


def lag_terms(th_lag, L, X):
    """Termes du critere qui dependent de Psi au-dela de y*.

    Rend (log|det A|, terme de moyenne, ok) :
      terme de moyenne = 1/2 log|X_y'X_y| - 1/2 log|X'X| (0 si non demande) ;
      ok = point dans la region admissible (booleen JAX).
    """
    if L.get("vide"):
        return jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(True)
    Psi = psi_from_theta(th_lag, L)
    q = L["q"]
    besoin_moy = L["criterion"] == "reml_y" and L["p"] > 0
    Xj = jnp.asarray(X)
    if L["chemin"] == "nilpotent":
        ldA, ok = jnp.asarray(0.0), jnp.asarray(True)
        if besoin_moy:
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
        # DOUBLE GARDE : calcul sur une matrice sure, choix ensuite (en-tete).
        Bs = jnp.where(ok, B, jnp.eye(q)[None])
        ldA = jnp.sum(jnp.linalg.slogdet(Bs)[1])
        if besoin_moy:
            Yk = jnp.linalg.solve(Bs, L["Xt"])                 # (N, q, p)
            G = jnp.einsum("ktp,ktm->pm", Yk, Yk)
            return ldA, 0.5 * (jnp.linalg.slogdet(G)[1] - L["logXX"]), ok
        return ldA, jnp.asarray(0.0), ok
    else:
        n = L["n"]
        A = jnp.eye(n) - _M_dense(Psi, L)
        lu, piv = jax.scipy.linalg.lu_factor(A)
        dg = jnp.diagonal(lu)
        n_ech = jnp.sum(piv != jnp.arange(n))
        signe = jnp.prod(jnp.sign(dg)) * (1 - 2 * (n_ech % 2))
        ok = (signe > 0) & jnp.all(dg != 0)
        ldA = jnp.sum(jnp.log(jnp.abs(jnp.where(dg == 0, 1.0, dg))))
        Xy = jax.scipy.linalg.lu_solve((lu, piv), Xj) if besoin_moy else None
    if besoin_moy:
        G = Xy.T @ Xy
        return ldA, 0.5 * (jnp.linalg.slogdet(G)[1] - L["logXX"]), ok
    return ldA, jnp.asarray(0.0), ok


def lag_report(th_lag, L, X, neg2):
    """Sorties du decalage au point final (numpy), y compris le rayon spectral de M."""
    Psi = np.asarray(psi_from_theta(jnp.asarray(th_lag), L))
    crit = L["criterion"]
    # Le terme de moyenne est rapporte meme pour les criteres qui ne l'incluent
    # pas, pour qu'on puisse passer d'un critere a l'autre sans reajuster.
    L2 = dict(L, criterion="reml_y")
    ldA, ldm, ok = lag_terms(jnp.asarray(th_lag), L2, X)
    ldA, ldm, ok = float(ldA), float(ldm), bool(ok)
    if L["chemin"] == "nilpotent":
        rho = 0.0
    elif L["chemin"] == "rapide":
        rho = float(np.max(np.abs(L["lam_np"])) * np.max(np.abs(np.linalg.eigvals(Psi))))
    else:
        # Chemin general : le spectre de M, une fois, donne le rayon spectral ET
        # la condition exacte de la region (seul det A > 0 a ete verifie
        # pendant l'optimisation).
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
    """Noms des entrees libres, dans l'ordre de theta : Psi[receveur<-emetteur]."""
    tn = L.get("traits") or ["T%d" % (k + 1) for k in range(int(L["q"]))]
    mi, mj = np.nonzero(np.asarray(L["mask"], dtype=bool).reshape(int(L["q"]), int(L["q"])))
    return ["Psi[%s<-%s]" % (tn[i], tn[j]) for i, j in zip(mi, mj)]
