"""Diagnostic de sparsite : quel moteur vaut le coup, decide AVANT tout calcul.

CE QUE CE MODULE REPOND, ET CE QU'IL NE REPOND PAS
--------------------------------------------------
remlax forme V dense (n x n) et la factorise : le cout est en n^3 quoi qu'il
arrive. Un moteur creux travaille sur les equations du modele mixte

    C = [ X'R^-1 X   X'R^-1 Z            ]
        [ Z'R^-1 X   Z'R^-1 Z + G^-1     ]

et son cout depend du REMPLISSAGE de la factorisation de C, pas de sa taille.
Ce module calcule ce remplissage par factorisation SYMBOLIQUE : aucune valeur
numerique n'est formee, seul le motif des non-nuls circule. Compter le
remplissage prend quelques secondes a m ~ 6000 dans cette implementation en
Python pur (les boucles sur les non-nuls dominent) contre des heures pour
l'ajustement lui-meme ; passer par SuiteSparse le ramenerait a des
millisecondes, et c'est l'optimisation evidente si l'appel devient routinier.

CE QU'IL NE REPOND PAS, ET C'EST IMPORTANT
------------------------------------------
1. Un RAPPORT DE FLOPS N'EST PAS UN RAPPORT DE TEMPS. Aux petites tailles il est
   grotesquement optimiste pour le creux : sur un facteur iid a q = 1000 il
   annonce 1e7 en faveur du creux, alors que le rapport de temps observe entre
   lme4 et remlax sur ce genre de modele est de l'ordre de 1e2 a 1e3. Les
   formats creux, les indirections et le trafic memoire ne sont pas comptes. Les
   constantes doivent etre CALIBREES sur des mesures avant toute prediction de
   temps.
2. Il ignore le NOMBRE d'evaluations que demandera l'optimiseur, qui depend
   fortement de la parametrisation : mesure sur cette grille de benchmarks,
   us(t=8) converge en 26 iterations et fa(t=8) en 258. Ce multiplicateur doit
   venir d'un ajustement, jamais de la theorie.

CE QUI RESSORT DU DIAGNOSTIC, ET QUI EST CONTRE-INTUITIF
--------------------------------------------------------
La regle « parente genomique donc modele dense » est FAUSSE telle quelle. Avec
n = 4000 observations et q = 1000 genotypes a parente genomique, C est
1001 x 1001 pleine : 3,4e8 flops contre 4,1e10 pour une factorisation de V en
n^3. Le creux gagne encore un facteur 100, simplement parce que le nombre
d'effets est bien inferieur au nombre d'observations — c'est l'argument classique
de Henderson en faveur des equations du modele mixte.

Ce qui tue le creux est autre chose : un TERME A AUTANT DE NIVEAUX QUE
D'OBSERVATIONS dont l'inverse est plein, typiquement un champ spatial a noyau
metrique. Mesure sur la forme reelle du modele IGE (genotypes a parente
genomique PLUS champ iexp a q = n), C devient 1,5 fois plus grande que V et
remplie a 81 % : le rapport de flops tombe a 4,3 et ne bouge plus avec la
taille. Un facteur 4 sur une matrice plus grande et presque pleine ne survit pas
aux surcouts des formats creux. C'est la l'explication structurelle du fait
mesure dans le projet d'origine — RTMB a ~36 h contre ~15 min pour le moteur
dense a n = 16211 — et elle tient au champ spatial, pas a la parente.

LE POINT QUI DECIDE : LA SPARSITE DE K^-1, PAS LA TAILLE DU PROBLEME
--------------------------------------------------------------------
G^-1 = Sigma^-1 (x) K^-1 apparait dans C. Ce qui compte est donc la sparsite de
l'INVERSE de la correlation entre niveaux, et elle differe radicalement d'une
structure a l'autre :

    iid                     K^-1 diagonale
    ar1, ar1(x)ar1          K^-1 tridiagonale, ou Kronecker de tridiagonales
    parente genealogique A  A^-1 creuse, ~5 non-nuls par individu (Henderson)
    parente GENOMIQUE K     K^-1 PLEINE
    noyaux metriques        K^-1 pleine

C'est l'explication structurelle d'une mesure du projet d'origine : a n = 16211
avec une parente genomique, le moteur RTMB a mis ~36 h contre ~15 min pour le
moteur dense sur A100. Ce n'est pas une contre-performance de TMB, c'est un
modele qui n'est pas creux : le bloc G^-1 est plein, donc la factorisation creuse
paie une factorisation quasi dense EN PLUS de sa machinerie d'indices.

ORDONNANCEMENT. Le remplissage depend de la permutation. On utilise Reverse
Cuthill-McKee, disponible dans scipy. Une vraie permutation de degre minimum
approche (AMD, dans SuiteSparse) ferait mieux sur les motifs irreguliers, donc
le remplissage rendu ici est une BORNE SUPERIEURE. C'est le sens prudent pour un
outil de decision : il sous-promet le creux plutot que de le survendre.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Sparsite de K^-1 par structure de niveaux. "dense" veut dire que le bloc se
# remplit entierement, quelle que soit la permutation.
INVERSE_SPARSITY = {
    "id": "diagonal", "iid": "diagonal",
    "ar1": "tridiagonal", "sar": "tridiagonal", "ma1": "tridiagonal",
    "ar2": "banded", "ar3": "banded", "ma2": "banded", "arma": "banded",
    "ar1ar1": "kron_tridiagonal",
    "cor": "dense", "corb": "dense", "corg": "dense",
    "exp": "dense", "gau": "dense", "iexp": "dense", "igau": "dense",
    "ieuc": "dense", "aexp": "dense", "agau": "dense", "mtrn": "dense",
    "sph": "dense", "cir": "dense", "lvr": "dense", "ilv": "dense",
    "fixed": "dense",          # une K fournie : genomique par defaut, donc pleine
    "pedigree": "sparse_5",    # a declarer explicitement : A^-1 de Henderson
}


def _bloc_inverse(kind, q, dims=None, nnz_per_row=5):
    """Motif de K^-1 pour un terme, en matrice creuse booleenne (q x q)."""
    s = INVERSE_SPARSITY.get(kind, "dense")
    if s == "diagonal":
        return sp.eye(q, dtype=bool, format="csr")
    if s == "tridiagonal":
        return sp.diags([1, 1, 1], [-1, 0, 1], shape=(q, q), dtype=bool, format="csr")
    if s == "banded":
        return sp.diags([1] * 5, [-2, -1, 0, 1, 2], shape=(q, q), dtype=bool, format="csr")
    if s == "kron_tridiagonal":
        if not dims:
            raise ValueError("ar1ar1 exige dims=(n_lignes, n_colonnes)")
        a, b = dims
        if a * b != q:
            raise ValueError("dims %s incompatible avec q = %d" % (dims, q))
        T = lambda k: sp.diags([1, 1, 1], [-1, 0, 1], shape=(k, k), dtype=bool)
        return sp.kron(T(a), T(b), format="csr")
    if s == "sparse_5":
        # A^-1 de Henderson : ~nnz_per_row non-nuls par ligne, places au hasard
        # avec un decalage court — le MOTIF exact importe peu, sa densite oui.
        off = list(range(-(nnz_per_row // 2), nnz_per_row // 2 + 1))
        return sp.diags([1] * len(off), off, shape=(q, q), dtype=bool, format="csr")
    return sp.csr_matrix(np.ones((q, q), dtype=bool))


def mme_pattern(terms, res, n, p_fixed=1, dense_X=True, _neutralise=None):
    """Motif des non-nuls de C, la matrice des equations du modele mixte.

    Rend (C, blocs) ou blocs decrit l'origine de chaque plage d'indices, ce qui
    permet de dire OU se produit le remplissage et pas seulement combien.

    _neutralise sert a l'attribution : "G" remplace G^-1 par une diagonale,
    "Z" remplace Z'R^-1 Z par sa diagonale. Comparer les remplissages des trois
    motifs dit laquelle des deux causes porte le remplissage observe.
    """
    lignes, cols, data = [], [], []
    offs, blocs = [], []
    off = p_fixed
    Zcols = []
    for tm in terms:
        t, q = tm["t"], tm["q"]
        Zcols.append((off, off + t * q))
        blocs.append(dict(name=tm.get("name", "?"), start=off, stop=off + t * q,
                          kind=(tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")),
                          t=t, q=q))
        off += t * q
    m = off

    # --- Z' R^-1 Z : deux colonnes de Z se rencontrent si elles partagent une
    # observation. On l'obtient exactement depuis les triplets, sans former Z.
    Zs = []
    for (a, b), tm in zip(Zcols, terms):
        Z = sp.csr_matrix((np.ones(len(tm["zi"]), dtype=bool),
                           (np.asarray(tm["zi"]), np.asarray(tm["zj"]))),
                          shape=(n, b - a))
        Zs.append((a, b, Z))
    if _neutralise == "Z":
        d0 = np.arange(p_fixed, m)
        lignes.append(d0); cols.append(d0)
    else:
        for i, (a, b, Zi) in enumerate(Zs):
            for j, (c, d, Zj) in enumerate(Zs):
                B = (Zi.T @ Zj).tocoo()
                if B.nnz:
                    lignes.append(B.row + a); cols.append(B.col + c)
    # --- G^-1 : Sigma^-1 (x) K^-1, donc le bloc de K^-1 repete t x t fois
    for (a, b, _), tm, bl in zip(Zs, terms, blocs):
        t, q = tm["t"], tm["q"]
        kind_eff = "id" if _neutralise == "G" else bl["kind"]
        Ki = _bloc_inverse(kind_eff, q, dims=tm.get("dims")).tocoo()
        for u in range(t):
            for v in range(t):
                lignes.append(Ki.row + a + u * q); cols.append(Ki.col + a + v * q)
    # --- X' R^-1 X et X' R^-1 Z
    if p_fixed:
        rf, cf = np.meshgrid(np.arange(p_fixed), np.arange(p_fixed), indexing="ij")
        lignes.append(rf.ravel()); cols.append(cf.ravel())
        if dense_X:
            # X dense : chaque effet fixe touche toutes les colonnes de Z
            for a, b, _ in Zs:
                r2, c2 = np.meshgrid(np.arange(p_fixed), np.arange(a, b), indexing="ij")
                lignes.append(r2.ravel()); cols.append(c2.ravel())
                lignes.append(c2.ravel()); cols.append(r2.ravel())

    r = np.concatenate(lignes); c = np.concatenate(cols)
    C = sp.coo_matrix((np.ones(len(r), dtype=bool), (r, c)), shape=(m, m)).tocsr()
    C = ((C + C.T) > 0).tocsr()                       # motif symetrise
    return C, blocs


def _arbre_elimination(A):
    """Arbre d'elimination du motif A (deja permute), par union-find compresse."""
    n = A.shape[0]
    parent = np.full(n, -1, dtype=np.int64)
    ancetre = np.full(n, -1, dtype=np.int64)
    Ai, Ap = A.indices, A.indptr
    for k in range(n):
        for idx in range(Ap[k], Ap[k + 1]):
            i = Ai[idx]
            if i >= k:
                continue
            # remonter jusqu'a la racine, en compressant le chemin
            r = i
            while ancetre[r] != -1 and ancetre[r] != k:
                nxt = ancetre[r]; ancetre[r] = k; r = nxt
            if ancetre[r] == -1:
                ancetre[r] = k; parent[r] = k
    return parent


def symbolic_cholesky_nnz(C, permute=True):
    """Non-nuls du facteur de Cholesky de C, SANS aucun calcul numerique.

    Algorithme montant classique : pour chaque ligne k, le motif de L[k, :] est
    l'union des sous-arbres de l'arbre d'elimination atteints depuis les
    non-nuls de C[k, :k]. Le coup en temps est O(nnz(L)).
    """
    C = sp.csr_matrix(C)
    n = C.shape[0]
    perm = reverse_cuthill_mckee(C.tocsr(), symmetric_mode=True) if permute \
        else np.arange(n)
    A = C[perm][:, perm].tocsr()
    parent = _arbre_elimination(A)
    colcount = np.zeros(n, dtype=np.int64)            # non-nuls SOUS la diagonale
    total = 0
    marque = np.full(n, -1, dtype=np.int64)
    Ai, Ap = A.indices, A.indptr
    for k in range(n):
        marque[k] = k
        nk = 0
        for idx in range(Ap[k], Ap[k + 1]):
            i = Ai[idx]
            if i >= k:
                continue
            while marque[i] != k:
                marque[i] = k
                colcount[i] += 1
                nk += 1
                i = parent[i]
                if i == -1:
                    break
        total += nk
    nnz_L = int(total + n)                            # + la diagonale
    # flops d'une Cholesky creuse : somme des carres des hauteurs de colonne
    flops = float(np.sum(colcount.astype(float) ** 2 + 2.0 * colcount))
    return dict(n=n, nnz_C=int(A.nnz), nnz_L=nnz_L,
                fill_ratio=nnz_L / (n * (n + 1) / 2.0),
                sparse_flops=flops, perm="rcm" if permute else "natural",
                colcount_max=int(colcount.max()) if n else 0)


def dense_cost(terms, n):
    """Flops par evaluation du moteur dense de remlax.

    Trois postes, et le premier est celui qu'on oublie : quand K est fournie,
    term_factor forme Z (I (x) L_K) puis B B'. Avec K = I il prend une branche
    qui ne forme AUCUN produit, ce qui rend un balayage a K = I trompeusement
    bon marche.
    """
    assemblage = 0.0
    for tm in terms:
        t, q = tm["t"], tm["q"]
        mrank = t if tm.get("rank", 0) in (0, None) else tm["rank"]
        if tm.get("LK") is not None or tm.get("lvl") not in (None, "id", "iid"):
            assemblage += n * t * q * q            # Z (I (x) L_K)
            assemblage += n * n * mrank * q        # B B'
        else:
            assemblage += n * n * mrank * q        # B B' seulement
    chol = n ** 3 / 3.0
    return dict(assembly_flops=assemblage, cholesky_flops=chol,
                total_flops=assemblage + chol)


def incidence_fill(terms, n):
    """Remplissage cause par l'INCIDENCE, independamment de la parente.

    DEUXIEME MECANISME, ET IL EST SOUVENT LE DOMINANT. Z' R^-1 Z relie deux
    niveaux des qu'ils cooccurrent dans une observation. Une incidence a k
    non-nuls par ligne produit donc jusqu'a k^2 liens par observation, et le
    bloc (q x q) se remplit MEME SI K^-1 est diagonale.

    C'est le cas des incidences explicites : une matrice de voisinage a l'ordre 1
    a 4 voisins par plante et reste creuse, mais a l'ordre 5 elle en a plusieurs
    dizaines et le bloc devient plein. Un moteur creux n'y gagne rien, quelle que
    soit la structure de parente — l'incidence seule impose un modele dense.
    """
    out = []
    for tm in terms:
        t, q = tm["t"], tm["q"]
        zi, zj = np.asarray(tm["zi"]), np.asarray(tm["zj"])
        Z = sp.csr_matrix((np.ones(len(zi), dtype=bool), (zi, zj)), shape=(n, t * q))
        ztz = (Z.T @ Z)
        par_ligne = len(zi) / float(n)
        dens = ztz.nnz / float((t * q) ** 2)
        out.append(dict(name=tm.get("name", "?"), t=t, q=q,
                        z_nnz_per_row=par_ligne,
                        ztz_nnz=int(ztz.nnz), ztz_density=dens,
                        incidence="dense" if dens > 0.5 else
                                  ("filling" if dens > 0.05 else "sparse")))
    return out


def sparsity_report(terms, res, n, p_fixed=1, n_eval=None, verbose=True):
    """Verdict : quel moteur, et pourquoi.

    Deux causes de remplissage sont distinguees, parce qu'elles se corrigent
    differemment : celle de la PARENTE (G^-1 plein, cas genomique) et celle de
    l'INCIDENCE (Z'Z plein, cas d'un voisinage d'ordre eleve). L'une des deux
    suffit a rendre le creux inutile.
    """
    inc = incidence_fill(terms, n)
    C, blocs = mme_pattern(terms, res, n, p_fixed=p_fixed)
    sym = symbolic_cholesky_nnz(C)
    dns = dense_cost(terms, n)
    ratio = dns["total_flops"] / max(sym["sparse_flops"], 1.0)
    verdict = "sparse" if ratio > 3.0 else ("dense" if ratio < 1.0 / 3 else "comparable")
    # ATTRIBUTION PAR CONTREFACTUEL, pas par seuil sur une densite. On refait la
    # factorisation symbolique en neutralisant une cause a la fois : la chute du
    # remplissage mesure sa contribution. Un seuil sur la densite de Z'Z ne
    # marche pas — a l'ordre 5, Z'Z n'est rempli qu'a 12 % et pourtant le
    # facteur l'est a 86 %, parce que le remplissage se propage.
    C_sans_G, _ = mme_pattern(terms, res, n, p_fixed=p_fixed, _neutralise="G")
    C_sans_Z, _ = mme_pattern(terms, res, n, p_fixed=p_fixed, _neutralise="Z")
    f_tot = sym["fill_ratio"]
    f_sans_G = symbolic_cholesky_nnz(C_sans_G)["fill_ratio"]
    f_sans_Z = symbolic_cholesky_nnz(C_sans_Z)["fill_ratio"]
    # Part de remplissage qui disparait quand on neutralise chaque cause.
    dG, dZ = f_tot - f_sans_G, f_tot - f_sans_Z
    attribution = dict(fill_ratio=f_tot, fill_without_Ginv=f_sans_G,
                       fill_without_ZtZ=f_sans_Z, drop_from_Ginv=dG, drop_from_ZtZ=dZ)
    if f_tot < 0.05:
        cause = "no cause: the factor stays sparse (fill %.1f %%)" % (100 * f_tot)
    elif dG > 2 * dZ:
        cause = ("the relationship: neutralising G^-1 drops fill from %.0f %% to %.0f %%"
                 % (100 * f_tot, 100 * f_sans_G))
    elif dZ > 2 * dG:
        cause = ("the incidence: neutralising Z'Z drops fill from %.0f %% to %.0f %%"
                 % (100 * f_tot, 100 * f_sans_Z))
    else:
        cause = ("both, comparably: fill %.0f %%, and %.0f %% / %.0f %% with G^-1 / Z'Z "
                 "neutralised" % (100 * f_tot, 100 * f_sans_G, 100 * f_sans_Z))

    out = dict(sym, **dns, flop_ratio_dense_over_sparse=ratio, verdict=verdict,
               fill_cause=cause, fill_attribution=attribution, incidence=inc,
               blocks=[dict(b, kind_inverse=INVERSE_SPARSITY.get(b["kind"], "dense"))
                       for b in blocs])
    if n_eval:
        out["n_eval"] = n_eval
        out["sparse_flops_total"] = sym["sparse_flops"] * n_eval
        out["dense_flops_total"] = dns["total_flops"] * n_eval
    if verbose:
        print("C : %d x %d, nnz %d" % (sym["n"], sym["n"], sym["nnz_C"]))
        for b, i in zip(out["blocks"], inc):
            print("  terme %-8s t=%d q=%-6d structure %-8s -> K^-1 %-16s | "
                  "Z : %.1f non-nuls/ligne, Z'Z rempli a %.1f %% -> %s"
                  % (b["name"], b["t"], b["q"], b["kind"], b["kind_inverse"],
                     i["z_nnz_per_row"], 100 * i["ztz_density"], i["incidence"]))
        print("cause du remplissage : %s" % cause)
        print("facteur creux : nnz(L) = %d, remplissage %.4f du triangle, "
              "colonne la plus haute %d" % (sym["nnz_L"], sym["fill_ratio"],
                                            sym["colcount_max"]))
        print("flops par evaluation : creux %.3g | dense %.3g "
              "(assemblage %.3g + Cholesky %.3g)"
              % (sym["sparse_flops"], dns["total_flops"],
                 dns["assembly_flops"], dns["cholesky_flops"]))
        print("rapport dense/creux = %.1f  ->  %s" % (ratio, verdict))
    return out
