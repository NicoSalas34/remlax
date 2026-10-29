"""Cholesky creuse et inversion selective supernodale, avec SciPy seul.

POURQUOI CE MODULE. Le moteur par les equations du modele mixte (mme.py) a
besoin, a chaque iteration, de log|C|, de resolutions et de C^-1 lue sur le motif
de C. Pour une C creuse (pedigree), l'inverse entiere serait dense et hors de
portee ; l'inversion selective de Takahashi ne calcule C^-1 que sur le motif du
facteur, pour un cout proche d'une factorisation. CHOLMOD n'est pas disponible
dans l'environnement Python du paquet (ni scikit-sparse), d'ou SuperLU (SciPy).

LA FACTORISATION. SuperLU factorise P C P' sans pivotage et en mode symetrique :
U = D L', donc C_p = L D L' et le facteur de Cholesky vaut L D^(1/2).
L'ordre P (minimum de degre sur C + C') est calcule une fois, par un premier
appel ; ensuite on factorise C deja permutee dans l'ordre naturel, si bien que
le motif de L ne change pas d'une iteration a l'autre.

LES SUPERNOEUDS. Des colonnes consecutives de meme structure forment un
supernoeud, stocke en bloc dense (lignes x colonnes, ordre colonne). Les petits
supernoeuds sont fusionnes avec leur parent quand le remplissage ajoute reste
modere (regles de CHOLMOD) : les blocs deviennent assez grands pour que BLAS
travaille, au prix de quelques zeros explicites.

L'INVERSION SELECTIVE. Z = (L L')^-1, supernoeud par supernoeud, du dernier au
premier :
    Lt   = L_RJ L_JJ^-1
    Z_RJ = - Z_RR Lt
    Z_JJ = (L_JJ L_JJ')^-1 - Lt' Z_RJ
R : lignes de J sous son bloc diagonal ; Z_RR est dans le motif des supernoeuds
ancetres, deja calcules.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_triangular


_NOYAU = None


def _noyau_c():
    """Noyau C de l'inversion selective (selinv_c.c), compile au premier usage.

    Le source est livre avec le paquet ; on le compile avec le compilateur C du
    systeme dans un cache (~/.cache/remlax, ou $REMLAX_CACHE), sans aucune
    bibliotheque : BLAS et LAPACK sont ceux de SciPy, passes par pointeurs. En
    l'absence de compilateur, la boucle NumPy prend le relais (meme resultat,
    5 a 15 fois plus lente). MESURE : pedigree m = 20031, 0,3 s contre 3,9 s.
    """
    global _NOYAU
    if _NOYAU is not None:
        return _NOYAU or None
    try:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selinv_c.c")
        with open(src, "rb") as f:
            h = hashlib.sha1(f.read()).hexdigest()[:12]
        caches = [os.environ.get("REMLAX_CACHE"),
                  os.path.join(os.path.expanduser("~"), ".cache", "remlax"),
                  os.path.join(os.environ.get("TMPDIR", "/tmp"), "remlax_cache")]
        so = None
        for c in caches:
            if not c:
                continue
            try:
                os.makedirs(c, exist_ok=True)
            except OSError:
                continue
            cand = os.path.join(c, "libselinv_%s.so" % h)
            if os.path.exists(cand) or os.access(c, os.W_OK):
                so = cand
                break
        if so is None:
            raise RuntimeError("aucun cache inscriptible")
        if not os.path.exists(so):
            cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
            if cc is None:
                raise RuntimeError("pas de compilateur C")
            tmp = "%s.%d.tmp" % (so, os.getpid())
            subprocess.run([cc, "-O2", "-shared", "-fPIC", src, "-o", tmp],
                           check=True, capture_output=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        fn = lib.rx_selinv_brut
        fn.restype = ctypes.c_int
        P = ctypes.POINTER
        fn.argtypes = [ctypes.c_int, P(ctypes.c_longlong), P(ctypes.c_int), P(ctypes.c_int),
                       P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double), P(ctypes.c_double),
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        from scipy.linalg import cython_blas, cython_lapack
        api = ctypes.pythonapi
        api.PyCapsule_GetName.restype = ctypes.c_char_p
        api.PyCapsule_GetName.argtypes = [ctypes.py_object]
        api.PyCapsule_GetPointer.restype = ctypes.c_void_p
        api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        def ptr(mod, nom):
            cap = mod.__pyx_capi__[nom]
            return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
        _NOYAU = (fn, ptr(cython_blas, "dgemm"), ptr(cython_blas, "dtrsm"),
                  ptr(cython_lapack, "dpotri"))
    except Exception:
        _NOYAU = False
    return _NOYAU or None


def _options_cholmod():
    """Options de CHOLMOD (globales dans cvxopt) : supernodal, sans post-ordre,
    et UNE seule methode d'ordre, celle fournie (nmethods = 1). Par defaut
    CHOLMOD compare l'ordre fourni a AMD et garde le meilleur, ce qui changeait
    l'ordre du facteur a notre insu."""
    from cvxopt import cholmod
    cholmod.options["supernodal"] = 2
    cholmod.options["postorder"] = False
    cholmod.options["nmethods"] = 1


def _splu(A, ordre):
    return spla.splu(A, permc_spec=ordre, diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def ordre_metis(A):
    """Ordre de dissection emboitee (METIS, via pymetis) du motif symetrique A,
    ou None si pymetis manque.

    POURQUOI. L'ordre de minimum de degre de SuperLU (MMD) se perd des que C
    contient des lignes tres remplies (termes genetiques a parente dense,
    voisinages IGE) : sur mvD7 (m = 32 869), le facteur atteignait 340 millions
    d'entrees et SuperLU abandonnait faute de memoire. METIS coupe le graphe de
    C par des separateurs, recursivement : 86 millions d'entrees, 0,7 Go,
    20 fois moins d'operations (mesure CHOLMOD, 2026-10-26). Rend perm, l'ordre
    a appliquer (ancienne inconnue en position k : perm[k]).
    """
    try:
        import pymetis
    except ImportError:
        return None
    G = sp.csr_matrix(A, dtype=np.float64)
    G = (G - sp.diags(G.diagonal())).tocsr()
    G.eliminate_zeros()
    if G.nnz == 0:
        return np.arange(A.shape[0], dtype=np.int64)
    try:
        adj = pymetis.CSRAdjacency(G.indptr.astype(np.int32), G.indices.astype(np.int32))
        perm, _ = pymetis.nested_dissection(adjacency=adj)
    except (AttributeError, TypeError):
        perm, _ = pymetis.nested_dissection(xadj=G.indptr, adjncy=G.indices)
    return np.asarray(perm, dtype=np.int64)


class FacteurCreux:
    """Factorisations successives de matrices de MEME motif.

    motif : matrice creuse symetrique (motif complet, les deux triangles).
    ii, jj : entrees voulues de C^-1 (indices d'origine, base 0).
    """

    def __init__(self, motif, ii=None, jj=None, ordre="auto", moteur="auto"):
        A = sp.csc_matrix(motif, dtype=np.float64).copy()
        A.data[:] = 1.0
        m = A.shape[0]
        self.m = m
        # matrice de meme motif, diagonalement dominante : sert a l'analyse
        dg = np.asarray(abs(A).sum(axis=1)).ravel() + 1.0
        A0 = (A - sp.diags(A.diagonal()) + sp.diags(dg)).tocsc()
        # ORDRE D'ELIMINATION : dissection emboitee de METIS quand pymetis est
        # installe (ordre="auto" ou "metis"), sinon minimum de degre de SuperLU.
        pm = ordre_metis(A0) if ordre in ("auto", "metis") else None
        if ordre == "metis" and pm is None:
            raise ImportError("ordre METIS demande mais pymetis n'est pas installe")
        self.ordre = "metis" if pm is not None else "mmd"
        if pm is not None:
            self.perm = pm
        else:
            lu0 = _splu(A0, "MMD_AT_PLUS_A")
            # CONVENTION DE SciPy : perm_c[i] est la NOUVELLE position de l'ancienne
            # colonne i (Pr A Pc = L U, Pc[i, perm_c[i]] = 1). L'ordre a appliquer,
            # ancienne colonne en position k, est donc argsort(perm_c). Lu a
            # l'envers, la permutation detruisait l'ordre de minimum de degre : facteur
            # presque dense (393 millions d'entrees pour m = 20031) et 150 s par appel.
            self.perm = np.argsort(np.asarray(lu0.perm_c, dtype=np.int64))
        self.pinv = np.empty(m, dtype=np.int64); self.pinv[self.perm] = np.arange(m)
        # MOTEUR DE FACTORISATION : CHOLMOD (via cvxopt, qui l'embarque) quand il
        # est installe, sinon SuperLU. CHOLMOD exploite la symetrie (un seul
        # facteur, la moitie de la memoire d'une LU) et n'a pas la limite
        # interne de SuperLU, qui refuse les facteurs de plus de ~2 Go : sur
        # mvD7, SuperLU echoue meme avec l'ordre METIS (86 M entrees par facteur)
        # quand CHOLMOD factorise en 8,6 s (mesure 2026-10-26).
        self.moteur = "superlu"
        if moteur in ("auto", "cholmod"):
            try:
                import cvxopt  # noqa: F401
                from cvxopt import cholmod  # noqa: F401
                self.moteur = "cholmod"
            except ImportError:
                if moteur == "cholmod":
                    raise
        Ap0 = A0[self.perm][:, self.perm].tocsc()
        if self.moteur == "cholmod":
            try:
                self._init_cholmod(Ap0)
                if ii is not None:
                    self.preparer_extraction(ii, jj)
                    self._entrees_cholmod(ii, jj)
                return
            except RuntimeError:
                if moteur == "cholmod":
                    raise
                self.moteur = "superlu"           # repli : SuperLU, meme ordre
        lu1 = _splu(Ap0, "NATURAL")
        if not (np.array_equal(lu1.perm_c, np.arange(m)) and np.array_equal(lu1.perm_r, np.arange(m))):
            # SuperLU a encore permute (post-ordre de l'arbre) : on compose
            q = np.asarray(lu1.perm_c, dtype=np.int64)
            self.perm = self.perm[np.argsort(q)]
            self.pinv[self.perm] = np.arange(m)
            Ap0 = A0[self.perm][:, self.perm].tocsc()
            lu1 = _splu(Ap0, "NATURAL")
        L = sp.csc_matrix(lu1.L)
        L.sort_indices()
        self._L_indices = L.indices.copy(); self._L_indptr = L.indptr.copy()
        self._preparer_supernoeuds(L)
        self.lu = None
        if ii is not None:
            self.preparer_extraction(ii, jj)

    # ------------------------------------------------------------------ CHOLMOD
    @staticmethod
    def _cvx(vals, I, J, m):
        from cvxopt import spmatrix, matrix
        return spmatrix(matrix(np.ascontiguousarray(vals, dtype=np.float64)), I, J, (m, m))

    def _init_cholmod(self, Ap0):
        """Analyse symbolique de CHOLMOD dans NOTRE ordre (identite sur la matrice
        deja permutee, sans post-ordre), puis motif du facteur L pour les
        supernoeuds de l'inversion selective."""
        from cvxopt import cholmod, matrix
        m = self.m
        _options_cholmod()
        # Valeurs hors diagonale aleatoires : avec des 1 partout, des annulations
        # exactes feraient disparaitre des entrees du motif rendu par getfactor.
        S = sp.triu(Ap0, 1).tocoo()
        v = np.random.default_rng(0).uniform(0.5, 1.5, S.nnz)
        S = sp.coo_matrix((v, (S.row, S.col)), shape=(m, m))
        S = S + S.T
        dg = np.asarray(S.sum(axis=1)).ravel() + 1.0
        T = sp.tril(S + sp.diags(dg)).tocoo()
        I = matrix(T.row.astype(np.int64)); J = matrix(T.col.astype(np.int64))
        A = self._cvx(T.data, I, J, m)
        self._p_id = matrix(np.arange(m, dtype=np.int64))
        self.F = cholmod.symbolic(A, p=self._p_id)
        cholmod.numeric(A, self.F)
        colptr, rowind, val = (np.array(v).ravel() for v in cholmod.getfactor(self.F).CCS)
        colptr = colptr.astype(np.int64); rowind = rowind.astype(np.int64)
        # lignes triees dans chaque colonne ? sinon on retient l'ordre une fois
        col = np.repeat(np.arange(m), np.diff(colptr))
        ordre = np.lexsort((rowind, col))
        self._Lordre = None if np.array_equal(ordre, np.arange(len(ordre))) else ordre
        if self._Lordre is not None:
            rowind = rowind[ordre]
        L = sp.csc_matrix((np.ones(len(rowind)), rowind, colptr), shape=(m, m))
        self._L_indices = L.indices.copy(); self._L_indptr = L.indptr.copy()
        self._diag_pos = self._L_indptr[:-1]                 # diagonale en tete de colonne
        self._cles_L = np.repeat(np.arange(m, dtype=np.int64), np.diff(self._L_indptr)) * m + self._L_indices
        if not np.array_equal(self._L_indices[self._diag_pos], np.arange(m)):
            raise RuntimeError("CHOLMOD : diagonale absente en tete de colonne")
        # GARDE-FOU : le facteur doit etre celui de NOTRE ordre (L L' v = A v).
        # Sans nmethods = 1, CHOLMOD essayait aussi ses propres ordres et gardait
        # le moins couteux : sur un C dense a 40 % (residuelle ar1 x ar1), le
        # facteur rendu etait dans un autre ordre et l'inversion selective
        # echouait.
        Lv = sp.csc_matrix((val[self._Lordre] if self._Lordre is not None else val,
                            self._L_indices, self._L_indptr), shape=(m, m))
        v = np.random.default_rng(1).normal(size=m)
        Af = (sp.tril(S + sp.diags(dg)) + sp.triu(S, 1)).tocsr()
        if np.linalg.norm(Lv @ (Lv.T @ v) - Af @ v) > 1e-8 * np.linalg.norm(Af @ v):
            raise RuntimeError("CHOLMOD : le facteur n'est pas dans l'ordre impose")
        self._preparer_supernoeuds(L)
        self.lu = None

    def _entrees_cholmod(self, ii, jj):
        """Indices permutes, triangle inferieur, des entrees (ii, jj) de C : les
        valeurs de C passent alors directement a CHOLMOD, sans reconstruire la
        matrice creuse complete a chaque evaluation."""
        # MOTIF FIXE, VALEURS RECOPIEES. Construire un spmatrix de cvxopt a
        # partir de triplets (doublons sommes) coutait 60 s par evaluation sur
        # mvD7 (108 M entrees de L, ~2 M de C), plus que tout le reste. Le motif
        # est donc pose une fois, dans l'ordre CCS de cvxopt (colonne puis
        # ligne), et chaque evaluation ne recopie que les valeurs, sommees par
        # bincount sur la position CCS de chaque entree.
        from cvxopt import matrix, spmatrix
        a = self.pinv[np.asarray(ii, dtype=np.int64)]
        b = self.pinv[np.asarray(jj, dtype=np.int64)]
        I = np.maximum(a, b); J = np.minimum(a, b)
        cle = J * self.m + I                                 # CCS : colonne majeure
        uniq, pos = np.unique(cle, return_inverse=True)
        self._E_pos = pos.astype(np.int64)
        self._E_nnz = len(uniq)
        Iu = (uniq % self.m).astype(np.int64); Ju = (uniq // self.m).astype(np.int64)
        self._A = spmatrix(matrix(np.ones(len(uniq))), matrix(Iu), matrix(Ju), (self.m, self.m))
        # cvxopt range ses valeurs en CCS trie : verifie une fois
        Ic = np.array(self._A.I).ravel(); Jc = np.array(self._A.J).ravel()
        if not (np.array_equal(Ic, Iu) and np.array_equal(Jc, Ju)):
            raise RuntimeError("CHOLMOD : ordre CCS de cvxopt inattendu")

    def factoriser_valeurs(self, x):
        """Factorise C donnee par ses valeurs aux entrees (ii, jj). Rend log|C|."""
        from cvxopt import matrix
        v = np.bincount(self._E_pos, weights=np.asarray(x, dtype=np.float64),
                        minlength=self._E_nnz)
        self._A.V = matrix(v)
        return self._numerique_cholmod(self._A)

    def _numerique_cholmod(self, A):
        from cvxopt import cholmod
        # getfactor convertit le facteur en place et le laisse symbolique
        # (cholmod_factor_to_sparse) : on refait l'analyse a chaque fois. Elle
        # est lineaire en nnz(A) dans notre ordre fixe, negligeable devant la
        # factorisation numerique.
        try:
            _options_cholmod()
            self.F = cholmod.symbolic(A, p=self._p_id)
            cholmod.numeric(A, self.F)
        except ArithmeticError:
            raise np.linalg.LinAlgError("C n'est pas definie positive (CHOLMOD)")
        colptr, rowind, val = (np.array(v).ravel() for v in cholmod.getfactor(self.F).CCS)
        if len(val) == len(self._L_indices) and self._Lordre is None and \
                np.array_equal(rowind, self._L_indices):
            pass                                          # meme motif qu'a l'analyse
        else:
            # La conversion du facteur supernodal ecarte les zeros numeriques
            # exacts : le motif rendu est INCLUS dans celui de l'analyse (faite
            # sur une matrice diagonalement dominante). On replace chaque valeur
            # par sa cle colonne x ligne, comme pour SuperLU.
            col = np.repeat(np.arange(self.m, dtype=np.int64), np.diff(colptr.astype(np.int64)))
            cle = col * self.m + rowind.astype(np.int64)
            pos = np.searchsorted(self._cles_L, cle)
            if np.any(pos >= len(self._cles_L)) or np.any(self._cles_L[np.minimum(pos, len(self._cles_L) - 1)] != cle):
                raise RuntimeError("CHOLMOD : le motif du facteur sort de celui de l'analyse")
            v2 = np.zeros(len(self._L_indices)); v2[pos] = val
            val = v2
        d = val[self._diag_pos]
        if np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise np.linalg.LinAlgError("C n'est pas definie positive")
        self._Lval = val
        self._flat = np.zeros(self.off[-1])
        self._flat[self._carte] = val
        return float(2.0 * np.sum(np.log(d)))

    def _resoudre_supernoeuds(self, Bp):
        """L L' X = Bp par supernoeuds : un trsm et un gemm BLAS par bloc. La
        resolution de SciPy (spsolve_triangular) avance ligne a ligne, sans
        BLAS : MESURE sur mvD7 (823 supernoeuds, 108 M entrees, AI a 198
        seconds membres), elle dominait l'evaluation."""
        from scipy.linalg import solve_triangular
        X = np.array(Bp, dtype=np.float64, order="C")
        un = X.ndim == 1
        if un:
            X = X[:, None]
        blocs = []
        for s_ in range(self.ns):
            nc, nr = int(self.ncol[s_]), int(self.nrow[s_])
            blk = self._flat[self.off[s_]:self.off[s_ + 1]].reshape(nc, nr).T
            blocs.append((int(self.sup[s_]), int(self.sup[s_ + 1]), blk[:nc], blk[nc:], self.lignes[s_][nc:]))
        for c0, c1, Ljj, Lr, R in blocs:
            Xc = solve_triangular(Ljj, X[c0:c1], lower=True, check_finite=False)
            X[c0:c1] = Xc
            if len(R):
                X[R] -= Lr @ Xc
        for c0, c1, Ljj, Lr, R in reversed(blocs):
            Xc = X[c0:c1]
            if len(R):
                Xc = Xc - Lr.T @ X[R]
            X[c0:c1] = solve_triangular(Ljj, Xc, lower=True, trans="T", check_finite=False)
        return X[:, 0] if un else X

    # ------------------------------------------------------------------ structure
    def _preparer_supernoeuds(self, L):
        m = self.m
        ip, ix = L.indptr, L.indices
        cnt = np.diff(ip)
        # supernoeuds fondamentaux
        debut = [0]
        for j in range(1, m):
            jm = j - 1
            r0 = ix[ip[jm]:ip[jm + 1]]
            # j prolonge le supernoeud si L[j, j-1] est structurellement non nul
            # et si les deux colonnes ont la meme structure sous la diagonale
            if cnt[jm] == cnt[j] + 1 and len(r0) > 1 and r0[1] == j:
                continue
            debut.append(j)
        debut.append(m)
        sup = np.array(debut, dtype=np.int64)
        lignes = []
        for s in range(len(sup) - 1):
            j0 = sup[s]
            lignes.append(ix[ip[j0]:ip[j0 + 1]].astype(np.int64))      # lignes de la 1re colonne
        # fusion relaxee avec le parent contigu (regles de CHOLMOD)
        def trapeze(nr, nc):          # entrees d'un bloc trapeze nr x nc
            return nc * nr - nc * (nc - 1) // 2
        fus_sup, fus_lig = [sup[0]], []
        cur_c0, cur_rows = sup[0], lignes[0]
        cur_struct = trapeze(len(lignes[0]), sup[1] - sup[0])
        for s in range(1, len(sup) - 1):
            c0, rows = sup[s], lignes[s]
            ncol_cur = c0 - cur_c0
            parent = len(cur_rows) > ncol_cur and cur_rows[ncol_cur] == c0
            fusion = False
            s_struct = trapeze(len(rows), sup[s + 1] - c0)
            if parent:
                # les lignes de J sous son bloc sont dans {colonnes de J+1} U
                # lignes de J+1 (propriete du facteur) : l'union est le bloc de
                # J+1 prolonge par les colonnes de J
                ncol = int(sup[s + 1] - cur_c0)
                nr_new = ncol_cur + len(rows)
                apres = trapeze(nr_new, ncol)
                zeros = (apres - cur_struct - s_struct) / max(apres, 1)
                fusion = (ncol <= 4 or (ncol <= 16 and zeros <= 0.8) or
                          (ncol <= 48 and zeros <= 0.1) or zeros <= 0.05)
            if fusion:
                cur_rows = np.concatenate([np.arange(cur_c0, c0, dtype=np.int64), rows])
                cur_struct = cur_struct + s_struct
            else:
                fus_lig.append(cur_rows); fus_sup.append(c0)
                cur_c0, cur_rows, cur_struct = c0, rows, s_struct
        fus_lig.append(cur_rows); fus_sup.append(m)
        self.sup = np.array(fus_sup, dtype=np.int64)
        self.lignes = fus_lig
        ns = len(self.sup) - 1
        self.ns = ns
        self.ncol = np.diff(self.sup)
        self.nrow = np.array([len(r) for r in fus_lig], dtype=np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.nrow * self.ncol)]).astype(np.int64)
        self.sn_of = np.repeat(np.arange(ns), self.ncol)
        # carte des entrees de L (ordre CSC) vers la rangee des blocs
        col_of = np.repeat(np.arange(m), cnt)
        J = self.sn_of[col_of]
        lrow = np.empty(len(ix), dtype=np.int64)
        for s in range(ns):
            a, b = ip[self.sup[s]], ip[self.sup[s + 1]]
            lrow[a:b] = np.searchsorted(self.lignes[s], ix[a:b])
        lcol = col_of - self.sup[J]
        self._carte = self.off[J] + lrow + self.nrow[J] * lcol
        self._col_of = col_of
        self._cles = col_of * m + ix.astype(np.int64)        # croissantes (CSC trie)
        # tableaux du noyau C (rangement CHOLMOD supernodal)
        self._c_sup = self.sup.astype(np.int32)
        self._c_pi = np.concatenate([[0], np.cumsum(self.nrow)]).astype(np.int32)
        self._c_s = np.concatenate(self.lignes).astype(np.int32)
        self._c_off = self.off.astype(np.int64)
        self._c_sn = self.sn_of.astype(np.int32)
        # rassemblement de Z_RR : pour chaque J, les supernoeuds K qui portent
        # les colonnes de R, et les positions dans leurs blocs
        self._rass = []
        for s in range(ns):
            nJ = self.ncol[s]; R = self.lignes[s][nJ:]
            if not len(R):
                self._rass.append(None); continue
            Ks = self.sn_of[R]
            groupes = []
            for K in np.unique(Ks):
                pc = np.where(Ks == K)[0]
                pr = np.where(R >= R[pc[0]])[0]
                rK = np.searchsorted(self.lignes[K], R[pr])
                if np.any(rK >= self.nrow[K]) or np.any(self.lignes[K][np.minimum(rK, self.nrow[K] - 1)] != R[pr]):
                    raise RuntimeError("selinv : motif non cordal (supernoeuds %d et %d)" % (s, K))
                cK = R[pc] - self.sup[K]
                groupes.append((int(K), pr, pc, rK, cK))
            self._rass.append(groupes)

    def preparer_extraction(self, ii, jj):
        """Indices, dans la rangee des blocs, des entrees voulues de C^-1."""
        pi_, pj_ = self.pinv[np.asarray(ii, dtype=np.int64)], self.pinv[np.asarray(jj, dtype=np.int64)]
        lo, co = np.maximum(pi_, pj_), np.minimum(pi_, pj_)
        K = self.sn_of[co]
        r = np.empty(len(lo), dtype=np.int64)
        for k in np.unique(K):
            sel = np.where(K == k)[0]
            rr = np.searchsorted(self.lignes[k], lo[sel])
            if np.any(rr >= self.nrow[k]) or np.any(self.lignes[k][np.minimum(rr, self.nrow[k] - 1)] != lo[sel]):
                raise RuntimeError("selinv : entree voulue hors du motif du facteur")
            r[sel] = rr
        self._ext = self.off[K] + r + self.nrow[K] * (co - self.sup[K])

    # ----------------------------------------------------------------- numerique
    def factoriser(self, C):
        """Factorise C (meme motif que a la construction). Rend log|C|."""
        Cp = sp.csc_matrix(C)[self.perm][:, self.perm].tocsc()
        if self.moteur == "cholmod":
            from cvxopt import matrix
            T = sp.tril(Cp).tocoo()
            return self._numerique_cholmod(self._cvx(T.data, matrix(T.row.astype(np.int64)),
                                                     matrix(T.col.astype(np.int64)), self.m))
        lu = _splu(Cp, "NATURAL")
        if not (np.array_equal(lu.perm_c, np.arange(self.m)) and np.array_equal(lu.perm_r, np.arange(self.m))):
            raise np.linalg.LinAlgError("selinv : SuperLU a permute, factorisation non symetrique")
        d = lu.U.diagonal()
        if np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise np.linalg.LinAlgError("C n'est pas definie positive")
        self.lu, self.d = lu, d
        return float(np.sum(np.log(d)))

    def resoudre(self, B):
        B = np.asarray(B, dtype=np.float64)
        if self.moteur == "cholmod":
            Xp = self._resoudre_supernoeuds(B[self.perm])
            X = np.empty_like(Xp); X[self.perm] = Xp
            return X
        Xp = self.lu.solve(B[self.perm])
        X = np.empty_like(Xp); X[self.perm] = Xp
        return X

    def inverse_selective(self):
        """C^-1 aux entrees preparees (preparer_extraction)."""
        if self.moteur == "cholmod":
            return self._selinv_flat(self._flat.copy())
        L = sp.csc_matrix(self.lu.L)
        L.sort_indices()
        if np.array_equal(L.indptr, self._L_indptr) and np.array_equal(L.indices, self._L_indices):
            carte, col_of = self._carte, self._col_of
        else:
            # SuperLU ecarte parfois des zeros exacts (40 entrees sur 2,2 millions
            # sur le modele IGE ble) : le motif reel est INCLUS dans celui de
            # l'analyse, et on y retrouve chaque entree par sa cle colonne x ligne.
            col_of = np.repeat(np.arange(self.m), np.diff(L.indptr))
            cle = col_of * self.m + L.indices
            pos = np.searchsorted(self._cles, cle)
            if np.any(pos >= len(self._cles)) or np.any(self._cles[np.minimum(pos, len(self._cles) - 1)] != cle):
                raise RuntimeError("selinv : le motif du facteur sort de celui de l'analyse")
            carte = self._carte[pos]
        flat = np.zeros(self.off[-1])
        flat[carte] = L.data * np.sqrt(self.d)[col_of]
        return self._selinv_flat(flat)

    def _selinv_flat(self, flat):
        """Inversion selective a partir du facteur de Cholesky range par
        supernoeuds (flat) ; rend C^-1 aux entrees preparees."""
        noyau = _noyau_c()
        if noyau is not None:
            fn, pg, pt, pp = noyau
            Z = np.zeros(self.off[-1])
            P = ctypes.POINTER
            st = fn(int(self.ns), self._c_off.ctypes.data_as(P(ctypes.c_longlong)),
                    self._c_sup.ctypes.data_as(P(ctypes.c_int)), self._c_pi.ctypes.data_as(P(ctypes.c_int)),
                    self._c_s.ctypes.data_as(P(ctypes.c_int)), self._c_sn.ctypes.data_as(P(ctypes.c_int)),
                    flat.ctypes.data_as(P(ctypes.c_double)), Z.ctypes.data_as(P(ctypes.c_double)),
                    pg, pt, pp)
            if st != 0:
                raise np.linalg.LinAlgError("selinv (noyau C) : statut %d" % st)
            return Z[self._ext]
        Zb = [None] * self.ns
        for s in range(self.ns - 1, -1, -1):
            nJ, nr = int(self.ncol[s]), int(self.nrow[s])
            Lb = flat[self.off[s]:self.off[s + 1]].reshape((nr, nJ), order="F")
            Ljj = np.tril(Lb[:nJ])
            Linv = solve_triangular(Ljj, np.eye(nJ), lower=True, check_finite=False)
            Zjj = Linv.T @ Linv
            if nr > nJ:
                nR = nr - nJ
                Lt = Lb[nJ:] @ Linv                                   # L_RJ L_JJ^-1
                Zrr = np.zeros((nR, nR))
                for (K, pr, pc, rK, cK) in self._rass[s]:
                    Zrr[np.ix_(pr, pc)] = Zb[K][np.ix_(rK, cK)]
                il = np.triu_indices(nR, 1)
                Zrr[il] = Zrr.T[il]
                ZRJ = -Zrr @ Lt
                Zjj = Zjj - Lt.T @ ZRJ
                Zb[s] = np.vstack([Zjj, ZRJ])
            else:
                Zb[s] = Zjj
        Z = np.empty(self.off[-1])
        for s in range(self.ns):
            Z[self.off[s]:self.off[s + 1]] = Zb[s].ravel(order="F")
        return Z[self._ext]
