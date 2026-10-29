"""Moteur par les EQUATIONS DU MODELE MIXTE (MME), AI-REML a la maniere d'asreml.

Le moteur ne forme jamais V. Avec W = [X Z] :
    C = W' R^-1 W + diag(0, G^-1),      G = (+)_k Sigma_k (x) K_k
    -2 logL = (n-p) log 2pi + log|R| + log|G| + log|C| + log|det R_x|^2 + y'Py
    y'Py = y'R^-1 y - r' C^-1 r,         r = W' R^-1 y
(X est remplacee par Q de sa decomposition QR : model.orthonormal_X.)

INTERFACE. Comme model.make_objective et ai.make_fga, ce module rend la valeur,
le gradient et la matrice AI en THETA, avec les conventions du moteur dense
(memes parametres, memes structures, memes bornes). L'optimiseur AI en echelle
des variances, les replis, le polissage et les diagnostics de fit.py servent
donc tels quels aux deux moteurs.

LES DERIVEES sont prises d'abord par rapport aux coordonnees INTERNES (les
entrees de chaque Sigma et les parametres de niveaux), ou les formules sont
simples, puis ramenees a theta par la jacobienne exacte de JAX :
    g_theta = J' s,      AI_theta = Q_theta' P Q_theta,  Q_theta = Q J.
  Terme k, entree (a, b) de Sigma_k :
    M = q Sigma^-1 - Sigma^-1 (T + U'K^-1U) Sigma^-1,  T_ab = <C^-1_{kk,ab}, K^-1>
    score : M_aa, ou 2 M_ab ; colonne de travail Z vec(U Sigma^-1 E_ab).
  Parametre de niveau i du terme k :
    score : t dlog|K| + <Sigma^-1, T'> + <Sigma^-1, U' dK^-1 U> ;
    colonne de travail Z vec(dK K^-1 U).
  Residuelle par unites, motif g : H_g = S^-1 (n_g S - E'E - D) S^-1.
  Residuelle structuree (R dense par section) :
    tr(R^-1 dR) - <C^-1, W'R^-1 dR R^-1 W> - Py' dR Py ; colonne dR Py.
  AI = Q'R^-1Q - (W'R^-1Q)' C^-1 (W'R^-1Q).

C EST UNE COMBINAISON DE COMPOSANTES sur un motif commun : composantes FIXES
(motifs residuels W_a'W_b, parties creuses de K^-1) dont seul le coefficient
change, et composantes VARIABLES (K^-1 dense d'une structure parametree, bloc
W'R^-1W d'une residuelle structuree) dont les valeurs changent. Les traces du
score sont des produits scalaires entre C^-1 lue sur ce motif et les composantes.

INVERSION. C dense : LAPACK par JAX, sur le peripherique choisi (GPU possible).
C creuse : Cholesky creuse de SuperLU et inversion selective supernodale
(selinv.py), sur CPU.

HORS PERIMETRE (bascule vers le moteur dense, cf. hors_perimetre) : residuelle
"none" (les equations exigent R^-1), structure rr (Sigma singuliere, sans
inverse) et decalage spatial (y transforme par un parametre).
"""
from . import _x64  # noqa: F401
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .model import (orthonormal_X, split_theta, term_n_params, res_sections,
                    sec_n_params, _section_V)
from .structures import build_sigma, n_params
from .levels import level_chol
from .selinv import FacteurCreux

LOG2PI = float(np.log(2.0 * np.pi))


# GARDE DE CONDITIONNEMENT des Sigma inversees par le moteur MME : rapport
# minimal des diagonales du facteur de Cholesky (1e-5, soit un conditionnement
# d'environ 1e10). Les optima au bord du cone restent admissibles : le plancher
# relatif des blocs us y est de 1e-8 sur les valeurs propres (ratio ~1e-4).
DIAG_CHOL_MIN = 1e-5


# FONCTIONS DE SIGMA COMPILEES, gardees par structure (type, t, rang). Elles ne
# dependent que de la structure : dans le solveur persistant (serveur.py), un
# second ajustement de meme structure les reprend au lieu de les recompiler
# (0,7 s par ajustement sur un modele a deux termes iid).
_SIGMA_CACHE = {}


def _fonctions_sigma(st, t, rk):
    cle = (st, int(t), int(rk))
    if cle not in _SIGMA_CACHE:
        jpa = jnp.asarray(_paires(st, t))

        def vech(th):
            S = build_sigma(th, st, t, rk)
            return S[jpa[:, 0], jpa[:, 1]]
        _SIGMA_CACHE[cle] = (jax.jit(lambda th: build_sigma(th, st, t, rk)),
                             jax.jit(jax.jacfwd(vech)))
    return _SIGMA_CACHE[cle]


def hors_perimetre(terms, res, lag=None):
    """Raison pour laquelle le modele releve du moteur dense, ou None."""
    if lag is not None:
        return "decalage spatial (rx_lag) : y transforme par un parametre"
    if res.get("struct") == "none":
        return "residuelle 'none' : les equations du modele mixte exigent R^-1"
    for tm in terms:
        if tm["struct"] == "rr":
            return "structure rr du terme '%s' : Sigma singuliere, sans inverse" % tm["name"]
    for sec in res_sections(res, 0):
        if sec["struct"] == "rr":
            return "structure rr de la residuelle : singuliere"
    # NOYAUX GAUSSIENS. Une correlation gaussienne est quasi singuliere des que
    # deux niveaux sont proches : le moteur dense n'en a besoin que d'un facteur
    # (avec un jitter), mais le MME l'INVERSE, et K^-1 perd alors ses chiffres.
    # Verifie par simulation : logLik fausse de 3 a 7 % sur gau, igau et agau.
    gauss = ("gau", "igau", "agau")
    for tm in terms:
        lvl = tm.get("lvl")
        if lvl in gauss or (lvl == "sep" and any(k in gauss for k, _ in (tm.get("lvl_parts") or []))):
            return "noyau gaussien du terme '%s' : K quasi singuliere, son inverse est inexploitable" % tm["name"]
    for sec in res_sections(res, 0):
        if sec.get("lvl") in gauss:
            return "noyau gaussien de la residuelle : R quasi singuliere"
    return None


# ------------------------------------------------------------------------------
# Structures de niveaux en forme creuse fermee (produits de id et d'ar1)
# ------------------------------------------------------------------------------
def _ar1_parts(n):
    """A^-1 = c0 I + c1 D + c2 T pour A_ij = rho^|i-j| (D : diagonale interieure)."""
    I = sp.identity(n, format="csr")
    dd = np.ones(n)
    dd[0] = 0.0; dd[-1] = 0.0
    D = sp.diags(dd, format="csr")
    T = (sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr")
         if n > 1 else sp.csr_matrix((1, 1)))
    return [I, D, T]


def _ar1_c(r):
    u = 1.0 - r * r
    return np.array([1.0 / u, r * r / u, -r / u])


def _ar1_dc(r):                                       # d c / d rho
    u = 1.0 - r * r
    return np.array([2 * r / u ** 2, 2 * r / u ** 2, -(1 + r * r) / u ** 2])


def _ar1_corr(r, n):
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return r ** h


def _ar1_dcorr(r, n):
    h = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.where(h == 0, 0.0, h * r ** np.maximum(h - 1, 0))


class _NiveauxCreux:
    """K = (x)_f A_f, chaque facteur id ou ar1, rho = tanh(theta) comme dans le
    moteur dense (levels.ar1_chol). K^-1 = somme de produits de Kronecker de
    parties fixes ; seuls leurs coefficients dependent de rho."""

    def __init__(self, facteurs):
        self.fac = [(k, int(d)) for k, d in facteurs]
        self.ar1 = [i for i, (k, _) in enumerate(self.fac) if k == "ar1"]
        self.np = len(self.ar1)
        self.q = int(np.prod([d for _, d in self.fac]))
        listes = [(_ar1_parts(d) if k == "ar1" else [sp.identity(d, format="csr")])
                  for (k, d) in self.fac]
        self.combi = [()]
        for L in listes:
            self.combi = [c + (j,) for c in self.combi for j in range(len(L))]
        self.parts = []
        for c in self.combi:
            M = None
            for f, j in enumerate(c):
                M = listes[f][j] if M is None else sp.kron(M, listes[f][j], format="csr")
            self.parts.append(sp.csr_matrix(M))

    def coef(self, th, i_der=None):
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        out = np.ones(len(self.combi))
        for h, c in enumerate(self.combi):
            for j, f in enumerate(self.ar1):
                if j == i_der:
                    out[h] *= _ar1_dc(rho[j])[c[f]] * (1.0 - rho[j] ** 2)
                else:
                    out[h] *= _ar1_c(rho[j])[c[f]]
        return out

    def logdet(self, th):
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        return sum((self.q // self.fac[f][1]) * (self.fac[f][1] - 1) * np.log(1 - rho[i] ** 2)
                   for i, f in enumerate(self.ar1))

    def dlogdet(self, th, i):
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        d = self.fac[self.ar1[i]][1]
        return -(self.q // d) * (d - 1) * 2.0 * rho[i]       # d/dtheta, rho' = 1 - rho^2

    def appliquer_dK(self, th, i, B):
        """(dK / dtheta_i) B sans matrice q x q : facteur par facteur ; K B si i est None."""
        rho = np.tanh(np.asarray(th, dtype=np.float64))
        mats = []
        for j, (k, d) in enumerate(self.fac):
            if k == "id":
                mats.append(None)
                continue
            ia = self.ar1.index(j)
            mats.append(_ar1_dcorr(rho[ia], d) * (1.0 - rho[ia] ** 2) if ia == i
                        else _ar1_corr(rho[ia], d))
        dims = [d for _, d in self.fac]
        B = np.asarray(B, dtype=np.float64)
        out = np.empty_like(B)
        for c in range(B.shape[1]):
            T = B[:, c].reshape(dims)
            for ax, Mx in enumerate(mats):
                if Mx is not None:
                    T = np.moveaxis(np.tensordot(Mx, T, axes=([1], [ax])), 0, ax)
            out[:, c] = T.ravel()
        return out


def _facteurs_creux(tm):
    lvl = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    if lvl == "ar1":
        return [("ar1", int(tm["q"]))]
    if lvl == "ar1ar1":
        return [("ar1", int(tm["dims"][0])), ("ar1", int(tm["dims"][1]))]
    if lvl == "sep" and tm.get("lvl_parts") and all(k in ("id", "ar1") for k, _ in tm["lvl_parts"]):
        return [(k, int(d)) for k, d in tm["lvl_parts"]]
    return None


def _paires(st, t):
    if st in ("iid", "diag"):
        return np.column_stack([np.arange(t), np.arange(t)])
    return np.array([(a, b) for a in range(t) for b in range(a + 1)], dtype=np.int64).reshape(-1, 2)


class _FaitGPU(Exception):
    """Sortie du bloc de factorisation : la voie GPU par blocs a tout calcule."""


def _inverse_depuis_facteur(LK):
    """K^-1 a partir d'un facteur K = LK LK'.

    Facteur triangulaire (le cas de rx_term, Cholesky de K) : LK^-1 par
    substitution (dtrsm, BLAS 3 parallele), puis K^-1 = LK^-T LK^-1 par dsyrk
    (moitie d'un produit). La version precedente resolvait LK X = I par une LU
    generale : 2,6 a 5 s pour 3768 genotypes. Le caractere triangulaire n'est
    pas teste sur la matrice entiere (np.triu sur 3768 x 3768 coutait 2,6 s) :
    on essaie, et on verifie K^-1 (LK LK' v) = v sur un vecteur aleatoire.
    Facteur quelconque, ou verification ratee : LU generale.
    """
    import scipy.linalg as _sl
    q = LK.shape[0]

    def plein(Li):
        Ki = _sl.blas.dsyrk(1.0, Li, trans=1)        # triangle superieur de Li' Li
        Ki += Ki.T
        Ki[np.diag_indices(q)] *= 0.5
        return Ki

    if LK.shape[0] == LK.shape[1] and q > 1:
        v = np.random.default_rng(0).normal(size=q)
        w = LK @ (LK.T @ v)
        # premiere ligne (resp. colonne) nulle hors diagonale : candidat
        # triangulaire inferieur (resp. superieur). Un facteur qui ne l'est pas
        # donne un residu d'ordre 1 ; un facteur triangulaire, un residu de
        # l'ordre de cond(K) x 1e-16.
        for bas, ok in ((True, (LK[0, 1:] == 0).all()), (False, (LK[1:, 0] == 0).all())):
            if not ok:
                continue
            Ki = plein(_sl.solve_triangular(LK, np.eye(q), lower=bas, check_finite=False))
            if np.linalg.norm(Ki @ w - v) <= 1e-4 * np.linalg.norm(v):
                return Ki
    Li = np.linalg.solve(LK, np.eye(q))
    return plein(Li)


def _bloc_vals(M, diag):
    """Valeurs d'un bloc q x q dans l'ordre des triplets de sa composante :
    triangle superieur ligne par ligne (bloc diagonal), bloc entier sinon."""
    if diag:
        return M[np.triu_indices(M.shape[0])]
    return M.ravel()


# ------------------------------------------------------------------------------
# Le moteur
# ------------------------------------------------------------------------------
class MME:
    def __init__(self, terms, res, y, X, inversion="auto", verbose=True, separateur=None):
        t0 = time.time()
        raison = hors_perimetre(terms, res)
        if raison:
            raise ValueError("moteur MME : " + raison)
        self.verbose = verbose
        self.y = y = np.asarray(y, dtype=np.float64)
        self.X = X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        self.n, self.p = n, p
        self.p_sortie = p
        Xq, self.ldRx = orthonormal_X(X)
        self.Rx = np.linalg.qr(X)[1]
        self.terms, self.res = terms, res
        self._n_obs0 = n                      # dispositif d'origine (decoupage hybride)
        self._separateur = separateur         # noms des termes du separateur impose (hybride)
        # RESIDUELLE SUR GRILLE (ar1, ar1 x ar1, produit id/ar1), une section iid a
        # un caractere : R^-1 = sigma^-2 (x) A_f^-1 est creuse et connue en forme
        # fermee. Les observations sont rangees dans leur cellule ; chaque cellule
        # MANQUANTE recoit un effet fixe indicateur qui l'absorbe. La REML du
        # modele augmente est exactement celle des donnees observees :
        # log|V_F| + log|M'V_F^-1 M| = log|V_O| (complement de Schur), et
        # n_F - p_F = n - p (verifie a 1e-13 sur un champ ar1 x ar1 a 15 % de
        # cellules vides). C devient creuse au lieu de recevoir un bloc
        # W'R^-1W dense, et R n'est plus jamais inversee.
        self.grille = None
        secs0 = res_sections(res, n)
        if len(secs0) == 1 and secs0[0]["struct"] == "iid" and int(secs0[0]["t"]) == 1:
            s0 = secs0[0]
            fac0 = _facteurs_creux(dict(s0, q=int(s0.get("n_unit", 0) or 0)))
            un0 = np.asarray(s0.get("unit", []), dtype=np.int64)
            if fac0 is not None and len(un0) == n:
                nF = int(np.prod([dd for _, dd in fac0]))
                if (nF > 0 and un0.min() >= 0 and un0.max() < nF
                        and len(np.unique(un0)) == n):
                    mis = np.setdiff1d(np.arange(nF), un0)
                    yF = np.zeros(nF); yF[un0] = y
                    XqF = np.zeros((nF, p + len(mis)))
                    XqF[un0, :p] = Xq
                    XqF[mis, p + np.arange(len(mis))] = 1.0
                    self.grille = dict(un=un0, mis=mis, nF=nF, fac=fac0)
                    y = self.y = yF
                    Xq = XqF
                    n, p = nF, p + len(mis)
                    self.n, self.p = n, p
        self.T = []
        off = p
        for tm in terms:
            t, q = int(tm["t"]), int(tm["q"])
            st, rk = tm["struct"], int(tm["rank"])
            nsig = n_params(st, t, rk)
            nlv = term_n_params(tm) - nsig
            pa = _paires(st, t)
            sig_f, sig_j = _fonctions_sigma(st, t, rk)
            d = dict(name=tm["name"], t=t, q=q, struct=st, rank=rk, nsig=nsig, nlv=nlv, off=off,
                     paires=pa, Z=sp.csc_matrix(
                         (tm["zx"], (self._ligne(tm["zi"]), tm["zj"])), shape=(n, t * q)),
                     sig=sig_f, sig_jac=sig_j)
            lvl = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
            fac = _facteurs_creux(tm)
            if tm.get("Kinv") is not None:
                # precision fournie (pedigree) : quel que soit le niveau declare
                d.update(forme="fixe", parts=[sp.csr_matrix(tm["Kinv"])],
                         ldK=float(tm["Kinv_logdet"]))
            elif lvl == "id":
                d.update(forme="fixe", parts=[sp.identity(q, format="csr")], ldK=0.0)
            elif lvl == "fixed":
                LK = np.asarray(tm["LK"], dtype=np.float64)
                d.update(forme="fixe", parts=[sp.csr_matrix(_inverse_depuis_facteur(LK))],
                         ldK=float(2.0 * np.sum(np.log(np.abs(np.diag(LK))))))
            elif fac is not None:
                nc = _NiveauxCreux(fac)
                if nc.q != q or nc.np != nlv:
                    raise ValueError("moteur MME : niveaux '%s' du terme '%s' incoherents"
                                     % (lvl, tm["name"]))
                d.update(forme="creux", niv=nc, parts=nc.parts)
            else:
                # K EXACTEMENT COMME LE MOTEUR DENSE : par le facteur de
                # level_chol, jitter compris. level_corr seul donnerait une autre
                # matrice des que le jitter compte, donc une autre vraisemblance.
                def Kde(th, lvl=lvl, q=q, tm=tm):
                    L = level_chol(th, lvl, q, dims=tm.get("dims"), order=tm.get("lvl_order", 0),
                                   coord=tm.get("coord"), parts=tm.get("lvl_parts"),
                                   opts=tm.get("lvl_opts"), expr=tm.get("lvl_expr"))
                    return L @ L.T
                d.update(forme="dense", K=jax.jit(Kde), dK=jax.jit(jax.jacfwd(Kde)),
                         parts=[sp.csr_matrix(np.ones((q, q)))])
            self.T.append(d)
            off += t * q
        self.m = off
        self.S = []
        for sec in res_sections(res, n):
            rows = np.asarray(sec["rows"], dtype=np.int64)
            st, t, rk = sec["struct"], int(sec["t"]), int(sec["rank"])
            nsig = n_params(st, t, rk)
            lvl = sec.get("lvl", "id")
            d = dict(rows=rows, struct=st, t=t, nsig=nsig, np_=sec_n_params(sec), lvl=lvl)
            if self.grille is not None:
                niv = _NiveauxCreux(self.grille["fac"])
                d.update(forme="grille", niv=niv, parts=niv.parts)
                self.S.append(d)
                continue
            if lvl == "id":
                pa = _paires(st, t)
                sig_f, sig_j = _fonctions_sigma(st, t, rk)
                d.update(paires=pa, sig=sig_f, sig_jac=sig_j)
                tr = np.asarray(sec["trait"], dtype=np.int64)
                un = np.asarray(sec["unit"], dtype=np.int64)
                par_u = {}
                for i_, (u_, a_) in enumerate(zip(un, tr)):
                    par_u.setdefault(int(u_), {})[int(a_)] = int(rows[i_])
                grp = {}
                for dct in par_u.values():
                    tau = tuple(sorted(dct))
                    grp.setdefault(tau, []).append([dct[a_] for a_ in tau])
                d["motifs"] = [dict(tau=np.array(tau, dtype=np.int64), I=np.array(Ls, dtype=np.int64),
                                    n=len(Ls)) for tau, Ls in grp.items()]
            else:
                d.update(R=jax.jit(lambda th, sec=sec: _section_V(th, sec)),
                         dR=jax.jit(jax.jacfwd(lambda th, sec=sec: _section_V(th, sec))))
            self.S.append(d)
        self.W = sp.hstack([sp.csc_matrix(Xq)] + [d["Z"] for d in self.T], format="csc")
        self._composantes()
        self._choisir_inversion(inversion)
        if verbose:
            m = self.m
            print("  [mme] m = %d | motif de C %.1f %% | inversion %s | %d composantes | preparation %.1f s"
                  % (m, 100.0 * min(1.0, (self.nU + self._n_entrees_blocs) / (m * (m + 1) / 2)),
                     self.inversion, self.n_comp, time.time() - t0), flush=True)

    def _ligne(self, i):
        """Ligne de W d'une observation : sa cellule en mode grille."""
        i = np.asarray(i, dtype=np.int64)
        return i if self.grille is None else self.grille["un"][i]

    # ------------------------------------------------------------------ motif de C
    def _composantes(self, differer=True):
        """Motif de C et composantes. differer : les blocs pleins (parente
        dense) n'y entrent pas encore ; ils ne servent qu'aux inversions creuse
        et hybride, et _choisir_inversion les materialise alors (sur Krause,
        178 millions d'entrees et 18 s de preparation pour rien en dense)."""
        self._blocs_differes = False
        self._n_entrees_blocs = 0
        W, m, y = self.W, self.m, self.y
        I_, J_, X_, var = [], [], [], []

        def ajouter(i, j, x, variable=False):
            I_.append(np.asarray(i, dtype=np.int64)); J_.append(np.asarray(j, dtype=np.int64))
            X_.append(np.asarray(x, dtype=np.float64)); var.append(variable)
            return len(I_) - 1

        def triu_trip(M):
            T = sp.triu(sp.csr_matrix(M)).tocoo()
            keep = T.data != 0
            return T.row[keep], T.col[keep], T.data[keep]
        for d in self.S:
            if d.get("forme") == "grille":
                d["ic"], d["Wy"], d["yPy"] = [], [], []
                for P in d["parts"]:
                    PW = P @ W
                    d["ic"].append(ajouter(*triu_trip(W.T @ PW)))
                    d["Wy"].append(np.asarray(PW.T @ y).ravel())
                    d["yPy"].append(float(y @ (P @ y)))
                d["yPy"] = np.asarray(d["yPy"])
                continue
            if d["lvl"] == "id":
                for mo in d["motifs"]:
                    tg = len(mo["tau"])
                    Wa = [W[mo["I"][:, a], :] for a in range(tg)]
                    Ya = y[mo["I"]]
                    mo["Wy"] = [np.asarray(Wa[a].T @ Ya) for a in range(tg)]
                    mo["yy"] = Ya.T @ Ya
                    mo["ic"] = -np.ones((tg, tg), dtype=np.int64)
                    for a in range(tg):
                        for b in range(a, tg):
                            M = Wa[a].T @ Wa[b]
                            if a != b:
                                M = M + M.T
                            mo["ic"][a, b] = mo["ic"][b, a] = ajouter(*triu_trip(M))
            else:
                Ws = sp.csr_matrix(W[d["rows"], :])          # par lignes : indices = colonnes
                cols = np.unique(Ws.indices)
                d["cols"], d["Ws"] = cols, Ws[:, cols].toarray()
                ii, jj = np.triu_indices(len(cols))
                d["triu"] = (ii, jj)
                d["ic"] = ajouter(cols[ii], cols[jj], np.zeros(len(ii)), True)
                d["ys"] = y[d["rows"]]
        self._bloc_de = {}                  # composante -> (terme, part, a, b) d'un bloc plein
        for k_, d in enumerate(self.T):
            t, q = d["t"], d["q"]
            plein = d["struct"] not in ("iid", "diag")
            d["ic"] = -np.ones((t, t, len(d["parts"])), dtype=np.int64)
            for h, P in enumerate(d["parts"]):
                Pt = sp.csr_matrix(P).tocoo()
                # BLOC PLEIN (parente dense, noyau) : en inversion dense, il
                # s'ajoute a C d'un seul tenant (voir _preparer_blocs)
                bloc_plein = Pt.nnz >= 0.5 * q * q and q >= 16
                for a in range(t):
                    for b in range(a, t):
                        if not plein and a != b:
                            continue
                        ia, ib = d["off"] + a * q, d["off"] + b * q
                        if bloc_plein and differer:
                            vide = np.zeros(0)
                            j = ajouter(vide, vide, vide, d["forme"] == "dense")
                            self._blocs_differes = True
                            self._n_entrees_blocs += (q * (q + 1)) // 2 if a == b else q * q
                        elif a == b:
                            sel = Pt.row <= Pt.col
                            j = ajouter(ia + Pt.row[sel], ia + Pt.col[sel], Pt.data[sel], d["forme"] == "dense")
                        else:
                            j = ajouter(ia + Pt.row, ib + Pt.col, Pt.data, d["forme"] == "dense")
                        d["ic"][a, b, h] = d["ic"][b, a, h] = j
                        if bloc_plein:
                            self._bloc_de[j] = (k_, h, a, b)
        ii = np.concatenate(I_ + [np.arange(m)])
        jj = np.concatenate(J_ + [np.arange(m)])
        U = sp.csc_matrix((np.ones(len(ii)), (ii, jj)), shape=(m, m))
        U.sum_duplicates(); U.sort_indices()
        self.ci = U.indices.astype(np.int64)
        self.cj = np.repeat(np.arange(m), np.diff(U.indptr)).astype(np.int64)
        self._U_indptr = U.indptr
        cle = self.cj * m + self.ci
        lens = np.array([len(x) for x in X_], dtype=np.int64)
        self.n_comp = len(X_)
        self.c_debut = np.concatenate([[0], np.cumsum(lens)])
        self.c_pos = np.searchsorted(cle, np.concatenate(J_) * m + np.concatenate(I_))
        self.c_id = np.repeat(np.arange(self.n_comp), lens)
        self.c_val0 = np.concatenate(X_)
        self.c_var = np.array(var, dtype=bool)
        self.nU = len(cle)
        self.poids = np.where(self.ci == self.cj, 1.0, 2.0)

    def _choisir_inversion(self, inversion):
        m = self.m
        self.facteur = None
        self._mode_blocs = False
        n_U = self.nU + getattr(self, "_n_entrees_blocs", 0)       # blocs differes compris
        if inversion == "auto" and m > 1000 and n_U <= 0.3 * m * (m + 1) / 2 \
                and getattr(self, "_blocs_differes", False):
            self._composantes(differer=False)
        if inversion in ("creuse", "hybride") and getattr(self, "_blocs_differes", False):
            self._composantes(differer=False)
        if inversion == "auto":
            if m <= 1000 or n_U > 0.3 * m * (m + 1) / 2:
                inversion = "dense"
            else:
                try:
                    F = FacteurCreux(self._motif(), self.ci, self.cj)
                    remp = F.off[-1] / (m * (m + 1) / 2)
                except MemoryError:
                    # SuperLU refuse un facteur trop rempli : C est presque pleine
                    F, remp = None, 1.0
                inversion = "dense" if remp > 0.5 else "creuse"
                if inversion == "creuse":
                    self.facteur = F
        elif inversion == "creuse":
            self.facteur = FacteurCreux(self._motif(), self.ci, self.cj)
        elif inversion == "hybride":
            # SEPARATEUR DENSE + BLOCS DENSES PAR LOTS (mme_hybride.py) : le
            # separateur est fait des effets fixes et des termes globaux trouves
            # par decoupage.py ; hors perimetre, retour au choix automatique.
            from .mme_hybride import FacteurHybride, ensemble_dense, HorsPerimetre
            from .decoupage import Decoupage
            if self._separateur is not None:
                noms = [tm["name"] for tm in self.terms]
                glob = [noms.index(nm) for nm in self._separateur]
            else:
                glob = Decoupage(self.terms, self.res, self._n_obs0).chercher(
                    espace="niveaux", p=self.p_sortie)["globaux"]
            try:
                self.facteur = FacteurHybride(self.ci, self.cj, m,
                                              ensemble_dense(self.T, self.p_sortie, glob),
                                              verbose=self.verbose)
            except HorsPerimetre as e:
                if self.verbose:
                    print("  [mme] hybride hors perimetre (%s) -> inversion automatique" % e, flush=True)
                return self._choisir_inversion("auto")
        self.inversion = inversion
        if inversion == "dense":
            self._jci, self._jcj = jnp.asarray(self.ci), jnp.asarray(self.cj)
            # SUR CPU, L'ALGEBRE DENSE PASSE PAR LAPACK (SciPy) ET NON PAR XLA.
            # XLA formait C par scatter, puis C^-1 par cho_solve contre
            # l'identite (2 m^3), et chaque resultat revenait sur l'hote par
            # conversion. MESURE sur drops2 (m = 1428) : 0,34 s par evaluation
            # complete. LAPACK forme C par indexation, la factorise (dpotrf) et
            # en tire l'inverse depuis le facteur (dpotri, m^3/3 de plus). Sur
            # GPU, XLA (cuSOLVER) reste le bon choix.
            try:
                dev = next(iter(jnp.zeros(()).devices()))
                self._dense_cpu = dev.platform == "cpu"
            except Exception:
                self._dense_cpu = False
            if self._bloc_de:
                self._preparer_blocs()

    # ------------------------------------------------------- inversion dense par blocs
    def _preparer_blocs(self):
        """Inversion dense sur CPU avec des blocs pleins (parente dense, noyau).

        Chaque composante (a, b) d'un terme a parente dense porte q^2 entrees.
        Passer par le motif creux de C les regroupait par bincount a chaque
        evaluation, puis les relisait une a une dans C^-1 pour les traces :
        178 millions d'entrees sur un modele a 5 traitements x 3768 genotypes.
        Ici les blocs s'ajoutent a C d'un seul tenant (C[A, B] += Sigma^-1[a, b]
        K^-1) et leurs traces se lisent sur les blocs de C^-1 (somme du produit
        terme a terme). Les autres composantes (residuelle, effets fixes) gardent
        le motif creux, restreint a leurs entrees.
        """
        m = self.m
        self._mode_blocs = True
        est_bloc = np.zeros(self.n_comp, dtype=bool)
        est_bloc[list(self._bloc_de)] = True
        ent = ~est_bloc[self.c_id]                       # entrees hors blocs
        self._sp_ent = np.nonzero(ent)[0]
        u, inv = np.unique(self.c_pos[self._sp_ent], return_inverse=True)
        self._sp_u, self._sp_inv = u, inv.astype(np.int64)
        self._sp_i, self._sp_j = self.ci[u], self.cj[u]
        self._sp_hors_diag = self._sp_i != self._sp_j
        self._sp_poids = np.where(self._sp_hors_diag, 2.0, 1.0)
        # blocs constants (parente fournie) : la matrice pleine, une fois par part
        self._P_plein = {}
        for j, (k, h, a, b) in self._bloc_de.items():
            d = self.T[k]
            if d["forme"] != "dense" and (k, h) not in self._P_plein:
                self._P_plein[(k, h)] = np.asarray(d["parts"][h].toarray(), dtype=np.float64)
        # diagonale de C (PEV) : positions directes
        self._diag_idx = np.arange(m)
        self._g = None
        if not self._dense_cpu:
            self._preparer_blocs_gpu()

    def _preparer_blocs_gpu(self):
        """Mode blocs sur GPU : C assemblee sur la carte, C^-1 n'en sort pas.

        La version precedente assemblait C sur l'hote (2,8 Go a m = 18 860),
        l'envoyait sur la carte, puis ramenait C^-1 pour les traces : 6,9 s par
        evaluation sur Krause (A100) pour ~2 s de calcul. Ici les blocs K^-1
        restent sur la carte, C y est assemblee par une fonction compilee
        (structure statique, valeurs en arguments), et seules reviennent les
        traces des blocs, les entrees de C^-1 hors blocs et sa diagonale.
        Un terme a noyau dense parametre (derivees de K) garde la voie hote.
        """
        if any(self.T[k]["forme"] == "dense" and self.T[k]["nlv"] > 0 for (k, h, a, b) in self._bloc_de.values()):
            return
        m = self.m
        cles = sorted({(k, h) for (k, h, a, b) in self._bloc_de.values()})
        idx_cle = {c: i for i, c in enumerate(cles)}
        ordre = sorted(self._bloc_de)
        # UNE OPERATION PAR TERME, et non par bloc : C[terme] += kron(S, K^-1),
        # S (t x t) portant les coefficients Sigma^-1[a, b] ; les traces de tous
        # les blocs d'un terme sortent d'un seul einsum. Deroulee bloc par bloc,
        # la fonction compilee portait 406 operations pour un fa a 28
        # environnements et sa compilation coutait ~30 s.
        groupes = []
        for (k, h) in cles:
            d = self.T[k]
            t, q = d["t"], d["q"]
            membres = [(n_, self._bloc_de[j][2], self._bloc_de[j][3]) for n_, j in enumerate(ordre)
                       if self._bloc_de[j][:2] == (k, h)]
            groupes.append((d["off"], t, q, idx_cle[(k, h)],
                            np.array([x[0] for x in membres], dtype=np.int32),
                            np.array([x[1] for x in membres], dtype=np.int32),
                            np.array([x[2] for x in membres], dtype=np.int32)))
        hd = self._sp_hors_diag
        si, sj = jnp.asarray(self._sp_i), jnp.asarray(self._sp_j)
        sih, sjh = jnp.asarray(self._sp_i[hd]), jnp.asarray(self._sp_j[hd])
        hd_j = jnp.asarray(np.nonzero(hd)[0])
        groupes = tuple(groupes)

        def assembler(coefs, xs, Ps, r):
            C = jnp.zeros((m, m), dtype=xs.dtype)
            C = C.at[si, sj].add(xs)
            C = C.at[sjh, sih].add(xs[hd_j])
            for (off, t, q, c_, n_i, a_i, b_i) in groupes:
                S = jnp.zeros((t, t), dtype=xs.dtype).at[a_i, b_i].set(coefs[n_i]).at[b_i, a_i].set(coefs[n_i])
                C = C.at[off:off + t * q, off:off + t * q].add(jnp.kron(S, Ps[c_]))
            L = jnp.linalg.cholesky(C)
            ld = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            sol = jax.scipy.linalg.cho_solve((L, True), r)
            return L, ld, sol

        def traces(L, Ps):
            Ci = jax.scipy.linalg.cho_solve((L, True), jnp.eye(m, dtype=L.dtype))
            tr = jnp.zeros(len(ordre), dtype=L.dtype)
            for (off, t, q, c_, n_i, a_i, b_i) in groupes:
                B = Ci[off:off + t * q, off:off + t * q].reshape(t, q, t, q)
                Tm = jnp.einsum("aibj,ij->ab", B, Ps[c_])
                tr = tr.at[n_i].set(Tm[a_i, b_i])
            return tr, Ci[sj, si], jnp.diag(Ci)

        self._g = dict(ordre=ordre, cles=cles, assembler=jax.jit(assembler), traces=jax.jit(traces),
                       Ps=None)

    def _Ps_gpu(self, termes):
        g = self._g
        if g["Ps"] is not None and not any(self.T[k]["forme"] == "dense" for (k, h) in g["cles"]):
            return g["Ps"]
        Ps = tuple(jnp.asarray(termes[k]["Kinv"] if self.T[k]["forme"] == "dense" else self._P_plein[(k, h)])
                   for (k, h) in g["cles"])
        g["Ps"] = Ps
        return Ps

    def _P_bloc(self, j, termes):
        k, h, a, b = self._bloc_de[j]
        if self.T[k]["forme"] == "dense":
            return termes[k]["Kinv"]
        return self._P_plein[(k, h)]

    def _assembler_blocs(self, coef, val, termes):
        m = self.m
        C = np.zeros((m, m), order="F")
        w = coef[self.c_id[self._sp_ent]] * val[self._sp_ent]
        xs = np.bincount(self._sp_inv, weights=w, minlength=len(self._sp_u))
        C[self._sp_i, self._sp_j] += xs
        h_ = self._sp_hors_diag
        C[self._sp_j[h_], self._sp_i[h_]] += xs[h_]
        for j, (k, h, a, b) in self._bloc_de.items():
            c = coef[j]
            if c == 0.0:
                continue
            d = self.T[k]
            q = d["q"]
            ia, ib = d["off"] + a * q, d["off"] + b * q
            P = self._P_bloc(j, termes)
            C[ia:ia + q, ib:ib + q] += c * P
            if a != b:
                C[ib:ib + q, ia:ia + q] += c * P.T
        return C

    def _cinv_plein(self, et):
        if "Cinv" not in et and not self._dense_cpu:
            # facteur sur le GPU : C^-1 y est formee, puis ramenee sur l'hote
            L = et["L"]
            et["Cinv"] = np.asarray(jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.m, dtype=L.dtype)))
        if "Cinv" not in et:
            import scipy.linalg as _sl
            Ci, info = _sl.lapack.dpotri(et["L"], lower=1)
            if info != 0:
                raise np.linalg.LinAlgError("dpotri : C singuliere")
            # dpotri ne remplit que le triangle inferieur
            Ci = np.tril(Ci)
            Ci += np.tril(Ci, -1).T
            et["Cinv"] = Ci
        return et["Cinv"]

    def _motif(self, x=None):
        m = self.m
        v = np.ones(self.nU) if x is None else x
        U = sp.csc_matrix((v, self.ci, self._U_indptr), shape=(m, m))
        return (U + sp.triu(U, 1).T).tocsc()

    def _rinv(self, B, et):
        B = np.asarray(B, dtype=np.float64)
        un = B.ndim == 1
        if un:
            B = B[:, None]
        out = np.zeros_like(B)
        for s, d in enumerate(self.S):
            if d.get("forme") == "grille":
                out[d["rows"]] = sum(ch * (P @ B[d["rows"]]) for ch, P in zip(et["Rinv"][s], d["parts"]))
                continue
            if d["lvl"] == "id":
                for g, mo in enumerate(d["motifs"]):
                    I = mo["I"]; tg = I.shape[1]
                    Bg = B[I.ravel()].reshape(mo["n"], tg, B.shape[1])
                    out[I.ravel()] = np.einsum("nbc,ba->nac", Bg, et["Sg_inv"][s][g]).reshape(-1, B.shape[1])
            else:
                out[d["rows"]] = et["Rinv"][s] @ B[d["rows"]]
        return out[:, 0] if un else out

    # ------------------------------------------------------------------ evaluation
    def evaluer(self, theta, complet=True):
        # DERNIERE EVALUATION EN CACHE. Apres l'AI, la finition puis les sorties
        # (beta, BLUP) reevaluaient le point final : deux factorisations de C de
        # plus, 27 + 11 s sur Krause us(5) en CPU. Meme theta, meme demande (ou
        # complete deja faite) : le resultat et l'etat (facteur de C) servent.
        cle = np.asarray(theta, dtype=np.float64).tobytes()
        der = getattr(self, "_dernier", None)
        if der is not None and der[0] == cle and (der[1] or not complet):
            self._etat = der[3]
            return dict(der[2])
        if der is not None and der[0] == cle and complet and der[4] is not None:
            # L'AI essaie un pas sur la valeur seule (factorisation de C), puis
            # l'evalue completement au meme point : on reprend la factorisation
            # et on ne calcule que les derivees. MESURE sur Krause fa2 en CPU :
            # ~10 s de Cholesky par iteration acceptee, sur ~38 s.
            t0_ = time.time()
            self._etat = der[3]
            g, AI = self._derivees(*der[4])
            out = dict(der[2]); out.update(g=g, AI=AI)
            self._durees = getattr(self, "_durees", [])
            if len(self._durees) < 200:
                self._durees.append(time.time() - t0_ + der[5])
            self._dernier = (cle, True, dict(out), self._etat, None, 0.0)
            return out
        t0_ = time.time()
        out = self._evaluer(theta, complet)
        if complet:
            # duree des seules evaluations completes calculees (pas du cache) :
            # c'est elle que rapporte eval_s
            self._durees = getattr(self, "_durees", [])
            if len(self._durees) < 200:
                self._durees.append(time.time() - t0_)
        if np.isfinite(out.get("f", np.inf)):
            self._dernier = (cle, complet, dict(out), self._etat,
                             None if complet else getattr(self, "_entrees_derivees", None),
                             time.time() - t0_)
        return out

    def _evaluer(self, theta, complet=True):
        theta = np.asarray(theta, dtype=np.float64)
        th_t, th_r, _ = split_theta(theta, self.terms, self.res)
        th_t = [np.asarray(v, dtype=np.float64) for v in th_t]
        th_r = np.asarray(th_r, dtype=np.float64)
        m, n, p = self.m, self.n, self.p
        et = dict(Sg_inv={}, Rinv={})
        coef = np.zeros(self.n_comp)
        val = self.c_val0.copy()
        ldG = 0.0
        termes = []
        for k, d in enumerate(self.T):
            ths, thl = th_t[k][:d["nsig"]], th_t[k][d["nsig"]:]
            Sig = np.asarray(d["sig"](jnp.asarray(ths)), dtype=np.float64)
            try:
                cS = np.linalg.cholesky(Sig)
            except np.linalg.LinAlgError:
                return dict(f=np.inf)
            dS = np.diag(cS)
            if dS.min() < DIAG_CHOL_MIN * dS.max():
                # Sigma presque singuliere : Sigma^-1 n'a plus de chiffres
                # significatifs et le gradient devient faux (fa a 5 caracteres,
                # banc 2026-10-22 : 8450 contre 148 pour le moteur dense). Le
                # point est refuse ; l'AI rejette le pas qui y mene.
                return dict(f=np.inf)
            Si = np.linalg.inv(Sig)
            info = dict(Si=Si, ths=ths, thl=thl)
            if d["forme"] == "fixe":
                cK, ldK = np.ones(1), d["ldK"]
            elif d["forme"] == "creux":
                cK, ldK = d["niv"].coef(thl), d["niv"].logdet(thl)
            else:
                K = np.asarray(d["K"](jnp.asarray(thl)), dtype=np.float64)
                try:
                    cKK = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return dict(f=np.inf)
                dg = np.diag(cKK)
                if dg.min() < 1e-6 * dg.max():
                    # conditionnement au-dela de 1e12 : K^-1 n'a plus de chiffres
                    # significatifs ; le point est traite comme inadmissible
                    return dict(f=np.inf)
                Kinv = np.linalg.inv(K)
                info["Kinv"] = Kinv
                cK, ldK = np.ones(1), 2.0 * np.sum(np.log(np.diag(cKK)))
            info["cK"] = cK
            ldG += d["q"] * 2.0 * np.sum(np.log(np.diag(cS))) + d["t"] * ldK
            for h in range(len(d["parts"])):
                for a in range(d["t"]):
                    for b in range(a, d["t"]):
                        j = d["ic"][a, b, h]
                        if j < 0:
                            continue
                        coef[j] = Si[a, b] * cK[h]
                        if d["forme"] == "dense" and not (self._mode_blocs and j in self._bloc_de):
                            val[self.c_debut[j]:self.c_debut[j + 1]] = _bloc_vals(info["Kinv"], a == b)
            termes.append(info)
        r = np.zeros(m)
        yRy = ldR = 0.0
        sections = []
        o = 0
        for s, d in enumerate(self.S):
            th_s = th_r[o:o + d["np_"]]
            o += d["np_"]
            info = dict(th=th_s)
            if d.get("forme") == "grille":
                s2 = float(np.exp(2.0 * th_s[0]))
                thl = th_s[1:]
                c = d["niv"].coef(thl) / s2
                ldR += n * np.log(s2) + d["niv"].logdet(thl)
                for h, j in enumerate(d["ic"]):
                    coef[j] = c[h]
                    r += c[h] * d["Wy"][h]
                yRy += float(c @ d["yPy"])
                et["Rinv"][s] = c
                info.update(s2=s2, thl=thl, c=c)
                sections.append(info)
                continue
            if d["lvl"] == "id":
                Sr = np.asarray(d["sig"](jnp.asarray(th_s[:d["nsig"]])), dtype=np.float64)
                info["Sr"] = Sr
                et["Sg_inv"][s] = []
                for mo in d["motifs"]:
                    Sg = Sr[np.ix_(mo["tau"], mo["tau"])]
                    try:
                        cg = np.linalg.cholesky(Sg)
                    except np.linalg.LinAlgError:
                        return dict(f=np.inf)
                    if np.diag(cg).min() < DIAG_CHOL_MIN * np.diag(cg).max():
                        return dict(f=np.inf)
                    ldR += mo["n"] * 2.0 * np.sum(np.log(np.diag(cg)))
                    Sgi = np.linalg.inv(Sg)
                    et["Sg_inv"][s].append(Sgi)
                    yRy += float(np.sum(Sgi * mo["yy"]))
                    tg = len(mo["tau"])
                    for a in range(tg):
                        r += mo["Wy"][a] @ Sgi[:, a]
                        for b in range(a, tg):
                            coef[mo["ic"][a, b]] = Sgi[a, b]
            else:
                R = np.asarray(d["R"](jnp.asarray(th_s)), dtype=np.float64)
                try:
                    cR = np.linalg.cholesky(R)
                except np.linalg.LinAlgError:
                    return dict(f=np.inf)
                Ri = np.linalg.inv(R)
                ldR += 2.0 * np.sum(np.log(np.diag(cR)))
                A = Ri @ d["Ws"]
                j = d["ic"]
                coef[j] = 1.0
                val[self.c_debut[j]:self.c_debut[j + 1]] = (d["Ws"].T @ A)[d["triu"]]
                Riy = Ri @ d["ys"]
                yRy += float(d["ys"] @ Riy)
                r[d["cols"]] += d["Ws"].T @ Riy
                et["Rinv"][s] = Ri
                info.update(Ri=Ri, A=A)
            sections.append(info)
        x = None if self._mode_blocs else np.bincount(self.c_pos, weights=coef[self.c_id] * val,
                                                      minlength=self.nU)
        try:
            if self.inversion == "dense" and self._dense_cpu:
                import scipy.linalg as _sl
                if self._mode_blocs:
                    C = self._assembler_blocs(coef, val, termes)
                else:
                    C = np.zeros((m, m))
                    C[self.ci, self.cj] = x
                    C[self.cj, self.ci] = x
                L, info = _sl.lapack.dpotrf(C, lower=1, clean=1, overwrite_a=1)
                if info != 0:
                    return dict(f=np.inf)
                ldC = float(2.0 * np.sum(np.log(np.diag(L))))
                if not np.isfinite(ldC):
                    return dict(f=np.inf)
                sol = _sl.cho_solve((L, True), r, check_finite=False)
                et["L"] = L
            elif self.inversion == "dense":
                if self._mode_blocs and self._g is not None:
                    g_ = self._g
                    w_ = coef[self.c_id[self._sp_ent]] * val[self._sp_ent]
                    xs = np.bincount(self._sp_inv, weights=w_, minlength=len(self._sp_u))
                    L, ld_, sol = g_["assembler"](jnp.asarray(coef[g_["ordre"]]), jnp.asarray(xs),
                                                  self._Ps_gpu(termes), jnp.asarray(r))
                    ldC = float(ld_)
                    if not np.isfinite(ldC):
                        return dict(f=np.inf)
                    sol = np.asarray(sol)
                    et["L"] = L
                    et["termes_g"] = termes
                    raise _FaitGPU()
                if self._mode_blocs:
                    # GPU sans voie compilee : C assemblee par blocs sur l'hote
                    C = jnp.asarray(self._assembler_blocs(coef, val, termes))
                else:
                    xj = jnp.asarray(x)
                    C = jnp.zeros((m, m)).at[self._jci, self._jcj].set(xj).at[self._jcj, self._jci].set(xj)
                L = jnp.linalg.cholesky(C)
                del C
                ldC = float(2.0 * jnp.sum(jnp.log(jnp.diag(L))))
                if not np.isfinite(ldC):
                    return dict(f=np.inf)
                sol = np.asarray(jax.scipy.linalg.cho_solve((L, True), jnp.asarray(r)))
                et["L"] = L
            elif self.inversion == "hybride":
                ldC = self.facteur.factoriser_valeurs(x)
                sol = self.facteur.resoudre(r)
            elif getattr(self.facteur, "moteur", "") == "cholmod":
                # CHOLMOD : les valeurs passent directement dans l'ordre permute
                ldC = self.facteur.factoriser_valeurs(x)
                sol = self.facteur.resoudre(r)
            else:
                ldC = self.facteur.factoriser(self._motif(x))
                sol = self.facteur.resoudre(r)
        except _FaitGPU:
            pass
        except (np.linalg.LinAlgError, RuntimeError):
            return dict(f=np.inf)
        f = (n - p) * LOG2PI + ldR + ldG + ldC + self.ldRx + yRy - float(r @ sol)
        out = dict(f=float(f), sol=sol)
        self._etat = et
        if not complet:
            # de quoi calculer les derivees plus tard au meme point (evaluer)
            self._entrees_derivees = (et, termes, sections, sol, val)
            return out
        self._entrees_derivees = None
        g, AI = self._derivees(et, termes, sections, sol, val)
        out.update(g=g, AI=AI)
        return out

    # ------------------------------------------------------------------ derivees
    def _resoudre(self, et, B):
        if self.inversion == "dense" and self._dense_cpu:
            import scipy.linalg as _sl
            return _sl.cho_solve((et["L"], True), np.asarray(B, dtype=np.float64), check_finite=False)
        if self.inversion == "dense":
            return np.asarray(jax.scipy.linalg.cho_solve((et["L"], True), jnp.asarray(B)))
        return self.facteur.resoudre(B)

    def _cinv_motif(self, et):
        if self.inversion == "dense" and self._dense_cpu:
            import scipy.linalg as _sl
            Ci, info = _sl.lapack.dpotri(et["L"], lower=1)
            if info != 0:
                raise np.linalg.LinAlgError("dpotri : C singuliere")
            # dpotri ne remplit que le triangle inferieur : (ci, cj) est le
            # triangle superieur de C (ci <= cj), lu en (cj, ci)
            return Ci[self.cj, self.ci]
        if self.inversion == "dense":
            Ci = jax.scipy.linalg.cho_solve((et["L"], True), jnp.eye(self.m))
            return np.asarray(Ci[self._jci, self._jcj])
        return self.facteur.inverse_selective()

    def _derivees(self, et, termes, sections, sol, val):
        n = self.n
        if self._mode_blocs and self._g is not None:
            # MODE BLOCS SUR GPU : traces et entrees de C^-1 calculees sur la carte
            tr_g, cv_u_g, dg_g = self._g["traces"](et["L"], self._Ps_gpu(termes))
            tr_g, cv_u, et["Cinv_diag"] = np.asarray(tr_g), np.asarray(cv_u_g), np.asarray(dg_g)
            e_ = self._sp_ent
            tr_comp = np.bincount(self.c_id[e_], weights=(cv_u * self._sp_poids)[self._sp_inv] * val[e_],
                                  minlength=self.n_comp)
            for n_, j in enumerate(self._g["ordre"]):
                k_, h_, a_, b_ = self._bloc_de[j]
                tr_comp[j] = float(tr_g[n_]) * (2.0 if a_ != b_ else 1.0)

            def cv_at(pos):
                return cv_u[np.searchsorted(self._sp_u, pos)]

            def trace(j, a, b, vals=None, plein=None):
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cv_at(pos) * np.where(self.ci[pos] == self.cj[pos], 1.0, 2.0)
                                    * vals)) / (1.0 if a == b else 2.0)
        elif self._mode_blocs:
            # MODE BLOCS : C^-1 pleine ; traces des blocs sur leurs blocs, des
            # autres composantes sur leurs seules entrees
            Ci = self._cinv_plein(et)
            cv_u = Ci[self._sp_j, self._sp_i]                    # aux entrees hors blocs
            e_ = self._sp_ent
            tr_comp = np.bincount(self.c_id[e_], weights=(cv_u * self._sp_poids)[self._sp_inv] * val[e_],
                                  minlength=self.n_comp)

            def bloc_ci(j):
                k, h, a, b = self._bloc_de[j]
                d = self.T[k]; q = d["q"]
                ia, ib = d["off"] + a * q, d["off"] + b * q
                return Ci[ia:ia + q, ib:ib + q]

            for j in self._bloc_de:
                tr_comp[j] = float(np.sum(bloc_ci(j) * self._P_bloc(j, termes)))
                k_, h_, a_, b_ = self._bloc_de[j]
                if a_ != b_:
                    tr_comp[j] *= 2.0                            # convention : divise plus bas

            def cv_at(pos):
                return Ci[self.cj[pos], self.ci[pos]]

            def trace(j, a, b, vals=None, plein=None):
                if j in self._bloc_de:
                    if vals is None:
                        return tr_comp[j] / (1.0 if a == b else 2.0)
                    return float(np.sum(bloc_ci(j) * plein))
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cv_at(pos) * np.where(self.ci[pos] == self.cj[pos], 1.0, 2.0)
                                    * vals)) / (1.0 if a == b else 2.0)
        else:
            cv = self._cinv_motif(et)
            cvw = cv * self.poids
            tr_comp = np.bincount(self.c_id, weights=cvw[self.c_pos] * val, minlength=self.n_comp)

            def cv_at(pos):
                return cv[pos]

            def trace(j, a, b, vals=None, plein=None):
                if vals is None:
                    return tr_comp[j] / (1.0 if a == b else 2.0)
                pos = self.c_pos[self.c_debut[j]:self.c_debut[j + 1]]
                return float(np.sum(cvw[pos] * vals)) / (1.0 if a == b else 2.0)

        e = self.y - self.W @ sol
        Py = self._rinv(e, et)
        blocs_J, s_int, Q = [], [], []
        for k, d in enumerate(self.T):
            info = termes[k]
            Si, q, t = info["Si"], d["q"], d["t"]
            U = sol[d["off"]:d["off"] + t * q].reshape(q, t, order="F")
            nparts = len(d["parts"])
            Tp = np.zeros((t, t, nparts))
            for h in range(nparts):
                for a in range(t):
                    for b in range(a, t):
                        j = d["ic"][a, b, h]
                        if j >= 0:
                            Tp[a, b, h] = Tp[b, a, h] = trace(j, a, b)
            if d["forme"] == "dense":
                KiU = info["Kinv"] @ U
                Tm = Tp[:, :, 0]
            else:
                PU = [P @ U for P in d["parts"]]
                KiU = sum(info["cK"][h] * PU[h] for h in range(nparts))
                Tm = np.tensordot(Tp, info["cK"], axes=([2], [0]))
            M = q * Si - Si @ (Tm + U.T @ KiU) @ Si
            for (a, b) in d["paires"]:
                E = np.zeros((t, t)); E[a, b] = 1.0; E[b, a] = 1.0
                s_int.append(M[a, a] if a == b else 2.0 * M[a, b])
                Q.append(d["Z"] @ (U @ Si @ E).ravel(order="F"))
            if d["nlv"]:
                if d["forme"] == "creux":
                    niv = d["niv"]
                    for i in range(d["nlv"]):
                        dc = niv.coef(info["thl"], i_der=i)
                        Td = np.tensordot(Tp, dc, axes=([2], [0]))
                        dKiU = sum(dc[h] * PU[h] for h in range(nparts))
                        s_int.append(t * niv.dlogdet(info["thl"], i) + float(np.sum(Si * Td))
                                     + float(np.sum(Si * (U.T @ dKiU))))
                        Q.append(d["Z"] @ niv.appliquer_dK(info["thl"], i, KiU).ravel(order="F"))
                else:
                    dK = np.asarray(d["dK"](jnp.asarray(info["thl"])), dtype=np.float64)
                    Ki = info["Kinv"]
                    for i in range(d["nlv"]):
                        dKi = dK[:, :, i]
                        dKinv = -Ki @ dKi @ Ki
                        Td = np.zeros((t, t))
                        for a in range(t):
                            for b in range(a, t):
                                j = d["ic"][a, b, 0]
                                if j >= 0:
                                    Td[a, b] = Td[b, a] = (
                                        trace(j, a, b, True, plein=dKinv) if (self._mode_blocs and j in self._bloc_de)
                                        else trace(j, a, b, _bloc_vals(dKinv, a == b)))
                        s_int.append(t * float(np.sum(Ki * dKi)) + float(np.sum(Si * Td))
                                     + float(np.sum(Si * (U.T @ (dKinv @ U)))))
                        Q.append(d["Z"] @ (dKi @ KiU).ravel(order="F"))
            Js = np.asarray(d["sig_jac"](jnp.asarray(info["ths"])), dtype=np.float64)
            Jk = np.zeros((len(d["paires"]) + d["nlv"], d["nsig"] + d["nlv"]))
            Jk[:len(d["paires"]), :d["nsig"]] = Js
            Jk[len(d["paires"]):, d["nsig"]:] = np.eye(d["nlv"])
            blocs_J.append(Jk)
        for s, d in enumerate(self.S):
            info = sections[s]
            if d.get("forme") == "grille":
                # d(-2logL)/dtheta = dlog|R| + sum_h dc_h [tr(C^-1 W'P_hW) + e'P_h e],
                # R^-1 = sum_h c_h P_h ; vecteurs de travail dR/dtheta P y, P y = R^-1 e
                c, thl, s2, niv = info["c"], info["thl"], info["s2"], d["niv"]
                trh = np.array([trace(j, 0, 0) for j in d["ic"]])
                ePe = np.array([float(e @ (P @ e)) for P in d["parts"]])
                s_int.append(2.0 * n - 2.0 * float(c @ (trh + ePe)))   # theta = log sigma
                Q.append(2.0 * e)
                for i in range(niv.np):
                    dc = niv.coef(thl, i_der=i) / s2
                    s_int.append(niv.dlogdet(thl, i) + float(dc @ (trh + ePe)))
                    Q.append(s2 * niv.appliquer_dK(thl, i, Py[:, None])[:, 0])
                blocs_J.append(np.eye(d["np_"]))
                continue
            if d["lvl"] == "id":
                t = d["t"]
                H = np.zeros((t, t))
                for g, mo in enumerate(d["motifs"]):
                    tg = len(mo["tau"])
                    Sgi = et["Sg_inv"][s][g]
                    Eg = e[mo["I"].ravel()].reshape(mo["n"], tg)
                    D = np.zeros((tg, tg))
                    for a in range(tg):
                        for b in range(a, tg):
                            D[a, b] = D[b, a] = trace(mo["ic"][a, b], a, b)
                    Sg = info["Sr"][np.ix_(mo["tau"], mo["tau"])]
                    H[np.ix_(mo["tau"], mo["tau"])] += Sgi @ (mo["n"] * Sg - Eg.T @ Eg - D) @ Sgi
                for (a, b) in d["paires"]:
                    s_int.append(H[a, a] if a == b else 2.0 * H[a, b])
                    qv = np.zeros(n)
                    for mo in d["motifs"]:
                        tg = len(mo["tau"])
                        Eab = np.zeros((tg, tg))
                        ia = np.where(mo["tau"] == a)[0]; ib = np.where(mo["tau"] == b)[0]
                        if len(ia) and len(ib):
                            Eab[ia[0], ib[0]] = 1.0; Eab[ib[0], ia[0]] = 1.0
                        Pg = Py[mo["I"].ravel()].reshape(mo["n"], tg)
                        qv[mo["I"].ravel()] = (Pg @ Eab).ravel()
                    Q.append(qv)
                Js = np.asarray(d["sig_jac"](jnp.asarray(info["th"][:d["nsig"]])), dtype=np.float64)
                Jk = np.zeros((len(d["paires"]), d["np_"]))
                Jk[:, :d["nsig"]] = Js
                blocs_J.append(Jk)
            else:
                dR = np.asarray(d["dR"](jnp.asarray(info["th"])), dtype=np.float64)
                Ri, A = info["Ri"], info["A"]
                ii, jj = d["triu"]
                nc = len(d["cols"])
                Cb = np.zeros((nc, nc))
                vb = cv_at(self.c_pos[self.c_debut[d["ic"]]:self.c_debut[d["ic"] + 1]])
                Cb[ii, jj] = vb; Cb[jj, ii] = vb
                Pys = Py[d["rows"]]
                for i in range(d["np_"]):
                    dRi = dR[:, :, i]
                    s_int.append(float(np.sum(Ri * dRi)) - float(np.sum(Cb * (A.T @ dRi @ A)))
                                 - float(Pys @ dRi @ Pys))
                    qv = np.zeros(n); qv[d["rows"]] = dRi @ Pys
                    Q.append(qv)
                blocs_J.append(np.eye(d["np_"]))
        nI = sum(b.shape[0] for b in blocs_J)
        nT = sum(b.shape[1] for b in blocs_J)
        J = np.zeros((nI, nT))
        oi = oj = 0
        for b in blocs_J:
            J[oi:oi + b.shape[0], oj:oj + b.shape[1]] = b
            oi += b.shape[0]; oj += b.shape[1]
        g = J.T @ np.asarray(s_int)
        Qm = np.column_stack(Q) @ J
        RQ = self._rinv(Qm, et)
        WRQ = np.asarray(self.W.T @ RQ)
        AI = Qm.T @ RQ - WRQ.T @ self._resoudre(et, WRQ)
        return g, 0.5 * (AI + AI.T)

    # ---------------------------------------------------------------- sorties finales
    def sorties(self, theta, pev=()):
        """beta (echelle de X), vbeta, BLUP par terme (q x t) et, pour les termes
        nommes dans `pev`, la variance d'erreur de prediction diagonale.

        PEV. C = W'R^-1W + diag(0, G^-1) est formee sur les vraies R et G, donc
        le bloc u de C^-1 est Var(u - u_chapeau) (Henderson 1975), comme dans
        asreml. Sa diagonale est lue dans l'inversion selective (le motif de C
        contient toujours la diagonale) ou dans l'inverse dense. Remplacer X par
        Q (decomposition QR) ne change pas ce bloc.
        """
        ev = self.evaluer(theta, complet=False)
        sol, p = ev["sol"], self.p_sortie
        E = np.zeros((self.m, p))
        E[np.arange(p), np.arange(p)] = 1.0
        Cqq = self._resoudre(self._etat, E)[:p, :]
        Ri = np.linalg.inv(self.Rx)
        vb = Ri @ Cqq @ Ri.T
        out = dict(beta=Ri @ sol[:p], vbeta=0.5 * (vb + vb.T),
                   blups={d["name"]: sol[d["off"]:d["off"] + d["t"] * d["q"]].reshape(d["q"], d["t"], order="F")
                          for d in self.T})
        noms = [d["name"] for d in self.T] if pev is True else list(pev or [])
        if noms:
            inconnus = set(noms) - {d["name"] for d in self.T}
            if inconnus:
                raise ValueError("pev : terme(s) inconnu(s) %s" % sorted(inconnus))
            if not hasattr(self, "_pos_diag"):
                diag = np.where(self.ci == self.cj)[0]
                self._pos_diag = np.full(self.m, -1, dtype=np.int64)
                self._pos_diag[self.ci[diag]] = diag
            out["pev"] = {}
            if self._mode_blocs:
                if self._g is not None:
                    if "Cinv_diag" not in self._etat:
                        self._etat["Cinv_diag"] = np.asarray(
                            self._g["traces"](self._etat["L"], self._Ps_gpu(self._etat["termes_g"]))[2])
                    dg = self._etat["Cinv_diag"]
                else:
                    dg = np.diag(self._cinv_plein(self._etat))
                for d in self.T:
                    if d["name"] in noms:
                        out["pev"][d["name"]] = np.maximum(dg[d["off"]:d["off"] + d["t"] * d["q"]], 0.0).reshape(
                            d["q"], d["t"], order="F")
                return out
            cv = self._cinv_motif(self._etat)
            for d in self.T:
                if d["name"] in noms:
                    pos = self._pos_diag[d["off"]:d["off"] + d["t"] * d["q"]]
                    out["pev"][d["name"]] = np.maximum(np.asarray(cv)[pos], 0.0).reshape(
                        d["q"], d["t"], order="F")
        return out


def make_fga_mme(terms, res, y, X, inversion="auto", verbose=True):
    """Rend (fga, fun_jac, moteur) : fga(theta) -> (f, g, AI) ; fun_jac(theta) -> (f, g)."""
    M = MME(terms, res, y, X, inversion=inversion, verbose=verbose)

    def fga(theta):
        r = M.evaluer(np.asarray(theta), complet=True)
        if not np.isfinite(r["f"]):
            k = len(theta)
            return np.inf, np.full(k, np.nan), np.full((k, k), np.nan)
        return r["f"], r["g"], r["AI"]

    def fun_jac(theta):
        f, g, _ = fga(theta)
        return float(f), np.asarray(g, dtype=np.float64)
    return fga, fun_jac, M
