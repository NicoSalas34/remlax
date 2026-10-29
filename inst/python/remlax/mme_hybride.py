"""Inversion HYBRIDE de C pour le moteur MME : separateur dense, blocs denses par lots.

LE DECOUPAGE. Les inconnues de C se partagent en deux ensembles :
  D : les effets fixes et les niveaux des termes GLOBAUX (ceux qui relient les
      blocs d'observations, trouves par decoupage.py) ;
  S : tout le reste.
Une fois D retire, la partie Css se decompose en composantes connexes
independantes (les bacs d'un essai, les environnements...). Le remplissage les
rend presque pleines : chacune est factorisee en DENSE, et les composantes de
tailles voisines sont traitees ensemble (vmap), comme dans blocs.py.

L'ELIMINATION (meme algebre que kmmr, Melsen et al. 2026, sections 3.3 a 3.6) :
  Css = blocs L_b L_b'           log|C| = somme log|Css_b| + log|Sch|
  Q   = Css^-1 Csd               Sch = Cdd - Cds Q  (complement de Schur, dense)
  C x = r :  g = Css^-1 r_s ;  x_d = Sch^-1 (r_d - Q' r_s) ;  x_s = g - Q x_d
  C^-1 :  dd = Sch^-1 ;  sd = -Q Sch^-1 ;  ss = Css^-1 + Q Sch^-1 Q'
Cds Css^-1 r_s = Q' r_s : Csd n'a pas a etre garde apres la factorisation.
Seules les entrees de C^-1 sur le motif de C sont rendues (inverse_selective),
comme pour l'inversion creuse ; les blocs ss sont calcules en dense par bloc.

DIFFERENCE AVEC kmmr. kmmr choisit le dense par une regle fixe (une structure
de Kronecker dense) et factorise la partie creuse par CHOLMOD sur CPU. Ici le
separateur vient du decoupage automatique, et la partie creuse, faite de blocs
independants, reste sur le GPU en lots denses.

PERIMETRE (version 1). Les composantes de Css doivent rester de taille moderee
(TAILLE_MAX_BLOC) ; sinon HorsPerimetre est leve et le moteur MME garde ses
inversions dense ou creuse.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import jax
import jax.numpy as jnp

from .blocs import _paquets

TAILLE_MAX_BLOC = 8000


class HorsPerimetre(Exception):
    pass


def ensemble_dense(MME_T, p_dense, globaux):
    """Indices de C du separateur : effets fixes [0, p_dense) et niveaux des
    termes globaux (indices dans l'ordre des termes de mme.MME)."""
    idx = [np.arange(p_dense, dtype=np.int64)]
    for k in globaux:
        d = MME_T[k]
        idx.append(np.arange(d["off"], d["off"] + d["t"] * d["q"], dtype=np.int64))
    return np.unique(np.concatenate(idx))


class FacteurHybride:
    """Meme interface que selinv.FacteurCreux : factoriser, resoudre,
    inverse_selective (entrees de C^-1 aux positions (ci, cj) du motif)."""

    def __init__(self, ci, cj, m, dense_idx, taille_max=TAILLE_MAX_BLOC, verbose=False):
        ci = np.asarray(ci, dtype=np.int64); cj = np.asarray(cj, dtype=np.int64)
        self.m = m
        isD = np.zeros(m, dtype=bool); isD[np.asarray(dense_idx, dtype=np.int64)] = True
        D = np.flatnonzero(isD); S = np.flatnonzero(~isD)
        self.D, self.S = D, S
        nD, nS = len(D), len(S)
        self.nD = nD
        dpos = np.full(m, -1, dtype=np.int64); dpos[D] = np.arange(nD)
        spos = np.full(m, -1, dtype=np.int64); spos[S] = np.arange(nS)
        cls_dd = isD[ci] & isD[cj]
        cls_ss = ~isD[ci] & ~isD[cj]
        cls_sd = ~(cls_dd | cls_ss)
        # composantes de Css
        e = np.flatnonzero(cls_ss)
        G = sp.coo_matrix((np.ones(len(e), dtype=np.int8), (spos[ci[e]], spos[cj[e]])), shape=(nS, nS))
        nc, lab = connected_components(G, directed=False)
        tailles = np.bincount(lab, minlength=nc)
        if nS and tailles.max() > taille_max:
            raise HorsPerimetre("composante de %d inconnues apres retrait du separateur (max %d)"
                                % (tailles.max(), taille_max))

        def cout(t):
            return t ** 3 / 3.0 + 2.0 * t * t * nD + t * nD * nD
        groupes = _paquets(tailles, cout) if nS else []
        # position de chaque inconnue de S : (paquet, emplacement, rang local)
        paq = np.full(nS, -1, dtype=np.int64); slot = np.zeros(nS, dtype=np.int64)
        loc = np.zeros(nS, dtype=np.int64)
        self.paquets = []
        for g_i, comps in enumerate(groupes):
            nmax = int(tailles[comps].max())
            lignes = -np.ones((len(comps), nmax), dtype=np.int64)
            for a, c in enumerate(comps):
                mem = np.flatnonzero(lab == c)
                paq[mem] = g_i; slot[mem] = a; loc[mem] = np.arange(len(mem))
                lignes[a, :len(mem)] = mem
            self.paquets.append(dict(nb=len(comps), m=nmax, lignes=lignes,
                                     masque=(lignes >= 0).astype(np.float64)))
        # entrees du motif, par classe
        self.e_dd = np.flatnonzero(cls_dd)
        self.dd_r, self.dd_c = dpos[ci[self.e_dd]], dpos[cj[self.e_dd]]
        es = np.flatnonzero(cls_sd)
        s_of = np.where(isD[ci[es]], cj[es], ci[es])
        d_of = np.where(isD[ci[es]], ci[es], cj[es])
        self.sd = []
        self.ss = []
        for g_i in range(len(self.paquets)):
            sel = paq[spos[s_of]] == g_i
            self.sd.append(dict(e=es[sel], b=slot[spos[s_of[sel]]], i=loc[spos[s_of[sel]]], d=dpos[d_of[sel]]))
            sel2 = paq[spos[ci[e]]] == g_i
            ee = e[sel2]
            self.ss.append(dict(e=ee, b=slot[spos[ci[ee]]], i=loc[spos[ci[ee]]], j=loc[spos[cj[ee]]]))
        self.n_entrees = len(ci)
        # rang dans S, par paquet, de chaque ligne de chaque bloc (pour resoudre)
        self._spos = spos
        self._dpos = dpos
        self._paq, self._slot, self._loc = paq, slot, loc
        self._construire()
        if verbose:
            print("  [mme] hybride : separateur %d inconnues | %d blocs en %d paquets (%s)"
                  % (nD, nc, len(self.paquets),
                     ", ".join("%d x %d" % (P["nb"], P["m"]) for P in self.paquets)), flush=True)

    # ------------------------------------------------------------------ calcul
    def _construire(self):
        # LES INDICES SONT DES ARGUMENTS des fonctions compilees, pas des
        # constantes capturees : XLA tenterait de replier des tableaux de
        # dizaines de millions d'entrees a la compilation (cf. model.make_neg2).
        nD = self.nD
        # entiers 32 bits : sur mvD7, ces tableaux comptent des dizaines de
        # millions d'entrees (1,6 Go en 64 bits sur le peripherique)
        i32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.int32))
        self._idx = dict(
            dd=tuple(i32(a) for a in (self.e_dd, self.dd_r, self.dd_c)),
            sd=[{k: i32(v) for k, v in d.items()} for d in self.sd],
            ss=[{k: i32(v) for k, v in d.items()} for d in self.ss],
            masque=[jnp.asarray(p["masque"]) for p in self.paquets])
        n_ent = self.n_entrees

        def cho_lot(L, B):
            return jax.vmap(lambda Lb, Bb: jax.scipy.linalg.cho_solve((Lb, True), Bb))(L, B)

        def factoriser(x, idx):
            dd_e, dd_r, dd_c = idx["dd"]
            Cdd = jnp.zeros((nD, nD)).at[dd_r, dd_c].set(x[dd_e]).at[dd_c, dd_r].set(x[dd_e])
            ld = jnp.zeros(())
            Ls, Qs = [], []
            for msk, a, b in zip(idx["masque"], idx["sd"], idx["ss"]):
                nb, m = msk.shape
                Css = jnp.zeros((nb, m, m))
                Css = Css.at[b["b"], b["i"], b["j"]].set(x[b["e"]]).at[b["b"], b["j"], b["i"]].set(x[b["e"]])
                Css = Css + jax.vmap(jnp.diag)(1.0 - msk)
                Csd = jnp.zeros((nb, m, nD)).at[a["b"], a["i"], a["d"]].set(x[a["e"]])
                L = jnp.linalg.cholesky(Css)
                ld = ld + 2.0 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(L)))
                Q = cho_lot(L, Csd)
                Cdd = Cdd - jnp.einsum("bnd,bne->de", Csd, Q)
                Ls.append(L); Qs.append(Q)
            Lsch = jnp.linalg.cholesky(Cdd)
            ld = ld + 2.0 * jnp.sum(jnp.log(jnp.diag(Lsch)))
            return ld, Ls, Qs, Lsch

        def resoudre(Ls, Qs, Lsch, Bs, Bd):
            Xi = Bd
            Gs = []
            for L, Q, B in zip(Ls, Qs, Bs):
                Gs.append(cho_lot(L, B))
                Xi = Xi - jnp.einsum("bnd,bnk->dk", Q, B)
            Td = jax.scipy.linalg.cho_solve((Lsch, True), Xi)
            Ts = [Gb - jnp.einsum("bnd,dk->bnk", Q, Td) for Gb, Q in zip(Gs, Qs)]
            return Ts, Td

        def inverse(Ls, Qs, Lsch, idx):
            dd_e, dd_r, dd_c = idx["dd"]
            Si = jax.scipy.linalg.cho_solve((Lsch, True), jnp.eye(nD))
            out = jnp.zeros(n_ent).at[dd_e].set(Si[dd_r, dd_c])
            for L, Q, a, b in zip(Ls, Qs, idx["sd"], idx["ss"]):
                nb, m = L.shape[0], L.shape[1]
                M = jnp.einsum("bnd,de->bne", Q, Si)                 # Q Sch^-1
                out = out.at[a["e"]].set(-M[a["b"], a["i"], a["d"]])
                Ci = cho_lot(L, jnp.broadcast_to(jnp.eye(m), (nb, m, m)))
                Ci = Ci + jnp.einsum("bnd,bkd->bnk", M, Q)
                out = out.at[b["e"]].set(Ci[b["b"], b["i"], b["j"]])
            return out

        self._f_fact = jax.jit(factoriser)
        self._f_res = jax.jit(resoudre)
        self._f_inv = jax.jit(inverse)

    def factoriser_valeurs(self, x):
        """x : valeurs de C sur le motif (ci, cj). Rend log|C|."""
        ld, Ls, Qs, Lsch = self._f_fact(jnp.asarray(x), self._idx)
        ld = float(ld)
        if not np.isfinite(ld):
            raise np.linalg.LinAlgError("C n'est pas definie positive (hybride)")
        self._etat = (Ls, Qs, Lsch)
        return ld

    def resoudre(self, B):
        B = np.asarray(B, dtype=np.float64)
        un = B.ndim == 1
        if un:
            B = B[:, None]
        Ls, Qs, Lsch = self._etat
        Bd = jnp.asarray(B[self.D])
        Bs = []
        for P in self.paquets:
            lig = P["lignes"]
            sel = np.where(lig >= 0, self.S[np.maximum(lig, 0)], 0)
            Bb = B[sel]
            Bb[lig < 0] = 0.0
            Bs.append(jnp.asarray(Bb))
        Ts, Td = self._f_res(Ls, Qs, Lsch, Bs, Bd)
        X = np.zeros_like(B)
        X[self.D] = np.asarray(Td)
        for P, T in zip(self.paquets, Ts):
            lig = P["lignes"]; ok = lig >= 0
            X[self.S[lig[ok]]] = np.asarray(T)[ok]
        return X[:, 0] if un else X

    def inverse_selective(self):
        Ls, Qs, Lsch = self._etat
        return np.asarray(self._f_inv(Ls, Qs, Lsch, self._idx))
