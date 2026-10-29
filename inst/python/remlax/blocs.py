"""Moteur par blocs : V = A + U U', blocs de A traites par lots (vmap).

LE CALCUL. decoupage.py separe les termes en LOCAUX et GLOBAUX et les
observations en blocs : termes locaux et residuelle ne couplent que des
observations d'un meme bloc. Alors

    V = A + U U',   A = diag(A_1, ..., A_B),   U = [B_k, k global]  (n x w)

avec A_b = R_b + somme des B_k B_k' locaux restreints au bloc b. Pour chaque
bloc on factorise A_b = L_b L_b' et on resout W_b = L_b^-1 [U_b, X_b, y_b] ; la
matrice des produits croises T = somme_b W_b' W_b vaut [U X y]' A^-1 [U X y].
La matrice BORDEE

    H = T + diag(1_w, 0_p, 0)

a pour premier bloc diagonal M = I + U'A^-1U, la capacite de Woodbury. Sa
Cholesky donne tout le reste par complements de Schur successifs :

    log|V|        = somme_b log|A_b| + log|M|      (diagonale de L_H, w premiers)
    log|X'V^-1X|  = diagonale de L_H, p suivants
    y'Py          = carre du dernier element diagonal de L_H

d'ou -2 logL REML sans jamais former V ni Sigma^-1 (une Sigma singuliere ne
gene pas : M >= I). Le gradient vient de la derivation automatique.

VMAP. Les blocs de tailles voisines sont regroupes en paquets et completes
jusqu'a la taille du plus grand du paquet : identite sur la diagonale de A,
zeros dans U, X et y, ce qui ne change ni log|A_b| ni W_b. Dans un paquet, vmap
fait de toutes les factorisations un seul lot (Cholesky et resolutions
batchees). Les paquets sont choisis par programmation dynamique pour limiter
le calcul perdu dans le remplissage. Un paquet se parcourt par lots de blocs
(lax.scan sous jax.checkpoint) : la memoire est bornee par un lot, le calcul
ne change pas ; un lot = tout le paquet par defaut.

PERIMETRE (version 1). Termes locaux a K = I (toute structure entre
caracteres) ou K fournie ; un terme local a structure entre niveaux
parametree (ar1, ...) est rendu global. Termes globaux : tous. Residuelle :
sections sans structure entre unites (lvl "id"), toute structure entre
caracteres. Hors de ce perimetre, hors_perimetre() le dit et fit_reml bascule
vers le moteur dense econome.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp

from .model import split_theta, res_sections, orthonormal_X, _section_S, sec_n_params
from .structures import chol_sigma, n_params
from .levels import level_chol
from .decoupage import Decoupage, largeur_facteur

LOG2PI = float(np.log(2.0 * np.pi))
PAQUETS_MAX = 4
SURCOUT_PAQUET = 0.10
OCTETS_LOT = 2 * 2 ** 30          # memoire visee par lot de blocs (scan)


def _genre(tm):
    """"id" (K = I), "fixed" (K fournie) ou le nom d'une structure parametree."""
    lvl = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    if tm.get("Kinv") is not None and tm.get("LK") is None:
        return "kinv"
    return "id" if lvl in ("", None) else lvl


def hors_perimetre(terms, res, lag=None):
    """Raison pour laquelle le modele sort du perimetre, ou None."""
    if lag is not None:
        return "decalage spatial"
    if res.get("struct") == "none":
        return "residuelle 'none'"
    for sec in res_sections(res, 1):
        if sec.get("lvl", "id") != "id":
            return "residuelle a structure entre unites (%s)" % sec.get("lvl")
    for tm in terms:
        if _genre(tm) == "kinv":
            return "terme '%s' donne par sa precision K^-1 seule" % tm["name"]
    return None


def _paquets(tailles, cout, kmax=PAQUETS_MAX):
    """Partition des blocs tries par taille en au plus kmax paquets contigus,
    minimisant la somme des couts (chaque bloc d'un paquet paie le cout du plus
    grand). Programmation dynamique, O(B^2 kmax)."""
    ordre = np.argsort(tailles, kind="stable")
    t = np.asarray(tailles)[ordre]
    B = len(t)
    inf = float("inf")
    best = [[inf] * (B + 1) for _ in range(kmax + 1)]
    arg = [[0] * (B + 1) for _ in range(kmax + 1)]
    best[0][0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(1, B + 1):
            for i in range(j):
                if best[k - 1][i] == inf:
                    continue
                v = best[k - 1][i] + (j - i) * cout(int(t[j - 1]))
                if v < best[k][j]:
                    best[k][j], arg[k][j] = v, i
    # Chaque paquet de plus coute une compilation et un lot de plus : il doit
    # faire gagner au moins SURCOUT_PAQUET du cout d'un paquet unique.
    k = min(range(1, kmax + 1), key=lambda kk: best[kk][B] * (1.0 + SURCOUT_PAQUET * (kk - 1)))
    groupes, j = [], B
    while j > 0:
        i = arg[k][j]
        groupes.append(ordre[i:j])
        j, k = i, k - 1
    return groupes[::-1]


def _z_dense(tm, n):
    Z = np.zeros((n, int(tm["t"]) * int(tm["q"])))
    np.add.at(Z, (np.asarray(tm["zi"], dtype=np.int64), np.asarray(tm["zj"], dtype=np.int64)),
              np.asarray(tm["zx"], dtype=np.float64))
    return Z


class PlanBlocs:
    """Decoupage, paquets et tableaux completes, calcules une fois sur l'hote."""

    def __init__(self, terms, res, y, X, decoupage=None, octets_lot=OCTETS_LOT, verbose=False):
        n = int(len(y))
        self.n, self.terms, self.res = n, terms, res
        Xq, self.ldRx = orthonormal_X(X)
        p = Xq.shape[1]
        self.p = p
        D = decoupage or Decoupage(terms, res, n).chercher()
        G = set(D["globaux"])
        # un terme local a structure entre niveaux parametree passe en global
        for k, tm in enumerate(terms):
            if k not in G and _genre(tm) not in ("id", "fixed"):
                G.add(k)
        if G != set(D["globaux"]):
            D = dict(D, blocs=Decoupage(terms, res, n).blocs(sorted(G)))
        self.globaux = sorted(G)
        self.locaux = [k for k in range(len(terms)) if k not in G and len(terms[k]["zi"])]
        lab = np.asarray(D["blocs"])
        self.n_blocs = int(lab.max()) + 1
        tailles = np.bincount(lab)
        self.w = int(sum(largeur_facteur(terms[k]) for k in self.globaux))
        r = self.w + p + 1
        self.r = r

        # termes locaux scalaires (t = 1, iid ou diag, K = I) : une matrice commune
        def scalaire(tm):
            return int(tm["t"]) == 1 and tm["struct"] in ("iid", "diag") and _genre(tm) == "id"
        self.loc_sc = [k for k in self.locaux if scalaire(terms[k])]
        self.loc_gen = [k for k in self.locaux if not scalaire(terms[k])]

        # residuelle : codes des paires (meme unite, meme bloc)
        secs = res_sections(res, n)
        self.secs = secs
        off_code, code_off = 0, []
        for sec in secs:
            code_off.append(off_code)
            off_code += int(sec["t"]) ** 2
        self.n_code = off_code

        # incidences denses : globales (parente fixe pliee) et locales generales
        Zfull = {}
        for k in self.globaux + self.loc_gen:
            tm = terms[k]
            Z = _z_dense(tm, n)
            if _genre(tm) == "fixed" and tm.get("LK") is not None:
                t, q = int(tm["t"]), int(tm["q"])
                Z = np.einsum("ntq,qp->ntp", Z.reshape(n, t, q),
                              np.asarray(tm["LK"])).reshape(n, t * q)
            Zfull[k] = Z

        # cout d'un bloc de taille m : Cholesky, resolution, produits croises
        def cout(m):
            return m ** 3 / 3.0 + m * m * r + m * r * r

        self.paquets = []
        for idx_blocs in _paquets(tailles, cout):
            self.paquets.append(self._paquet(idx_blocs, lab, tailles, Xq, np.asarray(y, float),
                                             Zfull, secs, code_off, octets_lot))
        if verbose:
            print("  [blocs] %s" % self.resume(), flush=True)

    # ------------------------------------------------------------------ paquet
    def _paquet(self, idx_blocs, lab, tailles, Xq, y, Zfull, secs, code_off, octets_lot):
        terms, n, r = self.terms, self.n, self.r
        nb = len(idx_blocs)
        m = int(tailles[idx_blocs].max())
        lignes = -np.ones((nb, m), dtype=np.int64)
        for a, b in enumerate(idx_blocs):
            rr = np.flatnonzero(lab == b)
            lignes[a, :len(rr)] = rr
        masque = lignes >= 0
        sel = np.where(masque, lignes, 0)

        def lire(M):                               # (n, ...) -> (nb, m, ...), zeros hors bloc
            out = M[sel]
            out[~masque] = 0.0
            return out

        P = dict(nb=nb, m=m)
        P["masque"] = masque.astype(np.float64)
        P["y"] = lire(y)
        P["X"] = lire(Xq)
        P["Zg"] = [lire(Zfull[k]).reshape(nb, m, int(terms[k]["t"]), int(terms[k]["q"]))
                   for k in self.globaux]

        # locaux scalaires : colonnes (terme, niveau) presentes dans chaque bloc
        n_sc = len(self.loc_sc)
        bloc_de = np.full(n, -1, dtype=np.int64)
        pos_de = np.full(n, -1, dtype=np.int64)
        for a in range(nb):
            rr = lignes[a][masque[a]]
            bloc_de[rr] = a
            pos_de[rr] = np.arange(len(rr))
        ent_a, ent_li, ent_cle, ent_x = [], [], [], []
        for j, k in enumerate(self.loc_sc):
            tm = terms[k]
            zi = np.asarray(tm["zi"], dtype=np.int64)
            a = bloc_de[zi]
            ok = a >= 0
            if not ok.any():
                continue
            ent_a.append(a[ok]); ent_li.append(pos_de[zi[ok]])
            ent_cle.append(np.asarray(tm["zj"], dtype=np.int64)[ok])
            ent_x.append(np.asarray(tm["zx"], dtype=np.float64)[ok])
            ent_cle[-1] = np.stack([np.full(ok.sum(), j), ent_cle[-1]], axis=1)
        if ent_a:
            ea = np.concatenate(ent_a); eli = np.concatenate(ent_li)
            ecle = np.concatenate(ent_cle); ex = np.concatenate(ent_x)
            # colonne = rang de (terme, niveau) parmi ceux du bloc
            cle_bloc = np.stack([ea, ecle[:, 0], ecle[:, 1]], axis=1)
            uniq, inv = np.unique(cle_bloc, axis=0, return_inverse=True)
            inv = inv.ravel()
            debut = np.searchsorted(uniq[:, 0], np.arange(nb))
            col_u = np.arange(len(uniq)) - debut[uniq[:, 0]]
            c_max = int(col_u.max()) + 1
            Zs = np.zeros((nb, m, c_max))
            np.add.at(Zs, (ea, eli, col_u[inv]), ex)
            Es = np.zeros((nb, c_max, n_sc + 1))
            Es[:, :, n_sc] = 1.0                       # colonne vide -> terme fictif nul
            Es[uniq[:, 0], col_u, n_sc] = 0.0
            Es[uniq[:, 0], col_u, uniq[:, 1]] = 1.0
        else:
            c_max = 1
            Zs = np.zeros((nb, m, 1))
            Es = np.zeros((nb, 1, n_sc + 1)); Es[:, :, n_sc] = 1.0
        P["Zs"], P["Es"] = Zs, Es

        # locaux generaux : niveaux presents dans le bloc (K = I) ou tous (K pliee)
        P["Zl"], P["loc_gen"] = [], []
        for k in self.loc_gen:
            tm = terms[k]
            t, q = int(tm["t"]), int(tm["q"])
            Zk = lire(Zfull[k]).reshape(nb, m, t, q)
            if not np.any(Zk):
                continue
            if _genre(tm) == "id":
                # niveaux utilises par bloc, completes a la largeur maximale
                niv = [np.flatnonzero(np.any(Zk[a] != 0, axis=(0, 1))) for a in range(nb)]
                qm = max(max(len(v) for v in niv), 1)
                Zc = np.zeros((nb, m, t, qm))
                for a, v in enumerate(niv):
                    Zc[a, :, :, :len(v)] = Zk[a][:, :, v]
                Zk = Zc
            P["Zl"].append(Zk)
            P["loc_gen"].append(k)

        # residuelle : paires (i, j) d'une meme unite dans le bloc, codes de Sigma
        I_, J_, C_ = [], [], []
        for a in range(nb):
            rr = lignes[a][masque[a]]
            pos = np.full(n, -1, dtype=np.int64)
            pos[rr] = np.arange(len(rr))
            ii, jj, cc = [], [], []
            for s, sec in enumerate(secs):
                rows = np.asarray(sec["rows"], dtype=np.int64)
                ok = pos[rows] >= 0
                if not ok.any():
                    continue
                lr = pos[rows[ok]]
                un = np.asarray(sec["unit"], dtype=np.int64)[ok]
                tr = np.asarray(sec["trait"], dtype=np.int64)[ok]
                t = int(sec["t"])
                o = np.argsort(un, kind="stable")
                lr, un, tr = lr[o], un[o], tr[o]
                deb = np.flatnonzero(np.r_[True, un[1:] != un[:-1]])
                fin = np.r_[deb[1:], len(un)]
                for d0, d1 in zip(deb, fin):
                    for x in range(d0, d1):
                        for z in range(d0, d1):
                            ii.append(lr[x]); jj.append(lr[z])
                            cc.append(code_off[s] + tr[x] * t + tr[z])
            I_.append(ii); J_.append(jj); C_.append(cc)
        pm = max(max(len(v) for v in I_), 1)
        Ip = np.zeros((nb, pm), dtype=np.int64)
        Jp = np.zeros((nb, pm), dtype=np.int64)
        Er = np.zeros((nb, pm, self.n_code))
        for a in range(nb):
            L = len(I_[a])
            Ip[a, :L], Jp[a, :L] = I_[a], J_[a]
            Er[a, np.arange(L), C_[a]] = 1.0          # paires vides : ligne nulle
        P["Ir"], P["Jr"], P["Er"] = Ip, Jp, Er

        # lots : nombre de blocs traites ensemble, d'apres la memoire visee
        c_tot = c_max + sum(z.shape[2] * z.shape[3] for z in P["Zl"])
        par_bloc = 8.0 * (3 * m * m + m * (c_tot + r) * 2 + r * r)
        lot = int(max(1, min(nb, octets_lot // max(par_bloc, 1.0))))
        n_lots = int(math.ceil(nb / lot))
        P["lot"], P["n_lots"] = lot, n_lots
        return P

    def resume(self):
        txt = ", ".join("%d x %d" % (P["nb"], P["m"]) for P in self.paquets)
        return ("%d blocs en %d paquets (%s) ; globaux %s, w = %d"
                % (self.n_blocs, len(self.paquets), txt,
                   [self.terms[k]["name"] for k in self.globaux], self.w))

    # ------------------------------------------------------------------ tableaux
    def tableaux(self):
        """Tableaux du peripherique, regroupes par lots : (n_lots, lot, ...)."""
        out = []
        for P in self.paquets:
            lot, nl, nb = P["lot"], P["n_lots"], P["nb"]
            pad = nl * lot - nb

            def lots(a):
                a = np.asarray(a)
                if pad:          # blocs fictifs : masque nul, donc A = I et W = 0
                    a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], dtype=a.dtype)])
                return jnp.asarray(a.reshape((nl, lot) + a.shape[1:]))
            out.append(dict(
                masque=lots(P["masque"]), y=lots(P["y"]), X=lots(P["X"]),
                Zg=[lots(z) for z in P["Zg"]], Zs=lots(P["Zs"]), Es=lots(P["Es"]),
                Zl=[lots(z) for z in P["Zl"]], Ir=lots(P["Ir"]), Jr=lots(P["Jr"]),
                Er=lots(P["Er"])))
        return out


# ==============================================================================
# -2 logL
# ==============================================================================
def make_neg2_blocs(plan):
    """f(theta, tableaux) -> -2 logL REML (constante comprise), fonction JAX pure."""
    terms, res, p, w, r = plan.terms, plan.res, plan.p, plan.w, plan.r
    n = plan.n
    glob, loc_sc = plan.globaux, plan.loc_sc
    loc_gen_par_paquet = [P["loc_gen"] for P in plan.paquets]

    def facteur_sigma(theta_k, tm):
        t, rk = int(tm["t"]), int(tm["rank"])
        return chol_sigma(theta_k[:n_params(tm["struct"], t, rk)], tm["struct"], t, rk)

    def lk_param(theta_k, tm):
        t, rk = int(tm["t"]), int(tm["rank"])
        ns = n_params(tm["struct"], t, rk)
        return level_chol(theta_k[ns:], _genre(tm), int(tm["q"]), dims=tm.get("dims"),
                          LK_fixed=tm.get("LK"), order=tm.get("lvl_order", 0),
                          coord=tm.get("coord"), parts=tm.get("lvl_parts"),
                          opts=tm.get("lvl_opts"), expr=tm.get("lvl_expr"))

    def f(theta, tabs):
        th_t, th_r, _ = split_theta(theta, terms, res)
        Ls = {k: facteur_sigma(th_t[k], terms[k]) for k in glob + plan.locaux}
        LKp = {k: lk_param(th_t[k], terms[k]) for k in glob if _genre(terms[k]) not in ("id", "fixed")}
        o = 0
        Svec = []
        for sec in plan.secs:
            pn = sec_n_params(sec)
            Svec.append(_section_S(th_r[o:o + pn], sec).reshape(-1))
            o += pn
        Svec = jnp.concatenate(Svec)
        sig_sc = jnp.concatenate([jnp.stack([Ls[k][0, 0] for k in loc_sc]) if loc_sc else jnp.zeros(0),
                                  jnp.zeros(1)])

        def bloc(d, loc_gen):
            m = d["y"].shape[0]
            # A = R + B_loc B_loc' + identite sur le remplissage
            vals = d["Er"] @ Svec
            A = jnp.zeros((m, m)).at[d["Ir"], d["Jr"]].add(vals)
            A = A + jnp.diag(1.0 - d["masque"])
            facs = [d["Zs"] * (d["Es"] @ sig_sc)[None, :]]
            for Zk, k in zip(d["Zl"], loc_gen):
                facs.append(jnp.einsum("ntq,tm->nmq", Zk, Ls[k]).reshape(m, -1))
            Bl = jnp.concatenate(facs, axis=1)
            A = A + Bl @ Bl.T
            # colonnes globales U, puis X et y
            us = []
            for Zk, k in zip(d["Zg"], glob):
                if k in LKp:
                    Zk = jnp.einsum("ntq,qp->ntp", Zk, LKp[k])
                us.append(jnp.einsum("ntq,tm->nmq", Zk, Ls[k]).reshape(m, -1))
            RHS = jnp.concatenate(us + [d["X"], d["y"][:, None]], axis=1)
            L = jnp.linalg.cholesky(A)
            Wb = jax.scipy.linalg.solve_triangular(L, RHS, lower=True)
            return Wb.T @ Wb, 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        T = jnp.zeros((r, r))
        ld = jnp.zeros(())
        for tab, loc_gen in zip(tabs, loc_gen_par_paquet):
            def corps(c, d, loc_gen=loc_gen):
                Tb, lb = jax.vmap(lambda dd: bloc(dd, loc_gen))(d)
                return (c[0] + Tb.sum(0), c[1] + lb.sum()), None
            (T, ld), _ = jax.lax.scan(jax.checkpoint(corps), (T, ld), tab)
        H = T + jnp.diag(jnp.concatenate([jnp.ones(w), jnp.zeros(p + 1)]))
        LH = jnp.linalg.cholesky(H)
        dH = jnp.diag(LH)
        return ((n - p) * LOG2PI + ld + 2.0 * jnp.sum(jnp.log(dH[:w + p]))
                + dH[-1] ** 2 + plan.ldRx)
    return f


def make_objective_blocs(plan, scale=None):
    """(fun_jac_scaled, fun_jac, sc), meme contrat que model.make_objective."""
    sc = float(plan.n) if scale is None else float(scale)
    f = make_neg2_blocs(plan)
    tabs = plan.tableaux()
    fg = jax.jit(jax.value_and_grad(lambda th, tb: f(th, tb) / sc))

    def fun_jac_scaled(theta):
        v, g = fg(jnp.asarray(theta), tabs)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc
    return fun_jac_scaled, fun_jac, sc
