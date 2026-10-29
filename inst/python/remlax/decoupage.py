"""Detection automatique de la structure d'un modele : termes globaux et blocs.

POURQUOI. Un modele mixte reel se decoupe souvent en blocs presque
independants (bacs, essais, sites), relies seulement par quelques termes
globaux (effets genetiques, parente). Les deux formes du calcul en tirent parti
sans rien savoir du dispositif :

- forme V : V = A + U U', avec A diagonale par blocs (termes locaux et
  residuelle) et U les facteurs des termes globaux. Woodbury remplace la
  Cholesky de V (n^3 / 3) par celles des blocs et d'une matrice de la largeur
  de U ;
- forme MME : eliminer les niveaux des blocs avant ceux des termes globaux
  borne le remplissage du facteur de C (les niveaux globaux forment un
  separateur du graphe de C).

Rien n'est suppose : ni bacs, ni noms de termes. Le decoupage se lit dans les
incidences, les parentes et la residuelle, et le choix des termes globaux se
fait par un modele de cout explicite.

COUPLAGE. Deux observations sont couplees par un terme LOCAL si elles portent
un meme niveau de ce terme (K = I), ou si le terme a une structure entre
niveaux non diagonale (parente, AR1, ...), auquel cas tout son support forme un
seul groupe. La residuelle couple les observations d'une meme unite (lvl "id")
ou toute sa section (structure entre unites). Elle est toujours locale : elle
est de rang plein et ne peut pas passer dans U.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .model import res_sections
from .structures import chol_sigma, theta0 as struct_theta0


def _k_identite(tm):
    lvl = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
    return lvl in ("id", "", None) and tm.get("LK") is None and tm.get("Kinv") is None


def largeur_facteur(tm):
    """Colonnes du facteur B = Z (L_Sigma (x) L_K) : m q, m = colonnes de L_Sigma."""
    t, rk = int(tm["t"]), int(tm["rank"])
    Ls = chol_sigma(np.asarray(struct_theta0(tm["struct"], t, rk)), tm["struct"], t, rk)
    return int(np.shape(Ls)[1]) * int(tm["q"])


def _aretes_terme(tm):
    """(observations, noeuds locaux) du couplage d'un terme."""
    zi = np.asarray(tm["zi"], dtype=np.int64)
    if _k_identite(tm):
        return zi, np.asarray(tm["zj"], dtype=np.int64) % int(tm["q"]), int(tm["q"])
    return zi, np.zeros(len(zi), dtype=np.int64), 1


def _aretes_residuelle(res, n):
    out = []
    for sec in res_sections(res, n):
        rows = np.asarray(sec["rows"], dtype=np.int64)
        if sec.get("lvl", "id") == "id":
            un = np.asarray(sec["unit"], dtype=np.int64)
            u, inv = np.unique(un, return_inverse=True)
            out.append((rows, inv, len(u)))
        else:
            out.append((rows, np.zeros(len(rows), dtype=np.int64), 1))
    return out


def _contracter(obs, noeud, n_noeuds, n):
    """(lignes, composantes) du couplage d'un seul terme : une arete par ligne."""
    if len(obs) == 0:
        return obs, noeud, 0
    G = sp.coo_matrix((np.ones(len(obs), dtype=np.int8), (obs, n + noeud)),
                      shape=(n + n_noeuds, n + n_noeuds))
    _, lab = connected_components(G, directed=False)
    lignes = np.unique(obs)
    _, comp = np.unique(lab[lignes], return_inverse=True)
    return lignes, comp, int(comp.max()) + 1


class Decoupage:
    """Couplages precalcules ; blocs(locaux) et cout(globaux) a la demande."""

    def __init__(self, terms, res, n):
        self.terms, self.n = terms, int(n)
        # CONTRACTION. L'union des couplages de plusieurs termes est l'union des
        # partitions que chaque terme induit seul : on remplace donc les aretes
        # d'un terme (une par entree de Z, des millions pour un terme IEE) par
        # une arete par ligne de son support vers sa composante. Meme graphe de
        # blocs, recherche ~10 fois plus rapide.
        self.ar_t = [_contracter(*_aretes_terme(tm), self.n) for tm in terms]
        self.ar_r = [_contracter(*a, self.n) for a in _aretes_residuelle(res, self.n)]
        self.w = np.array([largeur_facteur(tm) for tm in terms], dtype=np.int64)

    def blocs(self, globaux):
        """Etiquette de bloc de chaque observation quand `globaux` sont retires."""
        n = self.n
        I_, J_, off = [], [], n
        for k, (zi, nd, nn) in enumerate(self.ar_t):
            if k in globaux:
                continue
            I_.append(zi); J_.append(off + nd); off += nn
        for rows, nd, nn in self.ar_r:
            I_.append(rows); J_.append(off + nd); off += nn
        if not I_:
            return np.arange(n)
        I = np.concatenate(I_); J = np.concatenate(J_)
        G = sp.coo_matrix((np.ones(len(I), dtype=np.int8), (I, J)), shape=(off, off))
        _, lab = connected_components(G, directed=False)
        _, lab_obs = np.unique(lab[:n], return_inverse=True)
        return lab_obs

    def cout(self, globaux):
        """Operations et memoire (en doubles) de la forme V par blocs.

        Blocs : Cholesky et inverse de chaque bloc, n_b^3 ; A^-1 U, 2 n_b^2 w ;
        U'A^-1U, 2 n w^2 ; Cholesky de la capacite, w^3 / 3. Memoire : blocs,
        U et A^-1 U, capacite.
        """
        lab = self.blocs(globaux)
        nb = np.bincount(lab).astype(np.float64)
        w = float(self.w[list(globaux)].sum()) if globaux else 0.0
        n = float(self.n)
        ops = float((nb ** 3).sum() + 2.0 * (nb ** 2).sum() * w + 2.0 * n * w * w + w ** 3 / 3.0)
        mem = float((nb ** 2).sum() + 2.0 * n * w + w * w)
        return dict(ops=ops, mem=mem, n_blocs=int(len(nb)), plus_grand=int(nb.max()), w=int(w))

    def cout_niveaux(self, globaux, p=0):
        """Operations et memoire de l'inversion hybride de C (mme_hybride.py).

        Le separateur compte p effets fixes et les t q niveaux de chaque terme
        global ; chaque bloc compte les niveaux des termes locaux touches par
        ses observations. Cholesky et inverse des blocs, q_b^3 ; Q = Css^-1 Csd et
        complement de Schur, 2 q_b^2 nD ; Q Sch^-1 et sa correction, 2 q_b nD^2 ;
        Cholesky du separateur, nD^3 / 3.
        """
        lab = self.blocs(globaux)
        nB = int(lab.max()) + 1
        qb = np.zeros(nB)
        for k, tm in enumerate(self.terms):
            if k in globaux or not len(tm["zi"]):
                continue
            zi = np.asarray(tm["zi"], dtype=np.int64)
            zj = np.asarray(tm["zj"], dtype=np.int64)
            if not _k_identite(tm):
                qb[lab[zi[0]]] += int(tm["t"]) * int(tm["q"])
                continue
            cle = np.unique(lab[zi] * (int(tm["t"]) * int(tm["q"]) + 1) + zj)
            qb += np.bincount(cle // (int(tm["t"]) * int(tm["q"]) + 1), minlength=nB)
        nD = float(p + sum(int(self.terms[k]["t"]) * int(self.terms[k]["q"]) for k in globaux))
        ops = float((qb ** 3).sum() + 2.0 * (qb ** 2).sum() * nD + 2.0 * qb.sum() * nD * nD + nD ** 3 / 3.0)
        mem = float((qb ** 2).sum() + 2.0 * qb.sum() * nD + nD * nD)
        return dict(ops=ops, mem=mem, n_blocs=int((qb > 0).sum()), plus_grand=int(qb.max()), w=int(nD))

    def chercher(self, critere="ops", verbose=False, espace="observations", p=0):
        """Termes globaux par deux recherches gloutonnes ; la meilleure est rendue.

        EN AVANT : part d'aucun terme global et rend global, a chaque pas, le terme
        qui fait le plus baisser le critere. Elle reste coincee des que la coupure
        demande plusieurs termes a la fois : deux termes genetiques relient
        chacun tous les bacs, et en retirer un seul ne coupe rien (mesure sur le
        cas synthetique du test econome).
        EN ARRIERE : part de tous les termes globaux et rend local, a chaque pas,
        le terme dont le retour fait le plus baisser le critere. Les termes
        locaux reviennent un a un sans fusionner les blocs ; un terme qui relie
        les blocs fait monter le cout et reste global.
        """
        tous = [k for k in range(len(self.terms)) if len(self.ar_t[k][0])]
        # espace "observations" : forme V par blocs (blocs.py) ; espace
        # "niveaux" : inversion hybride de C (mme_hybride.py)
        cout = self.cout if espace == "observations" else (lambda G: self.cout_niveaux(G, p))

        def avant():
            # Un pas en avant coute une evaluation par terme ; il s'arrete en
            # general des le premier pas (aucun terme seul ne coupe le graphe).
            G, c = [], cout([])
            while True:
                cands = [(cout(G + [k]), k) for k in tous if k not in G]
                cands = [x for x in cands if x[0][critere] < c[critere]]
                if not cands:
                    return G, c
                ck, k = min(cands, key=lambda x: x[0][critere])
                G, c = G + [k], ck
                if verbose:
                    print("  [decoupage] + %s : %d blocs (max %d), w = %d, %.3g ops"
                          % (self.terms[k]["name"], c["n_blocs"], c["plus_grand"], c["w"], c["ops"]),
                          flush=True)

        def arriere():
            # PASSES CLASSEES. Chaque passe evalue une fois le retour de chaque
            # terme global, les classe par gain, puis les rend locaux dans cet
            # ordre en reverifiant chaque retour : ~2T evaluations par passe au
            # lieu de T par pas (T^2 au total, 33 s sur un modele a 51 termes).
            G, c = list(tous), cout(list(tous))
            while True:
                classe = sorted(((cout([g for g in G if g != k])[critere], k) for k in G))
                change = False
                for v, k in classe:
                    if v >= c[critere]:
                        break
                    G2 = [g for g in G if g != k]
                    c2 = cout(G2)
                    if c2[critere] < c[critere]:
                        G, c, change = G2, c2, True
                        if verbose:
                            print("  [decoupage] - %s : %d blocs (max %d), w = %d, %.3g ops"
                                  % (self.terms[k]["name"], c["n_blocs"], c["plus_grand"], c["w"],
                                     c["ops"]), flush=True)
                if not change:
                    return G, c

        Ga, ca = avant()
        Gr, cr = arriere()
        G, c = (Ga, ca) if ca[critere] <= cr[critere] else (Gr, cr)
        G = sorted(G)
        return dict(globaux=G, noms=[self.terms[k]["name"] for k in G], cout=c,
                    cout_dense=dict(ops=float(self.n) ** 3, mem=3.0 * float(self.n) ** 2),
                    espace=espace,
                    blocs=self.blocs(G))
