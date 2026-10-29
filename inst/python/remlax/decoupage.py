"""Automatic detection of the structure of a model: global terms and blocks.

A real mixed model often splits into nearly independent blocks (tanks,
trials, sites), linked only by a few global terms (genetic effects,
relationship matrices). Both forms of the computation use this split without
any knowledge of the design.

- V form: V = A + U U', with A block diagonal (local terms and residual) and
  U the factors of the global terms. Woodbury replaces the Cholesky
  factorisation of V (n^3 / 3) by those of the blocks and of one matrix of
  the width of U (blocs.py).
- MME form: eliminating the levels of the blocks before those of the global
  terms bounds the fill of the factor of C, because the global levels form a
  separator of the graph of C (mme_hybride.py).

Nothing is assumed about tanks or term names. The split is read from the
incidences, the relationship matrices and the residual. The global terms are
chosen by an explicit cost model.

Coupling. Two observations are coupled by a LOCAL term if they share a level
of that term (K = I). A term with a non-diagonal structure between levels
(relationship, AR1, ...) puts its whole support into one group. The residual
couples the observations of one unit (lvl "id"), or its whole section when
it has a structure between units. The residual is always local, because it
has full rank and cannot go into U.

Organisation. The helper functions turn each term and each residual section
into edges between observations and local nodes, then contract them to one
edge per observation. Decoupage keeps these contracted couplings and offers
blocs (labels for a given set of global terms), cout and cout_niveaux (cost
models of the V and MME forms) and chercher (greedy search of the global
terms).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .layout import level_kind, res_sections
from .structures import chol_sigma, theta0 as struct_theta0


def _identity_K(tm):
    """True when the term has no structure between levels (K = I)."""
    lvl = level_kind(tm)
    return lvl in ("id", "", None) and tm.get("LK") is None and tm.get("Kinv") is None


def largeur_facteur(tm):
    """Columns of the factor B = Z (L_Sigma (x) L_K): m q, with m the number
    of columns of L_Sigma."""
    t, rk = int(tm["t"]), int(tm["rank"])
    Ls = chol_sigma(np.asarray(struct_theta0(tm["struct"], t, rk)), tm["struct"], t, rk)
    return int(np.shape(Ls)[1]) * int(tm["q"])


def _term_edges(tm):
    """(observations, local nodes, number of nodes) of the coupling of a term."""
    zi = np.asarray(tm["zi"], dtype=np.int64)
    if _identity_K(tm):
        return zi, np.asarray(tm["zj"], dtype=np.int64) % int(tm["q"]), int(tm["q"])
    return zi, np.zeros(len(zi), dtype=np.int64), 1


def _residual_edges(res, n):
    """(observations, local nodes, number of nodes) of each residual section."""
    out = []
    for sec in res_sections(res, n):
        rows = np.asarray(sec["rows"], dtype=np.int64)
        if sec.get("lvl", "id") == "id":
            units = np.asarray(sec["unit"], dtype=np.int64)
            u, inv = np.unique(units, return_inverse=True)
            out.append((rows, inv, len(u)))
        else:
            out.append((rows, np.zeros(len(rows), dtype=np.int64), 1))
    return out


def _contract(obs, node, n_nodes, n):
    """(rows, components, number of components) of the coupling of a single
    term: one edge per row."""
    if len(obs) == 0:
        return obs, node, 0
    G = sp.coo_matrix((np.ones(len(obs), dtype=np.int8), (obs, n + node)),
                      shape=(n + n_nodes, n + n_nodes))
    _, lab = connected_components(G, directed=False)
    rows = np.unique(obs)
    _, comp = np.unique(lab[rows], return_inverse=True)
    return rows, comp, int(comp.max()) + 1


class Decoupage:
    """Precomputed couplings. blocs(globaux) and cout(globaux) on demand."""

    def __init__(self, terms, res, n):
        self.terms, self.n = terms, int(n)
        # Contraction. The union of the couplings of several terms equals the
        # union of the partitions that each term induces alone. The edges of
        # a term (one per entry of Z, millions for an IEE term) are therefore
        # replaced by one edge per row of its support towards its component.
        # The graph of blocks is the same and the search is much faster
        # (dev/notes.md).
        self.ar_t = [_contract(*_term_edges(tm), self.n) for tm in terms]
        self.ar_r = [_contract(*a, self.n) for a in _residual_edges(res, self.n)]
        self.w = np.array([largeur_facteur(tm) for tm in terms], dtype=np.int64)

    def blocs(self, globaux):
        """Block label of each observation once `globaux` are removed."""
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
        """Operations and memory (in doubles) of the V form by blocks.

        Operations: Cholesky and inverse of each block, n_b^3; A^-1 U,
        2 n_b^2 w; U'A^-1U, 2 n w^2; Cholesky of the capacitance matrix,
        w^3 / 3. Memory: blocks, U and A^-1 U, capacitance matrix.
        """
        lab = self.blocs(globaux)
        nb = np.bincount(lab).astype(np.float64)
        w = float(self.w[list(globaux)].sum()) if globaux else 0.0
        n = float(self.n)
        ops = float((nb ** 3).sum() + 2.0 * (nb ** 2).sum() * w + 2.0 * n * w * w + w ** 3 / 3.0)
        mem = float((nb ** 2).sum() + 2.0 * n * w + w * w)
        return dict(ops=ops, mem=mem, n_blocs=int(len(nb)), plus_grand=int(nb.max()), w=int(w))

    def cout_niveaux(self, globaux, p=0):
        """Operations and memory of the hybrid inversion of C (mme_hybride.py).

        The separator holds p fixed effects and the t q levels of each global
        term. Each block holds the levels of the local terms reached by its
        observations. Operations: Cholesky and inverse of the blocks, q_b^3;
        Q = Css^-1 Csd and the Schur complement, 2 q_b^2 nD; Q Sch^-1 and its
        correction, 2 q_b nD^2; Cholesky of the separator, nD^3 / 3.
        """
        lab = self.blocs(globaux)
        nB = int(lab.max()) + 1
        qb = np.zeros(nB)
        for k, tm in enumerate(self.terms):
            if k in globaux or not len(tm["zi"]):
                continue
            zi = np.asarray(tm["zi"], dtype=np.int64)
            zj = np.asarray(tm["zj"], dtype=np.int64)
            if not _identity_K(tm):
                qb[lab[zi[0]]] += int(tm["t"]) * int(tm["q"])
                continue
            key = np.unique(lab[zi] * (int(tm["t"]) * int(tm["q"]) + 1) + zj)
            qb += np.bincount(key // (int(tm["t"]) * int(tm["q"]) + 1), minlength=nB)
        nD = float(p + sum(int(self.terms[k]["t"]) * int(self.terms[k]["q"]) for k in globaux))
        ops = float((qb ** 3).sum() + 2.0 * (qb ** 2).sum() * nD + 2.0 * qb.sum() * nD * nD + nD ** 3 / 3.0)
        mem = float((qb ** 2).sum() + 2.0 * qb.sum() * nD + nD * nD)
        return dict(ops=ops, mem=mem, n_blocs=int((qb > 0).sum()), plus_grand=int(qb.max()), w=int(nD))

    def chercher(self, critere="ops", verbose=False, espace="observations", p=0):
        """Global terms by a forward and a backward greedy search. The better
        result is returned.

        Forward search: starts from no global term and, at each step, makes
        global the term that lowers the criterion most. It gets stuck as soon
        as the cut needs several terms at once. For example, two genetic
        terms each link all the tanks, and removing only one of them cuts
        nothing.
        Backward search: starts from all terms global and, at each step, makes
        local the term whose return lowers the criterion most. Local terms
        come back one by one without merging blocks. A term that links the
        blocks raises the cost and stays global.

        espace: "observations" for the V form by blocks (blocs.py), "niveaux"
        for the hybrid inversion of C (mme_hybride.py), with p fixed effects.
        """
        all_terms = [k for k in range(len(self.terms)) if len(self.ar_t[k][0])]
        cost = self.cout if espace == "observations" else (lambda G: self.cout_niveaux(G, p))

        def forward():
            # One forward step costs one evaluation per term. The search
            # usually stops at the first step, since no single term cuts the
            # graph.
            G, c = [], cost([])
            while True:
                cands = [(cost(G + [k]), k) for k in all_terms if k not in G]
                cands = [x for x in cands if x[0][critere] < c[critere]]
                if not cands:
                    return G, c
                ck, k = min(cands, key=lambda x: x[0][critere])
                G, c = G + [k], ck
                if verbose:
                    print("  [decoupage] + %s : %d blocs (max %d), w = %d, %.3g ops"
                          % (self.terms[k]["name"], c["n_blocs"], c["plus_grand"], c["w"], c["ops"]),
                          flush=True)

        def backward():
            # Ranked passes. Each pass evaluates once the return of every
            # global term and ranks the terms by gain. It then makes them local
            # in that order, checking each return again. This costs about 2T
            # evaluations per pass instead of T per step (T^2 in total).
            G, c = list(all_terms), cost(list(all_terms))
            while True:
                ranked = sorted(((cost([g for g in G if g != k])[critere], k) for k in G))
                changed = False
                for v, k in ranked:
                    if v >= c[critere]:
                        break
                    G2 = [g for g in G if g != k]
                    c2 = cost(G2)
                    if c2[critere] < c[critere]:
                        G, c, changed = G2, c2, True
                        if verbose:
                            print("  [decoupage] - %s : %d blocs (max %d), w = %d, %.3g ops"
                                  % (self.terms[k]["name"], c["n_blocs"], c["plus_grand"], c["w"],
                                     c["ops"]), flush=True)
                if not changed:
                    return G, c

        Ga, ca = forward()
        Gr, cr = backward()
        G, c = (Ga, ca) if ca[critere] <= cr[critere] else (Gr, cr)
        G = sorted(G)
        return dict(globaux=G, noms=[self.terms[k]["name"] for k in G], cout=c,
                    cout_dense=dict(ops=float(self.n) ** 3, mem=3.0 * float(self.n) ** 2),
                    espace=espace,
                    blocs=self.blocs(G))
