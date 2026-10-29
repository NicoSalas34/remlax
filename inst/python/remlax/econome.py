"""Moteur dense, mode econome en memoire GPU.

Meme vraisemblance, meme gradient et meme matrice AI que le moteur dense
standard (model.make_objective, ai.make_fga, fit.effets_denses). Seule la
facon de les calculer change, pour reduire le nombre de matrices n x n et de
facteurs vivants en meme temps sur le GPU. Le mode standard reste le defaut et
n'est pas modifie ; fit_reml(..., memoire="econome") choisit celui-ci.

CE QUE LE MODE STANDARD GARDE, ET QUE CELUI-CI EVITE. Mesure faite sur le
modele mvD7 (n = 12 760, 51 termes, 198 parametres), qui ne tenait pas sur une
tranche MIG de 10 Go :

1. Les incidences pleine hauteur Zs (dense_Z, n x somme des t q = 3,35 Go),
   creees sur le GPU par make_neg2 et gardees jusqu'aux BLUP de fin
   d'ajustement. Elles ne servent pas pendant l'optimisation. Ici elles ne
   sont jamais formees sur le GPU : les BLUP construisent l'incidence d'un
   terme a la fois, sur l'hote (z_hote).

2. La residuelle sans structure entre unites (lvl "id") formee en dense :
   _section_V construit l'egalite des unites (n_s x n_s), Sigma[tr, tr] et leur
   produit, puis la passe arriere rassemble dV[idx, idx]. Pour une residuelle
   us(caractere) par plante, R n'a pourtant qu'une entree par paire de
   caracteres d'une meme plante (65 000 sur mvD7 contre 10 911^2). Ici R est
   dispersee entree par entree, et la passe arriere ne lit que ces entrees.

3. Les facteurs B_k gardes entre la passe avant et la passe arriere. Ici chaque
   contribution de groupe est sous jax.checkpoint : B est recalcule dans la
   passe arriere, pour un cout de n x w operations, negligeable devant la
   Cholesky.

4. Le rassemblement dV[idx, idx] d'un groupe dont le support couvre une grande
   partie des lignes (le terme genetique de mvD7 les couvre presque toutes) :
   une copie n x n. Ici un tel facteur est etendu a toute la hauteur
   (n x w, 0,24 Go sur mvD7), et la passe arriere fait dV B sans copie.

5. Dans l'AI (ai.make_fga), V^-1 formee par cho_solve contre l'identite
   pendant que L reste vivante pour les resolutions de P Q, puis l'adjoint a
   cote. Ici P Q est calcule d'abord par resolutions triangulaires (n x p), et
   l'adjoint V^-1 - ... ne se forme qu'ensuite, quand L peut mourir.

CE QUI NE CHANGE PAS : l'algebre. V est la meme matrice. L'ordre des sommes
flottantes change, donc l'egalite se teste a une tolerance relative.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .model import (split_theta, res_sections, sec_n_params, support_groups,
                    restrict_Z, prefold_fixed_K, orthonormal_X, term_factor,
                    _section_V, _section_S, section_matvec)
from .core import _reml_fwd
from .ai import _sigma_et_K

# Un groupe dont le support couvre au moins cette fraction des lignes est etendu
# a toute la hauteur (point 4 de l'en-tete). En dessous, la dispersion d'un bloc
# |idx|^2 coute moins que n x w operations de plus.
FRACTION_HAUTEUR = 0.5
# Au-dela de cette taille, la matrice d'indicatrices de la residuelle creuse
# (paires x t^2) n'est pas formee : la valeur passe par un rassemblement.
OCTETS_MAX_INDICATRICES = 256 * 2 ** 20


def _paires_unites(sec):
    """Paires (i, j), indices locaux a la section, de deux lignes d'une meme unite.

    Toutes les paires, diagonale comprise et dans les deux sens : R est
    dispersee telle quelle, sans symetrisation.
    """
    un = np.asarray(sec["unit"], dtype=np.int64)
    ordre = np.argsort(un, kind="stable")
    u_tri = un[ordre]
    debut = np.flatnonzero(np.r_[True, u_tri[1:] != u_tri[:-1]])
    taille = np.diff(np.r_[debut, len(u_tri)])
    I_, J_ = [], []
    for m in np.unique(taille):
        d = debut[taille == m]
        blocs = ordre[d[:, None] + np.arange(m)[None, :]]          # (g, m)
        I_.append(np.repeat(blocs, m, axis=1).ravel())
        J_.append(np.tile(blocs, (1, m)).ravel())
    return np.concatenate(I_), np.concatenate(J_)


def z_hote(term, n):
    """Incidence pleine hauteur d'un terme, en NumPy sur l'hote (cf. model.dense_Z)."""
    Z = np.zeros((n, term["t"] * term["q"]))
    np.add.at(Z, (np.asarray(term["zi"], dtype=np.int64),
                  np.asarray(term["zj"], dtype=np.int64)), np.asarray(term["zx"]))
    return Z


class PlanEconome:
    """Tout ce qui ne depend pas de theta, calcule une fois.

    terms_v : termes apres pliage de la parente fixe dans l'incidence
    (model.prefold_fixed_K) ; Zr : incidences restreintes a leur support, sur le
    peripherique ; plans des sections residuelles et des groupes.
    """

    def __init__(self, terms, res, y, X):
        n = int(y.shape[0])
        self.n, self.res, self.terms0 = n, res, terms
        Xq, self.ldRx = orthonormal_X(X)
        self.yj, self.Xj = jnp.asarray(y), jnp.asarray(Xq)
        self.groupes = support_groups(terms, n)
        Zr = [None] * len(terms)
        rows_of = [None] * len(terms)
        for rows, ks in self.groupes:
            for k in ks:
                Zr[k] = restrict_Z(terms[k], rows, n)
                rows_of[k] = np.asarray(rows, dtype=np.int64)
        self.terms, self.Zr = prefold_fixed_K(terms, Zr)
        self.rows_of = rows_of
        tout = np.arange(n, dtype=np.int64)

        def plein(r):
            return r.shape[0] == n and np.array_equal(r, tout)

        # --- sections residuelles -------------------------------------------
        self.secs = res_sections(res, n)
        self.plan_secs = []
        for sec in self.secs:
            rows = np.asarray(sec["rows"], dtype=np.int64)
            if sec.get("lvl", "id") == "id":
                i, j = _paires_unites(sec)
                tr = np.asarray(sec["trait"], dtype=np.int64)
                t = int(sec["t"])
                d = dict(mode="paires", I=jnp.asarray(rows[i]), J=jnp.asarray(rows[j]))
                code = tr[i] * t + tr[j]
                if len(code) * t * t * 8 <= OCTETS_MAX_INDICATRICES:
                    # valeur = E vec(Sigma) : la passe arriere est un produit
                    # E' dvals, deterministe (le rassemblement Sigma[a, b] aurait
                    # une passe arriere par additions atomiques)
                    E = np.zeros((len(code), t * t))
                    E[np.arange(len(code)), code] = 1.0
                    d["E"] = jnp.asarray(E)
                else:
                    d["code"] = jnp.asarray(code)
                self.plan_secs.append(d)
            elif plein(rows):
                self.plan_secs.append(dict(mode="plein"))
            else:
                self.plan_secs.append(dict(mode="bloc", idx=jnp.asarray(rows)))

        # --- groupes de termes ----------------------------------------------
        self.plan_grp = []
        for rows, ks in self.groupes:
            rows = np.asarray(rows, dtype=np.int64)
            if plein(rows):
                mode = "plein"
            elif rows.shape[0] >= FRACTION_HAUTEUR * n:
                mode = "haut"
            else:
                mode = "bloc"
            self.plan_grp.append(dict(mode=mode, ks=list(ks), idx=jnp.asarray(rows),
                                      contrib=self._contribution(ks, rows, mode)))

    def _contribution(self, ks, rows, mode):
        """B B' d'un groupe, sous jax.checkpoint (B recalcule dans la passe arriere)."""
        terms, n = self.terms, self.n
        idx = jnp.asarray(rows)

        def f(ths, Zs_):
            B = jnp.concatenate([term_factor(th, terms[k], Z)
                                 for th, k, Z in zip(ths, ks, Zs_)], axis=1)
            if mode == "haut":
                B = jnp.zeros((n, B.shape[1]), B.dtype).at[idx].set(B)
            return B @ B.T
        return jax.checkpoint(f)

    def V(self, theta, Zr):
        """V = R + somme des contributions des groupes, assemblee en place."""
        n = self.n
        th_terms, th_res, _ = split_theta(theta, self.terms, self.res)
        V = jnp.zeros((n, n))
        o = 0
        for sec, d in zip(self.secs, self.plan_secs):
            pnb = sec_n_params(sec)
            th_s = th_res[o:o + pnb]
            o += pnb
            if d["mode"] == "paires":
                S = _section_S(th_s, sec)
                vals = d["E"] @ S.reshape(-1) if "E" in d else S.reshape(-1)[d["code"]]
                V = V.at[d["I"], d["J"]].add(vals)
            elif d["mode"] == "plein":
                V = V + _section_V(th_s, sec)
            else:
                idx = d["idx"]
                V = V.at[idx[:, None], idx[None, :]].add(_section_V(th_s, sec))
        for d in self.plan_grp:
            ks = d["ks"]
            M = d["contrib"]([th_terms[k] for k in ks], [Zr[k] for k in ks])
            if d["mode"] == "bloc":
                idx = d["idx"]
                V = V.at[idx[:, None], idx[None, :]].add(M)
            else:
                V = V + M
        return V

    def resume(self):
        """Texte court : modes retenus, pour le journal."""
        from collections import Counter
        s = Counter(d["mode"] for d in self.plan_secs)
        g = Counter(d["mode"] for d in self.plan_grp)
        return ("residuelle %s ; groupes %s ; adjoint %s"
                % (", ".join("%s %d" % kv for kv in sorted(s.items())) or "aucune",
                   ", ".join("%s %d" % kv for kv in sorted(g.items())) or "aucun",
                   adjoint_choisi()))


# ==============================================================================
# Noyau V -> -2logL, passe arriere par cho_solve (cf. core.reml_from_V)
# ==============================================================================
# MEME VALEUR, MEME GRADIENT que core.reml_from_V : la passe avant est la
# sienne (_reml_fwd). Seule la passe arriere change. Le noyau commun forme
# V^-1 = L^-T L^-1 par une resolution contre l'identite (n^3 operations) puis un
# produit general (2 n^3). cho_solve fait deux resolutions triangulaires
# (2 n^3 au total), dans le tampon de l'identite : meme nombre de matrices
# n x n vivantes (L et ce tampon), un tiers d'operations en moins. MESURE sur
# mvD7 (CPU, 32 coeurs) avant ce changement : AI econome 10,4 s contre 7,4 s
# pour l'AI standard, qui passait deja par cho_solve.
#
# MAIS cho_solve COUTE UNE MATRICE n x n DE PLUS SUR GPU : XLA ne fait pas les
# deux resolutions en place. MESURE sur mvD7, tranche MIG 10 Go : temporaires de
# l'objectif 3,82 Gio avec L^-1 puis L^-T L^-1, 5,03 Gio avec cho_solve, qui ne
# tient plus. Sur CPU au contraire, cho_solve gagne sans que la RAM compte
# (objectif 8,6 s contre 13,0 s pour le mode standard). D'ou ADJOINT : "inv"
# (memoire minimale) sur GPU, "cho" (moins d'operations) sur CPU ; la variable
# d'environnement REMLAX_ADJOINT force l'un ou l'autre.
def adjoint_choisi():
    import os
    v = os.environ.get("REMLAX_ADJOINT", "").strip().lower()
    if v in ("inv", "cho"):
        return v
    return "cho" if jax.default_backend() == "cpu" else "inv"


def _V_inverse(L, mode):
    """V^-1 depuis le facteur de Cholesky, selon le compromis choisi."""
    n = L.shape[0]
    if mode == "cho":
        return jax.scipy.linalg.cho_solve((L, True), jnp.eye(n))
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(n), lower=True)
    return Linv.T @ Linv


@jax.custom_vjp
def reml_from_V_cho(V, y, X):
    return _reml_fwd(V, y, X)[0]


def _fwd_cho(V, y, X):
    val, res = _reml_fwd(V, y, X)
    return val, res


def _bwd_commun(res, g, mode):
    L, ViX, La, r = res
    Z = jax.scipy.linalg.solve_triangular(La, ViX.T, lower=True)   # A^-1/2 ViX'
    W = _V_inverse(L, mode)                                          # V^-1
    W = W - Z.T @ Z                                                  # P
    W = W - jnp.outer(r, r)                                          # P - (Py)(Py)'
    return (g * W, 2.0 * g * r, None)


reml_from_V_cho.defvjp(_fwd_cho, lambda res, g: _bwd_commun(res, g, "cho"))


@jax.custom_vjp
def reml_from_V_inv(V, y, X):
    return _reml_fwd(V, y, X)[0]


reml_from_V_inv.defvjp(_fwd_cho, lambda res, g: _bwd_commun(res, g, "inv"))


# ==============================================================================
# Objectif (valeur et gradient), meme contrat que model.make_objective
# ==============================================================================
def make_neg2_econome(plan, adjoint=None):
    """-2logL (constante comprise) comme fonction JAX pure de (theta, Zr)."""
    noyau = reml_from_V_cho if (adjoint or adjoint_choisi()) == "cho" else reml_from_V_inv

    def f(theta, Zr_a):
        V = plan.V(jnp.asarray(theta), Zr_a)
        return noyau(V, plan.yj, plan.Xj) + plan.ldRx
    return f


def make_objective_econome(plan, scale=None, adjoint=None):
    """Rend (fun_jac_scaled, fun_jac, sc) ; meme mise a l'echelle par n que
    model.make_objective."""
    sc = float(plan.n) if scale is None else float(scale)
    f_brut = make_neg2_econome(plan, adjoint=adjoint)
    fg = jax.jit(jax.value_and_grad(lambda th, Z: f_brut(th, Z) / sc))
    Zr = plan.Zr

    def fun_jac_scaled(theta):
        v, g = fg(jnp.asarray(theta), Zr)
        return float(v), np.asarray(g, dtype=np.float64)

    def fun_jac(theta):
        v, g = fun_jac_scaled(theta)
        return v * sc, g * sc
    return fun_jac_scaled, fun_jac, sc


# ==============================================================================
# AI : valeur, gradient et matrice AI d'une factorisation (cf. ai.make_fga)
# ==============================================================================
def make_fga_econome(plan, adjoint=None):
    """fga(theta) -> (-2logL, gradient, AI), comme ai.make_fga.

    Colonnes de Q : memes formules que ai.make_ai (colonnes_terme,
    colonnes_section), recopiees ici pour partager les incidences du plan au
    lieu d'en construire une seconde copie sur le peripherique.
    """
    n, terms, secs, res = plan.n, plan.terms, plan.secs, plan.res
    yj, Xj = plan.yj, plan.Xj
    p_x = Xj.shape[1]
    mode_adj = adjoint or adjoint_choisi()
    rows_of = [jnp.asarray(r) for r in plan.rows_of]

    def colonnes_terme(th_k, term, Z, rows, v):
        t, q = term["t"], term["q"]
        U = (Z.T @ v[rows]).reshape(t, q)

        def g(th):
            S, K = _sigma_et_K(th, term)
            M = U if K is None else U @ K
            return (S @ M).reshape(t * q)
        J = jax.jacfwd(g)(th_k)
        return jnp.zeros((n, J.shape[1])).at[rows].set(Z @ J)

    def colonnes_section(th_s, sec, v):
        rows = jnp.asarray(np.asarray(sec["rows"], dtype=np.int64))
        J = jax.jacfwd(lambda th: section_matvec(th, sec, v[rows]))(th_s)
        return jnp.zeros((n, J.shape[1])).at[rows].set(J)

    def fga(theta, Zr):
        V, vjp_V = jax.vjp(lambda th: plan.V(th, Zr), theta)
        L = jnp.linalg.cholesky(V)
        del V
        logdetV = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        def sol(b):
            return jax.scipy.linalg.cho_solve((L, True), b)
        ViX = sol(Xj)
        A = Xj.T @ ViX
        La = jnp.linalg.cholesky(A)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(La)))
        ViXA = ViX @ jax.scipy.linalg.cho_solve((La, True), jnp.eye(p_x))
        Py = sol(yj) - ViXA @ (ViX.T @ yj)
        f = (n - p_x) * np.log(2.0 * np.pi) + logdetV + logdetA + plan.ldRx + yj @ Py
        th_terms, th_res, _ = split_theta(theta, terms, res)
        blocs = [colonnes_terme(th_terms[k], tm, Zr[k], rows_of[k], Py)
                 for k, tm in enumerate(terms)]
        o = 0
        for sec in secs:
            p_s = sec_n_params(sec)
            blocs.append(colonnes_section(th_res[o:o + p_s], sec, Py))
            o += p_s
        Q = jnp.concatenate(blocs, axis=1)
        PQ = sol(Q) - ViXA @ (ViX.T @ Q)
        AI = Q.T @ PQ
        # adjoint de V, forme EN DERNIER : L ne sert plus qu'a lui. Compromis
        # memoire / operations selon adjoint_choisi().
        W = _V_inverse(L, mode_adj)
        W = W - jnp.concatenate([ViXA, Py[:, None]], axis=1) \
            @ jnp.concatenate([ViX, Py[:, None]], axis=1).T
        g = vjp_V(W)[0]
        return f, g, 0.5 * (AI + AI.T)

    _fga = jax.jit(fga)
    Zr = plan.Zr

    def fga_appel(theta):
        return _fga(jnp.asarray(theta), Zr)
    return fga_appel


# ==============================================================================
# Sorties de fin d'ajustement (beta, BLUP, PEV), cf. fit.effets_denses
# ==============================================================================
def V_hote(plan, theta):
    """V au point theta, ramenee sur l'hote (NumPy) ; le GPU est libere ensuite."""
    Vf = jax.jit(plan.V)
    V = np.asarray(Vf(jnp.asarray(theta), plan.Zr))
    return V
