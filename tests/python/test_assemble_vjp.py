"""La regle de derivation de l'assemblage rend-elle EXACTEMENT le meme gradient ?

POURQUOI CE FICHIER EXISTE. L'assemblage de V porte desormais sa propre regle de
derivation, pour que la passe arriere ne conserve pas une matrice n x n par
terme — 65 termes a n = 16211 demandaient 127 Go sur le modele reel. Une regle
mal posee donne un gradient FAUX sans rien signaler : l'ajustement converge
ailleurs, ou pas du tout, et rien ne dit pourquoi.

LE CONTROLE. On compare a la chaine naive, terme a terme, sur un modele qui
melange les cas : `us` multi-caracteres, `iid`, une structure de niveaux a
parametres (ar1), et une residuelle a deux sections. La valeur doit etre
identique au bit pres ; le gradient a la precision de l'arithmetique.
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import jax
import jax.numpy as jnp
from remlax.model import (assemble_V, _assemble_V_fwd_val, split_theta,
                          term_factor, residual_V, dense_Z, n_theta)
from remlax.core import reml_from_V

ok = ko = 0
def chk(lbl, cond, det=""):
    global ok, ko
    if cond: ok += 1; print("  ok    %-46s %s" % (lbl, det))
    else:    ko += 1; print("  ECHEC %-46s %s" % (lbl, det))


def chaine_naive(theta, terms, Zs, res, n):
    """L'assemblage AVANT la regle : une addition n x n par terme."""
    th_terms, th_res, _ = split_theta(theta, terms, res)
    V = residual_V(th_res, res, n)
    for k, tm in enumerate(terms):
        B = term_factor(th_terms[k], tm, Zs[k])
        V = V + B @ B.T
    return V


def dispositif(seed=0):
    rng = np.random.default_rng(seed)
    n_unit, t, q = 60, 3, 12
    n = n_unit * t
    gu = np.repeat(np.arange(q), n_unit // q + 1)[:n_unit]
    tri = np.tile(np.arange(t), n_unit)
    uni = np.repeat(np.arange(n_unit), t)
    zj = np.array([tri[i] * q + gu[uni[i]] for i in range(n)])
    y = rng.normal(size=n)
    X = np.eye(t)[tri]
    # un terme us multi-caracteres, un terme iid, un terme a niveaux ar1
    terms = [
        dict(name="g", struct="us", rank=0, t=t, q=q, zi=np.arange(n), zj=zj,
             zx=np.ones(n), lvl="id", LK=None),
        dict(name="b", struct="iid", rank=0, t=1, q=6,
             zi=np.arange(n), zj=np.arange(n) % 6, zx=np.ones(n), lvl="id", LK=None),
        dict(name="s", struct="iid", rank=0, t=1, q=10,
             zi=np.arange(n), zj=np.arange(n) % 10, zx=np.ones(n),
             lvl="ar1", LK=None),
    ]
    res = dict(struct="diag", t=t, rank=0, trait=tri, unit=uni, lvl="id")
    return terms, res, y, X, n


def main():
    terms, res, y, X, n = dispositif()
    Zs = [dense_Z(tm, n) for tm in terms]
    p = n_theta(terms, res)
    print("modele : %d termes, %d parametres, n = %d\n" % (len(terms), p, n))

    rng = np.random.default_rng(7)
    for essai in range(3):
        th = jnp.asarray(rng.normal(scale=0.4, size=p))

        V_new = assemble_V(th, terms, Zs, res, n)
        V_ref = chaine_naive(th, terms, Zs, res, n)
        e = float(jnp.abs(V_new - V_ref).max())
        chk("essai %d : V identique" % essai, e == 0.0, "ecart max %.1e" % e)

        f_new = lambda t_: reml_from_V(assemble_V(t_, terms, Zs, res, n), y, X)
        f_ref = lambda t_: reml_from_V(chaine_naive(t_, terms, Zs, res, n), y, X)
        v_new, g_new = jax.value_and_grad(f_new)(th)
        v_ref, g_ref = jax.value_and_grad(f_ref)(th)
        chk("essai %d : -2logL identique" % essai,
            float(abs(v_new - v_ref)) == 0.0,
            "%.10f contre %.10f" % (v_new, v_ref))
        rel = float(jnp.abs(g_new - g_ref).max() /
                    jnp.maximum(jnp.abs(g_ref).max(), 1e-30))
        chk("essai %d : gradient identique" % essai, rel < 1e-10,
            "ecart relatif max %.2e" % rel)

    # LE CONTROLE QUI ATTRAPE UNE REGLE PLAUSIBLE MAIS FAUSSE : une difference
    # finie ne connait rien de la regle et ne peut pas partager son erreur.
    th = jnp.asarray(rng.normal(scale=0.3, size=p))
    f_new = lambda t_: reml_from_V(assemble_V(t_, terms, Zs, res, n), y, X)
    g = jax.grad(f_new)(th)
    h = 1e-6
    pires = []
    for j in range(p):
        e_j = jnp.zeros(p).at[j].set(h)
        gd = float((f_new(th + e_j) - f_new(th - e_j)) / (2 * h))
        pires.append(abs(gd - float(g[j])) / max(abs(gd), 1.0))
    chk("gradient contre differences finies", max(pires) < 1e-5,
        "ecart relatif max %.2e sur %d parametres" % (max(pires), p))

    print("\n%d verification(s), %d echec(s)" % (ok + ko, ko))
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())
