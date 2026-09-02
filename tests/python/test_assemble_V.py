"""L'assemblage en UN produit rend-il exactement la meme chose que la chaine ?

POURQUOI CE FICHIER EXISTE. L'assemblage utilise l'identite
Sum_k B_k B_k' = [B_1 ... B_K][B_1 ... B_K]', qui ne laisse qu'UN intermediaire
n x n au lieu d'un par terme — 65 termes a n = 16211 demandaient 127 Go sur le
modele reel. La reecriture est exacte en algebre, mais une concatenation dans le
mauvais ordre ou une largeur mal calculee donne un V FAUX sans rien signaler.

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
from remlax.model import (assemble_V, split_theta, term_factor,
                          residual_V, dense_Z, n_theta)
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


def dispositif_nombreux(n_unit=200, t=3, q=20, n_nuis=40, seed=5):
    """La FORME du modele reel : peu de termes genetiques, beaucoup de nuisances."""
    rng = np.random.default_rng(seed)
    n = n_unit * t
    tri = np.tile(np.arange(t), n_unit); uni = np.repeat(np.arange(n_unit), t)
    gu = np.repeat(np.arange(q), n_unit // q + 1)[:n_unit]
    zj = np.array([tri[i] * q + gu[uni[i]] for i in range(n)])
    terms = [dict(name="g", struct="us", rank=0, t=t, q=q, zi=np.arange(n),
                  zj=zj, zx=np.ones(n), lvl="id", LK=None)]
    for i in range(n_nuis):
        terms.append(dict(name="z%d" % i, struct="iid", rank=0, t=1, q=12,
                          zi=np.arange(n), zj=(np.arange(n) + i) % 12,
                          zx=np.ones(n), lvl="id", LK=None))
    res = dict(struct="diag", t=t, rank=0, trait=tri, unit=uni, lvl="id")
    return terms, res, rng.normal(size=n), np.eye(t)[tri], n


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
        # TOLERANCE ET NON EGALITE EXACTE, et ce n'est pas un relachement :
        # Sum_k B_k B_k' = [B_1...B_K][B_1...B_K]' est exact en ALGEBRE, mais
        # l'addition flottante n'est pas associative. Regrouper les termes en un
        # produit change l'ordre des sommes, donc les derniers bits. Exiger 0
        # exactement testerait l'associativite de l'arithmetique, pas l'identite.
        rel = float(jnp.abs(V_new - V_ref).max() / jnp.abs(V_ref).max())
        chk("essai %d : V identique a la chaine" % essai, rel < 1e-13,
            "ecart relatif max %.1e" % rel)

        f_new = lambda t_: reml_from_V(assemble_V(t_, terms, Zs, res, n), y, X)
        f_ref = lambda t_: reml_from_V(chaine_naive(t_, terms, Zs, res, n), y, X)
        v_new, g_new = jax.value_and_grad(f_new)(th)
        v_ref, g_ref = jax.value_and_grad(f_ref)(th)
        chk("essai %d : -2logL identique" % essai,
            float(abs(v_new - v_ref) / abs(v_ref)) < 1e-13,
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

    # LE CONTROLE QUE J'AVAIS OMIS LA PREMIERE FOIS, et c'est la raison meme du
    # changement : la correction ne suffit pas, il faut que le PIC BAISSE. Une
    # premiere tentative etait correcte et n'a rien change au pic ; verifier la
    # valeur et le gradient sans verifier la propriete visee ne prouve rien.
    # LE PIC SE MESURE SUR BEAUCOUP DE TERMES. Le gain croit avec leur nombre :
    # un modele a trois termes n'en montre rien, et c'est precisement pourquoi
    # la limite n'est apparue que sur le modele reel a 65 termes.
    terms_m, res_m, y_m, X_m, n_m = dispositif_nombreux()
    Zs_m = [dense_Z(tm, n_m) for tm in terms_m]
    th_m = jnp.asarray(np.random.default_rng(3).normal(
        scale=0.3, size=n_theta(terms_m, res_m)))

    def pic(f, t_):
        fj = jax.jit(jax.value_and_grad(f))
        v, _ = fj(t_); jax.block_until_ready(v)
        return fj.lower(t_).compile().memory_analysis().temp_size_in_bytes
    t_ref = pic(lambda t_: reml_from_V(
        chaine_naive(t_, terms_m, Zs_m, res_m, n_m), y_m, X_m), th_m)
    t_new = pic(lambda t_: reml_from_V(
        assemble_V(t_, terms_m, Zs_m, res_m, n_m), y_m, X_m), th_m)
    chk("le pic temporaire BAISSE (%d termes)" % len(terms_m), t_new < 0.5 * t_ref,
        "%.1f Mo contre %.1f, soit /%.1f" % (t_new/2**20, t_ref/2**20, t_ref/max(t_new,1)))

    print("\n%d verification(s), %d echec(s)" % (ok + ko, ko))
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())
