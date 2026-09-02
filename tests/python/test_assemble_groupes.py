"""L'assemblage par sous-blocs rend-il la MEME V, le MEME gradient, et un pic PLUS BAS ?

POURQUOI CES TROIS CONTROLES ET PAS DEUX. Une correction memoire anterieure de
ce solveur etait juste et INUTILE : verifiee sur la valeur et sur le gradient,
jamais sur le pic qu'elle etait censee reduire. Le troisieme controle est donc
le seul qui teste la propriete visee ; les deux premiers testent qu'on ne l'a
pas obtenue en cassant le calcul.

LES TOLERANCES SONT RELATIVES, PAS DES EGALITES. L'identite est exacte en
algebre, mais l'addition flottante n'est pas associative : disperser par
sous-blocs change l'ordre des sommations, donc les derniers bits.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "src")
from remlax import _x64  # noqa: F401  (active la double precision)
from remlax.model import (assemble_V, assemble_V_groupes, dense_Z, restrict_Z,
                          support_groups, n_theta, term_support)


def dispositif(n_par_bloc=140, n_blocs=5, n_traits=4, q_gen=60, q_iee=200, graine=0):
    """Un dispositif de la MEME FORME que le modele IGE : chaque terme ne
    concerne qu'un bloc et un caractere, ce qui est exactement la propriete que
    la restriction au support exploite."""
    rng = np.random.default_rng(graine)
    terms, n = [], n_par_bloc * n_blocs * n_traits
    lignes = np.arange(n).reshape(n_blocs, n_traits, n_par_bloc)
    # --- un terme genetique par espece, couvrant TOUS les blocs d'un caractere
    for esp, tr in ((0, range(n_traits // 2)), (1, range(n_traits // 2, n_traits))):
        rr = np.sort(lignes[:, list(tr), :].ravel())
        t = len(list(tr))
        zi = np.repeat(rr, 1)
        zj = rng.integers(0, t * q_gen, size=len(rr))
        terms.append(dict(name="gen%d" % esp, struct="us", t=t, rank=0, q=q_gen,
                          zi=zi, zj=zj, zx=np.ones(len(rr)), lvl="id", LK=None))
    # --- termes locaux : spatiaux (facteurs unitaires) et IEE (ponderes)
    for b in range(n_blocs):
        for tr in range(n_traits):
            rr = np.sort(lignes[b, tr, :])
            for q in (12, 48):                       # spatiaux : une entree/ligne
                terms.append(dict(name="spat_%d_%d_%d" % (b, tr, q), struct="iid",
                                  t=1, rank=0, q=q, zi=rr,
                                  zj=rng.integers(0, q, size=len(rr)),
                                  zx=np.ones(len(rr)), lvl="id", LK=None))
            k = 8                                     # IEE : k entrees ponderees
            zi = np.repeat(rr, k)
            terms.append(dict(name="iee_%d_%d" % (b, tr), struct="iid", t=1, rank=0,
                              q=q_iee, zi=zi,
                              zj=rng.integers(0, q_iee, size=len(zi)),
                              zx=rng.random(len(zi)), lvl="id", LK=None))
    # La residuelle a besoin de ses index : _section_V lit sec["trait"] et
    # sec["unit"] pour former Sigma_trait x C_unite terme a terme.
    tri = np.tile(np.repeat(np.arange(n_traits), n_par_bloc), n_blocs)
    res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, dtype=np.int64),
               unit=np.arange(n, dtype=np.int64), lvl="id")
    del tri
    y = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    return terms, res, y, X, n


def main():
    terms, res, y, X, n = dispositif()
    groupes = support_groups(terms, n)
    Zs = [dense_Z(t, n) for t in terms]
    Zr = [None] * len(terms)
    for rows, ks in groupes:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)

    p = n_theta(terms, res)
    rng = np.random.default_rng(7)
    th = jnp.asarray(rng.standard_normal(p) * 0.3)
    print("n = %d | %d termes | %d groupes de support | p = %d"
          % (n, len(terms), len(groupes), p))

    # ---------------------------------------------------------------- 1. valeur
    V0 = np.asarray(assemble_V(th, terms, Zs, res, n))
    V1 = np.asarray(assemble_V_groupes(th, terms, groupes, Zr, res, n))
    ech = max(abs(V0).max(), 1e-300)
    err = abs(V0 - V1).max() / ech
    print("\n1. V identique      : ecart relatif max %.3e" % err)
    assert err < 1e-12, "V differe : %.3e" % err

    # -------------------------------------------------------------- 2. gradient
    def f0(t_):
        return assemble_V(t_, terms, Zs, res, n).sum()

    def f1(t_):
        return assemble_V_groupes(t_, terms, groupes, Zr, res, n).sum()

    # SOUS COMPILATION : une identite peut tenir en interprete et casser une
    # fois fusionnee par XLA, qui reordonne les sommations.
    g0 = np.asarray(jax.jit(jax.grad(f0))(th))
    g1 = np.asarray(jax.jit(jax.grad(f1))(th))
    eg = abs(g0 - g1).max() / max(abs(g0).max(), 1e-300)
    print("2. gradient identique : ecart relatif max %.3e" % eg)
    assert eg < 1e-10, "gradient differe : %.3e" % eg

    # ------------------------------------------------------------------ 3. PIC
    # LE SEUL CONTROLE QUI TESTE LA PROPRIETE VISEE.
    mo_plein = sum(z.size for z in Zs) * 8 / 2**20
    mo_restr = sum(z.size for z in Zr) * 8 / 2**20
    print("\n3. facteurs pleine hauteur : %8.1f Mo" % mo_plein)
    print("   facteurs restreints     : %8.1f Mo  (/%.1f)"
          % (mo_restr, mo_plein / max(mo_restr, 1e-9)))
    assert mo_restr < mo_plein / 2, "aucun gain reel : %.1f contre %.1f" % (mo_restr, mo_plein)

    # LE PIC SE MESURE DANS UN PROCESSUS SEPARE, PAS AVEC UN COMPTEUR REMIS A
    # ZERO. peak_bytes_in_use est le haut niveau CUMULE du processus : une
    # seconde mesure dans le meme processus ne peut qu'etre superieure ou egale
    # a la premiere, donc la forme la plus econome rend le meme nombre que
    # l'autre. Mesure : 866.2 Mo pour les deux formes, alors que les facteurs
    # passent d'un facteur 14 — la mesure etait cassee, pas l'optimisation.
    # (Ma tentative de remise a zero, `d.memory_stats() and
    # d.clear_memory_stats()`, etait avalee par un try/except.)
    import subprocess
    GABARIT = """
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "src"); sys.path.insert(0, "tests/python")
from remlax import _x64
from remlax.model import (assemble_V, assemble_V_groupes, dense_Z,
                          restrict_Z, support_groups, n_theta)
from test_assemble_groupes import dispositif
terms, res, y, X, n = dispositif()
gr = support_groups(terms, n)
if "__FORME__" == "pleine":
    Zs = [dense_Z(t, n) for t in terms]
    f = lambda th: assemble_V(th, terms, Zs, res, n).sum()
else:
    Zr = [None] * len(terms)
    for rows, ks in gr:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)
    f = lambda th: assemble_V_groupes(th, terms, gr, Zr, res, n).sum()
th = jnp.asarray(np.random.default_rng(7).standard_normal(n_theta(terms, res)) * 0.3)
jax.block_until_ready(jax.jit(jax.grad(f))(th))
m = 0
for d in jax.local_devices():
    st = d.memory_stats() or {}
    m = max(m, st.get("peak_bytes_in_use", 0))
print("PIC", m / 2**20)
"""
    pics = {}
    for forme in ("pleine", "groupee"):
        r = subprocess.run([sys.executable, "-c", GABARIT.replace("__FORME__", forme)],
                           capture_output=True, text=True)
        lig = [l for l in r.stdout.splitlines() if l.startswith("PIC ")]
        if not lig:
            print("   (mesure du pic indisponible : %s)" % (r.stderr.strip().splitlines()[-1:] or [""])[0][:90])
        pics[forme] = float(lig[-1].split()[1]) if lig else 0.0

    p0, p1 = pics["pleine"], pics["groupee"]
    if p0 > 0 and p1 > 0:
        print("   pic du gradient, ancienne forme : %8.1f Mo" % p0)
        print("   pic du gradient, forme groupee  : %8.1f Mo  (/%.2f)"
              % (p1, p0 / max(p1, 1e-9)))
        if p1 >= p0:
            print("   ATTENTION : le pic n'a pas baisse. Les facteurs sont bien %.1f fois"
                  % (mo_plein / max(mo_restr, 1e-9)))
            print("   plus petits, donc si le pic ne suit pas, il est domine par un AUTRE")
            print("   poste — a chercher avant de conclure quoi que ce soit sur la VRAM.")
    else:
        print("   pic non instrumente sur ce peripherique (CPU) : mesure sur GPU")

    print("\n%d verifications, 0 echec" % 3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
