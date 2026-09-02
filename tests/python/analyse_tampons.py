"""Quels tampons XLA alloue-t-il, et de quelle taille ?

POURQUOI CE FICHIER EXISTE. La sonde sur tranche echoue desormais sur une
allocation de 4,02 Gio que mon decompte n'explique pas : ce n'est pas V (646 Mo
a n = 8987), ce n'est pas un facteur restreint (8,1 Mo au total sur un
dispositif de meme forme). J'ai une hypothese — la dispersion par indexation
avancee materialiserait un intermediaire pleine taille par groupe — mais une
hypothese ne se publie pas.

XLA sait le dire exactement. Et c'est une propriete de COMPILATION, donc
mesurable sur CPU sans GPU et sans attendre en file.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "src")
sys.path.insert(0, "tests/python")
from remlax import _x64  # noqa: F401
from remlax.model import (assemble_V, assemble_V_groupes, dense_Z, restrict_Z,
                          support_groups, n_theta, reml_from_V)
from test_assemble_groupes import dispositif


def analyse(nom, f, th):
    c = jax.jit(f).lower(th).compile()
    a = c.memory_analysis()
    mo = lambda v: (v or 0) / 2**20
    print("\n--- %s" % nom)
    for cle in ("temp_size_in_bytes", "argument_size_in_bytes",
                "output_size_in_bytes", "generated_code_size_in_bytes",
                "alias_size_in_bytes", "host_temp_size_in_bytes"):
        v = getattr(a, cle, None)
        if v:
            print("    %-30s %9.1f Mo" % (cle, mo(v)))
    return mo(getattr(a, "temp_size_in_bytes", 0))


def main():
    # Dispositif de la MEME FORME que le modele IGE, a l'echelle reelle : cinq
    # blocs, quatre caracteres, et des termes locaux a un bloc et un caractere.
    terms, res, y, X, n = dispositif(n_par_bloc=450, n_blocs=5, n_traits=4,
                                     q_gen=181, q_iee=1920)
    groupes = support_groups(terms, n)
    Zs = [dense_Z(t, n) for t in terms]
    Zr = [None] * len(terms)
    for rows, ks in groupes:
        for k in ks:
            Zr[k] = restrict_Z(terms[k], rows, n)
    p = n_theta(terms, res)
    th = jnp.asarray(np.random.default_rng(7).standard_normal(p) * 0.3)
    yj, Xj = jnp.asarray(y), jnp.asarray(X)
    print("n = %d | %d termes | %d groupes | p = %d | V = %.0f Mo"
          % (n, len(terms), len(groupes), p, n * n * 8 / 2**20))
    print("facteurs pleine hauteur %.0f Mo | restreints %.0f Mo"
          % (sum(z.size for z in Zs) * 8 / 2**20,
             sum(z.size for z in Zr) * 8 / 2**20))

    # L'OBJECTIF COMPLET, gradient compris : c'est ce que la sonde execute.
    def obj_plein(t_):
        V = assemble_V(t_, terms, Zs, res, n)
        return reml_from_V(V, yj, Xj)

    def obj_group(t_):
        V = assemble_V_groupes(t_, terms, groupes, Zr, res, n)
        return reml_from_V(V, yj, Xj)

    t0 = analyse("assemblage plein, valeur seule", obj_plein, th)
    t1 = analyse("assemblage groupe, valeur seule", obj_group, th)
    g0 = analyse("assemblage plein, GRADIENT", jax.grad(obj_plein), th)
    g1 = analyse("assemblage groupe, GRADIENT", jax.grad(obj_group), th)

    print("\n==================== BILAN ====================")
    print("  valeur   : plein %8.1f Mo | groupe %8.1f Mo | rapport %.2f"
          % (t0, t1, t0 / max(t1, 1e-9)))
    print("  gradient : plein %8.1f Mo | groupe %8.1f Mo | rapport %.2f"
          % (g0, g1, g0 / max(g1, 1e-9)))
    print("\n  V vaut %.0f Mo. Si le tampon du gradient groupe vaut plusieurs fois"
          % (n * n * 8 / 2**20))
    print("  V, c'est qu'un intermediaire pleine taille est alloue PAR GROUPE —")
    print("  la chaine d'additions n x n deguisee en chaine de dispersions.")
    print("  %d groupes x %.0f Mo = %.0f Mo attendus dans ce cas."
          % (len(groupes), n * n * 8 / 2**20, len(groupes) * n * n * 8 / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
