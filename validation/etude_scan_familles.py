"""Quelle covariable detecte quel mecanisme ? Etude des trois familles du scan.

LA QUESTION. Le modele de Sato et al. (2021, 2024) porte, par SNP, l'effet du
genotype focal et le PRODUIT focal x voisin — une similarite allelique, donc un
DGE x IGE. Le modele d'effets genetiques indirects au sens de Muir (2005) et
Bijma (2011) porte l'effet du genotype focal et la DOSE ALLELIQUE DU VOISIN,
additive. Ce sont deux hypotheses biologiques distinctes, et aucun des deux
modeles ne contient l'autre. Si le mecanisme reel est additif, le terme produit
de Sato le voit-il ? Si le mecanisme est une similarite, le terme additif le
voit-il ? Et le modele qui porte les trois familles perd-il quelque chose a
depenser des degres de liberte ?

CE QUI EST SIMULE. Un dispositif randomise en blocs, un fond polygenique DGE +
IGE avec covariance (structure `us` 2x2) et une residuelle AR1 dans les blocs.
Un SNP causal dont le mecanisme varie ; les autres SNP sont nuls. Le modele nul
du scan est LE VRAI modele : la question posee ici est celle de la covariable
d'effet fixe, pas celle de la specification des composantes de variance — ce
dernier point est deja tranche ailleurs (voir docs/scan.md).

L'EFFET EST CALIBRE EN VARIANCE, PAS EN COEFFICIENT. Les trois covariables
n'ont pas la meme dispersion : la dose du voisin est une moyenne sur J voisins,
donc moins dispersee que la dose du focal, et le produit moins encore. Comparer
a coefficient egal comparerait des signaux d'amplitude differente. On fixe donc
l'ecart-type de la CONTRIBUTION du terme causal, identique pour les quatre
mecanismes.

Lancement (depuis la racine du depot) :
    python validation/etude_scan_familles.py
"""
import importlib.util
import os
import sys
import time

import numpy as np
from scipy.stats import chi2 as chi2_dist

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "src"))
sys.path.insert(0, os.path.join(RACINE, "tests", "python"))


def _charger(nom, chemin):
    sp = importlib.util.spec_from_file_location(nom, chemin)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_scan = _charger("remlax_scan", os.path.join(RACINE, "src", "remlax", "scan.py"))
_t = _charger("t_scan", os.path.join(RACINE, "tests", "python", "test_scan.py"))
projeter, scan, lambda_gc = _scan.projeter, _scan.scan, _scan.lambda_gc

RNG = np.random.default_rng(90210)

# Les mecanismes reels au SNP causal.
MECANISMES = ("direct", "voisin additif", "similarite", "voisin + similarite")

# Les modeles candidats, en specifications de `scan`.
TESTS = [
    "dir",              # GWAS standard
    "ind",              # IGE additif seul (dose du voisin)
    "sim",              # similarite seule
    "dir+sim",          # le modele de Sato : focal + produit
    "dir+ind",          # le modele d'IGE additif : focal + dose du voisin
    "dir+ind+sim",      # les trois familles, 3 ddl
    "ind|dir+sim",      # y a-t-il un effet ADDITIF du voisin en plus de Sato ?
    "sim|dir+ind",      # y a-t-il une INTERACTION en plus des effets additifs ?
]

N_REP = 300
SIGNAL = 0.40       # ecart-type de la contribution causale, en unites de y


def contributions(d, j):
    """Les trois covariables du SNP j, au niveau observation."""
    u = d["Zdir"] @ d["Mdir"][:, j]
    v = d["Zind"] @ d["Mind"][:, j]
    return {"direct": u, "voisin additif": v, "similarite": u * v}


def main():
    t0 = time.time()
    d = _t.dispositif(q=90, n_bloc=4, nrow=9, ncol=10, p=300)
    V, _, _ = _t.ajuste(d)
    n, p, X = d["n"], d["p"], d["X"]
    Vi = np.linalg.solve(V, np.eye(n))
    XtVi = X.T @ Vi
    Am = XtVi @ X
    vbeta = np.linalg.pinv(Am)
    Lv = np.linalg.cholesky(V + 1e-9 * np.eye(n))
    b_fixe = np.array([1.0, 1.3, 0.7, -0.4])

    def fit_de(y):
        beta = np.linalg.solve(Am, XtVi @ y)
        resid = y - X @ beta
        Py = Vi @ resid - Vi @ X @ (vbeta @ (XtVi @ resid))
        return dict(Vi=Vi, vbeta=vbeta, Py=Py)

    def scanner(y):
        proj = projeter(fit_de(y), X, {"dir": d["Zdir"], "ind": d["Zind"]})
        return scan(proj, {"dir": d["Mdir"], "ind": d["Mind"]}, tests=TESTS)

    # ---- (I) calibration : aucun SNP causal -------------------------------
    lam = {t: [] for t in TESTS}
    for _ in range(N_REP):
        y = X @ b_fixe + Lv @ RNG.normal(size=n)
        r = scanner(y)
        for t in TESTS:
            lam[t].append(lambda_gc(r["chi2_" + t], int(r["ddl_" + t][0])))
    print("calibration (lambda, mediane sur %d repetitions, aucun SNP causal)" % N_REP)
    for t in TESTS:
        print("    %-14s %.3f" % (t, np.median(lam[t])))

    # ---- (II) puissance par mecanisme -------------------------------------
    det = {m: {t: [] for t in TESTS} for m in MECANISMES}
    for meca in MECANISMES:
        for _ in range(N_REP):
            j0 = int(RNG.integers(p))
            c = contributions(d, j0)
            if meca == "voisin + similarite":
                # les deux composantes a poids egal, chacune reduite d'abord
                sig = (c["voisin additif"] / c["voisin additif"].std()
                       + c["similarite"] / c["similarite"].std())
            else:
                sig = c[meca]
            # CALIBRATION EN VARIANCE : la contribution causale a toujours
            # l'ecart-type SIGNAL, quel que soit le mecanisme.
            y = X @ b_fixe + SIGNAL * (sig / sig.std()) + Lv @ RNG.normal(size=n)
            r = scanner(y)
            for t in TESTS:
                ddl = int(r["ddl_" + t][0])
                s = chi2_dist.isf(0.05 / p, ddl)
                det[meca][t].append(bool(r["chi2_" + t][j0] > s))

    print("\npuissance : detection du SNP causal au seuil de Bonferroni "
          "(%d repetitions, signal d'ecart-type %.2f)" % (N_REP, SIGNAL))
    entete = "  %-14s" % "modele teste" + "".join("%18s" % m for m in MECANISMES)
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    tab = {}
    for t in TESTS:
        ligne = "  %-14s" % t
        for m in MECANISMES:
            v = 100 * float(np.mean(det[m][t]))
            tab[(t, m)] = v
            ligne += "%17.0f%%" % v
        print(ligne)

    np.savez(os.path.join(RACINE, "validation", "results", "scan_familles.npz"),
             tests=np.array(TESTS), mecanismes=np.array(MECANISMES),
             lam=np.array([np.median(lam[t]) for t in TESTS]),
             puissance=np.array([[tab[(t, m)] for m in MECANISMES] for t in TESTS]),
             n_rep=N_REP, signal=SIGNAL, n=n, p=p, q=d["q"])
    print("\n%.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
