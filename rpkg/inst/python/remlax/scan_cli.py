"""Point d'entree du scan GWAS, POSE A COTE de `cli.py`.

Ce module n'importe `cli.py` ni ne le modifie. Il reutilise `Bundle`,
`fit_reml` et `scan`, et rien d'autre.

    python -m remlax.scan_cli <paquet> --tests "dir,ind,dir+ind+sim,ind|dir+sim"

POURQUOI UN SECOND POINT D'ENTREE PLUTOT QU'UNE OPTION DE `remlax`. Le scan a
besoin de `Vi` et `Py`, qui sont des matrices n x n vivant dans le PROCESSUS de
l'ajustement : elles ne traversent pas le fichier de resultats vers R (2,1 Go a
n = 16 000). Le scan doit donc s'executer la ou l'ajustement s'execute. Comme
on ne touche pas a `cli.py`, c'est un second executable qui refait le chemin
paquet -> ajustement -> inference, avec le scan a la place des tests de Wald.

POURQUOI IL NE REOPTIMISE PAS. `--theta-in --maxiter 0` relit le theta deja
estime et se contente de l'EVALUER : V est reassemblee, Vi et Py sont calcules,
et l'optimiseur est court-circuite. Le scan ne paie donc pas une seconde
estimation des composantes de variance — c'est la meme, celle de l'ajustement
que l'utilisateur a deja lance et valide.

FORMAT DES ENTREES DU SCAN. Le paquet ecrit par `rx_export` n'est pas touche :
les tableaux propres au scan vivent a cote, decrits par `scan_manifest.json`,
et sont relus ici par un lecteur local de six lignes. Aucun risque de corrompre
`manifest.json`.

    scan_manifest.json   {"arrays": [{"name":..., "dtype":"f8"|"i4"|"str", "shape":[...]}],
                          "meta": {"q_dir":..., "q_ind":..., "p":...}}
    scan_dir_zi/zj/zx    incidence du genotype porte par la plante, en COO
    scan_ind_zi/zj/zx    incidence des genotypes voisins, ponderee, en COO
    scan_Mdir            doses genotypiques (q_dir x p), CENTREES pour `sim`
    scan_Mind            doses des genotypes voisins (q_ind x p)
    scan_snp             noms des SNP (facultatif)
    scan_chr, scan_pos   carte (facultatif)
    scan_garder          masque 0/1 des SNP a traiter (facultatif)
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def _lire(paquet, man, nom):
    m = man[nom]
    if m["dtype"] == "str":
        with open(os.path.join(paquet, nom + ".txt")) as f:
            return [l.rstrip("\n") for l in f]
    dt = {"f8": np.float64, "i4": np.int32}[m["dtype"]]
    a = np.fromfile(os.path.join(paquet, nom + ".bin"), dtype=dt)
    shp = tuple(int(v) for v in m["shape"])
    return a.reshape(shp, order="F") if len(shp) > 1 else a


def _coo(paquet, man, pre, n, q):
    from scipy.sparse import coo_matrix
    i = _lire(paquet, man, pre + "_zi").astype(np.int64)
    j = _lire(paquet, man, pre + "_zj").astype(np.int64)
    x = _lire(paquet, man, pre + "_zx").astype(np.float64)
    return coo_matrix((x, (i, j)), shape=(n, q)).toarray()


def main(argv=None):
    from remlax.bundle import Bundle
    from remlax.fit import fit_reml
    from remlax.device import pick_device, device_report
    from remlax.scan import projeter, scan, lambda_gc, familles_requises

    ap = argparse.ArgumentParser(
        prog="remlax-scan",
        description="GWAS a V figee sur un paquet remlax (voir docs/scan.md).")
    ap.add_argument("bundle")
    ap.add_argument("--tests", default="dir,ind,sim,dir+ind+sim",
                    help="specifications separees par des virgules ; barre pour "
                         "un test conditionnel, par exemple 'ind|dir+sim'")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--theta-in", action="store_true",
                    help="relit in_theta.bin ; avec --maxiter 0 le theta est "
                         "seulement EVALUE, sans reoptimisation")
    ap.add_argument("--maxiter", type=int, default=0)
    ap.add_argument("--polish", type=int, default=0)
    ap.add_argument("--bloc", type=int, default=1024,
                    help="largeur des blocs de SNP pour la famille 'sim'")
    ap.add_argument("--sortie", default="scan_resultats.csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    import jax
    dev, plat = pick_device(a.backend)
    ctx = jax.default_device(dev)
    ctx.__enter__()
    if not a.quiet:
        print("[scan] backend retenu : %s (%s)" % (plat, dev), flush=True)

    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    besoin = familles_requises(tests)

    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    n = len(y)

    with open(os.path.join(a.bundle, "scan_manifest.json")) as f:
        sm = json.load(f)
    man = {d["name"]: d for d in sm["arrays"]}
    meta = sm.get("meta", {})

    # `scan_M` present = LES MEMES SNP des deux cotes (cas intra-espece) : la
    # matrice n'est ecrite qu'une fois, et un seul tableau est passe a `scan`.
    # Cela evite de dupliquer 292 Mo au format du projet, et cela leve toute
    # ambiguite sur l'alignement des colonnes pour les tests conjoints.
    M_partage = _lire(a.bundle, man, "scan_M") if "scan_M" in man else None
    incid, M = {}, {}
    for fam, pre, cq, cm in (("dir", "scan_dir", "q_dir", "scan_Mdir"),
                             ("ind", "scan_ind", "q_ind", "scan_Mind")):
        if pre + "_zi" not in man:
            continue
        incid[fam] = _coo(a.bundle, man, pre, n, int(meta[cq]))
        if M_partage is None:
            M[fam] = _lire(a.bundle, man, cm)
    if M_partage is not None:
        M = M_partage
    manquantes = [f for f in besoin if f in ("dir", "ind") and f not in incid]
    if "sim" in besoin:
        manquantes += [f for f in ("dir", "ind") if f not in incid]
    if manquantes:
        raise SystemExit(
            "[scan] tests %s demandes mais incidence(s) absente(s) du paquet : %s. "
            "Cote R, `incidences=` doit nommer les termes du modele qui les "
            "portent." % (tests, sorted(set(manquantes))))

    # --- ajustement (ou simple evaluation du theta fourni) -----------------
    th0 = None
    if a.theta_in:
        th0 = np.fromfile(os.path.join(a.bundle, "in_theta.bin"), dtype=np.float64)
    elif a.maxiter == 0:
        raise SystemExit("[scan] --maxiter 0 sans --theta-in : il n'y a alors "
                         "aucun theta a evaluer. Fournir in_theta.bin, ou "
                         "demander un vrai ajustement avec --maxiter > 0.")
    t0 = time.time()
    fit = fit_reml(terms, res, y, X, theta_init=th0, maxiter=a.maxiter,
                   polish=a.polish, verbose=not a.quiet,
                   hessian=False, blups=True, pev=False)
    t_fit = time.time() - t0
    if not a.quiet:
        print("[scan] modele nul : -2logL = %.6f en %.1f s"
              % (fit["neg2_reml"], t_fit), flush=True)

    # --- projection puis scan ----------------------------------------------
    t0 = time.time()
    proj = projeter(fit, X, incid)
    t_proj = time.time() - t0
    t0 = time.time()
    r = scan(proj, M, tests=tests, bloc=a.bloc, verbose=not a.quiet,
             garder=(_lire(a.bundle, man, "scan_garder").astype(bool)
                     if "scan_garder" in man else None))
    t_scan = time.time() - t0
    if not a.quiet:
        print("[scan] projection %.2f s | scan de %d SNP %.2f s"
              % (t_proj, int(meta["p"]), t_scan), flush=True)
        for t in tests:
            print("        lambda(%s) = %.3f"
                  % (t, lambda_gc(r["chi2_" + t], int(r["ddl_" + t][0]))), flush=True)

    # --- table de sortie ----------------------------------------------------
    p = int(meta["p"])
    cols = {}
    cols["snp"] = (_lire(a.bundle, man, "scan_snp") if "scan_snp" in man
                   else ["snp%d" % (i + 1) for i in range(p)])
    if "scan_chr" in man:
        cols["chr"] = _lire(a.bundle, man, "scan_chr")
    if "scan_pos" in man:
        cols["pos"] = _lire(a.bundle, man, "scan_pos")
    for k, v in r.items():
        cols[k] = v
    noms = list(cols)
    chemin = os.path.join(a.bundle, a.sortie)
    with open(chemin, "w") as f:
        f.write(",".join(noms) + "\n")
        for i in range(p):
            f.write(",".join(
                ("" if isinstance(cols[k][i], float) and not np.isfinite(cols[k][i])
                 else ("%s" % cols[k][i] if isinstance(cols[k][i], str)
                       else "%.10g" % cols[k][i]))
                for k in noms) + "\n")
    with open(os.path.join(a.bundle, "scan_meta.json"), "w") as f:
        json.dump(dict(tests=tests, n=n, p=p, secondes_fit=t_fit,
                       secondes_projection=t_proj, secondes_scan=t_scan,
                       neg2_reml=float(fit["neg2_reml"]),
                       lambda_gc={t: lambda_gc(r["chi2_" + t],
                                               int(r["ddl_" + t][0]))
                                  for t in tests}), f, indent=1)
    if not a.quiet:
        print("[scan] ecrit : %s" % chemin, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
