#!/usr/bin/env python3
"""Parite CPU / GPU du solveur, sur une batterie de dispositifs serialises.

CE QUE MESURE CE TEST. Le meme paquet, la meme vraisemblance, le meme
optimiseur : SEULE la machine change. Tout ecart au-dela du bruit de reduction
est un bug, pas de l'arrondi — un GPU ne reordonne les sommes qu'a l'interieur
d'une precision fixee, et on est en float64 des deux cotes.

DEUX QUANTITES, pas une :
  -2logL   compare l'OPTIMUM atteint. Deux backends peuvent y arriver par des
           chemins differents ; c'est la valeur qui compte.
  theta    compare le POINT. Un ecart sur theta avec une logLik identique
           signale une crete plate (structure surparametree), pas une erreur —
           d'ou les deux colonnes.

    python3 scripts/tests/parite_gpu.py <repertoire_de_paquets> [--tol 1e-8]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "inst", "python"))

from remlax import _x64  # noqa: F401,E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
from remlax.bundle import Bundle  # noqa: E402
from remlax.device import pick_device, device_report  # noqa: E402
from remlax.fit import fit_reml  # noqa: E402


def ajuste(paquet, backend):
    dev, plat = pick_device(backend)
    with jax.default_device(dev):
        b = Bundle(paquet)
        t0 = time.time()
        r = fit_reml(b.terms(), b.residual(), b.y, b.X, verbose=False,
                     hessian=False, blups=False)
        return dict(neg2=r["neg2_reml"], theta=np.asarray(r["theta"]),
                    plat=plat, s=time.time() - t0,
                    sig={k: np.asarray(v) for k, v in r["sigmas"].items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)

    print("peripheriques : %s" % device_report(), flush=True)
    dispo = [d.platform for d in jax.devices()]
    if "gpu" not in dispo and "cuda" not in dispo:
        print("AUCUN GPU visible : le test de parite n'a rien a comparer.", flush=True)
        return 2
    noms = sorted(d for d in os.listdir(a.bundles)
                  if os.path.exists(os.path.join(a.bundles, d, "manifest.json")))
    if a.only:
        garde = set(a.only.split(","))
        noms = [n for n in noms if n in garde]
    print("%d paquet(s) dans %s\n" % (len(noms), a.bundles), flush=True)
    print("%-14s %16s %16s %11s %11s %8s %8s" %
          ("modele", "-2logL CPU", "-2logL GPU", "ecart", "max|dtheta|", "s CPU", "s GPU"), flush=True)
    print("-" * 92, flush=True)
    ko, err = [], []
    for nm in noms:
        p = os.path.join(a.bundles, nm)
        try:
            c = ajuste(p, "cpu")
            g = ajuste(p, "gpu")
        except Exception as e:
            err.append((nm, "%s: %s" % (type(e).__name__, e)))
            print("%-14s %s" % (nm, "ECHEC : %s" % e), flush=True)
            continue
        d_ll = abs(c["neg2"] - g["neg2"])
        rel = d_ll / max(abs(c["neg2"]), 1.0)
        d_th = float(np.max(np.abs(c["theta"] - g["theta"]))) if len(c["theta"]) else 0.0
        drapeau = "" if rel <= a.tol else "   <-- ECART"
        if rel > a.tol:
            ko.append((nm, rel, d_th))
        print("%-14s %16.8f %16.8f %11.2e %11.2e %8.1f %8.1f%s"
              % (nm, c["neg2"], g["neg2"], rel, d_th, c["s"], g["s"], drapeau), flush=True)
    print("-" * 92, flush=True)
    if err:
        print("\n%d paquet(s) en erreur :" % len(err))
        for nm, e in err:
            print("   %-14s %s" % (nm, e))
    if ko:
        print("\n%d ecart(s) au-dela de %.1e :" % (len(ko), a.tol))
        for nm, rel, dt in ko:
            print("   %-14s -2logL rel %.2e, max|dtheta| %.2e" % (nm, rel, dt))
        return 1
    if err:
        return 1
    print("\nParite CPU/GPU verifiee sur %d modeles (tolerance relative %.0e)."
          % (len(noms), a.tol))
    return 0


if __name__ == "__main__":
    sys.exit(main())
