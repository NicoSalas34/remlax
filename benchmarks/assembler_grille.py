#!/usr/bin/env python3
"""Assemble les CSV de grille et separe ce qui doit l'etre.

POURQUOI CE SCRIPT PLUTOT QU'UN TABLEAU CROISE. Les nombres d'iterations
different entre backends sur une meme cellule (419 sur carte contre 317 sur
CPU, mesure). Un temps total melange donc DEUX effets : le cout unitaire d'une
evaluation, qui est ce qu'on veut comparer entre moteurs, et la longueur de la
trajectoire d'optimisation, qui depend du conditionnement et de l'ordre des
reductions. On rapporte les deux, et le temps par iteration qui les separe.

Il tourne sur des CSV PARTIELS : les jobs ecrivent cellule par cellule, donc une
grille en cours s'assemble et se lit sans attendre.
"""
import csv, glob, sys
from pathlib import Path

def lire(motifs):
    rows = []
    for m in motifs:
        for f in sorted(glob.glob(m)):
            for r in csv.DictReader(open(f)):
                if r.get("statut") == "ok":
                    for k in ("n", "q", "t", "p", "n_iter"):
                        r[k] = int(float(r[k])) if r[k] not in ("", "NA") else None
                    for k in ("fit_paroi_s", "fit_interne_s", "logLik"):
                        r[k] = float(r[k]) if r[k] not in ("", "NA") else float("nan")
                    rows.append(r)
    return rows

def main(motifs, sortie="grille_assemblee.csv"):
    rows = lire(motifs)
    if not rows:
        print("aucune ligne : les jobs n'ont peut-etre encore rien ecrit"); return 1
    for r in rows:
        # Le temps par iteration est la grandeur COMPARABLE entre moteurs : il
        # retire la longueur de trajectoire, qui n'est pas une propriete du
        # moteur mais de la surface et de l'ordre des reductions.
        r["s_par_iter"] = (r["fit_interne_s"] / r["n_iter"]) if r["n_iter"] else float("nan")

    champs = ["cellule", "cas", "moteur", "backend", "n", "q", "t", "p",
              "n_iter", "fit_interne_s", "fit_paroi_s", "s_par_iter", "logLik", "tag"]
    with open(sortie, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=champs, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["n"], r["q"], r["p"], r["cellule"])))

    print("%d cellules dans %s\n" % (len(rows), sortie))
    print("=== le nombre d'iterations contre p, par moteur ===")
    par = {}
    for r in rows:
        par.setdefault((r["moteur"], r["p"]), []).append(r["n_iter"])
    for k in sorted(par):
        v = par[k]
        print("  moteur %-6s p=%-4d : %3d cellules, iterations %4d a %4d (mediane %4d)"
              % (k[0], k[1], len(v), min(v), max(v), sorted(v)[len(v) // 2]))

    print("\n=== le prix du mauvais moteur : cas dense force sur le creux ===")
    ref = {(r["n"], r["q"], r["p"]): r for r in rows if r["cellule"] == "creux-creux"}
    for r in sorted(rows, key=lambda r: (r["n"], r["q"], r["p"])):
        if r["cellule"] != "dense-creux":
            continue
        b = ref.get((r["n"], r["q"], r["p"]))
        if b:
            print("  n=%-6d q=%-5d p=%-3d : cas dense %9.1f s contre cas creux %7.2f s  -> x%.0f"
                  % (r["n"], r["q"], r["p"], r["fit_interne_s"], b["fit_interne_s"],
                     r["fit_interne_s"] / max(b["fit_interne_s"], 1e-9)))

    print("\n=== carte entiere contre CPU, meme cas et MEME moteur dense ===")
    idx = {}
    for r in rows:
        if r["moteur"] == "dense":
            idx.setdefault((r["cas"], r["n"], r["q"], r["p"]), {})[r["backend"]] = r
    for k in sorted(idx):
        d = idx[k]
        if "cpu" in d and "gpu" in d:
            c, g = d["cpu"], d["gpu"]
            print("  cas %-6s n=%-6d q=%-5d p=%-3d | CPU %9.1f s (%3d it) | carte %8.1f s (%3d it)"
                  "  -> total x%.1f, par iteration x%.1f"
                  % (k[0], k[1], k[2], k[3], c["fit_interne_s"], c["n_iter"],
                     g["fit_interne_s"], g["n_iter"],
                     c["fit_interne_s"] / max(g["fit_interne_s"], 1e-9),
                     c["s_par_iter"] / max(g["s_par_iter"], 1e-9)))
    return 0

if __name__ == "__main__":
    m = sys.argv[1:] or ["hpc/*/remlax/out/grille*.csv", "hpc/*/grille*.csv"]
    sys.exit(main(m))
