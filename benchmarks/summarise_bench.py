#!/usr/bin/env python3
"""Tableau de synthese du banc a modele egal (bench_vs_reference.R).

    python3 benchmarks/summarise_bench.py benchmarks/results/bench_2026-09-28_*.csv [--md out.md]

Pour chaque (cas, n) : temps mural par logiciel, temps par evaluation, et
l'accord des optima : ecart de logLik remlax - asreml (meme convention), et
ecart relatif maximal des composantes de variance quand les deux logiciels
les rangent dans le meme ordre (iid, grm, ar1ar1 : oui ; us : asreml range
Sigma par lignes, remlax par colonnes du triangle inferieur, on compare
alors les ensembles tries)."""
import argparse, csv, glob, sys
from collections import OrderedDict

def lire(paths):
    rows = []
    for p in paths:
        for g in glob.glob(p):
            rows += list(csv.DictReader(open(g)))
    return rows

def comps(s):
    return [float(x) for x in s.split(";") if x not in ("", "NA")] if s else []

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+"); ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    rows = [r for r in lire(a.csv)]
    cles = OrderedDict()
    for r in rows:
        cles.setdefault((r["case"], int(float(r["n"]))), {})[(r["software"], r["backend"], r["tag"])] = r
    L = ["| case | n | software (backend) | wall s | solver s | evals | s / eval | logLik | agreement with asreml |", "|---|---|---|---|---|---|---|---|---|"]
    for (cas, n), d in cles.items():
        ref = next((r for (sw, bk, tg), r in d.items() if sw == "asreml" and r["status"] == "ok"), None)
        for (sw, bk, tg), r in d.items():
            if r["status"] != "ok":
                L.append("| %s | %d | %s | %s | | | | | %s |" % (cas, n, sw, r["status"], r.get("error", "")[:60]))
                continue
            acc = ""
            if ref is not None and sw != "asreml":
                try:
                    dll = float(r["logLik"]) - float(ref["logLik"])
                    a1, a2 = sorted(comps(r["components"])), sorted(comps(ref["components"]))
                    # asreml porte une composante de plus pour us (la variance residuelle fixee a 1) : on compare les n plus grandes en valeur absolue
                    k = min(len(a1), len(a2))
                    b1 = sorted(a1, key=abs)[-k:]; b2 = sorted(a2, key=abs)[-k:]
                    rel = max(abs(x - y) / max(abs(y), 1e-8) for x, y in zip(sorted(b1), sorted(b2)))
                    acc = "dlogLik %+.2e, comp rel %.1e" % (dll, rel) if sw == "remlax" else "comp rel %.1e" % rel
                except Exception as e:
                    acc = "n/a"
            L.append("| %s | %d | %s (%s%s) | %s | %s | %s | %s | %s | %s |" % (
                cas, n, sw, bk, ", " + tg if tg else "", r["wall_s"], r["solver_s"] if r["solver_s"] not in ("NA", "") else "",
                r["n_eval"], r["s_per_eval"], r["logLik"], acc))
    out = "\n".join(L)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")

if __name__ == "__main__":
    sys.exit(main())
