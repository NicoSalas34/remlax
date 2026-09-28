#!/usr/bin/env python3
"""Figure de validation : estimations remlax contre reference, tous controles.

    python3 validation/make_validation_fig.py --csv validation/results/checks_<date>.csv \
            --out validation/results/fig_validation.png [--pairs-out validation/results/pairs_<date>.csv]

Les paires (remlax, reference) sont lues dans la colonne `detail` du CSV, qui
cite la ligne du journal telle quelle. Trois ecritures y coexistent et sont
reconnues :
    "a vs b"                       une paire, remlax d'abord
    "remlax a b c | <ref> d e f"   deux listes de meme longueur
    "s2g a/b | s2e c/d"            des fractions remlax/reference
Une paire n'est retenue que si le libelle ne dit pas explicitement l'inverse
("asreml a <= remlax b" est un ordre, pas une paire ; "a <= b" est ignore).
Le panneau de droite porte les ecarts de -2logL (ou de logLik) pour les
controles dont le libelle contient "logL".

Chaque paire retenue est ecrite dans --pairs-out, pour que la figure soit
verifiable ligne a ligne.
"""
import argparse
import csv
import os
import re
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
RX_VS = re.compile(r"(?<![\w.])(%s)\s+vs\.?\s+(%s)(?![\w.])" % (NUM, NUM))
RX_FRAC = re.compile(r"(?<![\w./])(%s)/(%s)(?![\w./])" % (NUM, NUM))
RX_LISTS = re.compile(r"remlax(?: theta| phi)?\s+((?:%s\s*)+)\|\s*(?:asreml|nlme|lme4|sommer)(?: phi| theta)?\s+((?:%s\s*)+)" % (NUM, NUM))
RX_BAR = re.compile(r"(?:remlkit|remlax)\s+(%s)\s*\|\s*(?:R|asreml)\s+(%s)" % (NUM, NUM))

LOGICIEL = OrderedDict([
    ("asreml", ("asreml_structures", "asreml_catalogue", "asreml_rest", "asreml_blup", "ige_model")),
    ("lme4", ("lme4_sommer", "lme4_extra")),
    ("sommer", ("sommer_extra",)),
    ("nlme", ("nlme",)),
    ("closed form / internal", ("algebra", "sparse_parity", "sparse_vs_dense", "sparse_prec", "pytest")),
])
COULEUR = {"asreml": "#1f77b4", "lme4": "#2ca02c", "sommer": "#d62728", "nlme": "#9467bd",
           "closed form / internal": "#7f7f7f"}


def logiciel_de(suite, check):
    for sw, suites in LOGICIEL.items():
        if suite in suites:
            if suite == "lme4_sommer":
                return "sommer" if "sommer" in check.lower() else "lme4"
            return sw
    return "other"


EXCLURE = ("ameliore", "signe de", "emboitement", "domine", "moins bon", "constante",
           "a bouge", "decroit", "repliqu", "==", "identique a lme4",
           "voie explicite", "backend", "annonce", "ddl du terme", "degeneree", "disponible",
           "entre dans", "separees", "sections ameliorent", "GONFLE", "n'atteint pas")


def paires(detail, check):
    """Paires (remlax, reference) lues dans `detail`, ou [] si le libelle dit que
    la comparaison n'est pas remlax-contre-reference (deux modeles remlax, une
    verite simulee, un ordre)."""
    d = detail.replace("\\|", "|")
    out = []
    if "<=" in d or ">=" in d or any(k.lower() in check.lower() for k in EXCLURE):
        return out
    m = RX_LISTS.search(d)
    if m:
        a = [float(x) for x in m.group(1).split()]
        b = [float(x) for x in m.group(2).split()]
        if "ensemble" in check:              # le controle compare des ensembles tries
            a, b = sorted(a), sorted(b)
        if len(a) == len(b):
            return list(zip(a, b))
    m = RX_BAR.search(d)
    if m:
        return [(float(m.group(1)), float(m.group(2)))]
    for m in RX_VS.finditer(d):
        out.append((float(m.group(1)), float(m.group(2))))
    if out:
        return out
    for m in RX_FRAC.finditer(d):
        out.append((float(m.group(1)), float(m.group(2))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-out", default=None)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.csv, newline="")))

    pts, gaps = [], []
    for r in rows:
        if r["verdict"] != "pass" and r["verdict"] != "FAIL":
            continue
        if r["suite"] in ("stress", "pytest", "cpu_gpu_parity"):
            continue                         # compteurs, pas des paires remlax / reference
        sw = logiciel_de(r["suite"], r["check"])
        est_ll = "logl" in r["check"].lower() or "loglik" in r["detail"].lower()
        for x, y in paires(r["detail"], r["check"]):
            rec = dict(suite=r["suite"], check=r["check"], software=sw, remlax=x, reference=y,
                       kind="logLik" if est_ll else "estimate")
            if est_ll:
                gaps.append(rec)
            else:
                pts.append(rec)
    if a.pairs_out:
        with open(a.pairs_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "software", "suite", "check", "remlax", "reference"])
            w.writeheader()
            for rec in pts + gaps:
                w.writerow({k: rec[k] for k in w.fieldnames})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    for sw in list(LOGICIEL) + ["other"]:
        P = [p for p in pts if p["software"] == sw]
        if not P:
            continue
        x = np.array([p["reference"] for p in P]); y = np.array([p["remlax"] for p in P])
        ax1.scatter(x, y, s=18, alpha=0.75, color=COULEUR.get(sw, "k"), label="%s (%d)" % (sw, len(P)), zorder=3)
    allv = np.array([p["reference"] for p in pts] + [p["remlax"] for p in pts])
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    ax1.plot([lo, hi], [lo, hi], color="k", lw=0.8, zorder=1)
    ax1.set_xscale("symlog", linthresh=1e-2); ax1.set_yscale("symlog", linthresh=1e-2)
    ax1.set_xlabel("reference estimate"); ax1.set_ylabel("remlax estimate")
    ax1.set_title("Variance components, correlations, fixed effects, SE, BLUP\n(%d pairs read from the check log)" % len(pts))
    ax1.legend(frameon=False, fontsize=8)

    decal = 0
    for sw in list(LOGICIEL) + ["other"]:
        G = [g for g in gaps if g["software"] == sw]
        if not G:
            continue
        gap = np.abs(np.array([g["remlax"] - g["reference"] for g in G]))
        gap = np.where(gap == 0, 1e-13, gap)
        ax2.scatter(np.arange(len(G)) + decal, gap, s=18, color=COULEUR.get(sw, "k"),
                    label="%s (%d)" % (sw, len(G)), zorder=3)
        decal += len(G)
    ax2.axhline(1e-6, color="k", lw=0.8, ls="--"); ax2.text(0.5, 1.3e-6, "tolerance 1e-6 (lme4, sommer, nlme, closed form)", fontsize=7)
    ax2.axhline(1e-5, color="k", lw=0.8, ls=":"); ax2.text(0.5, 1.3e-5, "tolerance 1e-5 (asreml, own stopping rule)", fontsize=7)
    ax2.set_yscale("log"); ax2.set_ylim(5e-14, 10.0)
    ax2.set_xlabel("check (in table order)"); ax2.set_ylabel("|logLik gap| (remlax vs reference)")
    ax2.set_title("Log-likelihood at the optimum\n(%d checks; 0 drawn at 1e-13)" % len(gaps))
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("figure :", a.out, "| paires estimations", len(pts), "| ecarts logLik", len(gaps))
    if gaps:
        gg = np.abs(np.array([g["remlax"] - g["reference"] for g in gaps]))
        print("ecart logLik max :", gg.max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
