#!/usr/bin/env python3
"""Extrait le tableau de verifications des journaux des jobs du cluster.

    python3 validation/parse_logs.py <repertoire_des_journaux> \
            --out validation/results/checks_<date>.csv

POURQUOI UN PARSEUR PLUTOT QU'UNE SAISIE. Les suites de tests impriment une
ligne par verification, avec un verdict. Recopier ces lignes a la main dans un
tableau, c'est se donner l'occasion de faire une faute de frappe sur un chiffre
qui sera ensuite cite dans un article. Le parseur ne peut pas se tromper de
chiffre : il prend la ligne telle quelle.

CE QU'IL LIT. Deux formats seulement, ceux que les suites produisent :
  - une ligne de verdict :  deux espaces, un libelle, deux espaces au moins,
    puis OK ou ECHEC, puis un detail chiffre ;
  - un titre de section :   === 3. spline 2D ===
et pour le balayage aleatoire, qui n'imprime pas de verdicts nommes, les
compteurs de son resume.

Les libelles restent en francais : ce sont les chaines imprimees par les suites,
citees verbatim. Les traduire donnerait un tableau plus lisible et une trace
fausse.
"""
import argparse
import csv
import os
import re
from collections import Counter

# (fichier, suite, reference) — l'ordre fixe celui du tableau final.
SOURCES = [
    ("01_core.log", "algebra", "internal algebra"),
    ("02_stress_s0.log", "stress", "properties (no reference implementation)"),
    ("02_stress_s50.log", "stress", "properties (no reference implementation)"),
    ("02_stress_s100.log", "stress", "properties (no reference implementation)"),
    ("03_lme4_sommer.log", "lme4_sommer", "lme4 / sommer"),
    ("04_asreml.log", "asreml_structures", "asreml"),
    ("05_asreml2.log", "asreml_catalogue", "asreml / pbkrtest / R"),
    ("06_ige.log", "ige_model", "asreml + known truth"),
    ("21_parite.log", "cpu_gpu_parity", "the other backend (same input bundle)"),
]

VERDICT = re.compile(r"^  (?P<nom>.{2,60}?)\s{2,}(?P<verdict>OK|ECHEC)\s*(?P<detail>.*)$")
SECTION = re.compile(r"^=+\s*(?P<t>\d+[a-z]?\..*?)\s*=+\s*$")
PARITE = re.compile(r"^(?P<m>[a-z0-9_.]+)\s+(?P<c>-?\d+\.\d+)\s+(?P<g>-?\d+\.\d+)\s+"
                    r"(?P<rel>[\d.]+e[+-]\d+)\s+(?P<dth>[\d.]+e[+-]\d+)\s+"
                    r"(?P<sc>[\d.]+)\s+(?P<sg>[\d.]+)", re.M)
COMPTEURS = ["exception", "logLik non finie", "non convergé", "Sigma non PSD",
             "optimum instable"]


def trouver(racine, nom):
    """Le meme journal peut apparaitre a plusieurs profondeurs selon le job."""
    for d, _, fs in os.walk(racine):
        if nom in fs:
            return os.path.join(d, nom)
    return None


def lignes_verdict(txt, suite, ref, prov, src):
    section, out = "", []
    for line in txt.splitlines():
        m = SECTION.match(line.strip())
        if m:
            section = m.group("t").strip()
            continue
        m = VERDICT.match(line.rstrip())
        if m:
            out.append(dict(suite=suite, section=section, check=m.group("nom").strip(),
                            reference=ref,
                            verdict="pass" if m.group("verdict") == "OK" else "FAIL",
                            detail=m.group("detail").strip(),
                            provenance=prov, source_log=src))
    return out


def lignes_stress(txt, suite, ref, prov, src, seed):
    cnt = {}
    for k in COMPTEURS:
        m = re.search(r"^  %s\s+(\d+)\s*$" % re.escape(k), txt, re.M)
        cnt[k] = int(m.group(1)) if m else None
    m = re.search(r"^(\d+)/(\d+) sans aucun probleme", txt, re.M)
    ok, tot = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    trip = re.findall(r"^\s+\((\d+), ([\d.]+), ([\d.]+)\)\s*$", txt, re.M)
    impossible = ["exception", "logLik non finie", "non convergé", "Sigma non PSD"]
    detail = "clean %d/%d; unstable optima %d" % (ok, tot, cnt.get("optimum instable") or 0)
    if trip:
        detail += " (best-of-restart gains %s in -2logL)" % ", ".join(
            "%.4f" % (float(a) - float(b)) for _, a, b in trip)
    return [dict(suite=suite, section="random configuration sweep, seed %d" % seed,
                 check="%d configurations: no exception, finite logLik, PSD Sigma, converged" % tot,
                 reference=ref,
                 verdict="pass" if all((cnt.get(k) or 0) == 0 for k in impossible) else "FAIL",
                 detail=detail, provenance=prov, source_log=src)]


def lignes_parite(txt, suite, ref, prov, src, tol=1e-8):
    out = []
    for d in (m.groupdict() for m in PARITE.finditer(txt)):
        out.append(dict(
            suite=suite, section="structure catalogue, 34 serialised designs",
            check="design '%s': same -2logL on CPU and on an A100 MIG slice" % d["m"],
            reference=ref, verdict="pass" if float(d["rel"]) <= tol else "FAIL",
            detail="-2logL %s vs %s | rel gap %s | max|dtheta| %s | %s s CPU vs %s s MIG"
                   % (d["c"], d["g"], d["rel"], d["dth"], d["sc"], d["sg"]),
            provenance=prov, source_log=src))
    return out


COLS = ["suite", "section", "check", "reference", "verdict", "detail",
        "provenance", "source_log"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", help="repertoire contenant les journaux des jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="re-measured 2026-09-01, ssh:cluster, ige_reml.sif")
    a = ap.parse_args(argv)

    rows, manquants = [], []
    for nom, suite, ref in SOURCES:
        p = trouver(a.logs, nom)
        if p is None:
            manquants.append(nom)
            continue
        txt = open(p, errors="replace").read()
        if suite == "stress":
            seed = int(re.search(r"_s(\d+)\.log$", nom).group(1))
            rows += lignes_stress(txt, suite, ref, a.provenance, nom, seed)
        elif suite == "cpu_gpu_parity":
            rows += lignes_parite(txt, suite, ref, a.provenance, nom)
        else:
            rows += lignes_verdict(txt, suite, ref, a.provenance, nom)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    print("%d verifications ecrites dans %s" % (len(rows), a.out))
    print("par suite :", dict(Counter(r["suite"] for r in rows)))
    print("verdicts  :", dict(Counter(r["verdict"] for r in rows)))
    if manquants:
        print("journaux ABSENTS (suites non couvertes) :", ", ".join(manquants))
    ech = [r["check"] for r in rows if r["verdict"] == "FAIL"]
    if ech:
        print("ECHECS :", ech)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
