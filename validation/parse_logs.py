#!/usr/bin/env python3
"""Extrait le tableau de verifications des journaux des suites de validation.

    python3 validation/parse_logs.py <journaux>[=<provenance>] [<journaux2>[=<provenance2>] ...] \
            --out validation/results/checks_<date>.csv

POURQUOI UN PARSEUR PLUTOT QU'UNE SAISIE. Les suites de tests impriment une
ligne par verification, avec un verdict. Recopier ces lignes a la main dans un
tableau, c'est se donner l'occasion de faire une faute de frappe sur un chiffre
qui sera ensuite cite dans un article. Le parseur ne peut pas se tromper de
chiffre : il prend la ligne telle quelle.

CE QU'IL LIT. Les formats que les suites produisent :
  - une ligne de verdict :  deux espaces, un libelle, deux espaces au moins,
    puis OK ou ECHEC, puis un detail chiffre (suites R principales) ;
  - la variante `  [OK] libelle   detail` des suites du moteur creux ;
  - la variante `  OK    libelle  detail` de test_remlax_asreml_blup.R ;
  - un titre de section :   === 3. spline 2D ===
  - pour le balayage aleatoire, les compteurs de son resume ;
  - pour pytest, la ligne finale `N passed, M skipped` ;
  - pour la parite CPU/GPU, le tableau de parite_gpu.py.

PLUSIEURS REPERTOIRES. Le meme journal peut exister en plusieurs exemplaires
(cluster et local). Les repertoires sont lus dans l'ordre donne et le PREMIER
exemplaire trouve d'un journal fait foi ; donner le cluster d'abord. Chaque
repertoire porte sa provenance apres `=`.

Les libelles restent en francais : ce sont les chaines imprimees par les suites,
citees verbatim. Les traduire donnerait un tableau plus lisible et une trace
fausse.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, OrderedDict

# (fichier, suite, reference) — l'ordre fixe celui du tableau final.
SOURCES = [
    ("01_core.log", "algebra", "internal algebra"),
    ("00_pytest.log", "pytest", "internal algebra and closed forms (pytest)"),
    ("02_stress_s0.log", "stress", "properties (no reference implementation)"),
    ("02_stress_s50.log", "stress", "properties (no reference implementation)"),
    ("02_stress_s100.log", "stress", "properties (no reference implementation)"),
    ("03_lme4_sommer.log", "lme4_sommer", "lme4 / sommer"),
    ("07_nlme.log", "nlme", "nlme"),
    ("08_lme4_extra.log", "lme4_extra", "lme4"),
    ("09_sommer_extra.log", "sommer_extra", "sommer"),
    ("04_asreml.log", "asreml_structures", "asreml"),
    ("05_asreml2.log", "asreml_catalogue", "asreml / pbkrtest / R"),
    ("15_asreml3.log", "asreml_rest", "asreml"),
    ("06b_asreml_blup.log", "asreml_blup", "asreml"),
    ("06_ige.log", "ige_model", "asreml + known truth"),
    ("10_tmb_parity.log", "sparse_parity", "dense engine (same theta)"),
    ("11_tmb_vs_dense.log", "sparse_vs_dense", "dense engine (warm start both ways)"),
    ("12_tmb_prec.log", "sparse_prec", "independent dense-algebra REML"),
    ("21_parite.log", "cpu_gpu_parity", "the other backend (same input bundle)"),
]

VERDICT = re.compile(r"^  (?P<nom>.{2,70}?)\s{2,}(?P<verdict>OK|ECHEC)\s*(?P<detail>.*)$")
VERDICT_CROCHETS = re.compile(r"^\s*\[(?P<verdict>OK|ECHEC)\]\s+(?P<nom>.+?)(?:\s{2,}(?P<detail>.*))?$")
VERDICT_TETE = re.compile(r"^  (?P<verdict>OK|ECHEC)\s+(?P<nom>.+?)(?:\s{2,}(?P<detail>.*))?$")
SECTION = re.compile(r"^=+\s*(?P<t>[A-Z]?\d*[a-z]?\..*?)\s*=+\s*$")
PARITE = re.compile(r"^(?P<m>[a-z0-9_.]+)\s+(?P<c>-?\d+\.\d+)\s+(?P<g>-?\d+\.\d+)\s+"
                    r"(?P<rel>[\d.]+e[+-]\d+)\s+(?P<dth>[\d.]+e[+-]\d+)\s+"
                    r"(?P<sc>[\d.]+)\s+(?P<sg>[\d.]+)", re.M)
PYTEST = re.compile(r"^(?:=+ )?(?P<p>\d+) passed(?:, (?P<s>\d+) skipped)?(?:, (?P<f>\d+) failed)?"
                    r"(?:, (?P<e>\d+) errors?)? in (?P<t>[\d.]+)s", re.M)
COMPTEURS = ["exception", "logLik non finie", "non convergé", "Sigma non PSD",
             "optimum instable"]
# Versions des logiciels, lues dans les en-tetes que les suites impriment.
# Lues dans les HUIT premieres lignes de chaque journal (les suites impriment
# leurs versions en tete) et dans 00_versions.log entier, un fichier que le
# job ecrit avec la ligne [R] ... | asreml ... et la ligne [jax] ... .
VERSIONS = [
    ("R", re.compile(r"R version (\d+\.\d+\.\d+)")),
    ("R", re.compile(r"\| R (\d+) (\d+\.\d+)")),
    ("asreml", re.compile(r"asreml (\d+\.\d+\.\d+[\d.]*)")),
    ("lme4", re.compile(r"lme4 (\d+\.\d+[\d.-]*)")),
    ("sommer", re.compile(r"sommer (\d+\.\d+\.\d+)")),
    ("nlme", re.compile(r"nlme (\d+\.\d+[-.]\d+)")),
    ("pbkrtest", re.compile(r"pbkrtest (\d+\.\d+\.\d+)")),
    ("jax", re.compile(r"\[jax\] (\d+\.\d+\.\d+)")),
    ("RTMB", re.compile(r"RTMB (\d+\.\d+[\d.]*)")),
]
N_LIGNES_VERSIONS = 8


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
        m = VERDICT.match(line.rstrip()) or VERDICT_CROCHETS.match(line.rstrip()) \
            or VERDICT_TETE.match(line.rstrip())
        if m:
            out.append(dict(suite=suite, section=section, check=m.group("nom").strip(),
                            reference=ref,
                            verdict="pass" if m.group("verdict") == "OK" else "FAIL",
                            detail=(m.group("detail") or "").strip(),
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


def lignes_pytest(txt, suite, ref, prov, src):
    m = None
    for m in PYTEST.finditer(txt):
        pass
    if m is None:
        return []
    p, s, f, e = (int(m.group(k) or 0) for k in ("p", "s", "f", "e"))
    return [dict(suite=suite, section="pytest, tests/python",
                 check="%d tests collected by pytest" % (p + s + f + e), reference=ref,
                 verdict="pass" if f == 0 and e == 0 else "FAIL",
                 detail="%d passed, %d skipped, %d failed, %d errors in %s s" % (p, s, f, e, m.group("t")),
                 provenance=prov, source_log=src)]


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


def versions_de(txt):
    v = {}
    for nom, rx in VERSIONS:
        m = rx.search(txt)
        if m and nom not in v:
            v[nom] = ".".join(g for g in m.groups() if g)
    return v


COLS = ["suite", "section", "check", "reference", "verdict", "detail",
        "provenance", "source_log"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+",
                    help="repertoire(s) de journaux, chacun sous la forme chemin[=provenance]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="re-measured, provenance not given")
    ap.add_argument("--versions-out", default=None,
                    help="JSON des versions de logiciels lues dans les journaux (defaut : a cote de --out)")
    a = ap.parse_args(argv)

    racines = []
    for spec in a.logs:
        chemin, _, prov = spec.partition("=")
        racines.append((chemin, prov or a.provenance))

    rows, manquants, versions, origine = [], [], OrderedDict(), {}
    for nom, suite, ref in SOURCES:
        p, prov = None, None
        for racine, prov_r in racines:
            p = trouver(racine, nom)
            if p is not None:
                prov = prov_r
                break
        if p is None:
            manquants.append(nom)
            continue
        origine[nom] = p
        txt = open(p, errors="replace").read()
        for k, v in versions_de("\n".join(txt.splitlines()[:N_LIGNES_VERSIONS])).items():
            versions.setdefault(k, OrderedDict()).setdefault(prov, v)
        if suite == "stress":
            seed = int(re.search(r"_s(\d+)\.log$", nom).group(1))
            rows += lignes_stress(txt, suite, ref, prov, nom, seed)
        elif suite == "cpu_gpu_parity":
            rows += lignes_parite(txt, suite, ref, prov, nom)
        elif suite == "pytest":
            rows += lignes_pytest(txt, suite, ref, prov, nom)
        else:
            rows += lignes_verdict(txt, suite, ref, prov, nom)

    for racine, prov_r in racines:
        pv = trouver(racine, "00_versions.log")
        if pv:
            for k, v in versions_de(open(pv, errors="replace").read()).items():
                versions.setdefault(k, OrderedDict())[prov_r] = v
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    vout = a.versions_out or re.sub(r"checks_", "versions_", a.out).replace(".csv", ".json")
    with open(vout, "w") as f:
        json.dump({"versions": versions, "logs": origine, "missing": manquants}, f, indent=1)

    print("%d verifications ecrites dans %s" % (len(rows), a.out))
    print("par suite :", dict(Counter(r["suite"] for r in rows)))
    print("verdicts  :", dict(Counter(r["verdict"] for r in rows)))
    print("versions  :", json.dumps(versions))
    if manquants:
        print("journaux ABSENTS (suites non couvertes) :", ", ".join(manquants))
    ech = [r["check"] for r in rows if r["verdict"] == "FAIL"]
    if ech:
        print("ECHECS :", ech)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
