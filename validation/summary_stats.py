#!/usr/bin/env python3
"""Ecarts maximaux par logiciel de reference, lus dans pairs_<date>.csv et checks_<date>.csv.

    python3 validation/summary_stats.py validation/results/checks_2026-09-28.csv validation/results/pairs_2026-09-28.csv
"""
import csv, sys
from collections import OrderedDict
checks = list(csv.DictReader(open(sys.argv[1]))); pairs = list(csv.DictReader(open(sys.argv[2])))
SW = OrderedDict([("asreml", ("asreml_structures", "asreml_catalogue", "asreml_rest", "asreml_blup", "ige_model")),
                  ("lme4", ("lme4_sommer", "lme4_extra")), ("sommer", ("lme4_sommer", "sommer_extra")), ("nlme", ("nlme",)),
                  ("closed form / dense engine", ("algebra", "pytest", "sparse_parity", "sparse_vs_dense", "sparse_prec")),
                  ("properties only", ("stress",))])
print("| reference | checks | passed | max |logLik gap| | max relative gap on estimates | pairs read |")
print("|---|---|---|---|---|---|")
for sw, suites in SW.items():
    cs = [c for c in checks if c["suite"] in suites and (sw not in ("lme4", "sommer") or c["suite"] != "lme4_sommer" or (sw == "sommer") == ("sommer" in c["check"].lower()))]
    ps = [p for p in pairs if p["software"] == sw]
    ll = [abs(float(p["remlax"]) - float(p["reference"])) for p in ps if p["kind"] == "logLik"]
    est = [abs(float(p["remlax"]) - float(p["reference"])) / max(abs(float(p["reference"])), 1e-8) for p in ps if p["kind"] == "estimate"]
    print("| %s | %d | %d | %s | %s | %d |" % (sw, len(cs), sum(c["verdict"] == "pass" for c in cs),
          ("%.1e" % max(ll)) if ll else "-", ("%.1e" % max(est)) if est else "-", len(ps)))
print()
print("total checks", len(checks), "passed", sum(c["verdict"] == "pass" for c in checks))
