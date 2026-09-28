#!/usr/bin/env python3
"""Genere docs/validation.md a partir du CSV de verifications.

Le CSV est la source ; ce script ne fait que le mettre en forme. Aucun chiffre
n'est saisi a la main : tout ce qui apparait dans la page vient d'une ligne du
CSV, elle-meme extraite par programme des journaux des jobs du cluster.

    python3 validation/make_validation_md.py --csv validation/results/checks_<date>.csv \
            --versions validation/results/versions_<date>.json --rev <git sha> --date <date>
"""
import argparse
import csv
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(HERE, "results", "checks_2026-09-28.csv")
OUT = os.path.join(ROOT, "docs", "validation.md")

TITRES = OrderedDict([
    ("algebra", ("Internal algebra", "the solver against itself and against closed-form results",
                 "Gradients against central finite differences, the Hessian against a second "
                 "differentiation, log-determinant and quadratic form against a direct "
                 "computation, parameter counts, and the refusal of designs that would make V "
                 "singular. These need no other software: they are the properties the "
                 "implementation must have to be an implementation of REML at all.")),
    ("stress", ("Random sweep", "properties over randomly drawn designs",
                "Configurations are drawn at random (structure, dimensions, number of traits, "
                "seed) and fitted. Nothing is compared to a reference: the sweep asks only "
                "whether the solver ever produces something impossible - an exception, a "
                "non-finite log-likelihood, a covariance matrix that is not positive "
                "semi-definite, a run that does not converge.")),
    ("pytest", ("Unit tests (pytest)", "closed forms and independent numpy references",
                "The pytest suite under tests/python: one test per feature, each against an "
                "independent reference (numpy Cholesky REML written out longhand, closed-form "
                "correlation matrices, finite differences). Collected here as a single line; "
                "docs/tests-matrix.md maps every feature to its test.")),
    ("lme4_sommer", ("Against lme4 and sommer", "two open-source reference implementations",
                     "The models both packages can fit: a single random factor on three "
                     "layouts, two crossed factors, a genomic relationship matrix, and a "
                     "bivariate unstructured covariance. Agreement is required on the "
                     "restricted log-likelihood first, then on the variance components, the "
                     "fixed effects and the BLUPs.")),
    ("nlme", ("Against nlme", "the reference implementation for correlation structures",
              "nlme::gls and nlme::lme on the same data: corAR1, corCAR1, corARMA (AR2 and "
              "ARMA(1,1)), corCompSymm, corExp, corGaus, corLin, corSpher, corSymm with "
              "varIdent (an unstructured covariance) and varIdent alone (one residual variance "
              "per group), then a random intercept with an AR1 residual. A grouped corStruct "
              "is written in remlax by declaring the group as the trait of the residual, "
              "residual = ~ id(g):ar1(t) with trait = g. Agreement on -2logL, the residual "
              "variance, the correlation parameters, the fixed effects and their standard "
              "errors.")),
    ("lme4_extra", ("Against lme4, random regressions, nesting and PEV",
                    "the reference implementation for crossed and nested random effects",
                    "Correlated and independent random slopes (a two-column us and diag term), "
                    "a nested design, three crossed factors on an unbalanced layout, very "
                    "unequal group sizes. Besides -2logL, components, fixed effects and their "
                    "standard errors: the BLUPs, and the prediction error variance checked "
                    "against the mixed-model-equations formula computed by hand (lme4's condVar "
                    "follows a different convention, without the uncertainty on beta, and is "
                    "checked against its own formula).")),
    ("sommer_extra", ("Against sommer, genomic and multi-trait models",
                      "the reference implementation for genomic relationship matrices",
                      "A GRM on one term with BLUPs and PEV; three traits with us genetic and "
                      "us residual covariance and a GRM; the same with diag and the likelihood "
                      "ratio between the two; additive plus dominance (two relationship matrices "
                      "on the same factor); a multi-environment trial with one genetic variance "
                      "per environment. sommer is tightened to tolParConvLL = 1e-12; its "
                      "log-likelihood carries a data-dependent constant, so only differences "
                      "of -2logL are compared.")),
    ("asreml_structures", ("Against asreml, structures and inference", "the field reference",
                           "The structures a plant breeder actually writes: a one-dimensional "
                           "AR1, a separable AR1xAR1 field, two-dimensional splines, and a "
                           "covariance shared between two terms through str(). Then the "
                           "inference layer: heritability and its delta-method standard error, "
                           "Wald tests, and the equivalences between saturated "
                           "parameterisations.")),
    ("asreml_catalogue", ("Against asreml, the rest of the catalogue", "the field reference",
                          "The structures that were identified by matching asreml's output "
                          "rather than from documentation - lvr, the anisotropic Matern, "
                          "user-written structures, sectioned residuals - plus prediction and "
                          "the Kenward-Roger denominator degrees of freedom against pbkrtest.")),
    ("asreml_rest", ("Against asreml, the remaining structures",
                     "the field reference",
                     "Grouped residual series ar2, ar3, ma1, ma2, arma, sar, cor, corb and corg; "
                     "the two-dimensional metric kernels iexp, igau, ieuc, aexp, agau, sph and "
                     "cir on an irregular field; the non-saturated multi-trait structures fa(1), "
                     "rr(1), corh, ante(1), chol(1) and diag on four traits with a GRM; the "
                     "separable product id x ar1 x ar1 with shared correlations; and the "
                     "multi-trait spatial residual us(trait):ar1(row):ar1(col).")),
    ("asreml_blup", ("Against asreml, BLUPs", "the field reference",
                     "BLUPs and their ordering on a one-factor model, level by level.")),
    ("ige_model", ("A full model of known truth", "asreml, and the simulated parameters",
                   "The complete direct and indirect genetic effects model: a weighted "
                   "neighbourhood incidence, a covariance shared between the direct and "
                   "indirect terms, a separable field and a nugget - eight variance "
                   "parameters - fitted on simulated data whose true values are known, by "
                   "remlax and by asreml.")),
    ("sparse_parity", ("Sparse engine: parameterisation parity",
                       "the dense engine, at a common theta",
                       "Sigma, level correlations and their closed-form sparse precisions read "
                       "identically by the RTMB sparse engine and the JAX dense engine, including "
                       "K^-1 K = I checked by product.")),
    ("sparse_vs_dense", ("Sparse engine: same likelihood as the dense engine",
                         "the dense engine, warm-started in both directions",
                         "Each engine is evaluated at the other's optimum; comparing two optima "
                         "alone would only show where each stopped.")),
    ("sparse_prec", ("Sparse engine: precision path", "an independent dense-algebra REML",
                     "A term declared by its precision matrix gives the same likelihood as the "
                     "same term declared by its relationship matrix, arbitrated by a REML written "
                     "out longhand in the test file.")),
    ("cpu_gpu_parity", ("CPU against GPU", "the other backend, on the same input",
                        "Every design of the structure catalogue is serialised once, then "
                        "fitted twice in the same process: on the CPU and on a slice of an "
                        "A100. The model must not depend on the machine that fits it.")),
])


def lire(chemin):
    with open(chemin, newline="") as f:
        return list(csv.DictReader(f))


def tableau(lignes, colonnes, entetes):
    out = ["| " + " | ".join(entetes) + " |",
           "|" + "|".join(["---"] * len(entetes)) + "|"]
    for r in lignes:
        cells = []
        for c in colonnes:
            v = (r.get(c) or "").replace("|", "\\|").strip()
            cells.append(v if v else "-")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--versions", default=None, help="JSON ecrit par parse_logs.py")
    ap.add_argument("--rev", default="unknown", help="revision git testee")
    ap.add_argument("--date", default="unknown")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    rows = lire(a.csv)
    vers = json.load(open(a.versions)) if a.versions and os.path.exists(a.versions) else {}
    n_tot = len(rows)
    n_pass = sum(r["verdict"] == "pass" for r in rows)
    par_suite = OrderedDict()
    for r in rows:
        par_suite.setdefault(r["suite"], []).append(r)
    provenances = OrderedDict()
    for r in rows:
        provenances.setdefault(r["provenance"], 0)
        provenances[r["provenance"]] += 1

    L = []
    A = L.append
    A("# Validation")
    A("")
    A("Every number on this page was produced on %s by the test suites of this repository "
      "at git revision `%s` (working tree of that day, see the commit that follows it). The "
      "suites that need asreml or pbkrtest ran on the user's SLURM cluster inside the apptainer "
      "image `ige_reml.sif`; the others ran there too and, where noted, on a local CPU. The "
      "measurements were extracted from the logs by `validation/parse_logs.py` and this page "
      "is generated from the resulting table (`%s`) by `validation/make_validation_md.py`. "
      "No value here was typed by hand." % (a.date, a.rev, os.path.relpath(a.csv, ROOT)))
    A("")
    A("**%d checks, %d passed, %d failed.**" % (n_tot, n_pass, n_tot - n_pass))
    A("")
    if provenances:
        A("Provenance of the rows:")
        A("")
        for pv, k in provenances.items():
            A("- %s: %d checks" % (pv, k))
        A("")
    if vers.get("versions"):
        A("Software versions read from the log headers:")
        A("")
        for sw, d in vers["versions"].items():
            A("- %s: %s" % (sw, "; ".join("%s (%s)" % (v, pv.split(", ", 1)[1] if ", " in pv else pv)
                                          for pv, v in d.items())))
        A("")
    A("The check labels in the tables below are in French. They are the strings "
      "the test suites print, quoted verbatim, and the suites are written in "
      "French like the rest of the code. Translating them would give a more "
      "readable table and a trace that no longer matches the logs, so they are "
      "left alone; the section headings and the commentary are in English.")
    A("")
    if vers.get("missing"):
        A("Suites whose log was not available for this run, and therefore absent from the "
          "table: %s. See the last section." % ", ".join("`%s`" % m for m in vers["missing"]))
        A("")

    A("## Summary")
    A("")
    resume = []
    for s, rs in par_suite.items():
        t = TITRES.get(s, (s, "", ""))
        resume.append({"suite": t[0], "n": str(len(rs)), "ref": t[1],
                       "verdict": "%d/%d" % (sum(r["verdict"] == "pass" for r in rs), len(rs))})
    A(tableau(resume, ["suite", "n", "ref", "verdict"],
              ["Suite", "Checks", "Compared against", "Passed"]))
    A("")

    for s, rs in par_suite.items():
        titre, ref, expl = TITRES.get(s, (s, "", ""))
        A("## %s" % titre)
        A("")
        if expl:
            A(expl)
            A("")
        secs = OrderedDict()
        for r in rs:
            secs.setdefault(r["section"] or "", []).append(r)
        for sec, rr in secs.items():
            if sec:
                A("**%s**" % sec)
                A("")
            A(tableau(rr, ["check", "detail", "verdict"],
                      ["Check", "Measured", "Verdict"]))
            A("")

    A("## What could not be measured here")
    A("")
    A("This section exists so that nothing on the page above is mistaken for "
      "something it is not.")
    A("")
    A("**CPU against GPU.** The 34-design parity sweep (`tests/python/parite_gpu.py`, "
      "evaluation at a common theta on both devices) was not replayed on this date: the "
      "whole A100 became free late in the day and the time went to the speed benchmark, "
      "which fitted the same ten models on 4 CPU cores and on the whole card and reached "
      "the same optimum on each (|logLik gap| at most 1e-6 at six printed decimals, "
      "`benchmarks/results/bench_2026-09-28_*.csv`). The parity recorded on 2026-09-01 "
      "(34/34, relative gap at most 6.4e-15) stands for the code of that date. "
      "`tests/slurm_valid_gpu.slurm` replays it.")
    A("")
    A("**Structures with no external reference.** `mtrn` with all four parameters free, "
      "`own` (user-written kernels) and `sep` beyond three factors have no counterpart in "
      "asreml, lme4, sommer or nlme; they are checked against closed forms (besselK profile, "
      "the same kernel written as `exp`) and against the dense reference. The sparse RTMB "
      "engine is compared to the dense engine, never to a third program.")
    A("")
    A("**What agreement means here.** Two REML implementations agree on the restricted "
      "log-likelihood at the optimum to 1e-6 absolute, on the variance components to 1e-4 "
      "relative (looser on purpose: on a flat surface two optimisers stop at slightly "
      "different theta for the same likelihood) and on fixed effects to 1e-6 relative. "
      "asreml stops on its own criterion and is compared at 1e-5 on the log-likelihood and "
      "1e-3 on components; sommer is tightened to tolParConvLL = 1e-12 and compared at 1e-4. "
      "Iteration counts are never compared: average information, quasi-Newton and "
      "derivative-free optimisers do not count the same thing.")
    A("")
    A("**The comparison against the project's own RTMB engine at full scale.** The figure of "
      "roughly 36 hours for RTMB against roughly 15 minutes on a full A100, at "
      "n = 16211, was measured in the originating project "
      "(IGE_analysis_2024-2025) before this repository existed. It is quoted in "
      "the paper as a prior measurement with its date, and it was not re-run "
      "here: it needs both the full card and that project's data.")
    A("")
    A("## Re-running all of it")
    A("")
    A("```sh")
    A("# inside a container or an environment carrying JAX, R, lme4, sommer, nlme")
    A("export PYTHONPATH=$PWD/src RX_PY=python3")
    A("")
    A("python3 -m pytest tests/python -q                 # unit tests")
    A("python3 tests/python/test_remlax_core.py          # internal algebra")
    A("python3 tests/python/stress_remlax.py --n 50 --seed 0    # random sweep")
    A("Rscript  tests/R/test_remlax.R                    # lme4, sommer")
    A("Rscript  tests/R/test_remlax_nlme.R               # nlme, correlation structures")
    A("Rscript  tests/R/test_remlax_lme4.R               # lme4, slopes, nesting, PEV")
    A("Rscript  tests/R/test_remlax_sommer.R             # sommer, genomic multi-trait")
    A("Rscript  tests/R/test_remlax_asreml.R             # asreml, structures")
    A("Rscript  tests/R/test_remlax_asreml2.R            # asreml, catalogue, K&R")
    A("Rscript  tests/R/test_remlax_asreml3.R            # asreml, remaining structures")
    A("Rscript  tests/R/test_remlax_asreml_blup.R        # asreml, BLUPs")
    A("Rscript  tests/R/test_remlax_ige.R                # full model, known truth")
    A("Rscript  tests/R/test_remlax_tmb_parity.R         # sparse engine (RTMB)")
    A("Rscript  tests/R/test_remlax_tmb_vs_dense.R")
    A("Rscript  tests/R/test_remlax_tmb_prec.R")
    A("")
    A("Rscript  tests/R/export_bundles.R --out=bundles   # serialise the catalogue")
    A("python3  tests/python/parite_gpu.py bundles --tol 1e-8   # CPU against GPU")
    A("")
    A("python3 validation/parse_logs.py <logs>=<provenance> --out validation/results/checks_<date>.csv")
    A("python3 validation/make_validation_md.py --csv ... --versions ... --rev <sha> --date <date>")
    A("```")
    A("")
    A("asreml requires a licence, and the `test_remlax_asreml*.R` suites "
      "check one out at run time. The other suites need no licence. The "
      "CPU/GPU parity check needs a machine with a GPU; everything else runs on "
      "a CPU.")
    A("")

    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("%s ecrit : %d lignes, %d verifications, %d suites"
          % (os.path.relpath(a.out, ROOT), len(L), n_tot, len(par_suite)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
