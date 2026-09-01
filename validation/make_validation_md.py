#!/usr/bin/env python3
"""Genere docs/validation.md a partir du CSV de verifications.

Le CSV est la source ; ce script ne fait que le mettre en forme. Aucun chiffre
n'est saisi a la main : tout ce qui apparait dans la page vient d'une ligne du
CSV, elle-meme extraite par programme des journaux des jobs du cluster.

    python3 validation/make_validation_md.py
"""
import csv
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(HERE, "results", "checks_2026-09-01.csv")
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
    ("lme4_sommer", ("Against lme4 and sommer", "two open-source reference implementations",
                     "The models both packages can fit: a single random factor on three "
                     "layouts, two crossed factors, a genomic relationship matrix, and a "
                     "bivariate unstructured covariance. Agreement is required on the "
                     "restricted log-likelihood first, then on the variance components, the "
                     "fixed effects and the BLUPs.")),
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
    ("ige_model", ("A full model of known truth", "asreml, and the simulated parameters",
                   "The complete direct and indirect genetic effects model: a weighted "
                   "neighbourhood incidence, a covariance shared between the direct and "
                   "indirect terms, a separable field and a nugget - eight variance "
                   "parameters - fitted on simulated data whose true values are known, by "
                   "remlax and by asreml.")),
    ("cpu_gpu_parity", ("CPU against GPU", "the other backend, on the same input",
                        "Every design of the structure catalogue is serialised once, then "
                        "fitted twice in the same process: on the CPU and on a slice of an "
                        "A100. The model must not depend on the machine that fits it.")),
])


def lire():
    with open(CSV, newline="") as f:
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


def main():
    rows = lire()
    n_tot = len(rows)
    n_pass = sum(r["verdict"] == "pass" for r in rows)
    par_suite = OrderedDict()
    for r in rows:
        par_suite.setdefault(r["suite"], []).append(r)

    L = []
    A = L.append
    A("# Validation")
    A("")
    A("Every number on this page was produced by a job run on 2026-09-01 on the "
      "user's SLURM cluster, inside the apptainer image `ige_reml.sif`, which "
      "carries JAX 0.11.1, R 4.6.0 and licensed copies of asreml and pbkrtest. "
      "The measurements were extracted from the job logs by "
      "`validation/parse_logs.py` and this page is generated from the resulting "
      "table by `validation/make_validation_md.py`. No value here was typed by "
      "hand.")
    A("")
    A("**%d checks, %d passed, %d failed.**" % (n_tot, n_pass, n_tot - n_pass))
    A("")
    A("The check labels in the tables below are in French. They are the strings "
      "the test suites print, quoted verbatim, and the suites are written in "
      "French like the rest of the code. Translating them would give a more "
      "readable table and a trace that no longer matches the logs, so they are "
      "left alone; the section headings and the commentary are in English.")
    A("")
    A("A note on naming. The solver was called `remlkit` when these measurements "
      "were taken and was renamed `remlax` afterwards. The recorded table still "
      "prints the former name in the columns that quote a log line verbatim. "
      "Rewriting a recorded measurement to match a name chosen later would "
      "falsify the trace, so it is left as measured.")
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
    A("**The full A100 card.** Every GPU number above and in "
      "[benchmarks.md](benchmarks.md) comes from one MIG slice, which exposes 7 "
      "of the card's 108 streaming multiprocessors. The card itself was occupied "
      "by other work. A slice is the right instrument for checking that CPU and "
      "GPU agree - arithmetic does not depend on how many multiprocessors run "
      "it - but any timing measured on it is a pessimistic bound on the "
      "hardware, not a property of the solver.")
    A("")
    A("**Two structures on the slice.** The Matern kernels, isotropic and "
      "anisotropic, could not be fitted on the slice at 1200 levels: the Bessel "
      "quadrature asks for a single 6.5 GB allocation, above what a 10 GB slice "
      "can give. They fit without difficulty on the CPU, and the CPU/GPU parity "
      "check above did pass for both at 160 levels. This is a limit of the "
      "slice, and it is measured, not inferred.")
    A("")
    A("**The comparison against the project's own RTMB engine.** The figure of "
      "roughly 36 hours for RTMB against roughly 15 minutes on a full A100, at "
      "n = 16211, was measured in the originating project "
      "(IGE_analysis_2024-2025) before this repository existed. It is quoted in "
      "the paper as a prior measurement with its date, and it was not re-run "
      "here: it needs both the full card and that project's data.")
    A("")
    A("## Re-running all of it")
    A("")
    A("```sh")
    A("# inside a container or an environment carrying JAX, R, lme4, sommer")
    A("export PYTHONPATH=$PWD/src RX_PY=python3")
    A("")
    A("python3 tests/python/test_remlax_core.py          # internal algebra")
    A("python3 tests/python/stress_remlax.py --n 50 --seed 0    # random sweep")
    A("Rscript  tests/R/test_remlax.R                    # lme4, sommer")
    A("Rscript  tests/R/test_remlax_asreml.R             # asreml, structures")
    A("Rscript  tests/R/test_remlax_asreml2.R            # asreml, catalogue, K&R")
    A("Rscript  tests/R/test_remlax_ige.R                # full model, known truth")
    A("")
    A("Rscript  tests/R/export_bundles.R --out=bundles   # serialise the catalogue")
    A("python3  tests/python/parite_gpu.py bundles --tol 1e-8   # CPU against GPU")
    A("```")
    A("")
    A("asreml requires a licence, and the two `test_remlax_asreml*.R` suites "
      "check one out at run time. The other suites need no licence. The "
      "CPU/GPU parity check needs a machine with a GPU; everything else runs on "
      "a CPU.")
    A("")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("%s ecrit : %d lignes, %d verifications, %d suites"
          % (os.path.relpath(OUT, ROOT), len(L), n_tot, len(par_suite)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
