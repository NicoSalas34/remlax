"""Command line of the co-GWAS (cogwas.py), called by rx_cogwas from R.

    python -m remlax.cogwas_cli <bundle> --tests "pair|dir+ind" [--top 1000]
        [--backend auto|gpu|cpu] [--block-a 256] [--block-b 8192] [--quiet]

The bundle is rx_export of the null model, with in_theta.bin (evaluated,
not refitted) and the arrays of cogwas_manifest.json: the incidences
scan_dir_* and scan_ind_* (0-based COO), the focal doses cog_Ma (q_dir x
p_a) and the neighbour doses cog_Mb (q_ind x p_b). Writes cogwas_top.csv
(the top pairs of each test), cogwas_focal.csv and cogwas_neighbour.csv
(smallest p per locus and test) and cogwas_meta.json.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np


def _parse_args(argv):
    ap = argparse.ArgumentParser(prog="cogwas_cli")
    ap.add_argument("bundle")
    ap.add_argument("--tests", default="pair|dir+ind")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--block-a", type=int, default=1024)
    ap.add_argument("--block-b", type=int, default=8192)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _inputs(path, n):
    from remlax.scan_cli import _coo, _read_array
    with open(os.path.join(path, "cogwas_manifest.json")) as f:
        cm = json.load(f)
    man = {d["name"]: d for d in cm["arrays"]}
    meta = cm["meta"]
    Zg = _coo(path, man, "scan_dir", n, int(meta["q_dir"]))
    Zn = _coo(path, man, "scan_ind", n, int(meta["q_ind"]))
    Ma = _read_array(path, man, "cog_Ma")
    Mb = _read_array(path, man, "cog_Mb")
    return Zg, Zn, Ma, Mb


def _write_csv(path, cols):
    names = list(cols)
    m = len(cols[names[0]])
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(m):
            f.write(",".join(("%s" % cols[c][i]) if isinstance(cols[c][i], (str, np.integer, int))
                             else ("" if not np.isfinite(cols[c][i]) else "%.10g" % cols[c][i])
                             for c in names) + "\n")


def main(argv=None):
    from remlax.bundle import Bundle
    from remlax.device import pick_device
    from remlax.scan_cli import _null_fit
    from remlax.cogwas import PairProjection, cogwas
    import jax

    a = _parse_args(argv)
    dev, plat = pick_device(a.backend)
    jax.default_device(dev).__enter__()
    xp = None if plat == "cpu" else jax.numpy
    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    Zg, Zn, Ma, Mb = _inputs(a.bundle, len(y))
    fit, t_fit = _null_fit(SimpleNamespace(bundle=a.bundle, theta_in=True, maxiter=0, polish=0,
                                           quiet=a.quiet), terms, res, y, X)
    t0 = time.time()
    proj = PairProjection(fit, X, Zg, Zn, xp=xp)
    t_proj = time.time() - t0
    t0 = time.time()
    r = cogwas(proj, Ma, Mb, tests=tests, top=a.top, block_a=a.block_a, block_b=a.block_b,
               verbose=not a.quiet)
    t_pairs = time.time() - t0
    rows = {"test": [], "rank": [], "focal": [], "neighbour": [], "chi2": [], "p": [], "beta": [], "se": []}
    for t in tests:
        rt = r[t]
        for i in range(len(rt["chi2"])):
            rows["test"].append(t); rows["rank"].append(i + 1)
            rows["focal"].append(int(rt["a"][i]) + 1); rows["neighbour"].append(int(rt["b"][i]) + 1)
            for k in ("chi2", "p", "beta", "se"):
                rows[k].append(float(rt[k][i]))
    _write_csv(os.path.join(a.bundle, "cogwas_top.csv"), rows)
    _write_csv(os.path.join(a.bundle, "cogwas_focal.csv"),
               {("min_p_" + t): r[t]["min_p_a"] for t in tests})
    _write_csv(os.path.join(a.bundle, "cogwas_neighbour.csv"),
               {("min_p_" + t): r[t]["min_p_b"] for t in tests})
    with open(os.path.join(a.bundle, "cogwas_meta.json"), "w") as f:
        json.dump(dict(tests=tests, n=len(y), p_focal=int(Ma.shape[1]), p_neighbour=int(Mb.shape[1]),
                       n_pairs=int(r["n_pairs"]), backend=plat,
                       **{"lambda": {t: float(r[t]["lambda"]) for t in tests}},
                       seconds_fit=t_fit, seconds_projection=t_proj, seconds_pairs=t_pairs), f)
    if not a.quiet:
        print("[cogwas] %d pairs, %s | constants %.1f s, pairs %.1f s"
              % (r["n_pairs"], plat, t_proj, t_pairs), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
