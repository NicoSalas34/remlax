"""Command line of the batched exact refits (refit_batch.py), called by
rx_scan_exact(batch = ) from R.

    python -m remlax.refit_batch_cli <bundle> [--backend auto|gpu|cpu]
        [--batch 64] [--maxit 50] [--tol 1e-9] [--quiet]

The bundle is the one rx_export writes for the null model, plus:
  refit_meta.json   {"n", "k", "n_snp", "r"}
  refit_S.bin       SNP columns, float64, R order of an n x k x n_snp array
  refit_L.bin       tested hypothesis, float64, R order of an r x k matrix
  in_theta.bin      optional start; without it the null model is fitted
Writes refit_results.csv, one row per SNP in the order of refit_S.bin.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def _parse_args(argv):
    ap = argparse.ArgumentParser(prog="refit_batch_cli")
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--maxit", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _inputs(path):
    """SNP columns (n_snp, n, k), hypothesis L (r, k) and optional start."""
    meta = json.load(open(os.path.join(path, "refit_meta.json")))
    n, k, m, r = (int(meta[x]) for x in ("n", "k", "n_snp", "r"))
    S = np.fromfile(os.path.join(path, "refit_S.bin"), dtype=np.float64)
    if S.size != n * k * m:
        raise SystemExit("[refit] refit_S.bin holds %d values, %d x %d x %d expected"
                         % (S.size, n, k, m))
    S = S.reshape(m, k, n).transpose(0, 2, 1)           # R column-major n x k x m
    L = np.fromfile(os.path.join(path, "refit_L.bin"), dtype=np.float64).reshape(k, r).T
    th_path = os.path.join(path, "in_theta.bin")
    th0 = np.fromfile(th_path, dtype=np.float64) if os.path.exists(th_path) else None
    return S, L, th0


def _write(path, r, L):
    from scipy.stats import chi2 as _chi2
    rows = L.shape[0]
    cols = {"chi2_exact": r["chi2"], "df": np.full(len(r["chi2"]), float(rows))}
    cols["p_exact"] = _chi2.sf(r["chi2"], rows)
    cols["logLik"] = -0.5 * r["neg2_reml"]
    cols["neg2_reml"] = r["neg2_reml"]
    cols["converged"] = r["converged"].astype(float)
    cols["n_iter"] = r["n_iter"].astype(float)
    cols["decrement"] = r["decrement"]
    for a in range(rows):
        cols["Lb_%d" % (a + 1)] = r["Lb"][:, a]
        cols["se_%d" % (a + 1)] = r["se_Lb"][:, a]
    names = list(cols)
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(len(r["chi2"])):
            f.write(",".join("" if not np.isfinite(cols[c][i]) else "%.12g" % cols[c][i]
                             for c in names) + "\n")


def main(argv=None):
    from remlax.bundle import Bundle
    from remlax.device import pick_device
    from remlax.fit import fit_reml, _bounds
    from remlax.refit_batch import refit_batch
    import jax

    a = _parse_args(argv)
    dev, plat = pick_device(a.backend)
    jax.default_device(dev).__enter__()
    b = Bundle(a.bundle)
    if b.has("weights"):
        raise SystemExit("[refit] batched refits do not yet accept prior weights")
    if b.lag() is not None:
        raise SystemExit("[refit] batched refits do not accept a spatial lag (rx_lag)")
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    S, L, th0 = _inputs(a.bundle)
    t0 = time.time()
    if th0 is None:
        th0 = fit_reml(terms, res, y, X, verbose=False, hessian=False, blups=False)["theta"]
    bornes, th0 = _bounds(th0, len(th0), -12.0, 12.0, terms, res, y, None)
    lo = np.array([x[0] for x in bornes]); hi = np.array([x[1] for x in bornes])
    if not a.quiet:
        print("[refit] %s (%s) | %d SNP, n = %d, %d parameter(s), batch %d"
              % (plat, dev, S.shape[0], S.shape[1], len(th0), a.batch), flush=True)
    r = refit_batch(terms, res, y, X, S, L, th0, lo, hi, maxit=a.maxit, tol=a.tol,
                    batch=a.batch, verbose=not a.quiet)
    _write(os.path.join(a.bundle, "refit_results.csv"), r, L)
    if not a.quiet:
        print("[refit] %d of %d converged in %.1f s"
              % (int(r["converged"].sum()), len(r["converged"]), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
