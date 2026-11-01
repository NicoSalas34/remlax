"""Entry point of the genome-wide scan, beside `cli.py`.

This module neither imports nor modifies `cli.py`. It reuses `Bundle`,
`fit_reml` and `scan`, and nothing else.

    python3 remlax_run.py scan_cli <bundle> --tests "dir,ind,dir+ind+sim,ind|dir+sim"

The scan has its own entry point because it needs `Vi` and `Py`. These n x n
matrices live in the process of the fit and do not travel to R through the
result file (2 GB at n = 16 000). The scan must therefore run where the fit
runs. Since `cli.py` stays untouched, a second executable repeats the path
bundle -> fit -> inference, with the scan in place of the Wald tests.

`--theta-in --maxiter 0` reads the theta already estimated and only evaluates
it. V is assembled again, Vi and Py are computed, and the optimiser is
skipped. The scan therefore does not pay for a second estimation of the
variance components. It uses the components of the fit that the user has
already run and checked.

The bundle written by `rx_export` is left untouched. The arrays specific to
the scan live beside it, described by `scan_manifest.json`, and a small local
reader reads them back. `manifest.json` cannot be corrupted this way.

    scan_manifest.json   {"arrays": [{"name":..., "dtype":"f8"|"i4"|"str", "shape":[...]}],
                          "meta": {"q_dir":..., "q_ind":..., "p":...}}
    scan_dir_zi/zj/zx    incidence of the genotype carried by the plant, COO
    scan_ind_zi/zj/zx    weighted incidence of the neighbour genotypes, COO
    scan_<f>_t<k>_zi/... per-trait incidences of a multi-trait term (meta T_<f>)
    scan_trait, scan_traits  trait of each row (0-based codes) and trait names
    scan_Mdir            genotype doses (q_dir x p), CENTRED for `sim`
    scan_Mind            doses of the neighbour genotypes (q_ind x p)
    scan_M               doses shared by both families (replaces the two above)
    scan_snp             SNP names (optional)
    scan_chr, scan_pos   map (optional)
    scan_keep          0/1 mask of the SNPs to process (optional)
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def _read_array(bundle_dir, man, name):
    """One array of the scan manifest, in the format of Bundle.get."""
    m = man[name]
    if m["dtype"] == "str":
        with open(os.path.join(bundle_dir, name + ".txt")) as f:
            return [l.rstrip("\n") for l in f]
    dt = {"f8": np.float64, "i4": np.int32}[m["dtype"]]
    a = np.fromfile(os.path.join(bundle_dir, name + ".bin"), dtype=dt)
    shp = tuple(int(v) for v in m["shape"])
    return a.reshape(shp, order="F") if len(shp) > 1 else a


def _coo(bundle_dir, man, pre, n, q):
    """Dense (n x q) incidence from the COO triplets <pre>_zi, _zj, _zx."""
    from scipy.sparse import coo_matrix
    i = _read_array(bundle_dir, man, pre + "_zi").astype(np.int64)
    j = _read_array(bundle_dir, man, pre + "_zj").astype(np.int64)
    x = _read_array(bundle_dir, man, pre + "_zx").astype(np.float64)
    return coo_matrix((x, (i, j)), shape=(n, q)).toarray()


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="remlax-scan",
        description="GWAS at fixed V on a remlax bundle (see docs/scan.md).")
    ap.add_argument("bundle")
    ap.add_argument("--tests", default="dir,ind,sim,dir+ind+sim",
                    help="comma-separated specifications. A bar marks "
                         "a conditional test, for example 'ind|dir+sim'")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--theta-in", action="store_true",
                    help="rereads in_theta.bin. With --maxiter 0 theta is "
                         "only EVALUATED, without reoptimisation")
    ap.add_argument("--maxiter", type=int, default=0)
    ap.add_argument("--polish", type=int, default=0)
    ap.add_argument("--block", type=int, default=1024,
                    help="width of the SNP blocks for the family 'sim'")
    ap.add_argument("--output", default="scan_results.csv")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _scan_inputs(bundle_dir, n, tests, needed):
    """Manifest, incidences (per trait for a multi-trait term), doses and
    trait codes written by R beside the bundle."""
    with open(os.path.join(bundle_dir, "scan_manifest.json")) as f:
        sm = json.load(f)
    man = {d["name"]: d for d in sm["arrays"]}
    meta = sm.get("meta", {})
    # A `scan_M` array means the same SNPs on both sides (intra-species case).
    # The matrix is written once, and a single array is passed to `scan`. This
    # avoids duplicating a large matrix and removes any ambiguity about the
    # alignment of columns in joint tests.
    M_shared = _read_array(bundle_dir, man, "scan_M") if "scan_M" in man else None
    incid, M = {}, {}
    for family, pre, cq, cm in (("dir", "scan_dir", "q_dir", "scan_Mdir"),
                                ("ind", "scan_ind", "q_ind", "scan_Mind")):
        Tf = int(meta.get("T_" + family, 0))
        if Tf > 0:
            # multi-trait term: one incidence per trait
            incid[family] = [_coo(bundle_dir, man, "%s_t%d" % (pre, k), n, int(meta[cq]))
                             for k in range(Tf)]
        elif pre + "_zi" in man:
            incid[family] = _coo(bundle_dir, man, pre, n, int(meta[cq]))
        else:
            continue
        if M_shared is None:
            M[family] = _read_array(bundle_dir, man, cm)
    _check_incidences(tests, needed, incid)
    get = lambda name: _read_array(bundle_dir, man, name) if name in man else None
    return dict(man=man, meta=meta, incid=incid, M=M if M_shared is None else M_shared,
                trait=get("scan_trait"), traits=get("scan_traits"),
                keep=(get("scan_keep").astype(bool) if "scan_keep" in man else None))


def _check_incidences(tests, needed, incid):
    missing = [f for f in needed if f in ("dir", "ind") and f not in incid]
    if "sim" in needed:
        missing += [f for f in ("dir", "ind") if f not in incid]
    if missing:
        raise SystemExit(
            "[scan] tests %s requested but incidence(s) missing from the bundle: %s. "
            "On the R side, `incidences=` must name the terms of the model that "
            "carry them." % (tests, sorted(set(missing))))


def _null_fit(a, terms, res, y, X):
    """Fit of the null model, or plain evaluation of the supplied theta."""
    from remlax.fit import fit_reml
    th0 = None
    if a.theta_in:
        th0 = np.fromfile(os.path.join(a.bundle, "in_theta.bin"), dtype=np.float64)
    elif a.maxiter == 0:
        raise SystemExit("[scan] --maxiter 0 without --theta-in leaves "
                         "no theta to evaluate. Supply in_theta.bin, or "
                         "request a real fit with --maxiter > 0.")
    t0 = time.time()
    fit = fit_reml(terms, res, y, X, theta_init=th0, maxiter=a.maxiter,
                   polish=a.polish, verbose=not a.quiet,
                   hessian=False, blups=True, pev=False)
    t_fit = time.time() - t0
    if not a.quiet:
        print("[scan] null model: -2logL = %.6f in %.1f s"
              % (fit["neg2_reml"], t_fit), flush=True)
    return fit, t_fit


def _write_table(path, cols, p):
    """CSV with one row per SNP; non-finite numbers as empty fields."""
    names = list(cols)
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(p):
            f.write(",".join(
                ("" if isinstance(cols[k][i], float) and not np.isfinite(cols[k][i])
                 else ("%s" % cols[k][i] if isinstance(cols[k][i], str)
                       else "%.10g" % cols[k][i]))
                for k in names) + "\n")


def _output_columns(bundle_dir, man, r, p):
    cols = {}
    cols["snp"] = (_read_array(bundle_dir, man, "scan_snp") if "scan_snp" in man
                   else ["snp%d" % (i + 1) for i in range(p)])
    if "scan_chr" in man:
        cols["chr"] = _read_array(bundle_dir, man, "scan_chr")
    if "scan_pos" in man:
        cols["pos"] = _read_array(bundle_dir, man, "scan_pos")
    cols.update(r)
    return cols


def main(argv=None):
    from remlax.bundle import Bundle
    from remlax.device import pick_device
    from remlax.scan import project, scan, lambda_gc, required_families

    a = _parse_args(argv)
    import jax
    dev, plat = pick_device(a.backend)
    ctx = jax.default_device(dev)
    ctx.__enter__()
    if not a.quiet:
        print("[scan] backend chosen: %s (%s)" % (plat, dev), flush=True)

    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    n = len(y)
    inp = _scan_inputs(a.bundle, n, tests, required_families(tests))
    fit, t_fit = _null_fit(a, terms, res, y, X)

    t0 = time.time()
    proj = project(fit, X, inp["incid"], trait=inp["trait"], traits=inp["traits"])
    t_proj = time.time() - t0
    t0 = time.time()
    # The `sim` products run on the chosen device: on a GPU they go through
    # jax.numpy, on CPU through numpy (BLAS), as before.
    xp = None if plat == "cpu" else jax.numpy
    r = scan(proj, inp["M"], tests=tests, block=a.block, verbose=not a.quiet, keep=inp["keep"],
             xp=xp)
    t_scan = time.time() - t0
    p = int(inp["meta"]["p"])
    lam = {t: lambda_gc(r["chi2_" + t], int(r["df_" + t][0])) for t in tests}
    if not a.quiet:
        print("[scan] projection %.2f s | scan of %d SNP %.2f s" % (t_proj, p, t_scan), flush=True)
        for t in tests:
            print("        lambda(%s) = %.3f" % (t, lam[t]), flush=True)

    out_path = os.path.join(a.bundle, a.output)
    _write_table(out_path, _output_columns(a.bundle, inp["man"], r, p), p)
    with open(os.path.join(a.bundle, "scan_meta.json"), "w") as f:
        json.dump(dict(tests=tests, n=n, p=p, seconds_fit=t_fit,
                       seconds_projection=t_proj, seconds_scan=t_scan,
                       neg2_reml=float(fit["neg2_reml"]), lambda_gc=lam), f, indent=1)
    if not a.quiet:
        print("[scan] written: %s" % out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
