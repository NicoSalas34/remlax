"""Point d'entree du solveur : lit un dispositif serialise, ajuste, ecrit.

    python3 -m remlkit.cli <repertoire> [--backend auto|gpu|cpu] [--maxiter N]
                           [--polish N] [--no-hessian] [--no-blups] [--quiet]

Ecrit dans le meme repertoire : result.json (scalaires et dimensions) et des
binaires out_*.bin (theta, beta, Sigma, BLUPs) relus par R. Le format de sortie
est le miroir exact du format d'entree, pour que R n'ait besoin d'aucune
dependance Python et Python d'aucune dependance R.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from remlkit import _x64  # noqa: F401,E402
from remlkit.bundle import Bundle  # noqa: E402
from remlkit.device import pick_device, device_report  # noqa: E402
from remlkit.fit import fit_reml  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="solveur REML generique")
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--polish", type=int, default=25)
    ap.add_argument("--restarts", type=int, default=0,
                    help="redemarrages perturbes : seul moyen de detecter un optimum local")
    ap.add_argument("--restart-sd", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=-12.0)
    ap.add_argument("--ceil", type=float, default=12.0)
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--no-blups", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--vpredict", default=None,
                    help="expressions 'nom=formule' separees par ';', ex. "
                         "\"h2=V1/(V1+V2);rg=V2/sqrt(V1*V3)\"")
    ap.add_argument("--wald", action="store_true",
                    help="tests de Wald sur les effets fixes")
    ap.add_argument("--kenward-roger", action="store_true",
                    help="ddl du denominateur et covariance ajustee de beta (K&R 1997)")
    ap.add_argument("--only-predict", action="store_true",
                    help="AUCUN ajustement : relit in_theta.bin et ne fait que predire")
    ap.add_argument("--predict", action="store_true",
                    help="predictions : lit pred_L.bin (et pred_M_<terme>.bin) du paquet")
    ap.add_argument("--fixed-theta", default=None,
                    help="indices (1-based) des parametres FIXES a leur valeur de depart")
    a = ap.parse_args(argv)

    import jax
    dev, plat = pick_device(a.backend)
    # jax.default_device fixe le peripherique pour TOUTES les operations
    # suivantes : c'est le seul endroit ou le choix CPU/GPU intervient, et il
    # ne touche a rien d'autre que la machine.
    ctx = jax.default_device(dev)
    ctx.__enter__()
    if not a.quiet:
        print("[remlkit] peripheriques : %s" % device_report(), flush=True)
        print("[remlkit] backend retenu : %s (%s)" % (plat, dev), flush=True)

    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    if not a.quiet:
        print("[remlkit] n=%d, %d effet(s) fixe(s), %d terme(s) aleatoire(s)"
              % (len(y), X.shape[1], len(terms)), flush=True)
        for t in terms:
            print("          %-14s %-5s t=%-3d q=%-5d %s"
                  % (t["name"], t["struct"], t["t"], t["q"],
                     "K=I" if t["LK"] is None else "K fournie"), flush=True)

    fixe = None
    if a.fixed_theta:
        fixe = [int(v) - 1 for v in a.fixed_theta.split(",") if v.strip()]
    if a.only_predict:
        # Predire ne demande PAS de reajuster. Sans ce mode, chaque appel a
        # rk_predict() relancerait l'optimisation complete pour retrouver le
        # theta qu'on a deja — et rien ne garantirait qu'il retombe sur le meme
        # optimum.
        r = dict(theta=np.fromfile(os.path.join(a.bundle, "in_theta.bin"),
                                   dtype=np.float64),
                 n_par=0, n_obs=len(y), logLik=float("nan"), secondes=0.0,
                 only_predict=True)
        a.predict = True
    else:
        r = fit_reml(terms, res, y, X, maxiter=a.maxiter, polish=a.polish,
                     floor=a.floor, ceil=a.ceil, verbose=not a.quiet,
                     hessian=not a.no_hessian, blups=not a.no_blups,
                     n_restarts=a.restarts, restart_sd=a.restart_sd, fixed_idx=fixe)

    # --- inference ------------------------------------------------------------
    from remlkit.inference import vpredict as _vp, wald as _wald, component_names
    r["composantes_noms"] = component_names(terms, res)
    if a.only_predict and len(r["theta"]) != len(r["composantes_noms"]) and not a.quiet:
        print("[remlkit] mode predict seul : %d parametre(s) relus"
              % len(r["theta"]), flush=True)
    if a.vpredict:
        exprs = []
        for bloc in a.vpredict.split(";"):
            if not bloc.strip():
                continue
            nom, _, ex = bloc.partition("=")
            exprs.append((nom.strip(), ex.strip()))
        libre = np.ones(len(r["theta"]), dtype=bool)
        if fixe:
            libre[np.array(fixe)] = False
        r["vpredict"] = _vp(r["theta"], r.get("hessian"), terms, res, exprs, free=libre)
    if a.kenward_roger:
        from remlkit.inference import kenward_roger as _kr
        rk = _kr(r["theta"], terms, res, y, X, termes_fixes=b.fixed_groups())
        if rk.get("vbeta_kr") is not None:
            r["vbeta_kr"] = rk.pop("vbeta_kr")
        r["kenward_roger"] = rk
    if a.predict:
        from remlkit.inference import predict as _pred
        Lp = np.fromfile(os.path.join(a.bundle, "pred_L.bin"),
                         dtype=np.float64).reshape((-1, X.shape[1]), order="F")
        M = {}
        for t in terms:
            fp = os.path.join(a.bundle, "pred_M_%s.bin" % t["name"])
            if os.path.exists(fp):
                M[t["name"]] = np.fromfile(fp, dtype=np.float64).reshape(
                    (Lp.shape[0], t["t"] * t["q"]), order="F")
        pr = _pred(r["theta"], terms, res, y, X, Lp, M=M or None, vbeta=r.get("vbeta"))
        r["pred_cov"] = pr.pop("cov")
        r["predictions"] = pr
    if a.wald:
        import jax.numpy as _jnp
        from remlkit.model import assemble_V as _aV, dense_Z as _dZ
        Zs = [_dZ(t, len(y)) for t in terms]
        V = np.asarray(_aV(_jnp.asarray(r["theta"]), terms, Zs, res, len(y)))
        r["wald"] = _wald(y, X, V, termes_fixes=b.fixed_groups())

    out = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
           for k, v in r.items()
           if k not in ("theta", "beta", "sigmas", "sigma_res", "sigmas_res",
                        "blups", "hessian", "Py", "vbeta", "vbeta_kr", "Vi",
                        "pred_cov")}
    out["backend"] = plat
    out["devices"] = device_report()

    def wr(name, arr):
        np.asarray(arr, dtype=np.float64).ravel(order="F").tofile(
            os.path.join(a.bundle, name + ".bin"))

    wr("out_theta", r["theta"])
    if "beta" in r: wr("out_beta", r["beta"])
    if "vbeta" in r: wr("out_vbeta", r["vbeta"])
    if "vbeta_kr" in r: wr("out_vbeta_kr", r["vbeta_kr"])
    if "pred_cov" in r: wr("out_pred_cov", r["pred_cov"])
    out["sigma_dims"] = {}
    for nm, S in r.get("sigmas", {}).items():
        wr("out_sigma_%s" % nm, S); out["sigma_dims"][nm] = list(np.shape(S))
    if "sigma_res" in r: wr("out_sigma_res", r["sigma_res"])
    out["sigma_res_dims"] = {}
    for nm, S in r.get("sigmas_res", {}).items():
        wr("out_sigmares_%s" % nm, S); out["sigma_res_dims"][nm] = list(np.shape(S))
    out["blup_dims"] = {}
    for nm, U in r.get("blups", {}).items():
        wr("out_blup_%s" % nm, U); out["blup_dims"][nm] = list(np.shape(U))
    if "hessian" in r: wr("out_hessian", r["hessian"])

    # JSON n'a pas de NaN : json.dump en ecrit quand meme, et tout lecteur
    # conforme (jsonlite par exemple) refuse alors le fichier ENTIER. Un
    # diagnostic indisponible ferait donc perdre l'ajustement complet. On
    # convertit en null, que R relit en NA.
    def propre(o):
        if isinstance(o, dict):
            return {k: propre(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [propre(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open(os.path.join(a.bundle, "result.json"), "w") as f:
        json.dump(propre(out), f, indent=2, default=float)
    if not a.quiet and not a.only_predict:
        print("[remlkit] logLik = %.9f | %d parametres | %.1f s"
              % (r["logLik"], r["n_par"], r["secondes"]), flush=True)
    ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
