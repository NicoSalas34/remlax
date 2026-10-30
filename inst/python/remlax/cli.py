"""Entry point of the solver: reads a serialised model, fits it, writes the result.

    python3 remlax_run.py cli <bundle> [--backend auto|gpu|cpu] [--maxiter N]
                              [--polish N] [--no-hessian] [--no-blups] [--quiet]

The result goes into the bundle directory: result.json (scalars and the
dimensions of every array) and raw float64 files out_*.bin (theta, beta,
Sigma, BLUPs) read back by R. The output format mirrors the input format, so
that R needs no Python dependency and Python no R dependency.

main() runs these steps in order:

    _parser            command line
    _select_device     JAX device, entered as a context for the whole run
    _read_inputs       bundle, weights, spatial lag, checks of the options
    _parse_vpredict    vpredict expressions, checked before the fit
    _fit               REML fit, posterior mode, or no fit (--only-predict)
    _inference         component table, vpredict, Kenward-Roger, predict, Wald
    _sample_bayes      posterior sample (--bayes)
    _write_outputs     out_*.bin and result.json

A step that refuses the request raises _Refusal; main() prints its message on
stderr and returns 2.
"""
import time
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

# End of interpreter start-up, recorded by remlax_run.py before any heavy import.
_T_DEBUT = float(os.environ.get("RX_T_DEBUT", time.time()))

# Threads bounded to the allocated cores, before NumPy and JAX size their
# thread pools on the affinity of the process (fils.py).
from remlax.fils import bound_threads, _ETAT as _FILS  # noqa: E402
bound_threads()

import numpy as np  # noqa: E402

# Allocator ceiling for the low-memory mode and the blocks engine. By default
# JAX caps its pool at 75 % of the GPU memory, which is too little on a MIG
# slice (dev/notes.md). The variable is read when the backend initialises,
# which happens on the imports below, so it is set here. A value set by the
# caller is kept.
if "--memoire=econome" in sys.argv or "--moteur=blocs" in sys.argv or "--inversion=hybride" in sys.argv or any(
        (a, b) in (("--memoire", "econome"), ("--moteur", "blocs"), ("--inversion", "hybride"))
        for a, b in zip(sys.argv, sys.argv[1:])):
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from remlax import _x64  # noqa: F401,E402
from remlax import chrono  # noqa: E402
from remlax.bundle import Bundle  # noqa: E402
from remlax.device import pick_device, device_report, enable_compilation_cache, without_gpu_plugins  # noqa: E402
from remlax.fit import fit_reml  # noqa: E402

# Timing by stages (chrono.py). "demarrage" runs from the launch by R
# (RX_T_LANCEMENT) to the first line of this file; "imports" covers numpy, JAX
# and the solver. In server mode, serveur.py resets the timer at each request
# and both stages disappear.
chrono.reset()
chrono.set_at("demarrage", _T_DEBUT)
chrono.mark("imports")


class _Refusal(Exception):
    """A request the solver refuses; main() prints it and returns 2."""


def _error(msg):
    print("[remlax] ERREUR : %s" % msg, file=sys.stderr, flush=True)


def main(argv=None):
    a = _parser().parse_args(argv)
    import jax
    dev, plat = _select_device(a)
    with jax.default_device(dev):
        try:
            return _run(a, plat)
        except _Refusal as e:
            _error(str(e))
            return 2


def _run(a, plat):
    inp = _read_inputs(a)
    exprs = _parse_vpredict(a, inp)
    fixed = [int(v) - 1 for v in a.fixed_theta.split(",") if v.strip()] if a.fixed_theta else None
    r, Hu_map, prior = _fit(a, inp, fixed)
    _inference(a, r, inp, exprs, fixed)
    bayes = _sample_bayes(a, r, inp, exprs, fixed, plat, Hu_map, prior)
    chrono.mark("inference")
    _write_outputs(a, r, bayes, plat)
    if not a.quiet and not a.only_predict:
        print("[remlax] logLik = %.9f | %d parametres | %.1f s"
              % (r["logLik"], r["n_par"], r["secondes"]), flush=True)
    return 0


# ------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------

def _parser():
    ap = argparse.ArgumentParser(description="solveur REML generique")
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--polish", type=int, default=25)
    ap.add_argument("--optimiseur", default="lbfgsb", choices=["lbfgsb", "ai", "auto"])
    ap.add_argument("--moteur", default="dense", choices=["dense", "mme", "blocs", "auto"])
    ap.add_argument("--inversion", default="auto", choices=["auto", "dense", "creuse", "hybride"])
    ap.add_argument("--finition", default="newton", choices=["newton", "rc"],
                    help="fin d'ajustement : polissage de Newton apres L-BFGS-B (defaut), ou "
                         "arret sur progres lent et Newton a region de confiance (finition.py)")
    ap.add_argument("--arret-lent", default="100,0.05",
                    help="finition rc : fenetre (iterations) et gain minimal de -2logL")
    ap.add_argument("--memoire", default="standard", choices=["standard", "econome"],
                    help="moteur dense : 'econome' garde moins de matrices n x n sur le "
                         "peripherique (econome.py) ; meme vraisemblance")
    ap.add_argument("--restarts", type=int, default=0,
                    help="redemarrages perturbes : seul moyen de detecter un optimum local")
    ap.add_argument("--restart-sd", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=-12.0)
    ap.add_argument("--ceil", type=float, default=12.0)
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--no-blups", action="store_true")
    ap.add_argument("--pev-termes", dest="pev_termes", default=None,
                    help="liste de termes, separes par des virgules, pour "
                         "lesquels calculer la PEV. Preferable a --pev sur un "
                         "modele a nombreuses nuisances.")
    ap.add_argument("--pev", action="store_true",
                    help="variance d'erreur de prediction des BLUP (diagonale). "
                         "Necessaire a une heritabilite de Cullis ou a une "
                         "fiabilite ; coute une matrice (t*q) x n par terme.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--vpredict", default=None,
                    help="expressions 'nom=formule' separees par ';', ex. "
                         "\"h2=V1/(V1+V2);rg=V2/sqrt(V1*V3)\"")
    ap.add_argument("--wald", action="store_true",
                    help="tests de Wald sur les effets fixes")
    ap.add_argument("--kenward-roger", action="store_true",
                    help="ddl du denominateur et covariance ajustee de beta (K&R 1997)")
    ap.add_argument("--theta-in", action="store_true",
                    help="depart a chaud : lit in_theta.bin comme point initial "
                         "(avec --maxiter 0, evalue simplement la vraisemblance)")
    ap.add_argument("--only-predict", action="store_true",
                    help="AUCUN ajustement : relit in_theta.bin et ne fait que predire")
    ap.add_argument("--predict", action="store_true",
                    help="predictions : lit pred_L.bin (et pred_M_<terme>.bin) du paquet")
    ap.add_argument("--fixed-theta", default=None,
                    help="indices (1-based) des parametres FIXES a leur valeur de depart")
    ap.add_argument("--bayes", action="store_true",
                    help="loi a posteriori des parametres de variance par HMC (blackjax, "
                         "ChEES), depuis l'optimum REML")
    ap.add_argument("--depart", default="map", choices=["map", "reml"],
                    help="voie bayesienne : 'map' (defaut) part du mode a posteriori, SANS "
                         "ajustement REML ; 'reml' ajuste d'abord par REML")
    ap.add_argument("--chaines", type=int, default=0,
                    help="nombre de chaines ; 0 : 8 sur CPU, 32 sur GPU")
    ap.add_argument("--superchaines", type=int, default=4)
    ap.add_argument("--chauffe", type=int, default=150)
    ap.add_argument("--tirages", type=int, default=150)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--acceptation", type=float, default=0.651)
    ap.add_argument("--sauts-max", dest="sauts_max", type=int, default=1000)
    ap.add_argument("--prior", default=None,
                    help="fichier JSON de loi a priori, relatif au paquet (prior.json)")
    ap.add_argument("--bayes-effets", dest="bayes_effets", type=int, default=100,
                    help="tirages servant aux moments a posteriori de beta et des BLUP")
    return ap


def _pev_arg(a):
    """The --pev-termes list, or the --pev flag."""
    return a.pev_termes.split(",") if a.pev_termes else a.pev


def _slow_stop_arg(a):
    """--arret-lent as (window, minimal gain)."""
    w, g = a.arret_lent.split(",")
    return (int(w), float(g))


# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------

def _select_device(a):
    """The JAX device of the run. jax.default_device() in main() applies it to
    every later operation: the CPU/GPU choice enters nowhere else."""
    # CPU only when the whole process is (JAX_PLATFORMS=cpu, set by R for
    # backend = "cpu" in a subprocess): no CUDA plugin. The persistent server
    # serves CPU and GPU in the same process and keeps everything.
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        without_gpu_plugins()
    enable_compilation_cache()
    dev, plat = pick_device(a.backend)
    if not a.quiet:
        print("[remlax] peripheriques : %s" % device_report(), flush=True)
        if _FILS["message"]:
            print(_FILS["message"], flush=True)
        print("[remlax] backend retenu : %s (%s)" % (plat, dev), flush=True)
    chrono.mark("peripherique")
    return dev, plat


# ------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------

class _Inputs:
    """What the bundle holds: the model (terms, res, y, X), the prior weights,
    the prepared spatial lag, and the bundle itself for the optional files."""

    def __init__(self, bundle, terms, res, y, X, weights, lag):
        self.bundle, self.terms, self.res, self.y, self.X = bundle, terms, res, y, X
        self.weights, self.lag = weights, lag


def _read_inputs(a):
    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    chrono.mark("lecture")
    # Prior weights of the observations (rx_model(weights = )): fit.fit_reml.
    weights = np.asarray(b.get("poids"), dtype=np.float64) if b.has("poids") else None
    if weights is not None and a.bayes:
        raise _Refusal("la voie bayesienne ne prend pas encore les poids")
    lag = b.lag()
    if lag is not None:
        lag = _prepare_lag(a, lag, y, X)
    if not a.quiet:
        _print_model(terms, y, X)
    return _Inputs(b, terms, res, y, X, weights, lag)


def _prepare_lag(a, lag, y, X):
    """Spectral decomposition of W and long-format W, prepared once: the same
    object serves the fit and the inference tables."""
    from remlax.lag import prepare_lag
    lag = prepare_lag(lag, y, X)
    if not a.quiet:
        print("[remlax] decalage spatial : q=%d, %d entree(s) libre(s) de Psi, critere %s, "
              "chemin %s" % (lag["q"], int(lag["mask"].sum()), lag["criterion"],
                             lag["chemin"]), flush=True)
    if a.kenward_roger or a.predict or a.only_predict:
        raise _Refusal("Kenward-Roger et predict ne sont pas disponibles avec un "
                       "decalage spatial (rx_lag) : ils supposent y lineaire en beta et u a V "
                       "donnee, alors qu'ici seule y* = (I - M) y l'est.")
    return lag


def _print_model(terms, y, X):
    print("[remlax] n=%d, %d effet(s) fixe(s), %d terme(s) aleatoire(s)"
          % (len(y), X.shape[1], len(terms)), flush=True)
    for t in terms:
        print("          %-14s %-5s t=%-3d q=%-5d %s"
              % (t["name"], t["struct"], t["t"], t["q"],
                 "K=I" if t["LK"] is None else "K fournie"), flush=True)


def _parse_vpredict(a, inp):
    """The vpredict expressions as (name, formula) pairs, checked before the
    fit: a misspelt name must not cost a whole optimisation."""
    exprs = []
    if not a.vpredict:
        return exprs
    for block in a.vpredict.split(";"):
        if not block.strip():
            continue
        name, _, ex = block.partition("=")
        exprs.append((name.strip(), ex.strip()))
    from remlax.vpredict import check_vpredict
    try:
        check_vpredict(exprs, inp.terms, inp.res, lag=inp.lag)
    except ValueError as e:
        raise _Refusal(str(e))
    return exprs


# ------------------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------------------

def _read_theta(a):
    return np.fromfile(os.path.join(a.bundle, "in_theta.bin"), dtype=np.float64)


def _fit(a, inp, fixed):
    """The fit result r, plus the whitening Hu_map and the prior of the
    Bayesian route (None otherwise).

    --only-predict fits nothing: it reads theta back. Without it, every call to
    rx_predict() would rerun the whole optimisation to recover a theta already
    known, with no guarantee of reaching the same optimum."""
    th0 = None
    if a.only_predict:
        r = dict(theta=_read_theta(a), n_par=0, n_obs=len(inp.y), logLik=float("nan"),
                 secondes=0.0, only_predict=True)
        a.predict = True
    elif a.theta_in:
        th0 = _read_theta(a)
    prior, Hu_map = None, None
    if a.bayes and a.prior:
        with open(os.path.join(a.bundle, a.prior)) as fp:
            prior = json.load(fp)
    if a.bayes and a.depart == "map" and not a.only_predict:
        r, Hu_map = _fit_posterior_mode(a, inp, fixed, th0, prior)
    elif not a.only_predict:
        r = fit_reml(inp.terms, inp.res, inp.y, inp.X, theta_init=th0,
                     maxiter=a.maxiter, polish=a.polish,
                     floor=a.floor, ceil=a.ceil, verbose=not a.quiet,
                     hessian=not a.no_hessian, blups=not a.no_blups,
                     pev=_pev_arg(a),
                     n_restarts=a.restarts, restart_sd=a.restart_sd, fixed_idx=fixed,
                     lag=inp.lag, optimizer=a.optimiseur, moteur=a.moteur,
                     inversion=a.inversion, memoire=a.memoire, finition=a.finition,
                     poids=inp.weights, arret_lent=_slow_stop_arg(a))
    return r, Hu_map, prior


def _fit_posterior_mode(a, inp, fixed, th0, prior):
    """Bayesian route without REML. On a large model the REML fit alone takes
    hours: the posterior mode replaces it as starting point and whitening, and
    an evaluation at that point (maxiter = 0) gives Sigma, beta and the BLUPs
    at the mode."""
    from remlax.bayes import prepare, posterior_mode, available
    from remlax.fit import initial_theta
    if not available():
        raise _Refusal("la voie bayesienne exige blackjax (pip install blackjax)")
    if inp.lag is not None:
        raise _Refusal("la voie bayesienne n'accepte pas encore de decalage spatial")
    start = th0 if th0 is not None else initial_theta(inp.terms, inp.res, inp.y)
    th_map, Hu_map, info_map = posterior_mode(prepare(inp.terms, inp.res, inp.y, inp.X, prior),
                                               start, fixed_idx=fixed, floor=a.floor,
                                               ceil=a.ceil, verbose=not a.quiet)
    r = fit_reml(inp.terms, inp.res, inp.y, inp.X, theta_init=th_map, maxiter=0, polish=0,
                 floor=a.floor, ceil=a.ceil, verbose=False, hessian=False,
                 blups=not a.no_blups, pev=_pev_arg(a),
                 fixed_idx=fixed, optimizer=a.optimiseur, moteur=a.moteur,
                 inversion=a.inversion, memoire=a.memoire, finition=a.finition,
                 poids=inp.weights, arret_lent=_slow_stop_arg(a))
    r["depart_bayes"] = "map"
    r["mode_posterieur"] = info_map
    return r, Hu_map


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

def _inference(a, r, inp, exprs, fixed):
    """Adds to r the component names and table, the vpredict values, the
    Kenward-Roger adjustment, the predictions and the Wald tests requested."""
    from remlax.vpredict import vpredict
    from remlax.components import component_names
    terms, res, lag = inp.terms, inp.res, inp.lag
    r["composantes_noms"] = component_names(terms, res, lag=lag)
    if a.only_predict and len(r["theta"]) != len(r["composantes_noms"]) and not a.quiet:
        print("[remlax] mode predict seul : %d parametre(s) relus"
              % len(r["theta"]), flush=True)
    if not a.only_predict:
        _add_varcomp(a, r, inp, fixed)
    if exprs:
        free = np.ones(len(r["theta"]), dtype=bool)
        if fixed:
            free[np.array(fixed)] = False
        r["vpredict"] = vpredict(r["theta"], r.get("hessian"), terms, res, exprs, free=free,
                                 lag=lag, alias=r.get("alias"))
    if a.kenward_roger:
        from remlax.fixed_tests import kenward_roger
        t_w, y_w, X_w = _inference_model(inp)
        rk = kenward_roger(r["theta"], t_w, res, y_w, X_w,
                           termes_fixes=inp.bundle.fixed_groups())
        if rk.get("vbeta_kr") is not None:
            r["vbeta_kr"] = rk.pop("vbeta_kr")
        r["kenward_roger"] = rk
    if a.predict:
        _add_predictions(a, r, inp)
    if a.wald:
        _add_wald(r, inp)


def _inference_model(inp):
    """Terms, y and X for the tables that rebuild V (Wald, Kenward-Roger,
    predictions). With prior weights, the rows are scaled by sqrt(w): that is
    the model the solver fitted, whose V has the residual W^-1/2 R0 W^-1/2.
    Generalised least squares on it gives the weighted beta, its covariance and
    the BLUPs; the unweighted model would give other values."""
    if inp.weights is None:
        return inp.terms, inp.y, inp.X
    sw = np.sqrt(inp.weights)
    terms = [dict(tm, zx=np.asarray(tm["zx"], dtype=np.float64)
                  * sw[np.asarray(tm["zi"], dtype=np.int64)]) for tm in inp.terms]
    return terms, np.asarray(inp.y, dtype=np.float64) * sw, np.asarray(inp.X, dtype=np.float64) * sw[:, None]


def _add_varcomp(a, r, inp, fixed):
    from remlax.components import varcomp_table
    try:
        # The bounds of the fit, which the MME engine may have raised.
        r["varcomp"] = varcomp_table(r["theta"], r.get("hessian"), inp.terms, inp.res,
                                     floor=r.get("par_floor", a.floor),
                                     ceil=r.get("par_ceil", a.ceil), fixed_idx=fixed,
                                     lag=inp.lag, alias=r.get("alias"))
    except Exception as e:  # the table must never make a fit fail
        if not a.quiet:
            print("[remlax] table des composantes non calculee : %s" % e, flush=True)


def _add_predictions(a, r, inp):
    """Predictions L beta (+ M u): pred_L.bin, and pred_M_<term>.bin for the
    random terms in the classify."""
    from remlax.prediction import predict
    X = inp.X
    Lp = np.fromfile(os.path.join(a.bundle, "pred_L.bin"),
                     dtype=np.float64).reshape((-1, X.shape[1]), order="F")
    M = {}
    for t in inp.terms:
        fp = os.path.join(a.bundle, "pred_M_%s.bin" % t["name"])
        if os.path.exists(fp):
            M[t["name"]] = np.fromfile(fp, dtype=np.float64).reshape(
                (Lp.shape[0], t["t"] * t["q"]), order="F")
    t_w, y_w, X_w = _inference_model(inp)
    pr = predict(r["theta"], t_w, inp.res, y_w, X_w, Lp, M=M or None, vbeta=r.get("vbeta"))
    r["pred_cov"] = pr.pop("cov")
    r["predictions"] = pr


def _add_wald(r, inp):
    import jax.numpy as jnp
    from remlax.fixed_tests import wald
    from remlax.model import assemble_V, dense_Z
    t_w, y_w, X_w = _inference_model(inp)
    n = len(y_w)
    Zs = [dense_Z(t, n) for t in t_w]
    V = np.asarray(assemble_V(jnp.asarray(r["theta"]), t_w, Zs, inp.res, n))
    # With a spatial lag, the model linear in beta is that of y*.
    # y* is computed by the solver on the model it fitted (rows scaled when
    # there are weights).
    r["wald"] = wald(r.get("ystar", y_w), X_w, V, termes_fixes=inp.bundle.fixed_groups())


def _sample_bayes(a, r, inp, exprs, fixed, plat, Hu_map, prior):
    """Posterior sample of the variance parameters (--bayes), or None."""
    if not a.bayes or a.only_predict:
        return None
    from remlax.bayes import sample_posterior
    n_chains = a.chaines or (32 if plat == "gpu" else 8)
    if a.moteur == "mme" and not a.quiet:
        print("  [bayes] la loi a posteriori est evaluee par le moteur dense", flush=True)
    pev = (a.pev_termes.split(",") if a.pev_termes else
           ([t["name"] for t in inp.terms] if a.pev else ()))
    try:
        return sample_posterior(
            inp.terms, inp.res, inp.y, inp.X, r["theta"], H=r.get("hessian"), Hu=Hu_map,
            prior=prior, chains=n_chains, superchains=a.superchaines, warmup=a.chauffe,
            draws=a.tirages, seed=a.graine, target_accept=a.acceptation,
            max_leapfrog=a.sauts_max, fixed_idx=fixed, expressions=exprs,
            n_effets=a.bayes_effets, pev=pev, lag=inp.lag, verbose=not a.quiet)
    except (RuntimeError, ValueError) as e:
        raise _Refusal(str(e))


# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------

# Fields of r written as binary files, not into result.json.
_ARRAY_FIELDS = ("theta", "beta", "sigmas", "sigma_res", "sigmas_res",
                 "blups", "hessian", "Py", "vbeta", "vbeta_kr", "Vi",
                 "pred_cov", "loadings", "loadings_res", "pev", "lag", "ystar")

# Single arrays of r and their file names.
_SINGLE_ARRAYS = (("beta", "out_beta"), ("vbeta", "out_vbeta"),
                  ("vbeta_kr", "out_vbeta_kr"), ("pred_cov", "out_pred_cov"))


def _write_outputs(a, r, bayes, plat):
    """out_*.bin and result.json. Every array is written in binary with its
    dimensions in the JSON."""
    def wr(name, arr):
        np.asarray(arr, dtype=np.float64).ravel(order="F").tofile(
            os.path.join(a.bundle, name + ".bin"))

    out = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
           for k, v in r.items() if k not in _ARRAY_FIELDS}
    out["backend"] = plat
    out["devices"] = device_report()
    wr("out_theta", r["theta"])
    for key, name in _SINGLE_ARRAYS:
        if key in r:
            wr(name, r[key])
    out["sigma_dims"] = _write_named(wr, r.get("sigmas", {}), "out_sigma_")
    if "sigma_res" in r:
        wr("out_sigma_res", r["sigma_res"])
    out["sigma_res_dims"] = _write_named(wr, r.get("sigmas_res", {}), "out_sigmares_")
    out["loadings_dims"], out["psi_dims"] = _write_loadings(wr, r)
    out["pev_dims"] = _write_named(wr, r.get("pev") or {}, "out_pev_")
    out["blup_dims"] = _write_named(wr, r.get("blups", {}), "out_blup_")
    if "hessian" in r:
        wr("out_hessian", r["hessian"])
    if r.get("lag") is not None:
        out["lag"] = _write_lag(wr, r)
    if bayes is not None:
        out["bayes"] = _write_bayes(wr, bayes)
    chrono.mark("ecriture")
    out["chrono"] = chrono.stages_dict()
    # Cores actually used: the size of the affinity mask, after bounding.
    out["coeurs"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    out["fils_bornes"] = _FILS["message"]
    with open(os.path.join(a.bundle, "result.json"), "w") as f:
        json.dump(_json_safe(out), f, indent=2, default=float)


def _write_named(wr, arrays, prefix):
    """Writes <prefix><name>.bin for each named array; returns their dimensions."""
    dims = {}
    for nm, A in arrays.items():
        wr(prefix + nm, A)
        dims[nm] = list(np.shape(A))
    return dims


def _write_loadings(wr, r):
    """Loadings Lambda and specific variances psi of the fa/rr terms and of the
    residual sections (prefixed res_)."""
    loadings_dims, psi_dims = {}, {}
    for src, pref in (("loadings", ""), ("loadings_res", "res_")):
        for nm, lo in (r.get(src) or {}).items():
            wr("out_loadings_%s%s" % (pref, nm), lo["Lambda"])
            loadings_dims["%s%s" % (pref, nm)] = list(np.shape(lo["Lambda"]))
            if lo.get("psi") is not None:
                wr("out_psi_%s%s" % (pref, nm), lo["psi"])
                psi_dims["%s%s" % (pref, nm)] = [int(np.size(lo["psi"]))]
    return loadings_dims, psi_dims


def _write_lag(wr, r):
    """Spatial lag: Psi (q x q, units of W) and its standard errors, taken from
    the component table (delta method, covariance 2 H^-1), in binary; y*; the
    scalars go to the JSON."""
    Lr = r["lag"]
    q_ = Lr["Psi"].shape[0]
    se = np.full((q_, q_), np.nan)
    vc_se = {d["nom"]: d.get("se") for d in (r.get("varcomp") or [])}
    mi, mj = np.nonzero(Lr["mask"])
    for i_, j_ in zip(mi, mj):
        v_ = vc_se.get("Psi[%s<-%s]" % (Lr["traits"][i_], Lr["traits"][j_]))
        se[i_, j_] = np.nan if v_ is None else v_
    wr("out_lag_psi", Lr["Psi"])
    wr("out_lag_psi_se", se)
    wr("out_ystar", r["ystar"])
    out = {k: v for k, v in Lr.items() if k not in ("Psi", "mask")}
    out["q"] = int(q_)
    return out


def _write_bayes(wr, bayes):
    """Posterior draws of theta, of the components and of the expressions, and
    the posterior moments of beta and of the BLUPs."""
    th = bayes["theta"]
    C_, D_, p_ = th.shape
    wr("out_bayes_theta", th.reshape(C_ * D_, p_))
    wr("out_bayes_comp", bayes["composantes"])
    if bayes["expressions"].shape[1]:
        wr("out_bayes_vp", bayes["expressions"])
    jb = dict(resume=bayes["resume"], diagnostics=bayes["diagnostics"],
              prior=bayes["prior"], composantes_noms=bayes["composantes_noms"],
              expressions_noms=bayes["expressions_noms"],
              dims=dict(chaines=C_, tirages=D_, p=p_,
                        k=int(bayes["composantes"].shape[1]),
                        e=int(bayes["expressions"].shape[1])))
    ef = bayes["effets"]
    if ef is not None:
        wr("out_bayes_beta_moy", ef["beta_moyenne"])
        wr("out_bayes_beta_cov", ef["beta_cov"])
        jb["effets"] = dict(n=ef["n"], secondes=ef["secondes"], blup_dims={},
                            blup_sd=sorted(ef["blup_sd"]))
        for nm, U in ef["blup_moyenne"].items():
            wr("out_bayes_blup_moy_%s" % nm, U)
            jb["effets"]["blup_dims"][nm] = list(np.shape(U))
        for nm, U in ef["blup_sd"].items():
            wr("out_bayes_blup_sd_%s" % nm, U)
    return jb


def _json_safe(o):
    """NaN and infinities to null. JSON has no NaN: json.dump writes it anyway,
    and a conforming reader (jsonlite) then rejects the whole file, so one
    unavailable diagnostic would lose the fit. R reads null back as NA."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


if __name__ == "__main__":
    sys.exit(main())
