#!/usr/bin/env python3
"""Benchmarks de remlax : temps, compilation, memoire, par backend.

    python3 benchmarks/bench.py --backend cpu --suite all --out results/cpu.csv

AUCUNE dependance a l'hote : ce script tourne tel quel sur un noeud de calcul,
dans un conteneur, sans reseau. Il n'ecrit qu'un CSV et une ligne par mesure.

CE QUI EST MESURE, ET POURQUOI CE DECOUPAGE
-------------------------------------------
V est formee DENSE (n x n, float64) et factorisee : le cout est en O(n^3) par
evaluation, et l'empreinte en O(n^2). Les quatre suites separent les axes qui
decident du choix CPU/GPU :

  scaling      n croissant, modele fixe. L'axe dominant.
  params       nombre de parametres de variance croissant, n fixe. Le nombre
               d'iterations de L-BFGS-B en depend, pas le cout d'une evaluation.
  structures   une structure du catalogue par ligne, taille fixe. Isole le cout
               de CONSTRUCTION de V (noyaux, quadrature de Bessel) du cout de sa
               factorisation.
  compile      premier appel de l'objectif (compilation XLA incluse) contre les
               appels suivants. Sur GPU la compilation peut dominer largement le
               calcul pour de petits modeles : c'est la mesure qui dit a un
               utilisateur si le GPU vaut le detour.

PROTOCOLE. Graine fixee par cas. Un warm-up hors mesure, puis --reps repetitions
dont on rapporte la mediane et l'etendue. Le temps rapporte par `fit_*` est le
temps mur d'un ajustement COMPLET (initialisation, L-BFGS-B, polissage de
Newton, Hessien si demande), c'est-a-dire ce que paie l'utilisateur.
"""
import argparse
import csv
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


# ==============================================================================
# Dispositifs
# ==============================================================================
def design_iid(n, q=None, seed=0):
    """Un facteur aleatoire + residuelle iid : le modele le plus simple qui
    exerce tout le chemin (assemblage de V, Cholesky, gradient, polissage)."""
    q = q or max(5, n // 5)
    rng = np.random.default_rng(seed)
    lev = rng.integers(0, q, n)
    u = rng.normal(size=q)
    y = 2.0 + u[lev] + rng.normal(size=n, scale=0.8)
    term = dict(name="g", struct="iid", t=1, rank=0, q=q,
                zi=np.arange(n), zj=lev.astype(np.int64), zx=np.ones(n), LK=None)
    res = dict(struct="iid", t=1, rank=0,
               trait=np.zeros(n, np.int64), unit=np.arange(n))
    return [term], res, y, np.ones((n, 1))


def design_traits(n_unit, t, seed=0, struct="us"):
    """t caracteres en format long : n = n_unit * t observations, Sigma t x t
    libre sur le terme genetique et residuelle diagonale entre caracteres."""
    q = max(6, n_unit // 4)
    rng = np.random.default_rng(seed)
    n = n_unit * t
    unit = np.tile(np.arange(n_unit), t)
    trait = np.repeat(np.arange(t), n_unit)
    lev = rng.integers(0, q, n_unit)
    lev_long = np.tile(lev, t)
    zi = np.arange(n)
    zj = trait * q + lev_long                     # NIVEAU le plus rapide
    U = rng.normal(size=(q, t))
    y = 1.0 + U[lev_long, trait] + rng.normal(size=n, scale=0.7)
    term = dict(name="g", struct=struct, t=t, rank=(2 if struct == "fa" else 0),
                q=q, zi=zi, zj=zj.astype(np.int64), zx=np.ones(n), LK=None)
    res = dict(struct="diag", t=t, rank=0, trait=trait.astype(np.int64), unit=unit)
    X = np.zeros((n, t))
    X[np.arange(n), trait] = 1.0                  # une moyenne par caractere
    return [term], res, y, X


def design_genomic(n, q, t=1, seed=0, n_marq=None, ige=False):
    """Parente GENOMIQUE dense : le cas qui domine reellement le temps de calcul.

    POURQUOI CETTE SUITE EXISTE. Avec K = I, term_factor prend la branche
    `LK is None` et ne forme AUCUN produit : (L_Sigma (x) I) agit bloc par bloc.
    Un balayage a K = I mesure donc l'assemblage de V le moins cher possible, et
    sous-estime le cout du cas qui interesse un selectionneur. Avec une K dense,
    une evaluation paie en plus

        ZL = Z (I (x) L_K)      n t q^2
        B  = ZL (L_Sigma (x) I) n t m q
        V += B B'               n^2 m q

    A n = q = 2000 cela fait environ 16 Gflop contre 2,7 Gflop pour la Cholesky :
    l'assemblage domine. Et ce sont des produits matriciels denses, la ou un GPU
    est le plus favorise — d'ou l'interet de mesurer cet axe separement.

    ige=True donne la forme du modele a effets genetiques directs et indirects :
    UN terme a t = 2 dont la covariance 2x2 est libre, la premiere colonne de
    traits portant l'incidence directe et la seconde une incidence de voisinage
    PONDEREE. C'est ainsi qu'une covariance entre deux termes genetiques
    s'exprime ici, et cela multiplie par t^2 le cout du produit ci-dessus.
    """
    rng = np.random.default_rng(seed)
    n_marq = n_marq or max(200, q // 2)
    # K = W W' / m sur des marqueurs bialleliques, centree-reduite, plus une
    # ride minuscule : c'est la construction de VanRaden, celle qu'un
    # selectionneur fournit reellement, et elle est PLEINE.
    p = rng.uniform(0.05, 0.95, n_marq)
    W = rng.binomial(2, p, size=(q, n_marq)).astype(np.float64)
    W -= 2.0 * p
    W /= np.sqrt(2.0 * np.sum(p * (1.0 - p)))
    K = W @ W.T
    K[np.diag_indices(q)] += 1e-6 * np.trace(K) / q
    LK = np.linalg.cholesky(K)

    lev = rng.integers(0, q, n)                      # replications par genotype
    struct = "us" if t > 1 else "iid"
    if ige and t == 2:
        # trait 0 : effet direct du genotype de la plante
        # trait 1 : effet indirect, somme ponderee des voisins
        zi = np.concatenate([np.arange(n), np.arange(n)])
        vois = rng.integers(0, q, n)
        zj = np.concatenate([lev, q + vois]).astype(np.int64)
        w = rng.uniform(0.3, 1.0, n)                 # poids de voisinage
        zx = np.concatenate([np.ones(n), w])
    else:
        zi = np.concatenate([np.arange(n)] * t)
        zj = np.concatenate([k * q + lev for k in range(t)]).astype(np.int64)
        zx = np.ones(n * t)
    U = rng.normal(size=(q, t))
    y = 1.0 + U[lev, 0] + rng.normal(size=n, scale=0.8)
    term = dict(name="g", struct=struct, t=t, rank=0, q=q,
                zi=zi, zj=zj, zx=zx, LK=LK)
    res = dict(struct="iid", t=1, rank=0,
               trait=np.zeros(n, np.int64), unit=np.arange(n))
    return [term], res, y, np.ones((n, 1))


def design_level(n_row, n_col, kind, seed=0):
    """Champ structure entre NIVEAUX sur une grille n_row x n_col : ar1, produit
    separable, ou noyau metrique a coordonnees. Un niveau par cellule."""
    rng = np.random.default_rng(seed)
    q = n_row * n_col
    n = q
    rr, cc = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing="ij")
    coord = np.column_stack([rr.ravel().astype(float), cc.ravel().astype(float)])
    y = rng.normal(size=n)
    term = dict(name="f", struct="iid", t=1, rank=0, q=q,
                zi=np.arange(n), zj=np.arange(n), zx=np.ones(n), LK=None)
    term["lvl"] = kind
    if kind == "ar1ar1":
        term["dims"] = (n_row, n_col)
    elif kind == "ar1":
        pass                                       # 1D sur l'ordre des niveaux
    if kind in ("iexp", "igau", "ieuc", "sph", "cir", "aexp", "agau", "mtrn", "lvr"):
        term["coord"] = coord
    if kind == "exp" or kind == "gau":
        term["coord"] = coord[:, :1]
    if kind == "mtrn":
        term["lvl_opts"] = {"est_phi": 1.0, "est_nu": 1.0}
    if kind == "mtrn_aniso":
        term["lvl"] = "mtrn"
        term["coord"] = coord
        term["lvl_opts"] = {"est_phi": 1.0, "est_nu": 1.0,
                            "est_delta": 1.0, "est_alpha": 1.0}
    res = dict(struct="iid", t=1, rank=0,
               trait=np.zeros(n, np.int64), unit=np.arange(n))
    return [term], res, y, np.ones((n, 1))


# ==============================================================================
# Mesure
# ==============================================================================
def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def gpu_mem_mb():
    """Pic d'occupation du peripherique, quand JAX l'expose."""
    try:
        import jax
        st = jax.local_devices()[0].memory_stats() or {}
        peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
        return (peak / 1e6) if peak else float("nan")
    except Exception:
        return float("nan")


def timed_fit(terms, res, y, X, reps, hessian, **kw):
    """Ajustements repetes ; rend la mediane, le min, le max et le dernier resultat."""
    from remlax.fit import fit_reml
    ts = []
    out = None
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        out = fit_reml(terms, res, y, X, verbose=False, hessian=hessian,
                       blups=False, **kw)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts), out


def timed_objective(terms, res, y, X, reps):
    """Separe compilation et execution.

    make_objective rend une fonction jittee ; son PREMIER appel paie la trace et
    la compilation XLA, les suivants non. La difference est exactement ce qui
    fait qu'un petit modele est plus rapide sur CPU que sur GPU.
    """
    import jax
    from remlax import structures as S
    from remlax.model import make_objective, n_theta
    from remlax.fit import initial_theta
    fun_sc, fun, sc, Zs = make_objective(terms, res, y, X)
    th = np.asarray(initial_theta(terms, res, y))
    t0 = time.perf_counter()
    v, g = fun(th)
    jax.block_until_ready(v)
    t_first = time.perf_counter() - t0
    ts = []
    for k in range(reps):
        thk = th + 1e-3 * (k + 1)
        t0 = time.perf_counter()
        v, g = fun(thk)
        jax.block_until_ready(v)
        ts.append(time.perf_counter() - t0)
    return t_first, statistics.median(ts), float(v)


# ==============================================================================
# Suites
# ==============================================================================
def suite_scaling(rows, args, meta):
    ns = [int(v) for v in args.ns.split(",")] if args.ns else \
         [250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    ns = [n for n in ns if n <= args.nmax]
    for n in ns:
        terms, res, y, X = design_iid(n, seed=1)
        # Un ajustement complet coute O(n^3) par evaluation et plusieurs dizaines
        # d'evaluations. Repeter au-dela de n = 4000 achete une etendue au prix
        # d'heures de calcul : on garde les repetitions ou elles sont informatives.
        reps = args.reps if n <= args.reps_upto else 1
        try:
            med, lo, hi, out = timed_fit(terms, res, y, X, reps, hessian=False)
        except Exception as e:                      # OOM : on note la limite
            rows.append(dict(meta, suite="scaling", case="iid", n=n, n_par=2,
                             status="failed", error=type(e).__name__ + ": " + str(e)[:200]))
            print("[scaling] n=%d ECHEC %s" % (n, type(e).__name__), flush=True)
            break
        rows.append(dict(meta, suite="scaling", case="iid", n=n, q=terms[0]["q"],
                         n_par=int(out["n_par"]), status="ok",
                         fit_s_median=med, fit_s_min=lo, fit_s_max=hi,
                         # SANS n_iter, un ecart de temps d'ajustement entre deux
                         # backends est inexploitable : on ne sait pas s'il vient
                         # du cout d'une evaluation ou de leur nombre.
                         n_iter=int(out.get("n_iter") or -1),
                         logLik=float(out["logLik"]),
                         newton_decrement=float(out.get("newton_decrement") or float("nan")),
                         rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
        print("[scaling] n=%-6d %8.2f s  logLik %.6f" % (n, med, out["logLik"]), flush=True)


def suite_compile(rows, args, meta):
    ns = [int(v) for v in args.ns.split(",")] if args.ns else \
         [250, 500, 1000, 2000, 4000, 8000, 12000, 16000]
    ns = [n for n in ns if n <= args.nmax]
    for n in ns:
        terms, res, y, X = design_iid(n, seed=2)
        try:
            t_first, t_run, v = timed_objective(terms, res, y, X, args.reps)
        except Exception as e:
            rows.append(dict(meta, suite="compile", case="iid", n=n,
                             status="failed", error=type(e).__name__ + ": " + str(e)[:200]))
            break
        rows.append(dict(meta, suite="compile", case="iid", n=n, status="ok",
                         first_call_s=t_first, run_call_s=t_run,
                         compile_s=max(t_first - t_run, 0.0),
                         compile_share=(t_first - t_run) / t_first if t_first > 0 else float("nan"),
                         neg2logl=v, rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
        print("[compile] n=%-6d premier %7.3f s | execution %7.4f s | part compilation %5.1f %%"
              % (n, t_first, t_run, 100 * (t_first - t_run) / max(t_first, 1e-12)), flush=True)


def suite_params(rows, args, meta):
    n_unit = args.n_unit
    for t in [int(v) for v in (args.ts or "1,2,3,4,5,6,8").split(",")]:
        for struct in ("us", "fa") if t >= 3 else ("us",):
            terms, res, y, X = design_traits(n_unit, t, seed=3, struct=struct)
            n = len(y)
            if n > args.nmax:
                continue
            try:
                med, lo, hi, out = timed_fit(terms, res, y, X, args.reps, hessian=True)
            except Exception as e:
                rows.append(dict(meta, suite="params", case="%s(t=%d)" % (struct, t),
                                 n=n, status="failed",
                                 error=type(e).__name__ + ": " + str(e)[:200]))
                continue
            rows.append(dict(meta, suite="params", case="%s(t=%d)" % (struct, t),
                             n=n, n_traits=t, n_par=int(out["n_par"]), status="ok",
                             fit_s_median=med, fit_s_min=lo, fit_s_max=hi,
                             n_iter=int(out.get("n_iter") or -1),
                             logLik=float(out["logLik"]),
                             rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
            print("[params] %-8s t=%d n=%-6d p=%-3d %8.2f s"
                  % (struct, t, n, out["n_par"], med), flush=True)


def suite_structures(rows, args, meta):
    nr, nc = args.grid_rows, args.grid_cols
    cas = [("iid", None), ("us(3)", None), ("fa(4,2)", None),
           ("ar1", "ar1"), ("ar1ar1", "ar1ar1"),
           ("iexp", "iexp"), ("igau", "igau"), ("sph", "sph"),
           ("mtrn", "mtrn"), ("mtrn_aniso", "mtrn_aniso")]
    for nom, kind in cas:
        if kind is None:
            if nom == "iid":
                terms, res, y, X = design_iid(nr * nc, seed=4)
            elif nom == "us(3)":
                terms, res, y, X = design_traits((nr * nc) // 3, 3, seed=4, struct="us")
            else:
                terms, res, y, X = design_traits((nr * nc) // 4, 4, seed=4, struct="fa")
        else:
            terms, res, y, X = design_level(nr, nc, kind, seed=4)
        n = len(y)
        try:
            t_first, t_run, v = timed_objective(terms, res, y, X, args.reps)
            med, lo, hi, out = timed_fit(terms, res, y, X, 1, hessian=False)
            ok, err = "ok", ""
        except Exception as e:
            t_first = t_run = v = med = float("nan")
            out = {"n_par": -1, "logLik": float("nan")}
            ok, err = "failed", type(e).__name__ + ": " + str(e)[:200]
        rows.append(dict(meta, suite="structures", case=nom, n=n, status=ok, error=err,
                         first_call_s=t_first, run_call_s=t_run,
                         compile_s=(t_first - t_run) if t_first == t_first else float("nan"),
                         fit_s_median=med, n_par=int(out["n_par"]),
                         logLik=float(out["logLik"]),
                         rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
        print("[struct] %-12s n=%-5d compilation %8.3f s | execution %8.4f s | ajustement %8.2f s %s"
              % (nom, n, t_first - t_run if t_first == t_first else float("nan"),
                 t_run, med, err), flush=True)


def suite_genomic(rows, args, meta):
    """Parente dense : q croissant a n fixe, puis n croissant a q/n fixe."""
    qs = [int(v) for v in (args.qs or "250,500,1000,1500,2000,3000,4000").split(",")]
    for q in qs:
        n = args.n_gen
        if q > n:
            continue
        terms, res, y, X = design_genomic(n, q, t=1, seed=7)
        try:
            t_first, t_run, v = timed_objective(terms, res, y, X, args.reps)
            med, lo, hi, out = timed_fit(terms, res, y, X, 1, hessian=False)
            ok, err, ni = "ok", "", int(out.get("n_iter") or -1)
        except Exception as e:
            t_first = t_run = med = float("nan")
            out = {"n_par": -1, "logLik": float("nan")}
            ok, err, ni = "failed", type(e).__name__ + ": " + str(e)[:200], -1
        # Gflop d'assemblage attendus par evaluation, pour rapporter le temps
        # a la quantite d'arithmetique et non a la seule taille.
        gflop = (n * q * q + n * n * q) / 1e9
        rows.append(dict(meta, suite="genomic", case="iid+K(q=%d)" % q, n=n, q=q,
                         status=ok, error=err, first_call_s=t_first, run_call_s=t_run,
                         compile_s=(t_first - t_run) if t_first == t_first else float("nan"),
                         fit_s_median=med, n_iter=ni, n_par=int(out["n_par"]),
                         logLik=float(out["logLik"]), assembly_gflop=gflop,
                         rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
        print("[genomic] q=%-5d n=%-5d assemblage %7.1f Gflop | evaluation %8.4f s "
              "| ajustement %8.2f s %s" % (q, n, gflop, t_run, med, err), flush=True)


def suite_crossterm(rows, args, meta):
    """Covariance entre termes genetiques, avec parente dense.

    t = 1 : un seul terme genetique
    t = 2 : deux termes correles, forme du modele direct/indirect (incidence de
            voisinage ponderee sur le second)
    t = 3 : trois termes correles, covariance 3x3 libre
    """
    n, q = args.n_gen, args.q_gen
    for t in [int(v) for v in (args.ts or "1,2,3").split(",")]:
        for ige in ([False, True] if t == 2 else [False]):
            terms, res, y, X = design_genomic(n, q, t=t, seed=8, ige=ige)
            nom = "us(%d)+K%s" % (t, " [DGE/IGE]" if ige else "")
            try:
                t_first, t_run, v = timed_objective(terms, res, y, X, args.reps)
                med, lo, hi, out = timed_fit(terms, res, y, X, 1, hessian=True)
                ok, err, ni = "ok", "", int(out.get("n_iter") or -1)
            except Exception as e:
                t_first = t_run = med = float("nan")
                out = {"n_par": -1, "logLik": float("nan")}
                ok, err, ni = "failed", type(e).__name__ + ": " + str(e)[:200], -1
            gflop = (n * t * q * q + n * n * t * q) / 1e9
            rows.append(dict(meta, suite="crossterm", case=nom, n=n, q=q, n_traits=t,
                             status=ok, error=err, first_call_s=t_first, run_call_s=t_run,
                             compile_s=(t_first - t_run) if t_first == t_first else float("nan"),
                             fit_s_median=med, n_iter=ni, n_par=int(out["n_par"]),
                             logLik=float(out["logLik"]), assembly_gflop=gflop,
                             rss_mb=rss_mb(), gpu_mb=gpu_mem_mb()))
            print("[crossterm] %-18s p=%-3d assemblage %7.1f Gflop | evaluation %8.4f s "
                  "| ajustement %8.2f s %s"
                  % (nom, out["n_par"], gflop, t_run, med, err), flush=True)


SUITES = {"scaling": suite_scaling, "compile": suite_compile,
          "params": suite_params, "structures": suite_structures,
          "genomic": suite_genomic, "crossterm": suite_crossterm}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--suite", default="all",
                    help="all, ou une liste : scaling,compile,params,structures")
    ap.add_argument("--out", required=True, help="chemin du CSV de sortie")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reps-upto", dest="reps_upto", type=int, default=4000,
                    help="au-dela de ce n, une seule repetition (cout en n^3)")
    ap.add_argument("--nmax", type=int, default=16000,
                    help="borne superieure sur n (memoire du peripherique)")
    ap.add_argument("--ns", default=None, help="liste explicite de n")
    ap.add_argument("--ts", default=None, help="liste de nombres de caracteres")
    ap.add_argument("--qs", default=None, help="liste de tailles de parente (suite genomic)")
    ap.add_argument("--n-gen", dest="n_gen", type=int, default=4000,
                    help="n fixe des suites genomic et crossterm")
    ap.add_argument("--q-gen", dest="q_gen", type=int, default=2000,
                    help="q fixe de la suite crossterm")
    ap.add_argument("--n-unit", dest="n_unit", type=int, default=600)
    ap.add_argument("--grid-rows", dest="grid_rows", type=int, default=40)
    ap.add_argument("--grid-cols", dest="grid_cols", type=int, default=30)
    ap.add_argument("--tag", default="", help="etiquette libre (ex. a100-mig-1g10gb)")
    a = ap.parse_args(argv)

    import jax
    from remlax.device import pick_device, device_report
    dev, plat = pick_device(a.backend)
    ctx = jax.default_device(dev)
    ctx.__enter__()

    meta = dict(backend=plat, device=str(dev), device_kind=getattr(dev, "device_kind", ""),
                tag=a.tag, jax=jax.__version__, numpy=np.__version__,
                python=platform.python_version(), machine=platform.machine(),
                n_cpu=os.cpu_count(),
                xla_flags=os.environ.get("XLA_FLAGS", ""),
                omp=os.environ.get("OMP_NUM_THREADS", ""),
                # os.cpu_count() rend les coeurs de la MACHINE, pas ceux que le
                # cgroup alloue : sur un noeud de 192 coeurs alloue a 8, JAX
                # dimensionnerait son pool sur 192 et sur-souscrirait d'un facteur
                # 24. Les deux comptes sont donc journalises separement, et le
                # debit obtenu (mesure) dit lequel a servi.
                n_cpu_affinity=(len(os.sched_getaffinity(0))
                                if hasattr(os, "sched_getaffinity") else -1),
                mkl=os.environ.get("MKL_NUM_THREADS", ""),
                openblas=os.environ.get("OPENBLAS_NUM_THREADS", ""),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("[bench] %s" % json.dumps(meta), flush=True)
    print("[bench] peripheriques : %s" % device_report(), flush=True)

    # Warm-up hors mesure : la premiere compilation du processus paie aussi
    # l'initialisation du backend, qui n'a rien a voir avec le modele.
    w = design_iid(200, seed=99)
    timed_fit(*w, reps=1, hessian=False)

    rows = []
    noms = list(SUITES) if a.suite == "all" else a.suite.split(",")
    for nm in noms:
        if nm not in SUITES:
            raise SystemExit("suite inconnue : %s (au choix : %s)" % (nm, ", ".join(SUITES)))
        print("\n=== suite %s ===" % nm, flush=True)
        SUITES[nm](rows, a, meta)

    champs = []
    for r in rows:
        for k in r:
            if k not in champs:
                champs.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n[bench] %d mesures ecrites dans %s" % (len(rows), a.out), flush=True)
    ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
