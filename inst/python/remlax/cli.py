"""Point d'entree du solveur : lit un dispositif serialise, ajuste, ecrit.

    python3 remlax_run.py cli <repertoire> [--backend auto|gpu|cpu] [--maxiter N]
                              [--polish N] [--no-hessian] [--no-blups] [--quiet]

Ecrit dans le meme repertoire : result.json (scalaires et dimensions) et des
binaires out_*.bin (theta, beta, Sigma, BLUPs) relus par R. Le format de sortie
est le miroir exact du format d'entree, pour que R n'ait besoin d'aucune
dependance Python et Python d'aucune dependance R.
"""
import time
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

# fin du demarrage de l'interpreteur, notee par remlax_run.py avant tout import lourd
_T_DEBUT = float(os.environ.get("RX_T_DEBUT", time.time()))

# FILS BORNES AUX COEURS ALLOUES, avant l'import de NumPy et de JAX qui
# dimensionnent leurs groupes de fils sur l'affinite du processus (fils.py).
from remlax.fils import borner_fils, _ETAT as _FILS  # noqa: E402
borner_fils()

import numpy as np  # noqa: E402

# MODE ECONOME ET MOTEUR PAR BLOCS : PLAFOND DE L'ALLOCATEUR. JAX limite par defaut sa reserve a 75 %
# de la memoire du GPU, meme sans preallocation. Sur une tranche MIG de 10 Go la
# limite tombe a 7,65 Go : l'auto-reglage de la compilation y monte a 6,1 Go et
# fragmente la reserve, puis les temporaires de l'evaluation (4,1 Go d'un seul
# tenant sur mvD7) ne trouvent plus de place (mesure le 2026-10-25). A 95 %, le
# meme calcul passe. La variable est lue a l'initialisation du backend, qui a
# lieu des l'import des modules ci-dessous : elle se pose donc ici. Une valeur
# fixee par l'appelant est respectee.
if "--memoire=econome" in sys.argv or "--moteur=blocs" in sys.argv or "--inversion=hybride" in sys.argv or any(
        (a, b) in (("--memoire", "econome"), ("--moteur", "blocs"), ("--inversion", "hybride"))
        for a, b in zip(sys.argv, sys.argv[1:])):
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from remlax import _x64  # noqa: F401,E402
from remlax import chrono  # noqa: E402
from remlax.bundle import Bundle  # noqa: E402
from remlax.device import pick_device, device_report, activer_cache_compilation, sans_greffons_gpu  # noqa: E402
from remlax.fit import fit_reml  # noqa: E402

# CHRONOMETRAGE PAR ETAPES (chrono.py). "demarrage" : du lancement par R
# (RX_T_LANCEMENT) a la premiere ligne de ce fichier ; "imports" : numpy, JAX
# et le solveur. En mode serveur, serveur.py remet le chrono a zero a chaque
# requete et ces deux etapes disparaissent : c'est tout son interet.
chrono.raz()
chrono.poser("demarrage", _T_DEBUT)
chrono.marque("imports")


def main(argv=None):
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
    a = ap.parse_args(argv)

    import jax
    # CPU SEUL quand le processus entier l'est (JAX_PLATFORMS=cpu, pose par R
    # pour backend = "cpu" en sous-processus) : pas de greffon CUDA. Le serveur
    # persistant, lui, sert CPU et GPU dans le meme processus et garde tout.
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        sans_greffons_gpu()
    activer_cache_compilation()
    dev, plat = pick_device(a.backend)
    # jax.default_device fixe le peripherique pour TOUTES les operations
    # suivantes : c'est le seul endroit ou le choix CPU/GPU intervient, et il
    # ne touche a rien d'autre que la machine.
    ctx = jax.default_device(dev)
    ctx.__enter__()
    if not a.quiet:
        print("[remlax] peripheriques : %s" % device_report(), flush=True)
        if _FILS["message"]:
            print(_FILS["message"], flush=True)
        print("[remlax] backend retenu : %s (%s)" % (plat, dev), flush=True)
    chrono.marque("peripherique")

    b = Bundle(a.bundle)
    terms, res, y, X = b.terms(), b.residual(), b.y, b.X
    chrono.marque("lecture")
    # poids a priori des observations (rx_model(weights = )) : fit.fit_reml
    poids = np.asarray(b.get("poids"), dtype=np.float64) if b.has("poids") else None
    if poids is not None and a.bayes:
        print("[remlax] ERREUR : la voie bayesienne ne prend pas encore les poids", file=sys.stderr)
        return 2
    lag = b.lag()
    if lag is not None:
        # Prepare une fois (decomposition spectrale de W, W longue) : le meme
        # objet sert a l'ajustement et aux tables d'inference.
        from remlax.lag import prepare_lag
        lag = prepare_lag(lag, y, X)
        if not a.quiet:
            print("[remlax] decalage spatial : q=%d, %d entree(s) libre(s) de Psi, critere %s, "
                  "chemin %s" % (lag["q"], int(lag["mask"].sum()), lag["criterion"],
                                 lag["chemin"]), flush=True)
        if a.kenward_roger or a.predict or a.only_predict:
            print("[remlax] ERREUR : Kenward-Roger et predict ne sont pas disponibles avec un "
                  "decalage spatial (rx_lag) : ils supposent y lineaire en beta et u a V "
                  "donnee, alors qu'ici seule y* = (I - M) y l'est.",
                  file=sys.stderr, flush=True)
            return 2
    if not a.quiet:
        print("[remlax] n=%d, %d effet(s) fixe(s), %d terme(s) aleatoire(s)"
              % (len(y), X.shape[1], len(terms)), flush=True)
        for t in terms:
            print("          %-14s %-5s t=%-3d q=%-5d %s"
                  % (t["name"], t["struct"], t["t"], t["q"],
                     "K=I" if t["LK"] is None else "K fournie"), flush=True)

    # Les expressions vpredict sont lues et verifiees AVANT l'ajustement : une
    # faute de nom ne doit pas couter une optimisation entiere.
    exprs = []
    if a.vpredict:
        for bloc in a.vpredict.split(";"):
            if not bloc.strip():
                continue
            nom, _, ex = bloc.partition("=")
            exprs.append((nom.strip(), ex.strip()))
        from remlax.inference import check_vpredict
        try:
            check_vpredict(exprs, terms, res, lag=lag)
        except ValueError as e:
            print("[remlax] ERREUR : %s" % e, file=sys.stderr, flush=True)
            return 2

    fixe = None
    if a.fixed_theta:
        fixe = [int(v) - 1 for v in a.fixed_theta.split(",") if v.strip()]
    if a.only_predict:
        # Predire ne demande PAS de reajuster. Sans ce mode, chaque appel a
        # rx_predict() relancerait l'optimisation complete pour retrouver le
        # theta qu'on a deja — et rien ne garantirait qu'il retombe sur le meme
        # optimum.
        r = dict(theta=np.fromfile(os.path.join(a.bundle, "in_theta.bin"),
                                   dtype=np.float64),
                 n_par=0, n_obs=len(y), logLik=float("nan"), secondes=0.0,
                 only_predict=True)
        a.predict = True
    else:
        th0 = None
        if a.theta_in:
            th0 = np.fromfile(os.path.join(a.bundle, "in_theta.bin"), dtype=np.float64)
    prior, Hu_map = None, None
    if a.bayes and a.prior:
        with open(os.path.join(a.bundle, a.prior)) as fp:
            prior = json.load(fp)
    if a.bayes and a.depart == "map" and not a.only_predict:
        # VOIE BAYESIENNE SANS REML. Sur un gros modele, l'ajustement REML seul
        # prend des heures : le mode a posteriori le remplace comme point de
        # depart et comme blanchiment, puis une simple EVALUATION en ce point
        # (maxiter = 0) fournit Sigma, beta et les BLUP au mode.
        from remlax.bayes import preparer, mode_posterieur, disponible
        from remlax.fit import initial_theta
        if not disponible():
            print("[remlax] ERREUR : la voie bayesienne exige blackjax (pip install blackjax)",
                  file=sys.stderr, flush=True)
            return 2
        if lag is not None:
            print("[remlax] ERREUR : la voie bayesienne n'accepte pas encore de decalage spatial",
                  file=sys.stderr, flush=True)
            return 2
        depart = th0 if th0 is not None else initial_theta(terms, res, y)
        th_map, Hu_map, info_map = mode_posterieur(preparer(terms, res, y, X, prior), depart,
                                                   fixed_idx=fixe, floor=a.floor, ceil=a.ceil,
                                                   verbose=not a.quiet)
        r = fit_reml(terms, res, y, X, theta_init=th_map, maxiter=0, polish=0,
                     floor=a.floor, ceil=a.ceil, verbose=False, hessian=False,
                     blups=not a.no_blups,
                     pev=(a.pev_termes.split(",") if a.pev_termes else a.pev),
                     fixed_idx=fixe, optimizer=a.optimiseur, moteur=a.moteur,
                     inversion=a.inversion, memoire=a.memoire, finition=a.finition, poids=poids,
                     arret_lent=(int(a.arret_lent.split(',')[0]), float(a.arret_lent.split(',')[1])))
        r["depart_bayes"] = "map"
        r["mode_posterieur"] = info_map
    elif not a.only_predict:
        r = fit_reml(terms, res, y, X, theta_init=th0,
                     maxiter=a.maxiter, polish=a.polish,
                     floor=a.floor, ceil=a.ceil, verbose=not a.quiet,
                     hessian=not a.no_hessian, blups=not a.no_blups,
                     pev=(a.pev_termes.split(",") if a.pev_termes else a.pev),
                     n_restarts=a.restarts, restart_sd=a.restart_sd, fixed_idx=fixe,
                     lag=lag, optimizer=a.optimiseur, moteur=a.moteur,
                     inversion=a.inversion, memoire=a.memoire, finition=a.finition, poids=poids,
                     arret_lent=(int(a.arret_lent.split(',')[0]), float(a.arret_lent.split(',')[1])))

    # --- inference ------------------------------------------------------------
    from remlax.inference import vpredict as _vp, wald as _wald, component_names
    r["composantes_noms"] = component_names(terms, res, lag=lag)
    if a.only_predict and len(r["theta"]) != len(r["composantes_noms"]) and not a.quiet:
        print("[remlax] mode predict seul : %d parametre(s) relus"
              % len(r["theta"]), flush=True)
    if not a.only_predict:
        from remlax.inference import varcomp_table
        try:
            r["varcomp"] = varcomp_table(r["theta"], r.get("hessian"), terms, res,
                                         floor=a.floor, ceil=a.ceil, fixed_idx=fixe,
                                         lag=lag, alias=r.get("alias"))
        except Exception as e:  # la table ne doit jamais faire echouer un ajustement
            if not a.quiet:
                print("[remlax] table des composantes non calculee : %s" % e, flush=True)
    if exprs:
        libre = np.ones(len(r["theta"]), dtype=bool)
        if fixe:
            libre[np.array(fixe)] = False
        r["vpredict"] = _vp(r["theta"], r.get("hessian"), terms, res, exprs, free=libre,
                            lag=lag, alias=r.get("alias"))
    if a.kenward_roger:
        from remlax.inference import kenward_roger as _kr
        rk = _kr(r["theta"], terms, res, y, X, termes_fixes=b.fixed_groups())
        if rk.get("vbeta_kr") is not None:
            r["vbeta_kr"] = rk.pop("vbeta_kr")
        r["kenward_roger"] = rk
    if a.predict:
        from remlax.inference import predict as _pred
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
        from remlax.model import assemble_V as _aV, dense_Z as _dZ
        Zs = [_dZ(t, len(y)) for t in terms]
        V = np.asarray(_aV(_jnp.asarray(r["theta"]), terms, Zs, res, len(y)))
        # Avec un decalage, le modele lineaire en beta est celui de y*.
        r["wald"] = _wald(r.get("ystar", y), X, V, termes_fixes=b.fixed_groups())

    bayes = None
    if a.bayes and not a.only_predict:
        from remlax.bayes import sample_posterior
        nch = a.chaines or (32 if plat == "gpu" else 8)
        if a.moteur == "mme" and not a.quiet:
            print("  [bayes] la loi a posteriori est evaluee par le moteur dense", flush=True)
        try:
            bayes = sample_posterior(
                terms, res, y, X, r["theta"], H=r.get("hessian"), Hu=Hu_map, prior=prior,
                chains=nch, superchains=a.superchaines, warmup=a.chauffe, draws=a.tirages,
                seed=a.graine, target_accept=a.acceptation, max_leapfrog=a.sauts_max,
                fixed_idx=fixe, expressions=exprs, n_effets=a.bayes_effets,
                pev=(a.pev_termes.split(",") if a.pev_termes else
                     ([t["name"] for t in terms] if a.pev else ())),
                lag=lag, verbose=not a.quiet)
        except (RuntimeError, ValueError) as e:
            print("[remlax] ERREUR : %s" % e, file=sys.stderr, flush=True)
            return 2

    chrono.marque("inference")
    out = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
           for k, v in r.items()
           if k not in ("theta", "beta", "sigmas", "sigma_res", "sigmas_res",
                        "blups", "hessian", "Py", "vbeta", "vbeta_kr", "Vi",
                        "pred_cov", "loadings", "loadings_res", "pev", "lag", "ystar")}
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
    # LOADINGS ET PEV. Ecrits en binaire comme les autres tableaux, avec leurs
    # dimensions dans le JSON : le format de sortie est le miroir du format
    # d'entree, pour que R n'ait aucune dependance Python.
    out["loadings_dims"], out["psi_dims"] = {}, {}
    for src, pref in (("loadings", ""), ("loadings_res", "res_")):
        for nm, lo in (r.get(src) or {}).items():
            wr("out_loadings_%s%s" % (pref, nm), lo["Lambda"])
            out["loadings_dims"]["%s%s" % (pref, nm)] = list(np.shape(lo["Lambda"]))
            if lo.get("psi") is not None:
                wr("out_psi_%s%s" % (pref, nm), lo["psi"])
                out["psi_dims"]["%s%s" % (pref, nm)] = [int(np.size(lo["psi"]))]
    out["pev_dims"] = {}
    for nm, Pv in (r.get("pev") or {}).items():
        wr("out_pev_%s" % nm, Pv); out["pev_dims"][nm] = list(np.shape(Pv))
    out["blup_dims"] = {}
    for nm, U in r.get("blups", {}).items():
        wr("out_blup_%s" % nm, U); out["blup_dims"][nm] = list(np.shape(U))
    if "hessian" in r: wr("out_hessian", r["hessian"])
    # DECALAGE SPATIAL. Psi (q x q, unites de W) et ses erreurs-types (celles de
    # la table des composantes : methode delta, covariance 2 H^-1), en binaire ;
    # les scalaires dans le JSON.
    if r.get("lag") is not None:
        Lr = r["lag"]
        q_ = Lr["Psi"].shape[0]
        se = np.full((q_, q_), np.nan)
        noms_vc = {d["nom"]: d.get("se") for d in (r.get("varcomp") or [])}
        mi, mj = np.nonzero(Lr["mask"])
        for i_, j_ in zip(mi, mj):
            v_ = noms_vc.get("Psi[%s<-%s]" % (Lr["traits"][i_], Lr["traits"][j_]))
            se[i_, j_] = np.nan if v_ is None else v_
        wr("out_lag_psi", Lr["Psi"])
        wr("out_lag_psi_se", se)
        wr("out_ystar", r["ystar"])
        out["lag"] = {k: v for k, v in Lr.items() if k not in ("Psi", "mask")}
        out["lag"]["q"] = int(q_)

    if bayes is not None:
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
        out["bayes"] = jb

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

    chrono.marque("ecriture")
    out["chrono"] = chrono.dict_etapes()
    # coeurs reellement utilises : la taille du masque d'affinite, apres bornage
    out["coeurs"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    out["fils_bornes"] = _FILS["message"]
    with open(os.path.join(a.bundle, "result.json"), "w") as f:
        json.dump(propre(out), f, indent=2, default=float)
    if not a.quiet and not a.only_predict:
        print("[remlax] logLik = %.9f | %d parametres | %.1f s"
              % (r["logLik"], r["n_par"], r["secondes"]), flush=True)
    ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
