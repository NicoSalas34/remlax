"""Ou le GPU et le CPU different-ils, et pourquoi : le banc apparie.

POURQUOI CE FICHIER REMPLACE LA GRILLE PRECEDENTE
=================================================
La grille comparait des TEMPS D'AJUSTEMENT entre backends. C'est une mesure
biaisee, et un relecteur le verrait : les deux backends ne suivent pas la meme
trajectoire d'optimisation. Mesure sur une meme cellule, 419 iterations sur
carte contre 317 sur CPU, pour le meme optimum a 8,6e-10 pres en relatif. La
cause est que l'addition flottante n'est pas associative — les reductions se
font dans un ordre different — donc le gradient differe au dernier bit, la
recherche lineaire accepte un pas different, et les chemins divergent. Le
nombre d'iterations est decide en fin de course par le bruit d'arrondi.
Comparer deux temps totaux revient donc a comparer un cout materiel MULTIPLIE
par une longueur de chemin aleatoire.

L'UNITE DE MESURE PROPRE : L'EVALUATION A THETA IMPOSE
======================================================
Une evaluation de la vraisemblance au MEME theta fait exactement le meme
travail arithmetique des deux cotes. Pas d'optimiseur, pas de trajectoire : le
rapport est purement materiel, deterministe et repetable. Le theta commun est
obtenu UNE FOIS par un ajustement de reference, puis impose aux deux backends.

LA DECOMPOSITION EN TROIS PIECES, MESUREES SEPAREMENT
=====================================================
Plutot qu'une regression globale, on mesure trois quantites dont le produit
doit reconstruire le temps total. Chacune est verifiable seule, et la
reconstruction est un test que ce banc peut PERDRE.

  1. compilation   payee une fois, propriete du peripherique (XLA compile pour
                   la cible) ; plus chere sur GPU
  2. evaluation    regime etabli ; depend de la structure mais PAS du nombre de
                   parametres libres, la Cholesky de V dominant et ignorant p
  3. evaluations   n_iter + 2p : le Hessien par differences finies coute
                   exactement 2p gradients, donc cette part est PREDICTIBLE

HYPOTHESE TESTEE, ET C'EST LE RESULTAT ATTENDU
==============================================
Le GPU paie une compilation plus chere et gagne sur chaque evaluation. A petit
p il y a peu d'evaluations, la compilation domine, le GPU perd. A grand p le
nombre d'evaluations croit — par les iterations ET par les 2p du Hessien — la
compilation s'amortit, le GPU gagne. Le croisement CPU/GPU se produit donc en
p, et pas seulement en n. C'est refutable.

CE QUI REND LE DISPOSITIF DEFENDABLE
====================================
  - apparie      memes donnees, meme graine, meme theta ; seul le backend change
  - coeurs fixes 16 partout, puisqu'ils pilotent le temps de compilation
  - carte entiere assertee par le script appelant, qui sort en erreur sur une
                 tranche : une tranche donne des chiffres plausibles qui sont
                 silencieusement une borne pessimiste
  - repetitions  mediane et etendue, les temps variant d'une repetition a l'autre
  - cellule vide DECLAREE : le moteur creux n'a pas de voie GPU, CHOLMOD etant
                 CPU. La grille a six cellules et non huit, et cette absence est
                 un resultat : elle rend 'creux' et 'GPU' exclusifs.

COMMENT p EST FAIT VARIER, ET LE BIAIS QUI SUBSISTE
===================================================
A DONNEES STRICTEMENT IDENTIQUES : memes n, q, t, meme Z, meme y, meme graine.
Seule la STRUCTURE de covariance sur le terme genetique change, ce qui est
aussi ce que fait un utilisateur reel. A t caracteres :

    iid   1 parametre        us    t(t+1)/2 parametres
    diag  t parametres       fa(t,k)  entre les deux

Le biais qui subsiste, et qu'il faut ecrire plutot que cacher : changer la
structure change aussi le MOTIF de sparsite de G^-1 et la geometrie de la
surface. p n'est pas une dimension physique du probleme mais une propriete de
la structure, donc aucune manette ne le tourne seul. Deux garde-fous :
  - le cout par evaluation est rapporte separement du nombre d'evaluations,
    ce qui distingue un effet de motif (cout unitaire) d'un effet de dimension
    (nombre d'iterations) ;
  - la famille est indiquee dans chaque ligne, car le rang reduit ne se compare
    pas au rang plein : mesure anterieurement, fa a t=8 demandait 258 iterations
    contre 26 pour us, AVEC MOINS de parametres (31 contre 44).

Usage :
    python3 bench_backend.py --backend cpu --out cpu.csv --tag cpu16
    python3 bench_backend.py --backend gpu --out gpu.csv --tag carte
"""
import argparse, csv, json, os, platform, statistics, sys, time
import numpy as np


# ==============================================================================
# Dispositif : UN seul jeu de donnees par (n, q, t), reutilise pour tous les p
# ==============================================================================
def design(n_unit, t, q, seed=7, LK=None):
    """Multi-caracteres a q genotypes. Les donnees ne dependent PAS de la
    structure ajustee : c'est ce qui permet de faire varier p a donnees egales.

    ATTENTION AU PIEGE DEJA RENCONTRE : si le nombre d'unites descend sous q,
    des niveaux ne sont jamais observes et le systeme devient quasi singulier —
    ce qui se lit alors comme un cout de p alors que c'est un dispositif
    degenere. On l'interdit ici plutot que de le detecter apres coup.
    """
    if n_unit < 2 * q:
        raise ValueError("n_unit = %d < 2q = %d : replication insuffisante, "
                         "des niveaux seraient non observes ou non repliques"
                         % (n_unit, 2 * q))
    rng = np.random.default_rng(seed)
    n = n_unit * t
    unit = np.repeat(np.arange(n_unit), t)
    trait = np.tile(np.arange(t), n_unit)
    lev = unit % q
    # effets genetiques correles entre caracteres, structure pleine
    A = rng.normal(0, 1.0, (t, t)) / np.sqrt(t)
    G = A @ A.T + 0.5 * np.eye(t)
    u = rng.multivariate_normal(np.zeros(t), G, size=q)
    # La parente doit servir A LA GENERATION autant qu'a l'ajustement : ajuster
    # une K sur des donnees qui n'en portent pas ne change pas le COUT, le
    # travail arithmetique etant le meme, mais rend les vraisemblances et le
    # nombre d'iterations peu representatifs. Les deux autres fabriques le
    # faisaient deja ; celle-ci ne le faisait pas.
    if LK is not None:
        u = np.asarray(LK) @ u
    y = 1.5 + u[lev, trait] + rng.normal(0, 0.8, n)
    X = np.zeros((n, t))
    X[np.arange(n), trait] = 1.0                       # une moyenne par caractere
    # INDICE DE COLONNE : dense_Z construit Z avec t*q colonnes, l'effet aleatoire
    # etant le vecteur vec(Sigma (x) K) ordonne PAR CARACTERE — le code des BLUP
    # fait kron(Ls, Kq) puis reshape(t, q), donc la colonne est
    #     caractere * q + niveau
    # et non le niveau seul. Avec zj = lev, toutes les observations chargeaient le
    # meme bloc de q colonnes : le produit Z (Sigma (x) K) Z' ne faisait alors
    # intervenir que Sigma[0,0], donc iid, diag et us rendaient EXACTEMENT la
    # meme vraisemblance malgre des p differents. Defaut detecte par le temoin
    # d'appariement du banc, pas par une relecture.
    zj = (trait * q + lev).astype(np.int64)
    return dict(n=n, n_unit=n_unit, t=t, q=q, y=y, X=X,
                zi=np.arange(n), zj=zj, zx=np.ones(n),
                trait=trait.astype(np.int64), unit=unit)



# ==============================================================================
# PARENTE : le meme dispositif, avec ou sans K dense
# ==============================================================================
def kinship_chol(q, n_marq=1500, graine=11):
    """Facteur de Cholesky d'une parente genomique de VanRaden, DENSE.

    C'est l'axe structurel : le meme modele avec K = I est structurellement
    creux (precision diagonale), avec K genomique il est structurellement dense
    (K^-1 pleine). Le terme le recoit par le champ LK, que term_factor traite
    comme un niveau 'fixed'. ATTENTION, le code de model.py le signale : une K
    fournie SANS cle 'lvl' est traitee comme fixed, et faire retomber sur 'id'
    ferait ignorer la parente en silence — panne deja survenue sur le portage
    GPU du modele IGE, ou deux runs etiquetes 'kinship' etaient des doublons
    exacts des runs sans.
    """
    rng = np.random.default_rng(graine)
    fr = rng.uniform(0.05, 0.95, n_marq)
    M = rng.binomial(2, fr, size=(q, n_marq)).astype(np.float64)
    Zc = M - 2.0 * fr
    K = Zc @ Zc.T / (2.0 * np.sum(fr * (1.0 - fr)))
    K += np.eye(q) * 1e-6                      # regularisation minimale
    return np.linalg.cholesky(K)


# ==============================================================================
# Dispositif spatial : essai au champ en lignes x colonnes, residuelle ar1 x ar1
# ==============================================================================
def design_spatial(nr, nc, q, seed=13, LK=None):
    """L'essai au champ classique : une parcelle par cellule d'une grille.

    La residuelle porte ar1(row) x ar1(col), qui est LA structure des essais
    agronomiques depuis Gilmour et Cullis. Elle est separable, donc ses deux
    parametres de correlation s'ajoutent aux parametres de variance.
    """
    rng = np.random.default_rng(seed)
    n = nr * nc
    if n < 2 * q:
        raise ValueError("grille %dx%d = %d parcelles < 2q = %d" % (nr, nc, n, 2 * q))
    row = np.repeat(np.arange(nr), nc)
    col = np.tile(np.arange(nc), nr)
    lev = rng.permutation(np.repeat(np.arange(q), int(np.ceil(n / q)))[:n])
    u = rng.normal(0, 1.0, q) if LK is None else LK @ rng.normal(0, 1.0, q)
    y = 2.0 + u[lev] + rng.normal(0, 0.7, n)
    return dict(n=n, n_unit=n, t=1, q=q, y=y, X=np.ones((n, 1)),
                zi=np.arange(n), zj=lev.astype(np.int64), zx=np.ones(n),
                trait=np.zeros(n, np.int64), unit=np.arange(n),
                row=row, col=col, dims=(nr, nc), LK=LK)


# ==============================================================================
# Effets genetiques directs et indirects : deux incidences, une us(2)
# ==============================================================================
def design_ige(n_unit, q, voisins=4, seed=17, LK=None):
    """DGE et IGE : l'effet d'un genotype sur lui-meme et sur ses voisins.

    Un SEUL terme a t = 2, dont les deux 'caracteres' sont l'effet direct et
    l'effet indirect, avec une us(2) entre eux — c'est la parametrisation
    classique, et la correlation entre les deux est la quantite d'interet. Les
    deux colonnes de Z different : identite du genotype de la parcelle pour le
    direct, somme des voisins pour l'indirect. L'incidence indirecte est DENSE
    en colonnes meme quand la precision est diagonale, ce qui remplit le bloc
    des produits croises — le second mecanisme de remplissage identifie dans la
    session, distinct de la densite de K.
    """
    rng = np.random.default_rng(seed)
    if n_unit < 2 * q:
        raise ValueError("n_unit = %d < 2q = %d" % (n_unit, 2 * q))
    gen = rng.permutation(np.repeat(np.arange(q), int(np.ceil(n_unit / q)))[:n_unit])
    # voisinage : `voisins` parcelles tirees au hasard, ponderation 1/voisins
    vz_i, vz_j, vz_x = [], [], []
    for i in range(n_unit):
        vs = rng.choice(n_unit, size=voisins, replace=False)
        for v in vs:
            vz_i.append(i); vz_j.append(q + gen[v]); vz_x.append(1.0 / voisins)
    zi = np.concatenate([np.arange(n_unit), np.asarray(vz_i)])
    zj = np.concatenate([gen, np.asarray(vz_j)]).astype(np.int64)   # direct : 0..q-1
    zx = np.concatenate([np.ones(n_unit), np.asarray(vz_x)])        # indirect : q..2q-1
    ud = rng.normal(0, 1.0, q); ui = 0.6 * ud + 0.8 * rng.normal(0, 0.6, q)
    if LK is not None:
        ud, ui = LK @ ud, LK @ ui
    y = np.full(n_unit, 2.0)
    np.add.at(y, np.arange(n_unit), ud[gen])
    for i, j, x in zip(vz_i, vz_j, vz_x):
        y[i] += x * ui[j - q]
    y += rng.normal(0, 0.7, n_unit)
    return dict(n=n_unit, n_unit=n_unit, t=2, q=q, y=y, X=np.ones((n_unit, 1)),
                zi=zi, zj=zj, zx=zx,
                trait=np.zeros(n_unit, np.int64), unit=np.arange(n_unit), LK=LK)


def modele(d, struct, rank=0, res_struct="diag", res_lvl=None):
    """LK est propage DEPUIS le dispositif : c'est le seul endroit ou l'axe
    'avec ou sans parente' se decide, pour qu'il ne puisse pas etre perdu en
    route. Une K fournie sans cle 'lvl' est traitee comme un niveau fixe par
    term_factor, ce qui est le comportement voulu.
    """
    terms = [dict(name="g", struct=struct, t=d["t"], rank=rank, q=d["q"],
                  zi=d["zi"], zj=d["zj"], zx=d["zx"], LK=d.get("LK"))]
    res = dict(struct=res_struct, t=d["t"], rank=0,
               trait=d["trait"], unit=d["unit"])
    if res_lvl == "ar1ar1":
        # residuelle spatiale separable : C_unite = AR1(lignes) x AR1(colonnes)
        res.update(lvl="ar1ar1", lvl_parts=("ar1", "ar1"), dims=d["dims"])
    return terms, res


# ==============================================================================
# Mesures
# ==============================================================================
def chrono_eval(fit_reml, terms, res, d, theta, reps):
    """Evaluation A THETA IMPOSE : maxiter = 0 court-circuite l'optimiseur, donc
    ceci evalue et ne se deplace pas. C'est la mesure appariee entre backends.
    La premiere passe paie la compilation ; le solveur la rapporte lui-meme.
    """
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fit_reml(terms, res, d["y"], d["X"], theta_init=theta,
                       maxiter=0, polish=0, hessian=False, blups=False, verbose=False)
        ts.append(time.perf_counter() - t0)
    return out, ts


def chrono_fit(fit_reml, terms, res, d, maxiter, hessian):
    t0 = time.perf_counter()
    out = fit_reml(terms, res, d["y"], d["X"], maxiter=maxiter, polish=2,
                   hessian=hessian, blups=False, verbose=False)
    return out, time.perf_counter() - t0


def structures(t, familles):
    """(nom, struct, rank) pour chaque structure demandee, p croissant.

    SATURATION DU RANG REDUIT, a ne pas lire comme un defaut. fa(t, k) est une
    restriction de us(t) seulement si le rang est strictement insuffisant. A
    t = 2, un facteur de rang 1 plus les variances specifiques represente
    N'IMPORTE QUELLE matrice 2x2 : fa(2,1) et us(2) sont donc le MEME modele et
    rendent la meme vraisemblance, fa portant meme PLUS de parametres (6 contre
    5, mesure). Le controle d'emboitement doit l'admettre. Le cas est conserve
    dans le banc parce qu'il illustre exactement ce que la session a mesure par
    ailleurs : le rang reduit n'economise pas toujours des parametres, et il
    coute des iterations — 23 contre 14 ici a t=2, pour le meme optimum.
    """
    cat = []
    for f in familles:
        if f == "iid":
            cat.append(("iid", "iid", 0))
        elif f == "diag":
            cat.append(("diag", "diag", 0))
        elif f == "us":
            cat.append(("us", "us", 0))
        elif f.startswith("fa"):
            k = int(f[2:])
            if k < t:
                cat.append(("fa%d" % k, "fa", k))
    return cat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    # ON PILOTE PAR LE TOTAL, PAS PAR LE NOMBRE D'UNITES. Le moteur dense forme
    # V explicitement : a n = 32000 elle pese 8 Go en float64, et il en faut
    # trois vivantes. Fixer n_unit et laisser n = n_unit * t croitre avec t
    # rendait donc les grands t infaisables. En fixant n, l'axe p se parcourt a
    # taille de probleme CONSTANTE, ce qui est aussi ce qu'on veut mesurer.
    ap.add_argument("--n", type=int, default=4000,
                    help="observations TOTALES ; n_unit = n / t")
    # q doit tenir sous n_unit/2 pour le plus grand t demande, sinon des niveaux
    # seraient non repliques — le defaut qui a fausse la grille precedente.
    ap.add_argument("--q", type=int, default=200)
    ap.add_argument("--ts", default="2,4,8")
    ap.add_argument("--familles", default="iid,diag,fa1,fa2,fa4,us")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--no-fit", action="store_true",
                    help="evaluations seules ; l'ajustement complet est le poste cher")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import jax
    from remlax.device import pick_device, device_report
    from remlax.fit import fit_reml
    dev, plat = pick_device(a.backend)
    ctx = jax.default_device(dev)
    ctx.__enter__()

    # Le nombre de coeurs VUS pilote le temps de compilation : une valeur
    # differente entre deux jobs rend leurs temps incomparables. On l'enregistre.
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count()
    meta = dict(backend=plat, device=str(dev), tag=a.tag,
                device_kind=getattr(dev, "device_kind", ""),
                n_cpu_affinity=ncpu, omp=os.environ.get("OMP_NUM_THREADS", ""),
                jax=jax.__version__, python=platform.python_version(),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("[banc] %s" % json.dumps(meta), flush=True)
    print("[banc] peripheriques : %s" % device_report(), flush=True)

    # Rodage hors mesure : la toute premiere compilation du processus paie aussi
    # l'initialisation du backend, qui n'a rien a voir avec le modele.
    dw = design(200, 2, 40, seed=99)
    tw, rw = modele(dw, "diag")
    fit_reml(tw, rw, dw["y"], dw["X"], maxiter=3, polish=0, hessian=False,
             blups=False, verbose=False)

    lignes = []
    ts = [int(v) for v in a.ts.split(",")]
    n_unit_min = a.n // max(ts)
    if n_unit_min < 2 * a.q:
        raise SystemExit("q = %d trop grand : au plus grand t = %d il ne reste que %d unites, "
                         "il en faut au moins 2q = %d. Baisser --q ou monter --n."
                         % (a.q, max(ts), n_unit_min, 2 * a.q))
    for t in ts:
        d = design(a.n // t, t, a.q)
        for nom, struct, rank in structures(t, a.familles.split(",")):
            terms, res = modele(d, struct, rank)
            base = dict(meta, n=d["n"], n_unit=d["n_unit"], t=t, q=a.q,
                        structure=nom, famille=struct, rang=rank)
            # --- 1. ajustement de reference : donne le theta commun ET le total
            try:
                out, paroi = chrono_fit(fit_reml, terms, res, d, a.maxiter,
                                        hessian=not a.no_fit)
            except Exception as e:
                lignes.append(dict(base, phase="fit", statut="echec",
                                   erreur=type(e).__name__ + ": " + str(e)[:180]))
                print("[fit ] %-6s t=%d ECHEC %s" % (nom, t, type(e).__name__), flush=True)
                continue
            p = int(out["n_par"])
            lignes.append(dict(base, phase="fit", statut="ok", p=p,
                               n_iter=int(out.get("n_iter", -1)),
                               fit_paroi_s=paroi, fit_interne_s=float(out["secondes"]),
                               compile_s=float(out.get("compile_s", float("nan"))),
                               eval_s=float(out.get("eval_s", float("nan"))),
                               logLik=float(out["logLik"]),
                               newton_decrement=float(out.get("newton_decrement", float("nan"))),
                               grad_rel=float(out.get("grad_rel", float("nan"))),
                               conv_decrement=out.get("conv_decrement"),
                               tronque=bool(out.get("n_iter", 0) >= a.maxiter),
                               n_eval_predit=int(out.get("n_iter", 0)) + 2 * p * (0 if a.no_fit else 1)))
            print("[fit ] %-6s t=%d p=%-3d iter=%-4d interne %8.2f s (compil %6.2f s)"
                  % (nom, t, p, out.get("n_iter", -1), out["secondes"],
                     out.get("compile_s", float("nan"))), flush=True)

            # --- 2. evaluations A THETA IMPOSE : la mesure appariee
            th = np.asarray(out["theta"], dtype=np.float64)
            try:
                oe, ts = chrono_eval(fit_reml, terms, res, d, th, a.reps)
            except Exception as e:
                lignes.append(dict(base, phase="eval", statut="echec",
                                   erreur=type(e).__name__ + ": " + str(e)[:180]))
                continue
            lignes.append(dict(base, phase="eval", statut="ok", p=p, reps=a.reps,
                               eval_paroi_median_s=statistics.median(ts),
                               eval_paroi_min_s=min(ts), eval_paroi_max_s=max(ts),
                               compile_s=float(oe.get("compile_s", float("nan"))),
                               eval_s=float(oe.get("eval_s", float("nan"))),
                               logLik=float(oe["logLik"]),
                               # temoin d'appariement : la vraisemblance a theta
                               # impose doit etre IDENTIQUE entre backends. Si
                               # elle differe, ce n'est pas le meme calcul et le
                               # rapport de temps ne veut rien dire.
                               theta_impose=" ".join("%.10g" % v for v in th)))
            print("[eval] %-6s t=%d p=%-3d compil %7.3f s | evaluation %8.5f s | logLik %.9f"
                  % (nom, t, p, oe.get("compile_s", float("nan")),
                     oe.get("eval_s", float("nan")), oe["logLik"]), flush=True)

    champs = []
    for r in lignes:
        for k in r:
            if k not in champs:
                champs.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        for r in lignes:
            w.writerow(r)
    print("\n[banc] %d lignes ecrites dans %s" % (len(lignes), a.out), flush=True)
    ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
