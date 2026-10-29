"""Loi a posteriori des parametres de variance, par HMC (BlackJAX, ChEES).

PRINCIPE. Les effets fixes beta et les effets aleatoires u sont integres
ANALYTIQUEMENT : avec un a priori plat sur beta, la loi a posteriori de theta
est proportionnelle a la vraisemblance restreinte,

    p(theta | y)  ~  L_R(theta) p(theta),

ou L_R est exactement celle que maximise fit_reml (make_neg2). La chaine ne
porte donc que les parametres de variance, quelques dizaines au plus, et non
les milliers de niveaux de u : pas d'entonnoir de Neal entre une variance et
ses effets, la geometrie qui rend les modeles hierarchiques difficiles a
echantillonner quand u est explicite.

beta et u se deduisent ensuite tirage par tirage : sachant theta et y, ils sont
gaussiens (effets_denses donne moyenne et variance). On en tire les moments a
posteriori par Rao-Blackwell sur un sous-ensemble de tirages :

    E[u | y]   = moyenne des BLUP(theta_s)
    var(u | y) = moyenne des PEV(theta_s) + variance des BLUP(theta_s).

LOIS A PRIORI, posees sur l'echelle naturelle et ramenees a theta par le
jacobien du changement de variable :

    ecart-type (iid, diag, psi d'un fa, diagonale d'un us)   demi-t (defaut)
    matrice de correlation d'un bloc us                      LKJ(eta)
    loadings d'un fa ou d'un rr                               normale centree
    parametre de correlation en tanh (ar1, exp, ...)          uniforme sur (-1, 1)
    autres parametres (portees, Matern, structures rares)     normale sur theta

L'echelle par defaut des ecarts-types et des loadings est l'ecart-type des
residus des moindres carres ordinaires de y sur X.

ECHANTILLONNEUR. HMC a trajectoire gigotee (jittered), dont le pas et la
longueur de trajectoire sont adaptes par le critere ChEES sur l'ensemble des
chaines (Hoffman, Radul et Sountsov 2021), dans BlackJAX. Toutes les chaines
partagent le meme pas et la meme longueur : sous vmap, aucune n'attend une
trajectoire plus longue que les autres, ce qui est le defaut de NUTS vectorise.
D'ou le choix de NOMBREUSES CHAINES COURTES, batchees dans les memes appels XLA,
sur CPU comme sur GPU.

BLANCHIMENT. On echantillonne z, avec theta = theta_REML + L z et L L' = 2 H^-1
(covariance de Laplace a l'optimum REML). La loi de z est proche d'une normale
reduite : ChEES, qui n'adapte pas de matrice de masse pleine, y travaille dans
de bonnes conditions. Sans Hessien, L = 0,1 I.

CONVERGENCE. Les chaines sont groupees en K super-chaines : les M chaines d'une
super-chaine partent du MEME point, tire d'une loi surdispersee (z ~ N(0, 2^2 I)).
Le R-hat emboite (Margossian et al. 2024) compare les super-chaines entre
elles ; il reste valable quand chaque chaine est courte, la ou le R-hat
classique ne l'est plus.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .model import make_neg2, term_n_params, res_sections, sec_n_params
    from .structures import n_params, n_loadings, build_sigma
    from .levels import n_level_params
    from .inference import components_from_theta, component_names
except ImportError:                                             # appel hors paquet
    from model import make_neg2, term_n_params, res_sections, sec_n_params
    from structures import n_params, n_loadings, build_sigma
    from levels import n_level_params
    from inference import components_from_theta, component_names

# parametres entre niveaux dont la valeur est tanh(theta) (ou une image affine
# de tanh, comme la correlation uniforme `cor`) : uniforme sur la correlation
NIVEAUX_TANH = {"ar1", "sar", "ma1", "ar1ar1", "ma2", "arma", "ar2", "ar3", "cor",
                "exp", "gau", "aexp", "agau", "iexp", "igau", "ieuc"}

PRIOR_DEFAUT = {
    "sd": {"famille": "half_t", "df": 3.0, "echelle": None},
    "lkj": 2.0,
    "loading": {"echelle": None},
    "niveau": "uniforme",
    "generique": 2.0,
    "termes": {},
}
FAMILLES_SD = ("half_t", "half_normal", "half_cauchy", "inv_gamma", "uniforme")


def disponible():
    """BlackJAX est une dependance OPTIONNELLE : seule la voie bayesienne l'exige."""
    try:
        import blackjax  # noqa: F401
        import optax  # noqa: F401
        return True
    except ImportError:
        return False


# ==============================================================================
# Lois a priori
# ==============================================================================
def _log_sd(x, spec):
    """log densite (a une constante pres) d'un ecart-type x > 0."""
    fam = spec["famille"]
    if fam == "uniforme":
        return jnp.zeros_like(x)
    if fam == "inv_gamma":
        # sur la VARIANCE v = x^2, ramenee a x : log p(v) + log(dv/dx) = log p(v) + log(2x)
        a, b = float(spec.get("forme", 0.001)), float(spec.get("taux", 0.001))
        v = x * x
        return -(a + 1.0) * jnp.log(v) - b / v + jnp.log(x)
    s = float(spec["echelle"])
    if fam == "half_normal":
        return -0.5 * (x / s) ** 2
    df = 1.0 if fam == "half_cauchy" else float(spec.get("df", 3.0))
    return -0.5 * (df + 1.0) * jnp.log1p((x / s) ** 2 / df)


def _log_sech2(th):
    """log(1 - tanh(th)^2), stable pour |th| grand."""
    a = jnp.abs(th)
    return 2.0 * (jnp.log(2.0) - a - jnp.log1p(jnp.exp(-2.0 * a)))


def _resoudre_prior(prior, terms, res, y, X):
    """Complete la specification (defauts, echelles numeriques), par terme."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PRIOR_DEFAUT.items()}
    for k, v in (prior or {}).items():
        if k in ("sd", "loading") and isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    # echelle de reference : residus des moindres carres ordinaires
    Xn = np.asarray(X, dtype=float)
    b, *_ = np.linalg.lstsq(Xn, np.asarray(y, dtype=float), rcond=None)
    e = np.asarray(y, dtype=float) - Xn @ b
    s_ref = float(np.sqrt(np.sum(e * e) / max(len(e) - Xn.shape[1], 1)))
    p["echelle_reference"] = s_ref
    if p["sd"].get("echelle") is None:
        p["sd"]["echelle"] = s_ref
    if p["loading"].get("echelle") is None:
        p["loading"]["echelle"] = s_ref
    if p["sd"]["famille"] not in FAMILLES_SD:
        raise ValueError("prior : famille d'ecart-type inconnue %r (attendu %s)"
                         % (p["sd"]["famille"], ", ".join(FAMILLES_SD)))
    noms = [tm["name"] for tm in terms] + ["residuelle"]
    inconnus = set(p.get("termes") or {}) - set(noms)
    if inconnus:
        raise ValueError("prior : terme(s) inconnu(s) %s ; termes du modele : %s"
                         % (sorted(inconnus), noms))
    return p


def _spec_terme(p, nom):
    """Specification effective d'un terme : defauts, puis surcharges du terme."""
    sd, lkj, load = dict(p["sd"]), p["lkj"], dict(p["loading"])
    sur = (p.get("termes") or {}).get(nom) or {}
    if "sd" in sur:
        sd.update(sur["sd"])
        if sd.get("echelle") is None:
            sd["echelle"] = p["echelle_reference"]
    if "lkj" in sur:
        lkj = float(sur["lkj"])
    if "loading" in sur:
        load.update(sur["loading"])
    return sd, float(lkj), load


def _blocs(terms, res, p):
    """Decoupage de theta en blocs de loi a priori.

    Chaque bloc : (nature, debut, longueur, t, nom, specification). Natures :
    "sd", "us", "loading", "tanh", "generique".
    """
    blocs, o, rares = [], 0, []

    def sigma(o, struct, t, rank, nom):
        sd, lkj, load = _spec_terme(p, nom)
        ns = n_params(struct, t, rank)
        if not ns:
            return []
        if struct in ("iid", "diag") or (struct == "us" and t == 1):
            return [("sd", o, ns, t, nom, sd)]
        if struct == "us":
            return [("us", o, ns, t, nom, dict(sd=sd, lkj=lkj))]
        if struct == "fa":
            nl = n_loadings(t, rank)
            return [("loading", o, nl, t, nom, load), ("sd", o + nl, t, t, nom, sd)]
        if struct == "rr":
            return [("loading", o, ns, t, nom, load)]
        rares.append("%s (%s)" % (nom, struct))
        return [("generique", o, ns, t, nom, p["generique"])]

    def niveaux(o, kind, parts, nom, total):
        if total <= 0:
            return []
        if kind == "sep":
            out, oo = [], o
            for (k, _q) in parts or []:
                pk = n_level_params(k)
                if pk:
                    out += niveaux(oo, k, None, nom, pk)
                oo += pk
            return out
        if kind in NIVEAUX_TANH and p["niveau"] == "uniforme":
            return [("tanh", o, total, 0, nom, None)]
        return [("generique", o, total, 0, nom, p["generique"])]

    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        tot = term_n_params(tm)
        blocs += sigma(o, tm["struct"], tm["t"], tm["rank"], tm["name"])
        kind = tm.get("lvl") or ("fixed" if tm.get("LK") is not None else "id")
        blocs += niveaux(o + ns, kind, tm.get("lvl_parts"), tm["name"], tot - ns)
        o += tot
    for sec in res_sections(res, 0):
        tot = sec_n_params(sec)
        if tot == 0:
            continue
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        blocs += sigma(o, sec["struct"], sec["t"], sec["rank"], "residuelle")
        blocs += niveaux(o + ns, sec.get("lvl", "id"), sec.get("lvl_parts"), "residuelle", tot - ns)
        o += tot
    return blocs, o, rares


def make_log_prior(terms, res, p):
    """log p(theta), jacobien compris, comme fonction JAX de theta entier."""
    blocs, ntot, rares = _blocs(terms, res, p)

    def us_phi(th, t):
        S = build_sigma(th, "us", t, 0)
        sd = jnp.sqrt(jnp.diagonal(S))
        R = S / jnp.outer(sd, sd)
        ii, jj = np.tril_indices(t, -1)
        return jnp.concatenate([sd, R[ii, jj]])

    def lp(theta):
        tot = 0.0
        for (nat, o, ln, t, _nom, spec) in blocs:
            th = theta[o:o + ln]
            if nat == "sd":
                tot = tot + jnp.sum(_log_sd(jnp.exp(th), spec) + th)
            elif nat == "us":
                phi = us_phi(th, t)
                sd = phi[:t]
                S = build_sigma(th, "us", t, 0)
                R = S / jnp.outer(sd, sd)
                J = jax.jacfwd(lambda z: us_phi(z, t))(th)
                tot = (tot + jnp.sum(_log_sd(sd, spec["sd"]))
                       + (spec["lkj"] - 1.0) * jnp.linalg.slogdet(R)[1]
                       + jnp.linalg.slogdet(J)[1])
            elif nat == "loading":
                tot = tot - 0.5 * jnp.sum((th / float(spec["echelle"])) ** 2)
            elif nat == "tanh":
                tot = tot + jnp.sum(_log_sech2(th))
            else:
                tot = tot - 0.5 * jnp.sum((th / float(spec)) ** 2)
        return tot

    return lp, blocs, rares


# ==============================================================================
# Echantillonnage
# ==============================================================================
def rhat_emboite(x, K):
    """R-hat emboite (Margossian et al. 2024). x : (C, D), C = K * M chaines,
    les M chaines d'une super-chaine etant contigues et parties du meme point."""
    x = np.asarray(x, dtype=np.float64)
    C, D = x.shape
    M = C // K
    if K < 2 or D < 2:
        return np.nan
    x = x[:K * M].reshape(K, M, D)
    moy_ch = x.mean(axis=2)                                   # (K, M)
    moy_sc = moy_ch.mean(axis=1)                              # (K,)
    B = moy_sc.var(ddof=1)
    Bk = moy_ch.var(axis=1, ddof=1) if M > 1 else np.zeros(K)
    Wk = x.var(axis=2, ddof=1).mean(axis=1)
    W = np.mean(Bk + Wk)
    return float(np.sqrt(1.0 + B / W)) if W > 0 else np.nan


def _resume(x, K):
    """x : (chaines, tirages). Moyenne, ecart-type, quantiles, R-hat, ESS."""
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
    x = np.asarray(x, dtype=np.float64)
    plat = x.reshape(-1)
    q = np.quantile(plat, [0.025, 0.5, 0.975])
    ok = np.all(np.isfinite(plat)) and plat.std() > 0 and x.shape[1] >= 4
    rhat = float(potential_scale_reduction(jnp.asarray(x))) if ok else np.nan
    ess = float(effective_sample_size(jnp.asarray(x))) if ok else np.nan
    return dict(moyenne=float(plat.mean()), sd=float(plat.std(ddof=1)), q025=float(q[0]),
                mediane=float(q[1]), q975=float(q[2]), rhat=rhat,
                rhat_emboite=rhat_emboite(x, K) if ok else np.nan, ess=ess)


def echantillonner(logdens, z_init, warmup, draws, seed=0, target_accept=0.651,
                   max_leapfrog=1000, verbose=True):
    """HMC adapte par ChEES sur toutes les chaines, puis tirages par tranches.

    logdens : log densite JAX d'un vecteur z (d,). z_init : (C, d). Rend
    (z (C, D, d), infos, diagnostics). Chaque tranche est une boucle scan
    compilee une fois ; les temps sont pris apres block_until_ready, l'execution
    de JAX etant asynchrone (un chrono pris au retour d'appel ne mesure que
    l'envoi du calcul).
    """
    import blackjax
    import optax
    C, d = z_init.shape
    k_w, k_s = jax.random.split(jax.random.PRNGKey(seed))
    adapt = blackjax.chees_adaptation(logdens, num_chains=C, target_acceptance_rate=target_accept,
                                      max_leapfrog_steps=max_leapfrog)
    t1 = time.time()
    (etats, params), _ = adapt.run(k_w, jnp.asarray(z_init), step_size=0.3,
                                   optim=optax.adam(0.025), num_steps=warmup)
    jax.block_until_ready(etats)
    t_chauffe = time.time() - t1
    pas_ = float(np.asarray(params["step_size"]))
    if verbose:
        print("  [bayes] chauffe ChEES : %.1f s (compilation comprise) | pas %.3g" % (t_chauffe, pas_),
              flush=True)
    noyau = blackjax.dynamic_hmc(logdens, **params)
    n_tr = max(1, min(10, draws // 10))
    tranche = -(-draws // n_tr)

    @jax.jit
    def bloc(cle, etats):
        def un(etats, k):
            etats, info = jax.vmap(noyau.step)(jax.random.split(k, C), etats)
            return etats, (etats.position, info.is_divergent, info.acceptance_rate,
                           info.num_integration_steps)
        return jax.lax.scan(un, etats, jax.random.split(cle, tranche))

    pos, div, acc, nst = [], [], [], []
    t1, fait = time.time(), 0
    for i in range(n_tr):
        etats, (p_, dv, ac, ns) = bloc(jax.random.fold_in(k_s, i), etats)
        jax.block_until_ready(p_)
        ti = min(tranche, draws - fait)
        pos.append(np.asarray(p_)[:ti]); div.append(np.asarray(dv)[:ti])
        acc.append(np.asarray(ac)[:ti]); nst.append(np.asarray(ns)[:ti])
        fait += ti
        if verbose:
            ec = time.time() - t1
            print("  [bayes] tirages %5d / %d (%3.0f %%) | %.1f s, reste ~%.0f s | %d divergence(s)"
                  % (fait, draws, 100.0 * fait / draws, ec, ec / fait * (draws - fait),
                     int(sum(x.sum() for x in div))), flush=True)
    t_tirage = time.time() - t1
    z = np.concatenate(pos, axis=0).transpose(1, 0, 2)            # (C, D, d)
    div = np.concatenate(div, axis=0)
    acc = np.concatenate(acc, axis=0)
    nst = np.concatenate(nst, axis=0)
    diag = dict(algorithme="ChEES-HMC", divergences=int(div.sum()), acceptation=float(acc.mean()),
                pas=pas_, pas_integration_moyens=float(nst.mean()),
                secondes_chauffe=t_chauffe, secondes_tirages=t_tirage)
    return z, diag


def preparer(terms, res, y, X, prior=None):
    """Loi a priori resolue, log p(theta) et -2 logL comme fonctions JAX."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    p = _resoudre_prior(prior, terms, res, y, X)
    log_prior, blocs, rares = make_log_prior(terms, res, p)
    f, Zr, Zs = make_neg2(terms, res, y, X)
    Zr_j = [None if z_ is None else jnp.asarray(z_) for z_ in Zr]
    return dict(p=p, log_prior=log_prior, rares=rares, f=f, Zr=Zr_j, Zs=Zs, y=y, X=X)


def mode_posterieur(prep, theta0, fixed_idx=None, floor=-12.0, ceil=12.0, maxiter=2000,
                    eps=1e-4, verbose=True):
    """Mode de la loi a posteriori de theta, et Hessien de -log p(theta | y) en ce mode.

    C'est le point de depart et le blanchiment de la voie bayesienne SANS
    ajustement REML (rx_bayes(start = "map"), le defaut). L'a priori eloigne le
    mode des frontieres (variance nulle, correlation de +-1), ou le Hessien REML
    n'est pas defini positif. Le Hessien est pris par differences centrees du
    gradient (2 d gradients) : la derivee de la vraisemblance passe par un
    custom_vjp, que JAX ne sait pas deriver en mode direct.
    Rend (theta_map, Hu sur les parametres libres, info).
    """
    from scipy.optimize import minimize
    lp, f, Zr = prep["log_prior"], prep["f"], prep["Zr"]
    th0 = np.asarray(theta0, dtype=np.float64)
    ntot = len(th0)
    libre = np.ones(ntot, dtype=bool)
    if fixed_idx:
        libre[np.asarray(fixed_idx, dtype=int)] = False
    idx = np.flatnonzero(libre)
    base, jidx = jnp.asarray(th0), jnp.asarray(idx)
    n = len(prep["y"])

    def U(tf):                                   # -log p(theta | y), a une constante pres
        th = base.at[jidx].set(tf)
        return 0.5 * f(th, Zr) - lp(th)

    vg = jax.jit(jax.value_and_grad(U))

    def fun(tf):                                 # mis a l'echelle par n, comme le REML
        v, g = vg(jnp.asarray(tf))
        v, g = float(v), np.asarray(g, dtype=np.float64)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(g)
        return v / n, g / n

    t1 = time.time()
    r = minimize(fun, th0[idx], jac=True, method="L-BFGS-B",
                 bounds=[(floor, ceil)] * len(idx),
                 options=dict(maxiter=maxiter, maxfun=10 * maxiter, ftol=1e-14, gtol=1e-9))
    tf = np.asarray(r.x, dtype=np.float64)
    gr = jax.jit(jax.grad(U))
    d = len(idx)
    Hu = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps
        Hu[:, j] = (np.asarray(gr(jnp.asarray(tf + e))) - np.asarray(gr(jnp.asarray(tf - e)))) / (2 * eps)
    Hu = 0.5 * (Hu + Hu.T)
    th = th0.copy(); th[idx] = tf
    info = dict(iterations=int(r.nit), message=str(r.message), succes=bool(r.success),
                log_posterieur=-float(r.fun) * n, secondes=time.time() - t1)
    if verbose:
        print("  [bayes] mode a posteriori : %d iteration(s), %.1f s (%s)"
              % (info["iterations"], info["secondes"], info["message"]), flush=True)
    return th, Hu, info


def sample_posterior(terms, res, y, X, theta_hat, H=None, Hu=None, prior=None, chains=16,
                     superchains=4, warmup=300, draws=100, seed=0, target_accept=0.651,
                     max_leapfrog=1000, fixed_idx=None, expressions=(), n_effets=100,
                     pev=(), lag=None, verbose=True):
    """Tire la loi a posteriori de theta, depuis l'optimum REML.

    theta_hat : centre du blanchiment et des points de depart : le mode a
                posteriori (mode_posterieur) ou l'optimum REML.
    H         : Hessien de -2logL en theta_hat (optimum REML) ; 2 H^-1 blanchit.
    Hu        : Hessien de -log p(theta | y) sur les parametres libres (mode a
                posteriori) ; Hu^-1 blanchit. Prioritaire sur H. Si ni l'un ni
                l'autre n'est defini positif, le mode a posteriori est cherche
                depuis theta_hat et son Hessien sert au blanchiment.
    chains, superchains : C chaines groupees en K super-chaines (C multiple de K).
    warmup, draws : iterations d'adaptation ChEES, tirages gardes PAR CHAINE.
    fixed_idx : indices (base 0) tenus a leur valeur de theta_hat.
    expressions : [(nom, chaine)] facon vpredict, evaluees a chaque tirage.
    n_effets  : tirages servant aux moments a posteriori de beta et des BLUP.
    pev       : termes dont on veut l'ecart-type a posteriori des effets.
    """
    if not disponible():
        raise RuntimeError("la voie bayesienne exige blackjax : pip install blackjax "
                           "(ou rx_install_python(bayes = TRUE) depuis R)")
    if lag is not None:
        raise ValueError("la voie bayesienne n'accepte pas encore de decalage spatial (rx_lag)")
    K = max(1, int(superchains))
    if chains % K:
        raise ValueError("chains (%d) doit etre un multiple de superchains (%d)" % (chains, K))

    t0 = time.time()
    prep = preparer(terms, res, y, X, prior)
    y, X, p = prep["y"], prep["X"], prep["p"]
    log_prior, rares, f, Zs = prep["log_prior"], prep["rares"], prep["f"], prep["Zs"]
    if rares and verbose:
        print("  [bayes] loi normale sur theta (echelle interne) pour : %s" % ", ".join(rares),
              flush=True)
    th_hat = np.asarray(theta_hat, dtype=np.float64)
    ntot = len(th_hat)
    libre = np.ones(ntot, dtype=bool)
    if fixed_idx:
        libre[np.asarray(fixed_idx, dtype=int)] = False
    idx = np.flatnonzero(libre)
    d = len(idx)

    # BLANCHIMENT. Lw Lw' = covariance de Laplace : Hu^-1 au mode a posteriori,
    # ou 2 H^-1 a l'optimum REML. Un Hessien non defini positif (optimum REML a
    # une frontiere) fait chercher le mode a posteriori depuis theta_hat.
    def facteur(M, k):
        try:
            w, U_ = np.linalg.eigh(0.5 * (M + M.T))
        except np.linalg.LinAlgError:
            return None
        if not np.all(w > 1e-10 * max(abs(w).max(), 1.0)):
            return None
        return U_ * np.sqrt(k / w)

    Lw = None
    if Hu is not None:
        Lw = facteur(np.asarray(Hu), 1.0)
    elif H is not None:
        Lw = facteur(np.asarray(H)[np.ix_(idx, idx)], 2.0)
    if Lw is None:
        if verbose:
            print("  [bayes] Hessien absent ou non defini positif : recherche du mode a posteriori",
                  flush=True)
        th_hat, Hm, _ = mode_posterieur(prep, th_hat, fixed_idx=fixed_idx, verbose=verbose)
        Lw = facteur(Hm, 1.0)
    if Lw is None:
        Lw = 0.1 * np.eye(d)
        if verbose:
            print("  [bayes] Hessien du mode non defini positif : blanchiment par 0,1 I", flush=True)
    base = jnp.asarray(th_hat)
    jidx = jnp.asarray(idx)
    centre = jnp.asarray(th_hat[idx])
    Lw_j = jnp.asarray(Lw)
    Zr_j = prep["Zr"]

    def theta_de(z):
        return base.at[jidx].set(centre + Lw_j @ z)

    def logdens(z):
        th = theta_de(z)
        v = log_prior(th) - 0.5 * f(th, Zr_j)
        return jnp.where(jnp.isfinite(v), v, -jnp.inf)

    # points de depart : un par super-chaine, surdisperse, partage par ses M chaines
    rng = np.random.default_rng(seed)
    M = chains // K
    z_sc = 2.0 * rng.standard_normal((K, d))
    z_init = np.repeat(z_sc, M, axis=0)
    if verbose:
        print("  [bayes] ChEES-HMC : %d chaines (%d super-chaines x %d), %d chauffe + %d tirages, "
              "%d parametre(s) libre(s) | preparation %.1f s"
              % (chains, K, M, warmup, draws, d, time.time() - t0), flush=True)
    z, diag = echantillonner(logdens, z_init, warmup, draws, seed=seed,
                             target_accept=target_accept, max_leapfrog=max_leapfrog,
                             verbose=verbose)
    C, D = z.shape[0], z.shape[1]
    tf = th_hat[idx][None, None, :] + np.einsum("ij,cdj->cdi", Lw, z)
    th_draws = np.tile(th_hat, (C, D, 1))
    th_draws[:, :, idx] = tf

    # --- composantes et expressions, tirage par tirage -------------------------
    t1 = time.time()
    plats = th_draws.reshape(C * D, ntot)
    try:
        comp_fn = jax.jit(jax.vmap(lambda th: components_from_theta(th, terms, res)))
        comps = np.asarray(comp_fn(jnp.asarray(plats)))
    except Exception:            # composantes non vectorisables (rapports numpy des niveaux)
        un = jax.jit(lambda th: components_from_theta(th, terms, res))
        comps = np.stack([np.asarray(un(jnp.asarray(t_))) for t_ in plats])
    noms = component_names(terms, res)
    vp = []
    for (nom, expr) in expressions:
        env = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "abs": np.abs}
        env.update({"V%d" % (i + 1): comps[:, i] for i in range(comps.shape[1])})
        vp.append(np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=float)  # noqa: S307
                  * np.ones(C * D))
    vp = np.stack(vp, axis=1) if vp else np.zeros((C * D, 0))
    resume = []
    for i, nom in enumerate(noms):
        resume.append(dict(nom=nom, **_resume(comps[:, i].reshape(C, D), K)))
    for j, (nom, expr) in enumerate(expressions):
        resume.append(dict(nom=nom, expression=expr, **_resume(vp[:, j].reshape(C, D), K)))
    if verbose:
        print("  [bayes] composantes et expressions sur %d tirages : %.1f s"
              % (C * D, time.time() - t1), flush=True)

    def _max(cle):
        v = [r[cle] for r in resume if np.isfinite(r[cle])]
        return float(max(v)) if v else np.nan

    # --- beta et effets aleatoires, par Rao-Blackwell ---------------------------
    effets = None
    if n_effets and n_effets > 0:
        try:
            from .fit import effets_denses
        except ImportError:
            from fit import effets_denses
        t1 = time.time()
        k = int(min(n_effets, C * D))
        choix = np.unique(np.linspace(0, C * D - 1, k).round().astype(int))
        betas, vbetas, ub, upev = [], [], {}, {}
        for s_ in choix:
            e = effets_denses(plats[s_], terms, Zs, res, y, X, blups=True, pev=list(pev) or False)
            betas.append(e["beta"])
            vbetas.append(e["vbeta"])
            for nm, u in e["blups"].items():
                ub.setdefault(nm, []).append(u)
            for nm, pv in (e.get("pev") or {}).items():
                upev.setdefault(nm, []).append(pv)
        Bt = np.stack(betas)
        bcov = np.mean(vbetas, axis=0) + (np.cov(Bt.T) if len(choix) > 1 else 0.0)
        effets = dict(n=len(choix), beta_moyenne=Bt.mean(axis=0), beta_cov=np.atleast_2d(bcov),
                      blup_moyenne={nm: np.mean(v, axis=0) for nm, v in ub.items()},
                      blup_sd={nm: np.sqrt(np.mean(upev[nm], axis=0) + np.var(ub[nm], axis=0))
                               for nm in upev},
                      secondes=time.time() - t1)
        if verbose:
            print("  [bayes] moments de beta et des BLUP sur %d tirages : %.1f s"
                  % (len(choix), effets["secondes"]), flush=True)

    diag.update(chaines=C, superchaines=K, chauffe=int(warmup), tirages=D,
                rhat_max=_max("rhat"), rhat_emboite_max=_max("rhat_emboite"),
                ess_min=float(min([r["ess"] for r in resume if np.isfinite(r["ess"])] or [np.nan])),
                secondes=time.time() - t0, n_libres=int(d))
    prior_out = dict(sd=p["sd"], lkj=p["lkj"], loading=p["loading"], niveau=p["niveau"],
                     generique=p["generique"], echelle_reference=p["echelle_reference"],
                     termes=p.get("termes") or {}, loi_normale_theta=rares)
    return dict(theta=th_draws, composantes=comps, composantes_noms=noms,
                expressions=vp, expressions_noms=[n for n, _ in expressions],
                resume=resume, diagnostics=diag, prior=prior_out, effets=effets)
