"""Posterior distribution of the variance parameters, sampled by HMC (BlackJAX, ChEES).

Principle. The fixed effects beta and the random effects u are integrated out
analytically. With a flat prior on beta, the posterior of theta is
proportional to the restricted likelihood:

    p(theta | y)  ~  L_R(theta) p(theta),

where L_R is exactly the function that fit_reml maximises (make_neg2). The
chain therefore carries only the variance parameters, a few dozen at most. The
thousands of levels of u stay outside the chain. This removes Neal's funnel
between a variance and its effects. That funnel is the geometry that makes
hierarchical models hard to sample when u is explicit.

beta and u are then recovered draw by draw. Given theta and y they are
Gaussian, and dense_effects gives their mean and variance. Their posterior
moments follow by Rao-Blackwellisation over a subset of draws:

    E[u | y]   = mean of BLUP(theta_s)
    var(u | y) = mean of PEV(theta_s) + variance of BLUP(theta_s).

Priors. They are set on the natural scale and carried to theta by the Jacobian
of the change of variable:

    standard deviation (iid, diag, psi of fa, diagonal of us)   half-t (default)
    correlation matrix of a us block                            LKJ(eta)
    loadings of fa or rr                                        centred normal
    tanh correlation parameter (ar1, exp, ...)                  uniform on (-1, 1)
    other parameters (ranges, Matern, rare structures)          normal on theta

The default scale of standard deviations and loadings is the residual standard
deviation of the ordinary least squares fit of y on X.

Sampler. The sampler is jittered-trajectory HMC in BlackJAX. Its step size and
trajectory length are adapted by the ChEES criterion over all chains (Hoffman,
Radul and Sountsov 2021). All chains share the same step size and trajectory
length. Under vmap, vectorised NUTS makes every chain wait for the longest
trajectory. Shared settings remove that wait. Many short chains are therefore
run, batched in the same XLA calls, on CPU as on GPU.

Whitening. The chains sample z, with theta = theta_REML + L z and
L L' = 2 H^-1 (Laplace covariance at the REML optimum). When the centre is the
posterior mode, L L' = Hu^-1, with Hu the Hessian of -log p(theta | y). The
distribution of z is then close to a standard normal. ChEES adapts no dense
mass matrix and therefore needs a well-scaled target such as this one.
When no positive definite Hessian is available, L = 0.1 I.

Convergence. The chains are grouped in K superchains. The M chains of a
superchain start from the same point, drawn from an overdispersed distribution
(z ~ N(0, 2^2 I)). The nested R-hat (Margossian et al. 2024) compares the
superchains with each other. It stays valid when each chain is short, where
the classical R-hat does not.

Organisation. The first part builds the prior: densities of a standard
deviation, resolution of the user specification, cutting of theta into prior
blocks, and the log prior as a JAX function (make_log_prior). The second part
holds the sampler and its diagnostics (nested_rhat, run_sampler). The
entry points come last: prepare, posterior_mode and sample_posterior.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .layout import level_kind, term_n_params, res_sections, sec_n_params
from .model import make_neg2
from .structures import n_params, n_loadings, build_sigma
from .levels import n_level_params
from .components import components_from_theta, component_names

# Between-level parameters whose value is tanh(theta), or an affine image of
# tanh such as the uniform correlation `cor`. Their prior is uniform on the
# correlation.
LEVELS_TANH = {"ar1", "sar", "ma1", "ar1ar1", "ma2", "arma", "ar2", "ar3", "cor",
                "exp", "gau", "aexp", "agau", "iexp", "igau", "ieuc"}

PRIOR_DEFAULT = {
    "sd": {"family": "half_t", "df": 3.0, "scale": None},
    "lkj": 2.0,
    "loading": {"scale": None},
    "level": "uniform",
    "generic": 2.0,
    "terms": {},
}
SD_FAMILIES = ("half_t", "half_normal", "half_cauchy", "inv_gamma", "uniform")


def available():
    """Whether BlackJAX and optax are installed.

    They are optional dependencies. Only the Bayesian route needs them.
    """
    try:
        import blackjax  # noqa: F401
        import optax  # noqa: F401
        return True
    except ImportError:
        return False


# ==============================================================================
# Priors
# ==============================================================================
def _log_sd(x, spec):
    """Log density, up to a constant, of a standard deviation x > 0."""
    family = spec["family"]
    if family == "uniform":
        return jnp.zeros_like(x)
    if family == "inv_gamma":
        # The prior is on the variance v = x^2. Carried to x it becomes
        # log p(v) + log(dv/dx) = log p(v) + log(2x).
        a, b = float(spec.get("shape", 0.001)), float(spec.get("rate", 0.001))
        v = x * x
        return -(a + 1.0) * jnp.log(v) - b / v + jnp.log(x)
    s = float(spec["scale"])
    if family == "half_normal":
        return -0.5 * (x / s) ** 2
    df = 1.0 if family == "half_cauchy" else float(spec.get("df", 3.0))
    return -0.5 * (df + 1.0) * jnp.log1p((x / s) ** 2 / df)


def _log_sech2(th):
    """log(1 - tanh(th)^2), stable for large |th|."""
    a = jnp.abs(th)
    return 2.0 * (jnp.log(2.0) - a - jnp.log1p(jnp.exp(-2.0 * a)))


def _resolve_prior(prior, terms, res, y, X):
    """Complete the prior specification (defaults, numeric scales), term by term."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PRIOR_DEFAULT.items()}
    for k, v in (prior or {}).items():
        if k in ("sd", "loading") and isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    # Reference scale: residuals of the ordinary least squares fit.
    Xn = np.asarray(X, dtype=float)
    b, *_ = np.linalg.lstsq(Xn, np.asarray(y, dtype=float), rcond=None)
    e = np.asarray(y, dtype=float) - Xn @ b
    s_ref = float(np.sqrt(np.sum(e * e) / max(len(e) - Xn.shape[1], 1)))
    p["reference_scale"] = s_ref
    if p["sd"].get("scale") is None:
        p["sd"]["scale"] = s_ref
    if p["loading"].get("scale") is None:
        p["loading"]["scale"] = s_ref
    if p["sd"]["family"] not in SD_FAMILIES:
        raise ValueError("prior : famille d'ecart-type inconnue %r (attendu %s)"
                         % (p["sd"]["family"], ", ".join(SD_FAMILIES)))
    names = [tm["name"] for tm in terms] + ["residual"]
    unknown = set(p.get("terms") or {}) - set(names)
    if unknown:
        raise ValueError("prior : terme(s) inconnu(s) %s ; termes du modele : %s"
                         % (sorted(unknown), names))
    return p


def _term_spec(p, name):
    """Effective specification of one term: the defaults, then the term's overrides."""
    sd, lkj, load = dict(p["sd"]), p["lkj"], dict(p["loading"])
    override = (p.get("terms") or {}).get(name) or {}
    if "sd" in override:
        sd.update(override["sd"])
        if sd.get("scale") is None:
            sd["scale"] = p["reference_scale"]
    if "lkj" in override:
        lkj = float(override["lkj"])
    if "loading" in override:
        load.update(override["loading"])
    return sd, float(lkj), load


def _prior_blocks(terms, res, p):
    """Cut theta into prior blocks.

    Each block is (kind, start, length, t, name, specification). The kind is
    one of "sd", "us", "loading", "tanh" and "generic". Returns the blocks,
    the total length of theta and the list of terms that fall back on the
    generic normal prior.
    """
    blocks, o, rare = [], 0, []

    def sigma_blocks(o, struct, t, rank, name):
        sd, lkj, load = _term_spec(p, name)
        ns = n_params(struct, t, rank)
        if not ns:
            return []
        if struct in ("iid", "diag") or (struct == "us" and t == 1):
            return [("sd", o, ns, t, name, sd)]
        if struct == "us":
            return [("us", o, ns, t, name, dict(sd=sd, lkj=lkj))]
        if struct == "fa":
            nl = n_loadings(t, rank)
            return [("loading", o, nl, t, name, load), ("sd", o + nl, t, t, name, sd)]
        if struct == "rr":
            return [("loading", o, ns, t, name, load)]
        rare.append("%s (%s)" % (name, struct))
        return [("generic", o, ns, t, name, p["generic"])]

    def level_blocks(o, kind, parts, name, total):
        if total <= 0:
            return []
        if kind == "sep":
            out, oo = [], o
            for (k, _q) in parts or []:
                pk = n_level_params(k)
                if pk:
                    out += level_blocks(oo, k, None, name, pk)
                oo += pk
            return out
        if kind in LEVELS_TANH and p["level"] == "uniform":
            return [("tanh", o, total, 0, name, None)]
        return [("generic", o, total, 0, name, p["generic"])]

    for tm in terms:
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        tot = term_n_params(tm)
        blocks += sigma_blocks(o, tm["struct"], tm["t"], tm["rank"], tm["name"])
        kind = level_kind(tm)
        blocks += level_blocks(o + ns, kind, tm.get("lvl_parts"), tm["name"], tot - ns)
        o += tot
    for sec in res_sections(res, 0):
        tot = sec_n_params(sec)
        if tot == 0:
            continue
        ns = n_params(sec["struct"], sec["t"], sec["rank"])
        blocks += sigma_blocks(o, sec["struct"], sec["t"], sec["rank"], "residual")
        blocks += level_blocks(o + ns, sec.get("lvl", "id"), sec.get("lvl_parts"), "residual",
                               tot - ns)
        o += tot
    return blocks, o, rare


def make_log_prior(terms, res, p):
    """log p(theta), Jacobian included, as a JAX function of the whole theta."""
    blocks, ntot, rare = _prior_blocks(terms, res, p)

    def us_phi(th, t):
        # Natural parameters of a us block: standard deviations, then the
        # lower-triangle correlations.
        S = build_sigma(th, "us", t, 0)
        sd = jnp.sqrt(jnp.diagonal(S))
        R = S / jnp.outer(sd, sd)
        ii, jj = np.tril_indices(t, -1)
        return jnp.concatenate([sd, R[ii, jj]])

    def lp(theta):
        tot = 0.0
        for (kind, o, ln, t, _name, spec) in blocks:
            th = theta[o:o + ln]
            if kind == "sd":
                tot = tot + jnp.sum(_log_sd(jnp.exp(th), spec) + th)
            elif kind == "us":
                phi = us_phi(th, t)
                sd = phi[:t]
                S = build_sigma(th, "us", t, 0)
                R = S / jnp.outer(sd, sd)
                J = jax.jacfwd(lambda z: us_phi(z, t))(th)
                tot = (tot + jnp.sum(_log_sd(sd, spec["sd"]))
                       + (spec["lkj"] - 1.0) * jnp.linalg.slogdet(R)[1]
                       + jnp.linalg.slogdet(J)[1])
            elif kind == "loading":
                tot = tot - 0.5 * jnp.sum((th / float(spec["scale"])) ** 2)
            elif kind == "tanh":
                tot = tot + jnp.sum(_log_sech2(th))
            else:
                tot = tot - 0.5 * jnp.sum((th / float(spec)) ** 2)
        return tot

    return lp, blocks, rare


# ==============================================================================
# Sampling
# ==============================================================================
def nested_rhat(x, K):
    """Nested R-hat (Margossian et al. 2024).

    x has shape (C, D) with C = K * M chains. The M chains of one superchain
    are contiguous and started from the same point.
    """
    x = np.asarray(x, dtype=np.float64)
    C, D = x.shape
    M = C // K
    if K < 2 or D < 2:
        return np.nan
    x = x[:K * M].reshape(K, M, D)
    chain_means = x.mean(axis=2)                              # (K, M)
    super_means = chain_means.mean(axis=1)                    # (K,)
    B = super_means.var(ddof=1)
    Bk = chain_means.var(axis=1, ddof=1) if M > 1 else np.zeros(K)
    Wk = x.var(axis=2, ddof=1).mean(axis=1)
    W = np.mean(Bk + Wk)
    return float(np.sqrt(1.0 + B / W)) if W > 0 else np.nan


def _summarise(x, K):
    """Mean, standard deviation, quantiles, R-hat and ESS of x (chains, draws)."""
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    q = np.quantile(flat, [0.025, 0.5, 0.975])
    ok = np.all(np.isfinite(flat)) and flat.std() > 0 and x.shape[1] >= 4
    rhat = float(potential_scale_reduction(jnp.asarray(x))) if ok else np.nan
    ess = float(effective_sample_size(jnp.asarray(x))) if ok else np.nan
    return dict(mean=float(flat.mean()), sd=float(flat.std(ddof=1)), q025=float(q[0]),
                median=float(q[1]), q975=float(q[2]), rhat=rhat,
                nested_rhat=nested_rhat(x, K) if ok else np.nan, ess=ess)


def run_sampler(logdens, z_init, warmup, draws, seed=0, target_accept=0.651,
                   max_leapfrog=1000, verbose=True):
    """HMC adapted by ChEES over all chains, then draws in slices.

    logdens is the JAX log density of a vector z of length d. z_init has shape
    (C, d). Returns (z of shape (C, D, d), diagnostics). Each slice is a scan
    loop compiled once. Timings are taken after block_until_ready because JAX
    runs asynchronously: a clock read when the call returns only measures the
    dispatch of the computation.
    """
    import blackjax
    import optax
    C, d = z_init.shape
    k_w, k_s = jax.random.split(jax.random.PRNGKey(seed))
    adapt = blackjax.chees_adaptation(logdens, num_chains=C, target_acceptance_rate=target_accept,
                                      max_leapfrog_steps=max_leapfrog)
    t1 = time.time()
    (states, params), _ = adapt.run(k_w, jnp.asarray(z_init), step_size=0.3,
                                    optim=optax.adam(0.025), num_steps=warmup)
    jax.block_until_ready(states)
    t_warmup = time.time() - t1
    step_size = float(np.asarray(params["step_size"]))
    if verbose:
        print("  [bayes] chauffe ChEES : %.1f s (compilation comprise) | pas %.3g"
              % (t_warmup, step_size), flush=True)
    kernel = blackjax.dynamic_hmc(logdens, **params)
    n_slices = max(1, min(10, draws // 10))
    slice_len = -(-draws // n_slices)

    @jax.jit
    def run_slice(key, states):
        def one_step(states, k):
            states, info = jax.vmap(kernel.step)(jax.random.split(k, C), states)
            return states, (states.position, info.is_divergent, info.acceptance_rate,
                            info.num_integration_steps)
        return jax.lax.scan(one_step, states, jax.random.split(key, slice_len))

    pos, div, acc, nst = [], [], [], []
    t1, done = time.time(), 0
    for i in range(n_slices):
        states, (p_, dv, ac, ns) = run_slice(jax.random.fold_in(k_s, i), states)
        jax.block_until_ready(p_)
        take = min(slice_len, draws - done)
        pos.append(np.asarray(p_)[:take]); div.append(np.asarray(dv)[:take])
        acc.append(np.asarray(ac)[:take]); nst.append(np.asarray(ns)[:take])
        done += take
        if verbose:
            elapsed = time.time() - t1
            print("  [bayes] tirages %5d / %d (%3.0f %%) | %.1f s, reste ~%.0f s | %d divergence(s)"
                  % (done, draws, 100.0 * done / draws, elapsed, elapsed / done * (draws - done),
                     int(sum(x.sum() for x in div))), flush=True)
    t_draws = time.time() - t1
    z = np.concatenate(pos, axis=0).transpose(1, 0, 2)            # (C, D, d)
    div = np.concatenate(div, axis=0)
    acc = np.concatenate(acc, axis=0)
    nst = np.concatenate(nst, axis=0)
    diag = dict(algorithm="ChEES-HMC", divergences=int(div.sum()), acceptance=float(acc.mean()),
                step_size=step_size, mean_leapfrog_steps=float(nst.mean()),
                seconds_warmup=t_warmup, seconds_draws=t_draws)
    return z, diag


def prepare(terms, res, y, X, prior=None):
    """Resolved prior, with log p(theta) and -2 logL as JAX functions. The
    posterior is evaluated by the dense engine, which needs K."""
    from .checks import require_K
    require_K(terms, "voie bayesienne")
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    p = _resolve_prior(prior, terms, res, y, X)
    log_prior, _blocks, rare = make_log_prior(terms, res, p)
    f, Zr, Zs = make_neg2(terms, res, y, X)
    Zr_j = [None if z_ is None else jnp.asarray(z_) for z_ in Zr]
    return dict(p=p, log_prior=log_prior, rare=rare, f=f, Zr=Zr_j, Zs=Zs, y=y, X=X)


def posterior_mode(prep, theta0, fixed_idx=None, floor=-12.0, ceil=12.0, maxiter=2000,
                    eps=1e-4, verbose=True):
    """Posterior mode of theta, and the Hessian of -log p(theta | y) at that mode.

    The mode is the starting point and the whitening centre of the Bayesian
    route without a REML fit (rx_bayes(start = "map"), the default). The prior
    moves the mode away from the boundaries (zero variance, correlation of
    +-1), where the REML Hessian is not positive definite. The Hessian is
    taken by central differences of the gradient (2 d gradients). The
    derivative of the likelihood goes through a custom_vjp, which JAX cannot
    differentiate in forward mode.
    Returns (theta_map, Hu on the free parameters, info).
    """
    from scipy.optimize import minimize
    lp, f, Zr = prep["log_prior"], prep["f"], prep["Zr"]
    th0 = np.asarray(theta0, dtype=np.float64)
    ntot = len(th0)
    free = np.ones(ntot, dtype=bool)
    if fixed_idx:
        free[np.asarray(fixed_idx, dtype=int)] = False
    idx = np.flatnonzero(free)
    base, jidx = jnp.asarray(th0), jnp.asarray(idx)
    n = len(prep["y"])

    def U(tf):                                   # -log p(theta | y), up to a constant
        th = base.at[jidx].set(tf)
        return 0.5 * f(th, Zr) - lp(th)

    vg = jax.jit(jax.value_and_grad(U))

    def fun(tf):                                 # scaled by n, as in the REML fit
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
    info = dict(iterations=int(r.nit), message=str(r.message), success=bool(r.success),
                log_posterior=-float(r.fun) * n, seconds=time.time() - t1)
    if verbose:
        print("  [bayes] mode a posteriori : %d iteration(s), %.1f s (%s)"
              % (info["iterations"], info["seconds"], info["message"]), flush=True)
    return th, Hu, info


def sample_posterior(terms, res, y, X, theta_hat, H=None, Hu=None, prior=None, chains=16,
                     superchains=4, warmup=300, draws=100, seed=0, target_accept=0.651,
                     max_leapfrog=1000, fixed_idx=None, expressions=(), n_effects=100,
                     pev=(), lag=None, verbose=True):
    """Sample the posterior distribution of theta around theta_hat.

    theta_hat : centre of the whitening and of the starting points. It is the
                posterior mode (posterior_mode) or the REML optimum.
    H         : Hessian of -2logL at theta_hat (REML optimum). 2 H^-1 whitens.
    Hu        : Hessian of -log p(theta | y) on the free parameters (posterior
                mode). Hu^-1 whitens. It takes precedence over H. When neither
                is positive definite, the posterior mode is searched from
                theta_hat and its Hessian is used for whitening.
    chains, superchains : C chains grouped in K superchains (C a multiple of K).
    warmup, draws : ChEES adaptation iterations, and draws kept PER CHAIN.
    fixed_idx : indices (0-based) held at their value in theta_hat.
    expressions : [(name, string)] in the style of vpredict, evaluated at each draw.
    n_effects  : number of draws used for the posterior moments of beta and the BLUPs.
    pev       : terms whose effects get a posterior standard deviation.
    """
    if not available():
        raise RuntimeError("la voie bayesienne exige blackjax : pip install blackjax "
                           "(ou rx_install_python(bayes = TRUE) depuis R)")
    if lag is not None:
        raise ValueError("la voie bayesienne n'accepte pas encore de decalage spatial (rx_lag)")
    K = max(1, int(superchains))
    if chains % K:
        raise ValueError("chains (%d) doit etre un multiple de superchains (%d)" % (chains, K))

    t0 = time.time()
    prep = prepare(terms, res, y, X, prior)
    y, X, p = prep["y"], prep["X"], prep["p"]
    log_prior, rare, f, Zs = prep["log_prior"], prep["rare"], prep["f"], prep["Zs"]
    if rare and verbose:
        print("  [bayes] loi normale sur theta (echelle interne) pour : %s" % ", ".join(rare),
              flush=True)
    th_hat = np.asarray(theta_hat, dtype=np.float64)
    ntot = len(th_hat)
    free = np.ones(ntot, dtype=bool)
    if fixed_idx:
        free[np.asarray(fixed_idx, dtype=int)] = False
    idx = np.flatnonzero(free)
    d = len(idx)

    # Whitening. Lw Lw' is the Laplace covariance: Hu^-1 at the posterior mode,
    # or 2 H^-1 at the REML optimum. A Hessian that is not positive definite
    # (REML optimum on a boundary) triggers a search for the posterior mode
    # from theta_hat.
    def laplace_factor(M, k):
        try:
            w, U_ = np.linalg.eigh(0.5 * (M + M.T))
        except np.linalg.LinAlgError:
            return None
        if not np.all(w > 1e-10 * max(abs(w).max(), 1.0)):
            return None
        return U_ * np.sqrt(k / w)

    Lw = None
    if Hu is not None:
        Lw = laplace_factor(np.asarray(Hu), 1.0)
    elif H is not None:
        Lw = laplace_factor(np.asarray(H)[np.ix_(idx, idx)], 2.0)
    if Lw is None:
        if verbose:
            print("  [bayes] Hessien absent ou non defini positif : recherche du mode a posteriori",
                  flush=True)
        th_hat, Hm, _ = posterior_mode(prep, th_hat, fixed_idx=fixed_idx, verbose=verbose)
        Lw = laplace_factor(Hm, 1.0)
    if Lw is None:
        Lw = 0.1 * np.eye(d)
        if verbose:
            print("  [bayes] Hessien du mode non defini positif : blanchiment par 0,1 I", flush=True)
    base = jnp.asarray(th_hat)
    jidx = jnp.asarray(idx)
    center = jnp.asarray(th_hat[idx])
    Lw_j = jnp.asarray(Lw)
    Zr_j = prep["Zr"]

    def theta_of_z(z):
        return base.at[jidx].set(center + Lw_j @ z)

    def logdens(z):
        th = theta_of_z(z)
        v = log_prior(th) - 0.5 * f(th, Zr_j)
        return jnp.where(jnp.isfinite(v), v, -jnp.inf)

    # Starting points: one per superchain, overdispersed, shared by its M chains.
    rng = np.random.default_rng(seed)
    M = chains // K
    z_sc = 2.0 * rng.standard_normal((K, d))
    z_init = np.repeat(z_sc, M, axis=0)
    if verbose:
        print("  [bayes] ChEES-HMC : %d chaines (%d super-chaines x %d), %d chauffe + %d tirages, "
              "%d parametre(s) libre(s) | preparation %.1f s"
              % (chains, K, M, warmup, draws, d, time.time() - t0), flush=True)
    z, diag = run_sampler(logdens, z_init, warmup, draws, seed=seed,
                             target_accept=target_accept, max_leapfrog=max_leapfrog,
                             verbose=verbose)
    C, D = z.shape[0], z.shape[1]
    tf = th_hat[idx][None, None, :] + np.einsum("ij,cdj->cdi", Lw, z)
    th_draws = np.tile(th_hat, (C, D, 1))
    th_draws[:, :, idx] = tf

    # Variance components and expressions, draw by draw.
    t1 = time.time()
    flat_draws = th_draws.reshape(C * D, ntot)
    try:
        comp_fn = jax.jit(jax.vmap(lambda th: components_from_theta(th, terms, res)))
        comps = np.asarray(comp_fn(jnp.asarray(flat_draws)))
    except Exception:            # components that cannot be vectorised (numpy ratios of levels)
        one = jax.jit(lambda th: components_from_theta(th, terms, res))
        comps = np.stack([np.asarray(one(jnp.asarray(t_))) for t_ in flat_draws])
    names = component_names(terms, res)
    vp = []
    for (_name, expr) in expressions:
        env = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "abs": np.abs}
        env.update({"V%d" % (i + 1): comps[:, i] for i in range(comps.shape[1])})
        vp.append(np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=float)  # noqa: S307
                  * np.ones(C * D))
    vp = np.stack(vp, axis=1) if vp else np.zeros((C * D, 0))
    summary = []
    for i, name in enumerate(names):
        summary.append(dict(name=name, **_summarise(comps[:, i].reshape(C, D), K)))
    for j, (name, expr) in enumerate(expressions):
        summary.append(dict(name=name, expression=expr, **_summarise(vp[:, j].reshape(C, D), K)))
    if verbose:
        print("  [bayes] composantes et expressions sur %d tirages : %.1f s"
              % (C * D, time.time() - t1), flush=True)

    def max_finite(key):
        v = [r[key] for r in summary if np.isfinite(r[key])]
        return float(max(v)) if v else np.nan

    # beta and the random effects, by Rao-Blackwellisation.
    effects = None
    if n_effects and n_effects > 0:
        from .effects import dense_effects
        t1 = time.time()
        k = int(min(n_effects, C * D))
        picks = np.unique(np.linspace(0, C * D - 1, k).round().astype(int))
        betas, vbetas, blup_draws, pev_draws = [], [], {}, {}
        for s_ in picks:
            e = dense_effects(flat_draws[s_], terms, Zs, res, y, X, blups=True,
                              pev=list(pev) or False)
            betas.append(e["beta"])
            vbetas.append(e["vbeta"])
            for nm, u in e["blups"].items():
                blup_draws.setdefault(nm, []).append(u)
            for nm, pv in (e.get("pev") or {}).items():
                pev_draws.setdefault(nm, []).append(pv)
        Bt = np.stack(betas)
        bcov = np.mean(vbetas, axis=0) + (np.cov(Bt.T) if len(picks) > 1 else 0.0)
        effects = dict(n=len(picks), beta_mean=Bt.mean(axis=0), beta_cov=np.atleast_2d(bcov),
                       blup_mean={nm: np.mean(v, axis=0) for nm, v in blup_draws.items()},
                       blup_sd={nm: np.sqrt(np.mean(pev_draws[nm], axis=0)
                                            + np.var(blup_draws[nm], axis=0))
                                for nm in pev_draws},
                       seconds=time.time() - t1)
        if verbose:
            print("  [bayes] moments de beta et des BLUP sur %d tirages : %.1f s"
                  % (len(picks), effects["seconds"]), flush=True)

    diag.update(chains=C, superchains=K, warmup=int(warmup), draws=D,
                rhat_max=max_finite("rhat"), nested_rhat_max=max_finite("nested_rhat"),
                ess_min=float(min([r["ess"] for r in summary if np.isfinite(r["ess"])] or [np.nan])),
                seconds=time.time() - t0, n_free=int(d))
    prior_out = dict(sd=p["sd"], lkj=p["lkj"], loading=p["loading"], level=p["level"],
                     generic=p["generic"], reference_scale=p["reference_scale"],
                     terms=p.get("terms") or {}, theta_normal=rare)
    return dict(theta=th_draws, components=comps, component_names=names,
                expressions=vp, expression_names=[n for n, _ in expressions],
                summary=summary, diagnostics=diag, prior=prior_out, effects=effects)
