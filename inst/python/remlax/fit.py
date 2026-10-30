"""REML fit: optimisation, BLUPs, standard errors, diagnostics.

fit_reml() is the entry point; _fit_model() fits one model as a sequence
of named steps (engine and objective, starting point, optimisation,
finishing, result). The optimiser works on an unconstrained parametrisation
(structures.py): the wide bounds set here only keep a variance from escaping
to exp(-inf), where the gradient vanishes and the point looks like an optimum.

Related modules: start.py (starting values), checks.py (checks of the model),
effects.py (beta and BLUPs at a given theta), diagnostics.py (Hessian and
convergence criteria).
"""
from . import _x64  # noqa: F401
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .layout import level_kind, split_theta, n_theta, term_n_params, res_sections, sec_n_params
from .model import make_objective, assemble_V, aliased_entries
from .structures import build_sigma, n_params, sigma_loadings
from .levels import level_params_report
from .lag import (prepare_lag, lag_n_params, ystar as lag_ystar, lag_terms,
                  lag_report, theta_from_psi)
from .start import initial_theta, loading_bounds, _range_sweep
from .checks import validate, require_K
from .effects import dense_effects
from .diagnostics import hessian_fd, diagnostic, gradient_diagnostic


def _chrono(name):
    """Stage mark (chrono.py)."""
    from . import chrono
    chrono.mark(name)


# Relative gradient above which an MME fit not declared converged is resumed
# by the dense engine (converged fits are under 1e-6), and the largest n for
# which the dense engine still fits in memory (dev/notes.md).
GRAD_REL_DENSE_RESTART = 1e-5
N_MAX_DENSE_RESTART = 20000


_BLAS_ONE_THREAD = [False]        # NumpyBlasOneThread context already entered


def fit_reml(terms, res, y, X, *args, weights=None, **kwargs):
    """REML fit (options: see _fit_model).

    engine = "auto" or optimizer = "auto" pick the MME engine and AI-REML when
    the model is within the scope of the MME engine, the dense engine and
    L-BFGS-B otherwise (low-memory above n = 5000).

    weights: prior weights w_i > 0 of the observations (None: none). The residual
    becomes R = W^-1/2 R0 W^-1/2, W = diag(w), R0 the declared residual. With
    an iid residual whose variance is fixed at 1 (fixed_idx), R = diag(1/w):
    known residual variances, as in a two-stage analysis. Multiplying the rows
    of y, X and Z by sqrt(w_i) brings the residual back to R0, so no engine
    needs to know the weights. Only the likelihood changes, by a constant:
    -2logL(y) = -2logL(y*) - sum log w_i. beta, BLUPs, PEV and Hessian are
    unchanged; P y and V^-1 are brought back to the original scale
    (P y = W^1/2 P* y*, V^-1 = W^1/2 V*^-1 W^1/2).
    """
    kwargs = _resolve_auto(terms, res, y, kwargs)
    if kwargs.get("engine") == "mme" and not _BLAS_ONE_THREAD[0]:
        # MME engine: thin NumPy products, heavy work in LAPACK (SciPy); the two
        # OpenBLAS libraries must not compete for the cores (device.py).
        from .device import NumpyBlasOneThread
        _BLAS_ONE_THREAD[0] = True
        try:
            with NumpyBlasOneThread():
                return fit_reml(terms, res, y, X, *args, weights=weights, **kwargs)
        finally:
            _BLAS_ONE_THREAD[0] = False
    if weights is None:
        return _fit_model(terms, res, y, X, *args, **kwargs)
    return _fit_weighted(terms, res, y, X, weights, args, kwargs)


def _resolve_auto(terms, res, y, kwargs):
    """kwargs with engine = "auto" and optimizer = "auto" replaced by the
    engine and optimiser chosen for the model."""
    if kwargs.get("engine") == "auto" or kwargs.get("optimizer") == "auto":
        kwargs = dict(kwargs)
        from .mme import out_of_scope as _hp
        in_mme = _hp(terms, res, kwargs.get("lag")) is None
        if kwargs.get("engine") == "auto":
            kwargs["engine"] = "mme" if in_mme else "dense"
            if kwargs["engine"] == "dense" and len(y) > 5000 and kwargs.get("memory", "standard") == "standard":
                kwargs["memory"] = "low"
        if kwargs.get("optimizer") == "auto":
            kwargs["optimizer"] = "ai" if kwargs["engine"] == "mme" else "lbfgsb"
    return kwargs


def _fit_weighted(terms, res, y, X, weights, args, kwargs):
    """Fit on rows scaled by sqrt(w), then the likelihood, P y and V^-1 brought
    back to the original scale (see fit_reml)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != len(y) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("poids : %d valeurs finies et > 0 attendues" % len(y))
    sw = np.sqrt(w)
    y2 = np.asarray(y, dtype=np.float64) * sw
    X2 = np.asarray(X, dtype=np.float64) * sw[:, None]
    terms2 = [dict(tm, zx=np.asarray(tm["zx"], dtype=np.float64)
                   * sw[np.asarray(tm["zi"], dtype=np.int64)]) for tm in terms]
    out = _fit_model(terms2, res, y2, X2, *args, **kwargs)
    c = -float(np.sum(np.log(w)))                  # constant of -2logL
    for k in ("neg2_reml",):
        if k in out:
            out[k] = out[k] + c
    for k in ("logLik", "logLik_asreml"):
        if k in out:
            out[k] = out[k] - 0.5 * c
    if "Py" in out:
        out["Py"] = np.asarray(out["Py"]) * sw
    if "Vi" in out:
        out["Vi"] = np.asarray(out["Vi"]) * sw[:, None] * sw[None, :]
    out["weighted"] = True
    return out


def _fit_model(terms, res, y, X, theta_init=None, maxiter=3000,
               floor=-12.0, ceil=12.0, verbose=True, hessian=True,
               blups=True, pev=False, tol=1e-14, polish=25, check=True,
               n_restarts=0, restart_sd=0.5, seed=0, fixed_idx=None, lag=None,
               optimizer="lbfgsb", engine="dense", inversion="auto",
               memory="standard", finish="newton", slow_stop=(100, 0.05)):
    """REML fit of one model.

    finish: "newton" (default: L-BFGS-B to its tolerance, then Newton
      polishing) or "trust" (finition.py): L-BFGS-B stops as soon as its gain over
      the last slow_stop[0] iterations falls below slow_stop[1] (units of
      -2logL), then a trust-region Newton finishes, handling flat directions
      and vanishing variances.

    memory: "standard" (default) or "low" (dense engine only,
      econome.py). Same likelihood, gradient and AI matrix; the low-memory mode
      keeps fewer n x n matrices and no full-height incidence on the device.

    optimizer: "lbfgsb" (default) or "ai" (damped AI-REML, ai.py). Polishing,
      diagnostics and outputs are the same.

    engine: "dense" (V formed, default), "mme" (mixed model equations,
      mme.py; inversion "auto", "dense", "sparse" or "hybrid") or "blocks" (blocs.py:
      automatic split into blocks linked by global terms, V = A + U U' block by
      block). All give the same likelihood in theta. A model outside the scope
      of the MME or blocks engine falls back to the dense engine with a message.

    lag: spatial lag (lag.py). theta then ends with the free entries of Psi,
      the objective is the structural criterion, and beta, the BLUPs and the
      tests refer to y* = (I - M) y at the final point.

    n_restarts: restarts from perturbed points, keeping the best optimum. This
      is the only way to detect a local optimum: the Newton decrement measures
      the ascent available locally and is zero on top of a secondary hill.

    The steps, in order: engine and objective, starting point, compilation,
    optimisation, finishing (Newton polishing or trust region), then the
    result: likelihood and timings, identification, sigmas, fixed and random
    effects, Hessian and diagnostics, and the dense restart of a failed MME fit.
    """
    n = len(y)
    if check:
        validate(terms, res, y, X)
    lag = prepare_lag(lag, y, X)
    if n_restarts > 0:
        return _fit_with_restarts(
            terms, res, y, X, n_restarts, restart_sd, seed,
            dict(theta_init=theta_init, maxiter=maxiter, floor=floor, ceil=ceil,
                 verbose=verbose, hessian=hessian, blups=blups, pev=pev, tol=tol,
                 polish=polish, check=False, engine=engine, inversion=inversion,
                 memory=memory, finish=finish, slow_stop=slow_stop,
                 fixed_idx=fixed_idx, lag=lag, optimizer=optimizer))

    eng = _build_engine(terms, res, y, X, lag, engine, inversion, memory, optimizer,
                        floor, verbose)
    floor, engine, memory = eng.floor, eng.engine, eng.memory
    th0, p_cov, p = _start_theta(theta_init, terms, res, y, X, lag)
    _check_start(th0, terms, res, X, n, lag, p_cov, eng)
    obj = _Objective(eng, th0, p, verbose,
                     slow_on=(finish == "trust" and optimizer == "lbfgsb"),
                     slow_stop=slow_stop)
    bornes, th0 = _bounds(th0, p, floor, ceil, terms, res, y, fixed_idx)

    # maxiter = 0 means "do not optimise" and is short-circuited: L-BFGS-B
    # given maxiter = 0 still moves theta (dev/notes.md).
    if int(maxiter) <= 0:
        theta, r = np.asarray(th0, dtype=np.float64), None
    else:
        theta, r = _optimise(obj, th0, bornes, theta_init, terms, res, y, X, n, lag,
                             eng, optimizer, maxiter, tol, floor, fixed_idx, slow_stop,
                             verbose)
    t_optim = time.time() - obj.hist["t0"]
    _chrono("optimisation")

    polish_eff, H_cache = int(polish), None
    if int(maxiter) > 0 and optimizer == "ai" and bool(getattr(r, "success", False)):
        theta, r, polish_eff, H_cache = _check_ai_saddle(obj, theta, r, bornes, floor, ceil,
                                                         hessian, maxiter, tol, polish,
                                                         verbose)
    if finish == "trust" and int(maxiter) > 0 and polish_eff > 0:
        theta = _finish_trust_region(obj, theta, bornes, verbose)
        polish_eff = 0
    theta, n_polish = _newton_polish(obj, theta, polish_eff, p, floor, ceil, fixed_idx,
                                     verbose)

    f_end, g_end = obj.fun_jac(theta)
    _chrono("finish")
    out = _result_core(obj, eng, theta, r, f_end, g_end, X, y, n, p, maxiter, floor, ceil,
                       optimizer, t_optim, n_polish)
    n_plat = _add_identification(out, terms, n, p, verbose)
    y_fit = _add_lag_report(out, theta, lag, p_cov, X, y, f_end, verbose)
    _add_degenerate(out, theta, terms, p_cov, floor, fixed_idx, verbose)
    _add_sigmas(out, theta, terms, res, n)
    _add_effects(out, eng, theta, terms, res, y, y_fit, X, n, blups, hessian, pev)
    _chrono("effects")
    _add_hessian_diagnostics(out, obj, theta, g_end, H_cache, hessian, floor, ceil,
                             fixed_idx, n_plat)
    _chrono("hessian")
    return _dense_restart_if_needed(
        out, terms, res, y, X, engine, maxiter, verbose,
        dict(maxiter=maxiter, floor=floor, ceil=ceil, verbose=verbose, hessian=hessian,
             blups=blups, pev=pev, tol=tol, polish=polish, check=False, n_restarts=0,
             fixed_idx=fixed_idx, lag=lag, optimizer=optimizer, engine="dense",
             inversion=inversion, memory=memory, finish=finish,
             slow_stop=slow_stop))


# ------------------------------------------------------------------------------
# Restarts
# ------------------------------------------------------------------------------

def _fit_with_restarts(terms, res, y, X, n_restarts, restart_sd, seed, kw):
    """A base fit, then n_restarts fits from theta perturbed by N(0, restart_sd)
    (fixed entries kept); the best optimum is returned with the count of
    restarts that improved on the base."""
    floor, ceil, fixed_idx, verbose = kw["floor"], kw["ceil"], kw["fixed_idx"], kw["verbose"]
    base = fit_reml(terms, res, y, X, **kw)
    rng = np.random.default_rng(seed)
    best, n_better = base, 0
    kw_restart = dict(kw, verbose=False)
    for i in range(int(n_restarts)):
        th = np.clip(base["theta"] + rng.normal(0, restart_sd, len(base["theta"])),
                     floor, ceil)
        try:
            if fixed_idx:
                th[np.array(fixed_idx)] = base["theta"][np.array(fixed_idx)]
            cand = fit_reml(terms, res, y, X, **dict(kw_restart, theta_init=th))
        except Exception:
            continue
        if np.isfinite(cand["neg2_reml"]) and cand["neg2_reml"] < best["neg2_reml"] - 1e-8:
            best, n_better = cand, n_better + 1
    best["n_restarts"] = int(n_restarts)
    best["restart_gain"] = float(base["neg2_reml"] - best["neg2_reml"])
    best["restart_better"] = n_better
    if verbose and n_better:
        print("  [restarts] %d redemarrage(s) ont trouve mieux : gain %.4g en -2logL"
              % (n_better, best["restart_gain"]), flush=True)
    return best


# ------------------------------------------------------------------------------
# Engine and objective
# ------------------------------------------------------------------------------

def _build_engine(terms, res, y, X, lag, engine, inversion, memory, optimizer, floor,
                  verbose):
    """The engine actually used, after fallbacks, and its objective.

    Returns a namespace with engine, memory, floor (raised for the MME
    engine), fun_jac (-2logL and gradient), fun_sc (the same divided by sc,
    the scaled objective minimised by L-BFGS-B), sc, Zs (dense incidences, or
    None), and the engine objects M_mme, fga_mme, plan_eco, plan_blocs."""
    from types import SimpleNamespace
    n = len(y)
    eng = SimpleNamespace(engine=engine, memory=memory, floor=floor, Zs=None,
                          M_mme=None, fga_mme=None, plan_eco=None, plan_blocs=None)
    if eng.engine == "mme":
        _mme_fallback(eng, terms, res, lag, verbose)
    if eng.engine == "mme":
        # Variance floor of the MME engine, which uses Sigma^-1: a variance of
        # exp(2 x -12) = 4e-11 leaves C too ill-conditioned for log|C| to keep
        # its digits. Floor at 1e-8 times the least-squares residual variance.
        # Raised only once the engine is known to stay MME.
        _res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        _vy = float(_res @ _res) / max(n - X.shape[1], 1)
        eng.floor = max(floor, 0.5 * float(np.log(1e-8 * _vy)))
    if eng.engine == "blocks":
        _blocs_fallback(eng, terms, res, y, X, lag, optimizer, verbose)
    if eng.engine == "mme":
        from .mme import make_fga_mme
        eng.fga_mme, eng.fun_jac, eng.M_mme = make_fga_mme(terms, res, y, X,
                                                           inversion=inversion,
                                                           verbose=verbose)
        eng.sc = float(n)

        def fun_sc(theta, _fj=eng.fun_jac, _sc=eng.sc):
            v_, g_ = _fj(theta)
            return v_ / _sc, g_ / _sc
        eng.fun_sc = fun_sc
    elif eng.engine == "blocks":
        require_K(terms, "moteur par blocs")
        from .blocs import make_objective_blocks
        eng.fun_sc, eng.fun_jac, eng.sc = make_objective_blocks(eng.plan_blocs)
    else:
        require_K(terms, "moteur dense")
        _dense_objective(eng, terms, res, y, X, lag, verbose)
    return eng


def _mme_fallback(eng, terms, res, lag, verbose):
    """Outside the scope of the MME engine: dense engine, unless a term is
    given only by its precision Kinv, which the dense engine cannot use."""
    from .mme import out_of_scope
    reason = out_of_scope(terms, res, lag)
    if not reason:
        return
    if any(tm.get("Kinv") is not None and tm.get("LK") is None for tm in terms):
        raise ValueError("moteur MME hors perimetre (%s), et le moteur dense ne sait pas "
                         "utiliser une precision Kinv sans facteur de K : il ignorerait la "
                         "parente. Fournir K." % reason)
    if verbose:
        print("  [moteur] MME hors perimetre (%s) -> moteur dense" % reason, flush=True)
    eng.engine = "dense"


def _blocs_fallback(eng, terms, res, y, X, lag, optimizer, verbose):
    """The block plan, or the low-memory dense engine when the model is out of
    scope, splits into a single block, or asks for AI-REML."""
    from .blocs import out_of_scope as hp_blocs, BlockPlan
    reason = hp_blocs(terms, res, lag)
    if reason is None:
        eng.plan_blocs = BlockPlan(terms, res, y, X, verbose=verbose)
        if eng.plan_blocs.n_blocks < 2:
            reason = "aucun decoupage en blocs (un seul bloc)"
    if reason is None and optimizer == "ai":
        reason = "optimiseur AI non pris en charge par le moteur par blocs"
    if reason:
        if verbose:
            print("  [moteur] blocs : %s -> moteur dense econome" % reason, flush=True)
        eng.engine, eng.memory = "dense", "low"


def _dense_objective(eng, terms, res, y, X, lag, verbose):
    """Objective of the dense engine, standard or low-memory (econome.py, not
    available with a spatial lag)."""
    if eng.memory == "low" and lag is not None:
        if verbose:
            print("  [memoire] mode econome sans decalage spatial -> mode standard",
                  flush=True)
        eng.memory = "standard"
    if eng.memory == "low":
        from .econome import LowMemoryPlan, make_objective_low_memory
        eng.plan_eco = LowMemoryPlan(terms, res, y, X)
        eng.fun_sc, eng.fun_jac, eng.sc = make_objective_low_memory(eng.plan_eco)
        if verbose:
            print("  [memoire] econome : %s" % eng.plan_eco.summary(), flush=True)
    elif eng.memory == "standard":
        eng.fun_sc, eng.fun_jac, eng.sc, eng.Zs = make_objective(terms, res, y, X, lag=lag)
    else:
        raise ValueError("memory : 'standard' ou 'low', pas %r" % (eng.memory,))


# ------------------------------------------------------------------------------
# Starting point
# ------------------------------------------------------------------------------

def _start_theta(theta_init, terms, res, y, X, lag):
    """The starting theta, the number of covariance parameters p_cov and the
    total p (p_cov plus the free entries of Psi). Missing entries (NaN) of
    theta_init take the automatic start, so that fixing one parameter does not
    require a first call to learn the others."""
    p_cov = n_theta(terms, res)
    if theta_init is not None:
        th0 = np.array(theta_init, dtype=np.float64)
        if np.isnan(th0).any():
            _ti = initial_theta(terms, res, y, X)
            if th0.shape[0] == _ti.shape[0]:
                th0 = np.where(np.isnan(th0), _ti, th0)
    else:
        th0 = initial_theta(terms, res, y, X)
        if lag is not None:
            # Psi starts at `init` (units of W) when given, at 0 otherwise:
            # 0 is always admissible (A = I).
            ini = np.asarray(lag.get("init", np.zeros((lag["q"], lag["q"]))),
                             dtype=np.float64).reshape(lag["q"], lag["q"])
            ini = np.where(np.isfinite(ini), ini, 0.0)
            th0 = np.concatenate([th0, theta_from_psi(ini, lag)])
    p = p_cov + lag_n_params(lag)
    if th0.shape[0] != p:
        raise ValueError("theta initial de longueur %d, attendu %d" % (th0.shape[0], p))
    return th0, p_cov, p


def _check_start(th0, terms, res, X, n, lag, p_cov, eng):
    """Refuses a starting point outside the admissible region of Psi, and,
    without residual, a V that is not positive definite (said plainly rather
    than letting the Cholesky return NaN)."""
    if lag is not None:
        _ok = bool(lag_terms(jnp.asarray(th0[p_cov:]), lag, X)[2])
        if not _ok:
            raise ValueError(
                "Psi initial hors de la region admissible : I - M est singuliere ou de "
                "determinant negatif (pas d'equilibre). Partir de Psi = 0 ou d'une valeur "
                "plus petite.")
    if res.get("struct") == "none" and eng.engine == "dense":
        if eng.memory == "low":
            from .econome import V_on_host
            V0 = V_on_host(eng.plan_eco, th0)
        else:
            V0 = np.asarray(assemble_V(jnp.asarray(th0), terms, eng.Zs, res, n))
        try:
            np.linalg.cholesky(V0)
        except np.linalg.LinAlgError:
            w = np.linalg.eigvalsh(0.5 * (V0 + V0.T))
            raise ValueError(
                "residuelle 'none' : V n'est pas definie positive au point de depart "
                "(plus petite valeur propre %.3g, %d valeur(s) propre(s) <= 0 sur %d). "
                "Sans residuelle, les termes doivent donner a CHAQUE observation une "
                "variance propre : par exemple un terme de plante dont les incidences "
                "sont [I place sur le caractere t, W_t place sur le caractere t], "
                "structure us sur 2q colonnes." % (w.min(), int((w <= 0).sum()), n))


def _bounds(th0, p, floor, ceil, terms, res, y, fixed_idx):
    """Box bounds of L-BFGS-B, and th0 clipped to the loading bounds. Fixed
    parameters (asreml's F code) are bounded at their starting value: the
    algebra is unchanged, and the diagnostics see them at a bound and exclude
    them from the free subspace."""
    bornes = [(floor, ceil)] * p
    th0 = np.array(th0, dtype=np.float64)   # copy: th0 may be the caller's theta_init
    for (o_, ln_, B_) in loading_bounds(terms, res, y):
        for j in range(o_, o_ + ln_):
            bornes[j] = (-B_, B_)
            th0[j] = float(np.clip(th0[j], -B_, B_))
    if fixed_idx:
        for j in fixed_idx:
            bornes[j] = (float(th0[j]), float(th0[j]))
    return bornes, th0


# ------------------------------------------------------------------------------
# Counted, memoised objective
# ------------------------------------------------------------------------------

class _Objective:
    """The objective seen by the optimisers: counts and times the evaluations
    and remembers the last one.

    The first evaluation, at the starting point, pays XLA compilation plus one
    evaluation; it is timed apart (t_first) and kept, so the optimiser, which
    starts at the same point, finds it without recomputing. Compilation time is
    then t_first minus the median evaluation time.

    The memo has one slot per entry point: fun_jac (-2logL) and fun_sc (the
    scaled objective); fun_sc also reuses the fun_jac slot at the same theta.
    The order and number of evaluations are part of the result (n_eval): keep
    them when changing a caller."""

    def __init__(self, eng, th0, p, verbose, slow_on, slow_stop):
        _chrono("preparation")
        self._raw_fj, self._raw_sc, self.sc = eng.fun_jac, eng.fun_sc, eng.sc
        self.verbose, self.slow_stop = verbose, slow_stop
        _t = time.time()
        _v0 = self._raw_fj(th0)
        try:
            import jax as _jax
            _jax.block_until_ready(_v0[0])
        except Exception:
            pass
        self.t_first = time.time() - _t
        self._memo = {"fj": (np.array(th0, dtype=np.float64).tobytes(), _v0)}
        _chrono("compilation")
        # First sign of life of the solver: compilation can be long on a large
        # model, and a silent log cannot tell compiling from stuck.
        if verbose:
            print("  [compilation] premier appel (compilation + une evaluation) %.1f s | %d parametres"
                  % (self.t_first, p), flush=True)
        # hist: iterations, evaluations, last value seen (printed by the
        # callback at no cost), durations of the evaluations after the first.
        self.hist = {"n": 0, "t0": time.time(), "n_eval": 0,
                     "v": float("nan"), "gmax": float("nan")}
        self.hist["t_eval"] = []
        self._slow = dict(active=slow_on, f=[])

    def _evaluate(self, slot, th, raw, factor):
        key = np.asarray(th, dtype=np.float64).tobytes()
        m_ = self._memo.get(slot)
        if m_ is not None and m_[0] == key:
            return m_[1]
        if slot == "sc":
            m_ = self._memo.get("fj")
            if m_ is not None and m_[0] == key:
                v_, g_ = m_[1]
                return float(v_) / factor, np.asarray(g_, dtype=np.float64) / factor
        t_ = time.time()
        self.hist["n_eval"] += 1
        r_ = raw(th)
        if len(self.hist["t_eval"]) < 50:
            self.hist["t_eval"].append(time.time() - t_)
        self._memo[slot] = (key, r_)
        return r_

    def fun_jac(self, th):
        """-2logL and its gradient."""
        v, g = self._evaluate("fj", th, self._raw_fj, 1.0)
        self.hist["v"], self.hist["gmax"] = float(v), float(np.max(np.abs(g)))
        return v, g

    def fun_sc(self, th):
        """The scaled objective -2logL / sc minimised by L-BFGS-B; the value
        kept for printing is brought back to -2logL."""
        v, g = self._evaluate("sc", th, self._raw_sc, self.sc)
        self.hist["v"], self.hist["gmax"] = float(v) * self.sc, float(np.max(np.abs(g))) * self.sc
        return v, g

    def callback(self, _):
        """L-BFGS-B callback: slow-progress stop (finish "trust"), and a
        progress line whose period adapts to the measured evaluation time."""
        hist = self.hist
        hist["n"] += 1
        if self._slow["active"]:
            self._slow["f"].append(hist["v"])
            k_, seuil_ = self.slow_stop
            if len(self._slow["f"]) > k_ and self._slow["f"][-k_ - 1] - self._slow["f"][-1] < seuil_:
                hist["slow_stop"] = True
                raise StopIteration
        if not self.verbose:
            return
        te_ = float(np.median(hist["t_eval"])) if hist["t_eval"] else self.t_first
        pas = 1 if te_ > 0.25 else (10 if te_ > 0.02 else 50)
        if hist["n"] == 1 or hist["n"] % pas == 0:
            print("  iter %4d | -2logL %14.6f | max|grad| %.3e | %6.1f s | %d eval"
                  % (hist["n"], hist["v"], hist["gmax"],
                     time.time() - hist["t0"], hist["n_eval"]), flush=True)

    def lbfgsb(self, th, bornes, maxiter, tol):
        """One L-BFGS-B run on the scaled objective."""
        return minimize(self.fun_sc, th, jac=True, method="L-BFGS-B",
                        bounds=bornes, callback=self.callback,
                        options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                     ftol=tol, gtol=1e-10))


# ------------------------------------------------------------------------------
# Optimisation
# ------------------------------------------------------------------------------

def _optimise(obj, th0, bornes, theta_init, terms, res, y, X, n, lag, eng, optimizer,
              maxiter, tol, floor, fixed_idx, slow_stop, verbose):
    """Optimisation from th0: range sweep (automatic start only), check of a
    finite start, then AI-REML or L-BFGS-B, each followed by escapes from the
    variance floor. Returns theta and the optimiser result."""
    if theta_init is None:
        th0, _ = _range_sweep(obj.fun_jac, th0, terms, res, n, verbose=verbose,
                                  fixed_idx=fixed_idx, fun_sc=obj.fun_sc, bornes=bornes)
    # With a NaN gradient, L-BFGS-B returns the start at iteration 0 and
    # declares convergence: refused explicitly.
    _v0, _g0 = obj.fun_jac(th0)
    if not np.isfinite(float(_v0)) or not np.all(np.isfinite(np.asarray(_g0))):
        raise ValueError(
            "-2logL ou son gradient n'est pas fini au point de depart "
            "(valeur %r, %d composante(s) du gradient non finie(s)) : "
            "l'optimiseur ne peut pas demarrer. Verifier le noyau ou fournir "
            "theta_init." % (float(_v0), int(np.sum(~np.isfinite(np.asarray(_g0))))))
    if optimizer == "ai":
        r = _optimise_ai(obj, th0, bornes, terms, res, y, X, lag, eng, maxiter, tol, floor,
                         verbose)
    elif optimizer == "lbfgsb":
        r = _optimise_lbfgsb(obj, th0, bornes, terms, res, y, X, maxiter, tol, floor,
                             slow_stop, verbose)
    else:
        raise ValueError("optimizer inconnu : %r (lbfgsb, ai)" % (optimizer,))
    return np.asarray(r.x, dtype=np.float64), r


def _floor_reference(th0, terms, res, y, X):
    """Reference point of the floor escapes: the start, or the automatic start
    where it is higher (a theta_init itself at the floor says nothing)."""
    th_ref = np.array(th0, dtype=np.float64)
    _ti = initial_theta(terms, res, y, X)
    th_ref[:len(_ti)] = np.maximum(th_ref[:len(_ti)], _ti)
    return th_ref


def _optimise_lbfgsb(obj, th0, bornes, terms, res, y, X, maxiter, tol, floor, slow_stop,
                     verbose):
    """L-BFGS-B, then up to three escapes from the floor: a variance pushed
    near the floor in log scale does not come back, its gradient vanishes
    there (_floor_escape)."""
    r = obj.lbfgsb(th0, bornes, maxiter, tol)
    if obj.hist.get("slow_stop") and verbose:
        print("  [finition] L-BFGS-B arrete a l'iteration %d : gain < %g sur %d iterations"
              % (obj.hist["n"], slow_stop[1], slow_stop[0]), flush=True)
    th_ref = _floor_reference(th0, terms, res, y, X)
    for _tour in range(3):
        th_ev = _floor_escape(obj.fun_jac, np.asarray(r.x, dtype=np.float64), th_ref,
                                  bornes, terms, res, floor, verbose=verbose)
        if th_ev is None:
            break
        nit_av = int(r.nit)
        r = obj.lbfgsb(th_ev, bornes, maxiter, tol)
        r.nit = nit_av + int(r.nit)
    return r


def _optimise_ai(obj, th0, bornes, terms, res, y, X, lag, eng, maxiter, tol, floor, verbose):
    """AI-REML (ai.py): same objective and gradient, curvature from the average
    information, one factorisation per point. On failure (AI in the variance
    scale stalls on the edge of the cone of a us Sigma), L-BFGS-B resumes from
    the point reached. Then up to two escapes from the floor."""
    from types import SimpleNamespace
    from .ai import ai_reml_var
    if lag is not None:
        raise ValueError("optimizer='ai' : modele structurel (lag) non pris en charge.")
    fga, f_only = _ai_functions(obj, eng, terms, res, y, X)
    floor_us = 1e-8 if eng.engine == "mme" else 1e-10
    th_ai, info = ai_reml_var(fga, th0, bornes, terms, res, maxiter=maxiter,
                              verbose=verbose, counter=obj.hist, us_floor=floor_us,
                              fun_sc=obj.fun_sc, f_only=f_only)
    if not info["success"]:
        if verbose:
            print("  [ai] %s -> L-BFGS-B depuis ce point" % info["message"], flush=True)
        r2 = obj.lbfgsb(th_ai, bornes, maxiter, tol)
        th_ai = np.asarray(r2.x, dtype=np.float64)
        info = dict(nit=info["nit"] + int(r2.nit), success=False,
                    message="%s ; repli L-BFGS-B : %s" % (info["message"], r2.message))
    r = SimpleNamespace(x=th_ai, **info)
    th_ref = _floor_reference(th0, terms, res, y, X)
    for _tour in range(2):
        th_ev = _floor_escape(obj.fun_jac, np.asarray(r.x, dtype=np.float64), th_ref,
                                  bornes, terms, res, floor, verbose=verbose)
        if th_ev is None:
            break
        th_ai, info2 = ai_reml_var(fga, th_ev, bornes, terms, res, maxiter=maxiter,
                                   verbose=verbose, counter=obj.hist, us_floor=floor_us,
                                   fun_sc=obj.fun_sc, f_only=f_only)
        r = SimpleNamespace(x=th_ai, nit=int(getattr(r, "nit", 0)) + int(info2["nit"]),
                            success=info2["success"],
                            message="%s ; apres sortie du plancher : %s"
                            % (getattr(r, "message", ""), info2["message"]))
    return r


def _ai_functions(obj, eng, terms, res, y, X):
    """Value, gradient and AI matrix in one factorisation (fga) for the engine
    in use, and the value alone for the step trials (the MME engine without
    derivatives; elsewhere value and gradient)."""
    from .ai import make_fga
    if eng.engine == "mme":
        fga = eng.fga_mme
    elif eng.memory == "low":
        from .econome import make_fga_low_memory
        fga = make_fga_low_memory(eng.plan_eco)
    else:
        fga = make_fga(terms, res, y, X)
    if eng.engine == "mme":
        M_mme = eng.M_mme
        f_only = lambda th_: M_mme.evaluate(np.asarray(th_), full=False)["f"]
    else:
        f_only = lambda th_: float(obj.fun_jac(th_)[0])
    return fga, f_only


def _check_ai_saddle(obj, theta, r, bornes, floor, ceil, hessian, maxiter, tol, polish,
                     verbose):
    """After a converged AI-REML, no Newton polishing: AI stops on a decrement
    of 1e-6 and polishing has nothing left to gain. The AI matrix is positive
    semi-definite and cannot see negative curvature, so with hessian = TRUE
    the Hessian computed for the diagnostics checks for a saddle point first;
    a negative eigenvalue on the free subspace sends back to L-BFGS-B, then to
    polishing. Returns theta, r, the number of polishing steps and the cached
    Hessian (theta, H) or None."""
    from types import SimpleNamespace
    polish_eff, H_cache = 0, None
    lo_ = np.array([b[0] for b in bornes]); hi_ = np.array([b[1] for b in bornes])
    libre_ = (theta > lo_ + 1e-9) & (theta < hi_ - 1e-9)
    if hessian and libre_.any():
        H_ai = hessian_fd(obj.fun_jac, theta, floor=floor, ceil=ceil)
        H_cache = (theta.copy(), H_ai)
        w_ai = np.linalg.eigvalsh(H_ai[np.ix_(libre_, libre_)])
        if w_ai.min() < -1e-6 * max(1.0, np.abs(w_ai).max()):
            if verbose:
                print("  [ai] point selle (valeur propre du Hessien %.2e) -> L-BFGS-B depuis ce point"
                      % w_ai.min(), flush=True)
            r2 = obj.lbfgsb(theta, bornes, maxiter, tol)
            theta = np.asarray(r2.x, dtype=np.float64)
            r = SimpleNamespace(x=theta, nit=int(getattr(r, "nit", 0)) + int(r2.nit), success=False,
                                message="%s ; point selle, repli L-BFGS-B : %s"
                                % (getattr(r, "message", ""), r2.message))
            polish_eff = int(polish)
    return theta, r, polish_eff, H_cache


# ------------------------------------------------------------------------------
# Finishing
# ------------------------------------------------------------------------------

def _finish_trust_region(obj, theta, bornes, verbose):
    """Trust-region Newton finish (finition.py), recorded in hist."""
    from .finition import finish
    lo_ = np.array([b_[0] for b_ in bornes], dtype=np.float64)
    hi_ = np.array([b_[1] for b_ in bornes], dtype=np.float64)
    theta, info_fin = finish(obj.fun_jac, theta, lo_, hi_,
                             hessian=lambda th: hessian_fd(obj.fun_jac, th, floor=lo_, ceil=hi_),
                             verbose=verbose, counter=obj.hist)
    obj.hist["finish"] = info_fin
    return theta


def _newton_polish(obj, theta, n_steps, p, floor, ceil, fixed_idx, verbose):
    """Regularised Newton steps after L-BFGS-B, which stops on a relative
    tolerance about 1e-7 away from the optimum on the variance components.

    Each step costs a finite-difference Hessian (2p gradients) and is accepted
    only if it improves the objective (backtracking over alpha). Polishing
    stops when no step improves, the gradient vanishes, or after n_steps. One
    line per step is printed. Returns theta and the number of steps taken."""
    n_polish = 0
    if n_steps <= 0:
        return theta, n_polish
    if verbose:
        print("  polissage de Newton : jusqu'a %d pas, %d gradients par pas"
              % (n_steps, 2 * p), flush=True)
    for _ in range(n_steps):
        t_pas = time.time()
        f_cur, g_cur = obj.fun_jac(theta)
        H = hessian_fd(obj.fun_jac, theta, floor=floor, ceil=ceil)
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g_cur)):
            if verbose:
                print("  polissage arrete : Hessien ou gradient non fini", flush=True)
            break
        try:
            w, Vv = np.linalg.eigh(0.5 * (H + H.T))
        except np.linalg.LinAlgError:
            break
        # Regularisation: no step along a flat direction.
        lam = max(1e-10, 1e-8 * abs(w).max())
        w_reg = np.where(w > lam, w, lam)
        step = -(Vv @ ((Vv.T @ g_cur) / w_reg))
        done = False
        if fixed_idx:
            step[np.array(fixed_idx)] = 0.0
        for alpha in (1.0, 0.5, 0.25, 0.1):
            cand = np.clip(theta + alpha * step, floor, ceil)
            f_try, _ = obj.fun_jac(cand)
            # Absolute gain: a threshold relative to |f| (~1e4) would reject
            # gains of 1e-11 that still move the variances at the 7th digit.
            if np.isfinite(f_try) and f_try < f_cur - 1e-13:
                theta, done = cand, True
                break
        if not done:
            if verbose:
                print("  polissage arrete : aucun pas n'ameliore -2logL (deja au sommet)",
                      flush=True)
            break
        n_polish += 1
        if verbose:
            # Newton decrement g' H^+ g / 2 in logLik units, on the regularised
            # Hessian: the ascent still available under the quadratic model.
            dec = 0.5 * float(g_cur @ (Vv @ ((Vv.T @ g_cur) / w_reg)))
            print("  polish %3d | -2logL %14.6f | gain %.3e | decrement %.3e | %6.1f s | %d eval"
                  % (n_polish, f_try, f_cur - f_try, dec,
                     time.time() - t_pas, obj.hist["n_eval"]), flush=True)
        if float(g_cur @ g_cur) < 1e-24:
            break
    return theta, n_polish


# ------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------

def _result_core(obj, eng, theta, r, f_end, g_end, X, y, n, p, maxiter, floor, ceil,
                 optimizer, t_optim, n_polish):
    """Likelihood in both conventions, optimiser status, bounds and timings.

    remlax and lme4 include the constant (n - p)/2 log(2 pi) in logLik; asreml
    and sommer omit it (logLik_asreml). The bounds travel with the result:
    every consumer of the free subspace reads them instead of recoding them.
    With maxiter = 0 the optimiser did not run, and the status says so."""
    hist = obj.hist
    # Time of one evaluation: median of the optimiser's evaluations (for the
    # MME engine, of its full evaluations); compilation is the excess of the
    # first call over it.
    _t_eval = float(np.median(hist["t_eval"])) if hist["t_eval"] else float(obj.t_first)
    if eng.engine == "mme" and getattr(eng.M_mme, "_durations", None):
        _t_eval = float(np.median(eng.M_mme._durations))
    _t_compil = max(obj.t_first - _t_eval, 0.0)
    n_p = int(np.linalg.matrix_rank(X))
    cst = 0.5 * (len(y) - n_p) * np.log(2.0 * np.pi)
    no_optim = int(maxiter) <= 0
    return dict(theta=theta, neg2_reml=f_end, logLik=-0.5 * f_end, n_polish=n_polish,
                logLik_asreml=-0.5 * f_end + cst, const_2pi=float(cst),
                n_par=int(p), n_obs=int(n),
                max_grad=float(np.max(np.abs(g_end))),
                scipy_success=(True if no_optim else bool(r.success)),
                scipy_message=("evaluation seule (maxiter = 0), aucune optimisation"
                               if no_optim else str(r.message)),
                n_iter=(0 if no_optim else int(r.nit)),
                seconds=time.time() - hist["t0"],
                par_floor=float(floor), par_ceil=float(ceil),
                compile_s=_t_compil, eval_s=_t_eval, first_call_s=obj.t_first,
                n_eval=int(hist["n_eval"]),
                optimizer=str(optimizer), seconds_optim=float(t_optim))


def _add_identification(out, terms, n, p, verbose):
    """Entries not identified separately (equal patterns in V, aliased_entries):
    each group leaves expected flat directions in the Hessian, and the number
    of identified parameters (df of a LRT, AIC) accounts for them. Returns the
    number of expected flat directions."""
    alias = aliased_entries(terms, n)
    n_plat = int(sum(len(g["entries"]) - 1 for g in alias))
    _chrono("alias")
    out["alias"] = [dict(term=g["term"], entries=[[int(i), int(j)] for i, j in g["entries"]])
                    for g in alias]
    out["n_flat_expected"] = n_plat
    out["n_par_identified"] = int(p - n_plat)
    if n_plat and verbose:
        print("  [identification] %d direction(s) plate(s) par construction : %s"
              % (n_plat, "; ".join("%s %s" % (g["term"], " = ".join(
                  "[%d,%d]" % (i + 1, j + 1) for i, j in g["entries"])) for g in alias)),
              flush=True)
    return n_plat


def _add_lag_report(out, theta, lag, p_cov, X, y, f_end, verbose):
    """The analysed response: y, or with a spatial lag y* = (I - M) y at the
    final point, whose ordinary model gives beta, the BLUPs and P y. Adds the
    lag report. Returns the analysed response."""
    y_fit = (np.asarray(y, dtype=np.float64) if lag is None
             else np.asarray(lag_ystar(jnp.asarray(theta[p_cov:]), lag, y)))
    if lag is not None:
        out["lag"] = lag_report(theta[p_cov:], lag, X, f_end)
        out["ystar"] = y_fit
        if verbose:
            Lr = out["lag"]
            print("  [decalage] critere %s | chemin %s | log|det A| %.6f | rayon spectral de M %.4f"
                  % (Lr["criterion"], Lr["path"], Lr["logdet_A"], Lr["spectral_radius"]),
                  flush=True)
    return y_fit


def _add_degenerate(out, theta, terms, p_cov, floor, fixed_idx, verbose):
    """Terms with a parameter within 0.5 of the floor. A variance at the floor
    is not identified, not estimated at zero; asreml refuses to converge in
    that case, and remlax flags it instead."""
    seuil = floor + 0.5
    degen = []
    o = 0
    for k, tm in enumerate(terms):
        p_k = term_n_params(tm)
        if np.any(theta[o:o + p_k] <= seuil):
            degen.append(tm["name"])
        o += p_k
    if np.any(theta[o:p_cov] <= seuil):
        degen.append("residual")
    out["degenerate_components"] = degen
    out["n_fixed"] = 0 if not fixed_idx else len(fixed_idx)
    if degen and verbose:
        print("  [attention] composante(s) au plancher (non identifiee(s)) : %s"
              % ", ".join(degen), flush=True)


def _add_sigmas(out, theta, terms, res, n):
    """Sigma, loadings and level parameters of each term and of each residual
    section. Loadings are reported because the decomposition of an assembled
    Sigma is not unique. A residual "none" has no section, hence no sigma_res
    (a zero matrix would read as a variance estimated at zero)."""
    th_terms, th_res, _ = split_theta(theta, terms, res)
    out["sigmas"], out["rho"] = {}, {}
    for k, tm in enumerate(terms):
        ns = n_params(tm["struct"], tm["t"], tm["rank"])
        out["sigmas"][tm["name"]] = np.asarray(build_sigma(
            jnp.asarray(th_terms[k][:ns]), tm["struct"], tm["t"], tm["rank"]))
        kind_ = level_kind(tm)
        r_ = level_params_report(np.asarray(th_terms[k][ns:]), kind_,
                                 tm.get("lvl_order", 0), opts=tm.get("lvl_opts"),
                                 q=tm.get("q"), parts=tm.get("lvl_parts"))
        lo = sigma_loadings(np.asarray(th_terms[k][:ns]), tm["struct"],
                            tm["t"], tm["rank"])
        if lo is not None:
            out.setdefault("loadings", {})[tm["name"]] = lo
        if r_.get("pacf"):
            out.setdefault("pacf", {})[tm["name"]] = r_["pacf"]
        _store_level_params(out, tm["name"], kind_, r_)
    secs = res_sections(res, n)
    out["sigmas_res"], o_s = {}, 0
    for sec in secs:
        p_s = sec_n_params(sec)
        th_s = th_res[o_s:o_s + p_s]
        ns_r = n_params(sec["struct"], sec["t"], sec["rank"])
        nom = sec.get("name", "residual") if len(secs) > 1 else "residual"
        S = (np.asarray(build_sigma(jnp.asarray(th_s[:ns_r]), sec["struct"],
                                    sec["t"], sec["rank"]))
             if sec["struct"] in ("us", "fa", "diag")
             else np.array([[float(np.exp(2 * th_s[0]))]]))
        out["sigmas_res"][nom] = S
        lo_r = sigma_loadings(np.asarray(th_s[:ns_r]), sec["struct"],
                              sec["t"], sec["rank"])
        if lo_r is not None:
            out.setdefault("loadings_res", {})[nom] = lo_r
        r_res = level_params_report(np.asarray(th_s[ns_r:]), sec.get("lvl", "id"),
                                    sec.get("lvl_order", 0), opts=sec.get("lvl_opts"),
                                    q=sec.get("n_unit", len(np.unique(np.asarray(sec["unit"])))),
                                    parts=sec.get("lvl_parts"))
        _store_level_params(out, nom, sec.get("lvl", "id"), r_res)
        o_s += p_s
    if out["sigmas_res"]:
        out["sigma_res"] = out["sigmas_res"][list(out["sigmas_res"])[0]]


def _add_effects(out, eng, theta, terms, res, y, y_fit, X, n, blups, hessian, pev):
    """Engine, beta, vbeta, BLUPs and PEV. The MME engine reads them in the
    mixed model equations without forming V; the blocks engine and the
    low-memory mode form V once on the device and bring it to the host (a
    block-wise output remains to be written)."""
    engine, memory = eng.engine, eng.memory
    out["engine"] = engine
    out["memory"] = memory if engine == "dense" else "standard"
    if engine == "blocks":
        out["blocks"] = eng.plan_blocs.summary()
    if engine == "mme":
        so = eng.M_mme.outputs(theta, pev=(pev if blups else ()))
        out["beta"], out["vbeta"], out["V_singular"] = so["beta"], so["vbeta"], False
        if blups:
            out["blups"] = so["blups"]
            if "pev" in so:
                out["pev"] = so["pev"]
        out["inversion"] = eng.M_mme.inversion
    if engine in ("dense", "blocks") and (blups or hessian) and (memory == "low"
                                                                  or engine == "blocks"):
        from .econome import V_on_host, z_on_host, LowMemoryPlan
        plan_eco = LowMemoryPlan(terms, res, y, X) if engine == "blocks" else eng.plan_eco
        out.update(dense_effects(theta, terms, None, res, y_fit, X, blups=blups, pev=pev,
                                 V=V_on_host(plan_eco, theta),
                                 z_de=lambda k: z_on_host(terms[k], n)))
    elif engine == "dense" and (blups or hessian):
        out.update(dense_effects(theta, terms, eng.Zs, res, y_fit, X, blups=blups, pev=pev))


def _add_hessian_diagnostics(out, obj, theta, g_end, H_cache, hessian, floor, ceil,
                             fixed_idx, n_plat):
    """Hessian of -2logL and convergence diagnostics. Without the Hessian, the
    projected gradient criterion is still given (g_end is at hand), and the
    decrement and curvature fields say they are unavailable."""
    if hessian:
        if H_cache is not None and np.array_equal(H_cache[0], theta):
            H = H_cache[1]                      # the one of the saddle check
        else:
            H = hessian_fd(obj.fun_jac, theta, floor=floor, ceil=ceil)
        out["hessian"] = H
        out.update(diagnostic(H, g_end, theta, floor, ceil, fixed_idx=fixed_idx,
                               f_obj=out.get("neg2_reml"), n_flat_expected=n_plat))
    else:
        out.update(gradient_diagnostic(g_end, theta, floor, ceil,
                                        fixed_idx=fixed_idx,
                                        f_obj=out.get("neg2_reml")))
        out.update(dict(newton_decrement=np.nan, conv_decrement=None,
                        conv_hessian_ok=None, n_neg_eig=None, cond=np.nan))


def _dense_restart_if_needed(out, terms, res, y, X, engine, maxiter, verbose, kw):
    """Finish by the dense engine. An MME fit that the optimiser did not
    declare converged, with a relative gradient above GRAD_REL_DENSE_RESTART, is
    resumed by the dense engine from the point reached: the MME engine inverts
    Sigma and K and can lock near a rank-deficient Sigma. Not done above
    N_MAX_DENSE_RESTART observations (the dense engine no longer fits in memory),
    nor when a term has only its precision K^-1, which the dense engine would
    treat as independent (another model)."""
    gr_ = out.get("grad_rel")
    kinv_only = any(tm.get("Kinv") is not None and tm.get("LK") is None for tm in terms)
    failed = not bool(out.get("scipy_success", False))
    grad_ok = gr_ is not None and gr_ <= GRAD_REL_DENSE_RESTART
    if (engine == "mme" and int(maxiter) > 0 and kinv_only and verbose and failed
            and not grad_ok):
        print("  [moteur] MME non converge (gradient relatif %s), sans reprise dense : un terme "
              "n'a que sa precision K^-1" % ("%.2e" % gr_ if gr_ is not None else "n.d."), flush=True)
    if (engine == "mme" and int(maxiter) > 0 and not kinv_only and failed
            and len(y) <= N_MAX_DENSE_RESTART and not grad_ok):
        if verbose:
            print("  [moteur] MME non converge (gradient relatif %s) : fin par le moteur dense "
                  "depuis ce point" % ("%.2e" % gr_ if gr_ is not None else "n.d."), flush=True)
        out_d = fit_reml(terms, res, y, X, theta_init=out["theta"], **kw)
        out_d["engine"] = "mme+dense"
        out_d["inversion"] = out.get("inversion")
        out_d["seconds"] = float(out_d.get("seconds", 0.0)) + float(out.get("seconds", 0.0))
        return out_d
    return out


def _store_level_params(out, nom, kind, rapport):
    """Stores the level parameters in out["rho"] under unambiguous keys. A
    structure with one parameter keeps the bare key (rho["residual"]); with
    several (mtrn: phi, nu, delta, alpha) each key is prefixed: residual!phi,
    residual!nu, ..."""
    multi = kind == "mtrn"
    for cle, v in rapport.items():
        if cle == "pacf":
            continue
        if cle == "phi" and not multi:
            out["rho"][nom] = v
        else:
            out["rho"]["%s!%s" % (nom, cle)] = v


def _floor_escape(fun_jac, theta, th0, bornes, terms, res, floor, marge=3.0,
                  verbose=True):
    """Lifts off the floor the variances that do not belong there.

    A log-parametrised variance (iid, diag, specific variances of an fa) has a
    gradient in theta equal to 2 psi dF/dpsi: near the floor it vanishes
    whatever dF/dpsi, and L-BFGS-B declares convergence there (dev/notes.md).
    The sign of the gradient stays readable: when negative, raising the
    variance lowers -2logL. Those variances are raised together towards their
    start (psi0 e^-2, then e^-6, then e^-10); the point is kept as soon as
    -2logL decreases. Returns None when there is nothing to do or no gain."""
    from .ai import variance_scale_blocks
    blocs, p = variance_scale_blocks(terms, res)
    if p != len(theta):
        return None
    log_var = np.zeros(p, dtype=bool)
    for (o, ln, nat, t) in blocs:
        if nat in ("iid", "diag"):
            log_var[o:o + ln] = True
    lo = np.array([b[0] for b in bornes]); hi = np.array([b[1] for b in bornes])
    libre = hi > lo
    bas = log_var & libre & (theta <= floor + marge)
    if not bas.any():
        return None
    f0, g0 = fun_jac(theta)
    g0 = np.asarray(g0, dtype=np.float64)
    cand = bas & (g0 < 0)
    if not cand.any() or not np.isfinite(float(f0)):
        return None
    for recul in (1.0, 3.0, 5.0):
        th = theta.copy()
        th[cand] = np.clip(np.maximum(th0[cand] - recul, theta[cand]), lo[cand], hi[cand])
        f1, _ = fun_jac(th)
        if np.isfinite(float(f1)) and float(f1) < float(f0) - 1e-8 * max(1.0, abs(float(f0))):
            if verbose:
                print("  [plancher] %d variance(s) remontee(s) (log psi0 - %g) : -2logL %.6f -> %.6f"
                      % (int(cand.sum()), recul, float(f0), float(f1)), flush=True)
            return th
    return None
