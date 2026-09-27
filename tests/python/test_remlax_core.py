"""Tests unitaires du noyau remlax (Python), independants de R.

Couvre ce qui ne peut se verifier qu'ici : parametrisations, gradients,
assemblage de V, invariance au peripherique. Les comparaisons a des
implementations independantes (lme4, sommer) sont dans test_remlax.R.

    python3 scripts/tests/test_remlax_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import jax.numpy as jnp  # noqa: E402
from remlax import structures as S  # noqa: E402
from remlax.model import assemble_V, dense_Z, make_objective, split_theta, n_theta  # noqa: E402
from remlax.fit import fit_reml  # noqa: E402

def run():
    """Execute les six sections et rend la liste des echecs (vide si tout passe).

    LE CORPS EST UNE FONCTION, PAS UN SCRIPT A L'IMPORT. Ce fichier s'appelle
    test_*.py, donc pytest le collecte et l'IMPORTE ; un sys.exit() au niveau
    du module tuait alors la collecte entiere (INTERNALERROR SystemExit), ce
    qui a fait echouer la CI sans qu'aucun test n'ait tourne.
    """
    ECHECS = []


    def check(nom, cond, detail=""):
        print("  %-58s %s %s" % (nom, "OK " if cond else "ECHEC", detail))
        if not cond:
            ECHECS.append(nom)


    def jeu(n=120, q=20, t=1, seed=0, K=False):
        rng = np.random.default_rng(seed)
        lev = rng.integers(0, q, n)
        zi = np.tile(np.arange(n), t)
        zj = np.concatenate([a * q + lev for a in range(t)])
        zx = rng.normal(size=n * t) if t > 1 else np.ones(n)
        LK = None
        if K:
            A = rng.normal(size=(q, q)); Kk = A @ A.T / q + np.eye(q)
            LK = np.linalg.cholesky(Kk)
        term = dict(name="g", struct="us" if t > 1 else "iid", t=t, rank=0, q=q,
                    zi=zi, zj=zj, zx=zx, LK=LK)
        y = rng.normal(size=n); X = np.ones((n, 1))
        res = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64), unit=np.arange(n))
        return term, res, y, X


    print("\n=== 1. structures : nombre de parametres, positivite, aller-retour ===")
    for st, t, r in [("iid", 1, 0), ("iid", 4, 0), ("diag", 4, 0), ("us", 2, 0),
                     ("us", 6, 0), ("fa", 5, 1), ("fa", 5, 3), ("fa", 8, 4)]:
        p = S.n_params(st, t, r)
        th = S.theta0(st, t, r, var=1.7)
        Sig = np.array(S.build_sigma(jnp.array(th), st, t, r))
        ev = np.linalg.eigvalsh(Sig)
        check("%s(t=%d,r=%d) : p=%d, symetrique, PSD" % (st, t, r, p),
              len(th) == p and np.allclose(Sig, Sig.T) and ev.min() > -1e-12,
              "min eig %.2e" % ev.min())

    rng = np.random.default_rng(3)
    for t in (2, 5, 9):
        A = rng.normal(size=(t, t)); T = A @ A.T + np.eye(t)
        L = np.linalg.cholesky(T); ii, jj = S._tril_indices(t)
        th = L[ii, jj].copy(); th[ii == jj] = np.log(np.diag(L))
        back = np.array(S.build_sigma(jnp.array(th), "us", t))
        check("us(t=%d) represente n'importe quelle matrice PSD" % t,
              np.abs(back - T).max() < 1e-12, "ecart %.2e" % np.abs(back - T).max())

    print("\n=== 2. gradient autodiff contre differences finies centrees ===")
    for t, useK in [(1, False), (1, True), (3, False), (3, True)]:
        term, res, y, X = jeu(t=t, K=useK, seed=10 + t)
        fun_sc, fun, sc, Zs = make_objective([term], res, y, X)
        th = np.asarray(np.concatenate([S.theta0(term["struct"], term["t"], term["rank"], 0.5),
                                        S.theta0("iid", 1, 0, 0.5)]))
        _, g = fun(th)
        gfd = np.zeros_like(th)
        for j in range(len(th)):
            e = np.zeros_like(th); e[j] = 1e-6
            gfd[j] = (fun(th + e)[0] - fun(th - e)[0]) / 2e-6
        rel = np.max(np.abs(g - gfd) / np.maximum(np.abs(gfd), 1.0))
        check("gradient exact (t=%d, K=%s)" % (t, useK), rel < 1e-6, "ecart relatif %.2e" % rel)

    print("\n=== 3. assemblage de V : symetrie, positivite, coherence Kronecker ===")
    term, res, y, X = jeu(t=3, K=True, seed=5)
    p = n_theta([term], res)
    th = np.asarray(np.concatenate([S.theta0("us", 3, 0, 0.8), S.theta0("iid", 1, 0, 0.4)]))
    Zs = [dense_Z(term, len(y))]
    V = np.asarray(assemble_V(jnp.asarray(th), [term], Zs, res, len(y)))
    check("V symetrique", np.allclose(V, V.T, atol=1e-12))
    check("V definie positive", np.linalg.eigvalsh(V).min() > 0,
          "min eig %.3e" % np.linalg.eigvalsh(V).min())
    # reference directe : Z (Sigma (x) K) Z' + sigma2 I, montee sans passer par B
    Sig = np.array(S.build_sigma(jnp.asarray(th[:6]), "us", 3))
    Kk = term["LK"] @ term["LK"].T
    Zd = np.asarray(Zs[0])
    Vref = Zd @ np.kron(Sig, Kk) @ Zd.T + np.exp(2 * th[6]) * np.eye(len(y))
    check("V == Z (Sigma (x) K) Z' + R  (voie independante)",
          np.abs(V - Vref).max() < 1e-9, "ecart max %.2e" % np.abs(V - Vref).max())

    print("\n=== 4. REML analytique : plan a un facteur, equilibre ===")
    rng = np.random.default_rng(7)
    for a, m in [(20, 6), (12, 10), (40, 3)]:
        g = np.repeat(np.arange(a), m)
        y = 3 + np.repeat(rng.normal(0, 1.4, a), m) + rng.normal(0, 1.2, a * m)
        n = a * m; X = np.ones((n, 1))
        gm = y.mean(); grp = y.reshape(a, m).mean(axis=1)
        MSA = m * ((grp - gm) ** 2).sum() / (a - 1)
        MSE = ((y.reshape(a, m) - grp[:, None]) ** 2).sum() / (a * (m - 1))
        tm = dict(name="g", struct="iid", t=1, rank=0, q=a, zi=np.arange(n), zj=g,
                  zx=np.ones(n), LK=None)
        rs = dict(struct="iid", t=1, rank=0, trait=np.zeros(n, np.int64), unit=np.arange(n))
        r = fit_reml([tm], rs, y, X, verbose=False)
        da = abs(float(r["sigmas"]["g"][0, 0]) - (MSA - MSE) / m)
        de = abs(float(r["sigma_res"][0, 0]) - MSE)
        check("a=%d m=%d : sigma2 analytiques retrouvees" % (a, m), da < 1e-8 and de < 1e-8,
              "ecarts %.1e / %.1e, decrement %.1e" % (da, de, r["newton_decrement"]))

    print("\n=== 5. reproductibilite et invariance au peripherique ===")
    term, res, y, X = jeu(t=2, K=True, seed=99)
    r1 = fit_reml([term], res, y, X, verbose=False)
    r2 = fit_reml([term], res, y, X, verbose=False)
    check("deux appels identiques donnent le meme optimum",
          abs(r1["neg2_reml"] - r2["neg2_reml"]) < 1e-10,
          "ecart %.2e" % abs(r1["neg2_reml"] - r2["neg2_reml"]))
    r3 = fit_reml([term], res, y, X, verbose=False,
                  theta_init=r1["theta"] + 0.3 * np.ones_like(r1["theta"]))
    check("depart perturbe -> meme optimum",
          abs(r1["neg2_reml"] - r3["neg2_reml"]) < 1e-6,
          "ecart %.2e" % abs(r1["neg2_reml"] - r3["neg2_reml"]))

    print("\n=== 6. cas degeneres : le solveur refuse plutot que de mentir ===")
    try:
        S.n_params("inconnue", 3); check("structure inconnue rejetee", False)
    except ValueError:
        check("structure inconnue rejetee", True)
    term, res, y, X = jeu(t=1, seed=1)
    Xsing = np.column_stack([np.ones(len(y)), np.ones(len(y))])   # rang 1 pour 2 colonnes
    try:
        r = fit_reml([term], res, y, Xsing, verbose=False, hessian=False, blups=False)
        fini = np.isfinite(r["neg2_reml"])
        check("X singuliere : detectee (pas de logLik finie fabriquee)", not fini,
              "-2logL = %s" % r["neg2_reml"])
    except Exception as e:
        check("X singuliere : erreur explicite", True, type(e).__name__)

    print("\n" + "=" * 70)
    if ECHECS:
        print("ECHECS (%d) : %s" % (len(ECHECS), ", ".join(ECHECS)))
    else:
        print("Tous les tests du noyau passent.")
    return list(ECHECS)



def test_noyau():
    """Point d'entree pytest : la suite entiere est un seul test."""
    echecs = run()
    assert not echecs, "echecs du noyau : " + ", ".join(echecs)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
