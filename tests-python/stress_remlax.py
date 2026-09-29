"""Balayage aleatoire : le solveur tient-il sur des configurations variees ?

Les tests unitaires verifient des cas choisis ; celui-ci en tire au hasard des
centaines, y compris des cas volontairement penibles (variance nulle, colonnes
colineaires, un seul niveau, forte parente, plans tres desequilibres). On ne
verifie pas une valeur attendue — il n'y en a pas — mais des PROPRIETES qui
doivent tenir quoi qu'il arrive :

  1. l'ajustement se termine sans exception ;
  2. -2logL est finie ;
  3. le decrement de Newton est negligeable devant le seuil d'un LRT ;
  4. toutes les Sigma rendues sont symetriques et semi-definies positives ;
  5. relancer d'un autre point de depart retombe sur la meme logLik.

    python3 scripts/tests/stress_remlax.py [--n 200] [--seed 0]
"""
import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
from remlax.fit import fit_reml  # noqa: E402


def tirage(rng):
    n = int(rng.integers(60, 260))
    n_terms = int(rng.integers(1, 4))
    t_res = int(rng.choice([1, 1, 2, 3]))
    res_struct = str(rng.choice(["iid", "diag", "us"])) if t_res > 1 else "iid"
    # Format long VALIDE : chaque unite apparait au plus une fois par caractere.
    # Tirer unit et trait independamment cree des doublons (unite, caractere),
    # que le solveur refuse desormais explicitement — c'est ce refus qui a fait
    # apparaitre le defaut, garder le generateur fautif masquerait le reste.
    n_unit = max(n // t_res, 2)
    pairs = np.array([(u, t) for u in range(n_unit) for t in range(t_res)])
    sel = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    pairs = pairs[sel]
    n = len(pairs)
    unit, trait = pairs[:, 0], pairs[:, 1]
    terms = []
    for k in range(n_terms):
        q = int(rng.integers(3, 40))
        t = 1 if rng.random() < 0.5 else int(rng.integers(2, 5))
        struct = str(rng.choice(["iid", "diag", "us", "fa"])) if t > 1 else "iid"
        rank = int(rng.integers(1, t + 1)) if struct == "fa" else 0
        lev = rng.integers(0, q, n)
        zi = np.tile(np.arange(n), t)
        zj = np.concatenate([a * q + lev for a in range(t)])
        # cas penible : incidence parfois ponderee, parfois degeneree (colonne nulle)
        zx = np.ones(n * t) if rng.random() < 0.6 else rng.normal(size=n * t)
        if rng.random() < 0.15:
            zx = zx * (rng.random(n * t) > 0.3)
        LK = None
        if rng.random() < 0.4:
            A = rng.normal(size=(q, max(q // 2, 2)))
            Kk = A @ A.T / q + np.eye(q) * (0.01 if rng.random() < 0.3 else 1.0)
            LK = np.linalg.cholesky(Kk + 1e-8 * np.eye(q))
        terms.append(dict(name="t%d" % k, struct=struct, t=t, rank=rank, q=q,
                          zi=zi, zj=zj, zx=zx, LK=LK))
    p_fix = int(rng.integers(1, 4))
    X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(p_fix - 1)])
    # signal parfois nul : le solveur doit rendre une variance a zero, pas planter
    y = rng.normal(size=n) * (0.0 if rng.random() < 0.05 else 1.0) + rng.normal(size=n) * 0.3
    res = dict(struct=res_struct, t=t_res, rank=0, trait=trait, unit=unit)
    return terms, res, y, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n_ok = 0
    pb = {"exception": [], "logLik non finie": [], "non convergé": [],
          "Sigma non PSD": [], "optimum instable": []}
    for i in range(a.n):
        terms, res, y, X = tirage(rng)
        try:
            r = fit_reml(terms, res, y, X, verbose=False, blups=False, polish=25)
        except Exception:
            pb["exception"].append((i, traceback.format_exc().splitlines()[-1]))
            continue
        if not np.isfinite(r["neg2_reml"]):
            pb["logLik non finie"].append((i, r["neg2_reml"])); continue
        dec = r.get("newton_decrement", np.nan)
        if not (np.isfinite(dec) and abs(dec) < 1e-2):
            pb["non convergé"].append((i, dec))
        bad = False
        for nm, S in r["sigmas"].items():
            S = np.asarray(S)
            if not np.allclose(S, S.T, atol=1e-9) or np.linalg.eigvalsh(S).min() < -1e-8:
                pb["Sigma non PSD"].append((i, nm, float(np.linalg.eigvalsh(S).min())))
                bad = True
        try:
            r2 = fit_reml(terms, res, y, X, verbose=False, blups=False, hessian=False,
                          polish=25, theta_init=r["theta"] + 0.5)
            if abs(r2["neg2_reml"] - r["neg2_reml"]) > 1e-3 and r2["neg2_reml"] < r["neg2_reml"] - 1e-3:
                pb["optimum instable"].append((i, r["neg2_reml"], r2["neg2_reml"]))
        except Exception:
            pb["optimum instable"].append((i, "exception au redemarrage"))
        if not bad:
            n_ok += 1
        if (i + 1) % 25 == 0:
            print("  %3d/%d ..." % (i + 1, a.n), flush=True)

    print("\n%d configurations tirees" % a.n)
    total = 0
    for k, v in pb.items():
        print("  %-20s %d" % (k, len(v)))
        total += len(v)
        for item in v[:4]:
            print("      ", item)
    print("\n%d/%d sans aucun probleme." % (a.n - total, a.n))
    return 1 if pb["exception"] or pb["logLik non finie"] or pb["Sigma non PSD"] else 0


if __name__ == "__main__":
    sys.exit(main())
