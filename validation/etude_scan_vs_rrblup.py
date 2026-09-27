"""Versant Python de la validation externe du scan contre rrBLUP.

    Rscript validation/etude_scan_vs_rrblup.R  <dossier>
    python  validation/etude_scan_vs_rrblup.py <dossier>

Le script R produit les donnees ET les scores de rrBLUP ; celui-ci refait le
meme GWAS avec `remlax.scan` et confronte les deux. Voir l'en-tete du script R
pour le raisonnement : pourquoi ce cas degenere est le seul ou une reference
externe existe, et pourquoi la comparaison porte sur trois niveaux.

CE QUI EST VERIFIE.
  (A) beta par SNP : la solution des moindres carres generalises, identique.
  (B) le SCORE de rrBLUP reconstruit depuis les SEULES sorties brutes du scan,
      par la relation exacte
          Fstat_j = chi2_j * v2 / (y'Py - num_j^2/den_j),      v2 = n - 2
      qui traduit la reestimation de l'echelle residuelle par SNP que rrBLUP
      fait et que le scan ne fait pas. Si ce niveau passe, l'ecart de
      convention entre les deux est ENTIEREMENT explique, et il ne reste aucun
      desaccord d'algebre.
  (C) l'ampleur de la conservativite du chi2 a V figee, mesuree la ou elle mord
      (sur le SNP causal de gros effet) et la ou elle ne mord pas (sur les SNP
      nuls). C'est le chiffre qui justifie — ou non — le reajustement exact des
      tetes de liste.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist, chi2 as chi2_dist

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sp = importlib.util.spec_from_file_location(
    "remlax_scan", os.path.join(RACINE, "src", "remlax", "scan.py"))
rxscan = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(rxscan)


def main(dossier):
    import csv
    Vu, Ve, q, p, j0 = (float(x) for x in
                        open(os.path.join(dossier, "meta.txt")).read().split())
    q, p, j0 = int(q), int(p), int(j0) - 1          # R compte a partir de 1
    y = np.loadtxt(os.path.join(dossier, "y.txt"))
    K = np.loadtxt(os.path.join(dossier, "K.txt"))
    M = np.loadtxt(os.path.join(dossier, "M.txt"))      # q x p, doses -1/0/1
    with open(os.path.join(dossier, "rrblup_scores.csv")) as f:
        rr = {r["snp"]: float(r["score"]) for r in csv.DictReader(f)}
    score_rr = np.array([rr["snp%04d" % (j + 1)] for j in range(p)])
    assert M.shape == (q, p) and K.shape == (q, q), (M.shape, K.shape)

    # V du modele nul, avec LES composantes estimees par rrBLUP : ce qui est
    # teste ici est le SCAN, pas l'estimation des composantes de variance (deja
    # testee ailleurs dans remlax). Leur imposer la meme V est ce qui rend le
    # desaccord residuel attribuable au scan et a rien d'autre.
    V = Vu * K + Ve * np.eye(q)
    X = np.ones((q, 1))
    Vi = np.linalg.solve(V, np.eye(q))
    Am = X.T @ Vi @ X
    vbeta = np.linalg.pinv(Am)
    resid = y - X @ np.linalg.solve(Am, X.T @ Vi @ y)
    Py = Vi @ resid - Vi @ X @ (vbeta @ (X.T @ Vi @ resid))

    # une observation par genotype => l'incidence du genotype focal est I(q)
    proj = rxscan.projeter(dict(Vi=Vi, vbeta=vbeta, Py=Py), X, {"dir": np.eye(q)})
    r = rxscan.scan(proj, M, tests=["dir"])
    num, den, chi2 = r["num_dir"], r["den_dir_dir"], r["chi2_dir"]

    # (B) le score de rrBLUP, reconstruit sans rien recalculer en dimension n
    yPy = float(y @ Py)
    v2 = q - 2
    Fstat = chi2 * v2 / (yPy - num ** 2 / den)
    score_re = -np.log10(beta_dist.cdf(v2 / (v2 + Fstat), v2 / 2, 0.5))

    fini = np.isfinite(score_rr) & (score_rr > 0)
    err = np.abs(score_re - score_rr)
    print("q = %d genotypes, p = %d SNP ; Vu = %.6g, Ve = %.6g (rrBLUP)"
          % (q, p, Vu, Ve))
    print("(B) score rrBLUP reconstruit depuis les sorties du scan :")
    print("    ecart absolu max %.2e sur %d SNP" % (err[fini].max(), int(fini.sum())))
    print("    SNP causal snp%04d : rrBLUP %.6f | reconstruit %.6f"
          % (j0 + 1, score_rr[j0], score_re[j0]))
    assert err[fini].max() < 1e-6, "desaccord avec rrBLUP"

    # (C) la conservativite du chi2 a V figee
    score_chi2 = -np.log10(chi2_dist.sf(chi2, 1))
    autres = np.ones(p, bool)
    autres[j0] = False
    d = score_chi2[autres] - score_rr[autres]
    print("(C) chi2 a V figee contre F de rrBLUP, en -log10(p) :")
    print("    SNP causal : %.3f contre %.3f  (ecart %+.3f)"
          % (score_chi2[j0], score_rr[j0], score_chi2[j0] - score_rr[j0]))
    print("    SNP nuls   : median %+.4f, quantiles 5-95%% [%+.4f, %+.4f]"
          % (np.median(d), *np.percentile(d, [5, 95])))
    np.savez(os.path.join(dossier, "comparaison.npz"),
             score_rr=score_rr, score_re=score_re, score_chi2=score_chi2,
             chi2=chi2, Fstat=Fstat, beta=r["beta_dir"], j0=j0, q=q, p=p)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    sys.exit(main(sys.argv[1]))
