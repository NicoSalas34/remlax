"""Les deux sorties ajoutees pour le pipeline IGE : loadings et PEV.

POURQUOI CE FICHIER EXISTE. Ces deux sorties ont ete ajoutees parce que
l'inventaire du depot IGE a montre qu'elles manquaient : `H2_Cullis` n'est pas
calculable sans la variance d'erreur de prediction, et un tableau de loadings ne
se deduit pas de Sigma, dont la decomposition n'est pas unique. Elles sont donc
sous surveillance ici plutot que verifiees une fois a la main.

LA PEV SE VERIFIE PAR SES PROPRIETES, pas contre une valeur de reference. Quatre
proprietes la contraignent completement, et chacune attraperait une erreur
differente :

  1. un niveau SANS observation a une PEV egale a la variance a priori -- son
     prediction n'apporte aucune information ;
  2. son BLUP est exactement nul ;
  3. un niveau observe a une PEV strictement inferieure a la variance a priori ;
  4. la PEV DECROIT avec le nombre d'observations du niveau.

La quatrieme est la plus utile : une erreur de signe ou une confusion entre G et
son inverse passerait les trois premieres et echouerait celle-la.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from remlax.fit import fit_reml                                    # noqa: E402


def dispositif(nu=400, t=4, q=50, seed=7):
    """Dispositif a replication INEGALE, et volontairement des niveaux vides.

    L'inegalite est le levier du test : sans elle la propriete 4 serait vide de
    contenu, toutes les PEV etant egales.
    """
    rng = np.random.default_rng(seed)
    gu = np.repeat(np.arange(q), nu // q + 1)[:nu]
    n = nu * t
    tri = np.tile(np.arange(t), nu)
    uni = np.repeat(np.arange(nu), t)
    zj = np.array([tri[i] * q + gu[uni[i]] for i in range(n)])
    S = np.diag([1.0, 0.7, 0.5, 0.3][:t]) + 0.3
    G0 = rng.normal(size=(q, t)) @ np.linalg.cholesky(S).T
    y = G0[gu[uni], tri] + rng.normal(0, 0.5, n)
    X = np.eye(t)[tri]
    terms = [dict(name="g", struct="fa", rank=2, t=t, q=q, zi=np.arange(n),
                  zj=zj, zx=np.ones(n), lvl="id", LK=None)]
    res = dict(struct="diag", t=t, rank=0, trait=tri, unit=uni, lvl="id")
    return terms, res, y, X, gu, q, t


def main():
    terms, res, y, X, gu, q, t = dispositif()
    o = fit_reml(terms, res, y, X, maxiter=500, polish=2, hessian=False,
                 blups=True, pev=True, verbose=False)
    ok = ko = 0

    def chk(nom, cond, detail=""):
        nonlocal ok, ko
        if cond:
            ok += 1; print("  ok    %-52s %s" % (nom, detail))
        else:
            ko += 1; print("  ECHEC %-52s %s" % (nom, detail))

    # ---- 1. les loadings reconstruisent Sigma -------------------------------
    lo = o["loadings"]["g"]
    Lam, psi = lo["Lambda"], lo["psi"]
    Sig = o["sigmas"]["g"]
    if Sig.shape[0] != Sig.shape[1]:
        Sig = Sig @ Sig.T
    e = np.abs(Lam @ Lam.T + np.diag(psi) - Sig).max()
    chk("Lambda Lambda' + diag(psi) = Sigma", e < 1e-12, "ecart max %.2e" % e)
    chk("forme de Lambda", Lam.shape == (t, 2), "%s" % (Lam.shape,))
    # psi est une VARIANCE, convention des tableaux d'asreml : la diagonale de
    # Sigma vaut somme(V_k^2) + psi, et non + psi^2.
    e2 = np.abs((Lam ** 2).sum(1) + psi - np.diag(Sig)).max()
    chk("psi est bien une variance specifique", e2 < 1e-12, "ecart max %.2e" % e2)

    # ---- 2. les quatre proprietes de la PEV ---------------------------------
    pv = o["pev"]["g"]
    dS = np.diag(Sig)
    vus, cnt = np.unique(gu, return_counts=True)
    vides = sorted(set(range(q)) - set(vus.tolist()))
    chk("le dispositif a bien des niveaux vides", len(vides) > 0, "%d vides" % len(vides))
    chk("PEV positive partout", bool((pv >= 0).all()), "min %.3e" % pv.min())
    chk("PEV bornee par la variance a priori",
        bool((pv <= dS[None, :] + 1e-10).all()))
    e3 = np.abs(pv[vides] - dS[None, :]).max()
    chk("niveau vide : PEV = variance a priori", e3 < 1e-12, "ecart max %.2e" % e3)
    e4 = np.abs(o["blups"]["g"][vides]).max()
    chk("niveau vide : BLUP nul", e4 < 1e-12, "max|u| %.2e" % e4)
    obs = [i for i in range(q) if i not in vides]
    chk("niveau observe : PEV < variance a priori",
        bool((pv[obs] < dS[None, :] - 1e-8).all()),
        "max %.4f contre %.4f" % (pv[obs].max(), dS.min()))
    r = np.corrcoef(cnt, pv[vus].mean(1))[0, 1]
    chk("la PEV decroit avec la replication", r < -0.9, "correlation %.3f" % r)

    print("\n%d verification(s), %d echec(s)" % (ok + ko, ko))
    return 1 if ko else 0


def test_loadings_pev():
    """Point d'entree pytest : main() rend 0 si tout passe (convention du script)."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
