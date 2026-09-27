"""Le scan rend-il EXACTEMENT ce que rendrait le test de Wald a V figee ?

POURQUOI CE TEST EXISTE. `scan.py` remplace une boucle sur les SNP par des
produits matriciels en dimension genotype. Le gain est d'un facteur de l'ordre
de mille, et un gain de cette taille est exactement le genre de chose qui cache
une erreur d'algebre. La reference n'est donc pas un nombre attendu ecrit a la
main : c'est la formule de `inference.wald` appliquee au modele ou la colonne du
SNP a ete AJOUTEE A X. Les deux doivent coincider a la precision machine.

CE TEST NE DEMANDE PAS JAX. `scan.py` consomme la sortie de `fit_reml` — Vi,
vbeta, Py — et rien d'autre. Le dispositif est donc monte en numpy, V est
assemblee a la main, et le dict passe a `projeter` est construit EXACTEMENT
comme `fit_reml` le construit (lignes 504-530 de fit.py). Le test tourne sur
n'importe quelle machine, sans GPU et sans conteneur.

LE DISPOSITIF CHOISI. Une residuelle AR1 dans les blocs et DEUX noyaux
genetiques (direct et indirect) avec leur covariance : c'est-a-dire un V que ni
gaston ni statgenGWAS ne savent diagonaliser. Si le scan est juste ici, il est
juste sur les cas plus simples.
"""
import sys

import numpy as np

sys.path.insert(0, "src")
try:
    from remlax.scan import projeter, scan, lambda_gc, familles_requises
except ImportError:
    # `remlax/__init__.py` importe `_x64`, donc JAX. `scan.py` n'en a pas
    # besoin : on le charge alors par son chemin, ce qui garde ce test
    # executable sur une machine sans JAX — la propriete meme qu'il verifie.
    import importlib.util
    import os
    _p = os.path.join(os.path.dirname(__file__), "..", "..", "src", "remlax",
                      "scan.py")
    _sp = importlib.util.spec_from_file_location("remlax_scan", _p)
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    projeter, scan = _m.projeter, _m.scan
    lambda_gc, familles_requises = _m.lambda_gc, _m.familles_requises

RNG = np.random.default_rng(31415)
TOL = 1e-8


# ==============================================================================
# Dispositif et « fit » monte a la main
# ==============================================================================
def dispositif(q=90, n_bloc=4, nrow=9, ncol=10, p=200, q2=None):
    """q genotypes en n_bloc blocs randomises de nrow x ncol, p SNP en +/-1.

    q2 : si fourni, les VOISINS appartiennent a un second jeu de q2 genotypes
    (cas inter-especes : les voisins d'un ble sont des luzernes).
    """
    assert nrow * ncol == q
    n = q * n_bloc
    geno = np.concatenate([RNG.permutation(q) for _ in range(n_bloc)])
    bloc = np.repeat(np.arange(n_bloc), q)
    rr = np.tile(np.repeat(np.arange(nrow), ncol), n_bloc)
    cc = np.tile(np.tile(np.arange(ncol), nrow), n_bloc)

    Zdir = np.zeros((n, q))
    Zdir[np.arange(n), geno] = 1.0

    qv = q if q2 is None else q2
    genov = geno if q2 is None else RNG.integers(0, q2, n)
    pos = {(b, r, c): i for i, (b, r, c) in enumerate(zip(bloc, rr, cc))}
    Zind = np.zeros((n, qv))
    for i in range(n):
        b, r, c = bloc[i], rr[i], cc[i]
        vois = [pos.get((b, r + dr, c + dc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        vois = [j for j in vois if j is not None]
        for j in vois:
            Zind[i, genov[j]] += 1.0 / len(vois)

    Mdir = np.where(RNG.random((q, p)) < 0.5, -1.0, 1.0)
    Mind = Mdir if q2 is None else np.where(RNG.random((qv, p)) < 0.5, -1.0, 1.0)
    X = np.zeros((n, n_bloc))
    X[np.arange(n), bloc] = 1.0
    return dict(n=n, q=q, qv=qv, bloc=bloc, Zdir=Zdir, Zind=Zind,
                Mdir=Mdir, Mind=Mind, X=X, p=p)


def ajuste(d, sD=1.0, sI=0.6, sDI=-0.2, se=1.0, rho=0.4):
    """V, y, et le dict que `fit_reml` renverrait — memes formules que fit.py."""
    n, X = d["n"], d["X"]
    Zdir, Zind = d["Zdir"], d["Zind"]
    Kd = (d["Mdir"] @ d["Mdir"].T) / d["Mdir"].shape[1] + 1e-6 * np.eye(d["q"])
    Ki = (d["Mind"] @ d["Mind"].T) / d["Mind"].shape[1] + 1e-6 * np.eye(d["qv"])
    A = Zdir @ Kd @ Zdir.T
    B = Zind @ Ki @ Zind.T
    idx = np.arange(n)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    R[d["bloc"][:, None] != d["bloc"][None, :]] = 0.0
    V = sD**2 * A + sI**2 * B + se**2 * R
    if d["qv"] == d["q"]:
        Cx = Zdir @ Kd @ Zind.T
        V = V + sDI * (Cx + Cx.T)
        V = V + (1e-8 + abs(sDI)) * np.eye(n)   # garde la definie-positivite

    y = (X @ np.array([1.0, 1.3, 0.7, -0.4])
         + np.linalg.cholesky(V + 1e-9 * np.eye(n)) @ RNG.normal(size=n))

    # --- exactement ce que fait fit_reml apres l'optimisation ---------------
    Vi = np.linalg.solve(V, np.eye(n))
    XtVi = X.T @ Vi
    Am = XtVi @ X
    beta = np.linalg.solve(Am, XtVi @ y)
    vbeta = np.linalg.pinv(Am)
    resid = y - X @ beta
    Py = Vi @ resid - Vi @ X @ (vbeta @ (XtVi @ resid))
    return V, y, dict(Vi=Vi, vbeta=vbeta, Py=Py, beta=beta)


# ==============================================================================
# Reference : la formule de Wald de inference.wald, colonnes ajoutees a X
# ==============================================================================
def wald_ref(y, X, V, cols, pos_testees=None):
    """chi2 et beta pour les colonnes `cols` ajoutees a X, a V figee.

    `pos_testees` : indices (dans `cols`) reellement testes. None = toutes,
    c'est-a-dire le test conjoint. Un sous-ensemble donne le test CONDITIONNEL
    de ces colonnes sachant les autres — la convention de `inference.wald`.
    """
    Xa = np.hstack([X, cols])
    k = cols.shape[1]
    Vi = np.linalg.pinv(V)
    Am = Xa.T @ Vi @ Xa
    Ai = np.linalg.pinv(Am)
    beta = Ai @ (Xa.T @ Vi @ y)
    sel = np.arange(k) if pos_testees is None else np.asarray(pos_testees)
    off = Xa.shape[1] - k
    b = beta[off + sel]
    W = Ai[np.ix_(off + sel, off + sel)]
    return float(b @ np.linalg.pinv(W) @ b), b, np.sqrt(np.diag(W))


def _cols(d, j):
    u = d["Zdir"] @ d["Mdir"][:, j]
    v = d["Zind"] @ d["Mind"][:, j]
    return {"dir": u, "ind": v, "sim": u * v}


# ==============================================================================
# Les tests
# ==============================================================================
CAS = [
    ("dir",          ["dir"],              None),
    ("ind",          ["ind"],              None),
    ("sim",          ["sim"],              None),
    ("dir+ind",      ["dir", "ind"],       None),
    ("dir+ind+sim",  ["dir", "ind", "sim"], None),
    ("ind|dir",      ["dir", "ind"],       [1]),
    ("dir|ind",      ["dir", "ind"],       [0]),
    ("sim|dir+ind",  ["dir", "ind", "sim"], [2]),
]


def scan_egale_wald(n_snp=25, verbose=True):
    d = dispositif()
    V, y, fit = ajuste(d)
    proj = projeter(fit, d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    res = scan(proj, {"dir": d["Mdir"], "ind": d["Mind"]},
               tests=[c[0] for c in CAS])

    pires = {}
    for spec, ordre, sel in CAS:
        err_x, err_b = 0.0, 0.0
        for j in range(n_snp):
            c = _cols(d, j)
            cols = np.column_stack([c[f] for f in ordre])
            x2, b, se = wald_ref(y, d["X"], V, cols, sel)
            got = res["chi2_" + spec][j]
            err_x = max(err_x, abs(got - x2) / max(abs(x2), 1e-12))
            if sel is not None and len(sel) == 1 or len(ordre) == 1:
                err_b = max(err_b, abs(res["beta_" + spec][j] - b[0])
                            / max(abs(b[0]), 1e-12))
                err_b = max(err_b, abs(res["se_" + spec][j] - se[0])
                            / max(abs(se[0]), 1e-12))
        pires[spec] = (err_x, err_b)
        if verbose:
            print("  %-14s chi2 %.2e | beta,se %.2e" % (spec, err_x, err_b))
        assert err_x < TOL, "%s : chi2 s'ecarte de %.2e" % (spec, err_x)
        assert err_b < TOL, "%s : beta/se s'ecarte de %.2e" % (spec, err_b)
    return pires


def correlation_conditionnelle(n_snp=15, verbose=True):
    """r_dir_ind est-il la vraie correlation entre covariables sachant V et X ?"""
    d = dispositif()
    V, y, fit = ajuste(d)
    proj = projeter(fit, d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    res = scan(proj, {"dir": d["Mdir"], "ind": d["Mind"]}, tests=["dir+ind"])
    Vi, X, vbeta = fit["Vi"], d["X"], fit["vbeta"]
    P = Vi - (Vi @ X) @ vbeta @ (X.T @ Vi)
    err = 0.0
    for j in range(n_snp):
        c = _cols(d, j)
        u, v = c["dir"], c["ind"]
        r = (u @ P @ v) / np.sqrt((u @ P @ u) * (v @ P @ v))
        err = max(err, abs(res["r_dir_ind"][j] - r))
    if verbose:
        print("  r_dir_ind : ecart max %.2e" % err)
    assert err < TOL
    return err


def test_refus_doses_non_centrees():
    """La famille sim doit REFUSER un codage 0/1/2 plutot que rendre du bruit."""
    d = dispositif(p=40)
    _, _, fit = ajuste(d)
    proj = projeter(fit, d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    M012 = (d["Mdir"] + 1.0)          # 0 / 2 : non centre
    try:
        scan(proj, {"dir": M012, "ind": M012}, tests=["sim"])
    except ValueError as e:
        assert "centre" in str(e), str(e)
        print("  refus obtenu : %s..." % str(e)[:60])
        return
    raise AssertionError("la famille sim a accepte des doses non centrees")


def test_refus_conjoint_inter_especes():
    """Deux jeux de SNP differents : le test conjoint n'a pas de sens."""
    d = dispositif(p=40, q2=57)
    _, _, fit = ajuste(d)
    proj = projeter(fit, d["X"], {"dir": d["Zdir"], "ind": d["Zind"]})
    try:
        scan(proj, {"dir": d["Mdir"], "ind": d["Mind"]}, tests=["dir+ind"])
    except ValueError as e:
        assert "inter-espece" in str(e), str(e)
        print("  refus obtenu : %s..." % str(e)[:60])
    else:
        raise AssertionError("test conjoint accepte sur deux jeux de SNP")
    # mais chaque famille SEULE doit passer, et valoir la reference
    res = scan(proj, {"dir": d["Mdir"], "ind": d["Mind"]}, tests=["ind"])
    V, y, fit2 = ajuste(d)
    assert np.isfinite(res["chi2_ind"]).all()


def test_analyse_specifications():
    assert familles_requises(["sim|dir+ind"]) == ["dir", "ind", "sim"]
    assert familles_requises(["ind"]) == ["ind"]
    # "dir|" est VALIDE et vaut "dir" : rien a droite de la barre, donc rien
    # sur quoi conditionner. Ce qui doit etre refuse, c'est une famille
    # inconnue ou l'absence de terme teste a GAUCHE.
    assert familles_requises(["dir|"]) == ["dir"]
    for mauvais in ["", "abc", "|dir", "dir+truc"]:
        try:
            familles_requises([mauvais])
        except ValueError:
            continue
        raise AssertionError("specification '%s' acceptee" % mauvais)
    print("  specifications : analysees et refusees comme prevu")



def test_lambda_gc():
    x2 = RNG.chisquare(1, 200000)
    lam = lambda_gc(x2, 1)
    assert abs(lam - 1.0) < 0.02, lam
    print("  lambda sur du chi2 pur : %.4f" % lam)



def test_scan_egale_wald():
    """Enveloppe pytest sans argument : pytest lirait n_snp et verbose comme des fixtures."""
    scan_egale_wald(verbose=False)


def test_correlation_conditionnelle():
    correlation_conditionnelle(verbose=False)


if __name__ == "__main__":
    print("scan == wald (colonne ajoutee a X) :")
    scan_egale_wald()
    print("diagnostic de colinearite :")
    correlation_conditionnelle()
    print("garde-fous :")
    test_refus_doses_non_centrees()
    test_refus_conjoint_inter_especes()
    test_analyse_specifications()
    test_lambda_gc()
    print("\nTOUT PASSE")
