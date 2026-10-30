"""Genome-wide scan at fixed V, beside the solver.

No other module of remlax imports this file, and it modifies none. It reads
the output of `fit_reml` and nothing else. It needs neither JAX nor a GPU:
numpy is enough, because the solver has already done everything that had to be
differentiated.

The scan is an EMMAX-type GWAS (Kang et al. 2010). The variance components are
estimated once under the null hypothesis. Each SNP is then tested as a fixed
effect at fixed V. For a column s added to X,

    beta_s = (s'Ps)^-1 s'Py      var(beta_s) = (s'Ps)^-1      chi2 = (s'Py)^2/(s'Ps)
    P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

which is the formula of `fixed_tests.wald` written for one column. Since V does
not move, the score and the Wald statistics coincide.

The scan avoids a loop over SNPs in dimension n. The marker covariates live at
the genotype level (q of the order of a few hundred), while observations
number in the tens of thousands. P is projected once on the incidences of the
design,

    c_f = Z_f' P y        C_fg = Z_f' P Z_g

and the p statistics then follow from matrix products in dimension q:

    num_f(j)   = m_j' c_f
    den_fg(j)  = m_j' C_fg m_j

The cost drops from O(n^2 p) to O(q^2 p). The projection itself needs no new
factorisation, because `fit_reml` already returns `Vi`, `vbeta` and `Py`:

    C_fg = Z_f' Vi Z_g - (Z_f' Vi X) vbeta (X' Vi Z_g)

The covariates form families. With m_j the doses of SNP j, Z_dir the incidence
of the genotype carried by the plant and Z_ind the weighted incidence of the
genotypes of its neighbours:

    dir : s = Z_dir m_j                      DIRECT genetic effect of the SNP
    ind : s = Z_ind m_j                      INDIRECT genetic effect (neighbour dose)
    sim : s = (Z_dir m_j) * (Z_ind m_j)      allelic SIMILARITY focal x neighbour

`sim` is the covariate of Sato et al. (2021, 2024). With +/-1 coding the
product is +1 when the neighbour carries the same allele and -1 otherwise. It
is also the interaction term between the direct effect and the neighbour
effect, hence a DGE x IGE. The Sato model holds `dir` and `sim` without
`ind`. The additive IGE model of Muir (2005) and Bijma (2011) holds `dir` and
`ind` without `sim`. Neither tests what the other tests, and nested tests on
all three families together are more informative than either (docs/scan.md).

`dir` and `ind` are linear in m, so all their work happens in dimension q, in
O(q^2 p). `sim` is bilinear in m, because it depends on the product of the
focal and neighbour doses, so it does not reduce to genotype space. Its cross
terms with `dir` and `ind` stay cheap, because
    s_sim' P (Z_f m) = (s_sim' P Z_f) m
and the projection has already computed P Z_f. Only the term s_sim' P s_sim
needs a product in dimension n, which costs O(n^2) per SNP. It is the only
expensive step of the file. It is written as a blocked matrix product so that
BLAS, or a JAX array module passed as `xp`, can make the most of it.

`sim` is a similarity only if the doses are centred (+/-1 coding, or centred
doses). With 0/1/2 coding the product has no simple interpretation. `scan()`
therefore refuses the `sim` family on a dose matrix that is clearly not
centred, instead of returning a result that nothing flags as wrong.
"""
import numpy as np

FAMILIES = ("dir", "ind", "sim")

# Families linear in m: those that live entirely in genotype dimension.
LINEAR = ("dir", "ind")


# ==============================================================================
# Projection: P seen from the incidences of the design
# ==============================================================================
class Projection:
    """c_f = Z_f' P y, C_fg = Z_f' P Z_g, and P Z_f kept for the `sim` family.

    Attributes
    ----------
    c    : dict family -> vector (q_f,)
    C    : dict (family, family) -> matrix (q_f, q_g), symmetric in (f,g)
    PZ   : dict family -> matrix (n, q_f), used by the cross terms of `sim`
    Py   : vector (n,), taken from fit["Py"]
    q    : dict family -> q_f
    """

    def __init__(self, c, C, PZ, Py, Z, Vi, X, vbeta, ViX):
        self.c, self.C, self.PZ, self.Py, self.Z = c, C, PZ, Py, Z
        # References, no copies: Vi weighs 2 GB at n = 16 000 and already
        # exists in `fit`. `np.asarray` on a float64 array returns the same
        # object, so nothing is duplicated. Only the `sim` family uses these
        # attributes, because it alone needs P s for an s that is not an
        # incidence column.
        self._Vi, self._X, self._vbeta, self._ViX = Vi, X, vbeta, ViX
        self.q = {f: Z[f].shape[1] for f in Z}
        self.n = len(Py)

    def __repr__(self):
        return ("Projection(n=%d, %s)"
                % (self.n, ", ".join("%s: q=%d" % (f, q)
                                     for f, q in sorted(self.q.items()))))


def _dense(Z):
    """Accepts a numpy array, a numpy matrix or a scipy sparse matrix.
    Returns a 2-D array."""
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float64)


def project(fit, X, incidences):
    """Project P on the incidences, once for all families.

    Parameters
    ----------
    fit         : dict returned by `fit_reml`. It must hold `Vi`, `vbeta`, `Py`
                  (true as soon as `blups=True` or `hessian=True`, hence by
                  default). No other key is read.
    X           : fixed-effect matrix of the null model (n x p_fixed), the one
                  used for the fit.
    incidences  : dict {"dir": Z_dir, "ind": Z_ind}, with at least one key.
                  The two may have different numbers of columns. This is the
                  inter-species case, where the neighbours of a wheat plant
                  are alfalfa plants.

    Cost: one (n x n) by (n x q) product per family. No factorisation.
    """
    for key in ("Vi", "vbeta", "Py"):
        if key not in fit:
            raise KeyError(
                "fit['%s'] absent : `scan` lit la sortie de fit_reml et a besoin "
                "de Vi, vbeta et Py. Ils sont produits des que blups=True ou "
                "hessian=True (le defaut) ; un ajustement lance avec "
                "--no-blups --no-hessian ne les contient pas." % key)
    unknown = set(incidences) - set(LINEAR)
    if unknown:
        raise ValueError("incidences : cle(s) inconnue(s) %s ; attendu parmi %s"
                         % (sorted(unknown), list(LINEAR)))
    if not incidences:
        raise ValueError("incidences vide : fournir au moins 'dir' ou 'ind'.")

    Vi = np.asarray(fit["Vi"], dtype=np.float64)
    vbeta = np.asarray(fit["vbeta"], dtype=np.float64)
    Py = np.asarray(fit["Py"], dtype=np.float64).ravel()
    X = _dense(X)
    n = len(Py)
    if Vi.shape != (n, n):
        raise ValueError("Vi est %s mais Py a %d elements." % (Vi.shape, n))

    Z = {f: _dense(Zf) for f, Zf in incidences.items()}
    for f, Zf in Z.items():
        if Zf.shape[0] != n:
            raise ValueError("incidence '%s' a %d lignes, attendu %d."
                             % (f, Zf.shape[0], n))

    ViX = Vi @ X
    # P Z = Vi Z - ViX vbeta (X' Vi Z). This order avoids forming P (n x n).
    PZ, c, C = {}, {}, {}
    for f, Zf in Z.items():
        ViZ = Vi @ Zf
        PZ[f] = ViZ - ViX @ (vbeta @ (X.T @ ViZ))
        c[f] = Zf.T @ Py
    for f in Z:
        for g in Z:
            if (g, f) in C:
                C[(f, g)] = C[(g, f)].T
            else:
                C[(f, g)] = Z[f].T @ PZ[g]
    return Projection(c, C, PZ, Py, Z, Vi, X, vbeta, ViX)


# ==============================================================================
# Parsing of test specifications
# ==============================================================================
def _parse_test(spec):
    """'sim|dir+ind' -> (model, tested). 'dir+ind' -> (both, both).

    Left of the bar are the tested families. Right of the bar are the families
    conditioned on. The model is the union of both. Without a bar the whole
    model is tested, which is the joint test.
    """
    if "|" in spec:
        left, right = spec.split("|", 1)
        tested = [t.strip() for t in left.split("+") if t.strip()]
        cond = [t.strip() for t in right.split("+") if t.strip()]
    else:
        tested = [t.strip() for t in spec.split("+") if t.strip()]
        cond = []
    model = list(dict.fromkeys(tested + cond))
    bad = [f for f in model if f not in FAMILIES]
    if bad:
        raise ValueError("test '%s' : famille(s) inconnue(s) %s ; attendu parmi %s"
                         % (spec, bad, list(FAMILIES)))
    if not tested:
        raise ValueError("test '%s' : rien de teste a gauche de la barre." % spec)
    return model, tested


def required_families(tests):
    """The covariate families to compute for this list of tests."""
    needed = []
    for spec in tests:
        model, _ = _parse_test(spec)
        for f in model:
            if f not in needed:
                needed.append(f)
    return [f for f in FAMILIES if f in needed]


# ==============================================================================
# Scan
# ==============================================================================
def _markers(M, families):
    """Normalise M into a dict family -> matrix (q_f x p).

    A single array means the same SNPs on both sides (intra-species case). A
    dict gives one matrix per family. It allows the inter-species case, but
    then forbids joint tests, because column j of the two matrices does not
    refer to the same SNP. `scan` checks this.
    """
    if isinstance(M, dict):
        missing = [f for f in families if f not in M and f in LINEAR]
        if missing:
            raise ValueError("M : matrice de doses absente pour %s" % missing)
        out = {f: np.asarray(M[f], dtype=np.float64) for f in M}
    else:
        Ma = np.asarray(M, dtype=np.float64)
        out = {f: Ma for f in LINEAR}
    ps = {f: Mat.shape[1] for f, Mat in out.items()}
    if len(set(ps.values())) > 1:
        raise ValueError("M : nombres de SNP incompatibles entre familles : %s" % ps)
    return out, next(iter(ps.values()))


def _same_snps(Mf, families):
    """True if the families carry the same set of SNPs.

    Object identity is not a valid criterion. `scan_cli` reads the doses from
    two files, hence two distinct arrays, even in the intra-species case where
    the SNPs are the same. The content decides. The shapes are compared first,
    since a different number of genotypes marks the inter-species case. Then
    the arrays are compared element by element, in one pass.
    """
    lin = [f for f in families if f in LINEAR and f in Mf]
    if len(lin) < 2:
        return True
    ref = Mf[lin[0]]
    for f in lin[1:]:
        A = Mf[f]
        if A is ref:
            continue
        if A.shape != ref.shape or not np.array_equal(A, ref):
            return False
    return True


def scan(proj, M, tests=("dir", "ind", "sim"), keep=None, block=1024,
         xp=None, verbose=False):
    """Per-SNP statistics for the requested tests, at fixed V.

    Parameters
    ----------
    proj    : `Projection` returned by `project`.
    M       : matrix (q x p) of genotype doses, or dict family -> matrix.
              For the `sim` family the doses must be centred (typically +/-1
              coding). Otherwise the product is not a similarity.
    tests   : list of specifications, for example
                "dir"            direct effect alone, 1 df (standard GWAS)
                "ind"            additive indirect effect alone, 1 df
                "sim"            similarity alone, 1 df (Sato model)
                "dir+ind"        joint, 2 df
                "dir+ind+sim"    joint, 3 df
                "ind|dir"        indirect GIVEN the direct effect, 1 df
                "sim|dir+ind"    similarity given both additive effects
              Conditional tests are those of asreml and of
              `fixed_tests.wald`: each term given the others.
    keep    : boolean mask (p,) of the SNPs to process. The others come out
              as NaN. The allele frequency filter belongs to the caller.
    block   : width of the SNP blocks for the `sim` family (the only step in
              dimension n). It sets the peak memory and leaves the result
              unchanged.
    xp      : array module for the heavy product of `sim` (numpy by default).
              Passing `jax.numpy` moves it to the GPU and changes nothing else.

    Returns
    -------
    dict of vectors (p,):
      num_<f>, den_<f>_<g>          raw score quantities
      chi2_<test>, df_<test>, p_<test>
      beta_<test>, se_<test>        for 1-df tests only
      r_dir_ind                     correlation between the direct and
                                    indirect covariates, conditional on V and X
      df (degrees of freedom) and p are vectors so that they can be written
      to CSV as they are.
    """
    from scipy.stats import chi2 as _chi2
    xp = np if xp is None else xp
    needed = required_families(tests)
    Mf, p = _markers(M, needed)

    joint = [t for t in tests if len(_parse_test(t)[0]) > 1]
    if joint and not _same_snps(Mf, needed):
        raise ValueError(
            "tests conjoints %s demandes avec des jeux de marqueurs DIFFERENTS "
            "d'une famille a l'autre : la colonne j ne designe alors pas le meme "
            "SNP des deux cotes (cas inter-especes). Lancer les familles "
            "separement." % joint)

    for f in needed:
        if f in LINEAR and f not in proj.q:
            raise ValueError("famille '%s' demandee mais l'incidence correspondante "
                             "n'a pas ete projetee (incidences fournies : %s)"
                             % (f, sorted(proj.q)))
    if "sim" in needed:
        for f in LINEAR:
            if f not in proj.q:
                raise ValueError("la famille 'sim' est le produit des covariables "
                                 "directe et indirecte : les DEUX incidences sont "
                                 "necessaires (fournies : %s)" % sorted(proj.q))
        for f in LINEAR:
            mu = float(np.abs(Mf[f].mean()))
            sd = float(Mf[f].std())
            if sd > 0 and mu > 0.25 * sd:
                raise ValueError(
                    "famille 'sim' : les doses de '%s' ne sont pas centrees "
                    "(moyenne %.3g pour un ecart-type %.3g). Le produit des doses "
                    "n'est une similarite allelique qu'en codage centre (+/-1). "
                    "Centrer les doses, ou ne pas demander 'sim'." % (f, mu, sd))

    if keep is None:
        keep = np.ones(p, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (p,):
        raise ValueError("keep a %s elements, attendu (%d,)" % (keep.shape, p))
    idx = np.flatnonzero(keep)

    nan = lambda: np.full(p, np.nan)
    num = {f: nan() for f in needed}
    den = {}
    for i, f in enumerate(needed):
        for g in needed[i:]:
            den[(f, g)] = nan()

    # ---- linear families: everything in genotype dimension ---------------
    lin = [f for f in needed if f in LINEAR]
    for f in lin:
        num[f][idx] = Mf[f][:, idx].T @ proj.c[f]
    for i, f in enumerate(lin):
        for g in lin[i:]:
            A = proj.C[(f, g)] @ Mf[g][:, idx]
            den[(f, g)][idx] = np.einsum("ij,ij->j", Mf[f][:, idx], A)

    # ---- sim family: by blocks, the only step in dimension n -------------
    if "sim" in needed:
        Mdir, Mind = Mf["dir"], Mf["ind"]
        Zdir, Zind = proj.Z["dir"], proj.Z["ind"]
        Py = proj.Py
        PZdir, PZind = proj.PZ["dir"], proj.PZ["ind"]
        # P is never formed. P Z is already computed for the incidence
        # columns. s3 is not one of them, so the code computes
        #   P s3 = Vi s3 - ViX vbeta (X' Vi s3)
        # by blocks of SNPs. It is the only O(n^2) step of the file.
        Vi = proj._Vi
        for d0 in range(0, len(idx), int(block)):
            j = idx[d0:d0 + int(block)]
            u = Zdir @ Mdir[:, j]                       # (n, b)
            v = Zind @ Mind[:, j]                       # (n, b)
            S3 = u * v
            num["sim"][j] = S3.T @ Py
            # s_sim' P (Z_f m) = (s_sim' P Z_f) m: sum over the q genotypes,
            # one value per SNP of the block, hence "ij,ij->i" (rows are SNPs).
            if ("dir", "sim") in den:
                den[("dir", "sim")][j] = np.einsum(
                    "ij,ij->i", S3.T @ PZdir, Mdir[:, j].T)
            if ("ind", "sim") in den:
                den[("ind", "sim")][j] = np.einsum(
                    "ij,ij->i", S3.T @ PZind, Mind[:, j].T)
            S3x = xp.asarray(S3)
            PS3 = xp.asarray(Vi) @ S3x
            PS3 = PS3 - xp.asarray(proj._ViX) @ (
                xp.asarray(proj._vbeta) @ (xp.asarray(proj._X).T @ PS3))
            den[("sim", "sim")][j] = np.asarray(
                xp.einsum("ij,ij->j", S3x, PS3))
            if verbose:
                print("  [scan sim] %d / %d SNP" % (min(d0 + int(block), len(idx)),
                                                    len(idx)), flush=True)

    # ---- tests ------------------------------------------------------------
    out = {}
    for f in needed:
        out["num_" + f] = num[f]
    for (f, g), v in den.items():
        out["den_%s_%s" % (f, g)] = v

    def den_block(fams, j):
        k = len(fams)
        Dm = np.empty((len(j), k, k))
        for a, fa in enumerate(fams):
            for b, fb in enumerate(fams):
                key = (fa, fb) if (fa, fb) in den else (fb, fa)
                Dm[:, a, b] = den[key][j]
        return Dm

    for spec in tests:
        model, tested = _parse_test(spec)
        k, kt = len(model), len(tested)
        chi2v, betav, sev = nan(), nan(), nan()
        Dm = den_block(model, idx)
        Nv = np.stack([num[f][idx] for f in model], axis=1)        # (m, k)
        pos = [model.index(f) for f in tested]
        ok = np.isfinite(Dm).all(axis=(1, 2)) & np.isfinite(Nv).all(axis=1)
        if k == 1:
            d = Dm[:, 0, 0]
            good = ok & (d > 0)
            b = Nv[good, 0] / d[good]
            chi2v[idx[good]] = Nv[good, 0] ** 2 / d[good]
            betav[idx[good]] = b
            sev[idx[good]] = 1.0 / np.sqrt(d[good])
        else:
            # beta = D^-1 num, var(beta) = D^-1. Test of the subset A:
            #   chi2 = beta_A' [(D^-1)_AA]^-1 beta_A
            Di = np.full_like(Dm, np.nan)
            good = ok.copy()
            if good.any():
                sel = np.flatnonzero(good)
                try:
                    Di[sel] = np.linalg.inv(Dm[sel])
                except np.linalg.LinAlgError:
                    for s in sel:
                        try:
                            Di[s] = np.linalg.inv(Dm[s])
                        except np.linalg.LinAlgError:
                            good[s] = False
            beta = np.einsum("mab,mb->ma", Di, Nv)
            sub = Di[:, pos][:, :, pos]
            bA = beta[:, pos]
            for m in np.flatnonzero(good):
                try:
                    W = np.linalg.inv(sub[m])
                except np.linalg.LinAlgError:
                    continue
                chi2v[idx[m]] = float(bA[m] @ W @ bA[m])
            if kt == 1:
                betav[idx[good]] = bA[good, 0]
                d = sub[good, 0, 0]
                sev[idx[good]] = np.sqrt(np.maximum(d, 0.0))
        out["chi2_" + spec] = chi2v
        out["df_" + spec] = np.full(p, float(kt))
        with np.errstate(invalid="ignore"):
            out["p_" + spec] = _chi2.sf(chi2v, kt)
        if kt == 1:
            out["beta_" + spec] = betav
            out["se_" + spec] = sev

    if ("dir", "ind") in den:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r_dir_ind"] = den[("dir", "ind")] / np.sqrt(
                den[("dir", "dir")] * den[("ind", "ind")])
    return out


def lambda_gc(chi2, df=1):
    """Genomic control factor: median(chi2) / theoretical median."""
    from scipy.stats import chi2 as _chi2
    v = np.asarray(chi2, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v) / _chi2.ppf(0.5, df))
