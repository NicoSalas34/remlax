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
#
# Multi-trait models. In long format (one row per unit x trait, T traits), the
# covariate of family f for trait k is the family covariate kept on the rows
# of trait k, s_{f,k} = D_k s_f. Its incidence D_k Z_f is the per-trait
# incidence that a multi-trait term already holds. Each family thus becomes T
# BASE COLUMNS (f, k), and every quantity below is indexed by them:
#     c_(f,k) = Z_f' D_k P y        C_(f,k),(g,l) = Z_f' D_k P D_l Z_g
# A univariate model is the case T = 1, with the same arithmetic as before.
class Projection:
    """c, C and P Z indexed by base columns (family, trait), and P Z kept for
    the `sim` family.

    Attributes
    ----------
    c       : dict (f, k) -> vector (q_f,)
    C       : dict ((f, k), (g, l)) -> matrix (q_f, q_g)
    PZ      : dict (f, k) -> matrix (n, q_f), used by the cross terms of `sim`
    Z       : dict (f, k) -> per-trait incidence (n, q_f)
    Zfull   : dict f -> full-height incidence sum_k Z_(f,k), used by `sim`
    masks   : list of T boolean vectors (n,), rows of each trait
    traits  : list of T trait names ("" when T = 1)
    Py      : vector (n,), taken from fit["Py"]
    q       : dict family -> q_f
    """

    def __init__(self, c, C, PZ, Py, Z, Zfull, masks, traits, Vi, X, vbeta, ViX):
        self.c, self.C, self.PZ, self.Py, self.Z = c, C, PZ, Py, Z
        self.Zfull, self.masks, self.traits = Zfull, masks, traits
        # References, no copies: Vi weighs 2 GB at n = 16 000 and already
        # exists in `fit`. `np.asarray` on a float64 array returns the same
        # object, so nothing is duplicated. Only the `sim` family uses these
        # attributes, because it alone needs P s for an s that is not an
        # incidence column.
        self._Vi, self._X, self._vbeta, self._ViX = Vi, X, vbeta, ViX
        self.q = {f: Zf.shape[1] for f, Zf in Zfull.items()}
        self.n = len(Py)
        self.T = len(traits)

    def __repr__(self):
        return ("Projection(n=%d, T=%d, %s)"
                % (self.n, self.T, ", ".join("%s: q=%d" % (f, q)
                                             for f, q in sorted(self.q.items()))))


def _dense(Z):
    """Accepts a numpy array, a numpy matrix or a scipy sparse matrix.
    Returns a 2-D array."""
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float64)


def _trait_masks(inc, n, trait, traits):
    """(masks, names, T). The rows of each trait come from the trait codes
    when given, otherwise from the per-trait incidences."""
    T_inc = max(len(v) for v in inc.values())
    if trait is not None:
        masks, names = _masks_from_codes(trait, n, traits)
        if T_inc > 1 and len(masks) != T_inc:
            raise ValueError("%d traits in `trait`, but %d per-trait incidences."
                             % (len(masks), T_inc))
        return masks, names, len(masks)
    if T_inc == 1:
        return [np.ones(n, dtype=bool)], [""], 1
    names = [str(t) for t in traits] if traits is not None else [str(k + 1) for k in range(T_inc)]
    return _masks_from_incidences(inc, n, T_inc), names, T_inc


def _masks_from_codes(trait, n, traits):
    """Integer codes 0..T-1 in the order of `traits`, or labels."""
    codes = np.asarray(trait)
    if codes.shape != (n,):
        raise ValueError("trait has %s elements, expected (%d,)" % (codes.shape, n))
    if traits is None:
        lev = list(dict.fromkeys(codes.tolist()))
        return [codes == l for l in lev], [str(l) for l in lev]
    names = [str(t) for t in traits]
    if codes.dtype.kind in "iu":
        return [codes == k for k in range(len(names))], names
    return [codes.astype(str) == nm for nm in names], names


def _masks_from_incidences(inc, n, T):
    """Rows where a per-trait incidence is nonzero: every record carries the
    genotype of its plant."""
    masks = []
    for k in range(T):
        m = np.zeros(n, dtype=bool)
        for v in inc.values():
            if len(v) == T:
                m |= np.abs(v[k]).sum(axis=1) > 0
        masks.append(m)
    return masks


def _fit_arrays(fit):
    """Vi, vbeta and Py of a fit, checked."""
    for key in ("Vi", "vbeta", "Py"):
        if key not in fit:
            raise KeyError(
                "fit['%s'] missing. `scan` reads the output of fit_reml and needs "
                "Vi, vbeta and Py. They are produced as soon as blups=True or "
                "hessian=True (the default). A fit run with "
                "--no-blups --no-hessian does not contain them." % key)
    Vi = np.asarray(fit["Vi"], dtype=np.float64)
    vbeta = np.asarray(fit["vbeta"], dtype=np.float64)
    Py = np.asarray(fit["Py"], dtype=np.float64).ravel()
    if Vi.shape != (len(Py), len(Py)):
        raise ValueError("Vi is %s but Py has %d elements." % (Vi.shape, len(Py)))
    return Vi, vbeta, Py


def _incidence_lists(incidences, n):
    """family -> list of dense per-trait incidences (one element when the
    family is given as a single matrix), checked."""
    unknown = set(incidences) - set(LINEAR)
    if unknown:
        raise ValueError("incidences: unknown key(s) %s. Expected among %s"
                         % (sorted(unknown), list(LINEAR)))
    if not incidences:
        raise ValueError("incidences empty: supply at least 'dir' or 'ind'.")
    inc = {f: ([_dense(z) for z in Zf] if isinstance(Zf, (list, tuple)) else [_dense(Zf)])
           for f, Zf in incidences.items()}
    for f, v in inc.items():
        for Zk in v:
            if Zk.shape[0] != n:
                raise ValueError("incidence '%s' has %d rows, expected %d."
                                 % (f, Zk.shape[0], n))
        if len({Zk.shape[1] for Zk in v}) > 1:
            raise ValueError("incidence '%s': per-trait matrices with different "
                             "numbers of columns." % f)
    if len({len(v) for v in inc.values()} - {1}) > 1:
        raise ValueError("incidences: different numbers of traits between families "
                         "(%s)." % {f: len(v) for f, v in inc.items()})
    return inc


def project(fit, X, incidences, trait=None, traits=None):
    """Project P on the incidences, once for all families.

    Parameters
    ----------
    fit         : dict returned by `fit_reml`. It must hold `Vi`, `vbeta`, `Py`
                  (true as soon as `blups=True` or `hessian=True`, hence by
                  default). No other key is read.
    X           : fixed-effect matrix of the null model (n x p_fixed), the one
                  used for the fit.
    incidences  : dict {"dir": Z_dir, "ind": Z_ind}, with at least one key.
                  Each value is a matrix (n x q_f), or, for a multi-trait
                  model, a list of T per-trait incidences (the `Zl` of a
                  multi-trait term). A single matrix with `trait` given is
                  split by trait. The two families may have different numbers
                  of columns: this is the inter-species case, where the
                  neighbours of a wheat plant are alfalfa plants.
    trait       : optional trait of each row (n,), codes or labels. Needed to
                  split a single matrix; otherwise read from the incidences.
    traits      : optional trait names, in the order of the per-trait lists.

    Cost: one (n x n) by (n x q) product per base column. No factorisation.
    """
    Vi, vbeta, Py = _fit_arrays(fit)
    X = _dense(X)
    n = len(Py)
    inc = _incidence_lists(incidences, n)
    masks, traits, T = _trait_masks(inc, n, trait, traits)
    # A single matrix in a multi-trait model is split by trait.
    for f in inc:
        if len(inc[f]) == 1 and T > 1:
            inc[f] = [inc[f][0] * masks[k][:, None] for k in range(T)]
    Z = {(f, k): inc[f][k] for f in inc for k in range(T)}
    Zfull = {f: (inc[f][0] if T == 1 else sum(inc[f])) for f in inc}

    ViX = Vi @ X
    # P Z = Vi Z - ViX vbeta (X' Vi Z). This order avoids forming P (n x n).
    PZ, c, C = {}, {}, {}
    for b, Zb in Z.items():
        ViZ = Vi @ Zb
        PZ[b] = ViZ - ViX @ (vbeta @ (X.T @ ViZ))
        c[b] = Zb.T @ Py
    for a in Z:
        for b in Z:
            if (b, a) in C:
                C[(a, b)] = C[(b, a)].T
            else:
                C[(a, b)] = Z[a].T @ PZ[b]
    return Projection(c, C, PZ, Py, Z, Zfull, masks, traits, Vi, X, vbeta, ViX)


# ==============================================================================
# Parsing of test specifications
# ==============================================================================
#   spec  := items [ "|" items ]         tested left of the bar, conditioned right
#   items := item { "+" item }
#   item  := family [ "[" trait {"," trait} "]" ] [ "@common" | "@het" ]
# With T traits:
#   dir          beta_dir = 0 on every trait (any effect, T df)
#   dir[a]       beta_dir,a = 0, the other traits of dir free (1 df)
#   dir@common   one beta shared by every trait = 0, model column sum_k s_dir,k (1 df)
#   dir@het      beta_dir,1 = ... = beta_dir,T (heterogeneity, T - 1 df)
# A conditioning family enters the model with all its traits (or its common
# column with @common). With T = 1 the grammar is the univariate one.
_ITEM = None


def _parse_item(item, spec):
    import re
    global _ITEM
    if _ITEM is None:
        _ITEM = re.compile(r"\s*([A-Za-z]+)\s*(?:\[([^\]]*)\])?\s*(?:@\s*([A-Za-z]+))?\s*")
    m = _ITEM.fullmatch(item)
    if not m:
        raise ValueError("test '%s': cannot read '%s'." % (spec, item))
    fam, tr, mod = m.groups()
    traits = [t.strip() for t in tr.split(",") if t.strip()] if tr is not None else None
    _check_item(spec, fam, traits, mod)
    return fam, traits, mod


def _check_item(spec, fam, traits, mod):
    if fam not in FAMILIES:
        raise ValueError("test '%s': unknown family(ies) %s. Expected among %s"
                         % (spec, [fam], list(FAMILIES)))
    if mod is not None and mod not in ("common", "het"):
        raise ValueError("test '%s': unknown modifier '@%s' (expected @common or @het)."
                         % (spec, mod))
    if traits is not None and mod is not None:
        raise ValueError("test '%s': '[traits]' and '@%s' cannot be combined." % (spec, mod))


def _parse_test_full(spec):
    """'sim|dir+ind' -> (tested items, conditioning items); each item is
    (family, traits or None, modifier or None)."""
    left, right = (spec.split("|", 1) + [""])[:2] if "|" in spec else (spec, "")
    tested = [_parse_item(t, spec) for t in left.split("+") if t.strip()]
    cond = [_parse_item(t, spec) for t in right.split("+") if t.strip()]
    if not tested:
        raise ValueError("test '%s': nothing tested left of the bar." % spec)
    return tested, cond


def _parse_test(spec):
    """'sim|dir+ind' -> (model families, tested families). Without a bar the
    whole model is tested, which is the joint test."""
    tested, cond = _parse_test_full(spec)
    t = [f for f, _, _ in tested]
    model = list(dict.fromkeys(t + [f for f, _, _ in cond]))
    return model, list(dict.fromkeys(t))


def required_families(tests):
    """The covariate families to compute for this list of tests."""
    needed = []
    for spec in tests:
        model, _ = _parse_test(spec)
        for f in model:
            if f not in needed:
                needed.append(f)
    return [f for f in FAMILIES if f in needed]


def _design(spec, T, traits):
    """Model columns of a test as combinations of base columns, and the matrix L
    of the tested linear hypothesis on the model coefficients.

    Returns (cols, L, labels): cols a list of dicts {(f, k): weight}, one per
    model column; L (r x m); labels of the tested rows (for the 1-df
    outputs)."""
    tested, cond = _parse_test_full(spec)
    cols, where = _model_columns(spec, tested + cond, T)
    rows, labels = [], []
    for item in tested:
        r, lab = _hypothesis_rows(spec, item, where, len(cols), T, list(traits))
        rows += r; labels += lab
    L = np.array(rows)
    _, keep = np.unique(L, axis=0, return_index=True)
    keep = np.sort(keep)
    return cols, L[keep], [labels[i] for i in keep]


def _model_columns(spec, items, T):
    """Model columns in order of appearance: T unit columns per family, or one
    common column (sum over traits) for a family with @common. `where` maps
    (family, "all" | "common") to the positions of its columns."""
    cols, where = [], {}
    for f, _, mod in items:
        mode = "common" if mod == "common" else "all"
        other = (f, "all" if mode == "common" else "common")
        if other in where:
            raise ValueError("test '%s': family '%s' enters the model both with and "
                             "without @common." % (spec, f))
        if (f, mode) in where:
            continue
        if mode == "common":
            where[(f, mode)] = [len(cols)]
            cols.append({(f, k): 1.0 for k in range(T)})
        else:
            where[(f, mode)] = list(range(len(cols), len(cols) + T))
            cols.extend({(f, k): 1.0} for k in range(T))
    return cols, where


def _hypothesis_rows(spec, item, where, m, T, names):
    """Rows of L for one tested item, and their labels."""
    f, tr, mod = item

    def unit(i, minus=None):
        r = np.zeros(m); r[i] = 1.0
        if minus is not None:
            r[minus] = -1.0
        return r
    if mod == "common":
        return [unit(where[(f, "common")][0])], ["%s@common" % f]
    idx = where[(f, "all")]
    if mod == "het":
        if T < 2:
            raise ValueError("test '%s': @het needs at least two traits." % spec)
        return ([unit(idx[k], idx[0]) for k in range(1, T)],
                ["%s[%s-%s]" % (f, names[k], names[0]) for k in range(1, T)])
    ks = range(T) if tr is None else [_trait_index(t, names, spec) for t in tr]
    return ([unit(idx[k]) for k in ks],
            [f if T == 1 else "%s[%s]" % (f, names[k]) for k in ks])


def _trait_index(t, names, spec):
    if t in names:
        return names.index(t)
    if t.isdigit() and 1 <= int(t) <= len(names):
        return int(t) - 1
    raise ValueError("test '%s': unknown trait '%s' (traits: %s)." % (spec, t, names))


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
            raise ValueError("M: dose matrix missing for %s" % missing)
        out = {f: np.asarray(M[f], dtype=np.float64) for f in M}
    else:
        Ma = np.asarray(M, dtype=np.float64)
        out = {f: Ma for f in LINEAR}
    ps = {f: Mat.shape[1] for f, Mat in out.items()}
    if len(set(ps.values())) > 1:
        raise ValueError("M: SNP counts inconsistent between families: %s" % ps)
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


def _key(b, T, names):
    f, k = b
    return f if T == 1 else "%s[%s]" % (f, names[k])


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
                "dir"            direct effect alone (1 df; T df with T traits)
                "ind"            additive indirect effect alone
                "sim"            similarity alone (Sato model)
                "dir+ind"        joint
                "ind|dir"        indirect GIVEN the direct effect
                "sim|dir+ind"    similarity given both additive effects
              and, with T traits (multi-trait null model):
                "dir[y1]"        one trait, the others free (1 df)
                "dir@common"     one effect shared by every trait (1 df)
                "dir@het"        heterogeneity between traits (T - 1 df)
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
      num_<b>, den_<b>_<b'>         raw score quantities, b a base column
                                    (family, or family[trait] with T traits)
      chi2_<test>, df_<test>, p_<test>
      beta_<test>, se_<test>        for 1-df tests only
      beta_<f>[<trait>], se_...     with T traits, per-trait effects of the
                                    model of each single-family test
      r_dir_ind                     correlation between the direct and
                                    indirect covariates (T = 1)
      df (degrees of freedom) and p are vectors so that they can be written
      to CSV as they are.
    """
    xp = np if xp is None else xp
    needed = required_families(tests)
    Mf, p = _markers(M, needed)
    T, names = proj.T, proj.traits
    _check_scan_inputs(proj, Mf, needed, tests)
    designs = {spec: _design(spec, T, names) for spec in tests}
    idx = _kept_snps(keep, p)

    base = [(f, k) for f in needed for k in range(T)]
    num, den = _empty_statistics(base, p)
    _linear_statistics(proj, Mf, base, idx, num, den)
    if "sim" in needed:
        _sim_statistics(proj, Mf, base, idx, num, den, block, xp, verbose)

    out = _raw_outputs(num, den, T, names)
    Nb, Db = _stack_statistics(num, den, base, idx)
    for spec in tests:
        out.update(_test_outputs(spec, designs[spec], Nb, Db, base, idx, p, T, names))
    if T == 1 and (("dir", 0), ("ind", 0)) in den:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r_dir_ind"] = den[(("dir", 0), ("ind", 0))] / np.sqrt(
                den[(("dir", 0), ("dir", 0))] * den[(("ind", 0), ("ind", 0))])
    return out


def _check_scan_inputs(proj, Mf, needed, tests):
    """Refuse what the scan cannot interpret: joint tests on different marker
    sets, families without their incidence, `sim` on doses that are not
    centred."""
    joint = [t for t in tests if len(_parse_test(t)[0]) > 1]
    if joint and not _same_snps(Mf, needed):
        raise ValueError(
            "joint tests %s requested with DIFFERENT marker sets "
            "from one family to the other. Column j then does not denote the same "
            "SNP on both sides (between-species case). Run the families "
            "separately." % joint)
    for f in needed:
        if f in LINEAR and f not in proj.q:
            raise ValueError("family '%s' requested but the matching incidence "
                             "was not projected (incidences given: %s)"
                             % (f, sorted(proj.q)))
    if "sim" in needed:
        _check_sim_inputs(proj, Mf)


def _check_sim_inputs(proj, Mf):
    """`sim` needs both incidences and centred doses."""
    for f in LINEAR:
        if f not in proj.q:
            raise ValueError("the family 'sim' is the product of the direct "
                             "and indirect covariates. BOTH incidences are "
                             "needed (given: %s)" % sorted(proj.q))
    for f in LINEAR:
        mu = float(np.abs(Mf[f].mean()))
        sd = float(Mf[f].std())
        if sd > 0 and mu > 0.25 * sd:
            raise ValueError(
                "family 'sim': the doses of '%s' are not centred "
                "(mean %.3g for a standard deviation %.3g). The product of the doses "
                "is an allelic similarity only under centred coding (+/-1). "
                "Centre the doses, or do not request 'sim'." % (f, mu, sd))


def _kept_snps(keep, p):
    """Indices of the SNPs to process."""
    if keep is None:
        keep = np.ones(p, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (p,):
        raise ValueError("keep has %s elements, expected (%d,)" % (keep.shape, p))
    return np.flatnonzero(keep)


def _empty_statistics(base, p):
    """num[b] and den[(a, b)] (a before b in `base`), NaN until computed."""
    num = {b: np.full(p, np.nan) for b in base}
    den = {}
    for i, a in enumerate(base):
        for b in base[i:]:
            den[(a, b)] = np.full(p, np.nan)
    return num, den


def _linear_statistics(proj, Mf, base, idx, num, den):
    """`dir` and `ind` columns: everything in genotype dimension."""
    lin = [b for b in base if b[0] in LINEAR]
    for b in lin:
        num[b][idx] = Mf[b[0]][:, idx].T @ proj.c[b]
    for i, a in enumerate(lin):
        for b in lin[i:]:
            A = proj.C[(a, b)] @ Mf[b[0]][:, idx]
            den[(a, b)][idx] = np.einsum("ij,ij->j", Mf[a[0]][:, idx], A)


def _sim_statistics(proj, Mf, base, idx, num, den, block, xp, verbose):
    """`sim` columns, by blocks of SNPs: the only step in dimension n.

    P is never formed. P Z is already computed for the incidence columns. s3
    is not one of them, so the code computes P s3 = Vi s3 - ViX vbeta (X' Vi s3)
    by blocks of SNPs, one block of columns per trait."""
    T = proj.T
    lin = [b for b in base if b[0] in LINEAR]
    Mdir, Mind = Mf["dir"], Mf["ind"]
    Zdir, Zind = proj.Zfull["dir"], proj.Zfull["ind"]
    for d0 in range(0, len(idx), int(block)):
        j = idx[d0:d0 + int(block)]
        S3 = (Zdir @ Mdir[:, j]) * (Zind @ Mind[:, j])            # (n, b)
        S3k = [S3 if T == 1 else S3 * proj.masks[k][:, None] for k in range(T)]
        PS3 = [_project_columns(proj, S3k[k], xp) for k in range(T)]
        for k in range(T):
            sk = ("sim", k)
            num[sk][j] = S3k[k].T @ proj.Py
            # s_sim' P (Z_b m) = (s_sim' P Z_b) m: sum over the q genotypes,
            # one value per SNP of the block, hence "ij,ij->i".
            for b in lin:
                key = (b, sk) if (b, sk) in den else (sk, b)
                den[key][j] = np.einsum("ij,ij->i", S3k[k].T @ proj.PZ[b],
                                        Mf[b[0]][:, j].T)
            for l in range(k, T):
                den[(sk, ("sim", l))][j] = np.asarray(
                    xp.einsum("ij,ij->j", xp.asarray(S3k[k]), PS3[l]))
        if verbose:
            print("  [scan sim] %d / %d SNP" % (min(d0 + int(block), len(idx)),
                                                len(idx)), flush=True)


def _project_columns(proj, S, xp):
    """P S = Vi S - ViX vbeta (X' Vi S), with the array module xp."""
    Sx = xp.asarray(S)
    PS = xp.asarray(proj._Vi) @ Sx
    return PS - xp.asarray(proj._ViX) @ (
        xp.asarray(proj._vbeta) @ (xp.asarray(proj._X).T @ PS))


def _raw_outputs(num, den, T, names):
    """num_<b> and den_<a>_<b>, named by family (and trait)."""
    out = {}
    for b, v in num.items():
        out["num_" + _key(b, T, names)] = v
    for (a, b), v in den.items():
        out["den_%s_%s" % (_key(a, T, names), _key(b, T, names))] = v
    return out


def _stack_statistics(num, den, base, idx):
    """Nb (m, B) and the symmetric Db (m, B, B) over the kept SNPs."""
    pos_of = {b: i for i, b in enumerate(base)}
    Nb = np.stack([num[b][idx] for b in base], axis=1)
    Db = np.empty((len(idx), len(base), len(base)))
    for (a, b), v in den.items():
        Db[:, pos_of[a], pos_of[b]] = v[idx]
        Db[:, pos_of[b], pos_of[a]] = v[idx]
    return Nb, Db


def _model_statistics(cols, base, Nb, Db):
    """num and den of the model columns. W maps base columns to model
    columns; a pure selection keeps the univariate arithmetic exactly."""
    pos_of = {b: i for i, b in enumerate(base)}
    W = np.zeros((len(base), len(cols)))
    for c_, wd in enumerate(cols):
        for b, w in wd.items():
            W[pos_of[b], c_] = w
    sel = [int(np.flatnonzero(W[:, c_])[0]) if (np.count_nonzero(W[:, c_]) == 1
           and W[:, c_].max() == 1.0) else -1 for c_ in range(W.shape[1])]
    if all(s >= 0 for s in sel):
        return Nb[:, sel], Db[:, sel][:, :, sel]
    return Nb @ W, np.einsum("ba,mbc,cd->mad", W, Db, W)


def _test_outputs(spec, design, Nb, Db, base, idx, p, T, names):
    """chi2, df and p of one test, beta and se when it has one df, and the
    per-trait effects of a single-family test with T traits."""
    from scipy.stats import chi2 as _chi2
    cols, L, _ = design
    kt = L.shape[0]
    Nv, Dm = _model_statistics(cols, base, Nb, Db)
    ok = np.isfinite(Dm).all(axis=(1, 2)) & np.isfinite(Nv).all(axis=1)
    if Dm.shape[1] == 1:
        chi2v, betav, sev = _one_column_test(Nv, Dm, ok, idx, p)
        out = {}
    else:
        chi2v, betav, sev, out = _block_test(spec, L, Nv, Dm, ok, idx, p, T, names)
    out["chi2_" + spec] = chi2v
    out["df_" + spec] = np.full(p, float(kt))
    with np.errstate(invalid="ignore"):
        out["p_" + spec] = _chi2.sf(chi2v, kt)
    if kt == 1:
        out["beta_" + spec] = betav
        out["se_" + spec] = sev
    return out


def _one_column_test(Nv, Dm, ok, idx, p):
    """beta = num/den, var(beta) = 1/den, chi2 = num^2/den."""
    chi2v, betav, sev = np.full(p, np.nan), np.full(p, np.nan), np.full(p, np.nan)
    d = Dm[:, 0, 0]
    good = ok & (d > 0)
    chi2v[idx[good]] = Nv[good, 0] ** 2 / d[good]
    betav[idx[good]] = Nv[good, 0] / d[good]
    sev[idx[good]] = 1.0 / np.sqrt(d[good])
    return chi2v, betav, sev


def _block_test(spec, L, Nv, Dm, ok, idx, p, T, names):
    """beta = D^-1 num, var(beta) = D^-1. Test of the hypothesis L beta = 0:
        chi2 = (L beta)' [L D^-1 L']^-1 (L beta)
    (a subset A of the coefficients when L selects them)."""
    chi2v, betav, sev = np.full(p, np.nan), np.full(p, np.nan), np.full(p, np.nan)
    Di, good = _batched_inverse(Dm, ok)
    beta = np.einsum("mab,mb->ma", Di, Nv)
    if np.all((L == 0) | (L == 1)) and np.all(L.sum(axis=1) == 1):
        pos = [int(np.flatnonzero(r)[0]) for r in L]
        sub = Di[:, pos][:, :, pos]
        bA = beta[:, pos]
    else:
        sub = np.einsum("ra,mab,sb->mrs", L, Di, L)
        bA = beta @ L.T
    for m_ in np.flatnonzero(good):
        try:
            Wm = np.linalg.inv(sub[m_])
        except np.linalg.LinAlgError:
            continue
        chi2v[idx[m_]] = float(bA[m_] @ Wm @ bA[m_])
    if L.shape[0] == 1:
        betav[idx[good]] = bA[good, 0]
        sev[idx[good]] = np.sqrt(np.maximum(sub[good, 0, 0], 0.0))
    return chi2v, betav, sev, _per_trait_effects(spec, beta, Di, good, idx, p, T, names)


def _batched_inverse(Dm, ok):
    """Inverses of the SNP blocks; a singular block leaves its SNP out."""
    Di = np.full_like(Dm, np.nan)
    good = ok.copy()
    if good.any():
        selg = np.flatnonzero(good)
        try:
            Di[selg] = np.linalg.inv(Dm[selg])
        except np.linalg.LinAlgError:
            for s_ in selg:
                try:
                    Di[s_] = np.linalg.inv(Dm[s_])
                except np.linalg.LinAlgError:
                    good[s_] = False
    return Di, good


def _per_trait_effects(spec, beta, Di, good, idx, p, T, names):
    """beta_<f>[<trait>] and se_<f>[<trait>] of a single-family test with T
    traits (no condition, no trait subset, no modifier)."""
    tested, cond = _parse_test_full(spec)
    if T == 1 or cond or len(tested) != 1 or tested[0][1] is not None \
            or tested[0][2] is not None:
        return {}
    f = tested[0][0]
    out = {}
    for kk in range(T):
        bv, sv = np.full(p, np.nan), np.full(p, np.nan)
        bv[idx[good]] = beta[good, kk]
        sv[idx[good]] = np.sqrt(np.maximum(Di[good, kk, kk], 0.0))
        out["beta_%s[%s]" % (f, names[kk])] = bv
        out["se_%s[%s]" % (f, names[kk])] = sv
    return out


def lambda_gc(chi2, df=1):
    """Genomic control factor: median(chi2) / theoretical median."""
    from scipy.stats import chi2 as _chi2
    v = np.asarray(chi2, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v) / _chi2.ppf(0.5, df))
