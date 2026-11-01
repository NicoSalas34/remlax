"""Co-GWAS at fixed V: a focal locus a crossed with a neighbour locus b.

For a focal plant i, the covariate of the pair (a, b) is the product of its
own dose at locus a and the dose of its neighbours at locus b:

    s_dir,a = Z_g m_a        focal dose (Z_g: plants x focal genotypes)
    s_ind,b = Z_n m'_b       neighbour dose (Z_n: plants x neighbour genotypes,
                             the neighbour weights)
    u_ab    = s_dir,a * s_ind,b

The neighbour markers m' may come from another species (wheat focal,
alfalfa neighbours) or from the same one. With the same locus and the same
species, u_aa is the `sim` covariate of scan.py (Sato et al. 2021).

Tests per pair, at fixed V (the null model), as in scan.py:
    pair             u alone
    pair|dir         u given s_dir,a
    pair|ind         u given s_ind,b
    pair|dir+ind     u given both
Each is the Wald (score) test of the last column of [conditioning..., u].

Cost. u_ab' P u_ab costs n^2 per pair when written in dimension n. The
doses only depend on genotypes, so with R = Z_g (row-wise Kronecker) Z_n,
an n x (q_g q_f) matrix, u_ab = R (m_a kron m_b) and every term reduces to
genotype space, computed once:
    G  = R' P y                (q_g, q_f)      -> u'Py   = m_a' G m'_b
    Hd = R' P Z_g              (q_g, q_f, q_g) -> u'P s_dir
    Hn = R' P Z_n              (q_g, q_f, q_f) -> u'P s_ind
    T  = R' P R                (q_g, q_f, q_g, q_f) -> u'Pu = m'_b' G_a m'_b,
                               G_a[f, e] = sum_gh m_ga m_ha T[g, f, h, e]
Per pair the cost becomes q_f^2 instead of n^2, with no approximation. The
tensors are kept as matrices on Khatri-Rao indices, so that a block of
pairs is a few large matrix products (_block_stats).
T holds (q_g q_f)^2 values: 2.9 GB for 181 lines x 106 families.

The p_a x p_b statistics are not all kept: the call returns the `top` pairs
of each test, the smallest p-value per focal and per neighbour locus, and a
histogram of chi2 for the genomic control lambda.
"""
import numpy as np

TESTS = ("pair", "pair|dir", "pair|ind", "pair|dir+ind")


class PairProjection:
    """Genotype-space constants of the co-GWAS for one null model."""

    def __init__(self, fit, X, Zg, Zn, xp=None):
        from .scan import _fit_arrays, _dense
        xp = np if xp is None else xp
        self.xp = xp
        Vi, vbeta, Py = _fit_arrays(fit)
        X = _dense(X)
        Zg, Zn = _dense(Zg), _dense(Zn)
        n, qg = Zg.shape
        qf = Zn.shape[1]
        if Zn.shape[0] != n or len(Py) != n:
            raise ValueError("cogwas: Z_g, Z_n and the fit must have the same rows.")
        self.n, self.qg, self.qf = n, qg, qf
        put = (lambda a: a) if xp is np else xp.asarray
        Vi_d, X_d, vb_d = put(Vi), put(X), put(vbeta)
        ViX = Vi_d @ X_d

        def P(A):
            VA = Vi_d @ A
            return VA - ViX @ (vb_d @ (X_d.T @ VA))
        Zg_d, Zn_d, Py_d = put(Zg), put(Zn), put(Py)
        PZg, PZn = P(Zg_d), P(Zn_d)
        self.c_d, self.c_n = Zg_d.T @ Py_d, Zn_d.T @ Py_d
        self.C_dd, self.C_nn, self.C_dn = Zg_d.T @ PZg, Zn_d.T @ PZn, Zg_d.T @ PZn
        R = (Zg_d[:, :, None] * Zn_d[:, None, :]).reshape(n, qg * qf)
        self.G = (R.T @ Py_d).reshape(qg, qf)
        # Stored as matrices on Khatri-Rao rows (g, h) and columns (f, e), so
        # that every per-block step is one matrix product (_block_stats).
        self.Hd2 = (R.T @ PZg).reshape(qg, qf, qg).transpose(0, 2, 1).reshape(qg * qg, qf)
        self.Hn2 = (R.T @ PZn).reshape(qg, qf * qf)
        self.T2 = (R.T @ P(R)).reshape(qg, qf, qg, qf).transpose(0, 2, 1, 3).reshape(qg * qg, qf * qf)


def _model_columns(test):
    if test not in TESTS:
        raise ValueError("cogwas: unknown test '%s' (expected %s)." % (test, ", ".join(TESTS)))
    cond = test.split("|")[1].split("+") if "|" in test else []
    return cond + ["pair"]


def _focal_stats(proj, Ma):
    """Terms of a block of focal loci Ma (q_g, A), shared by every neighbour block.

    With the Khatri-Rao columns KR_a = m_a kron m_a (q_g^2 x A):
        Ga = KR_a' T2  (A, q_f^2)      T2[(g,h), (f,e)] = T[g, f, h, e]
        Va = KR_a' Hd2 (A, q_f)        Hd2[(g,h), f]    = Hd[g, f, h]"""
    xp = proj.xp
    KRa = (Ma[:, None, :] * Ma[None, :, :]).reshape(proj.qg * proj.qg, -1)
    return dict(Ma=Ma, n_d=(proj.c_d @ Ma)[:, None],
                d_dd=xp.sum(Ma * (proj.C_dd @ Ma), axis=0)[:, None],
                Ga=KRa.T @ proj.T2, Va=KRa.T @ proj.Hd2,
                MaG=Ma.T @ proj.G, MaC=Ma.T @ proj.C_dn)


def _block_stats(proj, fa, Mb):
    """num and den entries of the focal block fa (from _focal_stats) crossed
    with the neighbour block Mb (q_f, B), on xp. With KR_b = m'_b kron m'_b:
        u'P u = Ga KR_b,   u'P s_dir = Va Mb,   u'P s_ind = Ma' (Hn2 KR_b)."""
    xp = proj.xp
    KRb = (Mb[:, None, :] * Mb[None, :, :]).reshape(proj.qf * proj.qf, -1)
    return dict(n_d=fa["n_d"], d_dd=fa["d_dd"],
                n_n=(proj.c_n @ Mb)[None, :],
                d_nn=xp.sum(Mb * (proj.C_nn @ Mb), axis=0)[None, :],
                n_u=fa["MaG"] @ Mb, d_dn=fa["MaC"] @ Mb,
                d_ud=fa["Va"] @ Mb, d_un=fa["Ma"].T @ (proj.Hn2 @ KRb),
                d_uu=fa["Ga"] @ KRb)


def _schur(s, cond, xp):
    """chi2, beta and se of the pair column given the conditioning columns,
    for every pair of the block, on the device (Frisch-Waugh: the last
    coefficient of the GLS is the one of u after projecting out the others).
        r_n = n_u - c' D11^-1 n_1      r_d = d_uu - c' D11^-1 c
        beta = r_n / r_d, var = 1 / r_d, chi2 = r_n^2 / r_d"""
    n_u, d_uu = s["n_u"], s["d_uu"]
    if not cond:
        r_n, r_d = n_u, d_uu
    elif cond == ["dir"]:
        r_n = n_u - s["d_ud"] * s["n_d"] / s["d_dd"]
        r_d = d_uu - s["d_ud"] ** 2 / s["d_dd"]
    elif cond == ["ind"]:
        r_n = n_u - s["d_un"] * s["n_n"] / s["d_nn"]
        r_d = d_uu - s["d_un"] ** 2 / s["d_nn"]
    else:
        dd, nn, dn = s["d_dd"], s["d_nn"], s["d_dn"]
        ud, un = s["d_ud"], s["d_un"]
        det = dd * nn - dn ** 2
        r_n = n_u - (ud * (nn * s["n_d"] - dn * s["n_n"]) + un * (dd * s["n_n"] - dn * s["n_d"])) / det
        r_d = d_uu - (ud ** 2 * nn - 2.0 * ud * un * dn + un ** 2 * dd) / det
    beta = r_n / r_d
    return r_n * beta, beta, 1.0 / xp.sqrt(r_d)


class _Top:
    """The `top` largest chi2 seen so far, with their (a, b), beta and se."""

    def __init__(self, top):
        self.top = int(top)
        self.chi2 = np.empty(0); self.a = np.empty(0, int); self.b = np.empty(0, int)
        self.beta = np.empty(0); self.se = np.empty(0)

    def add(self, chi2, a, b, beta, se):
        self.chi2 = np.concatenate([self.chi2, chi2])
        self.a = np.concatenate([self.a, a]); self.b = np.concatenate([self.b, b])
        self.beta = np.concatenate([self.beta, beta]); self.se = np.concatenate([self.se, se])
        keep = np.argsort(-self.chi2, kind="stable")[:self.top]
        for name in ("chi2", "a", "b", "beta", "se"):
            setattr(self, name, getattr(self, name)[keep])


def _block_top(chi2, beta, se, top, xp):
    """The `top` largest finite chi2 of a block, moved to the host with their
    flat indices, beta and se."""
    flat = xp.where(xp.isfinite(chi2), chi2, -xp.inf).ravel()
    k = min(int(top), flat.shape[0])
    if xp is np:
        idx = np.argpartition(-flat, k - 1)[:k]
    else:
        import jax
        idx = jax.lax.top_k(flat, k)[1]
    return (np.asarray(flat[idx]), np.asarray(idx), np.asarray(beta.ravel()[idx]),
            np.asarray(se.ravel()[idx]))


# Histogram of chi2 for the genomic control lambda: bins of 1e-4 up to 2 (the
# median of chi2_1 is 0.455, so lambda is resolved to 2e-4), coarser above.
BINS = np.concatenate([np.linspace(0, 2, 20001), np.linspace(2.01, 10, 800),
                       np.linspace(10.1, 200, 1900), [np.inf]])


def cogwas(proj, Ma, Mb, tests=("pair|dir+ind",), top=1000, block_a=1024, block_b=8192,
           verbose=False):
    """Every pair (focal locus a, neighbour locus b) at fixed V.

    Ma : (q_g, p_a) focal doses (centred, +/-1 or centred dosages).
    Mb : (q_f, p_b) neighbour doses, rows in the order of the columns of Z_n.
    Returns {test: {"chi2", "p", "a", "b", "beta", "se"} of the top pairs,
             "min_p_a", "min_p_b", "lambda"}, plus "n_pairs"."""
    from scipy.stats import chi2 as _chi2
    xp = proj.xp
    put = (lambda a: a) if xp is np else xp.asarray
    Ma = np.asarray(Ma, dtype=np.float64); Mb = np.asarray(Mb, dtype=np.float64)
    pa, pb = Ma.shape[1], Mb.shape[1]
    cols = {t: _model_columns(t) for t in tests}
    tops = {t: _Top(top) for t in tests}
    max_a = {t: np.zeros(pa) for t in tests}
    max_b = {t: np.zeros(pb) for t in tests}
    hist = {t: np.zeros(len(BINS) - 1) for t in tests}
    bins = put(BINS)
    for a0 in range(0, pa, int(block_a)):
        fa = _focal_stats(proj, put(Ma[:, a0:a0 + int(block_a)]))
        for b0 in range(0, pb, int(block_b)):
            s = _block_stats(proj, fa, put(Mb[:, b0:b0 + int(block_b)]))
            nA, nB = s["n_u"].shape
            for t in tests:
                chi2, beta, se = _schur(s, cols[t][:-1], xp)
                v, idx, bt, sd = _block_top(chi2, beta, se, top, xp)
                ok = np.isfinite(v)
                tops[t].add(v[ok], idx[ok] // nB + a0, idx[ok] % nB + b0, bt[ok], sd[ok])
                c = xp.where(xp.isfinite(chi2), chi2, 0.0)
                max_a[t][a0:a0 + nA] = np.maximum(max_a[t][a0:a0 + nA], np.asarray(c.max(axis=1)))
                max_b[t][b0:b0 + nB] = np.maximum(max_b[t][b0:b0 + nB], np.asarray(c.max(axis=0)))
                h = xp.histogram(xp.where(xp.isfinite(chi2), chi2, -1.0).ravel(), bins=bins)[0]
                hist[t] += np.asarray(h)
        if verbose:
            print("  [cogwas] %d / %d focal loci" % (min(a0 + int(block_a), pa), pa), flush=True)
    out = {"n_pairs": pa * pb}
    for t in tests:
        tp = tops[t]
        cum = np.cumsum(hist[t])
        med = BINS[np.searchsorted(cum, cum[-1] / 2.0)] if cum[-1] else np.nan
        out[t] = dict(chi2=tp.chi2, p=_chi2.sf(tp.chi2, 1), a=tp.a, b=tp.b, beta=tp.beta, se=tp.se,
                      min_p_a=_chi2.sf(max_a[t], 1), min_p_b=_chi2.sf(max_b[t], 1),
                      **{"lambda": med / _chi2.ppf(0.5, 1)})
    return out
