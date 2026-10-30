"""Reads a design serialised by R (R/bundle_write.R).

The bundle is a directory holding manifest.json and one file per array:
<name>.bin (raw binary) or <name>.txt (strings). Raw binary avoids any Python
dependency on the R side. Arrays are column-major, the native order of R.

R and Python exchange files because they can run in two separate containers.
Files also make parity checkable, since every backend then reads exactly the
same input.

Bundle.get reads one array. The other methods assemble the arrays into the
dicts used by the engine: terms() for the random terms, residual() for the
residual structure, lag() for the spatial lag, fixed_groups() for the fixed
terms.
"""
import json
import os
import numpy as np


class Bundle:
    """Design: y, X, random terms, residual structure."""

    def __init__(self, path):
        self.path = path
        mf = os.path.join(path, "manifest.json")
        if not os.path.exists(mf):
            raise FileNotFoundError("manifest.json missing from %s" % path)
        with open(mf) as f:
            man = json.load(f)
        self.man = {d["name"]: d for d in man} if isinstance(man, list) else man["arrays"]
        self.meta = {} if isinstance(man, list) else man.get("meta", {})

    def has(self, name):
        """True if the bundle holds the array `name`."""
        return name in self.man

    def get(self, name):
        """The array `name`: a list of strings, or a float64 or int32 array
        reshaped in column-major order."""
        m = self.man[name]
        if m["dtype"] == "str":
            with open(os.path.join(self.path, name + ".txt")) as f:
                return [l.rstrip("\n") for l in f]
        dt = np.float64 if m["dtype"] == "f8" else np.int32
        a = np.fromfile(os.path.join(self.path, name + ".bin"), dtype=dt)
        shp = tuple(int(v) for v in m["shape"])
        return a.reshape(shp, order="F") if len(shp) > 1 else a

    # ---------------------------------------------------------------- accessors
    @property
    def y(self):
        return self.get("y").astype(np.float64)

    @property
    def X(self):
        return np.atleast_2d(self.get("X")).astype(np.float64)

    @property
    def n(self):
        return int(self.y.shape[0])

    def fixed_groups(self):
        """[(term name, column indices)], or None if not supplied."""
        if not self.has("X_assign"):
            return None
        asg = self.get("X_assign").astype(int)
        names = self.get("X_terms")
        out, codes = [], sorted(set(asg.tolist()))
        for k, a in enumerate(codes):
            cols = [j for j in range(len(asg)) if asg[j] == a]
            out.append((names[k] if k < len(names) else "term %d" % a, cols))
        return out

    def terms(self):
        """List of the random terms, in the order declared on the R side.

        Each term holds name, struct, t (number of columns of Sigma), rank,
        q (number of levels), Z (n x t*q, COO in zi, zj, zx) and LK (Cholesky
        factor of the relationship matrix, or None for the identity).
        """
        out = []
        for nm in self.get("term_names"):
            d = dict(
                name=nm,
                struct=self.get("term_%s_struct" % nm)[0],
                t=int(self.get("term_%s_t" % nm)[0]),
                rank=int(self.get("term_%s_rank" % nm)[0]),
                q=int(self.get("term_%s_q" % nm)[0]),
                zi=self.get("term_%s_zi" % nm).astype(np.int64),   # row (0-based)
                zj=self.get("term_%s_zj" % nm).astype(np.int64),   # column (0-based)
                zx=self.get("term_%s_zx" % nm).astype(np.float64), # value
            )
            d["LK"] = (self.get("term_%s_LK" % nm).astype(np.float64)
                       if self.has("term_%s_LK" % nm) else None)
            # Sparse precision (pedigree). Only the MME engine uses it.
            if self.has("term_%s_Kinv_i" % nm):
                import scipy.sparse as _sp
                q_ = int(self.get("term_%s_q" % nm)[0])
                d["Kinv"] = _sp.csr_matrix((self.get("term_%s_Kinv_x" % nm).astype(np.float64),
                                            (self.get("term_%s_Kinv_i" % nm).astype(np.int64),
                                             self.get("term_%s_Kinv_j" % nm).astype(np.int64))),
                                           shape=(q_, q_))
                d["Kinv_logdet"] = float(self.get("term_%s_Kinv_logdet" % nm)[0])
            else:
                d["Kinv"], d["Kinv_logdet"] = None, None
            # Structure between levels: "id" (default), "fixed" (K supplied),
            # "ar1", "ar1ar1", ... Older bundles lack the key. The reader then
            # infers "fixed" if a K is present and "id" otherwise.
            d["lvl"] = (self.get("term_%s_lvl" % nm)[0] if self.has("term_%s_lvl" % nm)
                        else ("fixed" if d["LK"] is not None else "id"))
            d["dims"] = (tuple(int(v) for v in self.get("term_%s_dims" % nm))
                         if self.has("term_%s_dims" % nm) else None)
            d["lvl_order"] = (int(self.get("term_%s_lvlorder" % nm)[0])
                              if self.has("term_%s_lvlorder" % nm) else 0)
            # Coordinates of the levels, for exp()/gau(): q x 1 (1D) or q x 2 (2D).
            d["coord"] = (self.get("term_%s_coord" % nm).astype(np.float64)
                          if self.has("term_%s_coord" % nm) else None)
            d["lvl_opts"] = self._opts("term_%s" % nm)
            # Separable product: a list of (family, dimension), serialised as
            # two parallel vectors. It allows id (x) ar1 (x) ar1, a spatial
            # field replicated by block with shared correlations. `ar1ar1` is
            # limited to two factors and cannot express it.
            if self.has("term_%s_lvlpartk" % nm):
                pk = list(self.get("term_%s_lvlpartk" % nm))
                pq = [int(v) for v in self.get("term_%s_lvlpartq" % nm)]
                d["lvl_parts"] = list(zip(pk, pq))
            d["lvl_expr"] = (self.get("term_%s_lvlexpr" % nm)[0]
                             if self.has("term_%s_lvlexpr" % nm) else None)
            out.append(d)
        return out

    def _opts(self, pre):
        """Named settings of a level structure (mtrn, own, ...).

        They are serialised as two parallel arrays, keys and values. This
        format crosses the R/Python boundary without a JSON dependency on the
        R side, and it can grow without changes to the reader.
        """
        if not self.has(pre + "_lvloptk"):
            return None
        k = self.get(pre + "_lvloptk")
        v = self.get(pre + "_lvloptv")
        return {str(a): float(b) for a, b in zip(k, v)}

    def _residual_fields(self, pre):
        """Fields shared by a residual section and by the whole residual."""
        g = lambda s: pre + s                                     # noqa: E731
        r = dict(struct=self.get(g("_struct"))[0],
                 t=int(self.get(g("_t"))[0]),
                 rank=int(self.get(g("_rank"))[0]))
        r["lvl"] = self.get(g("_lvl"))[0] if self.has(g("_lvl")) else "id"
        r["lvl_order"] = int(self.get(g("_lvlorder"))[0]) if self.has(g("_lvlorder")) else 0
        r["coord"] = (self.get(g("_coord")).astype(np.float64)
                      if self.has(g("_coord")) else None)
        r["dims"] = (tuple(int(v) for v in self.get(g("_dims")))
                     if self.has(g("_dims")) else None)
        r["lvl_opts"] = self._opts(pre)
        r["lvl_expr"] = self.get(g("_lvlexpr"))[0] if self.has(g("_lvlexpr")) else None
        return r

    def residual(self):
        """Residual structure: struct, t, rank, trait index, unit index.

        With `res_nsec` > 1 the residual is a direct sum of sections (the dsum
        of asreml). Each section carries its own structure, its own parameters
        and the list of rows it covers.
        """
        r = self._residual_fields("res")
        r["trait"] = (self.get("res_trait").astype(np.int64)
                      if self.has("res_trait") else np.zeros(self.n, np.int64))
        r["unit"] = (self.get("res_unit").astype(np.int64)
                     if self.has("res_unit") else np.arange(self.n, dtype=np.int64))
        # n_unit is exported by R for a grid (cells without observation
        # included). Otherwise it is the largest code + 1.
        r["n_unit"] = (int(self.get("res_nunit")[0]) if self.has("res_nunit")
                       else int(r["unit"].max()) + 1)
        n_sec = int(self.get("res_nsec")[0]) if self.has("res_nsec") else 0
        if n_sec > 1:
            secs, names = [], (self.get("res_secnames")
                               if self.has("res_secnames") else None)
            for k in range(n_sec):
                pre = "res_s%d" % k
                s = self._residual_fields(pre)
                s["rows"] = self.get(pre + "_rows").astype(np.int64)
                s["trait"] = self.get(pre + "_trait").astype(np.int64)
                s["unit"] = self.get(pre + "_unit").astype(np.int64)
                s["n_unit"] = (int(self.get(pre + "_nunit")[0]) if self.has(pre + "_nunit")
                               else int(s["unit"].max()) + 1)
                s["name"] = names[k] if names is not None and k < len(names) else "section%d" % (k + 1)
                secs.append(s)
            cited = np.concatenate([s["rows"] for s in secs])
            if len(np.unique(cited)) != len(cited) or len(cited) != self.n:
                raise ValueError(
                    "dsum: the sections do not partition the %d observations "
                    "(%d rows cited, %d distinct). A missing row would leave "
                    "V with a zero variance. A duplicated row would count "
                    "twice." % (self.n, len(cited), len(np.unique(cited))))
            r["sections"] = secs
        return r

    def lag(self):
        """Spatial lag (rx_lag on the R side), or None.

        W travels as 0-based COO triplets, either one shared matrix or one per
        receiving trait. mask, value and init are q x q matrices (receiver in
        rows, emitter in columns), written column-major like everything else.
        """
        if not self.has("lag_q"):
            return None
        import scipy.sparse as sp
        q = int(self.get("lag_q")[0])
        N = int(self.get("lag_N")[0])
        nW = int(self.get("lag_nW")[0])
        Ws = []
        for k in range(nW):
            i = self.get("lag_W%d_i" % k).astype(np.int64)
            j = self.get("lag_W%d_j" % k).astype(np.int64)
            x = self.get("lag_W%d_x" % k).astype(np.float64)
            Ws.append(sp.csr_matrix((x, (i, j)), shape=(N, N)))
        mat = lambda nm, dt: np.asarray(self.get(nm), dtype=dt).reshape((q, q), order="F")  # noqa: E731
        return dict(q=q, N=N, W=Ws,
                    trait=self.get("lag_trait").astype(np.int64),
                    unit=self.get("lag_unit").astype(np.int64),
                    mask=mat("lag_mask", np.int64).astype(bool),
                    value=mat("lag_value", np.float64),
                    init=mat("lag_init", np.float64),
                    traits=list(self.get("lag_traits")),
                    criterion=self.get("lag_criterion")[0])
