"""Lecture d'un dispositif serialise par R (R/remlax.R).

FORMAT (repertoire) : manifest.json + <nom>.bin (binaire brut) / <nom>.txt.
C'est la convention deja utilisee par export_06f_design.R, reprise telle quelle :
binaire brut plutot que .npz pour n'exiger AUCUNE dependance Python cote R, et
column-major parce que c'est l'ordre natif de R.

POURQUOI DES FICHIERS ET PAS reticulate : R et JAX vivent dans deux conteneurs
distincts sur ce cluster. Le fichier est aussi ce qui rend la parite verifiable,
puisque les deux backends lisent alors STRICTEMENT la meme entree.
"""
import json
import os
import numpy as np


class Bundle:
    """Dispositif : y, X, termes aleatoires, structure residuelle."""

    def __init__(self, path):
        self.path = path
        mf = os.path.join(path, "manifest.json")
        if not os.path.exists(mf):
            raise FileNotFoundError("manifest.json absent de %s" % path)
        with open(mf) as f:
            man = json.load(f)
        self.man = {d["name"]: d for d in man} if isinstance(man, list) else man["arrays"]
        self.meta = {} if isinstance(man, list) else man.get("meta", {})

    def has(self, name):
        return name in self.man

    def get(self, name):
        m = self.man[name]
        if m["dtype"] == "str":
            with open(os.path.join(self.path, name + ".txt")) as f:
                return [l.rstrip("\n") for l in f]
        dt = np.float64 if m["dtype"] == "f8" else np.int32
        a = np.fromfile(os.path.join(self.path, name + ".bin"), dtype=dt)
        shp = tuple(int(v) for v in m["shape"])
        return a.reshape(shp, order="F") if len(shp) > 1 else a

    # ---------------------------------------------------------------- accesseurs
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
        """[(nom du terme, indices de colonnes)] ou None si non fourni."""
        if not self.has("X_assign"):
            return None
        asg = self.get("X_assign").astype(int)
        noms = self.get("X_termes")
        out, vus = [], sorted(set(asg.tolist()))
        for k, a in enumerate(vus):
            cols = [j for j in range(len(asg)) if asg[j] == a]
            out.append((noms[k] if k < len(noms) else "terme %d" % a, cols))
        return out

    def terms(self):
        """Liste des termes aleatoires, dans l'ordre declare cote R.

        Chaque terme : name, struct, t (nb de colonnes de Sigma), rank, q (nb de
        niveaux), Z (n x t*q, COO), L_K (facteur de Cholesky de la parente, ou
        None pour l'identite).
        """
        out = []
        for nm in self.get("term_names"):
            d = dict(
                name=nm,
                struct=self.get("term_%s_struct" % nm)[0],
                t=int(self.get("term_%s_t" % nm)[0]),
                rank=int(self.get("term_%s_rank" % nm)[0]),
                q=int(self.get("term_%s_q" % nm)[0]),
                zi=self.get("term_%s_zi" % nm).astype(np.int64),   # ligne (0-based)
                zj=self.get("term_%s_zj" % nm).astype(np.int64),   # colonne (0-based)
                zx=self.get("term_%s_zx" % nm).astype(np.float64), # valeur
            )
            d["LK"] = (self.get("term_%s_LK" % nm).astype(np.float64)
                       if self.has("term_%s_LK" % nm) else None)
            # Structure ENTRE NIVEAUX : "id" (defaut), "fixed" (K fournie),
            # "ar1", "ar1ar1". Les anciens paquets n'ont pas la cle : on deduit
            # alors "fixed" si une K est presente, "id" sinon.
            d["lvl"] = (self.get("term_%s_lvl" % nm)[0] if self.has("term_%s_lvl" % nm)
                        else ("fixed" if d["LK"] is not None else "id"))
            d["dims"] = (tuple(int(v) for v in self.get("term_%s_dims" % nm))
                         if self.has("term_%s_dims" % nm) else None)
            d["lvl_order"] = (int(self.get("term_%s_lvlorder" % nm)[0])
                              if self.has("term_%s_lvlorder" % nm) else 0)
            # Coordonnees des niveaux, pour exp()/gau() : q x 1 (1D) ou q x 2 (2D).
            d["coord"] = (self.get("term_%s_coord" % nm).astype(np.float64)
                          if self.has("term_%s_coord" % nm) else None)
            d["lvl_opts"] = self._opts("term_%s" % nm)
            # PRODUIT SEPARABLE : liste (famille, dimension), serialisee en deux
            # vecteurs paralleles. C'est ce qui permet id (x) ar1 (x) ar1, soit un
            # champ spatial replique par bloc a correlations PARTAGEES — que
            # `ar1ar1`, limite a deux facteurs, ne peut pas exprimer.
            if self.has("term_%s_lvlpartk" % nm):
                pk = list(self.get("term_%s_lvlpartk" % nm))
                pq = [int(v) for v in self.get("term_%s_lvlpartq" % nm)]
                d["lvl_parts"] = list(zip(pk, pq))
            d["lvl_expr"] = (self.get("term_%s_lvlexpr" % nm)[0]
                             if self.has("term_%s_lvlexpr" % nm) else None)
            out.append(d)
        return out

    def _opts(self, pre):
        """Reglages nommes d'une structure de niveaux (mtrn, own, ...).

        Serialises en deux tableaux paralleles, cles et valeurs : c'est le seul
        format qui traverse la frontiere R/Python sans dependance JSON cote R,
        et il reste extensible sans toucher au lecteur.
        """
        if not self.has(pre + "_lvloptk"):
            return None
        k = self.get(pre + "_lvloptk")
        v = self.get(pre + "_lvloptv")
        return {str(a): float(b) for a, b in zip(k, v)}

    def _struct_res(self, pre):
        """Champs communs a une section residuelle (ou a la residuelle entiere)."""
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
        """Structure residuelle : struct, t, rank, index de trait, index d'unite.

        Avec `res_nsec` > 1 la residuelle est une SOMME DIRECTE de sections
        (le dsum d'asreml) : chacune porte sa propre structure, ses propres
        parametres et la liste des lignes qu'elle couvre.
        """
        r = self._struct_res("res")
        r["trait"] = (self.get("res_trait").astype(np.int64)
                      if self.has("res_trait") else np.zeros(self.n, np.int64))
        r["unit"] = (self.get("res_unit").astype(np.int64)
                     if self.has("res_unit") else np.arange(self.n, dtype=np.int64))
        r["n_unit"] = int(r["unit"].max()) + 1
        n_sec = int(self.get("res_nsec")[0]) if self.has("res_nsec") else 0
        if n_sec > 1:
            secs, noms = [], (self.get("res_secnames")
                              if self.has("res_secnames") else None)
            for k in range(n_sec):
                pre = "res_s%d" % k
                s = self._struct_res(pre)
                s["rows"] = self.get(pre + "_rows").astype(np.int64)
                s["trait"] = self.get(pre + "_trait").astype(np.int64)
                s["unit"] = self.get(pre + "_unit").astype(np.int64)
                s["n_unit"] = int(s["unit"].max()) + 1
                s["name"] = noms[k] if noms is not None and k < len(noms) else "section%d" % (k + 1)
                secs.append(s)
            vues = np.concatenate([s["rows"] for s in secs])
            if len(np.unique(vues)) != len(vues) or len(vues) != self.n:
                raise ValueError(
                    "dsum : les sections ne partitionnent pas les %d observations "
                    "(%d lignes citees, %d distinctes). Une ligne oubliee sortirait "
                    "de V avec une variance nulle, une ligne doublee la compterait "
                    "deux fois." % (self.n, len(vues), len(np.unique(vues))))
            r["sections"] = secs
        return r
