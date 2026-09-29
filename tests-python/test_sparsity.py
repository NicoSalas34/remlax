"""Diagnostic de sparsite : motif des equations du modele mixte et remplissage
symbolique, verifies sur des motifs dont le remplissage est connu en forme
close (tridiagonale : aucun remplissage ; pleine : tout)."""
import numpy as np
import scipy.sparse as sp

import outils_dispositifs as od
from remlax.sparsity import mme_pattern, symbolic_cholesky_nnz, dense_cost, incidence_fill, sparsity_report


def test_symbolic_cholesky_sur_motifs_connus():
    q = 30
    T = sp.diags([1, 1, 1], [-1, 0, 1], shape=(q, q), dtype=bool, format="csr")
    r = symbolic_cholesky_nnz(T, permute=False)
    assert r["nnz_L"] == 2 * q - 1 and r["n"] == q
    assert r["colcount_max"] == 1
    F = sp.csr_matrix(np.ones((q, q), dtype=bool))
    r = symbolic_cholesky_nnz(F)
    assert r["nnz_L"] == q * (q + 1) // 2 and r["fill_ratio"] == 1.0
    # une fleche (premiere ligne et colonne pleines) : RCM la renverse, aucun
    # remplissage au-dela du motif ; en ordre naturel elle remplit tout
    A = sp.lil_matrix((q, q), dtype=bool); A.setdiag(True); A[0, :] = True; A[:, 0] = True
    assert symbolic_cholesky_nnz(A.tocsr(), permute=True)["nnz_L"] == 2 * q - 1
    assert symbolic_cholesky_nnz(A.tocsr(), permute=False)["nnz_L"] == q * (q + 1) // 2


def test_mme_pattern_dimensions_et_blocs():
    rng = np.random.default_rng(0)
    n, q = 60, 10
    tm = od.terme_facteur("g", np.tile(np.arange(q), n // q), q, lvl="ar1")
    res = od.residuelle_iid(n)
    C, blocs = mme_pattern([tm], res, n, p_fixed=2)
    assert C.shape == (2 + q, 2 + q)
    assert blocs[0]["name"] == "g" and blocs[0]["kind"] == "ar1" and blocs[0]["start"] == 2
    # Z'Z d'un facteur equilibre est diagonale ; K^-1 d'un ar1 tridiagonale ;
    # X dense touche tout : nnz = 4 (X'X) + 2*2*q (X'Z, Z'X) + tridiagonale
    assert C.nnz == 4 + 4 * q + (3 * q - 2)
    assert (C != C.T).nnz == 0
    inc = incidence_fill([tm], n)[0]
    assert inc["ztz_nnz"] == q and inc["ztz_density"] == 1.0 / q   # diagonale : 10 % a q = 10
    dc = dense_cost([tm], n)
    assert dc["cholesky_flops"] == n ** 3 / 3.0 and dc["total_flops"] > dc["cholesky_flops"]
    rep = sparsity_report([tm], res, n, p_fixed=2, verbose=False)
    assert rep["verdict"] in ("sparse", "dense", "comparable")
