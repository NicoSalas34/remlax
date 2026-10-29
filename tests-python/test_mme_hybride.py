"""Inversion hybride du moteur MME (mme_hybride.py) contre l'inversion dense.

Les petits dispositifs de test sont trop petits pour que le decoupage choisisse
de lui-meme des blocs (tout dense y est moins cher) : le separateur est donc
impose, pour que les blocs denses par lots soient bien exerces.
"""
import numpy as np
import pytest

import outils_dispositifs  # noqa: F401
from remlax import _x64  # noqa: F401
from remlax.mme import MME
from remlax.mme_hybride import FacteurHybride, HorsPerimetre, ensemble_dense
from remlax.fit import initial_theta
from test_econome import cas_ige, _rel
from test_blocs import cas_essais

CAS = {"ige": (cas_ige, ["gen_ble", "gen_luz"]),
       "essais": (cas_essais, ["geno", "geno_ar1", "rang_e0"])}


@pytest.mark.parametrize("nom", list(CAS))
def test_valeur_gradient_ai_identiques(nom):
    fab, sep = CAS[nom]
    terms, res, y, X = fab()
    Md = MME(terms, res, y, X, inversion="dense", verbose=False)
    Mh = MME(terms, res, y, X, inversion="hybride", verbose=False, separateur=sep)
    assert Mh.inversion == "hybride" and len(Mh.facteur.paquets) >= 1
    rng = np.random.default_rng(5)
    th0 = initial_theta(terms, res, y)
    for th in [th0, th0 + rng.normal(0, 0.3, len(th0))]:
        a, b = Md.evaluate(th), Mh.evaluate(th)
        assert abs(a["f"] - b["f"]) <= 1e-12 * abs(a["f"])
        assert _rel(b["g"], a["g"]) <= 1e-9
        assert _rel(b["AI"], a["AI"]) <= 1e-9


@pytest.mark.parametrize("nom", list(CAS))
def test_sorties_identiques(nom):
    fab, sep = CAS[nom]
    terms, res, y, X = fab()
    th = initial_theta(terms, res, y)
    sa = MME(terms, res, y, X, inversion="dense", verbose=False).outputs(th, pev=True)
    sb = MME(terms, res, y, X, inversion="hybride", verbose=False, separateur=sep).outputs(th, pev=True)
    assert _rel(sb["beta"], sa["beta"]) <= 1e-9
    for k in sa["blups"]:
        assert _rel(sb["blups"][k], sa["blups"][k]) <= 1e-9
        assert _rel(sb["pev"][k], sa["pev"][k]) <= 1e-9


def test_hors_perimetre_composante_trop_grande():
    terms, res, y, X = cas_ige()
    M = MME(terms, res, y, X, inversion="dense", verbose=False)
    with pytest.raises(HorsPerimetre):
        FacteurHybride(M.ci, M.cj, M.m, ensemble_dense(M.T, M.p_sortie, [0, 1]), taille_max=10)


def test_choix_automatique_sur_petit_modele():
    """Sans separateur impose, un petit modele reste en dense (moins cher)."""
    terms, res, y, X = cas_ige()
    M = MME(terms, res, y, X, inversion="hybride", verbose=False)
    assert M.inversion == "hybride" and M.facteur.paquets == []
