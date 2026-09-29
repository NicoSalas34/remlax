"""Les expressions vpredict sont verifiees avant l'ajustement."""
import pytest
from remlax.inference import check_vpredict

TERMES = [dict(name="gid", t=1, struct="iid", rank=0, q=10, LK=None)]
RES = dict(struct="iid", t=1, rank=0)


def _noms():
    from remlax.inference import component_names
    return component_names(TERMES, RES)


def test_expression_valide_acceptee():
    assert len(_noms()) == 2
    check_vpredict([("h2", "V1/(V1+V2)"), ("sd", "sqrt(V1)")], TERMES, RES)


def test_composante_absente_nommee_avec_la_liste():
    with pytest.raises(ValueError) as e:
        check_vpredict([("h2", "V1/(V1+V2+V3)")], TERMES, RES)
    msg = str(e.value)
    assert "V3" in msg and "V1 = gid" in msg and "2 composante" in msg


def test_fonction_non_admise_refusee():
    with pytest.raises(ValueError, match="open"):
        check_vpredict([("x", "open(V1)")], TERMES, RES)


def test_syntaxe_illisible():
    with pytest.raises(ValueError, match="illisible"):
        check_vpredict([("h2", "V1/(V1+")], TERMES, RES)
