"""remlax — solveur REML generique, independant du modele IGE.

Le paquet ne connait que des MATRICES D'INCIDENCE et des STRUCTURES DE
COVARIANCE. Aucune notion de blé, de luzerne, de voisinage ou de trait n'y
figure : ce qui est specifique au modele est construit cote R et serialise.
"""
from . import _x64  # noqa: F401  (active float64 AVANT tout le reste)
from .structures import STRUCTURES, n_params, build_sigma, chol_sigma
from .bundle import Bundle
from .model import assemble_V, neg2_reml, make_objective
from .device import pick_device, device_report
from .fit import fit_reml

__all__ = ["STRUCTURES", "n_params", "build_sigma", "chol_sigma", "Bundle",
           "assemble_V", "neg2_reml", "make_objective", "pick_device",
           "device_report", "fit_reml"]
