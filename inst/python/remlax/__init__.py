"""remlax: a generic REML solver, independent of the IGE model.

The package only knows incidence matrices and covariance structures. It holds
no notion of wheat, alfalfa, neighbourhood or trait. Everything specific to a
model is built on the R side and serialised.
"""
from . import _x64  # noqa: F401  (turns on float64 BEFORE anything else)
from .structures import STRUCTURES, n_params, build_sigma, chol_sigma
from .bundle import Bundle
from .model import assemble_V, neg2_reml, make_objective
from .device import pick_device, device_report
from .fit import fit_reml

__all__ = ["STRUCTURES", "n_params", "build_sigma", "chol_sigma", "Bundle",
           "assemble_V", "neg2_reml", "make_objective", "pick_device",
           "device_report", "fit_reml"]
