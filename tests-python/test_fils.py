"""Bornage des fils aux coeurs alloues (fils.py)."""
import os
import subprocess
import sys

import pytest

ICI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python")
sys.path.insert(0, ICI)
from remlax.fils import choose_cores  # noqa: E402

CODE = ("import os, sys; sys.path.insert(0, %r); from remlax.fils import bound_threads; "
        "m = bound_threads(verbose=False); import numpy; "
        "print(len(os.sched_getaffinity(0)), os.environ.get('OPENBLAS_NUM_THREADS'), m is not None)" % ICI)


def lancer(env_extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("REMLAX_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True)
    n, blas, borne = out.stdout.split()
    return int(n), blas, borne == "True"


pytestmark = pytest.mark.skipif(not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2,
                                reason="affinite indisponible ou un seul coeur")


def test_sans_variable_rien_ne_change():
    n0 = len(os.sched_getaffinity(0))
    n, blas, borne = lancer({})
    assert n == n0 and blas == "None" and not borne


def test_slurm_borne_le_masque():
    n, blas, borne = lancer({"SLURM_CPUS_PER_TASK": "1"})
    assert n == 1 and blas == "1" and borne


def test_remlax_threads_prime_et_tous_desactive():
    assert lancer({"SLURM_CPUS_PER_TASK": "1", "REMLAX_THREADS": "2"})[0] == 2
    n0 = len(os.sched_getaffinity(0))
    assert lancer({"SLURM_CPUS_PER_TASK": "1", "REMLAX_THREADS": "tous"})[0] == n0


def test_masque_deja_assez_petit():
    n0 = len(os.sched_getaffinity(0))
    n, blas, borne = lancer({"OMP_NUM_THREADS": str(n0 + 4)})
    assert n == n0 and not borne


def test_choix_dans_un_noeud_numa_et_coeurs_libres():
    masque = set(range(8))
    libres = {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.2, 4: 1.0, 5: 1.0, 6: 0.95, 7: 0.05}
    noeuds = [{0, 1, 2, 3}, {4, 5, 6, 7}]
    assert choose_cores(masque, 3, libres, noeuds) == [4, 5, 6]
    # aucun noeud n'a 5 coeurs : les 5 plus libres de tout le masque
    assert choose_cores(masque, 5, libres, noeuds) == [1, 2, 4, 5, 6]
