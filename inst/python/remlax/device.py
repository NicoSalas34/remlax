"""Choix du peripherique : GPU si disponible, CPU sinon, sans jamais echouer.

Le meme code tourne des deux cotes ; ce que change le peripherique, c'est la
vitesse et l'ordre des reductions (donc les derniers bits), pas le modele.
"""
from . import _x64  # noqa: F401
import jax


def pick_device(prefer="auto"):
    """prefer : 'auto' | 'gpu' | 'cpu'. Rend (device, nom_plateforme).

    ATTENTION, PIEGE CORRIGE LE 01/09/2026 : jax.devices() SANS ARGUMENT ne rend
    que les peripheriques de la plateforme par defaut. Des qu'un GPU est visible,
    elle ne liste plus aucun CPU — donc filtrer sa sortie sur platform == "cpu"
    rendait une liste vide, et `--backend=cpu` retombait silencieusement sur le
    GPU. Il faut demander explicitement jax.devices("cpu"), qui existe toujours.
    """
    gpus = []
    for plat in ("gpu", "cuda", "rocm", "tpu"):
        try:
            gpus += [d for d in jax.devices(plat)]
        except RuntimeError:
            pass
    try:
        cpus = list(jax.devices("cpu"))
    except RuntimeError:
        cpus = []
    if prefer == "cpu":
        if not cpus:
            raise RuntimeError("aucun peripherique CPU expose par JAX.")
        d = cpus[0]
    elif prefer == "gpu":
        if not gpus:
            raise RuntimeError("GPU demande mais JAX n'en voit aucun (jaxlib sans CUDA ?).")
        d = gpus[0]
    else:
        d = gpus[0] if gpus else (cpus[0] if cpus else jax.devices()[0])
    return d, d.platform


def device_report():
    """Tous les peripheriques, CPU compris (cf. le piege dans pick_device)."""
    out, vus = [], set()
    for plat in ("cpu", "gpu", "cuda", "rocm", "tpu"):
        try:
            for d in jax.devices(plat):
                if str(d) in vus:
                    continue
                vus.add(str(d))
                out.append({"platform": d.platform, "kind": d.device_kind})
        except RuntimeError:
            pass
    return out


def activer_cache_compilation():
    """Cache PERSISTANT des programmes compiles par XLA.

    Un ajustement recompile ses fonctions a chaque lancement du solveur. Avec ce
    cache, un second ajustement de meme forme (repetition d'un banc, grille,
    permutation, reprise) relit le programme sur disque. MESURE sur CPU
    (jax 0.11.1), une fonction a 30 Cholesky 300 x 300 et son gradient : premier
    appel 2,35 s sans cache, 0,42 s au second lancement.

    Dossier : $REMLAX_JAX_CACHE, sinon <cache remlax>/jax (meme cascade que le
    noyau C de selinv : $REMLAX_CACHE, ~/.cache/remlax, $TMPDIR/remlax_cache).
    REMLAX_JAX_CACHE=0 le desactive. Rend le dossier retenu, ou None.
    """
    import os
    choix = os.environ.get("REMLAX_JAX_CACHE")
    if choix is not None and choix.strip().lower() in ("", "0", "false", "non"):
        return None
    cands = [choix] if choix else [
        os.path.join(c, "jax") for c in (
            os.environ.get("REMLAX_CACHE"),
            os.path.join(os.path.expanduser("~"), ".cache", "remlax"),
            os.path.join(os.environ.get("TMPDIR", "/tmp"), "remlax_cache")) if c]
    for d in cands:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        if not os.access(d, os.W_OK):
            continue
        try:
            jax.config.update("jax_compilation_cache_dir", d)
            # tout programme est garde : les petits se compilent vite, mais un
            # ajustement en compile des dizaines
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        except Exception:
            return None
        return d
    return None


def sans_greffons_gpu():
    """Calcul sur CPU seul : JAX ne charge pas le greffon CUDA.

    JAX decouvre et initialise ses greffons (jax_plugins) avant de regarder la
    plateforme demandee, et celui de CUDA charge onze bibliotheques NVIDIA
    meme sous JAX_PLATFORMS=cpu : MESURE dans ige_reml.sif, 1,65 s sur les
    2,0 s de l'initialisation du backend, a chaque lancement du solveur.
    Marquer les greffons comme deja enregistres evite ce chargement.
    L'indicateur est interne a JAX : s'il disparait, rien ne change, sinon
    le temps perdu. A appeler avant toute operation JAX ; sans effet ensuite.
    """
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        jax.config.update("jax_platforms", "cpu")
    except Exception:
        pass
    try:
        from jax._src import xla_bridge as _xb
        if not _xb.backends_are_initialized() and hasattr(_xb, "_plugins_registered"):
            _xb._plugins_registered = True
    except Exception:
        pass


class BlasNumpyUnFil:
    """Contexte : l'OpenBLAS de NumPy a un fil, celui de SciPy inchange.

    Les roues de NumPy et de SciPy embarquent CHACUNE leur OpenBLAS, avec son
    propre groupe de fils en attente active. Les produits de NumPy et le LAPACK
    de SciPy (dpotrf, dpotri du moteur MME) se disputent alors deux fois plus
    de fils que de coeurs. MESURE sur un noeud genotoul (8 coeurs alloues, AMD
    EPYC 7513), evaluation MME complete : 1,46 s telle quelle contre 0,25 s
    NumPy a un fil a m = 1428 ; 0,77 contre 0,72 s a m = 3772. Les produits
    NumPy du moteur sont minces ; le gros calcul passe par LAPACK.

    Sans effet si les deux paquets partagent la meme bibliotheque, ou si le
    symbole est introuvable. REMLAX_BLAS_NUMPY=garder le desactive.
    """

    _lib = None
    _cherche = False

    @classmethod
    def _trouver(cls):
        if cls._cherche:
            return cls._lib
        cls._cherche = True
        import os
        if os.environ.get("REMLAX_BLAS_NUMPY", "").strip().lower() == "garder":
            return None
        try:
            import ctypes
            import numpy  # noqa: F401
            import scipy.linalg  # noqa: F401
            libs = []
            with open("/proc/self/maps") as f:
                for ligne in f:
                    p = ligne.split()[-1] if ligne.strip() else ""
                    if "openblas" in p and p.endswith(".so") or ".so." in p and "openblas" in p:
                        if p not in libs:
                            libs.append(p)
            np_lib = [p for p in libs if "numpy" in p]
            sp_lib = [p for p in libs if "scipy" in p]
            if not np_lib or not sp_lib or np_lib[0] == sp_lib[0]:
                return None
            L = ctypes.CDLL(np_lib[0])
            for nom_set, nom_get in (("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                                     ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
                                     ("openblas_set_num_threads", "openblas_get_num_threads")):
                if hasattr(L, nom_set) and hasattr(L, nom_get):
                    cls._lib = (getattr(L, nom_set), getattr(L, nom_get))
                    break
        except Exception:
            cls._lib = None
        return cls._lib

    def __enter__(self):
        self._avant = None
        f = self._trouver()
        if f is not None:
            try:
                import ctypes
                self._avant = int(f[1]())
                f[0](ctypes.c_int(1))
            except Exception:
                self._avant = None
        return self

    def __exit__(self, *exc):
        f = self._trouver()
        if f is not None and self._avant:
            try:
                import ctypes
                f[0](ctypes.c_int(self._avant))
            except Exception:
                pass
        return False
