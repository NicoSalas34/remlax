"""Device choice, compilation cache and BLAS thread control.

pick_device chooses the GPU when one is available and the CPU otherwise, and
never fails. The same code runs on both. The device changes the speed and the
order of reductions, and therefore the last bits, but leaves the model
unchanged.

The module also holds the start-up helpers used by cli.py
(enable_compilation_cache, without_gpu_plugins) and the NumpyBlasOneThread context
used by fit.py.
"""
from . import _x64  # noqa: F401
import jax


def pick_device(prefer="auto"):
    """prefer: 'auto' | 'gpu' | 'cpu'. Returns (device, platform_name).

    jax.devices() without an argument lists only the devices of the default
    platform. As soon as a GPU is visible it lists no CPU. Filtering its output
    on platform == "cpu" then gave an empty list, and `--backend=cpu` silently
    fell back to the GPU. The code therefore asks for jax.devices("cpu")
    explicitly, which always exists.
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
    """All devices, CPU included (see the note in pick_device)."""
    out, seen = [], set()
    for plat in ("cpu", "gpu", "cuda", "rocm", "tpu"):
        try:
            for d in jax.devices(plat):
                if str(d) in seen:
                    continue
                seen.add(str(d))
                out.append({"platform": d.platform, "kind": d.device_kind})
        except RuntimeError:
            pass
    return out


def enable_compilation_cache():
    """Turn on the persistent cache of programs compiled by XLA.

    A fit recompiles its functions each time the solver is launched. With this
    cache, a second fit of the same shape (repeated benchmark, grid,
    permutation, restart) reads the program back from disk (dev/notes.md).

    Folder: $REMLAX_JAX_CACHE, otherwise <remlax cache>/jax. The cascade is the
    same as for the C kernel of selinv: $REMLAX_CACHE, ~/.cache/remlax,
    $TMPDIR/remlax_cache. REMLAX_JAX_CACHE=0 turns the cache off. Returns the
    folder used, or None.
    """
    import os
    choice = os.environ.get("REMLAX_JAX_CACHE")
    if choice is not None and choice.strip().lower() in ("", "0", "false", "non"):
        return None
    cands = [choice] if choice else [
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
            # Every program is kept. Small ones compile fast, but a fit
            # compiles dozens of them.
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        except Exception:
            return None
        return d
    return None


def without_gpu_plugins():
    """CPU-only computation: JAX does not load the CUDA plugin.

    JAX discovers and initialises its plugins (jax_plugins) before it looks at
    the requested platform. The CUDA plugin loads eleven NVIDIA libraries even
    under JAX_PLATFORMS=cpu, which takes most of the backend start-up time at
    each launch of the solver (dev/notes.md). Marking the plugins as already
    registered avoids this loading. The flag is internal to JAX. If it
    disappears, nothing changes except the time lost. Call this before any JAX
    operation. It has no effect afterwards.
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


class NumpyBlasOneThread:
    """Context: the OpenBLAS of NumPy runs on one thread, SciPy's is unchanged.

    The NumPy and SciPy wheels each ship their own OpenBLAS, each with its own
    pool of busy-waiting threads. The NumPy products and the SciPy LAPACK
    (dpotrf, dpotri of the MME engine) then compete for twice as many threads
    as there are cores. The NumPy products of the engine are thin, and the
    heavy computation goes through LAPACK, so one NumPy thread costs little
    and removes the contention (dev/notes.md).

    No effect if both packages share the same library, or if the symbol cannot
    be found. REMLAX_BLAS_NUMPY=keep turns it off (the former value garder is
    still accepted).
    """

    _lib = None
    _searched = False

    @classmethod
    def _find(cls):
        if cls._searched:
            return cls._lib
        cls._searched = True
        import os
        if os.environ.get("REMLAX_BLAS_NUMPY", "").strip().lower() in ("keep", "garder"):
            return None
        try:
            import ctypes
            # Both imports load their OpenBLAS, so /proc/self/maps lists them.
            import numpy  # noqa: F401
            import scipy.linalg  # noqa: F401
            libs = []
            with open("/proc/self/maps") as f:
                for line in f:
                    p = line.split()[-1] if line.strip() else ""
                    if "openblas" in p and p.endswith(".so") or ".so." in p and "openblas" in p:
                        if p not in libs:
                            libs.append(p)
            np_lib = [p for p in libs if "numpy" in p]
            sp_lib = [p for p in libs if "scipy" in p]
            if not np_lib or not sp_lib or np_lib[0] == sp_lib[0]:
                return None
            L = ctypes.CDLL(np_lib[0])
            for set_name, get_name in (("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                                       ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
                                       ("openblas_set_num_threads", "openblas_get_num_threads")):
                if hasattr(L, set_name) and hasattr(L, get_name):
                    cls._lib = (getattr(L, set_name), getattr(L, get_name))
                    break
        except Exception:
            cls._lib = None
        return cls._lib

    def __enter__(self):
        self._before = None
        f = self._find()
        if f is not None:
            try:
                import ctypes
                self._before = int(f[1]())
                f[0](ctypes.c_int(1))
            except Exception:
                self._before = None
        return self

    def __exit__(self, *exc):
        f = self._find()
        if f is not None and self._before:
            try:
                import ctypes
                f[0](ctypes.c_int(self._before))
            except Exception:
                pass
        return False
