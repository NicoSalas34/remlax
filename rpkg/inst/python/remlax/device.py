"""Choix du peripherique : GPU si disponible, CPU sinon, sans jamais echouer.

Le meme code tourne des deux cotes ; ce que change le peripherique, c'est la
vitesse et l'ordre des reductions (donc les derniers bits), pas le modele.
"""
try:
    from . import _x64  # noqa: F401
except ImportError:
    import _x64          # noqa: F401
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
