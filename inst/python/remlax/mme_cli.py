"""Point d'entree du moteur MME en Python (remlax/mme.py), lance par R.

    python mme_cli.py <paquet> --backend {cpu,gpu} [--quiet]

Lit le paquet ecrit par .rx_mme_python (R/remlax_mme.R), mene les iterations
et ecrit resultat.json dans le meme repertoire.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remlax.device import pick_device  # noqa: E402
from remlax.bundle import Bundle        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paquet")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    import jax
    dev, plat = pick_device(a.backend)
    with jax.default_device(dev):
        from remlax.mme import PaquetMME, ajuster
        if not a.quiet:
            print("[remlax mme] peripherique : %s" % dev, flush=True)
        P = PaquetMME(Bundle(a.paquet))
        r = ajuster(P, verbose=not a.quiet)
    out = dict(phi=[float(v) for v in r["phi"]], f=r["f"], score=[float(v) for v in r["score"]],
               n_iter=int(r["n_iter"]), converge=r["converge"], message=r["message"],
               a_la_borne=r["a_la_borne"])
    with open(os.path.join(a.paquet, "resultat.json"), "w") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
