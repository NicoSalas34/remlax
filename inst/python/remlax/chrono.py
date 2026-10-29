"""Chronometrage par etapes d'un ajustement.

Chaque etape pose une marque ; le resultat rend, dans l'ordre, la duree ecoulee
depuis la marque precedente. L'interface R transmet l'instant ou elle lance le
solveur (variable RX_T_LANCEMENT, secondes depuis l'epoque) : la premiere etape
mesure alors le demarrage de l'interpreteur, invisible depuis Python seul.

Le module ne depend de rien et s'importe en premier dans cli.py, avant JAX.
"""
import os
import time

_marques = []


def raz(t0=None):
    """Repart de zero ; t0 (secondes depuis l'epoque) date l'origine."""
    del _marques[:]
    if t0 is None:
        try:
            t0 = float(os.environ.get("RX_T_LANCEMENT", ""))
        except ValueError:
            t0 = None
    if t0 is not None:
        _marques.append(("origine", t0))


def marque(nom):
    """Pose une marque ; rend l'instant."""
    t = time.time()
    _marques.append((nom, t))
    return t


def poser(nom, t):
    """Marque datee a un instant deja releve (t, secondes depuis l'epoque)."""
    _marques.append((nom, float(t)))


def etapes():
    """[(nom, secondes depuis la marque precedente)], marques consecutives de
    meme nom cumulees."""
    out = []
    for (n0, t0), (n1, t1) in zip(_marques, _marques[1:]):
        d = t1 - t0
        if out and out[-1][0] == n1:
            out[-1] = (n1, out[-1][1] + d)
        else:
            out.append((n1, d))
    return out


def dict_etapes():
    out = {}
    for n, d in etapes():
        out[n] = round(out.get(n, 0.0) + float(d), 6)
    return out


raz()
