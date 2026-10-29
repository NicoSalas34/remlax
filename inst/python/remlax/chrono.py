"""Timing of a fit by stages.

Each stage sets a mark. The result gives, in order, the time elapsed since the
previous mark. The R interface passes the instant at which it launches the
solver (variable RX_T_LANCEMENT, seconds since the epoch). The first stage then
measures the start-up of the interpreter, which Python alone cannot see.

The module has no dependency. cli.py imports it first, before JAX.
"""
import os
import time

_marks = []


def raz(t0=None):
    """Start again from zero. t0 (seconds since the epoch) dates the origin."""
    del _marks[:]
    if t0 is None:
        try:
            t0 = float(os.environ.get("RX_T_LANCEMENT", ""))
        except ValueError:
            t0 = None
    if t0 is not None:
        _marks.append(("origine", t0))


def marque(name):
    """Set a mark and return its instant."""
    t = time.time()
    _marks.append((name, t))
    return t


def poser(name, t):
    """Set a mark dated at an instant already read (t, seconds since the epoch)."""
    _marks.append((name, float(t)))


def etapes():
    """[(name, seconds since the previous mark)]. Consecutive marks with the
    same name are summed."""
    out = []
    for (n0, t0), (n1, t1) in zip(_marks, _marks[1:]):
        d = t1 - t0
        if out and out[-1][0] == n1:
            out[-1] = (n1, out[-1][1] + d)
        else:
            out.append((n1, d))
    return out


def dict_etapes():
    """The stages as a dict {name: seconds}, rounded to the microsecond."""
    out = {}
    for n, d in etapes():
        out[n] = round(out.get(n, 0.0) + float(d), 6)
    return out


raz()
