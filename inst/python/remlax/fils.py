"""Bound the compute threads to the allocated cores.

JAX does not read OMP_NUM_THREADS. Its CPU thread pool takes the size of the
affinity mask of the process, as OpenBLAS does when OPENBLAS_NUM_THREADS is
unset. A user who sets OMP_NUM_THREADS = 4 on a 32-core workstation would
therefore get 32 threads. When the scheduler already confines the job to its
cores (SLURM with task/affinity), the mask already has the right size and
this module does nothing.

This module does not protect against an overloaded node. Unconfined jobs on
the same node can spill onto the allocated cores and make timings unstable.
Only a reserved node (--exclusive) then gives reliable times (dev/notes.md).

bound_threads() restricts the affinity of the process to the requested number
of cores, before NumPy and JAX are imported. The number comes from
REMLAX_THREADS, then SLURM_CPUS_PER_TASK, then OMP_NUM_THREADS. Without any of
these variables, nothing changes. REMLAX_THREADS=all turns the bounding off
(the former value tous is still accepted).

The cores kept are the most idle ones of the mask, measured over 0.1 s in
/proc/stat. They are taken from a single NUMA node when one has enough.

This module imports only the standard library, because it must run before any
heavy import.
"""
import os
import sys
import time

# Shared state. cli.py imports it and reads the "message" key.
_ETAT = {"fait": False, "message": None}


def _requested_count():
    """(number of cores, name of the variable that gave it), or (None, None)."""
    v = os.environ.get("REMLAX_THREADS", "").strip().lower()
    if v in ("all", "0", "tous"):
        return None, None
    for var in ("REMLAX_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "").strip()
        try:
            n = int(v)
        except ValueError:
            continue
        if n > 0:
            return n, var
    return None, None


def _cpu_times():
    """{cpu: (idle, total)} read from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            fields = line.split()
            v = [int(x) for x in fields[1:]]
            out[int(fields[0][3:])] = (v[3] + (v[4] if len(v) > 4 else 0), sum(v))
    return out


def _idle_fractions(mask, delay=0.1):
    """Idle fraction of each core of the mask over `delay` seconds."""
    try:
        a = _cpu_times()
        time.sleep(delay)
        b = _cpu_times()
    except (OSError, ValueError):
        return {c: 0.0 for c in mask}
    out = {}
    for c in mask:
        if c in a and c in b and b[c][1] > a[c][1]:
            out[c] = (b[c][0] - a[c][0]) / (b[c][1] - a[c][1])
        else:
            out[c] = 0.0
    return out


def _numa_nodes():
    """[set of cores] per NUMA node, empty if the information is missing."""
    root = "/sys/devices/system/node"
    out = []
    try:
        for name in sorted(os.listdir(root)):
            if not (name.startswith("node") and name[4:].isdigit()):
                continue
            with open(os.path.join(root, name, "cpulist")) as f:
                text = f.read().strip()
            cores = set()
            for piece in text.split(","):
                if "-" in piece:
                    first, last = piece.split("-")
                    cores.update(range(int(first), int(last) + 1))
                elif piece:
                    cores.add(int(piece))
            out.append(cores)
    except OSError:
        return []
    return out


def choose_cores(mask, n, idle, nodes):
    """The n most idle cores of the mask. They come from a single NUMA node if
    one offers n of them (the most idle in sum), otherwise from the whole mask."""
    by_idleness = lambda cs: sorted(cs, key=lambda c: (-idle.get(c, 0.0), c))
    candidates = []
    for node in nodes:
        cs = by_idleness(node & mask)
        if len(cs) >= n:
            candidates.append((sum(idle.get(c, 0.0) for c in cs[:n]), cs[:n]))
    if candidates:
        return sorted(max(candidates, key=lambda x: x[0])[1])
    return sorted(by_idleness(mask)[:n])


def bound_threads(verbose=True):
    """Restrict the affinity of the process to n cores if the mask offers more.
    Call it before NumPy and JAX are imported. Returns the message issued, or
    None."""
    if _ETAT["fait"]:
        return _ETAT["message"]
    _ETAT["fait"] = True
    n, source = _requested_count()
    if n is None or not hasattr(os, "sched_getaffinity"):
        return None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        mask = set(os.sched_getaffinity(0))
    except OSError:
        return None
    if len(mask) <= n:
        return None
    cores = choose_cores(mask, n, _idle_fractions(mask), _numa_nodes())
    try:
        os.sched_setaffinity(0, cores)
    except OSError as e:
        msg = "[remlax] bornage des fils impossible (%s) : %d coeurs visibles pour %d demandes" % (e, len(mask), n)
    else:
        msg = "[remlax] fils bornes a %d coeurs (%s) sur les %d visibles : %s" % (
            n, source, len(mask), ",".join(map(str, cores)))
    _ETAT["message"] = msg
    if verbose:
        print(msg, file=sys.stderr, flush=True)
    return msg
