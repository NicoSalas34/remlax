"""Bornage des fils de calcul aux coeurs alloues.

JAX ne lit pas OMP_NUM_THREADS : son groupe de fils CPU prend la taille du
masque d'affinite du processus, comme OpenBLAS quand OPENBLAS_NUM_THREADS
manque. Un utilisateur qui pose OMP_NUM_THREADS = 4 sur une station de 32
coeurs obtenait donc 32 fils. Quand l'ordonnanceur confine deja le job a ses
coeurs (SLURM avec task/affinity, verifie sur genotoul et meso le
2026-10-27), le masque est deja a la bonne taille et ce module ne fait rien.

ATTENTION, CE MODULE NE PROTEGE PAS D'UN NOEUD SURCHARGE. Sur meso, des jobs
voisins non confines debordent sur les coeurs alloues (charge 300 pour 192
coeurs) : le meme ajustement y a varie d'un facteur 1 a 40 entre deux
passages, avec 8 coeurs d'affinite et 34 a 43 fils (essai A/B du
2026-10-27, srun --cpu-bind=cores sans effet). Seul un noeud reserve
(--exclusive) donne des temps fiables la-bas.

borner_fils() restreint donc l'affinite du processus au nombre de coeurs
demande, avant l'import de NumPy et de JAX. Le nombre vient, dans l'ordre, de
REMLAX_THREADS, SLURM_CPUS_PER_TASK, OMP_NUM_THREADS. Sans aucune de ces
variables, rien ne change. REMLAX_THREADS=tous desactive le bornage.

Les coeurs retenus sont les plus libres du masque, mesures sur 0,1 s dans
/proc/stat, et pris dans un meme noeud NUMA quand il en a assez.

Ce module n'importe que la bibliotheque standard : il doit pouvoir tourner
avant tout import lourd.
"""
import os
import sys
import time

_ETAT = {"fait": False, "message": None}


def _nombre_voulu():
    v = os.environ.get("REMLAX_THREADS", "").strip().lower()
    if v in ("tous", "all", "0"):
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


def _temps_cpu():
    """{cpu: (inactif, total)} lu dans /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for ligne in f:
            if not ligne.startswith("cpu") or ligne.startswith("cpu "):
                continue
            ch = ligne.split()
            v = [int(x) for x in ch[1:]]
            out[int(ch[0][3:])] = (v[3] + (v[4] if len(v) > 4 else 0), sum(v))
    return out


def _libres(masque, delai=0.1):
    """Fraction inactive de chaque coeur du masque sur `delai` secondes."""
    try:
        a = _temps_cpu()
        time.sleep(delai)
        b = _temps_cpu()
    except (OSError, ValueError):
        return {c: 0.0 for c in masque}
    out = {}
    for c in masque:
        if c in a and c in b and b[c][1] > a[c][1]:
            out[c] = (b[c][0] - a[c][0]) / (b[c][1] - a[c][1])
        else:
            out[c] = 0.0
    return out


def _noeuds_numa():
    """[ensemble de coeurs] par noeud NUMA, vide si l'information manque."""
    racine = "/sys/devices/system/node"
    out = []
    try:
        for nom in sorted(os.listdir(racine)):
            if not (nom.startswith("node") and nom[4:].isdigit()):
                continue
            with open(os.path.join(racine, nom, "cpulist")) as f:
                txt = f.read().strip()
            coeurs = set()
            for morceau in txt.split(","):
                if "-" in morceau:
                    d, f_ = morceau.split("-")
                    coeurs.update(range(int(d), int(f_) + 1))
                elif morceau:
                    coeurs.add(int(morceau))
            out.append(coeurs)
    except OSError:
        return []
    return out


def choisir_coeurs(masque, n, libres, noeuds):
    """Les n coeurs les plus libres du masque, dans un seul noeud NUMA si l'un
    d'eux en offre n (le plus libre en somme), sinon dans tout le masque."""
    tri = lambda cs: sorted(cs, key=lambda c: (-libres.get(c, 0.0), c))
    candidats = []
    for nd in noeuds:
        cs = tri(nd & masque)
        if len(cs) >= n:
            candidats.append((sum(libres.get(c, 0.0) for c in cs[:n]), cs[:n]))
    if candidats:
        return sorted(max(candidats, key=lambda x: x[0])[1])
    return sorted(tri(masque)[:n])


def borner_fils(verbose=True):
    """Restreint l'affinite du processus a n coeurs si le masque en offre plus.
    A appeler avant l'import de NumPy et de JAX. Rend le message emis (ou None)."""
    if _ETAT["fait"]:
        return _ETAT["message"]
    _ETAT["fait"] = True
    n, source = _nombre_voulu()
    if n is None or not hasattr(os, "sched_getaffinity"):
        return None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        masque = set(os.sched_getaffinity(0))
    except OSError:
        return None
    if len(masque) <= n:
        return None
    coeurs = choisir_coeurs(masque, n, _libres(masque), _noeuds_numa())
    try:
        os.sched_setaffinity(0, coeurs)
    except OSError as e:
        msg = "[remlax] bornage des fils impossible (%s) : %d coeurs visibles pour %d demandes" % (e, len(masque), n)
    else:
        msg = "[remlax] fils bornes a %d coeurs (%s) sur les %d visibles : %s" % (
            n, source, len(masque), ",".join(map(str, coeurs)))
    _ETAT["message"] = msg
    if verbose:
        print(msg, file=sys.stderr, flush=True)
    return msg
