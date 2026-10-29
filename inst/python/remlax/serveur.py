"""Solveur persistant : un processus Python par session R.

Lancer le solveur en sous-processus coute, a chaque ajustement, le demarrage de
Python, l'import de JAX, numpy et scipy et l'initialisation des peripheriques :
de l'ordre de 1,5 s, plus que le calcul d'un petit modele. Ce module garde un
processus ouvert pour la session R : les imports sont faits une fois, et chaque
ajustement n'est plus qu'une requete sur un socket local.

    python remlax_run.py serveur <fichier_port> <jeton> [<pid du processus R>]

Ecrit "<port> <pid>" dans <fichier_port> (de facon atomique) des qu'il ecoute,
sur 127.0.0.1 seulement. Chaque connexion porte UNE requete, une ligne JSON :

    {"jeton": "...", "argv": [...]}        execute cli.main(argv)
    {"jeton": "...", "ordre": "stop"}      arrete le serveur

La sortie de cli.main (stdout et stderr) est renvoyee ligne par ligne pendant
l'ajustement, puis une ligne "\\x1eRXFIN <code>" (0x1E : separateur ASCII,
lisible par R, qui ne lit pas l'octet nul). Une requete sans le bon jeton
est fermee sans reponse.

SURVIE DU PROCESSUS R. Un processus R qui meurt sans finaliseur (fils d'un
fork, session tuee) laissait son serveur orphelin, chacun gardant JAX en
memoire : neuf orphelins ont sature 28 Go de RAM pendant une suite de
verification parallele (2026-10-22). Le serveur verifie donc toutes les 2 s,
entre deux requetes, que le processus R qui l'a lance existe encore, et
s'arrete sinon (POSIX seulement : sous Windows, os.kill(pid, 0) enverrait un
Ctrl-C ; le finaliseur R y reste la seule protection).
"""
import json
import os
import socket
import sys
import traceback



class _Flux:
    """Pseudo-fichier texte qui envoie chaque ligne complete sur le socket."""

    def __init__(self, conn):
        self.conn, self.tampon = conn, ""

    def write(self, s):
        self.tampon += s
        while "\n" in self.tampon:
            ligne, self.tampon = self.tampon.split("\n", 1)
            self._envoyer(ligne + "\n")
        return len(s)

    def flush(self):
        if self.tampon:
            self._envoyer(self.tampon + "\n")
            self.tampon = ""

    def _envoyer(self, txt):
        try:
            self.conn.sendall(txt.replace("\x1e", "").encode("utf-8", "replace"))
        except OSError:
            pass                                 # R a ferme : on finit l'ajustement quand meme

    def isatty(self):
        return False


def _vivant(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fichier_port, jeton = argv[0], argv[1]
    pid_r = int(argv[2]) if len(argv) > 2 else 0
    surveiller = pid_r > 0 and os.name == "posix"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    if surveiller:
        srv.settimeout(2.0)
    from remlax import cli                       # jax, numpy, scipy : une seule fois
    import jax
    tmp = fichier_port + ".tmp"
    with open(tmp, "w") as f:
        f.write("%d %d\n" % (srv.getsockname()[1], os.getpid()))
    os.replace(tmp, fichier_port)
    n_fits = 0
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if surveiller and not _vivant(pid_r):
                break                            # le processus R n'existe plus
            continue
        conn.settimeout(None)
        try:
            ligne = conn.makefile("rb").readline()
            req = json.loads(ligne.decode("utf-8") or "{}")
        except (OSError, ValueError):
            conn.close()
            continue
        if req.get("jeton") != jeton:
            conn.close()
            continue
        if req.get("ordre") == "stop":
            conn.close()
            break
        flux = _Flux(conn)
        ancien = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = flux
        try:
            # chrono remis a zero : origine = lancement cote R, si transmis
            cli.chrono.raz(req.get("t_lancement"))
            cli.chrono.marque("transmission")
            code = cli.main([str(a) for a in req.get("argv", [])])
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:                    # noqa: B902 - tout echec est rapporte a R
            traceback.print_exc()
            code = 1
        finally:
            flux.flush()
            sys.stdout, sys.stderr = ancien
            # Les fonctions compilees par structure (mme._SIGMA_CACHE) restent :
            # c'est le gain du serveur. Les caches JAX ne sont vides qu'au-dela
            # d'un nombre d'ajustements, pour borner la memoire.
            n_fits += 1
            if n_fits % 200 == 0:
                try:
                    from remlax import mme as _m
                    _m._SIGMA_CACHE.clear()
                    jax.clear_caches()
                except Exception:                # noqa: BLE001
                    pass
        try:
            conn.sendall(("\x1eRXFIN %d\n" % int(code or 0)).encode())
        except OSError:
            pass
        conn.close()
    srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
