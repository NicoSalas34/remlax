"""Persistent solver: one Python process per R session.

Launching the solver as a subprocess pays, at each fit, the start-up of
Python, the import of JAX, numpy and scipy and the initialisation of the
devices. This costs more than the computation of a small model
(dev/notes.md). This module keeps one process open for the R session. The
imports are done once, and each fit becomes a request on a local socket.

    python remlax_run.py serveur <port_file> <token> [<pid of the R process>]

As soon as it listens, on 127.0.0.1 only, the server writes "<port> <pid>"
into <port_file> (atomically). Each connection carries one request, a JSON
line:

    {"token": "...", "argv": [...]}        runs cli.main(argv)
    {"token": "...", "order": "stop"}      stops the server

The output of cli.main (stdout and stderr) is sent back line by line during
the fit, then a line "\\x1eRXFIN <code>". 0x1E is the ASCII record separator,
which R can read, unlike the null byte. A request without the right token is
closed without an answer.

An R process that dies without a finaliser (child of a fork, killed session)
would leave its server orphaned, each orphan holding JAX in memory
(dev/notes.md). Between two requests the server therefore checks every 2 s
that the R process that launched it still exists, and stops otherwise. This
check is POSIX only. Under Windows os.kill(pid, 0) would send a Ctrl-C, so the
R finaliser remains the only protection there.
"""
import json
import os
import socket
import sys
import traceback



class _LineStream:
    """Text pseudo-file that sends each complete line over the socket."""

    def __init__(self, conn):
        self.conn, self.buffer = conn, ""

    def write(self, s):
        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._send(line + "\n")
        return len(s)

    def flush(self):
        if self.buffer:
            self._send(self.buffer + "\n")
            self.buffer = ""

    def _send(self, text):
        try:
            self.conn.sendall(text.replace("\x1e", "").encode("utf-8", "replace"))
        except OSError:
            pass                                 # R has closed: finish the fit anyway

    def isatty(self):
        return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    port_file, token = argv[0], argv[1]
    r_pid = int(argv[2]) if len(argv) > 2 else 0
    watch = r_pid > 0 and os.name == "posix"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    if watch:
        srv.settimeout(2.0)
    from remlax import cli                       # jax, numpy, scipy: once only
    import jax
    tmp = port_file + ".tmp"
    with open(tmp, "w") as f:
        f.write("%d %d\n" % (srv.getsockname()[1], os.getpid()))
    os.replace(tmp, port_file)
    n_fits = 0
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if watch and not _alive(r_pid):
                break                            # the R process no longer exists
            continue
        conn.settimeout(None)
        try:
            line = conn.makefile("rb").readline()
            req = json.loads(line.decode("utf-8") or "{}")
        except (OSError, ValueError):
            conn.close()
            continue
        if req.get("token") != token:
            conn.close()
            continue
        if req.get("order") == "stop":
            conn.close()
            break
        stream = _LineStream(conn)
        saved_streams = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            # chrono reset: the origin is the launch on the R side, if passed
            cli.chrono.reset(req.get("t_launch"))
            cli.chrono.mark("transfer")
            code = cli.main([str(a) for a in req.get("argv", [])])
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:                    # noqa: B902 - every failure is reported to R
            traceback.print_exc()
            code = 1
        finally:
            stream.flush()
            sys.stdout, sys.stderr = saved_streams
            # The functions compiled per structure (mme._SIGMA_CACHE) are kept,
            # because they are the gain of the server. The JAX caches are
            # cleared only after a number of fits, to bound memory.
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
