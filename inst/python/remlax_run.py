"""Entry point used by R: runs one module of the remlax package as a program.

    python remlax_run.py cli <arguments>        runs remlax.cli
    python remlax_run.py serveur <arguments>    runs remlax.serveur
    python remlax_run.py scan_cli <arguments>   runs remlax.scan_cli

Python puts the directory of this file, and only it, at the head of sys.path.
`remlax` is therefore imported as a package from this copy, every module has a
single name (remlax.<module>), and the current directory of the R session is
never searched, so a local file such as jax.py cannot shadow a dependency.
"""
import os
import sys
import time

# Start of the interpreter, read by remlax.cli for its timing of the start-up
# stage, before the package imports JAX, NumPy and SciPy.
os.environ.setdefault("RX_T_DEBUT", repr(time.time()))

import runpy  # noqa: E402

if len(sys.argv) < 2:
    sys.exit("usage: python remlax_run.py <module> [arguments]")
runpy.run_module("remlax." + sys.argv.pop(1), run_name="__main__", alter_sys=True)
