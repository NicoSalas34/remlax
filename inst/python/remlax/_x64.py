"""Turn on double precision before any JAX array is created.

JAX uses float32 by default, which is too coarse for a REML likelihood. The
log-determinant of an ill-conditioned matrix drifts quickly in float32
(dev/notes.md). __init__.py imports this module first, so importing any
submodule of remlax turns double precision on.
"""
import jax
jax.config.update("jax_enable_x64", True)
