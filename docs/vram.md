# Where the VRAM goes, measured

This page exists because three probes on a 10 GB GPU slice told us only "it does
not fit", while one buffer analysis on the CPU told us *what* does not fit, in
three minutes. Every figure below is measured, and where a figure of mine was
wrong it is recorded as wrong — the point of the page is to keep the next reader
from repeating the reasoning that produced it.

Model: the five-trait direct-and-indirect-effects design, `n = 8987`, 112
variance parameters, 37 random terms, unstructured genetic covariance of
dimension 9 and 6, unstructured residual. `V` is 616 MB.

## The buffer, by stage

| stage | temp buffer | multiple of `V` |
|---|---:|---:|
| REML core, gradient with respect to `V` | 1233 MB | 2.0 |
| assembly (support-restricted), gradient | 2185 MB | 3.5 |
| **whole chain, support-restricted** | **5457 MB** | **8.9** |
| whole chain, full-height factors | 6270 MB | 10.2 |

Read the last two rows together: restricting each term's factor to its support
buys a factor **1.15 on the chain**, not the factor 2.2 a synthetic bench of the
same nominal shape had promised. The bench was wrong for a reason worth naming —
see below.

## What the core costs, and why it is not the target

The core is 2.0 x `V`, and that is already the result of an optimisation: the
gradient of the objective with respect to `V` is written by hand as

    d(-2logL)/dV = P - (Py)(Py)',   P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1

so the backward pass keeps three n x n matrices instead of the ten that
automatic differentiation through a Cholesky factorisation and its triangular
solves would retain. A chunked variant of that backward pass was tried and
**measured as worse** (7.34 GB -> 8.11 GB at n = 16211): `fori_loop` carries the
full matrix in its state, so the complete matrix stays live *in addition* to the
blocks. It is left in the code, disabled, precisely so the idea is not retried
from scratch.

## What the factors cost

| | full height | restricted to support |
|---|---:|---:|
| all 37 terms | 1608 MB | **446 MB** |

A factor 3.6 on that line item, and it is exact — a term of this model concerns
one block and one trait, so its incidence is zero on every other row, and those
zeros were stored as double-precision floats.

Why it does not translate into a factor 3.6 on the chain: the two genetic terms
cover **every** row of their species (7147 of 8987 for wheat), so restricting
them gains almost nothing, and they carry the widest factors.

## Three of my own figures that were wrong

**"90 percent of the factor memory is in one-parameter nuisance terms whose
incidences are plain factors, so accumulate them by index comparison."** Wrong,
and measurement reversed it. The ten indirect-environmental terms hold 82
percent of the factor memory (1316 MB, `q = 1920` each) and are *weighted* with
11 to 12 entries per row at order 6 — the one class the index trick cannot
touch. The eighteen spatial terms, which *are* unit factors on their support,
hold 6 percent.

**"2.90 GB of constants are captured during lowering."** The warning was real,
its attribution was not: those were the full-height factors (2769 MB) captured
by the *reference* form my analysis script was compiling alongside for the
identity test. The objective's own restricted incidences are 161 MB. Passing
them as arguments instead was then measured as a small **loss** (4580 MB of temp
plus 161 MB of arguments, against 4548 MB), and was reverted.

**"22 support groups and 47 residual sections."** The real design has **6**
groups and **2** residual sections — one per species — and the larger residual
section alone is 7147 rows, 390 MB. The residual is two large blocks, not a
mosaic of small ones.

## The failure on a slice is fragmentation, not a shortage

Both probes plateaued at **8548 MiB, to the megabyte**, before and after the
optimisation. A number that does not move when consumption changes is not
consumption: it is the allocator's arena, capped at 95 percent of the slice.
The first probe then failed on a request of 1.58 GiB — exactly the full-height
concatenated factor — and the second on 4.02 GiB, which is the temp buffer.

So the quantity to reduce is **the largest contiguous buffer**, not total
memory. On a 10 GB slice the arena leaves room for roughly 4000 MB contiguous;
the chain wants 5457 MB.

## How to measure this, and how not to

    c = jax.jit(f).lower(args).compile()
    c.memory_analysis().temp_size_in_bytes

Compile-time, so it needs no GPU and no queue. Three minutes on a CPU named the
dominant stage; three slice submissions had only reported failure.

Two traps, both of which cost a submission here:

- `peak_bytes_in_use` from `device.memory_stats()` is the **process-cumulative**
  high-water mark. Measuring two variants in one process makes the second
  report the first's peak, so the cheaper form looks identical to the dearer
  one — measured, 866.2 MB for both, while the factors differed by 14x. Measure
  each variant in a separate process.
- Do not let a diagnostic assertion gate the measurement it accompanies. An
  assertion on the peak killed a probe before the real model was ever tried, so
  the question the job existed to answer went unanswered.

## What the numbers say to do

At `n = 8987` an evaluation costs about 1.1 s on a whole A100 against 4.84 s for
the nine-trait model, so four neighbourhood orders fit in two to four hours in
series on the whole card. A 1/7 slice exposes 7 of 108 compute units and is
roughly fifteen times slower, so parallelism across slices only buys the card's
queueing time — and it would still need another factor 1.4 on the largest
contiguous buffer, from an assembly whose chunked variant is already recorded as
having been measured worse.

Fit this model on the whole card. Slices are the right tool when the card is
held for many hours by another sweep, and only then.
