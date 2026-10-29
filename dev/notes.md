# Developer notes: measurements and history

These notes keep the measurements, dates and bug histories that used to sit in
code comments. The code keeps one or two sentences on why it is written as it
is; the evidence behind that choice lives here, grouped by file. Nothing here
is needed to use the package, and the directory is not part of the built
package (`.Rbuildignore`).

## R side

### counts.R

- `rx_n_theta()`. Imposing a common theta matters whenever two engines are
  compared: evaluated at their own starting points, their log-likelihoods
  diverged by 1e-5 to 9e-3, growing with the number of traits, which looked
  like a formulation defect. At a common theta they agree to 1e-12.
- `rx_n_theta()` once went through counters that only knew iid/diag/us and
  id/ar1/ar1ar1/prec/fixed. On fa, corh, cor, ar2, a us residual or a
  sectioned residual it stopped with "structure inconnue", although its
  documentation made it the length of theta for any model. It now uses the
  general counts of the dense engine, checked by test-comptages.R.
- `.rx_n_level_mtrn()`. A missing `est_*` option (rx_term() called directly
  with partial options) used to stop with "subscript out of bounds"; counting
  phi as fixed instead would have cut theta differently from the solver.

### model.R

- The parameter count of a model includes the between-level parameters of the
  residual (the rho of an ar1 residual), one set per section. Omitting them
  displayed 2 parameters for an ar1 x ar1 field that has 4.

### formula_residual.R

- The parser needs the column name of `trait` to recognise diag(trait). When
  only the vector was given, the trait factor was read as a unit factor and
  the residual became iid with 4 "units" instead of 180, a different model
  without any error. The name is now recovered from the columns of `data`.
- The options of a residual structure (mtrn, own) were read after the early
  exit of the 2D metric branch, so ~ mtrn(x, y, phi = ..., nu = ...) lost its
  options: four different parameter sets gave the same log-likelihood,
  -19.5572007, the sign that none of them arrived.
- The duplicate check of ar1(row):ar1(col) once tested the cell alone. In long
  format each cell appears once per trait by construction, so it refused
  us(trait):ar1(row):ar1(col), the most common spatial multi-trait model. It
  now tests the pair (cell, trait), as fit.py does.

### formula_random.R

- The 2D metric structures (iexp and its neighbours) were once tested inside a
  branch that only accepted ar1/ar2/.../gau, so they fell into "terme non
  reconnu" although both sides implemented them. A single list is now used.
- struct = on a correlation structure (asreml's v/h suffixes) once passed the
  raw factor whatever `struct`, so t = 1: ar1(col, struct = "diag") silently
  became ar1(col), one variance instead of one per trait.
- The name of a str() term once came from the raw text of the expression
  ("gid_mmZwnamepente"); it now joins the names of the incidences.

### bundle_write.R

- The unit of an ar1 x ar1 residual keeps its cell code. Renumbering it with
  factor() once dropped the empty cells and shifted the following ones, so the
  solver read the correlation at the wrong cells. Measured with 5 % missing
  cells: logLik -854.39 against -820.77 with the true geometry, and a row
  correlation of 0.18 for 0.6 simulated.

### bundle_read.R

- `$` matches partially on lists. Once the result carried `sigmas_res`,
  `r$sigmas` (not yet created) matched it: the matrices of the terms were
  added to the residual list, and `fit$sigmas[[1]]` returned the residual
  instead of the first term. Every positional comparison of the suites moved
  by one, without any error, while the log-likelihood stayed right.
- The Hessian is read back because the derived flag alone is not actionable:
  on a five-trait model the verdict "12 near-null directions out of 57" (no
  negative eigenvalue, condition number 2.4e9) only becomes useful with the
  eigenvectors, which say which parameters are not identified.

### python.R

- Without the check of the interpreter, a Python without jax gave a ten-line
  Python traceback ending in "No module named 'jax'" and "solveur REML :
  echec", without naming the interpreter tried or how to choose another. It
  happened at the first try of the installed package (system python3, no jax).

### launch.R

- Threads during R CMD check, measured on 20 cores: OMP_NUM_THREADS=2 left
  2.3 cores busy; one computation thread with Eigen threading off left 1.2.

### server.R

- Start-up cost of a subprocess, measured on 2026-10-22 at n = 500: 2.3 s
  outside the solver for 0.15 s of computation.

### fit.R

- The bounds of theta were accepted by the command line (--floor, --ceil) but
  no R argument passed them: a caller who set a floor at -8 got -12 without a
  word, and a pipeline recorded PAR_FLOOR=-8 in its specifications while the
  solver worked at -12.
- A term given only by Kinv was fitted by the dense engine as independent
  (level "id", "K=I" in the log). Measured on Hauteur.4: a genetic variance of
  0.3019, exactly that of the univariate fit without kinship (0.3013), instead
  of 0.1849 with kinship (0.1839); every heritability was inflated by a factor
  1.63 without any diagnostic.
- The interpreter is no longer probed before the first fit: the probe started
  a Python only to import jax, numpy and scipy (1.5 s per R session) before the
  solver imported them again. It now only explains a failed launch.
- pev takes term names: on the full IGE model, 21 of the 23 terms are nuisance
  terms, and asking for all their prediction error variances exhausted the
  memory at the first real run.

### scan.R

- A conditional test such as "ind|dir+sim" contains a vertical bar that the
  shell of system2() reads as a pipe: the command was cut in two and returned
  127, "command not found", with nothing pointing at the pipe. Every value is
  now quoted.
- read.csv() without check.names = FALSE turned `p_dir+ind+sim` into
  `p_dir.ind.sim`; a caller looping over `tests` then skipped the composite
  tests silently (three tests out of six missing from a summary).
- tempdir() can be relative when $TMPDIR is, and the path then resolved
  against the working directory of the Python process.
- Identical doses for both families are written once: 292 MB in the format of
  the IGE project.
