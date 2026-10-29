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
