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
