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

### lag.R

- Reciprocal Psi with IEE* (theorie_v2.md of the IGE project, T1b and study
  A): freeing the reciprocal entry raised the standard error of psi_21 by 8 to
  87 % at psi_v = -2, multiplied it by 6.4 at psi_v = -0.5, or made the
  information singular. Hence the warning of .rx_warn_tb_iee().
- Border covariates: see IGE_trait_paths_sim and revision_moyennes_bordures.md
  of the IGE project.

## Python side

### cli.py

- Allocator ceiling. JAX caps its pool at 75 % of the GPU memory by default,
  even without preallocation. On a 10 GB MIG slice the cap is 7.65 GB: the
  compilation autotuning reached 6.1 GB and fragmented the pool, then the
  evaluation temporaries (4.1 GB in one piece on mvD7) found no room
  (measured 2026-10-25). At 95 % the same computation runs.
- The device context used to be entered with `ctx.__enter__()` and left only
  at the end of a successful run; in the server process a refused request left
  it entered. It is now a `with` block.

### fit.py

- maxiter = 0 is short-circuited. L-BFGS-B given maxiter = 0 does not return
  the start: on an ar1 field theta moved by 1.6e-2 towards the optimum
  (maxfun = 10 x 0 is not read as "no evaluation"). With --theta-in this made
  the dense and sparse engines look 8e-2 apart, where an independent dense
  REML computation gave exactly the sparse value.
- Restarts: on 150 random configurations, 2 had a better optimum elsewhere, at
  0.009 and 0.09 of -2logL. The Newton decrement does not see a secondary
  hill.
- Compilation timing: measured on a dense kinship model, compilation reached
  66 % of the total time. The first evaluation is now kept in the memo and
  reused by the optimiser; the former extra evaluation at a perturbed point
  cost 60 s per fit on mvD7 (sparse MME).
- Counting evaluations: reconstructing compilation + n_eval x unit cost from
  n_iter + 2p failed by a factor two (median ratio 0.48 over 32 cells), since
  L-BFGS-B evaluates several times per iteration and each polishing step adds
  a full Hessian.
- Progress lines: a fixed period of 25 iterations left the log silent for
  52 minutes on the real model; the callback had also once been defined and
  never passed to minimize(). Newton polishing was silent too: 61.5 h and
  9 979 evaluations after the last iteration line on a 198-parameter model,
  read as a hang for four days.
- A NaN gradient at the start made L-BFGS-B return the start at iteration 0,
  declared converged (cir, 2026-09-28).
- AI fallback to L-BFGS-B: on us9 at n = 500, AI stalled on the edge of the
  cone; Newton polishing also got there but in 4 665 evaluations.
- No polishing after a converged AI: on alfalfa, 25 steps for gains of 1e-8,
  the total time multiplied by 4 to 5.
- AI saddle point: simulated fa of rank 2 on 5 traits, AI decrement under
  1e-6, a negative Hessian eigenvalue, logLik 1.6e-2 under the optimum.
- Floor escape (_floor_escape): on a weighted fa1 with 28 environments
  (drops2), the 28 specific variances stuck at exp(2 x -11.4), -2logL 5469
  against 3507 at the optimum, where they are 0.03 to 0.8.
- MME floor: a variance of exp(2 x -12) = 4e-11 left C too ill-conditioned for
  log|C|; the floor is 1e-8 times the least-squares residual variance. It is
  kept when the model falls back to the dense engine (defect listed in the
  cleanup plan).
- Dense restart of a failed MME fit: GRAD_REL_DENSE_RESTART was 1e-3, too lax.
  On mtigee8 (208 parameters, rank-deficient us(16) block) the three MME
  routes stopped at 2-5e-4, 0.13 to 0.20 logL units under the optimum,
  without triggering the restart (IEE bench, 2026-10-23). The MME engine can
  lock near a rank-deficient Sigma (fa on 5 traits, bench 2026-10-22). With a
  Kinv-only term the dense restart gave another model (verif_simulations,
  parente_pedigree_kinv: logLik -684.65 instead of -657.59). A converged AI
  fit can keep a relative gradient of a few 1e-5 (pedigree with 500
  individuals: decrement < 1e-6, 5e-7 from the optimum).
- Two logLik conventions: remlax and lme4 include (n - p)/2 log(2 pi), asreml
  and sommer omit it; 219.6 apart on a data set with n = 240, p = 1, checked
  to 1.3e-8.
- The bounds travel with the result: an R-side bound hard-coded at -8, where
  the solver bounds at -12, excluded eleven variance components from the
  standard errors.
- Degenerate components: the typical case is AR1 aliased with the nugget, the
  field absorbing the whole residual; asreml then refuses to converge.
- Without Hessian, the projected gradient is still reported: a 39-minute fit
  of the IGE order sweep once returned no criterion at all.

### start.py

- _sum_diag_ZKZ: on a GBLUP with 128 000 observations and 2 000 genotypes,
  nnz(Z) = 128 000 against nnz(G) = 2 000, and 4 s per call by the Z LK route.
- _exposure: the dense version formed Z (n x tq) and Z LK by einsum, 45 s of a
  100 s fit on a GBLUP with 3 768 genotypes. Dividing by n x t instead of the
  loaded pairs made every starting variance t times too large on a fa1 with 28
  environments (drops2): specific variances 39 against 0.03 to 0.7 at the
  optimum, -2logL 10136 against 5595, 13 damped AI iterations before the first
  full step.
- Loading bounds: with the common bound of 12, loadings of a fa on 5 traits
  reached 12 in a few AI steps, eigenvalues 0, 0, 0, 261, 759 for true
  variances of order 1 (bench 2026-10-22).
- initial_theta from the residuals of X: on 5 treatments x 3 768 genotypes
  (Krause), the start from raw y gave -2logL 49 000 against 20 311 at the
  optimum, and 7 damped AI iterations (920 s of 1 216) before the first full
  step.
- rho^d start: theta = 0 put rho on the clipping point; an exponential field on
  irregular positions stayed at rho = 0, -2logL 65 units from the optimum
  found by nlme::gls(corExp) (2026-09-28).
- Range sweep: on the irregular field of asreml test B7 (90 positions), the
  circular kernel has three local maxima (ranges 5.0, 7.4 and 10.9, -2logL 6.1
  apart), and asreml and remlax both went to the first. On a 60-position field
  with four local minima, the best candidate at starting variances led to the
  hill at 104.4 instead of 101.8, hence the short descents.

### checks.py

- Two rows of the same unit and trait under an iid residual with a structure
  between units (dsum(~ ar1(col) | site), several rows per column) gave
  -2logL = NaN on CPU and GPU without a word; found by the parity sweep.
- The former check formed Z dense (n x tq): 2 GB and 4 s at n = 128 000,
  q = 2 000.

### effects.py

- PEV by list of terms: the full IGE model has 23 terms, 21 of them nuisances
  (spatial block effects, indirect environmental effects); asking for all of
  them killed the process for lack of memory on the first real run. The PEV
  exists because H2_Cullis needs it; the asreml validation suite flagged it as
  the only output asreml gave and remlax did not.

### components.py

- The Jacobian is compiled: in eager mode it chained hundreds of small JAX
  operations, 1.7 s on a fa1 with 28 environments for a table whose fit took
  3 s.
- Metric level families used to report tanh(theta), with an arbitrary sign,
  and vpredict differentiated the wrong function on the negative side.

### fixed_tests.py

- Expected against observed information: on a 27-observation trial the
  observed Hessian gave 21.83 denominator df, the expected information 21.769,
  the value of pbkrtest.
- Validation against pbkrtest::KRmodcomp on an unbalanced block design: df
  21.76897 against 21.76896, scale 0.9998304 against 0.9998304, adjusted
  standard errors equal to 1e-7.
- Second-order term: computed in the internal parametrisation it gave
  Phi_A = 0.4375 in variances against 0.3465 in log standard deviations on a
  block design, 21 % apart, while the first-order part agreed to 2e-16.
- Contracting Theta against P_i instead of dPhi gave A1 and A2 of order n^3
  instead of O(1), and negative df.
