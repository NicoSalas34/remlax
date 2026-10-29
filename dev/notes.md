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

### model.py, incidence.py, residual.py

- One product for all terms: the chain V = R; V = V + B_k B_k' creates a new
  n x n matrix at each step and XLA does not reuse the buffers, one buffer per
  term. On the full multivariate model of Salas et al. (2026), 65 terms at
  n = 16211, that is 65 x 1.96 = 127 GB; the job failed asking for 134 GiB on
  an 80 GB A100. With [B_1 ... B_K][B_1 ... B_K]' the concatenated matrix is
  16211 x 42186 (5.1 GB against 2.0 for V); on a 41-term model the peak went
  from 126.1 to 16.2 MB, same value and gradient. A custom derivative rule
  for the assembly was tried and was useless: the peak is in the forward pass.
- term_support: on the 5-trait model (n = 8987), an IEE term carries
  q = 1920 columns for about 1800 useful rows, 131.6 MB of which 80 % zeros;
  the ten IEE terms made 1316 MB, 82 % of the factor memory. Restricted:
  1598 MB of factors against about 400 MB.
- orthonormal_X: on the wheat IGE model, -2logL was wrong by 4e-7 inside and
  by 1.4e-4 on a sub-model without the QR.
- prefold_fixed_K: 28 s of compilation for n = 3000, q = 1500; 633 s at
  n = 8000 on MIG, and a failed compilation of the ige model at n = 8037. The
  former dense einsum cost n x t x q^2 operations outside BLAS, 66 s for
  n = 1000, q = 8000, done twice (likelihood and AI).
- Incidences as arguments: passing them as arguments gained nothing in memory
  (buffer 4580.3 MB against 4548.7, 160.8 MB more arguments); the warning
  "2.90 GB of constants were captured" came from the full-height factors of
  the reference form compiled beside it for the identity test. Captured, a
  dense 8037 x 8037 incidence (ar1 x ar1 field, 516 MB) made the compilation
  fail.
- Residual us: on the IGE model the us coupling between traits of a unit is
  worth 2228 LRT points for 22 parameters.
- Scaling by n: on a one-factor model, without effective scaling L-BFGS-B
  stopped at -2logL = 425.15 with |grad| = 74.6, against 414.3095 at the
  optimum.
- aliased_entries: the plant term [I, W] of VP+ has q(q-1)/2 flat directions
  as soon as W is symmetric (theorie_v2.md, T1b, IGE project); the genetic
  term [Z, WZ] has none once replication makes W Z Z' non-symmetric.
- level_kind defaults to "fixed" when K is given without "lvl": falling back
  to "id" is the failure that made two GPU runs labelled "kinship" exact
  duplicates of the runs without.

### levels.py

- sep and parts: level_chol reads the second element of parts as the
  dimension of the factor, n_level_params read it as the order; for id and ar1
  both agree by accident. A sep (corb, 4) x (ar1, 5) counted 1 parameter
  instead of 5. level_corr also passed q = None to each factor, so a residual
  with a sep structure was unreachable (found by
  test_levels_catalogue.py::test_sep_level_corr_egale_kron).
- cor: at q = 10, t = -0.3 gives an eigenvalue of -1.7. Reporting tanh(theta)
  gave a value outside the model for small q (at q = 3, tanh = -0.9 becomes
  -0.45).
- arma: the formula of appendix C, (th-ph)(1-th*ph)/(1+th^2-2*th*ph), gave
  non positive definite sequences on 204 draws out of 300 (rho_1 = -0.976 with
  rho_2 = -0.944).
- mtrn metric: three other conventions (delta on one axis, or not as a square
  root) shift logLik by 0.5 to 4 points; the chosen one matches asreml to
  3e-10 on (delta = 2, alpha = 0) and (delta = 2, alpha = 0.6). See
  docs/note_remlax.md of the IGE project for the logLik curves that fixed lvr
  and mtrn; nine families were tried for ilv without success.
- mtrn with every parameter fixed stopped on "index is out of bounds for axis
  0 with size 0" when phi was read before the kind was known.
- cir gradient: clipping t = min(d/phi, 1) gave NaN gradients, n_iter = 0 and
  -2logL 6.4 from the optimum (asreml3 test B7, 2026-09-28).
- sep reported nothing in rho before each factor was reported by position.

### mme.py

- Gaussian kernels are out of scope: simulated, logLik wrong by 3 to 7 % on
  gau, igau and agau through K^-1.
- DIAG_CHOL_MIN: a nearly singular Sigma gave a wrong gradient (fa on 5 traits,
  bench 2026-10-22: 8450 against 148 for the dense engine).
- Grid residual: the REML of the augmented model matches the observed data to
  1e-13 on an ar1 x ar1 field with 15 % empty cells.
- Deferred full blocks: on Krause, 178 million entries and 18 s of
  preparation for nothing in the dense inversion.
- LAPACK on CPU: XLA formed C by scatter and C^-1 by cho_solve against the
  identity; 0.34 s per full evaluation on drops2 (m = 1428).
- GPU block mode: assembling C on the host (2.8 GB at m = 18 860), sending it
  and bringing C^-1 back cost 6.9 s per evaluation on Krause (A100) for about
  2 s of computation. Unrolled block by block, the compiled function had 406
  operations for a fa on 28 environments and compiled in about 30 s.
- _inverse_from_factor: the former general LU took 2.6 to 5 s for 3 768
  genotypes; np.triu on 3768 x 3768 cost 2.6 s.
- Cache of the last evaluation: after AI, the finish then the outputs
  re-evaluated the final point, two more factorisations (27 + 11 s on Krause
  us(5) on CPU). Reusing the value-only factorisation saves about 10 s of
  Cholesky per accepted AI iteration on Krause fa2 (CPU), out of about 38 s.
- Compiled Sigma functions cached by structure: 0.7 s per fit saved in the
  server on a two-term iid model.

### Python safety net (remlax-validation/refacto/golden_py.py)

- Engine values at fixed theta (dense, low-memory, blocks, the three MME
  inversions), MME outputs and whole AI fits, compared to the bit.
- The script pins itself to one CPU core. Allowed to use several cores, XLA
  on CPU was not deterministic despite the thread flags: the same commit
  differed by up to 3e-13 on the dense AI matrix, the blocks gradient and one
  AI fit in about one run out of six (10 of 10 identical runs once pinned).
  golden.R was stable for the same reason: the launcher bounds the solver to
  one core.

### ai.py

- Salvos: on fa(2) with 5 traits, n = 1500 and 2000, three bench fits ended
  460 to 690 logL units under the optimum (AI stuck, early L-BFGS-B
  fallback), depending on the BLAS and the number of threads.
- Variance scale: AI in theta crawls on us9, us12 at n ~ 500 and mv3_ref.
- REDUCTION_MAX: without it, an AI step sent the residual variance of a
  section to 1e-8 times its value; log|R| and log|C| become huge and cancel,
  and AI declared convergence on a wrong point (Krause, 5 treatments, 600
  genotypes: 3364.93 against 3338.25 at the optimum). The factor also decides
  the optimum reached when there are several: on DROPS one-stage us(5) (ar1
  fields with rho near 1), /10 reaches -1146.19, /100 -1384.0 and /1000
  -1189.6, three stationary points confirmed by the dense engine; /10 is the
  only factor right on the three cases tested, and it slows an fa2 (104
  iterations against 48 at /1000).
- LEVEL_STEP_MAX: on DROPS one-stage (ar1 fields, rho ~ 0.997), a full AI step
  moved rho from one basin to another, and the optimum depended on the number
  of BLAS threads (-1146.19, -1305.6 or -1462.5 for the same model); clamped
  at 0.5, right on DROPS fa1 and us with 6 and 8 threads.
- RATIO_MIN: changed no optimum on the cases tested and cost up to 60 % of
  time, hence optional.
- N_CLAMP_TO_FLOOR: Heywood case of an fa, 104 -> 50 iterations on Krause fa2
  (600 genotypes), same optimum; no other bench case changes.
- f_only: on drops2 (dense MME, m = 1428) the value alone costs 0.10 s against
  0.49 s for the full evaluation, and a damped iteration tries up to 20 steps.
- Faces of the cone: without them the Newton step points out of the cone and
  the loop stalls (us9 at n = 500: 1.4 units under the optimum). Without the
  rotations, AI stopped after 9 iterations on the face, followed by 490
  L-BFGS-B iterations and 4 polishing steps for 1.98 units of -2logL (absolute
  bench, us(8), q = 1000, A100); us(12) up to 3000 iterations. With a face
  threshold equal to the projection floor (MME, 1e-8) the rank changed at each
  iteration (us(12): 11, 9, 10). Requiring all null directions on the same
  side freed the whole block (us(12), MME). us9 at n = 500: -798.76 on the
  face, -796.94 after the fallback, before the rotations.
- LM damping: fa of rank 2, n = 500, the decrement stayed at 1.26e3 with steps
  of 3e-5 for twelve iterations without it. Keeping mu: fa2 at t = 8,
  q = 1000, three damped iterations of 20 to 35 s against 4 s for a full one.
  The full Newton step first after a damped iteration: drops2 fa1 7.2 s in v10
  against 14.5 s with mu kept alone.
- make_fga without forming P: on ar1 x ar1 at n = 2000, 7.25 n x n
  temporaries before, 3.7 GB at n = 8000 and a failed AI on a 10 GB MIG slice.

### finition.py

- decrement regularised: the truncated version ignored directions of
  curvature under rel * max|w|; on a fa with 28 environments (drops2_fa1) the
  finish concluded at -2logL 5598.78 when the optimum is 5595.41.
