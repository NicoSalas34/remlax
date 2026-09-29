## Submission

This is a new submission.

## Test environments

* local Linux, R 4.5.3, with and without the Python solver
* win-builder, R-devel (to do before submission)

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.

## Python engine

The solver is written in Python with 'JAX' and ships with the package in
`inst/python`. It needs a Python interpreter with 'jax', 'numpy' and 'scipy',
declared in SystemRequirements. Without it, the package loads, examples that
need the solver are skipped by `@examplesIf rx_python_check(quiet = TRUE)$ok`,
tests are skipped, and the vignettes show precomputed results.

When the solver runs under R CMD check (`_R_CHECK_LIMIT_CORES_` set), it is
limited to one computation thread. Python is started with
`PYTHONDONTWRITEBYTECODE=1`, so nothing is written to the library.

`rx_install_python()` writes a virtual environment to
`tools::R_user_dir("remlax", "data")` only when called, asks for confirmation
in interactive sessions, and `rx_remove_python()` deletes it. Its example is
in `\dontrun{}` because it downloads several hundred MB from PyPI.

## asreml

'asreml' is commercial and is not a dependency. The comparisons with it are
shipped as precomputed tables and read by `rx_validation_results()`.
