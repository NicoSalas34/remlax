# remlax, the R package

Installable form of the R interface that lives in `../R/`. The files in
`R/remlax.R`, `R/remlax_tmb.R` and `R/remlax_scan.R` here are **generated** by
`tools/sync_sources.R` from the scripts at the repository root, and a copy of
the Python solver is shipped in `inst/python/`. Do not edit them here.

## Install

```r
# from a clone
install.packages("rpkg", repos = NULL, type = "source")
# or, once the repository is public
remotes::install_github("NicoSalas34/remlax", subdir = "rpkg")
```

The package needs a Python interpreter (>= 3.10) with `jax`, `numpy` and
`scipy`. Point `RX_PY` at it if it is not the default `python3`:

```r
Sys.setenv(RX_PY = "/path/to/venv/bin/python")
library(remlax)
rx_solver_args()     # shows which solver copy will be used
```

Or let the package create one: `exe <- rx_install_python()` builds
`~/.remlax/venv` with jax, numpy and scipy (`cuda = TRUE` for the GPU build)
and prints the `RX_PY` line to keep. `rx_python_check()` reports what the
chosen interpreter imports; a Python without jax is refused before the solver
starts, with the interpreter tried and the line to fix.

If the Python package `remlax` is pip-installed under that interpreter, it is
used; otherwise the copy shipped with the R package is used. The sparse engine
(`rx_fit_sparse()`) additionally needs `RTMB`.

## Keep the package in step with the scripts

```sh
Rscript rpkg/tools/sync_sources.R           # regenerate
Rscript rpkg/tools/sync_sources.R --check   # verify (also run by the tests)
```

## Check

```sh
cd rpkg && Rscript -e 'roxygen2::roxygenise()' && cd .. && R CMD build rpkg && R CMD check remlax_*.tar.gz
```
