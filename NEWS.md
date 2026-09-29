# remlax 0.1.0

* First submission to CRAN.
* REML fitting of linear mixed models on the dense marginal covariance, with a
  'JAX' solver shipped in `inst/python` and called as a subprocess.
* Covariance structures of the 'ASReml-R' catalogue, neighbourhood incidences
  for direct and indirect genetic effects, ratios with standard errors, and a
  genome scan at fixed variance components.
* `rx_validate()` and the validation vignette compare the fits with 'lme4',
  'nlme' and 'sommer'; precomputed comparisons with 'asreml' are shipped in
  `inst/extdata`.
