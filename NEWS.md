# remlax 0.1.0

* First submission to CRAN.
* REML fitting of linear mixed models on the dense marginal covariance, with a
  'JAX' solver shipped in `inst/python` and called as a subprocess.
* Covariance structures of the 'ASReml-R' catalogue, neighbourhood incidences
  for direct and indirect genetic effects, ratios with standard errors, and a
  genome scan at fixed variance components.
* `summary()` of a fit, in the layout of `summary.asreml`: log-likelihood in
  both conventions, AIC, BIC, convergence, variance components with standard
  errors (delta method, exact Jacobian) and `P`/`U`/`B`/`F` codes, fixed
  effects, and BLUPs with `coef = TRUE`.
* `spl2d(x, y, nseg =, deg =, pord =, name =)` in the random formula of
  `rx_reml()`: a two-dimensional P-spline surface whose null part joins the
  fixed effects; `rx_predict()` averages it over the field.
* `rx_validate()` and the validation vignette compare the fits with 'lme4',
  'nlme' and 'sommer'; precomputed comparisons with 'asreml' are shipped in
  `inst/extdata`.
