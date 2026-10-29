# ==============================================================================
# reml.R - rx_reml(): fit a model given by formulas
# ==============================================================================
# rx_reml() is the main entry point. It builds the fixed design from `fixed`,
# drops incomplete rows, extracts the spl2d() surfaces, parses the random and
# residual formulas into an rx_model(), and fits it with rx_fit(), to which
# every other argument is passed. It keeps on the fit what rx_predict()
# needs: the fixed formula, the cleaned data and the factor levels.
# ==============================================================================

rx_reml <- function(fixed, random = NULL, residual = "units", data,
                    trait = NULL, unit = NULL,
                    backend = c("auto", "gpu", "cpu"), lag = NULL, ...) {
  backend <- match.arg(backend)
  fx <- .rx_fixed_design(fixed, data)
  cleaned <- .rx_drop_incomplete(fx, data, trait, unit, lag)
  data <- cleaned$data; y <- cleaned$y; X <- cleaned$X
  trait <- cleaned$trait; unit <- cleaned$unit
  tr <- if (is.character(trait) && length(trait) == 1L) data[[trait]] else trait
  un <- if (is.character(unit)  && length(unit)  == 1L) data[[unit]]  else unit

  # A spl2d() surface gives five random terms and fixed columns (its null
  # part); it is extracted before the other terms, on the cleaned data.
  spl <- .rx_extract_spl2d(random, data, trait = tr)
  random <- spl$random
  .rx_check_spl2d_at(spl, fixed, random)
  X <- .rx_add_spl2d_columns(X, spl)

  terms_l <- c(.rx_parse_random(random, data, trait = tr), spl$terms)
  res_obj <- .rx_parse_residual(residual, data,
                                trait = if (!is.null(trait)) trait else tr,
                                unit  = if (!is.null(unit))  unit  else un)
  mod <- rx_model(y, X, terms_l, res_obj,
                  name = deparse(fixed[[2]])[1], lag = lag)
  # A spl2d() surface needs the PEV (effective dimensions, generalised
  # heritability): they are computed by default unless pev = is given.
  dots <- list(...)
  if (!is.null(spl$X) && is.null(dots$pev)) dots$pev <- TRUE
  fit <- do.call(rx_fit, c(list(mod, backend = backend), dots))
  fit$model <- mod
  # The factor levels make a prediction grid give the same columns as X.
  fit$fixed   <- fixed
  fit$data    <- data
  fit$xlevels <- stats::.getXlevels(terms(fixed, data = data),
                                    model.frame(fixed, data, na.action = stats::na.pass))
  fit$call    <- match.call()
  fit$spl2d   <- if (!is.null(spl$X)) list(columns = colnames(spl$X), terms = unique(spl$groups),
                                           surfaces = spl$surfaces)
  fit
}

# Response and fixed design. X carries "assign" and "termes" (the term of each
# column), which the Wald tests use to test whole terms.
.rx_fixed_design <- function(fixed, data) {
  mf <- model.frame(fixed, data, na.action = na.pass)
  y  <- model.response(mf)
  X  <- model.matrix(fixed, data)
  assign <- attr(X, "assign")
  labels <- c("(Intercept)", attr(terms(fixed), "term.labels"))
  attr(X, "termes") <- labels[sort(unique(assign)) + 1L]
  list(y = y, X = X, assign = assign, labels = labels)
}

# Drops the rows where y or X is missing, with a message. Subsetting X loses
# its attributes, which are restored. A spatial lag describes the rows of
# `data`, so incomplete rows are refused with one.
.rx_drop_incomplete <- function(fx, data, trait, unit, lag) {
  y <- fx$y; X <- fx$X
  ok <- stats::complete.cases(y, X)
  if (!all(ok) && !is.null(lag))
    stop("rx_reml : ", sum(!ok), " ligne(s) incomplete(s) et un decalage spatial. Les ",
         "lignes de rx_lag() decrivent celles de `data` ; retirer les lignes incompletes ",
         "AVANT de construire rx_lag(), ou passer par rx_model().", call. = FALSE)
  if (!all(ok)) {
    message("rx_reml : ", sum(!ok), " ligne(s) incompletes retirees.")
    data <- data[ok, , drop = FALSE]
    y <- y[ok]; X <- X[ok, , drop = FALSE]
    attr(X, "assign") <- fx$assign
    attr(X, "termes") <- fx$labels[sort(unique(fx$assign)) + 1L]
    if (!is.null(trait)) trait <- trait[ok]
    if (!is.null(unit))  unit  <- unit[ok]
  }
  list(data = data, y = y, X = X, trait = trait, unit = unit)
}

# With spl2d(at =), the null part of each level carries x, y and x*y only: the
# mean of the level must come from elsewhere, usually the fixed formula.
.rx_check_spl2d_at <- function(spl, fixed, random) {
  for (a in unique(unlist(lapply(spl$surfaces, function(z) z$at$name))))
    if (!a %in% c(all.vars(fixed), if (!is.null(random)) all.vars(random)))
      message("spl2d(at = ", a, ") : ", a, " n'est ni dans les effets fixes ni dans les ",
              "effets aleatoires ; la moyenne de chaque surface n'est donc pas estimee. ",
              "Ajouter ", a, " a la formule fixe, en general.")
}

# Appends the null part of the surfaces to X, one fixed term per surface (one
# Wald test for its columns).
.rx_add_spl2d_columns <- function(X, spl) {
  if (is.null(spl$X)) return(X)
  asg <- attr(X, "assign"); lab <- attr(X, "termes"); ctr <- attr(X, "contrasts")
  X <- cbind(X, spl$X)
  attr(X, "assign") <- c(asg, max(asg) + as.integer(factor(spl$groups, levels = unique(spl$groups))))
  attr(X, "termes") <- c(lab, unique(spl$groups))
  attr(X, "contrasts") <- ctr
  X
}
