# Ajustement REML par formule
#
# @param fixed    formule des effets fixes (reponse a gauche)
# @param random   formule des termes aleatoires (cf. grammaire ci-dessus)
# @param residual "units" (iid), "diag" ou "us" ; ou une formule ~units
# @param trait    nom de la colonne de caractere (format long), ou NULL
# @param unit     nom de la colonne d'unite ; deux lignes de la meme unite sur
#                 des caracteres differents sont correlees sous residual="us"
# @param backend  "auto" | "gpu" | "cpu" - choix de MACHINE, jamais de modele
rx_reml <- function(fixed, random = NULL, residual = "units", data,
                    trait = NULL, unit = NULL,
                    backend = c("auto", "gpu", "cpu"), lag = NULL, ...) {
  # `...` transmet vpredict, wald, fixed_theta, n_restarts, verbose... a rx_fit.
  backend <- match.arg(backend)

  mf <- model.frame(fixed, data, na.action = na.pass)
  y  <- model.response(mf)
  X  <- model.matrix(fixed, data)
  .asg <- attr(X, "assign")
  .lab <- c("(Intercept)", attr(terms(fixed), "term.labels"))
  attr(X, "termes") <- .lab[sort(unique(.asg)) + 1L]
  ok <- stats::complete.cases(y, X)
  if (!all(ok) && !is.null(lag))
    stop("rx_reml : ", sum(!ok), " ligne(s) incomplete(s) et un decalage spatial. Les ",
         "lignes de rx_lag() decrivent celles de `data` ; retirer les lignes incompletes ",
         "AVANT de construire rx_lag(), ou passer par rx_model().", call. = FALSE)
  if (!all(ok)) {
    message("rx_reml : ", sum(!ok), " ligne(s) incompletes retirees.")
    data <- data[ok, , drop = FALSE]
    y <- y[ok]; X <- X[ok, , drop = FALSE]
    # Le sous-ensemblage de X PERD ses attributs : sans eux, wald() testerait
    # chaque colonne isolement au lieu du terme entier. `.keep_attr` n'existait
    # nulle part - toute execution avec une ligne incomplete s'arretait ici sur
    # "objet introuvable".
    attr(X, "assign") <- .asg; attr(X, "termes") <- .lab[sort(unique(.asg)) + 1L]
    if (!is.null(trait)) trait <- trait[ok]
    if (!is.null(unit))  unit  <- unit[ok]
  }
  tr <- if (is.character(trait) && length(trait) == 1L) data[[trait]] else trait
  un <- if (is.character(unit)  && length(unit)  == 1L) data[[unit]]  else unit
  # spl2d() dans `random` : une surface lisse donne CINQ termes aleatoires et
  # des colonnes FIXES (la partie nulle). On l'extrait avant l'analyse des autres
  # termes, sur les donnees deja nettoyees, et la partie nulle rejoint X comme
  # un terme fixe a part entiere (un seul test de Wald pour ses colonnes).
  spl <- .rx_extraire_spl2d(random, data, trait = tr)
  random <- spl$random
  # Par niveau, la partie nulle de chaque surface ne porte que x, y et x*y :
  # la moyenne du niveau doit venir d'ailleurs (bloc fixe ou aleatoire).
  for (a in unique(unlist(lapply(spl$surfaces, function(z) z$at$name))))
    if (!a %in% c(all.vars(fixed), if (!is.null(random)) all.vars(random)))
      message("spl2d(at = ", a, ") : ", a, " n'est ni dans les effets fixes ni dans les ",
              "effets aleatoires ; la moyenne de chaque surface n'est donc pas estimee. ",
              "Ajouter ", a, " a la formule fixe, en general.")
  if (!is.null(spl$X)) {
    asg <- attr(X, "assign"); lab <- attr(X, "termes"); ctr <- attr(X, "contrasts")
    X <- cbind(X, spl$X)
    attr(X, "assign") <- c(asg, max(asg) + as.integer(factor(spl$groups, levels = unique(spl$groups))))
    attr(X, "termes") <- c(lab, unique(spl$groups))
    attr(X, "contrasts") <- ctr
  }
  terms_l <- c(.rx_parse_random(random, data, trait = tr), spl$terms)
  res_obj <- .rx_parse_residual(residual, data,
                                trait = if (!is.null(trait)) trait else tr,
                                unit  = if (!is.null(unit))  unit  else un)
  mod <- rx_model(y, X, terms_l, res_obj,
                  name = deparse(fixed[[2]])[1], lag = lag)
  # Une surface spl2d() demande les PEV (dimensions effectives, heritabilite
  # generalisee) : on les calcule par defaut, sauf si l'appel dit pev =.
  dots <- list(...)
  if (!is.null(spl$X) && is.null(dots$pev)) dots$pev <- TRUE
  fit <- do.call(rx_fit, c(list(mod, backend = backend), dots))
  fit$model <- mod
  # Conserve de quoi predire : formule des effets fixes, donnees nettoyees et
  # niveaux des facteurs. Sans les xlevels, une prediction construite sur une
  # grille reduite n'aurait PAS les memes colonnes que X, et L beta melangerait
  # les coefficients sans rien signaler.
  fit$fixed   <- fixed
  fit$data    <- data
  fit$xlevels <- stats::.getXlevels(terms(fixed, data = data),
                                    model.frame(fixed, data, na.action = stats::na.pass))
  fit$call    <- match.call()
  fit$spl2d   <- if (!is.null(spl$X)) list(columns = colnames(spl$X), terms = unique(spl$groups),
                                             surfaces = spl$surfaces)
  fit
}
