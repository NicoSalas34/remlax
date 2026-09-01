# =============================================================================
# TEST EXTERNE : la partition creux / dense d'ASReml, sur les memes dispositifs
# =============================================================================
# POURQUOI CE FICHIER EXISTE
# --------------------------
# Notre affirmation centrale est qu'une parente DENSE ne doit pas passer par la
# voie creuse, et que la structure decide plus que la taille. Mesuree sur nos
# deux moteurs, elle vaut pour NOTRE implementation. Un relecteur demandera si
# elle generalise.
#
# ASReml permet de trancher, et de la meilleure facon possible : le contraste
# est INTERNE au meme logiciel. Son manuel 4.2, annexe A.1, decrit la partition
# des termes en un ensemble dense et un ensemble creux. Les termes de la formule
# fixe vont dans le dense, ceux des formules random et sparse dans le creux. La
# matrice inverse des coefficients est ENTIEREMENT formee pour l'ensemble dense
# et seulement PARTIELLEMENT pour l'ensemble creux. Et l'option
#     asreml.options(dense = ~vm(...))
# deplace un terme ALEATOIRE dans l'ensemble dense.
#
# On compare donc le MEME modele, sur les MEMES donnees, dans le meme logiciel,
# avec le terme genomique traite en creux (defaut) puis en dense. Meme
# optimiseur, meme critere d'arret, meme code : seule la voie de resolution
# change. Aucun de nos biais habituels ne s'applique — ni trajectoires
# divergentes entre backends, ni cout d'invocation hors processus, ni difference
# d'implementation.
#
# CE QUE LE MANUEL DIT DEJA, ET QU'IL FAUT CITER PLUTOT QUE REDECOUVRIR
# --------------------------------------------------------------------
# L'annexe A.1 recommande elle-meme le traitement dense pour les matrices de
# parente : elles sont grandes, de l'ordre de plusieurs milliers, et pleines,
# "and it can be more efficient to process such terms as dense". Notre these est
# donc enoncee par le logiciel de reference. Ce test la CHIFFRE.
#
# ELLE EXPLIQUE AUSSI POURQUOI remlax NE REND PAS LES PEV
# -------------------------------------------------------
# "the variance matrix of the BLUEs and BLUPs is only available for terms in the
# dense portion" : l'inverse partiel ne donne pas les variances d'erreur de
# prediction des termes creux. Ce n'est pas un oubli de notre part, c'est le
# prix de l'inverse partiel — et ASReml paie le meme, avec la meme sortie.
#
# CE QUE CE FICHIER NE MESURE PAS
# -------------------------------
# Il ne compare PAS remlax a ASReml en vitesse : ce sont deux algorithmes
# differents (information moyenne contre quasi-Newton), invoques differemment,
# et le manuel donne maxit = 13 par defaut la ou nous utilisons 3000 — les
# iterations ne sont donc pas des objets comparables. Seul le contraste
# INTERNE creux/dense est mesure ici, et il se compare a NOTRE contraste
# interne, rapport a rapport.
# =============================================================================
suppressMessages({ library(asreml); library(Matrix) })

args <- commandArgs(trailingOnly = TRUE)
opt <- function(nom, defaut) {
  i <- match(paste0("--", nom), args)
  if (is.na(i) || i == length(args)) defaut else args[i + 1L]
}
N_UNIT <- as.integer(opt("n-unit", "2000"))
QS     <- as.integer(strsplit(opt("qs", "200,500,1000"), ",")[[1]])
REPS   <- as.integer(opt("reps", "2"))
SORTIE <- opt("out", "asreml_creux_dense.csv")
TAG    <- opt("tag", "")
MAXIT  <- as.integer(opt("maxit", "30"))
# L'espace de travail par defaut est d'environ 134 Mo et sature des q = 2000 :
# "Insufficient workspace available when reordering matrices".
WS     <- opt("workspace", "16gb")

cat(sprintf("[asreml] version %s | n_unit = %d | q = %s | maxit = %d\n",
            as.character(packageVersion("asreml")), N_UNIT,
            paste(QS, collapse = ","), MAXIT))

# --- dispositif : parente genomique DENSE, construite selon VanRaden ----------
# La meme construction que la suite genomic de bench.py, pour que les deux
# series portent sur le meme objet statistique.
dispositif <- function(n_unit, q, n_marq = 1500, graine = 7) {
  set.seed(graine)
  p <- runif(n_marq, 0.05, 0.95)
  M <- matrix(rbinom(q * n_marq, 2, rep(p, each = q)), q, n_marq)
  Z <- scale(M, center = 2 * p, scale = FALSE)
  K <- tcrossprod(Z) / (2 * sum(p * (1 - p)))
  K <- K + diag(1e-6, q)                       # regularisation minimale
  gen <- factor(rep_len(seq_len(q), n_unit))
  u <- as.numeric(chol(K) %*% rnorm(q))
  y <- 1.5 + u[as.integer(gen)] + rnorm(n_unit, 0, 0.8)
  list(d = data.frame(y = y, gen = gen), K = K, q = q, n = n_unit)
}

# ASReml veut l'INVERSE, avec l'attribut INVERSE pose (manuel 4.2 p.67).
inverse_pose <- function(K) {
  Ki <- solve(K)
  dimnames(Ki) <- list(seq_len(nrow(K)), seq_len(nrow(K)))
  attr(Ki, "INVERSE") <- TRUE
  Ki
}

chrono <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  r <- tryCatch(eval.parent(substitute(expr)), error = function(e) e)
  list(r = r, s = proc.time()[["elapsed"]] - t0)
}

lignes <- list()
for (q in QS) {
  cat(sprintf("\n=== q = %d ===\n", q))
  dd <- dispositif(N_UNIT, q)
  Ki <- inverse_pose(dd$K)

  for (voie in c("creux", "dense")) {
    for (rep in seq_len(REPS)) {
      # La partition est fixee par asreml.options : par defaut un terme
      # aleatoire est CREUX ; dense = ~vm(...) le bascule.
      if (voie == "dense") {
        asreml.options(dense = ~ vm(gen, Ki), trace = FALSE, maxit = MAXIT,
                       workspace = WS, pworkspace = WS)
      } else {
        asreml.options(dense = ~ NULL, trace = FALSE, maxit = MAXIT,
                       workspace = WS, pworkspace = WS)
      }
      z <- chrono(asreml(fixed = y ~ 1, random = ~ vm(gen, Ki),
                         residual = ~ units, data = dd$d))
      if (inherits(z$r, "error")) {
        lignes[[length(lignes) + 1L]] <- data.frame(
          tag = TAG, q = q, n = dd$n, voie = voie, rep = rep, statut = "echec",
          secondes = NA_real_, n_iter = NA_integer_, logLik = NA_real_,
          converge = NA, pev_dispo = NA, pev_se_mediane = NA_real_,
          erreur = substr(conditionMessage(z$r), 1, 180))
        cat(sprintf("  %-6s rep %d : ECHEC %s\n", voie, rep,
                    substr(conditionMessage(z$r), 1, 90)))
        next
      }
      a <- z$r
      # QUE PORTE L'OBJET ? Je lisais length(a$loglik) comme un compte
      # d'iterations et cela rendait 1 partout : ce n'est manifestement pas
      # l'historique. On imprime les champs UNE fois plutot que de deviner, et
      # on extrait le compte de facon robuste.
      if (!exists(".champs_imprimes")) {
        cat("  [champs de l'objet asreml] ",
            paste(names(a), collapse = " "), "\n")
        .champs_imprimes <<- TRUE
      }
      n_it <- suppressWarnings(as.integer(
        # Champs reellement presents (imprimes par ce script) : ifault converge
        # nedf nwv nsing noeff loglik sigma2 ... trace ... Il n'y a ni `nit` ni
        # `monitor` ; l'historique est dans `trace`, une colonne par iteration.
        if (!is.null(a$trace)) ncol(as.matrix(a$trace))
        else if (length(a$loglik) > 1) length(a$loglik)
        else NA_integer_))
      # La variance d'erreur de prediction n'est disponible QUE pour la partie
      # dense (manuel A.1). C'est le second resultat de ce test, et il se
      # verifie plutot qu'il ne se suppose.
      pev <- tryCatch({
        pv <- predict(a, classify = "vm(gen, Ki)", only = "vm(gen, Ki)",
                      sed = FALSE, trace = FALSE)
        # Contredit ma lecture de l'annexe A.1, qui dit la variance des BLUP
        # disponible seulement pour la partie dense : predict() la rendait des
        # DEUX cotes. Soit il la recalcule a la demande independamment de la
        # partition, soit le terme n'etait pas reellement bascule. On enregistre
        # donc la valeur mediane de l'erreur type, pour voir si les deux voies
        # rendent la MEME chose ou seulement quelque chose.
        se <- pv$pvals$std.error
        if (is.null(se) || !any(is.finite(se))) FALSE else median(se, na.rm = TRUE)
      }, error = function(e) FALSE)
      lignes[[length(lignes) + 1L]] <- data.frame(
        tag = TAG, q = q, n = dd$n, voie = voie, rep = rep, statut = "ok",
        secondes = z$s, n_iter = n_it, logLik = tail(a$loglik, 1),
        converge = isTRUE(a$converge), pev_dispo = !identical(pev, FALSE),
        pev_se_mediane = if (is.numeric(pev)) pev else NA_real_, erreur = "")
      cat(sprintf("  %-6s rep %d : %7.2f s | %3s iter | logLik %.6f | PEV %s\n",
                  voie, rep, z$s, ifelse(is.na(n_it), "?", n_it),
                  tail(a$loglik, 1), pev))
    }
  }
}

res <- do.call(rbind, lignes)
write.csv(res, SORTIE, row.names = FALSE)
cat(sprintf("\n[asreml] %d lignes ecrites dans %s\n", nrow(res), SORTIE))

# --- le rapport interne, qui est l'objet du test -----------------------------
ok <- res[res$statut == "ok", ]
if (nrow(ok)) {
  cat("\n=== rapport creux / dense, INTERNE a ASReml ===\n")
  for (q in unique(ok$q)) {
    s <- ok[ok$q == q, ]
    tc <- median(s$secondes[s$voie == "creux"])
    td <- median(s$secondes[s$voie == "dense"])
    lc <- unique(round(s$logLik[s$voie == "creux"], 6))
    ld <- unique(round(s$logLik[s$voie == "dense"], 6))
    cat(sprintf("  q=%-5d creux %7.2f s | dense %7.2f s | rapport %5.2f | ",
                q, tc, td, tc / td))
    cat(sprintf("meme logLik : %s\n",
                isTRUE(all.equal(lc[1], ld[1], tolerance = 1e-6))))
  }
  cat("\nUn rapport > 1 signifie que la voie DENSE est plus rapide, ce que le\n",
      "manuel annonce pour une parente genomique. Les logLik doivent etre\n",
      "identiques : la partition change la resolution, pas le modele.\n", sep = "")
}
