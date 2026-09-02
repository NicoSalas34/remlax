# Quelle part du temps d'ajustement creux est le rapport d'ecarts-types ?
# L'enjeu est une ASYMETRIE DE COMPARAISON : le banc appelle le moteur dense
# avec hessian = FALSE et blups = FALSE, alors que rx_fit_sparse calcule
# sdreport() systematiquement. Si cette part est grande, le moteur creux fait un
# travail que le dense ne fait pas, et l'ecart de temps mesure le sous-estime.
suppressMessages({source("R/remlax.R"); source("R/remlax_tmb.R")})
src <- readLines("benchmarks/bench_engines.R")
i <- grep("^cas_traits <- function", src); j <- grep("^lignes <- list", src)[1]
eval(parse(text = paste(src[i:(j - 1L)], collapse = "\n")))
a <- commandArgs(TRUE)
arg <- function(k, d) { i <- match(paste0("--", k), a); if (is.na(i)) d else a[i + 1L] }
NU <- as.integer(arg("n-unit", "800")); QT <- as.integer(arg("q-trait", "150"))
cat(sprintf("%3s %7s %8s %10s %10s %9s\n", "t", "n", "q_total", "avec sdrep", "sans", "part"))
for (t in as.integer(strsplit(arg("ts", "2,4,6"), ",")[[1]])) {
  set.seed(77L)
  d  <- cas_traits(NU, t, QT, K = NULL)
  ms <- rx_model(d$y, d$X, terms = d$terms_s, residual = d$residual, name = "s")
  f1 <- rx_fit_sparse(ms, maxiter = 3000L, verbose = FALSE)
  f2 <- rx_fit_sparse(ms, maxiter = 3000L, verbose = FALSE, sdreport = FALSE)
  cat(sprintf("%3d %7d %8d %10.3f %10.3f %8.0f%%\n", t, d$n, QT * t,
              f1$secondes, f2$secondes, 100 * (f1$secondes - f2$secondes) / f1$secondes))
}
