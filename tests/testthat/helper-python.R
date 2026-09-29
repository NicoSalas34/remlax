# Un Python porteur de jax est-il joignable ? Sinon les tests d'ajustement
# s'ignorent (skip) au lieu d'echouer : le paquet R doit pouvoir passer
# R CMD check sur une machine sans JAX, et dire clairement ce qu'il n'a pas
# pu verifier.
rx_python_a_jax <- function() {
  py <- tryCatch(rx_python_cmd(), error = function(e) NULL)
  if (is.null(py)) return(FALSE)
  ok <- suppressWarnings(try(system2(py[1], shQuote(c(py[-1], "-c", "import jax, numpy, scipy")),
                                     stdout = FALSE, stderr = FALSE), silent = TRUE))
  identical(as.integer(ok), 0L)
}
skip_si_pas_de_jax <- function() {
  if (!rx_python_a_jax())
    skip("aucun Python avec jax joignable (definir RX_PY)")
}
