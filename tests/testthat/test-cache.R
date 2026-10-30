test_that("le cache va dans tempdir pendant un check ou sur demande", {
  withr_env <- function(vars, code) {
    old <- Sys.getenv(names(vars), unset = NA, names = TRUE)
    on.exit(for (k in names(old))
      if (is.na(old[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv, stats::setNames(list(old[[k]]), k)))
    do.call(Sys.setenv, as.list(vars))
    force(code)
  }
  Sys.unsetenv("REMLAX_CACHE")
  withr_env(c(`_R_CHECK_PACKAGE_NAME_` = "remlax"),
            expect_equal(remlax:::.rx_cache_dir(), file.path(tempdir(), "remlax_cache")))
  withr_env(c(REMLAX_CACHE = "/chemin/impose"),
            expect_equal(remlax:::.rx_cache_dir(), "/chemin/impose"))
  op <- options(remlax.cache = FALSE); on.exit(options(op), add = TRUE)
  expect_equal(remlax:::.rx_cache_dir(), file.path(tempdir(), "remlax_cache"))
})

test_that("la purge retire les fichiers anciens et garde les recents", {
  d <- file.path(tempdir(), "test_purge_cache")
  dir.create(d, showWarnings = FALSE)
  vieux <- file.path(d, "vieux"); neuf <- file.path(d, "neuf")
  writeLines("a", vieux); writeLines("b", neuf)
  Sys.setFileTime(vieux, Sys.time() - 40 * 86400)
  expect_equal(remlax:::.rx_cache_prune(d, days = 30), 1L)
  expect_false(file.exists(vieux))
  expect_true(file.exists(neuf))
  unlink(d, recursive = TRUE)
})
