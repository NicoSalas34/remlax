# Le paquet est une copie transformee des scripts vivants (R/*.R, src/remlax).
# Depuis le depot (variable REMLAX_REPO_ROOT, ou detection de ../../..), on
# verifie que rpkg/ est a jour ; installe seul, le test s'ignore.
test_that("rpkg/ est synchronise avec les scripts vivants", {
  racine <- Sys.getenv("REMLAX_REPO_ROOT", "")
  if (!nzchar(racine)) {
    cand <- normalizePath(file.path(testthat::test_path(), "..", "..", ".."), mustWork = FALSE)
    if (file.exists(file.path(cand, "R", "remlax.R")) &&
        file.exists(file.path(cand, "rpkg", "tools", "sync_sources.R"))) racine <- cand
  }
  if (!nzchar(racine)) skip("depot source introuvable (definir REMLAX_REPO_ROOT)")
  st <- system2(file.path(R.home("bin"), "Rscript"),
                c(shQuote(file.path(racine, "rpkg", "tools", "sync_sources.R")), "--check"),
                stdout = TRUE, stderr = TRUE)
  expect_identical(attr(st, "status"), NULL, info = paste(st, collapse = "\n"))
})
