# Builds the remlax Reference Manual: knitr runs the R chunks against the
# working tree of the package (figures in figures/, cache in cache/), then
# latexmk compiles the LaTeX with biber.
#
#   Rscript manual/build.R            # from the root of the repository
#   Rscript manual/build.R --clean    # drop the knitr cache first
#
# The PDF is a build product and is not versioned (see .gitignore).
args <- commandArgs(TRUE)
ici <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
setwd(ici)
if ("--clean" %in% args) unlink(c("cache", "figures"), recursive = TRUE)
knitr::knit("remlax-manual.Rnw", output = "remlax-manual.tex", quiet = TRUE)
rc <- system2("latexmk", c("-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", "remlax-manual.tex"))
if (rc != 0) stop("latexmk failed; see manual/remlax-manual.log")
message("manual/remlax-manual.pdf")
