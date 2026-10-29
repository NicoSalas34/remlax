# remlax Reference Manual

Sources of the reference manual, in LaTeX with knitr chunks. The chunks run
against the working tree of the package, so every figure and every printed
output comes from the code it documents.

## Build

From the root of the repository:

```
Rscript manual/build.R            # knitr, then latexmk with biber
Rscript manual/build.R --clean    # drop the knitr cache first
```

Requirements: a TeX distribution with pdfLaTeX, biber and latexmk; the R
packages knitr, pkgload and ggplot2; the Python solver of remlax for the
chapters that fit models (`rx_python_check()`).

The PDF, the generated `remlax-manual.tex`, `figures/` and `cache/` are build
products and are not versioned.

## Layout

- `remlax-manual.Rnw`: title page, contents, and the list of chapters.
- `preamble.tex`: packages, notation macros, the `syntax` and `caveat` boxes.
- `R/setup.R`: knitr options, loading of the package, figure theme.
- `chapters/`: one file per chapter and per appendix.
- `references.bib`: bibliography; each DOI is checked against CrossRef when an
  entry is added.
