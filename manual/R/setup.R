# Common setup of the manual: knitr options, the remlax working tree, and the
# plotting helpers shared by the chapters. Sourced by the first chunk of
# remlax-manual.Rnw, with the working directory set to manual/ by build.R.

suppressPackageStartupMessages({
  library(knitr)
  library(ggplot2)
})

# the manual documents the working tree of the package, not an installed copy
suppressMessages(pkgload::load_all("..", quiet = TRUE, export_all = FALSE))
rx_version <- as.character(utils::packageVersion("remlax"))

opts_chunk$set(
  echo = TRUE, comment = "#>", collapse = FALSE, tidy = FALSE,
  fig.path = "figures/", cache.path = "cache/", cache = TRUE,
  dev = "pdf", fig.align = "center", fig.width = 6, fig.height = 3.6,
  out.width = "0.85\\linewidth", size = "small", message = FALSE, warning = FALSE
)
options(width = 78, digits = 4)

# code chunks in the colours of the manual
knit_hooks$set(document = function(x) sub("\\usepackage[]{color}", "", x, fixed = TRUE))

# theme of every figure
theme_rx <- theme_bw(base_size = 9) +
  theme(panel.grid.minor = element_blank(), strip.background = element_rect(fill = "grey95", colour = NA),
        legend.position = "bottom")
theme_set(theme_rx)
RX_BLUE <- "#1F4E79"
RX_AMBER <- "#B45309"

# a fit needs the Python solver; chapters that fit models test this first
rx_solver_ready <- tryCatch(isTRUE(rx_python_check(quiet = TRUE)$ok), error = function(e) FALSE)
