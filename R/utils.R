# Value of `a`, or `b` when `a` is NULL (base R has it from 4.4.0 only).
`%||%` <- function(a, b) if (is.null(a)) b else a
