# What must be true before this solver is publishable

This page is the checklist the validation and benchmark suites answer to. It
exists because "the tests pass" is not a claim anybody can check: a reader needs
to know *which* claim each test defends, and what a failure would have meant.

Every row names the file that produces the evidence. Nothing here is a plan for
code that does not exist; where a suite has not run yet on the reference machine,
the row says so.

## 1. The dense engine computes REML

| Claim | Evidence | Status |
|---|---|---|
| Internal algebra is self-consistent | `tests/python/test_remlax_core.py` | 25 checks, passing |
| No configuration crashes or yields a non-PSD covariance | `tests/python/stress_remlax.py` | 150 random configurations, passing |
| Agrees with `lme4` and `sommer` where their models overlap | `tests/R/test_remlax.R` | 31 checks, passing |
| Agrees with `asreml`, including Kenward–Roger against `pbkrtest` | `tests/R/test_remlax_asreml.R` | 50 checks, passing |
| Recovers known truth on the full indirect-effects model | `tests/R/test_remlax_ige.R` | 11 checks, passing |
| CPU and GPU give the same answer | `tests/python/parite_gpu.py` | 34 checks, max relative gap 6.4e-15 |

## 2. The sparse engine computes the *same* REML

This is the part where a defect is easiest to miss, because a wrong likelihood
can leave the parameter estimates correct.

| Claim | Why it can fail silently | Evidence |
|---|---|---|
| The parameterisation is read identically by both engines | A permuted or rescaled θ gives a plausible fit at the wrong point | `tests/R/test_remlax_tmb_parity.R` — 30 checks, including K⁻¹K = I by product rather than by inspecting the sparsity pattern |
| The closed-form sparse precisions really invert the dense correlations | A tridiagonal matrix can look right and not be the inverse | same file |
| Both engines evaluate the same function | Comparing two optima only shows where each stopped | `tests/R/test_remlax_tmb_vs_dense.R` — 16 checks, warm start in BOTH directions |
| A term declared by its precision gives the same likelihood as the same term declared by its relationship matrix | A missing constant shifts the likelihood without touching the gradient | `tests/R/test_remlax_tmb_prec.R` — 9 checks, arbitrated by an independent dense-algebra REML |

Three defects were found this way, and two of them were in code that had already
passed a test on estimates:

1. a likelihood-convention mismatch, which showed up as a ratio of exactly
   2.0000000000 across three models — a constant factor is a convention, never a
   formulation difference;
2. a warm start in the **dense** engine that did not evaluate at the imposed θ
   but drifted toward the optimum, which invalidates precisely the comparison the
   warm start exists to make;
3. a missing `log|K|` in the sparse engine's precision path, a *constant* offset:
   θ agreed to five digits and only the likelihood was wrong — so every model
   comparison, likelihood-ratio test and information criterion was wrong too.

The lesson worth carrying: two engines that agree prove less than one engine
checked against an independent computation. The arbiter is a dense-algebra REML
written out longhand in the test file, and it settled all three cases.

### Known limit, measured rather than assumed

The mixed model equations contain R⁻¹. As the residual variance goes to zero the
coefficient matrix becomes ill-conditioned and the sparse engine loses accuracy,
where the dense engine — working on V — does not. Imposing θ rather than hoping
the optimiser visits that region:

| σ²_e | 3.4e-04 | 6.1e-06 | 1.1e-07 | 2.1e-09 | 3.8e-11 |
|---|---:|---:|---:|---:|---:|
| likelihood gap | 3.8e-11 | 1.3e-09 | 1.2e-07 | 5.8e-07 | 1.7e-04 |

This is structural, not a defect to fix, and it is why `asreml` parameterises the
variance *ratio*.

## 3. The decision grid

The question the grid answers is not "which engine is faster" but "which engine
for which model", and the answer is decided by structure, not size.

|  | structurally sparse case | structurally dense case |
|---|---|---|
| dense engine, CPU | ✓ | ✓ |
| dense engine, GPU | ✓ | ✓ |
| sparse engine, CPU | ✓ | ✓ (forced, outside the declared scope) |
| sparse engine, GPU | — does not exist | — does not exist |

**Six cells, not eight.** The sparse engine factorises through CHOLMOD, which is
CPU-only. That empty cell is a result: it makes "sparse" and "GPU" mutually
exclusive, so the choice is a trade-off rather than two independent axes.

The most informative cell is the structurally dense case *forced* onto the sparse
engine. The declared scope excludes it on principle — a genomic relationship
matrix has a full inverse — and forcing it is the quantitative answer to "why not
use sparse everywhere". It is reachable through the `Kinv` slot.

Produced by `benchmarks/bench_engines.R`. Two measurement traps it handles
explicitly:

- **Out-of-process cost.** The dense engine is invoked through a CLI, so every
  evaluation pays interpreter start-up and XLA compilation — measured at ~2.9 s
  against 0.07 s for an in-process sparse evaluation. The grid records the
  solver's own internal timing *and* wall time in separate columns; the gap
  between them is the invocation cost, which matters on its own since a
  model-comparison loop pays it on every call.
- **Slice versus card.** A 1/7 MIG slice exposes 7 of the A100's 108 SMs, so
  every GPU figure obtained on a slice is a pessimistic bound. The factor must be
  measured on a whole card, not extrapolated: extrapolating a slice linearly
  exceeds the card's advertised peak. That reference run waits in the queue for
  the full card, and until it lands every GPU number here is labelled as a bound.

## 4. What calibration the decision tool still needs

`remlax.sparsity` reports the fill-in exactly and attributes it — by
counterfactual re-factorisation rather than a density threshold — but it converts
that into a *flop* ratio, and a flop ratio is not a time ratio. It is absurdly
optimistic for the sparse route at small sizes. The grid above supplies the
coefficients that turn it into a time prediction; until those are fitted, the
tool reports structure and cost, never a recommended engine.
