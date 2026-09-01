# What the paired benchmark has established, and what is still running

This page is the working record of the measurement campaign of 2026-09-01. It
exists so that a finding is never carried in someone's head between the job that
produced it and the section that cites it. Every number below is read from a
CSV in `benchmarks/results/`; nothing here is an estimate.

## Why the protocol was rebuilt

Comparing **fit times** across backends is not a valid measurement. The two
backends follow different optimisation trajectories on the same model — 419
iterations on the card against 317 on the CPU, reaching the same optimum to
8.6e-10 relative — because floating-point addition is not associative: a
reduction is ordered differently on thousands of GPU threads than on sixteen CPU
cores, the gradient differs in its last bits, the line search accepts a
different step, and the paths diverge. Near a flat optimum the stopping test
then operates at the noise floor. A total time therefore mixes a hardware cost
with a path length settled by rounding noise.

The paired unit is an **objective evaluation at an imposed theta**, which does
identical arithmetic on both devices.

## Established

**The dense engine is indifferent to model structure.** At equal size and
structure, an identity relationship matrix and a dense genomic one cost the
same per evaluation: 0.01347 against 0.01364 s at n = 2000, and 0.23843 against
0.23817 s at n = 8000. Likewise the covariance family does not matter: at t = 6,
`iid` with 7 parameters and `us` with 27 both cost 0.2314 s. This is expected —
the engine forms and factorises V whatever the sparsity — and it is precisely
why it wins on dense models and loses on sparse ones.

**The parameter count acts on the number of evaluations, never on their price.**
Measured across 32 cells and three sizes.

**Evaluation cost is sub-cubic and becomes less so with size.** Per doubling of
n the factor is 3.8 then 5.1, against 8 for a pure Cholesky. The card is not
saturated at n = 2000, where fixed launch costs dominate, and approaches its
compute limit by n = 8000.

**Compilation is not a fixed cost.** It grows with n (0.375 s at n = 2000
against 1.393 s at n = 8000, so nearly linear) and with p (0.317 s at p = 3
against 0.445 s at p = 27). This weakens the amortisation argument behind the
expected CPU/GPU crossover, which must be redone on the numbers.

## The reconstruction test, and its failure

The cost model claims: compilation + evaluation count x unit cost = total. On
the 32 paired CPU cells at n = 2000 the predicted/measured ratio was **0.48** in
median — the model explained half the time. The cause was that the evaluation
count was PREDICTED as `n_iter + 2p`, whereas L-BFGS-B performs several
evaluations per iteration in its line search and each Newton polish step adds a
full Hessian. Measured on a case with p = 14 and 32 iterations: 60 predicted,
**102 actual**.

`fit_reml` now counts its objective calls and returns `n_eval`. With the real
count the reconstruction reaches **0.94**. The residual 6 % is Z assembly, the
initial theta and the BLUPs.

Counting rather than fitting a correction factor was deliberate: a factor of
1.70 would have closed the budget while hiding the missing item.

## The external test, and its negative result

ASReml's own sparse/dense partition produces **no measurable difference** on a
dense genomic relationship matrix, at any size tested. 24 valid measurements,
q from 500 to 4000 at n = 8000, three repetitions:

| q | sparse | dense | ratio | iterations | same logLik |
|---:|---:|---:|---:|:--:|:--:|
| 500 | 14.25 s | 14.71 s | 0.97 | 10 / 10 | yes |
| 1000 | 16.20 s | 16.07 s | 1.01 | 10 / 10 | yes |
| 2000 | 26.42 s | 25.31 s | 1.04 | 10 / 10 | yes |
| 4000 | 76.73 s | 74.86 s | 1.02 | 7 / 7 | yes |

This does **not** independently replicate our central claim, and the paper must
say so. What it does establish is that the partition changes only the solve
path: identical log-likelihoods and identical iteration counts throughout.

One caveat we cannot fully close: "the partition has no effect" and "the option
was not applied" would look the same here. The evidence that it did apply is
thin but real — the median prediction error variances are bit-identical at
q = 500 and 1000 but differ in the tenth digit at q = 2000 and 4000
(0.3489389175584 against 0.3489389176814), which is the signature of a
different arithmetic path rather than a no-op.

**Appendix A.1 does not explain remlax's missing PEVs.** The appendix says the
coefficient matrix inverse is only partially formed for sparse terms, so BLUP
variances are available only for the dense portion. Measured: prediction error
variances come back from BOTH paths, with the same median standard error to ten
digits. `predict()` evidently recomputes what it needs regardless of the
partition. The partial inverse must therefore NOT be offered as the structural
reason remlax does not expose PEVs — that absence is an implementation gap, and
should be written as one.

## Not established — do not cite

Nothing outstanding on the external test. The CPU/card paired comparison is
still running at the largest sizes.
