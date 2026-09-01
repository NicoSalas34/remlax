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

## Not established — do not cite

**ASReml's sparse/dense partition.** The first run showed no dense advantage at
q <= 1000 (ratios 0.61, 1.03, 1.10) and the fits were 0.07 to 1.7 s, too small
for the manual's claim, which concerns relationship matrices "of the order of
several thousand". Rerunning to q = 4000.

**Prediction error variances.** Appendix A.1 says the coefficient matrix inverse
is only partially formed for sparse terms, so BLUP variances are available only
for the dense portion. The measurement contradicts this: the sparse path
returned a finite median standard error of 0.2002. Either `predict()` recomputes
on demand independently of the partition, or the term was not actually moved.
Until this is settled, the partial inverse must NOT be offered as the structural
explanation for remlax not exposing PEVs.

**ASReml iteration counts.** The first run read 1 everywhere because the length
of the log-likelihood vector was taken for an iteration history. The history is
in the object's `trace` field, one column per iteration.
