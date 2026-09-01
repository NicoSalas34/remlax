# Benchmarks

## What was measured, and on what

All timings come from one SLURM cluster, on 2026-09-01, inside the apptainer
image `ige_reml.sif` (JAX 0.11.1, NumPy 2.5.2,
Python 3.12.3).

| | CPU | GPU |
|---|---|---|
| hardware | AMD EPYC, **8 cores allotted** of a 192-core node | **one MIG slice** of an A100 80GB PCIe: 10 GB, 7 of 108 SMs |
| threads | `OMP_NUM_THREADS=8`, verified by probe (see below) | - |
| raw data | `benchmarks/results/bench_cpu_*_2026-09-01.csv` | `benchmarks/results/bench_a100-mig-1g10gb_2026-09-01.csv` |

The full card was occupied by other work and could not be measured. **Every GPU
number on this page is therefore a pessimistic bound**, not a property of the
hardware: a slice exposes one seventh of the multiprocessors while paying the
same compilation cost. The paper says so wherever these numbers appear.

### A threading trap, and why it is documented rather than hidden

The first CPU run recorded `OMP_NUM_THREADS=1` inside the container although the
submission script exported 8, and achieved about 70 GFLOP/s on a dense
Cholesky - roughly two cores' worth of an eight-core allocation. SLURM's own
accounting put its CPU efficiency at 34.5%. `apptainer` does not pass a host
export into the container unless it is prefixed `APPTAINERENV_`.

The run was repeated with a probe that times one evaluation at n = 4000 under
each candidate setting and then runs the sweep under the fastest:

| setting | one evaluation at n = 4000 |
|---|---:|
| `OMP_NUM_THREADS=1` | 2.531 s |
| `OMP_NUM_THREADS=8` | **1.155 s** |

The single-thread figure matches the first run's 2.47 s to within 3%, which is
the direct evidence that it was effectively serial. Correcting it **halved the
apparent GPU advantage**: the per-evaluation ratio at n = 8000 fell from
14.2x to 5.9x. A benchmark whose baseline is accidentally serial does not
measure a solver, it measures a mistake.

## Growth in n

A single random factor plus an iid residual - the simplest model that exercises
the whole path. `fit` is a complete fit: initialisation, L-BFGS-B, Newton
polish. `eval` is one call of the objective and its gradient, compilation
excluded.

| n | fit CPU (s) | fit MIG (s) | gain | eval CPU (s) | eval MIG (s) | gain | iters |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.2599 | 0.3212 | 0.81 | 0.0030 | 0.0012 | 2.53 | 7 |
| 500 | 0.6334 | 0.3546 | 1.79 | 0.0116 | 0.0025 | 4.70 | 7 |
| 1000 | 2.1016 | 0.3980 | 5.28 | 0.0720 | 0.0080 | 8.98 | 7 |
| 2000 | 6.1289 | 1.2601 | 4.86 | 0.2847 | 0.0383 | 7.43 | 7 |
| 3000 | 14.0239 | 2.3137 | 6.06 | 0.6156 | 0.0988 | 6.23 | 7 |
| 4000 | 25.8020 | 6.3897 | 4.04 | 1.1849 | 0.2055 | 5.77 | 7 |
| 6000 | 70.6511 | 43.7662 | 1.61 | 3.3975 | 0.5877 | 5.78 | 7 |
| 8000 | 154.9669 | 94.1949 | 1.65 | 7.4594 | 1.2664 | 5.89 | 7 |

Two facts to read alongside these columns. The **iteration count is 7 at every
n on both backends**, and the restricted log-likelihoods agree to the last
printed digit (max |dlogL| = 0). So no part of the table below is explained by
one backend converging differently from the other.

### Three crossovers, not one

| definition | crossover |
|---|---|
| one objective call, compilation included | **n = 893** |
| one objective call, compilation amortised | below n = 250 - the slice wins at every size measured |
| a complete fit | **n = 301** |

### Compilation is nearly flat, and repaid almost immediately

Fitted on the log-log grid: XLA compilation grows as **n^0.48** on the CPU and
**n^0.40** on the slice, against **n^2.65** and **n^2.73** for one evaluation.
So compilation is not strictly constant in n, but it is asymptotically
negligible - which is the sense that matters.

Its median cost is 0.197 s on the CPU and 0.301 s on the slice. The slice's
extra 0.104 s is repaid after **1.6 evaluations at n = 1000** and immediately
above that. This corrects an expectation worth stating plainly: compilation is
*not* what keeps the GPU from paying off at moderate n.

### What does keep the slice from paying off

The per-evaluation ratio reaches 5.9x at n = 8000 while the whole-fit ratio is
only 1.65x, with the same 7 iterations on both sides. The gap is time the fit
spends outside objective evaluations, and it grows with n on the slice. That is
a measurement, not an explanation: the candidate causes - host-device
synchronisation at each L-BFGS-B iteration, transfer of the incidence matrices,
the spectral diagnostic - have not been separated, and profiling them is the
obvious next step rather than a conclusion to assert.

## Cost per structure of the catalogue

Fixed size, 1200 levels on a 40 x 30 grid, so the differences are the cost of
*building* V rather than of factorising it.

| structure | fit CPU (s) | fit MIG (s) | compile CPU (s) | params |
|---:|---:|---:|---:|---:|
| iid | 3.24 | 1.34 | 0.181 | 2 |
| us(3) | 11.02 | 1.55 | 0.241 | 9 |
| fa(4,2) | 51.49 | 4.67 | 0.255 | 15 |
| ar1 | 20.49 | 2.30 | 0.391 | 3 |
| ar1ar1 | 16.00 | 2.42 | 0.332 | 4 |
| iexp | 10.97 | 1.56 | 0.311 | 3 |
| igau | 10.71 | 0.95 | 0.269 | 3 |
| sph | 100.44 | 7.22 | 0.312 | 3 |
| mtrn | 1125.03 | OOM | -0.386 | 4 |
| mtrn_aniso | 172.99 | OOM | -0.450 | 6 |

The Matern kernels are the expensive ones, and on the slice they do not run at
all: the Bessel quadrature asks for a single 6.5 GB allocation, above what 10 GB
can give. They fit without difficulty on the CPU, and CPU/GPU parity does hold
for both at 160 levels (see [validation.md](validation.md)). This is a limit of
the slice, measured.

![Benchmarks]({/home/salas/.claude-science/orgs/272a30aa-a064-49b2-85e7-6680deef4d83/artifacts/proj_ff10b1313946/eee8f524-c58d-4484-a43d-44816c17f8c0/v79a0c968_bench_remlax.png})

## What this means in practice

1. **Below a few hundred observations, use the CPU.** A fit is a fraction of a
   second either way and the GPU adds a compilation to every new model shape.
2. **Between roughly 300 and a few thousand, the choice barely matters** on a
   MIG slice. It would matter on a full card, which was not measurable here.
3. **Above a few thousand, the GPU wins on arithmetic** - 5.9x per evaluation at
   n = 8000 - but only 1.65x on a whole fit on a slice, because of per-iteration
   overhead outside the objective.
4. **A one-seventh slice is close to not worth the detour** for a single fit.
   Its right use is running seven independent models, or seven restarts, at once.
5. **Avoid the Matern kernels on a small slice** and fit them on the CPU, or on
   a card with enough memory for the quadrature.

## Re-running the benchmarks

```sh
# CPU. The APPTAINERENV_ prefix is load-bearing inside a container.
APPTAINERENV_OMP_NUM_THREADS=8 APPTAINERENV_MKL_NUM_THREADS=8 \
  apptainer exec $SIF python3 benchmarks/bench.py --backend cpu --suite all \
  --ns 250,500,1000,2000,3000,4000,6000,8000 --reps 3 --reps-upto 4000 \
  --tag cpu-8c --out results/cpu.csv

# GPU
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  apptainer exec --nv $SIF python3 benchmarks/bench.py --backend gpu --suite all \
  --nmax 12000 --tag a100-mig --out results/gpu.csv
```

`--suite` takes any subset of `scaling`, `compile`, `params`, `structures`.
Every measurement is printed as it is taken, so a job killed by a walltime still
yields its data through the log - which is how the first CPU run survived.
