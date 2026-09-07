# The measurement environment

## What the numbers depend on

The article makes two kinds of claim, and they do not depend on the environment
in the same way.

**Domain-metric claims** — the oracle bound of 93.7 %, the 20.0 % of the
pipeline as implemented, the 35.8 % and 88.6 % of the two corrections, the
degradation curve — depend on the machine only through the trained networks.
Reusing the shipped model artefacts (`./run_all.sh bench`) reproduces them
exactly, digit for digit. Retraining first (`./run_all.sh all`) does not:
TensorFlow training is not bit-reproducible even at a fixed seed, so the
retrained networks differ slightly and every number that depends on them moves
by a point or two. A deviation larger than the reported confidence interval is a
bug, not a hardware difference.

**Timing claims** — the latency of the five variants, the fitted cost model, the
measured crossover, the concurrency sweep — are properties of a machine, and the
article's own argument is that they must be calibrated rather than assumed. They
will not reproduce numerically on different hardware. What should reproduce is
their *shape*: the fixed per-invocation cost dominating the per-candidate cost by
three orders of magnitude, the batched planner beating twenty sequential
invocations by roughly two orders of magnitude, and the crossover landing where
Eq. (1) predicts from the constants fitted on that machine.

If you want to check the crossover claim on your own hardware, run
`bench_ensemble_cost.py` and `bench_runtimes.py` first, read the fitted
`c_fixed` and `c_unit` out of the result JSON, and compare the predicted
crossover with what `bench_planner.py` measures. That comparison is
machine-independent even though the constants are not.

## The machine used for the reported measurements

| | |
|---|---|
| CPU | Intel Xeon @ 2.80 GHz, **2 vCPU** |
| Memory | 8 GB |
| Operating system | Ubuntu 24.04.4 LTS, kernel 6.18 |
| Python | 3.11.15 |
| Accelerator | none; TensorFlow CPU build throughout |
| Threads | TensorFlow left at its defaults |

Two vCPU is not an accident of convenience. The concurrency results are reported
against that budget because it is the scale at which the platform is intended to
run, and the whole point of Section 5.5 is what a cohort-sized burst does to a
single modest server. Running the concurrency sweep on a large machine will
raise every number and destroy the comparison with the two deadlines.

## Package versions

`requirements.txt` pins the exact versions used. Later versions of NumPy,
SciPy, pandas, scikit-learn and Matplotlib have no effect on any result. The
TensorFlow version does matter for the timings, because the fixed
per-invocation cost that the article measures is a property of the framework's
dispatch path, not of the arithmetic — that is the finding of Section 5.4, and
`bench_runtimes.py` demonstrates it by re-measuring the same forward pass
through ONNX Runtime and a plain NumPy implementation.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The ONNX packages are needed only for two of the four arms of
`bench_runtimes.py`, and FastAPI, Uvicorn and httpx only for the optional HTTP
cross-check. `python3 tools/check.py` reports which of these are missing and
what they would skip.

## Sources of run-to-run variation

**Load.** The three scripts that fit a cost model —
`bench_ensemble_cost.py`, `bench_runtimes.py` and `bench_planner.py` — take the
*minimum* over five timing blocks rather than the mean of one. That matters more
than it sounds: with the mean of a single block, repeated runs of
`bench_runtimes.py` on the same machine disagreed on the fitted marginal cost by
a factor of six, because a co-tenant scheduled during one block inflates it. The
minimum block is the standard robust estimator for a microbenchmark whose true
cost is a floor with additive noise above it, and with it the fitted crossovers
repeat to within about 20 %. `bench_latency.py` and the concurrency sweep report
medians and percentiles over many requests instead. None of this compensates for
a uniformly busy machine: run on an idle host, and prefer a dedicated instance
over a burstable shared one.

**Graph retracing.** TensorFlow re-traces the computation graph when the input
shape changes, and a timed call that also pays for a retrace can be several
times slower than the steady state. Every enumeration measurement here is
preceded by a warm-up at the *identical* batch shape. This matters: an earlier
version of these measurements warmed up at shape (1, ·) and timed at the full
grid, which inflated the enumeration arm and moved the apparent crossover by
about an order of magnitude. If you adapt these scripts, keep the shape-matched
warm-up.

**Seeds and sampling.** The learner population is drawn from a fixed seed, so
the domain-metric numbers are reproducible. Different runs of the tier
comparison and the degradation sweep draw independent learner samples: the two
full-availability points, 31.1 % and 35.8 %, differ by about two confidence
interval widths across seeds, and the article says so.

**Memory.** `bench_baselines_warm.py` measures peak allocation at grid sizes up to
2 457 000 candidates, and the full-enumeration arm allocates about 253 MB there.
`bench_planner.py` was run without the largest 4096 × 1280 specification, which
needs more memory than the 2 vCPU machine has.

## Expected runtime

See the table in `README.md`. On the machine above the full `./run_all.sh` takes
about 45 minutes, dominated by the twelve conditions of the ensemble-weight
sweep (13 min) and the concurrency sweep (7 min). `./run_all.sh bench` reuses
the shipped model artefacts and skips the five-minute training stage.
