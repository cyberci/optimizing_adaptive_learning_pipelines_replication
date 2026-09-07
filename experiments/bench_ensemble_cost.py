"""Cost model of the full ensemble scoring call used by the planner.

Both networks are invoked for every candidate, so the quantity that governs the
search is the cost of one *ensemble* scoring call as a function of the number of
candidates it carries. We fit c(n) = c_fixed + n*c_unit over the regime the
catalogued descriptors occupy and again over the large-batch regime, and derive
the enumeration/search crossover from the fitted constants.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import engine

eng = engine.Engine("V3", desc={**engine.DESCRIPTOR, "id": "medusa_B_corrected"})
m1, m2, meta = eng.cache["medusa_B_corrected"]
PREV = np.array([10, 3], np.float32)
rng = np.random.default_rng(11)
AB = rng.normal(1, .15, 10).astype(np.float32)


def probs(cand):
    B = cand.shape[0]
    x1 = np.concatenate([np.repeat(AB[None, :], B, 0), cand], 1).astype(np.float32)
    x2p = np.concatenate([np.repeat(PREV[None, :], B, 0), np.ones((B, 1), np.float32)], 1)
    a = np.asarray(m1(x1, training=False))[:, 0]
    g = np.asarray(m2([x2p, cand.astype(np.float32)], training=False))[:, 0]
    return 0.5 * a + 0.5 * g


SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
REPS = {1: 40, 2: 40, 4: 40, 8: 40, 16: 40, 32: 40, 64: 30, 128: 30, 256: 20,
        512: 20, 1024: 15, 4096: 10, 16384: 6, 65536: 3}

BLOCKS = 5

rows = []
for n in SIZES:
    x = np.zeros((n, 2), np.float32)
    probs(x)                                # warm up at this exact shape
    # Minimum over BLOCKS timing blocks rather than the mean of one: on a shared
    # machine a co-tenant scheduled during a block inflates it, and the fitted
    # constants then move by more than the effect being measured.
    best = float("inf")
    for _ in range(BLOCKS):
        t = time.perf_counter()
        for _ in range(REPS[n]):
            probs(x)
        best = min(best, (time.perf_counter() - t) / REPS[n] * 1e3)
    rows.append((n, best))


def fit(rs):
    A = np.vstack([np.ones(len(rs)), np.array([r[0] for r in rs], float)]).T
    sol, *_ = np.linalg.lstsq(A, np.array([r[1] for r in rs], float), rcond=None)
    return float(sol[0]), float(sol[1])


small = [r for r in rows if r[0] <= 1024]
big = [r for r in rows if r[0] >= 1024]
cf, cu = fit(small)
cu_big = (big[-1][1] - big[0][1]) / (big[-1][0] - big[0][0])

for n, t in rows:
    print(f"  n={n:>6d}  {t:9.3f} ms   {t/n*1e3:9.2f} us/candidate")
print(f"\nc_fixed {cf:.3f} ms   c_unit(small) {cu*1e3:.3f} us   "
      f"c_unit(large) {cu_big*1e3:.3f} us   ratio {cf/cu:,.0f}")

C_BOOK = 35.1 / 20.0
for T in (20, 50, 100, 200):
    g = (T * (cf + cu + C_BOOK) - cf) / cu_big
    print(f"  analytic crossover T={T:3d}: |G|* = {int(g):,d}")

json.dump(dict(sizes=[r[0] for r in rows], times=[round(r[1], 4) for r in rows],
               c_fixed=round(cf, 4), c_unit_small=round(cu, 7),
               c_unit_large=round(cu_big, 7),
               crossover={T: int((T * (cf + cu + C_BOOK) - cf) / cu_big)
                          for T in (20, 50, 100, 200)}),
          open(results("results_ensemble_cost.json"), "w"), indent=2)
print("\nwrote results_ensemble_cost.json")
