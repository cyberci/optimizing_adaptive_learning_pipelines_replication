"""Re-measures the enumeration arms with a shape-matched warm-up.

In the first pass the exhaustive arm was warmed up with a single candidate and
then timed on the full grid, so the timed call also paid for a graph retrace at
the new input shape. A catalogued descriptor has a fixed grid size, so the shape
is stable across requests and the steady-state cost is the operative one. Every
enumeration measurement below is therefore preceded by a warm-up at the identical
shape; the search arms are unaffected because they always score one candidate.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os, time, tracemalloc, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, optuna
from itertools import product
import engine
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = 0.7
eng = engine.Engine("V3", desc={**engine.DESCRIPTOR, "id": "medusa_B_corrected"})
m1, m2, meta = eng.cache["medusa_B_corrected"]
rng = np.random.default_rng(23)
PREV = np.array([10, 3], np.float32)
CHUNK = 65536


def probs(ab, cand):
    B = cand.shape[0]
    x1 = np.concatenate([np.repeat(ab[None, :], B, 0), cand], 1).astype(np.float32)
    x2p = np.concatenate([np.repeat(PREV[None, :], B, 0), np.ones((B, 1), np.float32)], 1)
    a = np.asarray(m1(x1, training=False))[:, 0]
    g = np.asarray(m2([x2p, cand.astype(np.float32)], training=False))[:, 0]
    return 0.5 * a + 0.5 * g


def chunked(ab, grid):
    best = 1.0
    for i in range(0, len(grid), CHUNK):
        best = min(best, float(np.abs(probs(ab, grid[i:i + CHUNK]) - TARGET).min()))
    return best


# ------------------------------------------------- 1 enumeration reference
def exhaustive_reference(n_rep=8):
    grid = np.array(list(product(np.linspace(5, 20, 400), np.linspace(1, 5, 200))), np.float32)
    ts, es = [], []
    for k in range(n_rep):
        ab = rng.normal(1, .15, 10).astype(np.float32)
        probs(ab, grid)                       # shape-matched warm-up
        t = time.perf_counter()
        p = probs(ab, grid)
        ts.append((time.perf_counter() - t) * 1e3)
        es.append(float(np.abs(p - TARGET).min()))
    print(f"  exhaustive |G|={len(grid)}  {np.mean(ts):8.1f} ms   err {np.mean(es):.6f}")
    return dict(sampler="exhaustive batched", trials=int(len(grid)),
                ms=float(np.mean(ts)), err=float(np.mean(es)), err_sd=float(np.std(es)))


# --------------------------------------------------------- 2 large grids
def large_grids():
    rows = []
    ab = rng.normal(1, .15, 10).astype(np.float32)
    for G in (10 ** 5, 10 ** 6, 2_457_000):
        g1 = np.linspace(5, 20, int(np.sqrt(G)))
        g2 = np.linspace(1, 5, int(np.ceil(G / len(g1))))
        grid = np.array(list(product(g1, g2)), np.float32)[:G]

        probs(ab, grid)
        tracemalloc.start()
        t = time.perf_counter(); p = probs(ab, grid); t_full = (time.perf_counter() - t) * 1e3
        peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
        e_full = float(np.abs(p - TARGET).min())
        del p

        chunked(ab, grid)
        tracemalloc.start()
        t = time.perf_counter(); e_chunk = chunked(ab, grid)
        t_chunk = (time.perf_counter() - t) * 1e3
        peak_chunk = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()

        # stratified sample, then a local sweep around the incumbent
        idx = rng.choice(len(grid), size=4096, replace=False)
        sub = grid[idx]
        probs(ab, sub)
        c0 = sub[0]
        loc0 = np.array(list(product(np.linspace(c0[0] - .5, c0[0] + .5, 64),
                                     np.linspace(c0[1] - .2, c0[1] + .2, 64))), np.float32)
        probs(ab, loc0)
        tracemalloc.start()
        t = time.perf_counter()
        ps = probs(ab, sub)
        j = int(np.argmin(np.abs(ps - TARGET)))
        c = sub[j]
        loc = np.array(list(product(np.linspace(c[0] - .5, c[0] + .5, 64),
                                    np.linspace(c[1] - .2, c[1] + .2, 64))), np.float32)
        pl = probs(ab, loc)
        e_strat = float(min(np.abs(ps - TARGET).min(), np.abs(pl - TARGET).min()))
        t_strat = (time.perf_counter() - t) * 1e3
        peak_strat = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()

        def obj(tr):
            c = np.array([[tr.suggest_float("a", 5, 20), tr.suggest_float("b", 1, 5)]], np.float32)
            return float(abs(probs(ab, c)[0] - TARGET))
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=1))
        probs(ab, grid[:1])
        t = time.perf_counter(); s.optimize(obj, n_trials=20)
        t_tpe = (time.perf_counter() - t) * 1e3

        rows.append(dict(grid=int(len(grid)), full_ms=round(t_full, 1), full_mb=round(peak, 1),
                         full_err=e_full, chunk_ms=round(t_chunk, 1),
                         chunk_mb=round(peak_chunk, 1), chunk_err=e_chunk,
                         strat_ms=round(t_strat, 1), strat_mb=round(peak_strat, 1),
                         strat_err=e_strat,
                         tpe20_ms=round(t_tpe, 1), tpe20_err=float(s.best_value)))
        print(f"  |G|={len(grid):>9,d}  full {t_full:9.1f} ms /{peak:7.1f} MB (err {e_full:.6f}) | "
              f"chunk {t_chunk:8.1f} ms /{peak_chunk:6.1f} MB (err {e_chunk:.6f}) | "
              f"strat {t_strat:7.1f} ms /{peak_strat:5.1f} MB (err {e_strat:.6f}) | "
              f"TPE20 {t_tpe:6.1f} ms (err {s.best_value:.6f})", flush=True)
    return rows


if __name__ == "__main__":
    print("1  enumeration reference with shape-matched warm-up")
    a = exhaustive_reference()
    print("\n2  grids beyond the crossover")
    b = large_grids()
    json.dump(dict(exhaustive=a, large_grids=b),
              open(results("results_baselines_warm.json"), "w"), indent=2)
    print("\nwrote results_baselines_warm.json")
