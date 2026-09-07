"""In-process concurrency measurement.

A FastAPI endpoint declared with `def` is dispatched to an anyio worker thread
pool, so a ThreadPoolExecutor over the same engine call reproduces the
server-side queueing behaviour. HTTP framing is excluded; the substitution is
validated against the HTTP measurement for V4 in loadtest.py.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, json, sys, time, threading
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import engine

DEADLINE_MS, INTERTASK_MS = 3000, 600
MODEL = "medusa_B_corrected"


def measure(variant, levels, per_worker):
    eng = engine.Engine(variant, desc={**engine.DESCRIPTOR, "id": MODEL})
    prev = {"number_of_medusas": 10, "time_limit": 3}
    for _ in range(2):
        eng.suggest(np.ones(10, np.float32), prev, True)
    rows = []
    for c in levels:
        rng = np.random.default_rng(99 + c)
        abs_ = [rng.normal(1, .15, 10).astype(np.float32) for _ in range(c * per_worker)]
        lat = []
        lock = threading.Lock()

        def task(ab):
            t = time.perf_counter()
            eng.suggest(ab, prev, True)
            d = (time.perf_counter() - t) * 1e3
            with lock:
                lat.append(d)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            list(ex.map(task, abs_))
        wall = time.perf_counter() - t0
        a = np.array(lat)
        r = dict(variant=variant, concurrency=c, n=len(a), wall_s=round(wall, 2),
                 throughput_rps=round(len(a) / wall, 2),
                 p50=round(float(np.percentile(a, 50)), 1),
                 p95=round(float(np.percentile(a, 95)), 1),
                 p99=round(float(np.percentile(a, 99)), 1),
                 within_600ms=float((a <= INTERTASK_MS).mean()),
                 within_3s=float((a <= DEADLINE_MS).mean()))
        rows.append(r)
        print(f"{variant} c={c:4d} n={r['n']:5d} thr={r['throughput_rps']:7.2f} rps  "
              f"p50={r['p50']:9.1f} p95={r['p95']:9.1f} ms  <=600ms {r['within_600ms']:6.1%}"
              f"  <=3s {r['within_3s']:6.1%}", flush=True)
    return rows


if __name__ == "__main__":
    out = {}
    out["V4"] = measure("V4", [1, 2, 4, 8, 16, 32, 64, 100], 8)
    out["V0"] = measure("V0", [1, 2, 4, 8, 16, 30], 2)
    json.dump(out, open(results("results_concurrency.json"), "w"), indent=2)
    print("done")
