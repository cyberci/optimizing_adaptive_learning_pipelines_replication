"""How does the descriptor-driven planner scale with the size of the candidate grid,
and how does its solution quality compare with fixed-budget TPE?"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import numpy as np, optuna
from itertools import product
import engine
optuna.logging.set_verbosity(optuna.logging.WARNING)

e = engine.Engine("V3")                    # direct-call runtime, no planner
m1, m2, meta = e.cache["medusa"]
rng = np.random.default_rng(11)
prev = np.array([10,3], np.float32)
TARGET = 0.7
BLOCKS = 5

def probs(ab, cand):
    B = cand.shape[0]
    x1 = np.concatenate([np.repeat(ab[None,:],B,0), cand],1).astype(np.float32)
    x2p = np.concatenate([np.repeat(prev[None,:],B,0), np.ones((B,1),np.float32)],1)
    a = np.asarray(m1(x1, training=False))[:,0]
    g = np.asarray(m2([x2p, cand.astype(np.float32)], training=False))[:,0]
    return 0.5*a + 0.5*g

# synthetic descriptors of growing grid size, always 2 tunable dims (model input is fixed)
SPECS = [(4,2),(8,3),(16,5),(32,10),(64,20),(128,40),(256,80),(512,160),(1024,320),(2048,640)]
rows=[]
for n1,n2 in SPECS:
    g1 = np.linspace(5,20,n1); g2 = np.linspace(1,5,n2)
    grid = np.array(list(product(g1,g2)), np.float32)
    G = grid.shape[0]
    ab = rng.normal(1,.15,10).astype(np.float32)
    # Shape-matched warm-up. A catalogued descriptor has a fixed grid size, so the
    # traced input shape is stable across requests and the steady-state cost is
    # the operative one; warming up at a different shape would charge the timed
    # call for a graph retrace it would not pay in service.
    probs(ab, grid)
    # Minimum over BLOCKS repeats. A single timed call is not a stable estimator
    # on a shared machine, and the crossover this sweep locates is read off the
    # point where two such curves meet.
    t_ex = float("inf")
    for _ in range(BLOCKS):
        t = time.perf_counter(); pr = probs(ab, grid)
        t_ex = min(t_ex, (time.perf_counter()-t)*1e3)
    err_ex = float(np.abs(pr-TARGET).min())

    def obj(tr):
        c=np.array([[tr.suggest_float("a",5,20), tr.suggest_float("b",1,5)]],np.float32)
        return abs(probs(ab,c)[0]-TARGET)
    t_tpe, err_tpe = float("inf"), None
    for r in range(BLOCKS):
        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=1))
        t = time.perf_counter(); s.optimize(obj, n_trials=20)
        t_tpe = min(t_tpe, (time.perf_counter()-t)*1e3)
        err_tpe = float(s.best_value)      # seeded, so identical across repeats
    rows.append(dict(grid=G, exhaustive_ms=round(t_ex,2), tpe20_ms=round(t_tpe,2),
                     err_exhaustive=round(err_ex,5), err_tpe20=round(err_tpe,5)))
    print(f"|G|={G:6d}  exhaustive={t_ex:8.1f} ms (err {err_ex:.4f})   TPE20={t_tpe:7.1f} ms (err {err_tpe:.4f})")
json.dump(rows, open(results("results_planner.json"),"w"), indent=2)
