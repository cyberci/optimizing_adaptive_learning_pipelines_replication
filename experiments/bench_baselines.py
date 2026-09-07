"""Search baselines, large grids and a non-monotone objective.

  1  Model-based baselines: TPE at several budgets, multivariate TPE, CMA-ES and
     random search, all scored through the identical inference path.
  2  Strategies for grids beyond the crossover: full enumeration, chunked
     enumeration, stratified sampling with a local sweep, and search. Peak
     memory is recorded.
  3  Robustness of the planner when the objective is non-monotone in the task
     parameters and dominated by interactions.
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


def probs(ab, cand):
    """Identical inference path for every arm: direct model call, batched."""
    B = cand.shape[0]
    x1 = np.concatenate([np.repeat(ab[None, :], B, 0), cand], 1).astype(np.float32)
    x2p = np.concatenate([np.repeat(PREV[None, :], B, 0), np.ones((B, 1), np.float32)], 1)
    a = np.asarray(m1(x1, training=False))[:, 0]
    g = np.asarray(m2([x2p, cand.astype(np.float32)], training=False))[:, 0]
    return 0.5 * a + 0.5 * g


def loss(ab, cand):
    return float(abs(probs(ab, cand)[0] - TARGET))


# ================================================== 1  model-based baselines
def search(ab, sampler, n_trials, lo, hi):
    def obj(tr):
        c = np.array([[tr.suggest_float("a", lo[0], hi[0]),
                       tr.suggest_float("b", lo[1], hi[1])]], np.float32)
        return loss(ab, c)
    s = optuna.create_study(direction="minimize", sampler=sampler)
    t = time.perf_counter()
    s.optimize(obj, n_trials=n_trials)
    return (time.perf_counter() - t) * 1e3, float(s.best_value)


def baselines(n_rep=8):
    lo, hi = (5.0, 1.0), (20.0, 5.0)
    grid = np.array(list(product(np.linspace(5, 20, 400), np.linspace(1, 5, 200))), np.float32)
    rows = []
    for name, mk in [("TPE", lambda: optuna.samplers.TPESampler(seed=1)),
                     ("TPE multivariate", lambda: optuna.samplers.TPESampler(multivariate=True, group=True, seed=1)),
                     ("CMA-ES", lambda: optuna.samplers.CmaEsSampler(seed=1)),
                     ("random", lambda: optuna.samplers.RandomSampler(seed=1))]:
        for T in (20, 50, 100, 200):
            ts, es = [], []
            for k in range(n_rep):
                ab = rng.normal(1, .15, 10).astype(np.float32)
                probs(ab, grid[:1])
                t, e = search(ab, mk(), T, lo, hi)
                ts.append(t); es.append(e)
            rows.append(dict(sampler=name, trials=T, ms=float(np.mean(ts)),
                             err=float(np.mean(es)), err_sd=float(np.std(es))))
            print(f"  {name:17s} T={T:4d}  {np.mean(ts):8.1f} ms   err {np.mean(es):.5f}"
                  f" (sd {np.std(es):.5f})", flush=True)
    # exhaustive reference on the same 80 000-point grid, same inference path
    ts, es = [], []
    for k in range(n_rep):
        ab = rng.normal(1, .15, 10).astype(np.float32)
        probs(ab, grid[:1])
        t = time.perf_counter(); p = probs(ab, grid); t = (time.perf_counter() - t) * 1e3
        ts.append(t); es.append(float(np.abs(p - TARGET).min()))
    rows.append(dict(sampler="exhaustive batched", trials=len(grid),
                     ms=float(np.mean(ts)), err=float(np.mean(es)), err_sd=float(np.std(es))))
    print(f"  {'exhaustive':17s} |G|={len(grid)}  {np.mean(ts):8.1f} ms   err {np.mean(es):.5f}")
    return rows


# ======================================================= 2  very large grids
def large_grids():
    rows = []
    ab = rng.normal(1, .15, 10).astype(np.float32)
    # the largest catalogued descriptor: 8 parameters, |G| = 2 457 000
    for G in (10 ** 5, 10 ** 6, 2_457_000):
        g1 = np.linspace(5, 20, int(np.sqrt(G)))
        g2 = np.linspace(1, 5, int(np.ceil(G / len(g1))))
        grid = np.array(list(product(g1, g2)), np.float32)[:G]
        probs(ab, grid[:1])
        tracemalloc.start()
        t = time.perf_counter(); p = probs(ab, grid); t_full = (time.perf_counter() - t) * 1e3
        peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
        e_full = float(np.abs(p - TARGET).min())

        # chunked enumeration, 65 536 candidates per invocation
        tracemalloc.start()
        t = time.perf_counter()
        best = 1.0
        for i in range(0, len(grid), 65536):
            best = min(best, float(np.abs(probs(ab, grid[i:i + 65536]) - TARGET).min()))
        t_chunk = (time.perf_counter() - t) * 1e3
        peak_chunk = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()

        # stratified sample then local sweep around the incumbent
        t = time.perf_counter()
        idx = rng.choice(len(grid), size=min(4096, len(grid)), replace=False)
        sub = grid[idx]
        ps = probs(ab, sub)
        j = int(np.argmin(np.abs(ps - TARGET)))
        c = sub[j]
        loc = np.array(list(product(np.linspace(c[0] - .5, c[0] + .5, 64),
                                    np.linspace(c[1] - .2, c[1] + .2, 64))), np.float32)
        pl = probs(ab, loc)
        e_strat = float(min(np.abs(ps - TARGET).min(), np.abs(pl - TARGET).min()))
        t_strat = (time.perf_counter() - t) * 1e3

        t_tpe, e_tpe = search(ab, optuna.samplers.TPESampler(seed=1), 20, (5, 1), (20, 5))
        rows.append(dict(grid=int(len(grid)), full_ms=round(t_full, 1), full_mb=round(peak, 1),
                         full_err=e_full, chunk_ms=round(t_chunk, 1), chunk_mb=round(peak_chunk, 1),
                         strat_ms=round(t_strat, 1), strat_err=e_strat,
                         tpe20_ms=round(t_tpe, 1), tpe20_err=e_tpe))
        print(f"  |G|={len(grid):>9,d}  full {t_full:8.1f} ms / {peak:6.1f} MB (err {e_full:.5f}) | "
              f"chunked {t_chunk:8.1f} ms / {peak_chunk:6.1f} MB | "
              f"stratified {t_strat:7.1f} ms (err {e_strat:.5f}) | "
              f"TPE20 {t_tpe:6.1f} ms (err {e_tpe:.5f})", flush=True)
    return rows


# ================================================== 3  non-monotone objective
def non_monotone():
    """A deliberately adversarial response surface: the effect of parameter 1
    reverses beyond its midpoint and the two parameters interact multiplicatively.
    The planner is agnostic to this because it enumerates; a rule tier that
    assumes monotone direction is not."""
    def surface(c):
        u = (c[:, 0] - 5) / 15.0
        v = (c[:, 1] - 1) / 4.0
        base = 0.5 + 0.45 * np.sin(2 * np.pi * u) * (1 - 0.6 * v) - 0.25 * (u * v)
        return np.clip(base, 0.01, 0.99)

    grid = np.array(list(product(range(5, 21), range(1, 6))), np.float32)
    p = surface(grid)
    j = int(np.argmin(np.abs(p - TARGET)))
    exhaustive_err = float(abs(p[j] - TARGET))

    # monotone-assuming rule tier: step one parameter toward "harder"
    cur = np.array([10.0, 3.0]); errs = []
    for _ in range(40):
        pc = float(surface(cur[None, :])[0])
        errs.append(abs(pc - TARGET))
        step = 1.0 if pc > TARGET else -1.0
        k = int(rng.integers(2))
        cur[k] = np.clip(cur[k] + step * (1 if k == 0 else 1), [5, 1][k], [20, 5][k])
    rule_err = float(np.mean(errs[10:]))

    def obj(tr):
        c = np.array([[tr.suggest_int("a", 5, 20), tr.suggest_int("b", 1, 5)]], np.float32)
        return float(abs(surface(c)[0] - TARGET))
    s = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1))
    s.optimize(obj, n_trials=20)
    tpe_err = float(s.best_value)

    print(f"  non-monotone surface: exhaustive err {exhaustive_err:.5f} | "
          f"TPE20 err {tpe_err:.5f} | monotone rule tier err {rule_err:.5f}")
    return dict(exhaustive_err=exhaustive_err, tpe20_err=tpe_err, rule_err=rule_err)


if __name__ == "__main__":
    print("1  model-based baselines (identical inference path)")
    a = baselines()
    print("\n2  grids beyond the crossover")
    b = large_grids()
    print("\n3  non-monotone, interaction-dominated objective")
    c = non_monotone()
    json.dump(dict(baselines=a, large_grids=b, nonmonotone=c),
              open(results("results_baselines.json"), "w"), indent=2)
    print("\nwrote results_baselines.json")
