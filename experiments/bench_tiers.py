"""Definitive tier comparison, all through the *same* planner and the same
longitudinal loop, so that only the predictor differs.

  oracle        - planner given the true response model (upper bound of the architecture)
  corrected     - planner given ensemble B (Bernoulli labels, early stopping)
  as_deployed   - planner given ensemble A, the configuration as implemented
                  (threshold labels, EPOCHS=2)
  rule          - the rule tier T2 alone
  default       - the static descriptor defaults, tier T4, alone
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from itertools import product
import engine, cognitive_sim as cs

EX = np.array([0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75])
W = np.array([0, .1, 0, 0, .1, 0, .2, 0, .3, .3])
PT = [cs.ExpParam(6, 1.5, 5, 20), cs.ExpParam(8, 0.5, 1, 5)]
CFG = [dict(name="number_of_medusas", easiest=5, hardest=20, initial=10, increment=1),
       dict(name="time_limit",        easiest=5, hardest=1,  initial=3,  increment=-1)]
GRID = np.array(list(product(range(5, 21), range(1, 6))), dtype=float)
TARGET, LO, HI = 0.7, 0.6, 0.8
STEPS, LEARNERS, BURN = 45, 120, 10


def true_p(ab, cfg_vec, rng=None):
    theta = EX.copy()
    for pt, v in zip(PT, cfg_vec):
        theta[pt.ability_index] *= pt.multiplier(v)
    act = theta > 0
    w = W[act] / W[act].sum()
    beta = ab[act]
    if rng is not None:
        beta = beta + rng.uniform(-cs.NOISE_EPS, cs.NOISE_EPS, size=beta.shape)
    sig = 1.0 / (1.0 + np.exp(-cs.DISCRIMINATION_A * (beta - theta[act])))
    return float(np.exp((w * np.log(np.clip(sig, 1e-12, 1.0))).sum()))


def oracle_choice(ab):
    ps = np.array([true_p(ab, g) for g in GRID])
    return GRID[int(np.argmin(np.abs(ps - TARGET)))]


def clamp(v, c):
    lo, hi = min(c["easiest"], c["hardest"]), max(c["easiest"], c["hardest"])
    return int(np.clip(v, lo, hi))


def rule_choice(prev, success, rng):
    nxt = dict(prev); c = CFG[rng.integers(len(CFG))]
    nxt[c["name"]] = clamp(prev[c["name"]] + (c["increment"] if success else -c["increment"]), c)
    return nxt


def run(mode, eng=None, seed=7):
    rng = np.random.default_rng(seed)
    ab = cs.generate_abilities(LEARNERS, 10, rng)
    prev = [{c["name"]: c["initial"] for c in CFG} for _ in range(LEARNERS)]
    succ = [True] * LEARNERS
    inband = np.zeros(LEARNERS); realised = [[] for _ in range(LEARNERS)]
    for t in range(STEPS):
        for i in range(LEARNERS):
            if mode == "oracle":
                v = oracle_choice(ab[i]); cfg = {"number_of_medusas": int(v[0]), "time_limit": int(v[1])}
            elif mode == "rule":
                cfg = rule_choice(prev[i], succ[i], rng)
            elif mode == "default":
                cfg = {c["name"]: c["initial"] for c in CFG}
            else:
                cp, _, _ = eng.suggest(ab[i], prev[i], succ[i])
                cfg = {k: int(v) for k, v in cp.items()}
            p = true_p(ab[i], [cfg["number_of_medusas"], cfg["time_limit"]], rng)
            if t >= BURN:
                inband[i] += (LO <= p <= HI); realised[i].append(p)
            succ[i] = rng.random() < p
            prev[i] = cfg
    n = STEPS - BURN
    band = inband / n
    dev = np.array([np.mean(np.abs(np.array(r) - TARGET)) for r in realised])
    return dict(mode=mode, flow_band=float(band.mean()),
                ci95=float(1.96 * band.std(ddof=1) / np.sqrt(LEARNERS)),
                mean_abs_dev=float(dev.mean()),
                mean_p=float(np.mean([np.mean(r) for r in realised])))


if __name__ == "__main__":
    rows = []
    for mode in ["oracle", "rule", "default"]:
        t = time.perf_counter(); r = run(mode); r["wall_s"] = round(time.perf_counter() - t, 1)
        rows.append(r); print(f"{mode:12s} flow-band {r['flow_band']:6.1%} +/-{r['ci95']:.3f}  "
                              f"mean|p-0.7| {r['mean_abs_dev']:.3f}  mean p {r['mean_p']:.3f} [{r['wall_s']}s]", flush=True)
    for tag in ["B_corrected", "A_as_deployed"]:
        eng = engine.Engine("V4", desc={**engine.DESCRIPTOR, "id": f"medusa_{tag}"})
        t = time.perf_counter(); r = run(tag, eng); r["wall_s"] = round(time.perf_counter() - t, 1)
        rows.append(r); print(f"{tag:12s} flow-band {r['flow_band']:6.1%} +/-{r['ci95']:.3f}  "
                              f"mean|p-0.7| {r['mean_abs_dev']:.3f}  mean p {r['mean_p']:.3f} [{r['wall_s']}s]", flush=True)
    json.dump(rows, open(results("results_tiers.json"), "w"), indent=2)
    print("done")
