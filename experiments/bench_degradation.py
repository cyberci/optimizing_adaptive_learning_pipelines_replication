"""Pedagogical quality of each tier of the graceful-degradation chain:

  T1 machine-learning optimiser  - descriptor-driven batched sweep
  T2 rule tier                   - +/- one declared increment on one random parameter
  T3 replay tier                 - the learner's last completed configuration
  T4 static tier                 - the descriptor's initial values

The availability of T1 is swept from 1.0 down to 0.0. Because every learner is
simulated, degradation is reported in flow-band retention instead of in an HTTP
success rate.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import engine, cognitive_sim as cs

EX = [0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75]
W = [0, .1, 0, 0, .1, 0, .2, 0, .3, .3]
PT = [cs.ExpParam(6, 1.5, 5, 20), cs.ExpParam(8, 0.5, 1, 5)]
CFG = [dict(name="number_of_medusas", easiest=5, hardest=20, initial=10, increment=1),
       dict(name="time_limit",        easiest=5, hardest=1,  initial=3,  increment=-1)]
TARGET, BAND = 0.7, (0.6, 0.8)
STEPS, LEARNERS, BURN = 45, 120, 10
RULE_TIER_AVAIL = 0.98


def clamp(v, c):
    lo, hi = min(c["easiest"], c["hardest"]), max(c["easiest"], c["hardest"])
    return int(np.clip(v, lo, hi))


def tier_auto(prev, success, rng):
    nxt = dict(prev)
    c = CFG[rng.integers(len(CFG))]
    nxt[c["name"]] = clamp(prev[c["name"]] + (c["increment"] if success else -c["increment"]), c)
    return nxt


def tier_default():
    return {c["name"]: c["initial"] for c in CFG}


def run(eng, avail, seed):
    rng = np.random.default_rng(seed)
    ab = cs.generate_abilities(LEARNERS, 10, rng)
    prev = [tier_default() for _ in range(LEARNERS)]
    last = [None] * LEARNERS
    succ = [True] * LEARNERS
    inband = np.zeros(LEARNERS)
    realised = [[] for _ in range(LEARNERS)]
    tiers = np.zeros(4)
    for t in range(STEPS):
        for i in range(LEARNERS):
            if rng.random() < avail:
                cp, _, _ = eng.suggest(ab[i], prev[i], succ[i])
                cfg = {k: int(v) for k, v in cp.items()}
                tiers[0] += 1
            elif rng.random() < RULE_TIER_AVAIL:
                cfg = tier_auto(prev[i], succ[i], rng)
                tiers[1] += 1
            elif last[i] is not None:
                cfg = dict(last[i])
                tiers[2] += 1
            else:
                cfg = tier_default()
                tiers[3] += 1
            p, _ = cs.success_probability(ab[i], EX, W, PT,
                                          [cfg["number_of_medusas"], cfg["time_limit"]], rng=rng)
            p = float(p[0])
            if t >= BURN:
                inband[i] += (BAND[0] <= p <= BAND[1])
                realised[i].append(p)
            succ[i] = rng.random() < p
            last[i] = prev[i]
            prev[i] = cfg
    n = STEPS - BURN
    band = inband / n
    dev = np.array([np.mean(np.abs(np.array(r) - TARGET)) for r in realised])
    ci = 1.96 * band.std(ddof=1) / np.sqrt(LEARNERS)
    return dict(availability=avail, flow_band=float(band.mean()), ci95=float(ci),
                mean_abs_dev=float(dev.mean()),
                mean_realised_p=float(np.mean([np.mean(r) for r in realised])),
                tier_share=[round(float(x / tiers.sum()), 4) for x in tiers])


if __name__ == "__main__":
    out = {}
    for tag, levels in [("B_corrected", [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]),
                        ("A_as_deployed", [1.0])]:
        eng = engine.Engine("V4", desc={**engine.DESCRIPTOR, "id": f"medusa_{tag}"})
        rows = []
        for a in levels:
            t = time.perf_counter()
            r = run(eng, a, 100 + int(a * 100))
            r["wall_s"] = round(time.perf_counter() - t, 1)
            rows.append(r)
            print(f"[{tag}] ML avail {a:5.0%}  flow-band {r['flow_band']:6.1%} +/-{r['ci95']:.3f}"
                  f"  mean|p-0.7| {r['mean_abs_dev']:.3f}  mean p {r['mean_realised_p']:.3f}"
                  f"  [{r['wall_s']}s]", flush=True)
        out[tag] = rows
    json.dump(out, open(results("results_degradation.json"), "w"), indent=2)
    print("done")
