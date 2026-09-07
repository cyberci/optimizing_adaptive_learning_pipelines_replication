"""The ensemble weight as a second localised configuration choice.

The ensemble combines an abilities network and a games network with a convex
weight alpha on the abilities branch. The value is a free configuration
parameter: it is fixed when the models are trained, written into the model
metadata and restored when they are loaded. Its default is 0.5, nothing in the
training procedure chooses it, and nothing validates it.
Only the abilities branch is learner-specific, so the weight controls how much of
each prediction depends on the learner at all.

The longitudinal loop of bench_tiers.py is re-run with the weight as a free
parameter so that the domain metric can be reported at the class default
(alpha = 0.5, the reference configuration), for the games network alone
(alpha = 0) and for the abilities network alone (alpha = 1).
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import engine, cognitive_sim as cs
from bench_tiers import CFG, GRID, TARGET, LO, HI, STEPS, LEARNERS, BURN, true_p


def run(eng, alpha, seed=7):
    rng = np.random.default_rng(seed)
    ab = cs.generate_abilities(LEARNERS, 10, rng)
    prev = [{c["name"]: c["initial"] for c in CFG} for _ in range(LEARNERS)]
    succ = [True] * LEARNERS
    inband = np.zeros(LEARNERS)
    realised = [[] for _ in range(LEARNERS)]
    distinct = set()
    for t in range(STEPS):
        for i in range(LEARNERS):
            cp, _, _ = eng.suggest(ab[i], prev[i], succ[i], alpha=alpha)
            cfg = {k: int(v) for k, v in cp.items()}
            if t >= BURN:
                distinct.add((cfg["number_of_medusas"], cfg["time_limit"]))
            p = true_p(ab[i], [cfg["number_of_medusas"], cfg["time_limit"]], rng)
            if t >= BURN:
                inband[i] += (LO <= p <= HI)
                realised[i].append(p)
            succ[i] = rng.random() < p
            prev[i] = cfg
    n = STEPS - BURN
    b = inband / n
    dev = np.array([np.mean(np.abs(np.array(r) - TARGET)) for r in realised])
    return dict(flow_band=float(b.mean()),
                ci95=float(1.96 * b.std(ddof=1) / np.sqrt(LEARNERS)),
                mean_abs_dev=float(dev.mean()),
                mean_p=float(np.mean([np.mean(r) for r in realised])),
                distinct_configs=len(distinct))


if __name__ == "__main__":
    rows = []
    for tag in ("A_as_deployed", "B_corrected"):
        eng = engine.Engine("V4", desc={**engine.DESCRIPTOR, "id": f"medusa_{tag}"})
        for alpha in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            t = time.perf_counter()
            r = run(eng, alpha)
            r.update(tag=tag, alpha=alpha, wall_s=round(time.perf_counter() - t, 1))
            rows.append(r)
            print(f"  {tag:14s} alpha={alpha:.1f}  flow-band {r['flow_band']:6.1%} "
                  f"+/-{r['ci95']:.3f}   mean p {r['mean_p']:.3f}   "
                  f"distinct configs {r['distinct_configs']:3d}  [{r['wall_s']}s]", flush=True)
    json.dump(rows, open(results("results_tiers_alpha.json"), "w"), indent=2)
    print("\nwrote results_tiers_alpha.json")
