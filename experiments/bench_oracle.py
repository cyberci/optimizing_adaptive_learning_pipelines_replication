"""Oracle bound: what flow-band retention is achievable at all, given a
descriptor's integer grid and the reconstructed response model? Also quantifies
how much finer descriptor granularity buys. Vectorised over learners."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json
import numpy as np
from itertools import product
import cognitive_sim as cs

EX = np.array([0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75])
W = np.array([0, .1, 0, 0, .1, 0, .2, 0, .3, .3])
PT = [cs.ExpParam(6, 1.5, 5, 20), cs.ExpParam(8, 0.5, 1, 5)]
TARGET, LO, HI, REPS = 0.7, 0.6, 0.8, 40
rng = np.random.default_rng(5)
AB = cs.generate_abilities(400, 10, rng)


def p_matrix(grid, noise_rng=None):
    """(n_learners, n_grid) success probabilities."""
    out = np.empty((len(AB), len(grid)))
    for j, g in enumerate(grid):
        theta = EX.copy()
        for pt, v in zip(PT, g):
            theta[pt.ability_index] *= pt.multiplier(v)
        act = theta > 0
        w = W[act] / W[act].sum()
        beta = AB[:, act]
        if noise_rng is not None:
            beta = beta + noise_rng.uniform(-cs.NOISE_EPS, cs.NOISE_EPS, size=beta.shape)
        sig = 1.0 / (1.0 + np.exp(-cs.DISCRIMINATION_A * (beta - theta[act])))
        out[:, j] = np.exp((w * np.log(np.clip(sig, 1e-12, 1.0))).sum(axis=1))
    return out


def sweep(nm, tl, label):
    grid = list(product(nm, tl))
    P0 = p_matrix(grid)                       # noise-free
    best = np.argmin(np.abs(P0 - TARGET), axis=1)
    chosen = [grid[b] for b in best]
    nf = np.abs(P0[np.arange(len(AB)), best] - TARGET)
    r = np.empty((len(AB), REPS))
    for k in range(REPS):
        pk = p_matrix(chosen, noise_rng=rng)
        r[:, k] = pk[np.arange(len(AB)), np.arange(len(AB))] if False else np.diag(pk)
    band = ((r >= LO) & (r <= HI)).mean(axis=1)
    dev = np.abs(r - TARGET).mean(axis=1)
    res = dict(label=label, grid=len(grid), oracle_flow_band=float(band.mean()),
               mean_abs_dev=float(dev.mean()), noise_free_gap=float(nf.mean()))
    print(f"{label:24s} |G|={len(grid):7d}  oracle flow-band={band.mean():6.1%}  "
          f"mean|p-0.7|={dev.mean():.3f}  noise-free gap={nf.mean():.4f}", flush=True)
    return res


if __name__ == "__main__":
    rows = [sweep(list(range(5, 21)), list(range(1, 6)), "catalogued integer grid"),
            sweep(np.arange(5, 20.01, 0.5), np.arange(1, 5.01, 0.25), "2x refinement"),
            sweep(np.arange(5, 20.01, 0.1), np.arange(1, 5.01, 0.05), "10x refinement")]
    json.dump(rows, open(results("results_oracle.json"), "w"), indent=2)
    print("done")
