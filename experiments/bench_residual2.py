"""Decomposing the residual error of the corrected predictor.

The ensemble averages an abilities network and a games network. Each is scored
separately against the true response model over the candidate grid the planner
actually enumerates, and the effect of a single-parameter post-hoc calibration
(a shared intercept on the logit scale, fitted on a held-out learner sample) is
measured in the domain metric.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from itertools import product
import engine, cognitive_sim as cs
from bench_residual import true_p_matrix, GRID, TARGET, LO, HI, band

N_FIT, N_EVAL = 200, 400
rng = np.random.default_rng(77)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def components(eng, AB, prev=np.array([10, 3], np.float32), result=True):
    """Returns (abilities-network, games-network, ensemble) prediction matrices."""
    m1, m2, meta = eng.cache[eng.desc["id"]]
    B = len(GRID)
    A = np.empty((len(AB), B)); Gm = np.empty((len(AB), B))
    x2p = np.concatenate([np.repeat(prev[None, :], B, 0),
                          np.full((B, 1), float(result), np.float32)], 1)
    g = np.asarray(m2([x2p, GRID], training=False))[:, 0]
    for i, ab in enumerate(AB):
        x1 = np.concatenate([np.repeat(ab.astype(np.float32)[None, :], B, 0), GRID], 1)
        A[i] = np.asarray(m1(x1, training=False))[:, 0]
        Gm[i] = g
    return A, Gm, 0.5 * A + 0.5 * Gm


def stats(name, P, P_true):
    e = P - P_true
    sel = np.argmin(np.abs(P - TARGET), axis=1)
    r = np.arange(len(P))
    print(f"    {name:34s} MAE {np.abs(e).mean():.4f}   bias {e.mean():+.4f}   "
          f"flow-band {band(P_true[r, sel]):6.1%}")
    return dict(name=name, mae=float(np.abs(e).mean()), bias=float(e.mean()),
                flow_band=band(P_true[r, sel]))


if __name__ == "__main__":
    AB_fit = cs.generate_abilities(N_FIT, 10, rng)
    AB_ev = cs.generate_abilities(N_EVAL, 10, rng)
    T_fit, T_ev = true_p_matrix(AB_fit), true_p_matrix(AB_ev)

    out = {}
    for tag in ("B_corrected", "A_as_deployed"):
        eng = engine.Engine("V4", desc={**engine.DESCRIPTOR, "id": f"medusa_{tag}"})
        print(f"\n  {tag}")
        Af, Gf, Ef = components(eng, AB_fit)
        Ae, Ge, Ee = components(eng, AB_ev)
        rows = [stats("abilities network alone", Ae, T_ev),
                stats("games network alone", Ge, T_ev),
                stats("ensemble", Ee, T_ev)]

        # one-parameter recalibration: shared intercept on the logit scale,
        # fitted on the held-out learner sample by matching the mean logit
        b = float((logit(T_fit) - logit(Ef)).mean())
        Ec = sigmoid(logit(Ee) + b)
        rows.append(stats(f"ensemble + logit intercept ({b:+.3f})", Ec, T_ev))
        out[tag] = dict(intercept=b, rows=rows)

    sel = np.argmin(np.abs(T_ev - TARGET), axis=1)
    out["oracle_flow_band"] = band(T_ev[np.arange(len(T_ev)), sel])
    print(f"\n    {'oracle (exact response model)':34s} "
          f"{'':32s}flow-band {out['oracle_flow_band']:6.1%}")

    json.dump(out, open(results("results_residual2.json"), "w"), indent=2)
    print("\nwrote results_residual2.json")
