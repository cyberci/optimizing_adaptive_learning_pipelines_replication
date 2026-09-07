"""Where does the residual error of the corrected predictor go?

Correcting the training labels reduces mean absolute error against the true
success probability from 0.208 to 0.038, yet flow-band retention rises only from
20.0 to 35.8 per cent against an architectural bound of 93.7. This script asks
why. Two hypotheses are separable:

  H1  the residual error is simply too large for a band of half-width 0.10;
  H2  the planner selects the candidate that minimises |p_hat - 0.7|, so it
      preferentially picks candidates at which the predictor's error points
      towards the target - the optimiser's curse. Under H2 the error the
      learner actually experiences is much larger than the average error.

H1 is tested by a counterfactual in which the true probabilities are perturbed
by independent noise calibrated to the same mean absolute error and the planner
then selects on the perturbed values. If retention under that counterfactual is
far above the measured 35.8 per cent, the magnitude of the error does not
explain the loss and the structure of the error does.
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

EX = np.array([0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75])
W = np.array([0, .1, 0, 0, .1, 0, .2, 0, .3, .3])
PT = [cs.ExpParam(6, 1.5, 5, 20), cs.ExpParam(8, 0.5, 1, 5)]
GRID = np.array(list(product(range(5, 21), range(1, 6))), dtype=np.float32)
TARGET, LO, HI = 0.7, 0.6, 0.8
N_LEARNERS = 400
rng = np.random.default_rng(31)


def true_p_matrix(AB):
    """(n_learners, |G|) noise-free success probabilities from the response model."""
    out = np.empty((len(AB), len(GRID)))
    for j, g in enumerate(GRID):
        theta = EX.copy()
        for pt, v in zip(PT, g):
            theta[pt.ability_index] *= pt.multiplier(v)
        act = theta > 0
        w = W[act] / W[act].sum()
        sig = 1.0 / (1.0 + np.exp(-cs.DISCRIMINATION_A * (AB[:, act] - theta[act])))
        out[:, j] = np.exp((w * np.log(np.clip(sig, 1e-12, 1.0))).sum(axis=1))
    return out


def pred_matrix(eng, AB, prev, result=True, alpha=0.5):
    m1, m2, meta = eng.cache[eng.desc["id"]]
    out = np.empty((len(AB), len(GRID)))
    for i, ab in enumerate(AB):
        out[i] = eng._p(m1, m2, alpha, ab.astype(np.float32), prev, result, GRID)
    return out


def band(p):
    return float(((p >= LO) & (p <= HI)).mean())


def report(tag, P_hat, P_true):
    sel = np.argmin(np.abs(P_hat - TARGET), axis=1)
    rows = np.arange(len(P_hat))
    p_sel_true = P_true[rows, sel]
    p_sel_hat = P_hat[rows, sel]
    err_all = np.abs(P_hat - P_true)
    err_sel = np.abs(p_sel_hat - p_sel_true)
    bias_all = float((P_hat - P_true).mean())
    bias_sel = float((p_sel_hat - p_sel_true).mean())
    print(f"  {tag}")
    print(f"    MAE over all candidates      {err_all.mean():.4f}")
    print(f"    MAE at the selected candidate{err_sel.mean():10.4f}"
          f"   ({err_sel.mean()/err_all.mean():.2f}x)")
    print(f"    bias over all candidates     {bias_all:+.4f}")
    print(f"    bias at selected candidate   {bias_sel:+.4f}")
    print(f"    flow-band retention          {band(p_sel_true):.1%}")
    return dict(tag=tag, mae_all=float(err_all.mean()), mae_sel=float(err_sel.mean()),
                inflation=float(err_sel.mean() / err_all.mean()),
                bias_all=bias_all, bias_sel=bias_sel,
                flow_band=band(p_sel_true))


if __name__ == "__main__":
    AB = cs.generate_abilities(N_LEARNERS, 10, rng)
    P_true = true_p_matrix(AB)
    prev = np.array([10, 3], np.float32)

    print("residual error of the corrected predictor")
    out = {}
    for tag in ("B_corrected", "A_as_deployed"):
        eng = engine.Engine("V4", desc={**engine.DESCRIPTOR, "id": f"medusa_{tag}"})
        P_hat = pred_matrix(eng, AB, prev)
        out[tag] = report(tag, P_hat, P_true)

    # oracle reference on the same grid and learners
    sel = np.argmin(np.abs(P_true - TARGET), axis=1)
    out["oracle"] = dict(tag="oracle", flow_band=band(P_true[np.arange(len(AB)), sel]))
    print(f"  oracle (exact response model)  flow-band retention {out['oracle']['flow_band']:.1%}")

    # H1 counterfactual: independent noise at the same mean absolute error
    print("\n  counterfactual: independent noise at matched mean absolute error")
    rows = []
    for mae in (0.038, 0.076, 0.150, 0.208):
        scale = mae / np.sqrt(2 / np.pi)          # MAE of a zero-mean Gaussian
        r = []
        for rep in range(20):
            P_n = np.clip(P_true + rng.normal(0, scale, P_true.shape), 1e-4, 1 - 1e-4)
            s = np.argmin(np.abs(P_n - TARGET), axis=1)
            r.append(band(P_true[np.arange(len(AB)), s]))
        rows.append(dict(mae=mae, flow_band=float(np.mean(r)), sd=float(np.std(r))))
        print(f"    MAE {mae:.3f}  ->  flow-band {np.mean(r):.1%}  (sd {np.std(r):.3f})")
    out["iid_counterfactual"] = rows

    json.dump(out, open(results("results_residual.json"), "w"), indent=2)
    print("\nwrote results_residual.json")
