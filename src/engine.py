"""Recommendation engine variants.

V0  as implemented: per-request load_model(), Optuna TPE (n_trials=20), model.predict()
V1  +cache       : models loaded once at start-up
V2  +warmup      : cache + traced warm-up call
V3  +directcall  : warmup + model(x, training=False) instead of .predict()
V4  +planner     : descriptor-driven planner - exhaustive batched sweep when the
                   descriptor's integer grid is small, TPE otherwise
"""
import os, pickle, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from repro_paths import MODELS, RESULTS, model, results
import numpy as np, optuna
from itertools import product
from tensorflow import keras
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = str(MODELS)
TARGET = 0.7
N_TRIALS = 20                      # trial budget of the search as implemented
GRID_BUDGET = 4096                 # planner threshold on |candidate grid|

DESCRIPTOR = {                     # one entry of the task-descriptor catalogue
    "id": "medusa",
    "params": [
        {"name": "number_of_medusas", "min": 5, "max": 20},
        {"name": "time_limit",        "min": 1, "max": 5},
    ],
}


def grid_size(desc):
    n = 1
    for p in desc["params"]:
        n *= int(p["max"]) - int(p["min"]) + 1
    return n


class Engine:
    def __init__(self, variant="V0", desc=DESCRIPTOR, model_dir=MODEL_DIR):
        self.variant, self.desc, self.model_dir = variant, desc, model_dir
        self.cache = {}
        if variant != "V0":
            self._load(desc["id"])
        if variant in ("V2", "V3", "V4"):
            self._warmup(desc["id"])

    # ---------- model management ----------
    def _load(self, gid):
        m1 = keras.models.load_model(f"{self.model_dir}/{gid}/abilities_model.h5", compile=False)
        m2 = keras.models.load_model(f"{self.model_dir}/{gid}/games_model.h5", compile=False)
        with open(f"{self.model_dir}/{gid}/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        self.cache[gid] = (m1, m2, meta)
        return self.cache[gid]

    def _get(self, gid):
        if self.variant == "V0":          # as implemented: reload on every request
            return self._load(gid)
        return self.cache[gid]

    def _warmup(self, gid):
        m1, m2, meta = self.cache[gid]
        d = len(meta["param_col_names"])
        m1(np.zeros((1, 10 + d), np.float32), training=False)
        m2([np.zeros((1, d + 1), np.float32), np.zeros((1, d), np.float32)], training=False)

    # ---------- inference ----------
    def _p(self, m1, m2, alpha, ab, prev, res, cand):
        """cand: (B, d) array of candidate parameter vectors -> (B,) probabilities"""
        B = cand.shape[0]
        x1 = np.concatenate([np.repeat(ab[None, :], B, 0), cand], 1).astype(np.float32)
        x2p = np.concatenate([np.repeat(prev[None, :], B, 0),
                              np.full((B, 1), float(res), np.float32)], 1).astype(np.float32)
        if self.variant in ("V0", "V1", "V2"):
            a = m1.predict(x1, verbose=0)[:, 0]
            g = m2.predict([x2p, cand.astype(np.float32)], verbose=0)[:, 0]
        else:
            a = np.asarray(m1(x1, training=False))[:, 0]
            g = np.asarray(m2([x2p, cand.astype(np.float32)], training=False))[:, 0]
        return alpha * a + (1.0 - alpha) * g

    # ---------- search ----------
    def suggest(self, abilities, prev_params, result, alpha=0.5, gid=None):
        gid = gid or self.desc["id"]
        m1, m2, meta = self._get(gid)
        ab = np.asarray(abilities, np.float32)
        prev = np.asarray([prev_params[p["name"]] for p in self.desc["params"]], np.float32)
        ps = self.desc["params"]

        if self.variant == "V4" and grid_size(self.desc) <= GRID_BUDGET:
            grid = np.array(list(product(*[range(int(p["min"]), int(p["max"]) + 1) for p in ps])),
                            dtype=np.float32)
            pr = self._p(m1, m2, alpha, ab, prev, result, grid)
            i = int(np.argmin(np.abs(pr - TARGET)))
            return ({p["name"]: float(grid[i, j]) for j, p in enumerate(ps)},
                    float(pr[i]), grid.shape[0])

        def objective(trial):
            cand = np.array([[trial.suggest_int(p["name"], int(p["min"]), int(p["max"]))
                              for p in ps]], np.float32)
            return abs(self._p(m1, m2, alpha, ab, prev, result, cand)[0] - TARGET)

        study = optuna.create_study(direction="minimize")   # default sampler = TPE
        study.optimize(objective, n_trials=N_TRIALS)
        best = np.array([[study.best_params[p["name"]] for p in ps]], np.float32)
        return ({p["name"]: float(v) for p, v in zip(ps, best[0])},
                float(self._p(m1, m2, alpha, ab, prev, result, best)[0]), N_TRIALS)
