"""Trains the two ensemble members with the training protocol as implemented:
the same layer sizes, optimiser, loss, metric, callbacks, batch size, epoch
count and split seeds that the platform uses."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, pickle, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, brier_score_loss
from tensorflow import keras
from keras.metrics import F1Score
import cognitive_sim as cs

EPOCHS, BATCH_SIZE = 2, 64          # values of the protocol as implemented
K, N = 10, 1000                     # ability dimensions, learners per epoch
OUT = model("medusa"); os.makedirs(OUT, exist_ok=True)

EXERCISE = [0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75]
WEIGHTS  = [0, 0.1, 0, 0, 0.1, 0, 0.2, 0, 0.3, 0.3]
PARAMS = {"number_of_medusas": cs.ExpParam(6, 1.5, 5, 20),
          "time_limit":        cs.ExpParam(8, 0.5, 1, 5)}

def simulate():
    rng = np.random.default_rng(42)
    abilities = cs.generate_abilities(N, K, rng)
    grids = [range(int(p.min), int(p.max) + 1) for p in PARAMS.values()]
    frames = []
    for combo in product(*grids):
        _, res, mod = cs.simulate_exercise_params(
            N, abilities, EXERCISE, WEIGHTS, list(PARAMS.values()), list(combo), rng=rng)
        d = cs.save_to_df(abilities, res, mod)
        for name, v in zip(PARAMS.keys(), combo):
            d[f"p_{name}"] = v
        frames.append(d)
    df = pd.concat(frames).reset_index(drop=True)
    return df

def build_m1(input_dim):
    m = keras.Sequential([keras.Input(shape=(input_dim,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=[F1Score(threshold=0.5)])
    return m

def build_m2(d_prev, d_new):
    pi = keras.layers.Input(shape=(d_prev,)); ni = keras.layers.Input(shape=(d_new,))
    px = keras.layers.Dense(64, activation="relu")(pi)
    nx = keras.layers.Dense(64, activation="relu")(ni)
    x = keras.layers.concatenate([px, nx])
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    o = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs=[pi, ni], outputs=o)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=[F1Score(threshold=0.5)])
    return m

def main():
    t0 = time.perf_counter()
    df = simulate()
    print(f"simulated rows: {len(df):,}  ({time.perf_counter()-t0:.1f}s)")
    pcols = [c for c in df.columns if c.startswith("p_")]
    acols = [c for c in df.columns if c.startswith("ability_")]

    # ---- Model 1 (ability-based) : split protocol as implemented ----
    X = df[acols + pcols]; y = df["result"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xte, yte, test_size=0.5, random_state=42)
    m1 = build_m1(Xtr.shape[1])
    m1.fit(Xtr, (ytr >= .5).astype(int), validation_data=(Xva, (yva >= .5).astype(int)),
           batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_f1_score", patience=10,
                                                    restore_best_weights=True)])
    p1 = m1.predict(Xte, verbose=0)[:, 0]
    print(f"M1 MAE={mean_absolute_error(yte,p1):.4f} Brier={brier_score_loss((yte>=.5).astype(int),p1):.4f}")

    # ---- Model 2 (history-based) : pairing protocol as implemented ----
    md = df[["person_id"] + pcols + ["result"]]
    merged = md.merge(md, on="person_id", suffixes=("", "_new"), how="inner")
    merged = merged.sample(frac=0.1, random_state=42).reset_index(drop=True)
    npcols = [f"{c}_new" for c in pcols]
    tr = merged[pcols + ["result"] + npcols]; tgt = merged["result_new"]
    Xtr, Xte, ytr, yte = train_test_split(tr, tgt, test_size=0.2, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xte, yte, test_size=0.5, random_state=42)
    m2 = build_m2(len(pcols) + 1, len(npcols))
    m2.fit([Xtr[pcols + ["result"]], Xtr[npcols]], (ytr >= .5).astype(int),
           validation_data=([Xva[pcols + ["result"]], Xva[npcols]], (yva >= .5).astype(int)),
           batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_f1_score", patience=10,
                                                    restore_best_weights=True)])
    p2 = m2.predict([Xte[pcols + ["result"]], Xte[npcols]], verbose=0)[:, 0]
    print(f"M2 MAE={mean_absolute_error(yte,p2):.4f} Brier={brier_score_loss((yte>=.5).astype(int),p2):.4f}")

    m1.save(f"{OUT}/abilities_model.h5"); m2.save(f"{OUT}/games_model.h5")
    with open(f"{OUT}/metadata.pkl", "wb") as f:
        pickle.dump({"ability_accuracy": 0.5, "param_col_names": pcols}, f)
    df.to_csv(work("sim_results.csv"), index=False)
    print("artifacts:", {f: os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)})

main()
