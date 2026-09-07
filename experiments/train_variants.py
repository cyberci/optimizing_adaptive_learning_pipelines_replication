"""Two training protocols on identical data:

  A  as implemented : label = 1[p >= 0.5], EPOCHS = 2
  B  corrected   : label ~ Bernoulli(p), early stopping on val_loss, EPOCHS <= 80

Both are evaluated against the *true* latent success probability p.
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os, pickle, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, brier_score_loss
from tensorflow import keras
from keras.metrics import F1Score

df = pd.read_csv(work("sim_results.csv"))
pcols = [c for c in df.columns if c.startswith("p_")]
acols = [c for c in df.columns if c.startswith("ability_")]


def m1_arch(d):
    return keras.Sequential([keras.Input(shape=(d,)),
                             keras.layers.Dense(64, activation="relu"),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1, activation="sigmoid")])


def m2_arch(dp, dn):
    pi = keras.layers.Input(shape=(dp,)); ni = keras.layers.Input(shape=(dn,))
    x = keras.layers.concatenate([keras.layers.Dense(64, activation="relu")(pi),
                                  keras.layers.Dense(64, activation="relu")(ni)])
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    return keras.Model([pi, ni], keras.layers.Dense(1, activation="sigmoid")(x))


def ece(y_bin, p_hat, bins=10):
    e = 0.0; n = len(p_hat)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p_hat >= lo) & (p_hat < lo + 1.0 / bins)
        if m.sum():
            e += m.sum() / n * abs(y_bin[m].mean() - p_hat[m].mean())
    return float(e)


rng = np.random.default_rng(3)
out = {}
for tag, as_implemented in [("A_as_deployed", True), ("B_corrected", False)]:
    X = df[acols + pcols].values.astype(np.float32)
    p = df["result"].values.astype(np.float32)
    y = (p >= 0.5).astype(np.float32) if as_implemented else (rng.random(len(p)) < p).astype(np.float32)
    Xtr, Xte, ytr, yte, ptr, pte = train_test_split(X, y, p, test_size=0.2, random_state=42)
    Xva, Xte, yva, yte, pva, pte = train_test_split(Xte, yte, pte, test_size=0.5, random_state=42)
    m = m1_arch(X.shape[1])
    if as_implemented:
        m.compile("adam", "binary_crossentropy", metrics=[F1Score(threshold=0.5)])
        m.fit(Xtr, ytr.reshape(-1,1), validation_data=(Xva, yva.reshape(-1,1)), batch_size=64, epochs=2, verbose=0)
    else:
        m.compile("adam", "binary_crossentropy")
        m.fit(Xtr, ytr, validation_data=(Xva, yva), batch_size=256, epochs=80, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                       restore_best_weights=True)])
    ph = m.predict(Xte, verbose=0)[:, 0]
    sel = (pte > 0.68) & (pte < 0.72)
    out[tag] = {"MAE_vs_true_p": round(float(mean_absolute_error(pte, ph)), 4),
                "Brier_vs_outcome": round(float(brier_score_loss((pte >= 0.5).astype(int), np.clip(ph, 0, 1))), 4),
                "ECE": round(ece((pte >= 0.5).astype(float), ph), 4),
                "mean_pred_where_true_p_is_0.7": round(float(ph[sel].mean()), 4)}
    d = model(f"medusa_{tag}"); os.makedirs(d, exist_ok=True)
    m.save(f"{d}/abilities_model.h5")

    md = df[["person_id"] + pcols + ["result"]]
    mg = md.merge(md, on="person_id", suffixes=("", "_new")).sample(frac=0.1, random_state=42).reset_index(drop=True)
    npc = [f"{c}_new" for c in pcols]
    Pp = mg[pcols + ["result"]].values.astype(np.float32)
    Pn = mg[npc].values.astype(np.float32)
    pn = mg["result_new"].values.astype(np.float32)
    yn = (pn >= 0.5).astype(np.float32) if as_implemented else (rng.random(len(pn)) < pn).astype(np.float32)
    itr, ite = train_test_split(np.arange(len(pn)), test_size=0.2, random_state=42)
    m2 = m2_arch(Pp.shape[1], Pn.shape[1])
    if as_implemented:
        m2.compile("adam", "binary_crossentropy", metrics=[F1Score(threshold=0.5)])
        m2.fit([Pp[itr], Pn[itr]], yn[itr].reshape(-1,1), batch_size=64, epochs=2, verbose=0)
    else:
        m2.compile("adam", "binary_crossentropy")
        m2.fit([Pp[itr], Pn[itr]], yn[itr], validation_split=0.1, batch_size=256, epochs=80, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                        restore_best_weights=True)])
    m2.save(f"{d}/games_model.h5")
    pickle.dump({"ability_accuracy": 0.5, "param_col_names": pcols}, open(f"{d}/metadata.pkl", "wb"))
    print(tag, out[tag], flush=True)

json.dump(out, open(results("results_training.json"), "w"), indent=2)
print("done")
