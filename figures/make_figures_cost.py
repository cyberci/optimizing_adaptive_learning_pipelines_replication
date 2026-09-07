"""Regenerates Figure 4 from the shape-matched planner sweep and builds the new
Figure 8 on runtime generality and the analytic crossover."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2563EB", "#EA580C", "#0D9488", "#9333EA"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FIG = str(FIGURES)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# ================================================================== Figure 4
pl = json.load(open(results("results_planner.json")))
ens = json.load(open(results("results_ensemble_cost.json")))
desc = json.load(open(data("descriptors.json")))
G = np.array([r["grid"] for r in pl], float)
te = np.array([r["exhaustive_ms"] for r in pl])
tt = np.array([r["tpe20_ms"] for r in pl])
ee = np.array([r["err_exhaustive"] for r in pl])
et = np.array([r["err_tpe20"] for r in pl])
cross20 = ens["crossover"]["20"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
gg = np.logspace(0.9, 6.3, 200)
a1.plot(gg, ens["c_fixed"] + gg * ens["c_unit_large"], "-", color=C[1], lw=0.9,
        alpha=0.45, zorder=2)
a1.plot(G, te, "-o", color=C[1], lw=1.8, ms=4.5, label="exhaustive batched sweep", zorder=4)
a1.plot(G, tt, "-s", color=C[0], lw=1.8, ms=4.5, label="TPE, 20 trials", zorder=4)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_ylim(5, 4e3)
for r in desc:
    a1.axvline(r["grid"], color=GRID, lw=0.8, zorder=1)
a1.axvline(cross20, color=C[2], lw=1.2, ls=(0, (4, 2)), zorder=3)
a1.text(cross20 * 1.4, 6.6, "predicted crossover\n$|G|^*=" +
        f"{cross20:,d}".replace(",", "\\,") + "$",
        fontsize=6.6, color=C[2], ha="left", va="bottom")
a1.set_xlabel(r"candidate-grid size $|G|$"); a1.set_ylabel("search time (ms, log)")
a1.legend(fontsize=7.5, loc="upper left"); a1.grid(zorder=0); despine(a1)
a1.set_xlim(5, 1e8)
a1.set_title("(a) Search cost", loc="left", fontsize=9, color=INK, pad=6)

a2.plot(G, np.maximum(ee, 1e-5), "-o", color=C[1], lw=1.8, ms=4.5, label="exhaustive batched sweep")
a2.plot(G, np.maximum(et, 1e-5), "-s", color=C[0], lw=1.8, ms=4.5, label="TPE, 20 trials")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel(r"candidate-grid size $|G|$")
a2.set_ylabel(r"attained $|\hat{p}-0.7|$ (log)")
a2.legend(fontsize=7.5, loc="lower left"); a2.grid(zorder=0); despine(a2)
a2.set_title("(b) Solution quality", loc="left", fontsize=9, color=INK, pad=6)
save(fig, "fig4_planner")

# ================================================================== Figure 8
rt = json.load(open(results("results_runtimes.json")))
cross = rt["_crossover"]
LAB = {"keras_predict": "Keras predict()", "keras_call": "Keras direct call",
       "onnxruntime": "ONNX Runtime", "numpy": "NumPy reference"}
COL = {"keras_predict": C[0], "keras_call": C[1], "onnxruntime": C[2], "numpy": C[3]}
MK = {"keras_predict": "s", "keras_call": "o", "onnxruntime": "^", "numpy": "D"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for k in ("keras_predict", "keras_call", "onnxruntime", "numpy"):
    v = rt[k]
    a1.plot(v["sizes"], v["times"], "-", marker=MK[k], color=COL[k], lw=1.6, ms=4.0,
            label=LAB[k])
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("candidates per invocation $n$")
a1.set_ylabel("invocation cost (ms, log)")
a1.legend(fontsize=7.0, loc="center left"); a1.grid(zorder=0); despine(a1)
a1.set_title("(a) Invocation cost under four runtimes", loc="left", fontsize=9, color=INK, pad=6)

Ts = [20, 50, 100, 200]
w = 0.2
xs = np.arange(len(Ts))
for i, k in enumerate(("keras_predict", "keras_call", "onnxruntime", "numpy")):
    vals = [cross[k][str(T)] for T in Ts]
    a2.bar(xs + (i - 1.5) * w, vals, width=w, color=COL[k], label=LAB[k], zorder=3)
a2.set_yscale("log")
a2.set_xticks(xs); a2.set_xticklabels([f"$T={T}$" for T in Ts])
a2.set_ylabel(r"predicted crossover $|G|^*$ (log)")
a2.axhline(2457000, color=INK, lw=0.9, ls=(0, (4, 2)))
a2.set_ylim(8e3, 6e7)
a2.text(3.45, 3.0e6, "largest catalogued descriptor", fontsize=6.4, color=INK, ha="right")
a2.legend(fontsize=6.6, loc="upper left", ncol=2); a2.grid(axis="y", zorder=0); despine(a2)
a2.set_title("(b) Predicted crossover by runtime", loc="left", fontsize=9, color=INK, pad=6)
save(fig, "fig8_runtimes")
