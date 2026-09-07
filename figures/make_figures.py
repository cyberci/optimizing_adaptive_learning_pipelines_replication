
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# validated categorical palette (dataviz six-checks, light surface): all PASS
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


# ---------------------------------------------------------------- Fig 3
lat = json.load(open(results("results_latency.json")))
d = lat["decomposition"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

variants = ["V0", "V1", "V2", "V3", "V4"]
labels = ["V0 as implemented", "V1 +model cache", "V2 +warm-up", "V3 +direct call", "V4 +planner"]
p50 = [lat[v]["p50"] for v in variants]
p95 = [lat[v]["p95"] for v in variants]
y = np.arange(len(variants))[::-1]
a1.barh(y, p50, height=0.55, color=[C[0]] * 3 + [C[2], C[1]], zorder=3)
a1.errorbar(p50, y, xerr=[[0] * 5, np.array(p95) - np.array(p50)], fmt="none",
            ecolor=MUTED, elinewidth=0.9, capsize=2.5, zorder=4)
a1.set_xscale("log"); a1.set_xlim(4, 12000)
a1.set_yticks(y); a1.set_yticklabels(labels)
a1.axvline(600, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
a1.text(560, -0.62, "600 ms budget", fontsize=6.8, color=INK, ha="right")
for yy, v in zip(y, p50):
    a1.text(v * 1.15, yy, f"{v:,.0f} ms", va="center", fontsize=7.5, color=INK)
a1.set_xlabel("recommendation latency, p50 with p95 whisker (ms, log)")
a1.grid(axis="x", zorder=0); despine(a1)
a1.set_title("(a) Latency of the five variants", loc="left", fontsize=9, color=INK, pad=6)

comp = [("model load / request", d["model_load_ms"], C[0]),
        ("predict() per cand.", d["predict_api_per_eval_ms"], C[0]),
        ("direct call per cand.", d["direct_call_per_eval_ms"], C[2]),
        ("batched per cand.", d["batch80_per_eval_ms"], C[1]),
        ("TPE bookkeeping (20)", d["tpe_overhead_20trials_ms"], MUTED)]
x = np.arange(len(comp))
a2.bar(x, [c[1] for c in comp], width=0.6, color=[c[2] for c in comp], zorder=3)
a2.set_yscale("log"); a2.set_ylim(0.05, 400)
a2.set_xticks(x); a2.set_xticklabels([c[0] for c in comp], fontsize=6.4, rotation=28, ha="right")
for xx, c in zip(x, comp):
    a2.text(xx, c[1] * 1.25, f"{c[1]:.2f}" if c[1] < 10 else f"{c[1]:.0f}",
            ha="center", fontsize=7.5, color=INK)
a2.set_ylabel("cost (ms, log)")
a2.grid(axis="y", zorder=0); despine(a2)
a2.set_title("(b) Cost decomposition", loc="left", fontsize=9, color=INK, pad=6)
save(fig, "fig3_latency")

# ---------------------------------------------------------------- Fig 5
con = json.load(open(results("results_concurrency.json")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for tag, col, mk, lab in [("V4", C[1], "o", "V4 descriptor-driven planner"),
                          ("V0", C[0], "s", "V0 as implemented")]:
    rows = con[tag]
    c = [r["concurrency"] for r in rows]
    a1.plot(c, [r["throughput_rps"] for r in rows], "-", marker=mk, color=col, lw=1.8, ms=4.5, label=lab)
    a2.plot(c, [r["p50"] for r in rows], "-", marker=mk, color=col, lw=1.8, ms=4.5, label=lab)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("concurrent learners"); a1.set_ylabel("throughput (req/s, log)")
a1.legend(fontsize=7.5, loc="center left"); a1.grid(zorder=0); despine(a1)
a1.set_title("(a) Throughput on 2 vCPU", loc="left", fontsize=9, color=INK, pad=6)

a2.set_xscale("log"); a2.set_yscale("log")
a2.axhline(600, color=INK, lw=0.9, ls=(0, (4, 2)))
a2.axhline(3000, color=INK, lw=0.9, ls=(0, (1, 2)))
a2.text(115, 660, "600 ms synchronous budget", fontsize=6.6, color=INK, ha="right")
a2.text(115, 3300, "3 s asynchronous polling deadline", fontsize=6.6, color=INK, ha="right")
a2.axvspan(28, 34, color=C[2], alpha=0.10, zorder=0)
a2.text(31, 8, "cohort\n(30)", fontsize=6.8, color=C[2], ha="center")
a2.set_xlabel("concurrent learners"); a2.set_ylabel("median latency (ms, log)")
a2.legend(fontsize=7.2, loc="upper left"); a2.grid(zorder=0); despine(a2); a2.set_xlim(0.85, 150)
a2.set_title("(b) Latency against the two deadlines", loc="left", fontsize=9, color=INK, pad=6)
save(fig, "fig5_concurrency")

# ---------------------------------------------------------------- Fig 7
deg = json.load(open(results("results_degradation.json")))["B_corrected"]
a = np.array([r["availability"] for r in deg]) * 100
fb = np.array([r["flow_band"] for r in deg]) * 100
ci = np.array([r["ci95"] for r in deg]) * 100
o = json.load(open(results("results_tiers.json")))
orc = [r for r in o if r["mode"] == "oracle"][0]["flow_band"] * 100
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.fill_between(a, fb - ci, fb + ci, color=C[1], alpha=0.16, lw=0, zorder=2)
ax.plot(a, fb, "-o", color=C[1], lw=1.9, ms=5, zorder=4, label="delivered (corrected ensemble)")
ax.axhline(orc, color=C[2], lw=1.4, ls=(0, (4, 2)), zorder=3,
           label="architectural upper bound (exact model)")
ax.set_xlabel("availability of the ML tier T1 (%)")
ax.set_ylabel("flow-band retention (%)")
ax.set_ylim(0, 100); ax.set_xlim(-3, 103)
ax.legend(fontsize=7.2, loc="center left")
ax.grid(zorder=0); despine(ax)
save(fig, "fig7_degradation")
print("all figures done")
