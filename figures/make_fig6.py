"""Rebuilds Figure 6: the tier ladder, now including the jointly corrected
predictor, and the sweep of the ensemble weight for both training protocols."""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

FIG = str(FIGURES)

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


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


ti = {r["mode"]: r for r in json.load(open(results("results_tiers.json")))}
al = json.load(open(results("results_tiers_alpha.json")))
A = {(r["tag"], r["alpha"]): r for r in al}

bars = [("oracle", "T1 planner + exact response model\n(architectural upper bound)",
         ti["oracle"], C[2]),
        ("joint", "T1 planner + both choices corrected\n(labels, $\\alpha=1$)",
         A[("B_corrected", 1.0)], C[1]),
        ("labels", "T1 planner + corrected labels only\n(default $\\alpha=0.5$)",
         A[("B_corrected", 0.5)], C[1]),
        ("default", "T4 static descriptor defaults", ti["default"], MUTED),
        ("rule", "T2 rule-based ($\\pm$ one parameter)", ti["rule"], MUTED),
        ("implemented", "T1 planner + predictor as implemented\n(default $\\alpha=0.5$)",
         A[("A_as_deployed", 0.5)], C[0])]

fig, (ax, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
y = np.arange(len(bars))[::-1]
vals = [b[2]["flow_band"] * 100 for b in bars]
err = [b[2]["ci95"] * 100 for b in bars]
ax.barh(y, vals, height=0.62, color=[b[3] for b in bars], zorder=3)
ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
ax.set_yticks(y); ax.set_yticklabels([b[1] for b in bars], fontsize=6.9)
for yy, v, e in zip(y, vals, err):
    ax.text(v + e + 1.8, yy, f"{v:.1f}%", va="center", fontsize=7.4, color=INK)
ax.axvline(ti["oracle"]["flow_band"] * 100, color=C[2], lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.set_xlim(0, 112)
ax.set_xlabel("interactions inside the 0.6\u20130.8 flow band (%)")
ax.grid(axis="x", zorder=0); despine(ax)
ax.set_title("(a) Flow-band retention by predictor configuration", loc="left", fontsize=9, color=INK, pad=6)

for tag, col, mk, lab in [("B_corrected", C[1], "o", "corrected labels"),
                          ("A_as_deployed", C[0], "s", "labels as implemented")]:
    rs = sorted([r for r in al if r["tag"] == tag], key=lambda r: r["alpha"])
    x = [r["alpha"] for r in rs]
    v = [r["flow_band"] * 100 for r in rs]
    c = [r["ci95"] * 100 for r in rs]
    a2.errorbar(x, v, yerr=c, fmt="-", marker=mk, color=col, lw=1.8, ms=4.5,
                elinewidth=0.9, capsize=2.5, label=lab, zorder=4)
a2.axhline(ti["oracle"]["flow_band"] * 100, color=C[2], lw=1.2, ls=(0, (4, 2)), zorder=3)
a2.text(0.03, 96, "architectural upper bound", fontsize=6.8, color=C[2], va="bottom")
a2.axvline(0.5, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=2)
a2.text(0.47, 4, "class default", fontsize=6.8, color=INK, va="bottom", ha="right")
a2.set_xlabel("abilities-network weight " + r"$\alpha$", fontsize=8.2)
a2.set_ylabel("flow-band retention (%)")
a2.set_ylim(0, 105); a2.set_xlim(-0.04, 1.04)
a2.legend(fontsize=7.2, loc="upper left", bbox_to_anchor=(0.02, 0.88))
a2.grid(zorder=0); despine(a2)
a2.set_title("(b) Retention against the ensemble weight $\\alpha$", loc="left", fontsize=9, color=INK, pad=6)
save(fig, "fig6_tiers")
