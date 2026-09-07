
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
from repro_paths import (MODELS, RESULTS, FIGURES, WORK, DATA,   # noqa: E402
                         results, model, figure, work, data)     # noqa: E402

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

C = ["#2563EB", "#EA580C", "#0D9488", "#9333EA"]
INK, MUTED, GRID, SURF = "#1a1a1a", "#5c5c5c", "#d8d8d6", "#f6f6f4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
FIG = str(FIGURES)


def box(ax, x, y, w, h, text, fc=SURF, ec=MUTED, fs=7.6, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, style="-|>", color=MUTED, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=color, lw=lw, ls=ls,
                                 mutation_scale=8, shrinkA=1, shrinkB=1, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


# ================================================= Fig 1: architecture
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.add_patch(Rectangle((0.005, 0.06), 0.20, 0.86, fc="none", ec=GRID, lw=0.8, ls=(0, (3, 2))))
ax.text(0.105, 0.945, "clients", fontsize=7.2, color=MUTED, ha="center")
box(ax, 0.02, 0.62, 0.17, 0.20, "Android client\n(Kotlin / Compose)\n8 parameterised games", fs=7.0)
box(ax, 0.02, 0.36, 0.17, 0.18, "Web client\n(Angular)\nteacher / researcher", fs=7.0)
box(ax, 0.02, 0.12, 0.17, 0.16, "OAuth2 / OIDC\nauth server", fs=7.0)

ax.add_patch(Rectangle((0.245, 0.06), 0.40, 0.86, fc="none", ec=GRID, lw=0.8, ls=(0, (3, 2))))
ax.text(0.445, 0.945, "resource server  (Spring Boot)",
        fontsize=7.2, color=MUTED, ha="center")
box(ax, 0.26, 0.70, 0.37, 0.14, "recommendation controller\nasync hand-off  +  renderable-game negotiation",
    fc="#eef4ff", ec=C[0], fs=7.0)
box(ax, 0.26, 0.26, 0.37, 0.37, "", fc="#fff", ec=C[1])
ax.text(0.445, 0.595, "degradation chain", fontsize=7.0,
        color=C[1], ha="center", fontweight="bold")
for i, (lbl, col) in enumerate([("T1  machine-learning optimiser", C[1]),
                                ("T2  rule tier  ($\\pm$ one increment)", MUTED),
                                ("T3  replay of the last configuration", MUTED),
                                ("T4  descriptor initial values", MUTED)]):
    box(ax, 0.275, 0.495 - i * 0.062, 0.34, 0.046, lbl, fc=SURF, ec=col, fs=6.5)
box(ax, 0.26, 0.09, 0.175, 0.14, "profile updater\nincremental + batch\nz-recalibration", fs=6.8)
box(ax, 0.455, 0.09, 0.175, 0.14, "relational store\ntask descriptors\n25 config items", fs=6.8)

ax.add_patch(Rectangle((0.695, 0.06), 0.30, 0.86, fc="none", ec=GRID, lw=0.8, ls=(0, (3, 2))))
ax.text(0.845, 0.945, "recommendation service (FastAPI)", fontsize=7.2, color=MUTED, ha="center")
box(ax, 0.710, 0.66, 0.270, 0.18, "descriptor-driven planner\n$|G| \\leq$ budget: exhaustive batched\notherwise: TPE",
    fc="#fff4ec", ec=C[1], fs=6.8)
box(ax, 0.710, 0.40, 0.270, 0.20, "ensemble predictor\nM1 ability MLP 64-32-1\nM2 history MLP (dual branch)\nconvex weight alpha", fs=6.9)
box(ax, 0.710, 0.14, 0.270, 0.18, "speculative profile update\nabilities_if_success /\nabilities_if_failure", fs=6.9)

arrow(ax, (0.19, 0.74), (0.26, 0.77))
arrow(ax, (0.19, 0.45), (0.26, 0.72))
arrow(ax, (0.445, 0.70), (0.445, 0.635))
arrow(ax, (0.632, 0.518), (0.710, 0.74), rad=-0.15, color=C[1], lw=1.2)
arrow(ax, (0.710, 0.69), (0.632, 0.486), rad=-0.15, color=C[1], lw=1.2)
arrow(ax, (0.845, 0.66), (0.845, 0.60), color=MUTED)
arrow(ax, (0.845, 0.40), (0.845, 0.32), color=MUTED)
arrow(ax, (0.710, 0.20), (0.632, 0.17), rad=0.10, color=MUTED, ls=(0, (3, 2)))

ax.plot([0.665,0.665],[0.06,0.90], color=MUTED, lw=0.8, ls=(0,(1,3)))
ax.text(0.672, 0.30, "process boundary", fontsize=6.2, color=MUTED, ha="left", rotation=90, va="center")
fig.savefig(f"{FIG}/fig1_architecture.pdf"); fig.savefig(f"{FIG}/fig1_architecture.png")
plt.close(fig); print("wrote fig1")

# ================================================= Fig 2: async sequence
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
lanes = [("game client", 0.10), ("resource server", 0.40), ("recommendation\nservice", 0.70), ("descriptor\nstore", 0.93)]
for name, x in lanes:
    box(ax, x - 0.085, 0.905, 0.17, 0.075, name, fc=SURF, fs=7.0, bold=True)
    ax.plot([x, x], [0.06, 0.90], color=GRID, lw=0.8, zorder=1)

def msg(y, x0, x1, text, col=MUTED, ls="-", up=True):
    arrow(ax, (x0, y), (x1, y), color=col, ls=ls, lw=1.0)
    ax.text((x0 + x1) / 2, y + (0.018 if up else -0.030), text, ha="center",
            fontsize=6.6, color=INK)

msg(0.860, 0.10, 0.40, "POST /result  (outcome of the finished task)")
msg(0.800, 0.40, 0.10, "202  {recommendedGameId}   <- returns immediately", col=C[0])
ax.text(0.115, 0.752, "learner keeps playing / sees the outcome screen",
        fontsize=6.6, color=C[0], ha="left", style="italic")

ax.add_patch(Rectangle((0.375, 0.30), 0.05, 0.40, fc="#fff4ec", ec=C[1], lw=0.9, zorder=2))
ax.text(0.352, 0.50, "degradation\nchain", fontsize=6.4, color=C[1], ha="right", va="center")
msg(0.660, 0.425, 0.70, "T1  POST /games/{id}/suggest", col=C[1])
msg(0.575, 0.70, 0.425, "suggested_params, success_rate,\nabilities_if_success / _if_failure", col=C[1], ls=(0, (3, 2)), up=False)
ax.text(0.435, 0.44, "on exception or empty result: T2 rules $\\rightarrow$ T3 replay $\\rightarrow$ T4 defaults",
        fontsize=6.4, color=MUTED, ha="left")
msg(0.360, 0.42, 0.93, "persist config + profile-update rows")

ax.add_patch(Rectangle((0.085, 0.055), 0.03, 0.25, fc="#eef4ff", ec=C[0], lw=0.9, zorder=2))
for k, y in enumerate([0.265, 0.205, 0.145]):
    msg(y, 0.115, 0.38, f"GET /config/{{id}}   poll {k+1}", ls=(0, (2, 2)))
ax.text(0.42, 0.20, "bounded polling: $10\\times300$ ms\n= 3 s hard deadline",
        fontsize=6.4, color=C[0], ha="left", va="center")
msg(0.075, 0.40, 0.10, "config + distraction set + XP", col=C[2])
fig.savefig(f"{FIG}/fig2_sequence.pdf"); fig.savefig(f"{FIG}/fig2_sequence.png")
plt.close(fig); print("wrote fig2")
