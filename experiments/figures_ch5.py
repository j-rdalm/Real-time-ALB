"""Additional thesis figures (Ch. 5) from the experiment results.

    fig_architecture        decision architecture of the real-time approach (diagram)
    fig_shift               accumulated exposure of each operator during one shift, three policies
                            (5 operators, seed 0, after 5 shifts of learning; runs the simulator)
    fig_sensitivity_panels  sensitivity analysis in four panels (gap reduction, minutes at REBA >= 4,
                            units per hour, plan changes), from results_sensitivity/summary.csv

Run from the repo root after the experiments:  python -m experiments.figures_ch5 [arch] [shift] [sens]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from experiments.figures import POL, INK, MUTED, GRID, LABELS     # same style as the other figures
from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

EXP = Path(__file__).resolve().parent
FIG = EXP.parent / "figures"
HIDDEN = {20: +3, 38: +4, 44: +2}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---------- 1. decision architecture ----------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.set_xlim(0, 10.45); ax.set_ylim(-0.25, 5.4); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.25)

    def arrow(p, q, text=None, dx=0.0, dy=0.0, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha="center", va="center",
                    fontsize=6.8, color=MUTED, style="italic")

    def path(pts, text=None, tx=None, ty=None):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], color=MUTED, lw=0.9)
        arrow(pts[-2], pts[-1])
        if text:
            ax.text(tx, ty, text, ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # data sources
    ax.text(1.2, 5.25, "Data", ha="center", fontsize=8.5, weight="bold", color=MUTED)
    box(0.1, 3.75, 2.2, 1.25, "Operators", "Borg CR-10 rating\nat each break", "#eef4fb")
    box(0.1, 2.15, 2.2, 1.25, "MotionSuit", "REBA score\nper task execution", "#eef4fb")
    box(0.1, 0.55, 2.2, 1.25, "Smartwatch", "energy per operator\nper cycle", "#eef4fb")
    # estimation
    ax.text(4.25, 5.25, "Every cycle", ha="center", fontsize=8.5, weight="bold", color=MUTED)
    box(3.05, 0.55, 2.4, 2.85, "Task estimates",
        "time, REBA and energy\nof each task\n(Eq. 5.1 and 5.2)\n\ninitial values as prior,\nupdated with each\nmeasurement", "#f4f3ef")
    # decisions
    ax.text(8.25, 5.25, "Decisions", ha="center", fontsize=8.5, weight="bold", color=MUTED)
    box(6.6, 3.75, 3.3, 1.25, "Every break: rotation",
        "which operator works on which\nloop in the next block\n(measured exposure, Borg rule)", "#fdf0e9")
    box(6.6, 2.15, 3.3, 1.25, "Start of shift: balancing",
        "plan for the headcount of the day;\nMILP solved again only when\nneeded", "#e9f6f0")
    box(6.6, 0.55, 3.3, 1.25, "Assembly line",
        "operators work on their loops", "#ffffff")

    arrow((2.3, 4.37), (6.6, 4.37), "Borg ratings", dy=0.14)
    arrow((2.3, 2.77), (3.05, 2.77))
    arrow((2.3, 1.17), (3.05, 1.17))
    arrow((5.45, 3.1), (6.6, 4.0), "exposure\nforecast", dx=-0.15, dy=0.3)
    arrow((5.45, 2.6), (6.6, 2.6), "estimates", dx=-0.05, dy=-0.17)
    arrow((7.6, 2.15), (7.6, 1.8), "loops", dx=0.32)
    path([(9.9, 4.37), (10.2, 4.37), (10.2, 1.17), (9.9, 1.17)])
    ax.text(10.33, 2.77, "assignment", rotation=90, ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    path([(8.25, 0.55), (8.25, 0.05), (1.2, 0.05), (1.2, 0.55)], "new measurements every cycle", 4.25, -0.12)
    save(fig, "fig_architecture")


# ---------- 2. cumulative exposure in one shift ----------
def fig_shift(seed=0, m=5, warmup=5):
    line = load_bosch()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    for ax, pol in zip(axes, ("static", "rotation", "realtime")):
        sim = LineSimulator(line.tasks, SimParams(reba_shift=HIDDEN), n_people=8, seed=seed)
        ctl = Controller(line, sim, pol)
        for d in range(1, warmup + 1):
            ctl.run_shift(d, m)
        ctl.trace = []
        ctl.run_shift(warmup + 1, m)
        cyc = [ev for ev in ctl.trace if ev["type"] == "cycle"]
        ops = sorted(cyc[0]["cum"], key=lambda s: int(s[1:]))
        t = np.array([0.0] + [ev["t_s"] / 3600 for ev in cyc])
        finals = []
        for k, o in enumerate(ops):
            y = np.array([0.0] + [ev["cum"][o] for ev in cyc])
            ax.plot(t, y, lw=1.1, color=plt.cm.viridis(k / (len(ops) - 1) * 0.85))
            finals.append(y[-1])
        for b in (2.5, 5.0):
            ax.axvline(b, color=MUTED, lw=0.7, ls=":")
        gap = max(finals) - min(finals)
        ax.set_title(f"{POL[pol][0]}\nfinal gap {gap:.0f}", loc="left", color=INK)
        ax.set_xlabel("Time in shift (h)")
        ax.set_xticks([0, 2.5, 5, 7.5])
        print(pol, "final cum per operator", [round(f, 1) for f in finals], "gap", round(gap, 1))
    axes[0].set_ylabel("Accumulated exposure\n(exposure-minutes)")
    fig.tight_layout()
    save(fig, "fig_shift")


# ---------- 4. sensitivity, four panels ----------
def fig_sensitivity_panels():
    s = pd.read_csv(EXP / "results_sensitivity" / "summary.csv", index_col=0)
    s = s.loc[[c for c in LABELS if c in s.index]]
    y = np.arange(len(s))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.4, 1.1, 1.0, 1.0]})
    panels = [
        ("(a) Reduction of the\nexposure gap (%)", [("red_rotation_%", "rotation"), ("red_realtime_%", "realtime")], (0, 100)),
        ("(b) Min/h at REBA ≥ 4,\nmost exposed", [("reba4_rotation", "rotation"), ("reba4_min_per_h_max", "realtime")], None),
        ("(c) Units per hour", [("units_per_hour", "realtime")], None),
        ("(d) Plan changes\nin 20 shifts", [("rebalances", "realtime")], None),
    ]
    for ax, (title, series, xlim) in zip(axes, panels):
        for k, (col, pol) in enumerate(series):
            lab, c, mk = POL[pol]
            off = 0.0 if len(series) == 1 else (0.17 if k == 0 else -0.17)
            ax.errorbar(s[f"{col}_mean"], y + off, xerr=s[f"{col}_std"], fmt=mk, color=c, ms=4, lw=0.9,
                        capsize=0, label=lab)
        ax.axhspan(y[0] - 0.5, y[0] + 0.5, color=GRID, alpha=0.6, zorder=0)
        ax.axvline(s.loc["base", f"{series[-1][0]}_mean"], color=MUTED, lw=0.6, ls=":", zorder=0)
        ax.set_title(title, loc="left", color=INK, fontsize=8)
        ax.grid(axis="y", visible=False)
        if xlim:
            ax.set_xlim(*xlim)
    axes[0].set_yticks(y, [LABELS[i] for i in s.index])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.6)
    save(fig, "fig_sensitivity_panels")


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "shift", "sens"]
    if "arch" in which: fig_architecture()
    if "shift" in which: fig_shift()
    if "sens" in which: fig_sensitivity_panels()
