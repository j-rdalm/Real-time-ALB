"""Figures for the thesis (PDF for LaTeX + PNG preview) from the experiment results.

    fig_estimates     estimates of the three hidden-error tasks over the observations (trimmed mean vs median)
    fig_sensitivity   reduction of the exposure gap per sensitivity scenario (offline vs real-time rotation)
    fig_regret        regret of the plan in use when the Bosch data has errors (Bosch plan vs learned plan)
    fig_workforce     KPIs by headcount (4..7 operators), three policies

Run from the repo root after the experiments:  python -m experiments.figures
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

EXP = Path(__file__).resolve().parent
FIG = EXP.parent / "figures"
# validated categorical slots 1-3 (all-pairs CVD-safe); markers give a second, colour-free cue
POL = {"realtime": ("Real-time", "#2a78d6", "o"), "rotation": ("Offline rotation", "#eb6834", "s"),
       "static": ("No rotation", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---------- 1. estimates of the hidden-error tasks ----------
def estimate_traces():
    cache = EXP / "results_v1" / "traces_hidden_tasks.csv"
    if cache.exists():
        return pd.read_csv(cache)
    line = load_bosch()
    hidden = {20: +3, 38: +4, 44: +2}
    headcount = np.random.default_rng(1007).integers(4, 8, 20)
    rows = []
    for central in ("trimmed", "median"):
        sim = LineSimulator(line.tasks, SimParams(reba_shift=hidden), n_people=8, seed=7)
        ctl = Controller(line, sim, "realtime", estimator_central=central)

        def hook(c):
            for i in hidden:
                rows.append({"central": central, "task": i, "n": c.est.n[i],
                             "estimate": c.est.estimate_task(i, "reba")})
        ctl.on_cycle = hook
        for d, m in enumerate(headcount, 1):
            ctl.run_shift(d, int(m))
    df = pd.DataFrame(rows).drop_duplicates(["central", "task", "n"])
    df.to_csv(cache, index=False)
    return df


def fig_estimates():
    df = estimate_traces()
    prior = load_bosch().tasks
    hidden = {20: 3, 38: 4, 44: 2}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), sharex=True)
    for ax, (i, d) in zip(axes, hidden.items()):
        bosch, true = prior.at[i, "reba"], prior.at[i, "reba"] + d
        ax.axhline(true, color=INK, lw=1.0, label="True value")
        ax.axhline(bosch, color=MUTED, lw=1.0, ls="--", label="Initial (observed) value")
        for central, col, lab in (("median", "#eb6834", "Median"), ("trimmed", "#2a78d6", "Trimmed mean")):
            s = df[(df.task == i) & (df.central == central)]
            ax.plot(s.n, s.estimate, color=col, lw=1.2, label=lab)
        ax.set_title(f"Task {i}", loc="left", color=INK)
        ax.set_xlabel("Observations (cycles)")
        ax.set_ylim(bosch - 0.8, true + 1.6)
    axes[0].set_ylabel("REBA estimate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    save(fig, "fig_estimates")


# ---------- 2. sensitivity ----------
LABELS = {
    "base": "Base scenario", "time_cv=0.025": "Time variation 2.5%", "time_cv=0.1": "Time variation 10%",
    "reba_sd=0.5": "REBA noise 0.5", "reba_sd=1.5": "REBA noise 1.5", "watch=0.05": "Watch error 5%",
    "watch=0.25": "Watch error 25%", "borg_sd=0.5": "Borg noise 0.5", "borg_sd=1.5": "Borg noise 1.5",
    "fatigue=moderate": "Fatigue moderate", "fatigue=high": "Fatigue high", "reba_thr=3": "REBA threshold 3",
    "no_skills": "No skills/restrictions", "borg_weight=0.5": "Borg in objective (0.5)",
    "borg_weight=1.0": "Borg in objective (1.0)", "min_gain=0.0": "Min gain 0%", "min_gain=0.05": "Min gain 5%",
    "drift=0.02": "Drift threshold 2%", "drift=0.1": "Drift threshold 10%",
}


def fig_sensitivity():
    s = pd.read_csv(EXP / "results_sensitivity" / "summary.csv", index_col=0)
    s = s.loc[[c for c in LABELS if c in s.index]]
    y = np.arange(len(s))[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    for k, (col, pol) in enumerate((("red_rotation_%", "rotation"), ("red_realtime_%", "realtime"))):
        lab, c, mk = POL[pol]
        off = 0.17 if k == 0 else -0.17
        ax.errorbar(s[f"{col}_mean"], y + off, xerr=s[f"{col}_std"], fmt=mk, color=c, ms=5, lw=1.0,
                    capsize=0, label=lab)
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, color=GRID, alpha=0.6, zorder=0)
    ax.set_yticks(y, [LABELS[i] for i in s.index])
    ax.set_xlabel("Reduction of the exposure gap between operators vs no rotation (%, mean ± SD, 10 seeds)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_sensitivity")


# ---------- 3. regret ----------
def fig_regret():
    """Bars = mean over the 20 seeds with errors; dots = each seed (the distribution is skewed: most
    seeds have zero regret, a few have more)."""
    r = pd.read_csv(EXP / "results_error_test" / "runs.csv")
    r = r[r.with_errors]
    ms = [4, 5, 6, 7]
    x = np.arange(len(ms))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for k, pol in enumerate(("static", "realtime")):
        lab, c, _ = POL[pol]
        pos = x + (k - 0.5) * 0.36
        vals = [r[f"regret_{pol}_m{m}_%"].to_numpy() for m in ms]
        ax.bar(pos, [v.mean() for v in vals], width=0.34, color=c, alpha=0.85,
               label="Plan from Bosch values" if pol == "static" else "Plan learned in real time")
        for p_, v in zip(pos, vals):
            ax.scatter(p_ + rng.uniform(-0.09, 0.09, len(v)), v, s=7, color=INK, alpha=0.55,
                       linewidths=0, zorder=3)
    ax.set_xticks(x, [f"{m} operators" for m in ms])
    ax.set_ylabel("Regret (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_regret")


# ---------- 4. workforce ----------
def fig_workforce():
    s = pd.read_csv(EXP / "results_workforce" / "summary.csv")
    panels = [("units_per_hour", "Units per hour"), ("exposure_range", "Exposure gap between operators\n(exposure-minutes per hour)"),
              ("exposure_per_h_max", "Exposure of the most exposed\noperator (exposure-minutes per hour)"),
              ("reba4_min_per_h_max", "Minutes per hour at REBA ≥ 4\n(most exposed operator)")]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for pol in ("static", "rotation", "realtime"):
            lab, c, mk = POL[pol]
            d = s[s.policy == pol].sort_values("m")
            ax.errorbar(d.m, d[f"{col}_mean"], yerr=d[f"{col}_std"], color=c, marker=mk, ms=5, capsize=0,
                        lw=1.4, elinewidth=0.8, label=lab)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks([4, 5, 6, 7])
    for ax in axes[1]:
        ax.set_xlabel("Operators on the shift")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_workforce")


if __name__ == "__main__":
    for f in (fig_estimates, fig_sensitivity, fig_regret, fig_workforce):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
