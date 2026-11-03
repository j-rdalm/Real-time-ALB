"""Sensitivity analysis (one parameter at a time, 10 seeds per configuration).

Scenario = E1: Bosch line, 20 shifts, headcount 4..7 drawn per day, three tasks whose real REBA is
higher than the Bosch value. For each configuration and seed the three policies are run.
Configurations: the base scenario (Ch. 3-aligned), the simulator parameters and fatigue, the REBA
exposure threshold, skills/medical restrictions off, the stability rule of the rebalancing (min gain,
drift threshold) and the Borg weight in the rotation objective.

Run from the repo root:  python -m experiments.sensitivity  [--seeds 10] [--workers 2]
"""
import argparse, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

OUT = Path(__file__).resolve().parent / "results_sensitivity"
HIDDEN = {20: +3, 38: +4, 44: +2}
N_DAYS, MIN_OPS, MAX_OPS = 20, 4, 7
FATIGUE = {"moderate": dict(fatigue_time=0.06, fatigue_reba=0.6, fatigue_energy=0.05),
           "high": dict(fatigue_time=0.12, fatigue_reba=1.2, fatigue_energy=0.10)}

# (name, parameter, level, SimParams overrides, Controller overrides, policies)
ALL = ("static", "rotation", "realtime")
CONFIGS = [("base", "-", "base", {}, {}, ALL)]
for lvl in (0.025, 0.10):
    CONFIGS.append((f"time_cv={lvl}", "time variation (SD)", lvl, {"time_cv": lvl}, {}, ALL))
for lvl in (0.5, 1.5):
    CONFIGS.append((f"reba_sd={lvl}", "REBA noise (SD)", lvl, {"reba_sd": lvl}, {}, ALL))
for lvl in (0.05, 0.25):
    CONFIGS.append((f"watch={lvl}", "watch error (SD)", lvl, {"watch_error": lvl}, {}, ALL))
for lvl in (0.5, 1.5):
    CONFIGS.append((f"borg_sd={lvl}", "Borg noise (SD)", lvl, {"borg_sd": lvl}, {}, ALL))
for lvl, kw in FATIGUE.items():
    CONFIGS.append((f"fatigue={lvl}", "fatigue", lvl, kw, {}, ALL))
CONFIGS.append(("reba_thr=3", "REBA exposure threshold", 3, {}, {"exposure_thresholds": {"reba": 3, "borg": 2}}, ALL))
CONFIGS.append(("no_skills", "skills and medical restrictions", "none", {"skill_prob": 1.0, "medical_share": 0.0}, {}, ALL))
# Borg term in the rotation objective (alternative to the base, where Borg is only a constraint)
for lvl in (0.5, 1.0):
    CONFIGS.append((f"borg_weight={lvl}", "Borg weight in rotation", lvl, {}, {"borg_weight": lvl}, ("realtime",)))
# the stability rule only affects the realtime policy
for lvl in (0.0, 0.05):
    CONFIGS.append((f"min_gain={lvl}", "min gain to rebalance", lvl, {}, {"min_gain": lvl}, ("realtime",)))
for lvl in (0.02, 0.10):
    CONFIGS.append((f"drift={lvl}", "drift threshold", lvl, {}, {"drift_threshold": lvl}, ("realtime",)))

_LINE = None


def _line():
    global _LINE
    if _LINE is None:
        _LINE = load_bosch()
    return _LINE


def run_one(job):
    name, param, level, sim_kw, ctl_kw, policy, seed = job
    line = _line()
    headcount = np.random.default_rng(1000 + seed).integers(MIN_OPS, MAX_OPS + 1, N_DAYS)
    sim = LineSimulator(line.tasks, SimParams(reba_shift=HIDDEN, **sim_kw), n_people=MAX_OPS + 1, seed=seed)
    ctl = Controller(line, sim, policy, **ctl_kw)
    for d, m in enumerate(headcount, 1):
        ctl.run_shift(d, int(m))
    df = pd.DataFrame(ctl.log_shifts)
    row = {"config": name, "parameter": param, "level": level, "policy": policy, "seed": seed,
           "rebalances": int(df["rebalanced"].sum()),
           "relaxed_borg_rule": int(df["relaxed_borg_rule"].sum()),
           "relaxed_eligibility": int(df["relaxed_eligibility"].sum())}
    for k in ("units_per_hour", "line_efficiency", "exposure_per_h_max", "exposure_range",
              "exposure_gini", "reba4_min_per_h_max", "borg_max"):
        row[k] = float(df[k].mean())
    if policy == "realtime":
        est = ctl.est.estimates()
        row["hidden_reba_abs_error"] = float(np.mean([abs(est.at[i, "reba"] - sim.truth.at[i, "reba"]) for i in HIDDEN]))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-realtime", action="store_true",
                    help="rerun only the realtime policy and reuse static/rotation runs from runs.csv "
                         "(valid when a change affects only the realtime policy)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = [(c[0], c[1], c[2], c[3], c[4], pol, s) for c in CONFIGS for pol in c[5] for s in range(a.seeds)
            if not (a.only_realtime and pol != "realtime")]
    old = pd.read_csv(OUT / "runs.csv") if a.only_realtime else None
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    runs = pd.DataFrame(rows)
    if old is not None:
        runs = pd.concat([old[old.policy != "realtime"], runs], ignore_index=True)
    runs.to_csv(OUT / "runs.csv", index=False)
    # policies not affected by the stability rule: reuse the base static/rotation runs of the same seed
    base_other = runs[(runs.config == "base") & (runs.policy != "realtime")]
    extra = []
    for c in CONFIGS:
        if c[5] == ("realtime",):
            extra.append(base_other.assign(config=c[0], parameter=c[1], level=c[2]))
    full = pd.concat([runs] + extra, ignore_index=True)
    piv = full.pivot_table(index=["config", "seed"], columns="policy", values="exposure_range")
    red = pd.DataFrame({"red_rotation_%": 100 * (1 - piv["rotation"] / piv["static"]),
                        "red_realtime_%": 100 * (1 - piv["realtime"] / piv["static"])}).reset_index()
    rt = full[full.policy == "realtime"][["config", "seed", "units_per_hour", "reba4_min_per_h_max",
                                          "rebalances", "hidden_reba_abs_error", "relaxed_borg_rule",
                                          "relaxed_eligibility"]]
    rot = full[full.policy == "rotation"][["config", "seed", "reba4_min_per_h_max"]].rename(
        columns={"reba4_min_per_h_max": "reba4_rotation"})
    m = red.merge(rt, on=["config", "seed"]).merge(rot, on=["config", "seed"])
    order = [c[0] for c in CONFIGS]
    summ = m.groupby("config").agg(["mean", "std"]).drop(columns="seed", level=0).reindex(order)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.round(3).to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    show = summ[["red_rotation_%_mean", "red_rotation_%_std", "red_realtime_%_mean", "red_realtime_%_std",
                 "reba4_rotation_mean", "reba4_min_per_h_max_mean", "rebalances_mean", "hidden_reba_abs_error_mean",
                 "relaxed_borg_rule_mean", "relaxed_eligibility_mean"]]
    print(show.round(2).to_string())
    print(f"\n{len(jobs)} runs in {time.time() - t0:.0f} s")
