"""Experiment v1 on the Bosch line.

E1  Policies over 20 shifts with a headcount that changes every day (4..7 operators) and a hidden
    error in the Bosch data: three tasks have a higher real REBA than the video-based value.
E2  Workforce variation: the same line run with 4, 5, 6 and 7 operators (realtime policy, after a
    warm-up), with every KPI expressed per hour so that days with different headcount compare.

Run from the repo root:
    python -m experiments.run_v1                                   # Bosch line
    python -m experiments.run_v1 --dataset other.xlsx --hidden none --min-ops 3 --max-ops 6
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

OUT = Path(__file__).resolve().parent / "results_v1"
OUT.mkdir(exist_ok=True)
HIDDEN = {20: +3, 38: +4, 44: +2}       # real REBA higher than the Bosch (video) value
N_DAYS, SEED = 20, 7
MIN_OPS, MAX_OPS = 4, 7


def e1(line):
    rng = np.random.default_rng(SEED)
    headcount = rng.integers(MIN_OPS, MAX_OPS + 1, N_DAYS)   # one headcount per day
    rows, blocks, est_err = [], [], []
    for policy in ("static", "rotation", "realtime"):
        sim = LineSimulator(line.tasks, SimParams(reba_shift=HIDDEN), n_people=MAX_OPS + 1, seed=SEED)
        ctl = Controller(line, sim, policy)
        for d, m in enumerate(headcount, 1):
            rows.append(ctl.run_shift(d, int(m)))
            if policy == "realtime":
                est = ctl.est.estimates()
                for i in HIDDEN:
                    est_err.append({"day": d, "task": i, "bosch": line.tasks.at[i, "reba"],
                                    "true": sim.truth.at[i, "reba"], "estimate": est.at[i, "reba"]})
        blocks += ctl.log_blocks
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E1_shifts.csv", index=False)
    pd.DataFrame(blocks).to_csv(OUT / "E1_blocks.csv", index=False)
    pd.DataFrame(est_err).to_csv(OUT / "E1_estimates_hidden_tasks.csv", index=False)
    return df


def e2(line, shifts_per_m=5):
    sim = LineSimulator(line.tasks, SimParams(reba_shift=HIDDEN), n_people=MAX_OPS + 1, seed=SEED + 1)
    ctl = Controller(line, sim, "realtime")
    for d in range(1, 6):                               # warm-up so that the estimates have converged
        ctl.run_shift(-d, (MIN_OPS + MAX_OPS + 1) // 2)   # middle headcount (6 for 4..7)
    rows = []
    for m in range(MIN_OPS, MAX_OPS + 1):
        for k in range(shifts_per_m):
            rows.append(ctl.run_shift(k + 1, m))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E2_workforce.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="xlsx with sheets Tasks and Precedence (default: Bosch line in Fuzzy-ALB)")
    ap.add_argument("--hidden", default="20:3,38:4,44:2",
                    help='tasks whose real REBA is higher than the data, "task:points,..." or "none"')
    ap.add_argument("--min-ops", type=int, default=MIN_OPS)
    ap.add_argument("--max-ops", type=int, default=MAX_OPS)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--out", help="results folder (default experiments/results_v1)")
    a = ap.parse_args()
    HIDDEN = {} if a.hidden == "none" else {int(k): int(v) for k, v in (x.split(":") for x in a.hidden.split(","))}
    MIN_OPS, MAX_OPS, N_DAYS = a.min_ops, a.max_ops, a.days
    if a.out:
        OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    line = load_bosch(a.dataset) if a.dataset else load_bosch()
    unknown = set(HIDDEN) - set(line.tasks.index)
    if unknown:
        raise SystemExit(f"--hidden refers to tasks that do not exist in this dataset: {sorted(unknown)}")
    d1 = e1(line)
    cols = ["units_per_hour", "units_per_operator_hour", "line_efficiency", "exposure_per_h_max", "exposure_per_unit_max",
            "exposure_range", "exposure_gini", "reba4_min_per_h_max", "borg_max"]
    print("E1 - mean over 20 shifts\n", d1.groupby("policy")[cols].mean().round(3).to_string())
    print("rebalances:", d1.groupby("policy")["rebalanced"].sum().to_dict())
    d2 = e2(line)
    print("\nE2 - by headcount\n", d2.groupby("m")[cols + ["mean_cycle_s"]].mean().round(3).to_string())
