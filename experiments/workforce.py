"""E2 with replications: the same line run with 4, 5, 6 and 7 operators, every KPI per hour or per
unit so that days with different headcount compare. 10 seeds; for each seed and policy the estimates
are warmed up with 5 shifts (middle headcount) and then 5 shifts are run for each headcount.

Run from the repo root:  python -m experiments.workforce  [--seeds 10] [--workers 2]
"""
import argparse, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

OUT = Path(__file__).resolve().parent / "results_workforce"
HIDDEN = {20: +3, 38: +4, 44: +2}
HEADCOUNTS, SHIFTS_PER_M, WARMUP = (4, 5, 6, 7), 5, 5
_LINE = None


def run_one(job):
    global _LINE
    policy, seed = job
    if _LINE is None:
        _LINE = load_bosch()
    sim = LineSimulator(_LINE.tasks, SimParams(reba_shift=HIDDEN), n_people=max(HEADCOUNTS) + 1, seed=seed)
    ctl = Controller(_LINE, sim, policy)
    for d in range(1, WARMUP + 1):
        ctl.run_shift(-d, 6)
    rows = []
    for m in HEADCOUNTS:
        for k in range(SHIFTS_PER_M):
            r = ctl.run_shift(k + 1, m)
            rows.append({**r, "seed": seed})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = [(p, s) for p in ("static", "rotation", "realtime") for s in range(a.seeds)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / "shifts.csv", index=False)
    cols = ["units_per_hour", "units_per_operator_hour", "line_efficiency", "exposure_per_h_max",
            "exposure_per_unit_max", "exposure_range", "reba4_min_per_h_max"]
    per_seed = df.groupby(["policy", "m", "seed"])[cols].mean().reset_index()
    summ = per_seed.groupby(["policy", "m"])[cols].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.round(3).to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(2).to_string())
    print(f"\n{len(jobs)} runs in {time.time() - t0:.0f} s")
