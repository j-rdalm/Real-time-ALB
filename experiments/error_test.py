"""Test of hidden errors in the initial (Bosch) data.

For each seed, 5 random tasks get a REBA error (-3..+4 points, never 0) and 5 other tasks a time
error (+-20%) in the simulator's "truth"; the system only knows the Bosch values. A control run with
no errors uses the same seed. The realtime policy runs 20 shifts (headcount 4..7 per day).

Detection rule (1 standard deviation of a single measurement, with at least 10 observations):
    REBA: |estimate - Bosch| > SEM = 0.94 points          time: |estimate - Bosch| / Bosch > 5%
Metrics
    detection   observations (= cycles) until the task is flagged and stays flagged
    convergence observations until |estimate - truth| < SEM (REBA) / < 5% of time, and stays
                (same tolerance as detection: closer to the truth than the noise of one measurement)
    false alarms share of tasks WITHOUT error flagged at the end (error runs and control runs)
    regret      % by which the plan in use is worse than the best plan, both evaluated with the
                TRUE values, for 4..7 operators: Bosch-based plan (static) vs plan learned in real time

Run from the repo root:  python -m experiments.error_test  [--seeds 20] [--workers 2] [--central trimmed|median]
"""
import argparse, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

from realtime_alb.data_io import load_bosch
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller
from realtime_alb.indicators import milp_inputs
from realtime_alb.balancing import solve_balance
from realtime_alb.config import ESTIMATOR_CENTRAL

OUT = Path(__file__).resolve().parent / "results_error_test"
N_DAYS, MIN_OPS, MAX_OPS, N_REBA, N_TIME = 20, 4, 7, 5, 5
SEM, TIME_TOL, MIN_N = 0.94, 0.05, 10
W = (0.5, 0.5)
_LINE, _STATIC = None, {}


def draw_errors(prior, seed):
    rng = np.random.default_rng(5000 + seed)
    tasks = rng.choice(prior.index.to_numpy(), N_REBA + N_TIME, replace=False)
    reba = {}
    for i in tasks[:N_REBA]:
        while True:
            d = int(rng.choice([-3, -2, -1, 1, 2, 3, 4]))
            if 1 <= prior.at[i, "reba"] + d <= 15:
                reba[int(i)] = d
                break
    time_ = {int(i): float(rng.choice([0.8, 1.2])) for i in tasks[N_REBA:]}
    return reba, time_


def plan_objective(assign, t, e, m):
    T = [sum(t[i] for i in t if assign[i] == s) for s in range(1, m + 1)]
    E = [sum(e[i] for i in e if assign[i] == s) for s in range(1, m + 1)]
    return W[0] * max(T) + W[1] * max(E)


CENTRAL = ESTIMATOR_CENTRAL


def run_one(job):
    global _LINE
    seed, with_errors, central = job if len(job) == 3 else (*job, CENTRAL)
    if _LINE is None:
        _LINE = load_bosch()
    line = _LINE
    prior = line.tasks
    reba_err, time_err = draw_errors(prior, seed) if with_errors else ({}, {})
    headcount = np.random.default_rng(1000 + seed).integers(MIN_OPS, MAX_OPS + 1, N_DAYS)
    sim = LineSimulator(prior, SimParams(reba_shift=reba_err, time_shift=time_err), n_people=MAX_OPS + 1, seed=seed)
    ctl = Controller(line, sim, "realtime", estimator_central=central)

    track = {(i, "reba"): [] for i in reba_err} | {(i, "time"): [] for i in time_err}

    def hook(c):
        for (i, k), hist in track.items():
            n = c.est.n[i]
            if hist and hist[-1][0] == n:
                continue
            hist.append((n, c.est.estimate_task(i, k)))
    ctl.on_cycle = hook
    for d, m in enumerate(headcount, 1):
        ctl.run_shift(d, int(m))

    def flagged(i, k, est, n):
        if n < MIN_N:
            return False
        b = float(prior.at[i, k])
        return abs(est - b) > SEM if k == "reba" else abs(est - b) / b > TIME_TOL

    def close(i, k, est):
        tr = float(sim.truth.at[i, k])
        return abs(est - tr) < SEM if k == "reba" else abs(est - tr) / tr < TIME_TOL

    def first_stable(hist, cond):
        ok_from = None
        for n, est in hist:
            if cond(n, est):
                ok_from = n if ok_from is None else ok_from
            else:
                ok_from = None
        return ok_from

    task_rows = []
    for (i, k), hist in track.items():
        err = reba_err[i] if k == "reba" else time_err[i] - 1
        task_rows.append({"seed": seed, "task": i, "indicator": k, "error": err,
                          "detected_after_obs": first_stable(hist, lambda n, e: flagged(i, k, e, n)),
                          "converged_after_obs": first_stable(hist, lambda n, e: close(i, k, e)),
                          "final_estimate": hist[-1][1] if hist else None,
                          "truth": float(sim.truth.at[i, k]), "bosch": float(prior.at[i, k])})

    est = ctl.est.estimates()
    clean = [i for i in prior.index if i not in reba_err and i not in time_err]
    fa_reba = np.mean([flagged(i, "reba", est.at[i, "reba"], ctl.est.n[i]) for i in clean])
    fa_time = np.mean([flagged(i, "time", est.at[i, "time"], ctl.est.n[i]) for i in clean])

    # regret, evaluated with the true values (energy follows the true time at the MEM rate)
    truth = prior.astype(float).copy()
    truth["reba"] = sim.truth["reba"]
    truth["time"] = sim.truth["time"]
    truth["energy"] = prior["energy"] * sim.truth["time"] / prior["time"]
    t_true, e_true = milp_inputs(truth, ctl.ref)
    t_b, e_b = milp_inputs(prior.astype(float), ctl.ref)
    ws = prior["ws"].to_dict()
    reg = {}
    for m in range(MIN_OPS, MAX_OPS + 1):
        opt = solve_balance(t_true, e_true, ws, line.precedence, m, *W)
        z_opt = plan_objective(opt["assignment"], t_true, e_true, m)
        if m not in _STATIC:
            _STATIC[m] = solve_balance(t_b, e_b, ws, line.precedence, m, *W)["assignment"]
        rt_plan = ctl.library[m]["assignment"] if m in ctl.library else \
            solve_balance(*milp_inputs(est, ctl.ref), ws, line.precedence, m, *W)["assignment"]
        reg[f"regret_static_m{m}_%"] = 100 * (plan_objective(_STATIC[m], t_true, e_true, m) / z_opt - 1)
        reg[f"regret_realtime_m{m}_%"] = 100 * (plan_objective(rt_plan, t_true, e_true, m) / z_opt - 1)
    run_row = {"seed": seed, "with_errors": with_errors, "false_alarm_reba": fa_reba,
               "false_alarm_time": fa_time, **reg}
    return task_rows, run_row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--central", default=ESTIMATOR_CENTRAL, choices=["median", "trimmed"])
    ap.add_argument("--out", help="results folder (default experiments/results_error_test)")
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, w, a.central) for s in range(a.seeds) for w in (True, False)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    tasks = pd.DataFrame([r for tr, _ in res for r in tr])
    runs = pd.DataFrame([rr for _, rr in res])
    tasks.to_csv(OUT / "tasks.csv", index=False)
    runs.to_csv(OUT / "runs.csv", index=False)

    tasks["detected"] = tasks["detected_after_obs"].notna()
    tasks["size"] = np.where(tasks.indicator == "reba", tasks.error.abs().astype(int).astype(str) + " pt",
                             (tasks.error * 100).round().astype(int).astype(str) + "%")
    print("Detection by error size")
    print(tasks.groupby(["indicator", "size"]).agg(
        n=("task", "size"), detected_rate=("detected", "mean"),
        median_obs_to_detect=("detected_after_obs", "median"),
        median_obs_to_converge=("converged_after_obs", "median")).round(2).to_string())
    print("\nFalse alarms (share of tasks without error flagged at the end)")
    print(runs.groupby("with_errors")[["false_alarm_reba", "false_alarm_time"]].mean().round(4).to_string())
    print("\nRegret (% worse than the best plan, true values)")
    cols = [c for c in runs.columns if c.startswith("regret")]
    print(runs.groupby("with_errors")[cols].agg(["mean", "max"]).round(2).T.to_string())
    print(f"\n{len(jobs)} runs in {time.time() - t0:.0f} s")
