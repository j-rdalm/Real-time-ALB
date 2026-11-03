"""Build the standalone dashboard of the real-time ALB chapter (Ch. 5).

Computes three data sets and embeds them in dashboard/rtalb_dashboard.html (opens in any browser,
no server needed):
  tasks  : per-task values, Ch. 4 ERI (FERI modal) and time-weighted exposure with/without thresholds
  plans  : the balance (plan library) for each headcount, with the initial values and with the true
           values of the main scenario (hidden REBA errors of tasks 20, 38 and 44 corrected)
  shifts : one example shift per headcount and policy, recorded cycle by cycle (realtime after a
           warm-up of 5 shifts so that the estimates have converged)

Run from the repo root:
    python -m experiments.build_dashboard
"""
import json
from pathlib import Path
import numpy as np

from realtime_alb.config import AHP_WEIGHTS, SHIFT_BLOCKS_MIN, BORG_LIMIT
from realtime_alb.data_io import load_bosch
from realtime_alb.indicators import reference_maxima, eri, exposure, milp_inputs
from realtime_alb.balancing import solve_balance
from realtime_alb.simulator import LineSimulator, SimParams
from realtime_alb.controller import Controller

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "dashboard" / "template.html"
OUT = ROOT / "dashboard" / "rtalb_dashboard.html"
HIDDEN = {20: +3, 38: +4, 44: +2}          # main scenario of Sec. 5.8 (same as run_v1)
HEADCOUNTS = range(4, 8)
POLICIES = ("static", "rotation", "realtime")
SEED, WARMUP = 7, 5
W = (0.5, 0.5)


def r(x, n=4):
    return round(float(x), n)


def task_table(line):
    pr = line.tasks
    ref = reference_maxima(pr)
    ref0 = reference_maxima(pr, thresholds={"reba": 0, "borg": 0})
    w = AHP_WEIGHTS
    rows = []
    for i in pr.index:
        t, e, rb, bg = (float(pr.at[i, c]) for c in ("time", "energy", "reba", "borg"))
        rate = e / (t / 60)
        comp = {  # exposure split by component (exposure-minutes per execution)
            "energy": w["energy"] * rate / ref["energy_rate"] * t / 60,
            "reba": w["reba"] * max(0, rb - ref["thr_reba"]) / (ref["reba"] - ref["thr_reba"]) * t / 60,
            "borg": w["borg"] * max(0, bg - ref["thr_borg"]) / (ref["borg"] - ref["thr_borg"]) * t / 60,
        }
        rows.append({
            "id": int(i), "station": str(pr.at[i, "station"]), "ws": int(pr.at[i, "ws"]),
            "time": r(t, 2), "energy": r(e), "rate": r(rate), "reba": int(rb), "borg": int(bg),
            "true_reba": int(rb + HIDDEN.get(i, 0)),
            "feri": r(eri(e, rb, bg, ref)),
            "exposure": r(exposure(e, rb, bg, t, ref), 5),
            "exposure_nothr": r(exposure(e, rb, bg, t, ref0), 5),
            "comp": {k: r(v, 5) for k, v in comp.items()},
            "comp_nothr": {
                "energy": r(comp["energy"], 5),
                "reba": r(w["reba"] * rb / ref0["reba"] * t / 60, 5),
                "borg": r(w["borg"] * bg / ref0["borg"] * t / 60, 5)},
        })
    for key in ("feri", "exposure", "exposure_nothr"):
        for k, row in enumerate(sorted(rows, key=lambda x: -x[key]), 1):
            row["rank_" + key] = k
    refs = {"energy_rate": r(ref["energy_rate"]), "reba": int(ref["reba"]), "borg": int(ref["borg"]),
            "energy": r(ref["energy"]), "thr_reba": ref["thr_reba"], "thr_borg": ref["thr_borg"],
            "eri_time": r(ref["eri_time"], 5), "weights": AHP_WEIGHTS}
    return rows, refs


def plans(line):
    pr = line.tasks
    ref = reference_maxima(pr)                 # fixed references from the initial data
    truth = pr.astype(float).copy()
    for i, d in HIDDEN.items():
        truth.at[i, "reba"] += d
    out = {}
    for label, vals in (("initial", pr.astype(float)), ("true", truth)):
        t, e = milp_inputs(vals, ref)
        x = {i: exposure(vals.at[i, "energy"], vals.at[i, "reba"], vals.at[i, "borg"], vals.at[i, "time"], ref)
             for i in vals.index}
        for m in HEADCOUNTS:
            sol = solve_balance(t, e, pr["ws"].to_dict(), line.precedence, m, *W)
            loops = {s: sorted(i for i, s2 in sol["assignment"].items() if s2 == s) for s in range(1, m + 1)}
            time_s = {s: float(vals.loc[ts, "time"].sum()) for s, ts in loops.items()}
            ct = max(time_s.values())
            out[f"{label}_{m}"] = {
                "data": label, "m": m, "objective": r(sol["objective"], 5), "CT_norm": r(sol["CT"], 5),
                "EL_norm": r(sol["EL"], 5), "ct_s": r(ct, 2),
                "efficiency": r(float(vals["time"].sum()) / (m * ct)),
                "loops": [{"loop": s, "tasks": [int(i) for i in ts],
                           "ws": sorted({int(pr.at[i, "ws"]) for i in ts}),
                           "time_s": r(time_s[s], 2),
                           "exposure_cycle": r(sum(x[i] for i in ts), 5),
                           "exposure_h": r(sum(x[i] for i in ts) * 3600 / ct, 4),
                           "e_norm": r(sol["eri_loads"][s], 5)} for s, ts in loops.items()],
            }
    return out


def shifts(line):
    out = {}
    for m in HEADCOUNTS:
        for pol in POLICIES:
            sim = LineSimulator(line.tasks, SimParams(reba_shift=HIDDEN), n_people=8, seed=SEED + m)
            ctl = Controller(line, sim, pol)
            for d in range(1, WARMUP + 1):
                ctl.run_shift(-d, m)
            ctl.trace = []
            rec = ctl.run_shift(1, m)
            bal = ctl.library[m]
            loops = {s: sorted(i for i, s2 in bal["assignment"].items() if s2 == s) for s in range(1, m + 1)}
            ops = sorted({o for ev in ctl.trace if ev["type"] == "block" for o in ev["operator_at"].values()},
                         key=lambda s: int(s[1:]))
            cycles = [{"t": r(ev["t_s"] / 60, 2), "cum": [r(ev["cum"][o], 3) for o in ops]}
                      for ev in ctl.trace if ev["type"] == "cycle"]
            blocks = []
            for ev in ctl.trace:
                if ev["type"] == "block":
                    blocks.append({"block": ev["block"], "start_min": ev["start_s"] / 60,
                                   "assign": {str(s): o for s, o in ev["operator_at"].items()},
                                   "forecast": {str(s): r(v, 3) for s, v in ev["forecast"].items()},
                                   "cum_before": {o: r(v, 3) for o, v in ev["cum_before"].items()},
                                   "borg_before": {o: int(v) for o, v in ev["borg_before"].items()},
                                   "relaxed": ev["relaxed"]})
                elif ev["type"] == "break":
                    blocks[-1]["borg_after"] = {o: int(v) for o, v in ev["borg"].items()}
                    blocks[-1]["block_exposure"] = {o: r(v, 3) for o, v in ev["block_exposure"].items()}
            last = [ev for ev in ctl.trace if ev["type"] == "cycle"][-1]
            hours = sum(SHIFT_BLOCKS_MIN) / 60
            out[f"{pol}_{m}"] = {
                "policy": pol, "m": m, "operators": ops, "loops": {str(s): ts for s, ts in loops.items()},
                "cycles": cycles, "blocks": blocks,
                "reba4_min_h": {o: r(last["reba4_s"][o] / 60 / hours, 2) for o in ops},
                "kpi": {k: r(rec[k], 4) for k in ("units_per_hour", "exposure_range", "exposure_gini",
                                                    "exposure_per_h_max", "reba4_min_per_h_max", "line_efficiency")},
            }
            print(f"shift {pol:9s} m={m}: gap {rec['exposure_range']:.2f}/h")
    return out


def seed_summary():
    """Mean over 10 seeds by headcount and policy (experiments/results_workforce/summary.csv), if present."""
    f = ROOT / "experiments" / "results_workforce" / "summary.csv"
    if not f.exists():
        return []
    import pandas as pd
    df = pd.read_csv(f)
    cols = ["policy", "m", "exposure_range_mean", "exposure_range_std", "reba4_min_per_h_max_mean"]
    return [{k: (r(v, 3) if isinstance(v, float) else (int(v) if k == "m" else v)) for k, v in row.items()}
            for row in df[cols].to_dict("records")]


def main():
    line = load_bosch()
    rows, refs = task_table(line)
    data = {"tasks": rows, "refs": refs, "precedence": [list(map(int, p)) for p in line.precedence],
            "hidden": {str(k): v for k, v in HIDDEN.items()}, "blocks_min": SHIFT_BLOCKS_MIN,
            "borg_limit": BORG_LIMIT, "plans": plans(line), "shifts": shifts(line),
            "seed_summary": seed_summary()}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print("written", OUT, f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
