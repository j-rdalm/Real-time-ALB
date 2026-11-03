"""Runs shifts on the simulated line under a decision policy and records KPIs.

Policies
  static    : balance from the Bosch values for today's headcount; no rotation; no learning
  rotation  : same balance; rotation at the breaks computed with the Bosch values (offline, Sec. 4.2 style)
  realtime  : estimates updated every cycle; balance re-solved at shift start when the headcount
              changes or the estimated loads drifted; rotation at the breaks with MEASURED
              cumulative exposure + Borg rule

Time scales: every cycle -> update estimates (no decision); every break -> rotation + Borg;
between shifts -> rebalance (MILP) if triggered.
"""
import numpy as np
import pandas as pd

from .config import (SHIFT_BLOCKS_MIN, REBA_MEDIUM, DEFAULT_NORMALISATION, ROTATION_BORG_WEIGHT,
                     BORG_LIMIT, ESTIMATOR_CENTRAL)
from .balancing import solve_balance
from .estimation import TaskEstimator
from .indicators import reference_maxima, milp_inputs, eri, exposure
from .rotation import rotate


def gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum())) if v.sum() > 0 else 0.0


class Controller:
    # Plans computed from the fixed Bosch values are the same in every run: share them across runs
    _offline_cache = {}

    def __init__(self, line, sim, policy="realtime", w_time=0.5, w_eri=0.5, drift_threshold=0.05,
                 min_gain=0.02, estimate_stat="median", blocks_min=SHIFT_BLOCKS_MIN, time_weighted=True,
                 normalisation=None, exposure_thresholds=None, estimator_central=ESTIMATOR_CENTRAL,
                 borg_weight=ROTATION_BORG_WEIGHT, use_eligibility=True):
        self.line, self.sim, self.policy = line, sim, policy
        self.prior = line.tasks
        self.ref = reference_maxima(self.prior, normalisation or DEFAULT_NORMALISATION, exposure_thresholds)
        self.w = (w_time, w_eri)
        self.drift_threshold = drift_threshold   # re-solve the MILP when a loop's load drifts more than this
        self.min_gain = min_gain                 # ...but change the line only if the new plan is this much better
        self.stat = estimate_stat
        self.blocks_min = blocks_min
        self.tw = time_weighted          # option B: ergonomic load and exposure weighted by task time
        self.est = TaskEstimator(self.prior, central=estimator_central)
        self.borg_weight = borg_weight           # weight of the Borg term in the rotation objective
        self.use_eligibility = use_eligibility   # skills and medical restrictions in the rotation
        self.library = {}           # headcount m -> balance solution (+ loads it was solved with)
        self.log_shifts, self.log_blocks = [], []
        self._borg_prior = self.prior["borg"].to_dict()
        self.on_cycle = None        # optional hook called after every cycle (used by the error test)
        self.trace = None           # optional list: set to [] to record every cycle (used by the dashboard)

    # ---------- decisions ----------
    def _values(self):
        return self.est.estimates(self.stat) if self.policy == "realtime" else self.prior.astype(float)

    def _balance(self, m):
        """Return (balance, changed?) for headcount m.

        Stability rule (realtime policy): if the loads of the stored plan drift more than
        drift_threshold under the new estimates, the MILP is solved again (a few seconds, nothing
        changes on the line). The new plan is adopted only if its objective is at least min_gain
        better than the stored plan evaluated with the same new estimates; otherwise the line keeps
        its plan, whose loads are refreshed so that drift is measured from the latest estimates."""
        vals = self._values()
        t, e = milp_inputs(vals, self.ref, self.tw)
        cached = self.library.get(m)
        ws = self.prior["ws"].to_dict()
        if cached is None:
            key = None
            if self.policy != "realtime":
                key = (id(self.line), m, self.w, self.tw, tuple(sorted(self.ref.items())))
                if key in Controller._offline_cache:
                    self.library[m] = Controller._offline_cache[key]
                    return self.library[m], True
            sol = solve_balance(t, e, ws, self.line.precedence, m, *self.w)
            self.library[m] = sol
            if key is not None:
                Controller._offline_cache[key] = sol
            return sol, True
        if self.policy != "realtime":
            return cached, False
        S = range(1, m + 1)
        loads_t = {s: sum(t[i] for i, s2 in cached["assignment"].items() if s2 == s) for s in S}
        loads_e = {s: sum(e[i] for i, s2 in cached["assignment"].items() if s2 == s) for s in S}
        drift = max(abs(loads_e[s] - cached["eri_loads"][s]) / cached["eri_loads"][s] for s in S)
        if drift <= self.drift_threshold:
            return cached, False
        old_obj = self.w[0] * max(loads_t.values()) + self.w[1] * max(loads_e.values())
        sol = solve_balance(t, e, ws, self.line.precedence, m, *self.w, warm_start=cached["assignment"])
        if sol["objective"] <= (1 - self.min_gain) * old_obj:
            self.library[m] = sol
            return sol, True
        kept = dict(cached, time_loads=loads_t, eri_loads=loads_e,
                    CT=max(loads_t.values()), EL=max(loads_e.values()), objective=old_obj)
        self.library[m] = kept
        return kept, False

    def _block_forecast(self, loops, block_s):
        """Expected exposure per loop in the next block, in the same units as the measured
        exposure: expected cycles x sum over the loop's tasks of the exposure of one execution."""
        vals = self._values()
        ct = max(vals.loc[tasks, "time"].sum() for tasks in loops.values())
        n_cycles = block_s / ct
        per_task = {i: exposure(vals.at[i, "energy"], vals.at[i, "reba"], vals.at[i, "borg"],
                                vals.at[i, "time"], self.ref, self.tw) for i in vals.index}
        return {s: n_cycles * sum(per_task[i] for i in tasks) for s, tasks in loops.items()}

    # ---------- one shift ----------
    def run_shift(self, day, m):
        operators = self.sim.attendance(m)
        bal, rebalanced = self._balance(m)
        loops = {s: [i for i, s2 in bal["assignment"].items() if s2 == s] for s in range(1, m + 1)}
        loop_ids = list(loops)
        eligible = (lambda o, s: self.sim.can_work(o, loops[s])) if self.use_eligibility else None
        cum = {o: 0.0 for o in operators}            # measured exposure so far in the shift
        cum_prior = {o: 0.0 for o in operators}      # exposure counted with Bosch values (offline policy)
        reba_med_s = {o: 0.0 for o in operators}
        borg = {}
        units, cycle_times, relaxed = 0, [], []

        for b, minutes in enumerate(self.blocks_min):
            block_s = minutes * 60
            forecast = self._block_forecast(loops, block_s)
            if b == 0:
                # first block: any assignment that respects skills and medical restrictions
                operator_at, rel = rotate(operators, loop_ids, cum, forecast, eligible=eligible)
            elif self.policy == "realtime":
                operator_at, rel = rotate(operators, loop_ids, cum, forecast, borg, BORG_LIMIT,
                                          self.borg_weight, eligible)
            elif self.policy == "rotation":   # offline: exposure counted with Bosch values, no Borg
                operator_at, rel = rotate(operators, loop_ids, cum_prior, forecast, eligible=eligible)
            else:                              # static: same assignment all shift
                rel = None
            if rel:
                relaxed.append(rel)

            elapsed, block_exp = 0.0, {o: 0.0 for o in operators}
            if self.trace is not None:
                self.trace.append({"type": "block", "day": day, "block": b + 1, "start_s": sum(self.blocks_min[:b]) * 60,
                                   "operator_at": dict(operator_at), "forecast": dict(forecast),
                                   "cum_before": dict(cum), "borg_before": dict(borg), "relaxed": rel or ""})
            while True:
                recs, ct = self.sim.run_cycle(loops, operator_at, elapsed / block_s)
                if elapsed + ct > block_s:
                    break
                elapsed += ct
                units += 1
                cycle_times.append(ct)
                if self.policy == "realtime":
                    self.est.update(recs)
                    if self.on_cycle:
                        self.on_cycle(self)
                for r in recs:
                    x = exposure(r["energy"], r["reba"], self._borg_prior[r["task"]], r["time"],
                                 self.ref, self.tw)
                    block_exp[r["operator"]] += x
                    if r["reba"] >= REBA_MEDIUM:
                        reba_med_s[r["operator"]] += r["time"]
                if self.trace is not None:
                    self.trace.append({"type": "cycle", "day": day, "block": b + 1,
                                       "t_s": sum(self.blocks_min[:b]) * 60 + elapsed, "ct": ct,
                                       "cum": {o: cum[o] + block_exp[o] for o in operators},
                                       "reba4_s": dict(reba_med_s)})
            for o in operators:
                cum[o] += block_exp[o]
            if self.policy == "rotation":
                for s, o in operator_at.items():
                    cum_prior[o] += forecast[s]
            # break: Borg per operator (exposure rate normalised by the line-average rate)
            mean_rate = np.mean(list(block_exp.values()))
            for o in operators:
                borg[o] = self.sim.borg(o, 0.5 * block_exp[o] / mean_rate if mean_rate else 0)
            if self.trace is not None:
                self.trace.append({"type": "break", "day": day, "block": b + 1, "borg": dict(borg),
                                   "block_exposure": dict(block_exp)})
            self.log_blocks.append({"day": day, "policy": self.policy, "m": m, "block": b + 1,
                                    **{f"loop_{s}": operator_at[s] for s in loop_ids},
                                    "relaxed": rel or "",
                                    "borg_mean": np.mean(list(borg.values())), "borg_max": max(borg.values())})

        hours = sum(self.blocks_min) / 60
        exp_h = np.array([cum[o] / hours for o in operators])
        rec = {
            "day": day, "policy": self.policy, "m": m, "rebalanced": rebalanced,
            "units": units, "units_per_hour": units / hours, "units_per_operator_hour": units / hours / m,
            "mean_cycle_s": float(np.mean(cycle_times)),
            "line_efficiency": float(self.prior["time"].sum() / (m * np.mean(cycle_times))),
            "exposure_per_unit_max": max(cum.values()) / units,
            "exposure_per_h_mean": exp_h.mean(), "exposure_per_h_max": exp_h.max(),
            "exposure_range": exp_h.max() - exp_h.min(), "exposure_gini": gini(exp_h),
            "reba4_min_per_h_max": max(reba_med_s.values()) / 60 / hours,
            "borg_max": max(borg.values()),
            "planned_EL": bal["EL"], "planned_CT": bal["CT"],
            "relaxed_borg_rule": relaxed.count("borg_rule"), "relaxed_eligibility": relaxed.count("eligibility"),
        }
        self.log_shifts.append(rec)
        return rec
