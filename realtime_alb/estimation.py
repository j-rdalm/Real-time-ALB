"""Estimation layer: from the stream of measurements to one value per task.

Start from the Bosch values (prior) and let the data take over as cycles accumulate:
    estimate = (n0 * prior + n * centre_of_last_N) / (n0 + n),   n = observations so far
centre = 10% trimmed mean (default) or median. Both ignore glitches; the median of integer REBA
scores jumps by ~1 point, the trimmed mean does not. p10 / p90 give the optimistic / pessimistic
scenarios that replace the fuzzy alpha-cuts. Borg stays at the Bosch interview value per task (Borg is now asked per operator at
the breaks, not per task, so it cannot update the task value).
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

MEASURED = ("time", "reba", "energy")


def _centre(values, q, central):
    if q == 50 and central == "trimmed":
        v = np.sort(np.asarray(values, float))
        k = int(0.1 * len(v))
        return float(v[k:len(v) - k].mean()) if len(v) > 2 * k else float(v.mean())
    return float(np.percentile(values, q))


class TaskEstimator:
    def __init__(self, prior, window=30, n0=5, central="median"):
        """central: "median" or "trimmed" (10% trimmed mean). The median of integer REBA scores can
        only take whole or half values, so it jumps by ~1 point when the window shifts slightly; the
        trimmed mean is as robust to outliers but changes smoothly."""
        self.central = central
        self.prior = prior
        self.window, self.n0 = window, n0
        self.buf = {k: defaultdict(lambda: deque(maxlen=window)) for k in MEASURED}
        self.n = defaultdict(int)

    def update(self, records):
        for r in records:
            for k in MEASURED:
                self.buf[k][r["task"]].append(r[k])
            self.n[r["task"]] += 1

    def estimates(self, stat="median"):
        """DataFrame (index task) with time, energy, reba, borg. stat: median | p10 | p90."""
        q = {"median": 50, "p10": 10, "p90": 90}[stat]
        out = self.prior[["time", "energy", "reba", "borg"]].astype(float).copy()
        for i in out.index:
            n = self.n[i]                      # all observations so far: the prior fades out
            if n == 0:
                continue
            a = n / (n + self.n0)
            for k in MEASURED:
                out.at[i, k] = (1 - a) * self.prior.at[i, k] + a * _centre(self.buf[k][i], q, self.central)
        return out

    def estimate_task(self, i, k, stat="median"):
        """Current estimate of one indicator k of task i (same rule as estimates())."""
        n = self.n[i]
        if n == 0:
            return float(self.prior.at[i, k])
        q = {"median": 50, "p10": 10, "p90": 90}[stat]
        a = n / (n + self.n0)
        return (1 - a) * float(self.prior.at[i, k]) + a * _centre(self.buf[k][i], q, self.central)

    def counts(self):
        return pd.Series({i: self.n[i] for i in self.prior.index})
