"""Simulator v1: pretends to be the MotionSuit + smartwatch on the Bosch line.

Ground truth = Bosch task values (optionally altered by a scenario, e.g. a task whose real REBA is
higher than the video-based value). For every cycle and task it produces what the devices would
deliver after task segmentation:
  * time   : actual duration of the task (s)
  * reba   : median REBA score of the task execution (integer 1..15, from the suit)
  * energy : kcal of the task, obtained by splitting the watch's per-cycle energy of that operator
             in proportion to the MEM values of the tasks the operator did in that cycle
Borg CR-10 is produced once per operator at each break.

v1 assumptions (decision 2A): every operator has the same task values. Optional fatigue (0 in the
base scenario) builds up within a rotation block and is recovered at the break.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SimParams:
    """Base scenario aligned with Ch. 3.4 (fuzzification spreads):
       time      SD 5% of the task time        (delta = 5%, read as a standard deviation)
       REBA      SD 0.94 points                 (SEM = 1.23 * sqrt(1 - 0.41), Ch. 3.4.3.2)
       watch     relative error 11%             (upper value for less accurate instruments, ref. [190])
       Borg      SD 1.0 point                   (+-1 by analogy with REBA, Ch. 3.4)
       fatigue   none in the base scenario      (no basis in Ch. 3; varied in the sensitivity analysis)
    The Borg-generation parameters (base, slope, personal bias) have no basis either; Borg is only
    used by the rotation (Borg rule and Borg term of the objective).
    Skills and medical restrictions are simulated (the Bosch data of this line has none)."""
    time_cv: float = 0.05            # cycle-to-cycle variation of task time (SD, relative)
    reba_sd: float = 0.94            # variation of the REBA score of a task execution (SD, points)
    watch_error: float = 0.11        # relative error of the smartwatch energy per cycle (SD)
    fatigue_time: float = 0.0        # extra task time at the end of a block (e.g. 0.06 = +6%)
    fatigue_reba: float = 0.0        # extra REBA points at the end of a block
    fatigue_energy: float = 0.0      # extra energy at the end of a block
    borg_base: float = 1.5           # Borg of a very light block
    borg_slope: float = 7.0          # Borg increase per unit of normalised exposure rate
    borg_sd: float = 1.0
    person_borg_bias_sd: float = 0.7 # individuals rate the same effort differently
    skill_prob: float = 0.95         # probability that an operator is qualified for a workstation
    medical_share: float = 0.25      # share of operators with a medical restriction ...
    medical_reba: int = 5            # ... that bars them from workstations with a task of REBA >= this (the 2 heaviest)
    reba_shift: dict = field(default_factory=dict)   # scenario: {task: +-points} on the true REBA
    time_shift: dict = field(default_factory=dict)   # scenario: {task: factor} on the true time


class LineSimulator:
    def __init__(self, prior, params=SimParams(), n_people=8, seed=0):
        self.rng = np.random.default_rng(seed)
        self.p = params
        self.prior_time = prior["time"].astype(float)     # MTM time: the MEM energy refers to it
        self.truth = prior.copy()
        self.truth["reba"] = self.truth["reba"].astype(float)
        self.truth["time"] = self.truth["time"].astype(float)
        for task, delta in params.reba_shift.items():
            self.truth.at[task, "reba"] = float(np.clip(self.truth.at[task, "reba"] + delta, 1, 15))
        for task, factor in params.time_shift.items():
            self.truth.at[task, "time"] *= factor
        # plain dicts for the inner loop (pandas .at is slow)
        self._t = self.truth["time"].to_dict()
        self._r = self.truth["reba"].to_dict()
        self._e = self.truth["energy"].to_dict()
        self._pt = self.prior_time.to_dict()
        self.people = [f"W{k + 1}" for k in range(n_people)]     # the operators of the line
        # Skills and medical restrictions per workstation (as in Job-Rotation-TL), simulated with a
        # separate random generator so that the measurement noise stream is not affected.
        rs = np.random.default_rng([seed, 777])
        wss = sorted(prior["ws"].unique())
        heavy = set(prior.loc[prior["reba"] >= params.medical_reba, "ws"])
        restricted = set(rs.choice(self.people, size=int(round(params.medical_share * n_people)), replace=False))
        self.skill = {o: {w: bool(rs.random() < params.skill_prob) for w in wss} for o in self.people}
        self.medical_ok = {o: {w: not (o in restricted and w in heavy) for w in wss} for o in self.people}
        self._ws = prior["ws"].to_dict()
        self.borg_bias = dict(zip(self.people, self.rng.normal(0, params.person_borg_bias_sd, n_people)))

    def can_work(self, operator, tasks):
        """Qualified and medically cleared for every workstation of the loop."""
        wss = {self._ws[i] for i in tasks}
        return all(self.skill[operator][w] and self.medical_ok[operator][w] for w in wss)

    def attendance(self, m):
        """Which m people come to work today."""
        if m > len(self.people):
            raise ValueError(f"headcount {m} is larger than the operator pool ({len(self.people)}): raise n_people")
        return sorted(self.rng.choice(self.people, size=m, replace=False), key=lambda s: int(s[1:]))

    def run_cycle(self, loops, operator_at, frac_block):
        """One cycle. loops: {loop: [tasks]}; operator_at: {loop: operator};
        frac_block: elapsed fraction of the rotation block (drives fatigue).
        Returns (records, cycle_duration_s)."""
        p, rng = self.p, self.rng
        Tt, Tr, Te, Pt = self._t, self._r, self._e, self._pt
        recs, work = [], {}
        for loop, tasks in loops.items():
            operator = operator_at[loop]
            f_t = 1 + p.fatigue_time * frac_block
            f_e = 1 + p.fatigue_energy * frac_block
            times = {i: Tt[i] * f_t * rng.lognormal(0, p.time_cv) for i in tasks}
            rebas = {i: int(np.clip(np.rint(rng.normal(Tr[i] + p.fatigue_reba * frac_block, p.reba_sd)), 1, 15))
                     for i in tasks}
            # same energy rate as in MEM, over the actual duration
            true_kcal = sum(Te[i] * f_e * times[i] / Pt[i] for i in tasks)
            watch_kcal = true_kcal * (1 + rng.normal(0, p.watch_error))       # what the watch reports
            mem_total = sum(Te[i] for i in tasks)
            for i in tasks:
                recs.append({"task": i, "loop": loop, "operator": operator, "time": times[i],
                             "reba": rebas[i], "energy": watch_kcal * Te[i] / mem_total})
            work[loop] = sum(times.values())
        return recs, max(work.values())      # paced line: the slowest loop sets the cycle

    def borg(self, operator, exposure_rate):
        """Borg CR-10 at a break. exposure_rate = ERI accumulated per hour in the block, normalised."""
        v = self.p.borg_base + self.p.borg_slope * exposure_rate + self.borg_bias[operator] + self.rng.normal(0, self.p.borg_sd)
        return int(np.clip(np.rint(v), 0, 10))
