"""Ergonomic risk index (ERI) and exposure.

Normalisation (same as Ch. 3.4 / Ch. 4.4, fixed from the initial Bosch data): time and energy by
1.05 x dataset max, REBA and Borg by dataset max + 1. Fixed references keep values comparable over
time; measured values may exceed 1 (a task heavier than any task in the initial study), accepted.

Two definitions:

* Ch. 4.4 (time_weighted=False): ERI of a task, counted once per execution
      ERI_i = wE*kcal_i/refE + wR*REBA_i/refR + wB*Borg_i/refB

* Real-time chapter (time_weighted=True, default): exposure = intensity above the no-risk level x time
      I_i = wE*(kcal/min)_i/ref_rate + wR*max(0, REBA_i - thR)/(refR - thR) + wB*max(0, Borg_i - thB)/(refB - thB)
      exposure_i = I_i * t_i                       (ERI-minutes)
  Energy enters as a RATE so that multiplying by time gives back kcal (no double counting of time).
  A long task with REBA 1 and Borg <= 2 only accumulates its energy term. The result does not depend
  on how the MTM operations were grouped into tasks (a task = the sum of its parts).
  MILP input: e_i = I_i * t_i / max_j(I_j * t_j)   (max over the Bosch data, fixed)
"""
from .config import AHP_WEIGHTS, SCALE_MAX, DEFAULT_NORMALISATION, EXPOSURE_THRESHOLDS


def reference_maxima(prior, mode=DEFAULT_NORMALISATION, thresholds=None):
    """REBA and Borg: dataset max + 1 (mode="data", Ch. 3.4) or scale maximum (mode="scale", not used)."""
    thr = thresholds or EXPOSURE_THRESHOLDS
    ref = {
        "time": 1.05 * prior["time"].max(),
        "energy": 1.05 * prior["energy"].max(),
        "energy_rate": 1.05 * (prior["energy"] / prior["time"] * 60).max(),   # kcal/min
        "reba": SCALE_MAX["reba"] if mode == "scale" else prior["reba"].max() + 1,
        "borg": SCALE_MAX["borg"] if mode == "scale" else prior["borg"].max() + 1,
        "thr_reba": thr["reba"], "thr_borg": thr["borg"],
    }
    ref["eri_time"] = max(exposure(prior.at[i, "energy"], prior.at[i, "reba"], prior.at[i, "borg"],
                                   prior.at[i, "time"], ref) for i in prior.index)
    return ref


def eri(energy, reba, borg, ref, w=AHP_WEIGHTS):
    """Ch. 4.4 ERI of one task execution (FERI modal formula)."""
    return (w["energy"] * energy / ref["energy"] + w["reba"] * reba / ref["reba"]
            + w["borg"] * borg / ref["borg"])


def intensity(energy, reba, borg, duration_s, ref, w=AHP_WEIGHTS):
    """Risk per unit of time, counting REBA and Borg only above their no-risk levels."""
    rate = energy / (duration_s / 60)
    return (w["energy"] * rate / ref["energy_rate"]
            + w["reba"] * max(0.0, reba - ref["thr_reba"]) / (ref["reba"] - ref["thr_reba"])
            + w["borg"] * max(0.0, borg - ref["thr_borg"]) / (ref["borg"] - ref["thr_borg"]))


def exposure(energy, reba, borg, duration_s, ref, time_weighted=True):
    """Exposure of one task execution: intensity x duration (ERI-minutes), or the Ch. 4.4 ERI."""
    if not time_weighted:
        return eri(energy, reba, borg, ref)
    return intensity(energy, reba, borg, duration_s, ref) * duration_s / 60


def milp_inputs(est, ref, time_weighted=True):
    """est: DataFrame (index task) with time, energy, reba, borg -> (t, e) dicts for solve_balance."""
    t = (est["time"] / ref["time"]).to_dict()
    if time_weighted:
        e = {i: exposure(est.at[i, "energy"], est.at[i, "reba"], est.at[i, "borg"], est.at[i, "time"], ref)
             / ref["eri_time"] for i in est.index}
    else:
        e = {i: eri(est.at[i, "energy"], est.at[i, "reba"], est.at[i, "borg"], ref) for i in est.index}
    return t, e
