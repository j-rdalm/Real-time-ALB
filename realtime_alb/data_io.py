"""Load the Bosch task dataset and (for validation) the OPL .dat files of Ch. 4."""
import re
import warnings
from dataclasses import dataclass, field
import pandas as pd

from .config import DEFAULT_DATASET


@dataclass
class LineData:
    tasks: pd.DataFrame                  # index = task id; columns time, energy, reba, borg, station, ws
    precedence: list                     # [(pred, succ), ...]
    n_workstations: int = field(init=False)

    def __post_init__(self):
        self.n_workstations = int(self.tasks["ws"].max())


def load_bosch(path=DEFAULT_DATASET) -> LineData:
    """Read Tasks + Precedence sheets (title on row 1, headers on row 2), as in Fuzzy-ALB."""
    sheets = pd.read_excel(path, sheet_name=["Tasks", "Precedence"], header=1)   # other sheets are ignored
    t = sheets["Tasks"].rename(columns={
        "Task ID": "task", "Station": "station", "Time (s)": "time",
        "Energy (kcal)": "energy", "REBA": "reba", "Borg": "borg"})
    t = t.set_index("task").sort_index()
    # Physical workstation index 1..p in ascending label order (same rule as milp_dat_generator_v2.py)
    labels = sorted(t["station"].unique())
    t["ws"] = t["station"].map({lab: k + 1 for k, lab in enumerate(labels)})
    p = sheets["Precedence"]
    prec = [(int(a), int(b)) for a, b in zip(p.iloc[:, 0], p.iloc[:, 1])]
    missing = {a for pair in prec for a in pair} - set(t.index)
    if missing:
        raise ValueError(f"Precedence refers to tasks that are not in the Tasks sheet: {sorted(missing)}")
    # The workstation-contiguity constraints assume that station labels follow the line order.
    # If many precedences go "backwards" in that order, the MILP piles tasks onto one operator.
    back = sum(t.at[a, "ws"] > t.at[b, "ws"] for a, b in prec)
    if back > 0.2 * len(prec):
        warnings.warn(f"{back} of {len(prec)} precedences go backwards in station order: "
                      "check that the Station numbers follow the order of the line")
    return LineData(t[["station", "ws", "time", "energy", "reba", "borg"]], prec)


def read_opl_dat(path):
    """Parse the Ch. 4 OPL .dat files (time_task, eri, taskWorkstation, precedence, weights)."""
    txt = open(path).read()

    def arr(name):
        body = re.search(rf"^{name}\s*=\s*\[(.*?)\];", txt, re.S | re.M).group(1)
        body = re.sub(r"//.*", "", body)
        return [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", body)]

    def scal(name):
        return float(re.search(rf"^{name}\s*=\s*([-\d.]+)", txt, re.M).group(1))

    prec = [(int(a), int(b)) for a, b in re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", txt)]
    return {
        "time": arr("time_task"), "eri": arr("eri"),
        "ws": [int(v) for v in arr("taskWorkstation")],
        "precedence": prec, "m": int(scal("n_stations")),
        "w_time": scal("w_time"), "w_eri": scal("w_eri"),
    }
