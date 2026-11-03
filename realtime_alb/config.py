"""Paths and default parameters. Everything here is read-only input or a tunable default."""
from pathlib import Path
import os

# Bosch dataset from the Fuzzy-ALB toolkit (read only, never modified).
# Override with the environment variable RTALB_DATASET if the repo lives elsewhere.
DEFAULT_DATASET = Path(os.environ.get(
    "RTALB_DATASET",
    Path(__file__).resolve().parents[2] / "Fuzzy-ALB" / "feri-alb-toolkit" / "feri" / "example_dataset.xlsx",
))

# AHP consensus weights from Ch. 3 (feri/ahp_weights.py, 7 of 8 experts retained)
AHP_WEIGHTS = {"energy": 0.3357, "reba": 0.4230, "borg": 0.2413}

# Shift structure. Rotation happens at the breaks (Borg is asked there too).
# Default: 3 blocks of 2h30, as in the job-rotation case study (Sec. 4.2).
SHIFT_BLOCKS_MIN = [150, 150, 150]

# REBA action levels: score >= 4 is "medium" risk or higher
REBA_MEDIUM = 4

# Normalisation of the ordinal scales in the ERI.
#   "data" (default, as in Ch. 3.4 / Ch. 4.4 / fuzzy_eri.py): divide by the dataset maximum + 1
#            (REBA 8, Borg 11 for Bosch), fixed from the initial data. Keeps the AHP weights with the
#            meaning they had in Ch. 4. Measured values may exceed 1: such a task is heavier than any
#            task of the initial study, which is accepted and informative.
#   "scale": divide by the scale maximum (REBA 15, Borg 10). Not used: on the Bosch line it lowers
#            the effective weight of REBA in the index from 34% to 21%.
SCALE_MAX = {"reba": 15, "borg": 10}
DEFAULT_NORMALISATION = "data"

# Exposure counts only what lies above the "no-risk" level of each ordinal scale (time-weighted):
#   REBA 1 = action level 0, "negligible risk, no action necessary" (Hignett & McAtamney, 2000)
#   Borg 2 = "weak, light" on the CR-10 scale (Borg, 1982)
# Energy has no threshold (the MAWL of 3.5 kcal/min, Sec. 4.3, is above every Bosch task).
# Alternative for the sensitivity analysis: {"reba": 3, "borg": 2} (REBA low-risk band counted as no risk).
EXPOSURE_THRESHOLDS = {"reba": 1, "borg": 2}

# Rotation at the breaks
ROTATION_BORG_WEIGHT = 0.0   # Borg is NOT in the rotation objective, only in the Borg rule below
                             # (0.5 and 1 are tested in the sensitivity analysis as alternatives)
BORG_LIMIT = 7               # Borg >= 7 ("very strong"): not sent to the heaviest loop in the next block

# Estimator of each task's value: "trimmed" (10% trimmed mean, default) or "median".
# The median of integer REBA scores jumps by ~1 point; the trimmed mean is as robust and smooth.
ESTIMATOR_CENTRAL = "trimmed"
