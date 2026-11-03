# Real-time ergonomic assembly line balancing (simulation, v1)

Proof of concept for the real-time ALB chapter. There is no MotionSuit or smartwatch data yet, so a
simulator produces the data the devices would deliver on the Bosch line (92 tasks, 20 workstations).

## Terminology

- **Loop**: the group of tasks the MILP assigns to one operator slot (fixed during the shift).
- **Operator**: a person. The rotation decides which operator works on which loop in each block.

## Time scales

| When | What happens |
|---|---|
| Every cycle | Measurements per task (time, REBA, energy) update the estimates. **No decision.** |
| Every break | Job rotation (which operator works on which loop) with the measured exposure; Borg CR-10 per operator |
| Shift start | Headcount m is known. Balance for m taken from the library. If m is new, the MILP is solved (~3 s). If the loads of the stored plan drifted > 5%, the MILP is solved again, but the line only changes when the new plan is at least 2% better (stability rule) |

## Ergonomic load weighted by task time (option B)

Ch. 4.4 sums the FERI of the tasks of an operator, once per task, whatever its duration. In this
chapter the load is an exposure: risk intensity above the no-risk level x time (Ch. 4 itself is unchanged).

    I_i = wE*(kcal/min)_i/ref_rate + wR*max(0, REBA_i - 1)/(refR - 1) + wB*max(0, Borg_i - 2)/(refB - 2)
    exposure_i = I_i * t_i   (ERI-minutes)

- Energy enters as a rate, so multiplying by time gives back kcal (time is not counted twice).
- REBA 1 = "negligible risk, no action necessary" (Hignett & McAtamney, 2000); Borg 2 = "weak, light"
  (Borg, 1982). A long task with REBA 1 and Borg <= 2 only accumulates its energy term.
  Thresholds in `config.EXPOSURE_THRESHOLDS`; REBA 3 is the alternative for the sensitivity analysis.
- Supported by exposure-above-threshold measures (Hoogendoorn et al., 2000, Spine 25:3087-3092) and
  by EAWS, which scores posture class x duration with 0 points for short upright standing.
- The result does not depend on how the MTM operations were grouped into tasks: a task gives the same
  exposure as the sum of its parts (in the Ch. 4.4 index, task 44 split in 10 parts counts ~7x more).
- MILP input: `e_i = I_i * t_i / max_j(I_j * t_j)` (max over the Bosch data, fixed).
- The measured exposure and the rotation forecast use the same function, so they share the same units.

`Controller(..., time_weighted=False)` gives back the Ch. 4.4 definition.

Normalisation is the same as in Ch. 4.4: REBA and Borg divided by the dataset maximum + 1, time and
energy by 1.05 x the dataset maximum, all fixed from the initial Bosch data. Measured values above 1
are accepted (a task heavier than any task in the initial study). The scale maxima (REBA 15, Borg 10)
are available with `Controller(..., normalisation="scale")` but not used, because on the Bosch line
they lower the effective weight of REBA from 34% to 21%.

## Inputs (read only)

The Bosch dataset is read from the Fuzzy-ALB repo, which must sit next to this one:
`../Fuzzy-ALB/feri-alb-toolkit/feri/example_dataset.xlsx`. Nothing in Fuzzy-ALB or Job-Rotation-TL is modified.
To use another location: `export RTALB_DATASET=/path/to/example_dataset.xlsx`.

## Modules (`realtime_alb/`)

| File | Contents |
|---|---|
| `balancing.py` | Sec. 4.4 MILP (OBJ-1, lexicographic) rewritten for PuLP + HiGHS; headcount m is a parameter |
| `simulator.py` | Suit + watch simulator: task time, REBA per execution, watch energy per cycle split by MEM shares, Borg per operator at breaks, optional fatigue within a block, simulated skills and medical restrictions |
| `estimation.py` | Bosch values as prior, replaced by the 10% trimmed mean of the last 30 cycles as data accumulates (the median is an option); p10 / p90 as optimistic / pessimistic scenarios (replace the fuzzy alpha-cuts) |
| `rotation.py` | Rotation at each break: minimise the spread of cumulative exposure adjusted by Borg (as in Job-Rotation-TL), with skills and medical restrictions; operator with Borg >= 7 not sent to the heaviest loop; exact by enumeration (m <= 7) |
| `indicators.py` | ERI per task execution with the AHP weights of Ch. 3; fixed normalisation references |
| `controller.py` | Runs shifts under a policy (`static`, `rotation`, `realtime`) and records KPIs |

## Run

```bash
pip install -r requirements.txt
python -m validation.validate_vs_cplex     # HiGHS port vs CPLEX results of Ch. 4
python -m experiments.run_v1               # E1 policies, E2 headcount 4..7 (~2 min, one seed)
python -m experiments.sensitivity          # sensitivity analysis, 10 seeds (~45 min on 2 cores)
python -m experiments.error_test           # hidden-error test, 20 seeds + 20 controls (~12 min)
python -m experiments.error_test --central median --out experiments/results_error_test_median
python -m experiments.workforce            # E2 with 10 seeds: KPIs by headcount, 3 policies (~3 min)
python -m experiments.figures              # thesis figures (PDF + PNG) in figures/
python -m experiments.build_dashboard      # standalone dashboard in dashboard/ (~1.5 min)
```

## Dashboard (`dashboard/rtalb_dashboard.html`)

Standalone HTML page (open it in a browser, no server needed), in the style of the Fuzzy-ALB dashboards.
`experiments/build_dashboard.py` computes the data and embeds it in `dashboard/template.html`:

* **Task exposure**: ERI of Sec. 4.4 vs time-weighted exposure (with and without the no-risk thresholds),
  stacked by component, rank comparison, and the effect of splitting a task into k equal parts.
* **Plans by headcount**: the balance for 4 to 7 operators (w_t = w_e = 0.5), with the initial values or with
  the true values of the main scenario (hidden REBA errors of tasks 20, 38 and 44 corrected): line layout,
  work content and exposure per loop, plan library.
* **Simulated shift**: one example shift per headcount and policy (realtime after 5 warm-up shifts), recorded
  cycle by cycle with `Controller.trace`: cumulative exposure per operator, rotation and Borg at each break, and
  the exposure gap of the three policies next to the 10-seed means of `results_workforce/summary.csv`.

## Figures (`figures/`, PDF for LaTeX + PNG)

| File | Content |
|---|---|
| `fig_estimates` | REBA estimates of the three hidden-error tasks over the observations, trimmed mean vs median, with the true and Bosch values |
| `fig_sensitivity` | Reduction of the exposure gap between operators vs no rotation, per sensitivity scenario (mean +- SD, 10 seeds) |
| `fig_regret` | Regret of the plan in use when the Bosch data has errors, Bosch-based vs learned plan, 4-7 operators (bars = mean, dots = seeds) |
| `fig_workforce` | Units per hour, exposure gap, most exposed operator and minutes at REBA >= 4 by headcount, three policies (mean +- SD, 10 seeds). Units per hour are the same for the three policies (the lines overlap) |

## Base scenario (aligned with Ch. 3.4)

| Parameter | Base | Basis |
|---|---|---|
| Task time variation (SD) | 5% | delta = 5% of Ch. 3.4, read as a standard deviation (2.5% in the sensitivity) |
| REBA noise (SD) | 0.94 points | SEM = 1.23 x sqrt(1 - 0.41), Ch. 3.4.3.2 |
| Watch error (SD) | 11% | upper value for less accurate instruments, ref. [190] of Ch. 3.4 |
| Borg noise (SD) | 1.0 point | +-1 by analogy with REBA, Ch. 3.4 |
| Fatigue | none | no basis in Ch. 3; only in the sensitivity analysis |
| MILP weights | w_t = w_e = 0.5 | |

## Rotation at the breaks

Objective as in Job-Rotation-TL: minimise E_max - E_min of the measured cumulative exposure after
the next block. Borg is not in the objective; it enters only through the Borg rule below. (A Borg
term, E~_o = E_o + beta * (Borg_o / 10) * mean forecast exposure of one block, is available and
tested in the sensitivity with beta = 0.5 and 1.)
Constraints: skills and medical restrictions (an operator can work on a loop only if cleared for all
its workstations) and the Borg rule (Borg >= 7: not on the heaviest loop). If no assignment satisfies
them, the Borg rule is dropped first, then eligibility; relaxations are counted in the results.
Skills and restrictions are simulated (separate random generator): each operator is qualified for a
workstation with probability 0.95, and 25% of the operators may not work on the two workstations
with a task of REBA >= 5. With these values a feasible rotation exists on 97% of the days.

## Estimator

Each task's value is the 10% trimmed mean of its last 30 measurements, blended with the Bosch value
(weight n/(n+5)). The median was the first choice, but the median of integer REBA scores jumps by
about 1 point when the window shifts: in the hidden-error test REBA estimates took 600-770
observations to settle with the median against 7-14 with the trimmed mean, and false alarms fell
from about 1% to 0%. Errors of 1 REBA point (the size of the measurement noise, SEM = 0.94) are
detected in 59% of the cases with the trimmed mean: a known limitation of the detection rule.

## Sensitivity analysis (`experiments/sensitivity.py`)

One parameter at a time, 10 seeds, E1 scenario. Levels: time variation 2.5/10%, REBA noise
0.5/1.5, watch error 5/25%, Borg noise 0.5/1.5, fatigue moderate/high, REBA exposure threshold 3,
min gain 0/5%, drift threshold 2/10%. Output: `results_sensitivity/runs.csv` and `summary.csv`
(reduction of the exposure gap vs the static policy, REBA >= 4 minutes, rebalances, estimation error).

## Hidden-error test (`experiments/error_test.py`)

20 seeds: 5 random tasks with a REBA error (-3..+4) and 5 with a time error (+-20%), plus 20 control
runs without errors. Detection when the estimate departs from the Bosch value by more than one
standard deviation of a single measurement (SEM 0.94 / 5% of time) with >= 10 observations;
convergence when it is that close to the true value and stays. Regret: how much worse the plan in use
is than the best plan, both evaluated with the true values. Output: `results_error_test/` (trimmed
mean, default) and `results_error_test_median/` (median, for comparison).

Results are written to `experiments/results_v1/`.

## Using other data

Any line works if the Excel file has the same two sheets as the Fuzzy-ALB dataset (title on row 1,
headers on row 2; other sheets are ignored):

- `Tasks`: `Task ID`, `Station`, `Task Name`, `Time (s)`, `Energy (kcal)`, `REBA`, `Borg`
- `Precedence`: `Predecessor ID`, `Successor ID`

```bash
python -m experiments.run_v1 --dataset other_line.xlsx --hidden none --min-ops 3 --max-ops 6 --out experiments/results_other
```

| Option | Meaning |
|---|---|
| `--dataset` | Excel file of the line (default: Bosch line in Fuzzy-ALB) |
| `--hidden` | tasks whose real REBA is higher than the data, e.g. `5:3,12:2`, or `none` |
| `--min-ops`, `--max-ops` | range of daily headcount |
| `--days` | number of shifts in E1 |
| `--out` | results folder |

Things to check in new data:

- **Station numbers must follow the order of the line.** The workstation-contiguity constraints
  assume it. If more than 20% of the precedences go backwards in station order, the loader warns:
  the MILP would pile most tasks on one operator. (Bosch: 6 of 105 backwards. The random dataset of
  `create_example_dataset.py`: 56 of 124, so its stations must be renumbered first.)
- **No Energy or Borg?** Put 0 in the column; that indicator then drops out of the ERI.
- **Headcount above 8:** rotation tests every assignment (8! = 40,320 is fine, 10! is too slow);
  use the heuristic of Job-Rotation-TL instead.
- **Much larger lines:** 92 tasks solve in 2-3 s; a few hundred tasks may need a longer time limit.
- Other AHP weights or break times: `realtime_alb/config.py`.

Tested on a synthetic 50-task line (stations renumbered in line order), 3 to 6 operators.

## Validation of the MILP port (midpoint instances, 6 operators)

| w_t / w_e | objective HiGHS | objective CPLEX | CT | EL |
|---|---|---|---|---|
| 0 / 1 | 5.02805 | 5.02805 | 4.9937 | 5.0280 |
| 0.25 / 0.75 | 5.01946 | 5.01946 | 4.9937 | 5.0280 |
| 0.5 / 0.5 | 5.01087 | 5.01087 | 4.9937 | 5.0280 |
| 0.75 / 0.25 | 4.98326 | 4.98326 | 4.9255 | 5.1565 |
| 1 / 0 | 4.92550 | 4.92550 | 4.9255 | 5.1565 |

Identical to CPLEX in all five instances, 2-3 s per solve. The validation uses the Ch. 4.4 inputs
(ERI not weighted by time); the time-weighted version only changes the input values, not the model. The tie-break stage retries without
HiGHS presolve when presolve wrongly reports it infeasible (seen on a synthetic line, checked with CBC).

## v1 assumptions to revisit

- All operators have the same task values (decision 2A); individual differences are future work (ALWABP-type extension).
- REBA arrives as one value per task execution (frame-level series and sensor errors come in v3).
- Borg is used for the rotation rule only; the task Borg in the ERI stays at the interview value.
- Breaks: 3 blocks of 2h30 (`config.SHIFT_BLOCKS_MIN`), to be replaced by the real Bosch break times.
- Results come from one random seed; replications are needed before reporting numbers.
