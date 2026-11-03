"""Check the HiGHS port against the CPLEX results of Ch. 4 (midpoint instances, alpha = 1).

Run from the repo root:  python -m validation.validate_vs_cplex  <path to Fuzzy-ALB/feri-alb-toolkit/alb>
"""
import sys
from pathlib import Path
from realtime_alb.data_io import read_opl_dat
from realtime_alb.balancing import solve_balance

# CPLEX reference (outputCPLEX.txt, cut=1.0 midpoint): objective, cycleTime, maxEriLoad
CPLEX = {
    "wt000_we100": (5.028046, 4.9937, 5.028046),
    "wt025_we075": (5.0194595, 4.9937, 5.028046),
    "wt050_we050": (5.010873, 4.9937, 5.028046),
    "wt075_we025": (4.98325525, 4.9255, 5.156521),
    "wt100_we000": (4.9255, 4.9255, 5.156521),
}

if __name__ == "__main__":
    alb_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "Fuzzy-ALB/feri-alb-toolkit/alb"
    print(f"{'instance':14}{'obj HiGHS':>11}{'obj CPLEX':>11}{'CT':>9}{'EL':>9}{'status':>9}{'sec':>7}")
    for tag, (zc, ctc, elc) in CPLEX.items():
        d = read_opl_dat(alb_dir / f"assembly_line_{tag}_cut10.dat")
        ids = range(1, len(d["time"]) + 1)
        r = solve_balance(dict(zip(ids, d["time"])), dict(zip(ids, d["eri"])), dict(zip(ids, d["ws"])),
                          d["precedence"], d["m"], d["w_time"], d["w_eri"], time_limit=300)
        ok = "OK" if abs(r["objective"] - zc) < 1e-4 else "DIFF"
        print(f"{tag:14}{r['objective']:11.5f}{zc:11.5f}{r['CT']:9.4f}{r['EL']:9.4f}{r['status']:>9}{r['seconds']:7.1f}  {ok}"
              f"   (CPLEX CT={ctc}, EL={elc})")
