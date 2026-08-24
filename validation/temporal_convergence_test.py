"""
Fixed-mesh BDF2 temporal convergence test.

The spatial mesh is held fixed and the time step is refined.  Errors are
measured against a same-mesh reference solution computed with a much smaller
time step, which suppresses the spatial-error floor in the temporal EOC.

Expected:
    EOC_t ~ 2 for velocity and temperature.
"""
from pathlib import Path
import csv
import math
from mpi4py import MPI
from cbf_bdm_solver import CBFProblem, Parameters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

N = 64
tf = 1.0
dts = [0.04, 0.02, 0.01, 0.005]
dt_ref = 0.00125

comm = MPI.COMM_WORLD

ref_problem = CBFProblem(N=N, params=Parameters())
ref = ref_problem.run(dt=dt_ref, tf=tf, lambda_grad=0.0)
u_ref, T_ref = ref["u"], ref["T"]

rows = []
prev_eu = None
prev_eT = None
prev_dt = None

for dt in dts:
    prob = CBFProblem(N=N, params=Parameters())
    sol = prob.run(dt=dt, tf=tf, lambda_grad=0.0)

    eu = prob.vector_l2_difference(sol["u"], u_ref)
    eT = prob.scalar_l2_difference(sol["T"], T_ref)

    if prev_eu is None:
        eoc_u = ""
        eoc_T = ""
    else:
        denom = math.log(prev_dt / dt)
        eoc_u = math.log(prev_eu / eu) / denom
        eoc_T = math.log(prev_eT / eT) / denom

    row = {
        "dt": dt,
        "u_temporal_l2": eu,
        "u_eoc": eoc_u,
        "T_temporal_l2": eT,
        "T_eoc": eoc_T,
        "mean_snes": sol["mean_snes"],
        "max_snes": sol["max_snes"],
    }
    rows.append(row)

    if comm.rank == 0:
        print(
            f"dt={dt:8.5f}  Eu={eu:12.5e}  EOCu={str(eoc_u):>8}  "
            f"ET={eT:12.5e}  EOCT={str(eoc_T):>8}"
        )

    prev_eu, prev_eT, prev_dt = eu, eT, dt

if comm.rank == 0:
    path = OUT / "temporal_convergence.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
