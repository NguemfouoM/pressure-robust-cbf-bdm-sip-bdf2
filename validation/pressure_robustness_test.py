"""
Pressure-robustness test for Proposition 5.1.

Runs the same BDM-DG/SIPG-BDF2 problem for
    f_lambda = f + lambda * grad(phi),
lambda in {0, 1, 1e2, 1e4, 1e6},
and reports the change in velocity and temperature relative to lambda=0.

Expected result:
    ||u_h^lambda-u_h^0||_L2 and ||T_h^lambda-T_h^0||_L2
remain at the nonlinear/algebraic solver tolerance (ideally near roundoff).

The pressure changes by the projected gradient potential, up to the chosen
pressure gauge.
"""
from pathlib import Path
import csv
import numpy as np
from mpi4py import MPI
from cbf_bdm_solver import CBFProblem, Parameters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

N = 48
dt = 0.01
tf = 1.0
lambdas = [0.0, 1.0, 1.0e2, 1.0e4, 1.0e6]

comm = MPI.COMM_WORLD
rows = []

baseline_problem = CBFProblem(N=N, params=Parameters())
baseline = baseline_problem.run(dt=dt, tf=tf, lambda_grad=0.0)
u0, p0, T0 = baseline["u"], baseline["p"], baseline["T"]

for lam in lambdas:
    if lam == 0.0:
        prob = baseline_problem
        sol = baseline
    else:
        prob = CBFProblem(N=N, params=Parameters())
        sol = prob.run(dt=dt, tf=tf, lambda_grad=lam)

    du = prob.vector_l2_difference(sol["u"], u0)
    dT = prob.scalar_l2_difference(sol["T"], T0)

    row = {
        "lambda": lam,
        "u_invariance_l2": du,
        "T_invariance_l2": dT,
        "mean_snes": sol["mean_snes"],
        "max_snes": sol["max_snes"],
    }
    rows.append(row)

    if comm.rank == 0:
        print(
            f"lambda={lam:10.3e}  "
            f"||u_lam-u_0||={du:12.5e}  "
            f"||T_lam-T_0||={dT:12.5e}  "
            f"mean SNES={sol['mean_snes']:.3f}"
        )

if comm.rank == 0:
    path = OUT / "pressure_robustness.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
