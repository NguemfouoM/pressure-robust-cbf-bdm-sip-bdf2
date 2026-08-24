"""Generate publication-ready PDF figures from the two CSV files."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "validation"
F = ROOT / "figures" / "validation"
F.mkdir(parents=True, exist_ok=True)

# Pressure robustness
pfile = R / "pressure_robustness.csv"
if pfile.exists():
    rows = list(csv.DictReader(pfile.open()))
    lam = np.array([float(r["lambda"]) for r in rows])
    eu = np.array([float(r["u_invariance_l2"]) for r in rows])
    eT = np.array([float(r["T_invariance_l2"]) for r in rows])

    # Use 1+lambda so lambda=0 is visible on logarithmic axis.
    x = 1.0 + lam
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(x, eu, marker="o", label=r"$\|u_h^\lambda-u_h^0\|_{L^2}$")
    ax.loglog(x, eT, marker="s", label=r"$\|T_h^\lambda-T_h^0\|_{L^2}$")
    ax.set_xlabel(r"$1+\lambda$")
    ax.set_ylabel("Invariance defect")
    ax.set_title("Gradient-force pressure-robustness test")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "pressure_robustness.pdf")
    plt.close(fig)

# Temporal convergence
tfile = R / "temporal_convergence.csv"
if tfile.exists():
    rows = list(csv.DictReader(tfile.open()))
    dt = np.array([float(r["dt"]) for r in rows])
    eu = np.array([float(r["u_temporal_l2"]) for r in rows])
    eT = np.array([float(r["T_temporal_l2"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(dt, eu, marker="o", label=r"$u$")
    ax.loglog(dt, eT, marker="s", label=r"$T$")
    ref = eu[-1] * (dt / dt[-1])**2
    ax.loglog(dt, ref, linestyle="--", label=r"$O(\tau^2)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Same-mesh temporal $L^2$ error")
    ax.set_title("BDF2 temporal convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "temporal_convergence.pdf")
    plt.close(fig)

print(f"Figures written to {F}")
