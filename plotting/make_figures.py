from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from plotting.journal_style import COLORS, MARKERS, finish_axis, new_figure, save_figure


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")


def convergence_figure(csv_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(csv_path).sort_values("h", ascending=False)
    columns = ["h", "u_l2", "u_dg", "p_l2", "T_l2", "T_h1"]
    _require_columns(data, columns)
    fig, axes = new_figure()
    ax = axes[0]
    curves = [
        ("u_l2", r"$\|u-u_h\|_{L^2}$", "velocity_l2"),
        ("u_dg", r"$\|u-u_h\|_{1,h}$", "velocity_dg"),
        ("p_l2", r"$\|p-p_h\|_{L^2}$", "pressure_l2"),
    ]
    for i, (column, label, color) in enumerate(curves):
        ax.loglog(data.h, data[column], marker=MARKERS[i], color=COLORS[color], label=label)
    href = np.asarray(data.h)
    anchor = float(data.u_l2.iloc[-1])
    ax.loglog(href, anchor * (href / href[-1]) ** 2, "--", color=COLORS["reference"], label=r"$O(h^2)$")
    finish_axis(ax, xlabel=r"Mesh size $h$", ylabel="Error", title="(a) Flow variables")
    ax.legend(frameon=False)

    ax = axes[1]
    for i, (column, label, color) in enumerate(
        [("T_l2", r"$\|T-T_h\|_{L^2}$", "temperature_l2"),
         ("T_h1", r"$\|T-T_h\|_{H^1}$", "temperature_h1")]
    ):
        ax.loglog(data.h, data[column], marker=MARKERS[i + 3], color=COLORS[color], label=label)
    anchor_l2 = float(data.T_l2.iloc[-1])
    anchor_h1 = float(data.T_h1.iloc[-1])
    ax.loglog(href, anchor_l2 * (href / href[-1]) ** 2, "--", color=COLORS["reference"], label=r"$O(h^2)$")
    ax.loglog(href, anchor_h1 * (href / href[-1]), ":", color=COLORS["reference"], label=r"$O(h)$")
    finish_axis(ax, xlabel=r"Mesh size $h$", ylabel="Error", title="(b) Temperature")
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "manufactured_spatial_convergence")


def mass_figure(csv_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(csv_path).sort_values("h", ascending=False)
    _require_columns(data, ["h", "divergence_l2", "flux_imbalance"])
    fig, axes = new_figure(width=3.45, height=3.0, ncols=1)
    ax = axes[0]
    ax.loglog(data.h, data.divergence_l2, "o-", color=COLORS["mass"], label=r"$\|\nabla\!\cdot u_h\|_{L^2}$")
    ax.loglog(data.h, data.flux_imbalance, "s--", color=COLORS["velocity_l2"], label="Maximum cell-flux imbalance")
    finish_axis(ax, xlabel=r"Mesh size $h$", ylabel="Mass defect", title="Exact mass-conservation diagnostics")
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "manufactured_mass_conservation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    source = args.results / "manufactured_convergence.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run the manufactured experiment first: {source} does not exist")
    convergence_figure(source, args.figures)
    mass_figure(source, args.figures)


if __name__ == "__main__":
    main()
