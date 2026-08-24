from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150,
    "savefig.dpi": 450, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _tag(domain: str, beta: float) -> str:
    return f"{domain}_beta_{beta:g}".replace(".", "p")


def _nodal_gradient(tri: mtri.Triangulation, values: np.ndarray) -> np.ndarray:
    cells = tri.triangles
    xy = np.column_stack((tri.x, tri.y))
    out = np.zeros(len(values)); weight = np.zeros(len(values))
    for cell in cells:
        pts = xy[cell]
        A = np.column_stack((pts[:, 0], pts[:, 1], np.ones(3)))
        try:
            gx, gy, _ = np.linalg.solve(A, values[cell])
        except np.linalg.LinAlgError:
            continue
        magnitude = np.hypot(gx, gy)
        out[cell] += magnitude; weight[cell] += 1.0
    return out / np.maximum(weight, 1.0)


def field_figure(snapshot: Path, output: Path, title: str) -> None:
    data = np.load(snapshot)
    xy, cells = data["xy"], data["cells"].astype(int)
    velocity, temperature = data["velocity"], data["temperature"]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], cells)
    speed = np.linalg.norm(velocity, axis=1)
    grad_T = _nodal_gradient(tri, temperature)
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.15), constrained_layout=True)

    axes[0].triplot(tri, lw=0.18, color="0.25")
    axes[0].set_title("(a) Mesh")

    im = axes[1].tricontourf(tri, speed, levels=24, cmap="viridis")
    stride = max(1, len(xy) // 280)
    axes[1].quiver(xy[::stride, 0], xy[::stride, 1], velocity[::stride, 0],
                   velocity[::stride, 1], color="white", scale=None, width=0.006)
    fig.colorbar(im, ax=axes[1], shrink=0.76, pad=0.02)
    axes[1].set_title(r"(b) $|\mathbf{u}_h|$")

    im = axes[2].tricontourf(tri, temperature, levels=24, cmap="inferno")
    axes[2].tricontour(tri, temperature, levels=9, colors="white", linewidths=0.35, alpha=0.7)
    fig.colorbar(im, ax=axes[2], shrink=0.76, pad=0.02)
    axes[2].set_title(r"(c) $T_h$")

    im = axes[3].tricontourf(tri, grad_T, levels=24, cmap="magma")
    fig.colorbar(im, ax=axes[3], shrink=0.76, pad=0.02)
    axes[3].set_title(r"(d) $|\nabla T_h|$")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, y=1.02, fontsize=9.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def sensitivity_figure(csv_path: Path, output: Path) -> None:
    frame = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    labels = {"l_shape": "L-shaped cavity", "obstacle_enclosure": "Perforated enclosure"}
    styles = {"l_shape": "o-", "obstacle_enclosure": "s--"}
    for domain, group in frame.groupby("domain"):
        group = group.sort_values("beta_f")
        x = group.beta_f + 1.0
        axes[0].semilogx(x, group.velocity_l2, styles[domain], label=labels[domain])
        axes[1].semilogx(x, group.temperature_l2, styles[domain])
        axes[2].semilogx(x, group.boundary_heat_flux.abs(), styles[domain])
    axes[0].set_ylabel(r"$\|\mathbf{u}_h\|_{L^2}$")
    axes[1].set_ylabel(r"$\|T_h\|_{L^2}$")
    axes[2].set_ylabel("Absolute boundary heat flux")
    for i, ax in enumerate(axes):
        ax.set_xlabel(r"$1+\beta_F$")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"({chr(97+i)})")
    axes[0].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    fields = args.results / "fields"
    field_figure(fields / f"{_tag('l_shape', 1.0)}.npz",
                 args.figures / "l_shape_thermal_flow",
                 r"L-shaped porous cavity, $\beta_F=1$")
    field_figure(fields / f"{_tag('obstacle_enclosure', 1.0)}.npz",
                 args.figures / "obstacle_enclosure_thermal_flow",
                 r"Porous enclosure with cold inclusions, $\beta_F=1$")
    sensitivity_figure(args.results / "complex_domain_diagnostics.csv",
                       args.figures / "forchheimer_sensitivity")


if __name__ == "__main__":
    main()

