from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "velocity_l2": "#0072B2",
    "velocity_dg": "#56B4E9",
    "pressure_l2": "#009E73",
    "temperature_l2": "#D55E00",
    "temperature_h1": "#E69F00",
    "mass": "#CC79A7",
    "reference": "#4D4D4D",
}

MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.2,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.28,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
        }
    )


def new_figure(width: float = 6.8, height: float = 3.05, ncols: int = 2):
    apply_style()
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    return fig, axes


def finish_axis(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.12)
    ax.tick_params(direction="in", top=True, right=True)


def save_figure(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
