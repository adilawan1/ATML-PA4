"""Shared matplotlib style for report figures (static PDF/PNG for the LaTeX report).

Categorical colours use a fixed, CVD-validated order (never cycled; colour follows the entity);
magnitudes use a single-hue light->dark blue ramp; one y-axis per chart; recessive grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa4.common.utils import results_root  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 4, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def color_map(names):
    """Stable entity -> colour assignment in the fixed categorical order."""
    return {n: CAT[i % len(CAT)] for i, n in enumerate(names)}


def fig_dir(task: str) -> Path:
    p = results_root() / "figures" / task
    p.mkdir(parents=True, exist_ok=True)
    return p


def table_dir() -> Path:
    p = results_root() / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, task: str, name: str):
    d = fig_dir(task)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("figure:", d / f"{name}.pdf")


def write_table(df, name: str, float_fmt="%.4g", caption: str = ""):
    d = table_dir()
    df.to_csv(d / f"{name}.csv", index=False)
    try:
        tex = df.to_latex(index=False, float_format=float_fmt.__mod__ if isinstance(float_fmt, str) else float_fmt,
                          escape=True, caption=caption or None, label=f"tab:{name}")
        (d / f"{name}.tex").write_text(tex)
    except Exception as e:  # jinja2 missing etc.
        print("latex export failed:", e)
    print("table:", d / f"{name}.csv")
