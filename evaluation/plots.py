"""Chart generation for the README: SHAP feature importance and calibration
reliability diagrams. Kept separate from shap_report.py/metrics.py, which
compute the underlying tables independent of any plotting library.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Intermediate calibration tables (one model's report script writes its own
# table here; whichever of report_xgboost.py/report_torch.py runs second
# reads the other's table to assemble the combined comparison chart). Lives
# under artifacts/ since, like the model files there, it's a regenerable
# byproduct of a training/report run, not source.
ARTIFACT_CAL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "calibration"
ARTIFACT_CAL_DIR.mkdir(parents=True, exist_ok=True)

_BAR_COLOR = "#4C72B0"


def plot_shap_bar(table: pl.DataFrame, title: str, out_name: str) -> Path:
    """Horizontal bar chart of mean(|SHAP value|) per feature, most important
    at the top. `table` is the output of evaluation.shap_report.mean_abs_shap_table."""
    ordered = table.sort("mean_abs_shap")
    features = ordered["feature"].to_list()
    values = ordered["mean_abs_shap"].to_list()

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(features) + 1.2))
    ax.barh(features, values, color=_BAR_COLOR)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration(
    curves: dict[str, pl.DataFrame],
    out_name: str,
    title: str = "Calibration: predicted vs. actual delay rate",
) -> Path:
    """Reliability diagram for one or more models, each given as a
    (mean_predicted, frac_delayed) table from evaluation.metrics.calibration_table.
    Includes the y=x perfect-calibration reference line.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for (name, table), color in zip(curves.items(), colors):
        ax.plot(
            table["mean_predicted"].to_numpy(),
            table["frac_delayed"].to_numpy(),
            marker="o",
            color=color,
            label=name,
        )

    ax.set_xlabel("Mean predicted probability (per decile)")
    ax.set_ylabel("Actual delay rate")
    ax.set_title(title)
    ax.set_xlim(0, max(0.8, ax.get_xlim()[1]))
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
