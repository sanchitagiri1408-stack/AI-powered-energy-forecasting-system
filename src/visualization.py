"""Plotting helpers for pipeline artifacts.

Matplotlib is imported lazily so data generation and model tests can still run
in environments where plotting dependencies are not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _matplotlib():
    """Return pyplot configured for headless file output."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def create_plots(
    history: pd.DataFrame,
    test_data: pd.DataFrame,
    predictions: dict[str, Any],
    forecast: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Create the requested PNG charts and return their paths."""

    plt = _matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, filename: str) -> None:
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    history_sample = history.tail(min(len(history), 24 * 60))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(history_sample["timestamp"], history_sample["consumption_kwh"], color="#176b87", linewidth=1.2)
    ax.set(title="Synthetic Industrial Energy History", xlabel="Timestamp", ylabel="Consumption (kWh)")
    ax.grid(alpha=0.25)
    save(fig, "energy_history.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test_data["timestamp"], test_data["consumption_kwh"], label="Actual", color="#202a44", linewidth=1.3)
    for name, result in predictions.items():
        ax.plot(test_data["timestamp"], result["predictions"], label=name, linewidth=1.0, alpha=0.85)
    ax.set(title="Chronological Test Backtest", xlabel="Timestamp", ylabel="Consumption (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "actual_vs_predicted.png")

    metric_rows = [(name, result["rmse"]) for name, result in predictions.items()]
    metric_rows.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([row[0] for row in metric_rows], [row[1] for row in metric_rows], color="#e07a5f")
    ax.set(title="Model Comparison by RMSE", ylabel="RMSE (kWh)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(forecast["timestamp"], forecast["consumption_kwh"], color="#3a7d44", marker="o", markersize=3)
    ax.set(title="Recursive Future Energy Forecast", xlabel="Timestamp", ylabel="Forecast consumption (kWh)")
    ax.grid(alpha=0.25)
    save(fig, "future_forecast.png")

    selected = next(
        (result.get("model") for result in predictions.values() if hasattr(result.get("model"), "feature_importances_")),
        None,
    )
    if selected is not None:
        from .features import feature_columns

        importances = pd.Series(selected.feature_importances_, index=feature_columns())
        importances = importances.sort_values(ascending=False).head(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        importances.plot.barh(ax=ax, color="#4c956c")
        ax.set(title="Feature Importance (Tree Model)", xlabel="Importance")
        save(fig, "feature_importance.png")

    return paths
