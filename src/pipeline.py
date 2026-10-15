"""End-to-end synthetic energy forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DATA_FILE,
    DEFAULT_DAYS,
    DEFAULT_FORECAST_HORIZON,
    FEATURES_FILE,
    FORECAST_FILE,
    IMAGES_DIR,
    METRICS_CSV,
    METRICS_JSON,
    MODEL_FILE,
    OUTPUTS_DIR,
    ensure_project_directories,
)
from .data_generator import generate_synthetic_energy_data, load_energy_data
from .features import engineer_features, feature_columns
from .modeling import recursive_forecast, save_selected_model, train_and_evaluate
from .visualization import create_plots


def _metrics_records(results: dict[str, Any]) -> list[dict[str, float | str]]:
    """Make serializable model metric records."""

    return [
        {"model": name, "rmse": result["rmse"], "mae": result["mae"], "r2": result["r2"]}
        for name, result in results.items()
    ]


def run_pipeline(
    days: int = DEFAULT_DAYS,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    seed: int = 42,
    n_estimators: int = 100,
    data_path: Path = DATA_FILE,
) -> dict[str, Any]:
    """Run generation, feature engineering, training, forecasting, and plots.

    The source CSV is regenerated for every run to make ``days`` and ``seed``
    explicit and reproducible. If a caller supplies a custom ``data_path``,
    the same generated data is written there and then loaded through the normal
    cleaning path.
    """

    if days <= 0:
        raise ValueError("days must be positive")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    ensure_project_directories()
    generate_synthetic_energy_data(days=days, seed=seed, output_path=data_path)
    raw_data = load_energy_data(data_path)
    engineered = engineer_features(raw_data, drop_target_na=True)
    if len(engineered) < 20:
        raise ValueError("The generated data is too short after lag feature preparation")

    results, best_name, split = train_and_evaluate(
        engineered,
        seed=seed,
        n_estimators=n_estimators,
    )
    selected_model = results[best_name]["model"]

    save_selected_model(selected_model, MODEL_FILE)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_FILE.write_text(json.dumps(feature_columns(), indent=2), encoding="utf-8")

    future_forecast = recursive_forecast(selected_model, raw_data, forecast_horizon, seed=seed)
    future_forecast.to_csv(FORECAST_FILE, index=False)

    records = _metrics_records(results)
    metrics_payload = {
        "selected_model": best_name,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "seed": int(seed),
        "days": int(days),
        "forecast_horizon": int(forecast_horizon),
        "backtest_note": (
            "Test predictions are one-step-ahead backtests using observed lag history. "
            "Future predictions are recursive and feed predicted consumption into lag features."
        ),
        "models": records,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(METRICS_CSV, index=False)

    plot_paths = create_plots(raw_data, split.test, results, future_forecast, IMAGES_DIR)
    return {
        "raw_data": raw_data,
        "engineered_data": engineered,
        "results": results,
        "selected_model": best_name,
        "forecast": future_forecast,
        "metrics": metrics_payload,
        "plot_paths": plot_paths,
        "model_path": MODEL_FILE,
        "feature_path": FEATURES_FILE,
    }
