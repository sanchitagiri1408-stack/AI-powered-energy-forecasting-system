"""Fast tests for the synthetic energy forecasting pipeline."""

from __future__ import annotations

import json

import pandas as pd

from src.data_generator import REQUIRED_COLUMNS, generate_synthetic_energy_data
from src.features import engineer_features, feature_columns
from src.modeling import chronological_split
from src.pipeline import run_pipeline


def test_generated_schema_and_row_count() -> None:
    data = generate_synthetic_energy_data(days=3, seed=7, output_path=None)
    assert list(data.columns) == REQUIRED_COLUMNS
    assert len(data) == 3 * 24
    assert data["timestamp"].is_monotonic_increasing
    assert data["is_holiday"].dtype == bool


def test_feature_generation_has_no_target_nans_after_dropna() -> None:
    data = generate_synthetic_energy_data(days=15, seed=7, output_path=None)
    engineered = engineer_features(data)
    assert len(engineered) > 0
    assert not engineered[feature_columns()].isna().any().any()
    assert not engineered["consumption_kwh"].isna().any()


def test_chronological_split_preserves_time_order() -> None:
    data = generate_synthetic_energy_data(days=30, seed=7, output_path=None)
    engineered = engineer_features(data)
    split = chronological_split(engineered, test_fraction=0.2)
    assert split.train["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.test) == len(engineered) - split.split_index
    assert split.split_index == int(len(engineered) * 0.8)


def test_small_end_to_end_run_writes_metrics(tmp_path, monkeypatch) -> None:
    import src.config as config

    output_dir = tmp_path / "outputs"
    model_dir = tmp_path / "models"
    image_dir = tmp_path / "images"
    data_dir = tmp_path / "data"
    for directory in (output_dir, model_dir, image_dir, data_dir):
        directory.mkdir()
    monkeypatch.setattr(config, "OUTPUTS_DIR", output_dir)
    monkeypatch.setattr(config, "MODELS_DIR", model_dir)
    monkeypatch.setattr(config, "IMAGES_DIR", image_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)

    # pipeline imports artifact constants directly, so point those module-level
    # references at the same temporary paths for an isolated end-to-end test.
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", output_dir)
    monkeypatch.setattr(pipeline, "IMAGES_DIR", image_dir)
    monkeypatch.setattr(pipeline, "METRICS_JSON", output_dir / "metrics.json")
    monkeypatch.setattr(pipeline, "METRICS_CSV", output_dir / "metrics.csv")
    monkeypatch.setattr(pipeline, "FORECAST_FILE", output_dir / "forecast.csv")
    monkeypatch.setattr(pipeline, "MODEL_FILE", model_dir / "selected_model.joblib")
    monkeypatch.setattr(pipeline, "FEATURES_FILE", model_dir / "feature_list.json")

    result = run_pipeline(days=30, forecast_horizon=24, seed=11, n_estimators=20, data_path=data_dir / "energy.csv")
    assert result["forecast"].shape[0] == 24
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "forecast.csv").exists()
    assert (model_dir / "selected_model.joblib").exists()
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["selected_model"] in {"SeasonalNaive", "LinearRegression", "RandomForestRegressor"}
    assert {row["model"] for row in payload["models"]} == {"SeasonalNaive", "LinearRegression", "RandomForestRegressor"}
    assert pd.read_csv(output_dir / "metrics.csv").shape[0] == 3
