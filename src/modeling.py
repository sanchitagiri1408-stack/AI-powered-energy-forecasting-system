"""Model training, chronological evaluation, and recursive forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TEST_FRACTION, TARGET_COLUMN, TIMESTAMP_COLUMN
from .features import prepare_model_matrix


@dataclass(frozen=True)
class ChronologicalSplit:
    """Container for a chronological train/test split."""

    train: pd.DataFrame
    test: pd.DataFrame
    split_index: int


class SeasonalNaiveModel:
    """A feature-compatible seasonal naive estimator for 24-hour seasonality."""

    def __init__(self, season: int = 24) -> None:
        self.season = season

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict from the corresponding seasonal lag feature."""

        column = f"consumption_lag_{self.season}"
        if column not in features:
            raise ValueError(f"Seasonal naive prediction needs {column!r}")
        return features[column].to_numpy(dtype=float)


def chronological_split(data: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> ChronologicalSplit:
    """Split rows in time order, reserving the final fraction for testing."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    frame = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    split_index = int(len(frame) * (1.0 - test_fraction))
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("Not enough rows for the requested chronological split")
    return ChronologicalSplit(frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy(), split_index)


def seasonal_naive_predict(train: pd.DataFrame, test: pd.DataFrame, season: int = 24) -> np.ndarray:
    """Predict each test target from the observed value one seasonal period ago.

    The backtest intentionally uses observed history, including rows before the
    test window. This is a one-step-ahead backtest baseline, while future
    forecasts below are recursive and use predictions as lag values.
    """

    combined = pd.concat([train, test], ignore_index=True)
    values = combined[TARGET_COLUMN].to_numpy(dtype=float)
    start = len(train)
    predictions = []
    for position in range(start, len(combined)):
        source = position - season
        predictions.append(values[source] if source >= 0 else values[start - 1])
    return np.asarray(predictions, dtype=float)


def build_models(seed: int = 42, n_estimators: int = 100) -> dict[str, Any]:
    """Construct the baseline and supervised estimators."""

    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features=0.8,
        ),
    }


def calculate_metrics(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> dict[str, float]:
    """Calculate the requested regression metrics."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "r2": float(r2_score(actual_array, predicted_array)),
    }


def train_and_evaluate(
    engineered_data: pd.DataFrame,
    seed: int = 42,
    n_estimators: int = 100,
    test_fraction: float = TEST_FRACTION,
) -> tuple[dict[str, Any], str, ChronologicalSplit]:
    """Train all models, evaluate them, and return the best estimator by RMSE.

    Feature rows for the test period use observed target lags. This supports a
    fair one-step-ahead backtest and is explicitly different from the recursive
    future forecast path in :func:`recursive_forecast`.
    """

    split = chronological_split(engineered_data, test_fraction=test_fraction)
    x_train, y_train = prepare_model_matrix(split.train)
    x_test, y_test = prepare_model_matrix(split.test)

    results: dict[str, Any] = {}
    baseline_prediction = seasonal_naive_predict(split.train, split.test)
    results["SeasonalNaive"] = {
        **calculate_metrics(y_test, baseline_prediction),
        "predictions": baseline_prediction,
        "model": SeasonalNaiveModel(season=24),
    }

    for name, model in build_models(seed=seed, n_estimators=n_estimators).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            **calculate_metrics(y_test, predictions),
            "predictions": predictions,
            "model": model,
        }

    best_name = min(results, key=lambda name: results[name]["rmse"])
    return results, best_name, split


def save_selected_model(model: Any, path: Path) -> None:
    """Persist a fitted estimator using joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _future_exogenous_rows(last_timestamp: pd.Timestamp, periods: int, seed: int) -> pd.DataFrame:
    """Simulate deterministic exogenous drivers for future timestamps."""

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = day_of_week >= 5
    working_hours = (hour >= 7) & (hour <= 19)
    holiday_dates = {1, 26, 50, 90, 120, 160, 200, 250, 300, 330, 359}
    is_holiday = np.isin(day_of_year, list(holiday_dates))

    temperature = 28.0 + 7.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    temperature += 2.5 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.2, periods)
    humidity = 67.0 - 0.65 * (temperature - 26.0) + 7.0 * np.sin(2 * np.pi * (hour + 2) / 24)
    humidity = np.clip(humidity + rng.normal(0, 3.0, periods), 25.0, 95.0)
    occupancy = np.where(working_hours & ~is_weekend & ~is_holiday, 78.0, 18.0)
    occupancy = np.clip(occupancy + 8.0 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 4.0, periods), 0.0, 100.0)
    production = np.where(working_hours & ~is_weekend & ~is_holiday, 1.0, 0.18)
    production = np.clip(production + 0.12 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.06, periods), 0.05, 1.25)
    solar_profile = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12))
    renewable = 105.0 * solar_profile * (1.0 + 0.12 * np.sin(2 * np.pi * (day_of_year - 20) / 365.25))
    renewable = np.clip(renewable + rng.normal(0, 2.0, periods), 0.0, None)
    tariff = 7.0 + 2.1 * ((hour >= 9) & (hour <= 17)).astype(float) + 0.45 * np.sin(2 * np.pi * day_of_year / 365.25)
    tariff = np.clip(tariff + rng.normal(0, 0.08, periods), 4.5, 10.5)

    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "occupancy_pct": occupancy,
            "production_index": production,
            "renewable_generation_kwh": renewable,
            "tariff_inr_per_kwh": tariff,
            "is_holiday": is_holiday,
        }
    )


def _single_feature_row(frame: pd.DataFrame, target_history: list[float]) -> pd.DataFrame:
    """Build the feature row for one forecast timestamp from known history."""

    row = frame.copy()
    timestamp = row[TIMESTAMP_COLUMN]
    row["hour"] = timestamp.dt.hour.astype(int)
    row["day_of_week"] = timestamp.dt.dayofweek.astype(int)
    row["day_of_month"] = timestamp.dt.day.astype(int)
    row["day_of_year"] = timestamp.dt.dayofyear.astype(int)
    row["month"] = timestamp.dt.month.astype(int)
    row["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)
    row["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    row["is_business_day"] = (timestamp.dt.dayofweek < 5).astype(int)
    row["hour_sin"] = np.sin(2 * np.pi * row["hour"] / 24)
    row["hour_cos"] = np.cos(2 * np.pi * row["hour"] / 24)
    row["day_of_year_sin"] = np.sin(2 * np.pi * row["day_of_year"] / 365.25)
    row["day_of_year_cos"] = np.cos(2 * np.pi * row["day_of_year"] / 365.25)

    if len(target_history) < 168:
        raise ValueError("At least 168 historical target values are required for recursive forecasting")
    history = pd.Series(target_history, dtype=float)
    row["consumption_lag_1"] = history.iloc[-1]
    row["consumption_lag_24"] = history.iloc[-24]
    row["consumption_lag_168"] = history.iloc[-168]
    row["consumption_rolling_mean_24"] = history.iloc[-24:].mean()
    row["consumption_rolling_std_24"] = history.iloc[-24:].std()
    row["consumption_rolling_mean_168"] = history.iloc[-168:].mean()
    return row


def recursive_forecast(
    model: Any,
    history: pd.DataFrame,
    horizon: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Forecast future consumption recursively using predicted lag values.

    Future weather, operations, renewable generation, tariff, and holiday
    values are simulated deterministically from ``seed``. Each prediction is
    appended to target history before constructing the next row's lag features.
    """

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    ordered_history = history.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if len(ordered_history) < 168:
        raise ValueError("At least 168 historical rows are required for forecasting")
    future_drivers = _future_exogenous_rows(ordered_history[TIMESTAMP_COLUMN].iloc[-1], horizon, seed)
    target_history = ordered_history[TARGET_COLUMN].astype(float).tolist()
    predictions: list[float] = []

    for row_number in range(horizon):
        driver_row = future_drivers.iloc[[row_number]].copy()
        feature_row = _single_feature_row(driver_row, target_history)
        x_row, _ = prepare_model_matrix(feature_row.assign(**{TARGET_COLUMN: 0.0}))
        prediction = float(model.predict(x_row)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        target_history.append(prediction)

    forecast = future_drivers.copy()
    forecast[TARGET_COLUMN] = predictions
    return forecast
