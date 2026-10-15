"""Feature engineering for chronological energy forecasting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN, TIMESTAMP_COLUMN


CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "month",
    "week_of_year",
    "is_weekend",
    "is_business_day",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
WEATHER_FEATURES = ["temperature_c", "humidity_pct"]
OPERATIONAL_FEATURES = [
    "occupancy_pct",
    "production_index",
    "renewable_generation_kwh",
    "tariff_inr_per_kwh",
    "is_holiday",
]
LAG_FEATURES = ["consumption_lag_1", "consumption_lag_24", "consumption_lag_168"]
ROLLING_FEATURES = ["consumption_rolling_mean_24", "consumption_rolling_std_24", "consumption_rolling_mean_168"]


def engineer_features(data: pd.DataFrame, drop_target_na: bool = True) -> pd.DataFrame:
    """Add calendar, weather, operational, lag, and rolling features.

    Lagged and rolling features are built only from prior target observations:
    rolling windows are shifted by one hour before aggregation. Consequently,
    training rows never use the current target as an input. The first 168 rows
    have incomplete history and are dropped by default.
    """

    if TIMESTAMP_COLUMN not in data or TARGET_COLUMN not in data:
        raise ValueError(f"Data must contain {TIMESTAMP_COLUMN!r} and {TARGET_COLUMN!r}")

    frame = data.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN])
    frame = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    timestamp = frame[TIMESTAMP_COLUMN]
    iso_calendar = timestamp.dt.isocalendar()

    frame["hour"] = timestamp.dt.hour.astype(int)
    frame["day_of_week"] = timestamp.dt.dayofweek.astype(int)
    frame["day_of_month"] = timestamp.dt.day.astype(int)
    frame["day_of_year"] = timestamp.dt.dayofyear.astype(int)
    frame["month"] = timestamp.dt.month.astype(int)
    frame["week_of_year"] = iso_calendar.week.astype(int)
    frame["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    frame["is_business_day"] = (timestamp.dt.dayofweek < 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)

    target_history = frame[TARGET_COLUMN].shift(1)
    frame["consumption_lag_1"] = frame[TARGET_COLUMN].shift(1)
    frame["consumption_lag_24"] = frame[TARGET_COLUMN].shift(24)
    frame["consumption_lag_168"] = frame[TARGET_COLUMN].shift(168)
    frame["consumption_rolling_mean_24"] = target_history.rolling(24, min_periods=24).mean()
    frame["consumption_rolling_std_24"] = target_history.rolling(24, min_periods=24).std()
    frame["consumption_rolling_mean_168"] = target_history.rolling(168, min_periods=168).mean()

    if drop_target_na:
        frame = frame.dropna(subset=feature_columns() + [TARGET_COLUMN]).reset_index(drop=True)
    return frame


def feature_columns() -> list[str]:
    """Return the ordered feature names used by the estimators."""

    return CALENDAR_FEATURES + WEATHER_FEATURES + OPERATIONAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES


def prepare_model_matrix(data: pd.DataFrame, columns: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model matrix and target from an engineered dataframe."""

    selected = list(columns) if columns is not None else feature_columns()
    missing = [column for column in selected + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    x = data[selected].copy().astype(float)
    y = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(float)
    return x, y
