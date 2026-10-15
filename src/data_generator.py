"""Deterministic synthetic industrial energy data generation.

The generator creates hourly weather, operational, tariff, holiday, renewable
generation, and consumption signals. It has no network or external-data
dependency, making it suitable for demos, tests, and classroom exercises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_FILE, DEFAULT_DAYS, ensure_project_directories


REQUIRED_COLUMNS = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "occupancy_pct",
    "production_index",
    "renewable_generation_kwh",
    "tariff_inr_per_kwh",
    "is_holiday",
    "consumption_kwh",
]


def generate_synthetic_energy_data(
    days: int = DEFAULT_DAYS,
    seed: int = 42,
    start: str = "2024-01-01",
    output_path: Path | None = DATA_FILE,
) -> pd.DataFrame:
    """Generate an hourly synthetic industrial energy-use data set.

    Args:
        days: Number of complete days to generate. Must be positive.
        seed: Seed for all pseudo-random draws.
        start: First timestamp, interpreted as an hourly UTC-naive timestamp.
        output_path: Optional CSV destination. Pass ``None`` to avoid writing.

    Returns:
        A dataframe with exactly ``24 * days`` rows and :data:`REQUIRED_COLUMNS`.
    """

    if days <= 0:
        raise ValueError("days must be a positive integer")

    rng = np.random.default_rng(seed)
    periods = days * 24
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    # Seasonal temperature plus a repeatable daily cycle and small noise.
    seasonal_temperature = 28.0 + 7.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    daily_temperature = 2.5 * np.sin(2 * np.pi * (hour - 6) / 24)
    temperature = seasonal_temperature + daily_temperature + rng.normal(0, 1.2, periods)
    humidity = (
        67.0
        - 0.65 * (temperature - 26.0)
        + 7.0 * np.sin(2 * np.pi * (hour + 2) / 24)
        + rng.normal(0, 3.0, periods)
    )
    humidity = np.clip(humidity, 25.0, 95.0)

    is_weekend = day_of_week >= 5
    working_hours = (hour >= 7) & (hour <= 19)
    holiday_dates = {1, 26, 50, 90, 120, 160, 200, 250, 300, 330, 359}
    is_holiday = np.isin(day_of_year, list(holiday_dates))
    occupancy = np.where(working_hours & ~is_weekend & ~is_holiday, 78.0, 18.0)
    occupancy += 8.0 * np.sin(2 * np.pi * (hour - 8) / 24)
    occupancy += rng.normal(0, 4.0, periods)
    occupancy = np.clip(occupancy, 0.0, 100.0)

    production = np.where(working_hours & ~is_weekend & ~is_holiday, 1.0, 0.18)
    production += 0.12 * np.sin(2 * np.pi * (hour - 9) / 24)
    production += rng.normal(0, 0.06, periods)
    production = np.clip(production, 0.05, 1.25)

    solar_profile = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12))
    seasonal_solar = 1.0 + 0.12 * np.sin(2 * np.pi * (day_of_year - 20) / 365.25)
    renewable = 105.0 * solar_profile * seasonal_solar + rng.normal(0, 2.0, periods)
    renewable = np.clip(renewable, 0.0, None)

    peak_period = ((hour >= 9) & (hour <= 17)).astype(float)
    tariff = 7.0 + 2.1 * peak_period + 0.45 * np.sin(2 * np.pi * day_of_year / 365.25)
    tariff += rng.normal(0, 0.08, periods)
    tariff = np.clip(tariff, 4.5, 10.5)

    # Demand has a stable industrial base, weather/HVAC effects, production
    # load, occupancy, tariff response, and noise. Renewable output offsets
    # grid consumption but does not make total facility demand negative.
    base_load = 180.0
    hvac_load = 4.6 * np.maximum(temperature - 22.0, 0.0) + 1.2 * np.maximum(18.0 - temperature, 0.0)
    operational_load = 225.0 * production + 0.85 * occupancy
    tariff_response = -3.0 * (tariff - 7.0)
    daily_cycle = 18.0 * np.sin(2 * np.pi * (hour - 7) / 24)
    noise = rng.normal(0.0, 12.0, periods)
    consumption = base_load + hvac_load + operational_load + tariff_response + daily_cycle - 0.28 * renewable + noise
    consumption = np.clip(consumption, 50.0, None)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": np.round(temperature, 3),
            "humidity_pct": np.round(humidity, 3),
            "occupancy_pct": np.round(occupancy, 3),
            "production_index": np.round(production, 4),
            "renewable_generation_kwh": np.round(renewable, 3),
            "tariff_inr_per_kwh": np.round(tariff, 3),
            "is_holiday": is_holiday.astype(bool),
            "consumption_kwh": np.round(consumption, 3),
        }
    )
    data = data[REQUIRED_COLUMNS]

    if output_path is not None:
        ensure_project_directories()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
    return data


def load_energy_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load, validate, clean, and chronologically sort an energy CSV file."""

    if not Path(path).exists():
        raise FileNotFoundError(f"Energy data file does not exist: {path}")
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Energy data is missing required columns: {missing}")

    data = data[REQUIRED_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    numeric_columns = [column for column in REQUIRED_COLUMNS if column not in {"timestamp", "is_holiday"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["is_holiday"] = data["is_holiday"].astype("boolean")
    data = data.dropna(subset=REQUIRED_COLUMNS).drop_duplicates(subset=["timestamp"])
    data["is_holiday"] = data["is_holiday"].astype(bool)
    return data.sort_values("timestamp").reset_index(drop=True)
