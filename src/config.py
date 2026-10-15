"""Project configuration and path helpers.

The project deliberately keeps paths relative to the repository root so it can
be copied to another machine without changing source code. The root is inferred
from this module's location (``<project>/src/config.py``).
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "images"

DATA_FILE = DATA_DIR / "energy_consumption.csv"
METRICS_JSON = OUTPUTS_DIR / "metrics.json"
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
FORECAST_FILE = OUTPUTS_DIR / "forecast.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
FEATURES_FILE = MODELS_DIR / "feature_columns.json"

DEFAULT_SEED = 42
DEFAULT_DAYS = 365
DEFAULT_FORECAST_HORIZON = 24
TEST_FRACTION = 0.20
TARGET_COLUMN = "consumption_kwh"
TIMESTAMP_COLUMN = "timestamp"


def ensure_project_directories() -> None:
    """Create runtime directories if they do not already exist."""

    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
