"""Command-line entry point for the synthetic energy forecast project."""

from __future__ import annotations

import argparse

from src.config import DEFAULT_DAYS, DEFAULT_FORECAST_HORIZON, DEFAULT_SEED
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Generate, evaluate, and forecast synthetic industrial energy use.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of hourly synthetic history days to generate.")
    parser.add_argument("--forecast-horizon", type=int, default=DEFAULT_FORECAST_HORIZON, help="Number of future hourly forecasts.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed.")
    return parser.parse_args()


def main() -> None:
    """Run the pipeline and print a concise artifact summary."""

    args = parse_args()
    result = run_pipeline(days=args.days, forecast_horizon=args.forecast_horizon, seed=args.seed)
    print(f"Selected model: {result['selected_model']}")
    print("Metrics: outputs/metrics.json and outputs/metrics.csv")
    print(f"Forecast rows: {len(result['forecast'])} -> outputs/forecast.csv")
    print(f"Charts generated: {len(result['plot_paths'])} -> images/")


if __name__ == "__main__":
    main()
