# Data directory

This directory is the home for the hourly energy data created by the pipeline. The
project is designed to generate a deterministic **synthetic** data set locally; it does
not download, include, or imply access to a real industrial facility or utility.

## Expected data artifact

The requested public artifact name is:

```text
data/energy_consumption.csv
```

It should contain `24 * days` chronological rows for the selected run. The documented
full run uses 365 days (8,760 raw rows):

```bash
python main.py --days 365 --forecast-horizon 24 --seed 42
```

The smaller learning run is:

```bash
python main.py --days 30 --forecast-horizon 24 --seed 42
```

The source configuration uses this public filename by default. Inspect the generated file after each run so that screenshots and captions match the command you actually executed.

## Schema

| Column | Meaning |
| --- | --- |
| `timestamp` | Hourly UTC-naive timestamp |
| `temperature_c` | Simulated ambient temperature in °C |
| `humidity_pct` | Simulated relative humidity percentage |
| `occupancy_pct` | Simulated facility occupancy percentage |
| `production_index` | Simulated production activity index |
| `renewable_generation_kwh` | Simulated on-site renewable generation per hour |
| `tariff_inr_per_kwh` | Simulated tariff proxy in INR/kWh |
| `is_holiday` | Boolean holiday indicator |
| `consumption_kwh` | Synthetic facility consumption per hour; the target |

The exact required column order is defined by `REQUIRED_COLUMNS` in
`src/data_generator.py`. `load_energy_data` validates required columns, parses the
timestamp, converts numeric columns, removes invalid/duplicate rows, and sorts by time.

## Generation and reproducibility

The generator uses NumPy's seeded random generator and deterministic daily/annual
patterns. It simulates weekday working hours, holidays, weather, occupancy,
production, solar-like generation, tariffs, and a consumption equation. The same
`--days` and `--seed` produce the same synthetic history under a compatible environment.

This data is suitable for tutorials, tests, screenshots, and feature-engineering
practice. It is not suitable for operational decisions, billing, safety controls, or
claims about a real plant.

## Keeping data honest

- Label every chart and report as synthetic or simulated.
- Record the command, seed, date, and dependency environment beside any published
  artifact.
- Do not replace this file with a confidential meter export.
- If real data is used in a future experiment, document permissions, privacy controls,
  units, timezone, missingness, and data ownership separately.
