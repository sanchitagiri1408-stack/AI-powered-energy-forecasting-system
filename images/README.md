# Images directory

`src/visualization.py` writes headless PNG charts here after a successful pipeline run.
The charts are visual evidence for a synthetic simulation, not measurements from a
real facility.

## Generated PNGs

| File | What it shows |
| --- | --- |
| `energy_history.png` | Recent synthetic hourly consumption history, up to the trailing 60 days. |
| `actual_vs_predicted.png` | Actual target values and all three candidate predictions over the chronological test period. |
| `model_comparison.png` | RMSE bars for SeasonalNaive, LinearRegression, and RandomForestRegressor. |
| `future_forecast.png` | Recursive point forecast for the requested future horizon. |
| `feature_importance.png` | Top feature importances from the included RandomForestRegressor; normally created by the current pipeline and conditional only if the candidate set or plotting inputs change. |

## Generate them

```bash
python main.py --days 365 --forecast-horizon 24 --seed 42
```

For a quicker run:

```bash
python main.py --days 30 --forecast-horizon 24 --seed 42
```

The images use Matplotlib's non-interactive `Agg` backend, so no desktop plotting
window is required. Open the PNGs after the command finishes and add captions such as
“Synthetic industrial energy history, seed 42.”

## Screenshot checklist

- [ ] All charts are readable at their native resolution.
- [ ] Captions identify the data as synthetic/simulated.
- [ ] The actual-versus-predicted chart is described as a chronological backtest.
- [ ] The future chart is described as recursive and driver-simulated.
- [ ] No screenshot contains private paths, secrets, or real customer data.
