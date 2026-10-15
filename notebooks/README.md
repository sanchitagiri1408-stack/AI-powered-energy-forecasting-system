# Notebooks directory

The runnable project is script-first: use `main.py` and the modules in `src/`. This
directory is reserved for optional educational notebooks and is intentionally empty of
generated notebooks in the script-first setup.

## Suggested notebooks

If you add notebooks later, keep them thin and import functions from `src/` rather than
copying pipeline logic into cells:

1. `01_explore_synthetic_data.ipynb` — load the generated CSV, inspect schema,
   timestamps, distributions, and daily patterns.
2. `02_feature_engineering.ipynb` — display calendar, cyclical, lag, and rolling
   features and discuss why the current target must not be included.
3. `03_model_comparison.ipynb` — read `outputs/metrics.csv`, plot model errors, and
   compare the seasonal-naive baseline with supervised models.
4. `04_forecast_review.ipynb` — read `outputs/forecast.csv`, inspect the simulated
   future drivers, and display the forecast PNGs.

## Notebook hygiene

- Run notebooks from the project root so imports and relative paths match the script.
- Record the exact CLI command that generated the input artifacts.
- Never hide data generation or preprocessing in an untracked cell.
- Clear accidental secrets, local paths, and huge embedded outputs before committing.
- Keep `.ipynb_checkpoints/` out of Git; `.gitignore` already covers it.
- State clearly that all default data is virtual and synthetic.

## Quick start

Generate data and reports first:

```bash
python main.py --days 30 --forecast-horizon 24 --seed 42
```

Then open a notebook with Jupyter if you choose to install it separately. Jupyter is
not required by `requirements.txt` because the core pipeline and tests do not depend on
it.
