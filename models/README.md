# Models directory

This directory stores the selected fitted estimator and the ordered feature metadata
needed to understand how the estimator was trained. A model binary is a runtime
artifact, not proof that a production model has been validated.

## Requested artifacts

| File | Purpose |
| --- | --- |
| `models/best_model.joblib` | Fitted SeasonalNaive, LinearRegression, or RandomForestRegressor selected by lowest chronological-test RMSE. |
| `models/feature_columns.json` | Ordered feature names supplied to supervised estimators. |

The source configuration uses the public filenames shown above. Inspect the generated artifacts after each run; do not load a joblib file from an untrusted source.

## How the model is selected

The pipeline trains all three candidates on the first 80% of engineered rows and
evaluates the final 20% in chronological order. The selected model is the one with the
lowest RMSE. The seasonal-naive baseline is included so that a more complex estimator
has to demonstrate value against a simple daily-repeat rule.

## Loading guidance

Only load a joblib file you created or obtained from a trusted, reviewed source.
Joblib/pickle-based formats can execute arbitrary Python objects while loading. A
future production workflow should use a controlled model registry, artifact integrity
checks, dependency pinning, and access controls.

The saved estimator does not by itself document the full data contract. Keep the
feature metadata, source version, seed, split, dependency versions, and preprocessing
assumptions with any published experiment.

## Recreate the artifacts

```bash
python main.py --days 365 --forecast-horizon 24 --seed 42
```

The result is a model for the deterministic virtual simulation. It is not trained on
real corporate data and should not be connected directly to operational controls.
