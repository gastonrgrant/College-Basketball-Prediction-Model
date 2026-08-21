# Oliver Hybrid V4 Training Results

## Goal
Keep Dean Oliver's Four Factors as the core basketball theory while replacing arbitrary live-model weights with historically learned NCAA values.

## Validation Design
All reported 2022–2026 holdouts use strict forward validation: when predicting season T, the model is trained only on seasons prior to T. The V3 residual layer is trained only on expanding-window out-of-sample errors from the Four Factors model, preventing future-season information from leaking into earlier predictions.
For live predictions, the model automatically retrieves current-season team statistics from Sports Reference and uses data available up to the current date.

## Data Sources
- **Historical NCAA data:** Kaggle NCAA March Madness historical datasets, used to construct the 2003–2026 pregame training dataset and train the model's historical coefficients.
- **Current-season data:** Sports Reference College Basketball, used to retrieve up-to-date team, opponent, advanced, and roster statistics at prediction time.

## Models compared
1. **Oliver Original** — preserves 40% eFG, 25% turnovers, 20% offensive rebounding, 15% free-throw rate. Offense and defense are paired 50/50 inside each family. Historical data learns only point scaling and location.
2. **Oliver Learned** — NCAA history learns separate coefficients for offensive and defensive versions of all four factors.
3. **Oliver + V3 Residual** — starts with Oliver Learned and uses Net Rating difference, 3-point attempt-rate difference, mean pace, and pace difference only to predict the Oliver model's out-of-sample residual errors.

## Main tournament result (2022-2025, 268 NCAA Tournament games)

| Model | Winner accuracy | Margin MAE |
|---|---:|---:|
| Oliver Original | **63.43%** | 11.507 |
| Oliver Learned | 60.45% | 11.469 |
| Oliver + V3 Residual | 60.45% | **11.458** |

The unconstrained learned model and residual correction slightly improve margin error, but they reduce tournament winner accuracy. On the current evidence, the original Oliver factor ratios deserve to remain the tournament-prediction anchor rather than being replaced.

## Learned NCAA Four-Factor importance
When the eight offense/defense components are allowed to learn freely on all 102,607 historical pregame rows, their standardized coefficient magnitudes imply:

| Factor family | Oliver original | NCAA learned share |
|---|---:|---:|
| eFG | 40.0% | **38.55%** |
| Turnovers | 25.0% | **30.21%** |
| Offensive rebounding | 20.0% | **21.40%** |
| Free-throw rate | 15.0% | **9.84%** |

This is strikingly close to Oliver for eFG and rebounding, but historical NCAA margins put more weight on turnovers and less on free-throw rate.

## Learned offense/defense split inside each factor

| Factor | Offense share | Defense share |
|---|---:|---:|
| eFG | 57.41% | 42.59% |
| Turnovers | 53.41% | 46.59% |
| Rebounding | 58.73% | 41.27% |
| Free-throw rate | 48.64% | 51.36% |

This provides a mathematical replacement for the arbitrary 50/50 offense-defense blend if we decide to use it, while still keeping each Dean Oliver factor family explicit.

## V3 residual layer
Standardized residual coefficients:

- Net Rating difference: +0.118 points
- 3PA rate difference: +0.382 points
- Mean pace: +0.116 points
- Pace difference: -0.122 points

These are small relative to the Four-Factor coefficients. That supports using V3 information only as a correction layer rather than allowing it to dominate the model.

## Recommended live-model direction
Do **not** replace Oliver's 40/25/20/15 tournament core yet. The best next implementation is a theory-first live engine:

- preserve the original 40/25/20/15 Four-Factor family prior;
- replace the arbitrary 50/50 offense-defense blend only if additional tournament validation supports the learned splits;
- remove the current arbitrary 0.55 efficiency / 0.35 SRS / 0.10 Four-Factor final blend;
- calibrate the Four-Factor score directly into point margin from historical data;
- keep V3 variables available as diagnostics/correction candidates, but do not add their residual correction to winner selection until it demonstrates out-of-sample tournament improvement.

## Files
- `train_oliver_hybrid_v4.py` — reproducible training/backtest script (packaged one directory up in the ZIP).
- `oliver_model_forward_backtest.csv` — season-by-season strict forward results.
- `oliver_model_tournament_aggregate.csv` — aggregate tournament comparison.
- `learned_four_factor_family_weights.csv` — learned factor-family shares.
- `learned_four_factor_component_weights.csv` — learned offense/defense coefficients and splits.
- `learned_v3_residual_coefficients.csv` — correction-layer coefficients.
- `oliver_v3_hybrid_model_v4.joblib` — provisional trained artifact.
- `oliver_hybrid_summary.json` — machine-readable summary.
