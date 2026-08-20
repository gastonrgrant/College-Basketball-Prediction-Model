# V3 Repo Additions

This folder contains the first historical-training version of the model.

- `training/build_and_train_v3.py`: builds leakage-free pregame features from Kaggle detailed results.
- `model/ridge_margin_model_v3_1_clean.joblib`: provisional clean-core Ridge margin model.
- `artifacts/v3_baseline_results.csv`: whole-season holdout benchmark results.
- `artifacts/v3_feature_alpha_comparison.csv`: feature-set/regularization comparison.

The raw Kaggle CSVs are intentionally NOT included here.
Normal prediction users should eventually only need the saved model and live-data code.
Retraining developers will provide the raw historical data separately.
