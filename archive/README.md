# Archive

Earlier versions of the model, kept for reference. The current model is `predict_matchup.py` in the top-level folder.

## Models

- `model_v2.py`: hand-weighted blend of efficiency, SRS, and Four Factors. Prints to the terminal.
- `model_v4_1.py`: first version with Four Factor weights learned from history. Prints to the terminal.
- `model_v4_2.py`: the live model before V5. It writes the HTML report but treats every game as neutral and has no strength-of-schedule term. `model_v4_2_explained.md` walks through its math.

## Training

- `training/build_and_train_v3.py`: builds leakage-free pre-game features from Kaggle detailed results and fits the V3 ridge model.
- `training/train_oliver_hybrid_v4.py`: the V4 comparison of Oliver's 40/25/20/15 shares against learned weights. `v4_training_results.md` summarizes it.

Both scripts expect the raw Kaggle CSVs and an intermediate `historical_training_data_v3.csv`, neither of which is committed. They are not needed to run V5.

## Results and saved models

- `results/`: V3 and V4 holdout tables and learned weights.
- `saved_models/`: the V3 ridge model and V4 hybrid as joblib files. V5 does not load them. Its weights are written directly in `predict_matchup.py`.
