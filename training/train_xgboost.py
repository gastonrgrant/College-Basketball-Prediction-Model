"""Fit an XGBoost margin on the same features and holdout as V5.

Tree shape is chosen once on seasons before 2022. Each test year then picks
its own tree count by early stopping on the previous season, and refits on
every season before the test year. Every feature is constrained to increase
the margin. Predictions average both team orders so the margin flips when the
teams are swapped.
"""

from itertools import product
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from train_linear_model import (
    OUT_DIR,
    SPECS,
    TEST_SEASONS,
    build_rows,
    score,
)

COLUMNS = SPECS["eight_home_schedule"]
V5_WINNER = 0.7040939713249266
V5_MAE = 9.088151556647256
V5_TOURNEY_WINNER = 0.6567164179104478
V5_TOURNEY_MAE = 10.687619617788254
SUMMARY_PATH = OUT_DIR / "xgboost_comparison.json"

GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [25, 100],
    "reg_lambda": [1.0, 10.0],
}


def matrix(frame):
    return frame[COLUMNS].to_numpy(dtype=np.float32)


def augment(frame):
    flipped = frame.copy()
    flipped[COLUMNS] = -flipped[COLUMNS]
    flipped["Margin"] = -flipped["Margin"]
    return pd.concat([frame, flipped], ignore_index=True)


def symmetric_predict(model, frame):
    values = matrix(frame)
    return 0.5 * (model.predict(values) - model.predict(-values))


def make_model(settings, n_estimators, early_stopping_rounds=None):
    params = dict(settings)
    params.update(
        n_estimators=int(n_estimators),
        objective="reg:squarederror",
        tree_method="hist",
        subsample=0.8,
        random_state=0,
        n_jobs=4,
        monotone_constraints=tuple(1 for _ in COLUMNS),
    )
    if early_stopping_rounds:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**params)


def fit_model(train, settings, n_estimators, eval_frame=None):
    model = make_model(
        settings,
        n_estimators,
        early_stopping_rounds=40 if eval_frame is not None else None,
    )
    fit_args = {}
    if eval_frame is not None:
        fit_args["eval_set"] = [(matrix(eval_frame), eval_frame["Margin"].to_numpy())]
        fit_args["verbose"] = False
    model.fit(matrix(augment(train)), augment(train)["Margin"].to_numpy(), **fit_args)
    return model


def choose_settings(frame):
    train = frame[frame["Season"] < 2021]
    valid = frame[frame["Season"] == 2021]
    trials = []
    keys = list(GRID)
    for values in product(*[GRID[key] for key in keys]):
        settings = dict(zip(keys, values))
        model = fit_model(train, settings, n_estimators=800, eval_frame=valid)
        trees = int(model.best_iteration) + 1
        pred = symmetric_predict(model, valid)
        result = score(valid, pred)
        trial = {
            **settings,
            "n_estimators": trees,
            "valid_2021": result,
        }
        trials.append(trial)
        print(
            f"tune {settings} trees={trees} "
            f"acc={result['WinnerAccuracy']:.3f} mae={result['MarginMAE']:.2f}",
            flush=True,
        )
    chosen = min(
        trials,
        key=lambda trial: (
            trial["valid_2021"]["MarginMAE"],
            -trial["valid_2021"]["WinnerAccuracy"],
        ),
    )
    frozen = {key: chosen[key] for key in GRID}
    print(f"Frozen shape: {frozen}", flush=True)
    return frozen, trials


def forward_test(frame, tree_settings):
    seasons = []
    for season in TEST_SEASONS:
        prior = season - 1
        stopped = fit_model(
            frame[frame["Season"] < prior],
            tree_settings,
            n_estimators=800,
            eval_frame=frame[frame["Season"] == prior],
        )
        trees = int(stopped.best_iteration) + 1
        train = frame[frame["Season"] < season]
        test = frame[frame["Season"] == season]
        model = fit_model(train, tree_settings, trees)
        pred = symmetric_predict(model, test)
        row = {"Season": int(season), "n_estimators": trees, **score(test, pred)}
        seasons.append(row)
        print(
            f"{season}: trees {trees} acc {row['WinnerAccuracy']:.3f} mae {row['MarginMAE']:.2f}",
            flush=True,
        )
    return seasons


def pooled(rows):
    games = sum(row["N"] for row in rows)
    correct = sum(row["WinnerAccuracy"] * row["N"] for row in rows)
    mae = sum(row["MarginMAE"] * row["N"] for row in rows)
    tourney_games = sum(row["TourneyN"] for row in rows)
    tourney_correct = sum(row.get("TourneyWinnerAccuracy", 0) * row["TourneyN"] for row in rows)
    tourney_mae = sum(row.get("TourneyMarginMAE", 0) * row["TourneyN"] for row in rows)
    return {
        "N": games,
        "WinnerAccuracy": correct / games,
        "MarginMAE": mae / games,
        "TourneyN": tourney_games,
        "TourneyWinnerAccuracy": tourney_correct / tourney_games if tourney_games else None,
        "TourneyMarginMAE": tourney_mae / tourney_games if tourney_games else None,
    }


def main():
    print("Building pre-game rows...", flush=True)
    frame = build_rows()
    print("Choosing tree settings on seasons before 2022...", flush=True)
    frozen, trials = choose_settings(frame)
    print("Forward testing...", flush=True)
    seasons = forward_test(frame, frozen)
    summary = pooled(seasons)
    adopted = (
        summary["MarginMAE"] < V5_MAE
        and summary["WinnerAccuracy"] >= V5_WINNER
    )
    payload = {
        "model": "xgboost",
        "features": COLUMNS,
        "settings_chosen_on": "tree shape on 2021; tree count early-stopped on the season before each test year",
        "monotone_constraints": "increasing on every feature",
        "frozen_settings": frozen,
        "tuning_trials": trials,
        "forward_seasons": seasons,
        "forward_summary": summary,
        "v5_forward_summary": {
            "WinnerAccuracy": V5_WINNER,
            "MarginMAE": V5_MAE,
            "TourneyWinnerAccuracy": V5_TOURNEY_WINNER,
            "TourneyMarginMAE": V5_TOURNEY_MAE,
        },
        "adopted": adopted,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Adopted: {adopted}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
