"""Test blending each team's stats so far with its previous season.

blended = w * this season + (1 - w) * last season, w = games / (games + K).

K = 0 uses this season only. K = infinity uses last season only. Weights are
the V5 fits on earlier seasons. K is chosen on 2016-2021 and reported on
2022-2026. Games are scored from each team's second game on.
"""

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

from train_linear_model import OUT_DIR, SPECS, TEST_SEASONS, add_features, build_rows

COLUMNS = SPECS["eight_home_schedule"]
STATS = ["eFG", "Opp_eFG", "TOV", "Opp_TOV", "ORB", "Opp_ORB", "FTR", "Opp_FTR", "SOS"]
TUNE_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
K_GRID = [0, 1, 2, 3, 5, 8, 12, 20, 40, math.inf]
BUCKETS = [("1-4 games", 1, 4), ("5-9 games", 5, 9), ("10+ games", 10, 10_000)]
OUT = OUT_DIR / "season_blend.json"


def attach_prior(frame, final):
    for side in ("A", "B"):
        keys = list(zip(frame["Season"] - 1, frame[f"{side}_ID"]))
        frame[f"{side}_HasPrior"] = [key in final for key in keys]
        for stat in STATS:
            frame[f"{side}_Prior_{stat}"] = [
                final[key][stat] if key in final else np.nan for key in keys
            ]
    return frame


def blend(frame, k):
    out = frame.copy()
    for side in ("A", "B"):
        games = out[f"{side}_GP"].astype(float)
        if math.isinf(k):
            weight = pd.Series(0.0, index=out.index)
        else:
            weight = games / (games + k) if k > 0 else pd.Series(1.0, index=out.index)
        weight = weight.where(out[f"{side}_HasPrior"], 1.0)
        for stat in STATS:
            prior = out[f"{side}_Prior_{stat}"].fillna(out[f"{side}_{stat}"])
            out[f"{side}_{stat}"] = weight * out[f"{side}_{stat}"] + (1 - weight) * prior
    return add_features(out)


def fit_weights(standard, season):
    train = standard[standard["Season"] < season]
    return LinearRegression(fit_intercept=False).fit(train[COLUMNS].to_numpy(), train["Margin"].to_numpy())


def grade(frame):
    if frame.empty:
        return {"N": 0}
    pred = frame["Pred"].to_numpy()
    return {
        "N": int(len(frame)),
        "WinnerAccuracy": float(accuracy_score(frame["A_Won"], pred > 0)),
        "MarginMAE": float(mean_absolute_error(frame["Margin"], pred)),
    }


def score_seasons(rows, standard, seasons, k, models):
    blended = blend(rows[rows["Season"].isin(seasons)], k)
    parts = []
    for season in seasons:
        test = blended[blended["Season"] == season].copy()
        test["Pred"] = models[season].predict(test[COLUMNS].to_numpy())
        parts.append(test)
    held = pd.concat(parts, ignore_index=True)
    fewer = held[["A_GP", "B_GP"]].min(axis=1)
    out = {"all": grade(held)}
    for label, low, high in BUCKETS:
        out[label] = grade(held[(fewer >= low) & (fewer <= high)])
    return out


def label(k):
    return "last season only" if math.isinf(k) else ("this season only" if k == 0 else f"K={k}")


def main():
    print("Building the V5 rows used for weights...", flush=True)
    standard = build_rows()
    print("Building rows from each team's second game...", flush=True)
    rows, final = build_rows(min_games=1, return_final=True)
    rows = attach_prior(rows, final)
    seasons = TUNE_SEASONS + TEST_SEASONS
    models = {season: fit_weights(standard, season) for season in seasons}

    tuning = {}
    for k in K_GRID:
        tuning[label(k)] = score_seasons(rows, standard, TUNE_SEASONS, k, models)
        result = tuning[label(k)]["all"]
        print(f"tune {label(k)}: acc {result['WinnerAccuracy']:.3f} mae {result['MarginMAE']:.3f}", flush=True)
    chosen = min(K_GRID, key=lambda k: tuning[label(k)]["all"]["MarginMAE"])
    print(f"Chosen K: {label(chosen)}", flush=True)

    testing = {}
    for k in sorted({0, chosen, math.inf}):
        testing[label(k)] = score_seasons(rows, standard, TEST_SEASONS, k, models)
        print(f"\n2022-2026 with {label(k)}", flush=True)
        for bucket, result in testing[label(k)].items():
            print(
                f"  {bucket:10s} n={result['N']:6d} acc={result['WinnerAccuracy']:.3f} "
                f"mae={result['MarginMAE']:.2f}",
                flush=True,
            )

    payload = {
        "blend": "games / (games + K) on this season, the rest on last season's final stats",
        "k_chosen_on": TUNE_SEASONS,
        "k_grid": [label(k) for k in K_GRID],
        "chosen_k": None if math.isinf(chosen) else chosen,
        "tuning": tuning,
        "test_2022_2026": testing,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
