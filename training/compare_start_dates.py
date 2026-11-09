"""Score the shipped V5 formula on held-out games after each calendar cut.

The weights are the same year-by-year fits as the main forward test. Only the
games that count toward the score change.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from train_linear_model import (
    DATA_DIR,
    SPECS,
    TEST_SEASONS,
    build_rows,
    fit_predict,
)

COLUMNS = SPECS["eight_home_schedule"]
CUTS = [(12, 1), (12, 15), (1, 1), (1, 15), (2, 1)]
OUT = Path(__file__).resolve().parents[1] / "results" / "start_date_comparison.json"


def cut_on(season, month, day):
    year = season if month <= 6 else season - 1
    return datetime(year, month, day)


def score_mask(frame):
    y = frame["Margin"].to_numpy()
    won = frame["A_Won"].to_numpy()
    pred = frame["Pred"].to_numpy()
    regular = frame["IsTourney"].eq(0)
    out = {
        "N": int(len(frame)),
        "WinnerAccuracy": float(accuracy_score(won, pred > 0)) if len(frame) else None,
        "MarginMAE": float(mean_absolute_error(y, pred)) if len(frame) else None,
    }
    reg = frame[regular]
    out["RegularN"] = int(len(reg))
    if len(reg):
        out["RegularWinnerAccuracy"] = float(accuracy_score(reg["A_Won"], reg["Pred"] > 0))
        out["RegularMarginMAE"] = float(mean_absolute_error(reg["Margin"], reg["Pred"]))
    else:
        out["RegularWinnerAccuracy"] = None
        out["RegularMarginMAE"] = None
    return out


def main():
    seasons = pd.read_csv(DATA_DIR / "MSeasons.csv")
    day_zero = {
        int(row.Season): datetime.strptime(row.DayZero, "%m/%d/%Y")
        for row in seasons.itertuples(index=False)
    }
    print("Building pre-game rows...", flush=True)
    frame = build_rows()
    frame["Date"] = [
        day_zero[int(season)] + timedelta(days=int(day))
        for season, day in zip(frame["Season"], frame["DayNum"])
    ]
    held = []
    for season in TEST_SEASONS:
        train = frame[frame["Season"] < season]
        test = frame[frame["Season"] == season].copy()
        _, pred = fit_predict(train, test, COLUMNS)
        test["Pred"] = pred
        held.append(test)
        print(f"scored {season}: {len(test)} games", flush=True)
    held = pd.concat(held, ignore_index=True)
    payload = {"all_games": score_mask(held), "cuts": {}}
    print(f"\nAll held-out games: {payload['all_games']}", flush=True)
    for month, day in CUTS:
        label = f"{month:02d}-{day:02d}"
        mask = [
            date >= cut_on(int(season), month, day)
            for date, season in zip(held["Date"], held["Season"])
        ]
        later = held[mask]
        result = score_mask(later)
        payload["cuts"][label] = result
        print(
            f"on or after {label}: n={result['N']} "
            f"acc={result['WinnerAccuracy']:.3f} mae={result['MarginMAE']:.2f} "
            f"regular n={result['RegularN']} acc={result['RegularWinnerAccuracy']:.3f} "
            f"mae={result['RegularMarginMAE']:.2f}",
            flush=True,
        )
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
