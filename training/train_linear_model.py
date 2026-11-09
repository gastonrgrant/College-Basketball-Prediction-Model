"""Fit the v5 margin on Kaggle game logs.

The live report can see season rates, opponent rates, strength of schedule,
and where the game is played. These features are built the same way here,
from information available before each game.

A matchup feature is one team's offense against the other's defense, minus
the reverse. Schedule is the gap in strength of schedule, divided by how
spread out schedules are in that season. Home court is +1, 0, or -1.
"""

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# Folder with the Kaggle March Machine Learning Mania 2026 CSVs.
DATA_DIR = Path(os.environ.get(
    "KAGGLE_DATA_DIR",
    "/Users/gasto/Downloads/march-machine-learning-mania-2026",
))
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)

MIN_PRIOR_GAMES = 5
FT_COEF = 0.44
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]


def safe_div(num, den):
    return np.nan if den == 0 else num / den


def empty_state():
    return {
        "GP": 0.0, "PF": 0.0, "PA": 0.0, "FGM": 0.0, "FGA": 0.0, "FG3M": 0.0,
        "FTM": 0.0, "FTA": 0.0, "ORB": 0.0, "DRB": 0.0, "TOV": 0.0,
        "Opp_FGM": 0.0, "Opp_FGA": 0.0, "Opp_FG3M": 0.0, "Opp_FTA": 0.0,
        "Opp_ORB": 0.0, "Opp_DRB": 0.0, "Opp_TOV": 0.0, "Margin": 0.0,
        "SOS_Sum": 0.0, "SOS_N": 0.0,
    }


def summarize(state, min_games=MIN_PRIOR_GAMES):
    if state["GP"] < min_games:
        return None
    efg = safe_div(state["FGM"] + 0.5 * state["FG3M"], state["FGA"])
    opp_efg = safe_div(state["Opp_FGM"] + 0.5 * state["Opp_FG3M"], state["Opp_FGA"])
    tov = safe_div(state["TOV"], state["FGA"] + FT_COEF * state["FTA"] + state["TOV"])
    opp_tov = safe_div(state["Opp_TOV"], state["Opp_FGA"] + FT_COEF * state["Opp_FTA"] + state["Opp_TOV"])
    orb = safe_div(state["ORB"], state["ORB"] + state["Opp_DRB"])
    opp_orb = safe_div(state["Opp_ORB"], state["Opp_ORB"] + state["DRB"])
    ftr = safe_div(state["FTA"], state["FGA"])
    opp_ftr = safe_div(state["Opp_FTA"], state["Opp_FGA"])
    sos = safe_div(state["SOS_Sum"], state["SOS_N"]) if state["SOS_N"] else 0.0
    if any(np.isnan(v) for v in (efg, opp_efg, tov, opp_tov, orb, opp_orb, ftr, opp_ftr)):
        return None
    return {
        "eFG": efg, "Opp_eFG": opp_efg, "TOV": tov, "Opp_TOV": opp_tov,
        "ORB": orb, "Opp_ORB": opp_orb, "FTR": ftr, "Opp_FTR": opp_ftr,
        "SOS": 0.0 if np.isnan(sos) else sos,
        "Margin": safe_div(state["Margin"], state["GP"]),
        "GP": state["GP"],
    }


def add_game(state, offense, defense):
    state["GP"] += 1
    state["PF"] += offense["score"]
    state["PA"] += defense["score"]
    state["Margin"] += offense["score"] - defense["score"]
    for key in ("FGM", "FGA", "FG3M", "FTA", "ORB", "DRB", "TOV"):
        state[key] += offense[key]
        state[f"Opp_{key}"] += defense[key]


def pack(row, prefix):
    return {
        "score": float(getattr(row, f"{prefix}Score")),
        "FGM": float(getattr(row, f"{prefix}FGM")),
        "FGA": float(getattr(row, f"{prefix}FGA")),
        "FG3M": float(getattr(row, f"{prefix}FGM3")),
        "FTM": float(getattr(row, f"{prefix}FTM")),
        "FTA": float(getattr(row, f"{prefix}FTA")),
        "ORB": float(getattr(row, f"{prefix}OR")),
        "DRB": float(getattr(row, f"{prefix}DR")),
        "TOV": float(getattr(row, f"{prefix}TO")),
    }


def location_for_a(row, a_is_winner):
    loc = row.WLoc
    if loc == "N":
        return 0.0
    winner_home = 1.0 if loc == "H" else -1.0
    return winner_home if a_is_winner else -winner_home


def build_rows(min_games=MIN_PRIOR_GAMES, return_final=False):
    """Pre-game rows for games where both teams have min_games played.

    With return_final, also return each team's end-of-season profile keyed by
    (season, team id).
    """
    reg = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    tour = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    reg["IsTourney"] = 0
    tour["IsTourney"] = 1
    games = pd.concat([reg, tour], ignore_index=True)
    games = games.sort_values(["Season", "DayNum", "IsTourney"]).reset_index(drop=True)
    states = {}
    rows = []
    for game in games.itertuples(index=False):
        season = int(game.Season)
        winner = int(game.WTeamID)
        loser = int(game.LTeamID)
        team_a, team_b = (winner, loser) if winner < loser else (loser, winner)
        a_won = int(team_a == winner)
        state_a = states.setdefault((season, team_a), empty_state())
        state_b = states.setdefault((season, team_b), empty_state())
        profile_a = summarize(state_a, min_games)
        profile_b = summarize(state_b, min_games)
        if profile_a and profile_b:
            margin = (float(game.WScore) - float(game.LScore)) if a_won else (float(game.LScore) - float(game.WScore))
            rows.append({
                "Season": season,
                "A_ID": team_a,
                "B_ID": team_b,
                "DayNum": int(game.DayNum),
                "IsTourney": int(game.IsTourney),
                "A_Won": a_won,
                "Margin": margin,
                "Location": location_for_a(game, a_won),
                **{f"A_{k}": v for k, v in profile_a.items()},
                **{f"B_{k}": v for k, v in profile_b.items()},
            })
        winner_profile = summarize(states[(season, winner)])
        loser_profile = summarize(states[(season, loser)])
        # Strength of schedule uses the opponent's current scoring margin.
        if loser_profile:
            states[(season, winner)]["SOS_Sum"] += loser_profile["Margin"]
            states[(season, winner)]["SOS_N"] += 1
        elif states[(season, loser)]["GP"] > 0:
            states[(season, winner)]["SOS_Sum"] += states[(season, loser)]["Margin"] / states[(season, loser)]["GP"]
            states[(season, winner)]["SOS_N"] += 1
        if winner_profile:
            states[(season, loser)]["SOS_Sum"] += winner_profile["Margin"]
            states[(season, loser)]["SOS_N"] += 1
        elif states[(season, winner)]["GP"] > 0:
            states[(season, loser)]["SOS_Sum"] += states[(season, winner)]["Margin"] / states[(season, winner)]["GP"]
            states[(season, loser)]["SOS_N"] += 1
        add_game(states[(season, winner)], pack(game, "W"), pack(game, "L"))
        add_game(states[(season, loser)], pack(game, "L"), pack(game, "W"))
        if len(rows) and len(rows) % 25000 == 0:
            print(f"  {len(rows)} usable games so far", flush=True)
    frame = add_features(pd.DataFrame(rows))
    print(
        f"Built {len(frame)} games. SOS std median {frame.groupby('Season')['SOS_Std'].first().median():.2f}. "
        f"A_SOS range {frame['A_SOS'].quantile(0.05):.1f} to {frame['A_SOS'].quantile(0.95):.1f}."
    )
    if return_final:
        final = {}
        for key, state in states.items():
            profile = summarize(state, 1)
            if profile:
                final[key] = profile
        return frame, final
    return frame


def add_features(frame):
    """Matchup features from the A_ and B_ team rate columns."""
    sos_std = {}
    for season, group in frame.groupby("Season"):
        pooled = np.concatenate([group["A_SOS"].to_numpy(), group["B_SOS"].to_numpy()])
        sos_std[int(season)] = float(np.std(pooled)) or 1.0
    frame["SOS_Std"] = frame["Season"].map(sos_std)
    frame["SOS_ZDiff"] = (frame["A_SOS"] - frame["B_SOS"]) / frame["SOS_Std"]
    frame["eFG_Match"] = (frame["A_eFG"] - frame["B_Opp_eFG"]) - (frame["B_eFG"] - frame["A_Opp_eFG"])
    frame["TOV_Match"] = (frame["B_Opp_TOV"] - frame["A_TOV"]) - (frame["A_Opp_TOV"] - frame["B_TOV"])
    frame["ORB_Match"] = (frame["A_ORB"] - frame["B_Opp_ORB"]) - (frame["B_ORB"] - frame["A_Opp_ORB"])
    frame["FTR_Match"] = (frame["A_FTR"] - frame["B_Opp_FTR"]) - (frame["B_FTR"] - frame["A_Opp_FTR"])
    frame["eFG_Diff"] = frame["A_eFG"] - frame["B_eFG"]
    frame["Def_eFG"] = frame["B_Opp_eFG"] - frame["A_Opp_eFG"]
    frame["TOV_Diff"] = frame["B_TOV"] - frame["A_TOV"]
    frame["Def_TOV"] = frame["A_Opp_TOV"] - frame["B_Opp_TOV"]
    frame["ORB_Diff"] = frame["A_ORB"] - frame["B_ORB"]
    frame["Def_ORB"] = frame["B_Opp_ORB"] - frame["A_Opp_ORB"]
    frame["FTR_Diff"] = frame["A_FTR"] - frame["B_FTR"]
    frame["Def_FTR"] = frame["B_Opp_FTR"] - frame["A_Opp_FTR"]
    return frame


SPECS = {
    "current_eight_plus_home": [
        "eFG_Diff", "Def_eFG", "TOV_Diff", "Def_TOV", "ORB_Diff", "Def_ORB",
        "FTR_Diff", "Def_FTR", "Location",
    ],
    "matchup_plus_home": ["eFG_Match", "TOV_Match", "ORB_Match", "FTR_Match", "Location"],
    "matchup_home_schedule": [
        "eFG_Match", "TOV_Match", "ORB_Match", "FTR_Match", "SOS_ZDiff", "Location",
    ],
    "eight_home_schedule": [
        "eFG_Diff", "Def_eFG", "TOV_Diff", "Def_TOV", "ORB_Diff", "Def_ORB",
        "FTR_Diff", "Def_FTR", "SOS_ZDiff", "Location",
    ],
}


def fit_predict(train, test, columns):
    model = LinearRegression(fit_intercept=False)
    model.fit(train[columns].to_numpy(), train["Margin"].to_numpy())
    pred = model.predict(test[columns].to_numpy())
    return model, pred


def score(frame, pred):
    y = frame["Margin"].to_numpy()
    won = frame["A_Won"].to_numpy()
    tourney = frame["IsTourney"].eq(1).to_numpy()
    out = {
        "N": int(len(frame)),
        "WinnerAccuracy": float(accuracy_score(won, pred > 0)),
        "MarginMAE": float(mean_absolute_error(y, pred)),
    }
    if tourney.any():
        out["TourneyN"] = int(tourney.sum())
        out["TourneyWinnerAccuracy"] = float(accuracy_score(won[tourney], pred[tourney] > 0))
        out["TourneyMarginMAE"] = float(mean_absolute_error(y[tourney], pred[tourney]))
    else:
        out["TourneyN"] = 0
    return out


def forward_test(frame):
    results = {name: [] for name in SPECS}
    oof = {name: [] for name in SPECS}
    for season in TEST_SEASONS:
        train = frame[frame["Season"] < season]
        test = frame[frame["Season"] == season]
        if train.empty or test.empty:
            continue
        for name, columns in SPECS.items():
            model, pred = fit_predict(train, test, columns)
            row = {"Season": season, **score(test, pred)}
            results[name].append(row)
            kept = test[["Margin", "A_Won"]].copy()
            kept["Pred"] = pred
            oof[name].append(kept)
            print(f"{name} {season}: acc {row['WinnerAccuracy']:.3f} mae {row['MarginMAE']:.2f}", flush=True)
    return results, oof


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
    print("Forward testing...", flush=True)
    results, oof = forward_test(frame)
    summary = {name: pooled(rows) for name, rows in results.items()}
    print(json.dumps(summary, indent=2))
    # The schedule model is the one the live page can actually compute.
    ranked = sorted(summary, key=lambda name: (summary[name]["MarginMAE"], -summary[name]["WinnerAccuracy"]))
    chosen = ranked[0]
    print("Chosen by holdout margin error:", chosen)
    production = frame[frame["Season"] < 2026]
    columns = SPECS[chosen]
    model, _ = fit_predict(production, production.iloc[:1], columns)
    held = pd.concat(oof[chosen], ignore_index=True)
    # Wins are the positive side of the margin. Fit the probability slope on held-out games.
    logit = LogisticRegression(fit_intercept=False, max_iter=500)
    logit.fit(held[["Pred"]].to_numpy(), held["A_Won"].to_numpy())
    payload = {
        "chosen": chosen,
        "trained_on": "seasons before 2026",
        "coefficients": {col: float(coef) for col, coef in zip(columns, model.coef_)},
        "win_prob_slope": float(logit.coef_[0, 0]),
        "forward_summary": summary,
        "forward_seasons": results,
    }
    path = OUT_DIR / "linear_model.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
