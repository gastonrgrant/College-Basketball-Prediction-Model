# March Madness Prediction Model V5

A college basketball matchup model built around Dean Oliver's Four Factors. Historical NCAA games decide how many points each advantage is worth. The live script is `predict_matchup.py`.

The model answers one question:

> Given what two teams have done so far, and where the game is played, which team should win and by how much?

```text
Historical NCAA games
        ↓
Learn how much each statistical advantage is worth
        ↓
Current team statistics, plus home, away, or neutral
        ↓
Projected point margin
        ↓
Win probability
```

## Project layout

```text
predict_matchup.py              Run a matchup. This is the V5 model.
requirements.txt                Python packages.
training/
  train_linear_model.py         Fit and test the V5 weights on Kaggle history.
  train_xgboost.py              Same test with XGBoost, for comparison.
  compare_start_dates.py        Score V5 on games after different dates in the season.
  test_season_blend.py          Choose how fast this season replaces last season.
results/
  linear_model.json             V5 weights and year-by-year test scores.
  xgboost_comparison.json       XGBoost test scores.
  start_date_comparison.json    Scores by start date.
  season_blend.json             Scores for each blend setting.
archive/                        Earlier versions (V2, V4.1, V4.2) and their results.
```

Matchup reports are written to `reports/`, which is not committed.

## Run a matchup

```text
pip install -r requirements.txt
python predict_matchup.py
```

The program asks for two teams, then where the game is played:

```text
Where is it played?
  1  Duke home
  2  Connecticut home
  3  Neutral site
Choice [3]:
```

Enter `1`, `2`, `3`, a team name, or press Enter for a neutral site. A home game names that school's arena and city on the report. A neutral game says "Neutral site" and does not invent a building.

From July through December the loader tries the upcoming season, then falls back to the latest season Sports-Reference has published. Leave `SEASON_OVERRIDE` as `None` unless you want to force a year.

Early in a season, each team's stats are blended with last season. See [Early season](#early-season).

If the live pages fail, the script falls back to local CSV files and says so.

## Early season

Two games of stats are mostly noise. Each stat the model uses is a mix of this season and last season:

```text
this-season weight = games played / (games played + 5)
stat = weight × this season + (1 − weight) × last season
```

| Games played | This season | Last season |
|---:|---:|---:|
| 0 | 0% | 100% |
| 2 | 29% | 71% |
| 5 | 50% | 50% |
| 10 | 67% | 33% |
| 20 | 80% | 20% |
| 35 | 88% | 12% |

A team with no previous Division I season uses this season only.

### Choosing 5

`training/test_season_blend.py` tried values from 0 (this season only) to never trusting this season. It picked the value on 2016–2021 and then scored 2022–2026, with each season's weights fit only on earlier seasons. Here, unlike the holdout below, games are scored from each team's first game.

| Setting | All games | Teams with 1–4 games | 5–9 games | 10 or more |
|---|---:|---:|---:|---:|
| This season only | 70.5% / 9.51 | 71.0% / 11.98 | 73.0% / 9.57 | 69.7% / 8.98 |
| **Blend, 5** | **71.0% / 9.17** | **73.6% / 9.98** | **73.1% / 9.43** | **69.9% / 8.94** |
| Last season only | 66.4% / 10.13 | | | |

Each cell is winners picked, then average margin miss. The pool is 27,020 games. The blend wins in every group. It helps most in the first four games, where the average miss falls by 2 points. Results are in `results/season_blend.json`.

### Roster changes and transfers

Last season's team stats describe last season's roster. For the two teams in a matchup, the script adjusts last season before blending:

1. It reads who has played minutes this season.
2. A returning player's line comes from last season's team page. A transfer's line comes from his own player page, at his old school.
3. Offense moves by how differently this season's players shot, protected the ball, rebounded, and drew fouls last season, weighted by this season's minutes.
4. Defense is the minutes-weighted average of the team defenses those players were on last season. A transfer brings his old team's defense.
5. A freshman, or anyone without a previous college season, counts as unchanged from his team's last season.

John Blackwell's 2025–26 season at Wisconsin counts toward Duke's starting point. Returning-minutes counts would treat him like a freshman. The report and the console say how much each team's numbers come from this season, and name its main transfers.

Kaggle has no player data, so this roster step is not in the test above. The blend itself is tested. The roster step only changes last season's side of the blend, so it fades as games are played: by 20 games it is a fifth of each stat.

These page reads make the first matchup slow, about a minute and a half, because Sports-Reference allows about 20 requests a minute. Finished-season pages are saved in `.cache/` and are not fetched again.

## What the margin is

Every matchup starts at 0. A positive margin favors Team A. Swapping the teams flips the sign.

```text
Projected margin =
    Four Factor margin
  + home court
  + strength of schedule
```

The result is capped at 35 points in either direction. The report then turns that margin into a win probability. Running a matchup does not train the model again.

Net rating, three-point attempt rate, and pace have coefficients of zero in V5. They do not move the margin. Pace is still shown as expected possessions.

## The eight Four Factor inputs

Each factor has an offensive side and a defensive side. The two sides keep separate weights.

| Factor | Offense | Defense |
|---|---|---|
| Shooting | Team eFG% | Opponent eFG% allowed |
| Turnovers | Team TOV% | Opponent TOV% forced |
| Rebounding | Team ORB% | Opponent ORB% allowed |
| Free throws | Team FT/FGA | Opponent FT/FGA allowed |

```text
eFG% = (FGM + 0.5 × 3PM) / FGA
TOV% = TOV / (FGA + 0.44 × FTA + TOV)
ORB% = ORB / (ORB + opponent DRB)
FT/FGA = FTA / FGA
```

Higher offensive eFG%, ORB%, and FT/FGA are better. Lower offensive TOV% is better. On defense the signs reverse: allowing less efficient shooting, fewer offensive rebounds, and fewer free throws is better, and forcing more turnovers is better.

Sports-Reference prints some of these as 0–100 and others as 0–1. The script converts them to fractions before applying the weights.

### Matchup equations

Positive means an advantage for Team A.

```text
eFG_Diff      = A eFG% − B eFG%
Def_eFG_Adv   = B opponent eFG% − A opponent eFG%
TOV_Adv       = B TOV% − A TOV%
ForceTOV_Adv  = A opponent TOV% − B opponent TOV%
ORB_Diff      = A ORB% − B ORB%
DefORB_Adv    = B opponent ORB% − A opponent ORB%
FTR_Diff      = A FT/FGA − B FT/FGA
DefFTR_Adv    = B opponent FT/FGA − A opponent FT/FGA
```

V5 does not collapse each factor into one offense-versus-defense number. That version was fit on the same games and picked fewer winners, so it was left out. The eight weights already compare one team's offense with the other's defense, and they let those two sides count differently. Shooting offense is worth more per unit than shooting defense. Turnovers follow the same pattern.

## Coefficients in the live script

These slopes were fit on pre-game observations from seasons before 2026, with no intercept. The script stores the full values. Rounded to four decimals:

| Component | Points per unit |
|---|---:|
| Offensive eFG | 75.2734 |
| Defensive eFG | 58.1182 |
| Offensive turnover advantage | 88.1139 |
| Defensive turnover creation | 78.5127 |
| Offensive rebounding | 32.5365 |
| Defensive rebounding | 32.7217 |
| Offensive FT/FGA | 7.7893 |
| Defensive FT/FGA | 9.2455 |
| Home court | 3.6737 |
| Schedule, per standard deviation | 1.8889 |

A one-percentage-point gap is 0.01 of a unit:

| Advantage | Points |
|---|---:|
| +1 point of offensive eFG | +0.753 |
| +1 point of defensive eFG | +0.581 |
| +1 point of offensive turnover advantage | +0.881 |
| +1 point of turnover creation | +0.785 |
| +1 point of offensive rebounding | +0.325 |
| +1 point of defensive rebounding | +0.327 |
| +1 point of offensive FT/FGA | +0.078 |
| +1 point of defensive FT/FGA | +0.092 |

A team that turns the ball over at 12% against a team at 16% gets `0.04 × 88.1139`, about **3.52** points, before the other seven terms.

Home court is `+3.674` when Team A is at home, `−3.674` when Team A is away, and `0` on a neutral floor.

Schedule is Sports-Reference strength of schedule, z-scored against the teams loaded for that run:

```text
schedule margin = 1.8889 × (A SOS − B SOS) / standard deviation of SOS
```

In training, strength of schedule was the average scoring margin of opponents faced so far. The live page uses Sports-Reference's SOS column. Both enter the model as a within-sample z-score, so they do not have to share a raw scale.

## Margin to win probability

```text
P(Team A wins) = 1 / (1 + e^(−0.1489804539 × projected margin))
```

| Projected margin | Win probability |
|---:|---:|
| 0 | 50.0% |
| +1 | 53.7% |
| +2 | 57.4% |
| +5 | 67.8% |
| +10 | 81.6% |
| +15 | 90.3% |
| +20 | 95.2% |

The probability does not make a second pick. It only translates the margin.

## Holdout

The comparison used 102,607 pre-game rows. A team needed five prior games before it could be scored. Each test season was predicted only from earlier seasons. The 2026 file in this Kaggle drop has regular-season games and no tournament games, so the tournament count is 268 games from 2022 through 2025.

| Model | Winners | Margin miss | Tournament winners | Tournament miss |
|---|---:|---:|---:|---:|
| Eight factors + home court | 69.7% | 9.25 | 60.8% | 11.46 |
| One number per factor + home court | 61.1% | 10.92 | 56.7% | 12.45 |
| One number per factor + home court + schedule | 61.0% | 10.85 | 64.9% | 11.72 |
| **Eight factors + home court + schedule (V5)** | **70.4%** | **9.09** | **65.7%** | **10.69** |

The pool is 23,156 games. V5's tournament line is 176 of 268.

Season by season for V5:

| Season | Games | Winners | Margin miss | Tournament winners |
|---:|---:|---:|---:|---:|
| 2022 | 4,419 | 70.4% | 8.85 | 40 of 67 |
| 2023 | 4,677 | 71.0% | 9.10 | 43 of 67 |
| 2024 | 4,691 | 70.3% | 9.19 | 46 of 67 |
| 2025 | 4,713 | 70.7% | 9.11 | 47 of 67 |
| 2026 | 4,656 | 69.6% | 9.19 | — |

70.4% is a solid ordinary result for a season-stats model. Most of those games are mismatches, and those easy games pull the number up. The tournament rate, 65.7%, is the harder score. The average miss is still about 9 points, so a game the model calls by 3 or 4 points remains a slight favorite.

The shipped weights are one fit on every season before 2026. The table above refits before each test season. Both use the same features.

Dean Oliver's 40/25/20/15 shares are a benchmark from *Basketball on Paper*, not the live formula. An earlier comparison of that benchmark is in `archive/v4_training_results.md`.

### Against simpler predictions

| Prediction | Winners | Margin miss |
|---|---:|---:|
| Always pick the home team | 62.1% of non-neutral games (V5: 70.8% on the same games) | — |
| Call every game a tie | — | 11.49 |
| Each team's scoring margin so far, plus home court | 69.9% | 9.22 |
| **V5** | **70.4%** | **9.09** |

### XGBoost

`training/train_xgboost.py` fits gradient-boosted trees on the same ten inputs and the same year-by-year test. Each game is trained in both team orders, and a larger gap in any stat is required to mean more points.

| Model | Winners | Margin miss | Tournament winners | Tournament miss |
|---|---:|---:|---:|---:|
| XGBoost | 70.3% | 9.12 | 180 of 268 | 10.71 |
| **Linear regression (V5)** | **70.4%** | **9.09** | **176 of 268** | **10.69** |

The trees lost on both overall numbers, so V5 stays linear. The same trainer with XGBoost's linear setting reproduces V5's 70.4% and 9.09 exactly, which rules out a pipeline difference.

### Start of season

`training/compare_start_dates.py` scores V5 only on games after a chosen date. From December 1 through February 1 the winner rate stays between 69.5% and 70.4% and the margin miss falls only to about 8.9 points. Dropping early-season games does not improve the model.

## What changes the pick

| Input | Changes the margin? |
|---|---|
| The eight Four Factor gaps | Yes |
| Home, away, or neutral | Yes, 3.674 points |
| Strength of schedule | Yes |
| Net rating, 3-point attempt rate, pace | No |
| AP rank, NCAA seed | No. Seeds and ranks label the report and can mark an upset. |
| Who is on the roster | Early in the season, through last season's side of the blend |
| Height, points per game | No. Shown on the report. |
| Arena and city | No. Shown when a home team is chosen. |

An upset label needs a real ranking gap: an unranked team over a ranked team, an AP gap of 10 or more, a seed gap of 3 or more, or an unseeded team over a seeded team. Two 1 seeds are not an upset.

## Limits

Season rates include every game already played. An injury, a rest day, or a recent slump is not a separate adjustment.

The roster adjustment is untested, and it only applies to the two teams in the matchup. Freshmen count as average for their team. A transfer's defense is his old team's defense.

The margin is capped at ±35. A hypothetical neutral game has no arena, because there is no specific building to look up. Home arenas come from Wikipedia and can miss a school.

The 2026 holdout slice is regular season only.

## Retrain

`training/train_linear_model.py` rebuilds the pre-game table from the Kaggle March Machine Learning Mania 2026 files and writes `results/linear_model.json`. The raw files are not in this repo. Point the script at them with an environment variable:

```text
KAGGLE_DATA_DIR=/path/to/march-machine-learning-mania-2026 python training/train_linear_model.py
```

That folder needs `MRegularSeasonDetailedResults.csv`, `MNCAATourneyDetailedResults.csv`, and `MSeasons.csv`.

Copy a new fit into `predict_matchup.py` only after the forward test beats the table above.

## Data

- **History:** Kaggle's March Machine Learning Mania 2026 competition. Regular-season and NCAA tournament box scores, seasons 2003–2026.
- **Live team stats:** Sports-Reference College Basketball.
- **Framework:** Dean Oliver's Four Factors, from *Basketball on Paper*.
- **Arenas:** Wikipedia infoboxes for the home team's men's basketball page.
