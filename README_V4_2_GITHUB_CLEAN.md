# March Madness Prediction Model V4.2

A college basketball matchup model built around Dean Oliver's Four Factors, with the actual point values of those factors learned from historical NCAA games instead of being assigned by hand.

The model is designed to answer one question:

> Given what two teams have done up to the current date, which team should win and by how much?

The model combines two types of information:

- **Historical NCAA data** teaches the model how basketball statistics translate into point margin.
- **Current Sports Reference data** describes the teams at the time the prediction is made.

In simple terms:

```text
Historical NCAA games
        ↓
Learn how much each statistical advantage is worth
        ↓
Current team statistics
        ↓
Compare Team A and Team B
        ↓
Projected point margin
        ↓
Win probability
```

---

## 1. Core idea

Every matchup starts at a projected margin of 0.

The model compares Team A and Team B using the offensive and defensive sides of Dean Oliver's Four Factors:

- Shooting efficiency
- Turnovers
- Offensive rebounding
- Free-throw pressure

Historical NCAA games determine how many projected points each advantage is worth.

A much smaller secondary correction then accounts for a few variables that historically explained errors left over after the Four Factors.

The final structure is:

```text
Projected Margin = Four Factor Margin + V3 Residual Correction
```

A positive margin favors Team A.

A negative margin favors Team B.

---

## 2. Historical training vs. live data

This is one of the most important parts of the model.

### Historical data teaches the model

The currently embedded coefficients were learned from approximately **102,607 pregame NCAA observations from the 2003-2026 seasons**.

For each historical observation, the model sees:

```text
What Team A looked like before the game
What Team B looked like before the game
What actually happened
```

The historical data is used to learn the mathematical relationship between team statistics and actual scoring margin.

The full historical dataset does not have to be loaded every time a prediction is made because the learned coefficients are embedded directly in the prediction program.

### Live data describes today's teams

When the model is run normally, it attempts to pull the current NCAA season from Sports Reference College Basketball.

Examples:

```text
January 2027 → 2027 season statistics
February 2027 → 2027 season statistics
January 2029 → 2029 season statistics
```

The overall idea is:

```text
Historical Learning + Current Team Data = Current Prediction
```

During July through December, the program first checks the upcoming season. If Sports Reference has not populated that season yet, it falls back to the most recent available season.

If the live scrape fails completely, the model falls back to local CSV files and prints a warning that those files may be stale.

---

## 3. Dean Oliver's Four Factors

The prediction core is built around four basketball concepts.

### Effective Field Goal Percentage

Effective field goal percentage gives extra credit to made three-pointers because a three is worth more than a two.

```text
eFG% = (FGM + 0.5 × 3PM) / FGA
```

Higher offensive eFG% is better.

Lower opponent eFG% allowed is better defensively.

### Turnover Rate

Turnover rate measures how often an offense loses a possession through a turnover.

Lower offensive TOV% is better.

On defense, a higher opponent TOV% means the defense is forcing more turnovers.

### Offensive Rebounding Rate

ORB% measures how often a team recovers available offensive rebounds.

Higher offensive ORB% is better because it creates additional possessions.

On defense, lower opponent ORB% is better because the defense is finishing possessions with defensive rebounds.

### Free-Throw Pressure

The current model uses FT/FGA.

Higher offensive FT/FGA is treated as stronger free-throw pressure.

Lower opponent FT/FGA allowed is better defensively.

---

## 4. There are eight Four Factor inputs, not four

Each Four Factor has an offensive and defensive side.

| Factor | Offensive measurement | Defensive measurement |
|---|---|---|
| Shooting | eFG% | Opponent eFG% allowed |
| Turnovers | TOV% | Opponent TOV% forced |
| Rebounding | ORB% | Opponent ORB% allowed |
| Free throws | FT/FGA | Opponent FT/FGA allowed |

This produces eight separate matchup variables.

The offense and defense sides are not forced into a 50/50 blend in the final prediction.

Each side has its own coefficient learned from historical NCAA results.

---

## 5. Team A orientation

Every matchup variable is constructed so that:

```text
Positive value = advantage for Team A
Negative value = advantage for Team B
```

This keeps the model symmetric.

If Duke vs. North Carolina produces:

```text
Duke +4.2
```

then reversing the matchup should produce:

```text
North Carolina -4.2
```

The order in which the teams are entered should not create a hidden advantage.

---

## 6. The eight matchup equations

Let Team A and Team B be the two teams being compared.

### Offensive shooting advantage

```text
eFG_Diff = Team A eFG% - Team B eFG%
```

### Defensive shooting advantage

Lower opponent eFG% is better, so the subtraction is reversed:

```text
Def_eFG_Adv = Team B Opp eFG% - Team A Opp eFG%
```

### Offensive turnover advantage

Lower offensive TOV% is better:

```text
TOV_Adv = Team B TOV% - Team A TOV%
```

### Defensive turnover creation

Higher opponent TOV% means the defense forces more turnovers:

```text
ForceTOV_Adv = Team A Opp TOV% - Team B Opp TOV%
```

### Offensive rebounding advantage

```text
ORB_Diff = Team A ORB% - Team B ORB%
```

### Defensive rebounding advantage

Lower opponent ORB% is better:

```text
DefORB_Adv = Team B Opp ORB% - Team A Opp ORB%
```

### Offensive free-throw advantage

```text
FTR_Diff = Team A FT/FGA - Team B FT/FGA
```

### Defensive free-throw advantage

Lower opponent FT/FGA allowed is better:

```text
DefFTR_Adv = Team B Opp FT/FGA - Team A Opp FT/FGA
```

---

## 7. Historically learned Four Factor coefficients

Historical NCAA results determine how much each difference contributes to projected point margin.

The coefficients currently embedded in V4.2 are:

| Component | Coefficient |
|---|---:|
| Offensive eFG advantage | 73.4620 |
| Defensive eFG advantage | 58.3245 |
| Offensive turnover advantage | 91.8850 |
| Defensive turnover creation | 73.4088 |
| Offensive rebounding advantage | 33.2480 |
| Defensive rebounding advantage | 30.1818 |
| Offensive FT/FGA advantage | 12.5770 |
| Defensive FT/FGA advantage | 12.1923 |

These values look large because the rate statistics are expressed as decimals.

For example:

```text
1 percentage point = 0.01
```

A one-percentage-point offensive eFG advantage contributes:

```text
0.01 × 73.4620 = 0.735 projected points
```

### Approximate value of a one-percentage-point advantage

| Advantage | Projected margin effect |
|---|---:|
| +1 percentage point offensive eFG | +0.735 points |
| +1 percentage point defensive eFG | +0.583 points |
| +1 percentage point offensive TOV advantage | +0.919 points |
| +1 percentage point turnover creation | +0.734 points |
| +1 percentage point offensive rebounding | +0.332 points |
| +1 percentage point defensive rebounding | +0.302 points |
| +1 percentage point offensive FT/FGA | +0.126 points |
| +1 percentage point defensive FT/FGA | +0.122 points |

---

## 8. Complete Four Factor margin equation

The main prediction core is:

```text
Four Factor Margin =

  73.4620 × eFG_Diff
+ 58.3245 × Def_eFG_Adv
+ 91.8850 × TOV_Adv
+ 73.4088 × ForceTOV_Adv
+ 33.2480 × ORB_Diff
+ 30.1818 × DefORB_Adv
+ 12.5770 × FTR_Diff
+ 12.1923 × DefFTR_Adv
```

This produces the model's Four Factor projected margin.

The model does **not** use a hand-written 40/25/20/15 equation for the final prediction.

Instead, historical NCAA results learned the direct point-margin relationships shown above.

---

## 9. Example: turnover advantage

Suppose:

```text
Team A TOV% = 12%
Team B TOV% = 16%
```

The offensive turnover advantage for Team A is:

```text
0.16 - 0.12 = 0.04
```

The contribution to projected margin is:

```text
0.04 × 91.8850 = 3.68 points
```

All else equal, that four-percentage-point ball-security advantage contributes approximately:

```text
Team A +3.68 points
```

This is one reason the model can identify lower-seeded teams that protect the ball extremely well.

---

## 10. The V3 residual correction

The Four Factors make the primary prediction.

The model then asks:

> What did the Four Factor model systematically miss in historical games?

For a historical game:

```text
Residual = Actual Margin - Four Factor Prediction
```

A secondary model was trained to explain only those leftover errors.

The current residual variables are:

- Net Rating difference
- Three-point attempt-rate difference
- Pace difference

This structure is intentional.

It prevents general team-strength statistics from competing with or overwhelming the Four Factors.

---

## 11. Net Rating correction

For each team:

```text
Net Rating = Offensive Rating - Opponent Offensive Rating
```

Then:

```text
NetRtg_Diff = Team A Net Rating - Team B Net Rating
```

The current coefficient is:

```text
0.0081545
```

Even a large 10-point Net Rating advantage contributes only:

```text
10 × 0.0081545 = 0.082 projected points
```

The coefficient is small because much of the information inside Net Rating is already represented by shooting, turnovers, rebounding, and free throws.

---

## 12. Three-point attempt-rate correction

The matchup feature is:

```text
P3Ar_Diff = Team A 3PAr - Team B 3PAr
```

The current coefficient is:

```text
4.91246
```

A ten-percentage-point difference contributes:

```text
0.10 × 4.91246 = 0.491 projected points
```

---

## 13. Pace correction

The model uses:

```text
Pace_Diff = Team A Pace - Team B Pace
```

The current coefficient is:

```text
-0.028766
```

For example, a five-possession pace difference contributes:

```text
5 × -0.028766 = -0.144 points
```

The effect is intentionally small.

---

## 14. Complete V3 residual equation

```text
V3 Residual Correction =

  0.0081545 × NetRtg_Diff
+ 4.91246 × P3Ar_Diff
- 0.028766 × Pace_Diff
```

---

## 15. Final projected margin

The final prediction is:

```text
Projected Margin = Four Factor Margin + V3 Residual Correction
```

Example:

```text
Four Factor margin:       Team A +4.6
V3 residual correction:   Team B +0.3
--------------------------------------
Final projected margin:   Team A +4.3
```

The team on the positive side of the final margin is the predicted winner.

---

## 16. Margin to win probability

After calculating projected margin, V4.2 converts it into win probability with a logistic equation calibrated from forward-held-out historical predictions.

The equation is:

```text
P(Team A wins) = 1 / (1 + e^(-0.1432050386 × Projected Margin))
```

Examples:

| Projected margin | Approx. win probability |
|---:|---:|
| 0 | 50.0% |
| +1 | 53.6% |
| +2 | 57.1% |
| +5 | 67.2% |
| +10 | 80.7% |
| +15 | 89.5% |
| +20 | 94.6% |

The probability calculation does not make a second prediction.

It only translates projected point margin into an estimated chance of winning.

---

## 17. How the coefficients are learned

At a simplified level, historical regression is trying to make:

```text
Predicted Margin
```

as close as possible to:

```text
Actual Margin
```

for all historical games.

The model predicts each historical margin as a weighted combination of the eight Four Factor matchup variables.

The training process chooses the coefficient values that minimize the total squared prediction error across the historical dataset.

In plain English:

> Find the mathematical weights that make predicted margins as close as possible to actual historical margins.

That is where coefficients such as 73.4620, 91.8850, and 33.2480 come from.

They are not manually selected basketball boosts.

---

## 18. Pregame construction

Historical training observations are built using team information available **before the game being predicted**.

Conceptually:

```text
Team statistics before Game X
        ↓
Predict Game X
        ↓
Compare prediction with Game X result
```

The result of Game X should not be included in the statistics used to predict Game X.

This is essential for meaningful training.

---

## 19. Validation design

Reported 2022-2026 holdouts use strict forward validation.

When predicting season T, the model is trained only on seasons before T.

Examples:

```text
Test 2022 → train only on seasons before 2022
Test 2023 → train only on seasons before 2023
Test 2024 → train only on seasons before 2024
Test 2025 → train only on seasons before 2025
Test 2026 → train only on seasons before 2026
```

This is designed to imitate real deployment.

The model should never use future seasons to claim that it predicted an earlier season.

The V3 residual layer is also trained from expanding-window, out-of-sample errors from the Four Factor model.

---

## 20. What currently affects the prediction

| Feature | Affects projected winner? |
|---|---|
| Offensive eFG% | Yes |
| Defensive eFG% | Yes |
| Offensive TOV% | Yes |
| Defensive turnover creation | Yes |
| Offensive ORB% | Yes |
| Defensive rebounding | Yes |
| Offensive FT/FGA | Yes |
| Defensive FT/FGA | Yes |
| Net Rating | Yes, small residual correction |
| 3P attempt rate | Yes, small residual correction |
| Pace difference | Yes, small residual correction |
| Elo | No |
| SRS | No |
| SOS | No |
| NCAA tournament seed | No |
| Coach rating | No |
| Star-player rating | No |
| Player height | No |
| Player PPG | No |
| Player names | No |

SRS, SOS, player information, team profiles, and physical-size information may appear in the reporting layer, but they do not currently change the projected winner.

Tournament seeds are used for upset labels after the prediction. They do not determine the projected margin.

---

## 21. Descriptive offense/defense blends

The code contains learned offense-share values:

```text
eFG offense share: 57.41%
TOV offense share: 53.41%
ORB offense share: 58.73%
FTR offense share: 48.64%
```

These values are used for descriptive matchup projections shown in the report.

They do **not** determine the final point margin.

The final prediction uses the eight separate historically learned coefficients directly.

---

## 22. Expected possessions

The program displays an Expected Possessions estimate based on the two teams' pace.

In the current V4.2 implementation, this value is descriptive.

It does not directly multiply the final projected margin.

Pace enters the prediction only through the small Pace_Diff residual term.

---

## 23. What the current model intentionally does not do

V4.2 does not currently add manual bonuses such as:

```text
+3 points because a team has a superstar
+2 points because a coach is considered elite
+4 points because a program has tournament experience
```

It also does not currently use Elo.

The goal is to avoid counting the same underlying team quality multiple times through correlated boosts.

New variables should be added only if they demonstrate additional predictive value under strict out-of-sample testing.

---

## 24. Current limitations

The model is still a work in progress.

### Player availability and injuries

Current team statistics are season-to-date averages.

If an important player becomes unavailable, the team's averages still contain games played with that player.

A future version should test a mathematically learned availability adjustment rather than an arbitrary star-player bonus.

### Home-court advantage

The current matchup prediction does not ask whether the game is at Team A, at Team B, or on a neutral court.

That is mostly acceptable for NCAA Tournament games, but a general regular-season model should eventually include a historically learned location effect.

### Frozen-date historical backtests

For a true historical tournament evaluation, team data must be frozen at the date the prediction would actually have been made.

For example:

```text
Selection Sunday 2026 backtest
→ use only information available on Selection Sunday 2026
```

Using end-of-season statistics would introduce future information.

### Periodic retraining

Live Sports Reference inputs automatically move forward with the current season.

The embedded historical coefficients should eventually be refreshed as additional completed seasons become available.

---

## 25. Future-use goal

The long-term objective is for the model to work at any future date.

Example:

```text
Run in January 2027
        ↓
Load 2027 team statistics available on that date
        ↓
Apply historically learned NCAA relationships
        ↓
Predict Team A vs Team B
```

As future seasons accumulate, the historical training dataset can also be updated and the coefficients retrained.

The intended system is:

```text
EVERY PREDICTION

Current Sports Reference data
        ↓
Apply trained model
        ↓
Current matchup prediction
```

and periodically:

```text
New completed NCAA games
        ↓
Add to historical training data
        ↓
Retrain
        ↓
Forward validate
        ↓
Deploy updated coefficients
```

---

## 26. The model in one paragraph

> The model was trained on more than 100,000 historical NCAA pregame observations. It compares two teams using Dean Oliver's Four Factors on both offense and defense: shooting efficiency, turnovers, offensive rebounding, and free-throw pressure. Historical NCAA results determine how many projected points each statistical advantage is worth. Those eight point contributions are added together. A much smaller secondary model adjusts for Net Rating, three-point attempt rate, and pace only where those variables historically explained errors left by the Four Factors. The resulting value is the projected point spread, and a historically calibrated logistic equation converts that margin into a win probability. Current-season Sports Reference statistics are plugged into those learned relationships whenever the model is run.

The shortest explanation is:

> Compare the teams in the things that historically win basketball games, let NCAA history determine how many points each advantage is worth, add the points together, and convert the projected margin into a win probability.

---

## Data sources

- **Historical NCAA data:** NCAA men's basketball historical game data obtained through Kaggle and used to construct the pregame training dataset.
- **Current-season data:** Sports Reference College Basketball team, opponent, advanced, and roster statistics.
- **Basketball framework:** Dean Oliver's Four Factors, described in *Basketball on Paper*.

The exact Kaggle dataset or competition URL should be added here so the historical training source is fully reproducible.

---

## Model version

This README describes:

```text
March Madness Prediction Model V4.2
Historically Trained Oliver Hybrid
Future-Ready Live Sports Reference Inputs
```

The prediction-time coefficients documented above match the coefficients embedded in the current V4.2 model.
