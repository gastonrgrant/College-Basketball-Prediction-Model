# March Madness Prediction Model V4.2

A college basketball matchup model built around **Dean Oliver's Four Factors**, with the actual point values of those factors learned from historical NCAA games rather than assigned by hand.

The model is designed to answer a simple question:

> **Given what two teams have done up to the current date, which team should win and by how much?**

The model uses two different types of data:

1. **Historical NCAA data** teaches the model how basketball statistics translate into point margin.
2. **Current-season Sports Reference data** describes the two teams at the moment the prediction is made.

In simple terms:

```text
Historical NCAA games
        ↓
Learn what each statistical advantage is worth
        ↓
Current team statistics
        ↓
Compare Team A vs Team B
        ↓
Projected point margin
        ↓
Win probability
```

---

## 1. Core idea

Every matchup starts at a projected margin of **0**.

The model compares Team A and Team B in the offensive and defensive sides of Dean Oliver's Four Factors:

- Shooting efficiency
- Turnovers
- Offensive rebounding
- Free-throw pressure

Historical NCAA games determine how many projected points each advantage is worth.

The model then adds a much smaller secondary correction for a few variables that historically explained errors left over after the Four Factors.

The final equation is:

$$
\boxed{
\text{Projected Margin}
=
\text{Four Factor Margin}
+
\text{V3 Residual Correction}
}
$$

A positive margin favors Team A.  
A negative margin favors Team B.

---

# 2. Historical training vs. live data

This distinction is central to the model.

## Historical data teaches the model

The currently embedded coefficients were learned from approximately **102,607 pregame NCAA observations from the 2003-2026 seasons**.

For each historical observation, the model sees:

```text
What Team A looked like before the game
What Team B looked like before the game
What actually happened
```

The historical data is used to determine the mathematical relationship between team statistics and actual scoring margin.

It is **not** necessary to load the full historical dataset every time a prediction is made because the learned coefficients are embedded in the prediction program.

## Live data describes today's teams

When the model is run normally, it attempts to pull the current NCAA season from **Sports Reference College Basketball**.

For example:

```text
January 2027 → 2027 season statistics
February 2027 → 2027 season statistics
January 2029 → 2029 season statistics
```

The model therefore combines:

$$
\boxed{
\text{Historical Learning}
+
\text{Current Team Data}
=
\text{Current Prediction}
}
$$

During July-December, the program first checks the upcoming season. If Sports Reference has not populated that season yet, it falls back to the most recent available season.

If the live scrape fails completely, the program falls back to local CSV files and prints a warning that those files may be stale.

---

# 3. Dean Oliver's Four Factors

The prediction core is based on four basketball concepts.

## Effective field-goal percentage

Effective field-goal percentage gives extra credit to three-point field goals because a made three is worth more than a made two.

$$
eFG\% = \frac{FGM + 0.5(3PM)}{FGA}
$$

Higher offensive eFG% is better.

Lower opponent eFG% allowed is better defensively.

---

## Turnover rate

Turnover rate measures how often an offense loses a possession through a turnover.

Lower offensive TOV% is better.

On defense, a higher opponent TOV% means the defense is forcing more turnovers.

---

## Offensive rebounding rate

ORB% measures how often a team recovers available offensive rebounds.

Higher offensive ORB% is better because it creates additional possessions.

On defense, lower opponent ORB% is better because it means the defense is finishing possessions with defensive rebounds.

---

## Free-throw pressure

The current model uses **FT/FGA**.

Higher offensive FT/FGA is treated as stronger free-throw pressure.

Lower opponent FT/FGA allowed is better defensively.

---

# 4. There are eight Four Factor inputs, not four

Each Four Factor has an offensive and defensive side.

| Factor | Offensive measurement | Defensive measurement |
|---|---|---|
| Shooting | eFG% | Opponent eFG% allowed |
| Turnovers | TOV% | Opponent TOV% forced |
| Rebounding | ORB% | Opponent ORB% allowed |
| Free throws | FT/FGA | Opponent FT/FGA allowed |

This produces eight separate matchup variables.

The offense and defense sides are **not forced into a 50/50 blend** in the final prediction.

Each side has its own coefficient learned from historical NCAA results.

---

# 5. Team A orientation

Every matchup variable is constructed so:

```text
Positive value = advantage for Team A
Negative value = advantage for Team B
```

This is important because it makes the model symmetric.

If Duke vs. North Carolina produces:

```text
Duke +4.2
```

then reversing the teams should produce:

```text
North Carolina -4.2
```

The order in which the teams are entered should not create a hidden advantage.

---

# 6. The eight matchup equations

Let Team A and Team B be the two teams being compared.

## Offensive shooting advantage

$$
X_1=eFG_A-eFG_B
$$

## Defensive shooting advantage

Because lower opponent eFG% is better:

$$
X_2=OppEFG_B-OppEFG_A
$$

## Offensive turnover advantage

Because lower offensive TOV% is better:

$$
X_3=TOV_B-TOV_A
$$

## Defensive turnover creation

Because higher opponent TOV% means the defense forces more turnovers:

$$
X_4=OppTOV_A-OppTOV_B
$$

## Offensive rebounding advantage

$$
X_5=ORB_A-ORB_B
$$

## Defensive rebounding advantage

Because lower opponent ORB% is better:

$$
X_6=OppORB_B-OppORB_A
$$

## Offensive free-throw advantage

$$
X_7=FTR_A-FTR_B
$$

## Defensive free-throw advantage

Because lower opponent FT/FGA allowed is better:

$$
X_8=OppFTR_B-OppFTR_A
$$

---

# 7. Historically learned Four Factor coefficients

Historical NCAA results determine how much each of those differences contributes to projected point margin.

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

These values may look large because rate statistics are expressed as decimals.

For example:

$$
1\%=0.01
$$

Therefore a one-percentage-point offensive eFG advantage contributes approximately:

$$
0.01(73.462)=0.735
$$

projected points.

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

# 8. Complete Four Factor margin equation

The main prediction core is:

$$
\begin{aligned}
M_{FF}={}&
73.4620(eFG_A-eFG_B)\\
&+58.3245(OppEFG_B-OppEFG_A)\\
&+91.8850(TOV_B-TOV_A)\\
&+73.4088(OppTOV_A-OppTOV_B)\\
&+33.2480(ORB_A-ORB_B)\\
&+30.1818(OppORB_B-OppORB_A)\\
&+12.5770(FTR_A-FTR_B)\\
&+12.1923(OppFTR_B-OppFTR_A)
\end{aligned}
$$

This value is the model's **Four Factor projected margin**.

The model does **not** use a hand-written `40/25/20/15` scoring equation for the final pick.

Instead, historical NCAA results learned the direct point-margin relationships above.

---

# 9. Example: why turnover advantages can matter so much

Suppose:

```text
Team A TOV% = 12%
Team B TOV% = 16%
```

The offensive turnover advantage for Team A is:

$$
0.16-0.12=0.04
$$

The point contribution is:

$$
0.04(91.8850)=3.68
$$

So, all else equal, the four-percentage-point ball-security advantage contributes approximately:

**Team A +3.68 points**

This is one reason the model can identify dangerous lower-seeded teams that protect the ball extremely well.

---

# 10. The V3 residual correction

The Four Factors make the primary prediction.

The model then asks:

> **What did the Four Factor model systematically miss in historical games?**

A secondary model was trained on those leftover errors.

For a historical game:

$$
Residual
=
ActualMargin-FourFactorPrediction
$$

The V3 layer attempts to explain only that residual.

The current residual variables are:

- Net Rating difference
- Three-point attempt-rate difference
- Pace difference

This design is intentional.

Instead of allowing general team-strength statistics to compete with or overwhelm the Four Factors, they are used only as a small correction layer.

---

# 11. Net Rating correction

For each team:

$$
NetRating=ORtg-OppORtg
$$

Then:

$$
NetRtgDiff=NetRating_A-NetRating_B
$$

Current coefficient:

$$
0.0081545
$$

So even a large 10-point Net Rating advantage contributes only:

$$
10(0.0081545)=0.082
$$

projected points.

This coefficient is deliberately small because much of the information contained in Net Rating is already represented by shooting, turnovers, rebounding and free throws.

---

# 12. Three-point attempt-rate correction

Three-point attempt rate is represented by 3PAr.

The matchup feature is:

$$
3PArDiff=3PAr_A-3PAr_B
$$

Current coefficient:

$$
4.91246
$$

A ten-percentage-point difference in three-point attempt rate therefore contributes:

$$
0.10(4.91246)=0.491
$$

projected points.

---

# 13. Pace correction

The model uses:

$$
PaceDiff=Pace_A-Pace_B
$$

Current coefficient:

$$
-0.028766
$$

For example, a five-possession pace difference contributes only:

$$
5(-0.028766)=-0.144
$$

points.

The effect is intentionally small.

---

# 14. Complete V3 residual equation

$$
\begin{aligned}
M_{V3}={}&
0.0081545(NetRtgDiff)\\
&+4.91246(3PArDiff)\\
&-0.028766(PaceDiff)
\end{aligned}
$$

---

# 15. Final projected margin

The final margin is simply:

$$
\boxed{
M=M_{FF}+M_{V3}
}
$$

Example:

```text
Four Factor margin:       Team A +4.6
V3 residual correction:   Team B +0.3
--------------------------------------
Final projected margin:   Team A +4.3
```

The team on the positive side of the final margin is the predicted winner.

---

# 16. Margin to win probability

After calculating projected margin, V4.2 converts it to win probability with a logistic equation calibrated from forward-held-out historical predictions.

The current equation is:

$$
\boxed{
P(A)=
\frac{1}
{1+e^{-0.1432050386M}}
}
$$

where:

- $M$ = Team A projected margin
- $P(A)$ = probability Team A wins

Some examples:

| Projected margin | Approx. win probability |
|---:|---:|
| 0 | 50.0% |
| +1 | 53.6% |
| +2 | 57.1% |
| +5 | 67.2% |
| +10 | 80.7% |
| +15 | 89.5% |
| +20 | 94.6% |

The probability calculation does not make a separate prediction.

It simply translates the projected point margin into an estimated chance of winning.

---

# 17. How the coefficients are learned

At a simplified level, historical regression is solving:

$$
ActualMargin_i
\approx
\beta_1X_{1i}
+\beta_2X_{2i}
+\cdots
+\beta_8X_{8i}
$$

The model chooses the coefficients $\beta$ that minimize historical prediction error.

Using ordinary least squares, the objective is:

$$
\min_{\beta}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
$$

In plain English:

> **Find the mathematical weights that make predicted margins as close as possible to actual historical margins.**

That is where coefficients such as `73.4620`, `91.8850`, and `33.2480` come from.

They are not manually selected basketball "boosts."

---

# 18. Pregame construction prevents direct outcome leakage

Historical training observations are built from team information available **before the game being predicted**.

Conceptually:

```text
Team statistics before Game X
        ↓
Predict Game X
        ↓
Compare with Game X result
```

The result of Game X is not supposed to be included in the statistics used to predict Game X.

This is essential for meaningful historical training.

---

# 19. Validation design

Reported 2022-2026 holdouts use **strict forward validation**.

When predicting season $T$, the model is trained only on seasons before $T$.

For example:

```text
Test 2022 → train on seasons before 2022
Test 2023 → train on seasons before 2023
Test 2024 → train on seasons before 2024
Test 2025 → train on seasons before 2025
Test 2026 → train on seasons before 2026
```

This is designed to imitate real deployment:

> **The model should never use future seasons to claim that it predicted an earlier season.**

The V3 residual layer is also trained from expanding-window, out-of-sample errors from the Four Factor model.

---

# 20. What currently affects the prediction

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
| Elo | **No** |
| SRS | **No** |
| SOS | **No** |
| NCAA tournament seed | **No** |
| Coach rating | **No** |
| Star-player rating | **No** |
| Player height | **No** |
| Player PPG | **No** |
| Player names | **No** |

SRS, SOS, player information, team profiles and physical-size information may appear in the program's reporting layer, but they do not currently change the projected winner.

Tournament seeds are used for upset labels after the prediction; they do not determine the predicted margin.

---

# 21. Descriptive offense/defense blends are not the final prediction

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

# 22. Expected possessions

The program displays:

```text
Expected Possessions
```

using the average pace of the two teams.

In the current V4.2 implementation, this value is **descriptive** and does not directly multiply the final projected margin.

Pace enters the prediction only through the small `Pace_Diff` residual term.

---

# 23. What the current model intentionally does not do

V4.2 currently does not add manual bonuses such as:

```text
+3 points for a superstar
+2 points for an elite coach
+4 points for tournament experience
```

It also does not currently use Elo.

The goal is to prevent the same underlying team quality from being counted repeatedly through multiple correlated "boosts."

New variables should be added only if they demonstrate additional predictive value under strict out-of-sample testing.

---

# 24. Current limitations

The model is still a work in progress.

Important limitations include:

### Player availability and injuries

Current team statistics are season-to-date averages.

If an important player becomes unavailable, the team's historical season averages still contain games played with that player.

A future version should test a mathematically learned availability adjustment rather than an arbitrary "star-player bonus."

### Home-court advantage

The current matchup prediction does not ask whether a game is:

- at Team A
- at Team B
- neutral site

This is mostly acceptable for NCAA Tournament games, but a general regular-season model should eventually include a historically learned location effect.

### Frozen-date historical backtests

For true historical tournament evaluation, team data must be frozen at the date the prediction would actually have been made.

For example, a Selection Sunday 2026 backtest should use only information available on Selection Sunday 2026.

Using end-of-season statistics would introduce future information.

### Periodic retraining

Live Sports Reference inputs automatically move forward with the current season.

The embedded historical coefficients, however, should eventually be refreshed as additional completed seasons become available.

---

# 25. Future-use goal

The long-term objective is for the model to work at any future date.

For example:

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


PERIODICALLY
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

# 26. The model in one paragraph

The simplest complete explanation is:

> **The model was trained on more than 100,000 historical NCAA pregame observations. It compares two teams using Dean Oliver's Four Factors on both offense and defense: shooting efficiency, turnovers, offensive rebounding and free-throw pressure. Historical NCAA results determine how many projected points each statistical advantage is worth. Those eight point contributions are added together. A much smaller secondary model adjusts for Net Rating, three-point attempt rate and pace only where those variables historically explained errors left by the Four Factors. The resulting value is the projected point spread, and a historically calibrated logistic equation converts that margin into a win probability. Current-season Sports Reference statistics are plugged into those learned relationships whenever the model is run.**

The shortest possible explanation is:

> **Compare the teams in the things that historically win basketball games, let NCAA history determine how many points each advantage is worth, add the points together, and convert the projected margin into a win probability.**

---

# Data sources

- **Historical NCAA data:** NCAA men's basketball historical game data obtained through Kaggle and used to construct the pregame training dataset.
- **Current-season data:** Sports Reference College Basketball team, opponent, advanced and roster statistics.
- **Basketball framework:** Dean Oliver's Four Factors, described in *Basketball on Paper*.

> The exact Kaggle dataset/competition URL used for the historical data should be added here so the training source is fully reproducible.

---

# Model version

This document describes:

```text
March Madness Prediction Model V4.2
Historically Trained Oliver Hybrid
Future-Ready Live Sports Reference Inputs
```

The prediction-time coefficients documented above match the coefficients embedded in the current V4.2 model.
