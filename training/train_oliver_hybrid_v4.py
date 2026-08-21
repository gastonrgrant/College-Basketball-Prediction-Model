from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# ============================================================
# PATHS
# ============================================================
DATA_CSV = Path('historical_training_data_v3.csv')
OUT_DIR = Path('oliver_hybrid_outputs')
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# MODEL PHILOSOPHY
# ============================================================
# Dean Oliver's Four Factors remain the core basketball model:
#   eFG%, turnover rate, offensive rebounding, free-throw rate.
#
# Benchmark A: Oliver Original
#   - fixed 40/25/20/15 factor-family ratios
#   - 50/50 offense/defense within each family
#   - only global point calibration + location are learned
#
# Benchmark B: Oliver Learned
#   - same Four Factor families
#   - historical NCAA data learns offense/defense contribution and
#     each component's point-margin coefficient
#
# Benchmark C: Oliver + V3 Residual
#   - Oliver Learned makes the base prediction
#   - a second model learns ONLY the residual errors from genuinely
#     out-of-sample, expanding-window Oliver predictions
#   - correction features are non-Four-Factor V3 clean-core signals
#
# No future season is used to predict an earlier held-out season.
# ============================================================

TARGET = 'ActualMargin'
LOCATION = 'A_Location'

# Every feature is oriented so positive favors Team A.
FACTOR_COMPONENTS = [
    'eFG_Diff', 'Def_eFG_Adv',
    'TOV_Adv', 'ForceTOV_Adv',
    'ORB_Diff', 'DefORB_Adv',
    'FTR_Diff', 'DefFTR_Adv',
]

FACTOR_FAMILIES = {
    'eFG': ('eFG_Diff', 'Def_eFG_Adv'),
    'TOV': ('TOV_Adv', 'ForceTOV_Adv'),
    'ORB': ('ORB_Diff', 'DefORB_Adv'),
    'FTR': ('FTR_Diff', 'DefFTR_Adv'),
}

OLIVER_ORIGINAL_WEIGHTS = {
    'eFG': 0.40,
    'TOV': 0.25,
    'ORB': 0.20,
    'FTR': 0.15,
}

# Deliberately excludes Four Factor variables. This layer is allowed to
# explain only what the Dean Oliver core systematically misses.
V3_RESIDUAL_FEATURES = [
    'NetRtg_Diff',
    'P3Ar_Diff',
    'Pace_Mean',
    'Pace_Diff',
]

TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
MIN_OOF_TRAIN_SEASONS = 5

USECOLS = [
    'Season', 'DayNum', 'IsTourney', 'TeamAID', 'TeamBID',
    LOCATION, TARGET, 'A_Won',
    *FACTOR_COMPONENTS,
    *V3_RESIDUAL_FEATURES,
]


def safe_scale_fit(df: pd.DataFrame, cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[cols])
    return scaler


def transform_components(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    arr = scaler.transform(df[FACTOR_COMPONENTS])
    return pd.DataFrame(arr, columns=FACTOR_COMPONENTS, index=df.index)


def fit_oliver_original(train: pd.DataFrame):
    """Fixed 40/25/20/15 Oliver ratios; learn only point scale + location."""
    scaler = safe_scale_fit(train, FACTOR_COMPONENTS)
    z = transform_components(train, scaler)

    family = {}
    for fam, (off_col, def_col) in FACTOR_FAMILIES.items():
        # Original benchmark keeps a neutral 50/50 offense-defense blend.
        family[fam] = 0.5 * (z[off_col] + z[def_col])

    score = sum(OLIVER_ORIGINAL_WEIGHTS[f] * family[f] for f in FACTOR_FAMILIES)
    X = np.column_stack([train[LOCATION].to_numpy(), score.to_numpy()])
    model = LinearRegression().fit(X, train[TARGET])
    return {'component_scaler': scaler, 'model': model}


def predict_oliver_original(fitted, df: pd.DataFrame) -> np.ndarray:
    z = transform_components(df, fitted['component_scaler'])
    family = {}
    for fam, (off_col, def_col) in FACTOR_FAMILIES.items():
        family[fam] = 0.5 * (z[off_col] + z[def_col])
    score = sum(OLIVER_ORIGINAL_WEIGHTS[f] * family[f] for f in FACTOR_FAMILIES)
    X = np.column_stack([df[LOCATION].to_numpy(), score.to_numpy()])
    return fitted['model'].predict(X)


def fit_oliver_learned(train: pd.DataFrame):
    """Learn eight offense/defense Four-Factor coefficients from NCAA history."""
    scaler = safe_scale_fit(train, FACTOR_COMPONENTS)
    z = transform_components(train, scaler)
    X = np.column_stack([train[LOCATION].to_numpy(), z[FACTOR_COMPONENTS].to_numpy()])
    model = LinearRegression().fit(X, train[TARGET])
    return {'component_scaler': scaler, 'model': model}


def predict_oliver_learned(fitted, df: pd.DataFrame) -> np.ndarray:
    z = transform_components(df, fitted['component_scaler'])
    X = np.column_stack([df[LOCATION].to_numpy(), z[FACTOR_COMPONENTS].to_numpy()])
    return fitted['model'].predict(X)


def fit_residual_correction(residual_train: pd.DataFrame):
    scaler = safe_scale_fit(residual_train, V3_RESIDUAL_FEATURES)
    X = scaler.transform(residual_train[V3_RESIDUAL_FEATURES])
    model = LinearRegression().fit(X, residual_train['OliverResidual'])
    return {'scaler': scaler, 'model': model}


def predict_residual_correction(fitted, df: pd.DataFrame) -> np.ndarray:
    X = fitted['scaler'].transform(df[V3_RESIDUAL_FEATURES])
    return fitted['model'].predict(X)


def metrics(y: np.ndarray, pred: np.ndarray, won: np.ndarray) -> dict:
    return {
        'N_Games': int(len(y)),
        'WinnerAccuracy': float(accuracy_score(won, pred > 0)),
        'MarginMAE': float(mean_absolute_error(y, pred)),
    }


def evaluate_prediction(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TARGET].to_numpy()
    won = df['A_Won'].to_numpy()
    out = metrics(y, pred, won)
    t = df['IsTourney'].eq(1).to_numpy()
    out.update({
        'N_TourneyGames': int(t.sum()),
        'TourneyWinnerAccuracy': float(accuracy_score(won[t], pred[t] > 0)) if t.any() else np.nan,
        'TourneyMarginMAE': float(mean_absolute_error(y[t], pred[t])) if t.any() else np.nan,
    })
    return out


def build_expanding_oliver_oof(data: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window OOF predictions for residual learning."""
    seasons = sorted(data['Season'].unique())
    rows = []
    for i, season in enumerate(seasons):
        if i < MIN_OOF_TRAIN_SEASONS:
            continue
        train = data[data['Season'] < season]
        test = data[data['Season'] == season]
        if train.empty or test.empty:
            continue
        fitted = fit_oliver_learned(train)
        pred = predict_oliver_learned(fitted, test)
        block = test[['Season', 'DayNum', 'IsTourney', 'TeamAID', 'TeamBID', TARGET, 'A_Won', *V3_RESIDUAL_FEATURES]].copy()
        block['OliverLearned_OOF'] = pred
        block['OliverResidual'] = block[TARGET] - block['OliverLearned_OOF']
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def learned_weight_report(fitted) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Model columns: location first, then standardized factor components.
    coef = fitted['model'].coef_
    location_coef = float(coef[0])
    comp_coef = pd.Series(coef[1:], index=FACTOR_COMPONENTS, dtype=float)

    component_rows = []
    family_rows = []
    total_family_strength = 0.0
    family_strength = {}

    for fam, (off_col, def_col) in FACTOR_FAMILIES.items():
        off = float(comp_coef[off_col])
        deff = float(comp_coef[def_col])
        strength = abs(off) + abs(deff)
        family_strength[fam] = strength
        total_family_strength += strength
        within_denom = strength if strength > 0 else 1.0
        component_rows.extend([
            {'FactorFamily': fam, 'Component': off_col, 'StdCoefficientPoints': off,
             'WithinFamilyShare': abs(off) / within_denom},
            {'FactorFamily': fam, 'Component': def_col, 'StdCoefficientPoints': deff,
             'WithinFamilyShare': abs(deff) / within_denom},
        ])

    for fam in FACTOR_FAMILIES:
        family_rows.append({
            'FactorFamily': fam,
            'OliverOriginalWeight': OLIVER_ORIGINAL_WEIGHTS[fam],
            'LearnedImportanceShare': family_strength[fam] / total_family_strength if total_family_strength else np.nan,
            'CombinedAbsStdCoefficientPoints': family_strength[fam],
        })

    components = pd.DataFrame(component_rows)
    families = pd.DataFrame(family_rows)
    families['LocationCoefficientPoints'] = location_coef
    return families, components


def main():
    print('Loading historical V3 data...')
    data = pd.read_csv(DATA_CSV, usecols=USECOLS)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f'Rows: {len(data):,}; seasons {data.Season.min()}-{data.Season.max()}')

    print('Building expanding-window Oliver OOF residuals...')
    oof = build_expanding_oliver_oof(data)
    oof.to_csv(OUT_DIR / 'oliver_learned_expanding_oof.csv', index=False)

    reports = []
    prediction_rows = []

    for test_season in TEST_SEASONS:
        if test_season not in set(data['Season']):
            continue
        train = data[data['Season'] < test_season]
        test = data[data['Season'] == test_season].copy()
        if train.empty or test.empty:
            continue

        original_fit = fit_oliver_original(train)
        original_pred = predict_oliver_original(original_fit, test)

        learned_fit = fit_oliver_learned(train)
        learned_pred = predict_oliver_learned(learned_fit, test)

        residual_train = oof[oof['Season'] < test_season].copy()
        residual_fit = fit_residual_correction(residual_train)
        correction = predict_residual_correction(residual_fit, test)
        hybrid_pred = learned_pred + correction

        for name, pred in [
            ('Oliver Original', original_pred),
            ('Oliver Learned', learned_pred),
            ('Oliver + V3 Residual', hybrid_pred),
        ]:
            row = {'Model': name, 'TestSeason': test_season, **evaluate_prediction(test, pred)}
            reports.append(row)

        p = test[['Season', 'DayNum', 'IsTourney', 'TeamAID', 'TeamBID', TARGET, 'A_Won']].copy()
        p['OliverOriginal_PredMargin'] = original_pred
        p['OliverLearned_PredMargin'] = learned_pred
        p['V3Residual_Correction'] = correction
        p['OliverV3Hybrid_PredMargin'] = hybrid_pred
        prediction_rows.append(p)

        print(f'Finished forward holdout {test_season}')

    report_df = pd.DataFrame(reports)
    report_df.to_csv(OUT_DIR / 'oliver_model_forward_backtest.csv', index=False)
    pred_df = pd.concat(prediction_rows, ignore_index=True)
    pred_df.to_csv(OUT_DIR / 'oliver_model_forward_predictions.csv', index=False)

    # Aggregate 2022-2025 tournament only (2026 tournament absent in this snapshot).
    agg_rows = []
    tour = pred_df[(pred_df['IsTourney'] == 1) & (pred_df['Season'].isin([2022, 2023, 2024, 2025]))]
    pred_cols = {
        'Oliver Original': 'OliverOriginal_PredMargin',
        'Oliver Learned': 'OliverLearned_PredMargin',
        'Oliver + V3 Residual': 'OliverV3Hybrid_PredMargin',
    }
    for name, col in pred_cols.items():
        m = metrics(tour[TARGET].to_numpy(), tour[col].to_numpy(), tour['A_Won'].to_numpy())
        agg_rows.append({'Model': name, 'Scope': '2022-2025 NCAA Tournament', **m})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / 'oliver_model_tournament_aggregate.csv', index=False)

    # Final deployable parameters use all currently available historical rows.
    final_oliver = fit_oliver_learned(data)
    families, components = learned_weight_report(final_oliver)
    families.to_csv(OUT_DIR / 'learned_four_factor_family_weights.csv', index=False)
    components.to_csv(OUT_DIR / 'learned_four_factor_component_weights.csv', index=False)

    # Residual learner uses only out-of-sample Oliver errors, including 2026 regular-season OOF rows.
    final_residual = fit_residual_correction(oof)
    residual_coef = pd.DataFrame({
        'Feature': V3_RESIDUAL_FEATURES,
        'StdCoefficientPoints': final_residual['model'].coef_,
    })
    residual_coef.to_csv(OUT_DIR / 'learned_v3_residual_coefficients.csv', index=False)

    artifact = {
        'version': 'V4-Oliver-Hybrid-provisional',
        'philosophy': 'Dean Oliver Four Factors are stage-1 core; V3 features only correct out-of-sample Oliver residuals.',
        'training_rows': int(len(data)),
        'training_seasons': [int(data.Season.min()), int(data.Season.max())],
        'factor_components': FACTOR_COMPONENTS,
        'factor_families': FACTOR_FAMILIES,
        'original_oliver_weights': OLIVER_ORIGINAL_WEIGHTS,
        'v3_residual_features': V3_RESIDUAL_FEATURES,
        'oliver_component_scaler': final_oliver['component_scaler'],
        'oliver_model': final_oliver['model'],
        'residual_scaler': final_residual['scaler'],
        'residual_model': final_residual['model'],
        'residual_training_is_expanding_oof': True,
    }
    joblib.dump(artifact, OUT_DIR / 'oliver_v3_hybrid_model_v4.joblib')

    summary = {
        'rows': int(len(data)),
        'seasons': [int(data.Season.min()), int(data.Season.max())],
        'oof_residual_rows': int(len(oof)),
        'validation': 'strict expanding/forward season validation; test season uses only earlier seasons',
        'tournament_aggregate_2022_2025': agg_df.to_dict(orient='records'),
        'learned_factor_family_weights': families.to_dict(orient='records'),
        'learned_component_coefficients': components.to_dict(orient='records'),
        'residual_coefficients': residual_coef.to_dict(orient='records'),
    }
    with open(OUT_DIR / 'oliver_hybrid_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Forward backtest ===')
    print(report_df.to_string(index=False))
    print('\n=== Tournament aggregate 2022-2025 ===')
    print(agg_df.to_string(index=False))
    print('\n=== Learned Four Factor family weights ===')
    print(families.to_string(index=False))
    print('\n=== Learned component coefficients ===')
    print(components.to_string(index=False))
    print('\n=== Residual correction coefficients ===')
    print(residual_coef.to_string(index=False))


if __name__ == '__main__':
    main()
