"""
V3 historical training pipeline for the College Basketball Prediction Model.

Expected repo layout:
    data/
        MTeams.csv
        MRegularSeasonDetailedResults.csv
        MNCAATourneyDetailedResults.csv

Run:
    python build_and_train_v3.py

Outputs:
    artifacts/historical_training_data_v3.csv
    model/ridge_margin_model_v3.joblib
    artifacts/v3_baseline_results.csv

Important:
- Every game feature is created BEFORE the current game's result updates team state.
- Team A/B orientation is based on TeamID, NOT winner/loser, preventing target leakage.
- This is a provisional V3 baseline. Ridge alpha and feature selection will be tuned
  with time-aware validation in the next stage.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import json

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("model")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TEAMS_CSV = DATA_DIR / "MTeams.csv"
REG_CSV = DATA_DIR / "MRegularSeasonDetailedResults.csv"
TOURNEY_CSV = DATA_DIR / "MNCAATourneyDetailedResults.csv"

MIN_PRIOR_GAMES = 5
POSS_FT_COEF = 0.475

def safe_div(a, b):
    return np.nan if b == 0 else a / b

def side_stats(row, prefix, opp_prefix):
    score, opp_score = float(row[f"{prefix}Score"]), float(row[f"{opp_prefix}Score"])
    fgm, fga = float(row[f"{prefix}FGM"]), float(row[f"{prefix}FGA"])
    fg3m, fg3a = float(row[f"{prefix}FGM3"]), float(row[f"{prefix}FGA3"])
    ftm, fta = float(row[f"{prefix}FTM"]), float(row[f"{prefix}FTA"])
    orb, drb, tov = float(row[f"{prefix}OR"]), float(row[f"{prefix}DR"]), float(row[f"{prefix}TO"])
    ofgm, ofga = float(row[f"{opp_prefix}FGM"]), float(row[f"{opp_prefix}FGA"])
    ofg3m, ofg3a = float(row[f"{opp_prefix}FGM3"]), float(row[f"{opp_prefix}FGA3"])
    oftm, ofta = float(row[f"{opp_prefix}FTM"]), float(row[f"{opp_prefix}FTA"])
    oorb, odrb, otov = float(row[f"{opp_prefix}OR"]), float(row[f"{opp_prefix}DR"]), float(row[f"{opp_prefix}TO"])
    poss_team = fga - orb + tov + POSS_FT_COEF * fta
    poss_opp = ofga - oorb + otov + POSS_FT_COEF * ofta
    poss = (poss_team + poss_opp) / 2.0
    return {
        "GP":1.0,"PF":score,"PA":opp_score,"Poss":poss,
        "FGM":fgm,"FGA":fga,"FG3M":fg3m,"FG3A":fg3a,"FTM":ftm,"FTA":fta,"ORB":orb,"DRB":drb,"TOV":tov,
        "Opp_FGM":ofgm,"Opp_FGA":ofga,"Opp_FG3M":ofg3m,"Opp_FG3A":ofg3a,"Opp_FTM":oftm,"Opp_FTA":ofta,
        "Opp_ORB":oorb,"Opp_DRB":odrb,"Opp_TOV":otov,
        "Wins":1.0 if score > opp_score else 0.0,"Margin":score-opp_score
    }

def empty_state():
    sample_keys = ["GP","PF","PA","Poss","FGM","FGA","FG3M","FG3A","FTM","FTA","ORB","DRB","TOV",
                   "Opp_FGM","Opp_FGA","Opp_FG3M","Opp_FG3A","Opp_FTM","Opp_FTA","Opp_ORB","Opp_DRB","Opp_TOV",
                   "Wins","Margin"]
    return {k:0.0 for k in sample_keys}

def add_to_state(state, x):
    for k,v in x.items():
        state[k] += v

def summarize_state(s):
    if s["GP"] <= 0:
        return None
    gp=s["GP"]
    ortg=100*safe_div(s["PF"],s["Poss"])
    drtg=100*safe_div(s["PA"],s["Poss"])
    efg=safe_div(s["FGM"]+.5*s["FG3M"],s["FGA"])
    opp_efg=safe_div(s["Opp_FGM"]+.5*s["Opp_FG3M"],s["Opp_FGA"])
    tov=safe_div(s["TOV"],s["FGA"]+POSS_FT_COEF*s["FTA"]+s["TOV"])
    opp_tov=safe_div(s["Opp_TOV"],s["Opp_FGA"]+POSS_FT_COEF*s["Opp_FTA"]+s["Opp_TOV"])
    orb=safe_div(s["ORB"],s["ORB"]+s["Opp_DRB"])
    opp_orb=safe_div(s["Opp_ORB"],s["Opp_ORB"]+s["DRB"])
    ftr=safe_div(s["FTM"],s["FGA"])
    opp_ftr=safe_div(s["Opp_FTM"],s["Opp_FGA"])
    p3=safe_div(s["FG3M"],s["FG3A"])
    opp_p3=safe_div(s["Opp_FG3M"],s["Opp_FG3A"])
    p3ar=safe_div(s["FG3A"],s["FGA"])
    opp_p3ar=safe_div(s["Opp_FG3A"],s["Opp_FGA"])
    ts=safe_div(s["PF"],2*(s["FGA"]+POSS_FT_COEF*s["FTA"]))
    opp_ts=safe_div(s["PA"],2*(s["Opp_FGA"]+POSS_FT_COEF*s["Opp_FTA"]))
    return {
        "GP":gp,"ORtg":ortg,"DRtg":drtg,"NetRtg":ortg-drtg,
        "eFG":efg,"Opp_eFG":opp_efg,"TOVpct":tov,"Opp_TOVpct":opp_tov,
        "ORBpct":orb,"Opp_ORBpct":opp_orb,"FTR":ftr,"Opp_FTR":opp_ftr,
        "P3":p3,"Opp_P3":opp_p3,"P3Ar":p3ar,"Opp_P3Ar":opp_p3ar,
        "TS":ts,"Opp_TS":opp_ts,"Pace":safe_div(s["Poss"],gp),
        "WinPct":safe_div(s["Wins"],gp),"AvgMargin":safe_div(s["Margin"],gp)
    }

def a_location_value(row,a_is_winner):
    if row["WLoc"]=="N": return 0.0
    winner_loc=1.0 if row["WLoc"]=="H" else -1.0
    return winner_loc if a_is_winner else -winner_loc

def build_dataset():
    teams=pd.read_csv(TEAMS_CSV)
    reg=pd.read_csv(REG_CSV); reg["IsTourney"]=0
    tour=pd.read_csv(TOURNEY_CSV); tour["IsTourney"]=1
    names=dict(zip(teams.TeamID,teams.TeamName))
    games=pd.concat([reg,tour],ignore_index=True).sort_values(["Season","DayNum","IsTourney"]).reset_index(drop=True)
    states=defaultdict(empty_state); out=[]
    for _,g in games.iterrows():
        season=int(g.Season); wid=int(g.WTeamID); lid=int(g.LTeamID)
        aid,bid=sorted([wid,lid]); a_is_winner=(aid==wid)
        A=summarize_state(states[(season,aid)]); B=summarize_state(states[(season,bid)])
        if A and B and A["GP"]>=MIN_PRIOR_GAMES and B["GP"]>=MIN_PRIOR_GAMES:
            a_score=float(g.WScore if a_is_winner else g.LScore)
            b_score=float(g.LScore if a_is_winner else g.WScore)
            r={"Season":season,"DayNum":int(g.DayNum),"IsTourney":int(g.IsTourney),
               "TeamAID":aid,"TeamBID":bid,"TeamA":names.get(aid,str(aid)),"TeamB":names.get(bid,str(bid)),
               "A_Location":a_location_value(g,a_is_winner),"ActualMargin":a_score-b_score,"A_Won":int(a_score>b_score)}
            for prefix,s in [("A",A),("B",B)]:
                for k,v in s.items(): r[f"{prefix}_{k}"]=v
            r.update({
                "GP_Min":min(A["GP"],B["GP"]),"GP_Diff":A["GP"]-B["GP"],
                "ORtg_Diff":A["ORtg"]-B["ORtg"],"DRtg_Adv":B["DRtg"]-A["DRtg"],
                "NetRtg_Diff":A["NetRtg"]-B["NetRtg"],"eFG_Diff":A["eFG"]-B["eFG"],
                "Def_eFG_Adv":B["Opp_eFG"]-A["Opp_eFG"],"TOV_Adv":B["TOVpct"]-A["TOVpct"],
                "ForceTOV_Adv":A["Opp_TOVpct"]-B["Opp_TOVpct"],"ORB_Diff":A["ORBpct"]-B["ORBpct"],
                "DefORB_Adv":B["Opp_ORBpct"]-A["Opp_ORBpct"],"FTR_Diff":A["FTR"]-B["FTR"],
                "DefFTR_Adv":B["Opp_FTR"]-A["Opp_FTR"],"P3_Diff":A["P3"]-B["P3"],
                "DefP3_Adv":B["Opp_P3"]-A["Opp_P3"],"P3Ar_Diff":A["P3Ar"]-B["P3Ar"],
                "TS_Diff":A["TS"]-B["TS"],"DefTS_Adv":B["Opp_TS"]-A["Opp_TS"],
                "Pace_Mean":(A["Pace"]+B["Pace"])/2,"Pace_Diff":A["Pace"]-B["Pace"],
                "WinPct_Diff":A["WinPct"]-B["WinPct"],"AvgMargin_Diff":A["AvgMargin"]-B["AvgMargin"]
            })
            out.append(r)
        add_to_state(states[(season,wid)],side_stats(g,"W","L"))
        add_to_state(states[(season,lid)],side_stats(g,"L","W"))
    return pd.DataFrame(out).replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)

FEATURES=["A_Location","GP_Diff","ORtg_Diff","DRtg_Adv","NetRtg_Diff","eFG_Diff","Def_eFG_Adv",
          "TOV_Adv","ForceTOV_Adv","ORB_Diff","DefORB_Adv","FTR_Diff","DefFTR_Adv","P3_Diff",
          "DefP3_Adv","P3Ar_Diff","TS_Diff","DefTS_Adv","Pace_Mean","Pace_Diff","WinPct_Diff","AvgMargin_Diff"]

if __name__=="__main__":
    data=build_dataset()
    data.to_csv(ARTIFACT_DIR/"historical_training_data_v3.csv",index=False)
    report=[]
    for test_season in [s for s in [2022,2023,2024,2025,2026] if s in set(data.Season)]:
        train=data[data.Season!=test_season]; test=data[data.Season==test_season]
        model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
        model.fit(train[FEATURES],train["ActualMargin"])
        pred=model.predict(test[FEATURES])
        t=test.IsTourney.eq(1)
        report.append({
            "TestSeason":test_season,"N_Games":len(test),
            "WinnerAccuracy":accuracy_score(test.A_Won,pred>0),
            "MarginMAE":mean_absolute_error(test.ActualMargin,pred),
            "N_TourneyGames":int(t.sum()),
            "TourneyWinnerAccuracy":accuracy_score(test.loc[t,"A_Won"],pred[t]>0) if t.any() else np.nan,
            "TourneyMarginMAE":mean_absolute_error(test.loc[t,"ActualMargin"],pred[t]) if t.any() else np.nan
        })
    pd.DataFrame(report).to_csv(ARTIFACT_DIR/"v3_baseline_results.csv",index=False)
    final=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    final.fit(data[FEATURES],data["ActualMargin"])
    joblib.dump({"model":final,"features":FEATURES,"min_prior_games":MIN_PRIOR_GAMES,
                 "poss_ft_coef":POSS_FT_COEF,"training_rows":len(data),
                 "training_seasons":[int(data.Season.min()),int(data.Season.max())]},
                MODEL_DIR/"ridge_margin_model_v3.joblib")
    print(pd.DataFrame(report).to_string(index=False))
