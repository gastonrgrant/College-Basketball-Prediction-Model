import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from functools import lru_cache


# =========================================================
# CONFIG
# =========================================================

# =========================================================
# LIVE SCRAPING URLS (replaces CSVs)
# =========================================================

TEAM_BASIC_URL = "https://www.sports-reference.com/cbb/seasons/men/2026-school-stats.html?per_game=1"
OPP_BASIC_URL = "https://www.sports-reference.com/cbb/seasons/men/2026-opponent-stats.html?per_game=1"
TEAM_ADV_URL = "https://www.sports-reference.com/cbb/seasons/men/2026-advanced-school-stats.html"
OPP_ADV_URL = "https://www.sports-reference.com/cbb/seasons/men/2026-advanced-opponent-stats.html"

TEAM_BASIC_CSV = "basic_team_stats.csv"
OPP_BASIC_CSV = "basic_opponent_stats.csv"
TEAM_ADV_CSV = "advanced_team_stats.csv"
OPP_ADV_CSV = "advanced_opponent_stats.csv"

# If your exact column names differ slightly, update these.
TEAM_ADV_KEEP = [
    "School", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%",
    "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%", "FT/FGA"
]

OPP_ADV_KEEP = [
    "School", "ORtg", "FTr", "3PAr", "TS%", "TRB%",
    "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%", "FT/FGA"
]

TEAM_BASIC_KEEP = [
    "School", "Tm.", "Opp.", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "ORB", "TRB", "AST", "STL", "BLK", "TOV", "PF"
]

OPP_BASIC_KEEP = [
    "School", "Tm.", "Opp.", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "ORB", "TRB", "AST", "STL", "BLK", "TOV", "PF"
]


# =========================================================
# CLEANING HELPERS
# =========================================================

def normalize_team_name(name: str) -> str:
    """Standardize team names and strip NCAA tag."""
    if pd.isna(name):
        return name

    name = str(name)
    name = name.replace("\xa0", " ").strip()

    # Remove trailing NCAA marker if present
    name = re.sub(r"\s*NCAA\s*$", "", name).strip()

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)

    return name


def canonical_lookup_name(name: str) -> str:
    """
    Case-insensitive lookup normalization.
    Treats St, St., State, and Saint the same for lookup purposes.
    """
    if pd.isna(name):
        return ""

    name = normalize_team_name(str(name)).lower()
    name = name.replace("&", "and")
    name = re.sub(r"[.\',()/-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    tokens = []
    for token in name.split():
        if token in {"st", "state", "saint"}:
            tokens.append("st")
        else:
            tokens.append(token)

    return " ".join(tokens)


def has_ncaa_tag(name: str) -> bool:
    if pd.isna(name):
        return False
    return bool(re.search(r"\bNCAA\b", str(name)))


def clean_sportsref_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated header rows and empty junk rows."""
    df = df.copy()

    if "School" not in df.columns:
        raise ValueError("Expected a 'School' column in the CSV.")

    # Remove repeated header rows
    df = df[df["School"].astype(str).str.strip() != "School"]

    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.strip() != "Rk"]

    df = df[df["School"].notna()].copy()
    df["Original_School"] = df["School"].astype(str)
    df["Is_Tourney_Team"] = df["Original_School"].apply(has_ncaa_tag)
    df["Team"] = df["School"].apply(normalize_team_name)

    return df


def keep_columns(df: pd.DataFrame, keep_cols: list, label: str) -> pd.DataFrame:
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing expected columns: {missing}")
    return df[keep_cols + ["Team", "Original_School", "Is_Tourney_Team"]].copy()


def convert_numeric(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    if exclude is None:
        exclude = []

    df = df.copy()
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

from bs4 import Comment

def scrape_table(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)

    soup = BeautifulSoup(res.text, "lxml")

    tables = []

    # grab commented tables
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in c:
            temp = BeautifulSoup(c, "lxml")
            tables.extend(temp.find_all("table"))

    # grab normal tables
    tables.extend(soup.find_all("table"))

    for t in tables:
        if "School" in t.text:
            df = pd.read_html(str(t))[0]

            # 🔥 FIX: flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)

            df.columns = [str(c).strip() for c in df.columns]

            return df

    raise ValueError(f"Table not found at {url}")

# =========================================================
# LOAD + MERGE
# =========================================================

def load_and_prepare_data(team_basic_csv, opp_basic_csv, team_adv_csv, opp_adv_csv):
    try:
        team_basic = scrape_table(TEAM_BASIC_URL)
        opp_basic = scrape_table(OPP_BASIC_URL)
        team_adv = scrape_table(TEAM_ADV_URL)
        opp_adv = scrape_table(OPP_ADV_URL)
        print("Using LIVE scraped data.\n")
    except:
        print("Scrape failed — falling back to CSV.\n")
        team_basic = pd.read_csv(team_basic_csv)
        opp_basic = pd.read_csv(opp_basic_csv)
        team_adv = pd.read_csv(team_adv_csv)
        opp_adv = pd.read_csv(opp_adv_csv)

    team_basic = clean_sportsref_df(team_basic)
    opp_basic = clean_sportsref_df(opp_basic)
    team_adv = clean_sportsref_df(team_adv)
    opp_adv = clean_sportsref_df(opp_adv)

    team_basic = keep_columns(team_basic, TEAM_BASIC_KEEP, "TEAM_BASIC")
    opp_basic = keep_columns(opp_basic, OPP_BASIC_KEEP, "OPP_BASIC")
    team_adv = keep_columns(team_adv, TEAM_ADV_KEEP, "TEAM_ADV")
    opp_adv = keep_columns(opp_adv, OPP_ADV_KEEP, "OPP_ADV")

    # Rename team basic columns
    team_basic = team_basic.rename(columns={
        "Tm.": "Pts_For",
        "Opp.": "Pts_Against",
        "FG": "FGM",
        "FGA": "FGA",
        "FG%": "FG%",
        "3P": "3PM",
        "3PA": "3PA",
        "3P%": "3P%",
        "FT": "FTM",
        "FTA": "FTA",
        "FT%": "FT%",
        "ORB": "ORB_pg",
        "TRB": "TRB_pg",
        "AST": "AST_pg",
        "STL": "STL_pg",
        "BLK": "BLK_pg",
        "TOV": "TOV_pg",
        "PF": "PF_pg"
    })

    # Rename opponent basic columns
    opp_basic = opp_basic.rename(columns={
        "Tm.": "Opp_Pts_For",
        "Opp.": "Opp_Pts_Against",
        "FG": "Opp_FGM",
        "FGA": "Opp_FGA",
        "FG%": "Opp_FG%",
        "3P": "Opp_3PM",
        "3PA": "Opp_3PA",
        "3P%": "Opp_3P%",
        "FT": "Opp_FTM",
        "FTA": "Opp_FTA",
        "FT%": "Opp_FT%",
        "ORB": "Opp_ORB_pg",
        "TRB": "Opp_TRB_pg",
        "AST": "Opp_AST_pg",
        "STL": "Opp_STL_pg",
        "BLK": "Opp_BLK_pg",
        "TOV": "Opp_TOV_pg",
        "PF": "Opp_PF_pg"
    })

    # Rename opponent advanced columns
    opp_adv = opp_adv.rename(columns={
        "ORtg": "Opp_ORtg",
        "FTr": "Opp_FTr",
        "3PAr": "Opp_3PAr",
        "TS%": "Opp_TS%",
        "TRB%": "Opp_TRB%",
        "AST%": "Opp_AST%",
        "STL%": "Opp_STL%",
        "BLK%": "Opp_BLK%",
        "eFG%": "Opp_eFG%",
        "TOV%": "Opp_TOV%",
        "ORB%": "Opp_ORB%",
        "FT/FGA": "Opp_FT/FGA"
    })

    # Keep only one copy of team metadata
    team_meta = team_adv[["Team", "Original_School", "Is_Tourney_Team"]].copy()

    team_adv = team_adv.drop(columns=["Original_School", "Is_Tourney_Team"])
    team_basic = team_basic.drop(columns=["Original_School", "Is_Tourney_Team"])
    opp_basic = opp_basic.drop(columns=["Original_School", "Is_Tourney_Team"])
    opp_adv = opp_adv.drop(columns=["Original_School", "Is_Tourney_Team"])

    merged = team_meta.merge(team_adv, on="Team", how="inner")
    merged = merged.merge(team_basic, on=["Team", "School"], how="inner")
    merged = merged.merge(opp_adv, on=["Team", "School"], how="inner")
    merged = merged.merge(opp_basic, on=["Team", "School"], how="inner")

    merged = convert_numeric(
        merged,
        exclude=["Team", "School", "Original_School", "Is_Tourney_Team"]
    )

    # Remove exact duplicates if any
    merged = merged.drop_duplicates(subset=["Team"]).reset_index(drop=True)

    # Canonical lookup key
    merged["LookupKey"] = merged["Team"].apply(canonical_lookup_name)

    full_df = merged.copy()
    tourney_df = merged[merged["Is_Tourney_Team"]].copy()

    return full_df, tourney_df


# =========================================================
# BASELINES
# =========================================================

def build_baselines(df: pd.DataFrame):
    """Build means, stds, and percentiles using all D1 teams."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    baselines = {}

    for col in numeric_cols:
        s = df[col].dropna().astype(float)
        if len(s) == 0:
            continue

        baselines[col] = {
            "mean": s.mean(),
            "std": s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0
        }

    percentile_cols = [
        "ORtg", "eFG%", "TOV%", "ORB%", "TRB%", "FT/FGA", "3PAr", "AST%", "TS%",
        "Pace", "SRS", "SOS", "FT%", "3P%", "ORB_pg", "TRB_pg", "AST_pg",
        "STL_pg", "BLK_pg", "TOV_pg", "Pts_For", "3PA",
        "Opp_ORtg", "Opp_eFG%", "Opp_TOV%", "Opp_ORB%", "Opp_TRB%", "Opp_FT/FGA", "Opp_3PAr",
        "Opp_3P%", "Opp_TOV_pg", "Opp_ORB_pg", "Opp_Pts_Against"
    ]

    percentile_cols = [c for c in percentile_cols if c in df.columns]

    percentile_tables = {}
    for col in percentile_cols:
        percentile_tables[col] = df[col].rank(pct=True)

    return baselines, percentile_tables


def zscore(value, col, baselines):
    if col not in baselines:
        return 0.0
    return (value - baselines[col]["mean"]) / baselines[col]["std"]


def percentile_of_team(df, percentile_tables, team_name, col):
    if col not in percentile_tables:
        return None
    row_idx = df.index[df["Team"] == team_name]
    if len(row_idx) == 0:
        return None
    return float(percentile_tables[col].loc[row_idx[0]])


def oriented_percentile(df, percentile_tables, team_name, col, lower_is_better=False):
    p = percentile_of_team(df, percentile_tables, team_name, col)
    if p is None:
        return None
    return 1 - p if lower_is_better else p


# =========================================================
# PHYSICAL PROFILE HELPERS
# =========================================================

def inches_to_feet_string(inches):
    if inches is None:
        return "N/A"
    feet = int(inches // 12)
    rem_inches = int(round(inches - feet * 12))
    if rem_inches == 12:
        feet += 1
        rem_inches = 0
    return f"{feet}'{rem_inches}\" ({inches:.1f} in)"


from bs4 import BeautifulSoup, Comment
import requests
import re
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def get_team_physical_profile(team_name, year=2026, min_mp=10.0):

    TEAM_SLUG_MAP = {
        "UCF": "central-florida",
        "USC": "southern-california",
        "Ole Miss": "mississippi",
        "UConn": "connecticut",
        "BYU": "brigham-young",
        "SMU": "southern-methodist",
        "LSU": "louisiana-state",
        "UNLV": "nevada-las-vegas",
        "UTEP": "texas-el-paso",
        "UTSA": "texas-san-antonio",
        "UAB": "alabama-birmingham",
        "St. John's": "st-johns-ny",
        "St John's": "st-johns-ny",
        "VCU": "virginia-commonwealth",
    }

    def build_slug(name):
        if name in TEAM_SLUG_MAP:
            return TEAM_SLUG_MAP[name]

        name = name.lower()
        name = name.replace("&", "and")
        name = re.sub(r"[.\']", "", name)
        name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
        return name

    def try_fetch(slug):
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{year}.html"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return res
        except:
            pass
        return None

    # 🔥 FIXED NAME CLEANING (handles Jr, Sr, etc.)
    def clean_name(n):
        n = re.sub(r"\s+", " ", n.strip())
        n = re.sub(r"\b(Jr|Sr|II|III|IV)\b\.?", "", n)
        return n.strip()

    def try_fetch(slug):
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{year}.html"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return res
        except:
            pass
        return None

    # Fetch page
    slug = build_slug(team_name)
    res = try_fetch(slug)

    if res is None:
        fallback_names = [
            team_name.replace("UCF", "Central Florida"),
            team_name.replace("USC", "Southern California"),
            team_name.replace("Ole Miss", "Mississippi"),
        ]
        for alt in fallback_names:
            slug = build_slug(alt)
            res = try_fetch(slug)
            if res:
                break

    if res is None:
        return None

    soup = BeautifulSoup(res.content, "lxml")

    # Extract tables (including commented)
    all_tables = []
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in c:
            temp = BeautifulSoup(c, "lxml")
            all_tables.extend(temp.find_all("table"))

    all_tables.extend(soup.find_all("table"))

    pg_table = None
    roster_table = None

    for t in all_tables:
        if t.get("id") == "players_per_game":
            pg_table = t
        if t.get("id") == "roster":
            roster_table = t

    if pg_table is None or roster_table is None:
        return None

    minutes = {}
    stats = {}
    positions = {}

    # =========================
    # PER GAME STATS
    # =========================
    for row in pg_table.find("tbody").find_all("tr"):
        name = row.find("td", {"data-stat": "name_display"})
        mp = row.find("td", {"data-stat": "mp_per_g"})

        if not name or not mp:
            continue

        n = clean_name(name.get_text())
        m = float(mp.get_text())

        if m < min_mp:
            continue

        minutes[n] = m

        def get(stat):
            cell = row.find("td", {"data-stat": stat})
            return float(cell.get_text()) if cell and cell.get_text() else 0

        stats[n] = {
            "pts": get("pts_per_g"),
            "ast": get("ast_per_g"),
            "trb": get("trb_per_g"),
            "fg": get("fg_pct"),
            "fg3": get("fg3_pct"),
            "fg3a": get("fg3a_per_g")
        }

        pos = row.find("td", {"data-stat": "pos"})
        positions[n] = pos.get_text(strip=True) if pos else "?"

    # =========================
    # HEIGHT PARSING (FIXED)
    # =========================
    heights = {}

    for row in roster_table.find("tbody").find_all("tr"):
        name = row.find("th", {"data-stat": "player"}) or row.find("td", {"data-stat": "player"})
        h = row.find("td", {"data-stat": "height"})

        if name and h:
            heights[clean_name(name.get_text())] = h.get_text(strip=True)

    def to_inches(h):
        try:
            h = str(h).strip()
            if "-" not in h:
                return None
            ft, inch = h.split("-", 1)
            return float(ft) * 12 + float(inch)
        except:
            return None

    # =========================
    # BUILD PROFILE
    # =========================
    h_vals = []
    weights = []
    pos_groups = {"G": [], "F": [], "C": []}
    stars = []

    for n in minutes:
        if n not in heights:
            continue

        h_in = to_inches(heights[n])
        if h_in is None:
            continue

        h_vals.append(h_in)
        weights.append(minutes[n])

        pos = positions.get(n, "?")
        if pos in pos_groups:
            pos_groups[pos].append(h_in)

        # star logic (unchanged)
        if minutes[n] >= 25 or stats[n]["pts"] >= 12:
            s = stats[n]
            stars.append(
                f"{n} ({pos}) "
                f"{s['pts']:.1f} ppg, {s['ast']:.1f} ast, {s['trb']:.1f} reb, "
                f"{s['fg']:.3f} fg%, {s['fg3']:.3f} 3p%, {s['fg3a']:.1f} 3PA"
            )

    if not h_vals:
        return None

    avg_height = np.average(h_vals, weights=weights)

    def avg_group(lst):
        if not lst:
            return None
        val = np.mean(lst)
        return f"{int(val//12)}'{int(val%12)}\""

    return {
        "avg_height_inches": float(avg_height),
        "avg_height_str": f"{int(avg_height//12)}'{int(avg_height%12)}\"",
        "rotation_size": len(h_vals),
        "guards": avg_group(pos_groups["G"]),
        "forwards": avg_group(pos_groups["F"]),
        "centers": avg_group(pos_groups["C"]),
        "stars": stars[:6]
    }
# =========================================================
# MATCHUP MODEL — V4.1 HISTORICALLY TRAINED OLIVER HYBRID
# =========================================================
# IMPORTANT:
# Dean Oliver Four Factors are the trained core; V3 features are a small
# residual correction. All prediction-time coefficients are embedded below.


# =========================================================
# HISTORICALLY LEARNED MATCHUP PARAMETERS
# =========================================================
# Trained from 102,607 pre-game NCAA observations (2003-2026).
# The Four Factors remain the core of the model.  Coefficients below are
# direct point-margin slopes learned from historical NCAA games while
# enforcing Team-A/Team-B symmetry (swapping teams flips the prediction).
#
# IMPORTANT UNIT NOTE:
# Sports-Reference prints some percentage stats as 0-100 values (for example
# TOV% and ORB%), while others are already 0-1 decimals (for example eFG%).
# _as_fraction() normalizes them before applying the historical coefficients.

FOUR_FACTOR_POINT_COEFS = {
    # Shooting efficiency
    "eFG_Diff": 73.4620226969,
    "Def_eFG_Adv": 58.3245101598,

    # Turnovers
    "TOV_Adv": 91.8850117640,
    "ForceTOV_Adv": 73.4087928355,

    # Offensive / defensive rebounding
    "ORB_Diff": 33.2479851361,
    "DefORB_Adv": 30.1817552091,

    # Free-throw pressure
    "FTR_Diff": 12.5769872520,
    "DefFTR_Adv": 12.1923107577,
}

# These secondary V3 signals were trained ONLY to explain what the Four
# Factors missed. Their coefficients are intentionally tiny relative to the
# Four-Factor core, keeping Dean Oliver's framework central to the model.
V3_RESIDUAL_POINT_COEFS = {
    "NetRtg_Diff": 0.0081545121,
    "P3Ar_Diff": 4.9124626635,
    "Pace_Diff": -0.0287659650,
}

# Forward-held-out calibration of predicted margin -> win probability.
# logit(P[Team A wins]) = WIN_PROB_LOGIT_SLOPE * projected_margin
WIN_PROB_LOGIT_SLOPE = 0.1432050386

# Learned offense share inside each Four-Factor family.  These are used only
# for descriptive matchup projections; the final margin uses the separate
# offense and defense coefficients above directly.
FOUR_FACTOR_OFFENSE_SHARE = {
    "eFG": 0.5741319935,
    "TOV": 0.5341281235,
    "ORB": 0.5872674213,
    "FTR": 0.4863930149,
}


def _as_fraction(value):
    """Normalize Sports-Reference rate stats to 0-1 units."""
    x = float(value)
    return x / 100.0 if abs(x) > 1.5 else x


def predict_stat(off_value, def_allowed_value, offense_weight):
    """Blend offense and opponent defense using a historically learned share."""
    defense_weight = 1.0 - offense_weight
    return offense_weight * float(off_value) + defense_weight * float(def_allowed_value)


def clamp(x, low, high):
    return max(low, min(high, x))


def build_matchup_features(A, B, baselines):
    """Build live matchup features in the same orientation as historical V4.1.

    Every signed feature is positive when it favors Team A.  The final margin
    is built primarily from Dean Oliver's Four Factors, with a small V3
    residual correction learned from historical errors.
    """

    # Normalize rates so current Sports-Reference data matches historical units.
    a_efg, b_efg = _as_fraction(A["eFG%"]), _as_fraction(B["eFG%"])
    a_def_efg, b_def_efg = _as_fraction(A["Opp_eFG%"]), _as_fraction(B["Opp_eFG%"])

    a_tov, b_tov = _as_fraction(A["TOV%"]), _as_fraction(B["TOV%"])
    a_force_tov, b_force_tov = _as_fraction(A["Opp_TOV%"]), _as_fraction(B["Opp_TOV%"])

    a_orb, b_orb = _as_fraction(A["ORB%"]), _as_fraction(B["ORB%"])
    a_opp_orb, b_opp_orb = _as_fraction(A["Opp_ORB%"]), _as_fraction(B["Opp_ORB%"])

    a_ftr, b_ftr = _as_fraction(A["FT/FGA"]), _as_fraction(B["FT/FGA"])
    a_opp_ftr, b_opp_ftr = _as_fraction(A["Opp_FT/FGA"]), _as_fraction(B["Opp_FT/FGA"])

    a_3par, b_3par = _as_fraction(A["3PAr"]), _as_fraction(B["3PAr"])

    # ---------------------------------------------------------
    # Dean Oliver Four Factors: separate offense and defense.
    # ---------------------------------------------------------
    efg_diff = a_efg - b_efg
    def_efg_adv = b_def_efg - a_def_efg

    # Lower offensive turnover rate is better.
    tov_adv = b_tov - a_tov
    # Higher opponent turnover rate means the defense forces more turnovers.
    force_tov_adv = a_force_tov - b_force_tov

    orb_diff = a_orb - b_orb
    # Lower opponent ORB% means better defensive rebounding.
    def_orb_adv = b_opp_orb - a_opp_orb

    ftr_diff = a_ftr - b_ftr
    # Lower opponent FT/FGA allowed is better.
    def_ftr_adv = b_opp_ftr - a_opp_ftr

    # ---------------------------------------------------------
    # Secondary V3 residual features.
    # ---------------------------------------------------------
    a_net = float(A["ORtg"]) - float(A["Opp_ORtg"])
    b_net = float(B["ORtg"]) - float(B["Opp_ORtg"])
    netrtg_diff = a_net - b_net
    p3ar_diff = a_3par - b_3par
    pace_diff = float(A["Pace"]) - float(B["Pace"])

    # Descriptive offense-vs-defense projections. These are NOT what determines
    # the final margin; they exist for the report/commentary layer.
    pred_A_efg = predict_stat(a_efg, b_def_efg, FOUR_FACTOR_OFFENSE_SHARE["eFG"])
    pred_B_efg = predict_stat(b_efg, a_def_efg, FOUR_FACTOR_OFFENSE_SHARE["eFG"])
    pred_A_tov = predict_stat(a_tov, b_force_tov, FOUR_FACTOR_OFFENSE_SHARE["TOV"])
    pred_B_tov = predict_stat(b_tov, a_force_tov, FOUR_FACTOR_OFFENSE_SHARE["TOV"])
    pred_A_orb = predict_stat(a_orb, b_opp_orb, FOUR_FACTOR_OFFENSE_SHARE["ORB"])
    pred_B_orb = predict_stat(b_orb, a_opp_orb, FOUR_FACTOR_OFFENSE_SHARE["ORB"])
    pred_A_ftr = predict_stat(a_ftr, b_opp_ftr, FOUR_FACTOR_OFFENSE_SHARE["FTR"])
    pred_B_ftr = predict_stat(b_ftr, a_opp_ftr, FOUR_FACTOR_OFFENSE_SHARE["FTR"])

    # ORtg midpoint is descriptive only; it does not enter the learned margin.
    pred_A_ortg = (float(A["ORtg"]) + float(B["Opp_ORtg"])) / 2.0
    pred_B_ortg = (float(B["ORtg"]) + float(A["Opp_ORtg"])) / 2.0

    factor_raw = {
        "eFG_Diff": efg_diff,
        "Def_eFG_Adv": def_efg_adv,
        "TOV_Adv": tov_adv,
        "ForceTOV_Adv": force_tov_adv,
        "ORB_Diff": orb_diff,
        "DefORB_Adv": def_orb_adv,
        "FTR_Diff": ftr_diff,
        "DefFTR_Adv": def_ftr_adv,
    }

    factor_point_contrib = {
        name: factor_raw[name] * FOUR_FACTOR_POINT_COEFS[name]
        for name in factor_raw
    }

    four_factor_margin = sum(factor_point_contrib.values())

    residual_raw = {
        "NetRtg_Diff": netrtg_diff,
        "P3Ar_Diff": p3ar_diff,
        "Pace_Diff": pace_diff,
    }
    residual_point_contrib = {
        name: residual_raw[name] * V3_RESIDUAL_POINT_COEFS[name]
        for name in residual_raw
    }
    v3_residual_correction = sum(residual_point_contrib.values())

    # Family-level contributions are easier to interpret in the report.
    efg_margin = factor_point_contrib["eFG_Diff"] + factor_point_contrib["Def_eFG_Adv"]
    tov_margin = factor_point_contrib["TOV_Adv"] + factor_point_contrib["ForceTOV_Adv"]
    orb_margin = factor_point_contrib["ORB_Diff"] + factor_point_contrib["DefORB_Adv"]
    ftr_margin = factor_point_contrib["FTR_Diff"] + factor_point_contrib["DefFTR_Adv"]

    return {
        "pred_A_ortg": pred_A_ortg,
        "pred_B_ortg": pred_B_ortg,
        "pred_A_efg": pred_A_efg,
        "pred_B_efg": pred_B_efg,
        "pred_A_tov": pred_A_tov,
        "pred_B_tov": pred_B_tov,
        "pred_A_orb": pred_A_orb,
        "pred_B_orb": pred_B_orb,
        "pred_A_ftr": pred_A_ftr,
        "pred_B_ftr": pred_B_ftr,
        "four_factor_score": four_factor_margin,
        "four_factor_margin": four_factor_margin,
        "v3_residual_correction": v3_residual_correction,
        "factor_point_contrib": factor_point_contrib,
        "residual_point_contrib": residual_point_contrib,
        "efg_margin": efg_margin,
        "tov_margin": tov_margin,
        "orb_margin": orb_margin,
        "ftr_margin": ftr_margin,
        "NetRtg_Diff": netrtg_diff,
        "P3Ar_Diff": p3ar_diff,
        "Pace_Diff": pace_diff,
        # Keep familiar names for downstream report compatibility.
        "efg_edge": efg_margin,
        "tov_edge": tov_margin,
        "orb_edge": orb_margin,
        "ftr_edge": ftr_margin,
        "ortg_edge": netrtg_diff,
        "srs_edge_raw": float(A["SRS"]) - float(B["SRS"]) if "SRS" in A.index else 0.0,
        "srs_edge_std": (
            zscore(A["SRS"], "SRS", baselines) - zscore(B["SRS"], "SRS", baselines)
            if "SRS" in A.index else 0.0
        ),
        "sos_edge_std": (
            zscore(A["SOS"], "SOS", baselines) - zscore(B["SOS"], "SOS", baselines)
            if "SOS" in A.index else 0.0
        ),
    }


def historically_projected_margin(features, expected_possessions=None, baselines=None):
    """Return the learned NCAA point margin.

    Stage 1: Dean Oliver Four Factors (dominant core).
    Stage 2: small V3 residual correction trained on Four-Factor errors.

    No 0.55/0.35/0.10 heuristic blend and no arbitrary Four-Factor scaling.
    """
    four_factor_margin = float(features["four_factor_margin"])
    v3_correction = float(features["v3_residual_correction"])
    projected_margin = four_factor_margin + v3_correction

    return projected_margin, {
        "four_factor_margin": four_factor_margin,
        "v3_residual_correction": v3_correction,
        "eFG_margin": float(features["efg_margin"]),
        "turnover_margin": float(features["tov_margin"]),
        "rebounding_margin": float(features["orb_margin"]),
        "free_throw_margin": float(features["ftr_margin"]),
        "net_rating_correction": float(features["residual_point_contrib"]["NetRtg_Diff"]),
        "three_point_rate_correction": float(features["residual_point_contrib"]["P3Ar_Diff"]),
        "pace_correction": float(features["residual_point_contrib"]["Pace_Diff"]),
    }


# Backwards-compatible alias so the rest of the program still works if this
# function name is referenced elsewhere.
def heuristic_projected_margin(features, expected_possessions, baselines):
    return historically_projected_margin(features, expected_possessions, baselines)


def margin_to_win_probability(projected_margin):
    """Historically calibrated win probability from forward-held-out margins."""
    logit = WIN_PROB_LOGIT_SLOPE * float(projected_margin)
    return 1.0 / (1.0 + np.exp(-logit))


def compute_matchup(team_a, team_b, full_df, baselines, percentile_tables,
                    seed_a=None, seed_b=None):
    if team_a not in full_df["Team"].values:
        raise ValueError(f"{team_a} not found.")
    if team_b not in full_df["Team"].values:
        raise ValueError(f"{team_b} not found.")

    A = full_df.loc[full_df["Team"] == team_a].iloc[0]
    B = full_df.loc[full_df["Team"] == team_b].iloc[0]

    features = build_matchup_features(A, B, baselines)

    expected_possessions = (
        (A["Pace"] + B["Pace"]) / 2.0
        if "Pace" in full_df.columns else 68.0
    )

    projected_margin, margin_components = heuristic_projected_margin(
        features,
        expected_possessions,
        baselines,
    )

    # No arbitrary blowout multiplier and no second pace-confidence adjustment.
    # Pace is already reflected when efficiency is converted into points.
    projected_margin = clamp(projected_margin, -35.0, 35.0)
    win_prob_a = margin_to_win_probability(projected_margin)

    winner = team_a if projected_margin >= 0 else team_b
    loser = team_b if winner == team_a else team_a
    win_prob = win_prob_a if winner == team_a else 1.0 - win_prob_a
    spread = abs(projected_margin)

    # The model score is the historically learned signed point margin itself.
    # This removes the old arbitrary 0.55 / 0.35 / 0.10 display blend.
    model_score = projected_margin

    # =========================================================
    # UPSET RISK LOGIC
    # =========================================================
    upset_risk = "Low"
    upset_type = "Non"

    if seed_a is not None and seed_b is not None:
        larger_seed_number = max(seed_a, seed_b)

        # Check 1-vs-16 BEFORE the generic 13+ condition.
        if {seed_a, seed_b} == {1, 16}:
            upset_type = "Historic"
        elif {seed_a, seed_b} in [{8, 9}, {4, 5}, {6, 7}]:
            upset_type = "Non"
        elif {seed_a, seed_b} in [{7, 10}, {6, 11}, {5, 12}]:
            upset_type = "Mild"
        elif larger_seed_number >= 13:
            upset_type = "True"
        else:
            upset_type = "Moderate"

        favorite_seed = seed_a if projected_margin > 0 else seed_b
        other_seed = seed_b if projected_margin > 0 else seed_a
        abs_margin = abs(projected_margin)

        # A numerically larger seed being favored is an upset pick.
        if favorite_seed > other_seed:
            upset_risk = "Very High"
        elif abs_margin <= 2:
            upset_risk = "High"
        elif abs_margin <= 6:
            upset_risk = "Medium"
        else:
            upset_risk = "Low"

        if upset_type == "Non":
            upset_risk = "Low"

    else:
        if win_prob < 0.57:
            upset_risk = "High"
        elif win_prob < 0.63:
            upset_risk = "Medium"
        else:
            upset_risk = "Low"

    return {
        "team_a": team_a,
        "team_b": team_b,
        "winner": winner,
        "loser": loser,
        "win_prob": win_prob,
        "projected_margin": spread,
        "signed_projected_margin_a": projected_margin,
        "win_prob_a": win_prob_a,
        "projected_spread_text": f"{winner} -{spread:.1f}",
        "expected_possessions": expected_possessions,
        "pred_A_ortg": features["pred_A_ortg"],
        "pred_B_ortg": features["pred_B_ortg"],
        "model_score": model_score,
        "margin_components": margin_components,
        "raw_edges": {
            "Shooting (points)": features["efg_margin"],
            "Turnovers (points)": features["tov_margin"],
            "Rebounding (points)": features["orb_margin"],
            "Free Throw Pressure (points)": features["ftr_margin"],
            "Four Factors Core (points)": features["four_factor_margin"],
            "V3 Residual Correction (points)": features["v3_residual_correction"],
        },
        "upset_risk": upset_risk,
        "upset_type": upset_type,
    }

# =========================================================
# COMMENTARY ENGINE
# =========================================================

def pct_label(p):
    if p is None:
        return None

    pct = int(round(p * 100))

    if 10 <= pct % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")

    return f"{pct}{suffix} percentile"


def pace_descriptor(p):
    if p is None:
        return "unknown"
    if p >= 0.85:
        return "very fast"
    if p >= 0.65:
        return "fast"
    if p >= 0.35:
        return "average"
    if p >= 0.15:
        return "slow"
    return "very slow"


def label_strength(p):
    if p is None:
        return None
    if p >= 0.85: return "Elite"
    if p >= 0.70: return "Strong"
    if p >= 0.55: return "Above Avg"
    if p >= 0.45: return "Average"
    if p >= 0.30: return "Below Avg"
    return "Poor"




def style_tags(team_name, row, full_df, percentile_tables):
    sections = {
        "Offense": [],
        "Ball Control": [],
        "Rebounding": [],
        "Defense": [],
        "Tempo": [],
        "Concerns": []
    }

    def pct(col, lower=False):
        val = oriented_percentile(
            full_df,
            percentile_tables,
            team_name,
            col,
            lower_is_better=lower
        )
        return val if val is not None else None

    def add(section, text):
        if text not in sections[section]:
            sections[section].append(text)

    # =========================
    # OFFENSE
    # =========================
    p_efg = pct("eFG%")
    if p_efg is not None:
        label = label_strength(p_efg)
        add("Offense", f"{label} efficiency (eFG% {row['eFG%']:.3f}, {pct_label(p_efg)})")

    p_3p = pct("3P%")
    if p_3p is not None:
        label = label_strength(p_3p)
        add("Offense", f"{label} 3pt shooting (3P% {row['3P%']:.3f}, {pct_label(p_3p)})")

    p_3par = pct("3PAr")
    if p_3par is not None:
        if p_3par >= 0.70:
            add("Offense", f"Heavy 3pt reliance (3PAr {row['3PAr']:.3f})")
        elif p_3par <= 0.35:
            add("Offense", f"Paint-focused offense (3PAr {row['3PAr']:.3f})")

    p_ftr = pct("FT/FGA")
    if p_ftr is not None:
        label = label_strength(p_ftr)
        add("Offense", f"{label} free throw pressure (FT/FGA {row['FT/FGA']:.3f}, {pct_label(p_ftr)})")

    # =========================
    # BALL CONTROL
    # =========================
    p_tov = pct("TOV%", lower=True)
    if p_tov is not None:
        label = label_strength(p_tov)
        add("Ball Control", f"{label} ball security (TOV% {row['TOV%']:.3f}, {pct_label(p_tov)})")

    p_ast = pct("AST%")
    if p_ast is not None:
        label = label_strength(p_ast)
        add("Ball Control", f"{label} ball movement (AST% {row['AST%']:.3f}, {pct_label(p_ast)})")

    # =========================
    # REBOUNDING
    # =========================
    p_orb = pct("ORB%")
    if p_orb is not None:
        label = label_strength(p_orb)
        add("Rebounding", f"{label} offensive rebounding (ORB% {row['ORB%']:.3f}, {pct_label(p_orb)})")

    p_trb = pct("TRB%")
    if p_trb is not None:
        label = label_strength(p_trb)
        add("Rebounding", f"{label} total rebounding (TRB% {row['TRB%']:.3f}, {pct_label(p_trb)})")

    # =========================
    # DEFENSE
    # =========================
    p_def_efg = pct("Opp_eFG%", lower=True)
    if p_def_efg is not None:
        label = label_strength(p_def_efg)
        add("Defense", f"{label} shot defense (Opp eFG% {row['Opp_eFG%']:.3f}, {pct_label(p_def_efg)})")

    p_def_tov = pct("Opp_TOV%")
    if p_def_tov is not None:
        label = label_strength(p_def_tov)
        add("Defense", f"{label} turnover creation (Opp TOV% {row['Opp_TOV%']:.3f}, {pct_label(p_def_tov)})")

    p_def_orb = pct("Opp_ORB%", lower=True)
    if p_def_orb is not None:
        label = label_strength(p_def_orb)
        add("Defense", f"{label} defensive rebounding (Opp ORB% {row['Opp_ORB%']:.3f}, {pct_label(p_def_orb)})")

    # =========================
    # TEMPO
    # =========================
    p_pace = pct("Pace")

    if p_pace is not None:
        pct_str = pct_label(p_pace)

        if p_pace >= .90:
            add("Tempo", f"Very fast-paced ({pct_str}, {row['Pace']:.1f} possessions per game)")
        elif p_pace >= 0.70:
            add("Tempo", f"Fast-paced ({pct_str}, {row['Pace']:.1f} possessions per game)")
        elif p_pace >= 0.60:
            add("Tempo", f"Above average pace ({pct_str}, {row['Pace']:.1f} possessions per game)")
        elif p_pace <= 0.10:
            add("Tempo", f"Very slow-paced ({pct_str}, {row['Pace']:.1f} possessions per game)")
        elif p_pace <= 0.30:
            add("Tempo", f"Slow-paced ({pct_str}, {row['Pace']:.1f} possessions per game)")
        elif p_pace <= 0.40:
            add("Tempo", f"Below average pace ({pct_str}, {row['Pace']:.1f} possessions per game)")
        else:
            add("Tempo", f"Average pace ({pct_str}, {row['Pace']:.1f} possessions per game)")

    # =========================
    # CONCERNS
    # =========================
    if p_tov is not None and p_tov <= 0.35:
        add("Concerns", f"Turnover issues (TOV% {row['TOV%']:.3f}, {pct_label(p_tov)})")

    if p_def_orb is not None and p_def_orb <= 0.40:
        add("Concerns", f"Allows offensive rebounds (Opp ORB% {row['Opp_ORB%']:.3f}, {pct_label(p_def_orb)})")

    if p_efg is not None and p_efg <= 0.35:
        add("Concerns", f"Inefficient offense (eFG% {row['eFG%']:.3f}, {pct_label(p_efg)})")

    return {k: v[:3] for k, v in sections.items() if v}

def matchup_tensions(team_a, team_b, A, B, full_df, percentile_tables):
    notes = []

    def pct(team, col, lower=False):
        return oriented_percentile(
            full_df,
            percentile_tables,
            team,
            col,
            lower_is_better=lower
        )

    def fmt(team, col, val, p):
        return f"{team} {col} {val:.3f} ({pct_label(p)})"

    def mismatch(a_p, b_p):
        return a_p is not None and b_p is not None and a_p >= 0.55 and b_p <= 0.45

    def strong_edge(a_p, b_p):
        return a_p is not None and b_p is not None and a_p >= 0.70 and b_p <= 0.30

    # =========================
    # OFFENSE VS DEFENSE
    # =========================
    stat_map = [
        ("eFG%", "Opp_eFG%", True, "shooting efficiency", "should be more efficient scoring"),
        ("TOV%", "Opp_TOV%", False, "ball control", "should take better care of the ball"),
        ("ORB%", "Opp_ORB%", True, "offensive rebounding", "should generate more second chances"),
        ("FT/FGA", "Opp_FT/FGA", True, "free throw pressure", "should get to the line more"),
    ]

    for off_stat, def_stat, lower_def, label, phrase in stat_map:

        lower_off = (off_stat == "TOV%")

        a_off = pct(team_a, off_stat, lower_off)
        b_off = pct(team_b, off_stat, lower_off)

        a_def = pct(team_a, def_stat, lower_def)
        b_def = pct(team_b, def_stat, lower_def)

        if mismatch(a_off, b_def):
            notes.append(
                f"{team_a} {phrase} against {team_b} "
                f"({fmt(team_a, off_stat, A[off_stat], a_off)} vs {fmt(team_b, def_stat, B[def_stat], b_def)})."
            )

        if strong_edge(a_off, b_def):
            notes.append(
                f"{team_a} has a major advantage in {label} "
                f"({fmt(team_a, off_stat, A[off_stat], a_off)} vs {fmt(team_b, def_stat, B[def_stat], b_def)})."
            )

        if mismatch(b_off, a_def):
            notes.append(
                f"{team_b} {phrase} against {team_a} "
                f"({fmt(team_b, off_stat, B[off_stat], b_off)} vs {fmt(team_a, def_stat, A[def_stat], a_def)})."
            )

        if strong_edge(b_off, a_def):
            notes.append(
                f"{team_b} has a major advantage in {label} "
                f"({fmt(team_b, off_stat, B[off_stat], b_off)} vs {fmt(team_a, def_stat, A[def_stat], a_def)})."
            )

    # =========================
    # OFFENSE VS OFFENSE
    # =========================
    pure_stats = [
        ("eFG%", False),
        ("TOV%", True),
        ("ORB%", False),
        ("FT/FGA", False),
        ("AST%", False),
        ("3P%", False),
        ("3PAr", False),
    ]

    for stat, lower in pure_stats:
        a_p = pct(team_a, stat, lower)
        b_p = pct(team_b, stat, lower)

        if a_p is None or b_p is None:
            continue

        if mismatch(a_p, b_p):
            notes.append(
                f"{team_a} has the edge in {stat} "
                f"({fmt(team_a, stat, A[stat], a_p)} vs {fmt(team_b, stat, B[stat], b_p)})."
            )

        if mismatch(b_p, a_p):
            notes.append(
                f"{team_b} has the edge in {stat} "
                f"({fmt(team_b, stat, B[stat], b_p)} vs {fmt(team_a, stat, A[stat], a_p)})."
            )

    # =========================
    # DEFENSE VS DEFENSE
    # =========================
    def_stats = [
        ("Opp_eFG%", True),
        ("Opp_TOV%", False),
        ("Opp_ORB%", True),
        ("Opp_FT/FGA", True),
        ("Opp_3P%", True),
    ]

    for stat, lower in def_stats:
        a_p = pct(team_a, stat, lower)
        b_p = pct(team_b, stat, lower)

        if a_p is None or b_p is None:
            continue

        if mismatch(a_p, b_p):
            notes.append(
                f"{team_a} has the stronger defense in {stat} "
                f"({fmt(team_a, stat, A[stat], a_p)} vs {fmt(team_b, stat, B[stat], b_p)})."
            )

        if mismatch(b_p, a_p):
            notes.append(
                f"{team_b} has the stronger defense in {stat} "
                f"({fmt(team_b, stat, B[stat], b_p)} vs {fmt(team_a, stat, A[stat], a_p)})."
            )

    # =========================
    # PACE
    # =========================
    if "Pace" in A.index and "Pace" in B.index:
        if abs(A["Pace"] - B["Pace"]) >= 2:
            faster = team_a if A["Pace"] > B["Pace"] else team_b
            slower = team_b if faster == team_a else team_a
            notes.append(
                f"{faster} will try to push tempo while {slower} prefers slower play "
                f"({team_a} {A['Pace']:.1f} vs {team_b} {B['Pace']:.1f})."
            )

    # =========================
    # CLEAN + SPACING
    # =========================
    seen = set()
    deduped = []
    for n in notes:
        if n not in seen:
            deduped.append(n + "\n")   # 🔥 adds spacing between each point
            seen.add(n)

    if not deduped:
        deduped.append("No major mismatches detected.\n")

    return deduped[:30]

def game_script(team_a, team_b, A, B, full_df, percentile_tables, matchup_result):
    notes = []

    avg_pace = matchup_result["expected_possessions"]
    pace_pct = float((full_df["Pace"] <= avg_pace).mean())

    if abs(A["Pace"] - B["Pace"]) >= 3.0:
        faster = team_a if A["Pace"] > B["Pace"] else team_b
        slower = team_b if faster == team_a else team_a
        notes.append(f"There is a real tempo clash here. {faster} wants to speed the game up, while {slower} is more comfortable in a slower setting.")

    if pace_pct <= 0.35:
        notes.append("This projects as a lower possession game, which naturally raises volatility and gives the underdog more room to hang around.")
    elif pace_pct >= 0.65:
        notes.append("This projects as a higher possession game, which usually gives the better team more chances to separate.")

    # Variance
    a_3par = oriented_percentile(full_df, percentile_tables, team_a, "3PAr")
    b_3par = oriented_percentile(full_df, percentile_tables, team_b, "3PAr")
    if (a_3par is not None and a_3par >= 0.80) or (b_3par is not None and b_3par >= 0.80):
        notes.append("At least one side is highly dependent on the three, so shot variance could swing this game more than usual.")

    return notes[:15]


def generate_game_drivers(team_a, team_b, A, B, full_df, percentile_tables, winner):
    drivers = []

    def pct(team, col, lower=False):
        return oriented_percentile(
            full_df,
            percentile_tables,
            team,
            col,
            lower_is_better=lower
        )

    def add(text):
        if text not in drivers:
            drivers.append(text)

    # identify winner/loser
    w = winner
    l = team_b if winner == team_a else team_a

    # =========================
    # SHOOTING (LESS STRICT)
    # =========================
    w_efg = pct(w, "eFG%")
    l_def_efg = pct(l, "Opp_eFG%", lower=True)

    if w_efg and l_def_efg:
        if w_efg > l_def_efg + 0.10:
            add(f"{w} should have a clear scoring efficiency advantage in this matchup.")
        elif w_efg > 0.60:
            add(f"{w} brings strong shooting efficiency into this game.")

    # =========================
    # TURNOVERS (LESS STRICT)
    # =========================
    w_tov = pct(w, "TOV%", lower=True)
    l_def_tov = pct(l, "Opp_TOV%")

    if w_tov and l_def_tov:
        if w_tov > l_def_tov + 0.15:
            add(f"{w} should control the game with superior ball security.")
        elif w_tov > 0.70:
            add(f"{w} is likely to take care of the ball and limit mistakes.")

    # =========================
    # REBOUNDING
    # =========================
    w_orb = pct(w, "ORB%")
    l_def_orb = pct(l, "Opp_ORB%", lower=True)

    if w_orb and l_def_orb:
        if w_orb > l_def_orb + 0.10:
            add(f"{w} may create extra possessions through offensive rebounding.")

    # =========================
    # OVERALL EFFICIENCY (NEW)
    # =========================
    w_ortg = pct(w, "ORtg")
    l_ortg = pct(l, "ORtg")

    if w_ortg and l_ortg and w_ortg > l_ortg:
        add(f"{w} holds the stronger overall offensive profile in this matchup.")

    # =========================
    # TEMPO
    # =========================
    w_pace = pct(w, "Pace")
    l_pace = pct(l, "Pace")

    if w_pace and l_pace:
        if w_pace > 0.65 and l_pace > 0.65:
            add(f"A faster game benefits {w} by increasing possession volume.")
        elif w_pace < 0.35 and l_pace < 0.35:
            add(f"A slower game reduces possessions and increases variance.")

    # =========================
    # FALLBACK (guarantee at least 2–3 lines)
    # =========================
    if len(drivers) < 2:
        add(f"{w} projects as the more complete team across key efficiency metrics.")

    if len(drivers) < 3:
        add(f"{w} is better positioned to dictate how this game is played.")

    return drivers[:5]

def build_commentary(team_a, team_b, full_df, baselines, percentile_tables, matchup_result):
    A = full_df.loc[full_df["Team"] == team_a].iloc[0]
    B = full_df.loc[full_df["Team"] == team_b].iloc[0]

    winner = matchup_result["winner"]
    loser = matchup_result["loser"]

    winner_row = A if winner == team_a else B
    loser_row = B if winner == team_a else A

    winner_tags = style_tags(winner, winner_row, full_df, percentile_tables)
    loser_tags = style_tags(loser, loser_row, full_df, percentile_tables)

    tensions = matchup_tensions(team_a, team_b, A, B, full_df, percentile_tables)
    script = game_script(team_a, team_b, A, B, full_df, percentile_tables, matchup_result)

    # =========================================================
    # 🔥 NEW GAME DRIVERS (REPLACES TOP EDGES)
    # =========================================================
    top_edge_text = generate_game_drivers(
        team_a,
        team_b,
        A,
        B,
        full_df,
        percentile_tables,
        winner
    )
    # =========================================================
    # SIZE PROFILE
    # =========================================================
    phys_winner = get_team_physical_profile(winner)
    phys_loser = get_team_physical_profile(loser)

    physical_notes = []

    def build_block(team, phys):
        block = []

        block.append(
            f"{team} averages {phys['avg_height_str']} "
            f"({phys['avg_height_inches']:.1f} in) "
            f"across its {phys['rotation_size']} rotation players."
        )

        parts = []
        if phys.get("guards"):
            parts.append(f"G: {phys['guards']}")
        if phys.get("forwards"):
            parts.append(f"F: {phys['forwards']}")
        if phys.get("centers"):
            parts.append(f"C: {phys['centers']}")

        if parts:
            block.append("Position Size → " + " | ".join(parts))

        return block

    if phys_winner:
        physical_notes.extend(build_block(winner, phys_winner))

    if phys_loser:
        physical_notes.extend(build_block(loser, phys_loser))

    return {
        "winner_tags": winner_tags,
        "loser_tags": loser_tags,
        "tensions": tensions,
        "game_script": script,
        "top_edges": top_edge_text[:5],
        "physical_notes": physical_notes
    }

# =========================================================
# DISPLAY
# =========================================================
def print_team_profile(team_name, row, full_df, percentile_tables):
    print(f"\n{team_name} Profile")
    print("-" * 60)

    # =========================================================
    # 🔥 KEY PLAYERS (ADDED)
    # =========================================================
    phys = get_team_physical_profile(team_name)
    if phys and phys.get("stars"):
        print("Key Players:")
        for p in phys["stars"]:
            print(f"  - {p}")

    def pct(col, lower=False):
        return pct_label(
            oriented_percentile(full_df, percentile_tables, team_name, col, lower_is_better=lower)
        )

    print("\nRatings:")
    print(f"  Offensive Rating: {row['ORtg']:.1f} ({pct('ORtg')}) (scores {row['ORtg']:.1f} points per 100 possessions)")
    print(f"  Defensive Rating: {row['Opp_ORtg']:.1f} ({pct('Opp_ORtg', True)}) (allows {row['Opp_ORtg']:.1f} points per 100 possessions)")

    print("\nStrength:")
    if "SRS" in row.index:
        print(f"  SRS: {row['SRS']:.2f} ({pct('SRS')})")
    if "SOS" in row.index:
        print(f"  Strength of Schedule: {row['SOS']:.2f} ({pct('SOS')})")

    print("\nShooting:")
    print(f"  eFG%: {row['eFG%']:.3f} ({pct('eFG%')})")
    print(f"  3P%: {row['3P%']:.3f} ({pct('3P%')})")
    print(f"  3PA: {row['3PA']:.1f} per game ({pct('3PA') if '3PA' in percentile_tables else ''})")
    print(f"  3P Attempt Rate: {row['3PAr']:.3f} ({pct('3PAr')})")
    print(f"  FT/FGA: {row['FT/FGA']:.3f} ({pct('FT/FGA')})")

    print("\nBall Control:")
    print(f"  TOV%: {row['TOV%']:.3f} ({pct('TOV%', True)})")
    print(f"  Turnovers: {row['TOV_pg']:.1f} per game")
    print(f"  AST%: {row['AST%']:.3f} ({pct('AST%')})")
    print(f"  Assists: {row['AST_pg']:.1f} per game ({pct('AST_pg')})")

    print("\nRebounding:")
    print(f"  ORB%: {row['ORB%']:.3f} ({pct('ORB%')})")
    print(f"  TRB%: {row['TRB%']:.3f} ({pct('TRB%')})")
    print(f"  Offensive Rebounds: {row['ORB_pg']:.1f} per game ({pct('ORB_pg')})")
    print(f"  Total Rebounds: {row['TRB_pg']:.1f} per game ({pct('TRB_pg')})")

    print("\nDefense:")
    print(f"  Opp eFG%: {row['Opp_eFG%']:.3f} ({pct('Opp_eFG%', True)})")
    print(f"  Opp TOV%: {row['Opp_TOV%']:.3f} ({pct('Opp_TOV%')})")
    print(f"  Opp Turnovers Forced: {row['Opp_TOV_pg']:.1f} per game")
    print(f"  Opp ORB%: {row['Opp_ORB%']:.3f} ({pct('Opp_ORB%', True)})")
    print(f"  Opp FT/FGA: {row['Opp_FT/FGA']:.3f} ({pct('Opp_FT/FGA', True)})")
    print(f"  Opp 3P%: {row['Opp_3P%']:.3f} ({pct('Opp_3P%', True)})")
    print(f"  Opp 3PA: {row['Opp_3PA']:.1f} per game")
    print(f"  Opp 3PAr: {row['Opp_3PAr']:.3f} ({pct('Opp_3PAr', True)})")

    print("\nTempo:")
    print(f"  Pace: {row['Pace']:.1f} possessions per game ({pct('Pace')})")
    


def print_sections(tag_dict):
    for section, items in tag_dict.items():
        print(f"\n{section}:")
        for item in items:
            print(f"  - {item}")
            
def print_matchup_report(team_a, team_b, result, commentary, profile_a, profile_b):
    print("\n" + "=" * 70)
    print(f"{team_a} vs {team_b}")
    print("=" * 70)

    print(f"Prediction: {result['winner']}")
    print(f"Win Probability: {result['win_prob']*100:.1f}%")
    print(f"Projected Spread: {result['projected_spread_text']}")
    print(f"Expected Possessions: {result['expected_possessions']:.1f}")
   

    print("\nTop Edges")
    print("-" * 70)
    for edge in commentary["top_edges"]:
        print(f"- {edge}")

    print("\nSize Profile")
    print("-" * 70)
    if profile_a and profile_b:
        print(f"- {team_a} averages {profile_a['avg_height_str']} across its {profile_a['rotation_size']} rotation players.")
        print(f"- Position Size → G: {profile_a['guards']} | F: {profile_a['forwards']} | C: {profile_a['centers']}")
        print(f"- {team_b} averages {profile_b['avg_height_str']} across its {profile_b['rotation_size']} rotation players.")
        print(f"- Position Size → G: {profile_b['guards']} | F: {profile_b['forwards']} | C: {profile_b['centers']}")
    else:
        print("- Physical profile unavailable for one or both teams.")

    print(f"\n{team_a} Profile")
    print("-" * 60)

    if profile_a and profile_a.get("stars"):
        print("Key Players:")
        for s in profile_a["stars"]:
            print(f"  - {s}")
        print()

    print("\nKey Strengths")
    print("-" * 70)
    print_sections(commentary["winner_tags"] if result["winner"] == team_a else commentary["loser_tags"])

    print("\n" + "=" * 70)

    print(f"\n{team_b} Profile")
    print("-" * 60)

    if profile_b and profile_b.get("stars"):
        print("Key Players:")
        for s in profile_b["stars"]:
            print(f"  - {s}")
        print()

    print("\nKey Strengths")
    print("-" * 70)
    print_sections(commentary["winner_tags"] if result["winner"] == team_b else commentary["loser_tags"])

    print("\n" + "=" * 70)

    print("\nMatchup Notes")
    print("-" * 70)
    for note in commentary["tensions"]:
        print(f"- {note}")

    print("\nGame Script")
    print("-" * 70)
    for note in commentary["game_script"]:
        print(f"- {note}")

    print("\n" + "=" * 70 + "\n")


# =========================================================
# TEAM LOOKUP HELPERS
# =========================================================

def find_team_name(user_input, df):
    lookup = canonical_lookup_name(user_input)
    teams = df["Team"].tolist()

    exact = [t for t in teams if canonical_lookup_name(t) == lookup]
    if exact:
        return exact[0]

    contains = [t for t in teams if lookup in canonical_lookup_name(t)]
    if len(contains) == 1:
        return contains[0]
    elif len(contains) > 1:
        raise ValueError(f"Multiple teams matched '{user_input}': {contains[:10]}")
    else:
        raise ValueError(f"No team matched '{user_input}'.")


# =========================================================
# MAIN
# =========================================================

def main():
    full_df, tourney_df = load_and_prepare_data(
        TEAM_BASIC_CSV,
        OPP_BASIC_CSV,
        TEAM_ADV_CSV,
        OPP_ADV_CSV
    )

    baselines, percentile_tables = build_baselines(full_df)

    print(f"\nLoaded {len(full_df)} total teams.")
    print(f"Detected {len(tourney_df)} tournament tagged teams.\n")

    while True:
        team_a = input("Enter Team A (or 'quit'): ").strip()
        if team_a.lower() == "quit":
            break

        team_b = input("Enter Team B: ").strip()

        

        try:
            # 🔥 FIX TEAM NAME MATCHING (you already wrote this function)
            team_a = find_team_name(team_a, full_df)
            team_b = find_team_name(team_b, full_df)

            # 🔥 THIS IS YOUR ACTUAL MODEL FUNCTION
            result = compute_matchup(
                team_a,
                team_b,
                full_df,
                baselines,
                percentile_tables
                
            )

            # 🔥 BUILD COMMENTARY (YOU WERE MISSING THIS)
            commentary = build_commentary(
                team_a,
                team_b,
                full_df,
                baselines,
                percentile_tables,
                result
            )

            # 🔥 PROFILES
            profile_a = get_team_physical_profile(team_a)
            profile_b = get_team_physical_profile(team_b)

            # 🔥 PRINT
            print_matchup_report(
                team_a,
                team_b,
                result,
                commentary,
                profile_a,
                profile_b
            )

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()