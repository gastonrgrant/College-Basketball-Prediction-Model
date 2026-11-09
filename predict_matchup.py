import html
import pandas as pd
import numpy as np
import re
import requests
import webbrowser
from bs4 import BeautifulSoup, Comment
from functools import lru_cache
from datetime import date
from io import StringIO
from pathlib import Path
import hashlib
import time


# =========================================================
# CONFIG
# =========================================================
# V5 is the live model. V4.2 is archive/model_v4_2.py and does not apply home court or schedule.
# This script asks where the game is played, adds the fitted home-court shift,
# and scores the Four Factors with separate offense and defense weights.
# Schedule is a separate fitted term. Neutral games do not invent an arena.

# =========================================================
# LIVE SCRAPING URLS (replaces CSVs)
# =========================================================

# Set this to a season-ending year (for example 2027) only when you want to
# force a specific live season. Leave as None for normal use.
SEASON_OVERRIDE = None

# Historical coefficients currently embedded in this file were learned from
# pre-game NCAA observations spanning these seasons. Live team inputs are
# refreshed separately every time the program runs.
TRAINING_START_SEASON = 2003
TRAINING_END_SEASON = 2026
ACTIVE_SEASON_YEAR = None
DATA_SOURCE_URL = None

# Each stat is games / (games + K) this season and the rest last season.
# K = 5 was chosen on 2016-2021 and tested on 2022-2026 (training/test_season_blend.py).
SEASON_BLEND_K = 5.0
CURRENT_RAW_DF = None
PRIOR_SEASON_DF = None
BLEND_NOTES = {}

# Sports-Reference blocks clients that make more than about 20 requests a minute.
SR_REQUEST_GAP_SECONDS = 3.1
_SR_LAST_REQUEST = [0.0]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def infer_current_ncaa_season(today=None):
    """Return the NCAA season-ending year that should be live today.

    Jan-Jun belongs to the season ending in the current calendar year.
    Jul-Dec first attempts the upcoming season (year + 1). If Sports-Reference
    has not published that season yet, the loader automatically falls back to
    the most recent available season.
    """
    today = today or date.today()
    return today.year if today.month <= 6 else today.year + 1


def build_live_urls(season_year):
    base = f"https://www.sports-reference.com/cbb/seasons/men/{int(season_year)}"
    return {
        "team_basic": f"{base}-school-stats.html?per_game=1",
        "opp_basic": f"{base}-opponent-stats.html?per_game=1",
        "team_adv": f"{base}-advanced-school-stats.html",
        "opp_adv": f"{base}-advanced-opponent-stats.html",
    }


def live_season_candidates():
    if SEASON_OVERRIDE is not None:
        return [int(SEASON_OVERRIDE)]

    inferred = infer_current_ncaa_season()
    # In the summer, the upcoming season page may not exist yet. Walking back
    # one season keeps the program usable without silently hard-coding a year.
    return [inferred, inferred - 1]


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
            # pandas 3 removed errors="ignore". Keep the old behavior:
            # convert the column when every value parses, otherwise leave it.
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df

def sr_get_text(url, persist=False, cache_key=None):
    """Fetch a Sports-Reference page, spaced out to stay under their rate limit.

    persist keeps the page in .cache/ for pages that no longer change, such as
    a finished season. cache_key separates copies of a page that keeps growing.
    """
    cached = CACHE_DIR / f"{hashlib.sha1((cache_key or url).encode()).hexdigest()}.html"
    if persist and cached.exists():
        return cached.read_text()
    wait = SR_REQUEST_GAP_SECONDS - (time.monotonic() - _SR_LAST_REQUEST[0])
    if wait > 0:
        time.sleep(wait)
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    finally:
        _SR_LAST_REQUEST[0] = time.monotonic()
    if res.status_code != 200:
        raise ValueError(f"HTTP {res.status_code} at {url}")
    if persist:
        CACHE_DIR.mkdir(exist_ok=True)
        cached.write_text(res.text)
    return res.text


def scrape_table(url, persist=False):
    soup = BeautifulSoup(sr_get_text(url, persist=persist), "lxml")

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
            df = pd.read_html(StringIO(str(t)))[0]

            # 🔥 FIX: flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)

            df.columns = [str(c).strip() for c in df.columns]

            return df

    raise ValueError(f"Table not found at {url}")

# =========================================================
# LOAD + MERGE
# =========================================================

def scrape_season_tables(season_year, persist=False):
    """Return the four Sports-Reference season tables once games have been played."""
    urls = build_live_urls(season_year)
    tables = [
        scrape_table(urls[key], persist=persist)
        for key in ("team_basic", "opp_basic", "team_adv", "opp_adv")
    ]
    # A real Division-I season table should contain many teams. This
    # avoids accepting an empty/placeholder upcoming-season page.
    if min(len(table) for table in tables) < 50:
        raise ValueError("season tables are not populated yet")
    # Before opening night the page lists every school with blank stats.
    team_basic = tables[0]
    if "G" not in team_basic.columns:
        raise ValueError("season table has no games column")
    played = pd.to_numeric(team_basic["G"], errors="coerce")
    if int((played > 0).sum()) < 50:
        raise ValueError("no games have been played yet")
    return tables, urls["team_basic"]


def load_and_prepare_data(team_basic_csv, opp_basic_csv, team_adv_csv, opp_adv_csv):
    global ACTIVE_SEASON_YEAR, DATA_SOURCE_URL, CURRENT_RAW_DF, PRIOR_SEASON_DF

    live_error_messages = []
    tables = None

    # Try the season appropriate for today's date first. During the summer,
    # Sports-Reference may not have the upcoming season page yet, so try the
    # previous season automatically before using local CSVs.
    for season_year in live_season_candidates():
        try:
            tables, source_url = scrape_season_tables(season_year)
        except Exception as exc:
            live_error_messages.append(f"{season_year}: {exc}")
            continue
        ACTIVE_SEASON_YEAR = int(season_year)
        DATA_SOURCE_URL = source_url
        for message in live_error_messages:
            print(f"Skipped {message}.")
        print(
            f"Using LIVE Sports-Reference {ACTIVE_SEASON_YEAR} season data "
            f"(retrieved {date.today().isoformat()})."
        )
        break

    CURRENT_RAW_DF = None
    PRIOR_SEASON_DF = None
    if tables is None:
        ACTIVE_SEASON_YEAR = None
        DATA_SOURCE_URL = None
        print("Live scrape failed — falling back to local CSV files.")
        if live_error_messages:
            print("Live attempts: " + " | ".join(live_error_messages))
        print(
            "WARNING: local CSVs may be stale. For a current prediction, "
            "verify they were downloaded for the current season.\n"
        )
        tables = [pd.read_csv(path) for path in (team_basic_csv, opp_basic_csv, team_adv_csv, opp_adv_csv)]
        return prepare_season_tables(*tables)

    full_df, tourney_df = prepare_season_tables(*tables)
    CURRENT_RAW_DF = full_df.copy()
    try:
        prior_tables, _ = scrape_season_tables(ACTIVE_SEASON_YEAR - 1, persist=True)
        PRIOR_SEASON_DF, _ = prepare_season_tables(*prior_tables)
    except Exception as exc:
        print(f"Could not load the {ACTIVE_SEASON_YEAR - 1} season to blend with: {exc}")
    if PRIOR_SEASON_DF is not None and "G" in full_df.columns:
        full_df = blend_with_prior_season(full_df, PRIOR_SEASON_DF)
        typical = float(full_df["G"].median())
        weight = typical / (typical + SEASON_BLEND_K)
        print(
            f"Blending with the {_season_name(ACTIVE_SEASON_YEAR - 1)} season: "
            f"the typical team has played {typical:.0f} games, so this season counts "
            f"{weight:.0%} and last season {1 - weight:.0%}."
        )
    print()
    tourney_df = full_df[full_df["Is_Tourney_Team"]].copy()
    return full_df, tourney_df


def blend_columns(full_df, prior_df):
    numeric = full_df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric if col in prior_df.columns and col != "G"]


def season_weight(games):
    games = 0.0 if pd.isna(games) else float(games)
    return games / (games + SEASON_BLEND_K)


def blend_with_prior_season(full_df, prior_df):
    """Blend every team's stats with its own final stats from last season."""
    cols = blend_columns(full_df, prior_df)
    prior = prior_df.drop_duplicates("Team").set_index("Team").reindex(full_df["Team"])
    current = full_df[cols].to_numpy(dtype=float)
    last = prior[cols].to_numpy(dtype=float)
    current, last = np.where(np.isnan(current), last, current), np.where(np.isnan(last), current, last)
    weight = full_df["G"].map(season_weight).to_numpy()[:, None]
    out = full_df.copy()
    out[cols] = weight * current + (1.0 - weight) * last
    return out


def prepare_season_tables(team_basic, opp_basic, team_adv, opp_adv):
    team_basic = clean_sportsref_df(team_basic)
    games_played = None
    if "G" in team_basic.columns:
        games_played = pd.DataFrame({
            "Team": team_basic["Team"],
            "G": pd.to_numeric(team_basic["G"], errors="coerce"),
        }).drop_duplicates("Team")
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
    if games_played is not None:
        merged = merged.merge(games_played, on="Team", how="left")

    # Canonical lookup key
    merged["LookupKey"] = merged["Team"].apply(canonical_lookup_name)

    full_df = merged.copy()
    tourney_df = merged[merged["Is_Tourney_Team"]].copy()

    return full_df, tourney_df


# =========================================================
# ROSTER STARTING POINT
# =========================================================
# Last season's team stats, adjusted for the players in this season's rotation.
# Transfers bring their own stats from their old school. This adjustment is not
# in the Kaggle test, because Kaggle has no player-level data.

PLAYER_COUNT_STATS = ("mp", "fg", "fga", "fg3", "ft", "fta", "orb", "tov")
OFFENSE_ROSTER_COLUMNS = (("eFG%", "efg"), ("TOV%", "tov"), ("ORB%", "orb"), ("FT/FGA", "ftr"))
DEFENSE_ROSTER_COLUMNS = ("Opp_eFG%", "Opp_TOV%", "Opp_ORB%", "Opp_FT/FGA")


def _find_table(soup, table_id):
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if f'id="{table_id}"' in comment:
            return BeautifulSoup(comment, "lxml").find("table", id=table_id)
    return None


def _cell_number(cell):
    text = cell.text.strip() if cell is not None else ""
    try:
        return float(text) if text else 0.0
    except ValueError:
        return 0.0


def _player_lines(soup, table_id, stats):
    """{player page link: line} from a team page's player table."""
    table = _find_table(soup, table_id)
    lines = {}
    if table is None or table.find("tbody") is None:
        return lines
    for row in table.find("tbody").find_all("tr"):
        link = row.find("a", href=re.compile(r"^/cbb/players/"))
        if link is None:
            continue
        line = {stat: _cell_number(row.find(attrs={"data-stat": stat})) for stat in stats}
        line["name"] = link.text.strip()
        lines[link["href"]] = line
    return lines


def _season_label(year):
    return f"{int(year) - 1}-{str(int(year))[-2:]}"


def player_previous_season(href, year):
    """A player's totals for the season before year, and the school he played for."""
    label = _season_label(int(year) - 1)
    try:
        page = sr_get_text(
            f"https://www.sports-reference.com{href}",
            persist=True,
            cache_key=f"{href}|{label}",
        )
    except Exception:
        return None
    soup = BeautifulSoup(page, "lxml")
    totals = _find_table(soup, "players_totals")
    if totals is None or totals.find("tbody") is None:
        return None
    line = {stat: 0.0 for stat in PLAYER_COUNT_STATS}
    schools = {}
    for row in totals.find("tbody").find_all("tr"):
        season = row.find(attrs={"data-stat": "year_id"})
        if season is None or season.text.strip() != label:
            continue
        for stat in PLAYER_COUNT_STATS:
            line[stat] += _cell_number(row.find(attrs={"data-stat": stat}))
        school = row.find(attrs={"data-stat": "team_name_abbr"})
        if school is not None:
            name = school.text.strip()
            schools[name] = schools.get(name, 0.0) + _cell_number(row.find(attrs={"data-stat": "mp"}))
    if line["mp"] <= 0:
        return None
    orb_total = orb_minutes = 0.0
    advanced = _find_table(soup, "players_advanced")
    if advanced is not None and advanced.find("tbody") is not None:
        for row in advanced.find("tbody").find_all("tr"):
            season = row.find(attrs={"data-stat": "year_id"})
            if season is None or season.text.strip() != label:
                continue
            minutes = _cell_number(row.find(attrs={"data-stat": "mp"}))
            orb_total += minutes * _cell_number(row.find(attrs={"data-stat": "orb_pct"}))
            orb_minutes += minutes
    line["orb_pct"] = orb_total / orb_minutes if orb_minutes else np.nan
    line["school"] = max(schools, key=schools.get) if schools else None
    return line


def _lineup_rates(entries):
    """Team rates, as fractions, from (minutes weight, season line) pairs."""
    entries = [(weight, line) for weight, line in entries if weight > 0 and line["mp"] > 0]
    if not entries:
        return None
    sums = {
        stat: sum(weight * line[stat] / line["mp"] for weight, line in entries)
        for stat in ("fg", "fga", "fg3", "ft", "fta", "orb", "tov")
    }
    if sums["fga"] <= 0:
        return None
    rebounders = [(weight, line["orb_pct"]) for weight, line in entries if not np.isnan(line["orb_pct"])]
    rebound_weight = sum(weight for weight, _ in rebounders)
    # Player ORB% is his share of available rebounds while on the floor; five play at once.
    orb = 5.0 * sum(weight * value for weight, value in rebounders) / rebound_weight / 100.0 if rebound_weight else np.nan
    return {
        "efg": (sums["fg"] + 0.5 * sums["fg3"]) / sums["fga"],
        "tov": sums["tov"] / (sums["fga"] + 0.44 * sums["fta"] + sums["tov"]),
        "ftr": sums["ft"] / sums["fga"],
        "orb": orb,
    }


def _prior_team_row(prior_df, school):
    if not school:
        return None
    match = prior_df.loc[prior_df["LookupKey"] == canonical_lookup_name(school)]
    return match.iloc[0] if not match.empty else None


def roster_starting_point(team, prior_df):
    """Last season's stats for team, adjusted for this season's rotation.

    Offense moves by how differently this season's players shot, protected the
    ball, rebounded, and drew fouls last season. Defense averages the defenses
    those players were on last season. Players with no previous season count as
    unchanged from the team's last season.
    """
    year = ACTIVE_SEASON_YEAR
    prior = _prior_team_row(prior_df, team)
    if year is None or prior is None:
        return None
    now_page = team_page_html(team, year)
    last_page = team_page_html(team, int(year) - 1)
    if now_page is None or last_page is None:
        return None
    playing = _player_lines(BeautifulSoup(now_page, "lxml"), "players_totals", ("mp",))
    last_soup = BeautifulSoup(last_page, "lxml")
    last_lines = _player_lines(last_soup, "players_totals", PLAYER_COUNT_STATS)
    last_advanced = _player_lines(last_soup, "players_advanced", ("mp", "orb_pct"))
    for href, line in last_lines.items():
        advanced = last_advanced.get(href)
        line["orb_pct"] = advanced["orb_pct"] if advanced and advanced["mp"] > 0 else np.nan
        line["school"] = team

    minutes_now = sum(line["mp"] for line in playing.values())
    baseline = _lineup_rates([(line["mp"], line) for line in last_lines.values()])
    if minutes_now <= 0 or baseline is None:
        return None

    covered, defenses, transfers = [], [], []
    for href, line in playing.items():
        if line["mp"] <= 0:
            continue
        history = last_lines.get(href)
        if history is None:
            history = player_previous_season(href, year)
        school_row = prior
        if history is not None:
            covered.append((line["mp"], history))
            if history["school"] and canonical_lookup_name(history["school"]) != canonical_lookup_name(team):
                transfers.append((line["name"], history["school"], line["mp"] / minutes_now))
                school_row = _prior_team_row(prior_df, history["school"])
                if school_row is None:
                    school_row = prior
        defenses.append((line["mp"], school_row))

    covered_share = sum(weight for weight, _ in covered) / minutes_now
    lineup = _lineup_rates(covered)
    values = {}
    for col, key in OFFENSE_ROSTER_COLUMNS:
        if lineup is None or col not in prior.index or pd.isna(prior[col]):
            continue
        if np.isnan(lineup[key]) or np.isnan(baseline[key]):
            continue
        change = covered_share * (lineup[key] - baseline[key])
        scale = 100.0 if abs(float(prior[col])) > 1.5 else 1.0
        values[col] = float(prior[col]) + change * scale
    total = sum(weight for weight, _ in defenses)
    for col in DEFENSE_ROSTER_COLUMNS:
        if col in prior.index and total > 0:
            values[col] = sum(weight * float(row[col]) for weight, row in defenses) / total
    return {"values": values, "transfers": transfers, "covered_share": covered_share}


def apply_roster_starting_point(full_df, team):
    """Re-blend team's Four Factors against its roster-adjusted last season."""
    if team in BLEND_NOTES or CURRENT_RAW_DF is None or PRIOR_SEASON_DF is None:
        return
    raw = CURRENT_RAW_DF.loc[CURRENT_RAW_DF["Team"] == team]
    rows = full_df.index[full_df["Team"] == team]
    if raw.empty or len(rows) == 0 or "G" not in raw.columns:
        return
    raw = raw.iloc[0]
    has_prior = _prior_team_row(PRIOR_SEASON_DF, team) is not None
    weight = season_weight(raw["G"]) if has_prior else 1.0
    note = {"games": raw["G"], "weight": weight, "has_prior": has_prior, "transfers": []}
    start = roster_starting_point(team, PRIOR_SEASON_DF) if has_prior and weight < 1.0 else None
    if start:
        for col, value in start["values"].items():
            current = raw[col] if col in raw.index and pd.notna(raw[col]) else value
            full_df.loc[rows, col] = weight * float(current) + (1.0 - weight) * value
        note["transfers"] = start["transfers"]
        note["covered_share"] = start["covered_share"]
    BLEND_NOTES[team] = note


def blend_note(team):
    """One sentence on how much of team's numbers come from this season."""
    note = BLEND_NOTES.get(team)
    if not note or ACTIVE_SEASON_YEAR is None:
        return ""
    if not note["has_prior"]:
        return f"{team} has no previous Division I season, so its numbers are this season only."
    games = 0 if pd.isna(note["games"]) else int(note["games"])
    text = (
        f"{team} has played {games} game{'s' if games != 1 else ''}, so this season counts "
        f"{note['weight']:.0%} and last season {1 - note['weight']:.0%}."
    )
    transfers = sorted(note["transfers"], key=lambda item: item[2], reverse=True)
    if transfers:
        names = ", ".join(f"{name} ({school})" for name, school, _ in transfers[:3])
        text += f" Last season's numbers include transfers {names}."
    return text


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

_TEAM_STANDINGS = {}


def _standing_key(team_name):
    year = ACTIVE_SEASON_YEAR or infer_current_ncaa_season()
    return (str(team_name), int(year))


def _parse_team_standing(html_text):
    """Read the AP rank and NCAA seed from a Sports-Reference team page."""
    ap_rank = None
    ap_is_final = False
    rank_match = re.search(
        r"Rank:</strong>\s*(\d+)(?:st|nd|rd|th)?\s+in the\s*(?:<a[^>]*>)?\s*([^<]*?)\s*AP Poll",
        html_text,
        flags=re.IGNORECASE,
    )
    if rank_match:
        ap_rank = int(rank_match.group(1))
        # In-season labels carry the poll date, e.g. "Oct 19th AP Poll".
        ap_is_final = "final" in rank_match.group(2).lower()
    seed = None
    seed_match = re.search(r"\(#(\d+)\s+seed\b", html_text, flags=re.IGNORECASE)
    if seed_match:
        seed = int(seed_match.group(1))
    return {"ap_rank": ap_rank, "ap_is_final": ap_is_final, "seed": seed}


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


def build_team_slug(name):
    if name in TEAM_SLUG_MAP:
        return TEAM_SLUG_MAP[name]
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[.\']", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name


@lru_cache(maxsize=None)
def team_page_html(team_name, year):
    """Sports-Reference team season page, or None if no slug matches."""
    persist = ACTIVE_SEASON_YEAR is not None and int(year) < int(ACTIVE_SEASON_YEAR)
    names = [
        team_name,
        team_name.replace("UCF", "Central Florida"),
        team_name.replace("USC", "Southern California"),
        team_name.replace("Ole Miss", "Mississippi"),
    ]
    tried = set()
    for name in names:
        slug = build_team_slug(name)
        if slug in tried:
            continue
        tried.add(slug)
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{int(year)}.html"
        try:
            return sr_get_text(url, persist=persist)
        except Exception:
            continue
    return None


@lru_cache(maxsize=None)
def get_team_physical_profile(team_name, year=None, min_mp=10.0):
    # Use the same live season as the team-stat model unless explicitly forced.
    if year is None:
        year = ACTIVE_SEASON_YEAR or infer_current_ncaa_season()

    # 🔥 FIXED NAME CLEANING (handles Jr, Sr, etc.)
    def clean_name(n):
        n = re.sub(r"\s+", " ", n.strip())
        n = re.sub(r"\b(Jr|Sr|II|III|IV)\b\.?", "", n)
        return n.strip()

    page = team_page_html(team_name, year)
    if page is None:
        return None

    _TEAM_STANDINGS[_standing_key(team_name)] = _parse_team_standing(page)

    soup = BeautifulSoup(page, "lxml")

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
    players = []

    for n in minutes:
        if n not in heights:
            continue

        h_in = to_inches(heights[n])
        if h_in is None:
            continue

        h_vals.append(h_in)
        weights.append(minutes[n])

        pos = positions.get(n, "?")
        primary = re.split(r"[-/]", str(pos).strip())[0].upper()
        if primary.startswith("G"):
            primary = "G"
        elif primary.startswith("F"):
            primary = "F"
        elif primary.startswith("C"):
            primary = "C"
        else:
            primary = None
        if primary in pos_groups:
            pos_groups[primary].append(h_in)

        # star logic (unchanged)
        if minutes[n] >= 25 or stats[n]["pts"] >= 12:
            s = stats[n]
            stars.append(
                f"{n} ({pos}) "
                f"{s['pts']:.1f} ppg, {s['ast']:.1f} ast, {s['trb']:.1f} reb, "
                f"{s['fg']:.3f} fg%, {s['fg3']:.3f} 3p%, {s['fg3a']:.1f} 3PA"
            )
            players.append({
                "name": n,
                "pos": pos,
                "pts": s["pts"],
                "ast": s["ast"],
                "trb": s["trb"],
                "fg": s["fg"],
                "fg3": s["fg3"],
                "fg3a": s["fg3a"],
            })

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
        "stars": stars[:6],
        "players": players[:6],
    }
# =========================================================
# MATCHUP MODEL — V4.1 HISTORICALLY TRAINED OLIVER HYBRID
# =========================================================
# IMPORTANT:
# Dean Oliver's Four Factors are the trained core. Net rating, three-point
# attempt rate, and pace were left at zero after the V5 holdout.
# All prediction-time coefficients are embedded below.


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
    # Refit on pre-2026 games. Offense and defense keep separate weights.
    "eFG_Diff": 75.27342715920912,
    "Def_eFG_Adv": 58.11817300325095,
    "TOV_Adv": 88.11389453132519,
    "ForceTOV_Adv": 78.51270993680325,
    "ORB_Diff": 32.53647172486484,
    "DefORB_Adv": 32.72169243839539,
    "FTR_Diff": 7.789341293906361,
    "DefFTR_Adv": 9.2455258147597,
}

# These secondary V3 signals were trained ONLY to explain what the Four
# Factors missed. Their coefficients are intentionally tiny relative to the
# Four-Factor core, keeping Dean Oliver's framework central to the model.
# The v5 holdout did not use these. They stay at zero so the tested model is the one that runs.
V3_RESIDUAL_POINT_COEFS = {
    "NetRtg_Diff": 0.0,
    "P3Ar_Diff": 0.0,
    "Pace_Diff": 0.0,
}

# Forward-held-out calibration of predicted margin -> win probability.
# logit(P[Team A wins]) = WIN_PROB_LOGIT_SLOPE * projected_margin
WIN_PROB_LOGIT_SLOPE = 0.14898045394540615

# Same fit as the Four Factor slopes. +1 is team A at home, 0 is neutral, -1 is team A away.
HOME_COURT_POINTS = 3.6737017735691313
# Points for a one-standard-deviation gap in strength of schedule, team A minus team B.
SCHEDULE_Z_POINTS = 1.8889355683185807

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

    contrib = features["factor_point_contrib"]
    return projected_margin, {
        "four_factor_margin": four_factor_margin,
        "v3_residual_correction": v3_correction,
        "eFG_margin": float(features["efg_margin"]),
        "eFG_offense": float(contrib["eFG_Diff"]),
        "eFG_defense": float(contrib["Def_eFG_Adv"]),
        "turnover_margin": float(features["tov_margin"]),
        "turnover_offense": float(contrib["TOV_Adv"]),
        "turnover_defense": float(contrib["ForceTOV_Adv"]),
        "rebounding_margin": float(features["orb_margin"]),
        "rebounding_offense": float(contrib["ORB_Diff"]),
        "rebounding_defense": float(contrib["DefORB_Adv"]),
        "free_throw_margin": float(features["ftr_margin"]),
        "free_throw_offense": float(contrib["FTR_Diff"]),
        "free_throw_defense": float(contrib["DefFTR_Adv"]),
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
                    seed_a=None, seed_b=None, location=0):
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
    # Home court is the fitted location coefficient from the same regression.
    location_code = 0 if location is None else int(location)
    home_court_margin = HOME_COURT_POINTS * location_code
    schedule_std = float(full_df["SOS"].std(ddof=0)) if "SOS" in full_df.columns else 0.0
    if schedule_std > 0 and "SOS" in A.index and "SOS" in B.index and pd.notna(A["SOS"]) and pd.notna(B["SOS"]):
        schedule_margin = SCHEDULE_Z_POINTS * (float(A["SOS"]) - float(B["SOS"])) / schedule_std
    else:
        schedule_margin = 0.0
    projected_margin = float(projected_margin) + home_court_margin + schedule_margin
    margin_components["home_court_margin"] = home_court_margin
    margin_components["schedule_margin"] = schedule_margin
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
        "location": location_code,
        "home_court_margin": home_court_margin,
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
    if p >= 0.85:
        return "elite"
    if p >= 0.70:
        return "strong"
    if p >= 0.55:
        return "above average"
    if p >= 0.45:
        return "average"
    if p >= 0.30:
        return "below average"
    return "poor"


def _profile_number(value):
    value = float(value)
    if abs(value) > 1.5:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def style_tags(team_name, row, full_df, percentile_tables):
    sections = {
        "Offense": [],
        "Ball Control": [],
        "Rebounding": [],
        "Defense": [],
        "Tempo": [],
        "Concerns": [],
    }

    def pct(col, lower=False):
        return oriented_percentile(
            full_df,
            percentile_tables,
            team_name,
            col,
            lower_is_better=lower,
        )

    def add(section, text, score):
        if text not in [existing for _, existing in sections[section]]:
            sections[section].append((score, text))

    def say(section, sentence, percentile):
        if percentile is None:
            return
        add(section, sentence, abs(float(percentile) - 0.5))

    p_efg = pct("eFG%")
    if p_efg is not None:
        say(
            "Offense",
            f"Shooting efficiency is {label_strength(p_efg)}. "
            f"The effective field-goal percentage is {_profile_number(row['eFG%'])} ({pct_label(p_efg)}).",
            p_efg,
        )

    p_3p = pct("3P%")
    if p_3p is not None:
        say(
            "Offense",
            f"Three-point shooting is {label_strength(p_3p)}. "
            f"The 3-point percentage is {_profile_number(row['3P%'])} ({pct_label(p_3p)}).",
            p_3p,
        )

    p_3par = pct("3PAr")
    if p_3par is not None and (p_3par >= 0.70 or p_3par <= 0.35):
        style = "relies heavily on threes" if p_3par >= 0.70 else "works through the paint"
        say(
            "Offense",
            f"The offense {style}. "
            f"The 3-point rate is {_profile_number(row['3PAr'])} ({pct_label(p_3par)}).",
            p_3par,
        )

    p_ftr = pct("FT/FGA")
    if p_ftr is not None:
        say(
            "Offense",
            f"Free-throw pressure is {label_strength(p_ftr)}. "
            f"The free-throw rate is {_profile_number(row['FT/FGA'])} ({pct_label(p_ftr)}).",
            p_ftr,
        )

    p_tov = pct("TOV%", lower=True)
    if p_tov is not None:
        say(
            "Ball Control",
            f"Ball security is {label_strength(p_tov)}. "
            f"The turnover rate is {_profile_number(row['TOV%'])} ({pct_label(p_tov)}).",
            p_tov,
        )

    p_ast = pct("AST%")
    if p_ast is not None:
        say(
            "Ball Control",
            f"Ball movement is {label_strength(p_ast)}. "
            f"The assist rate is {_profile_number(row['AST%'])} ({pct_label(p_ast)}).",
            p_ast,
        )

    p_orb = pct("ORB%")
    if p_orb is not None:
        say(
            "Rebounding",
            f"Offensive rebounding is {label_strength(p_orb)}. "
            f"The offensive-rebound rate is {_profile_number(row['ORB%'])} ({pct_label(p_orb)}).",
            p_orb,
        )

    p_trb = pct("TRB%")
    if p_trb is not None:
        say(
            "Rebounding",
            f"Total rebounding is {label_strength(p_trb)}. "
            f"The total-rebound rate is {_profile_number(row['TRB%'])} ({pct_label(p_trb)}).",
            p_trb,
        )

    p_def_efg = pct("Opp_eFG%", lower=True)
    if p_def_efg is not None:
        say(
            "Defense",
            f"Shot defense is {label_strength(p_def_efg)}. "
            f"Opponents have an effective field-goal percentage of {_profile_number(row['Opp_eFG%'])} ({pct_label(p_def_efg)}).",
            p_def_efg,
        )

    p_def_tov = pct("Opp_TOV%")
    if p_def_tov is not None:
        say(
            "Defense",
            f"Turnover pressure is {label_strength(p_def_tov)}. "
            f"The forced-turnover rate is {_profile_number(row['Opp_TOV%'])} ({pct_label(p_def_tov)}).",
            p_def_tov,
        )

    p_def_orb = pct("Opp_ORB%", lower=True)
    if p_def_orb is not None:
        say(
            "Defense",
            f"Defensive rebounding is {label_strength(p_def_orb)}. "
            f"Opponents have an offensive-rebound rate of {_profile_number(row['Opp_ORB%'])} ({pct_label(p_def_orb)}).",
            p_def_orb,
        )

    p_pace = pct("Pace")
    if p_pace is not None:
        if p_pace >= 0.90:
            pace_word = "very fast"
        elif p_pace >= 0.70:
            pace_word = "fast"
        elif p_pace >= 0.60:
            pace_word = "above average"
        elif p_pace <= 0.10:
            pace_word = "very slow"
        elif p_pace <= 0.30:
            pace_word = "slow"
        elif p_pace <= 0.40:
            pace_word = "below average"
        else:
            pace_word = "average"
        say(
            "Tempo",
            f"Pace is {pace_word}. "
            f"The team plays at {row['Pace']:.1f} possessions per game ({pct_label(p_pace)}).",
            p_pace,
        )

    if p_tov is not None and p_tov <= 0.35:
        add(
            "Concerns",
            f"Turnovers are a problem. The turnover rate is {_profile_number(row['TOV%'])} ({pct_label(p_tov)}).",
            1,
        )
    if p_def_orb is not None and p_def_orb <= 0.40:
        add(
            "Concerns",
            f"Defensive rebounding is soft. Opponents have an offensive-rebound rate of {_profile_number(row['Opp_ORB%'])} ({pct_label(p_def_orb)}).",
            1,
        )
    if p_efg is not None and p_efg <= 0.35:
        add(
            "Concerns",
            f"Shooting efficiency is a problem. The effective field-goal percentage is {_profile_number(row['eFG%'])} ({pct_label(p_efg)}).",
            1,
        )

    chosen = {}
    for section, items in sections.items():
        ranked = sorted(items, key=lambda item: item[0], reverse=True)
        if section == "Concerns":
            texts = [text for _, text in ranked[:3]]
        else:
            notable = [text for score, text in ranked if score >= 0.06]
            texts = notable[:3] or [text for _, text in ranked[:1]]
        if texts:
            chosen[section] = texts
    return chosen

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

    readable = {
        "eFG%": "effective field-goal percentage",
        "Opp_eFG%": "opponent effective field-goal percentage",
        "TOV%": "turnover rate",
        "Opp_TOV%": "forced-turnover rate",
        "ORB%": "offensive-rebound rate",
        "Opp_ORB%": "opponent offensive-rebound rate",
        "FT/FGA": "free-throw rate",
        "Opp_FT/FGA": "opponent free-throw rate",
        "AST%": "assist rate",
        "3P%": "3-point percentage",
        "3PAr": "3-point rate",
        "Opp_3P%": "opponent 3-point percentage",
    }

    def fmt(team, col, val, p):
        name = readable.get(col, col.replace("_", " "))
        return f"{team}'s {name} is {_profile_number(val)} ({pct_label(p)})"

    def mismatch(a_p, b_p):
        return a_p is not None and b_p is not None and (a_p - b_p) >= 0.25

    def strong_edge(a_p, b_p):
        return a_p is not None and b_p is not None and a_p >= 0.70 and b_p <= 0.30

    used_stats = set()

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

        if strong_edge(a_off, b_def):
            notes.append(
                f"{team_a} has a major advantage in {label}. "
                f"{fmt(team_a, off_stat, A[off_stat], a_off)}, while {fmt(team_b, def_stat, B[def_stat], b_def)}."
            )
            used_stats.update((off_stat, def_stat))
        elif mismatch(a_off, b_def):
            notes.append(
                f"{team_a} {phrase} against {team_b}. "
                f"{fmt(team_a, off_stat, A[off_stat], a_off)}, while {fmt(team_b, def_stat, B[def_stat], b_def)}."
            )
            used_stats.update((off_stat, def_stat))

        if strong_edge(b_off, a_def):
            notes.append(
                f"{team_b} has a major advantage in {label}. "
                f"{fmt(team_b, off_stat, B[off_stat], b_off)}, while {fmt(team_a, def_stat, A[def_stat], a_def)}."
            )
            used_stats.update((off_stat, def_stat))
        elif mismatch(b_off, a_def):
            notes.append(
                f"{team_b} {phrase} against {team_a}. "
                f"{fmt(team_b, off_stat, B[off_stat], b_off)}, while {fmt(team_a, def_stat, A[def_stat], a_def)}."
            )
            used_stats.update((off_stat, def_stat))

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
        if stat in used_stats:
            continue
        a_p = pct(team_a, stat, lower)
        b_p = pct(team_b, stat, lower)

        if a_p is None or b_p is None:
            continue

        if mismatch(a_p, b_p):
            notes.append(
                f"{team_a} has the edge in {readable.get(stat, stat)}. "
                f"{fmt(team_a, stat, A[stat], a_p)}, while {fmt(team_b, stat, B[stat], b_p)}."
            )

        if mismatch(b_p, a_p):
            notes.append(
                f"{team_b} has the edge in {readable.get(stat, stat)}. "
                f"{fmt(team_b, stat, B[stat], b_p)}, while {fmt(team_a, stat, A[stat], a_p)}."
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
        if stat in used_stats:
            continue
        a_p = pct(team_a, stat, lower)
        b_p = pct(team_b, stat, lower)

        if a_p is None or b_p is None:
            continue

        if mismatch(a_p, b_p):
            notes.append(
                f"{team_a} has the defensive edge in {readable.get(stat, stat)}. "
                f"{fmt(team_a, stat, A[stat], a_p)}, while {fmt(team_b, stat, B[stat], b_p)}."
            )

        if mismatch(b_p, a_p):
            notes.append(
                f"{team_b} has the defensive edge in {readable.get(stat, stat)}. "
                f"{fmt(team_b, stat, B[stat], b_p)}, while {fmt(team_a, stat, A[stat], a_p)}."
            )

    # =========================
    # PACE
    # =========================
    if "Pace" in A.index and "Pace" in B.index:
        if abs(A["Pace"] - B["Pace"]) >= 3:
            faster = team_a if A["Pace"] > B["Pace"] else team_b
            slower = team_b if faster == team_a else team_a
            notes.append(
                f"{faster} will try to push the tempo, while {slower} prefers a slower game. "
                f"{team_a} plays at {A['Pace']:.1f} possessions per game, and {team_b} plays at {B['Pace']:.1f}."
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

    if pace_pct <= 0.35:
        notes.append(
            f"The projected pace is {avg_pace:.1f} possessions, slower than about {round((1 - pace_pct) * 100)}% of Division I teams. "
            "Fewer possessions give the trailing team fewer chances to get buried."
        )
    elif pace_pct >= 0.65:
        notes.append(
            f"The projected pace is {avg_pace:.1f} possessions, faster than about {round(pace_pct * 100)}% of Division I teams. "
            "More possessions give the better team more chances to pull away."
        )

    # Variance
    a_3par = oriented_percentile(full_df, percentile_tables, team_a, "3PAr")
    b_3par = oriented_percentile(full_df, percentile_tables, team_b, "3PAr")
    if (a_3par is not None and a_3par >= 0.80) or (b_3par is not None and b_3par >= 0.80):
        notes.append("At least one team takes a high share of its shots from three. A hot or cold night from deep can move the score more than these season numbers suggest.")

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

    if w_efg and l_def_efg and w_efg >= l_def_efg + 0.25:
        add(f"{w} should have a clear scoring efficiency advantage in this matchup.")

    # =========================
    # TURNOVERS
    # =========================
    w_tov = pct(w, "TOV%", lower=True)
    l_def_tov = pct(l, "Opp_TOV%")

    if w_tov and l_def_tov and w_tov >= 0.80 and w_tov >= l_def_tov + 0.25:
        add(f"{w} should control the game with superior ball security.")

    # =========================
    # REBOUNDING
    # =========================
    w_orb = pct(w, "ORB%")
    l_def_orb = pct(l, "Opp_ORB%", lower=True)

    if w_orb and l_def_orb and w_orb >= l_def_orb + 0.25:
        add(f"{w} may create extra possessions through offensive rebounding.")

    # =========================
    # OVERALL EFFICIENCY (NEW)
    # =========================
    w_ortg = pct(w, "ORtg")
    l_ortg = pct(l, "ORtg")

    if w_ortg and l_ortg and w_ortg >= l_ortg + 0.15:
        add(f"{w} holds the stronger overall offensive profile in this matchup.")

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
            
_REPORT_CSS = """
:root {
  color-scheme: light;
  --ink: #1a1814;
  --muted: #5e584f;
  --line: #ddd4c6;
  --paper: #e9e2d6;
  --sheet: #fbf8f4;
  --track: #e6dfd4;
  --a: #1d3557;
  --b: #8c3a2f;
  --sans: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
  --serif: "Iowan Old Style", Palatino, "Palatino Linotype", "Book Antiqua", Georgia, serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 16px;
  line-height: 1.5;
}
.sheet {
  position: relative;
  max-width: 1080px;
  margin: 28px auto;
  padding: 36px 40px 28px;
  background: var(--sheet);
  border-top: 4px solid var(--ink);
}
.data-from {
  position: absolute;
  top: 8px;
  right: 14px;
  max-width: 58%;
  margin: 0;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 10px;
  line-height: 1.3;
  text-align: right;
}
.data-from a { color: inherit; text-decoration: none; }
.data-from a:hover { text-decoration: underline; }
.kicker, .meta, .verdict, .factor, .tags h3, .section-label, table, .stat, .footer, .pos, .size {
  font-family: var(--sans);
}
.top, .split-labels, .factor-top, .compare-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}
.kicker, .section-label, .tags h3, .pos span, .players th {
  margin: 0;
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.meta, .footer, .sub { color: var(--muted); }
.meta { margin: 4px 0 0; font-size: 14px; }
.hero {
  display: grid;
  grid-template-columns: 1fr minmax(240px, 300px) 1fr;
  gap: 28px;
  align-items: center;
  padding: 28px 0 8px;
}
.hero > * { min-width: 0; }
.side { margin: 0 0 6px; }
h1 {
  margin: 0;
  font-size: clamp(1.8rem, 3vw, 3rem);
  font-weight: 600;
  letter-spacing: -0.03em;
  line-height: 1.05;
  overflow-wrap: break-word;
}
.side-a h1 { color: var(--a); }
.side-b h1, .side-b { text-align: right; }
.side-b h1 { color: var(--b); }
.pick {
  display: inline-block;
  margin-top: 10px;
  padding: 3px 8px;
  border: 1px solid currentColor;
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.verdict { text-align: center; }
.spread {
  margin: 0;
  font-family: var(--serif);
  font-size: 2rem;
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
}
.prob { margin: 4px 0 14px; font-size: 15px; }
.venue { margin: 8px 0 0; color: var(--muted); font-size: 13px; letter-spacing: 0.01em; }
.split {
  display: flex;
  height: 14px;
  background: var(--track);
}
.seg { min-width: 4px; }
.seg.a { background: var(--a); }
.seg.b { background: var(--b); }
.split-labels { margin-top: 8px; font-size: 13px; font-variant-numeric: tabular-nums; }
.sub { margin: 12px 0 0; font-size: 13px; }
section { padding: 26px 0; border-top: 1px solid var(--line); }
.section-label { margin-bottom: 14px; }
.factor { margin: 0 0 14px; }
.factor-top { font-size: 14px; margin-bottom: 2px; }
.factor-split { margin: 0 0 6px; color: var(--muted); font-size: 12px; line-height: 1.4; }
.diverge {
  position: relative;
  display: grid;
  grid-template-columns: 1fr 1fr;
  height: 10px;
  background: var(--track);
}
.half { display: flex; height: 100%; }
.half.left { justify-content: flex-end; }
.half.right { justify-content: flex-start; }
.half span { display: block; height: 100%; }
.half.left span { background: var(--a); }
.half.right span { background: var(--b); }
.diverge::after {
  content: "";
  position: absolute;
  left: 50%;
  top: -3px;
  bottom: -3px;
  width: 1px;
  background: var(--ink);
}
.summary { margin: 8px 0 0; max-width: 70ch; }
.group { margin-top: 22px; }
.group h3 {
  margin: 0 0 10px;
  font-family: var(--sans);
  font-size: 15px;
  font-weight: 600;
}
.cmp-head, .cmp-row {
  display: grid;
  grid-template-columns: minmax(190px, 1.2fr) 1fr 1fr;
  gap: 22px;
}
.cmp-head {
  align-items: end;
  margin-bottom: 2px;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 700;
}
.cmp-head .a { color: var(--a); }
.cmp-head .b { color: var(--b); }
.cmp-row {
  align-items: center;
  padding: 13px 0;
  border-top: 1px solid var(--line);
}
.cmp-name strong {
  display: block;
  font-family: var(--serif);
  font-size: 1.08rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.cmp-name span {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 12px;
  line-height: 1.35;
}
.cmp-cell {
  display: grid;
  grid-template-columns: 52px minmax(64px, 1fr) 118px;
  gap: 8px;
  align-items: center;
  font-family: var(--sans);
  font-size: 16px;
  font-variant-numeric: tabular-nums;
}
.cmp-cell .num { font-weight: 700; }
.cmp-cell.a .num, .cmp-cell.a .mini-name { color: var(--a); }
.cmp-cell.b .num, .cmp-cell.b .mini-name { color: var(--b); }
.cmp-cell .pct { color: var(--muted); font-size: 12px; line-height: 1.2; text-align: right; }
.rank { margin: 8px 0 0; font-family: var(--sans); font-size: 14px; }
.mini-name { display: none; font-size: 12px; font-weight: 700; }
.meter { height: 9px; background: var(--track); }
.meter span { display: block; height: 100%; }
.cmp-cell.a .meter span, .stat-line.a .meter span { background: var(--a); }
.cmp-cell.b .meter span, .stat-line.b .meter span { background: var(--b); }
.hint { margin: 0 0 8px; color: var(--muted); font-family: var(--sans); font-size: 13px; line-height: 1.45; }
.teams { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
.team-card { min-width: 0; padding-top: 12px; }
.team-card.a { border-top: 3px solid var(--a); }
.team-card.b { border-top: 3px solid var(--b); }
.team-card h2 { margin: 0 0 6px; font-size: 1.6rem; letter-spacing: -0.03em; }
.team-card.a h2 { color: var(--a); }
.team-card.b h2 { color: var(--b); }
.size { margin: 0 0 12px; font-size: 14px; }
.pos { display: flex; gap: 18px; margin-bottom: 14px; }
.pos strong { display: block; color: var(--ink); font-size: 16px; letter-spacing: 0; text-transform: none; }
.players { overflow-x: auto; }
.players table { width: 100%; min-width: 420px; border-collapse: collapse; }
.players th, .players td { padding: 6px 4px; border-bottom: 1px solid var(--line); text-align: right; }
.players th:first-child, .players td:first-child { text-align: left; }
.players td:first-child { font-family: var(--serif); }
.tags ul { list-style: none; margin: 0; padding: 0; }
.tags li { margin: 7px 0; padding-left: 12px; border-left: 2px solid var(--line); }
.narrative { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; }
.narrative p { margin: 0 0 12px; }
.footer { margin-top: 8px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 12px; }
.reading { max-width: 72ch; }
@media (max-width: 800px) {
  .sheet { margin: 0; padding: 22px 16px 18px; }
  .data-from { position: static; max-width: none; margin: 0 0 12px; }
  .top { align-items: flex-start; flex-direction: column; gap: 8px; }
  .hero, .teams, .narrative { grid-template-columns: 1fr; }
  .side-b, .side-b h1 { text-align: left; }
  .verdict { text-align: left; }
  .cmp-head { display: none; }
  .cmp-row { grid-template-columns: 1fr; gap: 8px; }
  .mini-name { display: block; }
}
@media print {
  body { background: white; }
  .sheet { margin: 0; border-top: 0; }
}
"""

_COMPARE_GROUPS = [
    ("Team strength", [
        ("Offensive rating", "Points scored per 100 possessions", "ORtg", False),
        ("Defensive rating", "Points allowed per 100 possessions", "Opp_ORtg", True),
        ("SRS", "Point margin against an average team", "SRS", False),
        ("Strength of schedule", "Quality of the opponents faced", "SOS", False),
        ("Pace", "Possessions per game", "Pace", False),
    ]),
    ("Four Factors", [
        ("Effective FG%", "Offense. Shooting efficiency, counting threes as 1.5", "eFG%", False),
        ("Effective FG% allowed", "Defense. A longer bar means opponents shoot worse", "Opp_eFG%", True),
        ("Turnover rate", "Offense. A longer bar means fewer turnovers", "TOV%", True),
        ("Forced turnover rate", "Defense. A longer bar means the defense takes the ball more often", "Opp_TOV%", False),
        ("Offensive rebound %", "Offense. Share of missed shots the offense grabs", "ORB%", False),
        ("Defensive rebounding", "Defense. A longer bar means opponents grab fewer offensive rebounds", "Opp_ORB%", True),
        ("Free throw rate", "Offense. Free throws attempted per field-goal attempt", "FT/FGA", False),
        ("Free throw rate allowed", "Defense. A longer bar means opponents get to the line less", "Opp_FT/FGA", True),
    ]),
    ("Shooting", [
        ("3-point %", "Share of threes that go in", "3P%", False),
        ("3-point attempt rate", "Share of shots that are threes", "3PAr", False),
    ]),
]


# Official school colors, chosen dark enough to read on the cream page.
# A school that is not listed keeps the default navy or brick.
TEAM_COLORS = {
    "Air Force": "#003087", "Akron": "#041E42", "Alabama": "#9E1B32", "Arizona": "#AB0520",
    "Arizona State": "#8C1D40", "Arkansas": "#9D2235", "Army": "#000000", "Auburn": "#E87722",
    "Baylor": "#154734", "Boise State": "#0033A0", "Boston College": "#8C2232", "Bradley": "#A50034",
    "Butler": "#13294B", "BYU": "#002E5D", "California": "#003262", "Charlotte": "#005035",
    "Cincinnati": "#E00122", "Clemson": "#F56600", "Colorado": "#000000", "Colorado State": "#1E4D2B",
    "Connecticut": "#000E2F", "Creighton": "#005CA9", "Davidson": "#AC1A2F", "Dayton": "#CE1141",
    "DePaul": "#005EB8", "Drake": "#004477", "Duke": "#003087", "Duquesne": "#BA0C2F",
    "East Carolina": "#592A8A", "Florida": "#FA4616", "Florida Atlantic": "#003366",
    "Florida State": "#782F40", "Fordham": "#860038", "Fresno State": "#DB0032",
    "George Mason": "#006633", "George Washington": "#002654", "Georgetown": "#041E42",
    "Georgia": "#BA0C2F", "Georgia Tech": "#003057", "Gonzaga": "#C8102E", "Grand Canyon": "#522398",
    "Harvard": "#A51C30", "Houston": "#C8102E", "Illinois": "#E84A27", "Illinois State": "#CE1126",
    "Indiana": "#990000", "Indiana State": "#003DA5", "Iowa": "#000000", "Iowa State": "#C8102E",
    "Kansas": "#0051BA", "Kansas State": "#512888", "Kentucky": "#0033A0", "Liberty": "#C41230",
    "Louisville": "#AD0000", "Loyola Chicago": "#922247", "LSU": "#461D7C", "Marquette": "#003366",
    "Maryland": "#E03A3E", "Massachusetts": "#881C1C", "Memphis": "#003087", "Miami (FL)": "#F47321",
    "Miami (OH)": "#C3142D", "Michigan": "#00274C", "Michigan State": "#18453B", "Minnesota": "#7A0019",
    "Mississippi State": "#660000", "Missouri": "#000000", "Missouri State": "#5E0009",
    "Murray State": "#002144", "Navy": "#00205B", "NC State": "#CC0000", "Nebraska": "#E41C38",
    "Nevada": "#003366", "New Mexico": "#BA0C2F", "North Carolina": "#4B9CD3", "North Texas": "#00853E",
    "Northern Iowa": "#4B116F", "Northwestern": "#4E2A84", "Notre Dame": "#0C2340", "Ohio State": "#BB0000",
    "Oklahoma": "#841617", "Oklahoma State": "#FF6600", "Ole Miss": "#CE1126", "Oregon": "#154733",
    "Oregon State": "#DC4405", "Penn State": "#041E42", "Pittsburgh": "#003594", "Princeton": "#E87722",
    "Providence": "#000000", "Purdue": "#000000", "Rhode Island": "#002868", "Rice": "#00205B",
    "Richmond": "#990000", "Rutgers": "#CC0033", "Saint Joseph's": "#9E1B32", "Saint Louis": "#003DA5",
    "Saint Mary's": "#003366", "San Diego State": "#A6192E", "San Francisco": "#00543C",
    "Santa Clara": "#862633", "Seton Hall": "#004488", "SMU": "#0033A0", "South Carolina": "#73000A",
    "South Florida": "#006747", "St. Bonaventure": "#582C83", "St. John's": "#C8102E",
    "St. John's (NY)": "#C8102E", "Stanford": "#8C1515", "Syracuse": "#F76900", "TCU": "#4D1979",
    "Temple": "#9D2235", "Tennessee": "#FF8200", "Texas": "#BF5700", "Texas A&M": "#500000",
    "Texas Tech": "#CC0000", "Tulane": "#006747", "Tulsa": "#002D72", "UAB": "#1E6B52",
    "UCF": "#000000", "UCLA": "#2D68C4", "UConn": "#000E2F", "UMass": "#881C1C", "UNLV": "#B10202",
    "USC": "#990000", "Utah": "#CC0000", "Utah State": "#0F2439", "Vanderbilt": "#000000",
    "VCU": "#000000", "Villanova": "#00205B", "Virginia": "#232D4B", "Virginia Tech": "#630031",
    "Wake Forest": "#000000", "Washington": "#4B2E83", "Washington State": "#981E32",
    "West Virginia": "#002855", "Wichita State": "#000000", "Wisconsin": "#C5050C",
    "Wyoming": "#492F24", "Xavier": "#0C2340", "Yale": "#00356B",
}


def _esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def _clean_note(text):
    return re.sub(r"\s+", " ", str(text)).strip()


_NOTE_LABELS = (
    ("Opp_FT/FGA", "opponent free-throw rate"),
    ("Opp FT/FGA", "opponent free-throw rate"),
    ("Opp_3PAr", "opponent 3-point rate"),
    ("Opp 3PAr", "opponent 3-point rate"),
    ("Opp_3P%", "opponent 3-point percentage"),
    ("Opp 3P%", "opponent 3-point percentage"),
    ("Opp_eFG%", "opponent effective FG%"),
    ("Opp eFG%", "opponent effective FG%"),
    ("Opp_TOV%", "forced-turnover rate"),
    ("Opp TOV%", "forced-turnover rate"),
    ("Opp_ORB%", "opponent offensive-rebound rate"),
    ("Opp ORB%", "opponent offensive-rebound rate"),
    ("FT/FGA", "free-throw rate"),
    ("3PAr", "3-point rate"),
    ("3P%", "3-point percentage"),
    ("eFG%", "effective FG%"),
    ("TOV%", "turnover rate"),
    ("ORB%", "offensive-rebound rate"),
    ("AST%", "assist rate"),
    ("TRB%", "total-rebound rate"),
)


def _plain_note(text):
    text = _clean_note(text)
    for token, label in _NOTE_LABELS:
        text = text.replace(token, label)

    def trim(match):
        number = float(match.group(0))
        if abs(number) >= 10:
            return f"{number:.1f}"
        trimmed = f"{number:.3f}".rstrip("0").rstrip(".")
        return trimmed

    return re.sub(r"\d+\.\d+", trim, text)


def _fmt_value(col, value):
    value = float(value)
    if col in {"SRS", "SOS", "ORtg", "Opp_ORtg", "Pace"}:
        return f"{value:.1f}"
    if abs(value) > 1.5:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _shot_pct(value):
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100.0
    return f"{value:.1f}%"


def _ordinal(p):
    label = pct_label(p)
    if not label:
        return "—"
    return label.replace(" percentile", "")


def _point_side(value, team_a, team_b):
    value = float(value)
    if abs(value) < 0.05:
        return "even"
    team = team_a if value > 0 else team_b
    return f"{abs(value):.1f} pts toward {team}"


def _favor_phrase(value, team_a, team_b):
    if abs(value) < 0.05:
        return "are even"
    team = team_a if value > 0 else team_b
    return f"favor {team} by {abs(value):.1f} points"


def _home_court_sentence(result, team_a, team_b):
    location = int(result.get("location") or 0)
    points = f"{HOME_COURT_POINTS:.3f}"
    if location > 0:
        return f"{team_a} home adds {points} points."
    if location < 0:
        return f"{team_b} home adds {points} points."
    return "Neutral site, so home court adds nothing."


_VENUE_PAGES = {
    "Miami (FL)": "Miami Hurricanes men's basketball",
    "Miami (OH)": "Miami RedHawks men's basketball",
    "Connecticut": "UConn Huskies men's basketball",
    "UConn": "UConn Huskies men's basketball",
    "Ole Miss": "Ole Miss Rebels men's basketball",
    "UCF": "UCF Knights men's basketball",
    "USC": "USC Trojans men's basketball",
    "LSU": "LSU Tigers men's basketball",
    "UNLV": "UNLV Rebels men's basketball",
    "VCU": "VCU Rams men's basketball",
    "SMU": "SMU Mustangs men's basketball",
    "UAB": "UAB Blazers men's basketball",
    "BYU": "BYU Cougars men's basketball",
    "St. John's": "St. John's Red Storm men's basketball",
    "St. John's (NY)": "St. John's Red Storm men's basketball",
    "Saint Mary's": "Saint Mary's Gaels men's basketball",
}


def _wiki_plain(value):
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.I | re.S)
    value = re.sub(r"<ref[^>/]*/>", "", value, flags=re.I)
    value = re.sub(r"<br\s*/?>", " and ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\(\s*[\d,]+\s*\)", "", value)
    value = value.replace("'''", "").replace("''", "")
    value = re.sub(r"(\s+and\s+)+", " and ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" .")


def _wiki_infobox(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "redirects": 1,
        },
        headers={"User-Agent": "CollegeBasketballModel/1.0 (personal research)"},
        timeout=12,
    )
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    revisions = page.get("revisions") or []
    if not revisions:
        return None, None
    text = revisions[0].get("slots", {}).get("main", {}).get("*", "")
    arena = place = None
    for line in text.splitlines():
        arena_match = re.match(r"\|\s*arena\s*=\s*(.*)", line, flags=re.I)
        place_match = re.match(r"\|\s*location\s*=\s*(.*)", line, flags=re.I)
        if arena_match and not arena:
            arena = _wiki_plain(arena_match.group(1)) or None
        if place_match and not place:
            place = _wiki_plain(place_match.group(1)) or None
    return arena, place


@lru_cache(maxsize=None)
def lookup_home_venue(team_name):
    """Arena and city for a team's home floor. Neutral games do not use this."""
    bare = re.sub(r"\s*\([^)]*\)", "", str(team_name)).strip()
    queries = []
    if team_name in _VENUE_PAGES:
        queries.append(_VENUE_PAGES[team_name])
    queries.append(f"{bare} men's basketball")
    seen = set()
    try:
        for query in queries:
            found = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"},
                headers={"User-Agent": "CollegeBasketballModel/1.0 (personal research)"},
                timeout=12,
            )
            for hit in found.json().get("query", {}).get("search", []):
                title = hit.get("title") or ""
                if title in seen or re.search(r"\d{4}", title) or "basketball" not in title.lower():
                    continue
                seen.add(title)
                arena, place = _wiki_infobox(title)
                if arena or place:
                    return {"arena": arena, "place": place}
    except Exception:
        return {"arena": None, "place": None}
    return {"arena": None, "place": None}


def venue_line(location, team_a, team_b):
    if not location:
        return "Neutral site"
    home = team_a if int(location) > 0 else team_b
    venue = lookup_home_venue(home)
    arena = venue.get("arena")
    place = venue.get("place")
    if arena and place:
        return f"{arena} · {place}"
    if arena:
        return arena
    if place:
        return f"{home} home · {place}"
    return f"{home} home"


def _data_source_tag():
    if not DATA_SOURCE_URL:
        return '<p class="data-from">Data from local CSV files</p>'
    href = _esc(DATA_SOURCE_URL)
    visible = href.replace("/", "/<wbr>")
    return f'<p class="data-from">Data from <a href="{href}">{visible}</a></p>'


def _season_name(year):
    end = int(year)
    return f"{end - 1}–{str(end)[-2:]}"


def _season_line():
    retrieved = date.today().strftime("%B %d, %Y").replace(" 0", " ")
    year = ACTIVE_SEASON_YEAR
    if year is None:
        return f"Saved team stats on this computer · {retrieved}"
    season = _season_name(year)
    if int(year) < infer_current_ncaa_season():
        return f"{season} season · final team stats"
    return f"{season} season · team stats through {retrieved}"


def _channel(value):
    value = value / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    red, green, blue = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def _contrast(hex_a, hex_b):
    lighter = max(_luminance(hex_a), _luminance(hex_b))
    darker = min(_luminance(hex_a), _luminance(hex_b))
    return (lighter + 0.05) / (darker + 0.05)


def _darken(hex_color, factor):
    hex_color = hex_color.lstrip("#")
    red, green, blue = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(red * factor), int(green * factor), int(blue * factor)
    )


def _readable_color(hex_color, background="#fbf8f4", minimum=4.5):
    color = hex_color
    if _contrast(color, background) >= minimum:
        return color
    for step in range(1, 18):
        mixed = _darken(hex_color, 1 - step * 0.05)
        if _contrast(mixed, background) >= minimum:
            return mixed
    return "#1a1814"


def _team_color_pair(team_a, team_b):
    fallback_a, fallback_b = "#1d3557", "#8c3a2f"
    color_a = _readable_color(TEAM_COLORS[team_a]) if team_a in TEAM_COLORS else None
    color_b = _readable_color(TEAM_COLORS[team_b]) if team_b in TEAM_COLORS else None
    if color_a and color_b:
        if color_a.lower() == color_b.lower():
            color_b = _darken(color_b, 0.72)
        return color_a, color_b
    if color_a:
        other = fallback_b if _contrast(color_a, fallback_b) > _contrast(color_a, fallback_a) else fallback_a
        if other.lower() == color_a.lower():
            other = fallback_b
        return color_a, other
    if color_b:
        other = fallback_a if _contrast(color_b, fallback_a) > _contrast(color_b, fallback_b) else fallback_b
        if other.lower() == color_b.lower():
            other = fallback_a
        return other, color_b
    return fallback_a, fallback_b


def _comparison_html(team_a, team_b, row_a, row_b, full_df, percentile_tables):
    groups = []
    for title, specs in _COMPARE_GROUPS:
        rows = [
            "<div class=\"cmp-head\"><span></span>"
            f"<span class=\"a\">{_esc(team_a)}</span>"
            f"<span class=\"b\">{_esc(team_b)}</span></div>"
        ]
        any_row = False
        for label, gloss, col, lower in specs:
            if col not in row_a.index or col not in row_b.index:
                continue
            if pd.isna(row_a[col]) or pd.isna(row_b[col]):
                continue
            any_row = True
            cells = []
            for team, row, css_class in (
                (team_a, row_a, "a"),
                (team_b, row_b, "b"),
            ):
                percentile = oriented_percentile(
                    full_df, percentile_tables, team, col, lower_is_better=lower
                )
                width = 0 if percentile is None else max(0, min(100, round(percentile * 100)))
                cells.append(
                    "<div class=\"cmp-cell {css}\">"
                    "<span class=\"mini-name\">{team}</span>"
                    "<span class=\"num\">{value}</span>"
                    "<span class=\"meter\"><span style=\"width:{width}%\"></span></span>"
                    "<span class=\"pct\">{ordinal}</span>"
                    "</div>".format(
                        css=css_class,
                        team=_esc(team),
                        value=_fmt_value(col, row[col]),
                        width=width,
                        ordinal=_esc(pct_label(percentile) or "—"),
                    )
                )
            rows.append(
                "<div class=\"cmp-row\">"
                f"<div class=\"cmp-name\"><strong>{_esc(label)}</strong><span>{_esc(gloss)}</span></div>"
                + "".join(cells)
                + "</div>"
            )
        if any_row:
            groups.append(f"<div class=\"group\"><h3>{_esc(title)}</h3>{''.join(rows)}</div>")
    if not groups:
        return ""
    return (
        "<section>"
        "<p class=\"section-label\">Season profile</p>"
        "<p class=\"hint\">The text after each bar is a percentile, not a rank in the country. "
        "98th percentile means better than 98% of Division I teams. A longer bar is always the better result, "
        "including the defensive rows, where a longer bar means the defense allows less.</p>"
        + "".join(groups)
        + "</section>"
    )


def _factor_html(team_a, team_b, result):
    parts = result.get("margin_components") or {}
    factors = [
        ("Shooting", "eFG_margin", "eFG_offense", "eFG_defense"),
        ("Turnovers", "turnover_margin", "turnover_offense", "turnover_defense"),
        ("Rebounding", "rebounding_margin", "rebounding_offense", "rebounding_defense"),
        ("Free throws", "free_throw_margin", "free_throw_offense", "free_throw_defense"),
    ]
    totals = [float(parts.get(key, 0.0)) for _, key, _, _ in factors]
    scale = max([abs(value) for value in totals] + [1.0])
    rows = []
    for label, total_key, offense_key, defense_key in factors:
        value = float(parts.get(total_key, 0.0))
        magnitude = min(100.0, abs(value) / scale * 100.0)
        left = f"{magnitude:.1f}" if value > 0 else "0"
        right = f"{magnitude:.1f}" if value < 0 else "0"
        detail = "Even" if abs(value) < 0.05 else _point_side(value, team_a, team_b)
        split = (
            f"Offense: {_point_side(parts.get(offense_key, 0.0), team_a, team_b)}. "
            f"Defense: {_point_side(parts.get(defense_key, 0.0), team_a, team_b)}."
        )
        rows.append(
            "<div class=\"factor\">"
            f"<div class=\"factor-top\"><span>{_esc(label)}</span><span>{_esc(detail)}</span></div>"
            f"<p class=\"factor-split\">{_esc(split)}</p>"
            "<div class=\"diverge\" aria-hidden=\"true\">"
            f"<div class=\"half left\"><span style=\"width:{left}%\"></span></div>"
            f"<div class=\"half right\"><span style=\"width:{right}%\"></span></div>"
            "</div></div>"
        )
    four = float(parts.get("four_factor_margin", 0.0))
    home_court = float(parts.get("home_court_margin", 0.0))
    schedule = float(parts.get("schedule_margin", 0.0))
    if abs(home_court) < 0.05:
        home_line = "Neutral site, so home court adds nothing."
    else:
        home_line = f"Home court adds {abs(home_court):.3f} points toward {team_a if home_court > 0 else team_b}."
    if abs(schedule) < 0.05:
        schedule_line = "Strength of schedule does not move the margin."
    else:
        schedule_line = (
            f"Strength of schedule adds {abs(schedule):.1f} points toward "
            f"{team_a if schedule > 0 else team_b}."
        )
    summary = (
        f"Added together, shooting, turnovers, rebounding, and free throws {_favor_phrase(four, team_a, team_b)}. "
        f"{home_line} {schedule_line}"
    )
    return (
        "<section>"
        "<p class=\"section-label\">Where the margin comes from</p>"
        f"<p class=\"hint\">Each bar is that factor’s effect on the projected margin, in points. "
        f"Left of center favors {_esc(team_a)}. Right of center favors {_esc(team_b)}. "
        "The line under each bar splits the same points into offense and defense.</p>"
        + "".join(rows)
        + f"<p class=\"summary\">{_esc(summary)}</p>"
        + "</section>"
    )


def _players_html(profile):
    players = sorted(profile.get("players") or [], key=lambda player: player["pts"], reverse=True)
    if not players:
        return ""
    body = []
    for player in players:
        body.append(
            "<tr>"
            f"<td>{_esc(player['name'])}</td>"
            f"<td>{_esc(player['pos'])}</td>"
            f"<td>{player['pts']:.1f}</td>"
            f"<td>{player['trb']:.1f}</td>"
            f"<td>{player['ast']:.1f}</td>"
            f"<td>{_shot_pct(player['fg'])}</td>"
            f"<td>{_shot_pct(player['fg3'])}</td>"
            "</tr>"
        )
    return (
        "<div class=\"players\"><table>"
        "<thead><tr><th>Player</th><th>Pos</th><th>PTS</th><th>REB</th><th>AST</th><th>FG</th><th>3P</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _positions_html(profile):
    groups = [
        ("Guards", profile.get("guards")),
        ("Forwards", profile.get("forwards")),
        ("Centers", profile.get("centers")),
    ]
    cells = [
        f"<div><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>"
        for label, value in groups
        if value
    ]
    if not cells:
        return ""
    return f"<div class=\"pos\">{''.join(cells)}</div>"


def _tags_html(tag_dict):
    if not tag_dict:
        return ""
    blocks = []
    for section, items in tag_dict.items():
        lis = "".join(f"<li>{_esc(_plain_note(item))}</li>" for item in items)
        blocks.append(f"<h3>{_esc(section)}</h3><ul>{lis}</ul>")
    return f"<div class=\"tags\">{''.join(blocks)}</div>"


def _team_card(team, css_class, profile, tags):
    parts = [f"<article class=\"team-card {css_class}\"><h2>{_esc(team)}</h2>"]
    if profile:
        parts.append(
            f"<p class=\"size\">{_esc(profile['avg_height_str'])} average"
            f" · {int(profile['rotation_size'])} rotation players</p>"
        )
        parts.append(_positions_html(profile))
        parts.append(_players_html(profile))
    else:
        parts.append("<p class=\"hint\">Roster and height data were not available for this team.</p>")
    parts.append(_tags_html(tags))
    parts.append("</article>")
    return "".join(parts)


def _notes_html(commentary):
    columns = []
    for title, key in (("Matchup notes", "tensions"), ("Projected game shape", "game_script")):
        notes = [_plain_note(note) for note in commentary.get(key) or []]
        notes = [note for note in notes if note]
        if not notes:
            continue
        paragraphs = "".join(f"<p>{_esc(note)}</p>" for note in notes)
        columns.append(f"<div><p class=\"section-label\">{title}</p>{paragraphs}</div>")
    edges = [_plain_note(edge) for edge in commentary.get("top_edges") or []]
    edges = [edge for edge in edges if edge]
    blocks = []
    if edges:
        paragraphs = "".join(f"<p>{_esc(edge)}</p>" for edge in edges)
        blocks.append(
            f"<section><p class=\"section-label\">Why this pick</p><div class=\"reading\">{paragraphs}</div></section>"
        )
    if columns:
        blocks.append(f"<section><div class=\"narrative\">{''.join(columns)}</div></section>")
    return "".join(blocks)


def _report_slug(name):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "team"


def _pick_context_sentence(winner, loser, win_prob, upset):
    if win_prob < 0.55:
        sentence = f"This is a toss-up. A {loser} win would be a normal result."
    elif win_prob < 0.65:
        sentence = f"{winner} is a slight favorite, and {loser} still wins this matchup often."
    elif win_prob < 0.80:
        sentence = f"{winner} is a clear favorite."
    else:
        sentence = f"{winner} is a strong favorite, by a comfortable margin."
    if upset:
        sentence += f" It is an upset pick because {winner} is clearly the lower-ranked team."
    return sentence


def _postseason_is_set(standing_a, standing_b):
    return bool((standing_a or {}).get("seed") or (standing_b or {}).get("seed"))


def _rank_line(standing, postseason):
    season = _season_name(ACTIVE_SEASON_YEAR) if ACTIVE_SEASON_YEAR else "this"
    if not standing:
        return "Ranking unavailable"
    if postseason:
        seed = standing.get("seed")
        if seed:
            return f"{season} NCAA tournament, {seed} seed"
        return f"Not in the {season} NCAA tournament"
    rank = standing.get("ap_rank")
    if not rank:
        return f"Unranked in the {season} AP poll"
    label = "Final AP" if standing.get("ap_is_final") else "AP"
    return f"{label} #{rank} in the {season} poll"


def _is_upset_pick(winner, loser, standing_winner, standing_loser, postseason):
    winner_standing = standing_winner or {}
    loser_standing = standing_loser or {}
    if postseason:
        winner_seed = winner_standing.get("seed")
        loser_seed = loser_standing.get("seed")
        if winner_seed and loser_seed:
            return winner_seed - loser_seed >= 3
        return winner_seed is None and loser_seed is not None
    winner_rank = winner_standing.get("ap_rank")
    loser_rank = loser_standing.get("ap_rank")
    if loser_rank is not None and winner_rank is None:
        return True
    if winner_rank is not None and loser_rank is not None:
        return winner_rank - loser_rank >= 10
    return False


def build_matchup_page(team_a, team_b, result, commentary, profile_a, profile_b, full_df, percentile_tables, venue_text=""):
    row_a = full_df.loc[full_df["Team"] == team_a].iloc[0]
    row_b = full_df.loc[full_df["Team"] == team_b].iloc[0]
    prob_a = float(result["win_prob_a"])
    prob_b = 1.0 - prob_a
    winner = result["winner"]
    pick_a = "<p class=\"pick\">Model pick</p>" if winner == team_a else ""
    pick_b = "<p class=\"pick\">Model pick</p>" if winner == team_b else ""
    tags_a = commentary["winner_tags"] if winner == team_a else commentary["loser_tags"]
    tags_b = commentary["winner_tags"] if winner == team_b else commentary["loser_tags"]
    flex_a = max(prob_a, 0.02)
    flex_b = max(prob_b, 0.02)
    color_a, color_b = _team_color_pair(team_a, team_b)
    standing_a = _TEAM_STANDINGS.get(_standing_key(team_a))
    standing_b = _TEAM_STANDINGS.get(_standing_key(team_b))
    postseason = _postseason_is_set(standing_a, standing_b)
    rank_a = _rank_line(standing_a, postseason)
    rank_b = _rank_line(standing_b, postseason)
    winner_standing = standing_a if result["winner"] == team_a else standing_b
    loser_standing = standing_b if result["winner"] == team_a else standing_a
    upset = _is_upset_pick(result["winner"], result["loser"], winner_standing, loser_standing, postseason)
    pick_context = _pick_context_sentence(result["winner"], result["loser"], float(result["win_prob"]), upset)
    favored_by = f"{result['winner']} is favored by {float(result['projected_margin']):.1f} points."
    winner_chance = f"{result['winner']} wins {float(result['win_prob']) * 100:.1f}% of the time in the model."
    home_court_line = _home_court_sentence(result, team_a, team_b)
    blend_lines = "".join(
        f'<p class="meta">{_esc(line)}</p>' for line in (blend_note(team_a), blend_note(team_b)) if line
    )
    body = f"""
    <header class="top">
      <div>
        <p class="kicker">Matchup report</p>
        <p class="meta">{_esc(_season_line())}</p>
        {blend_lines}
      </div>
      <p class="meta">Season offense and defense, weighted by past games</p>
    </header>
    <section class="hero">
      <div class="side-a">
        <p class="side">First team</p>
        <h1>{_esc(team_a)}</h1>
        <p class="rank">{_esc(rank_a)}</p>
        {pick_a}
      </div>
      <div class="verdict">
        <p class="kicker">Projected margin</p>
        <p class="spread">{_esc(result['projected_spread_text'].replace(' -', ' −'))}</p>
        {f'<p class="venue">{_esc(venue_text)}</p>' if venue_text else ''}
        <p class="prob">{_esc(favored_by)} {_esc(winner_chance)} {_esc(pick_context)}</p>
        <div class="split" role="img" aria-label="{_esc(team_a)} {prob_a * 100:.1f} percent, {_esc(team_b)} {prob_b * 100:.1f} percent">
          <div class="seg a" style="flex:{flex_a:.4f} 1 0"></div>
          <div class="seg b" style="flex:{flex_b:.4f} 1 0"></div>
        </div>
        <div class="split-labels">
          <span>{_esc(team_a)} wins {prob_a * 100:.1f}% of the time</span>
          <span>{_esc(team_b)} wins {prob_b * 100:.1f}% of the time</span>
        </div>
        <p class="sub">{_esc(home_court_line)} The game is projected at {result['expected_possessions']:.1f} possessions.</p>
      </div>
      <div class="side-b">
        <p class="side">Second team</p>
        <h1>{_esc(team_b)}</h1>
        <p class="rank">{_esc(rank_b)}</p>
        {pick_b}
      </div>
    </section>
    {_factor_html(team_a, team_b, result)}
    {_comparison_html(team_a, team_b, row_a, row_b, full_df, percentile_tables)}
    <section>
      <p class="section-label">Teams</p>
      <div class="teams">
        {_team_card(team_a, "a", profile_a, tags_a)}
        {_team_card(team_b, "b", profile_b, tags_b)}
      </div>
    </section>
    {_notes_html(commentary)}
    <p class="footer">Team stats come from Sports-Reference. The margin uses separate offense and defense weights for shooting, turnovers, rebounding, and free throws, plus home court and strength of schedule. Those weights were fit on college games before 2026. Running a matchup does not train the model again. Bar colors are school colors when the program knows them.</p>
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(team_a)} vs {_esc(team_b)}</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>
  <main class="sheet" style="--a:{color_a};--b:{color_b}">
    {_data_source_tag()}
    {body}
  </main>
</body>
</html>
"""


def show_matchup_report(
    team_a,
    team_b,
    result,
    commentary,
    profile_a,
    profile_b,
    full_df,
    percentile_tables,
    open_browser=True,
    venue_text="",
):
    page = build_matchup_page(
        team_a,
        team_b,
        result,
        commentary,
        profile_a,
        profile_b,
        full_df,
        percentile_tables,
        venue_text=venue_text,
    )
    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{_report_slug(team_a)}_vs_{_report_slug(team_b)}.html"
    path.write_text(page, encoding="utf-8")

    print(f"\n{team_a} vs {team_b}")
    print(f"Pick: {result['projected_spread_text']}  ·  {result['win_prob'] * 100:.1f}% to win")
    print(f"Report: {path}")
    if open_browser:
        try:
            opened = webbrowser.open(path.as_uri())
        except Exception:
            opened = False
        if not opened:
            print("Open that file in a browser to read the full report.")
    print()
    return path


# =========================================================
# TEAM LOOKUP HELPERS
# =========================================================

def ask_site(team_a, team_b):
    """Return +1, -1, or 0 for team A home, team B home, or neutral."""
    print("Where is it played?")
    print(f"  1  {team_a} home")
    print(f"  2  {team_b} home")
    print("  3  Neutral site")
    while True:
        raw = input("Choice [3]: ").strip().lower()
        if raw in {"", "3", "neutral", "neutral site", "n"}:
            return 0
        if raw in {"1", team_a.lower(), f"{team_a.lower()} home"}:
            return 1
        if raw in {"2", team_b.lower(), f"{team_b.lower()} home"}:
            return -1
        print("Enter 1, 2, 3, or press Enter for a neutral site.")


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

    print(
        f"Historical model training: {TRAINING_START_SEASON}-{TRAINING_END_SEASON} "
        "pre-game NCAA data."
    )
    if ACTIVE_SEASON_YEAR is not None:
        print(
            f"Live prediction inputs: {ACTIVE_SEASON_YEAR} team stats up to "
            f"{date.today().isoformat()}."
        )
    else:
        print("Live prediction inputs: LOCAL CSV FALLBACK (check freshness).")

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
            added = [team for team in (team_a, team_b) if team not in BLEND_NOTES]
            for team in added:
                apply_roster_starting_point(full_df, team)
            if added:
                baselines, percentile_tables = build_baselines(full_df)
            for team in (team_a, team_b):
                if blend_note(team):
                    print(blend_note(team))
            location = ask_site(team_a, team_b)
            played_at = venue_line(location, team_a, team_b)

            result = compute_matchup(
                team_a,
                team_b,
                full_df,
                baselines,
                percentile_tables,
                location=location,
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
            show_matchup_report(
                team_a,
                team_b,
                result,
                commentary,
                profile_a,
                profile_b,
                full_df,
                percentile_tables,
                venue_text=played_at,
            )

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()