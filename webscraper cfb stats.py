import cfbd
import pandas as pd
import time
from cfbd.rest import ApiException

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CFBD_API_KEY = "ZEqw4jpOI+JuOOCB3IClWZ/czZLe4JFr81stOSrlC4PoM0mthANbJYevdrmE+Fwi"
OUTPUT_CSV   = r"C:\Users\jcohen\Downloads\college_player_stats.csv"
YEARS        = list(range(2010, 2025))
POSITIONS    = ["QB", "RB", "WR", "TE"]

# ─── SET UP CFBD CLIENT ──────────────────────────────────────────────────────
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)

stats_api  = cfbd.StatsApi(api_client)
teams_api  = cfbd.TeamsApi(api_client)

all_seasons = []

for year in YEARS:
    print(f"\n⟳ Fetching stats for {year}…")
    try:
        raw_pass  = stats_api.get_player_season_stats(year=year, category="passing")
        raw_rush  = stats_api.get_player_season_stats(year=year, category="rushing")
        raw_recv  = stats_api.get_player_season_stats(year=year, category="receiving")
    except ApiException as e:
        print(f"  ⚠️  APIException on {year}: {e.status} {e.reason}")
        continue

    # DEBUG: peek at the very first two raw records
    print("  → passing[0:2]:", [r.to_dict() for r in raw_pass[:2]])
    print("  → rushing[0:2]:", [r.to_dict() for r in raw_rush[:2]])
    print("  → receiving[0:2]:", [r.to_dict() for r in raw_recv[:2]])

    # Convert lists of PlayerStat objects to DataFrame
    df_pass = pd.DataFrame([r.to_dict() for r in raw_pass])
    df_rush = pd.DataFrame([r.to_dict() for r in raw_rush])
    df_recv = pd.DataFrame([r.to_dict() for r in raw_recv])

    # Common key cols in each
    key_cols = ["playerId","player","team","conference","season"]

    def pivot_stats(df, prefix, mapping):
        """
        Pivot a long df of (statType, stat) into wide form,
        rename according to mapping, and drop unmapped stats.
        """
        df = df[key_cols + ["statType","stat"]].copy()
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        wide = (
            df
            .pivot_table(
                index=key_cols,
                columns="statType",
                values="stat",
                aggfunc="sum"
            )
            .reset_index()
        )
        # rename only the columns we care about
        rename_dict = {k: f"{prefix}_{v}" for k, v in mapping.items()}
        wide = wide.rename(columns=rename_dict)
        return wide

    # Define how each statType should be renamed
    pass_map = {
        "CMP": "completions",
        "ATT": "attempts",
        "YDS": "yards",
        "TD":  "touchdowns",
        "INT": "interceptions"
    }
    rush_map = {
        "ATT": "attempts",
        "YDS": "yards",
        "TD":  "touchdowns"
    }
    recv_map = {
        "REC": "receptions",
        "YDS": "yards",
        "TD":  "touchdowns"
    }

    dfp = pivot_stats(df_pass, "pass", pass_map)
    dfr = pivot_stats(df_rush, "rush", rush_map)
    dfe = pivot_stats(df_recv, "receiving", recv_map)

    # DEBUG: peek at the pivoted
    print("  → pass wide columns:", dfp.columns.tolist()[:10], "…")
    print("  → rush wide columns:", dfr.columns.tolist()[:10], "…")
    print("  → recv wide columns:", dfe.columns.tolist()[:10], "…")

    # Merge the three on the key columns
    merged = dfp.merge(dfr, on=key_cols, how="outer") \
                .merge(dfe, on=key_cols, how="outer") \
                .fillna(0)

    # Fetch roster once per year, map playerId → position
    try:
        roster = teams_api.get_roster(year=year)
    except ApiException as e:
        print(f"  ⚠️  Roster error {year}: {e.status}")
        continue

    pos_map = {str(p.id): p.position for p in roster}
    merged["position"] = merged["playerId"].astype(str).map(pos_map)
    merged = merged[merged["position"].isin(POSITIONS)]

    all_seasons.append(merged)
    time.sleep(1)

# concat & select
full = pd.concat(all_seasons, ignore_index=True)
wanted = [
"player","team","conference","position","season",
     "pass_completions","pass_attempts","pass_yards","pass_touchdowns","pass_interceptions",
     "rush_attempts","rush_yards","rush_touchdowns",
     "receiving_receptions","receiving_yards","receiving_touchdowns",

]
keep = [c for c in wanted if c in full.columns]
final_df = full[keep]

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(final_df)} rows to {OUTPUT_CSV}")
