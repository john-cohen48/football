import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── suppress sklearn “invalid feature names” warnings ────────────────────────
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# === USER SETTINGS ===
your_draft_slot = 3   # 1–10 in a 10-team snake draft
NUM_TEAMS      = 10
TOTAL_PICKS    = 15

# Age cutoffs by position for YOUR picks
AGE_CUTOFFS = {
    'QB': 36,
    'RB': 32,
    'WR': 35,
    'TE': 34,
    'FB': 30
}

# === 1) LOAD & CLEAN DATA ===
history   = pd.read_csv(r"C:\Users\jcohen\Downloads\fantasy_stats_2015_2024.csv")
big_board = pd.read_csv(r"C:\Users\jcohen\Downloads\big board 2025.csv")

# Normalize column names
for df in (history, big_board):
    df.columns = df.columns.str.strip().str.upper()

# Normalize player names
history['PLAYER NAME']   = history['PLAYER'].str.replace(r"[^\w\s]", "", regex=True).str.strip()
big_board['PLAYER NAME'] = big_board['PLAYER NAME'].str.replace(r"[^\w\s]", "", regex=True).str.strip()

# Standardize POS
big_board['POS'] = (
    big_board['POS']
      .str.replace(r"\d+", "", regex=True)
      .str.strip()
      .str.upper()
      .replace({
        'DEF':'DST','D/ST':'DST','DEFENSE':'DST',
        'PK':'K','KICKER':'K'
      })
)

# === 2) AGGREGATE LAST 3 SEASONS & EXTRACT 2024 AGE+1 ===
# sort so that first row per player is their 2024 season
hist_sorted = history.sort_values(['PLAYER NAME','YEAR'], ascending=[True, False])

# take up to 3 most recent seasons for volume stats
hist_top3   = hist_sorted.groupby('PLAYER NAME').head(3)

# aggregate volume & rate stats over those 3 seasons
agg_stats = (
    hist_top3
      .groupby('PLAYER NAME')
      .agg({
        'G':    'sum',
        'CMP':  'sum',
        'ATT':  'sum',
        'YDS':  'sum',
        'TD':   'sum',
        'INT':  'sum',
        'TGT':  'sum',
        'REC':  'sum',
        'Y/R':  'mean',
        'FMB':  'sum',
        'FL':   'sum',
        'PPR':  'sum'
      })
      .rename(columns={'PPR':'PPR_TOTAL','Y/R':'Y_R_MEAN'})
)

# pull each player’s 2024 age, then add 1 for 2025
hist_latest = (
    hist_sorted
      .groupby('PLAYER NAME')
      .first()[['AGE']]
      .rename(columns={'AGE':'AGE_2025'})
)
hist_latest['AGE_2025'] += 1

# combine them
agg = agg_stats.join(hist_latest, how='left')
agg['PPR_PER_GAME'] = agg['PPR_TOTAL'] / agg['G']


# === 3) MERGE & APPLY AGE CUTS ===
merged = pd.merge(big_board, agg, on='PLAYER NAME', how='left')

# keep only non-K/DST, players with ≥4 games, and under your cutoff
merged = (
    merged[~merged['POS'].isin(['K','DST'])]
          .query("G >= 4")
          .loc[lambda df: df.apply(
              lambda r: r['AGE_2025'] <= AGE_CUTOFFS.get(r['POS'], 99),
              axis=1
          )]
)

# prepare features & target
features = ['AGE_2025','G','CMP','ATT','YDS','TD','INT',
            'TGT','REC','Y_R_MEAN','FMB','FL']
df2 = merged.dropna(subset=features + ['PPR_PER_GAME']).copy()
X, y = df2[features], df2['PPR_PER_GAME']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4) TRAIN & EVALUATE ===
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    max_features=0.5,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
mean_ppr  = y_test.mean()
pct_error = rmse / mean_ppr * 100

print(f"RMSE (PPR/game): {rmse:.2f}")
print(f"Mean PPR/game (test): {mean_ppr:.2f}")
print(f"Pct Error: {pct_error:.1f}%")

# === 5) PREDICT & ADJUST ===
merged['PRED_PPR_PER_GAME']   = merged.apply(
    lambda r: model.predict([r[features].values])[0],
    axis=1
)
merged['PRED_FANTASY_POINTS'] = merged['PRED_PPR_PER_GAME'] * merged['G']
merged['ADJ_FANTASY_POINTS']  = merged.apply(
    lambda r: r['PRED_FANTASY_POINTS'] * (
        1+0.10 if r['RK']<=10 else
        1+0.05 if r['RK']<=20 else
        1-0.05 if r['RK']<=30 else
        1-0.10
    ),
    axis=1
)

# build fallback board from these same age-filtered players
fallback_board = merged[['PLAYER NAME','POS','RK']].copy()
kd_board      = fallback_board[fallback_board['POS'].isin(['K','DST'])]

# === 6) DRAFT SIMULATION ===
pos_limits = {'QB':2,'RB':4,'WR':5,'TE':2,'DST':1,'K':1}
counts     = {p:0 for p in pos_limits}
taken      = set()
picks      = []

round_num = 1
while len(picks) < TOTAL_PICKS and round_num <= 30:
    order = (list(range(1,NUM_TEAMS+1)) if round_num%2==1
             else list(range(NUM_TEAMS,0,-1)))
    for team_pick in order:
        if len(picks)>=TOTAL_PICKS: break

        if team_pick != your_draft_slot:
            other = big_board[~big_board['PLAYER NAME'].isin(taken)]
            if not other.empty:
                choice = other.sort_values('RK').iloc[0]
                taken.add(choice['PLAYER NAME'])
            continue

        rem = TOTAL_PICKS - len(picks)
        must = None
        if rem==2:
            if counts['K']==0:    must='K'
            elif counts['DST']==0: must='DST'
        elif rem==1:
            if counts['DST']==0:  must='DST'
            elif counts['K']==0:   must='K'

        if must in ('K','DST'):
            avail = kd_board[~kd_board['PLAYER NAME'].isin(taken)]
            if avail.empty: continue
            choice = avail[avail['POS']==must].sort_values('RK').iloc[0]
        else:
            ml_avail = merged[~merged['PLAYER NAME'].isin(taken)]
            ml_avail = ml_avail[ml_avail['POS'].map(lambda p: counts[p] < pos_limits[p])]
            ml_avail = ml_avail.dropna(subset=['ADJ_FANTASY_POINTS'])
            if not ml_avail.empty:
                choice = ml_avail.sort_values('ADJ_FANTASY_POINTS', ascending=False).iloc[0]
            else:
                fb = fallback_board[~fallback_board['PLAYER NAME'].isin(taken)]
                fb = fb[fb['POS'].map(lambda p: counts[p] < pos_limits[p])]
                if fb.empty: continue
                choice = fb.sort_values('RK').iloc[0]

        picks.append({
            'Round': round_num,
            'Pick':  f"Round {round_num} Pick {your_draft_slot}",
            'PLAYER NAME': choice['PLAYER NAME'],
            'POS': choice['POS'],
            'Adjusted Fantasy Points':
                merged.loc[merged['PLAYER NAME']==choice['PLAYER NAME'],
                           'ADJ_FANTASY_POINTS'].iloc[0]
        })
        taken.add(choice['PLAYER NAME'])
        counts[choice['POS']] += 1

    round_num += 1

# === 7) OUTPUT ===
draft_df = pd.DataFrame(picks)
print("\nYour Draft Picks:")
print(draft_df[['Pick','Round','PLAYER NAME','POS','Adjusted Fantasy Points']])
