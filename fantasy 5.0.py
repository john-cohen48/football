import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# === SETTINGS ===
your_draft_slot = 7
num_teams = 10
total_picks = 15
age_cutoffs = {'QB':36,'RB':32,'WR':35,'TE':34,'FB':30}
pos_limits = {'QB':2,'RB':4,'WR':5,'TE':2,'K':1,'DST':1}

# === LOAD DATA ===
history = pd.read_csv(r"C:\Users\jcohen\Downloads\fantasy_stats_2015_2024.csv")
big_board = pd.read_csv(r"C:\Users\jcohen\Downloads\big board 2025.csv")

for df in (history, big_board):
    df.columns = df.columns.str.strip().str.upper()

history['PLAYER NAME'] = history['PLAYER'].str.replace(r"[^\w\s]", "", regex=True).str.strip()
big_board['PLAYER NAME'] = big_board['PLAYER NAME'].str.replace(r"[^\w\s]", "", regex=True).str.strip()
big_board['POS'] = big_board['POS'].str.replace(r"\d+", "", regex=True).str.strip().str.upper().replace({
    'DEF':'DST','D/ST':'DST','DEFENSE':'DST','PK':'K','KICKER':'K'
})

# === AGGREGATE LAST 3 SEASONS ===
hist_sorted = history.sort_values(['PLAYER NAME','YEAR'], ascending=[True, False])
hist_top3 = hist_sorted.groupby('PLAYER NAME').head(3)

feature_agg = hist_top3.groupby('PLAYER NAME').agg({
    'G':'sum','CMP':'sum','ATT':'sum','YDS':'sum','TD.3':'sum','INT':'sum',
    'TGT':'sum','REC':'sum','Y/R':'mean','FMB':'sum','FL':'sum'
}).rename(columns={'Y/R':'Y_R_MEAN','TD.3':'TD_TOTAL'})

ppr_agg = hist_top3.groupby('PLAYER NAME').agg({
    'YDS':'sum','TD':'sum','YDS.1':'sum','TD.1':'sum','REC':'sum',
    'YDS.2':'sum','TD.2':'sum'
}).rename(columns={
    'YDS':'PASS_YDS','TD':'PASS_TD','YDS.1':'RUSH_YDS','TD.1':'RUSH_TD',
    'REC':'REC_REC','YDS.2':'REC_YDS','TD.2':'REC_TD'
}).fillna(0)

ppr_agg['PPR_TOTAL'] = (
    ppr_agg['REC_REC'] + ppr_agg['REC_YDS']/10 + ppr_agg['REC_TD']*6 +
    ppr_agg['RUSH_YDS']/10 + ppr_agg['RUSH_TD']*6 +
    ppr_agg['PASS_YDS']/25 + ppr_agg['PASS_TD']*4
)

hist_latest = hist_sorted.groupby('PLAYER NAME').first()[['AGE']].rename(columns={'AGE':'AGE_2025'})
hist_latest['AGE_2025'] += 1

agg = feature_agg.join(ppr_agg['PPR_TOTAL']).join(hist_latest)
agg['PPR_PER_GAME'] = agg['PPR_TOTAL'] / agg['G']
agg = agg.join(ppr_agg[['PASS_YDS','RUSH_YDS','REC_YDS','PASS_TD','RUSH_TD','REC_TD']])
agg['PASS_YDS_PER_G'] = agg['PASS_YDS'] / agg['G']
agg['RUSH_YDS_PER_G'] = agg['RUSH_YDS'] / agg['G']
agg['REC_YDS_PER_G']  = agg['REC_YDS'] / agg['G']

# === MODEL ===
merged = pd.merge(big_board, agg, on='PLAYER NAME', how='left')
merged = (
    merged[~merged['POS'].isin(['K','DST'])]
          .query("G >= 4")
          .loc[lambda df: df.apply(
              lambda r: r['AGE_2025'] <= age_cutoffs.get(r['POS'], 99),
              axis=1
          )]
)

features = [
    'AGE_2025','G','CMP','ATT','PASS_YDS_PER_G','RUSH_YDS_PER_G','REC_YDS_PER_G',
    'TD_TOTAL','PASS_TD','RUSH_TD','REC_TD','INT','TGT','REC','Y_R_MEAN','FMB','FL'
]
df2 = merged.dropna(subset=features + ['PPR_PER_GAME']).copy()
X, y = df2[features], df2['PPR_PER_GAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=500, max_depth=10, max_features=0.5,
    min_samples_split=5, min_samples_leaf=1, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
mean_ppr = y_test.mean()
pct_err  = rmse / mean_ppr * 100

print(f"RMSE (PPR/game): {rmse:.2f}")
print(f"Mean PPR/game (test): {mean_ppr:.2f}")
print(f"Pct Error: {pct_err:.1f}%")

# === PREDICT & CREATE FINAL BOARD ===
merged['PRED_PPR_PER_GAME'] = merged.apply(lambda r: model.predict([r[features].values])[0], axis=1)
merged['PRED_FANTASY_POINTS'] = merged['PRED_PPR_PER_GAME'] * merged['G']
merged['ADJ_FANTASY_POINTS'] = merged.apply(
    lambda r: r['PRED_FANTASY_POINTS'] * (
        1+0.10 if r['RK'] <= 10 else
        1+0.05 if r['RK'] <= 20 else
        1-0.05 if r['RK'] <= 30 else
        1-0.10
    ), axis=1
)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import matplotlib
matplotlib.use('Qt5Agg')  # ✅ Set backend before importing pyplot
import matplotlib.pyplot as plt
from collections import defaultdict

# === Load data ===
nfl_df = pd.read_csv(r"C:\Users\jcohen\Downloads\fantasy_stats_2015_2024.csv")
cfb_df = pd.read_csv(r"C:\Users\jcohen\Downloads\college_player_stats.csv")
big_board_df = pd.read_csv(r"C:\Users\jcohen\Downloads\big board 2025.csv")

# === Standardize and clean names ===
nfl_df["Player"] = nfl_df["Player"].str.replace(r"[^\w\s]", "", regex=True).str.strip().str.upper()
cfb_df["player"] = cfb_df["player"].str.replace(r"[^\w\s]", "", regex=True).str.strip().str.upper()
big_board_df["PLAYER NAME"] = big_board_df["PLAYER NAME"].str.replace(r"[^\w\s]", "", regex=True).str.strip().str.upper()

# === Identify rookies ===
nfl_players = set(nfl_df["Player"])
rookies_df = cfb_df[~cfb_df["player"].isin(nfl_players)].copy()
rookies_df = rookies_df.sort_values("season").groupby("player").tail(1)
big_board_df = big_board_df.rename(columns={"TEAM": "nfl_team"})
rookies_data = rookies_df.merge(big_board_df, left_on="player", right_on="PLAYER NAME", how="inner")

# === Prepare training data ===
nfl_with_cfb = nfl_df[nfl_df["Player"].isin(cfb_df["player"])].copy()
nfl_with_cfb["FirstYear"] = nfl_with_cfb.groupby("Player")["Year"].transform("min")
nfl_with_cfb = nfl_with_cfb.rename(columns={"Tm": "nfl_team"})
rookie_nfl_stats = nfl_with_cfb[nfl_with_cfb["Year"] == nfl_with_cfb["FirstYear"]]
cfb_latest = cfb_df.sort_values("season").groupby("player").tail(1)
train_data = rookie_nfl_stats.merge(cfb_latest, left_on="Player", right_on="player", how="inner")

# === Feature engineering ===
def compute_score(row):
    if row["position"] == "QB":
        return row["pass_yards"] + 50 * row["pass_touchdowns"] - 100 * row["pass_interceptions"]
    elif row["position"] == "RB":
        return row["rush_yards"] + row["receiving_yards"] + 20 * (row["rush_touchdowns"] + row["receiving_touchdowns"])
    else:
        return row["receiving_yards"] + 20 * row["receiving_touchdowns"]

train_data["PerfScore"] = train_data.apply(compute_score, axis=1)
train_data["Rank"] = train_data.groupby("Year")["PerfScore"].rank(method="dense", ascending=False)

# === Compute fantasy points manually (PPR scoring) ===
train_data["PPR_TOTAL"] = (
    train_data["receiving_receptions"] * 1 +
    train_data["receiving_yards"] / 10 +
    train_data["receiving_touchdowns"] * 6 +
    train_data["rush_yards"] / 10 +
    train_data["rush_touchdowns"] * 6 +
    train_data["pass_yards"] / 25 +
    train_data["pass_touchdowns"] * 4
)

# === Select features ===
num_features = [
    "pass_attempts", "pass_yards", "pass_touchdowns", "pass_interceptions",
    "rush_yards", "rush_touchdowns",
    "receiving_receptions", "receiving_yards", "receiving_touchdowns"
]
cat_features = ["position", "conference", "nfl_team"]

# Drop rows with missing features for training
train_data = train_data.dropna(subset=num_features + cat_features)

# Build matrices
X_train = train_data[num_features + cat_features].copy()
y_train = train_data["PPR_TOTAL"]
X_test = rookies_data[num_features + cat_features].copy()

# Encode categorical features
X_train = pd.get_dummies(X_train, columns=cat_features, prefix=cat_features)
X_test = pd.get_dummies(X_test, columns=cat_features, prefix=cat_features)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# === Train and evaluate RandomForest ===
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae = -cross_val_score(best_model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
r2 = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='r2').mean()

print("RandomForest Cross-Validation Results Rookies:")
print(f"MAE = {round(mae, 2)}, R² = {round(r2, 2)}")

# === Train model and predict ===
best_model.fit(X_train, y_train)
rookies_data["PredictedFantPt"] = best_model.predict(X_test)

# === Output ===
# === Prepare output with RK included ===
output = rookies_data[["PLAYER NAME", "nfl_team", "POS", "PredictedFantPt", "RK"]]
output = output.sort_values("PredictedFantPt", ascending=False).reset_index(drop=True)

# === Add additional fields to match veterans ===
output['G'] = 17  # Assume full season
output['AGE_2025'] = 22  # Default rookie age
output['PRED_PPR_PER_GAME'] = output['PredictedFantPt'] / output['G']
output['PRED_FANTASY_POINTS'] = output['PredictedFantPt']

# Adjust fantasy points based on rank
def adjust_points(row):
    if row['RK'] <= 10:
        return row['PredictedFantPt'] * 1.10
    elif row['RK'] <= 20:
        return row['PredictedFantPt'] * 1.05
    elif row['RK'] <= 30:
        return row['PredictedFantPt'] * 0.95
    else:
        return row['PredictedFantPt'] * 0.90

output['ADJ_FANTASY_POINTS'] = output.apply(adjust_points, axis=1)
output['POS'] = output['POS'].str.replace(r"\d+", "", regex=True).str.strip().str.upper()
output.rename(columns={'nfl_team': 'TEAM'}, inplace=True)

# Ensure column names match
rookie_columns = ["PLAYER NAME", "POS", "TEAM", "G", "AGE_2025", "RK", "PRED_PPR_PER_GAME", "PRED_FANTASY_POINTS", "ADJ_FANTASY_POINTS"]
rookies_formatted = output[rookie_columns]
common_cols = list(set(merged.columns) & set(rookies_formatted.columns))
merged = pd.concat([merged, rookies_formatted[common_cols]], ignore_index=True)




# === FINAL BIG BOARD SORTING & RANKING ===
final_board = merged.sort_values(
    by=['RK', 'PRED_PPR_PER_GAME', 'PLAYER NAME'],
    ascending=[True, False, True]
).reset_index(drop=True)

final_board['NEW_RANK'] = final_board.index + 1

final_board.to_csv("C:/Users/jcohen/Downloads/final_big_board_with_predictions.csv", index=False)

# === DRAFT SIMULATION BASED ON NEW RANK ===
taken = set()
counts = {p: 0 for p in pos_limits}
draft_picks = []

i = your_draft_slot - 1
while len(draft_picks) < total_picks and i < len(final_board):
    row = final_board.iloc[i]
    name, pos = row['PLAYER NAME'], row['POS']
    if name not in taken and counts[pos] < pos_limits.get(pos, 0):
        draft_picks.append({
            'Pick': len(draft_picks)+1,
            'PLAYER NAME': name,
            'POS': pos,
            'ADJ_FANTASY_POINTS': row['ADJ_FANTASY_POINTS']
        })
        taken.add(name)
        counts[pos] += 1
    i += num_teams

draft_df = pd.DataFrame(draft_picks)
print("\nYour Draft Picks:")
print(draft_df)
