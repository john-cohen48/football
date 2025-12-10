import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
college_df = pd.read_csv(r"C:\Users\jcohen\Downloads\college_player_stats.csv")
nfl_df = pd.read_csv(r"C:\Users\jcohen\Downloads\fantasy_stats_2015_2024.csv")
big_board_2025 = pd.read_csv(r"C:\Users\jcohen\Downloads\big board 2025.csv")

# === Clean Column Names ===
college_df.columns = college_df.columns.str.upper().str.strip()
nfl_df.columns = nfl_df.columns.str.upper().str.strip()
big_board_2025.columns = big_board_2025.columns.str.upper().str.strip()

# === Normalize Names ===
def normalize_name(name):
    return re.sub(r"[^A-Z]", "", str(name).upper())

college_df["PLAYER_ID"] = college_df["PLAYER"].apply(normalize_name)
nfl_df["PLAYER_ID"] = nfl_df["PLAYER"].apply(normalize_name)
big_board_2025["PLAYER_ID"] = big_board_2025["PLAYER NAME"].apply(normalize_name)

# === Aggregate NFL Fantasy Points ===
nfl_df_filtered = nfl_df[["PLAYER_ID", "FANTPOS", "PPR"]].dropna(subset=["PPR"])
nfl_agg = nfl_df_filtered.groupby(["PLAYER_ID", "FANTPOS"]).agg(CAREER_PPR=("PPR", "sum")).reset_index()

# === Aggregate College Stats ===
offensive_positions = ["QB", "RB", "WR", "TE"]
college_df = college_df[college_df["POSITION"].isin(offensive_positions)]
college_agg = college_df.groupby(["PLAYER_ID", "POSITION"]).agg({
    "PASS_YARDS": "sum",
    "PASS_TOUCHDOWNS": "sum",
    "PASS_INTERCEPTIONS": "sum",
    "RUSH_YARDS": "sum",
    "RUSH_TOUCHDOWNS": "sum",
    "RECEIVING_RECEPTIONS": "sum",
    "RECEIVING_YARDS": "sum",
    "RECEIVING_TOUCHDOWNS": "sum"
}).reset_index()

# === Merge College and NFL ===
merged_df = pd.merge(
    college_agg,
    nfl_agg,
    left_on=["PLAYER_ID", "POSITION"],
    right_on=["PLAYER_ID", "FANTPOS"],
    how="inner"
).drop(columns=["FANTPOS"])

# === Train Models by Position ===
models = {}
metrics = {}

for pos in offensive_positions:
    pos_df = merged_df[merged_df["POSITION"] == pos].copy()
    X = pos_df.drop(columns=["PLAYER_ID", "POSITION", "CAREER_PPR"])
    y = pos_df["CAREER_PPR"]
    X.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    models[pos] = model
    metrics[pos] = {"RMSE": rmse, "MAE": mae, "R2": r2}

for pos in metrics:
    print(f"--- {pos} Model ---")
    print(f"RMSE: {metrics[pos]['RMSE']:.2f}")
    print(f"MAE : {metrics[pos]['MAE']:.2f}")
    print(f"R²  : {metrics[pos]['R2']:.4f}")
    print()


metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Position"})

# === Helper to Split Names ===
def split_name(full_name):
    parts = str(full_name).strip().split()
    return parts[0], parts[-1] if len(parts) >= 2 else ("", "")

big_board_2025["IS_ROOKIE"] = ~big_board_2025["PLAYER_ID"].isin(merged_df["PLAYER_ID"])
big_board_2025["FIRST_NAME"], big_board_2025["LAST_NAME"] = zip(*big_board_2025["PLAYER NAME"].map(split_name))

# === Predict 2025 Rookies ===
rookie_ids = set(big_board_2025[big_board_2025["IS_ROOKIE"]]["PLAYER_ID"])
rookie_features = college_agg[college_agg["PLAYER_ID"].isin(rookie_ids)].copy()

rookie_predictions = []
for pos in offensive_positions:
    pos_rookies = rookie_features[rookie_features["POSITION"] == pos].copy()
    if pos_rookies.empty:
        continue
    X = pos_rookies.drop(columns=["PLAYER_ID", "POSITION"])
    X.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = models[pos]
    y_pred = model.predict(X_scaled)
    temp_df = pos_rookies[["PLAYER_ID"]].copy()
    temp_df["Position"] = pos
    temp_df["Predicted_Career_PPR"] = y_pred
    rookie_predictions.append(temp_df)

rookie_predictions_df = pd.concat(rookie_predictions, ignore_index=True)
rookie_predictions_df = pd.merge(
    rookie_predictions_df,
    big_board_2025[["PLAYER_ID", "PLAYER NAME", "FIRST_NAME", "LAST_NAME"]],
    on="PLAYER_ID",
    how="left"
)

# === Get Best Predicted Rookie per Position ===
best_rookies = rookie_predictions_df.sort_values(["Position", "Predicted_Career_PPR"], ascending=[True, False])
best_rookies_top1 = best_rookies.groupby("Position").first().reset_index()
best_rookies_top1 = pd.merge(best_rookies_top1, metrics_df, on="Position", how="left")

# === Save to CSV ===
best_rookies_top1.to_csv(r"C:\Users\jcohen\Downloads\best_predicted_rookies_2025.csv", index=False)

# === Optional: Display or Print Results ===
print(best_rookies_top1)
