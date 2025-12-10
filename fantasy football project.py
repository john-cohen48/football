# Complete updated script with fallback for must_pick_candidates

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === USER SETTING ===
your_draft_slot = 5  # Change this to your draft slot (1-10)

# === Load and Clean Data ===
history = pd.read_csv(r"C:\Users\jcohen\Downloads\fantasy_stats_2015_2024.csv")
big_board = pd.read_csv(r"C:\Users\jcohen\Downloads\big board 2025.csv")

history['PLAYER NAME'] = history['Player'].str.replace(r'[^\w\s]', '', regex=True).str.strip()
big_board['PLAYER NAME'] = big_board['PLAYER NAME'].str.replace(r'[^\w\s]', '', regex=True).str.strip()
big_board['POS'] = big_board['POS'].str.replace(r'\d+', '', regex=True).str.strip().str.upper()
big_board['POS'] = big_board['POS'].replace({'DEF': 'DST', 'D/ST': 'DST', 'DEFENSE': 'DST'})

merged = pd.merge(big_board, history, on='PLAYER NAME', how='left')
print("Columns in merged:", merged.columns.tolist())

# === Keep most recent season per player ===
merged = merged.sort_values(by='Year', ascending=False).drop_duplicates(subset='PLAYER NAME')

# === Filter out players with too few games ===
merged = merged[merged['G'] >= 4]

# === Remove outliers in PPR using IQR ===
Q1 = merged['PPR'].quantile(0.25)
Q3 = merged['PPR'].quantile(0.75)
IQR = Q3 - Q1
merged = merged[(merged['PPR'] >= Q1 - 1.5 * IQR) & (merged['PPR'] <= Q3 + 1.5 * IQR)]

# === Feature Engineering ===
merged['PPR_per_game'] = merged['PPR'] / merged['G']
features = ['Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Tgt', 'Rec', 'Y/R', 'Fmb', 'FL']
target = 'PPR_per_game'
model_data = merged[features + [target]].dropna()
X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (PPR/game): {rmse:.2f}")

# Predict full-season points assuming 17 games, then apply age penalty
def predict_total(row):
    if all(pd.notnull(row[f]) for f in features):
        pts = model.predict(pd.DataFrame([row[features]]))[0] * 17
        if row['Age'] >= 35:
            pts *= 0.80
        elif row['Age'] >= 30:
            pts *= 0.90
        return pts
    return None

merged['Predicted Fantasy Points'] = merged.apply(predict_total, axis=1)

# Adjust points by big board rank
def adjust_points(row):
    pts = row['Predicted Fantasy Points']
    if pd.isnull(pts): return None
    rank = row['RK']
    if rank <= 10: return pts * 1.10
    elif rank <= 20: return pts * 1.05
    elif rank <= 30: return pts * 0.95
    elif rank <= 40: return pts * 0.90
    else: return pts * 0.85

merged['Adjusted Fantasy Points'] = merged.apply(adjust_points, axis=1)

# === Rank Groups for each round ===
max_rank = int(merged['RK'].max())
rank_groups = {}
for start in range(1, max_rank + 1, 10):
    label = f"{start}-{start+9}"
    rank_groups[label] = []

for _, row in merged.iterrows():
    rank = row['RK']
    if pd.notnull(rank):
        key = f"{int((rank-1)//10*10+1)}-{int((rank-1)//10*10+10)}"
        if key in rank_groups:
            rank_groups[key].append(row)

round_to_rank_group = {i: grp for i, grp in enumerate(list(rank_groups.keys())[:15], start=1)}

position_limits = {'QB': 2, 'RB': 4, 'WR': 5, 'TE': 2, 'DST': 1, 'K': 1}
positions_count = {pos: 0 for pos in position_limits}
taken_players = set()
selected_players = []

# === Draft Simulator ===
round_num = 1
max_rounds = 30
while sum(positions_count.values()) < 15 and round_num <= max_rounds:
    your_pick = your_draft_slot if round_num % 2 == 1 else 11 - your_draft_slot
    draft_order = list(range(1, 11)) if round_num % 2 == 1 else list(range(10, 0, -1))
    rank_group = round_to_rank_group.get(round_num, None)

    for pick_num in draft_order:
        if sum(positions_count.values()) >= 15:
            break

        if pick_num != your_draft_slot:
            bot_pick = merged[~merged['PLAYER NAME'].isin(taken_players)]
            bot_pick = bot_pick.dropna(subset=['Adjusted Fantasy Points']).sort_values(by='RK').head(1)
            if not bot_pick.empty:
                taken_players.add(bot_pick.iloc[0]['PLAYER NAME'])
        else:
            # determine if must pick DST/K
            must_pick = None
            rem = 15 - sum(positions_count.values())
            if rem == 2 and positions_count['DST'] == 0:
                must_pick = 'DST'
            elif rem == 2 and positions_count['K'] == 0:
                must_pick = 'K'
            elif rem == 1 and positions_count['DST'] == 0:
                must_pick = 'DST'
            elif rem == 1 and positions_count['K'] == 0:
                must_pick = 'K'

            candidates = merged[~merged['PLAYER NAME'].isin(taken_players)]
            candidates = candidates[candidates['POS'].isin(position_limits)]
            candidates = candidates[candidates['POS'].map(lambda p: positions_count[p] < position_limits[p])]

            if must_pick:
                must_candidates = candidates[candidates['POS'] == must_pick]
                if must_candidates.empty:
                    # fallback to any available of that position
                    fallback = merged[
                        (merged['POS'] == must_pick) &
                        (~merged['PLAYER NAME'].isin(taken_players))
                    ]
                    if fallback.empty:
                        print(f"[Round {round_num}] No valid players for {must_pick}, skipping.")
                        continue
                    best_pick = fallback.sort_values(by='RK').iloc[0]
                else:
                    # pick by adjusted points if available else by rank
                    if must_candidates['Adjusted Fantasy Points'].notna().any():
                        best_pick = must_candidates.sort_values(
                            by='Adjusted Fantasy Points', ascending=False
                        ).iloc[0]
                    else:
                        best_pick = must_candidates.sort_values(by='RK').iloc[0]
            else:
                group = rank_groups.get(rank_group, [])
                group_avail = [p for p in group if p['PLAYER NAME'] not in taken_players and positions_count[p['POS']] < position_limits[p['POS']]]
                if group_avail:
                    best_pick = max(group_avail, key=lambda p: p['Adjusted Fantasy Points'] or 0)
                else:
                    best_pick = candidates.sort_values(
                        by='Adjusted Fantasy Points', ascending=False
                    ).iloc[0]

            # record pick
            selected_players.append({
                'Pick': f"Round {round_num} Pick {your_draft_slot}",
                'Round': round_num,
                'PLAYER NAME': best_pick['PLAYER NAME'],
                'POS': best_pick['POS'],
                'Adjusted Fantasy Points': best_pick['Adjusted Fantasy Points']
            })
            taken_players.add(best_pick['PLAYER NAME'])
            positions_count[best_pick['POS']] += 1

    round_num += 1

# === Results ===
final_draft = pd.DataFrame(selected_players)
print("\nYour Draft Picks:")
print(final_draft[['Pick', 'Round', 'PLAYER NAME', 'POS', 'Adjusted Fantasy Points']])
