from pathlib import Path

import pandas as pd

base_path = Path(r"C:\Users\Acer\Documents\Code-Main\Python\Football Machine\data\raw")

master_df: pd.DataFrame = pd.DataFrame()

for file in base_path.rglob("*.csv"):
    current_df = pd.read_csv(file)
    master_df = pd.concat([master_df, current_df], axis=0, ignore_index=True)


# print(master_df.shape)
# print(list(master_df.columns))
# print(master_df.columns.duplicated().sum())


drop_cols = [
    "Div",
    "Time",
    "B365H",
    "B365D",
    "B365A",
    "BWH",
    "BWD",
    "BWA",
    "IWH",
    "IWD",
    "IWA",
    "PSH",
    "PSD",
    "PSA",
    "WHH",
    "WHD",
    "WHA",
    "VCH",
    "VCD",
    "VCA",
    "MaxH",
    "MaxD",
    "MaxA",
    "AvgH",
    "AvgD",
    "AvgA",
    "B365>2.5",
    "B365<2.5",
    "P>2.5",
    "P<2.5",
    "Max>2.5",
    "Max<2.5",
    "Avg>2.5",
    "Avg<2.5",
    "AHh",
    "B365AHH",
    "B365AHA",
    "PAHH",
    "PAHA",
    "MaxAHH",
    "MaxAHA",
    "AvgAHH",
    "AvgAHA",
    "B365CH",
    "B365CD",
    "B365CA",
    "BWCH",
    "BWCD",
    "BWCA",
    "IWCH",
    "IWCD",
    "IWCA",
    "PSCH",
    "PSCD",
    "PSCA",
    "WHCH",
    "WHCD",
    "WHCA",
    "VCCH",
    "VCCD",
    "VCCA",
    "MaxCH",
    "MaxCD",
    "MaxCA",
    "AvgCH",
    "AvgCD",
    "AvgCA",
    "B365C>2.5",
    "B365C<2.5",
    "PC>2.5",
    "PC<2.5",
    "MaxC>2.5",
    "MaxC<2.5",
    "AvgC>2.5",
    "AvgC<2.5",
    "AHCh",
    "B365CAHH",
    "B365CAHA",
    "PCAHH",
    "PCAHA",
    "MaxCAHH",
    "MaxCAHA",
    "AvgCAHH",
    "AvgCAHA",
    "Referee",
]

master_df = master_df.drop(columns=drop_cols)
master_df = master_df.dropna()
cols = list(master_df.columns)


move_cols = ["FTHG", "FTAG", "FTR"]
for col in move_cols:
    cols.remove(col)
    cols.append(col)

master_df = master_df[cols]

master_df["Date"] = pd.to_datetime(master_df["Date"])
master_df = master_df.sort_values(by="Date").reset_index(drop=True)


# home_results = master_df[['Date', 'Home_Team', 'Home_Result']].rename(columns={'Home_Team': 'Team', 'Home_Result': 'Result'})
# away_results = master_df[['Date', 'Away_Team', 'Away_Result']].rename(columns={'Away_Team': 'Team', 'Away_Result': 'Result'})
# timeline = pd.concat([home_results, away_results]).sort_values(by='Date')


# # 3. Create a helper function to fetch a team's last 3 matches BEFORE the current date
# def get_past_form(row, team_col):
#     team = row[team_col]
#     current_date = row['Date']

#     # Filter for this team's matches strictly before today
#     past_matches = timeline[(timeline['Team'] == team) & (timeline['Date'] < current_date)]

#     # Grab the last 3 results and string them together (e.g., "W-L-W")
#     last_3 = past_matches.tail(3)['Result'].tolist()
#     return "-".join(last_3) if last_3 else "N/A"

# # 4. Apply directly to your original DataFrame
# df['Home_Form'] = df.apply(get_past_form, team_col='Home_Team', axis=1)
# df['Away_Form'] = df.apply(get_past_form, team_col='Away_Team', axis=1)


# ==========================================
# PHASE 1: CHRONOLOGICAL SETUP (YOUR CODE)
# ==========================================
master_df["Date"] = pd.to_datetime(master_df["Date"])
master_df = master_df.sort_values(by="Date").reset_index(drop=True)
master_df["Match_ID"] = master_df.index

home_side = master_df[["Match_ID", "Date", "HomeTeam", "FTR"]].rename(
    columns={"HomeTeam": "Team"}
)
home_side["Outcome"] = home_side["FTR"].map({"H": "W", "A": "L", "D": "D"})
home_side["Venue"] = "H"

away_side = master_df[["Match_ID", "Date", "AwayTeam", "FTR"]].rename(
    columns={"AwayTeam": "Team"}
)
away_side["Outcome"] = away_side["FTR"].map({"A": "W", "H": "L", "D": "D"})
away_side["Venue"] = "A"

timeline = pd.concat([home_side, away_side]).sort_values(by="Date")

# ==========================================
# PHASE 2: ML NUMERIC ENCODING
# ==========================================
# Map individual outcomes directly to numeric points for ML math
points_map = {"W": 3, "D": 1, "L": 0}
timeline["Points_Earned"] = timeline["Outcome"].map(points_map)

# Track individual game history positions (Match 1 = most recent, Match 3 = oldest)
timeline["M1_Pts"] = timeline.groupby("Team")["Points_Earned"].shift(1)
timeline["M2_Pts"] = timeline.groupby("Team")["Points_Earned"].shift(2)
timeline["M3_Pts"] = timeline.groupby("Team")["Points_Earned"].shift(3)

# Calculate the overall total form score out of 9
timeline["Total_Form_Score"] = timeline[["M1_Pts", "M2_Pts", "M3_Pts"]].sum(
    axis=1, min_count=1
)

# Handle early season games (fill missing history with 0 points)
cols_to_fill = ["M1_Pts", "M2_Pts", "M3_Pts", "Total_Form_Score"]
timeline[cols_to_fill] = timeline[cols_to_fill].fillna(0)

# ==========================================
# PHASE 3: MAPPING BACK TO ORIGINAL SCHEDULE
# ==========================================
# Separate timeline back into distinct Home and Away data lookup tables
home_timeline = timeline[timeline["Venue"] == "H"].set_index("Match_ID")
away_timeline = timeline[timeline["Venue"] == "A"].set_index("Match_ID")

# Map the numeric features back into your master training dataframe
master_df["Home_M1_Pts"] = master_df["Match_ID"].map(home_timeline["M1_Pts"])
master_df["Home_M2_Pts"] = master_df["Match_ID"].map(home_timeline["M2_Pts"])
master_df["Home_M3_Pts"] = master_df["Match_ID"].map(home_timeline["M3_Pts"])
master_df["Home_Form_Score"] = master_df["Match_ID"].map(
    home_timeline["Total_Form_Score"]
)

master_df["Away_M1_Pts"] = master_df["Match_ID"].map(away_timeline["M1_Pts"])
master_df["Away_M2_Pts"] = master_df["Match_ID"].map(away_timeline["M2_Pts"])
master_df["Away_M3_Pts"] = master_df["Match_ID"].map(away_timeline["M3_Pts"])
master_df["Away_Form_Score"] = master_df["Match_ID"].map(
    away_timeline["Total_Form_Score"]
)


print(master_df)
print(f"List of columns : {list(master_df.columns)}")
# print(f" Shape : {master_df.shape}")
# print(master_df.isnull().sum())
