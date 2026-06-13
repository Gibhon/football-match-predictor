from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def combine_df(base_path=BASE_DIR / "data" / "raw") -> pd.DataFrame:
    raw_dfs: pd.DataFrame = [pd.read_csv(file) for file in base_path.rglob("*.csv")]
    if not raw_dfs:
        raise FileNotFoundError(f"No CSV files found in {base_path}")
    return pd.concat(raw_dfs, ignore_index=True)


def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Dropping unnecessary columns and rows with na.
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
    clean_df = raw_df.drop(columns=drop_cols)
    clean_df = clean_df.dropna()
    return clean_df


def modify_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    df = clean_df.sort_values("Date").reset_index(drop=True)
    df["Match_id"] = df.index

    home_side = df[["Match_id", "Date", "HomeTeam", "FTR", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GoalsScored", "FTAG": "GoalsConceded"}
    )
    home_side["Outcome"] = home_side["FTR"].map({"H": "W", "A": "L", "D": "D"})
    home_side["Venue"] = "H"

    away_side = df[["Match_id", "Date", "AwayTeam", "FTR", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GoalsScored", "FTHG": "GoalsConceded"}
    )
    away_side["Outcome"] = away_side["FTR"].map({"A": "W", "H": "L", "D": "D"})
    away_side["Venue"] = "A"

    timeline = pd.concat([home_side, away_side]).sort_values(by="Date")
    timeline["PointsEarned"] = timeline["Outcome"].map({"W": 3, "L": 0, "D": 1})
    timeline["TotalWins"] = timeline["Outcome"].map({"W": 1, "L": 0, "D": 0})

    grouped_timeline = timeline.groupby("Team")
    shifts = [grouped_timeline["PointsEarned"].shift(i) for i in range(1, 6)]
    shifts_scored = [grouped_timeline["GoalsScored"].shift(i) for i in range(1, 6)]
    shifts_conceded = [grouped_timeline["GoalsConceded"].shift(i) for i in range(1, 6)]
    shifts_wins = [grouped_timeline["TotalWins"].shift(i) for i in range(1, 6)]

    timeline["TotalFormScore"] = (
        pd.concat(shifts, axis=1).sum(axis=1, min_count=1).fillna(0)
    )
    timeline["GoalsAVG"] = (
        (pd.concat(shifts_scored, axis=1).sum(axis=1, min_count=1)) / 5
    ).fillna(0)
    timeline["GoalsConcededAVG"] = (
        (pd.concat(shifts_conceded, axis=1).sum(axis=1, min_count=1)) / 5
    ).fillna(0)
    timeline["WinRate"] = (
        (pd.concat(shifts_wins, axis=1).sum(axis=1, min_count=1)) / 5
    ).fillna(0)

    home_timeline = timeline[timeline["Venue"] == "H"].set_index("Match_id")
    away_timeline = timeline[timeline["Venue"] == "A"].set_index("Match_id")

    df["HomeFormScore"] = df["Match_id"].map(home_timeline["TotalFormScore"])
    df["AwayFormScore"] = df["Match_id"].map(away_timeline["TotalFormScore"])
    df["HomeGoalsAVG"] = df["Match_id"].map(home_timeline["GoalsAVG"])
    df["AwayGoalsAVG"] = df["Match_id"].map(away_timeline["GoalsAVG"])
    df["HomeGoalsConcededAVG"] = df["Match_id"].map(home_timeline["GoalsConcededAVG"])
    df["AwayGoalsConcededAVG"] = df["Match_id"].map(away_timeline["GoalsConcededAVG"])
    df["WinRateHome"] = df["Match_id"].map(home_timeline["WinRate"])
    df["WinRateAway"] = df["Match_id"].map(away_timeline["WinRate"])

    # Sliding the Labels and non-training data to the back
    cols = list(df.columns)
    cols_tobe_moved = ["FTHG", "FTAG", "FTR"]
    for col in cols_tobe_moved:
        cols.remove(col)
        cols.append(col)
    df = df[cols]

    df = df.drop(columns="Match_id")
    return df


df = modify_df(clean_df(combine_df()))
df1 = clean_df(combine_df())
print(df)
print(df1)
