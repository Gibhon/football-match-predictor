from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def get_h2h_wr(df, current_row, N=5):
    current_date = current_row["Date"]
    t1, t2 = current_row["HomeTeam"], current_row["AwayTeam"]

    past_matches = df[
        (df["Date"] < current_date)
        & (
            ((df["HomeTeam"] == t1) & (df["AwayTeam"] == t2))
            | ((df["HomeTeam"] == t2) & (df["AwayTeam"] == t1))
        )
    ]

    latest_h2h = past_matches.sort_values("Date").tail(N)

    if len(latest_h2h) == 0:
        return 0.0, 0.0

    home_wins = 0
    away_wins = 0

    for _, match in latest_h2h.iterrows():
        if match["HomeTeam"] == t1:
            if match["FTR"] == "H":
                home_wins += 1
            elif match["FTR"] == "A":
                away_wins += 1
        else:
            if match["FTR"] == "A":
                home_wins += 1
            elif match["FTR"] == "H":
                away_wins += 1

    total_games = len(latest_h2h)
    home_h2h_wr = home_wins / total_games
    away_h2h_wr = away_wins / total_games

    return home_h2h_wr, away_h2h_wr


def modify_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    # Index Preperation
    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    df = clean_df.sort_values("Date").reset_index(drop=True)
    df["Match_id"] = df.index

    # Making separate df for two sides
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

    # Combining Home and Away df
    timeline = pd.concat([home_side, away_side]).sort_values(by="Date")
    timeline["PointsEarned"] = timeline["Outcome"].map({"W": 3, "L": 0, "D": 1})
    timeline["TotalWins"] = timeline["Outcome"].map({"W": 1, "L": 0, "D": 0})

    grouped_timeline = timeline.groupby("Team")
    shifts_points = [grouped_timeline["PointsEarned"].shift(i) for i in range(1, 6)]
    shifts_scored = [grouped_timeline["GoalsScored"].shift(i) for i in range(1, 6)]
    shifts_conceded = [grouped_timeline["GoalsConceded"].shift(i) for i in range(1, 6)]
    shifts_wins = [grouped_timeline["TotalWins"].shift(i) for i in range(1, 6)]

    timeline["TotalFormScore"] = (
        pd.concat(shifts_points, axis=1).sum(axis=1, min_count=1).fillna(0)
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

    h2h_stats = df.apply(lambda r: get_h2h_wr(df, r, N=5), axis=1)
    df["H2H_Home_WinRate"] = [x[0] for x in h2h_stats]
    df["H2H_Away_WinRate"] = [x[1] for x in h2h_stats]

    df["Season"] = df["Date"].dt.year
    # Sliding the Labels and non-training data to the back
    cols = list(df.columns)
    cols_tobe_moved = ["FTHG", "FTAG", "FTR"]
    for col in cols_tobe_moved:
        cols.remove(col)
        cols.append(col)
    df = df[cols]

    df = df.drop(columns="Match_id")
    return df
