from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent


def combine_df(base_path=base_dir / "data" / "raw") -> pd.DataFrame:
    raw_dfs = []
    needed_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "AvgH",
        "AvgD",
        "AvgA",
    ]
    for file in base_path.rglob("*.csv"):
        temp_df = pd.read_csv(file, encoding="latin1")
        temp_df = temp_df.rename(
            columns={"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"}
        )
        temp_df = temp_df[needed_cols]
        raw_dfs.append(temp_df)
    if not raw_dfs:
        raise FileNotFoundError(f"No CSV files found in {base_path}")
    return pd.concat(raw_dfs, ignore_index=True)


def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.dropna()


def save_df(df, file_path=base_dir / "data" / "processed" / "processed_df.csv"):
    df.to_csv(file_path)
