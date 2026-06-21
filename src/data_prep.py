from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent


def combine_df(base_path=base_dir / "data" / "raw") -> pd.DataFrame:
    needed_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    raw_dfs: pd.DataFrame = [
        pd.read_csv(file, encoding="latin1", usecols=needed_cols)
        for file in base_path.rglob("*.csv")
    ]
    if not raw_dfs:
        raise FileNotFoundError(f"No CSV files found in {base_path}")
    return pd.concat(raw_dfs, ignore_index=True)


def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.dropna()


def save_df(df, file_path=base_dir / "data" / "processed" / "processed_df.csv"):
    df.to_csv(file_path)
