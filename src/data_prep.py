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
        "HTAG",
        "HTHG",
        "HTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    ]
    clean_df = raw_df.drop(columns=drop_cols)
    clean_df = clean_df.dropna()
    return clean_df


def save_df(df, file_path=BASE_DIR / "data" / "processed" / "processed_df.csv"):
    df.to_csv(file_path)
