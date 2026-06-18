from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FootballDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        features = torch.tensor(features, dtype=torch.float32)
        self.features = (features - features.mean(dim=0)) / features.std(dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def read_data(
    file_path=Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "processed_df.csv",
):
    data_df = pd.read_csv(file_path)
    data_df = data_df.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTAG", "FTHG"])
    data_np = data_df.to_numpy()

    features_np_final = data_np[:, :-1].astype(np.float32)
    labels_np = data_np[:, -1]

    conditions = [(labels_np == "H"), (labels_np == "A"), (labels_np == "D")]
    choices = [1, 2, 0]

    labels_np_collapsed = np.select(conditions, choices)
    labels_np_final = labels_np_collapsed

    return features_np_final.dtype, labels_np_final.dtype


def load_data(dateset):
    pass
