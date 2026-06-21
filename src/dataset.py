from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class FootballDataset(Dataset):
    def __init__(self, features, labels, mean=None, std=None):
        super().__init__()
        features = torch.tensor(features, dtype=torch.float32)
        if mean is None and std is None:
            self.mean = features.mean(0)
            self.std = features.std(0)
        else:
            self.mean = mean
            self.std = std
        self.features = (features - self.mean) / self.std
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
    condition = data_df["Season"] == 2024
    train_data_df = data_df[~condition]
    val_data_df = data_df[condition]

    def data_processor(data_df):
        data_df = data_df.drop(
            columns=["Date", "HomeTeam", "AwayTeam", "FTAG", "FTHG", "Season"]
        )
        data_np = data_df.to_numpy()

        features_np = data_np[:, :-1].astype(np.float32)
        labels_np = data_np[:, -1]

        conditions = [(labels_np == "D"), (labels_np == "H"), (labels_np == "A")]
        choices = [0, 1, 2]
        labels_np = np.select(conditions, choices)

        return features_np, labels_np

    train_features_np, train_labels_np = data_processor(train_data_df)
    val_features_np, val_labels_np = data_processor(val_data_df)

    return train_features_np, train_labels_np, val_features_np, val_labels_np


def load_data(dataset, shuffle, num_workers=1):
    return DataLoader(dataset, shuffle=shuffle, batch_size=32, num_workers=num_workers)
