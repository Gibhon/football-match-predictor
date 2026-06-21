import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.dataset import read_data

train_features, train_labels, val_features, val_labels = read_data()

unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))
