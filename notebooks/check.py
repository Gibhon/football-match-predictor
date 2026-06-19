import numpy as np
from dataset import read_data

train_features, train_labels, val_features, val_labels = read_data()

unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))


print(17339 / (3 * 4692))  # draw

print(17339 / (3 * 7411))  # home
print(17339 / (3 * 5236))  # away
