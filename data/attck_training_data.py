import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import util

import os
import pandas as pd


class AttackData(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.array(util.fromPickle(x_path), dtype="float64")
        self.x = self.x.reshape(self.x.shape[0], 2)
        y = np.array(util.fromPickle(y_path), dtype="float64")
        self.y = y.reshape(y.shape[0], )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        prediction_vector = self.x[idx]
        in_or_out = self.y[idx]
        return prediction_vector, in_or_out
