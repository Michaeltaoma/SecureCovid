from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import util

import os
import pandas as pd


class AttackData(Dataset):
    def __init__(self, x_path, y_path):
        self.x = util.fromPickle(x_path)
        self.y = util.fromPickle(y_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        prediction_vector = self.x[idx]
        in_or_out = self.y[idx]
        return prediction_vector, in_or_out


train_path = "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/partition/covid_y_pred.pkl"
target_path = "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/partition/covid_target.pkl"
train_data = AttackData(train_path, target_path)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
print("hello")