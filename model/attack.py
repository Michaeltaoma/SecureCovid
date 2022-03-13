import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackModel(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(AttackModel, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.last = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.last(x)
        return x
