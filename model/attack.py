import torch.nn as nn


class AttackModel(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(AttackModel, self).__init__()

        # Number of input features is 12.
        self.layer_1 = nn.Linear(n_feature, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, n_hidden)
        self.layer_out = nn.Linear(n_hidden, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
