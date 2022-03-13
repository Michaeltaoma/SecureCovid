import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import preprocess
import train
from data.attck_training_data import AttackData
from model.attack import AttackModel

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Training on GPU... Ready for HyperJump...")
else:
    device = torch.device("cpu")
    print("Training on CPU... May the force be with you...")

train_path = "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/partition/covid_y_pred.pkl"
target_path = "/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/partition/covid_target.pkl"
train_data = AttackData(train_path, target_path)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
dataloader = {"train": train_dataloader}
attack = AttackModel(2, 128, 2)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(attack.parameters(), lr=1e-3)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.07)

best_attack = train.train_model(device, attack, criterion, optimizer, exp_lr_scheduler, )


