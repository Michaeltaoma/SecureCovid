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
import preprocess

parser = argparse.ArgumentParser(description='Secure Covid')
parser.add_argument('--input_path',
                    default='/content/drive/MyDrive/MEDICAL/attack_train/partition/covid_target.pkl',
                    type=str, help='Path to store the data')
parser.add_argument('--target_path',
                    default='/content/drive/MyDrive/MEDICAL/attack_train/partition/covid_y_pred.pkl',
                    type=str, help='Path to store the data')
parser.add_argument('--out_path', default='/content/drive/MyDrive/MEDICAL/trained', type=str,
                    help='Path to store the trained model')
parser.add_argument('--weight_path',
                    default='/content/drive/MyDrive/MEDICAL/trained/best_shadow_1647045058.8686106.pth', type=str,
                    help='Path to load the trained model')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--valid_size', default=.2, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--epoch', default=10, type=int, help='epoch number')
parser.add_argument('--name', default="best_shadow", type=str, help='Name of the model')
args = parser.parse_args()

train_path = Path(args.input_path)
target_path = Path(args.target_path)

train_data = AttackData(train_path, target_path)
train_dataloader, val_dataloader, dataset_size = preprocess.load_attack_set(train_data, args.valid_size)
dataloaders = {"train": train_dataloader, "val": val_dataloader}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Training on GPU... Ready for HyperJump...")
else:
    device = torch.device("cpu")
    print("Training on CPU... May the force be with you...")

if args.mode.__eq__("train"):

    saved_path = Path(args.out_path)

    saved_path = saved_path.joinpath("{}_{}.pth".format(args.name, time.time()))

    learning_rate = args.learning_rate

    attack = AttackModel(2, 64, 1)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(attack.parameters(), lr=learning_rate)

    best_attack = train.train_attack_model(device, attack, criterion, optimizer, dataloaders, 20)

    torch.save(best_attack.state_dict(), saved_path)

    print("Attack Model saved to {}".format(saved_path))


elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    print("infer")
