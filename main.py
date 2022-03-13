import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import preprocess
import train
from model import pretrained

parser = argparse.ArgumentParser(description='Secure Covid')
parser.add_argument('--data_path', default='/content/COVID19-DATASET/', type=str, help='Path to store the data')
parser.add_argument('--out_path', default='/content/drive/MyDrive/MEDICAL/trained', type=str,
                    help='Path to store the trained model')
parser.add_argument('--weight_path',
                    default='/content/drive/MyDrive/MEDICAL/trained/best_shadow_1647045058.8686106.pth', type=str,
                    help='Path to load the trained model')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--valid_size', default=.2, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--step_size', default=7, type=int, help='Default step size')
parser.add_argument('--gamma', default=0.1, type=float, help='Default gamma')
parser.add_argument('--epoch', default=10, type=int, help='epoch number')
parser.add_argument('--name', default="best_shadow", type=str, help='Name of the model')
args = parser.parse_args()

# For what should be in this dir, refer to shadow.ipynb
DATA_PATH = Path(args.data_path)
TRAIN_PATH = DATA_PATH.joinpath("train")
TEST_PATH = DATA_PATH.joinpath("test")

trainloader, valloader, dataset_size = preprocess.load_split_train_test(TRAIN_PATH, args.valid_size)
dataloaders = {"train": trainloader, "val": valloader}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}
class_names = trainloader.dataset.classes

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
    step_size = args.step_size
    gamma = args.gamma
    epoch = args.epoch

    shadow = pretrained.dense_shadow(device, class_names, pretrained=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(shadow.parameters(), lr=learning_rate)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_shadow = train.train_model(device, shadow, criterion, optimizer, exp_lr_scheduler, data_sizes, dataloaders, num_epochs=epoch)

    torch.save(best_shadow.state_dict(), saved_path)

    print("Model saved to {}".format(saved_path))

elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    print("to infer")
