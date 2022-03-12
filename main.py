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
    saved_path = saved_path.joinpath("best_shadow_{}.pth".format(time.time()))

    shadow = pretrained.dense_shadow(device, class_names, pretrained=True)

    criterion = nn.CrossEntropyLoss()

    # Specify optimizer which performs Gradient Descent
    optimizer = optim.Adam(shadow.parameters(), lr=1e-3)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Learning Scheduler

    best_shadow = train.train_model(device, shadow, criterion, optimizer, exp_lr_scheduler, data_sizes, dataloaders, num_epochs=10)

    torch.save(best_shadow.state_dict, saved_path)

elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    print("to infer")
