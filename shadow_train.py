import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import preprocess
from trainer import train
from model import pretrained, covid_net, cnn
import util
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


parser = argparse.ArgumentParser(description='Secure Covid Shadow Train')
# parser.add_argument('--data_path', default='/content/COVID19-DATASET', type=str, help='Path to store the data')
parser.add_argument('--data_path', default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/data/content/COVID19-DATASET', type=str, help='Path to store the data')
parser.add_argument('--out_path', default='/content/drive/MyDrive/MEDICAL/trained', type=str,
                    help='Path to store the trained model')
parser.add_argument('--weight_path',
                    default='/content/drive/MyDrive/MEDICAL/trained/best_shadow_1647045058.8686106.pth', type=str,
                    help='Path to load the trained model')
parser.add_argument('--res_path', default='/content/drive/MyDrive/EECE571J/m2_result/final_folder', type=str,
                    help='Path to store the training result')
parser.add_argument('--mode', default='train', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--model', default='dense', type=str, help='Select which model to use')
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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Training on GPU... Ready for HyperJump...")
else:
    device = torch.device("cpu")
    print("Training on CPU... May the force be with you...")

if args.mode.__eq__("train"):
    learning_rate = args.learning_rate
    step_size = args.step_size
    gamma = args.gamma
    epoch = args.epoch

    if args.model.__eq__("dense"):
        trainloader, valloader, dataset_size = preprocess.load_split_train_test(TRAIN_PATH, args.valid_size,
                                                                                preprocess.data_transforms)
        dataloaders = {"train": trainloader, "val": valloader}
        data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}
        class_names = trainloader.dataset.classes
        shadow = pretrained.dense_shadow(device, class_names, pretrained=True)
    elif args.model.__eq__("covidnet"):
        trainloader, valloader, dataset_size = preprocess.load_split_train_test(TRAIN_PATH, args.valid_size,
                                                                                transform=preprocess.covid_data_transforms)
        dataloaders = {"train": trainloader, "val": valloader}
        data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}
        class_names = trainloader.dataset.classes
        shadow = covid_net.CovidNet(model='small', n_classes=2)
        shadow = shadow.to(device)
    else:
        trainloader, valloader, dataset_size = preprocess.load_split_train_test(TRAIN_PATH, args.valid_size)
        dataloaders = {"train": trainloader, "val": valloader}
        data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}
        class_names = trainloader.dataset.classes
        shadow = cnn.ConvNet()

    saved_path = Path(args.out_path)
    saved_path = saved_path.joinpath("{}_{}_{}.pth".format(args.mode, args.name, time.time()))

    result_path = Path(args.res_path)
    result_path = result_path.joinpath("{}_{}_{}.png".format(args.mode, args.name, time.time()))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(shadow.parameters(), lr=learning_rate)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_shadow, epoch_loss_record, epoch_acc_record = train.train_model(device, shadow, criterion, optimizer,
                                                                         exp_lr_scheduler, data_sizes, dataloaders,
                                                                         num_epochs=epoch)

    util.toFig(epoch_loss_record, epoch_acc_record, result_path)

    torch.save(best_shadow.state_dict(), saved_path)

    print("Shadow Model saved to {}".format(saved_path))

    print("Result image saved to {}".format(result_path))

elif args.mode.__eq__("eval"):
    print("to eval")

elif args.mode.__eq__("infer"):
    print("to infer")
