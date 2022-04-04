import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm.notebook as tqdm

from trainer import train
from data.attck_training_data import AttackData
from model.attack import AttackModel
from preprocess import preprocess
import util

parser = argparse.ArgumentParser(description='Secure Covid')
parser.add_argument('--input_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/attack_train/test/covid_y_pred.pkl',
                    type=str, help='Path to store the data')
parser.add_argument('--target_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/data/attack_train/test/covid_target.pkl',
                    type=str, help='Path to store the data')
parser.add_argument('--out_path', default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/temp', type=str,
                    help='Path to store the trained model')
parser.add_argument('--weight_path',
                    default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/temp/five/covid_attack_1649110011.8203351.pth', type=str,
                    help='Path to load the trained model')
parser.add_argument('--res_path', default='/Users/michaelma/Desktop/Workspace/School/UBC/courses/2021-22-Winter-Term2/EECE571J/project/SecureCovid/image', type=str, help='Path to store the result')
parser.add_argument('--mode', default='eval', type=str, help='Select whether to train, evaluate, inference the model')
parser.add_argument('--label', default='no_covid', type=str, help='Select the label for the attack model')
parser.add_argument('--valid_size', default=.25, type=float, help='Proportion of data used as validation set')
parser.add_argument('--learning_rate', default=.003, type=float, help='Default learning rate')
parser.add_argument('--epoch', default=100, type=int, help='epoch number')
parser.add_argument('--name', default="attack", type=str, help='Name of the model')
args = parser.parse_args()



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Training on GPU... Ready for HyperJump...")
else:
    device = torch.device("cpu")
    print("Training on CPU... May the force be with you...")

if args.mode.__eq__("train"):

    train_path = Path(args.input_path)
    target_path = Path(args.target_path)

    train_data = AttackData(train_path, target_path)
    train_dataloader, val_dataloader, dataset_size = preprocess.load_attack_set(train_data, args.valid_size)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}

    saved_path = Path(args.out_path)
    saved_path = saved_path.joinpath("{}_{}_{}.pth".format(args.label, args.name, time.time()))

    result_path = Path(args.res_path)
    train_result_path = result_path.joinpath("loss_{}_{}_{}.png".format(args.label, args.name, time.time()))
    val_result_path = result_path.joinpath("acc_{}_{}_{}.png".format(args.label, args.name, time.time()))

    learning_rate = args.learning_rate

    attack = AttackModel(2, 64, 1)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(attack.parameters(), lr=learning_rate)

    best_attack, loss_stat, accuracy_stat = train.train_attack_model(device, attack, criterion, optimizer, dataloaders, args.epoch)

    util.toFig_smooth(loss_stat['train'], loss_stat['val'], train_result_path, 0, "Loss", "Loss")

    util.toFig_smooth(accuracy_stat['train'], accuracy_stat['val'], val_result_path, 1, "Accuracy", "Accuracy")

    torch.save(best_attack.state_dict(), saved_path)

    print("{} Attack Model saved to {}".format(args.label, saved_path))

    print("Result image saved to {}".format(result_path))


elif args.mode.__eq__("eval"):
    test_path = Path(args.input_path)
    target_path = Path(args.target_path)

    test_data = AttackData(test_path, target_path)
    test_dataloader = preprocess.load_all_attack(test_data, batch_size = 8)

    attack = AttackModel(2, 64, 1)

    attack.load_state_dict(torch.load(args.weight_path))
    attack = attack.to(device)

    criterion = nn.BCEWithLogitsLoss()
    batch_acc = []
    batch_loss = []
    with torch.no_grad():
        """
        Out-of-training-data data
        """
        for x_batch, y_batch in tqdm.tqdm(test_dataloader, leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.float()
            y_batch = y_batch.unsqueeze(1)
            y_test_pred = attack(x_batch.float())
            acc = train.binary_acc(y_test_pred, y_batch)
            loss = criterion(y_test_pred, y_batch)
            batch_acc.append(acc)
            batch_loss.append(loss)

    print("Accuracy is {}%, loss is {}".format(sum(batch_acc)/len(batch_acc), sum(batch_loss)/len(batch_loss)))

elif args.mode.__eq__("infer"):
    print("infer")
