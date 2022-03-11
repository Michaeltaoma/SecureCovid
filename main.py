import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import pretrained
import preprocess, util, train


def main():
    DATA_PATH = ""

    trainloader, valloader, dataset_size = preprocess.load_split_train_test(DATA_PATH, .2)
    dataloaders = {"train": trainloader, "val": valloader}
    data_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'val']}
    class_names = trainloader.dataset.classes

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Training on GPU... Ready for HyperJump...")
    else:
        device = torch.device("cpu")
        print("Training on CPU... May the force be with you...")

    shadow = pretrained.dense_shadow(class_names, pretrained=True)

    criterion = nn.CrossEntropyLoss()

    # Specify optimizer which performs Gradient Descent
    optimizer = optim.Adam(shadow.parameters(), lr=1e-3)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Learning Scheduler

    best_shadow = train.train_model(shadow, criterion, optimizer, exp_lr_scheduler, data_sizes, dataloaders, num_epochs=10)

    torch.save(best_shadow.state_dict, "trained/best_shadow_{}.pth".format(time.time()))


if __name__ == '__main__':
    main()
