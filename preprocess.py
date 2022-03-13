import numpy as np

import torch
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

data_transforms = {"train": transforms.Compose([
    transforms.Resize((150, 150)),  # Resizes all images into same dimension
    transforms.RandomRotation(10),  # Rotates the images upto Max of 10 Degrees
    transforms.RandomHorizontalFlip(p=0.4),  # Performs Horizantal Flip over images
    transforms.ToTensor(),  # Coverts into Tensors
    transforms.Normalize(mean=mean_nums, std=std_nums)]),  # Normalizes
    "val": transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.CenterCrop(150),  # Performs Crop at Center and resizes it to 150x150
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])}

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_nums, std=std_nums)
])


def load_attack_set(dataset, valid_size):
    train_data = dataset
    test_data = dataset
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    dataset_size = {"train": len(train_idx), "val": len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=8)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=8)
    return trainloader, testloader, dataset_size


def load_split_train_test(datadir, valid_size=.2):
    train_data = datasets.ImageFolder(datadir,
                                      transform=data_transforms['train'])
    test_data = datasets.ImageFolder(datadir,
                                     transform=data_transforms['val'])
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    dataset_size = {"train": len(train_idx), "val": len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=8)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=8)
    return trainloader, testloader, dataset_size


def load_all_train(datadir, valid_size=0.0):
    train_data = datasets.ImageFolder(datadir,
                                      transform=preprocess.data_transforms['train'])
    test_data = datasets.ImageFolder(datadir,
                                     transform=preprocess.data_transforms['val'])
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    dataset_size = {"train": len(train_idx), "val": len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=1)
    return trainloader
