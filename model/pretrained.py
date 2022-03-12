import torch.nn as nn
from torchvision import models


def dense_shadow(device, class_names, pretrained=True):
    """load pretrained densenet"""
    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    return model
