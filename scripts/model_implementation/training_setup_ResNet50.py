import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from lib.ResNet50 import ResNet, get_ResNet

from src import Plate_Image_Dataset, helper_training_functions

from torchvision.transforms.v2 import functional as F
from pathlib import Path
from typing import Dict, Union
from torchvision.transforms import v2 as T
import lib.utils as utils
from lib.engine import train_one_epoch, evaluate 
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image


if __name__ == "__main__":
    model_dir = 'resnet50-11ad3fa6.pth'
    model = get_ResNet(model_dir, BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
    print(model)

    model.eval()  # or 
    model.train() # if training

# see recipe here: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

#  Setup optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=2e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=600)

#  Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Setup augmentations for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(176),
        transforms.TrivialAugmentWide(),
        transforms.RandomApply([transforms.RandomErasing(p=0.1)], p=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])