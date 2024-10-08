"""
Implementation of a 'Faster R-CNN' (as here: https://arxiv.org/pdf/1506.01497).
    Trained on: 'train/images/positives/'

    Before running this script:
        - Run 'pip install -e .' so that custom imports from ./src/ work 

Custom imports from src:
- Plate_Image_Dataset.py, class Plate_Image_Dataset to load and provide easy access to images.

"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from plate_detect import Negatives_Image_Dataset, helper_training_functions
import torch
# torchvision.transforms.v2 now recommend (faster, do more) (see here:https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor)
from torchvision.transforms.v2 import functional as F
from pathlib import Path
from typing import Dict, Union
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2 as T
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import torch.nn as nn
import torchvision_deps.T_utils.utils as utils

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = weights.transforms()
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

project_dir_root: Path= Path.cwd() # 'SC_TSL_15092024_Plate_Detect/' type PosixPath for UNIX, WindowsPath for windows...

img_dir: Path= project_dir_root.joinpath('train', 'images', 'new_negatives')  # 'SC_TSL_15092024_Plate_Detect/train/images/positives/' on UNIX systems

num_class = 2 # plate or background
# creates resnet50 v2 faster r cnn model with new head for class classification
model, preprocess = helper_training_functions.get_model_instance_object_detection(num_class)
print("I've got the model")
# move model to the right device
model.to(device)

dataset: Negatives_Image_Dataset_copy = Negatives_Image_Dataset.Negatives_Image_Dataset(
    img_dir=str(img_dir), 
    transforms=preprocess, # converts Tensor image, PIL image, NumPy ndarray into FloatTensor and scales pixel intensities in range [0.,1.].
)

# split the dataset in train and test set
dataset_size = len(dataset)
print(len(dataset))
test_size = min(50, int(dataset_size // 5))  # Use 20% of data for testing, or 50 samples, whichever is smaller
indices = [int(i) for i in torch.randperm(dataset_size).tolist()]
dataset_test = torch.utils.data.Subset(dataset, indices[-test_size:])
dataset = torch.utils.data.Subset(dataset, indices[:-test_size])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)
save_dir = 'path/to/model.pth'
num_epochs = 10
precedent_epoch=0

epoch, loss_metrics = helper_training_functions.train(model, data_loader, data_loader_test, device, num_epochs, precedent_epoch)

eval_metrics = helper_training_functions.evaluate_model(model, data_loader_test,device)

helper_training_functions.plot_eval_metrics(eval_metrics, 0)

print("\n -end-")